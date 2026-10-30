"""
Modular Model Predictive Controller (`pure_mpc_controller.py`).
LTV-MPC with 1st-Order Actuator Dynamics Embedded in State Equations:
  z(k) = [v(k), u_act(k-1)]^T
  z(k+1) = Az * z(k) + Bz * u_cmd(k) + rz(k)
"""
from __future__ import annotations

import numpy as np


class PureLiuMPC:
    def __init__(
        self,
        n_p: int = 10,
        n_c: int = 5,
        q1: float = 160.0,
        q2: float = 0.05,
        q3: float = 0.35,
        adaptive_shift: bool = True,
        u_min: float = -1.0,
        u_max: float = 1.0,
        dt: float = 0.5,
        mass_kg: float = 85000.0,
        gamma_rot: float = 0.08,
        davis_c_const: float = 1450.0,
        davis_b_lin: float = 185.0,
        davis_a_quad: float = 14.5,
        max_traction_accel: float = 1.85,
        max_braking_decel: float = 1.65,
    ):
        self.n_p = n_p
        self.n_c = min(n_c, n_p)
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.adaptive_shift = adaptive_shift
        self.u_min = u_min
        self.u_max = u_max
        self.dt = dt
        self.M = mass_kg
        self.gamma = gamma_rot

        self.a_a = davis_c_const / self.M
        self.b_a = davis_b_lin / self.M
        self.c_a = davis_a_quad / self.M

        self.f_a_max_unit = max_traction_accel * (1.0 + self.gamma)
        self.f_b_max_unit = max_braking_decel * (1.0 + self.gamma)
        self.g = 9.81
        self.prev_u: float = 0.0
        self.prev_v: float = 0.0
        self.u_act: float = 0.0

        self.D = np.eye(self.n_c, dtype=np.float64)
        for i in range(1, self.n_c):
            self.D[i, i - 1] = -1.0

        # Moderate proximal weighting for cruising/shifts (0.75^i)
        self.prox_weights = np.array([0.75 ** i for i in range(self.n_p)], dtype=np.float64)
        self.prox_weights /= np.mean(self.prox_weights)

        # Balanced stopping horizon weights (0.90^i) so MPC sees the entire stopping S-curve
        self.stop_weights = np.array([0.90 ** i for i in range(self.n_p)], dtype=np.float64)
        self.stop_weights /= np.mean(self.stop_weights)

    def reset(self) -> None:
        self.prev_u = 0.0
        self.prev_v = 0.0
        self.u_act = 0.0

    def _gamma_a_xi(self, v: float, u: float) -> float:
        if u >= 0.0:
            return self.f_a_max_unit if v <= 14.0 else self.f_a_max_unit * (14.0 / max(v, 14.0))
        return self.f_b_max_unit

    def _d_gamma_a_dv(self, v: float, u: float) -> float:
        if u > 0.0 and v > 14.0:
            return -self.f_a_max_unit * 14.0 / (v ** 2)
        return 0.0

    def f_nonlinear(
        self, v: float, u: float, slope_iota: float = 0.0, dist_unit: float = 0.0
    ) -> float:
        u_clamped = float(np.clip(u, self.u_min, self.u_max))
        gamma_xi = self._gamma_a_xi(v, u_clamped)
        h_omega = -(self.a_a + self.b_a * v + self.c_a * (v ** 2)) if v > 1e-4 else 0.0
        accel = (u_clamped * gamma_xi + h_omega - self.g * slope_iota + dist_unit) / (1.0 + self.gamma)
        return float(max(0.0, v + self.dt * accel))

    def linearize_step(self, x0: float, u0: float, slope_iota: float = 0.0) -> tuple[float, float, float]:
        x0_s = max(0.0, float(x0))
        u0_s = float(np.clip(u0, self.u_min, self.u_max))

        d_gamma = self._d_gamma_a_dv(x0_s, u0_s)
        gamma_val = self._gamma_a_xi(x0_s, u0_s)

        A = 1.0 + (self.dt / (1.0 + self.gamma)) * (u0_s * d_gamma - self.b_a - 2.0 * self.c_a * x0_s)
        B = (self.dt / (1.0 + self.gamma)) * gamma_val

        f0 = self.f_nonlinear(x0_s, u0_s, slope_iota=slope_iota)
        r_k = f0 - A * x0_s - B * u0_s
        return float(A), float(B), float(r_k)

    def build_augmented_prediction_matrices(
        self, A: float, B: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast O(N_p) lifted horizon matrices for state z(k) = [v(k), u_act(k-1)]^T:
            v(k+1) = A*v(k) + 0.28*B*u_act(k-1) + 0.72*B*u_cmd(k) + r(k)
            u_act(k) = 0.28*u_act(k-1) + 0.72*u_cmd(k)
        """
        Az = np.array([[A, 0.28 * B], [0.0, 0.28]], dtype=np.float64)
        Bz = np.array([[0.72 * B], [0.72]], dtype=np.float64)

        F = np.zeros((self.n_p, 2), dtype=np.float64)
        Phi = np.zeros((self.n_p, self.n_c), dtype=np.float64)
        Psi = np.zeros((self.n_p, 1), dtype=np.float64)

        step_impulses = np.zeros(self.n_p, dtype=np.float64)
        A_pow = np.eye(2, dtype=np.float64)
        psi_val = 0.0

        for i in range(self.n_p):
            step_impulses[i] = A_pow[0, 0] * Bz[0, 0] + A_pow[0, 1] * Bz[1, 0]
            A_pow = A_pow @ Az
            F[i, 0] = A_pow[0, 0]
            F[i, 1] = A_pow[0, 1]

            psi_val = psi_val * A + 1.0
            Psi[i, 0] = psi_val

            for j in range(min(i + 1, self.n_c)):
                val = step_impulses[i - j]
                if j < self.n_c - 1 or i == j:
                    Phi[i, j] = val
                else:
                    Phi[i, j] += val

        return F, Phi, Psi

    def _solve_box_constrained_qp(
        self,
        Phi: np.ndarray,
        E_0: np.ndarray,
        q1_vec: np.ndarray,
        q2_val: float,
        q3_val: float,
    ) -> float:
        d_0 = np.zeros((self.n_c, 1), dtype=np.float64)
        d_0[0, 0] = self.prev_u

        Phi_w = Phi * q1_vec[:, None]
        H = (Phi.T @ Phi_w) + q2_val * np.eye(self.n_c) + q3_val * (self.D.T @ self.D)
        g = (Phi_w.T @ E_0) - q3_val * (self.D.T @ d_0)

        U_unc = -np.linalg.solve(H + 1e-8 * np.eye(self.n_c), g)
        if np.all((U_unc >= self.u_min) & (U_unc <= self.u_max)):
            return float(U_unc[0, 0])

        U = np.clip(U_unc, self.u_min, self.u_max)
        step_lr = 1.0 / (float(np.max(np.linalg.eigvalsh(H))) + 1e-6)
        for _ in range(12):
            U_new = np.clip(U - step_lr * (H @ U + g), self.u_min, self.u_max)
            if float(np.max(np.abs(U_new - U))) < 1e-5:
                U = U_new
                break
            U = U_new
        return float(np.clip(U[0, 0], self.u_min, self.u_max))

    def select_action(
        self,
        v_curr: float,
        v_ref_horizon: np.ndarray,
        slope_iota: float = 0.0,
        dt_step: float | None = None,
        u_act_prev: float | None = None,
    ) -> float:
        ref = np.asarray(v_ref_horizon, dtype=np.float64).flatten()
        if len(ref) == 0:
            ref = np.full(self.n_p, float(v_curr), dtype=np.float64)
        elif len(ref) < self.n_p:
            ref = np.pad(ref, (0, self.n_p - len(ref)), mode="edge")
        else:
            ref = ref[: self.n_p]

        if dt_step is not None:
            self.dt = float(dt_step)

        u_act_curr = float(self.u_act if u_act_prev is None else u_act_prev)
        dv_curr = float(v_curr - self.prev_v)
        self.prev_v = float(v_curr)

        X_ref = ref.reshape(self.n_p, 1).copy()
        dv_01 = float(ref[min(1, len(ref) - 1)] - ref[0])

        # Causal Pre-Shift Guard: when current plateau is flat, only allow 1-step pre-charging
        if self.adaptive_shift and abs(dv_01) < 0.025 and abs(float(ref[0]) - v_curr) < 0.15:
            X_ref[2:, 0] = float(ref[1])

        v_eff = float(v_curr) + (0.14 * dv_curr if self.adaptive_shift else 0.0)
        z0 = np.array([[v_eff], [u_act_curr]], dtype=np.float64)

        A, B, r_k = self.linearize_step(v_curr, u_act_curr, slope_iota=slope_iota)
        F, Phi, Psi = self.build_augmented_prediction_matrices(A, B)

        E_0 = F @ z0 + Psi * r_k - X_ref

        if self.adaptive_shift:
            err_0 = abs(float(ref[0]) - float(v_curr))
            is_shift = err_0 > 1.2 or (abs(dv_curr) > 0.45 and abs(dv_01) < 0.05)
            is_stopping_ramp = (float(np.min(ref)) <= 3.5 and dv_01 < -0.02) or (float(ref[0]) <= 3.2)
            if is_shift:
                q1_vec = (self.q1 * 2.4) * self.prox_weights
                q2_eff, q3_eff = 1e-4, 0.02
            elif is_stopping_ramp:
                q1_vec = (self.q1 * 3.8) * self.stop_weights
                q2_eff, q3_eff = 1e-4, 0.03
            else:
                q1_vec = (self.q1 * 1.6) * self.prox_weights
                q2_eff, q3_eff = self.q2 * 0.5, self.q3 * 0.5
        else:
            q1_vec = np.full(self.n_p, self.q1, dtype=np.float64)
            q2_eff, q3_eff = self.q2, self.q3

        u_opt = self._solve_box_constrained_qp(Phi, E_0, q1_vec, q2_eff, q3_eff)

        if self.adaptive_shift and float(ref[0]) > 2.5 and float(ref[-1]) > 2.5:
            err_now = float(ref[0]) - float(v_curr)
            if abs(dv_01) < 0.05 and abs(dv_curr) > 0.25:
                if (dv_curr > 0.25 and 0.0 <= err_now < 0.65 * dv_curr) or (
                    dv_curr < -0.25 and 0.65 * dv_curr < err_now <= 0.0
                ):
                    u_opt = float(np.clip(u_opt - 0.50 * np.sign(dv_curr), self.u_min, self.u_max))

        self.prev_u = u_opt
        self.u_act = 0.72 * u_opt + 0.28 * u_act_curr
        return u_opt