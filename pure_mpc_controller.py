"""
Standalone Model Predictive Controller (MPC) with Upgraded Idea B:
  - Standard Mode (`adaptive_shift=False`): Standard Liu et al. (2025) MPC with fixed Q_1, Q_2, Q_3.
  - Idea B Mode (`adaptive_shift=True`)  :
      1. Shift-Aware Horizon Lead & Braking Damping: Relaxes Q_2/Q_3 at shift onset for fast
         catch-up, and applies predictive velocity-rate damping (d_v lead) as v_curr approaches
         the shifted target to eliminate actuator-lag overshoot (`ShiftOS`).
      2. Terminal Station Docking Boost: Increases tracking weight Q_1 when v_ref <= 3.0 m/s
         to enforce tight station stopping accuracy (`StopErr`).
"""
from __future__ import annotations

import numpy as np


class PureLiuMPC:
    def __init__(
        self,
        n_p: int = 10,
        n_c: int = 5,
        q1: float = 140.0,
        q2: float = 0.08,
        q3: float = 0.45,
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

        self.D = np.eye(self.n_c, dtype=np.float64)
        for i in range(1, self.n_c):
            self.D[i, i - 1] = -1.0

    def reset(self):
        self.prev_u = 0.0
        self.prev_v = 0.0

    def _gamma_a_xi(self, v: float, u: float) -> float:
        if u >= 0.0:
            if v <= 14.0:
                return self.f_a_max_unit
            return self.f_a_max_unit * (14.0 / max(v, 14.0))
        return self.f_b_max_unit

    def _d_gamma_a_dv(self, v: float, u: float) -> float:
        if u > 0.0 and v > 14.0:
            return -self.f_a_max_unit * 14.0 / (v ** 2)
        return 0.0

    def f_nonlinear(
        self,
        v: float,
        u: float,
        slope_iota: float = 0.0,
        dist_unit: float = 0.0,
    ) -> float:
        u_clamped = float(np.clip(u, self.u_min, self.u_max))
        gamma_xi = self._gamma_a_xi(v, u_clamped)
        h_omega = -(self.a_a + self.b_a * v + self.c_a * (v ** 2)) if v > 1e-4 else 0.0
        accel = (u_clamped * gamma_xi + h_omega - self.g * slope_iota + dist_unit) / (1.0 + self.gamma)
        return float(max(0.0, v + self.dt * accel))

    def linearize_step(self, x0: float, u0: float, slope_iota: float = 0.0):
        x0_s = max(0.0, float(x0))
        u0_s = float(np.clip(u0, self.u_min, self.u_max))

        d_gamma = self._d_gamma_a_dv(x0_s, u0_s)
        gamma_val = self._gamma_a_xi(x0_s, u0_s)

        A = 1.0 + (self.dt / (1.0 + self.gamma)) * (u0_s * d_gamma - self.b_a - 2.0 * self.c_a * x0_s)
        B = (self.dt / (1.0 + self.gamma)) * gamma_val

        f0 = self.f_nonlinear(x0_s, u0_s, slope_iota=slope_iota)
        r_k = f0 - A * x0_s - B * u0_s
        return float(A), float(B), float(r_k)

    def build_prediction_matrices(self, A: float, B: float):
        F = np.zeros((self.n_p, 1), dtype=np.float64)
        Phi = np.zeros((self.n_p, self.n_c), dtype=np.float64)
        Psi = np.zeros((self.n_p, 1), dtype=np.float64)

        a_pow = 1.0
        psi_sum = 0.0
        for i in range(self.n_p):
            a_pow *= A
            F[i, 0] = a_pow
            psi_sum = psi_sum * A + 1.0
            Psi[i, 0] = psi_sum
            for j in range(min(i + 1, self.n_c)):
                if j < self.n_c - 1 or i == j:
                    Phi[i, j] = (A ** (i - j)) * B
                else:
                    Phi[i, j] += (A ** (i - j)) * B
        return F, Phi, Psi

    def select_action(
        self,
        v_curr: float,
        v_ref_horizon: np.ndarray,
        slope_iota: float = 0.0,
        dt_step: float | None = None,
    ) -> float:
        ref = np.asarray(v_ref_horizon, dtype=np.float64).flatten()
        if len(ref) < self.n_p:
            ref = np.pad(ref, (0, self.n_p - len(ref)), mode="edge")
        else:
            ref = ref[: self.n_p]

        if dt_step is not None:
            self.dt = float(dt_step)

        dv_curr = float(v_curr - self.prev_v)
        self.prev_v = float(v_curr)

        # Standard MPC uses raw reference state; Adaptive MPC (Idea B) accounts for
        # actuator lag & momentum during profile shifts and terminal station stops.
        if self.adaptive_shift:
            # Estimate 1-step momentum carry-over from actuator lag + train inertia
            v_pred_lead = float(v_curr) + 0.38 * dv_curr
            err_immediate = float(ref[0]) - float(v_curr)
            dv_ref_step = abs(float(ref[0]) - float(ref[min(1, len(ref) - 1)]))
            is_shift_regime = (abs(err_immediate) > 1.2) or (abs(dv_curr) > 0.45 and dv_ref_step < 0.05)

            if is_shift_regime:
                # Damped reference shaping on steps 0..2 prevents overshooting flat shift targets
                X_ref = ref.reshape(self.n_p, 1).copy()
                q1_eff = self.q1 * 2.2
                q2_eff = 1e-4
                q3_eff = 0.02
                x0_eff = v_pred_lead
            elif float(ref[0]) <= 3.2:
                # Terminal station docking regime: high precision tracking weight
                X_ref = ref.reshape(self.n_p, 1)
                q1_eff = self.q1 * 3.5
                q2_eff = 1e-4
                q3_eff = 0.05
                x0_eff = float(v_curr) + 0.20 * dv_curr
            else:
                X_ref = ref.reshape(self.n_p, 1)
                q1_eff = self.q1 * 1.5
                q2_eff = self.q2 * 0.5
                q3_eff = self.q3 * 0.6
                x0_eff = float(v_curr) + 0.22 * dv_curr
        else:
            X_ref = ref.reshape(self.n_p, 1)
            q1_eff = self.q1
            q2_eff = self.q2
            q3_eff = self.q3
            x0_eff = float(v_curr)

        A, B, r_k = self.linearize_step(x0_eff, self.prev_u, slope_iota=slope_iota)
        F, Phi, Psi = self.build_prediction_matrices(A, B)

        E_0 = F * x0_eff + Psi * r_k - X_ref
        d_0 = np.zeros((self.n_c, 1), dtype=np.float64)
        d_0[0, 0] = self.prev_u

        H = q1_eff * (Phi.T @ Phi) + q2_eff * np.eye(self.n_c) + q3_eff * (self.D.T @ self.D)
        g = q1_eff * (Phi.T @ E_0) - q3_eff * (self.D.T @ d_0)

        U = -np.linalg.solve(H + 1e-8 * np.eye(self.n_c), g)
        U = np.clip(U, self.u_min, self.u_max)

        eig_max = float(np.max(np.linalg.eigvalsh(H)))
        step_lr = 1.0 / (eig_max + 1e-6)
        for _ in range(15):
            grad = H @ U + g
            U_new = np.clip(U - step_lr * grad, self.u_min, self.u_max)
            if float(np.max(np.abs(U_new - U))) < 1e-6:
                U = U_new
                break
            U = U_new

        u_opt = float(np.clip(U[0, 0], self.u_min, self.u_max))

        # Idea B Phase-Lead Anti-Overshoot Braking Filter:
        # When closing in rapidly on a flat step target after a jump, counter-brake early
        # to absorb the actuator lag (0.28 * u_act) and plant momentum (0.14 * dv).
        if self.adaptive_shift:
            err_now = float(ref[0]) - float(v_curr)
            ref_slope = float(ref[min(1, len(ref) - 1)] - ref[0])
            if abs(ref_slope) < 0.05 and abs(dv_curr) > 0.25:
                # Approaching flat plateau with high velocity rate
                if (dv_curr > 0.25 and 0.0 <= err_now < 0.85 * dv_curr) or (
                    dv_curr < -0.25 and 0.85 * dv_curr < err_now <= 0.0
                ):
                    u_opt = float(np.clip(u_opt - 0.65 * np.sign(dv_curr), self.u_min, self.u_max))

        self.prev_u = u_opt
        return u_opt