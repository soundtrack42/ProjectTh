"""
Modular Model Predictive Controller (`pure_mpc_controller.py`).

References & Mathematical Formulation:
  - Liu et al. (2025), "Model Predictive Control-Based Speed Tracking Control for Subway
    Trains Subject to Actuator Constraints," Actuators 14(3), 134, Eqs. (1)-(16).
  - State-Augmented LTV-MPC Formulation (Wabersich & Zeilinger, 2018; Liu et al., 2025 Eq. 12-14):
    Embeds the 1st-order actuator lag `u_act(k) = 0.72 * u_cmd(k) + 0.28 * u_act(k-1)` and
    longitudinal inertia momentum `0.14 * dv(k)` directly into the lifted state-space dynamics:
        z(k) = [v(k), u_act(k-1)]^T
        z(k+1) = Az * z(k) + Bz * u_cmd(k) + rz(k)
        X(k+1) = F * z(k) + Phi * U(k) + Psi * r(k)
"""
from __future__ import annotations

import numpy as np


class PureLiuMPC:
    def __init__(
        self,
        n_p: int = 5,           # 0.28^5 is 0.0017, so 5-step horizon is sufficient for actuator lag decay
        n_c: int = 3,
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

        # Mass-normalized Davis basic resistance coefficients (Liu et al. 2025 Eq. 4-5; Havaei 2022 Eq. 5)
        self.a_a = davis_c_const / self.M
        self.b_a = davis_b_lin / self.M
        self.c_a = davis_a_quad / self.M

        self.f_a_max_unit = max_traction_accel * (1.0 + self.gamma)
        self.f_b_max_unit = max_braking_decel * (1.0 + self.gamma)
        self.g = 9.81
        self.prev_u: float = 0.0
        self.prev_v: float = 0.0
        self.u_act: float = 0.0

        # First-difference matrix D for control variation penalty ||Delta U||^2_Q3 (Liu et al. 2025 Eq. 14)
        self.D = np.eye(self.n_c, dtype=np.float64)
        for i in range(1, self.n_c):
            self.D[i, i - 1] = -1.0
        self.DtD = self.D.T @ self.D
        self.I_nc = np.eye(self.n_c, dtype=np.float64)

        # Pre-allocated buffers for zero-allocation receding-horizon matrix construction
        self._F = np.zeros((self.n_p, 2), dtype=np.float64)
        self._Phi = np.zeros((self.n_p, self.n_c), dtype=np.float64)
        self._Psi = np.zeros((self.n_p, 1), dtype=np.float64)
        self._step_impulses = np.zeros(self.n_p, dtype=np.float64)
        self._E0 = np.zeros((self.n_p, 1), dtype=np.float64)
        self._X_ref = np.zeros((self.n_p, 1), dtype=np.float64)

        # Proximal horizon decay weights so the 2-step actuator lag window dominates tracking
        self.prox_weights = np.array([0.65 ** i for i in range(self.n_p)], dtype=np.float64)
        self.prox_weights /= np.mean(self.prox_weights)

        # Balanced stopping horizon weights (0.85^i) for smooth terminal station docking
        self.stop_weights = np.array([0.85 ** i for i in range(self.n_p)], dtype=np.float64)
        self.stop_weights /= np.mean(self.stop_weights)

        self._q1_shift = (self.q1 * 5.5) * self.prox_weights
        self._q1_stop = (self.q1 * 5.0) * self.stop_weights
        self._q1_cruise = (self.q1 * 2.8) * self.prox_weights
        self._q1_fixed = np.full(self.n_p, self.q1, dtype=np.float64)

    def reset(self) -> None:
        self.prev_u = 0.0
        self.prev_v = 0.0
        self.u_act = 0.0

    # =========================================================================
    # 1. Nonlinear Traction/Braking Characteristics & Taylor Linearization
    #    (Liu et al. 2025 Eqs. 2-3, 9, 11, 12)
    # =========================================================================
    def _gamma_a_xi(self, v: float, u: float) -> float:
        """Maximum traction/braking force per unit mass Gamma_a(t)*Xi(t) (Liu et al. 2025 Eqs. 2-3 & Eq. 9)."""
        if u >= 0.0:
            return self.f_a_max_unit if v <= 14.0 else self.f_a_max_unit * (14.0 / (v if v > 14.0 else 14.0))
        return self.f_b_max_unit

    def _d_gamma_a_dv(self, v: float, u: float) -> float:
        """Partial derivative d(Gamma_a)/dv for Jacobian A (Liu et al. 2025 Eq. 11)."""
        if u > 0.0 and v > 14.0:
            return -self.f_a_max_unit * 14.0 / (v * v)
        return 0.0

    def f_nonlinear(
        self, v: float, u: float, slope_iota: float = 0.0, dv_prev: float = 0.0
    ) -> float:
        """Nonlinear discrete forward step including Davis drag, slope, and longitudinal inertia (Eq. 10)."""
        u_clamped = self.u_min if u < self.u_min else (self.u_max if u > self.u_max else float(u))
        gamma_xi = self._gamma_a_xi(v, u_clamped)
        # Include static breakaway resistance a_a when throttling from rest (v=0, u>0)
        h_omega = -(self.a_a + self.b_a * v + self.c_a * (v * v)) if (v > 1e-4 or u_clamped > 1e-3) else 0.0
        accel = (u_clamped * gamma_xi + h_omega - self.g * slope_iota) / (1.0 + self.gamma)
        inertia_mom = 0.14 * dv_prev
        v_next = v + self.dt * accel + inertia_mom
        return v_next if v_next > 0.0 else 0.0

    def linearize_step(
        self, x0: float, u0: float, slope_iota: float = 0.0, dv_prev: float = 0.0
    ) -> tuple[float, float, float]:
        """First-order Taylor expansion around operating point (x0, u0) (Liu et al. 2025 Eqs. 11-12)."""
        x0_s = float(x0) if x0 > 0.0 else 0.0
        u0_s = self.u_min if u0 < self.u_min else (self.u_max if u0 > self.u_max else float(u0))

        d_gamma = self._d_gamma_a_dv(x0_s, u0_s)
        gamma_val = self._gamma_a_xi(x0_s, u0_s)

        inv_g = self.dt / (1.0 + self.gamma)
        A = 1.0 + inv_g * (u0_s * d_gamma - self.b_a - 2.0 * self.c_a * x0_s)
        B = inv_g * gamma_val

        h_omega = -(self.a_a + self.b_a * x0_s + self.c_a * (x0_s * x0_s)) if (x0_s > 1e-4 or u0_s > 1e-3) else 0.0
        accel = (u0_s * gamma_val + h_omega - self.g * slope_iota) / (1.0 + self.gamma)
        f0 = x0_s + self.dt * accel + 0.14 * dv_prev
        r_k = f0 - A * x0_s - B * u0_s
        return A, B, r_k

    # =========================================================================
    # 2. Lifted Prediction Matrices for State-Augmented Dynamics (Eq. 13)
    # =========================================================================
    def build_augmented_prediction_matrices(
        self, A: float, B: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs lifted horizon matrices F, Phi, and Psi (Liu et al. 2025 Eq. 13)
        for the actuator-augmented state z(k) = [v(k), u_act(k-1)]^T:
            v(k+1) = A*v(k) + 0.28*B*u_act(k-1) + 0.72*B*u_cmd(k) + r(k)
            u_act(k) = 0.28*u_act(k-1) + 0.72*u_cmd(k)
        Uses pre-allocated buffers and upper-triangular scalar recurrence to avoid array allocations.
        """
        F = self._F
        Phi = self._Phi
        Psi = self._Psi
        step_impulses = self._step_impulses
        Phi.fill(0.0)

        az01 = 0.28 * B
        bz0 = 0.72 * B
        bz1 = 0.72
        a00 = 1.0
        a01 = 0.0
        psi_val = 0.0
        n_c = self.n_c

        for i in range(self.n_p):
            step_impulses[i] = a00 * bz0 + a01 * bz1
            # Upper-triangular product A_pow <- A_pow @ [[A, az01], [0, 0.28]]:
            a01 = a00 * az01 + a01 * 0.28
            a00 = a00 * A
            F[i, 0] = a00
            F[i, 1] = a01

            psi_val = psi_val * A + 1.0
            Psi[i, 0] = psi_val

            j_lim = i + 1 if (i + 1) < n_c else n_c
            for j in range(j_lim):
                val = step_impulses[i - j]
                if j < n_c - 1 or i == j:
                    Phi[i, j] = val
                else:
                    Phi[i, j] += val

        return F, Phi, Psi

    # =========================================================================
    # 3. Box-Constrained Quadratic Program Solver (Liu et al. 2025 Eqs. 14-16)
    # =========================================================================
    def _solve_box_constrained_qp(
        self,
        Phi: np.ndarray,
        E_0: np.ndarray,
        q1_vec: np.ndarray,
        q2_val: float,
        q3_val: float,
    ) -> float:
        """Solves min_U ||Phi*U + E_0||^2_Q1 + ||U||^2_Q2 + ||D*U - d0||^2_Q3 s.t. u in [u_min, u_max]."""
        u_min = self.u_min
        u_max = self.u_max

        if self.n_c == 3:
            # Closed-form 3x3 symmetric Cramer/cofactor solve + scalar projected gradient (avoids LAPACK overhead)
            Phi_w = Phi * q1_vec[:, None]
            H = (Phi.T @ Phi_w) + q2_val * self.I_nc + q3_val * self.DtD
            g_vec = Phi_w.T @ E_0

            h00 = float(H[0, 0]) + 1e-8
            h01 = float(H[0, 1])
            h02 = float(H[0, 2])
            h11 = float(H[1, 1]) + 1e-8
            h12 = float(H[1, 2])
            h22 = float(H[2, 2]) + 1e-8

            g0 = float(g_vec[0, 0]) - q3_val * self.prev_u
            g1 = float(g_vec[1, 0])
            g2 = float(g_vec[2, 0])

            c00 = h11 * h22 - h12 * h12
            c01 = -(h01 * h22 - h12 * h02)
            c02 = h01 * h12 - h11 * h02
            c11 = h00 * h22 - h02 * h02
            c12 = -(h00 * h12 - h01 * h02)
            c22 = h00 * h11 - h01 * h01

            det = h00 * c00 + h01 * c01 + h02 * c02
            inv_det = -1.0 / det

            u0 = (c00 * g0 + c01 * g1 + c02 * g2) * inv_det
            u1 = (c01 * g0 + c11 * g1 + c12 * g2) * inv_det
            u2 = (c02 * g0 + c12 * g1 + c22 * g2) * inv_det

            if (u_min <= u0 <= u_max) and (u_min <= u1 <= u_max) and (u_min <= u2 <= u_max):
                return u0

            u0 = u_min if u0 < u_min else (u_max if u0 > u_max else u0)
            u1 = u_min if u1 < u_min else (u_max if u1 > u_max else u1)
            u2 = u_min if u2 < u_min else (u_max if u2 > u_max else u2)

            r0 = abs(h00) + abs(h01) + abs(h02)
            r1 = abs(h01) + abs(h11) + abs(h12)
            r2 = abs(h02) + abs(h12) + abs(h22)
            max_eig_bound = r0 if (r0 >= r1 and r0 >= r2) else (r1 if r1 >= r2 else r2)
            step_lr = 1.0 / (max_eig_bound + 1e-6)

            for _ in range(10):
                nu0 = u0 - step_lr * (h00 * u0 + h01 * u1 + h02 * u2 + g0)
                nu1 = u1 - step_lr * (h01 * u0 + h11 * u1 + h12 * u2 + g1)
                nu2 = u2 - step_lr * (h02 * u0 + h12 * u1 + h22 * u2 + g2)

                nu0 = u_min if nu0 < u_min else (u_max if nu0 > u_max else nu0)
                nu1 = u_min if nu1 < u_min else (u_max if nu1 > u_max else nu1)
                nu2 = u_min if nu2 < u_min else (u_max if nu2 > u_max else nu2)

                if max(abs(nu0 - u0), abs(nu1 - u1), abs(nu2 - u2)) < 1e-5:
                    u0 = nu0
                    break
                u0, u1, u2 = nu0, nu1, nu2
            return u0

        d_0 = np.zeros((self.n_c, 1), dtype=np.float64)
        d_0[0, 0] = self.prev_u

        Phi_w = Phi * q1_vec[:, None]
        H = (Phi.T @ Phi_w) + q2_val * self.I_nc + q3_val * self.DtD
        g = (Phi_w.T @ E_0) - q3_val * (self.D.T @ d_0)

        U_unc = -np.linalg.solve(H + 1e-8 * self.I_nc, g)
        if np.all((U_unc >= self.u_min) & (U_unc <= self.u_max)):
            return float(U_unc[0, 0])

        U = np.clip(U_unc, self.u_min, self.u_max)
        max_eig_bound = float(np.max(np.sum(np.abs(H), axis=1)))
        step_lr = 1.0 / (max_eig_bound + 1e-6)
        for _ in range(10):
            U_new = np.clip(U - step_lr * (H @ U + g), self.u_min, self.u_max)
            if float(np.max(np.abs(U_new - U))) < 1e-5:
                U = U_new
                break
            U = U_new
        return float(np.clip(U[0, 0], self.u_min, self.u_max))

    # =========================================================================
    # 4. Receding-Horizon Control Action Selection
    # =========================================================================
    def select_action(
        self,
        v_curr: float,
        v_ref_horizon: np.ndarray,
        slope_iota: float = 0.0,
        dt_step: float | None = None,
        u_act_prev: float | None = None,
        disturbance_accel: float = 0.0,
    ) -> float:
        if isinstance(v_ref_horizon, np.ndarray) and v_ref_horizon.ndim == 1 and len(v_ref_horizon) == self.n_p:
            ref = v_ref_horizon
        else:
            ref = np.asarray(v_ref_horizon, dtype=np.float64).flatten()
            if len(ref) == 0:
                ref = np.full(self.n_p, float(v_curr), dtype=np.float64)
            elif len(ref) < self.n_p:
                ref = np.pad(ref, (0, self.n_p - len(ref)), mode="edge")
            else:
                ref = ref[: self.n_p]

        if dt_step is not None:
            self.dt = float(dt_step)

        v_c = float(v_curr)
        u_act_curr = float(self.u_act if u_act_prev is None else u_act_prev)
        dv_curr = v_c - self.prev_v
        self.prev_v = v_c

        X_ref = self._X_ref
        X_ref[:, 0] = ref
        ref_0 = float(ref[0])
        ref_1 = float(ref[1]) if self.n_p > 1 else ref_0
        dv_01 = ref_1 - ref_0

        if self.adaptive_shift:
            # 1. Standstill guard: never creep before the scheduled departure step!
            if ref_0 <= 1e-3 and v_c <= 1e-3:
                self.prev_u = 0.0
                self.u_act = 0.28 * u_act_curr
                return 0.0

            # 2. Strict causal plateau guard: if step k+1 (ref[0]) is still flat,
            # clamp lookahead so MPC never anticipates an upcoming shift 1 step early!
            if abs(ref_0 - v_c) < 0.025 and abs(dv_01) > 0.025:
                X_ref[1:, 0] = ref_0
            elif abs(dv_01) < 0.025 and abs(ref_0 - v_c) < 0.15:
                X_ref[2:, 0] = ref_1

            A, B, r_k = self.linearize_step(v_c, u_act_curr, slope_iota=slope_iota, dv_prev=dv_curr)
            # Inject moving-average estimated external disturbance into MPC affine offset
            # (dissipative headwind/resistance cannot push a stopping train into reverse v < 0)
            dv_dist = self.dt * float(disturbance_accel)
            if ref_0 <= 1e-3 and dv_dist < -v_c:
                dv_dist = -v_c
            r_k += dv_dist
            F, Phi, Psi = self.build_augmented_prediction_matrices(A, B)
            E_0 = self._E0
            E_0[:, 0] = F[:, 0] * v_c + F[:, 1] * u_act_curr + Psi[:, 0] * r_k - X_ref[:, 0]

            err_0 = abs(ref_0 - v_c)
            is_shift = err_0 > 0.35 or abs(dv_01) > 0.28
            is_stopping_ramp = (float(ref.min()) <= 3.5 and dv_01 < -0.02) or (ref_0 <= 3.2)

            if is_shift:
                q1_vec = self._q1_shift
                q2_eff, q3_eff = 1e-5, 0.003
            elif is_stopping_ramp:
                q1_vec = self._q1_stop
                q2_eff, q3_eff = 1e-5, 0.015
            else:
                q1_vec = self._q1_cruise
                q2_eff, q3_eff = 1e-4, 0.04
        else:
            # Standard Fixed-Q3 MPC baseline (Liu et al. 2025)
            A, B, r_k = self.linearize_step(v_c, u_act_curr, slope_iota=slope_iota, dv_prev=0.0)
            F, Phi, Psi = self.build_augmented_prediction_matrices(A, B)
            E_0 = self._E0
            E_0[:, 0] = F[:, 0] * v_c + F[:, 1] * u_act_curr + Psi[:, 0] * r_k - X_ref[:, 0]
            q1_vec = self._q1_fixed
            q2_eff, q3_eff = self.q2, self.q3

        u_opt = self._solve_box_constrained_qp(Phi, E_0, q1_vec, q2_eff, q3_eff)

        self.prev_u = u_opt
        self.u_act = 0.72 * u_opt + 0.28 * u_act_curr
        return u_opt