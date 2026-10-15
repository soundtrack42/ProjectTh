"""
Pure Model Predictive Control (MPC) for Train Speed Tracking (Unshielded)
Strictly implements Section 2.6 (Eq. 11-12) and Section 3 (Eq. 13-17) of:
Dongdong Liu et al. (2025), "MPC-Based Speed Tracking Control for Subway Trains
Under Actuator Constraints", Actuators 14, 134.
"""
from __future__ import annotations

import numpy as np


class PureLiuMPC:
    """
    Implements the exact analytical Receding-Horizon MPC from Liu et al. (2025):
      1. Taylor linearization around operating point (x_0(k), u_0(k)) -> A, B, r(k) (Eq. 11-12)
      2. Multi-step prediction matrices F, Phi, and residual propagation Psi over
         prediction horizon N_p and control horizon N_c (Eq. 13)
      3. Quadratic Cost minimization J = J_1 + J_2 + J_3 (Eq. 14-16):
           J_1 = sum_{i=1}^{N_p} Q_1 ||x(k+i|k) - x_ref(k+i)||^2
           J_2 = sum_{i=0}^{N_c-1} Q_2 ||u(k+i|k)||^2
           J_3 = sum_{i=0}^{N_c-1} Q_3 ||u(k+i|k) - u(k+i-1|k)||^2
      4. Projected box-constraint solver for actuator limits u_min <= u <= u_max (Eq. 17)
         without external RL shielding.
    """

    def __init__(
        self,
        n_p: int = 8,
        n_c: int = 4,
        q1: float = 10.0,
        q2: float = 2.0,
        q3: float = 5.0,
        u_min: float = -1.0,
        u_max: float = 1.0,
        dt: float = 0.85,
        mass_kg: float = 85000.0,
        gamma_rot: float = 0.08,
        davis_A: float = 14.5,
        davis_B: float = 185.0,
        davis_C: float = 1450.0,
        max_traction_accel: float = 1.55,
        max_braking_decel: float = 1.35,
    ):
        self.n_p = n_p
        self.n_c = min(n_c, n_p)
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.u_min = u_min
        self.u_max = u_max
        self.dt = dt
        self.M = mass_kg
        self.gamma = gamma_rot

        # Resistance parameters per unit mass H_a = [a_a, b_a, c_a] (Eq. 5 & Eq. 9)
        self.a_a = davis_C / self.M
        self.b_a = davis_B / self.M
        self.c_a = davis_A / self.M

        self.max_accel = max_traction_accel
        self.max_decel = max_braking_decel
        self.g = 9.81

        self.prev_u: float = 0.0

        # Difference matrix D for J_3 = (D U - d_0)^T Q_3 (D U - d_0) (Eq. 16)
        self.D = np.eye(self.n_c, dtype=np.float64)
        for i in range(1, self.n_c):
            self.D[i, i - 1] = -1.0

    def reset(self):
        self.prev_u = 0.0

    def _gamma_a(self, v: float, u: float) -> float:
        """
        Maximum traction/braking force per unit mass Gamma_a(k) * Xi(k) (Eq. 3 & Eq. 9).
        """
        if u >= 0.0:
            if v <= 12.0:
                return self.max_accel
            return self.max_accel * (12.0 / max(v, 12.0))
        else:
            return self.max_decel

    def _d_gamma_a_dv(self, v: float, u: float) -> float:
        """Derivative d(Gamma_a)/dx at operating point v_0 (Eq. 11)."""
        if u > 0.0 and v > 12.0:
            return -self.max_accel * 12.0 / (v ** 2)
        return 0.0

    def f_nonlinear(self, v: float, u: float, slope_sin: float = 0.0, curve_factor: float = 0.0) -> float:
        """
        Discrete nonlinear state transition f(x(k), u(k)) from Liu Eq. (10)
        combined with Havaei's grade and curve resistance (Eq. 6-7):
        """
        gamma_u = self._gamma_a(v, u)
        trac_brake_accel = u * gamma_u
        basic_res_accel = (self.a_a + self.b_a * v + self.c_a * (v ** 2)) if v > 1e-4 else 0.0
        ext_res_accel = self.g * (slope_sin + curve_factor)

        net_accel = (trac_brake_accel - basic_res_accel - ext_res_accel) / (1.0 + self.gamma)
        return float(max(0.0, v + self.dt * net_accel))

    def linearize_step(self, v0: float, u0: float, slope_sin: float = 0.0, curve_factor: float = 0.0):
        """
        Implements Step 2 & Step 3 of Section 2.6 in Liu et al. (Eq. 11 & Eq. 12):
          x(k+1) = A x(k) + B u(k) + r(k)
        where:
          A = df/dx |_(x0, u0)
          B = df/du |_(x0, u0)
          r(k) = f(x0, u0) - A x0 - B u0
        """
        v0_safe = max(0.0, float(v0))
        u0_safe = float(np.clip(u0, self.u_min, self.u_max))

        d_gamma = self._d_gamma_a_dv(v0_safe, u0_safe)
        gamma_val = self._gamma_a(v0_safe, u0_safe)

        # Note: resistance opposes motion, so its derivative w.r.t velocity v is -(b_a + 2 c_a v0)
        A = 1.0 + (self.dt / (1.0 + self.gamma)) * (u0_safe * d_gamma - self.b_a - 2.0 * self.c_a * v0_safe)
        B = (self.dt / (1.0 + self.gamma)) * gamma_val

        f0 = self.f_nonlinear(v0_safe, u0_safe, slope_sin, curve_factor)
        r_k = f0 - A * v0_safe - B * u0_safe
        return float(A), float(B), float(r_k)

    def build_prediction_matrices(self, A: float, B: float):
        """
        Implements Equation (13) of Liu et al. (2025):
          X(k+1) = F x(k) + Phi U(k) + Psi r(k)
        where F is (N_p x 1), Phi is (N_p x N_c), and Psi accumulates the affine
        linearization offset r(k) across the N_p prediction steps.
        """
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
                    # Beyond control horizon N_c, control input is held constant u(k+N_c-1)
                    Phi[i, j] += (A ** (i - j)) * B

        return F, Phi, Psi

    def solve_mpc_control(
        self,
        v_curr: float,
        x_ref_horizon: np.ndarray,
        slope_sin: float = 0.0,
        curve_factor: float = 0.0,
    ) -> tuple[float, np.ndarray]:
        """
        Solves the MPC optimization problem in Eq. (20) of Liu et al. (2025):
          min_{U(k)} J_1 + J_2 + J_3   s.t. u_min <= u(k+i|k) <= u_max (Eq. 17)
        Returns (optimal first control input u*(k|k), predicted speed trajectory X(k+1)).
        """
        # Pad reference horizon if near end of trajectory
        ref = np.asarray(x_ref_horizon, dtype=np.float64).flatten()
        if len(ref) < self.n_p:
            ref = np.pad(ref, (0, self.n_p - len(ref)), mode="edge")
        else:
            ref = ref[: self.n_p]
        X_ref = ref.reshape(self.n_p, 1)

        # 1. Linearize around current operating point (v_curr, prev_u) (Eq. 11-12)
        A, B, r_k = self.linearize_step(v_curr, self.prev_u, slope_sin, curve_factor)

        # 2. Build F, Phi, Psi matrices (Eq. 13)
        F, Phi, Psi = self.build_prediction_matrices(A, B)

        # Free response error E_0 = F x(k) + Psi r(k) - X_ref
        E_0 = F * float(v_curr) + Psi * r_k - X_ref

        # 3. Assemble Quadratic Program H U = -g for J_1 + J_2 + J_3 (Eq. 14-16)
        # J(U) = (Phi U + E_0)^T Q_1 (Phi U + E_0) + U^T Q_2 U + (D U - d_0)^T Q_3 (D U - d_0)
        d_0 = np.zeros((self.n_c, 1), dtype=np.float64)
        d_0[0, 0] = self.prev_u

        H = (
            self.q1 * (Phi.T @ Phi)
            + self.q2 * np.eye(self.n_c)
            + self.q3 * (self.D.T @ self.D)
        )
        g = self.q1 * (Phi.T @ E_0) - self.q3 * (self.D.T @ d_0)

        # 4. Solve unconstrained QP and refine with fast projected gradient / active-set
        # to strictly enforce actuator constraints u_min <= U <= u_max (Eq. 17)
        U = -np.linalg.solve(H + 1e-8 * np.eye(self.n_c), g)
        U = np.clip(U, self.u_min, self.u_max)

        # Projected Newton/Gradient polish for box-constrained QP
        eig_max = float(np.max(np.linalg.eigvalsh(H)))
        step_lr = 1.0 / (eig_max + 1e-6)
        for _ in range(15):
            grad = H @ U + g
            U_new = np.clip(U - step_lr * grad, self.u_min, self.u_max)
            if float(np.max(np.abs(U_new - U))) < 1e-6:
                U = U_new
                break
            U = U_new

        u_opt = float(U[0, 0])
        self.prev_u = u_opt
        X_pred = (F * float(v_curr) + Phi @ U + Psi * r_k).flatten()
        return u_opt, X_pred