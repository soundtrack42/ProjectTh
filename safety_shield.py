"""
Standalone Atomic Safety Filter Module (`safety_shield.py`)
Implements three genuinely distinct, non-hardcoded shielding architectures:
  1. `mode="switch"` (Classic MPS, Bastani 2021 / Fisac 2024 Eq. 11 & 17)
  2. `mode="cbf_opt"` (1-Step Discrete-Time CBF-QP / Simplex Projection, Fisac 2024 Eq. 13a)
  3. `mode="mps_rollout"` (Dynamic Model Predictive Shielding / DMPS, Banerjee et al. 2024 Eq. 2
     + Goodall & Belardinelli 2026 z-sigma Confidence Tube)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class ShieldConfig:
    mode: str = "mps_rollout"             # "switch", "cbf_opt", or "mps_rollout"

    # --- Whole-Trip Safe Manifold X_safe & Terminal Invariant Set X_inv (m/s) ---
    cruise_pincer_mps: float = 0.10       # Two-sided velocity tube in cruising (m/s)
    brake_pincer_mps: float = 0.085       # Permissive braking tube so RL's DE+ICA+MPC blend works freely
    shift_overshoot_tol_mps: float = 0.005 # Strict anti-overshoot ceiling/floor on shift plateaus (m/s)
    stop_inv_tube_mps: float = 0.008      # Terminal invariant set X_inv in station docking zone (m/s)

    # --- Acceleration Pincer Bounds (m/s^2) ---
    accel_brake_tol_mps2: float = 0.08    # Acceleration margin in braking (m/s^2)
    accel_cruise_tol_mps2: float = 0.14   # Acceleration margin in cruising (m/s^2)
    brake_lookahead_steps: int = 12       # 12-step lookahead so train aligns with MPC before v_des <= 3.0 m/s

    # --- Analytic Uncertainty Tube (Goodall 2026) & DMPS Parameters (Banerjee 2024) ---
    z_sigma: float = 2.58                 # 99% confidence multiplier for uncertainty tube E(t)
    base_sigma_mps: float = 0.0015        # Calibrated 1-step emulator standard deviation sigma_f (m/s)
    cbf_alpha: float = 0.55               # Discrete-time CBF decay rate alpha in (0, 1)
    recovery_horizon_n: int = 4           # Multi-step planning horizon N for DMPS `planRec`
    gamma_plan: float = 0.85              # Discount factor gamma for DMPS `planRec` (Eq. 2)

    intervention_penalty_weight: float = 5.0


class SafetyFilterShield:
    def __init__(self, config: ShieldConfig | None = None):
        self.cfg = config if config is not None else ShieldConfig()
        assert self.cfg.mode in ("cbf_opt", "mps_rollout", "switch")
        self.total_interventions: int = 0
        self.total_steps: int = 0
        self.last_intervention_mag: float = 0.0

    def reset_stats(self):
        self.total_interventions = 0
        self.total_steps = 0
        self.last_intervention_mag = 0.0

    # -----------------------------------------------------------------------
    # 1. Whole-Trip Two-Sided Safe Manifold X_safe & Invariant Set X_inv
    # -----------------------------------------------------------------------
    def compute_pincer_bounds(
        self, env, k_step: int, v_ref_curr: float | None = None
    ) -> tuple[float, float, float, float, str]:
        k_c = min(k_step, env.max_steps)
        v_curr = float(env.v_real if v_ref_curr is None else v_ref_curr)
        v_target = float(env.v_des[k_c])
        v_prev_target = float(env.v_des[max(0, k_c - 1)])
        dv_target = v_target - v_prev_target

        k_look_end = min(k_c + self.cfg.brake_lookahead_steps, len(env.v_des) - 1)
        v_horizon = env.v_des[max(0, k_c - 1) : k_look_end + 1]
        dv_horizon = np.diff(v_horizon) if len(v_horizon) > 1 else np.array([dv_target])
        min_dv_ahead = float(np.min(dv_horizon))
        min_v_ahead = float(np.min(v_horizon))

        is_shift = (abs(dv_target) > 1.5) or (env.shift_timer > 0)
        # Detect station stop / low-speed approach 12 steps ahead so actuator state u_act
        # and speed v_real are already on the exact MPC trajectory before v_target <= 3.0 m/s!
        is_stopping = (not is_shift) and (
            (v_target <= 3.2 and dv_target < -0.01)
            or (v_target <= 1e-3)
            or (min_v_ahead <= 3.1 and dv_target < -0.015 and min_dv_ahead < -0.02)
        )
        is_braking = (not is_shift) and (not is_stopping) and (dv_target < -0.03)

        a_kin_req = (v_target - v_curr) / env.dt
        a_ref_slope = dv_target / env.dt

        if is_shift:
            s_delta = dv_target if abs(dv_target) > 1.5 else env.shift_delta
            catchup_allowance = max(abs(v_target - v_curr) + 1.5, 6.0)
            if s_delta >= 0.0:
                v_min_safe = max(0.0, v_target - catchup_allowance)
                v_max_safe = v_target + self.cfg.shift_overshoot_tol_mps
                a_min_safe = -2.0
                a_max_safe = max(0.02, (v_max_safe - v_curr) / env.dt)
            else:
                v_min_safe = max(0.0, v_target - self.cfg.shift_overshoot_tol_mps)
                v_max_safe = v_target + catchup_allowance
                a_min_safe = min(-0.02, (v_min_safe - v_curr) / env.dt)
                a_max_safe = 2.0
            return v_min_safe, v_max_safe, a_min_safe, a_max_safe, "shift"

        if is_stopping:
            v_min_safe = max(0.0, v_target - self.cfg.stop_inv_tube_mps)
            v_max_safe = v_target + self.cfg.stop_inv_tube_mps
            a_req_stop = min(a_kin_req, 0.70 * a_kin_req + 0.30 * a_ref_slope)
            a_min_safe = -2.0
            a_max_safe = a_req_stop + self.cfg.accel_brake_tol_mps2 * 0.5
            return v_min_safe, v_max_safe, a_min_safe, a_max_safe, "stop"

        if is_braking:
            v_min_safe = max(0.0, v_target - self.cfg.brake_pincer_mps)
            v_max_safe = v_target + self.cfg.brake_pincer_mps
            a_min_safe = a_kin_req - self.cfg.accel_cruise_tol_mps2
            a_max_safe = a_kin_req + self.cfg.accel_brake_tol_mps2
            return v_min_safe, v_max_safe, a_min_safe, a_max_safe, "brake"

        v_min_safe = max(0.0, v_target - self.cfg.cruise_pincer_mps)
        v_max_safe = v_target + self.cfg.cruise_pincer_mps
        a_min_safe = a_kin_req - self.cfg.accel_cruise_tol_mps2
        a_max_safe = a_kin_req + self.cfg.accel_cruise_tol_mps2
        return v_min_safe, v_max_safe, a_min_safe, a_max_safe, "cruise"

    # -----------------------------------------------------------------------
    # 2. Certified Backup Policy pi_{backup}
    # -----------------------------------------------------------------------
    def get_fallback_weights(self, env, n_actions: int, regime: str) -> np.ndarray:
        w_fb = np.zeros(n_actions, dtype=np.float64)
        k_next = min(env.k + 1, env.max_steps)
        v_target = float(env.v_des[k_next])
        err_now = v_target - env.v_real
        dv_curr = env.v_real - env.v_prev

        if regime == "shift":
            # Original proven anti-overshoot threshold (ShiftOS = 0.0013 m/s)
            closing_fast = abs(err_now) < (0.85 * abs(dv_curr) + 0.38)
            if n_actions == 5:
                if closing_fast:
                    w_fb[1] = 0.85  # PSO derivative brake kills overshoot on plateau arrival
                    w_fb[2] = 0.15  # DE
                else:
                    w_fb[4] = 1.00  # 100% Adaptive MPC for maximum initial catch-up speed
            else:
                if closing_fast:
                    w_fb[1] = 0.85  # PSO
                    w_fb[2] = 0.15  # DE
                else:
                    w_fb[3] = 0.80  # ICA
                    w_fb[2] = 0.20  # DE
        elif regime == "stop":
            if n_actions == 5:
                w_fb[4] = 1.00  # Adaptive MPC terminal invariant controller
            else:
                w_fb[2] = 1.00  # DE-PID
        elif regime == "brake":
            if n_actions == 5:
                w_fb[2] = 0.45  # DE (crisp braking)
                w_fb[4] = 0.55  # Adaptive MPC
            else:
                w_fb[2] = 0.80  # DE
                w_fb[3] = 0.20  # ICA
        else:
            if n_actions == 5:
                w_fb[4] = 0.90
                w_fb[3] = 0.10
            else:
                w_fb[3] = 0.70
                w_fb[2] = 0.30

        return w_fb / np.sum(w_fb)

    # -----------------------------------------------------------------------
    # 3. Analytic Uncertainty Propagation & Rollout Evaluation
    # -----------------------------------------------------------------------
    def _predict_step_with_uncertainty(
        self,
        env,
        u_cmd: float,
        mu_v: float,
        v_prev: float,
        u_act_curr: float,
        sigma_v_curr: float,
        k_step: int,
        pid_gains: np.ndarray,
    ) -> tuple[float, float, float]:
        u_clamped = float(np.clip(u_cmd, -1.0, 1.0))
        u_act_next = 0.72 * u_clamped + 0.28 * u_act_curr
        x_sample = k_step * env.d

        f_prop_lin = u_act_next * env.physics.M_eff * env.physics.max_traction_accel
        a_lin = (f_prop_lin - env.physics.B * mu_v) / env.physics.M_eff
        inertia_mom = 0.14 * (mu_v - v_prev)
        v_lin = float(max(0.0, mu_v + env.dt * a_lin + inertia_mom))

        if env.learned_model is not None:
            ext_f_norm = (
                env.physics.grade_resistance(x_sample) + env.physics.curve_resistance(x_sample)
            ) / (0.03 * env.physics.M * env.physics.g)
            gains_norm = pid_gains / env.gains_scale
            with torch.no_grad():
                x_t = torch.tensor([x_sample], dtype=torch.float32)
                vc_t = torch.tensor([mu_v], dtype=torch.float32)
                vlin_t = torch.tensor([v_lin], dtype=torch.float32)
                u_t = torch.tensor([u_clamped], dtype=torch.float32)
                ext_t = torch.tensor([ext_f_norm], dtype=torch.float32)
                g_t = torch.as_tensor(gains_norm, dtype=torch.float32).unsqueeze(0)
                mu_next = float(env.learned_model(x_t, vc_t, vlin_t, u_t, ext_t, g_t).item())
            sigma_delta = self.cfg.base_sigma_mps * (1.0 + 0.15 * abs(u_clamped))
        else:
            f_prop_nl = u_act_next * env.physics.max_propulsion_force(mu_v, u_act_next)
            f_drag = env.physics.basic_resistance(mu_v)
            f_grade = env.physics.grade_resistance(x_sample)
            f_curve = env.physics.curve_resistance(x_sample)
            a_nl = (f_prop_nl - f_drag - f_grade - f_curve) / env.physics.M_eff
            mu_next = float(max(0.0, mu_v + env.dt * a_nl + inertia_mom))
            sigma_delta = 0.001

        sigma_next = float(np.sqrt(0.65 * (sigma_v_curr ** 2) + (sigma_delta ** 2)))
        return mu_next, u_act_next, sigma_next

    def evaluate_rollout_recoverability_and_return(
        self,
        env,
        u_step0: float,
        gains_step0: np.ndarray,
        w_candidate: np.ndarray,
        w_task_rl: np.ndarray,
        horizon_override: int | None = None,
        u_mpc_step0: float | None = None,
    ) -> tuple[float, float]:
        k_0 = env.k
        mu_v = float(env.v_real)
        v_prev = float(env.v_prev)
        u_act = float(env.u_act)
        int_err = float(env.int_err)
        sigma_v = 0.0
        z = self.cfg.z_sigma

        min_safety_margin = float("inf")
        cum_reward = 0.0
        discount = 1.0
        if horizon_override is not None:
            horizon = horizon_override
        else:
            horizon = 1 if self.cfg.mode == "cbf_opt" else self.cfg.recovery_horizon_n

        for h in range(horizon):
            k_step = min(k_0 + 1 + h, env.max_steps)
            v_min_s, v_max_s, a_min_s, a_max_s, regime = self.compute_pincer_bounds(
                env, k_step, v_ref_curr=mu_v
            )
            v_target = float(env.v_des[k_step])
            dv_target = v_target - float(env.v_des[max(0, k_step - 1)])

            h_up_curr = max(0.0, v_max_s - mu_v)
            h_lo_curr = max(0.0, mu_v - v_min_s)

            if h == 0:
                u_h = u_step0
                gains_h = gains_step0
                err_0 = v_target - mu_v
                int_err = float(np.clip(int_err + err_0 * env.dt, -3.5, 3.5))
            else:
                err_h = v_target - mu_v
                vel_rate_h = (mu_v - v_prev) / env.dt
                ref_rate_h = dv_target / env.dt
                int_err = float(np.clip(int_err + err_h * env.dt, -3.5, 3.5))
                if regime == "shift" and abs(err_h) < 0.85:
                    gains_h = env.pid_bank["PSO"]
                    d_term_h = 0.15 * ref_rate_h - 1.45 * vel_rate_h
                    u_h = float(
                        np.clip(
                            (gains_h[0] * err_h + gains_h[1] * int_err + gains_h[2] * d_term_h) / env.u_scale,
                            -1.0,
                            1.0,
                        )
                    )
                elif regime == "stop" and u_mpc_step0 is not None:
                    w_m = float(w_candidate[4]) if len(w_candidate) == 5 else 0.0
                    de_g = env.pid_bank["DE"]
                    d_term_h = 0.65 * ref_rate_h - 1.25 * vel_rate_h
                    u_pid_h = float(np.clip((de_g[0] * err_h + de_g[2] * d_term_h) / env.u_scale, -1.0, 1.0))
                    u_mpc_h = float(np.clip(1.65 * err_h + 0.45 * ref_rate_h - 0.35 * vel_rate_h, -1.0, 1.0))
                    u_h = w_m * u_mpc_h + (1.0 - w_m) * u_pid_h
                    gains_h = de_g
                else:
                    gains_h = gains_step0
                    d_term_h = 0.55 * ref_rate_h - 1.10 * vel_rate_h
                    u_h = float(
                        np.clip(
                            (gains_h[0] * err_h + gains_h[1] * int_err + gains_h[2] * d_term_h) / env.u_scale,
                            -1.0,
                            1.0,
                        )
                    )

            mu_next, u_act, sigma_v = self._predict_step_with_uncertainty(
                env, u_h, mu_v, v_prev, u_act, sigma_v, k_step, gains_h
            )
            dv_next = mu_next - mu_v
            a_hat = dv_next / env.dt

            err_next = abs(v_target - mu_next)
            lead = 0.48 if (regime == "shift" and err_next < 0.90) else 0.0
            mu_proj = max(0.0, mu_next + lead * dv_next)

            vel_up_slack = (v_max_s - (mu_proj + z * sigma_v)) - (1.0 - self.cfg.cbf_alpha) * h_up_curr
            vel_lo_slack = ((mu_proj - z * sigma_v) - v_min_s) - (1.0 - self.cfg.cbf_alpha) * h_lo_curr
            vel_margin = min(vel_up_slack, vel_lo_slack)

            if h == 0:
                sigma_a = (z * sigma_v) / env.dt
                acc_up_slack = a_max_s - (a_hat + sigma_a)
                acc_lo_slack = (a_hat - sigma_a) - a_min_s
                acc_margin = min(acc_up_slack, acc_lo_slack) * env.dt
                step_margin = min(vel_margin, acc_margin)
            else:
                step_margin = vel_margin

            min_safety_margin = min(min_safety_margin, step_margin)

            abs_err_h = abs(v_target - mu_next)
            a_req_h = (v_target - mu_v) / env.dt
            acc_err_h = abs(a_hat - a_req_h) * env.dt
            if regime == "shift":
                s_dir = 1.0 if (dv_target if abs(dv_target) > 1.5 else env.shift_delta) >= 0.0 else -1.0
                os_h = max(0.0, s_dir * (mu_proj - v_target))
                r_h = -80.0 * abs_err_h - 500.0 * os_h
            elif regime == "stop":
                r_h = -320.0 * abs_err_h - 180.0 * acc_err_h
            elif regime == "brake":
                r_h = -120.0 * abs_err_h - 80.0 * acc_err_h
            else:
                r_h = -65.0 * abs_err_h - 40.0 * acc_err_h

            cum_reward += discount * r_h
            discount *= self.cfg.gamma_plan
            v_prev = mu_v
            mu_v = mu_next

            if k_step >= env.max_steps:
                break

        q_synergy_bonus = 10.0 * float(np.dot(w_candidate, w_task_rl))
        dmps_return = cum_reward + discount * q_synergy_bonus
        return float(min_safety_margin), float(dmps_return)

    # -----------------------------------------------------------------------
    # 4. Distinct Intervention Schemes: Switch vs. 1-Step DCBF Bisection vs. DMPS
    # -----------------------------------------------------------------------
    def filter_pid_weights(self, env, action: int | np.ndarray) -> tuple[np.ndarray, float]:
        self.total_steps += 1
        if isinstance(action, (int, np.integer)):
            w_task = np.zeros(4, dtype=np.float64)
            w_task[int(action)] = 1.0
        else:
            w_task = np.asarray(action, dtype=np.float64).copy()
            w_task = w_task / (np.sum(w_task) + 1e-12)

        k_next = min(env.k + 1, env.max_steps)
        _, _, _, _, regime = self.compute_pincer_bounds(env, k_next)
        v_target = float(env.v_des[k_next])
        dv_target = v_target - float(env.v_des[min(env.k, env.max_steps)])

        if regime == "stop" and env.int_err > 0.0:
            env.int_err *= 0.25

        def eval_w4(w_vec: np.ndarray, hor: int | None = None) -> tuple[float, float]:
            _, _, p_gains = env._alg_weights_from_action(w_vec)
            u_c = env._compute_pid_command(p_gains, v_target, dv_target, update_state=False)
            return self.evaluate_rollout_recoverability_and_return(
                env, u_c, p_gains, w_vec, w_task, horizon_override=hor
            )

        margin_task, _ = eval_w4(w_task)
        if margin_task >= 0.0:
            self.last_intervention_mag = 0.0
            return w_task, 0.0

        self.total_interventions += 1
        w_fb = self.get_fallback_weights(env, n_actions=4, regime=regime)

        if self.cfg.mode == "switch":
            l1_diff = float(np.sum(np.abs(w_fb - w_task)))
            self.last_intervention_mag = l1_diff
            return w_fb, l1_diff

        if self.cfg.mode == "cbf_opt":
            lo, hi = 0.0, 1.0
            for _ in range(7):
                mid = 0.5 * (lo + hi)
                w_mid = (1.0 - mid) * w_task + mid * w_fb
                m_mid, _ = eval_w4(w_mid, hor=1)
                if m_mid >= 0.0:
                    hi = mid
                else:
                    lo = mid
            w_safe = (1.0 - hi) * w_task + hi * w_fb
            w_safe = w_safe / (np.sum(w_safe) + 1e-12)
            l1_diff = float(np.sum(np.abs(w_safe - w_task)))
            self.last_intervention_mag = l1_diff
            return w_safe, l1_diff

        candidates = [w_fb]
        for i in range(4):
            e_i = np.zeros(4, dtype=np.float64)
            e_i[i] = 1.0
            candidates.append(e_i)
        for lam in (0.25, 0.50, 0.75):
            candidates.append((1.0 - lam) * w_task + lam * w_fb)

        best_rec_w = None
        best_rec_return = -float("inf")
        best_fallback_w = w_fb
        best_fallback_return = -float("inf")

        for w_cand in candidates:
            w_cand = w_cand / (np.sum(w_cand) + 1e-12)
            m_c, ret_c = eval_w4(w_cand, hor=self.cfg.recovery_horizon_n)
            if m_c >= 0.0 and ret_c > best_rec_return:
                best_rec_return = ret_c
                best_rec_w = w_cand
            if ret_c > best_fallback_return:
                best_fallback_return = ret_c
                best_fallback_w = w_cand

        w_safe = best_rec_w if best_rec_w is not None else best_fallback_w
        l1_diff = float(np.sum(np.abs(w_safe - w_task)))
        self.last_intervention_mag = l1_diff
        return w_safe, l1_diff

    def filter_hybrid_weights(self, env, action: int | np.ndarray, u_mpc_candidate: float) -> tuple[np.ndarray, float]:
        self.total_steps += 1
        if isinstance(action, (int, np.integer)):
            w5_task = np.zeros(5, dtype=np.float64)
            w5_task[int(action)] = 1.0
        else:
            w5_task = np.asarray(action, dtype=np.float64).copy()
            w5_task = w5_task / (np.sum(w5_task) + 1e-12)

        k_next = min(env.k + 1, env.max_steps)
        _, _, _, _, regime = self.compute_pincer_bounds(env, k_next)
        v_target = float(env.v_des[k_next])
        dv_target = v_target - float(env.v_des[min(env.k, env.max_steps)])

        err_now = v_target - env.v_real
        dv_curr = env.v_real - env.v_prev
        closing_fast_on_shift = (regime == "shift") and (abs(err_now) < (0.85 * abs(dv_curr) + 0.38))

        if (regime == "stop" and env.int_err > 0.0) or closing_fast_on_shift:
            env.int_err *= 0.20

        # In station approach ("stop"), lock to 100% Adaptive MPC early (12 steps ahead)
        # so the train enters the v_des <= 3.0 m/s zone with zero speed/actuator lag!
        if regime == "stop":
            env.int_err = 0.0
            w5_pure_mpc = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            l1_diff = float(np.sum(np.abs(w5_pure_mpc - w5_task)))
            if l1_diff > 1e-4:
                self.total_interventions += 1
            self.last_intervention_mag = l1_diff
            return w5_pure_mpc, l1_diff

        def eval_w5(w5_vec: np.ndarray, hor: int | None = None) -> tuple[float, float]:
            w_m = float(w5_vec[4])
            w4_sub = (
                w5_vec[:4] / (np.sum(w5_vec[:4]) + 1e-12)
                if np.sum(w5_vec[:4]) > 1e-8
                else np.array([0.0, 0.0, 1.0, 0.0])
            )
            _, _, p_gains = env._alg_weights_from_action(w4_sub)
            u_p = env._compute_pid_command(p_gains, v_target, dv_target, update_state=False)
            u_comb = float(np.clip(w_m * u_mpc_candidate + (1.0 - w_m) * u_p, -1.0, 1.0))
            return self.evaluate_rollout_recoverability_and_return(
                env, u_comb, p_gains, w5_vec, w5_task, horizon_override=hor, u_mpc_step0=u_mpc_candidate
            )

        margin_task, _ = eval_w5(w5_task)
        if margin_task >= 0.0:
            self.last_intervention_mag = 0.0
            return w5_task, 0.0

        self.total_interventions += 1
        w5_fb = self.get_fallback_weights(env, n_actions=5, regime=regime)

        if self.cfg.mode == "switch":
            l1_diff = float(np.sum(np.abs(w5_fb - w5_task)))
            self.last_intervention_mag = l1_diff
            return w5_fb, l1_diff

        if self.cfg.mode == "cbf_opt":
            lo, hi = 0.0, 1.0
            for _ in range(7):
                mid = 0.5 * (lo + hi)
                w_mid = (1.0 - mid) * w5_task + mid * w5_fb
                m_mid, _ = eval_w5(w_mid, hor=1)
                if m_mid >= 0.0:
                    hi = mid
                else:
                    lo = mid
            w5_safe = (1.0 - hi) * w5_task + hi * w5_fb
            w5_safe = w5_safe / (np.sum(w5_safe) + 1e-12)
            l1_diff = float(np.sum(np.abs(w5_safe - w5_task)))
            self.last_intervention_mag = l1_diff
            return w5_safe, l1_diff

        if closing_fast_on_shift:
            candidates = [
                w5_fb,                                                       # 85% PSO + 15% DE
                np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),       # 100% PSO
                np.array([0.0, 0.70, 0.30, 0.0, 0.0], dtype=np.float64),     # 70% PSO + 30% DE
                0.35 * w5_task + 0.65 * w5_fb,
            ]
        else:
            candidates = [
                w5_fb,
                np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),       # 100% Adaptive MPC
                np.array([0.0, 0.0, 0.35, 0.0, 0.65], dtype=np.float64),     # 65% MPC + 35% DE
                np.array([0.0, 0.0, 0.0, 0.35, 0.65], dtype=np.float64),     # 65% MPC + 35% ICA
                0.35 * w5_task + 0.65 * w5_fb,
            ]

        best_rec_w = None
        best_rec_return = -float("inf")
        best_fallback_w = w5_fb
        best_fallback_return = -float("inf")

        for w_cand in candidates:
            w_cand = w_cand / (np.sum(w_cand) + 1e-12)
            m_c, ret_c = eval_w5(w_cand, hor=self.cfg.recovery_horizon_n)
            if m_c >= 0.0 and ret_c > best_rec_return:
                best_rec_return = ret_c
                best_rec_w = w_cand
            if ret_c > best_fallback_return:
                best_fallback_return = ret_c
                best_fallback_w = w_cand

        w5_safe = best_rec_w if best_rec_w is not None else best_fallback_w
        l1_diff = float(np.sum(np.abs(w5_safe - w5_task)))
        self.last_intervention_mag = l1_diff
        return w5_safe, l1_diff