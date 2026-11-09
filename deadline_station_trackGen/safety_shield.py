"""
Standalone Atomic Safety Filter & Scissor Band Module (`safety_shield.py`).

References & Shielding Architectures:
  1. `mode="switch"`: Classic Model Predictive Shielding (MPS) (Bastani, 2021;
     Hsu, Hu, & Fisac, 2024, "The Safety Filter: A Unified View of Safety-Critical Control
     in Autonomous Systems," Annual Review of Control, Robotics, and Autonomous Systems, Eqs. 11 & 17).
  2. `mode="cbf_opt"`: 1-Step Discrete-Time Control Barrier Function (DCBF) Simplex Bisection
     (Hsu, Hu, & Fisac, 2024, Eq. 13a; Agrawal & Sreenath, 2017).
  3. `mode="mps_rollout"`: Dynamic Model Predictive Shielding (DMPS) with Multi-Step Recovery
     Planning `planRec` (Banerjee et al., 2024, Eq. 2) and Heteroscedastic Neural NLL z-sigma
     Confidence Tubes (Goodall & Belardinelli, 2026 / `sacktock/A2C-GP-Shield`, Sec. 4;
     Wang et al., 2023; Wabersich & Zeilinger, 2018).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class ShieldConfig:
    mode: str = "mps_rollout"              # "switch", "cbf_opt", or "mps_rollout"

    # --- Emulated-Trajectory Safe Manifold X_safe & Terminal Invariant Set X_inv (m/s) ---
    cruise_pincer_mps: float = 0.14        # Two-sided velocity tube around v_emu in cruising (m/s)
    brake_pincer_mps: float = 0.16         # Two-sided velocity tube around v_emu in braking (m/s)
    shift_overshoot_tol_mps: float = 0.005 # Strict anti-overshoot ceiling/floor on shift plateaus (m/s)
    stop_inv_tube_mps: float = 0.015       # Terminal invariant set X_inv in station docking zone (m/s)

    # --- Acceleration Pincer Bounds (m/s^2) ---
    accel_brake_tol_mps2: float = 0.35
    accel_cruise_tol_mps2: float = 0.30
    brake_lookahead_steps: int = 8

    # --- Heteroscedastic Uncertainty Tube (Goodall & Belardinelli 2026) & DMPS (Banerjee et al. 2024) ---
    z_sigma: float = 2.576                 # Exact 99% Gaussian quantile Phi^{-1}(0.995) = 2.576
    base_sigma_mps: float = 0.0045         # Baseline floor standard deviation sigma_0 (m/s)
    gamma_sigma: float = 0.65              # Contraction factor for multi-step variance recursion
    cbf_alpha: float = 0.55                # Discrete-time CBF decay rate alpha in (0, 1) (Fisac 2024)
    recovery_horizon_n: int = 3            # Multi-step planning horizon N for DMPS `planRec` (Banerjee 2024)
    gamma_plan: float = 0.85               # Discount factor gamma for DMPS `planRec` (Banerjee 2024 Eq. 2)
    intervention_penalty_weight: float = 5.0


class SafetyFilterShield:
    def __init__(self, config: ShieldConfig | None = None):
        self.cfg = config if config is not None else ShieldConfig()
        assert self.cfg.mode in ("cbf_opt", "mps_rollout", "switch")
        self.total_interventions: int = 0
        self.total_steps: int = 0
        self.last_intervention_mag: float = 0.0
        self._feat_buf = torch.zeros((1, 8), dtype=torch.float32)

    def reset_stats(self) -> None:
        self.total_interventions = 0
        self.total_steps = 0
        self.last_intervention_mag = 0.0

    # -----------------------------------------------------------------------
    # 1. Emulated Reference Trajectory & Learned Variance Rollout
    #    (Goodall & Belardinelli 2026 / A2C-GP-Shield; Wabersich & Zeilinger 2018)
    # -----------------------------------------------------------------------
    def _get_or_build_emulated_trajectory(self, env) -> np.ndarray:
        """
        Rolls out the Heteroscedastic Neural Physics Emulator (`env.learned_model`)
        along `env.v_des` to produce and cache:
          - `env._cached_v_emu`: Emulated Reference Trajectory mu_emu(k)
          - `env._cached_sigma_emu`: Propagated Heteroscedastic Standard Deviation sigma_tube(k)
        """
        sig = (
            len(env.v_des),
            float(env.v_des[0]),
            float(env.v_des[len(env.v_des) // 2]),
            float(env.v_des[-1]),
            id(env.learned_model),
        )
        if getattr(env, "_cached_v_emu_sig", None) == sig and hasattr(env, "_cached_v_emu"):
            return env._cached_v_emu

        n_pts = len(env.v_des)
        v_emu = np.zeros(n_pts, dtype=np.float64)
        sigma_emu = np.zeros(n_pts, dtype=np.float64)
        mu_v, v_prev, u_act, sigma_v = 0.0, 0.0, 0.0, self.cfg.base_sigma_mps

        mpc_emu = env.internal_mpc.__class__(
            n_p=env.internal_mpc.n_p,
            n_c=env.internal_mpc.n_c,
            q1=env.internal_mpc.q1,
            q2=env.internal_mpc.q2,
            q3=env.internal_mpc.q3,
            adaptive_shift=True,
            mass_kg=env.physics.M,
            max_traction_accel=env.physics.max_traction_accel,
            max_braking_decel=env.physics.max_braking_decel,
        )
        ones_gains = np.ones(3, dtype=np.float64) * env.gains_scale

        for k in range(1, n_pts):
            ref_h = env.v_des[k : min(k + mpc_emu.n_p, n_pts)]
            x_s = k * env.d
            slope_iota = (
                env.physics.grade_resistance(x_s) + env.physics.curve_resistance(x_s)
            ) / (env.physics.M * env.physics.g)
            u_cmd = mpc_emu.select_action(
                mu_v, ref_h, slope_iota=slope_iota, dt_step=env.dt, u_act_prev=u_act
            )
            mu_next, u_act, sigma_v = self._predict_step_with_uncertainty(
                env, u_cmd, mu_v, v_prev, u_act, sigma_v, k, ones_gains
            )
            v_emu[k] = mu_next
            sigma_emu[k] = sigma_v
            v_prev, mu_v = mu_v, mu_next

        env._cached_v_emu = v_emu
        env._cached_sigma_emu = sigma_emu
        env._cached_v_emu_sig = sig
        return v_emu

    def classify_driving_regime(
        self, env, k_c: int, v_target: float, dv_target: float
    ) -> str:
        """
        Classifies step `k_c` into ("shift", "stop", "brake", "cruise") across all 6 modes
        of Havaei & Sandidzadeh (2022) Eq. 53, including feasible multi-step shift ramps.
        """
        k_look_end = min(k_c + self.cfg.brake_lookahead_steps, len(env.v_des) - 1)
        v_horizon = env.v_des[max(0, k_c - 1) : k_look_end + 1]
        dv_horizon = np.diff(v_horizon) if len(v_horizon) > 1 else np.array([dv_target])
        min_v_ahead = float(np.min(v_horizon))
        dv_5step = float(v_target - env.v_des[max(0, k_c - 5)])

        is_shift_ramp = (
            (abs(dv_target) > 1.2)
            or (env.shift_timer > 0)
            or (abs(dv_target) > 0.58 and abs(dv_5step) > 2.2 and min_v_ahead > 1.5)
        )
        if is_shift_ramp:
            return "shift"
        if (
            (v_target <= 3.2 and dv_target < -0.01)
            or (v_target <= 1e-3)
            or (min_v_ahead <= 3.0 and dv_target < -0.02 and float(np.min(dv_horizon)) < -0.03)
        ):
            return "stop"
        if dv_target < -0.03:
            return "brake"
        return "cruise"

    def compute_pincer_bounds(
        self, env, k_step: int, v_ref_curr: float | None = None
    ) -> tuple[float, float, float, float, str]:
        k_c = min(k_step, env.max_steps)
        v_curr = float(env.v_real if v_ref_curr is None else v_ref_curr)
        v_target = float(env.v_des[k_c])
        dv_target = v_target - float(env.v_des[max(0, k_c - 1)])

        v_emu_traj = self._get_or_build_emulated_trajectory(env)
        v_emu_center = float(v_emu_traj[k_c])
        dv_emu = v_emu_center - float(v_emu_traj[max(0, k_c - 1)])

        # Learned 99% confidence radius r_99(k) at step k_c
        sigma_emu_arr = getattr(env, "_cached_sigma_emu", None)
        r99_k = self.cfg.z_sigma * float(sigma_emu_arr[k_c]) if sigma_emu_arr is not None else 0.03

        regime = self.classify_driving_regime(env, k_c, v_target, dv_target)
        a_kin_req = (v_emu_center - v_curr) / env.dt
        a_ref_slope = dv_emu / env.dt

        # Green Safe Envelope wraps outside the Yellow 99% Scissor Band (r99_k + regime tolerance)
        if regime == "shift":
            dv_ahead = float(env.v_des[min(k_c + 1, len(env.v_des) - 1)] - v_target)
            on_flat_plateau = (abs(dv_target) <= 0.08) and (abs(dv_ahead) <= 0.08)
            # Smooth transition from 0.16 on the ramp to cruise_pincer_mps (0.14) on the plateau
            rad_shift = r99_k + (self.cfg.cruise_pincer_mps if on_flat_plateau else 0.16)
            v_min = max(0.0, v_emu_center - rad_shift)
            v_max = v_emu_center + rad_shift
            return v_min, v_max, -2.0, 2.0, regime

        if regime == "stop":
            # Taper smoothly from brake_pincer_mps on the approach ramp down to stop_inv_tube_mps at standstill
            ramp_scale = float(np.clip(v_emu_center / 3.2, 0.0, 1.0))
            stop_margin = self.cfg.stop_inv_tube_mps + ramp_scale * (self.cfg.brake_pincer_mps - self.cfg.stop_inv_tube_mps)
            rad_stop = r99_k + stop_margin
            v_min = max(0.0, v_emu_center - rad_stop)
            v_max = v_emu_center + rad_stop
            a_req_stop = min(a_kin_req, 0.70 * a_kin_req + 0.30 * a_ref_slope)
            return v_min, v_max, -2.0, a_req_stop + 0.5 * self.cfg.accel_brake_tol_mps2, regime

        if regime == "brake":
            rad_brk = r99_k + self.cfg.brake_pincer_mps
            v_min = max(0.0, v_emu_center - rad_brk)
            v_max = v_emu_center + rad_brk
            a_lo = min(a_kin_req, a_ref_slope) - self.cfg.accel_cruise_tol_mps2
            a_hi = max(a_kin_req, a_ref_slope) + self.cfg.accel_brake_tol_mps2
            return v_min, v_max, a_lo, a_hi, regime

        rad_crs = r99_k + self.cfg.cruise_pincer_mps
        v_min = max(0.0, v_emu_center - rad_crs)
        v_max = v_emu_center + rad_crs
        a_lo = min(a_kin_req, a_ref_slope) - self.cfg.accel_cruise_tol_mps2
        a_hi = max(a_kin_req, a_ref_slope) + self.cfg.accel_cruise_tol_mps2
        return v_min, v_max, a_lo, a_hi, regime


    # -----------------------------------------------------------------------
    # 2. Certified Backup Policy pi_{backup} (Bastani 2021; Hsu et al. 2024)
    # -----------------------------------------------------------------------
    def get_fallback_weights(self, env, n_actions: int, regime: str) -> np.ndarray:
        if n_actions == 5:
            if regime == "shift":
                w_fb = [0.0, 0.15, 0.0, 0.0, 0.85]
            elif regime == "stop":
                w_fb = [0.0, 0.0, 0.0, 0.0, 1.0]
            elif regime == "brake":
                w_fb = [0.0, 0.0, 0.20, 0.0, 0.80]
            else:
                w_fb = [0.0, 0.0, 0.0, 0.10, 0.90]
        else:
            if regime == "shift":
                w_fb = [0.0, 0.80, 0.20, 0.0]
            elif regime == "stop":
                w_fb = [0.0, 0.0, 1.00, 0.0]
            elif regime == "brake":
                w_fb = [0.0, 0.0, 0.80, 0.20]
            else:
                w_fb = [0.0, 0.0, 0.30, 0.70]

        arr = np.asarray(w_fb, dtype=np.float64)
        return arr / np.sum(arr)

    @staticmethod
    def bleed_positive_integral_windup(env, regime: str) -> None:
        if regime in ("brake", "stop") and env.int_err > 0.0:
            env.int_err *= 0.20

    @staticmethod
    def build_dmps_candidate_pool(
        w_task: np.ndarray, w_fb: np.ndarray, n_actions: int, regime: str = "cruise"
    ) -> list[np.ndarray]:
        """Candidate recovery pool for DMPS `planRec` (Banerjee et al. 2024 Eq. 2)."""
        if n_actions == 4:
            cands = [w_fb] + [np.eye(4, dtype=np.float64)[i] for i in range(4)]
            cands.extend([(1.0 - lam) * w_task + lam * w_fb for lam in (0.30, 0.65)])
            return cands

        return [
            w_fb,
            np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.15, 0.0, 0.0, 0.85], dtype=np.float64),
            np.array([0.0, 0.0, 0.25, 0.0, 0.75], dtype=np.float64),
            0.40 * w_task + 0.60 * w_fb,
        ]

    # -----------------------------------------------------------------------
    # 3. Learned Heteroscedastic Uncertainty Propagation (Option A)
    #    (Goodall & Belardinelli 2026 / A2C-GP-Shield; Banerjee et al. 2024)
    # -----------------------------------------------------------------------
    def _predict_step_with_uncertainty(
        self, env, u_cmd: float, mu_v: float, v_prev: float, u_act_curr: float,
        sigma_v_curr: float, k_step: int, pid_gains: np.ndarray
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
            v_norm = mu_v / 15.0
            self._feat_buf[0, 0] = (v_lin - mu_v) / 1.5
            self._feat_buf[0, 1] = v_norm
            self._feat_buf[0, 2] = u_clamped
            self._feat_buf[0, 3] = ext_f_norm
            self._feat_buf[0, 4] = v_norm * v_norm
            with torch.no_grad():
                out = env.learned_model.net(self._feat_buf)
                dv_res = float(out[0, 0].item())
                if out.shape[-1] >= 2:
                    log_var = float(np.clip(out[0, 1].item(), -10.5, -3.2))
                    sigma_delta = float(np.exp(0.5 * log_var))
                else:
                    sigma_delta = self.cfg.base_sigma_mps
            mu_next = float(max(0.0, v_lin + dv_res))
        else:
            f_prop_nl = u_act_next * env.physics.max_propulsion_force(mu_v, u_act_next)
            f_drag = env.physics.basic_resistance(mu_v)
            f_grade = env.physics.grade_resistance(x_sample)
            f_curve = env.physics.curve_resistance(x_sample)
            a_nl = (f_prop_nl - f_drag - f_grade - f_curve) / env.physics.M_eff
            mu_next = float(max(0.0, v_lin + env.dt * a_nl + inertia_mom))
            sigma_delta = self.cfg.base_sigma_mps

        # Include real-time disturbance measurement when predicting next step online
        if sigma_v_curr == 0.0:
            d_est = float(getattr(env, "d_raw", getattr(env, "d_hat", 0.0)))
            if abs(d_est) > 1e-5:
                mu_next = float(max(0.0, mu_next + env.dt * d_est))

        sigma_next = float(np.sqrt(self.cfg.gamma_sigma * (sigma_v_curr ** 2) + (sigma_delta ** 2)))
        return mu_next, u_act_next, sigma_next

    @staticmethod
    def _compute_dmps_planner_reward(
        env, regime: str, v_emu_target: float, v_raw_target: float, dv_target: float,
        mu_next: float, a_hat: float, a_req_h: float
    ) -> float:
        abs_err_emu = abs(v_emu_target - mu_next)
        acc_err_h = abs(a_hat - a_req_h) * env.dt
        if regime == "shift":
            s_dir = 1.0 if (dv_target if abs(dv_target) > 0.58 else env.shift_delta) >= 0.0 else -1.0
            os_h = max(0.0, s_dir * (mu_next - v_raw_target))
            return -140.0 * abs_err_emu - 500.0 * os_h
        if regime == "stop":
            return -350.0 * abs_err_emu - 180.0 * acc_err_h
        if regime == "brake":
            return -180.0 * abs_err_emu - 90.0 * acc_err_h
        return -100.0 * abs_err_emu - 40.0 * acc_err_h

    def evaluate_rollout_recoverability_and_return(
        self, env, u_step0: float, gains_step0: np.ndarray, w_candidate: np.ndarray,
        w_task_rl: np.ndarray, horizon_override: int | None = None, u_mpc_step0: float | None = None
    ) -> tuple[float, float]:
        k_0 = env.k
        mu_v, v_prev, u_act, int_err = float(env.v_real), float(env.v_prev), float(env.u_act), float(env.int_err)
        sigma_v, z = 0.0, self.cfg.z_sigma

        min_safety_margin = float("inf")
        cum_reward, discount = 0.0, 1.0
        horizon = horizon_override if horizon_override is not None else (
            1 if self.cfg.mode == "cbf_opt" else self.cfg.recovery_horizon_n
        )
        v_emu_traj = self._get_or_build_emulated_trajectory(env)

        # Read full instantaneous disturbance acting on the train
        d_persistent = float(getattr(env, "d_raw", getattr(env, "d_hat", 0.0)))
        sigma_emu_arr = getattr(env, "_cached_sigma_emu", None)

        for h in range(horizon):
            k_step = min(k_0 + 1 + h, env.max_steps)
            v_min_s, v_max_s, a_min_s, a_max_s, regime = self.compute_pincer_bounds(env, k_step, v_ref_curr=mu_v)
            v_raw_target = float(env.v_des[k_step])
            v_emu_target = float(v_emu_traj[k_step])
            v_emu_prev = float(v_emu_traj[max(0, k_step - 1)])
            dv_emu = v_emu_target - v_emu_prev

            # Yellow 99th-Percentile Scissor Bounds around v_emu_target
            r99_k = self.cfg.z_sigma * float(sigma_emu_arr[k_step]) if sigma_emu_arr is not None else 0.05
            scis_lo = max(0.0, v_emu_target - r99_k)
            scis_hi = v_emu_target + r99_k

            err_h = v_emu_target - mu_v
            int_err = float(np.clip(int_err + err_h * env.dt, -3.5, 3.5))

            if h == 0:
                u_h, gains_h = u_step0, gains_step0
            else:
                ref_rate_h = dv_emu / env.dt
                vel_rate_h = (mu_v - v_prev) / env.dt
                rate_err_h = vel_rate_h - ref_rate_h
                u_feedforward = (ref_rate_h - d_persistent) / (
                    env.physics.max_braking_decel if (ref_rate_h - d_persistent) < 0.0 else env.physics.max_traction_accel
                )
                u_mpc_h = np.clip(u_feedforward + 1.60 * err_h - 0.35 * rate_err_h, -1.0, 1.0)
                w_m = float(w_candidate[4]) if len(w_candidate) == 5 else 0.0
                if w_m >= 0.50 and int(np.argmax(w_candidate)) == 4:
                    u_h = float(u_mpc_h)
                else:
                    u_pid_h = np.clip(
                        (gains_step0[0] * err_h + gains_step0[1] * int_err) / env.u_scale, -1.0, 1.0
                    )
                    u_h = float(w_m * u_mpc_h + (1.0 - w_m) * u_pid_h)
                gains_h = gains_step0

            mu_next, u_act, sigma_v = self._predict_step_with_uncertainty(
                env, u_h, mu_v, v_prev, u_act, sigma_v, k_step, gains_h
            )
            dv_next = mu_next - mu_v
            a_hat = dv_next / env.dt

            # Guard Yellow Scissor Bound during active wind gusts; guard Green Safe Envelope when calm
            if abs(d_persistent) > 0.05:
                hi_bound, lo_bound = scis_hi, scis_lo
                sig_mult = 0.35
            else:
                hi_bound, lo_bound = v_max_s, v_min_s
                sig_mult = 1.0

            dist_to_hi = hi_bound - (mu_next + sig_mult * z * sigma_v)
            dist_to_lo = (mu_next - sig_mult * z * sigma_v) - lo_bound
            step_margin = min(dist_to_hi, dist_to_lo)

            if h == 0:
                min_safety_margin = step_margin
            else:
                min_safety_margin = min(min_safety_margin, step_margin + 0.02 * h)

            cum_reward += discount * self._compute_dmps_planner_reward(
                env, regime, v_emu_target, v_raw_target, dv_emu, mu_next, a_hat, (v_emu_target - mu_v) / env.dt
            )
            discount *= self.cfg.gamma_plan
            v_prev, mu_v = mu_v, mu_next
            if k_step >= env.max_steps:
                break

        q_synergy_bonus = 6.0 * float(np.dot(w_candidate, w_task_rl))
        return float(min_safety_margin), float(cum_reward + discount * q_synergy_bonus)

    # -----------------------------------------------------------------------
    # 4. Intervention Execution: Switch vs. 1-Step DCBF vs. DMPS (Fisac 2024)
    # -----------------------------------------------------------------------
    def _execute_intervention_scheme(
        self, w_task: np.ndarray, w_fb: np.ndarray, candidates: list[np.ndarray], eval_fn
    ) -> tuple[np.ndarray, float]:
        margin_task, _ = eval_fn(w_task, 1)
        if margin_task >= 0.0:
            self.last_intervention_mag = 0.0
            return w_task, 0.0

        self.total_interventions += 1

        if self.cfg.mode == "switch":
            l1_diff = float(np.sum(np.abs(w_fb - w_task)))
            self.last_intervention_mag = l1_diff
            return w_fb, l1_diff

        if self.cfg.mode == "cbf_opt":
            lo, hi = 0.0, 1.0
            for _ in range(7):
                mid = 0.5 * (lo + hi)
                w_mid = (1.0 - mid) * w_task + mid * w_fb
                if eval_fn(w_mid, 1)[0] >= 0.0:
                    hi = mid
                else:
                    lo = mid
            w_safe = (1.0 - hi) * w_task + hi * w_fb
            w_safe = w_safe / (np.sum(w_safe) + 1e-12)
            l1_diff = float(np.sum(np.abs(w_safe - w_task)))
            self.last_intervention_mag = l1_diff
            return w_safe, l1_diff

        best_rec_w, best_rec_ret = None, -float("inf")
        best_fb_w, best_fb_ret = w_fb, -float("inf")
        for w_cand in candidates:
            w_cand = w_cand / (np.sum(w_cand) + 1e-12)
            m_c, ret_c = eval_fn(w_cand, self.cfg.recovery_horizon_n)
            if m_c >= 0.0 and ret_c > best_rec_ret:
                best_rec_ret, best_rec_w = ret_c, w_cand
            if ret_c > best_fb_ret:
                best_fb_ret, best_fb_w = ret_c, w_cand

        w_safe = best_rec_w if best_rec_w is not None else best_fb_w
        l1_diff = float(np.sum(np.abs(w_safe - w_task)))
        self.last_intervention_mag = l1_diff
        return w_safe, l1_diff

    @staticmethod
    def _to_simplex_weights(action: int | np.ndarray, n_actions: int) -> np.ndarray:
        if isinstance(action, (int, np.integer)):
            w = np.zeros(n_actions, dtype=np.float64)
            w[int(action)] = 1.0
            return w
        w = np.asarray(action, dtype=np.float64).copy()
        return w / (np.sum(w) + 1e-12)

    def _compute_full_physics_backup_u(self, env, k_next: int) -> float:
        """Computes exact tube-centering command u_fb compensating Davis drag, slope, and storm gust d_raw."""
        v_emu_traj = self._get_or_build_emulated_trajectory(env)
        v_emu_next = float(v_emu_traj[k_next])
        v_curr = float(env.v_real)
        d_full = float(getattr(env, "d_raw", getattr(env, "d_hat", 0.0)))
        inertia_mom = 0.14 * (v_curr - float(env.v_prev))

        # Target physical acceleration needed to land directly on v_emu_next
        a_target = ((v_emu_next - v_curr - inertia_mom) / env.dt) - d_full
        x_s = k_next * env.d
        f_resist = (
            env.physics.basic_resistance(v_curr)
            + env.physics.grade_resistance(x_s)
            + env.physics.curve_resistance(x_s)
        )
        a_resist = f_resist / env.physics.M_eff
        a_motor_req = a_target + a_resist

        f_max = env.physics.max_propulsion_force(v_curr, 1.0 if a_motor_req >= 0.0 else -1.0)
        scale_a = f_max / env.physics.M_eff
        u_act_req = a_motor_req / max(1e-4, scale_a)
        return float(np.clip((u_act_req - 0.28 * float(env.u_act)) / 0.72, -1.0, 1.0))

    def filter_mpc_command(
        self, env, u_mpc_candidate: float
    ) -> tuple[float, float]:
        """
        Direct safety shield over standalone MPC command u_mpc_candidate in [-1, 1].
        Triggers when heavy gusts push predicted speed outside the Yellow Scissor Band.
        """
        self.total_steps += 1
        u_task = float(np.clip(u_mpc_candidate, -1.0, 1.0))
        k_next = min(env.k + 1, env.max_steps)
        v_target = float(env.v_des[k_next])

        if v_target <= 0.05 and env.v_real <= 0.08:
            env.int_err = 0.0
            self.last_intervention_mag = 0.0
            return 0.0, 0.0

        v_min_s, v_max_s, _, _, regime = self.compute_pincer_bounds(env, k_next)
        self.bleed_positive_integral_windup(env, regime)

        v_emu_traj = self._get_or_build_emulated_trajectory(env)
        v_emu_next = float(v_emu_traj[k_next])
        sigma_emu_arr = getattr(env, "_cached_sigma_emu", None)
        r99_k = self.cfg.z_sigma * float(sigma_emu_arr[k_next]) if sigma_emu_arr is not None else 0.05
        scis_lo = max(0.0, v_emu_next - r99_k)
        scis_hi = v_emu_next + r99_k

        d_now = float(getattr(env, "d_raw", 0.0))
        if abs(d_now) > 0.05:
            hi_bound, lo_bound = scis_hi, scis_lo
            sig_mult = 0.35
        else:
            hi_bound, lo_bound = v_max_s, v_min_s
            sig_mult = 1.0

        # 1. Fast 1-step barrier check
        ones_gains = np.ones(3, dtype=np.float64) * env.gains_scale
        mu_next, _, sigma_v = self._predict_step_with_uncertainty(
            env, u_task, float(env.v_real), float(env.v_prev),
            float(env.u_act), 0.0, k_next, ones_gains
        )
        z = self.cfg.z_sigma
        if (hi_bound - (mu_next + sig_mult * z * sigma_v)) >= 0.0 and ((mu_next - sig_mult * z * sigma_v) - lo_bound) >= 0.0:
            self.last_intervention_mag = 0.0
            return u_task, 0.0

        # 2. Scissor bound breached by gust: compute full physics + disturbance backup command u_fb
        self.total_interventions += 1
        u_fb = self._compute_full_physics_backup_u(env, k_next)

        if self.cfg.mode == "switch":
            diff = abs(u_fb - u_task)
            self.last_intervention_mag = diff
            return u_fb, diff

        w5_pure_mpc = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        def eval_u(u_val: float, hor: int | None) -> tuple[float, float]:
            u_c = float(np.clip(u_val, -1.0, 1.0))
            return self.evaluate_rollout_recoverability_and_return(
                env, u_c, ones_gains, w5_pure_mpc, w5_pure_mpc,
                horizon_override=hor, u_mpc_step0=u_c
            )

        if self.cfg.mode == "cbf_opt":
            lo, hi = 0.0, 1.0
            for _ in range(4):
                mid = 0.5 * (lo + hi)
                u_mid = (1.0 - mid) * u_task + mid * u_fb
                if eval_u(u_mid, 1)[0] >= 0.0:
                    hi = mid
                else:
                    lo = mid
            u_safe = float(np.clip((1.0 - hi) * u_task + hi * u_fb, -1.0, 1.0))
            diff = abs(u_safe - u_task)
            self.last_intervention_mag = diff
            return u_safe, diff

        best_u, best_ret = u_fb, -float("inf")
        best_safe_u, best_safe_ret = None, -float("inf")
        for lam in (0.35, 0.70, 1.0):
            u_cand = float(np.clip((1.0 - lam) * u_task + lam * u_fb, -1.0, 1.0))
            m_c, ret_c = eval_u(u_cand, self.cfg.recovery_horizon_n)
            if m_c >= 0.0 and ret_c > best_safe_ret:
                best_safe_ret, best_safe_u = ret_c, u_cand
            if ret_c > best_ret:
                best_ret, best_u = ret_c, u_cand

        u_safe = best_safe_u if best_safe_u is not None else best_u
        diff = abs(u_safe - u_task)
        self.last_intervention_mag = diff
        return u_safe, diff

    def filter_pid_weights(self, env, action: int | np.ndarray) -> tuple[np.ndarray, float]:
        self.total_steps += 1
        w_task = self._to_simplex_weights(action, n_actions=4)
        k_next = min(env.k + 1, env.max_steps)
        _, _, _, _, regime = self.compute_pincer_bounds(env, k_next)
        v_target = float(env.v_des[k_next])
        dv_target = v_target - float(env.v_des[min(env.k, env.max_steps)])

        if v_target <= 1e-3 and env.v_real <= 0.02:
            env.int_err = 0.0
            self.last_intervention_mag = 0.0
            return w_task, 0.0

        self.bleed_positive_integral_windup(env, regime)
        w_fb = self.get_fallback_weights(env, n_actions=4, regime=regime)
        candidates = self.build_dmps_candidate_pool(w_task, w_fb, n_actions=4, regime=regime)

        def eval_w4(w_vec: np.ndarray, hor: int | None) -> tuple[float, float]:
            _, _, p_gains = env._alg_weights_from_action(w_vec)
            u_c = env._compute_pid_command(p_gains, v_target, dv_target, update_state=False)
            return self.evaluate_rollout_recoverability_and_return(
                env, u_c, p_gains, w_vec, w_task, horizon_override=hor
            )

        return self._execute_intervention_scheme(w_task, w_fb, candidates, eval_w4)

    def filter_hybrid_weights(
        self, env, action: int | np.ndarray, u_mpc_candidate: float
    ) -> tuple[np.ndarray, float]:
        self.total_steps += 1
        w5_task = self._to_simplex_weights(action, n_actions=5)
        k_next = min(env.k + 1, env.max_steps)
        _, _, _, _, regime = self.compute_pincer_bounds(env, k_next)
        v_target = float(env.v_des[k_next])
        dv_target = v_target - float(env.v_des[min(env.k, env.max_steps)])

        if v_target <= 0.05 and env.v_real <= 0.08:
            env.int_err = 0.0
            self.last_intervention_mag = 0.0
            return np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64), 0.0

        self.bleed_positive_integral_windup(env, regime)

        w5_fb = self.get_fallback_weights(env, n_actions=5, regime=regime)
        candidates = self.build_dmps_candidate_pool(w5_task, w5_fb, n_actions=5, regime=regime)

        def eval_w5(w5_vec: np.ndarray, hor: int | None) -> tuple[float, float]:
            w_m = float(w5_vec[4])
            act_i = int(np.argmax(w5_vec))
            w4_sub = (
                w5_vec[:4] / (np.sum(w5_vec[:4]) + 1e-12)
                if np.sum(w5_vec[:4]) > 1e-8
                else np.array([0.0, 0.0, 1.0, 0.0])
            )
            _, _, p_gains = env._alg_weights_from_action(w4_sub)
            if w_m >= 0.999 or (act_i == 4 and w_m >= 0.50):
                u_comb = float(np.clip(u_mpc_candidate, -1.0, 1.0))
            else:
                u_p = env._compute_pid_command(p_gains, v_target, dv_target, update_state=False)
                u_comb = float(np.clip(w_m * u_mpc_candidate + (1.0 - w_m) * u_p, -1.0, 1.0))
            return self.evaluate_rollout_recoverability_and_return(
                env, u_comb, p_gains, w5_vec, w5_task, horizon_override=hor, u_mpc_step0=u_mpc_candidate
            )

        return self._execute_intervention_scheme(w5_task, w5_fb, candidates, eval_w5)


# =============================================================================
# Analytic Heteroscedastic Scissor Band Suite (Option A — Zero Heuristics)
# (Goodall & Belardinelli 2026 / `sacktock/A2C-GP-Shield`; Wang et al. 2023)
# =============================================================================
def compute_and_print_shield_scissor_band(
    env,
    traj_dict: dict,
    shield: SafetyFilterShield | None = None,
    learned_policy=None,
) -> dict[str, np.ndarray]:
    """
    Computes the 99th-Percentile Scissor Band (Yellow) and Safe Envelope (Green)
    ANALYTICALLY from the Heteroscedastic Neural Physics Emulator's learned
    variance head `log_var = log(sigma_theta^2(s, a))` (Option A):
      - Yellow 99th-Pct Scissor: `v_emu(k) ± z_0.995 * sigma_tube(k)` (z_0.995 = 2.576)
      - Green Safe Manifold: `[v_min_safe(k), v_max_safe(k)]` from `compute_pincer_bounds`
    No sliding-window percentiles, no heuristic blends, and no convolution filters.
    """
    cfg = shield.cfg if shield is not None else ShieldConfig()
    sh_eval = shield if shield is not None else SafetyFilterShield(cfg)

    k_arr = np.asarray(traj_dict["k"], dtype=int)
    n_steps = len(k_arr)
    v_traj_arr = np.asarray(traj_dict["v"], dtype=np.float64)
    v_des_arr = np.asarray(traj_dict["v_des"], dtype=np.float64)

    v_emu_full = sh_eval._get_or_build_emulated_trajectory(env)
    sigma_emu_full = env._cached_sigma_emu

    v_emu_arr = np.zeros(n_steps, dtype=np.float64)
    scissor_rad = np.zeros(n_steps, dtype=np.float64)
    v_min_safe = np.zeros(n_steps, dtype=np.float64)
    v_max_safe = np.zeros(n_steps, dtype=np.float64)
    max_env_rad = np.zeros(n_steps, dtype=np.float64)
    step_err_arr = np.zeros(n_steps, dtype=np.float64)
    regimes = []

    env.shift_timer = 0
    env.shift_delta = 0.0
    for idx in range(n_steps):
        k_c = min(int(k_arr[idx]), len(v_emu_full) - 1)
        v_curr_emu = float(v_emu_full[k_c])
        sigma_k = float(sigma_emu_full[k_c])
        v_emu_arr[idx] = v_curr_emu

        v_t = float(v_des_arr[idx])
        dv_t = v_t - float(v_des_arr[max(0, idx - 1)])
        dv_5 = v_t - float(v_des_arr[max(0, idx - 5)])
        if abs(dv_t) > 1.2 or (abs(dv_t) > 0.58 and abs(dv_5) > 2.2):
            env.shift_timer = 1
            env.shift_delta = dv_t
        elif env.shift_timer > 0:
            env.shift_timer = (env.shift_timer + 1) if env.shift_timer < 6 else 0   #

        # Online safety barrier X_safe(k) and driving regime
        vmin_k, vmax_k, _, _, reg_k = sh_eval.compute_pincer_bounds(env, k_c, v_ref_curr=v_curr_emu)
        regimes.append(reg_k)

        r99_k = cfg.z_sigma * sigma_k
        scissor_rad[idx] = r99_k

        v_min_safe[idx] = vmin_k
        v_max_safe[idx] = vmax_k
        max_env_rad[idx] = max(vmax_k - v_curr_emu, v_curr_emu - vmin_k)

        step_err_arr[idx] = abs(v_traj_arr[idx] - v_curr_emu)

    regimes_arr = np.array(regimes)
    intervened_arr = np.asarray(
        traj_dict.get("shield_intervened", np.zeros(n_steps, dtype=bool)), dtype=bool
    )

    scissor_lo = np.maximum(0.0, v_emu_arr - scissor_rad)
    scissor_hi = v_emu_arr + scissor_rad

    emp_rmse = float(np.sqrt(np.mean(step_err_arr ** 2)))
    emp_p99_err = float(np.percentile(step_err_arr, 99.0))
    emp_max_err = float(np.max(step_err_arr))
    inside_green_pct = 100.0 * float(
        np.mean((v_traj_arr >= v_min_safe - 1e-3) & (v_traj_arr <= v_max_safe + 1e-3))
    )

    print("\n" + "=" * 132)
    print(
        f"HETEROSCEDASTIC NLL SHIELD SCISSOR DIAGNOSTICS | Emu-to-Real RMSE = {emp_rmse:.4f} m/s | "
        f"Emp 99th-Pct = {emp_p99_err:.4f} m/s | Max Err = {emp_max_err:.4f} m/s | "
        f"Real-in-Emu-Tube Containment: {inside_green_pct:.1f}%"
    )
    print("-" * 132)
    print(
        f"{'Regime':<10} | {'Steps':>7} | {'Emu-Real RMSE':>14} | {'Learned 99% (Yellow)':>21} | "
        f"{'Safe Envelope (Green)':>23} | {'Regime MaxErr':>15} | {'Interventions':>18}"
    )
    print("-" * 132)
    for reg_name in ("cruise", "shift", "brake", "stop"):
        mask = regimes_arr == reg_name
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        reg_rmse = float(np.sqrt(np.mean(step_err_arr[mask] ** 2)))
        r_max_v = float(np.max(step_err_arr[mask]))
        y_w = float(np.mean(scissor_rad[mask]))
        g_w = float(np.mean(max_env_rad[mask]))
        n_int = int(np.sum(intervened_arr[mask]))
        print(
            f"{reg_name:<10} | {cnt:7d} | {reg_rmse:10.4f} m/s | {'±' + f'{y_w:.4f} m/s':>21} | "
            f"{'±' + f'{g_w:.4f} m/s':>23} | {r_max_v:11.4f} m/s | "
            f"{n_int:5d} ({100.0 * n_int / max(1, cnt):5.1f}%)"
        )
    print("=" * 132)

    return {
        "k": k_arr,
        "v_pred": v_emu_arr,
        "scissor_lo": scissor_lo,
        "scissor_hi": scissor_hi,
        "scissor_rad": scissor_rad,
        "emp_rmse": emp_rmse,
        "emp_max_err": emp_max_err,
        "v_min": v_min_safe,
        "v_max": v_max_safe,
        "regimes": regimes_arr,
        "intervened": intervened_arr,
    }