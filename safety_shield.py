"""
Modular Runtime Safety Filter & Scissor Band Suite (`safety_shield.py`).
Safety Shielding centered strictly on the Emulated Trajectory of the Learned Physics Model.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class ShieldConfig:
    mode: str = "mps_rollout"              # "switch", "cbf_opt", or "mps_rollout"

    cruise_pincer_mps: float = 0.14
    brake_pincer_mps: float = 0.16
    shift_overshoot_tol_mps: float = 0.005
    stop_inv_tube_mps: float = 0.012

    accel_brake_tol_mps2: float = 0.35
    accel_cruise_tol_mps2: float = 0.30
    brake_lookahead_steps: int = 8

    z_sigma: float = 2.58
    base_sigma_mps: float = 0.0015
    cbf_alpha: float = 0.55
    recovery_horizon_n: int = 4
    gamma_plan: float = 0.85
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

    def _get_or_build_emulated_trajectory(self, env) -> np.ndarray:
        """
        Generates and caches the Emulated Reference Trajectory `v_emu_traj` by rolling out
        the Neural Physics Emulator (`env.learned_model`) along `env.v_des` using the
        actuator-lag-aware dynamics controller.
        """
        sig = (len(env.v_des), float(env.v_des[0]), float(env.v_des[len(env.v_des) // 2]), float(env.v_des[-1]))
        if getattr(env, "_cached_v_emu_sig", None) == sig and hasattr(env, "_cached_v_emu"):
            return env._cached_v_emu

        n_pts = len(env.v_des)
        v_emu = np.zeros(n_pts, dtype=np.float64)
        mu_v, v_prev, u_act = 0.0, 0.0, 0.0

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
            mu_next, u_act, _ = self._predict_step_with_uncertainty(
                env, u_cmd, mu_v, v_prev, u_act, 0.0, k, ones_gains
            )
            v_emu[k] = mu_next
            v_prev, mu_v = mu_v, mu_next

        env._cached_v_emu = v_emu
        env._cached_v_emu_sig = sig
        return v_emu

    def classify_driving_regime(
        self, env, k_c: int, v_target: float, dv_target: float
    ) -> str:
        """
        Classifies step `k_c` into ("shift", "stop", "brake", "cruise"), including
        multi-step profile shifts (`|dv_target| > 0.65` and `|dv_4step| > 2.2`) and the
        settling window (`env.shift_timer > 0`).
        """
        k_look_end = min(k_c + self.cfg.brake_lookahead_steps, len(env.v_des) - 1)
        v_horizon = env.v_des[max(0, k_c - 1) : k_look_end + 1]
        dv_horizon = np.diff(v_horizon) if len(v_horizon) > 1 else np.array([dv_target])
        min_v_ahead = float(np.min(v_horizon))
        dv_4step = float(v_target - env.v_des[max(0, k_c - 4)])

        is_shift_ramp = (
            (abs(dv_target) > 1.2)
            or (env.shift_timer > 0)
            or (abs(dv_target) > 0.65 and abs(dv_4step) > 2.2 and min_v_ahead > 1.5)
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
        """
        Centers the safety tube [v_min, v_max] strictly on the Emulated Trajectory
        `v_emu_traj[k_c]`, enforcing plateau anti-overshoot and terminal station docking.
        """
        k_c = min(k_step, env.max_steps)
        v_curr = float(env.v_real if v_ref_curr is None else v_ref_curr)
        v_target = float(env.v_des[k_c])
        dv_target = v_target - float(env.v_des[max(0, k_c - 1)])

        v_emu_traj = self._get_or_build_emulated_trajectory(env)
        v_emu_center = float(v_emu_traj[k_c])
        dv_emu = v_emu_center - float(v_emu_traj[max(0, k_c - 1)])

        regime = self.classify_driving_regime(env, k_c, v_target, dv_target)
        a_kin_req = (v_emu_center - v_curr) / env.dt
        a_ref_slope = dv_emu / env.dt

        if regime == "shift":
            dv_4step = float(v_target - env.v_des[max(0, k_c - 4)])
            s_delta = dv_target if abs(dv_target) > 0.65 else (env.shift_delta if env.shift_delta != 0.0 else dv_4step)
            if s_delta >= 0.0:
                v_min = max(0.0, v_emu_center - 0.18)
                v_max = min(v_emu_center + 0.18, v_target + self.cfg.shift_overshoot_tol_mps)
                return v_min, v_max, -2.0, max(0.02, (v_max - v_curr) / env.dt), regime
            v_min = max(0.0, v_emu_center - 0.18, v_target - self.cfg.shift_overshoot_tol_mps)
            v_max = v_emu_center + 0.18
            return v_min, v_max, min(-0.02, (v_min - v_curr) / env.dt), 2.0, regime

        if regime == "stop":
            v_min = max(0.0, v_emu_center - self.cfg.stop_inv_tube_mps)
            v_max = v_emu_center + self.cfg.stop_inv_tube_mps
            a_req_stop = min(a_kin_req, 0.70 * a_kin_req + 0.30 * a_ref_slope)
            return v_min, v_max, -2.0, a_req_stop + 0.5 * self.cfg.accel_brake_tol_mps2, regime

        if regime == "brake":
            v_min = max(0.0, v_emu_center - self.cfg.brake_pincer_mps)
            v_max = v_emu_center + self.cfg.brake_pincer_mps
            a_lo = min(a_kin_req, a_ref_slope) - self.cfg.accel_cruise_tol_mps2
            a_hi = max(a_kin_req, a_ref_slope) + self.cfg.accel_brake_tol_mps2
            return v_min, v_max, a_lo, a_hi, regime

        v_min = max(0.0, v_emu_center - self.cfg.cruise_pincer_mps)
        v_max = v_emu_center + self.cfg.cruise_pincer_mps
        a_lo = min(a_kin_req, a_ref_slope) - self.cfg.accel_cruise_tol_mps2
        a_hi = max(a_kin_req, a_ref_slope) + self.cfg.accel_cruise_tol_mps2
        return v_min, v_max, a_lo, a_hi, regime

    def get_fallback_weights(self, env, n_actions: int, regime: str) -> np.ndarray:
        if n_actions == 5:
            if regime == "shift":
                k_next = min(env.k + 1, env.max_steps)
                err_ahead = float(env.v_des[k_next]) - env.v_real
                w_fb = [0.0, 0.85, 0.15, 0.0, 0.0] if abs(err_ahead) < 0.45 else [0.0, 0.0, 0.0, 0.0, 1.0]
            elif regime == "stop":
                w_fb = [0.0, 0.0, 0.0, 0.0, 1.0]
            elif regime == "brake":
                w_fb = [0.0, 0.0, 0.35, 0.0, 0.65]
            else:
                w_fb = [0.0, 0.0, 0.0, 0.15, 0.85]
        else:
            if regime == "shift":
                w_fb = [0.0, 0.85, 0.15, 0.0]
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
        if regime == "stop" and env.int_err > 0.0:
            env.int_err *= 0.20

    @staticmethod
    def build_dmps_candidate_pool(
        w_task: np.ndarray, w_fb: np.ndarray, n_actions: int, regime: str = "cruise"
    ) -> list[np.ndarray]:
        if n_actions == 4:
            cands = [w_fb] + [np.eye(4, dtype=np.float64)[i] for i in range(4)]
            cands.extend([(1.0 - lam) * w_task + lam * w_fb for lam in (0.30, 0.65)])
            return cands

        if regime in ("brake", "stop"):
            return [
                w_fb,
                np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                np.array([0.0, 0.0, 0.25, 0.0, 0.75], dtype=np.float64),
                np.array([0.0, 0.0, 0.50, 0.0, 0.50], dtype=np.float64),
                0.35 * w_task + 0.65 * w_fb,
            ]

        return [
            w_fb,
            np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.35, 0.0, 0.65], dtype=np.float64),
            np.array([0.0, 0.85, 0.15, 0.0, 0.0], dtype=np.float64),
            0.40 * w_task + 0.60 * w_fb,
        ]

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
            self._feat_buf[0, 1] = v_norm
            self._feat_buf[0, 2] = u_clamped
            self._feat_buf[0, 3] = ext_f_norm
            self._feat_buf[0, 4] = v_norm * v_norm
            with torch.no_grad():
                dv_res = float(env.learned_model.net(self._feat_buf).item())
            mu_next = float(max(0.0, v_lin + dv_res))
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

    @staticmethod
    def _compute_dmps_planner_reward(
        env, regime: str, v_emu_target: float, v_raw_target: float, dv_target: float,
        mu_next: float, a_hat: float, a_req_h: float
    ) -> float:
        abs_err_emu = abs(v_emu_target - mu_next)
        acc_err_h = abs(a_hat - a_req_h) * env.dt
        if regime == "shift":
            s_dir = 1.0 if (dv_target if abs(dv_target) > 0.65 else env.shift_delta) >= 0.0 else -1.0
            os_h = max(0.0, s_dir * (mu_next - v_raw_target))
            return -120.0 * abs_err_emu - 500.0 * os_h
        if regime == "stop":
            return -350.0 * abs_err_emu - 180.0 * acc_err_h
        if regime == "brake":
            return -180.0 * abs_err_emu - 90.0 * acc_err_h
        return -90.0 * abs_err_emu - 40.0 * acc_err_h

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

        for h in range(horizon):
            k_step = min(k_0 + 1 + h, env.max_steps)
            v_min_s, v_max_s, a_min_s, a_max_s, regime = self.compute_pincer_bounds(env, k_step, v_ref_curr=mu_v)
            v_raw_target = float(env.v_des[k_step])
            v_emu_target = float(v_emu_traj[k_step])
            v_emu_prev = float(v_emu_traj[max(0, k_step - 1)])
            dv_emu = v_emu_target - v_emu_prev

            half_band = 0.5 * (v_max_s - v_min_s)
            h_up_curr = max(0.0, (v_emu_prev + half_band) - mu_v)
            h_lo_curr = max(0.0, mu_v - max(0.0, v_emu_prev - half_band))

            err_h = v_emu_target - mu_v
            int_err = float(np.clip(int_err + err_h * env.dt, -3.5, 3.5))

            if h == 0:
                u_h, gains_h = u_step0, gains_step0
            else:
                ref_rate_h = dv_emu / env.dt
                vel_rate_h = (mu_v - v_prev) / env.dt
                d_term_h = (
                    (1.10 * ref_rate_h - 0.90 * vel_rate_h)
                    if ref_rate_h < -0.04
                    else (0.55 * ref_rate_h - 1.10 * vel_rate_h)
                )
                u_pid_h = np.clip(
                    (gains_step0[0] * err_h + gains_step0[1] * int_err + gains_step0[2] * d_term_h) / env.u_scale,
                    -1.0,
                    1.0,
                )
                u_mpc_h = np.clip(1.65 * err_h + 0.50 * ref_rate_h - 0.35 * vel_rate_h, -1.0, 1.0)
                w_m = float(w_candidate[4]) if len(w_candidate) == 5 else 0.0
                u_h = float(w_m * u_mpc_h + (1.0 - w_m) * u_pid_h)
                gains_h = gains_step0

            mu_next, u_act, sigma_v = self._predict_step_with_uncertainty(
                env, u_h, mu_v, v_prev, u_act, sigma_v, k_step, gains_h
            )
            dv_next = mu_next - mu_v
            a_hat = dv_next / env.dt

            vel_up_slack = (v_max_s - (mu_next + z * sigma_v)) - (1.0 - self.cfg.cbf_alpha) * h_up_curr
            vel_lo_slack = ((mu_next - z * sigma_v) - v_min_s) - (1.0 - self.cfg.cbf_alpha) * h_lo_curr
            step_margin = min(vel_up_slack, vel_lo_slack)

            min_safety_margin = min(min_safety_margin, step_margin)
            cum_reward += discount * self._compute_dmps_planner_reward(
                env, regime, v_emu_target, v_raw_target, dv_emu, mu_next, a_hat, (v_emu_target - mu_v) / env.dt
            )
            discount *= self.cfg.gamma_plan
            v_prev, mu_v = mu_v, mu_next
            if k_step >= env.max_steps:
                break

        q_synergy_bonus = 6.0 * float(np.dot(w_candidate, w_task_rl))
        return float(min_safety_margin), float(cum_reward + discount * q_synergy_bonus)

    def _execute_intervention_scheme(
        self, w_task: np.ndarray, w_fb: np.ndarray, candidates: list[np.ndarray], eval_fn
    ) -> tuple[np.ndarray, float]:
        margin_task, _ = eval_fn(w_task, None)
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

        if v_target <= 1e-3 and env.v_real <= 0.02:
            env.int_err = 0.0
            self.last_intervention_mag = 0.0
            return np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64), 0.0

        self.bleed_positive_integral_windup(env, regime)

        if regime == "stop":
            env.int_err = 0.0
            w5_pure_mpc = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            if int(np.argmax(w5_task)) == 4 and float(w5_task[4]) >= 0.50:
                self.last_intervention_mag = 0.0
                return w5_pure_mpc, 0.0
            l1_diff = float(np.sum(np.abs(w5_pure_mpc - w5_task)))
            self.total_interventions += 1
            self.last_intervention_mag = l1_diff
            return w5_pure_mpc, l1_diff

        w5_fb = self.get_fallback_weights(env, n_actions=5, regime=regime)
        candidates = self.build_dmps_candidate_pool(w5_task, w5_fb, n_actions=5, regime=regime)

        def eval_w5(w5_vec: np.ndarray, hor: int | None) -> tuple[float, float]:
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

        return self._execute_intervention_scheme(w5_task, w5_fb, candidates, eval_w5)


# =============================================================================
# Diagnostic Scissor Band Suite Centered on the True Emulated Trajectory
# =============================================================================
def compute_and_print_shield_scissor_band(
    env,
    traj_dict: dict,
    shield: SafetyFilterShield | None = None,
    learned_policy=None,
) -> dict[str, np.ndarray]:
    """
    Uses the exact Emulated Reference Trajectory `_get_or_build_emulated_trajectory(env)`
    enforced by the runtime shield, properly updates `env.shift_timer` so all shift
    steps are classified, and centers the Yellow (99th-Pct) and Green (100% MaxEnv)
    Scissor Bands strictly on `v_emu_arr`.
    """
    cfg = shield.cfg if shield is not None else ShieldConfig()
    sh_eval = shield if shield is not None else SafetyFilterShield(cfg)

    k_arr = np.asarray(traj_dict["k"], dtype=int)
    n_steps = len(k_arr)
    v_traj_arr = np.asarray(traj_dict["v"], dtype=np.float64)
    v_des_arr = np.asarray(traj_dict["v_des"], dtype=np.float64)

    v_emu_full = sh_eval._get_or_build_emulated_trajectory(env)
    v_emu_arr = np.zeros(n_steps, dtype=np.float64)
    step_err_arr = np.zeros(n_steps, dtype=np.float64)
    regimes = []

    env.shift_timer = 0
    env.shift_delta = 0.0
    for idx in range(n_steps):
        k_c = min(int(k_arr[idx]), len(v_emu_full) - 1)
        v_curr_emu = float(v_emu_full[k_c])
        v_emu_arr[idx] = v_curr_emu

        v_t = float(v_des_arr[idx])
        dv_t = v_t - float(v_des_arr[max(0, idx - 1)])
        dv_4 = v_t - float(v_des_arr[max(0, idx - 4)])
        if abs(dv_t) > 1.2 or (abs(dv_t) > 0.65 and abs(dv_4) > 2.2):
            env.shift_timer = 1
            env.shift_delta = dv_t
        elif env.shift_timer > 0:
            env.shift_timer = (env.shift_timer + 1) if env.shift_timer < 6 else 0

        reg_k = sh_eval.classify_driving_regime(env, k_c, v_t, dv_t)
        regimes.append(reg_k)

        step_err_arr[idx] = abs(v_traj_arr[idx] - v_curr_emu)

    regimes_arr = np.array(regimes)
    intervened_arr = np.asarray(
        traj_dict.get("shield_intervened", np.zeros(n_steps, dtype=bool)), dtype=bool
    )

    reg_p99 = {}
    reg_max = {}
    for reg_name in ("cruise", "shift", "brake", "stop"):
        m = regimes_arr == reg_name
        if np.any(m):
            reg_p99[reg_name] = float(np.percentile(step_err_arr[m], 99.0))
            reg_max[reg_name] = float(np.max(step_err_arr[m]))
        else:
            reg_p99[reg_name] = 0.04
            reg_max[reg_name] = 0.08

    rad_99_raw = np.zeros(n_steps, dtype=np.float64)
    rad_max_raw = np.zeros(n_steps, dtype=np.float64)

    for idx in range(n_steps):
        reg_k = regimes[idx]
        w0, w1 = max(0, idx - 2), min(n_steps, idx + 3)
        loc_rms = float(np.sqrt(np.mean(step_err_arr[w0:w1] ** 2)))
        loc_max = float(np.max(step_err_arr[w0:w1]))

        r99 = max(
            loc_max * 0.90,
            0.60 * min(cfg.z_sigma * loc_rms, 1.15 * reg_p99[reg_k]) + 0.40 * reg_p99[reg_k],
            0.030,
        )
        r99 = float(np.clip(r99, 0.030, 0.18))
        if v_des_arr[idx] <= 1e-3 and v_emu_arr[idx] <= 0.02:
            r99 = 0.010
        rad_99_raw[idx] = r99

        box_tol = (
            cfg.cruise_pincer_mps
            if reg_k == "cruise"
            else (cfg.brake_pincer_mps if reg_k == "brake" else 0.08)
        )
        rmax = max(r99 + 0.035, loc_max * 1.08, 0.55 * loc_max + 0.45 * reg_max[reg_k], box_tol)
        rmax = float(np.clip(rmax, r99 + 0.025, 0.28))
        if v_des_arr[idx] <= 1e-3 and v_emu_arr[idx] <= 0.02:
            rmax = 0.020
        rad_max_raw[idx] = rmax

    kernel = np.array([0.20, 0.60, 0.20])
    scissor_rad = np.convolve(np.pad(rad_99_raw, 1, mode="edge"), kernel, mode="valid")
    max_env_rad = np.convolve(np.pad(rad_max_raw, 1, mode="edge"), kernel, mode="valid")

    v_center = v_emu_arr
    scissor_lo = np.maximum(0.0, v_center - scissor_rad)
    scissor_hi = v_center + scissor_rad

    v_min_safe = np.maximum(0.0, v_center - max_env_rad)
    v_max_safe = v_center + max_env_rad

    emp_rmse = float(np.sqrt(np.mean(step_err_arr ** 2)))
    emp_p99_err = float(np.percentile(step_err_arr, 99.0))
    emp_max_err = float(np.max(step_err_arr))
    inside_green_pct = 100.0 * float(
        np.mean((v_traj_arr >= v_min_safe - 1e-3) & (v_traj_arr <= v_max_safe + 1e-3))
    )

    print("\n" + "=" * 132)
    print(
        f"EMULATED-TRAJECTORY SHIELD SCISSOR DIAGNOSTICS | Emu-to-Real RMSE = {emp_rmse:.4f} m/s | "
        f"99th-Pct (Yellow) = {emp_p99_err:.4f} m/s | Max Err (Green) = {emp_max_err:.4f} m/s | "
        f"Real-in-Emu-Tube Containment: {inside_green_pct:.1f}%"
    )
    print("-" * 132)
    print(
        f"{'Regime':<10} | {'Steps':>7} | {'Emu-Real RMSE':>14} | {'99th-Pct (Yellow)':>21} | "
        f"{'100% MaxEnv (Green)':>23} | {'Regime MaxErr':>15} | {'Interventions':>18}"
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