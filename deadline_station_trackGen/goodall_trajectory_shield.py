"""
Module 3: Multi-Property Goodall & Belardinelli (2026) Heteroscedastic Trajectory Shield
(`goodall_trajectory_shield.py`).

Monitors 4 Operational Properties on the Trajectory & MPC Interface:
  1. `speed_limit`: 99% Scissor velocity tube vs. spatial speed limit & mass-scaled brake capacity.
  2. `smoothness`: Maintains non-annoying acceleration/braking rates & low jerk.
  3. `deadline`: Maintains on-time arrival trajectory inside the punctuality cone.
  4. `mpc_failure`: Detects when physical train speed diverges from the emulated tube
     (e.g., motor saturation under heavy passenger weight + headwind) and re-anchors the reference.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pure_mpc_controller import PureLiuMPC
from havaei_generalizing_rl import NonlinearTrainPhysics, LearnedLinearizationModel
from railroad_corridor import RailroadCorridor, generate_disjoint_wind_events
from cosine_trajectory_shaper import CosineTrajectoryShaper


@dataclass
class TrajectoryShieldConfig:
    mode: str = "mps_rollout"               # "switch", "cbf_opt", or "mps_rollout"
    z_sigma: float = 2.576                  # Exact 99% Gaussian quantile Phi^{-1}(0.995) = 2.576
    base_sigma_mps: float = 0.0045          # Floor standard deviation sigma_0 (m/s)
    gamma_sigma: float = 0.65               # Stochastic tube variance contraction factor (Report Eq. 4)
    cruise_pincer_mps: float = 0.14         # Safe envelope margin in cruise (m/s)
    shift_pincer_mps: float = 0.16          # Safe envelope margin in cosine shifts (m/s)
    brake_pincer_mps: float = 0.16          # Safe envelope margin in braking (m/s)
    stop_inv_tube_mps: float = 0.015        # Terminal invariant set margin at platform standstill (m/s)
    deadline_time_tol_s: float = 1.60       # Punctuality cone tolerance |T_proj - T_deadline| <= 1.6s
    recovery_horizon_n: int = 3             # Multi-step rollout horizon for DMPS `planRec`
    gamma_plan: float = 0.85                # Discount factor for `planRec`


class GoodallTrajectoryShield:
    def __init__(
        self,
        corridor: RailroadCorridor,
        emulator: LearnedLinearizationModel | None = None,
        config: TrajectoryShieldConfig | None = None,
    ):
        self.corridor = corridor
        self.emulator = emulator
        self.cfg = config if config is not None else TrajectoryShieldConfig()
        assert self.cfg.mode in ("switch", "cbf_opt", "mps_rollout")
        self._feat_buf = torch.zeros((1, 8), dtype=torch.float32)
        self.reset_stats()

    def reset_stats(self) -> None:
        self.total_steps: int = 0
        self.total_interventions: int = 0
        self.sigma_tube_curr: float = self.cfg.base_sigma_mps
        self.property_counts: dict[str, int] = {
            "speed_limit": 0,
            "smoothness": 0,
            "deadline": 0,
            "mpc_failure": 0,
        }

    def predict_emulator_step_with_uncertainty(
        self,
        physics: NonlinearTrainPhysics,
        x_curr: float,
        mu_v: float,
        v_prev: float,
        u_act_curr: float,
        u_cmd: float,
        sigma_v_curr: float,
        d_wind_hat: float = 0.0,
    ) -> tuple[float, float, float]:
        u_clamped = float(np.clip(u_cmd, -1.0, 1.0))
        u_act_next = 0.72 * u_clamped + 0.28 * u_act_curr
        f_prop_lin = u_act_next * physics.M_eff * physics.max_traction_accel
        a_lin = (f_prop_lin - physics.B * mu_v) / physics.M_eff
        inertia_mom = 0.14 * (mu_v - v_prev)
        v_lin = float(max(0.0, mu_v + self.corridor.dt * a_lin + inertia_mom))

        if self.emulator is not None:
            ext_f_norm = (
                physics.grade_resistance(x_curr) + physics.curve_resistance(x_curr)
            ) / (0.03 * physics.M * physics.g)
            v_norm = mu_v / 15.0
            self._feat_buf[0, 0] = (v_lin - mu_v) / 1.5
            self._feat_buf[0, 1] = v_norm
            self._feat_buf[0, 2] = u_clamped
            self._feat_buf[0, 3] = ext_f_norm
            self._feat_buf[0, 4] = v_norm * v_norm
            with torch.no_grad():
                out = self.emulator.net(self._feat_buf)
                dv_res = float(out[0, 0].item())
                log_var = float(np.clip(out[0, 1].item(), -10.5, -3.2))
                sigma_delta = float(np.exp(0.5 * log_var))
            mu_next = float(max(0.0, v_lin + dv_res))
        else:
            sigma_delta = self.cfg.base_sigma_mps
            mu_next = v_lin

        if mu_next > 1e-3 or u_clamped > 1e-3:
            mu_next = float(max(0.0, mu_next + self.corridor.dt * d_wind_hat))
            sigma_wind_extra = 0.007 * abs(d_wind_hat)
            sigma_step_tot = np.hypot(sigma_delta, sigma_wind_extra)
            sigma_next = float(np.sqrt(self.cfg.gamma_sigma * (sigma_v_curr ** 2) + (sigma_step_tot ** 2)))
        else:
            sigma_next = self.cfg.base_sigma_mps

        return mu_next, u_act_next, sigma_next

    def compute_regime_and_envelope_margin(
        self, v_curr: float, v_ref_next: float, in_landing: bool
    ) -> tuple[str, float]:
        dv = v_ref_next - v_curr
        if in_landing or v_ref_next <= 3.2:
            ramp_scale = float(np.clip(v_ref_next / 3.2, 0.0, 1.0))
            margin = self.cfg.stop_inv_tube_mps + ramp_scale * (self.cfg.brake_pincer_mps - self.cfg.stop_inv_tube_mps)
            return "stop", margin
        if abs(dv) > 0.12:
            return "shift", self.cfg.shift_pincer_mps
        if dv < -0.03:
            return "brake", self.cfg.brake_pincer_mps
        return "cruise", self.cfg.cruise_pincer_mps

    def check_four_property_margins(
        self,
        env,
        v_cmd_cand: float,
        a_start_cand: float,
        b_land_cand: float,
        v_pace_feasible: float,
    ) -> dict[str, float]:
        r99_now = self.cfg.z_sigma * self.sigma_tube_curr
        v_ceil = self.corridor.compute_smooth_spatial_ceiling(env.x_curr, b_land_cand)

        # Property 1: Speed & Brake Envelope
        h_speed = (v_ceil + 0.01) - (v_cmd_cand + r99_now)

        # Property 2: Smoothness & Comfort Flag
        a_comf_hi = self.corridor.comfort_accel_max
        b_comf_hi = self.corridor.comfort_braking_max
        h_smooth = min(a_comf_hi - a_start_cand, b_comf_hi - b_land_cand, b_land_cand - 0.60) * 3.0

        # Property 3: Deadline Punctuality Cone
        x_rem = max(1.0, self.corridor.station_x_m - env.x_curr)
        v_ref_sq = max(25.0, v_pace_feasible ** 2)
        tol_v_deadline = max(0.28, (self.cfg.deadline_time_tol_s * v_ref_sq) / max(300.0, x_rem))
        h_deadline = tol_v_deadline - abs(v_cmd_cand - v_pace_feasible)

        # Property 4: MPC Tracking Tube Failure Reference
        mpc_lag_err = abs(env.v_curr - env.shaper.ramp_v_goal) if env.shaper.ramp_u_phase >= 0.99 else 0.0
        is_saturated = abs(env.u_act) >= 0.96 and mpc_lag_err > 0.45
        h_mpc = -0.25 if is_saturated else 0.50

        return {
            "speed_limit": float(h_speed),
            "smoothness": float(h_smooth),
            "deadline": float(h_deadline),
            "mpc_failure": float(h_mpc),
        }

    def build_targeted_recovery_policy(
        self,
        env,
        violated_prop: str,
        v_cmd_task: float,
        a_start_task: float,
        b_land_task: float,
        v_pace_feasible: float,
    ) -> np.ndarray:
        r99_now = self.cfg.z_sigma * self.sigma_tube_curr
        v_ceil = self.corridor.compute_smooth_spatial_ceiling(env.x_curr, self.corridor.comfort_braking_decel)
        v_max_safe = max(1.5, v_ceil - r99_now - 0.02)

        a_comf = float(np.clip(a_start_task, 0.70, self.corridor.comfort_accel_max))
        b_comf = float(np.clip(b_land_task, 0.65, self.corridor.comfort_braking_max))
        v_on_time = float(np.clip(v_pace_feasible, 1.5, v_max_safe))

        if violated_prop == "speed_limit":
            return np.array([v_max_safe, a_comf, b_comf], dtype=np.float64)

        if violated_prop == "smoothness":
            return np.array([v_on_time, min(a_comf, 1.05), min(b_comf, 0.80)], dtype=np.float64)

        if violated_prop == "mpc_failure":
            v_reanchor = float(np.clip(0.5 * (env.v_curr + v_on_time), 1.5, v_max_safe))
            return np.array([v_reanchor, 0.95, 0.78], dtype=np.float64)

        return np.array([v_on_time, a_comf, b_comf], dtype=np.float64)

    def filter_trajectory_action(
        self,
        env,
        v_cmd_task: float,
        a_start_task: float,
        b_land_task: float,
        v_pace_feasible: float,
    ) -> tuple[float, float, float, bool, float, str]:
        self.total_steps += 1
        if env.shaper.in_landing:
            return v_cmd_task, a_start_task, b_land_task, False, 0.0, "none"

        margins = self.check_four_property_margins(
            env, v_cmd_task, a_start_task, b_land_task, v_pace_feasible
        )
        worst_prop = min(margins, key=margins.get)
        min_margin = margins[worst_prop]

        if min_margin >= 0.0:
            return v_cmd_task, a_start_task, b_land_task, False, 0.0, "none"

        self.total_interventions += 1
        self.property_counts[worst_prop] += 1

        p_task = np.array([v_cmd_task, a_start_task, b_land_task], dtype=np.float64)
        p_fb = self.build_targeted_recovery_policy(
            env, worst_prop, v_cmd_task, a_start_task, b_land_task, v_pace_feasible
        )

        if self.cfg.mode == "switch":
            diff = float(np.sum(np.abs(p_fb - p_task)))
            return float(p_fb[0]), float(p_fb[1]), float(p_fb[2]), True, diff, worst_prop

        if self.cfg.mode == "cbf_opt":
            lo, hi = 0.0, 1.0
            for _ in range(7):
                mid = 0.5 * (lo + hi)
                p_mid = (1.0 - mid) * p_task + mid * p_fb
                m_dict = self.check_four_property_margins(
                    env, float(p_mid[0]), float(p_mid[1]), float(p_mid[2]), v_pace_feasible
                )
                if min(m_dict.values()) >= 0.0:
                    hi = mid
                else:
                    lo = mid
            p_safe = (1.0 - hi) * p_task + hi * p_fb
            diff = float(np.sum(np.abs(p_safe - p_task)))
            return float(p_safe[0]), float(p_safe[1]), float(p_safe[2]), True, diff, worst_prop

        candidates = [(1.0 - lam) * p_task + lam * p_fb for lam in (0.35, 0.65, 0.85, 1.0)]
        best_p = p_fb
        best_score = -float("inf")

        for p_c in candidates:
            m_dict = self.check_four_property_margins(
                env, float(p_c[0]), float(p_c[1]), float(p_c[2]), v_pace_feasible
            )
            min_m = min(m_dict.values())
            multi_obj = (
                -10.0 * abs(float(p_c[0]) - v_pace_feasible)
                - 3.0 * abs(float(p_c[2]) - self.corridor.comfort_braking_decel)
                - 0.8 * float(np.sum(np.abs(p_c - p_task)))
            )
            if min_m >= -1e-4 and multi_obj > best_score:
                best_score = multi_obj
                best_p = p_c

        diff = float(np.sum(np.abs(best_p - p_task)))
        return float(best_p[0]), float(best_p[1]), float(best_p[2]), True, diff, worst_prop


def train_150_step_mpc_trajectory_emulator(
    corridor: RailroadCorridor,
    emulator_steps: int = 150,
    n_rollouts: int = 6,
    lr: float = 2.5e-3,
    seed: int = 42,
) -> tuple[LearnedLinearizationModel, list[float]]:
    rng = np.random.default_rng(seed)
    x_list, vc_list, vlin_list, u_list, ext_list, g_list, vnext_list, sdir_list = [], [], [], [], [], [], [], []
    ones_g = np.ones(3, dtype=np.float32)

    for ep in range(n_rollouts):
        mass_ep = float(rng.uniform(76000.0, 94000.0))
        cor_ep = copy.deepcopy(corridor)
        cor_ep.mass_kg = mass_ep
        physics = NonlinearTrainPhysics(mass_kg=mass_ep)
        mpc = PureLiuMPC(
            n_p=5, n_c=3, q1=160.0, q2=0.05, q3=0.35, adaptive_shift=True,
            dt=cor_ep.dt, mass_kg=mass_ep,
            max_traction_accel=cor_ep.max_traction_accel,
            max_braking_decel=cor_ep.max_service_braking,
        )
        shaper = CosineTrajectoryShaper(cor_ep)
        x_c, v_c, v_p, u_act, t_c, d_hat = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        v_target_ep = float(rng.uniform(11.8, 16.8))
        a_start_ep = float(rng.uniform(0.85, 1.18))
        b_land_ep = float(rng.uniform(0.72, 0.88))

        wind_ep, _, _ = (
            generate_disjoint_wind_events(550.0, cor_ep.dt, seed=seed + ep * 33)
            if ep % 2 == 1
            else (np.zeros(1200, dtype=np.float64), "", [])
        )

        k_step = 0
        while t_c < 460.0 and (cor_ep.station_x_m - x_c) > 0.20:
            if k_step % 70 == 0:
                v_target_ep = float(rng.uniform(11.8, 16.8))
            v_ref_h, _, _ = shaper.step_reference_horizon(
                x_c, v_c, t_c, v_target_ep, a_start_ep, b_land_ep, n_p=mpc.n_p, d_wind_hat=d_hat
            )
            slope_iota = (physics.grade_resistance(x_c) + physics.curve_resistance(x_c)) / (physics.M * physics.g)
            ext_f_norm = (physics.grade_resistance(x_c) + physics.curve_resistance(x_c)) / (0.03 * physics.M * physics.g)

            mpc.prev_v = float(v_p)
            u_cmd = mpc.select_action(
                v_c, v_ref_h, slope_iota=slope_iota, dt_step=cor_ep.dt,
                u_act_prev=u_act, disturbance_accel=0.90 * d_hat
            )
            u_clamped = float(np.clip(u_cmd, -1.0, 1.0))
            u_act = 0.72 * u_clamped + 0.28 * u_act

            f_prop_lin = u_act * physics.M_eff * physics.max_traction_accel
            a_lin = (f_prop_lin - physics.B * v_c) / physics.M_eff
            inertia_mom = 0.14 * (v_c - v_p)
            v_lin = float(max(0.0, v_c + cor_ep.dt * a_lin + inertia_mom))

            f_prop_nl = u_act * physics.max_propulsion_force(v_c, u_act)
            f_drag = physics.basic_resistance(v_c)
            f_grade = physics.grade_resistance(x_c)
            f_curve = physics.curve_resistance(x_c)
            a_nl_nom = (f_prop_nl - f_drag - f_grade - f_curve) / physics.M_eff

            d_ext = float(wind_ep[min(k_step, len(wind_ep) - 1)]) if (v_c > 1e-3 or u_clamped > 1e-3) else 0.0
            v_nl = float(max(0.0, v_c + cor_ep.dt * (a_nl_nom + d_ext) + inertia_mom))
            v_nl_nom = float(max(0.0, v_c + cor_ep.dt * a_nl_nom + inertia_mom))

            a_meas = ((v_nl - v_c) - inertia_mom) / cor_ep.dt
            d_hat = 0.70 * d_hat + 0.30 * (a_meas - a_nl_nom)

            dv_step = v_ref_h[0] - v_c
            s_dir = 1.0 if dv_step > 0.15 else (-1.0 if dv_step < -0.15 else 0.0)

            x_list.append(x_c)
            vc_list.append(v_c)
            vlin_list.append(v_lin)
            u_list.append(u_clamped)
            ext_list.append(ext_f_norm)
            g_list.append(ones_g)
            vnext_list.append(v_nl_nom)
            sdir_list.append(s_dir)

            v_p, v_c = v_c, v_nl
            x_c += 0.5 * (v_p + v_c) * cor_ep.dt
            t_c += cor_ep.dt
            k_step += 1
            if shaper.in_landing and v_c <= 0.01:
                break

    x_t = torch.tensor(x_list, dtype=torch.float32)
    vc_t = torch.tensor(vc_list, dtype=torch.float32)
    vlin_t = torch.tensor(vlin_list, dtype=torch.float32)
    u_t = torch.tensor(u_list, dtype=torch.float32)
    ext_t = torch.tensor(ext_list, dtype=torch.float32)
    g_t = torch.as_tensor(np.asarray(g_list), dtype=torch.float32)
    target_t = torch.tensor(vnext_list, dtype=torch.float32)
    sdir_t = torch.tensor(sdir_list, dtype=torch.float32)

    emulator = LearnedLinearizationModel(hidden_dim=64)
    optimizer = optim.AdamW(emulator.parameters(), lr=lr, weight_decay=2e-4)
    warmup = max(5, emulator_steps // 10)
    loss_curve = []
    best_rmse = float("inf")
    best_state = copy.deepcopy(emulator.state_dict())

    for step in range(emulator_steps):
        lr_mult = (
            float(step + 1) / float(warmup)
            if step < warmup
            else 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * float(step - warmup) / float(max(1, emulator_steps - warmup))))
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_mult

        pred_mu, _, log_var = emulator.forward_with_uncertainty(x_t, vc_t, vlin_t, u_t, ext_t, g_t)
        diff = pred_mu - target_t
        asym_w = torch.where((sdir_t * diff) < 0.0, 2.5, 1.0)
        mse_loss = torch.mean(asym_w * (diff ** 2))

        dyn_var_floor = (0.0080 + 0.0140 * torch.abs(target_t - vc_t) + 0.0055 * torch.abs(u_t)) ** 2
        res_sq_detached = diff.detach() ** 2 + dyn_var_floor
        nll_var_loss = 0.5 * torch.mean(log_var + res_sq_detached / (torch.exp(log_var) + 1e-8))
        loss = mse_loss + 0.25 * nll_var_loss

        with torch.no_grad():
            cur_rmse = float(torch.sqrt(nn.functional.mse_loss(pred_mu, target_t)).item())
            loss_curve.append(cur_rmse)
            if cur_rmse < best_rmse:
                best_rmse = cur_rmse
                best_state = copy.deepcopy(emulator.state_dict())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(emulator.parameters(), 0.5)
        optimizer.step()

    emulator.load_state_dict(best_state)
    emulator.eval()
    print(f"  [Heteroscedastic GP/NLL Physics Emulator ({emulator_steps} steps, Multi-Mass)] Fit RMSE: {best_rmse:.5f} m/s", flush=True)
    return emulator, loss_curve


def print_goodall_trajectory_scissor_diagnostics(run_dict: dict) -> None:
    tr = run_dict["traj"]
    v_real = tr["v"]
    v_emu = tr["v_emu"]
    scis_r = tr["scissor_rad"]
    env_r = tr["env_rad"]
    regimes = tr["regimes"]
    intervened = tr["intervened"]
    prop_counts = run_dict.get("property_counts", {})

    step_err = np.abs(v_real - v_emu)
    emp_rmse = float(np.sqrt(np.mean(step_err ** 2)))
    emp_p99 = float(np.percentile(step_err, 99.0))
    emp_max = float(np.max(step_err))
    in_green_pct = 100.0 * float(np.mean(step_err <= env_r + 1e-3))

    print("\n" + "=" * 142)
    print(
        f"GOODALL (2026) MULTI-PROPERTY TRAJECTORY-SHIELD DIAGNOSTICS (3-STATION MULTI-LEG ROUTE) | "
        f"Emu-to-Real RMSE = {emp_rmse:.4f} m/s | Emp 99th-Pct = {emp_p99:.4f} m/s | "
        f"Max Err = {emp_max:.4f} m/s | Physical Tube Containment: {in_green_pct:.1f}%"
    )
    if prop_counts:
        print(
            f"MULTI-PROPERTY RECOVERY BREAKDOWN -> SpeedLimit: {prop_counts.get('speed_limit', 0)} | "
            f"Smoothness: {prop_counts.get('smoothness', 0)} | DeadlineCone: {prop_counts.get('deadline', 0)} | "
            f"MPCFailure: {prop_counts.get('mpc_failure', 0)}"
        )
    print("-" * 142)
    print(
        f"{'Regime':<10} | {'Steps':>7} | {'Emu-Real RMSE':>14} | {'Learned 99% (Yellow)':>21} | "
        f"{'Safe Envelope (Green)':>23} | {'Regime MaxErr':>15} | {'Interventions':>18}"
    )
    print("-" * 142)
    for reg_name in ("cruise", "shift", "brake", "stop"):
        mask = regimes == reg_name
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        r_rmse = float(np.sqrt(np.mean(step_err[mask] ** 2)))
        r_max = float(np.max(step_err[mask]))
        y_w = float(np.mean(scis_r[mask]))
        g_w = float(np.mean(env_r[mask]))
        n_int = int(np.sum(intervened[mask]))
        print(
            f"{reg_name:<10} | {cnt:7d} | {r_rmse:10.4f} m/s | {'±' + f'{y_w:.4f} m/s':>21} | "
            f"{'±' + f'{g_w:.4f} m/s':>23} | {r_max:11.4f} m/s | "
            f"{n_int:5d} ({100.0 * n_int / max(1, cnt):5.1f}%)"
        )
    print("=" * 142, flush=True)