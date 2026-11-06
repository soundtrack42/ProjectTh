"""
Module 4 (Main Runner): Pure Model-Free PPO + GAE (3,000+ Epochs Overnight Training)
for the Multi-Station Long Corridor (15-17.5 km, 3 Landings, Dynamic Passenger Weight Changes),
Multi-Property Goodall Trajectory Shield, and 30-Scenario Benchmark (`dynamic_railroad_mpc.py`).

Pure Model-Free RL Highlights:
  - Zero supervised action targets (`targ_a_t` removed): The policy explores via a stochastic
    Gaussian actor `Normal(mu_theta(s), sigma_theta)` and updates strictly via the Clipped PPO
    surrogate loss + Generalized Advantage Estimation (GAE) from environment rewards.
  - Automatic Overnight Checkpointing: Saves `best_ppo_cosine_pilot.pt` and `latest_ppo_cosine_pilot.pt`.
"""
from __future__ import annotations

import argparse
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(1)

from pure_mpc_controller import PureLiuMPC
from havaei_generalizing_rl import NonlinearTrainPhysics, LearnedLinearizationModel
from railroad_corridor import (
    RailroadCorridor,
    generate_multi_station_scenario,
)
from cosine_trajectory_shaper import CosineTrajectoryShaper
from goodall_trajectory_shield import (
    TrajectoryShieldConfig,
    GoodallTrajectoryShield,
    train_150_step_mpc_trajectory_emulator,
    print_goodall_trajectory_scissor_diagnostics,
)


# ===========================================================================
# 1. Single-Leg & Multi-Leg RL Environment with Dynamic Train Mass
# ===========================================================================
class MPCTransitionRailroadEnv:
    """
    15-Feature RL Observation Interface:
      [0..1]   x_curr / L, x_rem / L
      [2]      t_rem / 450.0
      [3..6]   v_curr, v_naive, v_kin_hint, (v_kin_hint - v_naive)
      [7]      (v_curr - v_naive)
      [8..9]   v_lim_now, v_lim_lookahead
      [10]     tanh((x_brake - x_curr) / 500)
      [11]     local track grade & curve resistance
      [12..13] online wind estimate d_hat & wind trend dot{d}_hat
      [14]     normalized passenger vehicle mass ratio (M_curr - 85000) / 15000
    """
    def __init__(
        self,
        corridor: RailroadCorridor | None = None,
        traj_shield: GoodallTrajectoryShield | None = None,
        emulator: LearnedLinearizationModel | None = None,
    ):
        self.corridor = corridor if corridor is not None else RailroadCorridor()
        self.traj_shield = traj_shield
        self.emulator = emulator
        self.obs_dim = 15
        self.act_dim = 3
        self.wind_profile = np.zeros(1600, dtype=np.float64)
        self._sync_physics_and_mpc()
        self.reset()

    def _sync_physics_and_mpc(self) -> None:
        self.physics = NonlinearTrainPhysics(mass_kg=self.corridor.mass_kg)
        self.physics.max_traction_accel = self.corridor.max_traction_accel
        self.physics.max_braking_decel = self.corridor.max_service_braking
        self.shaper = CosineTrajectoryShaper(self.corridor)
        self.mpc = PureLiuMPC(
            n_p=5, n_c=3, q1=160.0, q2=0.05, q3=0.35, adaptive_shift=True,
            dt=self.corridor.dt, mass_kg=self.corridor.mass_kg,
            max_traction_accel=self.corridor.max_traction_accel,
            max_braking_decel=self.corridor.max_service_braking,
        )

    def reset(
        self,
        corridor_override: RailroadCorridor | None = None,
        wind_profile: np.ndarray | None = None,
    ) -> np.ndarray:
        if corridor_override is not None:
            self.corridor = corridor_override
            if self.traj_shield is not None:
                self.traj_shield.corridor = corridor_override
        self._sync_physics_and_mpc()

        if wind_profile is not None:
            self.wind_profile = np.asarray(wind_profile, dtype=np.float64).copy()
        else:
            self.wind_profile = np.zeros(1600, dtype=np.float64)

        self.t_curr = float(self.corridor.start_time_s)
        self.k_elapsed = 0
        self.x_curr = 0.0
        self.v_curr = 0.0
        self.v_prev = 0.0
        self.u_act = 0.0
        self.d_hat = 0.0
        self.d_hat_prev = 0.0
        self.prev_accel = 0.0
        self.energy_kwh = 0.0
        self.sigma_tube = 0.0045
        self.shaper.reset()
        self.mpc.reset()
        return self._get_obs()

    def compute_feasible_deadline_pace(self, a_start: float, b_land: float) -> tuple[float, float]:
        """Used by GoodallTrajectoryShield to define the punctuality safety cone."""
        x_rem = max(0.0, self.corridor.station_x_m - self.x_curr)
        t_rem = max(self.corridor.dt, self.corridor.deadline_s - self.t_curr)
        v_naive = x_rem / t_rem
        v_local_cap = self.corridor.compute_smooth_spatial_ceiling(self.x_curr, b_land)

        if x_rem <= 10.0 or self.shaper.in_landing:
            return float(np.clip(v_naive, 0.0, v_local_cap)), float(v_naive)

        _, _, _, b_eff_land = CosineTrajectoryShaper.dynamic_cosine_braking_spot(
            self.corridor, self.v_curr, b_land, d_wind_hat=self.d_hat
        )

        def estimate_remaining_time(v_c: float) -> float:
            v_throat = min(v_c, self.corridor.speed_zones[-1].v_limit_mps)
            dx_land = (np.pi * (v_throat ** 2)) / (4.0 * b_eff_land)
            t_land = (np.pi * v_throat) / (2.0 * b_eff_land)
            x_b = max(self.x_curr + 1.0, self.corridor.station_x_m - dx_land)
            if self.x_curr >= x_b:
                return (2.0 * x_rem) / max(0.5, self.v_curr + 0.5)

            t_acc = 0.0
            sim_v = self.v_curr
            for z in self.corridor.speed_zones:
                seg_a = max(self.x_curr, z.x_start)
                seg_b = min(x_b, z.x_end)
                if seg_b <= seg_a:
                    continue
                seg_len = seg_b - seg_a
                v_seg = min(v_c, z.v_limit_mps)
                dv = abs(v_seg - sim_v)
                if dv > 0.15:
                    rate = a_start if v_seg >= sim_v else b_land
                    t_ramp = (np.pi * dv) / (2.0 * max(0.45, rate))
                    dx_ramp = 0.5 * (sim_v + v_seg) * t_ramp
                    if dx_ramp < seg_len:
                        t_acc += t_ramp + (seg_len - dx_ramp) / max(0.5, v_seg)
                    else:
                        t_acc += seg_len / max(0.5, 0.5 * (sim_v + v_seg))
                else:
                    t_acc += seg_len / max(0.5, v_seg)
                sim_v = v_seg
            return t_acc + t_land

        lo, hi = 2.0, 17.8
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if estimate_remaining_time(mid) > t_rem:
                lo = mid
            else:
                hi = mid
        v_feas = 0.5 * (lo + hi)
        return float(np.clip(v_feas, 1.5, v_local_cap)), float(v_naive)

    def _get_obs(self) -> np.ndarray:
        x_rem = max(0.0, self.corridor.station_x_m - self.x_curr)
        t_rem = max(self.corridor.dt, self.corridor.deadline_s - self.t_curr)
        v_lim_now = self.corridor.get_speed_limit(self.x_curr)
        v_lim_300 = self.corridor.get_speed_limit(min(self.corridor.station_x_m, self.x_curr + 300.0))

        x_brake, dx_brake, t_land, _ = CosineTrajectoryShaper.dynamic_cosine_braking_spot(
            self.corridor, self.v_curr, self.corridor.comfort_braking_decel, d_wind_hat=self.d_hat
        )
        v_naive = x_rem / t_rem
        # Purely local kinematic pace to braking spot (no zone-integration solver)
        t_to_brake_local = max(1.0, t_rem - t_land)
        x_to_brake_local = max(0.0, x_rem - dx_brake)
        v_kin_hint = x_to_brake_local / t_to_brake_local if x_to_brake_local > 20.0 else v_naive

        slope_iota = (
            self.physics.grade_resistance(self.x_curr) + self.physics.curve_resistance(self.x_curr)
        ) / (self.physics.M * self.physics.g)
        d_trend = (self.d_hat - self.d_hat_prev) / self.corridor.dt
        mass_norm = (self.corridor.mass_kg - 85000.0) / 15000.0

        return np.array([
            self.x_curr / self.corridor.station_x_m,
            x_rem / self.corridor.station_x_m,
            t_rem / 450.0,
            self.v_curr / 18.0,
            v_naive / 18.0,
            v_kin_hint / 18.0,
            (v_kin_hint - v_naive) / 4.0,
            (self.v_curr - v_naive) / 5.0,
            v_lim_now / 18.0,
            v_lim_300 / 18.0,
            np.tanh((x_brake - self.x_curr) / 500.0),
            slope_iota * 25.0,
            self.d_hat / 0.70,
            np.tanh(d_trend / 0.15),
            mass_norm,
        ], dtype=np.float32)

    def decode_rl_action(self, action: np.ndarray) -> tuple[float, float, float, float, float]:
        """
        100% Model-Free Action Decoding:
        Anchored strictly on `v_naive = x_rem / t_rem`. When the neural network outputs `a[0] = 0`,
        it commands `v_naive` (zero cruise overshoot). The PPO agent must learn through reward
        feedback alone how much overshoot `3.80 * a[0]` to command in each spatial zone!
        """
        a_clip = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        a_start_cos = 1.00 + 0.25 * float(a_clip[1])   # [0.75, 1.25] m/s^2
        b_land_cos = 0.79 + 0.13 * float(a_clip[2])    # [0.66, 0.92] m/s^2

        v_feas, v_naive = self.compute_feasible_deadline_pace(a_start_cos, b_land_cos)
        v_cap = self.corridor.compute_smooth_spatial_ceiling(self.x_curr, b_land_cos)

        rl_max_allow = v_cap + (0.45 if (self.corridor.is_infeasible_challenge and self.traj_shield is not None) else 0.0)
        overshoot_delta = 3.80 * float(a_clip[0])
        v_cmd_rl = float(np.clip(v_naive + overshoot_delta, 1.5, rl_max_allow))
        return v_cmd_rl, a_start_cos, b_land_cos, v_feas, v_naive

    def step(
        self,
        action: np.ndarray,
        record_micro: bool = False,
        mpc_wind_comp_gain: float = 0.95,
    ) -> tuple[np.ndarray, float, bool, dict]:
        v_cmd_rl, a_start_cos, b_land_cos, v_feas, v_naive = self.decode_rl_action(action)
        v_cmd_unshielded = v_cmd_rl

        shield_intervened = False
        shield_diff = 0.0
        violated_prop = "none"
        if self.traj_shield is not None:
            self.traj_shield.sigma_tube_curr = self.sigma_tube
            v_cmd_rl, a_start_cos, b_land_cos, shield_intervened, shield_diff, violated_prop = (
                self.traj_shield.filter_trajectory_action(
                    self, v_cmd_rl, a_start_cos, b_land_cos, v_pace_feasible=v_feas
                )
            )

        r99_now = 2.576 * self.sigma_tube if self.traj_shield is not None else 0.0
        x_brake_est, _, _, _ = CosineTrajectoryShaper.dynamic_cosine_braking_spot(
            self.corridor, self.v_curr, b_land_cos, v_scissor_rad=r99_now, d_wind_hat=self.d_hat
        )
        n_substeps = 1 if (self.shaper.in_landing or (x_brake_est - self.x_curr) < 180.0) else self.corridor.macro_steps

        macro_reward = 0.0
        micro_logs = []
        done = False

        for _ in range(n_substeps):
            t0 = time.perf_counter()
            r99_step = 2.576 * self.sigma_tube if self.traj_shield is not None else 0.0
            v_ref_h, x_brake_active, _ = self.shaper.step_reference_horizon(
                self.x_curr,
                self.v_curr,
                self.t_curr,
                v_cmd_rl,
                a_start_cos,
                b_land_cos,
                n_p=self.mpc.n_p,
                v_scissor_rad=r99_step,
                d_wind_hat=self.d_hat,
            )

            slope_iota = (
                self.physics.grade_resistance(self.x_curr) + self.physics.curve_resistance(self.x_curr)
            ) / (self.physics.M * self.physics.g)

            eff_ff_gain = 1.00 if self.shaper.in_landing else mpc_wind_comp_gain
            self.mpc.prev_v = float(self.v_prev)
            u_cmd = self.mpc.select_action(
                v_curr=self.v_curr,
                v_ref_horizon=v_ref_h,
                slope_iota=slope_iota,
                dt_step=self.corridor.dt,
                u_act_prev=self.u_act,
                disturbance_accel=eff_ff_gain * self.d_hat,
            )
            t1 = time.perf_counter()

            if self.traj_shield is not None:
                v_emu_step, _, self.sigma_tube = self.traj_shield.predict_emulator_step_with_uncertainty(
                    self.physics, self.x_curr, self.v_curr, self.v_prev,
                    self.u_act, u_cmd, self.sigma_tube, d_wind_hat=self.d_hat
                )
            elif self.emulator is not None:
                temp_sh = GoodallTrajectoryShield(self.corridor, self.emulator)
                v_emu_step, _, self.sigma_tube = temp_sh.predict_emulator_step_with_uncertainty(
                    self.physics, self.x_curr, self.v_curr, self.v_prev,
                    self.u_act, u_cmd, self.sigma_tube, d_wind_hat=self.d_hat
                )
            else:
                v_emu_step = float(v_ref_h[0])

            # Integrate Nonlinear Train Plant
            u_clamped = float(np.clip(u_cmd, -1.0, 1.0))
            self.u_act = 0.72 * u_clamped + 0.28 * self.u_act
            f_prop = self.u_act * self.physics.max_propulsion_force(self.v_curr, self.u_act)
            f_drag = self.physics.basic_resistance(self.v_curr)
            f_grade = self.physics.grade_resistance(self.x_curr)
            f_curve = self.physics.curve_resistance(self.x_curr)
            a_nom = (f_prop - f_drag - f_grade - f_curve) / self.physics.M_eff

            k_w = min(self.k_elapsed, len(self.wind_profile) - 1)
            d_ext = float(self.wind_profile[k_w]) if (self.v_curr > 1e-3 or u_clamped > 1e-3) else 0.0

            inertia_mom = 0.14 * (self.v_curr - self.v_prev)
            v_next = float(max(0.0, self.v_curr + self.corridor.dt * (a_nom + d_ext) + inertia_mom))
            x_next = self.x_curr + 0.5 * (self.v_curr + v_next) * self.corridor.dt

            a_real = (v_next - self.v_curr) / self.corridor.dt
            jerk = abs(a_real - self.prev_accel) / self.corridor.dt
            self.prev_accel = a_real

            self.d_hat_prev = self.d_hat
            if self.v_curr > 1e-3 or v_next > 1e-3:
                a_meas_no_mom = ((v_next - self.v_curr) - inertia_mom) / self.corridor.dt
                d_step = a_meas_no_mom - a_nom
                if abs(d_step) < 0.015:
                    self.d_hat *= 0.25
                else:
                    self.d_hat = float(0.55 * self.d_hat + 0.45 * d_step)
            else:
                self.d_hat = 0.0

            if f_prop > 0.0:
                power_w = f_prop * (0.5 * (self.v_curr + v_next))
                self.energy_kwh += (power_w * self.corridor.dt) / 3.6e6

            # Reward function for pure model-free PPO:
            # 1. Smoothness & comfort reward (penalize harsh decel/accel and high jerk)
            # 2. On-time trajectory progress reward toward the braking spot
            v_feas_now, v_naive_now = self.compute_feasible_deadline_pace(a_start_cos, b_land_cos)
            pace_err = abs(v_cmd_rl - v_feas_now) if not self.shaper.in_landing else 0.0
            harsh_brake_pen = max(0.0, -a_real - 0.95) ** 2
            annoying_acc_pen = max(0.0, a_real - 1.25) ** 2
            comfort_param_pen = 0.15 * ((a_start_cos - 1.00) ** 2 + (b_land_cos - 0.78) ** 2)

            step_rew = (
                -0.45 * pace_err
                - 1.20 * harsh_brake_pen
                - 0.80 * annoying_acc_pen
                - 0.03 * jerk
                - comfort_param_pen
            )
            macro_reward += step_rew * (self.corridor.dt / 4.0)

            if record_micro:
                scis_rad = 2.576 * self.sigma_tube
                reg_name, env_margin = (
                    self.traj_shield.compute_regime_and_envelope_margin(
                        self.v_curr, float(v_ref_h[0]), self.shaper.in_landing
                    )
                    if self.traj_shield is not None
                    else GoodallTrajectoryShield(self.corridor).compute_regime_and_envelope_margin(
                        self.v_curr, float(v_ref_h[0]), self.shaper.in_landing
                    )
                )
                micro_logs.append({
                    "t": self.t_curr,
                    "x": self.x_curr,
                    "v": v_next,
                    "v_emu": v_emu_step,
                    "v_ref0": float(v_ref_h[0]),
                    "v_cmd_unshielded": v_cmd_unshielded,
                    "v_cmd_rl": v_cmd_rl,
                    "v_overshoot": v_feas_now,
                    "v_naive": v_naive_now,
                    "x_brake": x_brake_active,
                    "u_act": self.u_act,
                    "accel": a_real,
                    "wind_true": d_ext,
                    "wind_hat": self.d_hat,
                    "mass_kg": self.corridor.mass_kg,
                    "scissor_rad": scis_rad,
                    "env_rad": scis_rad + env_margin,
                    "regime": reg_name,
                    "shield_intervened": shield_intervened,
                    "violated_prop": violated_prop,
                    "step_us": (t1 - t0) * 1e6,
                })

            self.v_prev = self.v_curr
            self.v_curr = v_next
            self.x_curr = x_next
            self.t_curr += self.corridor.dt
            self.k_elapsed += 1

            rem_dist = self.corridor.station_x_m - self.x_curr
            if (rem_dist <= 0.25 and self.v_curr <= 0.02 and self.k_elapsed > 80) or (
                self.t_curr >= self.corridor.deadline_s + 95.0
            ) or (self.shaper.in_landing and self.v_curr <= 0.005 and rem_dist <= 0.80):
                done = True
                break

        if shield_intervened:
            macro_reward -= 0.15 * shield_diff

        if done:
            sched_err = abs(self.t_curr - self.corridor.deadline_s)
            stop_err = abs(self.x_curr - self.corridor.station_x_m)
            terminal_rew = (
                35.0 * float(np.exp(-0.30 * sched_err))
                + 20.0 * float(np.exp(-3.0 * stop_err))
                - 1.50 * sched_err
                - 5.0 * stop_err
            )
            macro_reward += terminal_rew

        info = {
            "t_start": self.corridor.start_time_s,
            "t_deadline": self.corridor.deadline_s,
            "t_curr": self.t_curr,
            "x_curr": self.x_curr,
            "v_curr": self.v_curr,
            "schedule_error_s": self.t_curr - self.corridor.deadline_s,
            "stop_error_m": self.x_curr - self.corridor.station_x_m,
            "energy_kwh": self.energy_kwh,
            "micro_logs": micro_logs,
        }
        return self._get_obs(), float(macro_reward), done, info


# ===========================================================================
# 2. Pure Model-Free PPO + GAE Actor-Critic Policy & Overnight Trainer
# ===========================================================================
class CosineTrajectoryRLPolicy(nn.Module):
    def __init__(self, obs_dim: int = 15, act_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )
        # Zero initialization on output layer so Epoch 1 starts at a = [0, 0, 0] (pure v_naive!)
        nn.init.zeros_(self.actor_mean[-2].weight)
        nn.init.zeros_(self.actor_mean[-2].bias)

        self.log_std = nn.Parameter(torch.full((act_dim,), -0.85, dtype=torch.float32))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_action_and_value(self, obs: np.ndarray, deterministic: bool = True):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        mean = self.actor_mean(obs_t)
        val = self.critic(obs_t).squeeze(-1)
        if deterministic:
            return mean.detach().numpy(), 0.0, float(val.item())
        std = torch.exp(torch.clamp(self.log_std, -2.2, -0.35))
        dist = Normal(mean, std)
        raw_act = dist.sample()
        logp = dist.log_prob(raw_act).sum(dim=-1)
        return raw_act.detach().numpy(), float(logp.item()), float(val.item())

    def evaluate_actions(self, obs_t: torch.Tensor, act_t: torch.Tensor):
        mean = self.actor_mean(obs_t)
        std = torch.exp(torch.clamp(self.log_std, -2.2, -0.35))
        dist = Normal(mean, std)
        logp = dist.log_prob(act_t).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(obs_t).squeeze(-1)
        return logp, entropy, values


def train_pure_ppo_cosine_pilot(
    base_corridor: RailroadCorridor,
    emulator: LearnedLinearizationModel,
    epochs: int = 3000,
    lr: float = 7.5e-4,
    gamma: float = 0.98,
    gae_lambda: float = 0.94,
    clip_eps: float = 0.20,
    wind_mode: str = "disjoint",
    seed: int = 42,
    checkpoint_prefix: str = "ppo_cosine_pilot",
) -> tuple[CosineTrajectoryRLPolicy, list[dict]]:
    """
    100% Pure Model-Free Clipped PPO + GAE Trainer (Zero Supervised Action Targets).
    Designed for 3,000+ epochs overnight training across randomized 3-station, ~16 km
    journeys with passenger weight changes and disjoint spurious wind events.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = MPCTransitionRailroadEnv(corridor=base_corridor, traj_shield=None, emulator=emulator)
    policy = CosineTrajectoryRLPolicy(obs_dim=env.obs_dim, act_dim=env.act_dim, hidden_dim=64)
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.12)

    val_scen = generate_multi_station_scenario(seed=seed + 777, n_legs=3, wind_mode=wind_mode, allow_infeasible_breaks=False)
    history = []
    best_score = float("inf")
    best_state = copy.deepcopy(policy.state_dict())
    t0 = time.perf_counter()

    log_interval = 25 if epochs >= 500 else 10

    for epoch in range(1, epochs + 1):
        # 1. Deterministic evaluation on the 3-station validation journey
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            with torch.no_grad():
                leg_errs, leg_stops, det_ret = [], [], 0.0
                for leg in val_scen["legs"]:
                    det_obs = env.reset(corridor_override=leg["corridor"], wind_profile=leg["wind_profile"])
                    det_done = False
                    while not det_done:
                        det_act, _, _ = policy.get_action_and_value(det_obs, deterministic=True)
                        det_obs, r_step, det_done, det_info = env.step(det_act, record_micro=False)
                        det_ret += r_step
                    leg_errs.append(abs(det_info["schedule_error_s"]))
                    leg_stops.append(abs(det_info["stop_error_m"]))
                det_sched_err = float(np.mean(leg_errs))
                det_stop_err = float(np.mean(leg_stops))

            score = det_sched_err + 2.0 * det_stop_err
            if score <= best_score:
                best_score = score
                best_state = copy.deepcopy(policy.state_dict())
                torch.save(best_state, f"best_{checkpoint_prefix}.pt")

        history.append({
            "epoch": epoch,
            "sched_err": det_sched_err,
            "stop_err": det_stop_err,
            "return": det_ret,
        })

        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            elapsed_min = (time.perf_counter() - t0) / 60.0
            print(
                f"  [Pure Model-Free PPO | 3-Station ~16km] Epoch {epoch:4d}/{epochs} | "
                f"Val |Arrival - Deadline|: {det_sched_err:5.2f} s (Best: {best_score:5.2f}) | "
                f"Stop Err: {det_stop_err:5.3f} m | Return: {det_ret:6.2f} | Elapsed: {elapsed_min:5.2f} min",
                flush=True,
            )

        if epoch % 250 == 0:
            torch.save(policy.state_dict(), f"latest_{checkpoint_prefix}.pt")

        # 2. Collect on-policy stochastic rollouts across a randomized 3-station scenario (3 legs)
        obs_buf, act_buf, logp_buf, adv_buf, ret_buf = [], [], [], [], []
        scen = generate_multi_station_scenario(
            seed=seed * 1000 + epoch * 17, n_legs=3, wind_mode=wind_mode, allow_infeasible_breaks=True
        )

        for leg in scen["legs"]:
            obs = env.reset(corridor_override=leg["corridor"], wind_profile=leg["wind_profile"])
            done = False
            ep_o, ep_a, ep_lp, ep_r, ep_v = [], [], [], [], []

            while not done:
                act, lp, val = policy.get_action_and_value(obs, deterministic=False)
                next_obs, rew, done, _ = env.step(act, record_micro=False)
                ep_o.append(obs)
                ep_a.append(act)
                ep_lp.append(lp)
                ep_r.append(rew / 20.0)  # Scale rewards for critic stability
                ep_v.append(val)
                obs = next_obs

            # Compute GAE(gamma, gae_lambda) for this station leg
            T_steps = len(ep_r)
            advs = np.zeros(T_steps, dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(T_steps)):
                next_val = 0.0 if t == T_steps - 1 else ep_v[t + 1]
                delta = ep_r[t] + gamma * next_val - ep_v[t]
                last_gae = delta + gamma * gae_lambda * last_gae
                advs[t] = last_gae
            rets = advs + np.asarray(ep_v, dtype=np.float32)

            obs_buf.extend(ep_o)
            act_buf.extend(ep_a)
            logp_buf.extend(ep_lp)
            adv_buf.extend(advs)
            ret_buf.extend(rets)

        # 3. Clipped PPO Surrogate Optimization
        obs_t = torch.as_tensor(np.asarray(obs_buf, dtype=np.float32))
        act_t = torch.as_tensor(np.asarray(act_buf, dtype=np.float32))
        old_lp_t = torch.as_tensor(np.asarray(logp_buf, dtype=np.float32))
        adv_t = torch.as_tensor(np.asarray(adv_buf, dtype=np.float32))
        ret_t = torch.as_tensor(np.asarray(ret_buf, dtype=np.float32))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        ent_coef = 0.008 * max(0.15, 1.0 - float(epoch) / float(epochs))
        for _ in range(4):
            new_lp_t, entropy_t, values_t = policy.evaluate_actions(obs_t, act_t)
            ratio = torch.exp(torch.clamp(new_lp_t - old_lp_t, -10.0, 10.0))
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(values_t, ret_t)
            entropy_loss = -entropy_t.mean()
            loss = actor_loss + 0.5 * critic_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        scheduler.step()

    policy.load_state_dict(best_state)
    policy.eval()
    return policy, history


# ===========================================================================
# 3. Multi-Station Episode Evaluator (Chains All 3 Landings & Dwell Stops)
# ===========================================================================
def evaluate_multi_station_journey(
    label: str,
    scenario: dict,
    mode: str,
    rl_policy: CosineTrajectoryRLPolicy | None = None,
    shield_mode: str | None = None,
    emulator: LearnedLinearizationModel | None = None,
    record_traj: bool = False,
    zero_wind_override: bool = False,
) -> dict:
    leg_sched_errs, leg_stop_errs, leg_land_decels, leg_peak_accels = [], [], [], []
    tot_energy_kwh = 0.0
    tot_sh_steps, tot_sh_ints = 0, 0
    prop_totals = {"speed_limit": 0, "smoothness": 0, "deadline": 0, "mpc_failure": 0}
    all_step_us = []
    chained_micro = []

    cum_x_offset = 0.0
    actual_clock_t = float(scenario["legs"][0]["t_start"])

    for leg in scenario["legs"]:
        cor_leg = copy.deepcopy(leg["corridor"])
        cor_leg.start_time_s = actual_clock_t
        wind_prof = np.zeros_like(leg["wind_profile"]) if zero_wind_override else leg["wind_profile"]

        sh_obj = (
            GoodallTrajectoryShield(cor_leg, emulator, TrajectoryShieldConfig(mode=shield_mode))
            if shield_mode is not None
            else None
        )
        env = MPCTransitionRailroadEnv(corridor=cor_leg, traj_shield=sh_obj, emulator=emulator)
        obs = env.reset(corridor_override=cor_leg, wind_profile=wind_prof)
        v_feas_init, _ = env.compute_feasible_deadline_pace(1.00, cor_leg.comfort_braking_decel)

        done = False
        leg_micro = []

        while not done:
            if mode == "RL_LEARNED":
                assert rl_policy is not None
                with torch.no_grad():
                    act, _, _ = rl_policy.get_action_and_value(obs, deterministic=True)
                mpc_gain = 0.95
            elif mode == "MAX_SPEED_GREEDY":
                act = np.array([1.0, 1.0, 0.0], dtype=np.float32)
                mpc_gain = 0.95
            elif mode == "NAIVE_NO_OVERSHOOT":
                act = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                mpc_gain = 0.95
            elif mode == "STATIC_NO_WIND_REPLAN":
                _, v_naive_curr = env.compute_feasible_deadline_pace(1.00, cor_leg.comfort_braking_decel)
                a0_static = float(np.clip((v_feas_init - v_naive_curr) / 3.80, -1.0, 1.0))
                act = np.array([a0_static, 0.0, 0.0], dtype=np.float32)
                mpc_gain = 0.20
            else:
                raise ValueError(f"Unknown mode: {mode}")

            obs, _, done, info = env.step(act, record_micro=True, mpc_wind_comp_gain=mpc_gain)
            leg_micro.extend(info["micro_logs"])

        leg_x = np.array([m["x"] for m in leg_micro])
        leg_acc = np.array([m["accel"] for m in leg_micro])
        all_step_us.extend([m["step_us"] for m in leg_micro])

        app_mask = leg_x >= (cor_leg.station_x_m - 600.0)
        peak_dec = float(-np.min(leg_acc[app_mask])) if np.any(app_mask) else 0.0
        leg_land_decels.append(peak_dec)
        leg_peak_accels.append(float(np.max(leg_acc)))
        leg_sched_errs.append(abs(info["schedule_error_s"]))
        leg_stop_errs.append(abs(info["stop_error_m"]))
        tot_energy_kwh += info["energy_kwh"]

        if sh_obj is not None:
            tot_sh_steps += sh_obj.total_steps
            tot_sh_ints += sh_obj.total_interventions
            for k_p in prop_totals:
                prop_totals[k_p] += sh_obj.property_counts[k_p]

        if record_traj:
            for m in leg_micro:
                m_copy = dict(m)
                m_copy["x"] = m["x"] + cum_x_offset
                m_copy["x_brake"] = m["x_brake"] + cum_x_offset
                chained_micro.append(m_copy)

            if leg["dwell_s"] > 0.0:
                n_dwell = int(leg["dwell_s"] / cor_leg.dt)
                for d_i in range(1, n_dwell + 1):
                    chained_micro.append({
                        "t": info["t_curr"] + d_i * cor_leg.dt,
                        "x": cum_x_offset + cor_leg.station_x_m,
                        "v": 0.0,
                        "v_emu": 0.0,
                        "v_ref0": 0.0,
                        "v_cmd_unshielded": 0.0,
                        "v_cmd_rl": 0.0,
                        "v_overshoot": 0.0,
                        "v_naive": 0.0,
                        "x_brake": cum_x_offset + cor_leg.station_x_m,
                        "u_act": 0.0,
                        "accel": 0.0,
                        "wind_true": 0.0,
                        "wind_hat": 0.0,
                        "mass_kg": leg["mass_kg"],
                        "scissor_rad": 0.011,
                        "env_rad": 0.026,
                        "regime": "stop",
                        "shield_intervened": False,
                        "violated_prop": "none",
                        "step_us": 0.0,
                    })

        cum_x_offset += cor_leg.station_x_m
        actual_clock_t = max(leg["deadline_s"] + leg["dwell_s"], info["t_curr"] + leg["dwell_s"])

    res = {
        "label": label,
        "total_length_m": scenario["total_length_m"],
        "has_infeasible_leg": scenario["has_infeasible_leg"],
        "abs_sched_err_s": float(np.mean(leg_sched_errs)),
        "abs_stop_err_m": float(np.mean(leg_stop_errs)),
        "peak_accel": float(np.max(leg_peak_accels)),
        "peak_land_decel": float(np.mean(leg_land_decels)),
        "energy_kwh": tot_energy_kwh,
        "shield_rate_pct": 100.0 * tot_sh_ints / max(1, tot_sh_steps) if shield_mode is not None else 0.0,
        "property_counts": prop_totals,
        "mean_step_us": float(np.mean(all_step_us)),
    }

    if record_traj:
        res["traj"] = {
            "t": np.array([m["t"] for m in chained_micro]),
            "x": np.array([m["x"] for m in chained_micro]),
            "v": np.array([m["v"] for m in chained_micro]),
            "v_emu": np.array([m["v_emu"] for m in chained_micro]),
            "v_ref": np.array([m["v_ref0"] for m in chained_micro]),
            "v_cmd_unshielded": np.array([m["v_cmd_unshielded"] for m in chained_micro]),
            "v_cmd": np.array([m["v_cmd_rl"] for m in chained_micro]),
            "v_overshoot": np.array([m["v_overshoot"] for m in chained_micro]),
            "v_naive": np.array([m["v_naive"] for m in chained_micro]),
            "x_brake": np.array([m["x_brake"] for m in chained_micro]),
            "u_act": np.array([m["u_act"] for m in chained_micro]),
            "accel": np.array([m["accel"] for m in chained_micro]),
            "wind_true": np.array([m["wind_true"] for m in chained_micro]),
            "wind_hat": np.array([m["wind_hat"] for m in chained_micro]),
            "mass_kg": np.array([m["mass_kg"] for m in chained_micro]),
            "scissor_rad": np.array([m["scissor_rad"] for m in chained_micro]),
            "env_rad": np.array([m["env_rad"] for m in chained_micro]),
            "regimes": np.array([m["regime"] for m in chained_micro]),
            "intervened": np.array([m["shield_intervened"] for m in chained_micro], dtype=bool),
        }
    return res


def save_dual_axis_wind_trajectory_shield_plot(
    scenario: dict,
    run_nominal_no_wind: dict,
    run_unshielded_wind: dict,
    run_shielded_wind: dict,
    out_path: str = "wind_trajectory_shield_dual_axis.png",
) -> None:
    tr_nom = run_nominal_no_wind["traj"]
    tr_unsh = run_unshielded_wind["traj"]
    tr_sh = run_shielded_wind["traj"]

    x_sh = tr_sh["x"]
    t_sh = tr_sh["t"]
    x_unique, idx_unique = np.unique(x_sh, return_index=True)
    t_unique = t_sh[idx_unique]

    def x_to_t(x_val):
        return np.interp(x_val, x_unique, t_unique)

    def t_to_x(t_val):
        return np.interp(t_val, t_unique, x_unique)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16.0, 9.6), gridspec_kw={"height_ratios": [3.0, 1.35]}, sharex=True
    )

    x_full_grid, v_lim_full, v_ceil_full = [], [], []
    offset_m = 0.0
    for leg in scenario["legs"]:
        cor: RailroadCorridor = leg["corridor"]
        xg = np.linspace(0.0, leg["leg_length_m"], 250)
        for xv in xg:
            x_full_grid.append(offset_m + xv)
            v_lim_full.append(cor.get_speed_limit(xv))
            v_ceil_full.append(cor.compute_smooth_spatial_ceiling(xv, cor.comfort_braking_decel))
        offset_m += leg["leg_length_m"]
        ax_top.axvline(
            offset_m, color="black", linestyle="--", alpha=0.55, linewidth=1.4,
            label=f"Station Stop #{leg['leg_idx']} ({offset_m:.0f}m, M={leg['mass_kg']/1000:.1f}t)"
        )

    ax_top.step(
        x_full_grid, v_lim_full, color="black", where="post", alpha=0.35, linewidth=1.3,
        label="Spatial Speed Limit v_lim(x)"
    )
    ax_top.plot(
        x_full_grid, v_ceil_full, color="dimgray", linestyle=":", linewidth=1.2,
        label="Smooth Cosine Zone Ceiling v_ceil(x)"
    )

    w_true = tr_sh["wind_true"]
    ax_top.fill_between(
        x_sh, 0.0, 18.2, where=(w_true < -0.04), color="deepskyblue", alpha=0.12,
        label="Disjoint Headwind Drag Event"
    )
    ax_top.fill_between(
        x_sh, 0.0, 18.2, where=(w_true > 0.04), color="mediumseagreen", alpha=0.12,
        label="Disjoint Tailwind Push Event"
    )

    ax_top.fill_between(
        x_sh, np.maximum(0.0, tr_sh["v_emu"] - tr_sh["env_rad"]),
        tr_sh["v_emu"] + tr_sh["env_rad"], color="mediumseagreen", alpha=0.25,
        label="Goodall Safe Envelope (Green)"
    )
    ax_top.fill_between(
        x_sh, np.maximum(0.0, tr_sh["v_emu"] - tr_sh["scissor_rad"]),
        tr_sh["v_emu"] + tr_sh["scissor_rad"], color="gold", alpha=0.55,
        label="Goodall 99% Scissor Band (Yellow)"
    )

    ax_top.plot(
        tr_nom["x"], tr_nom["v"], color="royalblue", linestyle="--", linewidth=1.9,
        label="1. Nominal Cosine Trajectory (No-Wind Reference)"
    )
    ax_top.plot(
        tr_unsh["x"], tr_unsh["v_cmd_unshielded"], color="darkorange", linestyle="-.", linewidth=1.8,
        label="2. Wind-Recomputed Dynamic Trajectory Target (Pure PPO Pilot)"
    )
    ax_top.plot(
        x_sh, tr_sh["v"], color="crimson", linestyle="-", linewidth=2.1,
        label="3. Goodall-Shielded Trajectory & MPC Actual Speed v(x, t)"
    )

    int_mask = tr_sh["intervened"]
    if np.any(int_mask):
        ax_top.scatter(
            x_sh[int_mask], tr_sh["v"][int_mask], color="red", s=16, zorder=6,
            label="Multi-Property Shield Recovery Intervention"
        )

    ax_top.set_ylim(-0.4, 18.4)
    ax_top.set_ylabel("Train Speed v (m/s)", fontsize=11, fontweight="bold")
    masses_str = " -> ".join([f"Leg{l['leg_idx']}:{l['mass_kg']/1000:.1f}t({l['passengers']}pax)" for l in scenario["legs"]])
    ax_top.set_title(
        f"Multi-Station ({scenario['total_length_m']/1000:.1f} km, 3 Landings) Wind-Adapted Cosine Trajectory & Goodall Shield\n"
        f"[Vehicle Mass Evolution: {masses_str} | Mean Leg Deadline Err = {run_shielded_wind['abs_sched_err_s']:.2f} s]",
        fontsize=11.0, fontweight="bold", pad=28,
    )
    ax_top.legend(fontsize=7.4, loc="lower left", ncol=3)
    ax_top.grid(True, alpha=0.3)

    sec_ax = ax_top.secondary_xaxis("top", functions=(x_to_t, t_to_x))
    sec_ax.set_xlabel("Clock Time t (s)  [Upper Horizontal Measure]", fontsize=10.5, fontweight="bold", color="navy")
    sec_ax.tick_params(axis="x", colors="navy", labelsize=9.5)

    ax_bot.axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
    ax_bot.fill_between(x_sh, 0.0, w_true, where=(w_true < 0.0), color="deepskyblue", alpha=0.35, label="Headwind Drag (m/s²)")
    ax_bot.fill_between(x_sh, 0.0, w_true, where=(w_true >= 0.0), color="mediumseagreen", alpha=0.35, label="Tailwind Push (m/s²)")
    ax_bot.plot(x_sh, w_true, color="navy", linewidth=1.5, label="Disjoint Wind d_wind(x)")
    ax_bot.plot(x_sh, tr_sh["u_act"], color="crimson", alpha=0.85, linewidth=1.4, label="MPC Throttle/Brake u_act(x)")

    ax_mass = ax_bot.twinx()
    ax_mass.step(x_sh, tr_sh["mass_kg"] / 1000.0, color="purple", linestyle=":", linewidth=1.8, label="Train Mass M(x) [tonnes]")
    ax_mass.set_ylabel("Train Mass (t)", color="purple", fontsize=9.5, fontweight="bold")
    ax_mass.tick_params(axis="y", labelcolor="purple")
    ax_mass.set_ylim(70.0, 100.0)

    ax_bot.set_ylabel("Wind (m/s²) / u_act", fontsize=10, fontweight="bold")
    ax_bot.set_xlabel("Cumulative Track Length x (m)  [Lower Horizontal Measure]", fontsize=11, fontweight="bold")
    ax_bot.legend(fontsize=7.8, loc="lower left", ncol=4)
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved Dual-Axis 3-Station ({scenario['total_length_m']/1000:.1f} km) Wind-Trajectory-Shield plot to: {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Overnight 3,000-Epoch Pure Model-Free PPO + Goodall Multi-Property Shield")
    ap.add_argument("--epochs", type=int, default=3000, help="Pure Model-Free PPO training epochs (default: 3000)")
    ap.add_argument("--eval-runs", type=int, default=30, help="Number of 3-station multi-km test scenarios")
    ap.add_argument("--n-legs", type=int, default=3, help="Number of station stops / legs per journey")
    ap.add_argument("--wind-mode", type=str, default="disjoint", choices=["disjoint", "continuous"])
    ap.add_argument("--emulator-steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", type=str, default="rl_cosine_mpc.png")
    ap.add_argument("--dual-plot", type=str, default="wind_trajectory_shield_dual_axis.png")
    args = ap.parse_args()

    base_corridor = RailroadCorridor()
    print("=" * 150)
    print(
        f"PURE MODEL-FREE PPO ({args.epochs} EPOCHS) | MULTI-STATION LONG CORRIDOR (~16 KM, {args.n_legs} LANDINGS, PASSENGER MASS) | "
        f"{args.eval_runs} RANDOMIZED SCENARIOS"
    )
    print("=" * 150, flush=True)

    print("\n=== 1. Training Heteroscedastic Neural Physics Emulator Across Varying Passenger Masses ===", flush=True)
    emulator, _ = train_150_step_mpc_trajectory_emulator(
        base_corridor, emulator_steps=args.emulator_steps, seed=args.seed
    )

    print(f"\n=== 2. Training 15-Feature Pure Model-Free PPO Pilot Across {args.epochs} Epochs ===", flush=True)
    rl_policy, train_hist = train_pure_ppo_cosine_pilot(
        base_corridor, emulator=emulator, epochs=args.epochs, wind_mode=args.wind_mode, seed=args.seed
    )

    method_specs = [
        ("1. Max-Speed Greedy Cosine + MPC", "MAX_SPEED_GREEDY", None),
        ("2. Naive Pace (No Cruise Overshoot) + MPC", "NAIVE_NO_OVERSHOOT", None),
        ("3. Static Plan (No Wind/Mass Replan) + MPC", "STATIC_NO_WIND_REPLAN", None),
        ("4. Pure PPO Cosine Pilot (Unshielded) + MPC", "RL_LEARNED", None),
        ("5. Goodall Multi-Prop Shield (Switch MPS)", "RL_LEARNED", "switch"),
        ("6. Goodall Multi-Prop Shield (1-Step DCBF)", "RL_LEARNED", "cbf_opt"),
        ("7. Goodall Multi-Prop Shield (DMPS Rollout)", "RL_LEARNED", "mps_rollout"),
    ]

    agg_results = {name: [] for name, _, _ in method_specs}
    sample_ep0_trajs = {}
    sample_ep0_scenario = None
    sample_ep0_no_wind = None
    multi_scen_dmps = []

    for ep in range(args.eval_runs):
        scen = generate_multi_station_scenario(
            seed=9000 + ep * 29,
            n_legs=args.n_legs,
            wind_mode=args.wind_mode,
            allow_infeasible_breaks=(ep > 0),
        )
        if ep == 0:
            sample_ep0_scenario = scen
            sample_ep0_no_wind = evaluate_multi_station_journey(
                "Nominal No-Wind", scen, "RL_LEARNED", rl_policy=rl_policy,
                shield_mode=None, emulator=emulator, record_traj=True, zero_wind_override=True
            )

        for name, mode_str, sh_mode in method_specs:
            need_traj = (ep == 0) or (sh_mode == "mps_rollout" and ep < 5)
            res = evaluate_multi_station_journey(
                label=name,
                scenario=scen,
                mode=mode_str,
                rl_policy=rl_policy,
                shield_mode=sh_mode,
                emulator=emulator,
                record_traj=need_traj,
            )
            agg_results[name].append(res)
            if ep == 0:
                sample_ep0_trajs[name] = res
            if sh_mode == "mps_rollout" and ep < 5:
                multi_scen_dmps.append(res)

    print("\n" + "=" * 150)
    print(f"SAMPLE OF THE {args.eval_runs} RANDOMIZED 3-STATION (~16 KM) SCENARIOS WITH PASSENGER WEIGHT CHANGES")
    print("-" * 150)
    print(
        f"{'Scen#':<5} | {'Total Len':>9} | {'Passenger Mass Evolution (Leg1->Leg2->Leg3)':<45} | "
        f"{'Mode':<10} | {'Mean |TimeErr|':>14} | {'Mean |StopErr|':>14} | {'Landing Decel':>13} | {'Shield%':>7}"
    )
    print("-" * 150)
    for ep_i in range(min(8, args.eval_runs)):
        r_d = agg_results["7. Goodall Multi-Prop Shield (DMPS Rollout)"][ep_i]
        scen_i = generate_multi_station_scenario(
            seed=9000 + ep_i * 29, n_legs=args.n_legs, wind_mode=args.wind_mode, allow_infeasible_breaks=(ep_i > 0)
        )
        m_str = " -> ".join([f"{l['mass_kg']/1000:.1f}t({l['passengers']}p)" for l in scen_i["legs"]])
        f_tag = "BREAK-TEST" if r_d["has_infeasible_leg"] else "FEASIBLE"
        print(
            f"{ep_i + 1:<5d} | {r_d['total_length_m']/1000:6.2f} km | {m_str:<45} | "
            f"{f_tag:<10} | {r_d['abs_sched_err_s']:12.2f} s | {r_d['abs_stop_err_m']:12.3f} m | "
            f"{r_d['peak_land_decel']:9.2f} m/s² | {r_d['shield_rate_pct']:6.1f}%"
        )
    print("=" * 150)

    print("\n" + "=" * 150)
    print(
        f"BATCH BENCHMARK ACROSS {args.eval_runs} MULTI-STATION (~16 KM, {args.n_legs} LANDINGS, PASSENGER WEIGHT & WIND) SCENARIOS (MEAN ± STD)"
    )
    print("-" * 150)
    print(
        f"{'Trajectory Pilot + Shield Method':<42} | {'Leg |Time Err| (s)':>18} | {'Feasible Only Err':>17} | "
        f"{'|Stop Err| (m)':>16} | {'Landing Decel':>13} | {'Shield %':>9} | {'Energy (kWh)':>12} | {'MPC Step':>9}"
    )
    print("-" * 150)
    for name, _, _ in method_specs:
        runs = agg_results[name]
        feas_runs = [r for r in runs if not r["has_infeasible_leg"]]
        abs_t_m = float(np.mean([r["abs_sched_err_s"] for r in runs]))
        abs_t_s = float(np.std([r["abs_sched_err_s"] for r in runs]))
        feas_t_m = float(np.mean([r["abs_sched_err_s"] for r in feas_runs])) if feas_runs else abs_t_m
        abs_x_m = float(np.mean([r["abs_stop_err_m"] for r in runs]))
        abs_x_s = float(np.std([r["abs_stop_err_m"] for r in runs]))
        dec_m = float(np.mean([r["peak_land_decel"] for r in runs]))
        sh_m = float(np.mean([r["shield_rate_pct"] for r in runs]))
        en_m = float(np.mean([r["energy_kwh"] for r in runs]))
        us_m = float(np.mean([r["mean_step_us"] for r in runs]))
        print(
            f"{name:<42} | {abs_t_m:7.2f} ± {abs_t_s:6.2f} s | {feas_t_m:14.2f} s | "
            f"{abs_x_m:6.3f} ± {abs_x_s:5.3f} m | {dec_m:9.2f} m/s² | "
            f"{sh_m:8.1f}% | {en_m:12.2f} | {us_m:6.2f} us"
        )
    print("=" * 150, flush=True)

    print_goodall_trajectory_scissor_diagnostics(sample_ep0_trajs["7. Goodall Multi-Prop Shield (DMPS Rollout)"])

    save_dual_axis_wind_trajectory_shield_plot(
        scenario=sample_ep0_scenario,
        run_nominal_no_wind=sample_ep0_no_wind,
        run_unshielded_wind=sample_ep0_trajs["4. Pure PPO Cosine Pilot (Unshielded) + MPC"],
        run_shielded_wind=sample_ep0_trajs["7. Goodall Multi-Prop Shield (DMPS Rollout)"],
        out_path=args.dual_plot,
    )

    tr_dmps = sample_ep0_trajs["7. Goodall Multi-Prop Shield (DMPS Rollout)"]["traj"]
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.0))
    plot_names = [
        "1. Max-Speed Greedy Cosine + MPC",
        "2. Naive Pace (No Cruise Overshoot) + MPC",
        "3. Static Plan (No Wind/Mass Replan) + MPC",
        "7. Goodall Multi-Prop Shield (DMPS Rollout)",
    ]
    cols = ["steelblue", "darkorange", "purple", "crimson"]
    lss = ["--", ":", "-.", "-"]

    for name, c, ls in zip(plot_names, cols, lss):
        r = sample_ep0_trajs[name]
        axes[0, 0].plot(r["traj"]["x"], r["traj"]["v"], color=c, linestyle=ls, linewidth=1.5, label=name)
    axes[0, 0].set_title("1. 3-Station Multi-Leg Speed Profile v(x) (~16 km)")
    axes[0, 0].set_xlabel("Cumulative Track Position x (m)")
    axes[0, 0].set_ylabel("Speed (m/s)")
    axes[0, 0].legend(fontsize=6.5, loc="lower left")
    axes[0, 0].grid(True, alpha=0.3)

    t_ep0 = tr_dmps["t"]
    axes[0, 1].plot(t_ep0, tr_dmps["accel"], color="crimson", linewidth=1.3, label="Longitudinal Accel a(t)")
    axes[0, 1].axhline(-0.78, color="seagreen", linestyle="--", label="Comfort Landing Target (-0.78 m/s²)")
    axes[0, 1].axhline(-0.92, color="darkorange", linestyle=":", label="Comfort Braking Limit (-0.92 m/s²)")
    axes[0, 1].set_title("2. Smooth Half-Cosine Shifts & 3 Station Landings a(t)")
    axes[0, 1].set_xlabel("Clock Time t (s)")
    axes[0, 1].set_ylabel("Acceleration (m/s²)")
    axes[0, 1].legend(fontsize=7.0, loc="lower left")
    axes[0, 1].grid(True, alpha=0.3)

    ep_cols = ["crimson", "royalblue", "seagreen", "purple", "darkorange"]
    for idx_r, (r_ep, col) in enumerate(zip(multi_scen_dmps, ep_cols)):
        tr_e = r_ep["traj"]
        axes[0, 2].plot(
            tr_e["t"], tr_e["x"], color=col, linewidth=1.6,
            label=f"Scen {idx_r+1}: L={r_ep['total_length_m']/1000:.1f}km, MeanErr={r_ep['abs_sched_err_s']:.2f}s"
        )
    axes[0, 2].set_title("3. Multi-Station Space-Time Trajectories x(t) (3 Stops)")
    axes[0, 2].set_xlabel("Clock Time t (s)")
    axes[0, 2].set_ylabel("Cumulative Distance x (m)")
    axes[0, 2].legend(fontsize=6.8, loc="upper left")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].step(t_ep0, tr_dmps["mass_kg"] / 1000.0, color="purple", linewidth=2.0, label="Vehicle Mass M(t) [tonnes]")
    ax_w2 = axes[1, 0].twinx()
    ax_w2.plot(t_ep0, tr_dmps["wind_true"], color="navy", alpha=0.6, linewidth=1.3, label="Disjoint Wind (m/s²)")
    ax_w2.set_ylabel("Wind Accel (m/s²)", color="navy", fontsize=8.5)
    axes[1, 0].set_title("4. Passenger Weight Changes at Stations & Disjoint Wind")
    axes[1, 0].set_xlabel("Clock Time t (s)")
    axes[1, 0].set_ylabel("Train Mass (tonnes)", color="purple")
    axes[1, 0].legend(fontsize=7.5, loc="upper left")
    axes[1, 0].grid(True, alpha=0.3)

    short_labels = ["Greedy", "Naive", "Static", "PPO Unsh", "PPO+Sw", "PPO+DCBF", "PPO+DMPS"]
    feas_means = [
        float(np.mean([r["abs_sched_err_s"] for r in agg_results[k] if not r["has_infeasible_leg"]]))
        for k, _, _ in method_specs
    ]
    bar_colors = ["steelblue", "darkorange", "purple", "seagreen", "salmon", "indianred", "crimson"]
    x_pos = np.arange(len(short_labels))
    axes[1, 1].bar(x_pos, feas_means, color=bar_colors, alpha=0.88)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(short_labels, fontsize=7.5)
    axes[1, 1].set_title(f"5. Mean Leg |Deadline Error| Across {args.eval_runs} 3-Station Runs")
    axes[1, 1].set_ylabel("Mean Leg |Deadline Error| (s)")
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    ep_x = [h["epoch"] for h in train_hist]
    sc_y = [h["sched_err"] for h in train_hist]
    axes[1, 2].plot(ep_x, sc_y, "r-", alpha=0.85, linewidth=1.4, label="3-Station Validation |Arrival - Deadline| (s)")
    axes[1, 2].set_title(f"6. Pure Model-Free PPO Learning Curve ({args.epochs} Epochs)")
    axes[1, 2].set_xlabel("Training Epoch")
    axes[1, 2].set_ylabel("Mean Leg Schedule Error (s)")
    axes[1, 2].legend(fontsize=7.8)
    axes[1, 2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    plt.close(fig)
    print(f"Saved 6-panel 3-Station Pure PPO benchmark plot to: {args.plot}", flush=True)


if __name__ == "__main__":
    main()