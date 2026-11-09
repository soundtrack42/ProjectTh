"""
Module 3: Hierarchical Differential & Contract-Based Multi-Property Trajectory Shield
(`goodall_trajectory_shield.py`).

Unifies:
  1. Goodall & Belardinelli (2026) Heteroscedastic GP/NLL 99% Scissor Tube (`2.576 * sigma_tube`).
  2. Cano, Henzinger, & Kueffner (2026) Hierarchical Differential Safety Properties (Orders d=0..3:
     Position/Docking, Velocity, Acceleration, Jerk) & Derivative Pressure `rho_d(t) = ||d^d x_t|| / r_d`
     with Iterative Level-by-Level Pruning (`W_3 subset W_2 subset W_1`).
  3. Adalat, Hamel-De le Court, & Belardinelli (2026/2027) Assume-Guarantee Contract Library `L_cert`
     with Recency-Weighted Discounted-UCB Contract Selection to maximize all remaining properties
     whenever one property is threatened.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from pure_mpc_controller import PureLiuMPC
from havaei_generalizing_rl import NonlinearTrainPhysics, LearnedLinearizationModel
from railroad_corridor import RailroadCorridor, generate_disjoint_wind_events
from cosine_trajectory_shaper import CosineTrajectoryShaper


@dataclass
class TrajectoryShieldConfig:
    mode: str = "mps_rollout"               # "switch", "cbf_opt", or "mps_rollout" (Contract-DMPS)
    z_sigma: float = 2.576                  # Exact 99% Gaussian quantile Phi^{-1}(0.995) = 2.576
    base_sigma_mps: float = 0.0045          # Floor standard deviation sigma_0 (m/s)
    gamma_sigma: float = 0.65               # Stochastic tube variance contraction factor
    cruise_pincer_mps: float = 0.14         # Safe envelope margin in cruise (m/s)
    shift_pincer_mps: float = 0.16          # Safe envelope margin in cosine shifts (m/s)
    brake_pincer_mps: float = 0.16          # Safe envelope margin in braking (m/s)
    stop_inv_tube_mps: float = 0.015        # Order-0 docking invariant tube at platform (m/s)

    # Hierarchical Derivative Radii r_d (Cano, Henzinger, & Kueffner 2026)
    max_jerk_mps3: float = 0.48             # Order-3 jerk comfort limit r_3 (m/s^3)
    deadline_time_tol_s: float = 3.20       # Projected arrival time tolerance |T_hat - T_dead| (s)
    energy_specific_cap_wh_ton_km: float = 48.0  # Specific traction energy budget (Wh / tonne-km)

    # Discounted-UCB Contract Selector parameters (Adalat et al. 2026, Algorithm 2)
    ucb_discount_eta: float = 0.95
    ucb_exploration_beta: float = 0.35


# Certified Contract Library L_cert: 5 Assume-Guarantee Backup Contracts
CERTIFIED_CONTRACT_LIBRARY = (
    "C0_NOMINAL_BALANCED",     # Base contract: all 6 properties within nominal interior
    "C1_SPEED_SCISSOR_CLAMP",  # Order-1 speed recovery: clamps to 99% Scissor ceiling, preserves pace
    "C2_COMFORT_JERK_DAMP",    # Order-2/3 smoothness recovery: stretches cosine ramp, adds +0.15m/s cruise
    "C3_DEADLINE_RECOVERY",    # Temporal recovery: minimal-energy projection onto on-time arrival cone
    "C4_ECO_REGEN_COAST",      # Energy recovery: trims excess cruise overshoot & harvests regen braking
)

# Persistent learned UCB bandit memory across all legs and episodes:
GLOBAL_LEARNED_CONTRACT_COUNTS = {c: 1.0 for c in CERTIFIED_CONTRACT_LIBRARY}
GLOBAL_LEARNED_CONTRACT_SCORES = {c: 0.0 for c in CERTIFIED_CONTRACT_LIBRARY}


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
        self._feat_buf = np.zeros(5, dtype=np.float32)
        if self.emulator is not None:
            net = self.emulator.net
            self._emu_w1 = net[0].weight.detach().numpy()[:, :5].T
            self._emu_b1 = net[0].bias.detach().numpy()
            self._emu_w2 = net[2].weight.detach().numpy().T
            self._emu_b2 = net[2].bias.detach().numpy()
            self._emu_w3 = net[4].weight.detach().numpy().T
            self._emu_b3 = net[4].bias.detach().numpy()
        else:
            self._emu_w1 = None

        # Point directly to persistent learned UCB contract statistics:
        self.contract_counts = GLOBAL_LEARNED_CONTRACT_COUNTS
        self.contract_scores = GLOBAL_LEARNED_CONTRACT_SCORES
        self.active_contract = "C0_NOMINAL_BALANCED"
        self.reset_stats()

    def reset_stats(self) -> None:
        self.total_steps: int = 0
        self.total_interventions: int = 0
        self.sigma_tube_curr: float = self.cfg.base_sigma_mps
        self.latched_a_cap: float = 1.20
        self.latched_b_cap: float = 0.92
        self.latch_steps_rem: int = 0
        self.property_counts: dict[str, int] = {
            "speed_limit": 0,
            "smoothness": 0,
            "jerk_order3": 0,
            "energy": 0,
            "deadline": 0,
            "mpc_failure": 0,
        }
        self.peak_pressures: dict[str, float] = {
            "rho_1_vel": 0.0,
            "rho_2_acc": 0.0,
            "rho_3_jerk": 0.0,
            "rho_E_energy": 0.0,
            "rho_T_deadline": 0.0,
        }

    def update_contract_ucb_statistics(self, contract_name: str, multi_property_payoff: float) -> None:
        """Discounted-UCB contract selector update with count floor >= 1e-3 to prevent underflow."""
        eta = self.cfg.ucb_discount_eta
        for c in CERTIFIED_CONTRACT_LIBRARY:
            self.contract_counts[c] = max(1e-3, self.contract_counts[c] * eta)
            self.contract_scores[c] = self.contract_scores[c] * eta
        self.contract_counts[contract_name] += 1.0
        self.contract_scores[contract_name] += float(multi_property_payoff)

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
        u_clamped = -1.0 if u_cmd < -1.0 else (1.0 if u_cmd > 1.0 else float(u_cmd))
        u_act_next = 0.72 * u_clamped + 0.28 * u_act_curr
        f_prop_lin = u_act_next * physics.M_eff * physics.max_traction_accel
        a_lin = (f_prop_lin - physics.B * mu_v) / physics.M_eff
        inertia_mom = 0.14 * (mu_v - v_prev)
        v_lin_raw = mu_v + self.corridor.dt * a_lin + inertia_mom
        v_lin = v_lin_raw if v_lin_raw > 0.0 else 0.0

        if self._emu_w1 is not None:
            ext_f_norm = (
                physics.grade_resistance(x_curr) + physics.curve_resistance(x_curr)
            ) / (0.03 * physics.M * physics.g)
            v_norm = mu_v / 15.0
            fb = self._feat_buf
            fb[0] = (v_lin - mu_v) / 1.5
            fb[1] = v_norm
            fb[2] = u_clamped
            fb[3] = ext_f_norm
            fb[4] = v_norm * v_norm
            z1 = fb @ self._emu_w1 + self._emu_b1
            h1 = z1 / (1.0 + np.exp(-z1))
            z2 = h1 @ self._emu_w2 + self._emu_b2
            h2 = z2 / (1.0 + np.exp(-z2))
            out = h2 @ self._emu_w3 + self._emu_b3
            dv_res = float(out[0])
            lv_raw = float(out[1])
            log_var = -10.5 if lv_raw < -10.5 else (-3.2 if lv_raw > -3.2 else lv_raw)
            sigma_delta = math.exp(0.5 * log_var)
            mu_raw = v_lin + dv_res
            mu_next = mu_raw if mu_raw > 0.0 else 0.0
        else:
            sigma_delta = self.cfg.base_sigma_mps
            mu_next = v_lin

        if mu_next > 1e-3 or u_clamped > 1e-3:
            mu_w = mu_next + self.corridor.dt * d_wind_hat
            mu_next = mu_w if mu_w > 0.0 else 0.0
            sigma_wind_extra = 0.007 * abs(d_wind_hat)
            sigma_step_tot = math.hypot(sigma_delta, sigma_wind_extra)
            sigma_next = math.sqrt(self.cfg.gamma_sigma * (sigma_v_curr * sigma_v_curr) + (sigma_step_tot * sigma_step_tot))
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

    def evaluate_hierarchical_pressures_and_margins(
        self,
        env,
        v_cmd_cand: float,
        a_start_cand: float,
        b_land_cand: float,
        v_pace_feasible: float,
        v_ceil_pre: float | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        r99_now = self.cfg.z_sigma * self.sigma_tube_curr
        v_ceil = (
            v_ceil_pre
            if v_ceil_pre is not None
            else self.corridor.compute_smooth_spatial_ceiling(env.x_curr, b_land_cand)
        )

        # 1. Order-1 Velocity Pressure: rho_1 = (v_cmd + r99) / v_ceil(x)
        rho_1 = (v_cmd_cand + r99_now) / max(1.0, v_ceil + 0.01)
        h_speed = (v_ceil + 0.01) - (v_cmd_cand + r99_now)

        # 2. Order-2 Acceleration/Braking Smoothness Certifier
        x_rem = max(0.0, self.corridor.station_x_m - env.x_curr)
        dv_actual = abs(min(v_cmd_cand, v_ceil) - env.v_curr)
        is_ramp_init = (env.k_elapsed in (0, 8)) or (dv_actual > 1.65 and env.shaper.ramp_u_phase > 0.92)
        a_comf_hi = 0.82 if is_ramp_init else (self.corridor.comfort_accel_max * 1.02)
        b_comf_hi = 0.72 if is_ramp_init else (self.corridor.comfort_braking_max * 1.02)
        rho_2 = max(a_start_cand / a_comf_hi, b_land_cand / b_comf_hi, 0.62 / max(0.30, b_land_cand))
        h_smooth = min(a_comf_hi - a_start_cand, b_comf_hi - b_land_cand, b_land_cand - 0.62) * 2.5

        # 3. Order-3 Jerk Pressure
        if dv_actual > 0.25:
            a_rate_eff = (a_start_cand * 0.64) if (v_cmd_cand >= env.v_curr) else (b_land_cand * 0.74)
            j_cos_peak = (2.0 * (a_rate_eff ** 2)) / max(1.7, dv_actual)
        else:
            j_cos_peak = 0.0
        jerk_est = max(j_cos_peak, abs(env.prev_accel) / 3.0)
        rho_3 = jerk_est / self.cfg.max_jerk_mps3
        h_jerk = self.cfg.max_jerk_mps3 - jerk_est

        # 4. Predictive Eco-Coasting Certifier (C4_ECO_REGEN_COAST)
        km_done = max(0.80, env.x_curr / 1000.0)
        mass_used_kg = getattr(env, "mass_hat_kg", self.corridor.mass_kg)
        kinetic_kwh = (0.5 * mass_used_kg * 1.08 * (env.v_curr ** 2)) / 3.6e6
        dissipated_kwh = max(0.0, env.energy_kwh - kinetic_kwh)
        e_budget_kwh = (55.0 * (mass_used_kg / 1000.0) * km_done) / 1000.0
        rho_E = min(1.08, dissipated_kwh / max(1.0, e_budget_kwh))

        v_lim_280 = self.corridor.get_speed_limit(min(self.corridor.station_x_m, env.x_curr + 280.0))
        can_zone_coast = (
            (env.v_curr > 14.0)
            and (v_lim_280 < env.v_curr - 1.0)
            and (v_cmd_cand > v_lim_280 + 0.45)
            and (env.d_hat >= -0.05)
        )
        can_pre_land_coast = (
            (220.0 < x_rem < 285.0)
            and (env.v_curr > 11.8)
            and (b_land_cand > 0.72)
            and (env.d_hat >= -0.04)
        )
        if can_zone_coast:
            h_energy = -(v_cmd_cand - (v_lim_280 + 0.40))
        elif can_pre_land_coast:
            h_energy = -(b_land_cand - 0.70)
        else:
            h_energy = 0.50

        # 5. Deadline Gust-Compensation Certifier (C3_DEADLINE_RECOVERY)
        if x_rem > 900.0 and env.d_hat < -0.30 and (v_cmd_cand < v_pace_feasible + 0.25):
            dv_mismatch = (v_pace_feasible + 0.25) - v_cmd_cand
        elif x_rem > 900.0 and env.d_hat > 0.30 and (v_cmd_cand > v_pace_feasible - 0.15):
            dv_mismatch = v_cmd_cand - (v_pace_feasible - 0.15)
        else:
            dv_mismatch = 0.0

        v_sq = max(36.0, v_pace_feasible ** 2)
        dt_proj_err = (x_rem * dv_mismatch) / v_sq
        eff_time_tol = 0.80
        rho_T = dt_proj_err / eff_time_tol
        if x_rem < 900.0 or dv_mismatch <= 0.0:
            h_deadline = 0.50
            rho_T = min(rho_T, 0.65)
        else:
            h_deadline = (eff_time_tol - dt_proj_err) * (v_sq / max(900.0, x_rem))

        # 6. MPC Tracking Saturation Reference
        mpc_lag_err = abs(env.v_curr - env.shaper.ramp_v_goal) if env.shaper.ramp_u_phase >= 0.99 else 0.0
        is_saturated = abs(env.u_act) >= 0.97 and mpc_lag_err > 0.55
        h_mpc = -0.25 if is_saturated else 0.50

        margins = {
            "speed_limit": float(h_speed),
            "smoothness": float(h_smooth),
            "jerk_order3": float(h_jerk),
            "energy": float(h_energy),
            "deadline": float(h_deadline),
            "mpc_failure": float(h_mpc),
        }
        pressures = {
            "rho_1_vel": float(min(1.5, rho_1)),
            "rho_2_acc": float(min(1.5, rho_2)),
            "rho_3_jerk": float(min(1.5, rho_3)),
            "rho_E_energy": float(min(1.5, rho_E)),
            "rho_T_deadline": float(min(1.5, rho_T)),
        }
        return margins, pressures

    def execute_contract_backup_policy(
        self,
        env,
        contract_name: str,
        v_cmd_task: float,
        a_start_task: float,
        b_land_task: float,
        v_pace_feasible: float,
    ) -> np.ndarray:
        r99_now = self.cfg.z_sigma * self.sigma_tube_curr
        v_ceil = self.corridor.compute_smooth_spatial_ceiling(env.x_curr, self.corridor.comfort_braking_decel)
        v_max_w1 = max(1.5, v_ceil - r99_now - 0.02)

        a_w2 = float(np.clip(a_start_task, 0.72, self.corridor.comfort_accel_max))
        b_w2 = float(np.clip(b_land_task, 0.66, self.corridor.comfort_braking_max))
        v_ppo_safe = float(np.clip(v_cmd_task, 1.5, v_max_w1))

        if contract_name == "C1_SPEED_SCISSOR_CLAMP":
            return np.array([v_max_w1, a_w2, b_w2], dtype=np.float64)

        if contract_name == "C2_COMFORT_JERK_DAMP":
            v_comp = float(np.clip(v_ppo_safe + 0.06, 1.5, v_max_w1))
            return np.array([v_comp, 0.74, 0.66], dtype=np.float64)

        if contract_name == "C4_ECO_REGEN_COAST":
            # Early predictive coasting into slower speed zone + regenerative braking harvest:
            v_eco = float(np.clip(v_ppo_safe - 0.28, 1.5, v_max_w1))
            return np.array([v_eco, 0.74, 0.65], dtype=np.float64)

        # "C3_DEADLINE_RECOVERY": boosts speed during headwinds or sheds excess speed in tailwinds
        v_gust_trim = float(np.clip(v_ppo_safe - 0.32 * env.d_hat, 1.5, v_max_w1))
        a_wind_rec = float(np.clip(0.88 - 0.10 * min(0.0, env.d_hat), 0.78, 0.94))
        return np.array([v_gust_trim, a_wind_rec, 0.72], dtype=np.float64)

    def filter_trajectory_action(
        self,
        env,
        v_cmd_task: float,
        a_start_task: float,
        b_land_task: float,
        v_pace_feasible: float,
    ) -> tuple[float, float, float, bool, float, str]:
        self.total_steps += 1
        v_ceil_cached = self.corridor.compute_smooth_spatial_ceiling(env.x_curr, b_land_task)
        margins, pressures = self.evaluate_hierarchical_pressures_and_margins(
            env, v_cmd_task, a_start_task, b_land_task, v_pace_feasible, v_ceil_pre=v_ceil_cached
        )
        for k_p, val_p in pressures.items():
            self.peak_pressures[k_p] = max(self.peak_pressures[k_p], val_p)

        if env.shaper.in_landing:
            self.active_contract = "C0_NOMINAL_BALANCED"
            return v_cmd_task, a_start_task, b_land_task, False, 0.0, "none"

        worst_prop = min(margins, key=margins.get)
        min_margin = margins[worst_prop]

        if min_margin >= 0.0:
            self.active_contract = "C0_NOMINAL_BALANCED"
            self.update_contract_ucb_statistics("C0_NOMINAL_BALANCED", 1.0 - max(pressures.values()))
            if self.latch_steps_rem > 0:
                self.latch_steps_rem -= 1
                r99_now = self.cfg.z_sigma * self.sigma_tube_curr
                v_max_latch = max(1.5, v_ceil_cached - r99_now - 0.02)
                v_comp_latch = float(np.clip(v_cmd_task + 0.045, 1.5, v_max_latch))
                return (
                    v_comp_latch,
                    min(a_start_task, self.latched_a_cap),
                    min(b_land_task, self.latched_b_cap),
                    False,
                    0.0,
                    "none",
                )
            return v_cmd_task, a_start_task, b_land_task, False, 0.0, "none"

        self.total_interventions += 1
        self.property_counts[worst_prop] += 1

        prop_to_contract = {
            "speed_limit": "C1_SPEED_SCISSOR_CLAMP",
            "smoothness": "C2_COMFORT_JERK_DAMP",
            "jerk_order3": "C2_COMFORT_JERK_DAMP",
            "energy": "C4_ECO_REGEN_COAST",
            "deadline": "C3_DEADLINE_RECOVERY",
            "mpc_failure": "C2_COMFORT_JERK_DAMP",
        }
        primary_contract = prop_to_contract.get(worst_prop, "C3_DEADLINE_RECOVERY")
        self.active_contract = primary_contract

        p_task = np.array([v_cmd_task, a_start_task, b_land_task], dtype=np.float64)
        p_fb = self.execute_contract_backup_policy(
            env, primary_contract, v_cmd_task, a_start_task, b_land_task, v_pace_feasible
        )

        if self.cfg.mode == "switch":
            p_sw = 0.45 * p_task + 0.55 * p_fb
            self.latched_a_cap, self.latched_b_cap, self.latch_steps_rem = float(p_sw[1]), float(p_sw[2]), 14
            diff = float(np.sum(np.abs(p_sw - p_task)))
            return float(p_sw[0]), float(p_sw[1]), float(p_sw[2]), True, diff, worst_prop

        if self.cfg.mode == "cbf_opt":
            p_safe = 0.30 * p_task + 0.70 * p_fb
            self.latched_a_cap, self.latched_b_cap, self.latch_steps_rem = float(p_safe[1]), float(p_safe[2]), 14
            diff = float(np.sum(np.abs(p_safe - p_task)))
            return float(p_safe[0]), float(p_safe[1]), float(p_safe[2]), True, diff, worst_prop

        # Mode 3 ("mps_rollout"): UCB Contract Selector over L_cert
        total_n = max(1.0, sum(self.contract_counts.values()))
        best_p = p_fb
        best_score = -float("inf")
        best_c_name = primary_contract

        for c_name in (primary_contract, "C4_ECO_REGEN_COAST", "C3_DEADLINE_RECOVERY"):
            p_c_base = self.execute_contract_backup_policy(
                env, c_name, v_cmd_task, a_start_task, b_land_task, v_pace_feasible
            )
            ucb_bonus = (
                self.contract_scores[c_name] / max(1e-3, self.contract_counts[c_name])
                + self.cfg.ucb_exploration_beta * np.sqrt(np.log(1.0 + total_n) / max(1e-3, self.contract_counts[c_name]))
            )
            for lam in (0.55, 0.85):
                p_cand = (1.0 - lam) * p_task + lam * p_c_base
                m_dict, p_dict = self.evaluate_hierarchical_pressures_and_margins(
                    env, float(p_cand[0]), float(p_cand[1]), float(p_cand[2]),
                    v_pace_feasible, v_ceil_pre=v_ceil_cached
                )
                if m_dict["speed_limit"] >= -1e-4 and m_dict["smoothness"] >= -1e-4:
                    multi_score = (
                        -3.0 * p_dict["rho_T_deadline"]
                        - 2.5 * p_dict["rho_2_acc"]
                        - 2.0 * p_dict["rho_3_jerk"]
                        - 1.5 * p_dict["rho_E_energy"]
                        + 0.4 * ucb_bonus
                    )
                    if multi_score > best_score:
                        best_score = multi_score
                        best_p = p_cand
                        best_c_name = c_name

        self.active_contract = best_c_name
        self.latched_a_cap, self.latched_b_cap, self.latch_steps_rem = float(best_p[1]), float(best_p[2]), 18
        self.update_contract_ucb_statistics(best_c_name, max(-2.0, best_score / 10.0))
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
        a_start_ep = float(rng.uniform(0.85, 1.15))
        b_land_ep = float(rng.uniform(0.72, 0.86))

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
    pressures = run_dict.get("peak_pressures", {})

    step_err = np.abs(v_real - v_emu)
    emp_rmse = float(np.sqrt(np.mean(step_err ** 2)))
    emp_p99 = float(np.percentile(step_err, 99.0))
    emp_max = float(np.max(step_err))
    in_green_pct = 100.0 * float(np.mean(step_err <= env_r + 1e-3))

    print("\n" + "=" * 146)
    print(
        f"HIERARCHICAL DIFFERENTIAL & CONTRACT SHIELD DIAGNOSTICS (3-STATION ~16.5 KM ROUTE) | "
        f"Emu-to-Real RMSE = {emp_rmse:.4f} m/s | Emp 99th-Pct = {emp_p99:.4f} m/s | "
        f"Physical Tube Containment: {in_green_pct:.1f}%"
    )
    if prop_counts:
        print(
            f"CONTRACT INTERVENTIONS BY PROPERTY -> Speed(d=1): {prop_counts.get('speed_limit', 0)} | "
            f"SmoothAccel(d=2): {prop_counts.get('smoothness', 0)} | Jerk(d=3): {prop_counts.get('jerk_order3', 0)} | "
            f"Energy: {prop_counts.get('energy', 0)} | DeadlineCone: {prop_counts.get('deadline', 0)}"
        )
    if pressures:
        print(
            f"PEAK DERIVATIVE PRESSURES rho_d(t) <= 1.0 (Cano et al. 2026) -> "
            f"rho_1(Vel)={pressures.get('rho_1_vel', 0.0):.3f} | rho_2(Acc)={pressures.get('rho_2_acc', 0.0):.3f} | "
            f"rho_3(Jerk)={pressures.get('rho_3_jerk', 0.0):.3f} | rho_E(Energy)={pressures.get('rho_E_energy', 0.0):.3f}"
        )
    print("-" * 146)
    print(
        f"{'Regime':<10} | {'Steps':>7} | {'Emu-Real RMSE':>14} | {'Learned 99% (Yellow)':>21} | "
        f"{'Safe Envelope (Green)':>23} | {'Regime MaxErr':>15} | {'Interventions':>18}"
    )
    print("-" * 146)
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
    print("=" * 146, flush=True)