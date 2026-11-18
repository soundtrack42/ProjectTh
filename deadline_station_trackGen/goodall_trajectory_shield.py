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

_ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT_DIR), str(_ROOT_DIR / "core")):
    if _p not in sys.path:
        sys.path.append(_p)

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
    "C2_COMFORT_JERK_DAMP",    # Order-2/3 smoothness recovery: stretches cosine ramp, caps accel/brake
    "C3_DEADLINE_RECOVERY",    # Temporal recovery: minimal-energy projection onto on-time arrival cone
    "C4_ECO_REGEN_COAST",      # Energy recovery: exploits downhill gravity / pre-zone coasting & regen
)
ACTIVE_RECOVERY_CONTRACTS = (
    "C1_SPEED_SCISSOR_CLAMP",
    "C2_COMFORT_JERK_DAMP",
    "C3_DEADLINE_RECOVERY",
    "C4_ECO_REGEN_COAST",
)


class GoodallTrajectoryShield:
    def __init__(
        self,
        corridor: RailroadCorridor,
        emulator: LearnedLinearizationModel | None = None,
        config: TrajectoryShieldConfig | None = None,
        ucb_state: dict | None = None,
    ):
        self.corridor = corridor
        self.emulator = emulator
        self.cfg = config if config is not None else TrajectoryShieldConfig()
        assert self.cfg.mode in ("switch", "cbf_opt", "mps_rollout")
        self._feat_buf = np.zeros(8, dtype=np.float32)
        self._z1_buf = np.zeros(64, dtype=np.float32)
        self._z2_buf = np.zeros(64, dtype=np.float32)
        self._tmp_buf = np.zeros(64, dtype=np.float32)
        self._out_buf = np.zeros(2, dtype=np.float32)
        if self.emulator is not None:
            net = self.emulator.net
            self._emu_w1 = np.ascontiguousarray(net[0].weight.detach().numpy().T)
            self._emu_b1 = np.ascontiguousarray(net[0].bias.detach().numpy())
            self._emu_w2 = np.ascontiguousarray(net[2].weight.detach().numpy().T)
            self._emu_b2 = np.ascontiguousarray(net[2].bias.detach().numpy())
            self._emu_w3 = np.ascontiguousarray(net[4].weight.detach().numpy().T)
            self._emu_b3 = np.ascontiguousarray(net[4].bias.detach().numpy())
        else:
            self._emu_w1 = None

        # Instance- or journey-scoped Discounted-UCB contract statistics:
        if ucb_state is not None:
            self.contract_counts = ucb_state.setdefault(
                "counts", {c: 1.0 for c in CERTIFIED_CONTRACT_LIBRARY}
            )
            self.contract_scores = ucb_state.setdefault(
                "scores", {c: 0.0 for c in CERTIFIED_CONTRACT_LIBRARY}
            )
        else:
            self.contract_counts: dict[str, float] = {c: 1.0 for c in CERTIFIED_CONTRACT_LIBRARY}
            self.contract_scores: dict[str, float] = {c: 0.0 for c in CERTIFIED_CONTRACT_LIBRARY}
        self.active_contract = "C0_NOMINAL_BALANCED"
        self._pending_ucb_contract: str | None = None
        self.reset_stats()

    def reset_stats(self) -> None:
        self.total_steps: int = 0
        self.total_interventions: int = 0
        self.total_tree_triggers: int = 0
        self.total_latched_steps: int = 0
        self.contract_selections: dict[str, int] = {c: 0 for c in ACTIVE_RECOVERY_CONTRACTS}
        self._pending_ucb_contract = None
        self.sigma_tube_curr: float = self.cfg.base_sigma_mps
        self.latched_a_cap: float = 1.20
        self.latched_b_cap: float = 0.92
        self.latched_contract: str = "C0_NOMINAL_BALANCED"
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
        """Discounted-UCB contract selector update with count floor >= 1e-3 across recovery contracts."""
        eta = self.cfg.ucb_discount_eta
        for c in ACTIVE_RECOVERY_CONTRACTS:
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
        ext_f_norm: float | None = None,
        rho_air: float | None = None,
        adh_mu: float | None = None,
    ) -> tuple[float, float, float]:
        u_clamped = -1.0 if u_cmd < -1.0 else (1.0 if u_cmd > 1.0 else float(u_cmd))
        u_act_next = 0.72 * u_clamped + 0.28 * u_act_curr
        f_prop_lin = u_act_next * physics.M_eff * physics.max_traction_accel
        a_lin = (f_prop_lin - physics.B * mu_v) / physics.M_eff
        inertia_mom = 0.14 * (mu_v - v_prev)
        v_lin_raw = mu_v + self.corridor.dt * a_lin + inertia_mom
        v_lin = v_lin_raw if v_lin_raw > 0.0 else 0.0

        if self._emu_w1 is not None:
            if ext_f_norm is None:
                ext_f_norm = (
                    physics.grade_resistance(x_curr) + physics.curve_resistance(x_curr)
                ) / (0.03 * physics.M * physics.g)
            if rho_air is None:
                temp_c = float(getattr(self.corridor, "temp_celsius", 18.0))
                rho_air = 293.15 / max(250.0, 273.15 + temp_c)
            if adh_mu is None:
                adh_mu = float(getattr(self.corridor, "adhesion_mu_factor", 1.0))

            v_norm = mu_v / 15.0
            fb = self._feat_buf
            fb[0] = (v_lin - mu_v) / 1.5
            fb[1] = v_norm
            fb[2] = u_clamped
            fb[3] = ext_f_norm
            fb[4] = v_norm * v_norm
            fb[5] = (physics.M / getattr(physics, "car_scale", 1.0) - 85000.0) / 10000.0
            fb[6] = (rho_air - 1.0) * 5.0
            fb[7] = (adh_mu - 1.0) * 5.0

            z1 = self._z1_buf
            z2 = self._z2_buf
            tmp = self._tmp_buf
            out = self._out_buf

            np.dot(fb, self._emu_w1, out=z1)
            z1 += self._emu_b1
            np.negative(z1, out=tmp)
            np.exp(tmp, out=tmp)
            tmp += 1.0
            np.divide(z1, tmp, out=z1)

            np.dot(z1, self._emu_w2, out=z2)
            z2 += self._emu_b2
            np.negative(z2, out=tmp)
            np.exp(tmp, out=tmp)
            tmp += 1.0
            np.divide(z2, tmp, out=z2)

            np.dot(z2, self._emu_w3, out=out)
            out += self._emu_b3

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

    @staticmethod
    def compute_regime_and_envelope_margin_default(
        v_curr: float, v_ref_next: float, in_landing: bool
    ) -> tuple[str, float]:
        dv = v_ref_next - v_curr
        if in_landing or v_ref_next <= 3.2:
            rs = v_ref_next / 3.2
            ramp_scale = 0.0 if rs < 0.0 else (1.0 if rs > 1.0 else rs)
            margin = 0.015 + ramp_scale * (0.16 - 0.015)
            return "stop", margin
        if abs(dv) > 0.12:
            return "shift", 0.16
        if dv < -0.03:
            return "brake", 0.16
        return "cruise", 0.14

    def compute_regime_and_envelope_margin(
        self, v_curr: float, v_ref_next: float, in_landing: bool
    ) -> tuple[str, float]:
        dv = v_ref_next - v_curr
        if in_landing or v_ref_next <= 3.2:
            rs = v_ref_next / 3.2
            ramp_scale = 0.0 if rs < 0.0 else (1.0 if rs > 1.0 else rs)
            margin = self.cfg.stop_inv_tube_mps + ramp_scale * (self.cfg.brake_pincer_mps - self.cfg.stop_inv_tube_mps)
            return "stop", margin
        if abs(dv) > 0.12:
            return "shift", self.cfg.shift_pincer_mps
        if dv < -0.03:
            return "brake", self.cfg.brake_pincer_mps
        return "cruise", self.cfg.cruise_pincer_mps

    @staticmethod
    def _mean_grade_ahead(physics: NonlinearTrainPhysics, x_curr: float, station_x_m: float, horizon_m: float = 250.0) -> float:
        x1 = min(station_x_m, x_curr + 0.33 * horizon_m)
        x2 = min(station_x_m, x_curr + 0.67 * horizon_m)
        x3 = min(station_x_m, x_curr + horizon_m)
        mg = physics.M * physics.g
        return (
            physics.grade_resistance(x1)
            + physics.grade_resistance(x2)
            + physics.grade_resistance(x3)
        ) / (3.0 * mg)

    @staticmethod
    def _schedule_slack_s(env) -> float:
        dur_leg_s = max(1.0, float(env.corridor.deadline_s - env.corridor.start_time_s))
        return (
            (env.x_curr / env.corridor.station_x_m)
            - ((env.t_curr - env.corridor.start_time_s) / dur_leg_s)
        ) * dur_leg_s

    def _compute_eco_speed_limit(
        self,
        env,
        v_cmd_cand: float,
        a_start: float,
        b_land: float,
        v_pace_feasible: float,
        v_obs: float,
        v_lim_180_pre: float | None = None,
        slope_120_pre: float | None = None,
        ot_slack_pre: float | None = None,
    ) -> float | None:
        """
        C4_ECO_REGEN_COAST recovery target (only used once another property has triggered the
        shield): pre-zone and downhill coasting when positive on-track schedule slack is available.
        The "going too fast" eco upper bound itself is applied upstream of the shield by
        `env.update_zone_eco_upper_bound` and is not a shield trigger.
        Returns capped speed target if commanding excess speed above the eco upper bound,
        or None if nominal control passes.
        """
        use_cos = getattr(env.shaper, "use_cosine", True)
        shaper_ready = (env.k_elapsed >= 8) and (not env.shaper.in_landing)
        if not (shaper_ready and env.d_hat >= -0.55):
            return None

        x_rem = max(0.0, self.corridor.station_x_m - env.x_curr)
        v_lim_180 = (
            v_lim_180_pre
            if v_lim_180_pre is not None
            else self.corridor.get_speed_limit(min(self.corridor.station_x_m, env.x_curr + 180.0))
        )
        slope_120 = (
            slope_120_pre
            if slope_120_pre is not None
            else (
                self._mean_grade_ahead(env.physics, env.x_curr, self.corridor.station_x_m, 120.0)
                if hasattr(env, "physics")
                else 0.0
            )
        )
        if ot_slack_pre is not None:
            ot_slack = ot_slack_pre
        else:
            ot_fn = getattr(env, "compute_on_track_slack_s", None)
            ot_slack = ot_fn(a_start, b_land) if ot_fn is not None else self._schedule_slack_s(env)

        # 1) Pre-zone natural coasting into upcoming lower speed section
        if env.d_hat >= -0.04 and v_obs > 13.0 and v_lim_180 < v_obs - 0.8 and ot_slack > 0.8:
            v_pre_targ = max(v_lim_180 + 0.15, v_obs - 0.18)
            if v_cmd_cand > v_pre_targ + 0.05:
                return min(v_cmd_cand, v_pre_targ)

        # 2) Downhill gravity assist: disengage traction when gravity sustains pace
        if env.d_hat >= -0.04 and slope_120 < -0.004 and x_rem > 320.0 and env.u_act > 0.02 and v_obs >= v_pace_feasible + 0.05 and ot_slack > 1.0:
            v_dh_targ = max(v_pace_feasible + 0.05, v_obs - 0.14)
            if v_cmd_cand > v_dh_targ + 0.05:
                return min(v_cmd_cand, v_dh_targ)
        return None

    @staticmethod
    def _estimate_macro_energy_proxy(
        env,
        v_obs: float,
        v_leaf: float,
        na: float,
        nb: float,
        dt_win: float,
    ) -> float:
        """Fast causal physics energy proxy (kWh) over the macro horizon for UCB tree scoring."""
        avg_v = max(0.2, 0.5 * (v_obs + v_leaf))
        a_req = (v_leaf - v_obs) / max(0.5, dt_win)
        x_mid = min(env.corridor.station_x_m, env.x_curr + 0.5 * avg_v * dt_win)
        f_drag = env.physics.basic_resistance(avg_v)
        f_gc = env.physics.grade_resistance(x_mid) + env.physics.curve_resistance(x_mid)
        f_prop_req = env.physics.M_eff * (a_req - env.d_hat) + f_drag + f_gc
        if f_prop_req >= 0.0:
            u_est = min(1.0, f_prop_req / max(1.0, env.physics.max_propulsion_force(avg_v, 1.0)))
            slew_l = 0.24 * (max(0.0, a_req - 0.52) ** 2) + 0.18 * max(0.0, na - 0.78)
            eta_t = max(0.58, min(0.92, 0.92 - 0.16 * (u_est ** 2) - slew_l - 0.010 * (max(0.0, avg_v - 13.4) ** 2)))
            return ((f_prop_req / eta_t) * avg_v * dt_win) / 3.6e6
        rb_raw = 1.0 - 1.6 * max(0.0, nb - 0.70) - 0.8 * max(0.0, abs(a_req) - 0.68)
        regen_b = max(0.35, min(1.0, rb_raw))
        f_elec = min(abs(f_prop_req), env.physics.M_eff * 0.72 * regen_b) if avg_v > 1.0 else 0.0
        return -(0.46 * regen_b * f_elec * avg_v * dt_win) / 3.6e6

    def _eval_pressures_margins_scalars(
        self,
        env,
        v_cmd_cand: float,
        a_start_cand: float,
        b_land_cand: float,
        v_pace_feasible: float,
        v_ceil_pre: float | None = None,
        rho_E_pre: float | None = None,
        v_lim_180_pre: float | None = None,
        slope_120_pre: float | None = None,
        x_brk_pre: float | None = None,
        ot_slack_pre: float | None = None,
    ) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
        is_ucb_dmps = (self.cfg.mode == "mps_rollout")
        r99_cmd = self.cfg.z_sigma * self.sigma_tube_curr
        v_ceil = (
            v_ceil_pre
            if v_ceil_pre is not None
            else self.corridor.compute_smooth_spatial_ceiling(env.x_curr, b_land_cand)
        )

        # 1. Order-1 Velocity Pressure: rho_1 = (v_cmd + r99_cmd) / v_ceil(x)
        rho_1 = (v_cmd_cand + r99_cmd) / max(1.0, v_ceil + 0.001)
        h_speed = (v_ceil + 0.001) - (v_cmd_cand + r99_cmd)

        # 2. Order-2 Acceleration/Braking Smoothness Certifier:
        x_rem = max(0.0, self.corridor.station_x_m - env.x_curr)
        v_obs = float(getattr(env, "v_meas", env.v_curr))
        dv_actual = abs((v_cmd_cand if v_cmd_cand < v_ceil else v_ceil) - v_obs)
        use_cos_loc = getattr(env.shaper, "use_cosine", True)
        shaper_ready = (env.shaper.ramp_u_phase > 0.92) if use_cos_loc else (abs(env.shaper.free_a_curr) < 0.15)
        is_ramp_acc = is_ucb_dmps and (dv_actual > 1.85 and shaper_ready and v_cmd_cand >= v_obs)
        is_ramp_brk = is_ucb_dmps and (dv_actual > 1.85 and shaper_ready and v_cmd_cand < v_obs)
        a_comf_hi = (0.94 if use_cos_loc else 1.04) if is_ramp_acc else (self.corridor.comfort_accel_max * 0.99)
        b_comf_hi = (0.76 if use_cos_loc else 0.80) if is_ramp_brk else (self.corridor.comfort_braking_max * 0.99)
        rho_2 = max(a_start_cand / a_comf_hi, b_land_cand / b_comf_hi, 0.62 / max(0.30, b_land_cand))
        h_smooth = min(a_comf_hi - a_start_cand, b_comf_hi - b_land_cand, b_land_cand - 0.62) * 2.5

        # 3. Order-3 Jerk Pressure
        if use_cos_loc:
            # Exact peak jerk pi^2|dv|/(2T^2) of the S-curve the cosine shaper would execute for this candidate:
            j_cos_peak = env.shaper.predict_shift_peak_jerk(
                v_obs, (v_cmd_cand if v_cmd_cand < v_ceil else v_ceil), a_start_cand, b_land_cand, env.d_hat
            )
        else:
            min_dv_jerk = (1.50 if shaper_ready else 2.20) if is_ucb_dmps else 0.25
            if dv_actual > min_dv_jerk:
                a_rate_eff = (a_start_cand * 0.64) if (v_cmd_cand >= v_obs) else (b_land_cand * 0.74)
                j_cos_peak = (2.0 * (a_rate_eff * a_rate_eff)) / max(1.7, dv_actual)
            else:
                j_cos_peak = 0.0
        jerk_est = max(j_cos_peak, abs(env.prev_accel) / 3.0)
        rho_3 = jerk_est / self.cfg.max_jerk_mps3
        h_jerk = self.cfg.max_jerk_mps3 - jerk_est

        # 4. Predictive Eco-Coasting & Regenerative Efficiency Certifier (C4_ECO_REGEN_COAST):
        if rho_E_pre is not None:
            rho_E_base = rho_E_pre
        else:
            km_done = max(0.80, env.x_curr / 1000.0)
            mass_used_kg = getattr(env, "mass_hat_kg", self.corridor.mass_kg)
            kinetic_kwh = (0.5 * mass_used_kg * 1.08 * (v_obs * v_obs)) / 3.6e6
            dissipated_kwh = max(0.0, env.energy_kwh - kinetic_kwh)
            e_budget_kwh = (55.0 * (mass_used_kg / 1000.0) * km_done) / 1000.0
            rho_E_base = min(1.08, dissipated_kwh / max(1.0, e_budget_kwh))
        cand_slew_e = (
            0.05 * max(0.0, v_cmd_cand - max(12.2, v_pace_feasible + 0.15))
            + 0.08 * max(0.0, a_start_cand - 0.78)
            + 0.12 * max(0.0, b_land_cand - 0.70)
        )
        rho_E = min(1.25, rho_E_base + cand_slew_e)

        # Compute braking spot ahead for both pre-landing regen and deadline cone:
        if x_brk_pre is not None:
            x_brk_est = x_brk_pre
        else:
            max_brk_est = env.physics.max_braking_decel if hasattr(env, "physics") else self.corridor.max_service_braking
            x_brk_est, _, _, _ = CosineTrajectoryShaper.dynamic_cosine_braking_spot(
                self.corridor, v_obs, b_land_cand, v_scissor_rad=0.0, d_wind_hat=env.d_hat, max_braking_decel=max_brk_est
            )

        if ot_slack_pre is not None:
            ot_slack_now = ot_slack_pre
        elif is_ucb_dmps:
            ot_fn = getattr(env, "compute_on_track_slack_s", None)
            ot_slack_now = ot_fn(a_start_cand, b_land_cand) if ot_fn is not None else self._schedule_slack_s(env)
        else:
            ot_slack_now = 0.0

        # Energy is scored (rho_E feeds the Tchebycheff tree objective) but is not a safety trigger:
        # the eco "going too fast" upper bound acts upstream of the shield (env.update_zone_eco_upper_bound).
        h_energy = 0.50

        # 5. Deadline Gust-Compensation & Schedule-Cone Certifier (C3_DEADLINE_RECOVERY)
        v_max_w1_est = max(1.5, v_ceil - r99_cmd)
        is_braking_station = (x_rem < 370.0 and v_cmd_cand < v_obs - 0.50)
        min_xrem_brk = 220.0 if use_cos_loc else 330.0
        before_brake = (
            (env.x_curr > (15.0 if use_cos_loc else 80.0))
            and (x_rem > min_xrem_brk)
            and (env.x_curr < x_brk_est - (35.0 if use_cos_loc else 55.0))
            and (True if use_cos_loc else (not is_braking_station))
        )
        dv_tol = 0.30 if use_cos_loc else 0.38
        ot_check = ot_slack_now if is_ucb_dmps else 0.0
        head_trigger = (
            (env.d_hat < -0.20 and ot_check < 1.8)
            or (is_ucb_dmps and ot_check < -0.55 and v_cmd_cand < v_pace_feasible + 0.12)
        )
        if before_brake and (
            head_trigger
            or v_cmd_cand < v_pace_feasible - dv_tol
            or (env.x_curr > 400.0 and v_obs < v_pace_feasible - (0.50 if use_cos_loc else 0.58))
        ):
            head_boost = 0.28 * max(0.0, -env.d_hat) + (0.08 if ot_check < -0.55 else 0.04)
            v_req_min = min(v_max_w1_est - 0.04, v_pace_feasible + head_boost)
            dv_mismatch = max(0.0, v_req_min - v_cmd_cand)
        else:
            dv_mismatch = 0.0

        v_sq = max(36.0, v_pace_feasible ** 2)
        dt_proj_err = (x_rem * dv_mismatch) / v_sq
        if is_ucb_dmps and dv_mismatch > 0.06 and ot_check < -0.55:
            dt_proj_err = max(dt_proj_err, -ot_check + 2.80)
        rho_T = dt_proj_err / self.cfg.deadline_time_tol_s
        if (not before_brake) or dv_mismatch <= 0.0:
            h_deadline = 0.50
            rho_T = min(rho_T, 0.65)
        else:
            h_deadline = (self.cfg.deadline_time_tol_s - dt_proj_err) * (v_sq / max(300.0, x_rem))

        # 6. MPC Tracking Saturation Reference
        mpc_lag_err = abs(v_obs - env.shaper.ramp_v_goal) if env.shaper.ramp_u_phase >= 0.99 else 0.0
        is_saturated = abs(env.u_act) >= 0.97 and mpc_lag_err > 0.55
        h_mpc = -0.25 if is_saturated else 0.50

        return (
            float(h_speed),
            float(h_smooth),
            float(h_jerk),
            float(h_energy),
            float(h_deadline),
            float(h_mpc),
            float(min(1.5, rho_1)),
            float(min(1.5, rho_2)),
            float(min(1.5, rho_3)),
            float(min(1.5, rho_E)),
            float(min(1.5, rho_T)),
        )

    def evaluate_hierarchical_pressures_and_margins(
        self,
        env,
        v_cmd_cand: float,
        a_start_cand: float,
        b_land_cand: float,
        v_pace_feasible: float,
        v_ceil_pre: float | None = None,
        v_lim_280_pre: float | None = None,
        rho_E_pre: float | None = None,
        v_lim_180_pre: float | None = None,
        slope_120_pre: float | None = None,
        x_brk_pre: float | None = None,
        ot_slack_pre: float | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        h_sp, h_sm, h_jk, h_en, h_dl, h_mp, r1, r2, r3, rE, rT = self._eval_pressures_margins_scalars(
            env,
            v_cmd_cand,
            a_start_cand,
            b_land_cand,
            v_pace_feasible,
            v_ceil_pre=v_ceil_pre,
            rho_E_pre=rho_E_pre,
            v_lim_180_pre=v_lim_180_pre,
            slope_120_pre=slope_120_pre,
            x_brk_pre=x_brk_pre,
            ot_slack_pre=ot_slack_pre,
        )
        margins = {
            "speed_limit": h_sp,
            "smoothness": h_sm,
            "jerk_order3": h_jk,
            "energy": h_en,
            "deadline": h_dl,
            "mpc_failure": h_mp,
        }
        pressures = {
            "rho_1_vel": r1,
            "rho_2_acc": r2,
            "rho_3_jerk": r3,
            "rho_E_energy": rE,
            "rho_T_deadline": rT,
        }
        return margins, pressures

    @staticmethod
    def _clamp(val: float, lo: float, hi: float) -> float:
        return lo if val < lo else (hi if val > hi else float(val))

    _PROP_NAMES = (
        "speed_limit",
        "smoothness",
        "jerk_order3",
        "energy",
        "deadline",
        "mpc_failure",
    )

    def _execute_contract_backup_scalars(
        self,
        env,
        contract_name: str,
        v_cmd_task: float,
        a_start_task: float,
        b_land_task: float,
        v_pace_feasible: float,
        v_ceil_pre: float | None = None,
        v_lim_180_pre: float | None = None,
        slope_120_pre: float | None = None,
        ot_slack_pre: float | None = None,
    ) -> tuple[float, float, float]:
        r99_cmd = self.cfg.z_sigma * self.sigma_tube_curr
        v_ceil = (
            v_ceil_pre
            if v_ceil_pre is not None
            else self.corridor.compute_smooth_spatial_ceiling(env.x_curr, self.corridor.comfort_braking_decel)
        )
        v_max_w1 = max(1.5, v_ceil - r99_cmd)

        a_w2 = self._clamp(a_start_task, 0.72, self.corridor.comfort_accel_max)
        b_w2 = self._clamp(b_land_task, 0.66, self.corridor.comfort_braking_max)
        v_ppo_safe = self._clamp(v_cmd_task, 1.5, v_max_w1)

        if contract_name == "C1_SPEED_SCISSOR_CLAMP":
            return v_ppo_safe, a_w2, b_w2

        use_cos = getattr(env.shaper, "use_cosine", True)
        x_rem_c = max(0.0, self.corridor.station_x_m - env.x_curr)

        if contract_name == "C2_COMFORT_JERK_DAMP":
            a_c2 = 0.85 if env.k_elapsed <= 8 else 0.75
            b_c2 = 0.66
            if use_cos:
                v_feas_damped, _ = env.compute_feasible_deadline_pace(a_c2, b_c2)
                dv_damped = max(0.0, v_feas_damped - v_pace_feasible)
                v_target_c2 = v_ppo_safe + 0.60 * dv_damped
            else:
                v_feas_damped, _ = env.compute_feasible_deadline_pace(0.84, 0.72)
                dv_damped = max(0.0, v_feas_damped - v_pace_feasible)
                v_target_c2 = (v_ppo_safe + 0.40 * dv_damped) if x_rem_c > 450.0 else (v_ppo_safe - 0.08)
            return self._clamp(v_target_c2, 1.5, v_max_w1), a_c2, b_c2

        if contract_name == "C4_ECO_REGEN_COAST":
            v_obs_now = float(getattr(env, "v_meas", env.v_curr))
            a_eco = 0.85 if env.k_elapsed <= 8 else min(a_w2, 0.90)
            b_eco = min(b_w2, 0.72)
            v_eco_lim = self._compute_eco_speed_limit(
                env,
                v_ppo_safe,
                a_eco,
                b_eco,
                v_pace_feasible,
                v_obs_now,
                v_lim_180_pre=v_lim_180_pre,
                slope_120_pre=slope_120_pre,
                ot_slack_pre=ot_slack_pre,
            )
            if v_eco_lim is not None:
                return self._clamp(v_eco_lim, 1.5, v_max_w1), a_eco, b_eco
            return v_ppo_safe, a_eco, b_eco

        cos_buf = 0.06 if use_cos else 0.0
        v_req_anchor = v_pace_feasible + 0.28 * max(0.0, -env.d_hat) + (0.10 if env.d_hat < -0.25 else cos_buf)
        v_rec_target = max(min(v_ppo_safe, v_pace_feasible + 0.28), v_req_anchor)
        v_gust_trim = self._clamp(v_rec_target, 1.5, v_max_w1)
        a_wind_rec = self._clamp(0.88 - 0.10 * min(0.0, env.d_hat), 0.78, 0.94)
        return v_gust_trim, a_wind_rec, min(b_land_task, 0.74)

    def execute_contract_backup_policy(
        self,
        env,
        contract_name: str,
        v_cmd_task: float,
        a_start_task: float,
        b_land_task: float,
        v_pace_feasible: float,
        v_ceil_pre: float | None = None,
    ) -> np.ndarray:
        v_b, a_b, b_b = self._execute_contract_backup_scalars(
            env, contract_name, v_cmd_task, a_start_task, b_land_task, v_pace_feasible, v_ceil_pre=v_ceil_pre
        )
        return np.array([v_b, a_b, b_b], dtype=np.float64)

    @staticmethod
    def _tchebycheff_scalars(rho_T: float, rho_2: float, rho_3: float, rho_E: float) -> float:
        d_T = abs(rho_T)
        d_2 = rho_2 - 0.42 if rho_2 > 0.42 else 0.0
        d_3 = rho_3 - 0.22 if rho_3 > 0.22 else 0.0
        d_E = rho_E - 0.40 if rho_E > 0.40 else 0.0
        g_te = max(0.36 * d_T, 0.22 * d_2, 0.20 * d_3, 0.22 * d_E)
        linear_comb = 2.8 * rho_T + 2.2 * rho_2 + 2.0 * rho_3 + 2.2 * rho_E
        return -(1.5 * g_te + linear_comb)

    @staticmethod
    def _tchebycheff_multi_obj_score(p_dict: dict[str, float]) -> float:
        """
        Tchebycheff Multi-Objective Decomposition (Wang et al., Biomimetics 2026, Eq. 11)
        combined with smooth derivative-pressure payoff across (Deadline, Accel, Jerk, Energy).
        """
        return GoodallTrajectoryShield._tchebycheff_scalars(
            p_dict["rho_T_deadline"],
            p_dict["rho_2_acc"],
            p_dict["rho_3_jerk"],
            p_dict["rho_E_energy"],
        )

    def _compute_latched_action(
        self,
        env,
        v_cmd_task: float,
        a_start_task: float,
        b_land_task: float,
        v_pace_feasible: float,
        v_ceil_cached: float,
        v_obs: float,
        use_cos: bool,
        v_lim_180_pre: float | None = None,
        slope_120_pre: float | None = None,
    ) -> tuple[float, float, float]:
        x_rem_now = max(0.0, self.corridor.station_x_m - env.x_curr)
        r99_now = self.cfg.z_sigma * self.sigma_tube_curr
        v_max_latch = max(1.5, v_ceil_cached - r99_now)
        a_lat = min(a_start_task, self.latched_a_cap)
        b_lat = min(b_land_task, self.latched_b_cap)
        min_xrem_lat = 220.0 if use_cos else 330.0
        if self.latched_contract == "C3_DEADLINE_RECOVERY" and x_rem_now > min_xrem_lat and (use_cos or not (x_rem_now < 370.0 and v_cmd_task < v_obs - 0.50)):
            v_feas_lat, _ = env.compute_feasible_deadline_pace(a_lat, b_lat)
            v_floor_lat = v_feas_lat + (0.25 if env.d_hat < -0.25 else (0.10 if use_cos else 0.0))
            v_target_lat = max(v_cmd_task, min(v_max_latch, v_floor_lat))
        elif self.latched_contract == "C4_ECO_REGEN_COAST":
            v_eco_lim = self._compute_eco_speed_limit(
                env,
                v_cmd_task,
                a_lat,
                b_lat,
                v_pace_feasible,
                v_obs,
                v_lim_180_pre=v_lim_180_pre,
                slope_120_pre=slope_120_pre,
            )
            v_target_lat = v_eco_lim if v_eco_lim is not None else v_cmd_task
        else:
            v_target_lat = v_cmd_task
        v_comp_latch = self._clamp(v_target_lat, 1.5, v_max_latch)
        return v_comp_latch, a_lat, b_lat

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
            self.active_contract = "C0_NOMINAL_BALANCED"
            return v_cmd_task, a_start_task, b_land_task, False, 0.0, "none"

        use_cos = getattr(env.shaper, "use_cosine", True)
        ceil_fn = (
            self.corridor.compute_smooth_spatial_ceiling
            if use_cos
            else self.corridor.compute_free_spatial_ceiling
        )
        v_ceil_cached = ceil_fn(env.x_curr, b_land_task)
        v_obs = float(getattr(env, "v_meas", env.v_curr))
        v_obs_prev = float(getattr(env, "v_meas_prev", env.v_prev))
        km_done_now = max(0.80, env.x_curr / 1000.0)
        mass_used_now = getattr(env, "mass_hat_kg", self.corridor.mass_kg)
        kin_kwh_now = (0.5 * mass_used_now * 1.08 * (v_obs * v_obs)) / 3.6e6
        diss_kwh_now = max(0.0, env.energy_kwh - kin_kwh_now)
        e_bud_now = (55.0 * (mass_used_now / 1000.0) * km_done_now) / 1000.0
        rho_E_cached = min(1.08, diss_kwh_now / max(1.0, e_bud_now))

        is_ucb_dmps = (self.cfg.mode == "mps_rollout")
        if is_ucb_dmps:
            v_lim_180_0 = self.corridor.get_speed_limit(min(self.corridor.station_x_m, env.x_curr + 180.0))
            slope_120_0 = (
                self._mean_grade_ahead(env.physics, env.x_curr, self.corridor.station_x_m, 120.0)
                if hasattr(env, "physics")
                else 0.0
            )
            ot_fn_0 = getattr(env, "compute_on_track_slack_s", None)
            ot_slack_0 = ot_fn_0(a_start_task, b_land_task) if ot_fn_0 is not None else self._schedule_slack_s(env)
        else:
            v_lim_180_0 = None
            slope_120_0 = None
            ot_slack_0 = None

        max_brk_est = env.physics.max_braking_decel if hasattr(env, "physics") else self.corridor.max_service_braking
        x_brk_0, _, _, _ = CosineTrajectoryShaper.dynamic_cosine_braking_spot(
            self.corridor, v_obs, b_land_task, v_scissor_rad=0.0, d_wind_hat=env.d_hat, max_braking_decel=max_brk_est
        )

        h0, h1, h2, h3, h4, h5, r1, r2, r3, rE, rT = self._eval_pressures_margins_scalars(
            env,
            v_cmd_task,
            a_start_task,
            b_land_task,
            v_pace_feasible,
            v_ceil_pre=v_ceil_cached,
            rho_E_pre=rho_E_cached,
            v_lim_180_pre=v_lim_180_0,
            slope_120_pre=slope_120_0,
            x_brk_pre=x_brk_0,
            ot_slack_pre=ot_slack_0,
        )
        pk = self.peak_pressures
        if r1 > pk["rho_1_vel"]:
            pk["rho_1_vel"] = r1
        if r2 > pk["rho_2_acc"]:
            pk["rho_2_acc"] = r2
        if r3 > pk["rho_3_jerk"]:
            pk["rho_3_jerk"] = r3
        if rE > pk["rho_E_energy"]:
            pk["rho_E_energy"] = rE
        if rT > pk["rho_T_deadline"]:
            pk["rho_T_deadline"] = rT

        margins_tuple = (h0, h1, h2, h3, h4, h5)
        worst_idx = 0
        min_margin = h0
        if h1 < min_margin:
            min_margin, worst_idx = h1, 1
        if h2 < min_margin:
            min_margin, worst_idx = h2, 2
        if h3 < min_margin:
            min_margin, worst_idx = h3, 3
        if h4 < min_margin:
            min_margin, worst_idx = h4, 4
        if h5 < min_margin:
            min_margin, worst_idx = h5, 5

        if min_margin >= 0.0:
            self.active_contract = "C0_NOMINAL_BALANCED"
            self.latch_steps_rem = 0
            self._pending_ucb_contract = None
            return v_cmd_task, a_start_task, b_land_task, False, 0.0, "none"

        worst_prop = self._PROP_NAMES[worst_idx]
        b_key_0 = round(b_land_task, 6)
        v_ceil_cache: dict[float, float] = {b_key_0: v_ceil_cached}
        x_brk_cache: dict[float, float] = {b_key_0: x_brk_0}

        # Short Zeno anti-chatter dwell (<= 2 macro steps) only while violation persists:
        if is_ucb_dmps and self.latch_steps_rem > 0:
            v_lat, a_lat, b_lat = self._compute_latched_action(
                env,
                v_cmd_task,
                a_start_task,
                b_land_task,
                v_pace_feasible,
                v_ceil_cached,
                v_obs,
                use_cos,
                v_lim_180_pre=v_lim_180_0,
                slope_120_pre=slope_120_0,
            )
            bk_lat = round(b_lat, 6)
            vc_lat = v_ceil_cache.get(bk_lat)
            if vc_lat is None:
                vc_lat = ceil_fn(env.x_curr, b_lat)
                v_ceil_cache[bk_lat] = vc_lat
            xb_lat = x_brk_cache.get(bk_lat)
            if xb_lat is None:
                xb_lat, _, _, _ = CosineTrajectoryShaper.dynamic_cosine_braking_spot(
                    self.corridor, v_obs, b_lat, v_scissor_rad=0.0, d_wind_hat=env.d_hat, max_braking_decel=max_brk_est
                )
                x_brk_cache[bk_lat] = xb_lat
            lh0, lh1, lh2, lh3, lh4, lh5, _, _, _, _, _ = self._eval_pressures_margins_scalars(
                env,
                v_lat,
                a_lat,
                b_lat,
                v_pace_feasible,
                v_ceil_pre=vc_lat,
                rho_E_pre=rho_E_cached,
                v_lim_180_pre=v_lim_180_0,
                slope_120_pre=slope_120_0,
                x_brk_pre=xb_lat,
                ot_slack_pre=ot_slack_0,
            )
            if min(lh0, lh1, lh2, lh3, lh4, lh5) >= -1e-4:
                self.latch_steps_rem -= 1
                self.total_latched_steps += 1
                self.active_contract = self.latched_contract
                diff_lat = abs(v_lat - v_cmd_task) + abs(a_lat - a_start_task) + abs(b_lat - b_land_task)
                if diff_lat > 1e-3:
                    self.total_interventions += 1
                    self.property_counts[worst_prop] += 1
                    if self.latched_contract in self.contract_selections:
                        self.contract_selections[self.latched_contract] += 1
                    self._pending_ucb_contract = self.latched_contract
                    return v_lat, a_lat, b_lat, True, diff_lat, worst_prop
                return v_lat, a_lat, b_lat, False, 0.0, "none"

        self.total_interventions += 1
        self.property_counts[worst_prop] += 1

        if self.cfg.mode == "switch":
            r99_now = self.cfg.z_sigma * self.sigma_tube_curr
            v_max_w1 = max(1.5, v_ceil_cached - r99_now - 0.02)
            v_sw = self._clamp(v_pace_feasible + 0.28 * max(0.0, -env.d_hat) + 0.06, 1.5, v_max_w1)
            a_nom = float(getattr(env, "a_start_center", 0.96))
            diff = abs(v_sw - v_cmd_task) + abs(a_nom - a_start_task) + abs(0.78 - b_land_task)
            return v_sw, a_nom, 0.78, True, diff, worst_prop

        if self.cfg.mode == "cbf_opt":
            r99_now = self.cfg.z_sigma * self.sigma_tube_curr
            v_max_w1 = max(1.5, v_ceil_cached - r99_now - 0.02)
            v_cbf_fb = self._clamp(v_pace_feasible + 0.30 * max(0.0, -env.d_hat) + 0.08, 1.5, v_max_w1)
            a_cbf_fb = min(a_start_task, 0.92)
            b_cbf_fb = min(b_land_task, 0.76)
            lo, hi = 0.0, 1.0
            for _ in range(4):
                mid = 0.5 * (lo + hi)
                om = 1.0 - mid
                vm = om * v_cmd_task + mid * v_cbf_fb
                am = om * a_start_task + mid * a_cbf_fb
                bm = om * b_land_task + mid * b_cbf_fb
                ch0, ch1, ch2, ch3, ch4, ch5, _, _, _, _, _ = self._eval_pressures_margins_scalars(
                    env, vm, am, bm, v_pace_feasible, v_ceil_pre=v_ceil_cached,
                    rho_E_pre=rho_E_cached, v_lim_180_pre=v_lim_180_0, slope_120_pre=slope_120_0,
                )
                c_tuple = (ch0, ch1, ch2, ch3, ch4, ch5)
                if c_tuple[worst_idx] >= -1e-4 and ch0 >= -1e-4:
                    hi = mid
                else:
                    lo = mid
            oh = 1.0 - hi
            vs = oh * v_cmd_task + hi * v_cbf_fb
            as_ = oh * a_start_task + hi * a_cbf_fb
            bs = oh * b_land_task + hi * b_cbf_fb
            diff = abs(vs - v_cmd_task) + abs(as_ - a_start_task) + abs(bs - b_land_task)
            return vs, as_, bs, True, diff, worst_prop

        # ======================================================================================
        # Mode 3 ("mps_rollout"): Safe Contract Searching Tree (Zhao 2024 Alg. 1, Eqs. 9-10)
        # + Tchebycheff Multi-Objective Evaluation (Wang et al. 2026 Eq. 11)
        # + Forward Robust Reachable Tube Pruning over the 8-Step Macro Window (Sun et al. 2024)
        # ======================================================================================
        prop_to_contract = (
            "C1_SPEED_SCISSOR_CLAMP",
            "C2_COMFORT_JERK_DAMP",
            "C2_COMFORT_JERK_DAMP",
            "C4_ECO_REGEN_COAST",
            "C3_DEADLINE_RECOVERY",
            "C2_COMFORT_JERK_DAMP",
        )
        primary_contract = prop_to_contract[worst_idx]
        self.active_contract = primary_contract
        fb_v, fb_a, fb_b = self._execute_contract_backup_scalars(
            env,
            primary_contract,
            v_cmd_task,
            a_start_task,
            b_land_task,
            v_pace_feasible,
            v_ceil_pre=v_ceil_cached,
            v_lim_180_pre=v_lim_180_0,
            slope_120_pre=slope_120_0,
            ot_slack_pre=ot_slack_0,
        )

        # Evaluate ALL 4 active recovery contracts so UCB chooses across the full library:
        cand_contracts: list[tuple[str, float, float, float]] = []
        for c_cand in ACTIVE_RECOVERY_CONTRACTS:
            if c_cand == primary_contract:
                cand_contracts.append((c_cand, fb_v, fb_a, fb_b))
            else:
                cv, ca, cb = self._execute_contract_backup_scalars(
                    env,
                    c_cand,
                    v_cmd_task,
                    a_start_task,
                    b_land_task,
                    v_pace_feasible,
                    v_ceil_pre=v_ceil_cached,
                    v_lim_180_pre=v_lim_180_0,
                    slope_120_pre=slope_120_0,
                    ot_slack_pre=ot_slack_0,
                )
                cand_contracts.append((c_cand, cv, ca, cb))

        total_n = max(1.0, sum(self.contract_counts[c] for c in ACTIVE_RECOVERY_CONTRACTS))
        log_total_n = math.log(1.0 + total_n)
        best_v, best_a, best_b = fb_v, fb_a, fb_b
        best_tree_return = -float("inf")
        best_c_name = primary_contract
        gamma_tree = 0.90

        rem_macro_steps = max(2, self.corridor.macro_steps - (env.k_elapsed % self.corridor.macro_steps))
        dt_horizon_window = rem_macro_steps * self.corridor.dt
        gam_pow = self.cfg.gamma_sigma ** rem_macro_steps
        ext_f_0 = (
            env.physics.grade_resistance(env.x_curr) + env.physics.curve_resistance(env.x_curr)
        ) / (0.03 * env.physics.M * env.physics.g)
        temp_c_0 = float(getattr(self.corridor, "temp_celsius", 18.0))
        rho_air_0 = 293.15 / max(250.0, 273.15 + temp_c_0)
        adh_mu_0 = float(getattr(self.corridor, "adhesion_mu_factor", 1.0))

        u_raw_now = (v_cmd_task - v_obs) * 0.45
        u_est_now = -1.0 if u_raw_now < -1.0 else (1.0 if u_raw_now > 1.0 else u_raw_now)
        _, _, sig_step_learned = self.predict_emulator_step_with_uncertainty(
            env.physics,
            env.x_curr,
            v_obs,
            v_obs_prev,
            env.u_act,
            u_est_now,
            0.0,
            d_wind_hat=env.d_hat,
            ext_f_norm=ext_f_0,
            rho_air=rho_air_0,
            adh_mu=adh_mu_0,
        )
        var_leaf = (
            gam_pow * (self.sigma_tube_curr ** 2)
            + ((1.0 - gam_pow) / (1.0 - self.cfg.gamma_sigma)) * (sig_step_learned ** 2)
        )
        r99_leaf = self.cfg.z_sigma * math.sqrt(max(1e-8, var_leaf))
        v_pace_sq = max(36.0, v_pace_feasible ** 2)

        v_ceil_leaf_cache: dict[tuple[float, float], float] = {}
        node_eval_cache: dict[tuple[float, float, float], tuple[bool, float]] = {}

        for c_name, bv, ba, bb in cand_contracts:
            if worst_idx == 3 and c_name == "C3_DEADLINE_RECOVERY":
                continue
            c_cnt = max(1e-3, self.contract_counts[c_name])
            ucb_bonus = (
                self.contract_scores[c_name] / c_cnt
                + self.cfg.ucb_exploration_beta * math.sqrt(log_total_n / c_cnt)
            )

            for lam in (0.55, 0.85, 1.0):
                om_l = 1.0 - lam
                nv = om_l * v_cmd_task + lam * bv
                na = om_l * a_start_task + lam * ba
                nb = om_l * b_land_task + lam * bb
                nkey = (round(nv, 5), round(na, 5), round(nb, 5))
                cached_node = node_eval_cache.get(nkey)
                if cached_node is None:
                    b_key = round(nb, 6)
                    vc_b = v_ceil_cache.get(b_key)
                    if vc_b is None:
                        vc_b = ceil_fn(env.x_curr, nb)
                        v_ceil_cache[b_key] = vc_b
                    xb_b = x_brk_cache.get(b_key)
                    if xb_b is None:
                        xb_b, _, _, _ = CosineTrajectoryShaper.dynamic_cosine_braking_spot(
                            self.corridor,
                            v_obs,
                            nb,
                            v_scissor_rad=0.0,
                            d_wind_hat=env.d_hat,
                            max_braking_decel=max_brk_est,
                        )
                        x_brk_cache[b_key] = xb_b

                    nh0, nh1, nh2, nh3, nh4, nh5, _, nr2, nr3, nrE, nrT = self._eval_pressures_margins_scalars(
                        env,
                        nv,
                        na,
                        nb,
                        v_pace_feasible,
                        v_ceil_pre=vc_b,
                        rho_E_pre=rho_E_cached,
                        v_lim_180_pre=v_lim_180_0,
                        slope_120_pre=slope_120_0,
                        x_brk_pre=xb_b,
                        ot_slack_pre=ot_slack_0,
                    )
                    nh_tuple = (nh0, nh1, nh2, nh3, nh4, nh5)
                    if nh0 < -1e-4 or nh1 < -1e-4 or nh_tuple[worst_idx] < -1e-4:
                        node_eval_cache[nkey] = (False, 0.0)
                        continue

                    dv_step = nv - v_obs
                    rate_lim = na * 0.62 if dv_step >= 0.0 else nb * 0.68
                    dv_achievable = math.copysign(min(abs(dv_step), rate_lim * dt_horizon_window), dv_step)
                    v_leaf = max(0.0, v_obs + dv_achievable + 0.25 * env.d_hat * self.corridor.dt)
                    x_leaf = min(self.corridor.station_x_m, env.x_curr + 0.5 * (v_obs + v_leaf) * dt_horizon_window)
                    l_key = (round(x_leaf, 2), b_key)
                    v_ceil_leaf = v_ceil_leaf_cache.get(l_key)
                    if v_ceil_leaf is None:
                        v_ceil_leaf = ceil_fn(x_leaf, nb)
                        v_ceil_leaf_cache[l_key] = v_ceil_leaf

                    if v_leaf + r99_leaf > v_ceil_leaf + 0.02:
                        node_eval_cache[nkey] = (False, 0.0)
                        continue

                    r_si_level1 = self._tchebycheff_scalars(nrT, nr2, nr3, nrE)
                    x_rem_leaf = max(0.0, self.corridor.station_x_m - x_leaf)
                    dv_leaf = abs((nv if nv < v_ceil_leaf else v_ceil_leaf) - v_leaf)
                    a_eff_leaf = (na * 0.64) if (nv >= v_leaf) else (nb * 0.74)
                    j_leaf = (2.0 * (a_eff_leaf ** 2)) / max(1.7, dv_leaf) if dv_leaf > 0.35 else 0.0
                    rho_3_leaf = j_leaf / self.cfg.max_jerk_mps3
                    dt_proj_leaf = (x_rem_leaf * abs(nv - v_pace_feasible)) / v_pace_sq
                    rho_T_leaf = min(0.85, dt_proj_leaf / self.cfg.deadline_time_tol_s)
                    e_win_kwh = self._estimate_macro_energy_proxy(
                        env, v_obs, v_leaf, na, nb, dt_horizon_window
                    )
                    dx_win_m = max(5.0, abs(x_leaf - env.x_curr))
                    wh_tkm_win = (e_win_kwh * 1000.0) / ((max(70000.0 * self.corridor.car_scale, mass_used_now) / 1000.0) * (dx_win_m / 1000.0))
                    rho_E_leaf = min(1.35, max(0.0, 0.35 * nrE + 0.65 * (wh_tkm_win / 52.0)))
                    r_si_leaf = self._tchebycheff_scalars(
                        rho_T_leaf, nr2, rho_3_leaf, rho_E_leaf
                    )
                    base_score = r_si_level1 + gamma_tree * (0.35 * r_si_leaf)
                    node_eval_cache[nkey] = (True, base_score)
                else:
                    ok, base_score = cached_node
                    if not ok:
                        continue

                r_ex_node = base_score + 0.40 * ucb_bonus
                if r_ex_node > best_tree_return:
                    best_tree_return = r_ex_node
                    best_v, best_a, best_b = nv, na, nb
                    best_c_name = c_name

        # 2-Step Heteroscedastic Neural Emulator Reachability Verification on Top Tree Candidate:
        if best_v > v_obs - 0.05:
            mu_r, vp_r, ua_r, sig_r = v_obs, v_obs_prev, env.u_act, self.sigma_tube_curr
            x_r = env.x_curr
            for step_r in range(2):
                u_raw = (best_v - mu_r) * 0.45
                u_est = -1.0 if u_raw < -1.0 else (1.0 if u_raw > 1.0 else u_raw)
                mu_next_r, ua_r, sig_r = self.predict_emulator_step_with_uncertainty(
                    env.physics,
                    x_r,
                    mu_r,
                    vp_r,
                    ua_r,
                    u_est,
                    sig_r,
                    d_wind_hat=env.d_hat,
                    ext_f_norm=(ext_f_0 if step_r == 0 else None),
                    rho_air=rho_air_0,
                    adh_mu=adh_mu_0,
                )
                if mu_next_r + self.cfg.z_sigma * sig_r > v_ceil_cached + 0.02:
                    best_v, best_a, best_b = fb_v, fb_a, fb_b
                    best_c_name = primary_contract
                    break
                x_r += 0.5 * (mu_r + mu_next_r) * self.corridor.dt
                vp_r, mu_r = mu_r, mu_next_r

        self.active_contract = best_c_name
        self.latched_contract = best_c_name
        self.latched_a_cap, self.latched_b_cap, self.latch_steps_rem = best_a, best_b, 2
        self.total_tree_triggers += 1
        if best_c_name in self.contract_selections:
            self.contract_selections[best_c_name] += 1
        self._pending_ucb_contract = best_c_name
        diff = abs(best_v - v_cmd_task) + abs(best_a - a_start_task) + abs(best_b - b_land_task)
        return best_v, best_a, best_b, True, diff, worst_prop


def train_150_step_mpc_trajectory_emulator(
    corridor: RailroadCorridor,
    emulator_steps: int = 150,
    n_rollouts: int = 12,
    lr: float = 2.5e-3,
    seed: int = 42,
) -> tuple[LearnedLinearizationModel, list[float]]:
    rng = np.random.default_rng(seed)
    x_list, vc_list, vlin_list, u_list, ext_list, g_list, vnext_list, sdir_list = [], [], [], [], [], [], [], []
    weather_regimes = [
        (18.0, 1.00),   # Clear Dry
        (14.0, 1.00),   # Windy Dry
        (8.0, 0.86),    # Rain WetRail
        (-6.0, 0.92),   # Cold Humid
        (-3.0, 0.80),   # Snow Slush
        (38.0, 0.94),   # Heat Thermal
    ]

    for ep in range(n_rollouts):
        mass_ep = float(rng.uniform(70500.0, 96000.0))
        temp_ep, adh_ep = weather_regimes[ep % len(weather_regimes)]
        rho_air_ep = 293.15 / max(250.0, 273.15 + temp_ep)
        cwr_thermal_ep = 1.0 + 0.0045 * max(0.0, temp_ep - 25.0)
        unmodeled_drag_ep = float(rng.uniform(0.97, 1.08))

        cor_ep = copy.deepcopy(corridor)
        cor_ep.mass_kg = mass_ep
        cor_ep.temp_celsius = temp_ep
        cor_ep.adhesion_mu_factor = adh_ep

        physics = NonlinearTrainPhysics(mass_kg=mass_ep)
        mass_ratio = 85000.0 / max(65000.0, mass_ep)
        physics.max_traction_accel = cor_ep.base_max_traction_accel * mass_ratio * adh_ep
        physics.max_braking_decel = cor_ep.base_max_service_braking * mass_ratio * adh_ep

        mpc = PureLiuMPC(
            n_p=5, n_c=3, q1=160.0, q2=0.05, q3=0.35, adaptive_shift=True,
            dt=cor_ep.dt, mass_kg=mass_ep,
            max_traction_accel=physics.max_traction_accel,
            max_braking_decel=physics.max_braking_decel,
        )
        shaper = CosineTrajectoryShaper(
            cor_ep,
            use_cosine=(ep % 2 == 0),
            max_traction_accel=physics.max_traction_accel,
            max_service_braking=physics.max_braking_decel,
        )
        x_c, v_c, v_p, u_act, t_c, d_hat = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        v_target_ep = float(rng.uniform(11.8, 16.8))
        a_start_ep = float(rng.uniform(0.75, 1.15))
        b_land_ep = float(rng.uniform(0.68, 0.88))

        wind_ep, _, _ = (
            generate_disjoint_wind_events(550.0, cor_ep.dt, seed=seed + ep * 33)
            if ep % 2 == 1
            else (np.zeros(1200, dtype=np.float64), "", [])
        )
        env_feat = np.array([
            (mass_ep - 85000.0) / 10000.0,
            (rho_air_ep - 1.0) * 5.0,
            (adh_ep - 1.0) * 5.0,
        ], dtype=np.float32)

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
            f_drag = physics.basic_resistance(v_c) * rho_air_ep * unmodeled_drag_ep
            f_grade = physics.grade_resistance(x_c)
            f_curve = physics.curve_resistance(x_c) * cwr_thermal_ep
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
            g_list.append(env_feat)
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

        res_sq_detached = diff.detach() ** 2 + 1e-8
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


def print_goodall_trajectory_scissor_diagnostics(
    runs_or_dict: dict | list[dict],
    subset_label: str = "",
) -> None:
    runs = [runs_or_dict] if isinstance(runs_or_dict, dict) else [r for r in runs_or_dict if r.get("traj") is not None]
    if not runs:
        return
    n_runs = len(runs)
    v_real_all = np.concatenate([r["traj"]["v"] for r in runs])
    v_emu_all = np.concatenate([r["traj"]["v_emu"] for r in runs])
    scis_r_all = np.concatenate([r["traj"]["scissor_rad"] for r in runs])
    env_r_all = np.concatenate([r["traj"]["env_rad"] for r in runs])
    regimes_all = np.concatenate([r["traj"]["regimes"] for r in runs])
    intervened_all = np.concatenate([r["traj"]["intervened"] for r in runs])
    is_dwell_all = np.concatenate([
        r["traj"].get("is_dwell", np.zeros(len(r["traj"]["v"]), dtype=bool)) for r in runs
    ])
    active_mask = ~is_dwell_all

    v_real = v_real_all[active_mask]
    v_emu = v_emu_all[active_mask]
    scis_r = scis_r_all[active_mask]
    env_r = env_r_all[active_mask]
    regimes = regimes_all[active_mask]
    intervened = intervened_all[active_mask]

    prop_counts: dict[str, int] = {}
    for r in runs:
        for k, v in r.get("property_counts", {}).items():
            prop_counts[k] = prop_counts.get(k, 0) + int(v)

    pressures: dict[str, float] = {}
    for r in runs:
        for k, v in r.get("peak_pressures", {}).items():
            pressures[k] = max(pressures.get(k, 0.0), float(v))

    step_err = np.abs(v_real - v_emu)
    emp_rmse = float(np.sqrt(np.mean(step_err ** 2)))
    emp_p99 = float(np.percentile(step_err, 99.0))
    in_green_pct = 100.0 * float(np.mean(step_err <= env_r + 1e-3))
    tot_km = float(np.sum([r.get("total_length_m", 0.0) for r in runs])) / 1000.0

    print("\n" + "=" * 154)
    tag_txt = f" [{subset_label}]" if subset_label else ""
    print(
        f"HIERARCHICAL DIFFERENTIAL & CONTRACT SHIELD DIAGNOSTICS{tag_txt} ({n_runs} SCENARIOS, {tot_km:.1f} KM TOTAL, {len(v_real)} STEPS) | "
        f"Emu-to-Real RMSE = {emp_rmse:.4f} m/s | Emp 99th-Pct = {emp_p99:.4f} m/s | "
        f"Physical Tube Containment: {in_green_pct:.2f}%"
    )
    if prop_counts:
        print(
            f"TOTAL CONTRACT INTERVENTIONS BY PROPERTY ({n_runs} Runs) -> Speed(d=1): {prop_counts.get('speed_limit', 0)} | "
            f"SmoothAccel(d=2): {prop_counts.get('smoothness', 0)} | Jerk(d=3): {prop_counts.get('jerk_order3', 0)} | "
            f"Energy: {prop_counts.get('energy', 0)} | DeadlineCone: {prop_counts.get('deadline', 0)}"
        )
    contract_counts_tot: dict[str, int] = {}
    for r in runs:
        for k, v in r.get("contract_selections", {}).items():
            contract_counts_tot[k] = contract_counts_tot.get(k, 0) + int(v)
    if contract_counts_tot:
        print(
            f"UCB RECOVERY CONTRACT MIX ({n_runs} Runs) -> "
            f"C1(Speed): {contract_counts_tot.get('C1_SPEED_SCISSOR_CLAMP', 0)} | "
            f"C2(Smooth/Jerk): {contract_counts_tot.get('C2_COMFORT_JERK_DAMP', 0)} | "
            f"C3(Deadline): {contract_counts_tot.get('C3_DEADLINE_RECOVERY', 0)} | "
            f"C4(Eco/Gravity): {contract_counts_tot.get('C4_ECO_REGEN_COAST', 0)}"
        )
    if pressures:
        print(
            f"PEAK DERIVATIVE PRESSURES rho_d(t) <= 1.0 (Cano et al. 2026) -> "
            f"rho_1(Vel)={pressures.get('rho_1_vel', 0.0):.3f} | rho_2(Acc)={pressures.get('rho_2_acc', 0.0):.3f} | "
            f"rho_3(Jerk)={pressures.get('rho_3_jerk', 0.0):.3f} | rho_E(Energy)={pressures.get('rho_E_energy', 0.0):.3f}"
        )
    print("-" * 154)
    print(
        f"{'Regime':<10} | {'Total Steps (Mean/Run)':>22} | {'Emu-Real RMSE':>14} | {'Learned 99% (Yellow)':>21} | "
        f"{'Safe Envelope (Green)':>23} | {'Regime P99 / MaxErr':>21} | {'Interventions':>18}"
    )
    print("-" * 154)
    for reg_name in ("cruise", "shift", "brake", "stop"):
        mask = regimes == reg_name
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        r_rmse = float(np.sqrt(np.mean(step_err[mask] ** 2)))
        r_p99 = float(np.percentile(step_err[mask], 99.0))
        r_max = float(np.max(step_err[mask]))
        y_w = float(np.mean(scis_r[mask]))
        g_w = float(np.mean(env_r[mask]))
        n_int = int(np.sum(intervened[mask]))
        steps_str = f"{cnt:6d} ({cnt / max(1, n_runs):5.1f}/run)"
        err_str = f"{r_p99:.4f} / {r_max:.4f} m/s"
        print(
            f"{reg_name:<10} | {steps_str:>22} | {r_rmse:10.4f} m/s | {'±' + f'{y_w:.4f} m/s':>21} | "
            f"{'±' + f'{g_w:.4f} m/s':>23} | {err_str:>21} | "
            f"{n_int:5d} ({100.0 * n_int / max(1, cnt):5.1f}%)"
        )
    print("=" * 154, flush=True)