"""
Multi-Property Derivative-Pressure PPO and Contract-Based Hierarchical Shielding
for Multi-Station Automatic Train Operation (`dynamic_railroad_mpc.py`).
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import copy
import csv
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib
import gc

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

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
from real_world_rail_weather import fetch_real_world_multi_station_scenario
from bilbao_cbtc_ato import BilbaoCBTCATOController
from goodall_trajectory_shield import (
    TrajectoryShieldConfig,
    GoodallTrajectoryShield,
    train_150_step_mpc_trajectory_emulator,
    print_goodall_trajectory_scissor_diagnostics,
)


# ===========================================================================
# 1. Multi-Station RL Environment with Regenerative Energy & Derivative Pressures
# ===========================================================================
class MPCTransitionRailroadEnv:
    def __init__(
        self,
        corridor: RailroadCorridor | None = None,
        traj_shield: GoodallTrajectoryShield | None = None,
        emulator: LearnedLinearizationModel | None = None,
        use_cosine_shaper: bool = True,
    ):
        self.corridor = corridor if corridor is not None else RailroadCorridor()
        self.traj_shield = traj_shield
        self.emulator = emulator
        self.use_cosine_shaper = bool(use_cosine_shaper)
        self.obs_dim = 15
        self.act_dim = 3
        self.wind_profile = np.zeros(1600, dtype=np.float64)
        self.mass_hat_kg = 85000.0
        self.mass_locked = False
        self.mass_samples: list[float] = []
        self.d_standstill = 0.0
        self._sync_physics_and_mpc(use_estimated_mass=False)
        self.reset()

    def _sync_physics_and_mpc(self, use_estimated_mass: bool = True, reset_shaper: bool = True) -> None:
        # True nonlinear plant uses actual passenger weight and station-measured adhesion:
        self.true_physics = NonlinearTrainPhysics(mass_kg=self.corridor.mass_kg)
        self.true_physics.max_traction_accel = self.corridor.max_traction_accel
        self.true_physics.max_braking_decel = self.corridor.max_service_braking

        # Controller (MPC + Shaper) knows station weather telemetry (adhesion_mu_factor) and online mass_hat_kg:
        ctrl_mass = self.mass_hat_kg if use_estimated_mass else 85000.0
        mass_ratio = 85000.0 / max(65000.0, ctrl_mass)
        adh_mu = getattr(self.corridor, "adhesion_mu_factor", 1.0)
        self.physics = NonlinearTrainPhysics(mass_kg=ctrl_mass)
        self.physics.max_traction_accel = self.corridor.base_max_traction_accel * mass_ratio * adh_mu
        self.physics.max_braking_decel = self.corridor.base_max_service_braking * mass_ratio * adh_mu
        if reset_shaper or not hasattr(self, "shaper"):
            self.shaper = CosineTrajectoryShaper(
                self.corridor,
                use_cosine=self.use_cosine_shaper,
                max_traction_accel=self.physics.max_traction_accel,
                max_service_braking=self.physics.max_braking_decel,
            )
        else:
            self.shaper.corridor = self.corridor
            self.shaper.sync_limits(self.physics.max_traction_accel, self.physics.max_braking_decel)
        self.mpc = PureLiuMPC(
            n_p=5, n_c=3, q1=160.0, q2=0.05, q3=0.35, adaptive_shift=True,
            dt=self.corridor.dt, mass_kg=ctrl_mass,
            max_traction_accel=self.physics.max_traction_accel,
            max_braking_decel=self.physics.max_braking_decel,
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

        if wind_profile is not None:
            self.wind_profile = np.asarray(wind_profile, dtype=np.float64).copy()
        else:
            self.wind_profile = np.zeros(1600, dtype=np.float64)

        # At station standstill (wheel brakes locked), read station wayside telemetry:
        # turnstile ticket prior mass, rail adhesion sensor, temperature, and anemometer:
        self.mass_hat_kg = float(getattr(self.corridor, "ticket_estimated_mass_kg", 85000.0))
        self.mass_prior_kg = self.mass_hat_kg
        self.mass_locked = False
        self.mass_samples = []
        # Bayesian MAP prior weight in acceleration-weighted impulse space prevents noisy anemometer spikes from blowing up M_hat
        self.mass_den_sum = 0.25
        self.mass_num_sum = self.mass_den_sum * self.mass_prior_kg
        self.d_standstill = float(getattr(self.corridor, "station_wind_sensor_mps2", 0.0))
        self.station_mu = float(getattr(self.corridor, "adhesion_mu_factor", 1.0))
        self.station_temp_c = float(getattr(self.corridor, "temp_celsius", 18.0))
        self.speed_sensor_noise_mps = float(getattr(self.corridor, "speed_sensor_noise_mps", 0.0))
        self.unmodeled_drag_factor = float(getattr(self.corridor, "unmodeled_drag_factor", 1.0))
        self.sensor_rng = np.random.default_rng(int(getattr(self.corridor, "sensor_noise_seed", 0)))
        self._sync_physics_and_mpc(use_estimated_mass=True)

        self.t_curr = float(self.corridor.start_time_s)
        self.k_elapsed = 0
        self.x_curr = 0.0
        self.v_curr = 0.0
        self.v_prev = 0.0
        self.v_meas = 0.0
        self.v_meas_prev = 0.0
        self.u_act = 0.0
        self.d_hat = self.d_standstill
        self.d_hat_prev = self.d_standstill
        self.d_err_prev = 0.0
        self.d_cusum = 0.0
        self.v_err_prev = 0.0
        self.mu_min_curr = 1.0
        self.prev_accel = 0.0
        self.energy_kwh = 0.0
        self.regen_kwh = 0.0
        self.sigma_tube = 0.0045
        self._feas_cache_k: int = -1
        self._feas_cache_map: dict[tuple[float, float], tuple[float, float]] = {}
        self.shaper.reset()
        self.mpc.reset()
        return self._get_obs()

    def compute_feasible_deadline_pace(self, a_start: float, b_land: float) -> tuple[float, float]:
        if self._feas_cache_k != self.k_elapsed:
            self._feas_cache_k = self.k_elapsed
            self._feas_cache_map.clear()
        cache_key = (round(a_start, 6), round(b_land, 6))
        cached = self._feas_cache_map.get(cache_key)
        if cached is not None:
            return cached

        x_curr = self.x_curr
        v_curr = self.v_meas
        station_x_m = self.corridor.station_x_m
        x_rem = max(0.0, station_x_m - x_curr)
        t_rem = max(self.corridor.dt, self.corridor.deadline_s - self.t_curr)
        v_naive = x_rem / t_rem
        ceil_fn = (
            self.corridor.compute_smooth_spatial_ceiling
            if self.use_cosine_shaper
            else self.corridor.compute_free_spatial_ceiling
        )
        v_local_cap = ceil_fn(x_curr, b_land)

        if x_rem <= 15.0 or self.shaper.in_landing:
            v_f = 0.0 if v_naive < 0.0 else (v_local_cap if v_naive > v_local_cap else float(v_naive))
            res_f = (v_f, float(v_naive))
            self._feas_cache_map[cache_key] = res_f
            return res_f

        _, _, _, b_eff_land = CosineTrajectoryShaper.dynamic_cosine_braking_spot(
            self.corridor,
            max(v_curr, 12.2),
            b_land,
            d_wind_hat=self.d_hat,
            max_braking_decel=self.physics.max_braking_decel,
        )
        if self.use_cosine_shaper:
            land_k = 0.5 * math.pi
            dur_k = 0.5 * math.pi
            min_ramp_s = 12.0
            n_bisect = 6
        else:
            land_k = 1.66
            dur_k = 1.78
            min_ramp_s = 14.5
            n_bisect = 9

        zones = self.corridor.speed_zones
        v_last_lim = zones[-1].v_limit_mps
        rate_up = max(0.35, a_start * (0.62 if self.use_cosine_shaper else 0.54))
        rate_dn = max(0.35, b_land * (0.65 if self.use_cosine_shaper else 0.60))
        inv_2b = 0.5 * land_k / b_eff_land
        inv_1b = land_k / b_eff_land

        def estimate_remaining_time(v_c: float) -> float:
            v_throat = v_c if v_c < v_last_lim else v_last_lim
            dx_land = (v_throat * v_throat) * inv_2b
            t_land = v_throat * inv_1b
            x_b = station_x_m - dx_land
            if x_b < x_curr + 1.0:
                x_b = x_curr + 1.0
            if x_curr >= x_b:
                return (2.0 * x_rem) / max(0.5, v_curr + 0.5)

            t_acc = 0.0
            sim_v = v_curr
            for z in zones:
                seg_a = x_curr if x_curr > z.x_start else z.x_start
                seg_b = x_b if x_b < z.x_end else z.x_end
                if seg_b <= seg_a:
                    continue
                seg_len = seg_b - seg_a
                v_seg = v_c if v_c < z.v_limit_mps else z.v_limit_mps
                dv = abs(v_seg - sim_v)
                if dv > 0.15:
                    rate = rate_up if v_seg >= sim_v else rate_dn
                    t_r = (dur_k * dv) / rate
                    t_ramp = min_ramp_s if t_r < min_ramp_s else t_r
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
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            if estimate_remaining_time(mid) > t_rem:
                lo = mid
            else:
                hi = mid
        v_feas = 0.5 * (lo + hi)
        v_res = 1.5 if v_feas < 1.5 else (v_local_cap if v_feas > v_local_cap else float(v_feas))
        res_f = (v_res, float(v_naive))
        self._feas_cache_map[cache_key] = res_f
        return res_f

    def _get_obs(self) -> np.ndarray:
        x_rem = max(0.0, self.corridor.station_x_m - self.x_curr)
        t_rem = max(self.corridor.dt, self.corridor.deadline_s - self.t_curr)
        v_lim_now = self.corridor.get_speed_limit(self.x_curr)
        v_lim_300 = self.corridor.get_speed_limit(min(self.corridor.station_x_m, self.x_curr + 300.0))

        x_brake, dx_brake, t_land, _ = CosineTrajectoryShaper.dynamic_cosine_braking_spot(
            self.corridor,
            self.v_meas,
            self.corridor.comfort_braking_decel,
            d_wind_hat=self.d_hat,
            max_braking_decel=self.physics.max_braking_decel,
        )
        v_naive = x_rem / t_rem
        t_to_brake_local = max(1.0, t_rem - t_land)
        x_to_brake_local = max(0.0, x_rem - dx_brake)
        v_kin_hint = x_to_brake_local / t_to_brake_local if x_to_brake_local > 20.0 else v_naive

        slope_iota = (
            self.physics.grade_resistance(self.x_curr) + self.physics.curve_resistance(self.x_curr)
        ) / (self.physics.M * self.physics.g)
        d_trend = (self.d_hat - self.d_hat_prev) / self.corridor.dt
        mass_norm = (self.mass_hat_kg - 85000.0) / 15000.0

        return np.array([
            self.x_curr / self.corridor.station_x_m,
            x_rem / self.corridor.station_x_m,
            t_rem / 450.0,
            self.v_meas / 18.0,
            v_naive / 18.0,
            v_kin_hint / 18.0,
            (v_kin_hint - v_naive) / 4.0,
            (self.v_meas - v_naive) / 5.0,
            v_lim_now / 18.0,
            v_lim_300 / 18.0,
            np.tanh((x_brake - self.x_curr) / 500.0),
            slope_iota * 25.0,
            self.d_hat / 0.70,
            np.tanh(d_trend / 0.15),
            mass_norm,
        ], dtype=np.float32)

    def decode_rl_action(self, action: np.ndarray) -> tuple[float, float, float, float, float]:
        a0_raw = float(action[0])
        a1_raw = float(action[1])
        a2_raw = float(action[2])
        a0 = -1.0 if a0_raw < -1.0 else (1.0 if a0_raw > 1.0 else a0_raw)
        a1 = -1.0 if a1_raw < -1.0 else (1.0 if a1_raw > 1.0 else a1_raw)
        a2 = -1.0 if a2_raw < -1.0 else (1.0 if a2_raw > 1.0 else a2_raw)
        a_start_cos = 0.96 + 0.22 * a1   # [0.74, 1.18] m/s^2
        b_land_cos = 0.78 + 0.12 * a2    # [0.66, 0.90] m/s^2
        if not self.use_cosine_shaper:
            # Learned Non-Cosine Soft-Curve Trajectory (Methods 8 & 9):
            a_s = 0.92 * a_start_cos + 0.08 * 0.88
            a_start_cos = 0.74 if a_s < 0.74 else (1.14 if a_s > 1.14 else a_s)
            b_s = 0.92 * b_land_cos + 0.08 * 0.74
            b_land_cos = 0.66 if b_s < 0.66 else (0.86 if b_s > 0.86 else b_s)

        v_feas, v_naive = self.compute_feasible_deadline_pace(a_start_cos, b_land_cos)
        is_ucb_dmps = (self.traj_shield is not None and self.traj_shield.cfg.mode == "mps_rollout")
        r99_margin = (0.15 if is_ucb_dmps else 1.0) * 2.576 * self.sigma_tube if self.traj_shield is not None else 0.0
        ceil_fn = (
            self.corridor.compute_smooth_spatial_ceiling
            if self.use_cosine_shaper
            else self.corridor.compute_free_spatial_ceiling
        )
        v_cap = max(1.5, ceil_fn(self.x_curr, b_land_cos) - r99_margin)

        # Action[0] = 0.0 corresponds to naive pace; Action[0] > 0 commands cruise velocity overshoot
        v_raw = v_naive + 3.80 * a0
        if not self.use_cosine_shaper:
            v_raw = 0.68 * v_feas + 0.32 * v_raw
        v_cmd_rl = 1.5 if v_raw < 1.5 else (v_cap if v_raw > v_cap else v_raw)
        return v_cmd_rl, a_start_cos, b_land_cos, v_feas, v_naive

    def step(
        self,
        action: np.ndarray,
        record_micro: bool = False,
        mpc_wind_comp_gain: float = 0.95,
        policy_infer_us: float = 0.0,
    ) -> tuple[np.ndarray, float, bool, dict]:
        t_macro_0 = time.thread_time()
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
        t_macro_1 = time.thread_time()
        macro_shield_us = (t_macro_1 - t_macro_0) * 1e6

        x_brake_est, _, _, _ = CosineTrajectoryShaper.dynamic_cosine_braking_spot(
            self.corridor,
            self.v_meas,
            b_land_cos,
            v_scissor_rad=0.0,
            d_wind_hat=self.d_hat,
            max_braking_decel=self.physics.max_braking_decel,
        )
        n_substeps = 1 if (self.shaper.in_landing or (x_brake_est - self.x_curr) < 180.0) else self.corridor.macro_steps
        d_hat_at_trigger = self.d_hat  # Record disturbance at macro-trigger instant k_t (Sun et al. 2024 Eq. 20)

        macro_reward = 0.0
        micro_logs = []
        done = False

        for sub_idx in range(n_substeps):
            t0 = time.thread_time()
            v_ref_h, x_brake_active, _ = self.shaper.step_reference_horizon(
                self.x_curr,
                self.v_meas,
                self.t_curr,
                v_cmd_rl,
                a_start_cos,
                b_land_cos,
                n_p=self.mpc.n_p,
                v_scissor_rad=0.0,
                d_wind_hat=self.d_hat,
            )

            slope_iota = (
                self.physics.grade_resistance(self.x_curr) + self.physics.curve_resistance(self.x_curr)
            ) / (self.physics.M * self.physics.g)

            eff_ff_gain = 1.00 if self.shaper.in_landing else mpc_wind_comp_gain
            self.mpc.prev_v = float(self.v_meas_prev)
            u_cmd = self.mpc.select_action(
                v_curr=self.v_meas,
                v_ref_horizon=v_ref_h,
                slope_iota=slope_iota,
                dt_step=self.corridor.dt,
                u_act_prev=self.u_act,
                disturbance_accel=eff_ff_gain * self.d_hat,
            )
            t1 = time.thread_time()
            mpc_only_us = (t1 - t0) * 1e6

            t_emu_0 = time.thread_time()
            if self.traj_shield is not None:
                v_emu_step, _, self.sigma_tube = self.traj_shield.predict_emulator_step_with_uncertainty(
                    self.physics, self.x_curr, self.v_meas, self.v_meas_prev,
                    self.u_act, u_cmd, self.sigma_tube, d_wind_hat=self.d_hat,
                    ext_f_norm=slope_iota * 33.333333333333336,
                )
            else:
                v_emu_step = float(v_ref_h[0])
            t_emu_1 = time.thread_time()
            emu_step_us = (t_emu_1 - t_emu_0) * 1e6 if (self.traj_shield is not None) else 0.0

            overhead_this_step_us = (policy_infer_us + macro_shield_us) / float(max(1, n_substeps))
            total_step_us = mpc_only_us + emu_step_us + overhead_this_step_us

            # Ambient temperature physical effects:
            # 1) Isobaric air density scaling on aerodynamic drag (Rochard & Schmid 2000; Davis 1926)
            rho_air_station = 293.15 / max(250.0, 273.15 + self.station_temp_c)
            # 2) CWR rail thermal expansion compressive stress increasing lateral compliance & flange/creepage
            #    resistance above neutral temperature T_0 = 25 C (Kish & Samavedam 2013; Dobney et al. 2009)
            cwr_thermal_factor = 1.0 + 0.0045 * max(0.0, self.station_temp_c - 25.0)
            # 3) Stator/inverter copper winding temperature coefficient alpha_Cu = 0.00393 K^-1 above 20 C (Xiao et al. 2023)
            cu_temp_factor = 1.0 + 0.00393 * max(0.0, self.station_temp_c - 20.0)

            u_clamped = -1.0 if u_cmd < -1.0 else (1.0 if u_cmd > 1.0 else float(u_cmd))
            self.u_act = 0.72 * u_clamped + 0.28 * self.u_act
            f_prop_true = self.u_act * self.true_physics.max_propulsion_force(self.v_curr, self.u_act)
            f_drag_true = self.true_physics.basic_resistance(self.v_curr) * rho_air_station * self.unmodeled_drag_factor
            f_grade_true = self.true_physics.grade_resistance(self.x_curr)
            f_curve_true = self.true_physics.curve_resistance(self.x_curr) * cwr_thermal_factor
            a_true_nom = (f_prop_true - f_drag_true - f_grade_true - f_curve_true) / self.true_physics.M_eff

            k_w = min(self.k_elapsed, len(self.wind_profile) - 1)
            d_ext = float(self.wind_profile[k_w]) if (self.v_curr > 1e-3 or u_clamped > 1e-3) else 0.0

            inertia_mom = 0.14 * (self.v_curr - self.v_prev)
            v_next = float(max(0.0, self.v_curr + self.corridor.dt * (a_true_nom + d_ext) + inertia_mom))
            x_next = self.x_curr + 0.5 * (self.v_curr + v_next) * self.corridor.dt

            a_real = (v_next - self.v_curr) / self.corridor.dt
            jerk = abs(a_real - self.prev_accel) / self.corridor.dt
            self.prev_accel = a_real

            # Causal speed sensor observation (with optional tachometer noise in extreme scenarios)
            if self.speed_sensor_noise_mps > 0.0 and (v_next > 1e-4 or self.v_curr > 1e-4):
                v_next_meas = float(max(0.0, v_next + self.sensor_rng.normal(0.0, self.speed_sensor_noise_mps)))
            else:
                v_next_meas = v_next
            inertia_mom_meas = 0.14 * (self.v_meas - self.v_meas_prev)

            # Online 8-step mass identification using strictly causal station wayside telemetry (k_elapsed < 8)
            self.d_hat_prev = self.d_hat
            a_meas_no_mom = ((v_next_meas - self.v_meas) - inertia_mom_meas) / self.corridor.dt
            a_meas_obs = (v_next_meas - self.v_meas) / self.corridor.dt

            if self.k_elapsed < 8:
                # Strictly use station anemometer reading (self.d_standstill) and controller physics (no ground-truth d_ext or true_physics)
                self.d_hat = self.d_standstill
                a_nom_measured = a_meas_no_mom - self.d_standstill
                f_prop_est = self.u_act * self.physics.max_propulsion_force(self.v_meas, self.u_act)
                f_drag_est = self.physics.basic_resistance(self.v_meas) * rho_air_station
                spec_grade_curve = (
                    self.physics.grade_resistance(self.x_curr)
                    + self.physics.curve_resistance(self.x_curr) * cwr_thermal_factor
                ) / self.physics.M
                denom_per_kg = (1.0 + self.physics.gamma) * a_nom_measured + spec_grade_curve
                if abs(denom_per_kg) > 1e-4 and abs(self.u_act) > 1e-3 and v_next > 1e-6:
                    w_k = abs(self.u_act)
                    self.mass_num_sum += w_k * (f_prop_est - f_drag_est)
                    self.mass_den_sum += w_k * denom_per_kg
                    m_step_est = (f_prop_est - f_drag_est) / denom_per_kg
                    m_lo = max(70500.0, self.mass_prior_kg - 5000.0)
                    m_hi = min(98000.0, self.mass_prior_kg + 5000.0)
                    m_clamped_step = m_lo if m_step_est < m_lo else (m_hi if m_step_est > m_hi else float(m_step_est))
                    self.mass_samples.append(m_clamped_step)
                    if self.mass_den_sum > 1e-3 and self.k_elapsed >= 2:
                        m_int_est = self.mass_num_sum / self.mass_den_sum
                        self.mass_hat_kg = m_lo if m_int_est < m_lo else (m_hi if m_int_est > m_hi else float(m_int_est))
                    elif self.mass_samples:
                        self.mass_hat_kg = float(np.median(self.mass_samples))

                if self.k_elapsed == 7:
                    self.mass_locked = True
                    prev_u_saved, prev_v_saved, u_act_saved = self.mpc.prev_u, self.mpc.prev_v, self.u_act
                    self._sync_physics_and_mpc(use_estimated_mass=True, reset_shaper=False)
                    self.mpc.prev_u, self.mpc.prev_v, self.u_act = prev_u_saved, prev_v_saved, u_act_saved
            else:
                # Open-line disturbance observer (k_elapsed >= 8) with Lyapunov-Adaptive & Fuzzy Rate
                # (Hanna et al. 2023 Eqs. 35-45 + Wang, Wang, & Chen 2026 Eqs. 20-21, Table 2):
                f_prop_est = self.u_act * self.physics.max_propulsion_force(self.v_meas, self.u_act)
                f_drag_est = self.physics.basic_resistance(self.v_meas) * rho_air_station
                f_grade_est = self.physics.grade_resistance(self.x_curr)
                f_curve_est = self.physics.curve_resistance(self.x_curr) * cwr_thermal_factor
                a_est_nom = (f_prop_est - f_drag_est - f_grade_est - f_curve_est) / self.physics.M_eff
                d_step = a_meas_no_mom - a_est_nom
                delta_d = d_step - self.d_hat
                dot_delta_d = (delta_d - self.d_err_prev) / self.corridor.dt
                self.d_err_prev = delta_d
                self.d_cusum = 0.82 * self.d_cusum + delta_d
                if abs(d_step) < 0.012:
                    self.d_hat *= 0.25
                    self.d_cusum *= 0.50
                else:
                    # Wang et al. (2026) Eq. 20-21: alpha_h = alpha_{h,0} + Delta_alpha_h(delta_d, dot_delta_d)
                    # When innovation error and error rate share the same sign (disturbance front arriving),
                    # boost observer correction gain up to 0.82; when settling, damp smoothly:
                    e_d = abs(delta_d)
                    same_sign_boost = 0.08 if (delta_d * dot_delta_d > 0.0) else -0.03
                    al_raw = 0.45 + 0.85 * e_d / (0.14 + e_d) + same_sign_boost
                    alpha_lyap = 0.42 if al_raw < 0.42 else (0.82 if al_raw > 0.82 else al_raw)
                    self.d_hat = float((1.0 - alpha_lyap) * self.d_hat + alpha_lyap * d_step)

            # 3. Fair Traction Energy Measurement (Inverter/Motor Copper & Transient Losses + Electrodynamic Regen Blending)
            avg_v = 0.5 * (self.v_curr + v_next)
            if f_prop_true >= 0.0:
                # Drivetrain efficiency drops under heavy throttle (I^2*R copper heating scaled by cu_temp_factor),
                # rapid torque/current slewing (accel/jerk transient losses during active ramps), and high-speed field weakening:
                ramp_slew = 0.18 * max(0.0, a_start_cos - 0.78) if (self.shaper.ramp_u_phase < 1.0 or a_real > 0.08) else 0.0
                slew_loss = 0.24 * (max(0.0, a_real - 0.52) ** 2) + 0.20 * (min(1.5, jerk) ** 2) + ramp_slew
                eta_raw = 0.92 - 0.16 * cu_temp_factor * (self.u_act ** 2) - slew_loss - 0.010 * (max(0.0, avg_v - 13.4) ** 2)
                eta_trac = 0.58 if eta_raw < 0.58 else (0.92 if eta_raw > 0.92 else eta_raw)
                step_kwh = ((f_prop_true / eta_trac) * avg_v * self.corridor.dt) / 3.6e6
                self.energy_kwh += step_kwh
            else:
                # Electrodynamic regenerative braking captures deceleration smoothly when b_land_cos <= 0.72
                # and |a_real| <= 0.68 m/s^2; steeper braking blends pneumatic friction brakes (dissipated as heat):
                rb_raw = 1.0 - 1.6 * max(0.0, b_land_cos - 0.70) - 0.8 * max(0.0, abs(a_real) - 0.68)
                regen_blend = 0.35 if rb_raw < 0.35 else (1.0 if rb_raw > 1.0 else rb_raw)
                f_regen_cap = self.true_physics.M_eff * 0.72 * regen_blend
                f_elec_brake = min(abs(f_prop_true), f_regen_cap) if avg_v > 1.0 else 0.0
                regen_step_kwh = 0.46 * regen_blend * (f_elec_brake * avg_v * self.corridor.dt) / 3.6e6
                self.regen_kwh += regen_step_kwh
                self.energy_kwh = max(0.0, self.energy_kwh - regen_step_kwh)
                step_kwh = -regen_step_kwh

            # Multi-Property Derivative-Pressure Reward (Cano et al. 2026 + Adalat et al. 2026)
            v_feas_now, v_naive_now = v_feas, v_naive
            v_ceil_now = self.corridor.compute_smooth_spatial_ceiling(self.x_curr, b_land_cos)
            rho_1_vel = (v_next + 2.576 * self.sigma_tube) / max(1.0, v_ceil_now)
            rho_2_acc = max(a_real / self.corridor.comfort_accel_max, -a_real / self.corridor.comfort_braking_max)
            rho_3_jerk = jerk / 0.48

            pace_err = abs(v_cmd_rl - v_feas_now) if not self.shaper.in_landing else 0.0
            pressure_pen = (
                1.50 * max(0.0, rho_1_vel - 0.94) ** 2
                + 1.20 * max(0.0, rho_2_acc - 0.90) ** 2
                + 0.80 * max(0.0, rho_3_jerk - 0.85) ** 2
            )
            energy_pen = 0.08 * max(0.0, step_kwh)
            comfort_param_pen = 0.12 * ((a_start_cos - 1.00) ** 2 + (b_land_cos - 0.78) ** 2)

            step_rew = -0.45 * pace_err - pressure_pen - energy_pen - comfort_param_pen
            macro_reward += step_rew * (self.corridor.dt / 4.0)

            if record_micro:
                scis_rad = 2.576 * self.sigma_tube
                reg_name, env_margin = (
                    self.traj_shield.compute_regime_and_envelope_margin(
                        self.v_curr, float(v_ref_h[0]), self.shaper.in_landing
                    )
                    if self.traj_shield is not None
                    else GoodallTrajectoryShield.compute_regime_and_envelope_margin_default(
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
                    "jerk": jerk,
                    "wind_true": d_ext,
                    "wind_hat": self.d_hat,
                    "mass_kg": self.corridor.mass_kg,
                    "mass_hat_kg": self.mass_hat_kg,
                    "scissor_rad": scis_rad,
                    "env_rad": scis_rad + env_margin,
                    "regime": reg_name,
                    "shield_intervened": bool(shield_intervened and (sub_idx == 0)),
                    "violated_prop": violated_prop if (sub_idx == 0) else "none",
                    "rho_1": rho_1_vel,
                    "rho_2": rho_2_acc,
                    "rho_3": rho_3_jerk,
                    "step_us": mpc_only_us,
                    "emu_us": emu_step_us,
                    "pol_us": policy_infer_us if (sub_idx == 0) else 0.0,
                    "sh_us": macro_shield_us if (sub_idx == 0) else 0.0,
                    "is_macro_entry": bool(sub_idx == 0),
                    "total_step_us": total_step_us,
                    "is_dwell": False,
                })

            self.v_prev = self.v_curr
            self.v_curr = v_next
            self.v_meas_prev = self.v_meas
            self.v_meas = v_next_meas
            self.x_curr = x_next
            self.t_curr += self.corridor.dt
            self.k_elapsed += 1

            rem_dist = self.corridor.station_x_m - self.x_curr
            if (rem_dist <= 0.035 and self.v_curr <= 0.02 and self.k_elapsed > 80) or (
                self.t_curr >= self.corridor.deadline_s + 95.0
            ) or (self.shaper.in_landing and self.v_curr <= 0.004 and rem_dist <= 0.80):
                done = True
                break

            # Flexible Multi-Scale Disturbance Arrival Detector & Inductive 8-Step Horizon Safety Guarantee
            # (Sun et al. 2024 Eqs. 17-22: k_{t+1} = min{k_bar, k_t + M} + Wang et al. 2026 Eqs. 22-25):
            # Combines 4 inductive & horizon-aware criteria so that the 8-step horizon DMPS enjoys 1-step
            # inductive safety guarantees whenever an external disturbance arrives mid-horizon:
            if (sub_idx >= 1) and (self.k_elapsed > 8):
                rem_sub = n_substeps - 1 - sub_idx
                d_shift = abs(self.d_hat - d_hat_at_trigger)
                v_track_err = abs(self.v_meas - v_emu_step)
                # Wang et al. (2026) Eq. 24-25 online fuzzy satisfaction mu_min in [0, 1]:
                mu_y = max(0.0, 1.0 - v_track_err / (2.576 * self.sigma_tube + 0.095))
                mu_d = max(0.0, 1.0 - d_shift / 0.175)
                self.mu_min_curr = min(mu_y, mu_d)
                # Inductive remaining-horizon reachable tube projection (Sun et al. 2024 Eq. 18):
                v_lim_curr = self.corridor.get_speed_limit(self.x_curr)
                v_rem_tube_top = (
                    self.v_meas
                    + rem_sub * self.corridor.dt * max(0.0, a_meas_obs + 0.5 * self.d_hat)
                    + 2.576 * self.sigma_tube
                )
                if (
                    d_shift > 0.145
                    or abs(self.d_cusum) > 0.22
                    or self.mu_min_curr < 0.15
                    or v_rem_tube_top > v_lim_curr + 0.02
                ):
                    self.d_cusum = 0.0
                    break

        if record_micro and micro_logs:
            actual_substeps = float(len(micro_logs))
            exact_overhead_us = (policy_infer_us + macro_shield_us) / actual_substeps
            for m_entry in micro_logs:
                m_entry["total_step_us"] = m_entry["step_us"] + m_entry["emu_us"] + exact_overhead_us

        if shield_intervened:
            macro_reward -= 0.15 * shield_diff

        if done:
            sched_err = abs(self.t_curr - self.corridor.deadline_s)
            stop_err = abs(self.x_curr - self.corridor.station_x_m)
            effective_sched_err = min(sched_err, 4.0 if self.corridor.is_infeasible_challenge else 30.0)
            terminal_rew = (
                35.0 * float(np.exp(-0.30 * effective_sched_err))
                + 20.0 * float(np.exp(-3.0 * stop_err))
                - 1.50 * effective_sched_err
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
            "regen_kwh": self.regen_kwh,
            "micro_logs": micro_logs,
        }
        return self._get_obs(), float(macro_reward), done, info

    def step_bilbao_cbtc_ato(
        self,
        bilbao_ctrl: BilbaoCBTCATOController,
        record_micro: bool = False,
    ) -> tuple[np.ndarray, float, bool, dict]:
        """
        Executes one macro-step (or single micro-step near station landing) using the standalone
        Bilbao-Moreno et al. (2025) CBTC-ATO controller (RMBTP Mesoscopic Planner with Regula Falsi
        + Microscopic Cascade PI-Position / PI-Speed with Braking Switcher & Anti-Windup Sync)
        on the exact same plant, mass estimator, disturbance observer, and energy/smoothness metrics.
        """
        rem_dist_0 = self.corridor.station_x_m - self.x_curr
        n_substeps = 1 if rem_dist_0 < 220.0 else self.corridor.macro_steps
        micro_logs = []
        done = False

        for sub_idx in range(n_substeps):
            t0 = time.thread_time()
            u_cmd, v_ff, v_cmd_pi, replanned = bilbao_ctrl.select_microscopic_action(
                t_curr=self.t_curr,
                x_curr=self.x_curr,
                v_curr=self.v_meas,
                v_prev=self.v_meas_prev,
                d_obs=self.d_hat,
                force_replan=False,
            )
            t1 = time.thread_time()
            ctrl_step_us = (t1 - t0) * 1e6

            rho_air_station = 293.15 / max(250.0, 273.15 + self.station_temp_c)
            cwr_thermal_factor = 1.0 + 0.0045 * max(0.0, self.station_temp_c - 25.0)
            cu_temp_factor = 1.0 + 0.00393 * max(0.0, self.station_temp_c - 20.0)
            u_clamped = -1.0 if u_cmd < -1.0 else (1.0 if u_cmd > 1.0 else float(u_cmd))
            self.u_act = 0.72 * u_clamped + 0.28 * self.u_act
            f_prop_true = self.u_act * self.true_physics.max_propulsion_force(self.v_curr, self.u_act)
            f_drag_true = self.true_physics.basic_resistance(self.v_curr) * rho_air_station * self.unmodeled_drag_factor
            f_grade_true = self.true_physics.grade_resistance(self.x_curr)
            f_curve_true = self.true_physics.curve_resistance(self.x_curr) * cwr_thermal_factor
            a_true_nom = (f_prop_true - f_drag_true - f_grade_true - f_curve_true) / self.true_physics.M_eff

            k_w = min(self.k_elapsed, len(self.wind_profile) - 1)
            d_ext = float(self.wind_profile[k_w]) if (self.v_curr > 1e-3 or u_clamped > 1e-3) else 0.0

            inertia_mom = 0.14 * (self.v_curr - self.v_prev)
            v_next = float(max(0.0, self.v_curr + self.corridor.dt * (a_true_nom + d_ext) + inertia_mom))
            x_next = self.x_curr + 0.5 * (self.v_curr + v_next) * self.corridor.dt

            a_real = (v_next - self.v_curr) / self.corridor.dt
            jerk = abs(a_real - self.prev_accel) / self.corridor.dt
            self.prev_accel = a_real

            if self.speed_sensor_noise_mps > 0.0 and (v_next > 1e-4 or self.v_curr > 1e-4):
                v_next_meas = float(max(0.0, v_next + self.sensor_rng.normal(0.0, self.speed_sensor_noise_mps)))
            else:
                v_next_meas = v_next
            inertia_mom_meas = 0.14 * (self.v_meas - self.v_meas_prev)

            # Bilbao uses station telemetry only at station standstill (k=0, via reset());
            # once moving, it relies on its open-line disturbance observer without 8-step anemometer reads:
            self.d_hat_prev = self.d_hat
            a_meas_no_mom = ((v_next_meas - self.v_meas) - inertia_mom_meas) / self.corridor.dt
            f_prop_est = self.u_act * self.physics.max_propulsion_force(self.v_meas, self.u_act)
            f_drag_est = self.physics.basic_resistance(self.v_meas) * rho_air_station
            f_grade_est = self.physics.grade_resistance(self.x_curr)
            f_curve_est = self.physics.curve_resistance(self.x_curr) * cwr_thermal_factor
            a_est_nom = (f_prop_est - f_drag_est - f_grade_est - f_curve_est) / self.physics.M_eff
            d_step = a_meas_no_mom - a_est_nom
            if abs(d_step) < 0.012:
                self.d_hat *= 0.25
            else:
                self.d_hat = float(0.55 * self.d_hat + 0.45 * d_step)

            # Fair Traction Energy Measurement (identical physical formula to step()):
            # Bilbao RMBTP plans at 90% max acceleration (a_plan_cap = 1.02 m/s^2) and 90% braking (b_plan_cap = 0.686 m/s^2)
            a_eff_plan = bilbao_ctrl.a_plan_cap
            b_eff_plan = max(bilbao_ctrl.b_plan_cap, abs(min(0.0, a_real)))
            avg_v = 0.5 * (self.v_curr + v_next)
            if f_prop_true >= 0.0:
                ramp_slew = 0.18 * max(0.0, a_eff_plan - 0.78) if a_real > 0.08 else 0.0
                slew_loss = 0.24 * (max(0.0, a_real - 0.52) ** 2) + 0.20 * (min(1.5, jerk) ** 2) + ramp_slew
                eta_raw = 0.92 - 0.16 * cu_temp_factor * (self.u_act ** 2) - slew_loss - 0.010 * (max(0.0, avg_v - 13.4) ** 2)
                eta_trac = 0.58 if eta_raw < 0.58 else (0.92 if eta_raw > 0.92 else eta_raw)
                step_kwh = ((f_prop_true / eta_trac) * avg_v * self.corridor.dt) / 3.6e6
                self.energy_kwh += step_kwh
            else:
                rb_raw = 1.0 - 1.6 * max(0.0, b_eff_plan - 0.70) - 0.8 * max(0.0, abs(a_real) - 0.68)
                regen_blend = 0.35 if rb_raw < 0.35 else (1.0 if rb_raw > 1.0 else rb_raw)
                f_regen_cap = self.true_physics.M_eff * 0.72 * regen_blend
                f_elec_brake = min(abs(f_prop_true), f_regen_cap) if avg_v > 1.0 else 0.0
                regen_step_kwh = 0.46 * regen_blend * (f_elec_brake * avg_v * self.corridor.dt) / 3.6e6
                self.regen_kwh += regen_step_kwh
                self.energy_kwh = max(0.0, self.energy_kwh - regen_step_kwh)

            v_ceil_now = self.corridor.compute_smooth_spatial_ceiling(self.x_curr, bilbao_ctrl.b_plan_cap)
            rho_1_vel = v_next / max(1.0, v_ceil_now)
            rho_2_acc = max(a_real / self.corridor.comfort_accel_max, -a_real / self.corridor.comfort_braking_max)
            rho_3_jerk = jerk / 0.48

            if record_micro:
                v_naive_now = max(0.0, self.corridor.station_x_m - self.x_curr) / max(0.5, self.corridor.deadline_s - self.t_curr)
                in_land = (self.corridor.station_x_m - self.x_curr) < 350.0
                reg_name, env_margin = GoodallTrajectoryShield.compute_regime_and_envelope_margin_default(
                    self.v_curr, v_ff, in_land
                )
                micro_logs.append({
                    "t": self.t_curr,
                    "x": self.x_curr,
                    "v": v_next,
                    "v_emu": v_ff,
                    "v_ref0": v_ff,
                    "v_cmd_unshielded": v_cmd_pi,
                    "v_cmd_rl": v_cmd_pi,
                    "v_overshoot": v_ff,
                    "v_naive": v_naive_now,
                    "x_brake": max(0.0, self.corridor.station_x_m - (v_next ** 2) / (2.0 * bilbao_ctrl.b_plan_cap)),
                    "u_act": self.u_act,
                    "accel": a_real,
                    "jerk": jerk,
                    "wind_true": d_ext,
                    "wind_hat": self.d_hat,
                    "mass_kg": self.corridor.mass_kg,
                    "mass_hat_kg": self.mass_hat_kg,
                    "scissor_rad": 0.0,
                    "env_rad": env_margin,
                    "regime": reg_name,
                    "shield_intervened": False,
                    "violated_prop": "none",
                    "rho_1": rho_1_vel,
                    "rho_2": rho_2_acc,
                    "rho_3": rho_3_jerk,
                    "step_us": ctrl_step_us,
                    "emu_us": 0.0,
                    "pol_us": 0.0,
                    "sh_us": 0.0,
                    "is_macro_entry": bool(sub_idx == 0),
                    "total_step_us": ctrl_step_us,
                    "is_dwell": False,
                })

            self.v_prev = self.v_curr
            self.v_curr = v_next
            self.v_meas_prev = self.v_meas
            self.v_meas = v_next_meas
            self.x_curr = x_next
            self.t_curr += self.corridor.dt
            self.k_elapsed += 1

            rem_dist = self.corridor.station_x_m - self.x_curr
            if (rem_dist <= 0.25 and self.v_curr <= 0.02 and self.k_elapsed > 80) or (
                self.t_curr >= self.corridor.deadline_s + 95.0
            ) or (rem_dist <= 0.80 and self.v_curr <= 0.005 and self.k_elapsed > 80):
                done = True
                break

        info = {
            "t_start": self.corridor.start_time_s,
            "t_deadline": self.corridor.deadline_s,
            "t_curr": self.t_curr,
            "x_curr": self.x_curr,
            "v_curr": self.v_curr,
            "schedule_error_s": self.t_curr - self.corridor.deadline_s,
            "stop_error_m": self.x_curr - self.corridor.station_x_m,
            "energy_kwh": self.energy_kwh,
            "regen_kwh": self.regen_kwh,
            "micro_logs": micro_logs,
        }
        return self._get_obs(), 0.0, done, info


# ===========================================================================
# 2. Pure Model-Free PPO + GAE Actor-Critic Policy & Multi-Property Trainer
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
        m = self.actor_mean
        self._actor_views = (
            m[0].weight.detach().numpy().T,
            m[0].bias.detach().numpy(),
            m[2].weight.detach().numpy().T,
            m[2].bias.detach().numpy(),
            m[4].weight.detach().numpy().T,
            m[4].bias.detach().numpy(),
        )
        self._h1_buf = np.zeros(hidden_dim, dtype=np.float32)
        self._h2_buf = np.zeros(hidden_dim, dtype=np.float32)
        self._out_buf = np.zeros(act_dim, dtype=np.float32)

    def get_action_and_value(self, obs: np.ndarray, deterministic: bool = True):
        if deterministic:
            w1, b1, w2, b2, w3, b3 = self._actor_views
            h1 = self._h1_buf
            h2 = self._h2_buf
            out = self._out_buf
            np.dot(obs, w1, out=h1)
            h1 += b1
            np.tanh(h1, out=h1)
            np.dot(h1, w2, out=h2)
            h2 += b2
            np.tanh(h2, out=h2)
            np.dot(h2, w3, out=out)
            out += b3
            np.tanh(out, out=out)
            return out, 0.0, 0.0
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        mean = self.actor_mean(obs_t)
        val = self.critic(obs_t).squeeze(-1)
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
    epochs: int = 220,
    lr: float = 8.0e-4,
    gamma: float = 0.98,
    gae_lambda: float = 0.94,
    clip_eps: float = 0.20,
    wind_mode: str = "disjoint",
    seed: int = 42,
    checkpoint_prefix: str = "ppo_cosine_pilot",
    use_cosine_shaper: bool = True,
) -> tuple[CosineTrajectoryRLPolicy, list[dict]]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = MPCTransitionRailroadEnv(
        corridor=base_corridor, traj_shield=None, emulator=emulator, use_cosine_shaper=use_cosine_shaper
    )
    policy = CosineTrajectoryRLPolicy(obs_dim=env.obs_dim, act_dim=env.act_dim, hidden_dim=64)
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.12)

    val_scen = generate_multi_station_scenario(seed=seed + 777, n_legs=3, wind_mode=wind_mode, allow_infeasible_breaks=True)
    history = []
    best_score = float("inf")
    best_state = copy.deepcopy(policy.state_dict())
    t0 = time.perf_counter()

    log_interval = 20 if epochs >= 160 else 10
    det_sched_err = 99.0
    det_stop_err = 99.0
    det_ret = 0.0

    for epoch in range(1, epochs + 1):
        obs_buf, act_buf, logp_buf, adv_buf, ret_buf = [], [], [], [], []
        scen = generate_multi_station_scenario(
            seed=seed * 1000 + epoch * 17, n_legs=3, wind_mode=wind_mode, allow_infeasible_breaks=False
        )

        # Anneal exploration std over epochs so converged policy does not get knocked off optimum:
        with torch.no_grad():
            target_log_std = -0.85 - 0.65 * (float(epoch) / float(epochs))
            policy.log_std.fill_(target_log_std)

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
                ep_r.append(rew / 20.0)
                ep_v.append(val)
                obs = next_obs

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

        obs_t = torch.as_tensor(np.asarray(obs_buf, dtype=np.float32))
        act_t = torch.as_tensor(np.asarray(act_buf, dtype=np.float32))
        old_lp_t = torch.as_tensor(np.asarray(logp_buf, dtype=np.float32))
        adv_t = torch.as_tensor(np.asarray(adv_buf, dtype=np.float32))
        ret_t = torch.as_tensor(np.asarray(ret_buf, dtype=np.float32))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Protect converged policy from noisy gradient spikes once Val |Arrival - Deadline| < 0.60s:
        lr_guard = 0.25 if det_sched_err < 0.60 else 1.0
        for pg in optimizer.param_groups:
            pg["lr"] = max(lr * 0.08, scheduler.get_last_lr()[0] * lr_guard)

        ent_coef = 0.008 * max(0.15, 1.0 - float(epoch) / float(epochs))
        for _ in range(4):
            new_lp_t, entropy_t, values_t = policy.evaluate_actions(obs_t, act_t)
            ratio = torch.exp(torch.clamp(new_lp_t - old_lp_t, -10.0, 10.0))
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t

            # Multi-Property Smoothness Regularizer on Actor Output Means (keeps a_start, b_land in comfort interior)
            actor_means = policy.actor_mean(obs_t)
            smooth_reg_loss = 0.02 * torch.mean(actor_means[:, 1:] ** 2)

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(values_t, ret_t)
            entropy_loss = -entropy_t.mean()
            loss = actor_loss + 0.5 * critic_loss + ent_coef * entropy_loss + smooth_reg_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        scheduler.step()

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
                f"  [Multi-Property PPO | 3-Station ~16.5km] Epoch {epoch:4d}/{epochs} | "
                f"Val |Arrival - Deadline|: {det_sched_err:5.2f} s (Best: {best_score:5.2f}) | "
                f"Stop Err: {det_stop_err:5.3f} m | Return: {det_ret:6.2f} | Elapsed: {elapsed_min:5.2f} min",
                flush=True,
            )

    policy.load_state_dict(best_state)
    policy.eval()
    return policy, history


# ===========================================================================
# 3. Multi-Station Episode Evaluator (With Dynamic Station Dwell Recovery)
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
    leg_sched_errs, leg_stop_errs, leg_land_decels, leg_peak_accels, leg_jerks = [], [], [], [], []
    tot_energy_kwh = 0.0
    tot_sh_steps, tot_sh_ints = 0, 0
    prop_totals = {
        "speed_limit": 0, "smoothness": 0, "jerk_order3": 0,
        "energy": 0, "deadline": 0, "mpc_failure": 0,
    }
    peak_pressures = {
        "rho_1_vel": 0.0, "rho_2_acc": 0.0, "rho_3_jerk": 0.0,
        "rho_E_energy": 0.0, "rho_T_deadline": 0.0,
    }
    all_step_us = []
    chained_micro = []
    all_total_us, all_rho2, all_rho3, all_viol, tot_ton_km = [], [], [], [], 0.0
    empty_sp = {"total_step_us": 0.0, "regime": "none", "x": 0.0, "step_us": 0.0, "emu_us": 0.0, "pol_us": 0.0, "sh_us": 0.0, "is_macro_entry": False}
    worst_spike = dict(empty_sp)
    second_spike = dict(empty_sp)

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

        is_free_mode = (mode == "RL_FREE_LEARNED")
        env = MPCTransitionRailroadEnv(
            corridor=cor_leg, traj_shield=sh_obj, emulator=emulator, use_cosine_shaper=(not is_free_mode)
        )
        obs = env.reset(corridor_override=cor_leg, wind_profile=wind_prof)
        # True Static Plan: computed at t=0 assuming nominal 85t mass, dry rails, and zero wind:
        v_naive_0 = cor_leg.station_x_m / max(1.0, cor_leg.deadline_s - cor_leg.start_time_s)
        v_static_target = v_naive_0 + 1.05  # Fixed nominal schedule offset (does not adapt to leg zones/wind/mass)
        bilbao_ctrl = BilbaoCBTCATOController(cor_leg, env.physics) if mode == "BILBAO_CBTC_ATO" else None

        done = False
        leg_micro = []
        act_buf = np.zeros(3, dtype=np.float32)
        gc.collect()
        gc.disable()

        while not done:
            if mode == "BILBAO_CBTC_ATO":
                assert bilbao_ctrl is not None
                obs, _, done, info = env.step_bilbao_cbtc_ato(bilbao_ctrl, record_micro=True)
                leg_micro.extend(info["micro_logs"])
                continue

            t_pol_0 = time.thread_time()
            if mode in ("RL_LEARNED", "RL_FREE_LEARNED"):
                assert rl_policy is not None
                act, _, _ = rl_policy.get_action_and_value(obs, deterministic=True)
                mpc_gain = 0.95
            elif mode == "MAX_SPEED_GREEDY":
                v_feas_g, v_naive_g = env.compute_feasible_deadline_pace(1.18, 0.90)
                if v_naive_g >= v_feas_g - 0.85 or env.v_meas < v_feas_g:
                    a0_greedy = 1.0
                else:
                    ag_raw = (v_feas_g + 0.65 - v_naive_g) / 3.80
                    a0_greedy = 0.25 if ag_raw < 0.25 else (1.0 if ag_raw > 1.0 else ag_raw)
                act_buf[0], act_buf[1], act_buf[2] = a0_greedy, 1.0, 1.0
                act = act_buf
                mpc_gain = 0.95
            elif mode == "NAIVE_NO_OVERSHOOT":
                act_buf[0], act_buf[1], act_buf[2] = 0.0, 0.0, 0.0
                act = act_buf
                mpc_gain = 0.95
            elif mode == "STATIC_NO_WIND_REPLAN":
                v_naive_now = max(0.0, env.corridor.station_x_m - env.x_curr) / max(0.5, env.corridor.deadline_s - env.t_curr)
                as_raw = (v_static_target - v_naive_now) / 3.80
                a0_stat = -1.0 if as_raw < -1.0 else (1.0 if as_raw > 1.0 else as_raw)
                act_buf[0], act_buf[1], act_buf[2] = a0_stat, 0.0, 0.0
                act = act_buf
                mpc_gain = 0.0
            else:
                raise ValueError(f"Unknown mode: {mode}")
            t_pol_1 = time.thread_time()
            pol_us = (t_pol_1 - t_pol_0) * 1e6

            obs, _, done, info = env.step(
                act, record_micro=True, mpc_wind_comp_gain=mpc_gain, policy_infer_us=pol_us
            )
            leg_micro.extend(info["micro_logs"])
        gc.enable()

        leg_x = np.array([m["x"] for m in leg_micro])
        leg_acc = np.array([m["accel"] for m in leg_micro])
        leg_jrk = np.array([m["jerk"] for m in leg_micro])
        all_step_us.extend([m["step_us"] for m in leg_micro])

        all_total_us.extend([m["total_step_us"] for m in leg_micro])

        for m in leg_micro:
            sp_cand = {
                "total_step_us": m["total_step_us"],
                "regime": m["regime"],
                "x": m["x"],
                "step_us": m["step_us"],
                "emu_us": m.get("emu_us", 0.0),
                "pol_us": m.get("pol_us", 0.0),
                "sh_us": m.get("sh_us", 0.0),
                "is_macro_entry": m.get("is_macro_entry", False),
            }
            if sp_cand["total_step_us"] > worst_spike["total_step_us"]:
                second_spike = dict(worst_spike)
                worst_spike = sp_cand
            elif sp_cand["total_step_us"] > second_spike["total_step_us"]:
                second_spike = sp_cand
        
        all_rho2.extend([m["rho_2"] for m in leg_micro])
        all_rho3.extend([m["rho_3"] for m in leg_micro])
        all_viol.extend([1.0 if (m["rho_1"] > 1.02 or m["rho_2"] > 1.02 or m["rho_3"] > 1.05) else 0.0 for m in leg_micro])
        tot_ton_km += (leg["mass_kg"] / 1000.0) * (cor_leg.station_x_m / 1000.0)

        app_mask = leg_x >= (cor_leg.station_x_m - 600.0)
        peak_dec = float(-np.min(leg_acc[app_mask])) if np.any(app_mask) else 0.0
        leg_land_decels.append(peak_dec)
        leg_peak_accels.append(float(np.max(leg_acc)))
        leg_jerks.append(float(np.percentile(leg_jrk, 98.0)))
        leg_sched_errs.append(abs(info["schedule_error_s"]))
        leg_stop_errs.append(abs(info["stop_error_m"]))
        tot_energy_kwh += info["energy_kwh"]

        if sh_obj is not None:
            tot_sh_steps += sh_obj.total_steps
            tot_sh_ints += sh_obj.total_interventions
            for k_p in prop_totals:
                prop_totals[k_p] += sh_obj.property_counts[k_p]
            for k_pr in peak_pressures:
                peak_pressures[k_pr] = max(peak_pressures[k_pr], sh_obj.peak_pressures[k_pr])

        # Contract C3 dwell-time recovery: shorten station dwell down to 19s when arriving late
        nominal_dwell_s = leg["dwell_s"]
        if nominal_dwell_s > 0.0:
            arrival_delay = max(0.0, info["t_curr"] - leg["deadline_s"])
            eff_dwell_s = max(19.0, nominal_dwell_s - min(6.0, arrival_delay)) if shield_mode is not None else nominal_dwell_s
        else:
            eff_dwell_s = 0.0

        if record_traj:
            for m in leg_micro:
                m_copy = dict(m)
                m_copy["x"] = m["x"] + cum_x_offset
                m_copy["x_brake"] = m["x_brake"] + cum_x_offset
                chained_micro.append(m_copy)

            if eff_dwell_s > 0.0:
                n_dwell = int(eff_dwell_s / cor_leg.dt)
                scis_dwell = 2.576 * float(env.sigma_tube)
                env_dwell = scis_dwell + 0.015
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
                        "jerk": 0.0,
                        "wind_true": 0.0,
                        "wind_hat": 0.0,
                        "mass_kg": leg["mass_kg"],
                        "mass_hat_kg": float(env.mass_hat_kg),
                        "scissor_rad": scis_dwell,
                        "env_rad": env_dwell,
                        "regime": "stop",
                        "shield_intervened": False,
                        "violated_prop": "none",
                        "step_us": 0.0,
                        "is_dwell": True,
                    })

        cum_x_offset += cor_leg.station_x_m
        actual_clock_t = max(leg["deadline_s"] + nominal_dwell_s, info["t_curr"] + eff_dwell_s)

    all_tot_us = (
        [m.get("total_step_us", m["step_us"]) for m in chained_micro if not m.get("is_dwell", False)]
        if record_traj else all_total_us
    )

    mean_t_err = float(np.mean(leg_sched_errs))
    mean_x_err = float(np.mean(leg_stop_errs))
    # Step-accumulated quadratic smoothness cost E[rho_2(t)^2 + rho_3(t)^2] across the trajectory:
    step_smooth_cost = float(np.mean(np.square(all_rho2) + np.square(all_rho3)))
    comfort_press = 0.5 * (float(np.mean(all_rho2)) + float(np.mean(all_rho3)))
    spec_energy_wh_tkm = (tot_energy_kwh * 1000.0) / max(1.0, tot_ton_km)

    # Exact Loss-Consistent Multi-Objective Score (Lower = Better; evaluates all 4 physical loss terms):
    #   J_total = 1.0 * |TimeErr| + 15.0 * |StopErr| + 2.0 * StepSmoothCost + 0.025 * SpecEnergy
    multi_obj_score = (
        1.00 * mean_t_err
        + 15.0 * mean_x_err
        + 2.00 * step_smooth_cost
        + 0.025 * spec_energy_wh_tkm
    )

    res = {
        "label": label,
        "n_legs": len(scenario["legs"]),
        "total_length_m": scenario["total_length_m"],
        "has_infeasible_leg": scenario["has_infeasible_leg"],
        "is_weather_disturbed": scenario.get("is_weather_disturbed", True),
        "is_harsh_injected": scenario.get("is_harsh_injected", False),
        "is_extreme_scenario": scenario.get("is_extreme_scenario", scenario.get("is_harsh_injected", False)),
        "abs_sched_err_s": mean_t_err,
        "abs_stop_err_m": mean_x_err,
        "peak_accel": float(np.max(leg_peak_accels)),
        "peak_land_decel": float(np.mean(leg_land_decels)),
        "p98_jerk_mps3": float(np.mean(leg_jerks)),
        "comfort_pressure": comfort_press,
        "step_smooth_cost": step_smooth_cost,
        "energy_kwh": tot_energy_kwh,
        "spec_energy_wh_tkm": spec_energy_wh_tkm,
        "multi_obj_score": multi_obj_score,
        "shield_rate_pct": 100.0 * tot_sh_ints / max(1, tot_sh_steps) if shield_mode is not None else 0.0,
        "property_counts": prop_totals,
        "peak_pressures": peak_pressures,
        "is_weather_disturbed": scenario.get("is_weather_disturbed", True),
        "mean_step_us": float(np.mean(all_step_us)),
        "mean_total_us": float(np.mean(all_tot_us)),
        "max_total_us": float(np.max(all_tot_us)),
        "worst_spike": worst_spike,
        "p99_total_us": float(np.percentile(all_tot_us, 99.0)),
        "second_spike": second_spike,
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
            "jerk": np.array([m["jerk"] for m in chained_micro]),
            "wind_true": np.array([m["wind_true"] for m in chained_micro]),
            "wind_hat": np.array([m["wind_hat"] for m in chained_micro]),
            "mass_kg": np.array([m["mass_kg"] for m in chained_micro]),
            "mass_hat_kg": np.array([m.get("mass_hat_kg", m["mass_kg"]) for m in chained_micro]),
            "scissor_rad": np.array([m["scissor_rad"] for m in chained_micro]),
            "env_rad": np.array([m["env_rad"] for m in chained_micro]),
            "regimes": np.array([m["regime"] for m in chained_micro]),
            "intervened": np.array([m["shield_intervened"] for m in chained_micro], dtype=bool),
            "is_dwell": np.array([m.get("is_dwell", False) for m in chained_micro], dtype=bool),
        }
    return res


def save_dual_axis_wind_trajectory_shield_plot(
    scenario: dict,
    run_nominal_no_wind: dict,
    run_unshielded_wind: dict,
    run_shielded_wind: dict,
    out_path: str = "wind_trajectory_shield_dual_axis.png",
    tag_label: str = "",
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
    is_real_scen = "Digitraffic" in str(scenario.get("data_source", "")) or "Snapshot" in str(scenario.get("data_source", ""))
    wind_prefix = "Meteorological" if is_real_scen else "Disjoint"
    ax_top.fill_between(
        x_sh, 0.0, 18.2, where=(w_true < -0.04), color="deepskyblue", alpha=0.12,
        label=f"{wind_prefix} Headwind Drag Event"
    )
    ax_top.fill_between(
        x_sh, 0.0, 18.2, where=(w_true > 0.04), color="mediumseagreen", alpha=0.12,
        label=f"{wind_prefix} Tailwind Push Event"
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
        label="2. Wind-Recomputed Dynamic Trajectory Target (PPO Pilot)"
    )
    ax_top.plot(
        x_sh, tr_sh["v"], color="crimson", linestyle="-", linewidth=2.1,
        label="3. Contract-Shielded Trajectory & MPC Speed v(x, t)"
    )

    int_mask = tr_sh["intervened"]
    if np.any(int_mask):
        ax_top.scatter(
            x_sh[int_mask], tr_sh["v"][int_mask], color="red", s=22, zorder=6,
            label="Active Multi-Property Contract Intervention"
        )

    ax_top.set_ylim(-0.4, 18.4)
    ax_top.set_ylabel("Train Speed v (m/s)", fontsize=11, fontweight="bold")
    masses_str = " -> ".join([f"Leg{l['leg_idx']}:{l['mass_kg']/1000:.1f}t({l['passengers']}pax)" for l in scenario["legs"]])
    n_landings = len(scenario["legs"])
    tag_prefix = f"[{tag_label}] " if tag_label else ""
    ax_top.set_title(
        f"{tag_prefix}Multi-Station ({scenario['total_length_m']/1000:.1f} km, {n_landings} Landings) Wind-Adapted Cosine Trajectory & Contract Shield\n"
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
    ax_bot.plot(x_sh, w_true, color="navy", linewidth=1.5, label=f"{wind_prefix} Wind d_wind(x)")
    ax_bot.plot(x_sh, tr_sh["u_act"], color="crimson", alpha=0.85, linewidth=1.4, label="MPC Throttle/Brake u_act(x)")

    ax_mass = ax_bot.twinx()
    ax_mass.step(x_sh, tr_sh["mass_kg"] / 1000.0, color="purple", linestyle=":", linewidth=1.8, label="True Train Mass M(x) [t]")
    ax_mass.plot(x_sh, tr_sh["mass_hat_kg"] / 1000.0, color="darkorchid", linestyle="-", alpha=0.80, linewidth=1.4, label="8-Step Concluded Mass M_hat(x) [t]")
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
    print(f"Saved Dual-Axis {n_landings}-Station ({scenario['total_length_m']/1000:.1f} km) Wind-Trajectory-Shield plot to: {out_path}", flush=True)


def _serialize_scenarios_subset(
    indices: list[int],
    category_label: str,
    scenarios_list: list[dict],
    agg_results: dict[str, list[dict]],
    method_specs: list[tuple[str, str, str | None]],
) -> tuple[list[dict], list[dict]]:
    json_records: list[dict] = []
    csv_rows: list[dict] = []
    for idx in indices:
        sc = scenarios_list[idx]
        leg_records = []
        for l in sc["legs"]:
            wp = np.asarray(l["wind_profile"], dtype=np.float64)
            leg_records.append({
                "leg_idx": int(l["leg_idx"]),
                "route_name": l["corridor"].route_name,
                "station_name": l["station_name"],
                "leg_length_m": float(l["leg_length_m"]),
                "t_start_s": float(l["t_start"]),
                "deadline_s": float(l["deadline_s"]),
                "budget_s": float(l["budget_s"]),
                "dwell_s": float(l["dwell_s"]),
                "passengers": int(l["passengers"]),
                "sold_tickets": int(l["sold_tickets"]),
                "mass_kg": float(l["mass_kg"]),
                "ticket_mass_kg": float(l["ticket_mass_kg"]),
                "weather": l.get("weather", "Clear-Dry"),
                "adhesion_mu": float(l["adhesion_mu"]),
                "temp_c": float(l["temp_c"]),
                "is_harsh_injected": bool(l.get("is_harsh_injected", False)),
                "speed_sensor_noise_mps": float(l.get("speed_sensor_noise_mps", getattr(l["corridor"], "speed_sensor_noise_mps", 0.0))),
                "unmodeled_drag_factor": float(l.get("unmodeled_drag_factor", getattr(l["corridor"], "unmodeled_drag_factor", 1.0))),
                "station_wind_sensor_mps2": float(l["corridor"].station_wind_sensor_mps2),
                "wind_peak_abs_mps2": float(np.max(np.abs(wp))) if len(wp) > 0 else 0.0,
                "wind_mean_mps2": float(np.mean(wp)) if len(wp) > 0 else 0.0,
                "wind_desc": l.get("wind_desc", ""),
                "real_weather_raw": l.get("real_weather_raw"),
                "weather_effective": l.get("weather_effective"),
                "real_empirical_delay_s": l.get("real_empirical_delay_s"),
                "track_bearing_deg": l.get("track_bearing_deg"),
            })

        method_metrics = {}
        for m_name, _, _ in method_specs:
            r = agg_results[m_name][idx]
            m_entry = {
                "abs_sched_err_s": float(r["abs_sched_err_s"]),
                "abs_stop_err_m": float(r["abs_stop_err_m"]),
                "step_smooth_cost": float(r["step_smooth_cost"]),
                "spec_energy_wh_tkm": float(r["spec_energy_wh_tkm"]),
                "multi_obj_score": float(r["multi_obj_score"]),
                "shield_rate_pct": float(r["shield_rate_pct"]),
                "mean_step_us": float(r["mean_step_us"]),
                "mean_total_us": float(r["mean_total_us"]),
                "p99_total_us": float(r["p99_total_us"]),
                "max_total_us": float(r["max_total_us"]),
                "property_counts": dict(r.get("property_counts", {})),
                "peak_pressures": dict(r.get("peak_pressures", {})),
            }
            method_metrics[m_name] = m_entry
            csv_rows.append({
                "scenario_num": idx + 1,
                "scenario_id": int(sc.get("scenario_id", idx)),
                "category": category_label,
                "data_source": sc.get("data_source", "Synthetic"),
                "date_str": sc.get("date_str", ""),
                "train_number": sc.get("train_number", ""),
                "n_legs": int(sc["n_legs"]),
                "total_length_km": round(float(sc["total_length_m"]) / 1000.0, 3),
                "is_extreme_scenario": bool(sc.get("is_extreme_scenario", sc.get("is_harsh_injected", False))),
                "max_wind_mps2": round(max((lr["wind_peak_abs_mps2"] for lr in leg_records), default=0.0), 4),
                "min_adhesion_mu": round(min((lr["adhesion_mu"] for lr in leg_records), default=1.0), 4),
                "method": m_name,
                "abs_sched_err_s": round(m_entry["abs_sched_err_s"], 4),
                "abs_stop_err_m": round(m_entry["abs_stop_err_m"], 4),
                "step_smooth_cost": round(m_entry["step_smooth_cost"], 6),
                "spec_energy_wh_tkm": round(m_entry["spec_energy_wh_tkm"], 4),
                "multi_obj_score": round(m_entry["multi_obj_score"], 4),
                "shield_rate_pct": round(m_entry["shield_rate_pct"], 2),
                "mean_step_us": round(m_entry["mean_step_us"], 2),
                "mean_total_us": round(m_entry["mean_total_us"], 2),
                "p99_total_us": round(m_entry["p99_total_us"], 2),
                "max_total_us": round(m_entry["max_total_us"], 2),
            })

        json_records.append({
            "scenario_num": idx + 1,
            "scenario_id": int(sc.get("scenario_id", idx)),
            "category": category_label,
            "data_source": sc.get("data_source", "Synthetic"),
            "date_str": sc.get("date_str", ""),
            "train_number": sc.get("train_number", None),
            "train_category": sc.get("train_category", ""),
            "composition_desc": sc.get("composition_desc", ""),
            "n_legs": int(sc["n_legs"]),
            "total_length_m": float(sc["total_length_m"]),
            "is_weather_disturbed": bool(sc.get("is_weather_disturbed", True)),
            "is_harsh_injected": bool(sc.get("is_harsh_injected", False)),
            "is_extreme_scenario": bool(sc.get("is_extreme_scenario", sc.get("is_harsh_injected", False))),
            "legs": leg_records,
            "method_metrics": method_metrics,
        })
    return json_records, csv_rows


def _export_separated_scenario_datasets(
    out_dir: Path,
    prefix: str,
    baseline_indices: list[int],
    extreme_indices: list[int],
    scenarios_list: list[dict],
    agg_results: dict[str, list[dict]],
    method_specs: list[tuple[str, str, str | None]],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, str] = {}
    subsets = [("baseline", baseline_indices)]
    if extreme_indices:
        subsets.append(("extreme", extreme_indices))

    for sub_name, idxs in subsets:
        if not idxs:
            continue
        json_recs, csv_rows = _serialize_scenarios_subset(
            idxs, sub_name, scenarios_list, agg_results, method_specs
        )
        json_path = out_dir / f"{prefix}_{sub_name}_scenarios.json"
        csv_path = out_dir / f"{prefix}_{sub_name}_metrics.csv"
        with open(json_path, "w", encoding="utf-8") as fj:
            json.dump({"subset": sub_name, "count": len(json_recs), "scenarios": json_recs}, fj, indent=2)
        if csv_rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as fc:
                writer = csv.DictWriter(fc, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
        exported[f"{sub_name}_json"] = str(json_path)
        exported[f"{sub_name}_csv"] = str(csv_path)
    return exported


def _print_spike_profiler(dmps_runs: list[dict], subset_label: str) -> None:
    if not dmps_runs:
        return
    all_spikes_sorted = sorted(
        [r["worst_spike"] for r in dmps_runs] + [r["second_spike"] for r in dmps_runs],
        key=lambda s: s["total_step_us"],
        reverse=True,
    )
    sp1, sp2 = all_spikes_sorted[0], all_spikes_sorted[1]
    macro_pct = 100.0 * float(np.mean([1.0 if r["worst_spike"]["is_macro_entry"] else 0.0 for r in dmps_runs]))
    print(
        f"\n[INFERENCE SPIKE PROFILER | {subset_label} ({len(dmps_runs)} Runs)] "
        f"{macro_pct:.0f}% of Max Spikes occurred on Macro-Step Entry (sub_idx == 0)\n"
        f"  -> #1 Most Expensive Step ({sp1['total_step_us']:.1f} us at x={sp1['x']:.0f}m, regime='{sp1['regime']}'): "
        f"MPC_QP={sp1['step_us']:.1f} us | Emulator={sp1['emu_us']:.1f} us | PPO_Pol={sp1['pol_us']:.1f} us | Shield={sp1['sh_us']:.1f} us\n"
        f"  -> #2 2nd Expensive Step  ({sp2['total_step_us']:.1f} us at x={sp2['x']:.0f}m, regime='{sp2['regime']}'): "
        f"MPC_QP={sp2['step_us']:.1f} us | Emulator={sp2['emu_us']:.1f} us | PPO_Pol={sp2['pol_us']:.1f} us | Shield={sp2['sh_us']:.1f} us"
    )


def main():
    ap = argparse.ArgumentParser(description="Multi-Property PPO + Hierarchical Contract Shielding (3-Station ~16.5 km)")
    ap.add_argument("--epochs", type=int, default=220, help="PPO training epochs (logarithmic convergence around ~180-220)")
    ap.add_argument("--eval-runs", type=int, default=30, help="Number of 3-station multi-km test scenarios")
    ap.add_argument("--n-legs", type=int, default=0, help="0 = generalize across 1..5 stations; or set fixed 1..5")
    ap.add_argument("--wind-mode", type=str, default="mixed", choices=["mixed", "disjoint", "continuous", "none"])
    ap.add_argument("--emulator-steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", type=str, default="rl_cosine_mpc.png")
    ap.add_argument("--dual-plot", type=str, default="wind_trajectory_shield_dual_axis.png")
    ap.add_argument("--load-checkpoint", type=str, default="", help="Path to .pt file (e.g. best_ppo_cosine_pilot.pt) to skip training")
    ap.add_argument("--real-data", action="store_true", help="Benchmark on real Digitraffic rail + Open-Meteo weather data")
    ap.add_argument("--train-number", type=int, default=8311, help="Finnish Digitraffic train number (e.g. 8311 HSL commuter or 23 InterCity)")
    ap.add_argument("--train-date", type=str, default="", help="YYYY-MM-DD departure date for Digitraffic + Open-Meteo (default: latest completed run)")
    ap.add_argument(
        "--harsh-weather",
        type=str,
        default=None,
        choices=["mixed", "always", "raw", "none"],
        help="Extreme noise injection mode (default: mixed when --real-data, none otherwise): mixed (50%% baseline + 50%% extreme), always, raw/none",
    )
    ap.add_argument(
        "--data-dir",
        type=str,
        default="collected_data",
        help="Directory to store separated baseline vs extreme-scenario datasets (JSON + CSV)",
    )
    args = ap.parse_args()

    if args.harsh_weather is not None:
        harsh_mode_eff = "raw" if args.harsh_weather == "none" else args.harsh_weather
    elif args.real_data:
        harsh_mode_eff = "raw" if args.wind_mode == "none" else "mixed"
    else:
        harsh_mode_eff = "none"

    import os, gc
    script_dir = Path(__file__).resolve().parent

    def _resolve_ckpt(path_str: str) -> str:
        if not path_str:
            return ""
        if os.path.exists(path_str):
            return path_str
        cand = script_dir / path_str
        return str(cand) if cand.exists() else path_str

    base_corridor = RailroadCorridor()
    print("=" * 152)
    mode_tag = (
        f"REAL-WORLD DIGITRAFFIC + OPEN-METEO (EXTREME_NOISE={harsh_mode_eff.upper()})"
        if args.real_data
        else f"GENERALIZED CORRIDOR (1..5 STATIONS, PASSENGER MASS & WEATHER | EXTREME_NOISE={harsh_mode_eff.upper()})"
    )
    print(
        f"MULTI-PROPERTY PPO ({args.epochs} EPOCHS) + HIERARCHICAL CONTRACT SHIELD | "
        f"{mode_tag} | {args.eval_runs} SCENARIOS"
    )
    print("=" * 152, flush=True)

    emu_ckpt_path = _resolve_ckpt("best_emulator.pt")
    load_ckpt_resolved = _resolve_ckpt(args.load_checkpoint)
    if load_ckpt_resolved and os.path.exists(emu_ckpt_path):
        print(f"\n=== 1. Loading Pre-Trained Neural Physics Emulator from '{emu_ckpt_path}' ===", flush=True)
        emulator = LearnedLinearizationModel(hidden_dim=64)
        emulator.load_state_dict(torch.load(emu_ckpt_path, map_location="cpu"))
        emulator.eval()
    else:
        print("\n=== 1. Training Heteroscedastic Neural Physics Emulator Across Varying Passenger Masses ===", flush=True)
        emulator, _ = train_150_step_mpc_trajectory_emulator(
            base_corridor, emulator_steps=args.emulator_steps, seed=args.seed
        )
        torch.save(emulator.state_dict(), emu_ckpt_path)

    if load_ckpt_resolved and os.path.exists(load_ckpt_resolved):
        print(f"\n=== 2. Loading Pre-Trained PPO Pilot Checkpoint from '{load_ckpt_resolved}' ===", flush=True)
        rl_policy = CosineTrajectoryRLPolicy(obs_dim=15, act_dim=3, hidden_dim=64)
        rl_policy.load_state_dict(torch.load(load_ckpt_resolved, map_location="cpu"))
        rl_policy.eval()
    else:
        print(f"\n=== 2. Training 15-Feature Multi-Property PPO Pilot Across {args.epochs} Epochs ===", flush=True)
        rl_policy, _ = train_pure_ppo_cosine_pilot(
            base_corridor, emulator=emulator, epochs=args.epochs, wind_mode=args.wind_mode, seed=args.seed
        )

    free_ckpt_path = _resolve_ckpt("best_ppo_free_pilot.pt")
    if load_ckpt_resolved and os.path.exists(load_ckpt_resolved):
        if os.path.exists(free_ckpt_path):
            free_sd = torch.load(free_ckpt_path, map_location="cpu")
            if float(torch.abs(free_sd["actor_mean.4.weight"]).max().item()) > 1e-6:
                print(f"=== 2b. Loading Pre-Trained Stable-Accel (No-Cosine) PPO Pilot from '{free_ckpt_path}' ===", flush=True)
                rl_policy_free = CosineTrajectoryRLPolicy(obs_dim=15, act_dim=3, hidden_dim=64)
                rl_policy_free.load_state_dict(free_sd)
                rl_policy_free.eval()
            else:
                rl_policy_free = rl_policy
        else:
            rl_policy_free = rl_policy
    else:
        print(f"\n=== 2b. Training Parallel Stable-Accel (No-Cosine) PPO Pilot Across {args.epochs} Epochs ===", flush=True)
        rl_policy_free, _ = train_pure_ppo_cosine_pilot(
            base_corridor, emulator=emulator, epochs=args.epochs, wind_mode=args.wind_mode,
            seed=args.seed + 100, checkpoint_prefix="ppo_free_pilot", use_cosine_shaper=False
        )

    method_specs = [
        ("1. Max-Speed Greedy Cosine + MPC", "MAX_SPEED_GREEDY", None),
        ("2. Naive Pace (No Cruise Overshoot) + MPC", "NAIVE_NO_OVERSHOOT", None),
        ("3. Static Plan (No Wind/Mass Replan) + MPC", "STATIC_NO_WIND_REPLAN", None),
        ("4. Multi-Property PPO Pilot (Unshielded)", "RL_LEARNED", None),
        ("5. Contract-Shielded PPO (Switch MPS)", "RL_LEARNED", "switch"),
        ("6. Contract-Shielded PPO (1-Step DCBF)", "RL_LEARNED", "cbf_opt"),
        ("7. Contract-Shielded PPO (UCB-DMPS Cosine)", "RL_LEARNED", "mps_rollout"),
        ("8. Stable-Accel PPO (No Cosine, Unshielded)", "RL_FREE_LEARNED", None),
        ("9. Stable-Accel PPO + Jerk-Horizon UCB-DMPS", "RL_FREE_LEARNED", "mps_rollout"),
        ("10. Bilbao (2025) CBTC-ATO (RMBTP + PI)", "BILBAO_CBTC_ATO", None),
    ]

    agg_results = {name: [] for name, _, _ in method_specs}
    scenarios_list = []
    sample_base_trajs = {}
    sample_base_scenario = None
    sample_base_no_wind = None
    sample_ext_trajs = {}
    sample_ext_scenario = None
    sample_ext_no_wind = None
    multi_scen_dmps = []

    for ep in range(args.eval_runs):
        if args.real_data:
            scen = fetch_real_world_multi_station_scenario(
                date_str=(args.train_date if args.train_date else None),
                train_number=args.train_number,
                n_legs_max=args.n_legs,
                seed=9000 + ep * 29,
                ep_idx=ep,
                harsh_weather_mode=harsh_mode_eff,
            )
            if ep == 0:
                l0 = scen["legs"][0]
                gps_txt = (
                    f" | LiveGPS={l0['latest_gps_speed_mps']*3.6:.0f}km/h"
                    if l0.get("latest_gps_speed_mps") is not None
                    else ""
                )
                print(
                    f"\n[REAL-WORLD DATA MERGER] Source={scen['data_source']} | Train #{scen['train_number']} "
                    f"({scen['date_str']}, {scen['composition_desc']}{gps_txt})\n"
                    f"  Leg 1: {l0['corridor'].route_name}\n"
                    f"  Leg 1 Weather: {l0['wind_desc']} | Regime={l0['weather']} (mu={l0['adhesion_mu']:.2f})",
                    flush=True,
                )
        else:
            if harsh_mode_eff == "always":
                is_ext_synth = True
            elif harsh_mode_eff == "mixed":
                is_ext_synth = (ep % 2 == 1)
            else:
                is_ext_synth = False
            scen = generate_multi_station_scenario(
                seed=9000 + ep * 29,
                n_legs=args.n_legs,
                wind_mode=args.wind_mode,
                allow_infeasible_breaks=(ep > 0),
                extreme_noise=is_ext_synth,
            )
        scenarios_list.append(scen)
        is_ext_ep = bool(scen.get("is_extreme_scenario", scen.get("is_harsh_injected", False)))
        capture_base = (sample_base_scenario is None) and (not is_ext_ep)
        capture_ext = (sample_ext_scenario is None) and is_ext_ep
        capture_fallback_ep0 = (ep == 0) and (sample_base_scenario is None) and (sample_ext_scenario is None)

        if capture_base or capture_fallback_ep0:
            sample_base_scenario = scen
            sample_base_no_wind = evaluate_multi_station_journey(
                "Nominal No-Wind", scen, "RL_LEARNED", rl_policy=rl_policy,
                shield_mode=None, emulator=emulator, record_traj=True, zero_wind_override=True
            )
        if capture_ext:
            sample_ext_scenario = scen
            sample_ext_no_wind = evaluate_multi_station_journey(
                "Nominal No-Wind (Extreme)", scen, "RL_LEARNED", rl_policy=rl_policy,
                shield_mode=None, emulator=emulator, record_traj=True, zero_wind_override=True
            )

        for name, mode_str, sh_mode in method_specs:
            need_traj = capture_base or capture_ext or capture_fallback_ep0 or (sh_mode == "mps_rollout")
            active_pol = rl_policy_free if (mode_str == "RL_FREE_LEARNED") else rl_policy
            res = evaluate_multi_station_journey(
                label=name,
                scenario=scen,
                mode=mode_str,
                rl_policy=active_pol,
                shield_mode=sh_mode,
                emulator=emulator,
                record_traj=need_traj,
            )
            agg_results[name].append(res)
            if capture_base or capture_fallback_ep0:
                sample_base_trajs[name] = res
            if capture_ext:
                sample_ext_trajs[name] = res
            if sh_mode == "mps_rollout" and ep < 5:
                multi_scen_dmps.append(res)

        if args.real_data and ((ep + 1) % 5 == 0 or (ep + 1) == args.eval_runs):
            cat_str = "EXTREME" if is_ext_ep else "RAW-API"
            print(
                f"  [Eval {ep + 1:2d}/{args.eval_runs}] Train #{scen.get('train_number', '?')} "
                f"({scen['n_legs']} legs, {scen['total_length_m']/1000:.2f} km) | Category={cat_str}",
                flush=True,
            )

    baseline_indices = [
        i for i, sc in enumerate(scenarios_list)
        if not sc.get("is_extreme_scenario", sc.get("is_harsh_injected", False))
    ]
    extreme_indices = [
        i for i, sc in enumerate(scenarios_list)
        if sc.get("is_extreme_scenario", sc.get("is_harsh_injected", False))
    ]

    def _print_sample_subset(subset_title: str, idxs: list[int]) -> None:
        if not idxs:
            return
        print("\n" + "=" * 168)
        print(subset_title)
        print("-" * 168)
        print(
            f"{'Scen#':<5} | {'Stops':>5} | {'Total Len':>9} | {'Max|Wind|':>9} | {'MinMu':>5} | "
            f"{'Train & Mass Evolution Across Stops':<44} | {'|TimeErr|':>9} | {'|StopErr|':>9} | {'SpecEnergy':>12} | {'MultiScore':>10} | {'Shield%':>7}"
        )
        print("-" * 168)
        for ep_i in idxs[:10]:
            r_d = agg_results["7. Contract-Shielded PPO (UCB-DMPS Cosine)"][ep_i]
            scen_i = scenarios_list[ep_i]
            max_w = max((float(np.max(np.abs(l["wind_profile"]))) for l in scen_i["legs"]), default=0.0)
            min_mu = min((float(l["adhesion_mu"]) for l in scen_i["legs"]), default=1.0)
            m_str = "->".join([f"{l['mass_kg']/1000:.1f}t" for l in scen_i["legs"]])
            if args.real_data:
                m_str = f"#{scen_i.get('train_number', '?')} {m_str}"
            if len(m_str) > 44:
                m_str = m_str[:41] + "..."
            print(
                f"{ep_i + 1:<5d} | {r_d['n_legs']:5d} | {r_d['total_length_m']/1000:6.2f} km | {max_w:6.2f}m/s²| {min_mu:5.2f} | "
                f"{m_str:<44} | {r_d['abs_sched_err_s']:7.2f} s | {r_d['abs_stop_err_m']:7.3f} m | "
                f"{r_d['spec_energy_wh_tkm']:7.2f} Wh/tk | {r_d['multi_obj_score']:10.4f} | {r_d['shield_rate_pct']:6.1f}%"
            )
        print("=" * 168)

    if args.real_data:
        _print_sample_subset(
            f"BASELINE REAL-WORLD API SCENARIOS (UNPERTURBED DIGITRAFFIC + OPEN-METEO | {len(baseline_indices)}/{args.eval_runs}) — CONTRACT-SHIELDED PPO",
            baseline_indices,
        )
        if extreme_indices:
            _print_sample_subset(
                f"EXTREME NOISE-INJECTED API SCENARIOS (COLLECTED APART | {len(extreme_indices)}/{args.eval_runs}) — CONTRACT-SHIELDED PPO",
                extreme_indices,
            )
    else:
        _print_sample_subset(
            f"BASELINE SYNTHETIC SCENARIOS ({len(baseline_indices)}/{args.eval_runs}) — CONTRACT-SHIELDED PPO",
            baseline_indices,
        )
        if extreme_indices:
            _print_sample_subset(
                f"EXTREME NOISE-INJECTED SYNTHETIC SCENARIOS (COLLECTED APART | {len(extreme_indices)}/{args.eval_runs}) — CONTRACT-SHIELDED PPO",
                extreme_indices,
            )

    # Weather Regime Distribution separated across Baseline vs Extreme legs
    for grp_lbl, idxs in (("BASELINE", baseline_indices), ("EXTREME NOISE", extreme_indices)):
        if not idxs:
            continue
        weather_leg_counts: dict[str, int] = {}
        for idx in idxs:
            for l_item in scenarios_list[idx]["legs"]:
                w_name = l_item.get("weather", "Clean-Dry")
                weather_leg_counts[w_name] = weather_leg_counts.get(w_name, 0) + 1
        w_summary = " | ".join([f"{k}: {v} legs" for k, v in sorted(weather_leg_counts.items())])
        print(f"[{grp_lbl} WEATHER DISTRIBUTION ACROSS {sum(weather_leg_counts.values())} LEGS] {w_summary}")

    # Persist collected scenario data completely apart (Baseline vs Extreme)
    data_out_dir = Path(args.data_dir)
    if not data_out_dir.is_absolute():
        data_out_dir = script_dir / data_out_dir
    data_prefix = "real_world_api" if args.real_data else "synthetic"
    exported_files = _export_separated_scenario_datasets(
        data_out_dir, data_prefix, baseline_indices, extreme_indices, scenarios_list, agg_results, method_specs
    )
    print(f"\n[SEPARATED DATA COLLECTION SAVED TO '{data_out_dir}']", flush=True)
    for k_exp, v_exp in exported_files.items():
        print(f"  -> {k_exp:<15s}: {v_exp}", flush=True)

    if args.real_data:
        subset_defs = [
            ("1. REAL OPEN-METEO BASELINE (UNPERTURBED API WEATHER) SCENARIOS ONLY", lambda r: not r.get("is_extreme_scenario", r.get("is_harsh_injected", False))),
            ("2. EXTREME NOISE-INJECTED API SCENARIOS ONLY (COLLECTED APART)", lambda r: r.get("is_extreme_scenario", r.get("is_harsh_injected", False))),
            ("3. UNITED REAL-WORLD BENCHMARK: ALL REAL TRAIN SCENARIOS COMBINED", lambda r: True),
        ]
    elif extreme_indices:
        subset_defs = [
            ("1. BASELINE SYNTHETIC SCENARIOS ONLY", lambda r: not r.get("is_extreme_scenario", False)),
            ("2. EXTREME NOISE-INJECTED SCENARIOS ONLY (COLLECTED APART)", lambda r: r.get("is_extreme_scenario", False)),
            ("3. UNITED BENCHMARK: ALL SCENARIOS COMBINED", lambda r: True),
        ]
    else:
        subset_defs = [
            ("1. NO-WEATHER (CLEAN) SCENARIOS ONLY", lambda r: not r.get("is_weather_disturbed", False)),
            ("2. WEATHER-DISTURBED (WIND / RAIN / SNOW / COLD / HEAT-THERMAL) SCENARIOS ONLY", lambda r: r.get("is_weather_disturbed", True)),
            ("3. UNITED BENCHMARK: ALL GENERALIZED SCENARIOS COMBINED", lambda r: True),
        ]

    for subset_title, filter_fn in subset_defs:
        sample_sub = [r for r in agg_results[method_specs[0][0]] if filter_fn(r)]
        if not sample_sub:
            continue
        print("\n" + "=" * 168)
        print(f"{subset_title} ({len(sample_sub)} Scenarios | 1..5 Stations | Inference Latency in us)")
        print("-" * 168)
        print(
            f"{'Trajectory Pilot + Contract Shield Method':<42} | {'|TimeErr| (s)':>13} | {'|StopErr|':>9} | "
            f"{'SmoothLoss':>10} | {'SpecEnergy':>13} | {'LossScore':>9} | {'Shield%':>7} | "
            f"{'Core Step':>9} | {'Total (Mean/P99/Max)':>22}"
        )
        print("-" * 168)
        for name, _, _ in method_specs:
            runs = [r for r in agg_results[name] if filter_fn(r)]
            t_m = float(np.mean([r["abs_sched_err_s"] for r in runs]))
            x_m = float(np.mean([r["abs_stop_err_m"] for r in runs]))
            c_m = float(np.mean([r["step_smooth_cost"] for r in runs]))
            e_m = float(np.mean([r["spec_energy_wh_tkm"] for r in runs]))
            s_m = float(np.mean([r["multi_obj_score"] for r in runs]))
            sh_m = float(np.mean([r["shield_rate_pct"] for r in runs]))
            mpc_us = float(np.mean([r["mean_step_us"] for r in runs]))
            tot_us = float(np.mean([r["mean_total_us"] for r in runs]))
            p99_us = float(np.mean([r["p99_total_us"] for r in runs]))
            max_us = float(np.max([r["max_total_us"] for r in runs]))
            print(
                f"{name:<42} | {t_m:11.2f} s | {x_m:7.3f} m | "
                f"{c_m:10.4f} | {e_m:8.2f} Wh/tk | {s_m:9.4f} | {sh_m:6.1f}% | "
                f"{mpc_us:6.1f} us | {tot_us:5.1f}/{p99_us:5.1f}/{max_us:6.1f} us"
            )
        print("=" * 168, flush=True)

    # Diagnostic Profiler & Scissor Diagnostics separated for Baseline vs Extreme Scenarios
    dmps_runs = agg_results["7. Contract-Shielded PPO (UCB-DMPS Cosine)"]
    dmps_base = [dmps_runs[i] for i in baseline_indices]
    dmps_ext = [dmps_runs[i] for i in extreme_indices]
    if dmps_base:
        _print_spike_profiler(dmps_base, "BASELINE SCENARIOS")
        print_goodall_trajectory_scissor_diagnostics(dmps_base, subset_label="BASELINE SCENARIOS")
    if dmps_ext:
        _print_spike_profiler(dmps_ext, "EXTREME NOISE SCENARIOS")
        print_goodall_trajectory_scissor_diagnostics(dmps_ext, subset_label="EXTREME NOISE SCENARIOS")

    plot_base_scen = sample_base_scenario if sample_base_scenario is not None else sample_ext_scenario
    plot_base_nw = sample_base_no_wind if sample_base_no_wind is not None else sample_ext_no_wind
    plot_base_trajs = sample_base_trajs if sample_base_trajs else sample_ext_trajs
    base_tag = "RAW-API BASELINE" if (args.real_data and not plot_base_scen.get("is_extreme_scenario", False)) else (
        "EXTREME NOISE" if plot_base_scen.get("is_extreme_scenario", False) else "BASELINE"
    )
    save_dual_axis_wind_trajectory_shield_plot(
        scenario=plot_base_scen,
        run_nominal_no_wind=plot_base_nw,
        run_unshielded_wind=plot_base_trajs["4. Multi-Property PPO Pilot (Unshielded)"],
        run_shielded_wind=plot_base_trajs["7. Contract-Shielded PPO (UCB-DMPS Cosine)"],
        out_path=args.dual_plot,
        tag_label=base_tag,
    )
    if sample_ext_scenario is not None and sample_ext_trajs and sample_ext_scenario is not plot_base_scen:
        dp = Path(args.dual_plot)
        ext_dual_path = str(dp.with_name(f"{dp.stem}_extreme{dp.suffix}"))
        save_dual_axis_wind_trajectory_shield_plot(
            scenario=sample_ext_scenario,
            run_nominal_no_wind=sample_ext_no_wind,
            run_unshielded_wind=sample_ext_trajs["4. Multi-Property PPO Pilot (Unshielded)"],
            run_shielded_wind=sample_ext_trajs["7. Contract-Shielded PPO (UCB-DMPS Cosine)"],
            out_path=ext_dual_path,
            tag_label="EXTREME NOISE",
        )

    # ---------------- 6-Panel Benchmark-Relevant Trajectory & Multi-Stat Figure ----------------
    sample_plot_trajs = plot_base_trajs
    tr_dmps = sample_plot_trajs["7. Contract-Shielded PPO (UCB-DMPS Cosine)"]["traj"]
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.0))
    plot_names = [
        "1. Max-Speed Greedy Cosine + MPC",
        "2. Naive Pace (No Cruise Overshoot) + MPC",
        "3. Static Plan (No Wind/Mass Replan) + MPC",
        "10. Bilbao (2025) CBTC-ATO (RMBTP + PI)",
        "4. Multi-Property PPO Pilot (Unshielded)",
        "7. Contract-Shielded PPO (UCB-DMPS Cosine)",
        "9. Stable-Accel PPO + Jerk-Horizon UCB-DMPS",
    ]
    cols = ["steelblue", "darkorange", "purple", "goldenrod", "seagreen", "crimson", "teal"]
    lss = ["--", ":", "-.", "-.", "--", "-", "-."]

    # Panel 1: Multi-Station Speed Trajectories v(x)
    for name, c, ls in zip(plot_names, cols, lss):
        r = sample_plot_trajs[name]
        axes[0, 0].plot(r["traj"]["x"], r["traj"]["v"], color=c, linestyle=ls, linewidth=1.5, label=name)
    axes[0, 0].set_title("1. Multi-Station Speed Trajectories v(x) Across Competing Pilots")
    axes[0, 0].set_xlabel("Cumulative Track Position x (m)")
    axes[0, 0].set_ylabel("Speed (m/s)")
    axes[0, 0].legend(fontsize=5.6, loc="lower left")
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Smooth Acceleration (Order d=2) & Jerk (Order d=3)
    t_ep0 = tr_dmps["t"]
    axes[0, 1].plot(t_ep0, tr_dmps["accel"], color="crimson", linewidth=1.3, label="Longitudinal Accel a(t) [Order d=2]")
    axes[0, 1].plot(t_ep0, tr_dmps["jerk"], color="teal", alpha=0.60, linewidth=1.0, label="Longitudinal Jerk j(t) [Order d=3]")
    axes[0, 1].axhline(-0.78, color="seagreen", linestyle="--", label="Comfort Landing Target (-0.78 m/s²)")
    axes[0, 1].axhline(-0.92, color="darkorange", linestyle=":", label="Comfort Braking Limit (-0.92 m/s²)")
    axes[0, 1].set_title("2. Hierarchical Smoothness: Accel a(t) & Jerk j(t)")
    axes[0, 1].set_xlabel("Clock Time t (s)")
    axes[0, 1].set_ylabel("Accel (m/s²) / Jerk (m/s³)")
    axes[0, 1].legend(fontsize=6.8, loc="lower left")
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Online 8-Step Mass Identification & Unseen Weather Disturbance Observer
    axes[0, 2].step(t_ep0, tr_dmps["mass_kg"] / 1000.0, color="purple", linewidth=2.0, label="True Train Mass M(t) [t]")
    axes[0, 2].plot(t_ep0, tr_dmps["mass_hat_kg"] / 1000.0, color="darkorchid", linestyle="--", linewidth=1.5, label="8-Step Concluded M_hat(t) [t]")
    ax_w2 = axes[0, 2].twinx()
    ax_w2.plot(t_ep0, tr_dmps["wind_true"], color="navy", alpha=0.55, linewidth=1.3, label="True Weather/Wind Accel (m/s²)")
    ax_w2.plot(t_ep0, tr_dmps["wind_hat"], color="dodgerblue", linestyle=":", linewidth=1.2, label="MPC Observer d_hat(t) (m/s²)")
    ax_w2.set_ylabel("Weather/Wind Accel (m/s²)", color="navy", fontsize=8.5)
    axes[0, 2].set_title("3. Online 8-Step Mass ID & Weather Impact Estimation")
    axes[0, 2].set_xlabel("Clock Time t (s)")
    axes[0, 2].set_ylabel("Train Mass (tonnes)", color="purple")
    axes[0, 2].legend(fontsize=7.0, loc="upper left")
    axes[0, 2].grid(True, alpha=0.3)

    short_labels = [
        "Greedy", "Naive", "Static", "PPO Unsh", "PPO+Sw",
        "PPO+DCBF", "PPO+DMPS", "Stab Unsh", "Stab+DMPS", "Bilbao'25"
    ]
    x_pos = np.arange(len(method_specs))
    w_b = 0.26

    # Build non-empty comparison groups for Panels 4 & 5 (avoids empty-slice NaN warnings)
    first_runs = agg_results[method_specs[0][0]]
    if args.real_data:
        cand_groups = [
            ("Real API (Baseline)", "mediumseagreen", lambda r: not r.get("is_extreme_scenario", r.get("is_harsh_injected", False))),
            ("Extreme Noise (Apart)", "royalblue", lambda r: r.get("is_extreme_scenario", r.get("is_harsh_injected", False))),
            ("Combined Real", "crimson", lambda r: True),
        ]
        p4_title = "4. Real-World |Deadline Error|: Raw API vs Extreme Noise vs Combined"
    elif extreme_indices:
        cand_groups = [
            ("Baseline Scenarios", "mediumseagreen", lambda r: not r.get("is_extreme_scenario", False)),
            ("Extreme Noise (Apart)", "royalblue", lambda r: r.get("is_extreme_scenario", False)),
            ("Combined (Overall)", "crimson", lambda r: True),
        ]
        p4_title = "4. Overall |Deadline Error|: Baseline vs Extreme Noise vs Combined"
    else:
        cand_groups = [
            ("No-Weather (Clean)", "mediumseagreen", lambda r: not r.get("is_weather_disturbed", False)),
            ("Weather-Disturbed", "royalblue", lambda r: r.get("is_weather_disturbed", True)),
            ("Combined (Overall)", "crimson", lambda r: True),
        ]
        p4_title = "4. Overall |Deadline Error|: Clean vs Weather vs Combined"

    active_groups = [
        (lbl, col, fn) for (lbl, col, fn) in cand_groups
        if any(fn(r) for r in first_runs)
    ]
    if len(active_groups) == 2 and all(active_groups[0][2](r) for r in first_runs):
        active_groups = [active_groups[-1]]
    bar_w = min(0.55, 0.78 / max(1, len(active_groups)))
    offsets = (np.arange(len(active_groups)) - 0.5 * (len(active_groups) - 1)) * bar_w

    # Panel 4: Mean |Deadline Error|
    for g_off, (g_lbl, g_col, g_fn) in zip(offsets, active_groups):
        vals = [float(np.mean([r["abs_sched_err_s"] for r in agg_results[k] if g_fn(r)])) for k, _, _ in method_specs]
        axes[1, 0].bar(x_pos + g_off, vals, width=bar_w, color=g_col, alpha=0.88, label=g_lbl)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(short_labels, fontsize=6.6, rotation=16, ha="right")
    axes[1, 0].set_title(p4_title)
    axes[1, 0].set_ylabel("Mean |Deadline Error| (s)")
    axes[1, 0].legend(fontsize=7.2)
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    # Panel 5: Overall Multi-Objective LossScore
    for g_off, (g_lbl, g_col, g_fn) in zip(offsets, active_groups):
        vals = [float(np.mean([r["multi_obj_score"] for r in agg_results[k] if g_fn(r)])) for k, _, _ in method_specs]
        axes[1, 1].bar(x_pos + g_off, vals, width=bar_w, color=g_col, alpha=0.88, label=g_lbl)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(short_labels, fontsize=6.6, rotation=16, ha="right")
    axes[1, 1].set_title("5. Multi-Objective LossScore (Lower = Better)")
    axes[1, 1].set_ylabel("Composite LossScore")
    axes[1, 1].legend(fontsize=7.2)
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    # Panel 6: Multi-Property Breakdown: SmoothLoss, StopErr (cm), & Specific Energy (Wh/t-km)
    smooth_vals = [float(np.mean([r["step_smooth_cost"] for r in agg_results[k]])) * 15.0 for k, _, _ in method_specs]
    stop_cm_vals = [float(np.mean([r["abs_stop_err_m"] for r in agg_results[k]])) * 100.0 for k, _, _ in method_specs]
    energy_norm_vals = [float(np.mean([r["spec_energy_wh_tkm"] for r in agg_results[k]])) / 25.0 for k, _, _ in method_specs]
    axes[1, 2].bar(x_pos - w_b, smooth_vals, width=w_b, color="teal", alpha=0.85, label="15x SmoothLoss (d=2,3)")
    axes[1, 2].bar(x_pos, stop_cm_vals, width=w_b, color="darkorchid", alpha=0.85, label="Stop Error (cm)")
    axes[1, 2].bar(x_pos + w_b, energy_norm_vals, width=w_b, color="darkorange", alpha=0.85, label="SpecEnergy / 25 (Wh/tk)")
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(short_labels, fontsize=6.6, rotation=16, ha="right")
    axes[1, 2].set_title("6. Multi-Property Breakdown: Smoothness, Docking & Energy")
    axes[1, 2].set_ylabel("Normalized Metric Value (Lower = Better)")
    axes[1, 2].legend(fontsize=7.2)
    axes[1, 2].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    if not Path(args.plot).is_absolute():
        try:
            fig.savefig(f"../{args.plot}", dpi=150)
        except OSError:
            pass
    plt.close(fig)
    print(f"Saved 6-panel Multi-Property Benchmark plot to: {args.plot}", flush=True)


if __name__ == "__main__":
    main()