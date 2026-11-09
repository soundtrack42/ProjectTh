"""
Module 2: True Phase-Tracked Half-Cosine S-Curve Shaper for All Shifts (Up & Down) and
Dynamic Speed-, Mass-, & Wind-Updated Station Landing (`cosine_trajectory_shaper.py`).
"""
from __future__ import annotations

import math
import numpy as np
from railroad_corridor import RailroadCorridor


class CosineTrajectoryShaper:
    """
    Generates C^1-smooth half-cosine velocity profiles for:
      1. Every departure and upward/downward speed shift (bounded by passenger comfort
         acceleration [0.65, 1.20] m/s^2 and vehicle mass capacity).
      2. Dynamic station landing: continuously updates the half-cosine landing curve
         `v(u) = 0.5 * v_entry_dyn * (1 + cos(pi * u))` based on current speed `v_curr`,
         remaining distance `rem_d`, remaining deadline time `t_rem`, and wind `d_wind_hat`,
         so landing deceleration is a smooth bell curve peaking at ~0.75-0.88 m/s^2.
    """
    def __init__(self, corridor: RailroadCorridor):
        self.corridor = corridor
        self._v_ref_buf = np.zeros(5, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.ramp_v_start: float = 0.0
        self.ramp_v_goal: float = 0.0
        self.ramp_t_dur: float = 1.0
        self.ramp_u_phase: float = 1.0

        self.in_landing: bool = False
        self.x_brake_dynamic: float = self.corridor.station_x_m
        self.v_entry_ref: float = 12.5
        self.dx_land_ref: float = 150.0
        self.t_land_ref: float = 24.0
        self.b_peak_ref: float = self.corridor.comfort_braking_decel

    def feasible_comfort_accel(self, v_mps: float, a_req: float, d_wind_hat: float = 0.0) -> float:
        scale = 1.0 if v_mps <= 14.0 else (14.0 / (v_mps if v_mps > 14.0 else 14.0))
        a_motor_max = self.corridor.max_traction_accel * scale * 0.90
        a_drag = 0.02 + 0.00018 * (v_mps * v_mps)
        w_neg = d_wind_hat if d_wind_hat < 0.0 else 0.0
        a_phys_cap = a_motor_max - a_drag + w_neg
        if a_phys_cap < 0.30:
            a_phys_cap = 0.30
        a_hi = self.corridor.comfort_accel_max if self.corridor.comfort_accel_max < a_phys_cap else a_phys_cap
        return 0.45 if a_req < 0.45 else (a_hi if a_req > a_hi else float(a_req))

    def feasible_comfort_braking(self, b_req: float, d_wind_hat: float = 0.0) -> float:
        w_pos = d_wind_hat if d_wind_hat > 0.0 else 0.0
        b_phys_cap = self.corridor.max_service_braking * 0.88 - w_pos
        if b_phys_cap < 0.35:
            b_phys_cap = 0.35
        b_hi = self.corridor.comfort_braking_max if self.corridor.comfort_braking_max < b_phys_cap else b_phys_cap
        return 0.50 if b_req < 0.50 else (b_hi if b_req > b_hi else float(b_req))

    @staticmethod
    def invert_cosine_distance_phase(rem_ratio: float) -> float:
        """
        Inverts the normalized half-cosine remaining distance equation:
            tilde_d(u) = 1 - u - sin(pi * u) / pi = rem_ratio  for u in [0, 1].
        """
        r = 0.0 if rem_ratio < 0.0 else (1.0 if rem_ratio > 1.0 else float(rem_ratio))
        if r >= 1.0 - 1e-7:
            return 0.0
        if r <= 1e-7:
            return 1.0
        pi_val = math.pi
        u = 1.0 - ((6.0 * r / (pi_val * pi_val)) ** (1.0 / 3.0)) if r < 0.12 else (1.0 - r)
        u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
        for _ in range(5):
            pi_u = pi_val * u
            f_val = (1.0 - u - math.sin(pi_u) / pi_val) - r
            df_val = -(1.0 + math.cos(pi_u))
            denom = df_val if df_val < -1e-6 else -1e-6
            u = u - f_val / denom
            u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
        return u

    @classmethod
    def dynamic_cosine_braking_spot(
        cls,
        corridor: RailroadCorridor,
        v_curr: float,
        b_land_cmd: float,
        t_rem_deadline: float | None = None,
        v_scissor_rad: float = 0.0,
        d_wind_hat: float = 0.0,
    ) -> tuple[float, float, float, float]:
        """
        Computes the dynamic half-cosine braking spot `x_brake` from current velocity,
        vehicle braking capacity, active wind `d_wind_hat`, and remaining time `t_rem_deadline`.
        """
        v_scis = v_scissor_rad if v_scissor_rad > 0.0 else 0.0
        v_eff = v_curr + v_scis
        if v_eff <= 1e-4:
            return corridor.station_x_m, 0.0, 0.0, b_land_cmd

        w_pos = d_wind_hat if d_wind_hat > 0.0 else 0.0
        b_phys_max = corridor.max_service_braking * 0.86 - w_pos
        if b_phys_max < 0.45:
            b_phys_max = 0.45
        b_comf_hi = corridor.comfort_braking_max if corridor.comfort_braking_max < b_phys_max else b_phys_max
        b_comf_lo = 0.62

        # Base wind-adjusted comfort deceleration
        b_raw = b_land_cmd - 0.12 * w_pos
        b_target = b_comf_lo if b_raw < b_comf_lo else (b_comf_hi if b_raw > b_comf_hi else float(b_raw))

        # Deadline-synchronized landing rate within [b_comf_lo, b_comf_hi]
        if t_rem_deadline is not None and 10.0 < t_rem_deadline < 45.0:
            b_for_exact_time = (math.pi * v_eff) / (2.0 * t_rem_deadline)
            b_blend = 0.75 * b_for_exact_time + 0.25 * b_target
            b_target = b_comf_lo if b_blend < b_comf_lo else (b_comf_hi if b_blend > b_comf_hi else float(b_blend))

        t_land = (math.pi * v_eff) / (2.0 * b_target)
        dx_brake = 0.5 * v_eff * t_land
        x_brake = corridor.station_x_m - dx_brake
        return float(x_brake), float(dx_brake), float(t_land), b_target

    def _update_smooth_shift_ramp(
        self,
        x_curr: float,
        v_curr: float,
        v_goal_now: float,
        a_start_cos: float,
        b_land_cos: float,
        d_wind_hat: float,
    ) -> None:
        """
        Advances a C^2 half-cosine S-curve for speed transitions:
            v(u) = v_start + (v_goal - v_start) * 0.5 * (1 - cos(pi * u)),  u in [0, 1]
        subject to Order-2 acceleration and Order-3 jerk bounds.
        """
        is_departing_from_rest = (v_curr < 1.35 and self.ramp_v_start < 1.0)
        retrigger_tol = 0.15 if self.ramp_u_phase >= 0.98 else 0.45

        if abs(v_goal_now - self.ramp_v_goal) > retrigger_tol:
            # Start a new half-cosine S-curve from the current smooth reference velocity
            v_anchor = (
                self.ramp_v_start + (self.ramp_v_goal - self.ramp_v_start) * 0.5 * (1.0 - math.cos(math.pi * self.ramp_u_phase))
                if self.ramp_u_phase < 0.99
                else float(v_curr)
            )
            dv = v_goal_now - v_anchor
            if abs(dv) <= 0.04:
                self.ramp_v_start = float(v_goal_now)
                self.ramp_v_goal = float(v_goal_now)
                self.ramp_t_dur = self.corridor.dt
                self.ramp_u_phase = 1.0
            else:
                if dv >= 0.0:
                    ar_raw = a_start_cos / 0.96
                    a_ratio = 0.72 if ar_raw < 0.72 else (1.28 if ar_raw > 1.28 else ar_raw)
                    cruise_scale = 1.0 if is_departing_from_rest else (0.62 * a_ratio)
                    rate = self.feasible_comfort_accel(v_curr, a_start_cos * cruise_scale, d_wind_hat)
                    j_comf = (0.38 if is_departing_from_rest else 0.20) * (a_ratio ** 1.6)
                    min_sinusoid_dur = (16.0 if is_departing_from_rest else 12.0) / a_ratio
                else:
                    br_raw = b_land_cos / 0.78
                    b_ratio = 0.75 if br_raw < 0.75 else (1.25 if br_raw > 1.25 else br_raw)
                    rate = self.feasible_comfort_braking(b_land_cos * (0.72 * b_ratio), d_wind_hat)
                    j_comf = 0.20 * (b_ratio ** 1.6)
                    min_sinusoid_dur = 12.0 / b_ratio

                # Order-3 Cosine Jerk bound: |j_max| = pi^2 * |dv| / (2 * T^2) <= j_comf
                abs_dv = abs(dv)
                t_from_acc = (math.pi * abs_dv) / (2.0 * (rate if rate > 0.30 else 0.30))
                t_from_jrk = math.pi * math.sqrt(abs_dv / (2.0 * j_comf))

                self.ramp_v_start = float(v_anchor)
                self.ramp_v_goal = float(v_goal_now)
                self.ramp_t_dur = float(max(min_sinusoid_dur, t_from_acc, t_from_jrk))
                self.ramp_u_phase = 0.0
        else:
            # Softly update goal for small drift while advancing phase u strictly by dt / T_ramp
            self.ramp_v_goal = float(0.85 * self.ramp_v_goal + 0.15 * v_goal_now)
            if self.ramp_u_phase < 1.0:
                u_n = self.ramp_u_phase + (self.corridor.dt / self.ramp_t_dur)
                self.ramp_u_phase = 1.0 if u_n > 1.0 else u_n

    def step_reference_horizon(
        self,
        x_curr: float,
        v_curr: float,
        t_curr: float,
        v_cmd_rl: float,
        a_start_cos: float,
        b_land_cos: float,
        n_p: int,
        v_scissor_rad: float = 0.0,
        d_wind_hat: float = 0.0,
    ) -> tuple[np.ndarray, float, float]:
        dt = self.corridor.dt
        rem_dist_now = max(0.0, self.corridor.station_x_m - x_curr)
        t_rem_now = max(dt, self.corridor.deadline_s - t_curr)
        pi_val = math.pi

        x_brake_now, dx_brake_now, t_land_now, b_eff_now = self.dynamic_cosine_braking_spot(
            self.corridor,
            v_curr,
            b_land_cos,
            t_rem_deadline=t_rem_now if rem_dist_now < 420.0 else None,
            v_scissor_rad=v_scissor_rad,
            d_wind_hat=d_wind_hat,
        )

        # Trigger landing when train reaches the dynamic braking spot
        if (not self.in_landing) and (x_curr >= x_brake_now - 0.5 * v_curr * dt) and (rem_dist_now < 450.0):
            self.in_landing = True
            self.x_brake_dynamic = float(x_curr)
            self.v_entry_ref = float(max(1.0, v_curr))
            self.dx_land_ref = float(max(1.0, rem_dist_now))
            self.b_peak_ref = float((pi_val * (self.v_entry_ref ** 2)) / (4.0 * self.dx_land_ref))
            self.t_land_ref = float((pi_val * self.v_entry_ref) / (2.0 * self.b_peak_ref))
        elif not self.in_landing:
            self.x_brake_dynamic = float(x_brake_now)

        v_ref = self._v_ref_buf if n_p == 5 else np.zeros(n_p, dtype=np.float64)

        if self.in_landing:
            u_now = self.invert_cosine_distance_phase(rem_dist_now / self.dx_land_ref)
            tilde_v = max(0.04, 0.5 * (1.0 + math.cos(pi_val * u_now)))
            if u_now < 0.88 and v_curr > 0.6:
                v_entry_implied = v_curr / tilde_v
                self.v_entry_ref = 0.80 * self.v_entry_ref + 0.20 * v_entry_implied
                b_cap = self.feasible_comfort_braking(self.corridor.comfort_braking_max, d_wind_hat)
                bp_raw = (pi_val * (self.v_entry_ref ** 2)) / (4.0 * self.dx_land_ref)
                self.b_peak_ref = 0.50 if bp_raw < 0.50 else (b_cap if bp_raw > b_cap else float(bp_raw))
                self.t_land_ref = float((pi_val * self.v_entry_ref) / (2.0 * self.b_peak_ref))

            du_step = dt / max(dt, self.t_land_ref)
            for i in range(n_p):
                u_i = min(1.0, u_now + (i + 1) * du_step)
                v_cos_i = 0.5 * self.v_entry_ref * (1.0 + math.cos(pi_val * u_i))
                if rem_dist_now <= 0.03 or u_i >= 0.999:
                    v_cos_i = 0.0
                v_ref[i] = v_cos_i if v_cos_i > 0.0 else 0.0

            t_land_rem = max(0.0, (1.0 - u_now) * self.t_land_ref)
            return v_ref, self.x_brake_dynamic, float(t_land_rem)

        # Open Cruising / Smooth Cosine Shift-Up & Shift-Down Phase
        v_smooth_ceil = self.corridor.compute_smooth_spatial_ceiling(x_curr, b_land_cos)
        v_goal_clamped = 0.0 if v_cmd_rl < 0.0 else (v_smooth_ceil if v_cmd_rl > v_smooth_ceil else float(v_cmd_rl))

        self._update_smooth_shift_ramp(
            x_curr, v_curr, v_goal_clamped, a_start_cos, b_land_cos, d_wind_hat
        )
        du_ramp = dt / max(dt, self.ramp_t_dur)
        sim_x = float(x_curr)
        sim_v = float(v_curr)
        land_thresh = 1.15 * max(80.0, dx_brake_now)

        for i in range(n_p):
            u_i = min(1.0, self.ramp_u_phase + i * du_ramp)
            cos_prog = 0.5 * (1.0 - math.cos(pi_val * u_i))
            v_cos = self.ramp_v_start + (self.ramp_v_goal - self.ramp_v_start) * cos_prog

            local_ceil = self.corridor.compute_smooth_spatial_ceiling(sim_x, b_land_cos)
            rem_i = max(0.0, self.corridor.station_x_m - sim_x)
            if rem_i <= land_thresh:
                v_land_env = math.sqrt(max(0.0, (4.0 * b_eff_now * rem_i) / pi_val))
            else:
                v_land_env = local_ceil
            v_next = min(v_cos, local_ceil, v_land_env)
            if v_next < 0.0:
                v_next = 0.0

            v_ref[i] = v_next
            sim_x += 0.5 * (sim_v + v_next) * dt
            sim_v = v_next

        return v_ref, x_brake_now, t_land_now