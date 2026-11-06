"""
Module 2: True Phase-Tracked Half-Cosine S-Curve Shaper for All Shifts (Up & Down) and
Dynamic Speed-, Mass-, & Wind-Updated Station Landing (`cosine_trajectory_shaper.py`).
"""
from __future__ import annotations

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
        scale = 1.0 if v_mps <= 14.0 else (14.0 / max(v_mps, 14.0))
        a_motor_max = self.corridor.max_traction_accel * scale * 0.90
        a_drag = 0.02 + 0.00018 * (v_mps ** 2)
        a_phys_cap = max(0.30, a_motor_max - a_drag + min(0.0, d_wind_hat))
        return float(np.clip(a_req, 0.45, min(self.corridor.comfort_accel_max, a_phys_cap)))

    def feasible_comfort_braking(self, b_req: float, d_wind_hat: float = 0.0) -> float:
        b_phys_cap = max(0.35, self.corridor.max_service_braking * 0.88 - max(0.0, d_wind_hat))
        return float(np.clip(b_req, 0.50, min(self.corridor.comfort_braking_max, b_phys_cap)))

    @staticmethod
    def invert_cosine_distance_phase(rem_ratio: float) -> float:
        """
        Inverts the normalized half-cosine remaining distance equation:
            tilde_d(u) = 1 - u - sin(pi * u) / pi = rem_ratio  for u in [0, 1].
        """
        r = float(np.clip(rem_ratio, 0.0, 1.0))
        if r >= 1.0 - 1e-7:
            return 0.0
        if r <= 1e-7:
            return 1.0
        u = 1.0 - ((6.0 * r / (np.pi ** 2)) ** (1.0 / 3.0)) if r < 0.12 else (1.0 - r)
        u = float(np.clip(u, 0.0, 1.0))
        for _ in range(5):
            f_val = (1.0 - u - np.sin(np.pi * u) / np.pi) - r
            df_val = -(1.0 + np.cos(np.pi * u))
            u = float(np.clip(u - f_val / min(-1e-6, df_val), 0.0, 1.0))
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
        Computes the dynamic half-cosine braking spot `x_brake` based on:
          - Current velocity `v_curr` (+ 99% Scissor radius `v_scissor_rad`),
          - Vehicle mass braking capability & active wind `d_wind_hat`,
          - Remaining time to deadline `t_rem_deadline` (adapts `b_eff` inside the
            comfortable range [0.64, 0.92] m/s^2 so landing duration matches `t_rem`!).
        """
        v_eff = max(0.0, v_curr + max(0.0, v_scissor_rad))
        if v_eff <= 1e-4:
            return corridor.station_x_m, 0.0, 0.0, b_land_cmd

        b_phys_max = max(0.45, corridor.max_service_braking * 0.86 - max(0.0, d_wind_hat))
        b_comf_hi = min(corridor.comfort_braking_max, b_phys_max)
        b_comf_lo = 0.62

        # Base wind-adjusted comfort deceleration
        b_target = float(np.clip(b_land_cmd - 0.12 * max(0.0, d_wind_hat), b_comf_lo, b_comf_hi))

        # Deadline-synchronized landing rate: if near the braking zone, modulate b_target
        # within [b_comf_lo, b_comf_hi] so T_land = pi * v / (2 * b) aligns with remaining time!
        if t_rem_deadline is not None and 10.0 < t_rem_deadline < 42.0:
            b_for_exact_time = (np.pi * v_eff) / (2.0 * t_rem_deadline)
            b_target = float(np.clip(0.65 * b_for_exact_time + 0.35 * b_target, b_comf_lo, b_comf_hi))

        t_land = (np.pi * v_eff) / (2.0 * b_target)
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
        if abs(v_goal_now - self.ramp_v_goal) > 0.10 or self.ramp_u_phase >= 0.999:
            dv = v_goal_now - v_curr
            if abs(dv) <= 0.03:
                self.ramp_v_start = v_goal_now
                self.ramp_v_goal = v_goal_now
                self.ramp_t_dur = self.corridor.dt
                self.ramp_u_phase = 1.0
            else:
                if dv >= 0.0:
                    rate = self.feasible_comfort_accel(v_curr, a_start_cos, d_wind_hat)
                else:
                    rate = self.feasible_comfort_braking(b_land_cos, d_wind_hat)
                self.ramp_v_start = float(v_curr)
                self.ramp_v_goal = float(v_goal_now)
                self.ramp_t_dur = max(self.corridor.dt * 3.0, (np.pi * abs(dv)) / (2.0 * rate))
                self.ramp_u_phase = 0.0
        else:
            span = self.ramp_v_goal - self.ramp_v_start
            if abs(span) > 0.06:
                p = float(np.clip((v_curr - self.ramp_v_start) / span, 0.0, 1.0))
                u_from_v = float(np.arccos(np.clip(1.0 - 2.0 * p, -1.0, 1.0)) / np.pi)
                min_step_u = self.ramp_u_phase + (self.corridor.dt / self.ramp_t_dur)
                self.ramp_u_phase = float(np.clip(max(u_from_v, min_step_u), 0.0, 1.0))
            else:
                self.ramp_u_phase = 1.0

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
            self.b_peak_ref = float((np.pi * (self.v_entry_ref ** 2)) / (4.0 * self.dx_land_ref))
            self.t_land_ref = float((np.pi * self.v_entry_ref) / (2.0 * self.b_peak_ref))
        elif not self.in_landing:
            self.x_brake_dynamic = float(x_brake_now)

        v_ref = np.zeros(n_p, dtype=np.float64)

        if self.in_landing:
            # True Phase-Tracked & Wind-Updated Half-Cosine Landing S-Curve:
            # 1. Invert current remaining distance ratio `rem_dist_now / dx_land_ref` for spatial phase `u_now`
            u_now = self.invert_cosine_distance_phase(rem_dist_now / self.dx_land_ref)

            # 2. Dynamically update `v_entry_dyn` and `t_land_dyn` from current speed `v_curr`
            #    so if wind pushes or drags the train during landing, the cosine curve adapts smoothly
            #    without ever jumping or dumping speed in the final step!
            tilde_v = max(0.04, 0.5 * (1.0 + float(np.cos(np.pi * u_now))))
            if u_now < 0.88 and v_curr > 0.6:
                v_entry_implied = v_curr / tilde_v
                self.v_entry_ref = 0.80 * self.v_entry_ref + 0.20 * v_entry_implied
                b_cap = self.feasible_comfort_braking(self.corridor.comfort_braking_max, d_wind_hat)
                self.b_peak_ref = float(
                    np.clip((np.pi * (self.v_entry_ref ** 2)) / (4.0 * self.dx_land_ref), 0.50, b_cap)
                )
                self.t_land_ref = float((np.pi * self.v_entry_ref) / (2.0 * self.b_peak_ref))

            du_step = dt / max(dt, self.t_land_ref)
            for i in range(n_p):
                u_i = min(1.0, u_now + (i + 1) * du_step)
                v_cos_i = 0.5 * self.v_entry_ref * (1.0 + float(np.cos(np.pi * u_i)))
                if rem_dist_now <= 0.03 or u_i >= 0.999:
                    v_cos_i = 0.0
                v_ref[i] = float(max(0.0, v_cos_i))

            t_land_rem = max(0.0, (1.0 - u_now) * self.t_land_ref)
            return v_ref, self.x_brake_dynamic, float(t_land_rem)

        # Open Cruising / Smooth Cosine Shift-Up & Shift-Down Phase
        v_smooth_ceil = self.corridor.compute_smooth_spatial_ceiling(x_curr, b_land_cos)
        v_goal_clamped = float(np.clip(v_cmd_rl, 0.0, v_smooth_ceil))

        self._update_smooth_shift_ramp(
            x_curr, v_curr, v_goal_clamped, a_start_cos, b_land_cos, d_wind_hat
        )
        du_ramp = dt / max(dt, self.ramp_t_dur)
        sim_x = float(x_curr)
        sim_v = float(v_curr)

        for i in range(n_p):
            u_i = min(1.0, self.ramp_u_phase + (i + 1) * du_ramp)
            cos_prog = 0.5 * (1.0 - float(np.cos(np.pi * u_i)))
            v_cos = self.ramp_v_start + (self.ramp_v_goal - self.ramp_v_start) * cos_prog

            local_ceil = self.corridor.compute_smooth_spatial_ceiling(sim_x, b_land_cos)
            rem_i = max(0.0, self.corridor.station_x_m - sim_x)
            # Only apply landing ceiling when within 1.15 * dx_brake_now so cruise speed is never clipped early
            if rem_i <= 1.15 * max(80.0, dx_brake_now):
                v_land_env = float(np.sqrt(max(0.0, (4.0 * b_eff_now * rem_i) / np.pi)))
            else:
                v_land_env = local_ceil
            v_next = float(np.clip(min(v_cos, local_ceil, v_land_env), 0.0, local_ceil))

            v_ref[i] = v_next
            sim_x += 0.5 * (sim_v + v_next) * dt
            sim_v = v_next

        return v_ref, x_brake_now, t_land_now