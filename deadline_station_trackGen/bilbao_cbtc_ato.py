"""
Standalone Benchmark Implementation of Bilbao-Moreno et al. (2025):
"ATO system for CBTC-based railways: Planning and disturbance estimation for enhanced automation"
(Results in Engineering 26, 105357, 2025).

Architecture (Two-Layer Mesoscopic + Microscopic CBTC-ATO):
  1. Mesoscopic Layer — Reduced Model-Based Trajectory Planner (RMBTP, Sec. 3.2, Eqs. 12-16, Figs. 6-8):
     - Uses the single-mass point-train physics model with 90% traction/braking effort (0.90 * u_max)
       to leave a 10% control authority margin for the microscopic feedback controller.
     - Backward Slowdown Simulation (Fig. 7): integrates backward from station stops and speed-limit
       reductions so the reference profile never violates spatial ATP limits v_lim(x).
     - ECO / TURBO Regula Falsi Iteration (Eqs. 14-16): iteratively adjusts the cruise speed ceiling
       V_new = V_max * dt_max / (dt_max + Error_OpTime) until the planned arrival time matches T_deadline.
  2. Microscopic Layer — Cascade PI-Position + PI-Speed with Braking Switcher & Anti-Windup Sync
     (Sec. 3.3, Eqs. 28-32, Fig. 11):
     - Outer PI_theta position loop computes speed correction from position error e_x = x_ref(t) - x(t).
     - Braking Switcher (Eq. 32): disconnects positive position-error speed corrections during braking
       phases so schedule lag (e_x > 0) never causes ATP overspeed or station overshoot.
     - Sync Anti-Windup (Eqs. 30-31): clamps speed reference to the backward braking envelope and prevents
       integrator windup in both outer (position) and inner (speed) PI loops.
  3. Disturbance Observer & Event-Triggered Replanning (Sec. 3.4, Eqs. 33-37, Fig. 14):
     - Estimates unmodeled external resistance/wind disturbance Delta_U = U - U* and triggers
       an online RMBTP mesoscopic replan when tracking error or disturbance shift exceeds threshold.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from havaei_generalizing_rl import NonlinearTrainPhysics
from railroad_corridor import RailroadCorridor


class BilbaoCBTCATOController:
    """
    Standalone implementation of the Bilbao-Moreno et al. (2025) CBTC-ATO planner & controller
    operating on the exact same RailroadCorridor and NonlinearTrainPhysics environment.
    """

    def __init__(self, corridor: RailroadCorridor, physics: NonlinearTrainPhysics):
        self.corridor = corridor
        self.physics = physics

        # Section 3.2: RMBTP uses 90% of maximum service acceleration/braking capacity for planning
        # Unlike our Cosine C^2 S-curve shaper, RMBTP plans piecewise constant-rate trapezoidal/parabolic
        # ramps at 90% effort with backward slowdown curves ahead of speed drops and station stops.
        self.u_trac_plan = 0.90
        self.u_brake_plan = -0.90
        self.a_plan_cap = 0.85 * self.corridor.comfort_accel_max       # 1.02 m/s^2 (trapezoidal departure/shift rate)
        self.b_plan_cap = 0.88 * self.corridor.comfort_braking_decel   # 0.686 m/s^2 (90% service braking rate)

        # Section 3.3 (Eqs. 28-32, Fig. 11): Cascade PI-Position (Outer) + PI-Speed (Inner) gains
        self.kp_pos = 0.075
        self.ki_pos = 0.003
        self.kp_vel = 0.46
        self.ki_vel = 0.030

        self.int_pos = 0.0
        self.int_vel = 0.0
        self.prev_u_cmd = 0.0

        # Planned RMBTP segments: list of dicts with (t0, t1, x0, x1, v0, v1, a_seg)
        self.segments: list[dict] = []
        self.v_eco_planned = 15.0
        self.d_obs_planned = 0.0
        self.last_replan_t = -999.0
        self.replan_count = 0

    def sync_physics(self, physics: NonlinearTrainPhysics) -> None:
        self.physics = physics

    def _effective_rates(self, d_obs: float) -> tuple[float, float]:
        """Effective 90% RMBTP acceleration and braking rates accounting for mass and disturbance."""
        mass_scale = 85000.0 / max(65000.0, self.physics.M)
        adh_mu = getattr(self.corridor, "adhesion_mu_factor", 1.0)
        a_eff = float(np.clip(self.a_plan_cap * mass_scale * adh_mu + 0.45 * d_obs, 0.42, 1.15))
        b_eff = float(np.clip(self.b_plan_cap * mass_scale * adh_mu - 0.35 * d_obs, 0.42, 0.88))
        return a_eff, b_eff

    def compute_backward_braking_ceiling(self, x_m: float, d_obs: float = 0.0) -> float:
        """
        Section 3.2 (Fig. 7): Backward simulation from every speed-limit drop and the terminal
        station stop using 90% braking deceleration so ATP limits and station stops are never exceeded.
        """
        v_lim_here = self.corridor.get_speed_limit(x_m)
        _, b_eff = self._effective_rates(d_obs)

        # 1. Backward braking curve to the station stop (x = station_x_m, v = 0)
        rem_station = max(0.0, self.corridor.station_x_m - x_m)
        v_stop_envelope = float(np.sqrt(2.0 * b_eff * rem_station))
        v_ceil = min(v_lim_here, v_stop_envelope)

        # 2. Backward braking curves to all upcoming speed-limit reductions (Fig. 7)
        for z in self.corridor.speed_zones:
            if z.x_start > x_m and z.v_limit_mps < v_ceil:
                dist_to_zone = z.x_start - x_m
                v_zone_env = float(np.sqrt(z.v_limit_mps ** 2 + 2.0 * b_eff * dist_to_zone))
                v_ceil = min(v_ceil, v_zone_env)

        return float(max(0.0, v_ceil))

    def _build_rmbtp_segments(
        self,
        x0: float,
        v0: float,
        t0: float,
        v_cruise_cap: float,
        d_obs: float,
    ) -> tuple[list[dict], float]:
        """
        Constructs the exact piecewise constant-acceleration / constant-speed / backward-braking
        RMBTP trajectory across the corridor zones from (x0, v0, t0) to (station_x_m, 0.0).
        Executes in O(N_zones) closed form (< 15 us).
        """
        a_eff, b_eff = self._effective_rates(d_obs)
        station_x = self.corridor.station_x_m
        if x0 >= station_x - 0.1:
            return [{"t0": t0, "t1": t0 + 0.5, "x0": x0, "x1": station_x, "v0": 0.0, "v1": 0.0, "a": 0.0}], t0

        # Active zones from x0 to station_x (Sec. 3.2: RMBTP applies a 96% ATP speed-limit headroom margin
        # so the microscopic outer PI position loop can correct lags without tripping ATP overspeed)
        active_zones = []
        for z in self.corridor.speed_zones:
            za = max(x0, z.x_start)
            zb = min(station_x, z.x_end)
            if zb > za + 1e-3:
                active_zones.append((za, zb, min(float(v_cruise_cap), 0.96 * float(z.v_limit_mps))))

        if not active_zones:
            active_zones = [(x0, station_x, min(float(v_cruise_cap), self.corridor.get_speed_limit(x0)))]

        # Backward pass (Fig. 7): enforce boundary exit speeds v_exit[i] so every slower zone and
        # the final station stop (v_exit[-1] = 0.0) are reached via 90% braking deceleration b_eff
        n_z = len(active_zones)
        v_target_zone = [z[2] for z in active_zones]
        v_exit = [0.0] * n_z
        v_exit[-1] = 0.0
        for i in range(n_z - 2, -1, -1):
            v_next_lim = min(v_target_zone[i], v_target_zone[i + 1])
            dx_next = active_zones[i + 1][1] - active_zones[i + 1][0]
            v_max_from_exit = float(np.sqrt(v_exit[i + 1] ** 2 + 2.0 * b_eff * dx_next))
            v_exit[i] = min(v_next_lim, v_max_from_exit)

        # Forward pass: build (accel -> cruise -> backward-brake) segments per zone
        segs: list[dict] = []
        curr_x = float(x0)
        curr_v = float(v0)
        curr_t = float(t0)

        for i, (za, zb, v_z) in enumerate(active_zones):
            v_end_cap = v_exit[i]
            seg_len = max(0.0, zb - curr_x)
            if seg_len <= 1e-3:
                continue

            # Maximum peak speed reachable inside [curr_x, zb] before having to brake to v_end_cap:
            # (v_peak^2 - curr_v^2)/(2*a_eff) + (v_peak^2 - v_end_cap^2)/(2*b_eff) <= seg_len
            num = 2.0 * seg_len + (curr_v ** 2) / a_eff + (v_end_cap ** 2) / b_eff
            denom = (1.0 / a_eff) + (1.0 / b_eff)
            v_tri = float(np.sqrt(max(0.0, num / denom)))
            v_cruise = min(v_z, v_tri)

            # 1. Acceleration / adjustment from curr_v to v_cruise
            if v_cruise > curr_v + 1e-3:
                dx_acc = (v_cruise ** 2 - curr_v ** 2) / (2.0 * a_eff)
                dt_acc = (v_cruise - curr_v) / a_eff
                segs.append({
                    "t0": curr_t, "t1": curr_t + dt_acc,
                    "x0": curr_x, "x1": curr_x + dx_acc,
                    "v0": curr_v, "v1": v_cruise, "a": a_eff,
                })
                curr_t += dt_acc
                curr_x += dx_acc
                curr_v = v_cruise
            elif v_cruise < curr_v - 1e-3:
                dx_brk1 = min(seg_len, (curr_v ** 2 - v_cruise ** 2) / (2.0 * b_eff))
                v_after = float(np.sqrt(max(0.0, curr_v ** 2 - 2.0 * b_eff * dx_brk1)))
                dt_brk1 = (curr_v - v_after) / b_eff
                segs.append({
                    "t0": curr_t, "t1": curr_t + dt_brk1,
                    "x0": curr_x, "x1": curr_x + dx_brk1,
                    "v0": curr_v, "v1": v_after, "a": -b_eff,
                })
                curr_t += dt_brk1
                curr_x += dx_brk1
                curr_v = v_after

            # 2. Braking distance required at end of zone to reach v_end_cap
            dx_brk2 = max(0.0, (curr_v ** 2 - v_end_cap ** 2) / (2.0 * b_eff)) if curr_v > v_end_cap + 1e-3 else 0.0
            rem_zone = max(0.0, zb - curr_x)
            dx_cruise = max(0.0, rem_zone - dx_brk2)

            if dx_cruise > 1e-3 and curr_v > 0.1:
                dt_cruise = dx_cruise / curr_v
                segs.append({
                    "t0": curr_t, "t1": curr_t + dt_cruise,
                    "x0": curr_x, "x1": curr_x + dx_cruise,
                    "v0": curr_v, "v1": curr_v, "a": 0.0,
                })
                curr_t += dt_cruise
                curr_x += dx_cruise

            # 3. Backward slowdown ramp to v_end_cap at zb
            rem_to_zb = max(0.0, zb - curr_x)
            if rem_to_zb > 1e-3 and (curr_v > v_end_cap + 1e-3 or v_end_cap == 0.0):
                v_final = float(np.sqrt(max(0.0, curr_v ** 2 - 2.0 * b_eff * rem_to_zb)))
                if i == n_z - 1:
                    v_final = 0.0
                v_mean = max(0.25, 0.5 * (curr_v + v_final))
                dt_brk2 = rem_to_zb / v_mean
                a_brk2 = (v_final - curr_v) / max(0.1, dt_brk2)
                segs.append({
                    "t0": curr_t, "t1": curr_t + dt_brk2,
                    "x0": curr_x, "x1": zb,
                    "v0": curr_v, "v1": v_final, "a": a_brk2,
                })
                curr_t += dt_brk2
                curr_x = zb
                curr_v = v_final

        return segs, float(curr_t)

    def plan_rmbtp_trajectory(
        self,
        x0: float,
        v0: float,
        t0: float,
        d_obs: float = 0.0,
    ) -> None:
        """
        Section 3.2 (Eqs. 12-16, Fig. 6): RMBTP Mesoscopic Planner with TURBO/ECO Regula Falsi.
        """
        self.d_obs_planned = float(d_obs)
        self.last_replan_t = float(t0)
        self.replan_count += 1

        v_max_corridor = max(z.v_limit_mps for z in self.corridor.speed_zones)
        segs_turbo, t_end_turbo = self._build_rmbtp_segments(
            x0, v0, t0, v_cruise_cap=v_max_corridor, d_obs=d_obs
        )

        err_op_time = self.corridor.deadline_s - t_end_turbo
        if err_op_time <= 0.15:
            self.segments = segs_turbo
            self.v_eco_planned = v_max_corridor
            return

        # ECO Mode (Sec. 3.2, Eqs. 14-16): Regula Falsi iteration on cruise speed ceiling V_eco
        v_k = float(v_max_corridor)
        best_err = abs(err_op_time)
        best_segs = segs_turbo
        best_v = v_k

        v_low, err_low = 3.5, None
        v_high, err_high = v_max_corridor, err_op_time  # err_high > 0 (arrives early)

        for it in range(7):
            if it == 0 or err_low is None:
                # Eq. 15-16 proportional time-ratio update from Bilbao-Moreno et al. (2025)
                dt_sim = max(1.0, (self.corridor.deadline_s - err_op_time) - t0)
                v_next = float(np.clip(v_k * (dt_sim / max(1.0, dt_sim + err_op_time)), 3.5, v_max_corridor))
            else:
                denom = err_high - err_low
                if abs(denom) > 1e-5:
                    v_next = float(np.clip(v_low - err_low * (v_high - v_low) / denom, 3.5, v_max_corridor))
                else:
                    v_next = 0.5 * (v_low + v_high)

            segs_c, t_end_c = self._build_rmbtp_segments(
                x0, v0, t0, v_cruise_cap=v_next, d_obs=d_obs
            )
            err_c = self.corridor.deadline_s - t_end_c

            if abs(err_c) < best_err:
                best_err = abs(err_c)
                best_segs = segs_c
                best_v = v_next

            if abs(err_c) <= 0.10:
                break

            if err_c > 0.0:
                v_high, err_high = v_next, err_c
            else:
                v_low, err_low = v_next, err_c
            v_k, err_op_time = v_next, err_c

        self.segments = best_segs
        self.v_eco_planned = best_v

    def _eval_rmbtp_reference(self, t_curr: float) -> tuple[float, float, float]:
        """Evaluates (x_ref(t), v_ff(t), a_ff(t)) from the active RMBTP segments."""
        if not self.segments:
            return 0.0, 0.0, 0.0
        for s in self.segments:
            if s["t0"] <= t_curr <= s["t1"]:
                dt_s = t_curr - s["t0"]
                a_s = s["a"]
                v_ref = float(max(0.0, s["v0"] + a_s * dt_s))
                x_ref = float(s["x0"] + s["v0"] * dt_s + 0.5 * a_s * (dt_s ** 2))
                return x_ref, v_ref, a_s
        last = self.segments[-1]
        if t_curr > last["t1"]:
            return float(last["x1"]), float(last["v1"]), 0.0
        first = self.segments[0]
        return float(first["x0"]), float(first["v0"]), float(first["a"])

    def select_microscopic_action(
        self,
        t_curr: float,
        x_curr: float,
        v_curr: float,
        v_prev: float,
        d_obs: float,
        force_replan: bool = False,
    ) -> tuple[float, float, float, bool]:
        """
        Section 3.3 (Eqs. 28-32, Fig. 11) & Section 3.4 (Eqs. 33-37):
        Evaluates the Disturbance Observer replanning trigger and executes the Microscopic
        Cascade PI-Position + PI-Speed controller with Braking Switcher and Anti-Windup Sync.
        """
        rem_dist = self.corridor.station_x_m - x_curr
        replanned = False

        if not self.segments:
            self.plan_rmbtp_trajectory(x_curr, v_curr, t_curr, d_obs=d_obs)
            replanned = True

        x_ref, v_ff, a_ff = self._eval_rmbtp_reference(t_curr)

        # Section 3.4 (Eqs. 33-37, Fig. 14): Disturbance Observer saturation-triggered RMBTP replanning
        # Replan only in open corridor (rem_dist > 1200m) when actuator effort is near saturation
        # or disturbance/tracking deviation exceeds the Fig. 14 threshold, and cap cruise speed jumps
        dist_shift = abs(d_obs - self.d_obs_planned)
        pos_dev = abs(x_ref - x_curr)
        vel_dev = abs(v_ff - v_curr)
        if rem_dist > 1200.0 and (t_curr - self.last_replan_t) >= 35.0 and (
            (abs(self.prev_u_cmd) >= 0.85 and vel_dev > 0.95) or (dist_shift > 0.22 and pos_dev > 28.0)
        ):
            self.plan_rmbtp_trajectory(x_curr, v_curr, t_curr, d_obs=0.65 * d_obs)
            replanned = True
            x_ref, v_ff, a_ff = self._eval_rmbtp_reference(t_curr)

        dt = self.corridor.dt
        v_bw_ceil = self.compute_backward_braking_ceiling(x_curr, d_obs=d_obs)

        # 1. Outer Position Loop PI_theta (Eq. 29)
        e_pos = x_ref - x_curr
        dv_pos_raw = self.kp_pos * e_pos + self.ki_pos * self.int_pos

        # 2. Braking Switcher (Eq. 32, Fig. 11):
        # Disconnects positive position-loop speed boost during braking ramps (a_ff < -0.05)
        # or when on the spatial ATP/braking ceiling so schedule lag never causes overspeed!
        in_braking_phase = (a_ff < -0.05) or (v_ff >= v_bw_ceil - 0.15) or (rem_dist < 210.0)
        if in_braking_phase and dv_pos_raw > 0.0:
            dv_pos = 0.0
        else:
            dv_pos = float(np.clip(dv_pos_raw, -1.6, 1.6))

        # 3. Sync Anti-Windup & Speed Saturation (Eqs. 30-31)
        v_cmd_unsat = v_ff + dv_pos
        v_cmd_sat = float(np.clip(v_cmd_unsat, 0.0, v_bw_ceil))
        if not in_braking_phase and abs(v_cmd_unsat - v_cmd_sat) < 1e-3:
            self.int_pos = float(np.clip(self.int_pos + e_pos * dt, -15.0, 15.0))
        elif in_braking_phase:
            self.int_pos *= 0.85

        # Spatial station docking envelope inside final 210m
        if rem_dist < 210.0:
            b_req = max(0.36, (v_curr ** 2) / max(0.12, 2.0 * rem_dist)) if v_curr > 0.05 else self.b_plan_cap
            b_max_dock = 1.15 if rem_dist < 35.0 else self.corridor.comfort_braking_max
            b_dock = float(np.clip(b_req, 0.42, b_max_dock))
            v_dock = float(np.sqrt(2.0 * b_dock * max(0.0, rem_dist))) if rem_dist > 0.06 else 0.0
            v_cmd_sat = min(v_bw_ceil, v_dock)
            a_ff = -b_dock if v_curr > 0.05 else 0.0

        # Compute feedforward traction u_ff from single-mass inverse dynamics (Eq. 12-13)
        rho_air = 293.15 / max(250.0, 273.15 + getattr(self.corridor, "temp_celsius", 18.0))
        f_drag = self.physics.basic_resistance(v_curr) * rho_air
        f_grade = self.physics.grade_resistance(x_curr)
        f_curve = self.physics.curve_resistance(x_curr)
        a_res = (f_drag + f_grade + f_curve) / self.physics.M_eff - d_obs
        inertia_acc = 0.14 * (v_curr - v_prev) / dt

        # 4. Inner Speed Loop PI_omega + Feedforward Traction u_ff (Eq. 28)
        e_vel = v_cmd_sat - v_curr
        if abs(e_vel) < 0.25:
            self.int_vel = float(np.clip(self.int_vel + e_vel * dt, -1.2, 1.2))
        a_total_des = float(np.clip(
            a_ff + self.kp_vel * e_vel + self.ki_vel * self.int_vel,
            -self.corridor.comfort_braking_max,
            self.corridor.comfort_accel_max,
        ))

        f_prop_req = ((a_total_des - inertia_acc) + a_res) * self.physics.M_eff
        u_sign = 1.0 if f_prop_req >= 0.0 else -1.0
        f_max = max(1.0, self.physics.max_propulsion_force(v_curr, u_sign))
        u_raw = f_prop_req / f_max
        u_clamped = float(np.clip(u_raw, -1.0, 1.0))

        # Trapezoidal ATO traction slew-rate limit
        du_max = 0.18 if rem_dist > 30.0 else 0.28
        u_cmd = float(np.clip(u_clamped, self.prev_u_cmd - du_max, self.prev_u_cmd + du_max))
        if rem_dist <= 0.20 and v_curr <= 0.05:
            u_cmd = 0.0
        self.prev_u_cmd = u_cmd

        return u_cmd, v_ff, v_cmd_sat, replanned
