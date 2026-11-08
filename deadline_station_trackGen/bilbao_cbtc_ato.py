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

        # Section 3.2: RMBTP uses 90% of maximum traction/braking capacity for planning
        # Note: RMBTP plans piecewise constant-rate trapezoidal/parabolic ramps (without Cosine C^2 S-curves)
        self.u_trac_plan = 0.90
        self.u_brake_plan = -0.90
        self.a_plan_cap = 0.90 * self.corridor.comfort_accel_max       # 1.08 m/s^2 (90% of comfort_accel_max)
        self.b_plan_cap = 0.90 * self.corridor.comfort_braking_max     # 0.828 m/s^2 (90% of comfort_braking_max)

        # Section 3.3 (Eqs. 28-32, Fig. 11): Cascade PI-Position (Outer) + PI-Speed (Inner) gains
        self.kp_pos = 0.22
        self.ki_pos = 0.015
        self.kp_vel = 0.82
        self.ki_vel = 0.12

        self.int_pos = 0.0
        self.int_vel = 0.0
        self.prev_u_cmd = 0.0

        # Planned temporal arrays from RMBTP
        self.plan_t = np.zeros(1, dtype=float)
        self.plan_x = np.zeros(1, dtype=float)
        self.plan_v = np.zeros(1, dtype=float)
        self.plan_u_ff = np.zeros(1, dtype=float)

        # Disturbance observer state (Sec. 3.4, Eqs. 33-37)
        self.d_obs_planned = 0.0
        self.last_replan_t = -999.0
        self.replan_count = 0

    def sync_physics(self, physics: NonlinearTrainPhysics) -> None:
        self.physics = physics

    def compute_backward_braking_ceiling(self, x_m: float, d_obs: float = 0.0, safety_margin: float = 0.0) -> float:
        """
        Section 3.2 (Fig. 7): Backward simulation from every speed-limit drop and the terminal
        station stop using 90% braking deceleration so ATP limits and station stops are respected.
        """
        v_lim_here = self.corridor.get_speed_limit(x_m)
        # Effective 90% braking deceleration accounting for observed wind/grade assistance
        b_eff = max(0.42, self.b_plan_cap - 0.35 * d_obs - safety_margin)

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

    def _simulate_rmbtp_forward(
        self,
        x0: float,
        v0: float,
        t0: float,
        v_cruise_cap: float,
        d_obs: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Lightweight mesoscopic forward simulation of the single-mass train model under cruise
        ceiling `v_cruise_cap` with 90% traction/braking effort (Sec. 3.2, Eqs. 12-13).
        Uses a 1.5s mesoscopic integration step for fast real-time Regula Falsi evaluation.
        """
        dt_plan = 1.50
        max_steps = int(max(150, (self.corridor.deadline_s - t0 + 160.0) / dt_plan))
        rho_air = 293.15 / max(250.0, 273.15 + getattr(self.corridor, "temp_celsius", 18.0))

        t_arr = [t0]
        x_arr = [x0]
        v_arr = [v0]
        u_arr = [0.0]

        x_c = float(x0)
        v_c = float(v0)
        t_c = float(t0)

        # Effective plant momentum multiplier (from 0.14*(v_curr - v_prev) per 0.5s step -> 1/(1-0.14) = 1.1628)
        mom_gain = 1.0 / (1.0 - 0.14)

        for _ in range(max_steps):
            rem = self.corridor.station_x_m - x_c
            if rem <= 0.20 or (rem <= 2.0 and v_c <= 0.08 and len(t_arr) > 3):
                break

            # Step-adaptive integration near the station stop for exact docking time prediction
            dt_step = 0.50 if rem < 180.0 else dt_plan
            v_bw_ceil = self.compute_backward_braking_ceiling(x_c, d_obs=d_obs, safety_margin=0.03)
            v_target = min(float(v_cruise_cap), v_bw_ceil)

            # Compute nominal resistance at x_c, v_c
            f_drag = self.physics.basic_resistance(v_c) * rho_air
            f_grade = self.physics.grade_resistance(x_c)
            f_curve = self.physics.curve_resistance(x_c)
            a_res = (f_drag + f_grade + f_curve) / self.physics.M_eff - d_obs

            # Desired acceleration bounded by 90% RMBTP caps
            a_des = (v_target - v_c) / dt_step
            a_cmd = float(np.clip(a_des, -self.b_plan_cap, self.a_plan_cap))

            # Convert desired net acceleration into feedforward dimensionless traction u_ff in [-0.90, 0.90]
            a_nom_req = (a_cmd / mom_gain) + a_res
            f_prop_req = a_nom_req * self.physics.M_eff
            u_sign = 1.0 if f_prop_req >= 0.0 else -1.0
            f_max = max(1.0, self.physics.max_propulsion_force(v_c, u_sign))
            u_ff = float(np.clip(f_prop_req / f_max, self.u_brake_plan, self.u_trac_plan))

            # Advance single-mass model
            f_prop = u_ff * self.physics.max_propulsion_force(v_c, u_ff)
            a_net = ((f_prop - f_drag - f_grade - f_curve) / self.physics.M_eff + d_obs) * mom_gain
            v_next = float(max(0.0, v_c + dt_step * a_net))
            x_next = float(min(self.corridor.station_x_m, x_c + 0.5 * (v_c + v_next) * dt_step))
            t_next = t_c + dt_step

            t_arr.append(t_next)
            x_arr.append(x_next)
            v_arr.append(v_next)
            u_arr.append(u_ff)

            x_c, v_c, t_c = x_next, v_next, t_next
            if x_c >= self.corridor.station_x_m - 0.15 and v_c <= 0.12:
                break

        u_arr[0] = u_arr[1] if len(u_arr) > 1 else 0.0
        return (
            np.array(t_arr, dtype=float),
            np.array(x_arr, dtype=float),
            np.array(v_arr, dtype=float),
            np.array(u_arr, dtype=float),
            float(t_arr[-1]),
        )

    def plan_rmbtp_trajectory(
        self,
        x0: float,
        v0: float,
        t0: float,
        d_obs: float = 0.0,
    ) -> None:
        """
        Section 3.2 (Eqs. 12-16, Fig. 6): RMBTP Mesoscopic Planner with TURBO/ECO Regula Falsi.
          1. Runs TURBO mode (V_max = max corridor speed) to find minimum possible arrival time T_turbo.
          2. If T_deadline <= T_turbo, keeps TURBO trajectory.
          3. Otherwise (ECO mode), applies Bilbao-Moreno et al.'s Regula Falsi update (Eqs. 14-16):
               Error_OpTime = T_deadline - T_sim(V_k)
               V_{k+1} = V_k * dt_rem / (dt_rem + Error_OpTime)
             to converge on the ECO cruise speed that hits T_deadline.
        """
        self.d_obs_planned = float(d_obs)
        self.last_replan_t = float(t0)
        self.replan_count += 1

        v_max_corridor = max(z.v_limit_mps for z in self.corridor.speed_zones)
        t_arr, x_arr, v_arr, u_arr, t_end_turbo = self._simulate_rmbtp_forward(
            x0, v0, t0, v_cruise_cap=v_max_corridor, d_obs=d_obs
        )

        err_op_time = self.corridor.deadline_s - t_end_turbo

        # If TURBO is already late or within 0.25s of deadline, keep TURBO setpoint trajectory
        if err_op_time <= 0.25:
            self.plan_t, self.plan_x, self.plan_v, self.plan_u_ff = t_arr, x_arr, v_arr, u_arr
            return

        # ECO Mode (Sec. 3.2, Eqs. 14-16): Regula Falsi iteration on cruise speed ceiling V_eco
        v_k = float(v_max_corridor)
        best_err = abs(err_op_time)
        best_plan = (t_arr, x_arr, v_arr, u_arr)

        v_low, err_low = 3.5, None
        v_high, err_high = v_max_corridor, err_op_time  # err_high > 0 (arrives early)

        for it in range(5):
            if it == 0 or err_low is None:
                # Eq. 15-16 proportional time-ratio update from Bilbao-Moreno et al. (2025)
                dt_sim = max(1.0, (self.corridor.deadline_s - err_op_time) - t0)
                v_next = float(np.clip(v_k * (dt_sim / max(1.0, dt_sim + err_op_time)), 3.5, v_max_corridor))
            else:
                # Secant / Regula Falsi root interpolation between (v_low, err_low < 0) and (v_high, err_high > 0)
                denom = err_high - err_low
                if abs(denom) > 1e-4:
                    v_next = float(np.clip(v_low - err_low * (v_high - v_low) / denom, 3.5, v_max_corridor))
                else:
                    v_next = 0.5 * (v_low + v_high)

            t_c, x_c, v_c, u_c, t_end_c = self._simulate_rmbtp_forward(
                x0, v0, t0, v_cruise_cap=v_next, d_obs=d_obs
            )
            err_c = self.corridor.deadline_s - t_end_c

            if abs(err_c) < best_err:
                best_err = abs(err_c)
                best_plan = (t_c, x_c, v_c, u_c)

            if abs(err_c) <= 0.25:
                break

            if err_c > 0.0:
                v_high, err_high = v_next, err_c
            else:
                v_low, err_low = v_next, err_c
            v_k, err_op_time = v_next, err_c

        self.plan_t, self.plan_x, self.plan_v, self.plan_u_ff = best_plan

    def select_microscopic_action(
        self,
        t_curr: float,
        x_curr: float,
        v_curr: float,
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

        # Interpolate current planned references
        if len(self.plan_t) <= 1:
            self.plan_rmbtp_trajectory(x_curr, v_curr, t_curr, d_obs=d_obs)
            replanned = True

        x_ref = float(np.interp(t_curr, self.plan_t, self.plan_x))
        v_ff = float(np.interp(t_curr, self.plan_t, self.plan_v))
        u_ff = float(np.interp(t_curr, self.plan_t, self.plan_u_ff))

        # Section 3.4 (Eqs. 33-37): Disturbance Observer event-triggered RMBTP replanning
        dist_shift = abs(d_obs - self.d_obs_planned)
        pos_dev = abs(x_ref - x_curr)
        vel_dev = abs(v_ff - v_curr)
        if rem_dist > 220.0 and (
            force_replan
            or ((t_curr - self.last_replan_t) >= 16.0 and (dist_shift > 0.09 or pos_dev > 16.0 or vel_dev > 0.85))
        ):
            self.plan_rmbtp_trajectory(x_curr, v_curr, t_curr, d_obs=d_obs)
            replanned = True
            x_ref = float(np.interp(t_curr, self.plan_t, self.plan_x))
            v_ff = float(np.interp(t_curr, self.plan_t, self.plan_v))
            u_ff = float(np.interp(t_curr, self.plan_t, self.plan_u_ff))

        dt = self.corridor.dt
        # Spatial ATP + station braking envelope with 10% control margin
        v_bw_ceil = self.compute_backward_braking_ceiling(x_curr, d_obs=d_obs, safety_margin=0.0)

        # 1. Outer Position Loop PI_theta (Eq. 29)
        e_pos = x_ref - x_curr
        dv_pos_raw = self.kp_pos * e_pos + self.ki_pos * self.int_pos

        # 2. Braking Switcher (Eq. 32, Fig. 11):
        # During braking (u_ff < -0.06) or when approaching the spatial ATP/braking ceiling,
        # open the positive position-loop correction so being behind schedule (e_pos > 0) CANNOT push
        # v_cmd above the braking curve or ATP limit!
        in_braking_phase = (u_ff < -0.06) or (v_ff + dv_pos_raw >= v_bw_ceil) or (rem_dist < 180.0)
        if in_braking_phase and dv_pos_raw > 0.0:
            dv_pos = 0.0
        else:
            dv_pos = float(np.clip(dv_pos_raw, -2.5, 2.5))

        # 3. Sync Anti-Windup & Speed Saturation (Eqs. 30-31)
        v_cmd_unsat = v_ff + dv_pos
        v_cmd_sat = float(np.clip(v_cmd_unsat, 0.0, v_bw_ceil))
        if not in_braking_phase and abs(v_cmd_unsat - v_cmd_sat) < 1e-3:
            self.int_pos = float(np.clip(self.int_pos + e_pos * dt, -20.0, 20.0))
        elif in_braking_phase:
            self.int_pos *= 0.85

        # Spatial station docking guidance inside final 160m so the train docks accurately at x = station_x_m
        if rem_dist < 160.0:
            b_dock = max(0.38, (v_curr ** 2) / max(0.25, 2.0 * rem_dist)) if v_curr > 0.3 else self.b_plan_cap
            b_dock_clamped = float(np.clip(b_dock, 0.45, self.corridor.comfort_braking_max))
            v_dock = float(np.sqrt(2.0 * b_dock_clamped * max(0.0, rem_dist)))
            v_cmd_sat = min(v_bw_ceil, v_dock)
            f_drag = self.physics.basic_resistance(v_curr)
            f_grade = self.physics.grade_resistance(x_curr)
            f_curve = self.physics.curve_resistance(x_curr)
            a_res = (f_drag + f_grade + f_curve) / self.physics.M_eff - d_obs
            a_req = (-b_dock_clamped * 0.86) + a_res
            f_max = max(1.0, self.physics.max_propulsion_force(v_curr, -1.0))
            u_ff = float(np.clip((a_req * self.physics.M_eff) / f_max, -0.95, 0.20))

        # 4. Inner Speed Loop PI_omega + Feedforward Traction u_ff (Eq. 28)
        e_vel = v_cmd_sat - v_curr
        u_raw = u_ff + self.kp_vel * e_vel + self.ki_vel * self.int_vel
        u_clamped = float(np.clip(u_raw, -1.0, 1.0))

        # Standard ATO traction/braking slew rate limit (trapezoidal ramp rate limit)
        du_max = 0.14 if rem_dist > 40.0 else 0.22
        u_cmd = float(np.clip(u_clamped, self.prev_u_cmd - du_max, self.prev_u_cmd + du_max))
        self.prev_u_cmd = u_cmd

        # Inner-loop anti-windup (only integrate when actuator is not saturated)
        if abs(u_raw - u_cmd) < 1e-3:
            self.int_vel = float(np.clip(self.int_vel + e_vel * dt, -2.5, 2.5))

        return u_cmd, v_ff, v_cmd_sat, replanned
