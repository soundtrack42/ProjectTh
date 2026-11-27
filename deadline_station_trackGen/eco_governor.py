"""Eco governor: an external, configurable operator between the pilot and the shield.

The governor is not part of the environment or of the pilot. The environment calls

    v_cmd, a_start, b_land, v_feas, capped = governor.apply(env, v_cmd, a_start, b_land, v_feas)

once per macro step, after the pilot and before the shield. Any object with `reset()`, `apply(...)` and the
counters `macro_steps` / `capped_steps` can be plugged in (``env.eco = MyGovernor()``), so a different vehicle or
operator policy only needs its own `EcoConfig` (e.g. from a JSON file) or its own governor class.

The governor never brakes and never acts as a safety layer: the shield still checks every command it outputs.
It only queries the environment's travel-time services (`compute_feasible_deadline_pace`,
`estimate_remaining_time_at_speed`, `compute_on_track_slack_s`) and its state (speed, position, schedule).

Two rules, both optional:
  1) Energy limits (every macro step): cap the departure acceleration rate (drivetrain ramp losses), optionally
     the planned landing deceleration, and keep a cruise command above the field-weakening knee of the motors
     only as far as the deadline needs it (v_feas + pace_reserve).
  2) Schedule bound (only when the train is ahead of schedule): hold the speed instead of accelerating, evict
     acceleration that would have to be braked away before a lower speed zone or the station, and, with enough
     slack, prefer gentler rates (slack levers).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields


@dataclass
class EcoConfig:
    # --- Vehicle / drivetrain (from the traction efficiency map of the rolling stock) ---
    v_knee: float = 13.4               # field-weakening onset of the motors (m/s)
    a_ramp: float = 0.78               # departure rate above which ramp (slew) losses grow (m/s^2)
    b_regen: float = 0.70              # landing deceleration above which regenerative braking blends out (m/s^2)
    k_fw: float = 0.010                # efficiency loss per (m/s)^2 above v_knee
    k_ramp: float = 0.18               # efficiency loss per m/s^2 above a_ramp
    k_regen: float = 1.6               # regeneration blend loss per m/s^2 above b_regen
    # --- Rule 1: energy limits (None = lever off) ---
    a_cap: float | None = None         # departure / shift-up acceleration rate cap (m/s^2)
    b_cap: float | None = None         # planned landing deceleration cap (m/s^2)
    fw_margin: float | None = None     # cruise above v_knee + fw_margin only if the deadline needs it (m/s)
    pace_reserve: float = 1.0          # ... i.e. down to max(v_knee + fw_margin, v_feas + pace_reserve) (m/s)
    # --- Rule 2: schedule bound ---
    schedule_bound: bool = False
    t_reserve: float = 4.0             # approach schedule reserve kept by the bound (s)
    slack_a_cap: float | None = None   # with >= t_reserve slack: gentler departure rate (m/s^2)
    slack_b_cap: float | None = None   # with >= t_reserve slack: gentler planned landing (m/s^2)
    slack_v_knee: float | None = None  # with >= t_reserve slack: cruise at most this speed (m/s)
    slack_reserve: float | None = None  # reserve for the slack levers (None: t_reserve)
    # --- Score-based mode (replaces rules 1 and 2): no tuned thresholds ---
    # Each macro step the governor compares the pilot's command with lowered versions of it and keeps the one with
    # the lowest predicted LossScore contribution of this leg: w_time * E|arrival error| + w_energy * energy
    # (Wh/t.km), with the LossScore weights. Energy is predicted from the vehicle model (resistance, drivetrain
    # efficiency with copper / ramp / field-weakening losses, regeneration), arrival from the planner's travel-time
    # estimate with an arrival-error distribution N(mu, sigma) calibrated on the tuning journeys.
    optimize: bool = False
    w_time: float = 1.0                # LossScore weight of |schedule error| (per s)
    w_energy: float = 0.025            # LossScore weight of specific energy (per Wh/t.km)
    eta_max: float = 0.92              # best drivetrain efficiency
    k_cu: float = 0.16                 # copper loss per (traction effort)^2
    k_slew: float = 0.24               # slew loss per (m/s^2)^2 above 0.52 m/s^2
    regen_eff: float = 0.46            # share of electric braking energy returned
    err_mu: float = 0.0                # calibrated mean arrival error per second of remaining time (s/s)
    err_b0: float = 0.0                # calibrated constant bias of the travel-time estimate (s)
    err_sig0: float = 0.5              # calibrated arrival error sd at zero remaining time (s)
    err_sig1: float = 0.0              # calibrated growth of the sd per second of remaining time (s/s)
    dv_step: float = 0.25              # candidate grid of the cruise speed (m/s) ...
    n_v: int = 8                       # ... down to n_v steps below the pilot's command
    a_steps: tuple = (0.0, 0.1, 0.2, 0.3)  # departure-rate reductions tried (m/s^2)
    # --- Adaptive mode (replaces the tuned numbers of rules 1 and 2; updated online during the journey) ---
    # The only fixed numbers are the vehicle's v_knee and a_ramp. The required pace is the lowest hold speed that
    # still arrives on time under the planner's travel-time estimate corrected by its own errors, learned online:
    #  - drift rate r: every macro step the estimate for the previous command is recomputed from the new state;
    #    the change of the predicted arrival per second driven is the estimate's error rate (running mean);
    #  - landing bias c: actual arrival minus the arrival predicted at landing entry, mean over completed legs.
    # Priors (from the calibration on the tuning journeys) count as prior_s seconds of driving and one leg.
    adaptive: bool = False
    prior_r: float = 0.012             # prior drift rate (s per s), calibration on the tuning journeys
    prior_c: float = 2.2               # prior landing bias (s), calibration on the tuning journeys
    prior_s: float = 100.0             # weight of prior_r, in seconds of driving
    # --- Physical mode (formulas from the train's physical data instead of tuned numbers) ---
    #  * eco departure rate a_eco: minimises ramp energy E_k / eta(a) plus the value of the ramp time lambda * v/(2a),
    #    where lambda = v^2 d(R/eta)/dv is the cruise energy needed per second recovered (Davis resistance R, efficiency
    #    map eta, identified mass, observed wind); applied only while the arrival keeps the margin below;
    #  * required pace v_req: lowest hold speed whose arrival, corrected by the planner's measured bias
    #    b(T) = err_b0 + err_mu T and spread sigma(T) = err_sig0 + err_sig1 T, is at least z * sigma(T) early, with the
    #    shield's 99% quantile z; cruise above v_knee is cut down to v_req, and with the schedule rule the train holds
    #    its speed once it is at v_req.
    physical: bool = False
    z: float = 2.576
    # --- Optional safety certification (off: the governor is an energy filter, the shield handles safety) ---
    # If on, the governor's command is executed only if, on every safety property the shield checks (speed,
    # acceleration, jerk, deadline), its margin is no worse than that of the pilot's command (`_no_worse` rule).
    certify: bool = False

    @property
    def has_energy_limits(self) -> bool:
        return self.a_cap is not None or self.b_cap is not None or self.fw_margin is not None

    @property
    def has_slack_levers(self) -> bool:
        return self.slack_a_cap is not None or self.slack_b_cap is not None or self.slack_v_knee is not None

    @classmethod
    def from_dict(cls, d: dict) -> "EcoConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown EcoConfig keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "EcoConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return asdict(self)


# Defaults of the benchmarked pipelines, selected on the tuning journeys (seeds 5000+29k):
#  - energy limits of the cosine pilots (methods 2-5): sweep a_cap 0.80/0.85, fw_margin 0.3/0.6,
#    pace_reserve 0.5/1.0/1.5; the non-cosine pilot (6, 7) arrives late with them, so it has none;
#  - schedule bound of UCB-DMPS: cosine (method 5) sweep t_reserve 3..5 s, slack_a_cap 0.65..0.90 -> 5 s, 0.70;
#    non-cosine (method 7): t_reserve 4 s, no slack lever improved it.
ENERGY_LIMITS_COSINE = dict(a_cap=0.85, fw_margin=0.3, pace_reserve=1.0)
SCHEDULE_BOUND_COSINE = dict(schedule_bound=True, t_reserve=5.0, slack_a_cap=0.70)
SCHEDULE_BOUND_FREE = dict(schedule_bound=True, t_reserve=4.0)


# Physical mode (default of the proposed pilots 2-7): only vehicle data, the planner's travel-time error measured
# once by constant-command driving on the tuning journeys (bias 2.22 s + 1.2% of the remaining time, spread
# 0.78 s + 0.55%), and the shield's 99% quantile.
PHYSICAL = dict(physical=True, err_b0=2.222, err_mu=0.01196, err_sig0=0.775, err_sig1=0.0055)


def physical_eco_config(schedule_bound: bool) -> EcoConfig:
    return EcoConfig(**PHYSICAL, schedule_bound=bool(schedule_bound))


def default_eco_config(cosine: bool, schedule_bound: bool, energy_limits: bool) -> EcoConfig | None:
    d: dict = {}
    if energy_limits and cosine:
        d.update(ENERGY_LIMITS_COSINE)
    if schedule_bound:
        d.update(SCHEDULE_BOUND_COSINE if cosine else SCHEDULE_BOUND_FREE)
    return EcoConfig(**d) if d else None


class EcoGovernor:
    def __init__(self, cfg: EcoConfig):
        self.cfg = cfg
        # Journey-level online statistics (kept across legs; the governor object lives for a whole journey).
        self.drift_sum = 0.0
        self.drift_time = 0.0
        self.land_bias = []
        self.margin_s = 0.0       # arrival margin (s), integral of the observed schedule errors of this journey
        self.reset()

    def drift_rate(self) -> float:
        c = self.cfg
        return (c.prior_r * c.prior_s + self.drift_sum) / (c.prior_s + self.drift_time)

    def landing_bias(self) -> float:
        return (self.cfg.prior_c + sum(self.land_bias)) / (1.0 + len(self.land_bias))

    def end_leg(self, env) -> None:
        """Called after a leg: learn the landing bias from the actual arrival, and feed the leg's own schedule
        error back into the arrival margin (late -> larger margin, early -> smaller)."""
        if self.cfg.adaptive and self._P_land is not None:
            self.land_bias.append(float(env.t_curr) - self._P_land)
        if self.cfg.adaptive:
            self.margin_s = max(0.0, self.margin_s + (float(env.t_curr) - float(env.corridor.deadline_s)))

    def _adaptive(self, env, v_cmd, a_start, b_land, v_feas):
        c = self.cfg
        cor = env.corridor
        t = float(env.t_curr)
        if env.shaper.in_landing:
            if self._P_land is None and self._last_P is not None:
                self._P_land = self._last_P
            self._prev = None
            return v_cmd, a_start, b_land, v_feas, False
        if self._prev is not None:
            tp, Pp, vp, ap, bp, v0 = self._prev
            # Learn only from steady cruising (speed at both ends of the macro step on the held command): the
            # departure ramp and zone transitions have their own, phase-specific errors.
            if abs(v0 - vp) < 0.3 and abs(float(env.v_meas) - vp) < 0.3:
                P_same = t + env.estimate_remaining_time_at_speed(vp, ap, bp)
                self.drift_sum += P_same - Pp
                self.drift_time += t - tp
        r, cb = self.drift_rate(), self.landing_bias()

        def arrival(v, a):
            return t + env.estimate_remaining_time_at_speed(v, a, b_land) * (1.0 + r) + cb + self.margin_s

        # Gentle departure: rate down to a_ramp only if the corrected arrival is still on time.
        if a_start > c.a_ramp and v_cmd > float(env.v_meas) + 0.3 and arrival(v_cmd, c.a_ramp) <= cor.deadline_s:
            a_start = c.a_ramp
            v_feas, _ = env.compute_feasible_deadline_pace(a_start, b_land)
        # Required pace: lowest hold speed whose corrected arrival is on time (bisection; inf if none).
        lo, hi = 1.5, max(1.5, float(v_cmd))
        if arrival(hi, a_start) > cor.deadline_s:
            v_req = math.inf
        else:
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                if arrival(mid, a_start) <= cor.deadline_s:
                    hi = mid
                else:
                    lo = mid
            v_req = hi
        capped = False
        v_new = v_cmd
        if v_new > c.v_knee:                  # field weakening only as far as the deadline needs it
            v_new = max(c.v_knee, min(v_new, v_req))
        if c.schedule_bound:                  # ahead of schedule: hold the speed, never brake
            self.macro_steps += 1
            v_hold = max(v_req, float(env.v_meas))
            if v_new > v_hold + 0.02:
                v_new, capped = v_hold, True
                self.capped_steps += 1
        P = t + env.estimate_remaining_time_at_speed(v_new, a_start, b_land)
        self._prev = (t, P, v_new, a_start, b_land, float(env.v_meas))
        self._last_P = P
        return v_new, a_start, b_land, v_feas, capped

    def reset(self) -> None:
        """Called by the environment at every leg reset."""
        self.macro_steps = 0      # macro steps checked by the schedule bound
        self.capped_steps = 0     # macro steps where the schedule bound lowered the command (Eco%)
        self.slack_steps = 0      # macro steps where a slack lever was applied
        self.certified_steps = 0  # macro steps where the governor changed the command and the change was certified
        self.rejected_steps = 0   # macro steps where the change was rejected by the certification
        self.inefficient_steps = 0  # macro steps where the change was rejected by the efficiency rule
        self.opt_steps = 0        # macro steps where the score-based mode lowered the command
        self._prev = None         # adaptive mode: (t, raw predicted arrival, v, a, b) of the previous macro step
        self._last_P = None
        self._P_land = None
        self._a_eco = None        # physical mode: eco departure rate of this leg (identified mass, observed wind)
        self.v_ub: float | None = None
        self._cache_k = -1
        self._cached = (15.0, False)
        self._anchor_v: float | None = None   # latched "no further acceleration" speed (None = released)

    # ------------------------------------------------------------------ physical mode
    def eco_departure_rate(self, env, v: float) -> float:
        """Energy-optimal departure rate for a ramp to v (see EcoConfig.physical)."""
        c = self.cfg
        ph = env.physics
        M, f_max = ph.M_eff, ph.M * ph.max_traction_accel
        f_w = -M * float(env.d_hat)
        def g(vv):
            f = ph.basic_resistance(vv) + f_w
            return f / self._eta(0.0, vv, max(0.0, f) / f_max)
        h = 1e-3
        lam = v * v * (g(v + h) - g(v - h)) / (2.0 * h)
        r_half = ph.basic_resistance(0.5 * v) + f_w

        def J(a):
            u = min(1.0, max(0.0, M * a + r_half) / f_max)
            return 0.5 * M * v * v / self._eta(a, 0.5 * v, u) + lam * v / (2.0 * a)

        # Coarse-to-fine search on [0.50, 1.20] m/s^2 (J is unimodal in a): 0.05 grid, then 0.01 around the best.
        coarse = min((0.50 + 0.05 * i for i in range(15)), key=J)
        return min((min(1.20, max(0.50, coarse + 0.01 * k)) for k in range(-4, 5)), key=J)

    def prepare(self, env) -> None:
        """At the station before departure: eco departure rate of the leg from the ticket-based mass prior and the
        station wind (physical mode), so that it is not computed inside the control loop."""
        if self.cfg.physical:
            v_top = max(z.v_limit_mps for z in env.corridor.speed_zones)
            self._a_eco = self.eco_departure_rate(env, min(self.cfg.v_knee, v_top))

    def _physical(self, env, v_cmd, a_start, b_land, v_feas):
        c = self.cfg
        cor = env.corridor
        if env.shaper.in_landing:
            return v_cmd, a_start, b_land, v_feas, False
        t = float(env.t_curr)

        args = {}

        def T_of(v, a):         # = env.estimate_remaining_time_at_speed, with the leg geometry built once per rate
            A = args.get(a)
            if A is None:
                A = args[a] = env._remaining_time_args(a, b_land)
            if A[0] <= 15.0:
                return float(A[0] / max(0.5, A[3] + 0.5))
            return float(env._eval_remaining_time_core(float(v), *A[2:]))

        def late_by(v, a):      # corrected arrival + z * sigma - deadline (<= 0: on time with 99% confidence)
            T = T_of(v, a)
            return t + T * (1.0 + c.err_mu) + c.err_b0 + c.z * (c.err_sig0 + c.err_sig1 * T) - cor.deadline_s

        if v_cmd > float(env.v_meas) + 0.3:
            if self._a_eco is None:
                self._a_eco = self.eco_departure_rate(env, min(c.v_knee, max(v_cmd, 1.0)))
            if a_start > self._a_eco and late_by(v_cmd, a_start) <= 0.0:
                # Gentlest of {a_eco, midpoint, a_pilot} that still arrives on time with 99% confidence.
                a_mid = round(0.5 * (self._a_eco + a_start), 3)
                a_new = a_start
                for a_try in (self._a_eco, a_mid):
                    if late_by(v_cmd, a_try) <= 0.0:
                        a_new = a_try
                        break
                a_start = a_new
                v_feas, _ = env.compute_feasible_deadline_pace(a_start, b_land)
        def required_pace(v_lo):
            """max(v_lo, v_req), v_req = lowest hold speed in [v_lo, v_cmd] that is on time (inf if none): one
            estimate when v_lo is already on time, else regula falsi on the bracket (at most 5 estimates)."""
            f_lo = late_by(v_lo, a_start)
            if f_lo <= 0.0:
                return v_lo
            v_hi = float(v_cmd)
            f_hi = late_by(v_hi, a_start)
            if f_hi > 0.0:
                return math.inf
            for _ in range(3):
                if v_hi - v_lo < 0.05:
                    break
                v_m = v_hi - f_hi * (v_hi - v_lo) / (f_hi - f_lo)
                v_m = min(v_hi - 0.01, max(v_lo + 0.01, v_m))
                f_m = late_by(v_m, a_start)
                if f_m <= 0.0:
                    v_hi, f_hi = v_m, f_m
                else:
                    v_lo, f_lo = v_m, f_m
            return v_hi

        capped = False
        v_new = v_cmd
        v_meas = float(env.v_meas)
        need_fw = v_new > c.v_knee
        need_sb = c.schedule_bound and v_new > v_meas + 0.02
        v_req = required_pace(min(c.v_knee, v_meas) if need_sb else c.v_knee) if (need_fw or need_sb) else 0.0
        if need_fw:                           # field weakening only as far as the deadline needs it
            v_new = min(v_new, max(c.v_knee, v_req))
        if c.schedule_bound:                  # at the required pace: hold the speed, never brake
            self.macro_steps += 1
            if need_sb:
                v_hold = max(v_meas, v_req)
                if v_new > v_hold + 0.02:
                    v_new, capped = v_hold, True
                    self.capped_steps += 1
        return v_new, a_start, b_land, v_feas, capped

    # ------------------------------------------------------------------ score-based mode
    def _eta(self, a: float, v: float, u: float) -> float:
        c = self.cfg
        e = (c.eta_max - c.k_cu * u * u - c.k_slew * max(0.0, a - 0.52) ** 2 - c.k_ramp * max(0.0, a - c.a_ramp)
             - c.k_fw * max(0.0, v - c.v_knee) ** 2)
        return min(c.eta_max, max(0.58, e))

    def predict_energy_kwh(self, env, v_c: float, a_start: float, b_land: float) -> float:
        """Traction energy (kWh) to the station when cruising at min(v_c, zone limit), departing / shifting up at
        a_start and landing at b_land: same geometry as the planner's travel-time estimate (zones, braking spot)."""
        c = self.cfg
        (x_rem, _, x_curr, v_curr, station_x_m, zones, v_last_lim, inv_2b, inv_1b,
         dur_k, min_ramp_s, rate_up, rate_dn) = env._remaining_time_args(a_start, b_land)
        ph = env.physics
        M = ph.M_eff
        f_max = ph.M * ph.max_traction_accel
        f_wind = -M * float(env.d_hat)          # disturbance observer, as a force opposing motion if > 0
        v_throat = min(v_c, v_last_lim)
        x_b = max(x_curr + 1.0, station_x_m - v_throat * v_throat * inv_2b)
        e_j, sim_v = 0.0, float(v_curr)
        for z in zones:
            seg_a, seg_b = max(x_curr, z.x_start), min(x_b, z.x_end)
            if seg_b <= seg_a:
                continue
            v_seg = min(v_c, z.v_limit_mps)
            if v_seg > sim_v + 0.05:            # ramp up: kinetic energy at the ramp efficiency
                v_bar = 0.5 * (sim_v + v_seg)
                f = M * a_start + ph.basic_resistance(v_bar) + f_wind
                e_j += 0.5 * M * (v_seg * v_seg - sim_v * sim_v) / self._eta(a_start, v_bar, min(1.0, f / f_max))
            elif v_seg < sim_v - 0.05:          # ramp down: part of the kinetic energy is regenerated
                e_j -= c.regen_eff * 0.5 * M * (sim_v * sim_v - v_seg * v_seg)
            f_c = ph.basic_resistance(v_seg) + f_wind
            if f_c > 0.0:
                e_j += f_c * (seg_b - seg_a) / self._eta(0.0, v_seg, min(1.0, f_c / f_max))
            sim_v = v_seg
        blend = min(1.0, max(0.35, 1.0 - 1.6 * max(0.0, b_land - 0.70) - 0.8 * max(0.0, b_land - 0.68)))
        e_j -= c.regen_eff * blend * 0.5 * M * v_throat * v_throat
        return e_j / 3.6e6

    def _expected_abs_err(self, mu: float, sig: float) -> float:
        z = mu / sig
        return sig * math.sqrt(2.0 / math.pi) * math.exp(-0.5 * z * z) + mu * math.erf(z / math.sqrt(2.0))

    def leg_cost(self, env, v_c: float, a_start: float, b_land: float) -> float:
        """Predicted LossScore contribution of this leg for a command held to the station."""
        c = self.cfg
        cor = env.corridor
        t_rem_pred = env.estimate_remaining_time_at_speed(v_c, a_start, b_land)
        mu = env.t_curr + t_rem_pred * (1.0 + c.err_mu) + c.err_b0 - cor.deadline_s
        sig = c.err_sig0 + c.err_sig1 * t_rem_pred
        ton_km = (float(env.mass_hat_kg) / 1000.0) * (cor.station_x_m / 1000.0)
        e_spec = self.predict_energy_kwh(env, v_c, a_start, b_land) * 1000.0 / max(1e-3, ton_km)
        return c.w_time * self._expected_abs_err(mu, sig) + c.w_energy * e_spec

    def _optimize(self, env, v_cmd, a_start, b_land, v_feas):
        c = self.cfg
        if env.shaper.in_landing or env.k_elapsed < 1:
            return v_cmd, a_start, b_land, v_feas
        best = (self.leg_cost(env, v_cmd, a_start, b_land), v_cmd, a_start)
        a_list = [a_start - da for da in c.a_steps if a_start - da >= 0.55 - 1e-9]
        for a in a_list:
            for i in range(0 if a < a_start else 1, c.n_v + 1):
                v = v_cmd - i * c.dv_step
                if v < 1.5:
                    break
                j = self.leg_cost(env, v, a, b_land)
                if j < best[0] - 1e-9:
                    best = (j, v, a)
        _, v_new, a_new = best
        if (v_new, a_new) != (v_cmd, a_start):
            if a_new != a_start:
                v_feas, _ = env.compute_feasible_deadline_pace(a_new, b_land)
            self.opt_steps += 1
        return v_new, a_new, b_land, v_feas

    # ------------------------------------------------------------------ rule 1
    def _energy_limits(self, env, v_cmd, a_start, b_land, v_feas):
        c = self.cfg
        if c.a_cap is not None and a_start > c.a_cap:
            a_start = c.a_cap
        if c.b_cap is not None and b_land > c.b_cap:
            b_land = max(0.63, c.b_cap)
        if c.a_cap is not None or c.b_cap is not None:
            v_feas, _ = env.compute_feasible_deadline_pace(a_start, b_land)
        if c.fw_margin is not None:
            v_fw = c.v_knee + c.fw_margin
            if v_cmd > v_fw:
                v_cmd = max(v_fw, min(v_cmd, v_feas + c.pace_reserve))
        return v_cmd, a_start, b_land, v_feas

    # ------------------------------------------------------------------ rule 2
    def upper_bound(self, env, a_start: float = 0.86, b_land: float = 0.74) -> tuple[float, bool]:
        """
        Upper bound on the commanded cruise speed, applied ONLY when the train is going too fast
        (cached once per step). Returns (v_ub, is_binding):
          1) v_track = minimal hold speed such that travelling at min(v_track, cap_i) over the remaining
             zones plus the stopping trajectory meets the deadline, scaled to keep t_reserve, plus a headwind
             margin from the disturbance observer.
          2) On-track test: holding min(v_meas, cap_i) + stop arrives early by `compute_on_track_slack_s`.
             If v_meas >= v_track and slack > t_res + 0.5 s, any further acceleration is useless:
             v_ub = max(v_track + 0.10, v_anchor), v_anchor = speed when the bound engaged (latched).
          3) Acceleration that would have to be braked away (an upcoming lower speed zone within braking
             reach, or the station braking spot) is evicted: v_ub <= max(v_target_after, v_anchor).
        The anchor is latched (no per-step re-anchoring on the measured speed, which produced a staircase of
        shaper ramps) and released with hysteresis (slack < t_res - 0.5 s). Otherwise v_ub = local speed cap.
        """
        if self._cache_k == env.k_elapsed:
            return self._cached

        cor = env.corridor
        x = env.x_curr
        v = max(0.0, float(env.v_meas))
        v_cur_cap = cor.get_speed_limit(x)
        res = (float(v_cur_cap), False)
        if env.k_elapsed < 8 or env.shaper.in_landing or (cor.station_x_m - x) <= 200.0 or v < 1.8:
            self._anchor_v = None
            self._cache_k = env.k_elapsed
            self._cached = res
            return res

        v_feas, _ = env.compute_feasible_deadline_pace(a_start, b_land)
        t_rem = max(cor.dt, cor.deadline_s - env.t_curr)
        t_res = self.cfg.t_reserve
        v_track = v_feas * (t_rem / max(0.5 * t_rem, t_rem - t_res)) + 0.25 * max(0.0, -float(env.d_hat))
        slack = env.compute_on_track_slack_s(a_start, b_land)
        latched = self._anchor_v is not None
        anchor = self._anchor_v if latched else v
        v_ub = v_cur_cap
        on_track = (slack > t_res - 0.5) if latched else (v >= v_track and slack > t_res + 0.5)
        if on_track:
            v_ub = max(v_track + 0.10, anchor)

        if slack > 0.0:
            # Next lower speed zone within braking reach: do not accelerate above current speed.
            k_land = 0.5 * math.pi if env.use_cosine_shaper else 1.66
            b_drop = max(0.38, b_land * (0.65 if env.use_cosine_shaper else 0.60))
            for z in cor.speed_zones:
                if z.x_start > x and z.v_limit_mps < v_cur_cap:
                    dx_brk = 0.5 * k_land * max(0.0, v * v - z.v_limit_mps * z.v_limit_mps) / b_drop
                    if (z.x_start - x) < dx_brk + 8.0 * v:
                        v_ub = min(v_ub, max(z.v_limit_mps + 0.08, anchor))
                    break
            # Station braking spot close ahead: no acceleration beyond what the schedule needs.
            _, b_eff_land = env._get_remaining_time_geom(b_land)
            dx_land = (0.5 * k_land / b_eff_land) * v * v
            if (cor.station_x_m - dx_land - x) < 7.5 * v:
                v_ub = min(v_ub, max(v_track + 0.06, anchor))

        binding = bool(v_ub < v_cur_cap - 0.05)
        if binding and not latched:
            self._anchor_v = v
        elif not binding:
            self._anchor_v = None
        res = (float(v_ub), binding)
        self.v_ub = res[0]
        self._cache_k = env.k_elapsed
        self._cached = res
        return res

    def _slack_levers(self, env, v_cmd, a_start, b_land):
        """Greedy levers (gentler acceleration, gentler planned landing, cruise below the knee), each kept only
        if holding the resulting plan still arrives `reserve` seconds before the deadline."""
        c = self.cfg
        reserve = c.t_reserve if c.slack_reserve is None else c.slack_reserve
        budget = (env.corridor.deadline_s - env.t_curr) - reserve
        if budget <= 0.0 or env.estimate_remaining_time_at_speed(v_cmd, a_start, b_land) > budget:
            return v_cmd, a_start, b_land
        changed = False
        if c.slack_a_cap is not None and a_start > c.slack_a_cap and v_cmd > float(env.v_meas) + 0.3:
            if env.estimate_remaining_time_at_speed(v_cmd, c.slack_a_cap, b_land) <= budget:
                a_start, changed = c.slack_a_cap, True
        if c.slack_b_cap is not None and b_land > c.slack_b_cap:
            if env.estimate_remaining_time_at_speed(v_cmd, a_start, c.slack_b_cap) <= budget:
                b_land, changed = c.slack_b_cap, True
        if c.slack_v_knee is not None and v_cmd > c.slack_v_knee and float(env.v_meas) <= c.slack_v_knee + 0.3:
            if env.estimate_remaining_time_at_speed(c.slack_v_knee, a_start, b_land) <= budget:
                v_cmd, changed = c.slack_v_knee, True
        if changed:
            self.slack_steps += 1
        return v_cmd, a_start, b_land

    # ------------------------------------------------------------------ entry point
    def predicted_loss(self, v_cmd: float, a_start: float, b_land: float) -> float:
        """Drivetrain-loss index of a command (vehicle parameters of the config): field weakening above v_knee,
        ramp losses above a_ramp, lost regeneration above b_regen. Non-decreasing in each argument."""
        c = self.cfg
        return (c.k_fw * max(0.0, v_cmd - c.v_knee) ** 2 + c.k_ramp * max(0.0, a_start - c.a_ramp)
                + c.k_regen * max(0.0, b_land - c.b_regen))

    def apply(self, env, v_cmd: float, a_start: float, b_land: float, v_feas: float):
        """Returns (v_cmd, a_start, b_land, v_feas, schedule_capped).

        Efficiency rule (always on): the governor is a filter that may only lower the command (cruise speed,
        departure rate, landing deceleration) and only if its predicted drivetrain loss does not increase;
        otherwise the pilot's command is passed on unchanged. Safety is left to the shield downstream."""
        pilot = (v_cmd, a_start, b_land, v_feas)
        cap_before = self.capped_steps
        out = self._apply(env, v_cmd, a_start, b_land, v_feas)
        if out[:4] == pilot:
            return out
        lowered = out[0] <= v_cmd + 1e-9 and out[1] <= a_start + 1e-9 and out[2] <= b_land + 1e-9
        if not lowered or self.predicted_loss(*out[:3]) > self.predicted_loss(v_cmd, a_start, b_land) + 1e-12:
            self.inefficient_steps += 1
            self.capped_steps = cap_before
            return pilot[0], pilot[1], pilot[2], pilot[3], False
        sh = getattr(env, "traj_shield", None)
        if not self.cfg.certify or sh is None:
            return out
        sh.sigma_tube_curr = env.sigma_tube
        h_pilot = sh._margins(env, pilot[0], pilot[1], pilot[2], pilot[3])
        h_eco = sh._margins(env, out[0], out[1], out[2], out[3])
        if sh._no_worse(h_eco, h_pilot):
            self.certified_steps += 1
            return out
        self.rejected_steps += 1
        self.capped_steps = cap_before
        return pilot[0], pilot[1], pilot[2], pilot[3], False

    def _apply(self, env, v_cmd: float, a_start: float, b_land: float, v_feas: float):
        c = self.cfg
        if c.physical:
            return self._physical(env, v_cmd, a_start, b_land, v_feas)
        if c.adaptive:
            return self._adaptive(env, v_cmd, a_start, b_land, v_feas)
        if c.optimize:
            v_cmd, a_start, b_land, v_feas = self._optimize(env, v_cmd, a_start, b_land, v_feas)
            return v_cmd, a_start, b_land, v_feas, False
        if c.has_energy_limits:
            v_cmd, a_start, b_land, v_feas = self._energy_limits(env, v_cmd, a_start, b_land, v_feas)
        capped = False
        if c.schedule_bound and not env.shaper.in_landing:
            self.macro_steps += 1
            v_ub, bind = self.upper_bound(env, a_start, b_land)
            if bind and v_cmd > v_ub + 0.02:
                v_cmd = v_ub
                capped = True
                self.capped_steps += 1
            if c.has_slack_levers:
                v_cmd, a_start, b_land = self._slack_levers(env, v_cmd, a_start, b_land)
        return v_cmd, a_start, b_land, v_feas, capped
