"""
Module 1: Multi-Station Long Railroad Corridor (14-18 km, 2-3 Station Stops),
Dynamic Passenger Weight Changes, and Disjoint Spurious Wind Generator (`railroad_corridor.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class SpeedLimitZone:
    x_start: float
    x_end: float
    v_limit_mps: float
    label: str = "Mainline"


@dataclass
class StationStopSpec:
    station_id: int
    name: str
    x_m: float                           # Station coordinate along the multi-km line (m)
    arrival_deadline_s: float            # Target arrival clock time at this station (s)
    dwell_time_s: float                  # Passenger boarding/alighting dwell time (s)
    passengers_boarding: int             # Passengers joining the train
    passengers_alighting: int            # Passengers leaving the train
    total_passengers_after: int          # Net passenger count for the departing leg
    train_mass_kg: float                 # Updated total vehicle mass M (kg) for the leg
    is_infeasible_leg: bool = False      # True if this leg's deadline is an infeasible break-test


@dataclass
class RailroadCorridor:
    """
    Represents either a single leg or a full Multi-Station Corridor (14.5 - 17.5 km, 3 stations)
    with stochastic passenger mass (sold tickets vs actual riders + weight variance) and weather forces.
    """
    route_name: str = "Express Line 1: Suburban -> Central -> Tech Park -> Airport (3 Stops)"
    station_x_m: float = 5200.0          # Current leg target distance (m)
    start_time_s: float = 0.0            # Current leg departure clock time (s)
    deadline_s: float = 420.0            # Current leg target arrival clock time (s)
    dt: float = 0.50                     # MPC & physics integration step (s)
    macro_steps: int = 8                 # RL decision interval during open cruising (4.0s)
    mass_kg: float = 85000.0             # True physical vehicle mass (tare + actual passengers & luggage)
    ticket_estimated_mass_kg: float = 85000.0  # Prior mass estimated from sold tickets (75kg * tickets)
    passengers: int = 170                # Actual passenger count on board (capped at max_capacity=360)
    sold_tickets: int = 170              # Sold ticket count reported at station turnstiles
    max_capacity: int = 360              # Maximum passenger capacity of the 4-car train
    weather_condition: str = "Dry"       # "Dry", "Rain", "Snow", or "ColdWind"
    adhesion_mu_factor: float = 1.00     # Rail friction multiplier (1.0=Dry, 0.82=Rain, 0.68=Snow)
    temp_celsius: float = 18.0           # Ambient temperature C (affects air density rho & drag A)
    station_wind_sensor_mps2: float = 0.0 # Wayside station anemometer reading at departure
    is_infeasible_challenge: bool = False

    # Base physical & comfort bounds at nominal 85,000 kg mass (m/s^2)
    base_max_traction_accel: float = 1.85
    base_max_service_braking: float = 1.65
    comfort_accel_max: float = 1.20      # Non-annoying passenger acceleration ceiling
    comfort_braking_decel: float = 0.78  # Smooth target landing deceleration
    comfort_braking_max: float = 0.92    # Upper bound of non-annoying braking range

    speed_zones: list[SpeedLimitZone] = field(default_factory=lambda: [
        SpeedLimitZone(0.0, 1200.0, 15.8, "Zone 1: Urban Departure"),
        SpeedLimitZone(1200.0, 2600.0, 17.5, "Zone 2: Express Trunk A"),
        SpeedLimitZone(2600.0, 3550.0, 12.8, "Zone 3: Viaduct Curve Restriction"),
        SpeedLimitZone(3550.0, 4600.0, 16.8, "Zone 4: Express Trunk B"),
        SpeedLimitZone(4600.0, 6500.0, 12.5, "Zone 5: Station Throat"),
    ])

    @property
    def max_traction_accel(self) -> float:
        """Physical traction limit scaled by vehicle mass and rail adhesion."""
        mass_scale = 85000.0 / max(65000.0, self.mass_kg)
        return float(self.base_max_traction_accel * mass_scale * self.adhesion_mu_factor)

    @property
    def max_service_braking(self) -> float:
        """Physical service braking limit scaled by vehicle mass and rail adhesion."""
        mass_scale = 85000.0 / max(65000.0, self.mass_kg)
        return float(self.base_max_service_braking * mass_scale * self.adhesion_mu_factor)

    def get_speed_limit(self, x_m: float) -> float:
        for z in self.speed_zones:
            if z.x_start <= x_m <= z.x_end:
                return z.v_limit_mps
        return self.speed_zones[-1].v_limit_mps

    def compute_smooth_spatial_ceiling(self, x_m: float, b_shift_down: float = 0.78) -> float:
        """
        C^1-smooth half-cosine anticipatory ceiling ahead of every slower speed zone.
        Enforces both deceleration comfort and Order-3 jerk comfort scaled by `b_shift_down`
        so gentler braking commands (`b_shift_down ~ 0.68`) trace wider, lower-jerk S-curves
        while greedy braking (`b_shift_down ~ 0.90`) brakes later and steeper.
        """
        v_here = self.get_speed_limit(x_m)
        zones = self.speed_zones
        has_near_drop = False
        for z in zones:
            if z.x_start > x_m and z.v_limit_mps < v_here and (z.x_start - x_m) < 380.0:
                has_near_drop = True
                break
        if not has_near_drop:
            return float(v_here)

        b_raw = b_shift_down / 0.78
        b_ratio = 0.75 if b_raw < 0.75 else (1.25 if b_raw > 1.25 else b_raw)
        bs_raw = b_shift_down * 0.76
        b_safe = 0.42 if bs_raw < 0.42 else (0.74 if bs_raw > 0.74 else bs_raw)
        j_max_zone = 0.22 * (b_ratio ** 1.6)
        min_drop_dur = 14.0 / b_ratio
        pi_val = math.pi

        for z in zones:
            if z.x_start > x_m and z.v_limit_mps < v_here:
                dx_to_zone = z.x_start - x_m
                if dx_to_zone >= 380.0:
                    continue
                dv_drop = v_here - z.v_limit_mps
                v_mean = 0.5 * (v_here + z.v_limit_mps)
                # Duration required by both comfort deceleration and comfort jerk:
                t_acc_req = (pi_val * dv_drop) / (2.0 * b_safe)
                t_jrk_req = pi_val * math.sqrt(dv_drop / (2.0 * j_max_zone))
                t_drop = min_drop_dur if min_drop_dur >= t_acc_req else t_acc_req
                if t_jrk_req > t_drop:
                    t_drop = t_jrk_req
                dx_drop = v_mean * t_drop

                if dx_to_zone < dx_drop and dx_drop > 1.0:
                    u_raw = 1.0 - (dx_to_zone / dx_drop)
                    u_s = 0.0 if u_raw < 0.0 else (1.0 if u_raw > 1.0 else u_raw)
                    v_cos_zone = z.v_limit_mps + dv_drop * 0.5 * (1.0 + math.cos(pi_val * u_s))
                    if v_cos_zone < v_here:
                        v_here = v_cos_zone
        return float(v_here)


def generate_disjoint_wind_events(
    total_duration_s: float = 600.0,
    dt: float = 0.50,
    calm_deadband_mps2: float = 0.34,
    seed: int | None = None,
) -> tuple[np.ndarray, str, list[tuple[float, float, float]]]:
    """
    Generates a continuous scalar wind field w_raw(t) where active gusts are casual,
    isolated events (~15-20% of the trip) separated by long stretches of calm air (0.0 m/s^2).
    """
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(total_duration_s / dt)) + 40
    t_grid = np.arange(n_steps, dtype=np.float64) * dt

    # Generate 2 distinct casual weather fronts per leg: one Headwind (<0) and one Tailwind (>0)
    first_sign = -1.0 if (rng.random() < 0.5) else 1.0
    w_raw = np.zeros(n_steps, dtype=np.float64)
    for f_i in range(2):
        center_s = float(rng.uniform(0.08 + 0.45 * f_i, 0.35 + 0.45 * f_i) * total_duration_s)
        width_s = float(rng.uniform(24.0, 36.0))
        sign = first_sign if (f_i == 0) else -first_sign
        peak = sign * float(rng.uniform(0.62, 0.82))
        w_raw += peak * np.exp(-0.5 * ((t_grid - center_s) / width_s) ** 2)

    # Smooth C^1 deadband thresholding
    wind_profile = np.where(
        np.abs(w_raw) > calm_deadband_mps2,
        np.sign(w_raw) * (np.abs(w_raw) - calm_deadband_mps2) * 1.25,
        0.0,
    )
    wind_profile = np.clip(wind_profile, -0.58, 0.52)

    # 2. Measure the emergent disjoint intervals [t_s, t_e] from the generated trace
    active_mask = np.abs(wind_profile) > 0.02
    event_meta = []
    descriptions = []
    k = 0
    ev_id = 1
    while k < n_steps:
        if active_mask[k]:
            k_start = k
            while k < n_steps and active_mask[k] and (np.sign(wind_profile[k]) == np.sign(wind_profile[k_start])):
                k += 1
            k_end = k
            if (k_end - k_start) * dt >= 8.0:
                t_s, t_e = k_start * dt, k_end * dt
                seg = wind_profile[k_start:k_end]
                peak_val = float(seg[np.argmax(np.abs(seg))])
                event_meta.append((t_s, t_e, peak_val))
                tag = "Headwind" if peak_val < 0 else "Tailwind"
                descriptions.append(f"E{ev_id}:{tag}({peak_val:+.2f}m/s²,[{t_s:.0f}s,{t_e:.0f}s])")
                ev_id += 1
        else:
            k += 1

    return wind_profile, (" | ".join(descriptions) if descriptions else "Calm"), event_meta


def generate_continuous_evolving_wind(
    max_duration_s: float = 600.0,
    dt: float = 0.50,
    seed: int | None = None,
) -> tuple[np.ndarray, str, list[tuple[float, float, float]]]:
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(max_duration_s / dt)) + 40
    t_rel = np.arange(n_steps, dtype=np.float64) * dt
    tau = t_rel / 400.0
    t_cross = float(rng.uniform(0.40, 0.60))
    width = 0.16

    if rng.random() < 0.5:
        amp_1, amp_2 = float(rng.uniform(0.32, 0.45)), -float(rng.uniform(0.38, 0.54))
        desc = f"Cont: Tailwind(+{amp_1:.2f})->Headwind({amp_2:.2f}m/s²)"
    else:
        amp_1, amp_2 = -float(rng.uniform(0.38, 0.54)), float(rng.uniform(0.30, 0.44))
        desc = f"Cont: Headwind({amp_1:.2f})->Tailwind(+{amp_2:.2f}m/s²)"

    w = 0.5 * (1.0 + np.tanh((tau - t_cross) / width))
    profile = (amp_1 * (1.0 - w) + amp_2 * w) * np.clip(t_rel / 8.0, 0.0, 1.0)
    return profile, desc, [(8.0, 400.0, amp_2)]


def _build_leg_zones(leg_length_m: float, rng: np.random.Generator) -> tuple[list[SpeedLimitZone], float]:
    """Builds 5 randomized spatial speed-limit zones for a single station-to-station leg."""
    p1 = float(np.round(leg_length_m * rng.uniform(0.19, 0.23)))
    p2 = float(np.round(leg_length_m * rng.uniform(0.44, 0.49)))
    p3 = float(np.round(leg_length_m * rng.uniform(0.62, 0.67)))
    p4 = float(np.round(leg_length_m * rng.uniform(0.84, 0.88)))

    v_z1 = float(np.round(rng.uniform(15.2, 16.5), 1))
    v_z2 = float(np.round(rng.uniform(16.8, 17.8), 1))
    v_z3 = float(np.round(rng.uniform(12.4, 13.6), 1))
    v_z4 = float(np.round(rng.uniform(16.4, 17.5), 1))
    v_z5 = float(np.round(rng.uniform(12.0, 13.0), 1))

    zones = [
        SpeedLimitZone(0.0, p1, v_z1, "Zone 1: Departure"),
        SpeedLimitZone(p1, p2, v_z2, "Zone 2: Trunk A"),
        SpeedLimitZone(p2, p3, v_z3, "Zone 3: Curve Restriction"),
        SpeedLimitZone(p3, p4, v_z4, "Zone 4: Trunk B"),
        SpeedLimitZone(p4, leg_length_m + 500.0, v_z5, "Zone 5: Station Throat"),
    ]
    min_zone_time = (
        (p1 / v_z1)
        + ((p2 - p1) / v_z2)
        + ((p3 - p2) / v_z3)
        + ((p4 - p3) / v_z4)
        + ((leg_length_m - p4) / v_z5)
    )
    min_feasible_s = float(np.ceil(min_zone_time + 31.0))
    return zones, min_feasible_s


def generate_multi_station_scenario(
    seed: int,
    n_legs: int | None = None,
    wind_mode: str = "disjoint",
    allow_infeasible_breaks: bool = True,
) -> dict:
    """
    Generates a generalized Multi-Station Railway Journey with:
      - Variable number of stations / legs `n_legs in {1, 2, 3, 4, 5}` (when n_legs is None or <= 0),
      - Generalized leg lengths in [3.8 km, 6.4 km] (total route length from ~4 km to ~31 km),
      - Stochastic passenger boarding/alighting mass changes at each station stop,
      - Continuous scalar wind field with emergent disjoint gusts.
    """
    rng = np.random.default_rng(seed)
    if n_legs is None or n_legs <= 0:
        n_legs = int(rng.integers(1, 6))

    tare_mass_kg = 72000.0
    max_train_capacity = 360
    curr_passengers = int(rng.integers(110, 190))
    curr_tickets = int(np.clip(curr_passengers + rng.integers(-18, 22), 40, max_train_capacity))
    t_clock = float(np.round(rng.uniform(0.0, 30.0) * 2.0) / 2.0)
    cum_x_m = 0.0

    weather_name, mu_adhesion, temp_c = "Dry", 1.00, 18.0
    legs = []
    station_names = [
        "Central Station", "Midtown Hub", "Tech Park",
        "University Pkwy", "Harbor Cross", "Airport Terminal",
    ]

    for leg_idx in range(n_legs):
        # Generalized leg length between 3,800 m and 6,400 m (total journey 3.8 km - 32.0 km)
        leg_len_m = float(np.round(rng.uniform(3800.0, 6400.0) / 50.0) * 50.0)
        zones, min_feas_s = _build_leg_zones(leg_len_m, rng)

        # Stochastic Passenger Boarding/Alighting across 1..5 stations
        if leg_idx == 0:
            n_leave, n_join = 0, curr_passengers
        elif leg_idx % 2 == 1:
            n_leave = int(rng.integers(20, min(70, curr_passengers - 10)))
            n_join = int(rng.integers(90, 175))
            curr_passengers = int(np.clip(curr_passengers - n_leave + n_join, 35, max_train_capacity))
        else:
            n_leave = int(rng.integers(75, min(190, curr_passengers - 15)))
            n_join = int(rng.integers(20, 80))
            curr_passengers = int(np.clip(curr_passengers - n_leave + n_join, 30, max_train_capacity))

        # 2. Sold Tickets vs. Missed-Train Passengers, Evaders & Per-Passenger Weight Variance
        missed_train_pax = int(rng.binomial(curr_passengers, 0.06))   # ~6% bought ticket but missed train
        untracked_pass_holders = int(rng.poisson(14.0))               # Untracked pass/evader riders
        curr_tickets = int(np.clip(curr_passengers + missed_train_pax - untracked_pass_holders, 30, max_train_capacity))

        avg_pax_weight_kg = float(rng.normal(82.0 if leg_idx == 2 else 75.5, 4.5))
        leg_mass_kg = float(tare_mass_kg + curr_passengers * avg_pax_weight_kg)
        ticket_est_mass_kg = float(tare_mass_kg + curr_tickets * 75.0)

        mass_time_adder = max(0.0, (leg_mass_kg - 85000.0) / 2200.0) + (4.0 if mu_adhesion < 0.85 else 0.0)
        eff_min_feas_s = min_feas_s + mass_time_adder

        # Stress-test leg (tight but physically solvable within +1.5s if shield coordinates speed & ramps)
        is_break = bool(allow_infeasible_breaks and (leg_idx == 1) and (rng.random() < 0.20))
        if is_break:
            leg_budget_s = float(np.round((eff_min_feas_s + rng.uniform(4.0, 8.0)) * 2.0) / 2.0)
        else:
            leg_budget_s = float(np.round((eff_min_feas_s + rng.uniform(16.0, 42.0)) * 2.0) / 2.0)

        leg_deadline_s = t_clock + leg_budget_s
        dwell_s = 25.0 if leg_idx < n_legs - 1 else 0.0

        if leg_idx == 0:
            is_scenario_weather = (wind_mode != "none") and (
                True if wind_mode in ("disjoint", "continuous") else ( (seed // 29) % 2 == 0 )
            )
        is_weather_active = is_scenario_weather

        if not is_weather_active:
            n_st = int(np.ceil((leg_budget_s + 150.0) / 0.50)) + 40
            w_prof, w_desc, w_meta = np.zeros(n_st, dtype=np.float64), "Clean (No Weather)", []
            weather_name, mu_adhesion, temp_c = "Clear-Dry", 1.00, 20.0
        else:
            if wind_mode == "continuous":
                w_prof, w_desc, w_meta = generate_continuous_evolving_wind(
                    leg_budget_s + 150.0, 0.50, seed=seed * 31 + leg_idx
                )
            else:
                w_prof, w_desc, w_meta = generate_disjoint_wind_events(
                    leg_budget_s + 150.0, 0.50, calm_deadband_mps2=0.22, seed=seed * 31 + leg_idx
                )
            # Sample unseen weather regime (Windy-Dry, Rain-WetRail, Cold-Humid, or Snow-Slush):
            w_type = int(rng.choice([0, 1, 2, 3], p=[0.30, 0.30, 0.20, 0.20]))
            if w_type == 0:
                weather_name, mu_adhesion, temp_c = "Windy-Dry", 1.00, 16.0
            elif w_type == 1:
                weather_name, mu_adhesion, temp_c = "Rain-WetRail", 0.86, 8.0
                w_prof = np.where(np.abs(w_prof) > 0.01, w_prof * 1.15, 0.0)
            elif w_type == 2:
                weather_name, mu_adhesion, temp_c = "Cold-Humid", 0.92, -6.0
                w_prof = np.where(np.abs(w_prof) > 0.01, w_prof * 1.12, 0.0)
            else:
                weather_name, mu_adhesion, temp_c = "Snow-Slush", 0.80, -3.0
                w_prof = np.where(np.abs(w_prof) > 0.01, w_prof * 1.20, 0.0)

        # Wayside station sensor at departure (t=0):
        station_wind_reading = float(w_prof[0] + rng.normal(0.0, 0.01)) if is_weather_active else 0.0

        corridor = RailroadCorridor(
            route_name=f"Leg {leg_idx+1} ({station_names[leg_idx]} -> {station_names[leg_idx+1]})",
            station_x_m=leg_len_m,
            start_time_s=t_clock,
            deadline_s=leg_deadline_s,
            mass_kg=leg_mass_kg,
            ticket_estimated_mass_kg=ticket_est_mass_kg,
            passengers=curr_passengers,
            sold_tickets=curr_tickets,
            max_capacity=max_train_capacity,
            weather_condition=weather_name,
            adhesion_mu_factor=mu_adhesion,
            temp_celsius=temp_c,
            station_wind_sensor_mps2=station_wind_reading,
            is_infeasible_challenge=is_break,
            speed_zones=zones,
        )

        cum_x_m += leg_len_m
        legs.append({
            "leg_idx": leg_idx + 1,
            "station_name": station_names[leg_idx + 1],
            "corridor": corridor,
            "leg_length_m": leg_len_m,
            "cum_station_x_m": cum_x_m,
            "t_start": t_clock,
            "deadline_s": leg_deadline_s,
            "budget_s": leg_budget_s,
            "dwell_s": dwell_s,
            "passengers": curr_passengers,
            "sold_tickets": curr_tickets,
            "n_join": n_join,
            "n_leave": n_leave,
            "mass_kg": leg_mass_kg,
            "ticket_mass_kg": ticket_est_mass_kg,
            "weather": weather_name,
            "adhesion_mu": mu_adhesion,
            "is_infeasible": is_break,
            "wind_profile": w_prof,
            "wind_desc": w_desc,
            "event_meta": w_meta,
        })
        t_clock = leg_deadline_s + dwell_s

    return {
        "scenario_id": seed,
        "n_legs": n_legs,
        "total_length_m": cum_x_m,
        "total_scheduled_time_s": legs[-1]["deadline_s"] - legs[0]["t_start"],
        "has_infeasible_leg": any(l["is_infeasible"] for l in legs),
        "is_weather_disturbed": is_weather_active,
        "legs": legs,
    }