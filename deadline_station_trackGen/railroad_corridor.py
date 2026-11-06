"""
Module 1: Multi-Station Long Railroad Corridor (14-18 km, 2-3 Station Stops),
Dynamic Passenger Weight Changes, and Disjoint Spurious Wind Generator (`railroad_corridor.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
        """Mass- and rail-adhesion-scaled traction limit (Rain/Snow reduces wheel-rail friction)."""
        mass_scale = 85000.0 / max(65000.0, self.mass_kg)
        return float(self.base_max_traction_accel * mass_scale * self.adhesion_mu_factor)

    @property
    def max_service_braking(self) -> float:
        """Mass- and rail-adhesion-scaled braking limit (Rain/Snow requires earlier braking spot)."""
        mass_scale = 85000.0 / max(65000.0, self.mass_kg)
        return float(self.base_max_service_braking * mass_scale * self.adhesion_mu_factor)

    def get_speed_limit(self, x_m: float) -> float:
        for z in self.speed_zones:
            if z.x_start <= x_m <= z.x_end:
                return z.v_limit_mps
        return self.speed_zones[-1].v_limit_mps

    def compute_smooth_spatial_ceiling(self, x_m: float, b_shift_down: float = 0.78) -> float:
        """
        C^1-smooth half-cosine anticipatory ceiling ahead of every slower speed zone
        so every downward speed shift is a gentle S-curve inside the comfort range.
        """
        v_here = self.get_speed_limit(x_m)
        b_safe = float(np.clip(b_shift_down, 0.50, min(self.comfort_braking_max, self.max_service_braking * 0.85)))

        for z in self.speed_zones:
            if z.x_start > x_m and z.v_limit_mps < v_here:
                dx_to_zone = z.x_start - x_m
                dx_drop = (np.pi * max(0.0, (v_here ** 2) - (z.v_limit_mps ** 2))) / (4.0 * b_safe)
                if dx_to_zone < dx_drop and dx_drop > 1.0:
                    u_s = float(np.clip(1.0 - (dx_to_zone / dx_drop), 0.0, 1.0))
                    v_cos_zone = z.v_limit_mps + (v_here - z.v_limit_mps) * 0.5 * (1.0 + float(np.cos(np.pi * u_s)))
                    v_here = min(v_here, v_cos_zone)
        return float(v_here)


def generate_disjoint_wind_events(
    total_duration_s: float = 600.0,
    dt: float = 0.50,
    n_events: int | None = None,
    min_event_dur_s: float = 35.0,
    max_event_dur_s: float = 65.0,
    min_calm_gap_s: float = 40.0,
    seed: int | None = None,
) -> tuple[np.ndarray, str, list[tuple[float, float, float]]]:
    """
    Generates strictly disjoint spurious wind events separated by calm zero-wind gaps.
    Events never overlap and have C^1-smooth cosine bell envelopes.
    """
    rng = np.random.default_rng(seed)
    if n_events is None:
        n_events = 2

    n_steps = int(np.ceil(total_duration_s / dt)) + 40
    wind_profile = np.zeros(n_steps, dtype=np.float64)

    usable_start_s = 16.0
    usable_end_s = max(usable_start_s + 160.0, total_duration_s - 150.0)
    cursor_s = usable_start_s

    base_sign = -1.0 if rng.random() < 0.55 else 1.0
    signs = [base_sign]
    for i in range(1, n_events):
        signs.append(-signs[-1] if rng.random() < 0.80 else signs[-1])

    event_meta = []
    descriptions = []

    for i in range(n_events):
        dur_s = float(rng.uniform(min_event_dur_s, max_event_dur_s))
        rem_events = n_events - 1 - i
        latest_start = usable_end_s - dur_s - rem_events * (min_event_dur_s + min_calm_gap_s)
        start_s = cursor_s if cursor_s >= latest_start else float(rng.uniform(cursor_s, latest_start))
        end_s = start_s + dur_s
        cursor_s = end_s + min_calm_gap_s

        sgn = signs[i]
        mag = float(rng.uniform(0.36, 0.58)) * sgn
        k_s = int(start_s / dt)
        k_e = min(n_steps, int(end_s / dt))
        k_len = max(4, k_e - k_s)

        tau = np.linspace(0.0, 1.0, k_len)
        bell_env = 0.5 * (1.0 - np.cos(2.0 * np.pi * tau))
        swell = 1.0 + 0.14 * np.sin(2.0 * np.pi * tau)
        wind_profile[k_s:k_e] = np.clip(mag * bell_env * swell, -0.65, 0.58)

        event_meta.append((start_s, end_s, mag))
        tag = "Headwind" if sgn < 0 else "Tailwind"
        descriptions.append(f"E{i+1}:{tag}({mag:+.2f}m/s²,[{start_s:.0f}s,{end_s:.0f}s])")

    return wind_profile, " | ".join(descriptions), event_meta


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
    n_legs: int = 3,
    wind_mode: str = "disjoint",
    allow_infeasible_breaks: bool = True,
) -> dict:
    """
    Generates a long 3-Station Multi-Leg Railway Journey (total length ~15 - 17.5 km,
    total duration ~1250 - 1450 s) with:
      - 3 consecutive station landings (`n_legs = 3`, each leg 4.8 - 5.9 km),
      - Realistic passenger boarding & alighting at each station stop, changing the
        physical mass of the 4-car train (`M_tare = 72,000 kg` + `75 kg * n_passengers`),
      - Disjoint spurious wind events on each leg, and
      - ~20% infeasible "rule-breaking" leg deadlines to test RL & Shield behavior.
    """
    rng = np.random.default_rng(seed)
    tare_mass_kg = 72000.0
    max_train_capacity = 360
    curr_passengers = int(rng.integers(120, 185))
    curr_tickets = int(np.clip(curr_passengers + rng.integers(-18, 22), 40, max_train_capacity))
    t_clock = float(np.round(rng.uniform(0.0, 30.0) * 2.0) / 2.0)
    cum_x_m = 0.0

    # Sample environmental weather & rail-friction condition for the journey
    weather_choices = [
        ("Dry", 1.00, float(rng.uniform(14.0, 24.0))),
        ("Rain", float(rng.uniform(0.82, 0.88)), float(rng.uniform(6.0, 13.0))),
        ("Snow", float(rng.uniform(0.72, 0.78)), float(rng.uniform(-6.0, 1.0))),
    ]
    w_idx = int(rng.choice([0, 1, 2], p=[0.55, 0.30, 0.15]))
    weather_name, mu_adhesion, temp_c = weather_choices[w_idx]

    legs = []
    station_names = ["Central Station", "Midtown Hub", "Tech Park Terminal", "Airport Express"]

    for leg_idx in range(n_legs):
        leg_len_m = float(np.round(rng.uniform(4900.0, 5800.0) / 50.0) * 50.0)
        zones, min_feas_s = _build_leg_zones(leg_len_m, rng)

        # 1. Stochastic Passenger Boarding/Alighting (Capped at max_train_capacity = 360)
        if leg_idx == 0:
            n_leave, n_join = 0, curr_passengers
        elif leg_idx == 1:
            n_leave = int(rng.integers(25, 65))
            n_join = int(rng.integers(115, 185))
            curr_passengers = int(np.clip(curr_passengers - n_leave + n_join, 40, max_train_capacity))
        else:
            n_leave = int(rng.integers(100, min(210, curr_passengers - 20)))
            n_join = int(rng.integers(20, 75))
            curr_passengers = int(np.clip(curr_passengers - n_leave + n_join, 30, max_train_capacity))

        # 2. Sold Tickets vs. Missed-Train Passengers, Evaders & Per-Passenger Weight Variance
        missed_train_pax = int(rng.binomial(curr_passengers, 0.06))   # ~6% bought ticket but missed train
        untracked_pass_holders = int(rng.poisson(14.0))               # Untracked pass/evader riders
        curr_tickets = int(np.clip(curr_passengers + missed_train_pax - untracked_pass_holders, 30, max_train_capacity))

        # Mean weight per passenger varies (e.g., heavier luggage on Leg 3 Airport Express, or overweight crowd)
        avg_pax_weight_kg = float(rng.normal(82.0 if leg_idx == 2 else 75.5, 4.5))
        leg_mass_kg = float(tare_mass_kg + curr_passengers * avg_pax_weight_kg)
        ticket_est_mass_kg = float(tare_mass_kg + curr_tickets * 75.0)  # Controller only knows sold tickets * 75kg!

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

        if wind_mode == "continuous":
            w_prof, w_desc, w_meta = generate_continuous_evolving_wind(
                leg_budget_s + 150.0, 0.50, seed=seed * 31 + leg_idx
            )
        else:
            w_prof, w_desc, w_meta = generate_disjoint_wind_events(
                leg_budget_s + 150.0, 0.50, seed=seed * 31 + leg_idx
            )

        # Wayside station anemometer reading at departure (with small sensor noise)
        station_wind_reading = float(w_prof[0] + rng.normal(0.0, 0.015))

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
        "legs": legs,
    }