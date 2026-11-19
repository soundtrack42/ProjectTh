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
    speed_sensor_noise_mps: float = 0.0  # Tachometer/radar measurement noise std (m/s)
    unmodeled_drag_factor: float = 1.0   # Plant-only aerodynamic/rolling drag multiplier
    sensor_noise_seed: int = 0           # Deterministic RNG seed for sensor noise per leg
    is_extreme_scenario: bool = False    # Tag separating extreme noise-injected scenarios
    n_cars: float = 4.0                  # Consist size (4 = calibrated reference unit)
    train_length_m: float = 0.0          # >0: distributed-mass train (grade averaging); 0 = point mass

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
    def car_scale(self) -> float:
        """n_cars / 4: traction, mass and rolling resistance scale with the number of (motor) cars."""
        return float(self.n_cars) / 4.0

    @property
    def max_traction_accel(self) -> float:
        """Physical traction limit scaled by vehicle mass and rail adhesion."""
        s = self.car_scale
        mass_scale = (85000.0 * s) / max(65000.0 * s, self.mass_kg)
        return float(self.base_max_traction_accel * mass_scale * self.adhesion_mu_factor)

    @property
    def max_service_braking(self) -> float:
        """Physical service braking limit scaled by vehicle mass and rail adhesion."""
        s = self.car_scale
        mass_scale = (85000.0 * s) / max(65000.0 * s, self.mass_kg)
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


    def compute_free_spatial_ceiling(self, x_m: float, b_shift_down: float = 0.78, alpha_soft: float = 0.36) -> float:
        """
        Shape-Free (Non-Cosine) C^2 Algebraic Soft-Curve spatial ceiling ahead of slower speed zones.
        Uses a purely algebraic C^2 Hermite-flare deceleration profile (zero trigonometric functions)
        bounded by Order-2 deceleration comfort and Order-3 jerk comfort.
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

        b_eff = max(0.72, float(b_shift_down))
        b_raw = b_eff / 0.78
        b_ratio = 0.75 if b_raw < 0.75 else (1.25 if b_raw > 1.25 else b_raw)
        bs_raw = b_eff * 0.76
        b_safe = 0.42 if bs_raw < 0.42 else (0.74 if bs_raw > 0.74 else bs_raw)
        j_max_zone = 0.22 * (b_ratio ** 1.6)
        min_drop_dur = 14.0 / b_ratio
        shape_k = 1.0 / (1.0 - alpha_soft)
        inv_1ma = shape_k

        for z in zones:
            if z.x_start > x_m and z.v_limit_mps < v_here:
                dx_to_zone = z.x_start - x_m
                if dx_to_zone >= 380.0:
                    continue
                dv_drop = v_here - z.v_limit_mps
                v_mean = 0.5 * (v_here + z.v_limit_mps)
                # Algebraic C^2 soft-curve duration required by comfort deceleration and peak flare jerk:
                t_acc_req = (shape_k * dv_drop) / b_safe
                t_jrk_req = math.sqrt((1.5 * shape_k * dv_drop) / (alpha_soft * j_max_zone))
                t_drop = min_drop_dur if min_drop_dur >= t_acc_req else t_acc_req
                if t_jrk_req > t_drop:
                    t_drop = t_jrk_req
                dx_drop = v_mean * t_drop

                if dx_to_zone < dx_drop and dx_drop > 1.0:
                    u_raw = 1.0 - (dx_to_zone / dx_drop)
                    u_s = 0.0 if u_raw < 0.0 else (1.0 if u_raw > 1.0 else u_raw)
                    # Evaluate algebraic C^2 soft progression psi_soft(1 - u_s; alpha_soft):
                    s = 1.0 - u_s
                    if s < alpha_soft:
                        tau = s / alpha_soft
                        tau2 = tau * tau
                        prog = alpha_soft * inv_1ma * (tau2 * tau - 0.5 * tau2 * tau2)
                    elif s <= 1.0 - alpha_soft:
                        prog = (s - 0.5 * alpha_soft) * inv_1ma
                    else:
                        tau = (1.0 - s) / alpha_soft
                        tau2 = tau * tau
                        prog = 1.0 - alpha_soft * inv_1ma * (tau2 * tau - 0.5 * tau2 * tau2)
                    v_alg_zone = z.v_limit_mps + dv_drop * prog
                    if v_alg_zone < v_here:
                        v_here = v_alg_zone
        return float(v_here)


# ===========================================================================
# Physical wind model: wind speed (m/s) -> longitudinal train disturbance (m/s^2)
# ===========================================================================
# The plant's own Davis aerodynamic coefficient (NonlinearTrainPhysics.A = 14.5 N s^2/m^2) and
# nominal effective mass M_eff = 85 t * (1 + 0.08) translate a relative-air-speed change into a
# longitudinal acceleration, so the wind disturbance is consistent with the plant's drag model:
#     d_wind = -[A * ((v_ref + w_head) |v_ref + w_head| - v_ref^2) + k_yaw * A * w_cross^2] * rho/rho_0 * c_reg / M_eff
# Reference speeds: v_ref = 0 m/s inside the sheltered station departure window (k = 0..7, t < 4 s,
# where the train is still below ~1 m/s), and v_ref = 15 m/s (typical cruise) on the open line.
# Wind speeds are capped at 30 m/s (108 km/h, upper end of Beaufort 10 "storm"), which bounds
# |d_wind| at about 0.28 m/s^2 head-on (x drag/air-density multipliers) and about 0.07 m/s^2 tailwind.
AERO_DAVIS_A_NS2_M2 = 14.5
WIND_REF_CRUISE_MPS = 15.0
WIND_REF_M_EFF_KG = 85000.0 * 1.08
WIND_YAW_SIDE_DRAG_K = 0.15
MAX_WIND_SPEED_MPS = 30.0
AERO_NOSE_TAIL_FRACTION = 0.5     # must match NonlinearTrainPhysics.AERO_NOSE_TAIL_FRACTION
CAR_LENGTH_M = 18.75              # 75 m four-car unit

# Train-size switch. None = the calibrated 4-car point-mass plant used for the paper benchmark.
# A number N enables the distributed-length model with N cars (CLI --train-cars N); 0 = use the real
# Digitraffic composition length on real-data journeys (synthetic journeys then use 4 cars).
TRAIN_CARS: float | None = None


def train_size_config(n_cars: float | None = None) -> tuple[float, float, bool]:
    """Returns (n_cars, length_m, length_effects) for the current switch."""
    n = TRAIN_CARS if n_cars is None else n_cars
    if n is None:
        return 4.0, 0.0, False
    n = 4.0 if n <= 0 else float(n)
    return n, n * CAR_LENGTH_M, True


def tail_clearance_time_s(zones: list, length_m: float) -> float:
    """Extra minimum running time from holding each lower limit for one train length longer."""
    if length_m <= 0.0:
        return 0.0
    return float(sum(length_m * (1.0 / zones[i].v_limit_mps - 1.0 / zones[i + 1].v_limit_mps)
                     for i in range(len(zones) - 1) if zones[i + 1].v_limit_mps > zones[i].v_limit_mps))


def apply_tail_clearance(zones: list, length_m: float) -> list:
    """
    A speed restriction holds until the tail has cleared it: every boundary where the limit rises is
    moved forward by the train length (the head position is the reference everywhere else).
    """
    if length_m <= 0.0:
        return zones
    zs = [SpeedLimitZone(z.x_start, z.x_end, z.v_limit_mps, z.label) for z in zones]
    for i in range(len(zs) - 1):
        if zs[i + 1].v_limit_mps > zs[i].v_limit_mps:
            b_new = min(zs[i].x_end + length_m, zs[i + 1].x_end - 1.0)
            zs[i].x_end = b_new
            zs[i + 1].x_start = b_new
    return zs
STATION_WINDOW_STEPS = 8          # k = 0..7 (t < 4.0 s): wayside station weather telemetry available
STATION_WINDOW_S = 4.0


def wind_speed_to_longitudinal_accel(
    w_head_mps,
    w_cross_mps=0.0,
    v_ref_mps: float = WIND_REF_CRUISE_MPS,
    rho_ratio: float = 1.0,
    drag_mult: float = 1.0,
    car_scale: float = 1.0,
):
    """
    Longitudinal disturbance (m/s^2, <0 = retarding) from the head- (+) / tail- (-) wind component.
    car_scale = n_cars/4: head/tail drag grows only through the skin-friction share of A, the crosswind
    (side-area) term grows with length, and the effective mass grows with the consist.
    """
    w_h = np.clip(np.asarray(w_head_mps, dtype=np.float64), -MAX_WIND_SPEED_MPS, MAX_WIND_SPEED_MPS)
    w_c = np.clip(np.abs(np.asarray(w_cross_mps, dtype=np.float64)), 0.0, MAX_WIND_SPEED_MPS)
    v_rel = v_ref_mps + w_h
    s = float(car_scale)
    a_long = AERO_DAVIS_A_NS2_M2 * (AERO_NOSE_TAIL_FRACTION + (1.0 - AERO_NOSE_TAIL_FRACTION) * s)
    f_long = a_long * (v_rel * np.abs(v_rel) - v_ref_mps * v_ref_mps)
    f_side = WIND_YAW_SIDE_DRAG_K * AERO_DAVIS_A_NS2_M2 * s * (w_c * w_c)
    d = -(f_long + f_side) * float(rho_ratio) * float(drag_mult) / (WIND_REF_M_EFF_KG * s)
    return float(d) if np.ndim(d) == 0 else d


def wind_speed_series_to_accel_profile(
    w_head_series: np.ndarray,
    w_head_station: float,
    w_cross: float = 0.0,
    dt: float = 0.50,
    rho_ratio: float = 1.0,
    drag_mult: float = 1.0,
    car_scale: float = 1.0,
) -> np.ndarray:
    """
    Maps a head-wind speed series (m/s) to the plant disturbance profile (m/s^2):
    constant station wind (v_ref = 0) over the k = 0..7 telemetry window, then a smooth 6 s blend
    to the open-line value (v_ref = 15 m/s) including gusts/turbulence.
    """
    n = len(w_head_series)
    t_grid = np.arange(n, dtype=np.float64) * dt
    d_station = wind_speed_to_longitudinal_accel(w_head_station, w_cross, 0.0, rho_ratio, drag_mult, car_scale)
    d_open = wind_speed_to_longitudinal_accel(w_head_series, w_cross, WIND_REF_CRUISE_MPS, rho_ratio, drag_mult, car_scale)
    gate = np.clip((t_grid - STATION_WINDOW_S) / 6.0, 0.0, 1.0)
    prof = (1.0 - gate) * d_station + gate * d_open
    prof[:STATION_WINDOW_STEPS] = d_station
    return prof


def _extract_wind_events(profile: np.ndarray, dt: float, thresh: float = 0.04, min_len_s: float = 6.0):
    event_meta, descriptions = [], []
    active = np.abs(profile) > thresh
    k, n, ev_id = 0, len(profile), 1
    while k < n:
        if active[k]:
            k0 = k
            while k < n and active[k] and (np.sign(profile[k]) == np.sign(profile[k0])):
                k += 1
            if (k - k0) * dt >= min_len_s:
                seg = profile[k0:k]
                pk = float(seg[np.argmax(np.abs(seg))])
                event_meta.append((k0 * dt, k * dt, pk))
                tag = "Headwind" if pk < 0 else "Tailwind"
                descriptions.append(f"E{ev_id}:{tag}({pk:+.2f}m/s²,[{k0 * dt:.0f}s,{k * dt:.0f}s])")
                ev_id += 1
        else:
            k += 1
    return event_meta, descriptions


def _disjoint_head_wind_speed(total_duration_s: float, dt: float, rng: np.random.Generator,
                              peak_lo: float = 10.0, peak_hi: float = 17.0,
                              calm_mps: float = 2.0) -> np.ndarray:
    """Two isolated gust fronts (one head-, one tail-wind), moderate 10-17 m/s (Beaufort 5-7), calm elsewhere."""
    n_steps = int(np.ceil(total_duration_s / dt)) + 40
    t_grid = np.arange(n_steps, dtype=np.float64) * dt
    first_sign = 1.0 if (rng.random() < 0.5) else -1.0
    w = np.zeros(n_steps, dtype=np.float64)
    for f_i in range(2):
        center_s = float(rng.uniform(0.08 + 0.45 * f_i, 0.35 + 0.45 * f_i) * total_duration_s)
        width_s = float(rng.uniform(24.0, 36.0))
        sign = first_sign if (f_i == 0) else -first_sign
        w += sign * float(rng.uniform(peak_lo, peak_hi)) * np.exp(-0.5 * ((t_grid - center_s) / width_s) ** 2)
    # Calm air below 2 m/s (light air, Beaufort 1) -> genuinely disjoint events:
    return np.sign(w) * np.maximum(0.0, np.abs(w) - calm_mps)


def generate_disjoint_wind_events(
    total_duration_s: float = 600.0,
    dt: float = 0.50,
    calm_deadband_mps2: float = 0.0,
    seed: int | None = None,
    drag_mult: float = 1.0,
) -> tuple[np.ndarray, str, list[tuple[float, float, float]]]:
    """
    Isolated head/tail gust fronts (moderate 10-17 m/s wind speed) mapped through the physical
    Davis-drag wind model; calm at the station departure window (k = 0..7).
    `calm_deadband_mps2` is kept for API compatibility (calm air is now defined in wind speed).
    """
    rng = np.random.default_rng(seed)
    w_head = _disjoint_head_wind_speed(total_duration_s, dt, rng)
    prof = wind_speed_series_to_accel_profile(w_head, w_head_station=0.0, dt=dt, drag_mult=drag_mult)
    meta, desc = _extract_wind_events(prof, dt)
    return prof, (" | ".join(desc) if desc else "Calm"), meta


def _continuous_head_wind_speed(max_duration_s: float, dt: float, rng: np.random.Generator):
    n_steps = int(np.ceil(max_duration_s / dt)) + 40
    tau = (np.arange(n_steps, dtype=np.float64) * dt) / 400.0
    t_cross = float(rng.uniform(0.40, 0.60))
    if rng.random() < 0.5:
        w1, w2 = -float(rng.uniform(9.0, 14.0)), float(rng.uniform(11.0, 17.0))
    else:
        w1, w2 = float(rng.uniform(11.0, 17.0)), -float(rng.uniform(9.0, 14.0))
    blend = 0.5 * (1.0 + np.tanh((tau - t_cross) / 0.16))
    return w1 * (1.0 - blend) + w2 * blend, w1, w2


def generate_continuous_evolving_wind(
    max_duration_s: float = 600.0,
    dt: float = 0.50,
    seed: int | None = None,
    drag_mult: float = 1.0,
) -> tuple[np.ndarray, str, list[tuple[float, float, float]]]:
    """Continuously veering wind (head <-> tail, 9-17 m/s) through the physical wind model."""
    rng = np.random.default_rng(seed)
    w_head, w1, w2 = _continuous_head_wind_speed(max_duration_s, dt, rng)
    prof = wind_speed_series_to_accel_profile(w_head, w_head_station=float(w_head[0]), dt=dt, drag_mult=drag_mult)
    tag = lambda w: ("Headwind" if w > 0 else "Tailwind") + f"({abs(w):.0f}m/s)"
    desc = f"Cont: {tag(w1)}->{tag(w2)} | d in [{prof.min():+.2f},{prof.max():+.2f}]m/s²"
    return prof, desc, [(8.0, 400.0, float(prof[-1]))]


def _ou_turbulence(n_steps: int, dt: float, sigma: float, tau_s: float, rng: np.random.Generator,
                   lp_tau_s: float = 3.2, start_k: int = STATION_WINDOW_STEPS) -> np.ndarray:
    """2nd-order band-limited Ornstein-Uhlenbeck turbulence (m/s), zero inside the station window."""
    phi = math.exp(-dt / tau_s)
    alpha = dt / (lp_tau_s + dt)
    s1 = 0.0
    out = np.zeros(n_steps, dtype=np.float64)
    for k in range(start_k, n_steps):
        s1 = phi * s1 + sigma * math.sqrt(1.0 - phi * phi) * float(rng.normal())
        out[k] = (1.0 - alpha) * out[k - 1] + alpha * s1
    return out


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
    extreme_noise: bool = False,
    n_cars: float | None = None,
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

    n_cars_eff, train_len_m, length_fx = train_size_config(n_cars)
    s_car = n_cars_eff / 4.0
    # Passenger flows are drawn for the 4-car reference (same RNG stream for every size) and scaled by s_car.
    tare_mass_kg = 72000.0 * s_car
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
        pax_phys = int(round(curr_passengers * s_car))
        tickets_phys = int(round(curr_tickets * s_car))
        leg_mass_kg = float(tare_mass_kg + pax_phys * avg_pax_weight_kg)
        ticket_est_mass_kg = float(tare_mass_kg + tickets_phys * 75.0)
        if length_fx:
            zones = apply_tail_clearance(zones, train_len_m)

        if leg_idx == 0:
            is_scenario_weather = (wind_mode != "none") and (
                True if wind_mode in ("disjoint", "continuous") else ((seed // 29) % 2 == 0)
            )
        is_weather_active = is_scenario_weather

        if not is_weather_active:
            weather_name, mu_adhesion, temp_c, w_type = "Clear-Dry", 1.00, 20.0, -1
        else:
            # Sample unseen weather regime before computing leg budget so adhesion adder uses current leg's mu_adhesion
            w_type = int(rng.choice([0, 1, 2, 3, 4], p=[0.22, 0.24, 0.18, 0.18, 0.18]))
            if w_type == 0:
                weather_name, mu_adhesion, temp_c = "Windy-Dry", 1.00, 16.0
            elif w_type == 1:
                weather_name, mu_adhesion, temp_c = "Rain-WetRail", 0.86, 8.0
            elif w_type == 2:
                weather_name, mu_adhesion, temp_c = "Cold-Humid", 0.92, -6.0
            elif w_type == 3:
                weather_name, mu_adhesion, temp_c = "Snow-Slush", 0.80, -3.0
            else:
                weather_name, mu_adhesion, temp_c = "Heat-Thermal", 0.94, 38.0

        mass_time_adder = max(0.0, (leg_mass_kg / s_car - 85000.0) / 2200.0) + (4.0 if mu_adhesion < 0.85 else 0.0)
        eff_min_feas_s = min_feas_s + mass_time_adder

        # Stress-test leg (tight but physically solvable within +1.5s if shield coordinates speed & ramps)
        is_break = bool(allow_infeasible_breaks and (leg_idx == 1) and (rng.random() < 0.20))
        if is_break:
            leg_budget_s = float(np.round((eff_min_feas_s + rng.uniform(4.0, 8.0)) * 2.0) / 2.0)
        else:
            leg_budget_s = float(np.round((eff_min_feas_s + rng.uniform(16.0, 42.0)) * 2.0) / 2.0)

        # Tail-clearance running time is added after the budget so every train size draws the same journeys.
        tail_s = float(np.round(tail_clearance_time_s(zones, train_len_m) * 2.0) / 2.0)
        leg_deadline_s = t_clock + leg_budget_s + tail_s
        dwell_s = 25.0 if leg_idx < n_legs - 1 else 0.0

        drag_mult_reg = {0: 1.00, 1: 1.15, 2: 1.12, 3: 1.20, 4: 1.05}.get(w_type, 1.0)
        rho_ratio_leg = 293.15 / max(250.0, 273.15 + temp_c)
        n_st = int(np.ceil((leg_budget_s + 150.0) / 0.50)) + 40
        w_cross_leg = 0.0
        if not is_weather_active:
            w_head = np.zeros(n_st, dtype=np.float64)
            w_desc = "Clean (No Weather)"
        elif wind_mode == "continuous":
            w_head, w1, w2 = _continuous_head_wind_speed(leg_budget_s + 150.0, 0.50, np.random.default_rng(seed * 31 + leg_idx))
            w_desc = f"Cont wind {w1:+.0f}->{w2:+.0f} m/s (head>0)"
        else:
            w_head = _disjoint_head_wind_speed(leg_budget_s + 150.0, 0.50, np.random.default_rng(seed * 31 + leg_idx))
            w_desc = "Disjoint gust fronts 10-17 m/s"
        w_head_station = float(w_head[0])

        # Wayside station sensor at departure (first 8 steps, t < 4.0 s):
        anem_std = 0.018 if extreme_noise else 0.008
        if extreme_noise:
            # Beaufort 9-10 storm: sustained 20-24 m/s at a random bearing, gust fronts up to 26-30 m/s
            # (<= 30 m/s cap) alternating head/tail on the open line, 12% turbulence intensity.
            n_st = max(n_st, len(w_head))
            t_grid = np.arange(n_st, dtype=np.float64) * 0.50
            v_sus = float(rng.uniform(20.0, 24.0))
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            w_head_sus = v_sus * math.cos(ang)
            w_cross_leg = abs(v_sus * math.sin(ang))
            w_head = np.full(n_st, w_head_sus, dtype=np.float64)
            open_line_gate = np.clip((t_grid - STATION_WINDOW_S) / 6.0, 0.0, 1.0)
            sgn0 = 1.0 if w_head_sus >= 0.0 else -1.0
            for g_i in range(3):
                c_s = float(rng.uniform(0.14 + 0.24 * g_i, 0.30 + 0.24 * g_i) * leg_budget_s)
                w_s = float(rng.uniform(16.0, 26.0))
                sgn = sgn0 if (g_i % 2 == 0) else -sgn0
                target = sgn * float(rng.uniform(26.0, 30.0))
                w_head = w_head + (target - w_head_sus) * np.exp(-0.5 * ((t_grid - c_s) / w_s) ** 2) * open_line_gate
            w_head = w_head + _ou_turbulence(n_st, 0.50, 0.12 * 28.0, 4.8, rng) * open_line_gate
            w_head = np.clip(w_head, -MAX_WIND_SPEED_MPS, MAX_WIND_SPEED_MPS)
            w_head_station = w_head_sus
            drag_mult_reg = max(drag_mult_reg, 1.10)
            mu_adhesion = float(np.clip(mu_adhesion * rng.uniform(0.86, 0.94), 0.72, 0.94))
            is_weather_active = True
            if weather_name == "Clear-Dry":
                weather_name = "Storm-Gale"

        w_prof = wind_speed_series_to_accel_profile(
            w_head, w_head_station, w_cross=w_cross_leg, dt=0.50, rho_ratio=rho_ratio_leg, drag_mult=drag_mult_reg,
            car_scale=s_car,
        )
        w_meta, ev_desc = _extract_wind_events(w_prof, 0.50)
        if extreme_noise:
            w_desc = (
                f"Extreme-Noise({weather_name}, sustained {abs(w_head_station):.0f}m/s head-comp, gusts<=30m/s, "
                f"d in [{w_prof.min():+.2f},{w_prof.max():+.2f}]m/s^2)"
            )
        elif ev_desc:
            w_desc = f"{w_desc}: " + " | ".join(ev_desc)

        station_wind_reading = float(w_prof[0] + rng.normal(0.0, anem_std)) if is_weather_active else 0.0
        speed_noise_std = 0.0025 if extreme_noise else 0.0
        unmodeled_drag = float(rng.uniform(1.03, 1.08)) if extreme_noise else 1.0

        corridor = RailroadCorridor(
            route_name=f"Leg {leg_idx+1} ({station_names[leg_idx]} -> {station_names[leg_idx+1]})",
            station_x_m=leg_len_m,
            start_time_s=t_clock,
            deadline_s=leg_deadline_s,
            mass_kg=leg_mass_kg,
            ticket_estimated_mass_kg=ticket_est_mass_kg,
            passengers=pax_phys,
            sold_tickets=tickets_phys,
            max_capacity=int(round(max_train_capacity * s_car)),
            weather_condition=weather_name,
            adhesion_mu_factor=mu_adhesion,
            temp_celsius=temp_c,
            station_wind_sensor_mps2=station_wind_reading,
            is_infeasible_challenge=is_break,
            speed_sensor_noise_mps=speed_noise_std,
            unmodeled_drag_factor=unmodeled_drag,
            sensor_noise_seed=int(seed * 101 + leg_idx * 17),
            is_extreme_scenario=bool(extreme_noise),
            speed_zones=zones,
            n_cars=n_cars_eff,
            train_length_m=train_len_m,
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
            "budget_s": leg_budget_s + tail_s,
            "dwell_s": dwell_s,
            "passengers": pax_phys,
            "sold_tickets": tickets_phys,
            "n_join": n_join,
            "n_leave": n_leave,
            "mass_kg": leg_mass_kg,
            "ticket_mass_kg": ticket_est_mass_kg,
            "weather": weather_name,
            "adhesion_mu": mu_adhesion,
            "temp_c": temp_c,
            "is_infeasible": is_break,
            "is_harsh_injected": bool(extreme_noise),
            "speed_sensor_noise_mps": speed_noise_std,
            "unmodeled_drag_factor": unmodeled_drag,
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
        "is_harsh_injected": bool(extreme_noise),
        "is_extreme_scenario": bool(extreme_noise),
        "legs": legs,
    }