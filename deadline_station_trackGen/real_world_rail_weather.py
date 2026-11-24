"""
Module 5: Real-World Railway Tracking (Finland Digitraffic API) + Meteorological
Disturbance Merger (Open-Meteo Forecast & ERA5 Archive API) + Harsh-Weather Stress
Noise Injector (`real_world_rail_weather.py`).

Selected Real-World Sources:
  1. Finland's Digitraffic (`rata.digitraffic.fi`):
     - `/api/v1/metadata/stations`: exact WGS-84 station coordinates
     - `/api/v1/trains/{date}` & `/trains/{date}/{num}`: real stop-by-stop schedules & actuals
     - `/api/v1/compositions/{date}` & `/compositions/{date}/{num}`: rolling stock classes, lengths & speed caps
     - `/api/v1/train-locations/latest/{num}`: live GPS speed & coordinates
  2. Open-Meteo (`api.open-meteo.com` & `archive-api.open-meteo.com`):
     - Hourly `temperature_2m`, `rain`, `snowfall`, `wind_speed_10m`, `wind_direction_10m`, `wind_gusts_10m`
       queried at each station-to-station leg's geodesic midpoint.

Includes calibrated Harsh-Weather Stress-Noise Injection (`harsh_weather_mode="mixed"|"always"|"raw"`)
so post-training benchmarks test both real meteorological conditions AND injected extreme wind gusts,
heatwaves (CWR rail thermal expansion), sub-zero cold snaps, heavy rain, and heavy snow/slush.
"""
from __future__ import annotations

import datetime
import gzip
import json
import math
import time
import urllib.parse
import urllib.request
import numpy as np

from railroad_corridor import (
    RailroadCorridor,
    _build_leg_zones,
    MAX_WIND_SPEED_MPS,
    wind_speed_to_longitudinal_accel,
    wind_speed_series_to_accel_profile,
    _drop_ou_warmup,
    anemometer_sigma_series,
    _extract_wind_events,
    train_size_config,
    apply_tail_clearance,
    tail_clearance_time_s,
    CAR_LENGTH_M,
)
import railroad_corridor as _rc

DIGITRAFFIC_HEADERS = {
    "Digitraffic-User": "ProjectTh-ATO-ContractShield-Research/1.0",
    "Accept": "application/json",
    "Accept-Encoding": "gzip",
}

# Verified Finnish VR / HSL rolling-stock approximate tare mass (tonnes) per unit/locomotive/coach
ROLLING_STOCK_TARE_TONNES = {
    "Sm5": 122.0,   # Stadler FLIRT 4-car electric commuter unit (75 m)
    "Sm4": 88.0,    # Alstom Coradia 2-car commuter EMU (54 m)
    "Sm2": 77.0,    # Valmet 2-car commuter EMU (53 m)
    "Sm3": 328.0,   # VR Pendolino 6-car tilting trainset (159 m)
    "Sm6": 363.0,   # Allegro 7-car high-speed trainset
    "Dm12": 57.0,   # Skoda/ČKD diesel railbus
    "Sr1": 84.0,    # Electric locomotive
    "Sr2": 84.0,    # Siemens/SLM Re460-based electric locomotive
    "Sr3": 89.0,    # Siemens Vectron electric locomotive
    "Dr16": 84.0,   # Diesel-electric locomotive
    "Dr19": 88.0,   # Stadler EuroDual/diesel locomotive
}

HARSH_REGIMES = ("Storm-Gale", "Heat-Thermal", "Arctic-Cold", "Heavy-Rain", "Blizzard-Snow")

# Physically meaningful railway extreme weather specifications (EN 14067-6 / Beaufort 8-10 storm scale;
# peak gusts strictly bounded <= 30 m/s [108 km/h], never unphysical >50-100 m/s hurricane/tornado forces):
# Physically defined extreme regimes. Wind is bounded by Beaufort 10 ("storm", 24.5-28.4 m/s) with
# gusts capped at 30 m/s (108 km/h); the train disturbance |d_wind| is not a free parameter but follows
# from the plant's Davis drag coefficient through `wind_speed_to_longitudinal_accel` (A = 14.5 N s^2/m^2,
# M_eff = 91.8 t, cruise v_ref = 15 m/s), air density rho(T) and the precipitation drag multiplier.
#   name: (sustained m/s, gust m/s, T_amb C, drag multiplier, mu_adh range, extra description)
_EXTREME_REGIME_DEFS = {
    "Storm-Gale": ((20.0, 24.0), (26.0, 30.0), (5.0, 15.0), 1.10, (0.86, 0.92), "Beaufort 9-10 gale/storm"),
    "Heavy-Rain": ((8.0, 14.0), (14.0, 21.0), (5.0, 16.0), 1.12, (0.76, 0.84), "torrential rain 18-35 mm/h, wet-rail lubrication"),
    "Blizzard-Snow": ((9.0, 15.0), (15.0, 23.0), (-12.0, -1.0), 1.14, (0.72, 0.80), "heavy snow/slush 5-12 cm/h, +14% snow drag"),
    "Arctic-Cold": ((7.0, 13.0), (12.0, 19.0), (-22.0, -12.0), 1.10, (0.82, 0.88), "sub-zero freeze, +9-17% dense-air drag, frost"),
    "Heat-Thermal": ((6.0, 11.0), (10.0, 17.0), (35.0, 42.0), 1.05, (0.89, 0.94), "heatwave, CWR rail 52-60 C (+5-8% curve drag, +6-9% Cu I^2R loss)"),
}


def _regime_spec_text(sus, gust, temp, drag, mu, extra) -> str:
    rho_hi = 293.15 / max(250.0, 273.15 + temp[0])
    d_head = wind_speed_to_longitudinal_accel(gust[1], 0.0, 15.0, rho_hi, drag)
    d_tail = wind_speed_to_longitudinal_accel(-gust[1], 0.0, 15.0, rho_hi, drag)
    d_stat = wind_speed_to_longitudinal_accel(sus[1], 0.0, 0.0, rho_hi, drag)
    return (
        f"{extra}; sustained {sus[0]:.0f}-{sus[1]:.0f} m/s, gusts {gust[0]:.0f}-{gust[1]:.0f} m/s "
        f"({gust[1] * 3.6:.0f} km/h), T_amb {temp[0]:.0f}..{temp[1]:.0f} C, mu_adh {mu[0]:.2f}-{mu[1]:.2f} -> "
        f"d_wind in [{d_head:+.2f}, {d_tail:+.2f}] m/s^2 at 15 m/s cruise (station |d| <= {abs(d_stat):.2f}); "
        f"plus tachometer noise 0.0025 m/s, anemometer noise 0.018 m/s^2, unmodeled drag +3-8%"
    )


EXTREME_WEATHER_REGIME_SPECS = {k: _regime_spec_text(*v) for k, v in _EXTREME_REGIME_DEFS.items()}

# In-process session caches to avoid redundant API hits across multi-run benchmarks
_STATIONS_CACHE: dict[str, dict] | None = None
_TRAINS_DATE_CACHE: dict[str, dict[int, dict]] = {}
_COMPS_DATE_CACHE: dict[str, dict[int, dict]] = {}
_CANDIDATES_DATE_CACHE: dict[str, list[int]] = {}
_WEATHER_CACHE: dict[tuple[float, float, str], tuple[dict, str]] = {}
_LOCATIONS_CACHE: dict[int, float] | None = None

# Verified Real-World Finnish Commuter Corridor Fallback (Helsinki HSL Coastal Line: Pasila -> Kauniainen)
FALLBACK_FINNISH_STATIONS = [
    {"code": "HKI", "name": "Helsinki", "lat": 60.1719, "lon": 24.9414},
    {"code": "PSL", "name": "Pasila", "lat": 60.1987, "lon": 24.9335},
    {"code": "HPL", "name": "Huopalahti", "lat": 60.2185, "lon": 24.8935},
    {"code": "LPV", "name": "Leppävaara", "lat": 60.2197, "lon": 24.8133},
    {"code": "KNI", "name": "Kauniainen", "lat": 60.2121, "lon": 24.7290},
]


def _http_get_json(url: str, headers: dict | None = None, timeout: float = 15.0, retries: int = 2):
    req_headers = dict(headers or {})
    if "Accept-Encoding" not in req_headers:
        req_headers["Accept-Encoding"] = "gzip"
    last_err: Exception | None = None
    for attempt in range(max(1, retries + 1)):
        try:
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                enc = (resp.headers.get("Content-Encoding") or "").lower()
                if "gzip" in enc or (len(raw) >= 2 and raw[:2] == b"\x1f\x8b"):
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(0.4 * (attempt + 1))
    assert last_err is not None
    raise last_err


def haversine_distance_and_bearing(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> tuple[float, float]:
    """Returns (geodesic_distance_m, initial_compass_bearing_deg in [0, 360))."""
    r_earth = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi * 0.5) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlam * 0.5) ** 2)
    dist_m = 2.0 * r_earth * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))

    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    bearing_deg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return float(dist_m), float(bearing_deg)


def _fetch_stations_map() -> dict[str, dict]:
    global _STATIONS_CACHE
    if _STATIONS_CACHE is not None:
        return _STATIONS_CACHE
    st_list = _http_get_json(
        "https://rata.digitraffic.fi/api/v1/metadata/stations",
        headers=DIGITRAFFIC_HEADERS,
        timeout=10.0,
    )
    out: dict[str, dict] = {}
    for s in st_list:
        out[s["stationShortCode"]] = {
            "code": s["stationShortCode"],
            "name": s["stationName"].replace(" asema", ""),
            "lat": float(s["latitude"]),
            "lon": float(s["longitude"]),
        }
    _STATIONS_CACHE = out
    return out


def _fetch_trains_and_comps_for_date(
    date_str: str, stations_map: dict[str, dict]
) -> tuple[dict[int, dict], dict[int, dict], list[int]]:
    if date_str in _TRAINS_DATE_CACHE:
        return (
            _TRAINS_DATE_CACHE[date_str],
            _COMPS_DATE_CACHE.get(date_str, {}),
            _CANDIDATES_DATE_CACHE.get(date_str, []),
        )

    tr_list = _http_get_json(
        f"https://rata.digitraffic.fi/api/v1/trains/{date_str}",
        headers=DIGITRAFFIC_HEADERS,
        timeout=25.0,
        retries=2,
    )
    trains_by_num: dict[int, dict] = {}
    commuter_cands: list[int] = []
    longdist_cands: list[int] = []

    for tr in tr_list:
        num = tr.get("trainNumber")
        if num is None:
            continue
        trains_by_num[int(num)] = tr
        if tr.get("cancelled"):
            continue
        cat = tr.get("trainCategory")
        if cat not in ("Commuter", "Long-distance"):
            continue
        stopping = [
            r for r in tr.get("timeTableRows", [])
            if r.get("trainStopping") and r.get("stationShortCode") in stations_map and not r.get("cancelled")
        ]
        # Need at least 3 distinct stopping stations (>= 2 legs)
        unique_st = {r["stationShortCode"] for r in stopping}
        if len(unique_st) < 3:
            continue
        if cat == "Commuter":
            commuter_cands.append(int(num))
        else:
            longdist_cands.append(int(num))

    comps_by_num: dict[int, dict] = {}
    try:
        comp_list = _http_get_json(
            f"https://rata.digitraffic.fi/api/v1/compositions/{date_str}",
            headers=DIGITRAFFIC_HEADERS,
            timeout=15.0,
        )
        if isinstance(comp_list, list):
            for c in comp_list:
                cnum = c.get("trainNumber")
                if cnum is not None:
                    comps_by_num[int(cnum)] = c
    except Exception:
        comps_by_num = {}

    commuter_cands.sort()
    longdist_cands.sort()
    candidates = commuter_cands + longdist_cands

    _TRAINS_DATE_CACHE[date_str] = trains_by_num
    _COMPS_DATE_CACHE[date_str] = comps_by_num
    _CANDIDATES_DATE_CACHE[date_str] = candidates
    return trains_by_num, comps_by_num, candidates


def _fetch_latest_gps_speeds() -> dict[int, float]:
    global _LOCATIONS_CACHE
    if _LOCATIONS_CACHE is not None:
        return _LOCATIONS_CACHE
    out: dict[int, float] = {}
    try:
        loc_list = _http_get_json(
            "https://rata.digitraffic.fi/api/v1/train-locations/latest",
            headers=DIGITRAFFIC_HEADERS,
            timeout=6.0,
        )
        if isinstance(loc_list, list):
            for entry in loc_list:
                tnum = entry.get("trainNumber")
                spd = entry.get("speed")
                if tnum is not None and spd is not None:
                    out[int(tnum)] = float(spd) / 3.6
    except Exception:
        pass
    _LOCATIONS_CACHE = out
    return out


def _resolve_train_and_date(
    date_str: str | None, train_number: int, ep_idx: int, stations_map: dict[str, dict]
) -> tuple[str, dict | None, dict | None]:
    """
    Resolves a valid Digitraffic departure date and real train entry.
    If `date_str` is omitted, prefers the most recent completed day (yesterday / -2d / -3d)
    that contains `train_number` so actual arrival/departure delays exist, falling back to today.
    For `ep_idx > 0`, selects distinct real Digitraffic trains across lines/corridors.
    """
    today = datetime.date.today()
    if date_str:
        candidate_dates = [date_str]
    else:
        candidate_dates = [
            (today - datetime.timedelta(days=1)).isoformat(),
            (today - datetime.timedelta(days=2)).isoformat(),
            (today - datetime.timedelta(days=3)).isoformat(),
            today.isoformat(),
        ]

    chosen_date = candidate_dates[0]
    trains_map: dict[int, dict] = {}
    comps_map: dict[int, dict] = {}
    candidates: list[int] = []

    for d in candidate_dates:
        try:
            t_m, c_m, cand = _fetch_trains_and_comps_for_date(d, stations_map)
        except Exception:
            continue
        if train_number in t_m and not t_m[train_number].get("cancelled"):
            chosen_date, trains_map, comps_map, candidates = d, t_m, c_m, cand
            break
        if not trains_map and cand:
            chosen_date, trains_map, comps_map, candidates = d, t_m, c_m, cand

    if not trains_map:
        return chosen_date, None, None

    # Build ordered diverse train selection starting with requested train_number if valid
    selected_num: int | None = None
    if ep_idx == 0 and train_number in trains_map and not trains_map[train_number].get("cancelled"):
        selected_num = train_number
    elif candidates:
        ordered = []
        if train_number in candidates:
            ordered.append(train_number)
        # Group by (category, commuterLineID) and round-robin across lines for corridor diversity
        buckets: dict[tuple[str, str], list[int]] = {}
        for cid in candidates:
            tr_c = trains_map[cid]
            key = (tr_c.get("trainCategory", ""), tr_c.get("commuterLineID", "") or "IC")
            buckets.setdefault(key, []).append(cid)
        bucket_keys = sorted(buckets.keys())
        max_b = max((len(v) for v in buckets.values()), default=0)
        for round_i in range(max_b):
            for bk in bucket_keys:
                b_list = buckets[bk]
                # Spread across morning/midday/evening departures within each line
                pos = (round_i * max(1, len(b_list) // 5)) % len(b_list)
                cand_id = b_list[pos]
                if cand_id not in ordered:
                    ordered.append(cand_id)
                if len(ordered) >= max(48, ep_idx + 2):
                    break
            if len(ordered) >= max(48, ep_idx + 2):
                break
        for cid in candidates:
            if len(ordered) >= max(48, ep_idx + 2):
                break
            if cid not in ordered:
                ordered.append(cid)
        selected_num = ordered[ep_idx % len(ordered)]

    if selected_num is None:
        return chosen_date, None, None

    tr_obj = trains_map.get(selected_num)
    comp_obj = comps_map.get(selected_num)
    if comp_obj is None:
        try:
            comp_cand = _http_get_json(
                f"https://rata.digitraffic.fi/api/v1/compositions/{chosen_date}/{selected_num}",
                headers=DIGITRAFFIC_HEADERS,
                timeout=5.0,
            )
            if isinstance(comp_cand, dict) and "journeySections" in comp_cand:
                comp_obj = comp_cand
        except Exception:
            comp_obj = None

    return chosen_date, tr_obj, comp_obj


def _estimate_composition_mass(comp_data: dict | None, train_category: str) -> tuple[float, float, str]:
    """
    Computes physical rolling-stock tare mass from Digitraffic composition (`locomotives`,
    `wagons`, `totalLength`, `maximumSpeed`) and returns:
      (normalized_tare_kg_in_plant_envelope, raw_real_tare_tonnes, composition_desc)
    """
    if not comp_data or not comp_data.get("journeySections"):
        raw_t = 122.0 if train_category == "Commuter" else 320.0
        return 72000.0, raw_t, f"Default-{train_category} (~{raw_t:.0f}t)", 75.0

    sec = comp_data["journeySections"][0]
    total_len_m = float(sec.get("totalLength") or 75.0)
    max_spd_kmh = int(sec.get("maximumSpeed") or 160)
    locos = sec.get("locomotives") or []
    wagons = sec.get("wagons") or []

    # Check if Digitraffic ever supplies totalMass directly; otherwise sum class tare weights
    if sec.get("totalMass"):
        raw_tonnes = float(sec["totalMass"])
        loco_label = locos[0].get("locomotiveType", "Train") if locos else "Train"
    else:
        loco_types = [l.get("locomotiveType", "") for l in locos if l.get("locomotiveType")]
        if loco_types and all(lt in ("Sm5", "Sm4", "Sm2", "Sm3", "Sm6", "Dm12") for lt in loco_types):
            # Coupled EMU/DMU trainsets: each entry in locomotives is one multi-car unit
            raw_tonnes = sum(ROLLING_STOCK_TARE_TONNES.get(lt, 95.0) for lt in loco_types)
            loco_label = f"{len(loco_types)}x{loco_types[0]}"
        elif loco_types:
            loco_t = sum(ROLLING_STOCK_TARE_TONNES.get(lt, 85.0) for lt in loco_types)
            wagon_t = 0.0
            for w in wagons:
                wt = w.get("wagonType", "")
                if wt.startswith("Ed"):
                    wagon_t += 55.0
                elif wt in ("Eil", "Eilf", "CEd", "CEmt", "Rx"):
                    wagon_t += 44.0
                else:
                    wagon_t += 46.0
            raw_tonnes = loco_t + wagon_t
            loco_label = f"{'+'.join(loco_types)}+{len(wagons)}w"
        else:
            raw_tonnes = max(60.0, 1.48 * total_len_m)
            loco_label = "Consist"

    # Map real fleet mass proportionally into our calibrated 4-car [71t, 76t] tare envelope
    # (so tare + [40..280] passengers stays cleanly inside the plant's [76t, 94t] envelope)
    norm_tare_kg = float(np.clip(70500.0 + (raw_tonnes / 500.0) * 5500.0, 70500.0, 76000.0))
    desc = f"{loco_label} ({total_len_m:.0f}m, Vmax={max_spd_kmh}km/h, RealTare={raw_tonnes:.0f}t)"
    return norm_tare_kg, float(raw_tonnes), desc, total_len_m


def fetch_open_meteo_corridor_weather(
    lat: float, lon: float, date_str: str, hour_utc: int = 12
) -> dict:
    """
    Queries Open-Meteo (free, no API key) for hourly wind speed, wind gusts, wind direction,
    temperature, rain, and snowfall at the rail leg's (lat, lon) midpoint.
    Caches the full 24h grid by (round(lat, 2), round(lon, 2), date_str).
    """
    today = datetime.date.today()
    try:
        req_date = datetime.date.fromisoformat(date_str)
    except ValueError:
        req_date = today

    cache_key = (round(float(lat), 2), round(float(lon), 2), req_date.isoformat())
    cached_w = _WEATHER_CACHE.get(cache_key)
    if cached_w is not None:
        hourly, source_label = cached_w
    else:
        hourly = None
        source_label = "Open-Meteo Live/Archive API"

    if hourly is None:
        days_ago = (today - req_date).days
        endpoints = (
            ["https://archive-api.open-meteo.com/v1/archive", "https://api.open-meteo.com/v1/forecast"]
            if days_ago > 5
            else ["https://api.open-meteo.com/v1/forecast", "https://archive-api.open-meteo.com/v1/archive"]
        )
        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start_date": req_date.isoformat(),
            "end_date": req_date.isoformat(),
            "hourly": "temperature_2m,precipitation,rain,snowfall,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        }
        qs = urllib.parse.urlencode(params)
        for base_url in endpoints:
            try:
                data = _http_get_json(f"{base_url}?{qs}", timeout=6.0)
                cand_h = data.get("hourly", {})
                if cand_h.get("temperature_2m") and any(x is not None for x in cand_h["temperature_2m"]):
                    hourly = cand_h
                    source_label = (
                        "Open-Meteo ERA5 Archive" if "archive-api" in base_url else "Open-Meteo Forecast API"
                    )
                    _WEATHER_CACHE[cache_key] = (hourly, source_label)
                    break
            except Exception:
                continue

    if hourly is not None:
        temps = hourly.get("temperature_2m", [])
        idx = min(max(0, int(hour_utc)), max(0, len(temps) - 1))

        def _val(key: str, default: float) -> float:
            arr = hourly.get(key, [])
            v = arr[idx] if idx < len(arr) else None
            return float(default if v is None else v)

        return {
            "temp_c": _val("temperature_2m", 6.0),
            "rain_mm": _val("rain", 0.0),
            "snow_cm": _val("snowfall", 0.0),
            "wind_speed_ms": _val("wind_speed_10m", 4.5),
            "wind_gust_ms": _val("wind_gusts_10m", 8.5),
            "wind_dir_deg": _val("wind_direction_10m", 245.0),
            "source": source_label,
        }

    return {
        "temp_c": 4.5,
        "rain_mm": 1.2,
        "snow_cm": 0.0,
        "wind_speed_ms": 6.8,
        "wind_gust_ms": 12.4,
        "wind_dir_deg": 255.0,
        "source": "Cached Open-Meteo Fallback",
    }


def _classify_raw_weather(weather_dict: dict) -> tuple[str, float, float]:
    """Maps meteorological observations to (weather_name, mu_adhesion, drag_multiplier)."""
    temp_c = float(weather_dict["temp_c"])
    rain_mm = float(weather_dict["rain_mm"])
    snow_cm = float(weather_dict["snow_cm"])
    if snow_cm > 0.05:
        mu = float(np.clip(0.82 - 0.03 * snow_cm, 0.68, 0.82))
        return "Snow-Slush", mu, 1.20
    if rain_mm > 0.10:
        mu = float(np.clip(0.88 - 0.015 * rain_mm, 0.76, 0.88))
        return "Rain-WetRail", mu, 1.15
    if temp_c < 0.0:
        return "Cold-Humid", 0.92, 1.12
    if temp_c >= 30.0:
        mu = float(np.clip(0.97 - 0.004 * (temp_c - 30.0), 0.91, 0.96))
        return "Heat-Thermal", mu, 1.05
    return "Windy-Dry", 1.00, 1.00


def apply_harsh_weather_stress_noise(
    weather_dict: dict,
    harsh_mode: str,
    leg_idx: int,
    ep_idx: int,
    rng: np.random.Generator,
) -> tuple[dict, str, float, float, bool]:
    """
    Injects physically calibrated harsh-weather perturbations (strong wind gusts, extreme high
    temperatures, extreme low temperatures, heavy rain, and heavy snowfall) on top of real
    Open-Meteo observations so post-training evaluation stress-tests all environmental extremes.

    Returns:
        (perturbed_weather_dict, weather_name, mu_adhesion, regime_drag_mult, is_harsh_injected)
    """
    raw_name, raw_mu, raw_drag = _classify_raw_weather(weather_dict)
    if harsh_mode in ("raw", "none"):
        return dict(weather_dict), raw_name, raw_mu, raw_drag, False

    if harsh_mode == "mixed" and (ep_idx % 2 == 0):
        # Balanced 50/50 split: keep even episodes as pure unperturbed Real Open-Meteo Baseline
        return dict(weather_dict), raw_name, raw_mu, raw_drag, False

    # Cycle across all 5 harsh-weather categories so even small eval suites cover all extremes
    regime_idx = (ep_idx * 3 + leg_idx) % len(HARSH_REGIMES)
    regime = HARSH_REGIMES[regime_idx]
    w = dict(weather_dict)

    if regime == "Storm-Gale":
        w["wind_speed_ms"] = float(np.clip(w["wind_speed_ms"] + rng.uniform(14.0, 18.0), 20.0, 24.0))
        w["wind_gust_ms"] = float(np.clip(max(w["wind_speed_ms"] + 4.0, w["wind_gust_ms"] + rng.uniform(14.0, 20.0)), 26.0, 30.0))
        w["wind_dir_deg"] = float((w["wind_dir_deg"] + rng.uniform(-25.0, 25.0)) % 360.0)
        w["temp_c"] = float(np.clip(float(w["temp_c"]), 5.0, 15.0))
        w["source"] = f"{weather_dict['source']} + Extreme[Storm-Gale <=30m/s]"
        mu_adhesion = float(np.clip(raw_mu * rng.uniform(0.88, 0.94), 0.86, 0.92))
        return w, "Storm-Gale", mu_adhesion, 1.10, True

    if regime == "Heat-Thermal":
        # Extreme summer heatwave: ambient 35..42 C -> CWR rail temp ~52..60 C, oxide-softened dry adhesion
        w["temp_c"] = float(np.clip(max(float(w["temp_c"]), 6.0) + rng.uniform(29.0, 36.0), 35.0, 42.0))
        w["rain_mm"] = 0.0
        w["snow_cm"] = 0.0
        w["wind_speed_ms"] = float(np.clip(w["wind_speed_ms"] + rng.uniform(3.0, 6.0), 6.0, 11.0))
        w["wind_gust_ms"] = float(np.clip(max(w["wind_speed_ms"] + 4.0, w["wind_gust_ms"] + rng.uniform(5.0, 9.0)), 10.0, 17.0))
        w["source"] = f"{weather_dict['source']} + Extreme[Heat-Thermal 35..42C]"
        mu_adhesion = float(rng.uniform(0.89, 0.94))
        return w, "Heat-Thermal", mu_adhesion, 1.05, True

    if regime == "Arctic-Cold":
        # Severe winter cold snap: -22..-12 C -> +9..+17% air density + frost/condensation adhesion loss
        w["temp_c"] = float(np.clip(min(float(w["temp_c"]), 2.0) - rng.uniform(14.0, 22.0), -22.0, -12.0))
        w["rain_mm"] = 0.0
        w["wind_speed_ms"] = float(np.clip(w["wind_speed_ms"] + rng.uniform(4.0, 8.0), 7.0, 13.0))
        w["wind_gust_ms"] = float(np.clip(max(w["wind_speed_ms"] + 5.0, w["wind_gust_ms"] + rng.uniform(6.0, 11.0)), 12.0, 19.0))
        w["source"] = f"{weather_dict['source']} + Extreme[Arctic-Cold -22..-12C]"
        mu_adhesion = float(rng.uniform(0.82, 0.88))
        return w, "Cold-Humid", mu_adhesion, 1.10, True

    if regime == "Heavy-Rain":
        w["rain_mm"] = float(np.clip(w["rain_mm"] + rng.uniform(18.0, 32.0), 18.0, 35.0))
        w["snow_cm"] = 0.0
        w["temp_c"] = float(np.clip(max(float(w["temp_c"]), 5.0), 5.0, 16.0))
        w["wind_speed_ms"] = float(np.clip(w["wind_speed_ms"] + rng.uniform(5.0, 9.0), 8.0, 14.0))
        w["wind_gust_ms"] = float(np.clip(max(w["wind_speed_ms"] + 5.0, w["wind_gust_ms"] + rng.uniform(7.0, 12.0)), 14.0, 21.0))
        w["source"] = f"{weather_dict['source']} + Extreme[Heavy-Rain 18..35mm/h]"
        mu_adhesion = float(np.clip(0.85 - 0.0025 * w["rain_mm"], 0.76, 0.84))
        return w, "Rain-WetRail", mu_adhesion, 1.12, True

    # Blizzard-Snow
    w["snow_cm"] = float(np.clip(w["snow_cm"] + rng.uniform(5.0, 11.0), 5.0, 12.0))
    w["rain_mm"] = 0.0
    w["temp_c"] = float(np.clip(min(float(w["temp_c"]), 0.0) - rng.uniform(2.0, 10.0), -12.0, -1.0))
    w["wind_speed_ms"] = float(np.clip(w["wind_speed_ms"] + rng.uniform(6.0, 10.0), 9.0, 15.0))
    w["wind_gust_ms"] = float(np.clip(max(w["wind_speed_ms"] + 6.0, w["wind_gust_ms"] + rng.uniform(8.0, 14.0)), 15.0, 23.0))
    w["source"] = f"{weather_dict['source']} + Extreme[Blizzard-Snow 5..12cm/h]"
    mu_adhesion = float(np.clip(0.81 - 0.007 * w["snow_cm"], 0.72, 0.80))
    return w, "Snow-Slush", mu_adhesion, 1.14, True


def build_directional_wind_acceleration_profile(
    total_duration_s: float,
    dt: float,
    track_bearing_deg: float,
    weather_dict: dict,
    seed: int = 42,
    regime_drag_mult: float = 1.0,
    is_harsh: bool = False,
    car_scale: float = 1.0,
) -> tuple[np.ndarray, str, list[tuple[float, float, float]], np.ndarray]:
    """
    Builds the 0.5 s longitudinal wind disturbance profile (m/s^2) from Open-Meteo wind *speeds*
    (sustained <= 24 m/s, gusts <= 30 m/s [108 km/h]) and direction relative to the track bearing,
    through the plant-consistent Davis-drag wind model `wind_speed_to_longitudinal_accel`:
      1) Sustained head/tail component + gust fronts whose wind speed reaches the reported gust value
         (alternating head/tail frontal shifts);
      2) Band-limited Ornstein-Uhlenbeck turbulence with 12% (harsh) / 8% turbulence intensity.
    Both act from the first step, including the station departure; the drag reference speed blends from
    v_ref = 0 (platform, first 4 s) to v_ref = 15 m/s (open line) over 6 s.
    Air density rho(T)/rho_0 and the precipitation drag multiplier scale the aerodynamic force.
    Also returns the per-step anemometer noise std (m/s^2) from the sensor spec (anemometer_sigma_series).
    """
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(total_duration_s / dt)) + 40
    t_grid = np.arange(n_steps, dtype=np.float64) * dt

    v_wind = min(24.0, float(weather_dict["wind_speed_ms"]))
    v_gust = min(MAX_WIND_SPEED_MPS, max(v_wind, float(weather_dict["wind_gust_ms"])))
    w_dir = float(weather_dict["wind_dir_deg"])
    temp_c = float(weather_dict["temp_c"])
    rho_ratio = 293.15 / max(250.0, 273.15 + temp_c)

    # Meteorological wind_dir is where wind blows FROM: cos(w_dir - track_bearing) > 0 => headwind.
    angle_rad = math.radians(w_dir - track_bearing_deg)
    head_proj = math.cos(angle_rad)
    cross_proj = abs(math.sin(angle_rad))
    w_head_sus = v_wind * head_proj
    w_cross = v_wind * cross_proj

    w_head = np.full(n_steps, w_head_sus, dtype=np.float64)

    # Gust fronts: the longitudinal wind component swings to +-v_gust (frontal shift alternates sign).
    sign_primary = 1.0 if head_proj >= -0.15 else -1.0
    n_gusts = 3 if is_harsh else 2
    for f_i in range(n_gusts):
        frac_lo = 0.14 + (0.24 if is_harsh else 0.30) * f_i
        frac_hi = min(0.86, frac_lo + (0.15 if is_harsh else 0.18))
        center_s = float(rng.uniform(frac_lo, frac_hi) * total_duration_s)
        width_s = float(rng.uniform(16.0, 26.0) if is_harsh else rng.uniform(20.0, 32.0))
        sign = sign_primary if (f_i % 2 == 0) else -sign_primary
        target = sign * v_gust * float(rng.uniform(0.88, 1.0))
        w_head += (target - w_head_sus) * np.exp(-0.5 * ((t_grid - center_s) / width_s) ** 2)

    ti = 0.12 if is_harsh else 0.08
    phi_ou = math.exp(-dt / (4.8 if is_harsh else 5.8))
    alpha_lp = dt / ((3.2 if is_harsh else 4.2) + dt)
    ou_stage1 = 0.0
    ou_smooth = np.zeros(n_steps, dtype=np.float64)
    for k in range(8, n_steps):
        ou_stage1 = phi_ou * ou_stage1 + ti * v_gust * math.sqrt(1.0 - phi_ou * phi_ou) * float(rng.normal())
        ou_smooth[k] = (1.0 - alpha_lp) * ou_smooth[k - 1] + alpha_lp * ou_stage1
    # Turbulence also acts during the station departure (same draws, warm-up dropped):
    ou_smooth = _drop_ou_warmup(ou_smooth, 8)
    w_head = np.clip(w_head + ou_smooth, -MAX_WIND_SPEED_MPS, MAX_WIND_SPEED_MPS)

    w_profile = wind_speed_series_to_accel_profile(
        w_head, w_head_sus, w_cross=w_cross, dt=dt, rho_ratio=rho_ratio, drag_mult=regime_drag_mult,
        car_scale=car_scale,
    )
    event_meta, _ = _extract_wind_events(w_profile, dt, thresh=0.06)

    desc = (
        f"{weather_dict['source']}: Wind={v_wind:.1f}m/s (Gust={v_gust:.1f}m/s @ {w_dir:.0f}deg vs "
        f"Track={track_bearing_deg:.0f}deg), T={temp_c:.1f}C, "
        f"Rain={weather_dict['rain_mm']:.1f}mm/h, Snow={weather_dict['snow_cm']:.1f}cm/h, "
        f"d in [{w_profile.min():+.2f},{w_profile.max():+.2f}]m/s^2"
    )
    anem_sig = anemometer_sigma_series(w_head, w_cross=w_cross, dt=dt, rho_ratio=rho_ratio,
                                       drag_mult=regime_drag_mult, car_scale=car_scale)
    return w_profile, desc, event_meta, anem_sig


def _extract_corridor_hops(
    rows: list[dict], stations_map: dict[str, dict], n_legs_max: int
) -> list[dict]:
    """
    Extracts station-to-station stops from Digitraffic `timeTableRows` and coalesces
    short suburban halts (< 2,800 m geodesic) into continuous multi-km express segments
    so leg geometry and scheduled running times remain physically faithful.
    """
    # Pair departures and arrivals at stopping stations
    stops: list[dict] = []
    for r in rows:
        if not r.get("trainStopping") or r.get("cancelled"):
            continue
        scode = r.get("stationShortCode")
        if scode not in stations_map:
            continue
        t_sched_s = r.get("scheduledTime")
        if not t_sched_s:
            continue
        t_act_s = r.get("actualTime") or r.get("liveEstimateTime") or t_sched_s
        dt_sched = datetime.datetime.fromisoformat(t_sched_s.replace("Z", "+00:00"))
        dt_act = datetime.datetime.fromisoformat(t_act_s.replace("Z", "+00:00"))
        ts_sched = dt_sched.timestamp()
        ts_act = dt_act.timestamp()

        if stops and stops[-1]["code"] == scode:
            if r.get("type") == "DEPARTURE":
                stops[-1]["dep_sched"] = ts_sched
                stops[-1]["dep_act"] = ts_act
                stops[-1]["hour_utc"] = dt_sched.hour
            elif r.get("type") == "ARRIVAL":
                stops[-1]["arr_sched"] = ts_sched
                stops[-1]["arr_act"] = ts_act
        else:
            stops.append({
                **stations_map[scode],
                "arr_sched": ts_sched,
                "arr_act": ts_act,
                "dep_sched": ts_sched,
                "dep_act": ts_act,
                "hour_utc": dt_sched.hour,
            })

    if len(stops) < 2:
        return []

    # Build contiguous segments, coalescing short (< 2.8 km) urban halts when enough stops remain
    hops: list[dict] = []
    i = 0
    while i < len(stops) - 1 and len(hops) < n_legs_max:
        s_start = stops[i]
        j = i + 1
        while j < len(stops) - 1:
            d_ij, _ = haversine_distance_and_bearing(
                s_start["lat"], s_start["lon"], stops[j]["lat"], stops[j]["lon"]
            )
            # Only coalesce if remaining stops still allow forming subsequent legs
            rem_stops = len(stops) - 1 - j
            if d_ij < 2800.0 and rem_stops >= max(1, (n_legs_max - len(hops) - 1)):
                j += 1
            else:
                break
        s_end = stops[j]
        geo_dist_m, bearing_deg = haversine_distance_and_bearing(
            s_start["lat"], s_start["lon"], s_end["lat"], s_end["lon"]
        )
        hops.append({
            "start": s_start,
            "end": s_end,
            "geo_dist_m": geo_dist_m,
            "bearing_deg": bearing_deg,
            "raw_sched_dur_s": max(60.0, s_end["arr_sched"] - s_start["dep_sched"]),
            "empirical_delay_s": float(s_end["arr_act"] - s_end["arr_sched"]),
            "hour_utc": int(s_start["hour_utc"]),
        })
        i = j
    return hops


def fetch_real_world_multi_station_scenario(
    date_str: str | None = None,
    train_number: int = 8311,
    n_legs_max: int = 4,
    seed: int = 42,
    ep_idx: int = 0,
    harsh_weather_mode: str = "mixed",
) -> dict:
    """
    Fetches a real multi-station railway journey from Finland's Digitraffic API and
    merges it with Open-Meteo hourly weather along each leg midpoint plus calibrated
    harsh-weather stress noise (strong wind, high/low temperatures, strong rain/snow).
    """
    rng = np.random.default_rng(seed)
    n_legs_target = max(1, min(5, int(n_legs_max) if n_legs_max and n_legs_max > 0 else int(rng.integers(1, 6))))

    stations_map: dict[str, dict] = {}
    hops: list[dict] = []
    actual_train_num = int(train_number)
    train_cat = "Commuter"
    commuter_line = ""
    comp_obj: dict | None = None
    data_source = "Digitraffic + Open-Meteo"
    used_date = date_str or (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    latest_gps_speed_mps: float | None = None

    try:
        stations_map = _fetch_stations_map()
        used_date, tr_obj, comp_obj = _resolve_train_and_date(
            date_str if date_str else None, int(train_number), ep_idx, stations_map
        )
        if tr_obj is not None:
            actual_train_num = int(tr_obj.get("trainNumber", train_number))
            train_cat = tr_obj.get("trainCategory", "Commuter")
            commuter_line = tr_obj.get("commuterLineID", "") or ""
            hops = _extract_corridor_hops(tr_obj.get("timeTableRows", []), stations_map, n_legs_target)
            gps_map = _fetch_latest_gps_speeds()
            latest_gps_speed_mps = gps_map.get(actual_train_num)
    except Exception:
        hops = []
        data_source = "Offline Fallback Snapshot"

    if len(hops) < 1:
        data_source = "Offline Fallback Snapshot"
        hops = []
        t0 = 0.0
        n_fb = min(n_legs_target, len(FALLBACK_FINNISH_STATIONS) - 1)
        for idx in range(n_fb):
            s_a = FALLBACK_FINNISH_STATIONS[idx]
            s_b = FALLBACK_FINNISH_STATIONS[idx + 1]
            geo_d, brg = haversine_distance_and_bearing(s_a["lat"], s_a["lon"], s_b["lat"], s_b["lon"])
            hops.append({
                "start": s_a,
                "end": s_b,
                "geo_dist_m": geo_d,
                "bearing_deg": brg,
                "raw_sched_dur_s": max(210.0, geo_d * 1.12 / 14.5 + 32.0),
                "empirical_delay_s": float(rng.uniform(-2.0, 6.0)),
                "hour_utc": 13,
            })
            t0 += 360.0

    norm_tare_kg, real_tare_tonnes, comp_desc, real_len_m = _estimate_composition_mass(comp_obj, train_cat)
    # Train size: --train-cars N forces N cars; --train-cars 0 takes the real composition length.
    if _rc.TRAIN_CARS is not None and _rc.TRAIN_CARS <= 0:
        n_cars_eff, train_len_m, length_fx = train_size_config(max(1.0, real_len_m / CAR_LENGTH_M))
    else:
        n_cars_eff, train_len_m, length_fx = train_size_config()
    s_car = n_cars_eff / 4.0
    norm_tare_kg *= s_car

    # Peak-hour commuter loading prior based on departure hour
    first_hour = hops[0]["hour_utc"]
    is_rush = (5 <= first_hour <= 8) or (13 <= first_hour <= 16)
    curr_passengers = int(rng.integers(150, 245) if is_rush else rng.integers(85, 170))

    t_clock = 0.0
    cum_x_m = 0.0
    legs = []
    any_harsh_injected = False

    for leg_idx, hop in enumerate(hops):
        s_a = hop["start"]
        s_b = hop["end"]
        geo_dist_m = hop["geo_dist_m"]
        bearing_deg = hop["bearing_deg"]

        # Apply rail alignment curvature factor (1.08x) and keep inside calibrated [3800m, 6400m] envelope
        curved_dist_m = geo_dist_m * 1.08
        leg_len_m = float(np.round(np.clip(curved_dist_m, 3800.0, 6400.0) / 50.0) * 50.0)
        zones, min_feas_s = _build_leg_zones(leg_len_m, rng)

        lat_mid = 0.5 * (s_a["lat"] + s_b["lat"])
        lon_mid = 0.5 * (s_a["lon"] + s_b["lon"])
        raw_weather = fetch_open_meteo_corridor_weather(
            lat_mid, lon_mid, date_str=used_date, hour_utc=hop["hour_utc"]
        )
        weather_eff, weather_name, mu_adhesion, drag_mult, is_harsh_leg = apply_harsh_weather_stress_noise(
            raw_weather,
            harsh_mode=harsh_weather_mode,
            leg_idx=leg_idx,
            ep_idx=ep_idx,
            rng=rng,
        )
        any_harsh_injected = any_harsh_injected or is_harsh_leg

        # Passenger boarding/alighting across intermediate stations
        if leg_idx == 0:
            n_leave, n_join = 0, curr_passengers
        else:
            n_leave = int(rng.integers(20, max(25, min(95, curr_passengers - 15))))
            n_join = int(rng.integers(35, 135 if is_rush else 95))
            curr_passengers = int(np.clip(curr_passengers - n_leave + n_join, 35, 360))

        missed_pax = int(rng.binomial(curr_passengers, 0.11 if is_harsh_leg else 0.06))
        untracked_pax = int(rng.poisson(24.0 if is_harsh_leg else 13.0))
        curr_tickets = int(np.clip(curr_passengers + missed_pax - untracked_pax, 30, 360))

        avg_pax_weight_kg = float(rng.normal(77.5, 5.5 if is_harsh_leg else 4.0))
        pax_phys = int(round(curr_passengers * s_car))
        tickets_phys = int(round(curr_tickets * s_car))
        leg_mass_kg = float(np.clip(norm_tare_kg + pax_phys * avg_pax_weight_kg, 76000.0 * s_car, 94000.0 * s_car))
        ticket_est_mass_kg = float(np.clip(norm_tare_kg + tickets_phys * 75.0, 74500.0 * s_car, 96000.0 * s_car))
        if length_fx:
            zones = apply_tail_clearance(zones, train_len_m)

        # Generate directional wind + turbulence profile
        raw_scaled_sched = hop["raw_sched_dur_s"] * (leg_len_m / max(500.0, curved_dist_m))
        mass_adder = max(0.0, (leg_mass_kg / s_car - 85000.0) / 2200.0) + tail_clearance_time_s(zones, train_len_m)
        adhesion_adder = max(0.0, (0.90 - mu_adhesion) * 38.0)

        w_prof, w_desc, w_meta, anem_sig = build_directional_wind_acceleration_profile(
            max(min_feas_s + 240.0, raw_scaled_sched + 200.0),
            dt=0.50,
            track_bearing_deg=bearing_deg,
            weather_dict=weather_eff,
            seed=seed * 31 + leg_idx,
            regime_drag_mult=drag_mult,
            is_harsh=is_harsh_leg,
            car_scale=s_car,
        )
        mean_headwind = max(0.0, -float(np.mean(w_prof[: int(min_feas_s * 2.0)])))
        headwind_adder = mean_headwind * 22.0
        short_leg_adder = max(0.0, (5000.0 - leg_len_m) / 140.0)
        eff_min_feas_s = min_feas_s + short_leg_adder + mass_adder + adhesion_adder + headwind_adder
        if is_harsh_leg:
            # Extreme scenarios face genuine schedule pressure while remaining physically solvable
            slack_lo, slack_hi = 15.0, 28.0
        else:
            slack_lo, slack_hi = 18.0, 36.0

        if eff_min_feas_s + 15.0 <= raw_scaled_sched <= eff_min_feas_s + 50.0:
            leg_budget_s = float(np.round(raw_scaled_sched * 2.0) / 2.0)
        else:
            leg_budget_s = float(np.round((eff_min_feas_s + rng.uniform(slack_lo, slack_hi)) * 2.0) / 2.0)

        leg_deadline_s = t_clock + leg_budget_s
        dwell_s = 25.0 if leg_idx < len(hops) - 1 else 0.0
        # Station anemometer reading at departure (k = 0); same draw as before, scaled to the sensor spec:
        station_wind_reading = float(w_prof[0] + anem_sig[0] * rng.normal(0.0, 1.0))
        speed_noise_std = 0.0025 if is_harsh_leg else 0.0
        unmodeled_drag = float(rng.uniform(1.03, 1.08)) if is_harsh_leg else 1.0
        temp_c = float(weather_eff["temp_c"])

        line_tag = f"Line {commuter_line} " if commuter_line else f"{train_cat} "
        corridor = RailroadCorridor(
            route_name=(
                f"Digitraffic {line_tag}#{actual_train_num} Leg {leg_idx + 1} "
                f"({s_a['name']} -> {s_b['name']}, Brg={bearing_deg:.0f}deg)"
            ),
            station_x_m=leg_len_m,
            start_time_s=t_clock,
            deadline_s=leg_deadline_s,
            mass_kg=leg_mass_kg,
            ticket_estimated_mass_kg=ticket_est_mass_kg,
            passengers=pax_phys,
            sold_tickets=tickets_phys,
            max_capacity=int(round(360 * s_car)),
            weather_condition=weather_name,
            adhesion_mu_factor=mu_adhesion,
            temp_celsius=temp_c,
            station_wind_sensor_mps2=station_wind_reading,
            is_infeasible_challenge=False,
            speed_sensor_noise_mps=speed_noise_std,
            unmodeled_drag_factor=unmodeled_drag,
            sensor_noise_seed=int(seed * 101 + leg_idx * 17),
            anemometer_sigma_mps2=anem_sig,
            is_extreme_scenario=bool(is_harsh_leg),
            speed_zones=zones,
            n_cars=n_cars_eff,
            train_length_m=train_len_m,
        )

        cum_x_m += leg_len_m
        legs.append({
            "leg_idx": leg_idx + 1,
            "station_name": s_b["name"],
            "corridor": corridor,
            "leg_length_m": leg_len_m,
            "real_geo_dist_m": geo_dist_m,
            "cum_station_x_m": cum_x_m,
            "t_start": t_clock,
            "deadline_s": leg_deadline_s,
            "budget_s": leg_budget_s,
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
            "is_infeasible": False,
            "wind_profile": w_prof,
            "wind_desc": w_desc,
            "event_meta": w_meta,
            "real_weather_raw": raw_weather,
            "weather_effective": weather_eff,
            "is_harsh_injected": is_harsh_leg,
            "speed_sensor_noise_mps": speed_noise_std,
            "unmodeled_drag_factor": unmodeled_drag,
            "real_empirical_delay_s": hop["empirical_delay_s"],
            "track_bearing_deg": bearing_deg,
            "lat_a": float(s_a["lat"]), "lon_a": float(s_a["lon"]),
            "lat_b": float(s_b["lat"]), "lon_b": float(s_b["lon"]),
            "hour_utc": int(hop.get("hour_utc", 12)), "date_str": used_date,
            "latest_gps_speed_mps": latest_gps_speed_mps,
        })
        t_clock = leg_deadline_s + dwell_s

    return {
        "scenario_id": seed,
        "train_number": actual_train_num,
        "train_category": train_cat,
        "commuter_line": commuter_line,
        "composition_desc": comp_desc,
        "real_tare_tonnes": real_tare_tonnes,
        "date_str": used_date,
        "data_source": data_source,
        "n_legs": len(legs),
        "total_length_m": cum_x_m,
        "total_scheduled_time_s": legs[-1]["deadline_s"] - legs[0]["t_start"],
        "has_infeasible_leg": False,
        "is_weather_disturbed": True,
        "is_harsh_injected": any_harsh_injected,
        "is_extreme_scenario": any_harsh_injected,
        "legs": legs,
    }
