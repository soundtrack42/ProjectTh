"""
Slow-weather update study: station-measured weather vs. weather fetched online along the route.

The plant gets a true slow-weather profile (temperature, rail adhesion) along every leg, shared by all policies:
  * along-route: real legs follow the Open-Meteo hourly data interpolated in time and between the departure and
    arrival stations (synthetic legs have no along-route data, so their weather stays constant);
  * front: additionally, a weather front passes during the leg (temperature -4..-10 degC and adhesion x0.75..0.88
    over 120 s, starting at 20-70% of the scheduled run time).
The controller knows the slow weather through one policy (dynamic_railroad_mpc.WX_POLICY):
  station  (measured at the departure station, held), online (station + change reported by the weather service,
  requests every 60 s, 20% fail -> keep last value), online-nowcast (same, without the 15-min slot), oracle.

    python weather_update_study.py fetch-real OUT.pkl         # build and cache the 30 real journeys (network)
    python weather_update_study.py run SUITE COND POLICY METHOD OUT.json [REAL.pkl]
"""
from __future__ import annotations

import copy
import json
import os
import pickle
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
import dynamic_railroad_mpc as D  # noqa: E402
import real_world_rail_weather as RW  # noqa: E402
from railroad_corridor import generate_multi_station_scenario  # noqa: E402

N_JOURNEYS = 30
FRONT_RAMP_S = 120.0

POLICIES = {
    "station": dict(WX_POLICY="station"),
    "online": dict(WX_POLICY="online", WX_API_SLOT_S=900.0, WX_FETCH_FAIL_P=0.2),
    "online_nowcast": dict(WX_POLICY="online", WX_API_SLOT_S=0.0, WX_FETCH_FAIL_P=0.2),
    "online_fail50": dict(WX_POLICY="online", WX_API_SLOT_S=0.0, WX_FETCH_FAIL_P=0.5),
    "oracle": dict(WX_POLICY="oracle"),
}
METHODS = {"m2": ("RL_LEARNED", None), "m5": ("RL_LEARNED", "mps_rollout"),
           "m6": ("RL_FREE_LEARNED", None), "m7": ("RL_FREE_LEARNED", "mps_rollout")}


def _hourly(lat: float, lon: float, date_str: str) -> dict | None:
    RW.fetch_open_meteo_corridor_weather(lat, lon, date_str=date_str, hour_utc=12)
    c = RW._WEATHER_CACHE.get((round(float(lat), 2), round(float(lon), 2), date_str))
    return None if c is None else c[0]


def _wx_at(h: dict | None, hour_f: float) -> dict | None:
    """Hourly Open-Meteo series linearly interpolated at fractional hour hour_f."""
    if h is None:
        return None
    out = {}
    for key, name, dflt in (("temperature_2m", "temp_c", 6.0), ("rain", "rain_mm", 0.0), ("snowfall", "snow_cm", 0.0)):
        arr = [dflt if v is None else float(v) for v in h.get(key, [])] or [dflt]
        i0 = int(np.clip(np.floor(hour_f), 0, len(arr) - 1))
        i1 = min(i0 + 1, len(arr) - 1)
        f = float(np.clip(hour_f - i0, 0.0, 1.0))
        out[name] = (1 - f) * arr[i0] + f * arr[i1]
    return out


def along_route_offsets(leg: dict, n: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Temperature offset (degC) and adhesion ratio along a real leg, relative to the departure station."""
    if "lat_a" not in leg:
        return np.zeros(n), np.ones(n)
    ha = _hourly(leg["lat_a"], leg["lon_a"], leg["date_str"])
    hb = _hourly(leg["lat_b"], leg["lon_b"], leg["date_str"])
    if ha is None or hb is None:
        return np.zeros(n), np.ones(n)
    t_sched = max(dt, float(leg["deadline_s"] - leg["t_start"]))
    T = np.empty(n)
    mu = np.empty(n)
    for k in range(n):
        t = k * dt
        hour_f = leg["hour_utc"] + t / 3600.0  # departure hour of this hop (minutes not in the timetable extract)
        f = min(1.0, t / t_sched)
        wa, wb = _wx_at(ha, hour_f), _wx_at(hb, hour_f)
        w = {key: (1 - f) * wa[key] + f * wb[key] for key in wa}
        T[k] = w["temp_c"]
        mu[k] = RW._classify_raw_weather(w)[1]
    return T - T[0], mu / max(1e-6, mu[0])


def attach_profiles(scen: dict, cond: str, seed: int) -> None:
    rng = np.random.default_rng((seed, 991))
    for leg in scen["legs"]:
        cor = leg["corridor"]
        n, dt = len(leg["wind_profile"]), cor.dt
        dT, rmu = along_route_offsets(leg, n, dt)
        T = cor.temp_celsius + dT
        mu = cor.adhesion_mu_factor * rmu
        if cond == "front":
            t_sched = float(leg["deadline_s"] - leg["t_start"])
            t_f = rng.uniform(0.2, 0.7) * t_sched
            s = np.clip((np.arange(n) * dt - t_f) / FRONT_RAMP_S, 0.0, 1.0)
            T = T - rng.uniform(4.0, 10.0) * s
            mu = mu * (1.0 - (1.0 - rng.uniform(0.75, 0.88)) * s)
        cor.slow_weather_profile = {"temp_c": T, "mu": np.clip(mu, 0.62, 1.0)}
        cor.wx_clock_offset_s = float(np.random.default_rng((seed, 992)).uniform(0.0, 900.0))


def scenarios(suite: str, real_pkl: str | None) -> list[dict]:
    if suite == "real":
        return pickle.load(open(real_pkl, "rb"))
    return [generate_multi_station_scenario(seed=5000 + 29 * ep, n_legs=0, wind_mode="mixed",
                                            allow_infeasible_breaks=(ep > 0), extreme_noise=(ep % 2 == 1))
            for ep in range(N_JOURNEYS)]


def fetch_real(out: str) -> None:
    scens = [RW.fetch_real_world_multi_station_scenario(date_str=None, train_number=8311, n_legs_max=0,
                                                         seed=9000 + ep * 29, ep_idx=ep, harsh_weather_mode="mixed")
             for ep in range(N_JOURNEYS)]
    for s in scens:  # warm the hourly cache for both stations of every leg (network only here)
        for leg in s["legs"]:
            if "lat_a" in leg:
                _hourly(leg["lat_a"], leg["lon_a"], leg["date_str"])
                _hourly(leg["lat_b"], leg["lon_b"], leg["date_str"])
    pickle.dump(scens, open(out, "wb"))
    pickle.dump(dict(RW._WEATHER_CACHE), open(out + ".wxcache", "wb"))
    print("saved", len(scens), "journeys,", sum(len(s["legs"]) for s in scens), "legs;", scens[0]["date_str"])


def run(suite: str, cond: str, policy: str, method: str, out: str, real_pkl: str | None) -> None:
    if real_pkl and os.path.exists(real_pkl + ".wxcache"):
        RW._WEATHER_CACHE.update(pickle.load(open(real_pkl + ".wxcache", "rb")))
    for k, v in POLICIES[policy].items():
        setattr(D, k, v)
    mode, shield = METHODS[method]
    emu = D.LearnedLinearizationModel(hidden_dim=64)
    emu.load_state_dict(torch.load("checkpoints/best_emulator.pt", map_location="cpu"))
    emu.eval()
    pol = D.CosineTrajectoryRLPolicy(obs_dim=18, act_dim=3, hidden_dim=64)
    ck = "best_ppo_free_pilot.pt" if mode == "RL_FREE_LEARNED" else "best_ppo_cosine_pilot.pt"
    pol.load_state_dict(torch.load(f"checkpoints/{ck}", map_location="cpu"))
    pol.eval()
    rows = []
    for ep, sc in enumerate(scenarios(suite, real_pkl)):
        sc = copy.deepcopy(sc)
        attach_profiles(sc, cond, seed=ep + (0 if suite == "syn" else 100000))
        r = D.evaluate_multi_station_journey(method, sc, mode, rl_policy=pol, shield_mode=shield, emulator=emu)
        prof = [l["corridor"].slow_weather_profile for l in sc["legs"]]
        rows.append(dict(loss=r["multi_obj_score"], T=r["abs_sched_err_s"], stop=r["abs_stop_err_m"],
                         smooth=r["step_smooth_cost"], E=r["spec_energy_wh_tkm"], shield=r["shield_rate_pct"],
                         max_us=r["max_total_us"],
                         dT_max=float(max(np.ptp(p["temp_c"]) for p in prof)),
                         dmu_max=float(max(np.ptp(p["mu"]) for p in prof))))
    json.dump(rows, open(out, "w"))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if sys.argv[1] == "fetch-real":
        fetch_real(sys.argv[2])
    else:
        run(*sys.argv[2:7], sys.argv[7] if len(sys.argv) > 7 else None)
