"""Select the historic extreme-weather suite: 3 real Digitraffic journeys per regime (storm, rain, snow, cold, heat),
each on a different date, chosen ONLY by the measured weather on their legs (no controller is run).
Usage: historic_suite_select.py daily.json out.json  (daily.json: 2016-2026 Open-Meteo daily archive at the 12 HUBXY hubs)"""
import json, sys, time, math, urllib.request, urllib.parse, datetime
import numpy as np
SRC = str(__import__("pathlib").Path(__file__).resolve().parent)
sys.path[:0] = [SRC, SRC + "/.."]
import real_world_rail_weather as RW

daily = json.load(open(sys.argv[1])); hubs = list(daily); T = daily[hubs[0]]["time"]
HUBXY = {"Helsinki":(60.17,24.94),"Tampere":(61.50,23.77),"Turku":(60.45,22.25),"Oulu":(65.01,25.47),"Kouvola":(60.87,26.70),
         "Seinajoki":(62.79,22.84),"Kuopio":(62.90,27.68),"Joensuu":(62.60,29.76),"Lahti":(60.98,25.66),"Jyvaskyla":(62.24,25.75),
         "Vaasa":(63.10,21.62),"Rovaniemi":(66.50,25.73)}
# regime: (daily key, sign, leg metric key, leg-metric sign, threshold on the worst leg)
REG = {"Storm":   ("wind_gusts_10m_max", +1, "wind_gust_ms", +1, 20.0),
       "Rain":    ("rain_sum", +1, "rain_mm", +1, 2.0),
       "Snow":    ("snowfall_sum", +1, "snow_cm", +1, 0.7),
       "Cold":    ("temperature_2m_min", -1, "temp_c", -1, -20.0),
       "Heat":    ("temperature_2m_max", +1, "temp_c", +1, 27.0)}
H_KEY = {"wind_gust_ms": "wind_gusts_10m", "rain_mm": "rain", "snow_cm": "snowfall", "temp_c": "temperature_2m"}

def hub_hourly(hub, date):
    la, lo = HUBXY[hub]
    q = urllib.parse.urlencode(dict(latitude=la, longitude=lo, start_date=date, end_date=date, wind_speed_unit="ms",
        hourly="temperature_2m,rain,snowfall,wind_gusts_10m", timezone="UTC"))
    for _ in range(4):
        try:
            return json.load(urllib.request.urlopen("https://archive-api.open-meteo.com/v1/archive?" + q, timeout=60))["hourly"]
        except Exception:
            time.sleep(20)
    return None

def candidate_dates(dkey, sgn, n=8):
    A = np.array([[np.nan if x is None else x for x in daily[h][dkey]] for h in hubs], float)
    best = np.nanmax(A * sgn, axis=0) * sgn; who = np.nanargmax(A * sgn, axis=0)
    out = []
    for i in np.argsort(-best * sgn):
        if T[i] < "2016-02-01" or any(abs((np.datetime64(T[i]) - np.datetime64(d)).astype(int)) <= 3 for d, _ in out):
            continue
        # hubs with this regime on that day (top 3 by value)
        hs = [hubs[j] for j in np.argsort(-A[:, i] * sgn)[:3]]
        out.append((T[i], hs))
        if len(out) >= n: break
    return out

stations = RW._fetch_stations_map()
plan, used_dates = [], set()
k = 0
for reg, (dkey, sgn, mkey, msgn, thr) in REG.items():
    got = 0
    for date, hs in candidate_dates(dkey, sgn):
        if got >= 3: break
        if date in used_dates: continue
        # peak hours at the worst hubs
        peak = {}
        for h in hs:
            hh = hub_hourly(h, date); time.sleep(1.5)
            if hh is None: continue
            v = np.array([np.nan if x is None else x for x in hh[H_KEY[mkey]]], float)
            peak[h] = set(int(t) for t in np.argsort(-v * msgn)[:6])
        try:
            trains, comps, cands = RW._fetch_trains_and_comps_for_date(date, stations)
        except Exception as e:
            print("digitraffic fail", date, e); continue
        # ep index this journey will take in the benchmark (odd episodes are the extreme ones)
        ep = 2 * (3 * list(REG).index(reg) + got) + 1
        seed = 9000 + ep * 29
        n_legs = int(np.random.default_rng(seed).integers(1, 6))
        scored = []
        for num in cands:
            hops = RW._extract_corridor_hops(trains[num].get("timeTableRows", []), stations, n_legs)
            if not hops: continue
            for h, hrs in peak.items():
                la, lo = HUBXY[h]
                d0, _ = RW.haversine_distance_and_bearing(la, lo, hops[0]["start"]["lat"], hops[0]["start"]["lon"])
                if d0 < 60000 and hops[0]["hour_utc"] in hrs:
                    scored.append((d0, num, h)); break
        scored.sort()
        best = None
        for d0, num, h in scored[:6]:              # check the actual leg weather the benchmark will use
            hops = RW._extract_corridor_hops(trains[num]["timeTableRows"], stations, n_legs)
            vals = []
            for hop in hops:
                w = RW.fetch_open_meteo_corridor_weather(0.5 * (hop["start"]["lat"] + hop["end"]["lat"]),
                                                         0.5 * (hop["start"]["lon"] + hop["end"]["lon"]), date, hop["hour_utc"])
                vals.append(w[mkey]); time.sleep(0.5)
            worst = max(v * msgn for v in vals) * msgn
            if best is None or worst * msgn > best[0] * msgn:
                best = (worst, num, h, len(hops), vals)
        if best is None or best[0] * msgn < thr * msgn:
            print(f"{reg} {date}: no journey past threshold ({None if best is None else round(best[0],1)})"); continue
        plan.append(dict(regime=reg, date=date, train_number=int(best[1]), hub=best[2], ep=ep, seed=seed,
                         n_legs=best[3], worst_leg_value=round(float(best[0]), 2), metric=mkey,
                         leg_values=[round(float(v), 2) for v in best[4]]))
        print(plan[-1], flush=True); used_dates.add(date); got += 1
json.dump(plan, open(sys.argv[2], "w"), indent=1); print("selected", len(plan))
