import os, sys, pickle
SRC = "/home/giorge/Desktop/ProjectTh/deadline_station_trackGen"
os.chdir(SRC); sys.path[:0] = [SRC, os.path.dirname(SRC), os.path.join(os.path.dirname(SRC), "core")]
import real_world_rail_weather as RW
scens = [RW.fetch_real_world_multi_station_scenario(date_str="2026-11-16", train_number=8311, n_legs_max=0,
         seed=9000 + ep * 29, ep_idx=ep, harsh_weather_mode="mixed") for ep in range(30)]
out = sys.argv[1]
pickle.dump(scens, open(out, "wb")); pickle.dump(dict(RW._WEATHER_CACHE), open(out + ".wxcache", "wb"))
print(len(scens), scens[0]["data_source"], scens[0]["date_str"], sum(s.get("is_extreme_scenario", False) for s in scens))
