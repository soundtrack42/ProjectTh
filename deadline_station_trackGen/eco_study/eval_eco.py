"""Evaluate one pilot checkpoint. Usage: SUITE(tune|syn|real) METHOD(m2|m5|m6|m7) CKPT ECO(default|off|latch|on) OUT.json
tune = 30 synthetic tuning journeys (seeds 5000+29k); syn = 30 synthetic test journeys (9000+29k, as the paper);
real = 30 Digitraffic + Open-Meteo journeys of 2026-11-16 (as the paper), from real30.pkl."""
import os, sys, json, pickle
SC = "/tmp/claude-1000/-home-giorge-Desktop-ProjectTh/b16d013a-8abd-404a-82c9-8d2177fa6873/scratchpad/eco"
SRC = "/home/giorge/Desktop/ProjectTh/deadline_station_trackGen"
os.chdir(SRC); sys.path[:0] = [SRC, os.path.dirname(SRC), os.path.join(os.path.dirname(SRC), "core")]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)
import dynamic_railroad_mpc as D
import real_world_rail_weather as RW
from railroad_corridor import generate_multi_station_scenario

suite, meth, ckpt, eco, out = sys.argv[1:6]
mode, shield = {"m2": ("RL_LEARNED", None), "m5": ("RL_LEARNED", "mps_rollout"),
                "m6": ("RL_FREE_LEARNED", None), "m7": ("RL_FREE_LEARNED", "mps_rollout")}[meth]
eco_flag = {"default": None, "off": False, "latch": True, "on": True}[eco]
if eco in ("latch", "on"):
    orig = D.MPCTransitionRailroadEnv.__init__
    def patched(self, *a, **k):
        orig(self, *a, **k)
        if eco == "latch":
            self.eco_a_cap, self.eco_slack_reserve_s = None, 4.0
        elif self.use_cosine_shaper:
            self.eco_a_cap, self.eco_slack_reserve_s = 0.70, 5.0
    D.MPCTransitionRailroadEnv.__init__ = patched

emu = D.LearnedLinearizationModel(hidden_dim=64)
emu.load_state_dict(torch.load("checkpoints/best_emulator.pt", map_location="cpu")); emu.eval()
pol = D.CosineTrajectoryRLPolicy(obs_dim=18, act_dim=3, hidden_dim=64)
pol.load_state_dict(torch.load(ckpt, map_location="cpu")); pol.eval(); pol._refresh_actor_views()
if suite == "real":
    RW._WEATHER_CACHE.update(pickle.load(open(f"{SC}/real30.pkl.wxcache", "rb")))
    scens = pickle.load(open(f"{SC}/real30.pkl", "rb"))
else:
    base = 5000 if suite == "tune" else 9000
    scens = [generate_multi_station_scenario(seed=base + 29 * ep, n_legs=0, wind_mode="mixed",
             allow_infeasible_breaks=(ep > 0), extreme_noise=(ep % 2 == 1)) for ep in range(30)]
rows = []
for ep, sc in enumerate(scens):
    r = D.evaluate_multi_station_journey(meth, sc, mode, rl_policy=pol, shield_mode=shield,
                                         emulator=emu, eco_governor=eco_flag)
    rows.append(dict(ep=ep, ext=bool(sc.get("is_extreme_scenario", sc.get("is_harsh_injected", False))),
                     loss=r["multi_obj_score"], T=r["abs_sched_err_s"], stop=r["abs_stop_err_m"],
                     smooth=r["step_smooth_cost"], E=r["spec_energy_wh_tkm"], kwh=r["energy_kwh"],
                     shield=r["shield_rate_pct"], eco=r["eco_rate_pct"], max_us=r.get("max_total_us", 0.0)))
json.dump(rows, open(out, "w"))
print("done", out)
