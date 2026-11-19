"""
Pilot selection on the tuning journeys (seeds 5000 + 29k; disjoint from training, validation and the
held-out benchmark seeds 9000 + 29k). Each pilot runs as method 2 (unshielded) and method 5 (UCB-DMPS).

    python eval_pilots.py PILOT OUT_JSON [free]
`free` evaluates a no-cosine pilot as methods 6 (unshielded) and 7 (UCB-DMPS).
PILOT is a MASA export (*.npz, deterministic tanh-MLP actor) or our PPO checkpoint (*.pt).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
os.chdir(PROJ)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
import dynamic_railroad_mpc as D  # noqa: E402
from railroad_corridor import generate_multi_station_scenario  # noqa: E402

N_TUNING = 30


class NumpyPilot:
    """Deterministic MASA actor exported by pilot_masa.py: tanh MLP, linear mean, clipped to [-1, 1]."""

    def __init__(self, path: str):
        w = np.load(path)
        n = len([k for k in w.files if k.startswith("W")])
        self.layers = [(w[f"W{i}"], w[f"b{i}"]) for i in range(n)]

    def get_action_and_value(self, obs, deterministic: bool = True):
        h = np.asarray(obs, dtype=np.float32)
        for i, (wi, bi) in enumerate(self.layers):
            h = h @ wi + bi
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        return np.clip(h, -1.0, 1.0).astype(np.float32), 0.0, 0.0


def load_pilot(path: str):
    if path.endswith(".npz"):
        return NumpyPilot(path)
    p = D.CosineTrajectoryRLPolicy(obs_dim=18, act_dim=3, hidden_dim=64)
    p.load_state_dict(torch.load(path, map_location="cpu"))
    p.eval()
    return p


def main():
    path, out = sys.argv[1], sys.argv[2]
    free = len(sys.argv) > 3 and sys.argv[3] == "free"
    mode = "RL_FREE_LEARNED" if free else "RL_LEARNED"
    emu = D.LearnedLinearizationModel(hidden_dim=64)
    emu.load_state_dict(torch.load("checkpoints/best_emulator.pt", map_location="cpu"))
    emu.eval()
    pilot = load_pilot(path)
    scens = [generate_multi_station_scenario(seed=5000 + 29 * ep, n_legs=0, wind_mode="mixed",
                                             allow_infeasible_breaks=(ep > 0), extreme_noise=(ep % 2 == 1))
             for ep in range(N_TUNING)]
    res = {"pilot": path}
    names = ("m6_unshielded", "m7_ucb_dmps") if free else ("m2_unshielded", "m5_ucb_dmps")
    for name, sh in zip(names, (None, "mps_rollout")):
        rs = [D.evaluate_multi_station_journey(name, sc, mode, rl_policy=pilot, shield_mode=sh, emulator=emu)
              for sc in scens]
        g = lambda k: float(np.mean([r[k] for r in rs]))
        res[name] = dict(loss=g("multi_obj_score"), T=g("abs_sched_err_s"), stop=g("abs_stop_err_m"),
                         smooth=g("step_smooth_cost"), E=g("spec_energy_wh_tkm"), shield_pct=g("shield_rate_pct"),
                         max_us=float(max(r["max_total_us"] for r in rs)))
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
