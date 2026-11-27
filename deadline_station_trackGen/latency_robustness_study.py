"""Does the evaluation machine affect the results? (appendix study)

Runs the same synthetic journeys (all 18 methods) under three machine conditions on ordinary user-space Linux:
  quiet   : benchmark pinned to one core, machine otherwise idle (the setting of the main benchmark)
  loaded  : every other core runs a CPU spinner or a memory/cache thrasher
  shared  : as loaded, plus a spinner pinned to the SAME core as the benchmark (it gets ~half the core and is
            preempted by the scheduler: the worst case a non-real-time OS can produce for a single task)
(quiet-r2 repeats the quiet run to measure run-to-run variation), and with two clocks for the per-step latency:
  cpu     : thread CPU time (what the main benchmark reports)
  wall    : wall-clock time (includes preemption and scheduler delays)

For each method it reports the response time of every 0.5 s control step, un-amortised (the pilot and shield cost
is charged to the step at which the decision is taken), as mean / p99 / p99.9 / max against the 0.5 s control period, and
whether the journey outcomes (punctuality, stop error, energy, comfort, safety counts, model evaluations) are
identical to the quiet run. Wall-clock-budgeted planners (Xiao, Zhong) also report how many calls hit the budget.

Usage: python latency_robustness_study.py --journeys 6 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import types
from pathlib import Path

import numpy as np
import torch

import dynamic_railroad_mpc as drm
from railroad_corridor import generate_multi_station_scenario

CTRL_PERIOD_US = 0.5e6
OUTCOME_KEYS = [
    "abs_sched_err_s", "abs_stop_err_m", "spec_energy_wh_tkm", "energy_kwh", "multi_obj_score", "step_smooth_cost",
    "shield_rate_pct", "jerk_exceed_steps", "accel_exceed_steps", "state_unsafe_steps", "unsafe_steps",
    "model_evals", "n_micro_steps",
]


def _spin(core: int) -> None:
    os.sched_setaffinity(0, {core})
    x = 0
    while True:
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF


def _thrash(core: int) -> None:
    os.sched_setaffinity(0, {core})
    a = np.ones(16 * 1024 * 1024 // 8)          # 16 MB: larger than the per-CCX L3 share, evicts the benchmark
    b = np.empty_like(a)
    while True:
        np.copyto(b, a)
        np.copyto(a, b)


def start_load(condition: str, bench_core: int) -> list[mp.Process]:
    if condition == "quiet":
        return []
    procs = []
    others = [c for c in sorted(os.sched_getaffinity(0)) if c != bench_core]
    for i, c in enumerate(others):
        procs.append(mp.Process(target=_thrash if i % 2 else _spin, args=(c,), daemon=True))
    if condition == "shared":
        procs.append(mp.Process(target=_spin, args=(bench_core,), daemon=True))
    for p in procs:
        p.start()
    time.sleep(1.0)
    return procs


RESP_US: list[float] = []   # un-amortised response time of every control step (pilot + shield + MPC + emulator)


def _wrap_step(fn):
    def wrapped(self, *a, **k):
        out = fn(self, *a, **k)
        for m in out[3].get("micro_logs", []):
            RESP_US.append(m["step_us"] + m["emu_us"] + m.get("pol_us", 0.0) + m.get("sh_us", 0.0))
        return out
    return wrapped


drm.MPCTransitionRailroadEnv.step = _wrap_step(drm.MPCTransitionRailroadEnv.step)
drm.MPCTransitionRailroadEnv.step_bilbao_cbtc_ato = _wrap_step(drm.MPCTransitionRailroadEnv.step_bilbao_cbtc_ato)


def set_clock(kind: str) -> None:
    if kind == "wall":
        shim = types.SimpleNamespace(**{k: getattr(time, k) for k in dir(time) if not k.startswith("_")})
        shim.thread_time = time.perf_counter
        drm.time = shim
    else:
        drm.time = time


def load_models(cos_ckpt: str, free_ckpt: str):
    ck = Path(drm.__file__).resolve().parent / "checkpoints"
    emu = drm.LearnedLinearizationModel(hidden_dim=64)
    emu.load_state_dict(torch.load(ck / "best_emulator.pt", map_location="cpu"))
    emu.eval()
    pols = []
    for f in (cos_ckpt, free_ckpt):
        p = drm.CosineTrajectoryRLPolicy(obs_dim=18, act_dim=3, hidden_dim=64)
        p.load_state_dict(torch.load(ck / f, map_location="cpu"))
        p.eval()
        pols.append(p)
    return emu, pols[0], pols[1]


def method_specs(emu):
    specs = [
        ("1. Max-Speed Greedy Cosine + MPC", "MAX_SPEED_GREEDY", None),
        ("2. Multi-Property PPO Pilot (Unshielded)", "RL_LEARNED", None),
        ("3. Contract-Shielded PPO (Switch MPS)", "RL_LEARNED", "switch"),
        ("4. Contract-Shielded PPO (1-Step DCBF)", "RL_LEARNED", "cbf_opt"),
        ("5. Contract-Shielded PPO (UCB-DMPS Cosine)", "RL_LEARNED", "mps_rollout"),
        ("6. Stable-Accel PPO (No Cosine, Unshielded)", "RL_FREE_LEARNED", None),
        ("7. Stable-Accel PPO + Jerk-Horizon UCB-DMPS", "RL_FREE_LEARNED", "mps_rollout"),
        ("8. Bilbao (2025) CBTC-ATO (RMBTP + PI)", "BILBAO_CBTC_ATO", None),
    ]
    from literature_baselines import load_extra_literature_pilots, load_literature_pilots
    lit = load_literature_pilots(emu, train_if_missing=False, log=lambda *a, **k: None)
    lit.update(load_extra_literature_pilots(emu, train_if_missing=False, log=lambda *a, **k: None))
    specs.extend((nm, "LIT_PILOT", None) for nm in lit)
    return specs, lit


def run_condition(args, condition: str, clock: str) -> dict:
    set_clock(clock)
    procs = start_load(condition.split("-")[0], args.core)
    try:
        emu, pol, pol_free = load_models(args.cos, args.free)
        specs, lit = method_specs(emu)
        out = {}
        for ep in range(args.journeys):
            scen = generate_multi_station_scenario(seed=9000 + ep * 29, n_legs=0, wind_mode="mixed",
                                                   allow_infeasible_breaks=(ep > 0), extreme_noise=(ep % 2 == 1))
            for name, mode, sh in specs:
                pilot = lit[name]() if mode == "LIT_PILOT" else None
                RESP_US.clear()
                res = drm.evaluate_multi_station_journey(
                    label=name, scenario=scen, mode=mode,
                    rl_policy=pol_free if mode == "RL_FREE_LEARNED" else pol,
                    shield_mode=sh, emulator=emu, record_traj=False, lit_pilot=pilot)
                rec = {k: float(res[k]) for k in OUTCOME_KEYS if k in res}
                r = np.asarray(RESP_US)
                rec.update(mean_us=float(r.mean()), p99_us=float(np.percentile(r, 99)), p999_us=float(np.percentile(r, 99.9)),
                           max_us=float(r.max()), amortised_max_us=res["max_total_us"], n_steps=int(r.size))
                if pilot is not None and hasattr(pilot, "escapes"):
                    rec["escapes"] = int(pilot.escapes)
                    rec["calls"] = int(getattr(pilot, "calls", 0) or getattr(pilot, "steps", 0))
                out.setdefault(name, []).append(rec)
            print(f"  [{condition}/{clock}] journey {ep + 1}/{args.journeys} done", flush=True)
        return out
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.join()


def summarise(runs: dict) -> dict:
    ref = runs["quiet/cpu"]
    table = {}
    for name in ref:
        row = {}
        for key, res in runs.items():
            recs = res[name]
            diffs = [max(abs(a[k] - b[k]) for k in OUTCOME_KEYS if k in a) for a, b in zip(recs, ref[name])]
            row[key] = {
                "mean_us": float(np.mean([r["mean_us"] for r in recs])),
                "p99_us": float(np.mean([r["p99_us"] for r in recs])),
                "p999_us": float(np.mean([r["p999_us"] for r in recs])),
                "max_us": float(np.max([r["max_us"] for r in recs])),
                "amortised_max_us": float(np.max([r["amortised_max_us"] for r in recs])),
                "max_pct_period": 100.0 * float(np.max([r["max_us"] for r in recs])) / CTRL_PERIOD_US,
                "outcomes_identical": bool(max(diffs) == 0.0),
                "max_outcome_diff": float(max(diffs)),
                "sched_err_s": float(np.mean([r["abs_sched_err_s"] for r in recs])),
                "energy_wh_tkm": float(np.mean([r["spec_energy_wh_tkm"] for r in recs])),
                "score": float(np.mean([r["multi_obj_score"] for r in recs])),
            }
            if "escapes" in recs[0]:
                row[key]["escape_pct"] = 100.0 * sum(r["escapes"] for r in recs) / max(1, sum(r["calls"] for r in recs))
        table[name] = row
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journeys", type=int, default=6)
    ap.add_argument("--core", type=int, default=max(os.sched_getaffinity(0)))
    ap.add_argument("--cos", default="best_ppo_cosine_pilot.pt")
    ap.add_argument("--free", default="best_ppo_free_pilot.pt")
    ap.add_argument("--out", default="data/latency_study")
    ap.add_argument("--conditions", default="quiet/cpu,quiet-r2/cpu,quiet/wall,loaded/cpu,loaded/wall,shared/cpu,shared/wall")
    args = ap.parse_args()
    os.sched_setaffinity(0, {args.core})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for cc in args.conditions.split(","):
        cond, clock = cc.split("/")
        t0 = time.perf_counter()
        runs[cc] = run_condition(args, cond, clock)
        print(f"{cc}: {time.perf_counter() - t0:.0f} s", flush=True)
        (out / "raw.json").write_text(json.dumps(runs, indent=1))
    table = summarise(runs)
    (out / "summary.json").write_text(json.dumps(table, indent=1))
    print(f"\n{'method':48s}" + "".join(f"{k:>22s}" for k in runs))
    for name, row in table.items():
        cells = []
        for k in runs:
            c = row[k]
            tag = "=" if c["outcomes_identical"] else "≠"
            esc = f" e{c['escape_pct']:.0f}%" if "escape_pct" in c else ""
            cells.append(f"{c['max_us']:8.0f}us {tag}{esc}")
        print(f"{name[:48]:48s}" + "".join(f"{s:>22s}" for s in cells))


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
