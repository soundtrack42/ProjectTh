"""
Departure mass-identification benchmark: station-only approach vs. second platform-end anemometer.

Both approaches use the same Kalman filter on (theta = 1000/M_eff, wind d), with the wind as a random walk
(gusts and turbulence act from the first step); they differ only in the sensors and therefore the window:
  * station-only  (x_a2 = 0):  station anemometer reading every step, fixed 8-step (4 s) window.
  * 2nd anemometer (x_a2 > 0): both wayside anemometers send a reading every step; the window runs until the train
                               passes x_a2 (10 m default, 15 m where the platform allows).
Anemometer noise follows a low-cost ultrasonic sensor spec (max(0.3 m/s, 3 %), see railroad_corridor.py).

Runs only the departure phase of every leg of the 30 + 30 benchmark journeys (same seeds as the main benchmark,
verified against data/*_scenarios.json), so it takes seconds per configuration. Closed-loop effects on the full
benchmark are obtained with `dynamic_railroad_mpc.py --anemometer2-x-m 0` (station-only) vs. the default.

Usage:
    python benchmark_mass_id.py                    # synthetic + real (real needs Digitraffic/Open-Meteo access)
    python benchmark_mass_id.py --suites synthetic
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dynamic_railroad_mpc as D
from railroad_corridor import generate_multi_station_scenario

CONFIGS = [("Station-only (8 steps)", 0.0), ("2nd anemometer @ 10 m", 10.0), ("2nd anemometer @ 15 m", 15.0)]


def build_scenarios(suite: str, train_date: str, train_number: int, historic: dict | None = None) -> list[dict]:
    scens = []
    for ep in range(30):
        if suite == "real" and historic and ep in historic:
            from real_world_rail_weather import fetch_real_world_multi_station_scenario

            h = historic[ep]
            sc = fetch_real_world_multi_station_scenario(
                date_str=h["date"], train_number=int(h["train_number"]), n_legs_max=0, seed=int(h["seed"]), ep_idx=0,
                harsh_weather_mode="historic",
            )
        elif suite == "real":
            from real_world_rail_weather import fetch_real_world_multi_station_scenario

            sc = fetch_real_world_multi_station_scenario(
                date_str=train_date, train_number=train_number, n_legs_max=0, seed=9000 + ep * 29, ep_idx=ep,
                harsh_weather_mode="mixed",
            )
        else:
            sc = generate_multi_station_scenario(
                seed=9000 + ep * 29, n_legs=0, wind_mode="mixed", allow_infeasible_breaks=(ep > 0),
                extreme_noise=(ep % 2 == 1),
            )
        scens.append(sc)
    return scens


def verify_against_benchmark(suite: str, scens: list[dict], data_dir: Path) -> int:
    pre = "real_world_api" if suite == "real" else "synthetic"
    saved = {}
    for sub in ("baseline", "extreme"):
        p = data_dir / f"{pre}_{sub}_scenarios.json"
        if p.exists():
            for s in json.load(open(p))["scenarios"]:
                saved[s["scenario_num"]] = s
    if not saved:
        return -1
    mism = 0
    for i, sc in enumerate(scens):
        sv = saved.get(i + 1)
        if sv is None:
            mism += 1
            continue
        for leg, ls in zip(sc["legs"], sv["legs"]):
            c = leg["corridor"]
            if abs(c.mass_kg - ls["mass_kg"]) > 1e-6 or abs(c.deadline_s - ls["deadline_s"]) > 1e-6:
                mism += 1
    return mism


def identify_departure_mass(emu, leg: dict, x_a2: float) -> dict:
    """Run one departure until the mass estimate locks (nominal pilot command); return the estimate."""
    D.STATION_ANEMOMETER2_X_M = x_a2
    cor = copy.deepcopy(leg["corridor"])
    env = D.MPCTransitionRailroadEnv(corridor=cor, traj_shield=None, emulator=emu, use_cosine_shaper=True)
    env.reset(corridor_override=cor, wind_profile=np.asarray(leg["wind_profile"], dtype=np.float64))
    for _ in range(60):
        if env.mass_locked:
            break
        if env.step(np.zeros(3, np.float32))[2]:
            break
    return {"est": float(env.mass_hat_kg), "prior": float(env.mass_prior_kg), "true": float(cor.mass_kg),
            "noisy": bool(env.speed_sensor_noise_mps > 0.0)}


def summarize(err: np.ndarray, true: np.ndarray) -> dict:
    a = np.abs(err)
    return {"n": len(a), "mae": a.mean(), "bias": err.mean(), "p90": np.percentile(a, 90), "max": a.max(),
            "rel": 100.0 * (a / true).mean()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="+", default=["synthetic", "real"], choices=["synthetic", "real"])
    ap.add_argument("--train-date", default="2026-11-16")
    ap.add_argument("--train-number", type=int, default=8311)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--fig", default="figures/mass_id_benchmark.png")
    ap.add_argument("--historic-suite", default="", help="Same JSON as dynamic_railroad_mpc.py --historic-suite")
    args = ap.parse_args()
    historic = {int(h["ep"]): h for h in json.load(open(args.historic_suite))} if args.historic_suite else None

    torch.set_num_threads(1)
    emu = D.LearnedLinearizationModel(hidden_dim=64)
    emu.load_state_dict(torch.load("checkpoints/best_emulator.pt", map_location="cpu"))
    emu.eval()

    rows, results = [], {}
    for suite in args.suites:
        scens = build_scenarios(suite, args.train_date, args.train_number, historic)
        mism = verify_against_benchmark(suite, scens, Path(args.data_dir))
        print(f"[{suite}] scenario check vs. benchmark: " + ("no saved scenarios" if mism < 0 else f"{mism} mismatches"))
        for si, sc in enumerate(scens):
            for li, leg in enumerate(sc["legs"]):
                rec = {"suite": suite, "scenario_num": si + 1, "leg": li + 1}
                for name, xa2 in CONFIGS:
                    r = identify_departure_mass(emu, leg, xa2)
                    rec.update(true_kg=r["true"], prior_kg=r["prior"], noisy=r["noisy"])
                    rec[f"est_{xa2:g}m"] = r["est"]
                rows.append(rec)
    D.STATION_ANEMOMETER2_X_M = 10.0

    out_csv = Path(args.data_dir) / "mass_id_benchmark.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hdr = f"{'Suite':10s} {'Telemetry':10s} {'Approach':24s} {'Legs':>4s} {'MAE kg':>8s} {'Bias':>7s} {'P90':>7s} {'Max':>7s} {'MAE %':>6s}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for suite in args.suites:
        for noisy, lab in ((False, "nominal"), (True, "extreme")):
            sel = [r for r in rows if r["suite"] == suite and r["noisy"] == noisy]
            if not sel:
                continue
            true = np.array([r["true_kg"] for r in sel])
            cols = [("Ticket prior", np.array([r["prior_kg"] for r in sel]))]
            cols += [(n, np.array([r[f"est_{x:g}m"] for r in sel])) for n, x in CONFIGS]
            for n, est in cols:
                s = summarize(est - true, true)
                results[(suite, lab, n)] = (s, np.abs(est - true))
                print(f"{suite:10s} {lab:10s} {n:24s} {s['n']:4d} {s['mae']:8.1f} {s['bias']:+7.0f} {s['p90']:7.0f} {s['max']:7.0f} {s['rel']:6.2f}")
    print(f"\nPer-leg results: {out_csv}")

    # Figure: empirical CDF of |mass error| on extreme-noise legs, one panel per suite (shared x scale).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Ticket prior": "#8a8a85", CONFIGS[0][0]: "#eb6834", CONFIGS[1][0]: "#2a78d6", CONFIGS[2][0]: "#1baf7a"}
    styles = {"Ticket prior": (0, (4, 3))}
    fig, axes = plt.subplots(1, len(args.suites), figsize=(5.2 * len(args.suites), 3.6), sharey=True, squeeze=False)
    for ax, suite in zip(axes[0], args.suites):
        for n in ["Ticket prior"] + [c[0] for c in CONFIGS]:
            if (suite, "extreme", n) not in results:
                continue
            a = np.sort(results[(suite, "extreme", n)][1])
            ax.step(a, np.arange(1, len(a) + 1) / len(a), where="post", lw=2, color=colors[n],
                    linestyle=styles.get(n, "-"), label=f"{n} (MAE {a.mean():.0f} kg)")
        ax.set_title(f"{'Real-world API' if suite == 'real' else 'Synthetic'}: extreme-noise legs", fontsize=10)
        ax.set_xlabel("|mass error| (kg)")
        ax.set_xlim(left=0)
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    axes[0][0].set_ylabel("Fraction of legs")
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=200)
    print(f"Figure: {args.fig}")


if __name__ == "__main__":
    main()
