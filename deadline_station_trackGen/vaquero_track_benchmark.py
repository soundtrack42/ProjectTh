"""
Benchmark on the real track of the public Vaquero-Serrano & Felez (2026) high-speed line dataset
(Zenodo 10.5281/zenodo.20507243; see vaquero_line.py for how the line is used).

Runs the unchanged benchmark of dynamic_railroad_mpc.py (all methods, pre-trained models, no training) on
the synthetic-suite journeys moved onto the real grade / curve / speed-limit profile.  Nothing in
dynamic_railroad_mpc.py is edited: this script only
  1. wraps generate_multi_station_scenario so each journey is placed on the real line, and
  2. wraps the environment's physics sync so the plant and the controller model use the real track.
Outputs go to data_vaquero/ and figures/vaquero_*.png, so the synthetic and real-world results are untouched.

Usage (from deadline_station_trackGen/):
    python vaquero_line.py --install                 # once: download + verify + convert the dataset
    python vaquero_track_benchmark.py --check        # dataset, checkpoints and journeys only (no simulation)
    python vaquero_track_benchmark.py                # full benchmark (same defaults as the synthetic suite)
Any other option is passed on to dynamic_railroad_mpc.py (e.g. --eval-runs, --n-legs, --train-cars).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "core"))

import vaquero_line as vq  # noqa: E402

# Same defaults as the synthetic benchmark in README.md; outputs redirected to their own paths.
DEFAULT_ARGS = {
    "--eval-runs": "30",
    "--n-legs": "0",
    "--wind-mode": "mixed",
    "--harsh-weather": "mixed",
    "--load-checkpoint": "best_ppo_cosine_pilot.pt",
    "--data-dir": str(SCRIPT_DIR / "data_vaquero"),
    "--plot": str(SCRIPT_DIR / "figures" / "vaquero_rl_cosine_mpc.png"),
    "--dual-plot": str(SCRIPT_DIR / "figures" / "vaquero_wind_trajectory_shield_dual_axis.png"),
}
# Without these, dynamic_railroad_mpc.py / literature_baselines.py would train and save new checkpoints.
REQUIRED_CKPTS = (
    "best_emulator.pt", "best_ppo_cosine_pilot.pt", "best_ppo_free_pilot.pt",
    "best_lit_eitop.pt", "best_lit_sppo.pt", "lit_cdq.pt", "lit_esmedrl.pt", "lit_ga_ddpg.pt",
    "lit_shield_sarsa.pt", "lit_srl_ietto.pt", "lit_ssa_drl.pt",
)


def _pop_flag(argv: list[str], flag: str) -> bool:
    if flag in argv:
        argv.remove(flag)
        return True
    return False


def _pop_opt(argv: list[str], opt: str, default: str) -> str:
    if opt in argv:
        i = argv.index(opt)
        val = argv[i + 1]
        del argv[i:i + 2]
        return val
    return default


def _with_defaults(argv: list[str]) -> list[str]:
    given = {a.split("=", 1)[0] for a in argv if a.startswith("--")}
    out = list(argv)
    for k, v in DEFAULT_ARGS.items():
        if k not in given:
            out += [k, v]
    return out


def preflight(argv: list[str]) -> list[str]:
    """Returns the problems that would stop a clean run (missing dataset or checkpoints)."""
    problems = []
    if not vq.NPZ_PATH.exists():
        problems.append(f"dataset missing: run `python vaquero_line.py --install` ({vq.NPZ_PATH})")
    ckpt_dir = SCRIPT_DIR / "checkpoints"
    names = list(REQUIRED_CKPTS)
    for opt in ("--load-checkpoint", "--free-checkpoint"):
        if opt in argv:
            names.append(Path(argv[argv.index(opt) + 1]).name)
    suffix = os.environ.get("LIT_CKPT_SUFFIX", "")
    if suffix:
        names = [n.replace(".pt", f"{suffix}.pt") if n.startswith(("lit_", "best_lit_")) else n for n in names]
    for n in sorted(set(names)):
        if not (ckpt_dir / n).exists() and not Path(n).exists():
            problems.append(f"checkpoint missing (would trigger training): checkpoints/{n}")
    return problems


def main() -> None:
    argv = sys.argv[1:]
    check_only = _pop_flag(argv, "--check")
    v_cap = float(_pop_opt(argv, "--v-cap", str(vq.DEFAULT_V_CAP_MPS)))
    service = _pop_opt(argv, "--service", "nonstop")
    if "--real-data" in argv:
        sys.exit("--real-data selects the Digitraffic suite; the Vaquero suite builds on the synthetic journeys.")
    argv = _with_defaults(argv)

    problems = preflight(argv)
    for p in problems:
        print(f"[vaquero] {p}")
    if check_only:
        if not vq.NPZ_PATH.exists():
            sys.exit(1)
        n_runs = int(argv[argv.index("--eval-runs") + 1])
        n_legs = int(argv[argv.index("--n-legs") + 1])
        vq._summary(n_runs, n_legs, v_cap)
        print("[vaquero] check done (no simulation run).", "Problems above." if problems else "Ready to run.")
        sys.exit(1 if problems else 0)
    if problems:
        sys.exit("[vaquero] refusing to run; fix the problems above.")

    import dynamic_railroad_mpc as drm
    vq.install_env_hook(drm.MPCTransitionRailroadEnv)
    drm.generate_multi_station_scenario = vq.make_scenario_generator(
        drm.generate_multi_station_scenario, v_cap=v_cap, service=service)
    print(f"[vaquero] real track: {vq.get_line(service).source} | speed cap {v_cap} m/s | service {service}")
    sys.argv = [str(SCRIPT_DIR / "dynamic_railroad_mpc.py")] + argv
    drm.main()


if __name__ == "__main__":
    main()
