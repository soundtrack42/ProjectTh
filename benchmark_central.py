"""
Centralized Benchmark Suite (`benchmark_central.py`)
Evaluates and compares:
  1. Single Fixed Heuristic PIDs (GA, PSO, DE, ICA)
  2. Standard Pure MPC (Liu et al. 2025, Fixed Jerk Weight Q_3)
  3. Idea C: Adaptive-Shift MPC (Liu et al. 2025 + Dynamic Shift-Aware Q_3 Relaxation)
  4. Hardcoded-Switch IPID (Havaei & Sandidzadeh 2022)
  5. Idea A (Way 1): RL-Switch with MPC Horizon Rollout on Paper Linear Approx (Batch-30)
  6. Idea A (Way 2): RL-Switch with MPC Horizon Rollout on 150-Step Emulator (Batch-30)
  7. Idea A + Idea C: 5-Action Hybrid RL-IPID-MPC Switch (['GA', 'PSO', 'DE', 'ICA', 'MPC'])
"""
from __future__ import annotations

import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pure_mpc_controller import PureLiuMPC
from havaei_generalizing_rl import (
    ALGORITHMS,
    ALG_TO_IDX,
    HYBRID_ALGORITHMS,
    HardcodedSwitchController,
    PathAgnosticHavaeiEnv,
    train_rl_switch_strict_two_ways,
)

MAX_ACCEPTABLE_EPISODE_MS = 1000.0
MAX_ACCEPTABLE_STEP_US = 500.0


def generate_truly_random_trajectory(
    total_horizon: int = 400,
    min_phase_len: int = 25,
    seed: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Randomly samples driving modes (1..6) and durations (>= min_phase_len) summing to `total_horizon`.
    Guarantees at least one Mode 2 (Shift Up) or Mode 5 (Shift Down) jump occurs in every track
    so `ShiftOS` and `ShiftAvg` are always actively tested!
    """
    rng = np.random.default_rng(seed)
    v_des = np.zeros(total_horizon + 2, dtype=np.float64)
    schedule = []

    cursor = 0
    curr_v = 0.0
    has_shift = False

    while cursor < total_horizon:
        rem_steps = total_horizon - cursor
        if rem_steps <= min_phase_len * 2:
            dur = rem_steps
            mode = 6 if curr_v > 0.5 else 3
        else:
            max_dur = min(60, rem_steps - min_phase_len)
            dur = int(rng.integers(min_phase_len, max_dur + 1))

            # Force a shift jump mid-journey if none has occurred yet
            if (not has_shift) and (cursor >= total_horizon // 3) and (curr_v >= 4.0):
                mode = 2 if curr_v <= 11.5 else 5
            elif curr_v <= 0.2:
                mode = int(rng.choice([1, 2, 6], p=[0.65, 0.20, 0.15]))
            elif curr_v >= 14.5:
                mode = int(rng.choice([3, 4, 5], p=[0.35, 0.35, 0.30]))
            else:
                mode = int(rng.integers(1, 7))

        if mode in (2, 5):
            has_shift = True

        t_loc = np.linspace(0.0, 1.0, dur, endpoint=False)
        start_k = cursor
        end_k = cursor + dur

        if mode == 1:
            low_v = min(curr_v + 1.5, 15.0)
            target_v = float(rng.uniform(low_v, 16.5))
            v_seg = curr_v + (target_v - curr_v) * (1.0 - np.exp(-3.2 * t_loc)) / (1.0 - np.exp(-3.2))
            curr_v = target_v
        elif mode == 2:
            target_v = float(min(16.8, curr_v + rng.uniform(3.0, 5.0)))
            v_seg = np.full(dur, target_v)
            curr_v = target_v
        elif mode == 3:
            v_seg = np.full(dur, curr_v) + 0.12 * np.sin(2.0 * np.pi * t_loc)
        elif mode == 4:
            target_v = float(rng.uniform(max(1.0, curr_v - 6.0), max(1.5, curr_v - 2.5)))
            v_seg = curr_v + (target_v - curr_v) * (t_loc ** 0.75)
            curr_v = target_v
        elif mode == 5:
            target_v = float(max(3.0, curr_v - rng.uniform(3.5, 5.0)))
            v_seg = np.full(dur, target_v)
            curr_v = target_v
        else:
            if curr_v > 0.5:
                stop_ramp = min(dur // 2, 16)
                v_seg = np.zeros(dur)
                v_seg[:stop_ramp] = curr_v * (1.0 - np.linspace(0, 1, stop_ramp) ** 0.8)
            else:
                v_seg = np.zeros(dur)
            curr_v = 0.0

        v_des[start_k:end_k] = v_seg
        schedule.append({"mode": mode, "start": start_k, "end": end_k - 1, "dur": dur, "v_target": float(curr_v)})
        cursor = end_k

    v_des[0] = 0.0
    v_des[total_horizon:] = 0.0
    return v_des, schedule


def run_controller_benchmark(
    method_name: str,
    v_des: np.ndarray,
    controller_type: str,
    rl_policy=None,
    mpc_ctrl: PureLiuMPC | None = None,
    lookahead_n: int = 15,
) -> dict:
    env = PathAgnosticHavaeiEnv(
        v_des_profile=v_des,
        plant_mode="nonlinear",
        lookahead_n=lookahead_n,
    )
    obs = env.reset()
    done = False

    hc = HardcodedSwitchController(transient_blend=0.0)
    hc.reset()
    if mpc_ctrl is not None:
        mpc_ctrl.reset()

    k_log, v_log, v_des_log, err_log, os_log, shift_err_log, brake_err_log, alg_log = [], [], [], [], [], [], [], []
    step_latencies_us = []

    while not done:
        t0 = time.perf_counter()

        if controller_type == "MPC":
            assert mpc_ctrl is not None
            k_curr = env.k
            ref_h = env.v_des[k_curr + 1 : min(k_curr + 1 + mpc_ctrl.n_p, len(env.v_des))]
            slope_iota = env.physics.grade_resistance(env.x_pos) / (env.physics.M * env.physics.g)
            curve_iota = env.physics.curve_resistance(env.x_pos) / (env.physics.M * env.physics.g)
            u_opt, s_boost = mpc_ctrl.select_action(env.v_real, ref_h, slope_iota=slope_iota + curve_iota)
            t1 = time.perf_counter()
            obs, _, term, trunc, info = env.step_mpc(u_opt, mpc_ctrl, shift_boost=s_boost)

        elif controller_type == "HARDCODED":
            action = hc.select_action(env)
            t1 = time.perf_counter()
            obs, _, term, trunc, info = env.step(action)

        elif controller_type == "RL_SOFT":
            assert rl_policy is not None
            with torch.no_grad():
                _, action, _, _ = rl_policy.act(obs, deterministic=True, soft_blend=True)
            t1 = time.perf_counter()
            obs, _, term, trunc, info = env.step(action)

        elif controller_type == "RL_HYBRID":
            assert rl_policy is not None
            with torch.no_grad():
                _, action, _, _ = rl_policy.act(obs, deterministic=True, soft_blend=True)
            t1 = time.perf_counter()
            obs, _, term, trunc, info = env.step_hybrid(action)

        elif controller_type.startswith("FIXED_"):
            alg_idx = int(controller_type.split("_")[1])
            t1 = time.perf_counter()
            obs, _, term, trunc, info = env.step(alg_idx)

        else:
            raise ValueError(f"Unknown controller_type: {controller_type}")

        done = term or trunc
        step_latencies_us.append((t1 - t0) * 1e6)

        k_log.append(info["k"])
        v_log.append(info["v_real"])
        v_des_log.append(info["v_des"])
        err_log.append(info["abs_err"])
        os_log.append(info["overshoot"])
        if info.get("shift_active", False):
            shift_err_log.append(info["abs_err"])
        if info.get("braking_active", False):
            brake_err_log.append(info["abs_err"])
        alg_log.append(info["alg"])

    err_arr = np.array(err_log)
    os_arr = np.array(os_log)
    lat_arr = np.array(step_latencies_us)

    mean_err = float(np.mean(err_arr))
    shift_os = float(np.max(os_arr)) if len(os_arr) > 0 else 0.0
    shift_avg_err = float(np.mean(shift_err_log)) if len(shift_err_log) > 0 else 0.0
    brake_err = float(np.mean(brake_err_log)) if len(brake_err_log) > 0 else 0.0
    stop_err = float(info["stopping_error"])

    paper_score = (mean_err + shift_os + shift_avg_err + brake_err + stop_err) / 5.0
    ep_infer_ms = float(np.sum(lat_arr) / 1000.0)
    mean_step_us = float(np.mean(lat_arr))
    passes_realtime = (ep_infer_ms <= MAX_ACCEPTABLE_EPISODE_MS) and (mean_step_us <= MAX_ACCEPTABLE_STEP_US)

    return {
        "Method": method_name,
        "Mean_Err": mean_err,
        "Shift_Overshoot": shift_os,
        "Shift_SettleErr": shift_avg_err,
        "Brake_Err": brake_err,
        "Stop_Err_m": stop_err,
        "Paper_Score": paper_score,
        "Infer_Mean_us": mean_step_us,
        "Episode_Infer_ms": ep_infer_ms,
        "Passes_Benchmark": "PASS" if passes_realtime else "FAIL",
        "traj": {
            "k": np.array(k_log),
            "v": np.array(v_log),
            "v_des": np.array(v_des_log),
            "err": err_arr,
            "alg": alg_log,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Centralized Benchmark: Idea A (MPC-Rollout RL) & Idea C (Adaptive MPC)")
    ap.add_argument("--horizon", type=int, default=400, help="Total working steps T")
    ap.add_argument("--lookahead-n", type=int, default=15, help="Length of lookahead acceleration vector A")
    ap.add_argument("--epochs", type=int, default=250, help="Total RL training epochs")
    ap.add_argument("--regen-every", type=int, default=30, help="Regenerate random trajectory every N epochs")
    ap.add_argument("--emulator-steps", type=int, default=150, help="Steps for Trajectory Emulator in Way 2")
    ap.add_argument("--eval-runs", type=int, default=30, help="Number of unseen random trajectories for Test 2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", type=str, default="central_benchmark_400.png")
    args = ap.parse_args()

    print("=" * 130)
    print(
        f"CENTRALIZED ATO BENCHMARK | Horizon T = {args.horizon} | "
        f"Lookahead A-Vector = {args.lookahead_n} steps | Epochs = {args.epochs} (Batch = {args.regen_every})"
    )
    print("=" * 130)

    v_des_fixed, schedule_fixed = generate_truly_random_trajectory(
        total_horizon=args.horizon, min_phase_len=25, seed=args.seed
    )
    print("\n[Generated Reference Multi-Phase Trajectory Schedule (Modes 1..6)]")
    for s in schedule_fixed:
        print(
            f"  * Mode {s['mode']}: steps [{s['start']:3d} .. {s['end']:3d}] "
            f"(duration = {s['dur']:2d} steps) | Target Speed = {s['v_target']:5.2f} m/s"
        )

    gen_fn = lambda seed: generate_truly_random_trajectory(
        total_horizon=args.horizon, min_phase_len=25, seed=seed
    )

    print(f"\n=== 1. Training RL Way 1: MPC-Horizon Rollout on Paper Linear Approx (Batch-{args.regen_every}) ===")
    rl_way1, hist_w1, _ = train_rl_switch_strict_two_ways(
        gen_fn,
        rl_way="paper_linear_only",
        traj_mode="batch_random",
        regen_every_n_epochs=args.regen_every,
        epochs=args.epochs,
        lookahead_n=args.lookahead_n,
        use_hybrid_mpc=False,
        seed=args.seed,
    )
    rl_way1.eval()

    print(f"\n=== 2. Training RL Way 2: MPC-Horizon Rollout on {args.emulator_steps}-Step Emulator (Batch-{args.regen_every}) ===")
    rl_way2, hist_w2, _ = train_rl_switch_strict_two_ways(
        gen_fn,
        rl_way="emulator_150_only",
        traj_mode="batch_random",
        regen_every_n_epochs=args.regen_every,
        epochs=args.epochs,
        emulator_steps=args.emulator_steps,
        lookahead_n=args.lookahead_n,
        use_hybrid_mpc=False,
        seed=args.seed,
    )
    rl_way2.eval()

    print(f"\n=== 3. Training Idea A+C Hybrid RL-IPID-MPC Switch (5 Actions: GA, PSO, DE, ICA, MPC) ===")
    rl_hybrid, hist_hyb, _ = train_rl_switch_strict_two_ways(
        gen_fn,
        rl_way="emulator_150_only",
        traj_mode="batch_random",
        regen_every_n_epochs=args.regen_every,
        epochs=args.epochs,
        emulator_steps=args.emulator_steps,
        lookahead_n=args.lookahead_n,
        use_hybrid_mpc=True,
        seed=args.seed,
    )
    rl_hybrid.eval()

    mpc_standard = PureLiuMPC(n_p=10, n_c=5, q1=120.0, q2=0.05, q3=0.25, adaptive_shift=False)
    mpc_idea_c = PureLiuMPC(n_p=10, n_c=5, q1=120.0, q2=0.05, q3=0.25, adaptive_shift=True)

    # ---------------- TEST 1: Reference Trajectory ----------------
    test1_results = [
        run_controller_benchmark("GA-PID (Fixed)", v_des_fixed, "FIXED_0", lookahead_n=args.lookahead_n),
        run_controller_benchmark("PSO-PID (Fixed)", v_des_fixed, "FIXED_1", lookahead_n=args.lookahead_n),
        run_controller_benchmark("DE-PID (Fixed)", v_des_fixed, "FIXED_2", lookahead_n=args.lookahead_n),
        run_controller_benchmark("ICA-PID (Fixed)", v_des_fixed, "FIXED_3", lookahead_n=args.lookahead_n),
        run_controller_benchmark("Pure MPC (Standard Fixed Q3)", v_des_fixed, "MPC", mpc_ctrl=mpc_standard, lookahead_n=args.lookahead_n),
        run_controller_benchmark("Adaptive MPC (Idea C: Shift-Q3)", v_des_fixed, "MPC", mpc_ctrl=mpc_idea_c, lookahead_n=args.lookahead_n),
        run_controller_benchmark("Hardcoded-Switch IPID (Havaei)", v_des_fixed, "HARDCODED", lookahead_n=args.lookahead_n),
        run_controller_benchmark("RL-Switch Way 1 (Linear + Idea A)", v_des_fixed, "RL_SOFT", rl_policy=rl_way1, lookahead_n=args.lookahead_n),
        run_controller_benchmark("RL-Switch Way 2 (150-Emu + Idea A)", v_des_fixed, "RL_SOFT", rl_policy=rl_way2, lookahead_n=args.lookahead_n),
        run_controller_benchmark("Hybrid RL-IPID-MPC (Idea A + C)", v_des_fixed, "RL_HYBRID", rl_policy=rl_hybrid, lookahead_n=args.lookahead_n),
    ]

    print("\n" + "=" * 150)
    print(f"TEST 1: PERFORMANCE ON REFERENCE GENERATED [0, {args.horizon}] TRAJECTORY")
    print("-" * 150)
    print(
        f"{'Controller Method':<35} | {'MeanErr':>7} | {'BrakeErr':>8} | {'ShiftOS':>7} | "
        f"{'ShiftAvg':>8} | {'StopErr':>7} | {'PaperScore':>10} | {'StepInfer':>12} | {'Status':>6}"
    )
    print("-" * 150)
    for r in test1_results:
        print(
            f"{r['Method']:<35} | {r['Mean_Err']:7.4f} | {r['Brake_Err']:8.4f} | "
            f"{r['Shift_Overshoot']:7.4f} | {r['Shift_SettleErr']:8.4f} | {r['Stop_Err_m']:7.4f} | "
            f"{r['Paper_Score']:10.4f} | {r['Infer_Mean_us']:9.2f} us | {r['Passes_Benchmark']:>6}"
        )
    print("=" * 150)

    # ---------------- TEST 2: 30 Unseen Random Trajectories ----------------
    eval_specs = [
        ("ICA-PID (Fixed)", "FIXED_3", None, None),
        ("DE-PID (Fixed)", "FIXED_2", None, None),
        ("Pure MPC (Standard Fixed Q3)", "MPC", None, mpc_standard),
        ("Adaptive MPC (Idea C: Shift-Q3)", "MPC", None, mpc_idea_c),
        ("Hardcoded-Switch IPID (Havaei)", "HARDCODED", None, None),
        ("RL-Switch Way 1 (Linear + Idea A)", "RL_SOFT", rl_way1, None),
        ("RL-Switch Way 2 (150-Emu + Idea A)", "RL_SOFT", rl_way2, None),
        ("Hybrid RL-IPID-MPC (Idea A + C)", "RL_HYBRID", rl_hybrid, None),
    ]
    agg = {name: [] for name, _, _, _ in eval_specs}

    for ep in range(args.eval_runs):
        v_rand, _ = generate_truly_random_trajectory(
            total_horizon=args.horizon, min_phase_len=25, seed=80000 + ep
        )
        for name, ctype, pol, m_ctrl in eval_specs:
            res = run_controller_benchmark(
                name, v_rand, ctype, rl_policy=pol, mpc_ctrl=m_ctrl, lookahead_n=args.lookahead_n
            )
            agg[name].append(res)

    print("\n" + "=" * 150)
    print(f"TEST 2: GENERALIZATION ACROSS {args.eval_runs} UNSEEN RANDOM [0, {args.horizon}] TRAJECTORIES (MODES 1..6)")
    print("-" * 150)
    print(
        f"{'Controller Method':<35} | {'MeanErr (m/s)':>15} | {'BrakeErr':>8} | {'ShiftOS':>7} | "
        f"{'ShiftAvg':>8} | {'StopErr':>7} | {'PaperScore':>10} | {'StepInfer':>12} | {'Status':>6}"
    )
    print("-" * 150)
    for name, _, _, _ in eval_specs:
        runs = agg[name]
        m_err = float(np.mean([r["Mean_Err"] for r in runs]))
        s_err = float(np.std([r["Mean_Err"] for r in runs]))
        b_err = float(np.mean([r["Brake_Err"] for r in runs]))
        os_val = float(np.mean([r["Shift_Overshoot"] for r in runs]))
        sa_val = float(np.mean([r["Shift_SettleErr"] for r in runs]))
        st_val = float(np.mean([r["Stop_Err_m"] for r in runs]))
        p_score = float(np.mean([r["Paper_Score"] for r in runs]))
        lat_us = float(np.mean([r["Infer_Mean_us"] for r in runs]))
        status = "PASS" if all(r["Passes_Benchmark"] == "PASS" for r in runs) else "FAIL"
        print(
            f"{name:<35} | {m_err:6.4f} ± {s_err:6.4f} | {b_err:8.4f} | {os_val:7.4f} | "
            f"{sa_val:8.4f} | {st_val:7.4f} | {p_score:10.4f} | {lat_us:9.2f} us | {status:>6}"
        )
    print("=" * 150)

    # ---------------- Save 6-Panel Plot ----------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    k_grid = test1_results[0]["traj"]["k"]
    v_des_ref = test1_results[0]["traj"]["v_des"]

    axes[0, 0].plot(k_grid, v_des_ref, "k--", linewidth=2.0, label="Target Profile")
    for idx_res in [4, 5, 6, 8, 9]:
        axes[0, 0].plot(k_grid, test1_results[idx_res]["traj"]["v"], linewidth=1.4, label=test1_results[idx_res]["Method"])
    axes[0, 0].set_title("1. Velocity Tracking on Random Multi-Mode [0, 400] Track")
    axes[0, 0].set_xlabel("Distance Sample")
    axes[0, 0].set_ylabel("Speed (m/s)")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    for idx_res in [4, 5, 6, 8, 9]:
        axes[0, 1].plot(k_grid, test1_results[idx_res]["traj"]["err"], linewidth=1.3, label=test1_results[idx_res]["Method"])
    axes[0, 1].set_title("2. Instantaneous Tracking Error |v - v_des|")
    axes[0, 1].set_xlabel("Distance Sample")
    axes[0, 1].set_ylabel("Error (m/s)")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    shift_seg = next((s for s in schedule_fixed if s["mode"] in (2, 5)), schedule_fixed[1])
    z0 = max(0, shift_seg["start"] - 5)
    z1 = min(len(k_grid), shift_seg["start"] + 15)
    sl = slice(z0, z1)
    axes[0, 2].plot(k_grid[sl], v_des_ref[sl], "k--", linewidth=2.2, label="Target Shift")
    for idx_res in [4, 5, 6, 8, 9]:
        axes[0, 2].plot(k_grid[sl], test1_results[idx_res]["traj"]["v"][sl], marker="o", markersize=3, label=test1_results[idx_res]["Method"])
    axes[0, 2].set_title("3. Zoom on Profile Shift Jump (Standard MPC vs Idea C vs RL)")
    axes[0, 2].set_xlabel("Distance Sample")
    axes[0, 2].legend(fontsize=7)
    axes[0, 2].grid(True, alpha=0.3)

    map_idx = {name: i for i, name in enumerate(HYBRID_ALGORITHMS)}
    hc_algs = [map_idx[a] for a in test1_results[6]["traj"]["alg"]]
    hyb_algs = [map_idx[a] for a in test1_results[9]["traj"]["alg"]]
    axes[1, 0].step(k_grid, hc_algs, "b:", where="mid", alpha=0.7, label="Hardcoded IPID")
    axes[1, 0].step(k_grid, hyb_algs, "r-", where="mid", alpha=0.85, label="Hybrid RL-IPID-MPC (Idea A+C)")
    axes[1, 0].set_yticks([0, 1, 2, 3, 4])
    axes[1, 0].set_yticklabels(list(HYBRID_ALGORITHMS))
    axes[1, 0].set_title("4. Active Controller Selection Along Track")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    names_bar = ["ICA", "DE", "Std MPC", "IdeaC MPC", "IPID", "RL W1", "RL W2", "Hybrid RL"]
    scores_bar = [float(np.mean([r["Paper_Score"] for r in agg[spec[0]]])) for spec in eval_specs]
    axes[1, 1].bar(names_bar, scores_bar, color=["gray", "gray", "orange", "darkorange", "steelblue", "mediumpurple", "seagreen", "crimson"])
    axes[1, 1].set_title("5. Test 2 Composite Score Across 30 Unseen Tracks")
    axes[1, 1].set_ylabel("Paper Score (Lower = Better)")
    axes[1, 1].tick_params(axis="x", labelsize=7.5)
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    lats = [
        test1_results[5]["Infer_Mean_us"],
        test1_results[6]["Infer_Mean_us"],
        test1_results[8]["Infer_Mean_us"],
        test1_results[9]["Infer_Mean_us"],
    ]
    axes[1, 2].bar(["IdeaC MPC", "IPID", "RL Way 2", "Hybrid RL"], lats, color=["darkorange", "steelblue", "seagreen", "crimson"])
    axes[1, 2].axhline(MAX_ACCEPTABLE_STEP_US, color="k", linestyle="--", label="500 us Limit")
    axes[1, 2].set_title("6. Online Inference Latency (us/step)")
    axes[1, 2].set_ylabel("Microseconds")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"\nSaved 6-panel benchmark plot to: {args.plot}")


if __name__ == "__main__":
    main()