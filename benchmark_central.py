"""
Centralized Multi-Trajectory Benchmark Suite (`benchmark_central.py`)
Compares:
  - Fixed Heuristic PIDs (GA, PSO, DE, ICA)
  - Fuzzy-PID (Standard Mamdani) & ILR-PSO-Fuzzy-PID (He et al., 2024)
  - Pure MPC & Adaptive MPC (Idea C)
  - Hardcoded-Switch IPID (Havaei & Sandidzadeh)
  - RL-Switch (PPO, Way 1 Linear, Way 2 Neural Emulator) & Hybrid RL-IPID-MPC
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
from fuzzy_pid_controller import FuzzyPIDController, OPTIMIZED_ILR_PSO_FUZZY_PARAMS
from havaei_generalizing_rl import (
    ALGORITHMS,
    ALG_TO_IDX,
    HYBRID_ALGORITHMS,
    COMPUTED_PID_BANK,
    HardcodedSwitchController,
    PathAgnosticHavaeiEnv,
    train_rl_switch_strict_two_ways,
)

MAX_ACCEPTABLE_EPISODE_MS = 1000.0
MAX_ACCEPTABLE_STEP_US = 500.0


def generate_truly_random_trajectory(
    total_horizon: int = 1000,
    min_phase_len: int = 25,
    seed: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
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
    fuzzy_ctrl: FuzzyPIDController | None = None,
    lookahead_n: int = 60,
    record_traj: bool = False,
) -> dict:
    env = PathAgnosticHavaeiEnv(
        v_des_profile=v_des,
        plant_mode="nonlinear",
        lookahead_n=lookahead_n,
    )
    obs = env.reset()
    done = False

    hc = HardcodedSwitchController(use_transient_chain=True)
    hc.reset()
    if mpc_ctrl is not None:
        mpc_ctrl.reset()
    if fuzzy_ctrl is not None:
        fuzzy_ctrl.reset()

    k_log, v_log, v_des_log, err_log, os_log, shift_err_log, brake_err_log, alg_log = [], [], [], [], [], [], [], []
    step_latencies_us = []

    while not done:
        t0 = time.perf_counter()

        if controller_type == "MPC":
            assert mpc_ctrl is not None
            k_curr = env.k
            x_sample = (k_curr + 1) * env.d
            ref_h = env.v_des[k_curr + 1 : min(k_curr + 1 + mpc_ctrl.n_p, len(env.v_des))]
            slope_iota = env.physics.grade_resistance(x_sample) / (env.physics.M * env.physics.g)
            curve_iota = env.physics.curve_resistance(x_sample) / (env.physics.M * env.physics.g)
            u_opt = mpc_ctrl.select_action(
                env.v_real, ref_h, slope_iota=slope_iota + curve_iota, dt_step=env.dt
            )
            t1 = time.perf_counter()
            obs, _, term, trunc, info = env.step_mpc(u_opt)

        elif controller_type in ("FUZZY_STD", "FUZZY_ILR_PSO"):
            if fuzzy_ctrl is None:
                fuzzy_ctrl = FuzzyPIDController(mode="ilr_pso" if controller_type == "FUZZY_ILR_PSO" else "standard")
            gains_k = fuzzy_ctrl.compute_gains(env)
            t1 = time.perf_counter()
            if hasattr(env, "step_custom_pid"):
                obs, _, term, trunc, info = env.step_custom_pid(gains_k, alg_name=controller_type)
            else:
                # Fallback if step_custom_pid was not added to PathAgnosticHavaeiEnv
                env.pid_bank["GA"] = gains_k
                obs, _, term, trunc, info = env.step(0)

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

        err_log.append(info["abs_err"])
        os_log.append(info["overshoot"])
        if info.get("shift_active", False):
            shift_err_log.append(info["abs_err"])
        if info.get("braking_active", False):
            brake_err_log.append(info["abs_err"])

        if record_traj:
            k_log.append(info["k"])
            v_log.append(info["v_real"])
            v_des_log.append(info["v_des"])
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

    res = {
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
    }
    if record_traj:
        res["traj"] = {
            "k": np.array(k_log),
            "v": np.array(v_log),
            "v_des": np.array(v_des_log),
            "err": err_arr,
            "alg": alg_log,
        }
    return res


def main():
    ap = argparse.ArgumentParser(description="Unified Multi-Trajectory ATO Benchmark: RL vs IPID, Fuzzy-PID & MPC")
    ap.add_argument("--horizon", type=int, default=1000, help="Total working steps T per trajectory")
    ap.add_argument("--lookahead-n", type=int, default=60, help="Length of lookahead acceleration vector A")
    ap.add_argument("--epochs", type=int, default=240, help="Total RL training epochs")
    ap.add_argument("--regen-every", type=int, default=40, help="Regenerate random trajectory every N epochs")
    ap.add_argument("--emulator-steps", type=int, default=150, help="Steps for Neural Trajectory Emulator in Way 2")
    ap.add_argument("--eval-runs", type=int, default=30, help="Number of unseen random trajectories for evaluation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", type=str, default="central_benchmark_multi.png")
    args = ap.parse_args()

    n_train_tracks = max(1, args.epochs // args.regen_every)
    print("=" * 136)
    print(
        f"CENTRALIZED ATO BENCHMARK | Horizon T = {args.horizon} | Lookahead A = {args.lookahead_n} steps | "
        f"Training: {args.epochs} Epochs ({n_train_tracks} Sequential Random Tracks) | Eval: {args.eval_runs} Unseen Tracks"
    )
    print("=" * 136)
    print("Computed Offline Heuristic PID Bank (Cardano Step-Response Optimization):")
    for alg_k, gains_v in COMPUTED_PID_BANK.items():
        print(f"  * {alg_k:<4}: Kp = {gains_v[0]:6.2f}, Ki = {gains_v[1]:6.2f}, Kd = {gains_v[2]:6.2f}")
    p_fuz = OPTIMIZED_ILR_PSO_FUZZY_PARAMS
    print(
        f"  * ILR-PSO-Fuzzy-PID (He et al. 2024): Kp0={p_fuz[0]:.2f}, Ki0={p_fuz[1]:.2f}, Kd0={p_fuz[2]:.2f} | "
        f"u_p={p_fuz[3]:.2f}, u_i={p_fuz[4]:.2f}, u_d={p_fuz[5]:.2f}"
    )

    gen_fn = lambda seed: generate_truly_random_trajectory(
        total_horizon=args.horizon, min_phase_len=25, seed=seed
    )

    print(f"\n=== 1. Training RL Way 1: Surrogate-Horizon on Paper Linear ARX Model (Batch-{args.regen_every}) ===")
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

    print(f"\n=== 2. Training RL Way 2: Surrogate-Horizon on {args.emulator_steps}-Step Neural Emulator (Batch-{args.regen_every}) ===")
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

    print(f"\n=== 3. Training Model-Free Clipped PPO + GAE on Nonlinear Plant (Batch-{args.regen_every}) ===")
    rl_ppo, hist_ppo, _ = train_rl_switch_strict_two_ways(
        gen_fn,
        rl_way="ppo_nonlinear",
        traj_mode="batch_random",
        regen_every_n_epochs=args.regen_every,
        epochs=args.epochs,
        lookahead_n=args.lookahead_n,
        use_hybrid_mpc=False,
        seed=args.seed,
    )
    rl_ppo.eval()

    print(f"\n=== 4. Training Idea A+C Hybrid RL-IPID-MPC Switch (5 Actions: GA, PSO, DE, ICA, MPC) ===")
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

    mpc_standard = PureLiuMPC(n_p=10, n_c=5, q1=140.0, q2=0.08, q3=0.45, adaptive_shift=False)
    mpc_idea_c = PureLiuMPC(n_p=10, n_c=5, q1=140.0, q2=0.08, q3=0.45, adaptive_shift=True)
    fuzzy_std = FuzzyPIDController(mode="standard")
    fuzzy_ilr_pso = FuzzyPIDController(mode="ilr_pso")

    # ---------------- MULTI-TRAJECTORY INFERENCE BENCHMARK (N Unseen Tracks) ----------------
    eval_specs = [
        ("GA-PID (Fixed)", "FIXED_0", None, None, None),
        ("PSO-PID (Fixed)", "FIXED_1", None, None, None),
        ("DE-PID (Fixed)", "FIXED_2", None, None, None),
        ("ICA-PID (Fixed)", "FIXED_3", None, None, None),
        ("Fuzzy-PID (Standard Mamdani)", "FUZZY_STD", None, None, fuzzy_std),
        ("ILR-PSO-Fuzzy-PID (He 2024)", "FUZZY_ILR_PSO", None, None, fuzzy_ilr_pso),
        ("Pure MPC (Standard Fixed Q3)", "MPC", None, mpc_standard, None),
        ("Adaptive MPC (Idea C: Shift-Q3)", "MPC", None, mpc_idea_c, None),
        ("Hardcoded-Switch IPID (Havaei)", "HARDCODED", None, None, None),
        ("RL-Switch (Model-Free PPO)", "RL_SOFT", rl_ppo, None, None),
        ("RL-Switch Way 1 (Linear + Idea A)", "RL_SOFT", rl_way1, None, None),
        ("RL-Switch Way 2 (150-Emu + Idea A)", "RL_SOFT", rl_way2, None, None),
        ("Hybrid RL-IPID-MPC (Idea A + C)", "RL_HYBRID", rl_hybrid, None, None),
    ]
    agg = {name: [] for name, _, _, _, _ in eval_specs}
    sample_trajs = {}
    sample_schedule = None

    for ep in range(args.eval_runs):
        v_rand, sched_rand = generate_truly_random_trajectory(
            total_horizon=args.horizon, min_phase_len=25, seed=80000 + ep
        )
        if ep == 0:
            sample_schedule = sched_rand
        for name, ctype, pol, m_ctrl, f_ctrl in eval_specs:
            res = run_controller_benchmark(
                name,
                v_rand,
                ctype,
                rl_policy=pol,
                mpc_ctrl=m_ctrl,
                fuzzy_ctrl=f_ctrl,
                lookahead_n=args.lookahead_n,
                record_traj=(ep == 0),
            )
            agg[name].append(res)
            if ep == 0:
                sample_trajs[name] = res["traj"]

    print("\n" + "=" * 154)
    print(f"INFERENCE BENCHMARK ACROSS {args.eval_runs} UNSEEN RANDOM [0, {args.horizon}] TRAJECTORIES (MODES 1..6, MEAN ± STD)")
    print("-" * 154)
    print(
        f"{'Controller Method':<35} | {'MeanErr (m/s)':>15} | {'BrakeErr':>8} | {'ShiftOS':>8} | "
        f"{'ShiftAvg':>8} | {'StopErr':>8} | {'PaperScore':>15} | {'StepInfer':>12} | {'Status':>6}"
    )
    print("-" * 154)
    for name, _, _, _, _ in eval_specs:
        runs = agg[name]
        m_err = float(np.mean([r["Mean_Err"] for r in runs]))
        s_err = float(np.std([r["Mean_Err"] for r in runs]))
        b_err = float(np.mean([r["Brake_Err"] for r in runs]))
        os_val = float(np.mean([r["Shift_Overshoot"] for r in runs]))
        sa_val = float(np.mean([r["Shift_SettleErr"] for r in runs]))
        st_val = float(np.mean([r["Stop_Err_m"] for r in runs]))
        p_score = float(np.mean([r["Paper_Score"] for r in runs]))
        p_std = float(np.std([r["Paper_Score"] for r in runs]))
        lat_us = float(np.mean([r["Infer_Mean_us"] for r in runs]))
        status = "PASS" if all(r["Passes_Benchmark"] == "PASS" for r in runs) else "FAIL"
        print(
            f"{name:<35} | {m_err:6.4f} ± {s_err:6.4f} | {b_err:8.4f} | {os_val:8.4f} | "
            f"{sa_val:8.4f} | {st_val:8.4f} | {p_score:6.4f} ± {p_std:6.4f} | {lat_us:9.2f} us | {status:>6}"
        )
    print("=" * 154)

    # ---------------- Save 6-Panel Plot ----------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    plot_methods = [
        "ILR-PSO-Fuzzy-PID (He 2024)",
        "Adaptive MPC (Idea C: Shift-Q3)",
        "Hardcoded-Switch IPID (Havaei)",
        "RL-Switch (Model-Free PPO)",
        "RL-Switch Way 2 (150-Emu + Idea A)",
        "Hybrid RL-IPID-MPC (Idea A + C)",
    ]
    k_grid = sample_trajs[plot_methods[0]]["k"]
    v_des_ref = sample_trajs[plot_methods[0]]["v_des"]

    axes[0, 0].plot(k_grid, v_des_ref, "k--", linewidth=2.0, label="Target Profile")
    for m_name in plot_methods:
        axes[0, 0].plot(k_grid, sample_trajs[m_name]["v"], linewidth=1.3, label=m_name)
    axes[0, 0].set_title(f"1. Unseen Sample Trajectory [0, {args.horizon}] Tracking")
    axes[0, 0].set_xlabel("Distance Sample")
    axes[0, 0].set_ylabel("Speed (m/s)")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    for m_name in plot_methods:
        axes[0, 1].plot(k_grid, sample_trajs[m_name]["err"], linewidth=1.2, label=m_name)
    axes[0, 1].set_title("2. Instantaneous Tracking Error |v - v_des|")
    axes[0, 1].set_xlabel("Distance Sample")
    axes[0, 1].set_ylabel("Error (m/s)")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    shift_seg = next((s for s in sample_schedule if s["mode"] in (2, 5)), sample_schedule[1])
    z0 = max(0, shift_seg["start"] - 5)
    z1 = min(len(k_grid), shift_seg["start"] + 15)
    sl = slice(z0, z1)
    axes[0, 2].plot(k_grid[sl], v_des_ref[sl], "k--", linewidth=2.2, label="Target Shift")
    for m_name in plot_methods:
        axes[0, 2].plot(k_grid[sl], sample_trajs[m_name]["v"][sl], marker="o", markersize=3, label=m_name)
    axes[0, 2].set_title("3. Zoom on Unseen Profile Shift Jump")
    axes[0, 2].set_xlabel("Distance Sample")
    axes[0, 2].legend(fontsize=7)
    axes[0, 2].grid(True, alpha=0.3)

    map_idx = {name: i for i, name in enumerate(HYBRID_ALGORITHMS)}
    hc_algs = [map_idx.get(a, 0) for a in sample_trajs["Hardcoded-Switch IPID (Havaei)"]["alg"]]
    hyb_algs = [map_idx.get(a, 0) for a in sample_trajs["Hybrid RL-IPID-MPC (Idea A + C)"]["alg"]]
    axes[1, 0].step(k_grid, hc_algs, "b:", where="mid", alpha=0.7, label="Hardcoded IPID (4-Step Chain)")
    axes[1, 0].step(k_grid, hyb_algs, "r-", where="mid", alpha=0.85, label="Hybrid RL-IPID-MPC")
    axes[1, 0].set_yticks([0, 1, 2, 3, 4])
    axes[1, 0].set_yticklabels(list(HYBRID_ALGORITHMS))
    axes[1, 0].set_title("4. Active Controller Selection on Unseen Track")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    bar_specs = eval_specs[4:]
    names_bar = ["Fuzzy", "ILR-Fuzzy", "Std MPC", "IdeaC MPC", "IPID", "RL PPO", "RL W1", "RL W2", "Hybrid RL"]
    scores_bar = [float(np.mean([r["Paper_Score"] for r in agg[spec[0]]])) for spec in bar_specs]
    axes[1, 1].bar(
        names_bar,
        scores_bar,
        color=["teal", "lightseagreen", "orange", "darkorange", "steelblue", "dodgerblue", "mediumpurple", "seagreen", "crimson"],
    )
    axes[1, 1].set_title(f"5. Composite PaperScore Across {args.eval_runs} Unseen Tracks")
    axes[1, 1].set_ylabel("PaperScore (Lower = Better)")
    axes[1, 1].tick_params(axis="x", labelsize=6.8)
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    axes[1, 2].plot(np.arange(1, len(hist_w1) + 1), -np.array(hist_w1), "b--", alpha=0.8, label="RL Way 1 (Linear Surrogate)")
    axes[1, 2].plot(np.arange(1, len(hist_w2) + 1), -np.array(hist_w2), "g-", alpha=0.85, label="RL Way 2 (Neural Emulator)")
    axes[1, 2].plot(np.arange(1, len(hist_ppo) + 1), -np.array(hist_ppo), "c:", linewidth=1.5, label="RL Model-Free PPO")
    axes[1, 2].plot(np.arange(1, len(hist_hyb) + 1), -np.array(hist_hyb), "r-", linewidth=1.6, label="Hybrid RL-IPID-MPC")
    axes[1, 2].set_title(f"6. Step Tracking Cost Across {args.epochs} Epochs ({n_train_tracks} Tracks)")
    axes[1, 2].set_xlabel("Training Epoch")
    axes[1, 2].set_ylabel("Mean Step Cost (Lower = Better)")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"\nSaved 6-panel multi-trajectory benchmark plot to: {args.plot}")


if __name__ == "__main__":
    main()