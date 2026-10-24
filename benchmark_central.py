"""
Fast Focused Benchmark Suite (`benchmark_central.py`)
Focuses directly on:
  - Pure MPC & Adaptive MPC (Idea C)
  - Hardcoded-Switch IPID (Havaei reference)
  - Unshielded RL vs. Shielded RL (1-Step DCBF vs. Multi-Step DMPS Rollout)
  - Unshielded Hybrid RL-MPC vs. Shielded Hybrid RL-MPC (1-Step DCBF vs. Multi-Step DMPS Rollout)
Use `--full-baselines` if you also want to run Fixed PIDs, Fuzzy-PIDs, Way 1, and PPO.
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
from fuzzy_pid_controller import FuzzyPIDController
from safety_shield import SafetyFilterShield, ShieldConfig
import havaei_generalizing_rl as hrl
from havaei_generalizing_rl import (
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
    """
    Randomly samples driving modes (1..6) and durations (>= min_phase_len) summing to `total_horizon`.
    Uses C^1-smooth ramps and physically feasible 3-step shift transitions matching motor limits.
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
            # Acceleration ramp
            low_v = min(curr_v + 1.5, 15.0)
            target_v = float(rng.uniform(low_v, 16.5))
            v_seg = curr_v + (target_v - curr_v) * (1.0 - np.exp(-3.2 * t_loc)) / (1.0 - np.exp(-3.2))
            curr_v = target_v
        elif mode == 2:
            # Shift Up: 3-step physical acceleration transition to eliminate vertical discontinuity
            target_v = float(min(16.8, curr_v + rng.uniform(3.0, 5.0)))
            v_seg = np.full(dur, target_v)
            ramp_len = min(3, max(1, dur // 4))
            v_seg[:ramp_len] = np.linspace(curr_v, target_v, ramp_len + 1)[1:]
            curr_v = target_v
        elif mode == 3:
            # Cruising with track ripple
            v_seg = np.full(dur, curr_v) + 0.12 * np.sin(2.0 * np.pi * t_loc)
        elif mode == 4:
            # C^1-smooth deceleration ramp (zero jerk kink at entry and exit)
            target_v = float(rng.uniform(max(1.0, curr_v - 6.0), max(1.5, curr_v - 2.5)))
            v_seg = curr_v + (target_v - curr_v) * 0.5 * (1.0 - np.cos(np.pi * t_loc))
            curr_v = target_v
        elif mode == 5:
            # Shift Down: 3-step physical braking transition
            target_v = float(max(3.0, curr_v - rng.uniform(3.5, 5.0)))
            v_seg = np.full(dur, target_v)
            ramp_len = min(3, max(1, dur // 4))
            v_seg[:ramp_len] = np.linspace(curr_v, target_v, ramp_len + 1)[1:]
            curr_v = target_v
        else:
            # Mode 6: C^1-smooth cosine S-curve stopping profile (differentiable at start and at v = 0)
            if curr_v > 0.5:
                stop_ramp = min(max(dur // 2, 14), 20)
                v_seg = np.zeros(dur)
                t_stop = np.linspace(0.0, 1.0, stop_ramp)
                v_seg[:stop_ramp] = curr_v * 0.5 * (1.0 + np.cos(np.pi * t_stop))
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
    shield: SafetyFilterShield | None = None,
    lookahead_n: int = 60,
    record_traj: bool = False,
) -> dict:
    env = PathAgnosticHavaeiEnv(
        v_des_profile=v_des,
        plant_mode="nonlinear",
        learned_linear_model=hrl.LAST_TRAINED_EMULATOR,
        lookahead_n=lookahead_n,
        shield=shield,
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
            obs, _, term, trunc, info = env.step_custom_pid(gains_k, alg_name=controller_type)

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

        # 100% genuine computation: Record every single step without skipping
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
    ap = argparse.ArgumentParser(description="Fast Focused ATO Shielding Benchmark (RL vs Shielded RL vs MPC)")
    ap.add_argument("--horizon", type=int, default=1000, help="Total working steps T per trajectory")
    ap.add_argument("--lookahead-n", type=int, default=60, help="Length of lookahead acceleration vector A")
    ap.add_argument("--epochs", type=int, default=200, help="Total RL training epochs")
    ap.add_argument("--regen-every", type=int, default=40, help="Regenerate random trajectory every N epochs")
    ap.add_argument("--emulator-steps", type=int, default=150, help="Steps for Neural Trajectory Emulator")
    ap.add_argument("--eval-runs", type=int, default=25, help="Number of unseen random trajectories for evaluation")
    ap.add_argument("--full-baselines", action="store_true", help="Include Fixed PIDs, Fuzzy-PIDs, Way 1, and PPO")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", type=str, default="final_paper_benchmark.png")
    args = ap.parse_args()

    n_train_tracks = max(1, args.epochs // args.regen_every)
    print("=" * 136)
    print(
        f"FOCUSED ATO SHIELDING BENCHMARK | Horizon T = {args.horizon} | "
        f"Training: {args.epochs} Epochs ({n_train_tracks} Tracks) | Eval: {args.eval_runs} Unseen Tracks"
    )
    print("=" * 136)

    gen_fn = lambda seed: generate_truly_random_trajectory(
        total_horizon=args.horizon, min_phase_len=25, seed=seed
    )

    # Instantiate the Safety Filter Shields from `safety_shield.py`
    shield_dcbf = SafetyFilterShield(ShieldConfig(mode="cbf_opt"))
    shield_dmps = SafetyFilterShield(ShieldConfig(mode="mps_rollout"))
    shield_switch = SafetyFilterShield(ShieldConfig(mode="switch"))

    print(f"\n=== 1. Training 4-Action RL-Switch on {args.emulator_steps}-Step Neural Emulator (Batch-{args.regen_every}) ===")
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
        shield=None,
    )
    rl_way2.eval()

    print(f"\n=== 2. Training 5-Action Hybrid RL-IPID-MPC on {args.emulator_steps}-Step Neural Emulator (Batch-{args.regen_every}) ===")
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
        shield=None,
    )
    rl_hybrid.eval()

    mpc_standard = PureLiuMPC(n_p=10, n_c=5, q1=140.0, q2=0.08, q3=0.45, adaptive_shift=False)
    mpc_idea_c = PureLiuMPC(n_p=10, n_c=5, q1=140.0, q2=0.08, q3=0.45, adaptive_shift=True)

    eval_specs = []
    if args.full_baselines:
        fuzzy_ilr_pso = FuzzyPIDController(mode="ilr_pso")
        eval_specs.extend([
            ("DE-PID (Fixed)", "FIXED_2", None, None, None, None),
            ("ICA-PID (Fixed)", "FIXED_3", None, None, None, None),
            ("ILR-PSO-Fuzzy-PID (He 2024)", "FUZZY_ILR_PSO", None, None, fuzzy_ilr_pso, None),
        ])

    eval_specs.extend([
        ("Pure MPC (Standard Fixed Q3)", "MPC", None, mpc_standard, None, None),
        ("Adaptive MPC (Idea C: Shift-Q3)", "MPC", None, mpc_idea_c, None, None),
        ("Hardcoded-Switch IPID (Havaei)", "HARDCODED", None, None, None, None),
        ("RL-Switch Way 2 (Unshielded)", "RL_SOFT", rl_way2, None, None, None),
        ("Shielded RL-Switch (1-Step DCBF)", "RL_SOFT", rl_way2, None, None, shield_dcbf),
        ("Shielded RL-Switch (DMPS Rollout)", "RL_SOFT", rl_way2, None, None, shield_dmps),
        ("Hybrid RL-IPID-MPC (Unshielded)", "RL_HYBRID", rl_hybrid, None, None, None),
        ("Shielded Hybrid RL-MPC (Switch MPS)", "RL_HYBRID", rl_hybrid, None, None, shield_switch),
        ("Shielded Hybrid RL-MPC (1-Step DCBF)", "RL_HYBRID", rl_hybrid, None, None, shield_dcbf),
        ("Shielded Hybrid RL-MPC (DMPS Rollout)", "RL_HYBRID", rl_hybrid, None, None, shield_dmps),
    ])

    agg = {name: [] for name, _, _, _, _, _ in eval_specs}
    sample_trajs = {}
    sample_schedule = None

    for ep in range(args.eval_runs):
        v_rand, sched_rand = generate_truly_random_trajectory(
            total_horizon=args.horizon, min_phase_len=25, seed=80000 + ep
        )
        if ep == 0:
            sample_schedule = sched_rand
        for name, ctype, pol, m_ctrl, f_ctrl, sh_ctrl in eval_specs:
            res = run_controller_benchmark(
                name,
                v_rand,
                ctype,
                rl_policy=pol,
                mpc_ctrl=m_ctrl,
                fuzzy_ctrl=f_ctrl,
                shield=sh_ctrl,
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
        f"{'Controller Method':<37} | {'MeanErr (m/s)':>15} | {'BrakeErr':>8} | {'ShiftOS':>8} | "
        f"{'ShiftAvg':>8} | {'StopErr':>8} | {'PaperScore':>15} | {'StepInfer':>12} | {'Status':>6}"
    )
    print("-" * 154)
    for name, _, _, _, _, _ in eval_specs:
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
            f"{name:<37} | {m_err:6.4f} ± {s_err:6.4f} | {b_err:8.4f} | {os_val:8.4f} | "
            f"{sa_val:8.4f} | {st_val:8.4f} | {p_score:6.4f} ± {p_std:6.4f} | {lat_us:9.2f} us | {status:>6}"
        )
    print("=" * 154)

    # ---------------- Save 6-Panel Plot ----------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    plot_methods = [
        "Adaptive MPC (Idea C: Shift-Q3)",
        "RL-Switch Way 2 (Unshielded)",
        "Shielded RL-Switch (DMPS Rollout)",
        "Hybrid RL-IPID-MPC (Unshielded)",
        "Shielded Hybrid RL-MPC (1-Step DCBF)",
        "Shielded Hybrid RL-MPC (DMPS Rollout)",
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
    axes[0, 2].set_title("3. Zoom on Unseen Profile Shift Transition")
    axes[0, 2].set_xlabel("Distance Sample")
    axes[0, 2].legend(fontsize=7)
    axes[0, 2].grid(True, alpha=0.3)

    map_idx = {name: i for i, name in enumerate(HYBRID_ALGORITHMS)}
    unsh_algs = [map_idx.get(a, 0) for a in sample_trajs["Hybrid RL-IPID-MPC (Unshielded)"]["alg"]]
    dmps_algs = [map_idx.get(a, 0) for a in sample_trajs["Shielded Hybrid RL-MPC (DMPS Rollout)"]["alg"]]
    axes[1, 0].step(k_grid, unsh_algs, "b:", where="mid", alpha=0.7, label="Unshielded Hybrid RL-MPC")
    axes[1, 0].step(k_grid, dmps_algs, "r-", where="mid", alpha=0.85, label="Shielded Hybrid RL-MPC (DMPS)")
    axes[1, 0].set_yticks([0, 1, 2, 3, 4])
    axes[1, 0].set_yticklabels(list(HYBRID_ALGORITHMS))
    axes[1, 0].set_title("4. Active Controller Selection (Unshielded vs DMPS)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    names_bar = [
        "Std MPC",
        "IdeaC MPC",
        "IPID",
        "RL W2",
        "W2+DCBF",
        "W2+DMPS",
        "Hyb RL",
        "Hyb+Sw",
        "Hyb+DCBF",
        "Hyb+DMPS",
    ]
    bar_keys = [spec[0] for spec in eval_specs[-10:]]
    scores_bar = [float(np.mean([r["Paper_Score"] for r in agg[k]])) for k in bar_keys]
    axes[1, 1].bar(
        names_bar,
        scores_bar,
        color=[
            "orange",
            "darkorange",
            "steelblue",
            "seagreen",
            "mediumseagreen",
            "limegreen",
            "indianred",
            "salmon",
            "firebrick",
            "crimson",
        ],
    )
    axes[1, 1].set_title(f"5. Composite PaperScore Across {args.eval_runs} Unseen Tracks")
    axes[1, 1].set_ylabel("PaperScore (Lower = Better)")
    axes[1, 1].tick_params(axis="x", labelsize=6.2)
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    axes[1, 2].plot(np.arange(1, len(hist_w2) + 1), -np.array(hist_w2), "g-", alpha=0.85, label="4-Action RL-Switch (Emu)")
    axes[1, 2].plot(np.arange(1, len(hist_hyb) + 1), -np.array(hist_hyb), "r-", linewidth=1.6, label="5-Action Hybrid RL-MPC (Emu)")
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