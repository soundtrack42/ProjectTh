"""
Standalone Verification & Plot (`test_emulator_150_vs_linear.py`):
Tests two core questions:
  1. Does 150 steps of gradient learning on a trajectory (`TrajectoryEmulatorNet`)
     reconstruct the true nonlinear train trajectory (Eq. 6-10) more accurately than
     the paper's fixed linear transfer-function approximation (Eq. 11)?
  2. When we train the RL Switch ONLY on the 150-step Emulator vs. ONLY on the Paper's
     Linear Approximation, which one performs better on the true nonlinear track
     (especially on ShiftOS and trajectory tracking error)?
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from havaei_nonlinear_two_phase import (
    ALGORITHMS,
    ALG_TO_IDX,
    HavaeiTwoPhaseEnv,
    HardcodedSwitchController,
    LearnedLinearizationModel,
    TwoPhaseRLSwitchPolicy,
)
from benchmark_central import generate_phased_trajectory, run_controller_benchmark
from havaei_generalizing_rl import learn_150_step_emulator_for_trajectory, train_rl_switch_strict_two_ways


def evaluate_model_approximation_error(v_des: np.ndarray, emulator: LearnedLinearizationModel) -> dict:
    """
    Runs all 4 heuristic PID algorithms [GA, PSO, DE, ICA] and random switches across
    the trajectory `v_des` and measures the exact step-by-step prediction error:
      - Error of Paper's Fixed Linear Approximation (`linear_paper`, Eq. 11) vs True Nonlinear Plant
      - Error of 150-Step Learned Trajectory Emulator (`linear_learned`) vs True Nonlinear Plant
    """
    env_true = HavaeiTwoPhaseEnv(v_des_profile=v_des, plant_mode="nonlinear")
    env_lin = HavaeiTwoPhaseEnv(v_des_profile=v_des, plant_mode="linear_paper")
    env_emu = HavaeiTwoPhaseEnv(v_des_profile=v_des, plant_mode="linear_learned", learned_linear_model=emulator)

    lin_errs, emu_errs = [], []
    v_true_log, v_lin_log, v_emu_log = [], [], []

    for mode_run in range(5):
        env_true.reset()
        done = False
        while not done:
            # Sync states so we test the exact 1-step-ahead trajectory prediction from the same state
            env_lin.k = env_emu.k = env_true.k
            env_lin.x_pos = env_emu.x_pos = env_true.x_pos
            env_lin.v_real = env_emu.v_real = env_true.v_real
            env_lin.v_prev = env_emu.v_prev = env_true.v_prev
            env_lin.shift_timer = env_emu.shift_timer = env_true.shift_timer
            env_lin.shift_delta = env_emu.shift_delta = env_true.shift_delta

            act = mode_run if mode_run < 4 else int((env_true.k // 15) % 4)
            _, _, term, trunc, info_true = env_true.step(act)
            _, _, _, _, info_lin = env_lin.step(act)
            _, _, _, _, info_emu = env_emu.step(act)
            done = term or trunc

            v_t = info_true["v_real"]
            v_l = info_lin["v_real"]
            v_e = info_emu["v_real"]

            lin_errs.append(abs(v_t - v_l))
            emu_errs.append(abs(v_t - v_e))

            if mode_run == 3:  # Log ICA rollout for plotting
                v_true_log.append(v_t)
                v_lin_log.append(v_l)
                v_emu_log.append(v_e)

    lin_arr = np.array(lin_errs)
    emu_arr = np.array(emu_errs)
    return {
        "lin_mae": float(np.mean(lin_arr)),
        "lin_rmse": float(np.sqrt(np.mean(lin_arr ** 2))),
        "lin_max": float(np.max(lin_arr)),
        "emu_mae": float(np.mean(emu_arr)),
        "emu_rmse": float(np.sqrt(np.mean(emu_arr ** 2))),
        "emu_max": float(np.max(emu_arr)),
        "v_true": np.array(v_true_log),
        "v_lin": np.array(v_lin_log),
        "v_emu": np.array(v_emu_log),
    }


def main():
    print("=" * 110)
    print("EXPERIMENT: Does 150 Steps of Trajectory Emulation Beat the Paper's Fixed Linear Approximation?")
    print("=" * 110)

    v_des, schedule = generate_phased_trajectory(total_horizon=400, seed=42)
    gen_fn = lambda seed: (v_des, schedule)

    # 1. Train the 150-step emulator on this trajectory
    t0 = time.perf_counter()
    emulator, loss_curve = learn_150_step_emulator_for_trajectory(
        v_des, emulator=None, emulator_steps=150, verbose=True, return_curve=True
    )
    t_emu = time.perf_counter() - t0

    # 2. Compare Model Approximation Accuracy against True Nonlinear Formula (Eq. 6-10)
    approx_stats = evaluate_model_approximation_error(v_des, emulator)
    print("\n" + "-" * 110)
    print("PART A: TRAJECTORY DYNAMICS APPROXIMATION ACCURACY (vs. True Nonlinear Formula Eq. 6-10)")
    print("-" * 110)
    print(
        f"  * Paper Fixed Linear Approx (Eq. 11)   -> MAE: {approx_stats['lin_mae']:.5f} m/s | "
        f"RMSE: {approx_stats['lin_rmse']:.5f} m/s | Max Error: {approx_stats['lin_max']:.5f} m/s"
    )
    print(
        f"  * 150-Step Learned Trajectory Emulator -> MAE: {approx_stats['emu_mae']:.5f} m/s | "
        f"RMSE: {approx_stats['emu_rmse']:.5f} m/s | Max Error: {approx_stats['emu_max']:.5f} m/s "
        f"(Trained in {t_emu:.2f}s)"
    )
    improvement = (1.0 - approx_stats["emu_rmse"] / max(1e-8, approx_stats["lin_rmse"])) * 100.0
    print(f"  => 150-Step Emulator reduces trajectory approximation error by {improvement:.1f}% over Paper Linear Approx!")
    print("-" * 110)

    # 3. Train RL Switch ONLY on Linear Approx vs ONLY on 150-Step Emulator
    print("\nPART B: Training RL Switch on Paper Linear Approx vs. 150-Step Trajectory Emulator (200 Epochs)...")
    rl_lin, _, _ = train_rl_switch_strict_two_ways(
        gen_fn, rl_way="paper_linear_only", traj_mode="fixed_generated", epochs=200, seed=42
    )
    rl_emu, _, _ = train_rl_switch_strict_two_ways(
        gen_fn, rl_way="emulator_150_only", traj_mode="fixed_generated", epochs=200, emulator_steps=150, seed=42
    )

    res_ipid = run_controller_benchmark("Hardcoded-Switch IPID (Havaei)", v_des, "HARDCODED")
    res_rl_lin = run_controller_benchmark("RL-Switch (Trained on Linear Approx)", v_des, "RL_SOFT", rl_policy=rl_lin)
    res_rl_emu = run_controller_benchmark("RL-Switch (Trained on 150-Step Emu)", v_des, "RL_SOFT", rl_policy=rl_emu)

    print("\n" + "=" * 115)
    print("PART B RESULTS: CLOSED-LOOP TRAJECTORY TRACKING ON TRUE NONLINEAR PLANT")
    print("-" * 115)
    print(
        f"{'Controller Method':<38} | {'MeanErr(m/s)':>12} | {'BrakeErr':>9} | "
        f"{'ShiftOS(m/s)':>12} | {'ShiftAvg':>9} | {'StopErr(m)':>10} | {'PaperScore':>10}"
    )
    print("-" * 115)
    for r in [res_ipid, res_rl_lin, res_rl_emu]:
        print(
            f"{r['Method']:<38} | "
            f"{r['Mean_Err']:12.4f} | "
            f"{r['Brake_Err']:9.4f} | "
            f"{r['Shift_Overshoot']:12.4f} | "
            f"{r['Shift_SettleErr']:9.4f} | "
            f"{r['Stop_Err_m']:10.4f} | "
            f"{r['Paper_Score']:10.4f}"
        )
    print("=" * 115)

    # Save diagnostic plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(np.arange(1, len(loss_curve) + 1), loss_curve, "r-", linewidth=1.8, label="150-Step Emulator RMSE")
    axes[0].axhline(approx_stats["lin_rmse"], color="b", linestyle="--", label="Paper Linear Approx RMSE (Eq. 11)")
    axes[0].set_title("1. 150-Step Emulator Learning vs Paper Linear Approx")
    axes[0].set_xlabel("Gradient Step (1..150)")
    axes[0].set_ylabel("Prediction RMSE (m/s)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    k_ax = np.arange(len(approx_stats["v_true"]))
    axes[1].plot(k_ax, np.abs(approx_stats["v_true"] - approx_stats["v_lin"]), "b--", alpha=0.8, label="|True - Paper Linear|")
    axes[1].plot(k_ax, np.abs(approx_stats["v_true"] - approx_stats["v_emu"]), "r-", alpha=0.9, label="|True - 150-Step Emulator|")
    axes[1].set_title("2. Model Prediction Error Along [0, 400] Track")
    axes[1].set_xlabel("Distance Sample")
    axes[1].set_ylabel("1-Step Velocity Prediction Error (m/s)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(res_ipid["traj"]["k"], res_ipid["traj"]["err"], "k:", linewidth=1.5, label="Hardcoded IPID")
    axes[2].plot(res_rl_lin["traj"]["k"], res_rl_lin["traj"]["err"], "b--", linewidth=1.4, label="RL (Paper Linear Approx)")
    axes[2].plot(res_rl_emu["traj"]["k"], res_rl_emu["traj"]["err"], "r-", linewidth=1.5, label="RL (150-Step Emulator)")
    axes[2].set_title("3. Resulting Closed-Loop Tracking Error |v - v_des|")
    axes[2].set_xlabel("Distance Sample")
    axes[2].set_ylabel("Tracking Error (m/s)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("emulator_150_vs_linear.png", dpi=150)
    print("\nSaved diagnostic comparison plot to emulator_150_vs_linear.png")


if __name__ == "__main__":
    main()