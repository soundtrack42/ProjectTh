"""
Standalone Benchmark (`test_emulator_150_vs_linear.py`):
Directly imports the stabilized physics emulator and cumulative multi-track RL trainer
from `havaei_generalizing_rl.py` and evaluates:
  1. Part A: 1-Step and 6-Step Surrogate Horizon RMSE across [100, 200, 300] emulator steps.
  2. Part B: Generalization across N unseen random trajectories (default: 30) with Mean ± Std.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_central import (
    generate_truly_random_trajectory,
    run_controller_benchmark,
)
from havaei_generalizing_rl import (
    LearnedLinearizationModel,
    PathAgnosticHavaeiEnv,
    learn_150_step_emulator_for_trajectory,
    train_rl_switch_strict_two_ways,
)


def compute_error_stats(errs: np.ndarray) -> Dict[str, float]:
    if len(errs) == 0:
        return {"mae": 0.0, "rmse": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mae": float(np.mean(errs)),
        "rmse": float(np.sqrt(np.mean(errs ** 2))),
        "p95": float(np.percentile(errs, 95)),
        "max": float(np.max(errs)),
    }


def rel_improvement_pct(baseline: float, candidate: float) -> float:
    return (1.0 - candidate / max(1e-8, baseline)) * 100.0


def evaluate_model_approximation_error(
    v_des: np.ndarray,
    emulators_by_steps: Dict[int, LearnedLinearizationModel],
    surrogate_horizon: int = 6,
) -> Dict[str, Any]:
    env_true = PathAgnosticHavaeiEnv(v_des_profile=v_des, plant_mode="nonlinear")
    env_lin = PathAgnosticHavaeiEnv(v_des_profile=v_des, plant_mode="linear_paper")
    envs_emu = {
        s: PathAgnosticHavaeiEnv(v_des_profile=v_des, plant_mode="linear_learned", learned_linear_model=mod)
        for s, mod in emulators_by_steps.items()
    }

    lin_1step, lin_grade, lin_shift, lin_horizon = [], [], [], []
    emu_1step = {s: [] for s in emulators_by_steps}
    emu_grade = {s: [] for s in emulators_by_steps}
    emu_shift = {s: [] for s in emulators_by_steps}
    emu_horizon = {s: [] for s in emulators_by_steps}

    v_true_log, v_lin_1step_log, v_lin_hor_log = [], [], []
    v_emu_1step_log = {s: [] for s in emulators_by_steps}
    v_emu_hor_log = {s: [] for s in emulators_by_steps}
    ext_force_log = []

    for mode_run in range(6):
        env_true.reset()
        env_lin.reset()
        for e in envs_emu.values():
            e.reset()
        done = False

        while not done:
            snap = env_true.get_internal_state()
            env_lin.set_internal_state(snap)
            for e in envs_emu.values():
                e.set_internal_state(snap)

            act = mode_run if mode_run < 5 else int((env_true.k // 15) % 5)
            _, _, term, trunc, info_true = env_true.step_hybrid(act)
            _, _, _, _, info_lin = env_lin.step_hybrid(act)
            infos_emu = {s: e.step_hybrid(act)[4] for s, e in envs_emu.items()}
            done = term or trunc

            v_t = info_true["v_real"]
            v_l = info_lin["v_real"]
            err_l = abs(v_t - v_l)

            lin_1step.append(err_l)
            if abs(info_true["ext_f_norm"]) > 1e-4:
                lin_grade.append(err_l)
            if info_true["shift_active"]:
                lin_shift.append(err_l)

            if mode_run == 5:
                v_true_log.append(v_t)
                v_lin_1step_log.append(v_l)
                ext_force_log.append(info_true["ext_f_norm"])

            for s, inf_e in infos_emu.items():
                v_e = inf_e["v_real"]
                err_e = abs(v_t - v_e)
                emu_1step[s].append(err_e)
                if abs(info_true["ext_f_norm"]) > 1e-4:
                    emu_grade[s].append(err_e)
                if info_true["shift_active"]:
                    emu_shift[s].append(err_e)
                if mode_run == 5:
                    v_emu_1step_log[s].append(v_e)

    for mode_run in range(6):
        env_true.reset()
        env_lin.reset()
        for e in envs_emu.values():
            e.reset()
        done = False
        step_counter = 0

        while not done:
            if step_counter % surrogate_horizon == 0:
                snap = env_true.get_internal_state()
                env_lin.set_internal_state(snap)
                for e in envs_emu.values():
                    e.set_internal_state(snap)

            act = mode_run if mode_run < 5 else int((env_true.k // 15) % 5)
            _, _, term, trunc, info_true = env_true.step_hybrid(act)
            _, _, _, _, info_lin = env_lin.step_hybrid(act)
            infos_emu = {s: e.step_hybrid(act)[4] for s, e in envs_emu.items()}
            done = term or trunc
            step_counter += 1

            v_t = info_true["v_real"]
            v_l = info_lin["v_real"]
            lin_horizon.append(abs(v_t - v_l))
            if mode_run == 5:
                v_lin_hor_log.append(v_l)

            for s, inf_e in infos_emu.items():
                v_e = inf_e["v_real"]
                emu_horizon[s].append(abs(v_t - v_e))
                if mode_run == 5:
                    v_emu_hor_log[s].append(v_e)

    return {
        "lin_1step": compute_error_stats(np.array(lin_1step)),
        "lin_grade": compute_error_stats(np.array(lin_grade)),
        "lin_shift": compute_error_stats(np.array(lin_shift)),
        "lin_horizon": compute_error_stats(np.array(lin_horizon)),
        "emu_1step": {s: compute_error_stats(np.array(v)) for s, v in emu_1step.items()},
        "emu_grade": {s: compute_error_stats(np.array(v)) for s, v in emu_grade.items()},
        "emu_shift": {s: compute_error_stats(np.array(v)) for s, v in emu_shift.items()},
        "emu_horizon": {s: compute_error_stats(np.array(v)) for s, v in emu_horizon.items()},
        "surrogate_horizon": surrogate_horizon,
        "traces": {
            "v_true": np.array(v_true_log),
            "v_lin_1step": np.array(v_lin_1step_log),
            "v_lin_hor": np.array(v_lin_hor_log),
            "v_emu_1step": {s: np.array(v) for s, v in v_emu_1step_log.items()},
            "v_emu_hor": {s: np.array(v) for s, v in v_emu_hor_log.items()},
            "ext_force": np.array(ext_force_log),
        },
    }


def print_part_a_table(approx_stats: Dict[str, Any], step_budgets: List[int]) -> None:
    H = approx_stats["surrogate_horizon"]
    print("\n" + "=" * 118)
    print(f"PART A: TRAJECTORY DYNAMICS APPROXIMATION ACCURACY ({step_budgets} STEPS)")
    print("-" * 118)
    print(
        f"{'Evaluation Regime':<36} | {'Surrogate Model':<24} | {'MAE (m/s)':>10} | "
        f"{'RMSE (m/s)':>10} | {'P95 (m/s)':>10} | {'Max (m/s)':>10} | {'RMSE Reduction':>14}"
    )
    print("-" * 118)

    regimes = [
        ("1-Step-Ahead (Full Track)", "lin_1step", "emu_1step"),
        ("1-Step-Ahead (Slope & Curve Zones)", "lin_grade", "emu_grade"),
        ("1-Step-Ahead (Profile Shift Jumps)", "lin_shift", "emu_shift"),
        (f"{H}-Step Surrogate Horizon Rollout", "lin_horizon", "emu_horizon"),
    ]
    for label, k_lin, k_emu in regimes:
        ml = approx_stats[k_lin]
        print(
            f"{label:<36} | {'Paper Linear Approx':<24} | {ml['mae']:10.5f} | "
            f"{ml['rmse']:10.5f} | {ml['p95']:10.5f} | {ml['max']:10.5f} | {'Baseline':>14}"
        )
        for s in step_budgets:
            me = approx_stats[k_emu][s]
            gain = rel_improvement_pct(ml["rmse"], me["rmse"])
            print(
                f"{'':<36} | {f'{s}-Step Physics Emulator':<24} | {me['mae']:10.5f} | "
                f"{me['rmse']:10.5f} | {me['p95']:10.5f} | {me['max']:10.5f} | {gain:+13.1f}%"
            )
        print("-" * 118)


def print_generalization_table(
    title: str,
    agg: Dict[str, List[Dict[str, Any]]],
    baseline_name: str,
) -> None:
    print("\n" + "=" * 138)
    print(title)
    print("-" * 138)
    print(
        f"{'Controller Method':<38} | {'MeanErr (m/s)':>16} | {'BrakeErr':>9} | "
        f"{'ShiftOS':>9} | {'ShiftAvg':>9} | {'StopErr (m)':>12} | {'PaperScore':>17} | {'vs Linear-RL':>14}"
    )
    print("-" * 138)

    base_score = float(np.mean([r["Paper_Score"] for r in agg[baseline_name]]))

    for name, runs in agg.items():
        m_err = float(np.mean([r["Mean_Err"] for r in runs]))
        s_err = float(np.std([r["Mean_Err"] for r in runs]))
        b_err = float(np.mean([r["Brake_Err"] for r in runs]))
        os_val = float(np.mean([r["Shift_Overshoot"] for r in runs]))
        sa_val = float(np.mean([r["Shift_SettleErr"] for r in runs]))
        st_val = float(np.mean([r["Stop_Err_m"] for r in runs]))
        p_score = float(np.mean([r["Paper_Score"] for r in runs]))
        p_std = float(np.std([r["Paper_Score"] for r in runs]))

        gain_str = (
            "Baseline"
            if name == baseline_name
            else f"{rel_improvement_pct(base_score, p_score):+13.1f}%"
        )
        print(
            f"{name:<38} | "
            f"{m_err:7.4f} ± {s_err:6.4f} | "
            f"{b_err:9.4f} | "
            f"{os_val:9.4f} | "
            f"{sa_val:9.4f} | "
            f"{st_val:12.4f} | "
            f"{p_score:7.4f} ± {p_std:6.4f} | "
            f"{gain_str:>14}"
        )
    print("=" * 138)


def main():
    ap = argparse.ArgumentParser(
        description="Physics-Informed Emulator & Cumulative Multi-Track RL Benchmark across N Unseen Tracks"
    )
    ap.add_argument("--horizon", type=int, default=400, help="Trajectory horizon steps T")
    ap.add_argument("--lookahead-n", type=int, default=60, help="Lookahead horizon N for observation vector")
    ap.add_argument("--epochs", type=int, default=200, help="RL training epochs")
    ap.add_argument(
        "--step-budgets",
        type=int,
        nargs="+",
        default=[100, 200, 300],
        help="Emulator gradient step budgets to benchmark",
    )
    ap.add_argument("--traj-mode", type=str, default="batch_random", choices=["fixed_generated", "batch_random"])
    ap.add_argument("--regen-every", type=int, default=40)
    ap.add_argument("--surrogate-horizon", type=int, default=6)
    ap.add_argument("--eval-runs", type=int, default=30, help="Number of unseen tracks for generalization benchmark")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", type=str, default="emulator_clean_vs_linear.png")
    args = ap.parse_args()

    step_budgets = sorted(list(set(args.step_budgets)))

    print("=" * 138)
    print(
        f"BENCHMARK: Cumulative Multi-Track Physics Emulator ({step_budgets} Steps) vs. Paper Linear Approx | "
        f"Evaluation: {args.eval_runs} Unseen Trajectories (Mean ± Std)"
    )
    print("=" * 138)

    v_des, _ = generate_truly_random_trajectory(
        total_horizon=args.horizon, min_phase_len=25, seed=args.seed
    )
    gen_fn = lambda seed: generate_truly_random_trajectory(
        total_horizon=args.horizon, min_phase_len=25, seed=seed
    )

    # 1. Train Stateless Physics Emulators & Report Part A Accuracy
    emulators_by_steps = {}
    loss_curves = {}
    for s in step_budgets:
        emu_s, curve_s = learn_150_step_emulator_for_trajectory(
            v_des,
            emulator=None,
            emulator_steps=s,
            verbose=False,
            return_curve=True,
        )
        emulators_by_steps[s] = emu_s
        loss_curves[s] = curve_s

    approx_stats = evaluate_model_approximation_error(
        v_des, emulators_by_steps, surrogate_horizon=args.surrogate_horizon
    )
    print_part_a_table(approx_stats, step_budgets)

    # 2. Train RL-Switch (4-Action) & Hybrid RL-MPC (5-Action) Policies
    print(f"\nPART B: Training RL Policies across {step_budgets} Emulator Steps ({args.epochs} Epochs)...")

    rl_lin, _, _ = train_rl_switch_strict_two_ways(
        gen_fn,
        rl_way="paper_linear_only",
        traj_mode=args.traj_mode,
        regen_every_n_epochs=args.regen_every,
        epochs=args.epochs,
        lookahead_n=args.lookahead_n,
        use_hybrid_mpc=False,
        seed=args.seed,
    )
    rl_lin.eval()

    hyb_lin, _, _ = train_rl_switch_strict_two_ways(
        gen_fn,
        rl_way="paper_linear_only",
        traj_mode=args.traj_mode,
        regen_every_n_epochs=args.regen_every,
        epochs=args.epochs,
        lookahead_n=args.lookahead_n,
        use_hybrid_mpc=True,
        seed=args.seed,
    )
    hyb_lin.eval()

    rl_emu_policies = {}
    hyb_emu_policies = {}

    for s in step_budgets:
        pol_rl, _, _ = train_rl_switch_strict_two_ways(
            gen_fn,
            rl_way="emulator_150_only",
            traj_mode=args.traj_mode,
            regen_every_n_epochs=args.regen_every,
            epochs=args.epochs,
            emulator_steps=s,
            lookahead_n=args.lookahead_n,
            use_hybrid_mpc=False,
            seed=args.seed,
        )
        pol_rl.eval()
        rl_emu_policies[s] = pol_rl

        pol_hyb, _, _ = train_rl_switch_strict_two_ways(
            gen_fn,
            rl_way="emulator_150_only",
            traj_mode=args.traj_mode,
            regen_every_n_epochs=args.regen_every,
            epochs=args.epochs,
            emulator_steps=s,
            lookahead_n=args.lookahead_n,
            use_hybrid_mpc=True,
            seed=args.seed,
        )
        pol_hyb.eval()
        hyb_emu_policies[s] = pol_hyb

    # 3. Comprehensive Evaluation Across N Unseen Trajectories
    specs = [
        ("Hardcoded-Switch IPID (Havaei)", "HARDCODED", None),
        ("Fuzzy-PID (Standard Mamdani)", "FUZZY_STD", None),
        ("ILR-PSO-Fuzzy-PID (He 2024)", "FUZZY_ILR_PSO", None),
        ("RL-Switch Way 1 (Linear Approx)", "RL_SOFT", rl_lin),
        *[(f"RL-Switch Way 2 ({s}-Step Emu)", "RL_SOFT", rl_emu_policies[s]) for s in step_budgets],
        ("Hybrid RL-MPC (Linear Approx)", "RL_HYBRID", hyb_lin),
        *[(f"Hybrid RL-MPC ({s}-Step Emu)", "RL_HYBRID", hyb_emu_policies[s]) for s in step_budgets],
    ]

    agg = {name: [] for name, _, _ in specs}
    sample_trajs = {}

    for ep in range(args.eval_runs):
        v_unseen, _ = generate_truly_random_trajectory(
            total_horizon=args.horizon, min_phase_len=25, seed=90000 + ep
        )
        for name, ctype, pol in specs:
            res = run_controller_benchmark(
                name,
                v_unseen,
                ctype,
                rl_policy=pol,
                lookahead_n=args.lookahead_n,
                record_traj=(ep == 0),
            )
            agg[name].append(res)
            if ep == 0 and "traj" in res:
                sample_trajs[name] = res["traj"]

    print_generalization_table(
        f"PART B: CLOSED-LOOP PERFORMANCE ACROSS {args.eval_runs} UNSEEN NONLINEAR TRAJECTORIES (MEAN ± STD)",
        agg=agg,
        baseline_name="RL-Switch Way 1 (Linear Approx)",
    )

    # 4. Diagnostic Figure
    max_s = step_budgets[-1]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5))
    ax1, ax2, ax3, ax4 = axes.flatten()
    traces = approx_stats["traces"]
    k_ax = np.arange(len(traces["v_true"]))
    H = approx_stats["surrogate_horizon"]

    for s in step_budgets:
        ax1.plot(np.arange(1, len(loss_curves[s]) + 1), loss_curves[s], linewidth=1.8, label=f"{s}-Step Emu Loss")
    ax1.axhline(approx_stats["lin_1step"]["rmse"], color="b", linestyle="--", label="Paper Linear Approx Baseline")
    ax1.set_title(f"1. Emulator Convergence ({step_budgets} Steps)")
    ax1.set_xlabel("Gradient Step")
    ax1.set_ylabel("Fit RMSE (m/s)")
    ax1.legend(fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_ax, np.abs(traces["v_true"] - traces["v_lin_1step"]), "b--", alpha=0.75, label="Paper Linear (1-Step)")
    for s in step_budgets:
        ax2.plot(k_ax, np.abs(traces["v_true"] - traces["v_emu_1step"][s]), alpha=0.85, label=f"{s}-Step Emu (1-Step)")
    ax2.set_title("2. 1-Step Velocity Error Along Track")
    ax2.set_xlabel("Distance Sample (k)")
    ax2.set_ylabel("Error (m/s)")
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    ax3.plot(k_ax, np.abs(traces["v_true"] - traces["v_lin_hor"]), "b--", alpha=0.75, label=f"Paper Linear ({H}-Step Drift)")
    for s in step_budgets:
        ax3.plot(k_ax, np.abs(traces["v_true"] - traces["v_emu_hor"][s]), alpha=0.85, label=f"{s}-Step Emu ({H}-Step Drift)")
    ax3.set_title(f"3. Compounding Error ({H}-Step Horizons)")
    ax3.set_xlabel("Distance Sample (k)")
    ax3.set_ylabel(f"{H}-Step Drift (m/s)")
    ax3.legend(fontsize=8.5)
    ax3.grid(True, alpha=0.3)

    if sample_trajs:
        ax4.plot(sample_trajs["RL-Switch Way 1 (Linear Approx)"]["k"], sample_trajs["RL-Switch Way 1 (Linear Approx)"]["err"], "b--", linewidth=1.3, label="RL-Switch (Linear)")
        ax4.plot(sample_trajs[f"RL-Switch Way 2 ({max_s}-Step Emu)"]["k"], sample_trajs[f"RL-Switch Way 2 ({max_s}-Step Emu)"]["err"], "g-", linewidth=1.4, label=f"RL-Switch ({max_s}-Step Emu)")
        ax4.plot(sample_trajs["Hybrid RL-MPC (Linear Approx)"]["k"], sample_trajs["Hybrid RL-MPC (Linear Approx)"]["err"], "m--", linewidth=1.3, label="Hybrid RL-MPC (Linear)")
        ax4.plot(sample_trajs[f"Hybrid RL-MPC ({max_s}-Step Emu)"]["k"], sample_trajs[f"Hybrid RL-MPC ({max_s}-Step Emu)"]["err"], "r-", linewidth=1.5, label=f"Hybrid RL-MPC ({max_s}-Step Emu)")
        ax4.set_title("4. Sample Unseen Track Closed-Loop Error |v - v_des|")
        ax4.set_xlabel("Distance Sample (k)")
        ax4.set_ylabel("Tracking Error (m/s)")
        ax4.legend(fontsize=8.5)
        ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"\nSaved benchmark plot to: {args.plot}")


if __name__ == "__main__":
    main()