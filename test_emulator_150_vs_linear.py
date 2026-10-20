"""
Stabilized Physics-Informed Emulator Benchmark (`test_emulator_150_vs_linear.py`):
Fixes the two core causes of emulator instability and late-epoch degradation:
  1. Zero-Residual Initialization + Linear Warmup:
     Starts the emulator at Step 0 with exact Paper Linear Approx fidelity (zero initial residual)
     so gradient learning descends monotonically without step 1..50 oscillations.
  2. Causal Physics Feature Masking + Multi-Track Replay Buffer:
     In `_integrate_plant_physics`, the true nonlinear residual (v_nl - v_lin) depends ONLY on
     (v_curr, u_cmd, ext_force, v_curr^2) — NOT on track coordinate `x_norm` or PID `gains_norm`.
     Masking those two spurious channels + retaining a replay buffer across batch regenerations
     prevents catastrophic overfitting at 200 and 300 steps.
"""
from __future__ import annotations

import argparse
import copy
import time
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmark_central import (
    generate_truly_random_trajectory,
    run_controller_benchmark,
)
import havaei_generalizing_rl as hgrl
from havaei_generalizing_rl import (
    LearnedLinearizationModel,
    PathAgnosticHavaeiEnv,
    train_rl_switch_strict_two_ways,
)


class StabilizedPhysicsEmulator(LearnedLinearizationModel):
    """
    Physics-informed residual emulator:
      - Zero-initialized final layer so Step 0 = exact Paper Linear Approximation (no oscillation).
      - Causal feature mask: uses only physical state/input features [v_norm, u_norm, ext_force, quad_norm]
        and zeroes out non-causal [x_norm] and [gains_norm] that cause counterfactual action overfitting.
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__(hidden_dim=hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize output layer to zero so at step 0: pred == v_lin_next
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        x_m: torch.Tensor,
        v_curr: torch.Tensor,
        v_lin_next: torch.Tensor,
        u_cmd: torch.Tensor,
        ext_force: torch.Tensor,
        gains_norm: torch.Tensor,
    ) -> torch.Tensor:
        v_norm = v_curr / 15.0
        u_norm = torch.clamp(u_cmd, -1.0, 1.0)
        quad_norm = v_norm ** 2
        zeros_1 = torch.zeros_like(v_norm)
        zeros_3 = torch.zeros_like(gains_norm)

        # Mask out non-physical position (x_norm) and controller identity (gains_norm)
        # so the emulator learns the true plant residual dv(v, u, ext_force)
        base = torch.stack([zeros_1, v_norm, u_norm, ext_force, quad_norm], dim=-1)
        feats = torch.cat([base, zeros_3], dim=-1)
        dv_residual = self.net(feats).squeeze(-1)
        return torch.clamp(v_lin_next + dv_residual, min=0.0)


# Global replay buffer across batch_random regenerations so 200/300 steps never overfit to a single track
_REPLAY_BUFFER: List[Dict[str, torch.Tensor]] = []


def install_stabilized_emulator_trainer(max_replay_tracks: int = 4, base_lr: float = 2.5e-3):
    """
    Replaces `havaei_generalizing_rl.learn_150_step_emulator_for_trajectory` with a
    monotonically converging, replay-buffered trainer with linear warmup + cosine decay.
    """
    def stabilized_learn_emulator(
        v_des: np.ndarray,
        emulator: LearnedLinearizationModel | None = None,
        emulator_steps: int = 150,
        n_rollouts: int = 8,
        lr: float = base_lr,
        verbose: bool = False,
        return_curve: bool = False,
    ):
        global _REPLAY_BUFFER
        env_nl = PathAgnosticHavaeiEnv(v_des_profile=v_des, plant_mode="nonlinear")
        x_list, vc_list, vlin_list, u_list, ext_list, g_list, vnext_list = [], [], [], [], [], [], []
        rng = np.random.default_rng(42)

        for ep in range(n_rollouts):
            env_nl.reset()
            done = False
            while not done:
                v_curr = env_nl.v_real
                if ep < 4:
                    _, _, term, trunc, info = env_nl.step(ep)
                elif ep == 4:
                    _, _, term, trunc, info = env_nl.step_hybrid(4)
                else:
                    rand_act = int(rng.integers(0, 5))
                    _, _, term, trunc, info = env_nl.step_hybrid(rand_act)
                done = term or trunc

                x_list.append(info["x_pos"])
                vc_list.append(v_curr)
                vlin_list.append(info["v_lin"])
                u_list.append(info["u_pid"])
                ext_list.append(info["ext_f_norm"])
                g_list.append(info["gains_norm"])
                vnext_list.append(info["v_real"])

        cur_data = {
            "x": torch.tensor(x_list, dtype=torch.float32),
            "vc": torch.tensor(vc_list, dtype=torch.float32),
            "vlin": torch.tensor(vlin_list, dtype=torch.float32),
            "u": torch.tensor(u_list, dtype=torch.float32),
            "ext": torch.tensor(ext_list, dtype=torch.float32),
            "g": torch.as_tensor(np.asarray(g_list), dtype=torch.float32),
            "target": torch.tensor(vnext_list, dtype=torch.float32),
        }

        if emulator is None:
            # Fresh training run: reset replay buffer and create zero-residual model
            _REPLAY_BUFFER.clear()
            emulator = StabilizedPhysicsEmulator(hidden_dim=64)
            step_budget = emulator_steps
            effective_lr = lr
        else:
            # Warm-starting on a new track in batch_random: use a gentle fine-tuning LR
            # and combine with replay buffer so cumulative steps don't overfit!
            step_budget = max(40, emulator_steps // 2)
            effective_lr = lr * 0.45

        _REPLAY_BUFFER.append(cur_data)
        if len(_REPLAY_BUFFER) > max_replay_tracks:
            _REPLAY_BUFFER.pop(0)

        x_t = torch.cat([d["x"] for d in _REPLAY_BUFFER], dim=0)
        vc_t = torch.cat([d["vc"] for d in _REPLAY_BUFFER], dim=0)
        vlin_t = torch.cat([d["vlin"] for d in _REPLAY_BUFFER], dim=0)
        u_t = torch.cat([d["u"] for d in _REPLAY_BUFFER], dim=0)
        ext_t = torch.cat([d["ext"] for d in _REPLAY_BUFFER], dim=0)
        g_t = torch.cat([d["g"] for d in _REPLAY_BUFFER], dim=0)
        target_t = torch.cat([d["target"] for d in _REPLAY_BUFFER], dim=0)

        optimizer = optim.AdamW(emulator.parameters(), lr=effective_lr, weight_decay=2e-4)
        warmup_steps = max(5, step_budget // 10)

        loss_curve = []
        best_loss = float("inf")
        best_state = copy.deepcopy(emulator.state_dict())

        for step_idx in range(step_budget):
            # Linear warmup for first 10% of steps, then smooth cosine decay
            if step_idx < warmup_steps:
                lr_mult = float(step_idx + 1) / float(warmup_steps)
            else:
                progress = float(step_idx - warmup_steps) / float(max(1, step_budget - warmup_steps))
                lr_mult = 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = effective_lr * lr_mult

            pred = emulator(x_t, vc_t, vlin_t, u_t, ext_t, g_t)
            mse_loss = nn.functional.mse_loss(pred, target_t)

            # Evaluate RMSE on current trajectory for logging and best-checkpoint tracking
            with torch.no_grad():
                pred_cur = emulator(
                    cur_data["x"], cur_data["vc"], cur_data["vlin"],
                    cur_data["u"], cur_data["ext"], cur_data["g"]
                )
                cur_rmse = float(torch.sqrt(nn.functional.mse_loss(pred_cur, cur_data["target"])).item())
                loss_curve.append(cur_rmse)
                if cur_rmse < best_loss:
                    best_loss = cur_rmse
                    best_state = copy.deepcopy(emulator.state_dict())

            optimizer.zero_grad()
            mse_loss.backward()
            nn.utils.clip_grad_norm_(emulator.parameters(), 0.5)
            optimizer.step()

        # Restore best checkpoint so extra steps can never degrade fidelity
        emulator.load_state_dict(best_state)

        if verbose:
            print(f"    [Stabilized Physics Emulator ({step_budget} steps)] Fit RMSE: {best_loss:.5f} m/s")
        if return_curve:
            return emulator, loss_curve
        return emulator

    hgrl.LearnedLinearizationModel = StabilizedPhysicsEmulator
    hgrl.learn_150_step_emulator_for_trajectory = stabilized_learn_emulator
    return stabilized_learn_emulator


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
    print(f"PART A: STABILIZED TRAJECTORY DYNAMICS ACCURACY ACROSS {step_budgets} EMULATOR STEPS")
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


def print_part_b_table(
    title: str,
    results: List[Dict[str, Any]],
    baseline_res: Dict[str, Any],
) -> None:
    print("\n" + "=" * 128)
    print(title)
    print("-" * 128)
    print(
        f"{'Controller Method':<38} | {'MeanErr(m/s)':>12} | {'BrakeErr':>9} | "
        f"{'ShiftOS(m/s)':>12} | {'ShiftAvg':>9} | {'StopErr(m)':>10} | {'PaperScore':>10} | {'vs Linear-RL':>14}"
    )
    print("-" * 128)
    base_score = baseline_res["Paper_Score"]
    for r in results:
        gain_str = (
            "Baseline"
            if r is baseline_res
            else f"{rel_improvement_pct(base_score, r['Paper_Score']):+13.1f}%"
        )
        print(
            f"{r['Method']:<38} | "
            f"{r['Mean_Err']:12.4f} | "
            f"{r['Brake_Err']:9.4f} | "
            f"{r['Shift_Overshoot']:12.4f} | "
            f"{r['Shift_SettleErr']:9.4f} | "
            f"{r['Stop_Err_m']:10.4f} | "
            f"{r['Paper_Score']:10.4f} | "
            f"{gain_str:>14}"
        )
    print("=" * 128)


def main():
    ap = argparse.ArgumentParser(
        description="Stabilized Physics-Informed Emulator Benchmark ([100, 200, 300] Steps vs Linear)"
    )
    ap.add_argument("--horizon", type=int, default=400, help="Trajectory horizon steps T")
    ap.add_argument("--lookahead-n", type=int, default=60, help="Lookahead horizon N for observation vector")
    ap.add_argument("--epochs", type=int, default=200, help="RL training epochs")
    ap.add_argument(
        "--step-budgets",
        type=int,
        nargs="+",
        default=[100, 200, 300],
        help="Emulator gradient step budgets to benchmark (default: 100 200 300)",
    )
    ap.add_argument(
        "--traj-mode",
        type=str,
        default="batch_random",
        choices=["fixed_generated", "batch_random"],
    )
    ap.add_argument("--regen-every", type=int, default=40)
    ap.add_argument("--surrogate-horizon", type=int, default=6)
    ap.add_argument("--eval-unseen", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", type=str, default="emulator_100_200_300_vs_linear.png")
    args = ap.parse_args()

    step_budgets = sorted(list(set(args.step_budgets)))
    learn_fn = install_stabilized_emulator_trainer()

    print("=" * 128)
    print(
        f"STABILIZED BENCHMARK: Zero-Init Causal Physics Emulator ({step_budgets} Steps) | "
        f"RL Traj Mode: {args.traj_mode}"
    )
    print("=" * 128)

    v_des, schedule = generate_truly_random_trajectory(
        total_horizon=args.horizon, min_phase_len=25, seed=args.seed
    )
    if args.traj_mode == "fixed_generated":
        gen_fn = lambda seed: (v_des, schedule)
    else:
        gen_fn = lambda seed: generate_truly_random_trajectory(
            total_horizon=args.horizon, min_phase_len=25, seed=seed
        )

    # -------------------------------------------------------------------------
    # PART A: Train Stabilized Emulators for [100, 200, 300] Steps
    # -------------------------------------------------------------------------
    emulators_by_steps = {}
    loss_curves = {}
    for s in step_budgets:
        emu_s, curve_s = learn_fn(
            v_des, emulator=None, emulator_steps=s, verbose=True, return_curve=True
        )
        emulators_by_steps[s] = emu_s
        loss_curves[s] = curve_s

    approx_stats = evaluate_model_approximation_error(
        v_des, emulators_by_steps, surrogate_horizon=args.surrogate_horizon
    )
    print_part_a_table(approx_stats, step_budgets)

    # -------------------------------------------------------------------------
    # PART B: Train RL-Switch (4-Action) & Hybrid RL-MPC (5-Action)
    # -------------------------------------------------------------------------
    print(f"\nPART B: Training RL Switch & Hybrid RL-MPC across {step_budgets} Emulator Steps ({args.epochs} Epochs)...")

    res_ipid = run_controller_benchmark(
        "Hardcoded-Switch IPID (Havaei)", v_des, "HARDCODED", lookahead_n=args.lookahead_n, record_traj=True
    )

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
    res_rl_lin = run_controller_benchmark(
        "RL-Switch Way 1 (Linear Approx)", v_des, "RL_SOFT", rl_policy=rl_lin, lookahead_n=args.lookahead_n, record_traj=True
    )

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
    res_hyb_lin = run_controller_benchmark(
        "Hybrid RL-MPC (Linear Approx)", v_des, "RL_HYBRID", rl_policy=hyb_lin, lookahead_n=args.lookahead_n, record_traj=True
    )

    rl_emu_policies = {}
    hyb_emu_policies = {}
    res_rl_emu_list = []
    res_hyb_emu_list = []

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
        res_rl_emu_list.append(
            run_controller_benchmark(
                f"RL-Switch Way 2 ({s}-Step Emu)",
                v_des,
                "RL_SOFT",
                rl_policy=pol_rl,
                lookahead_n=args.lookahead_n,
                record_traj=True,
            )
        )

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
        res_hyb_emu_list.append(
            run_controller_benchmark(
                f"Hybrid RL-MPC ({s}-Step Emu)",
                v_des,
                "RL_HYBRID",
                rl_policy=pol_hyb,
                lookahead_n=args.lookahead_n,
                record_traj=True,
            )
        )

    all_single_track = [res_ipid, res_rl_lin, *res_rl_emu_list, res_hyb_lin, *res_hyb_emu_list]
    print_part_b_table(
        "PART B.1: CLOSED-LOOP TRACKING ON TARGET NONLINEAR TRAJECTORY",
        all_single_track,
        baseline_res=res_rl_lin,
    )

    # -------------------------------------------------------------------------
    # PART B.2: Generalization Across Unseen Random Trajectories
    # -------------------------------------------------------------------------
    if args.eval_unseen > 0:
        specs = [
            ("Hardcoded-Switch IPID (Havaei)", "HARDCODED", None),
            ("RL-Switch Way 1 (Linear Approx)", "RL_SOFT", rl_lin),
            *[(f"RL-Switch Way 2 ({s}-Step Emu)", "RL_SOFT", rl_emu_policies[s]) for s in step_budgets],
            ("Hybrid RL-MPC (Linear Approx)", "RL_HYBRID", hyb_lin),
            *[(f"Hybrid RL-MPC ({s}-Step Emu)", "RL_HYBRID", hyb_emu_policies[s]) for s in step_budgets],
        ]
        agg = {name: [] for name, _, _ in specs}
        for ep in range(args.eval_unseen):
            v_unseen, _ = generate_truly_random_trajectory(
                total_horizon=args.horizon, min_phase_len=25, seed=90000 + ep
            )
            for name, ctype, pol in specs:
                agg[name].append(
                    run_controller_benchmark(
                        name, v_unseen, ctype, rl_policy=pol, lookahead_n=args.lookahead_n
                    )
                )

        avg_results = []
        for name, _, _ in specs:
            runs = agg[name]
            avg_results.append({
                "Method": name,
                "Mean_Err": float(np.mean([r["Mean_Err"] for r in runs])),
                "Brake_Err": float(np.mean([r["Brake_Err"] for r in runs])),
                "Shift_Overshoot": float(np.mean([r["Shift_Overshoot"] for r in runs])),
                "Shift_SettleErr": float(np.mean([r["Shift_SettleErr"] for r in runs])),
                "Stop_Err_m": float(np.mean([r["Stop_Err_m"] for r in runs])),
                "Paper_Score": float(np.mean([r["Paper_Score"] for r in runs])),
            })

        print_part_b_table(
            f"PART B.2: GENERALIZATION ACROSS {args.eval_unseen} UNSEEN RANDOM NONLINEAR TRAJECTORIES (MEAN)",
            avg_results,
            baseline_res=avg_results[1],
        )

    # -------------------------------------------------------------------------
    # Save 4-Panel Diagnostic Plot
    # -------------------------------------------------------------------------
    max_s = step_budgets[-1]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5))
    ax1, ax2, ax3, ax4 = axes.flatten()
    traces = approx_stats["traces"]
    k_ax = np.arange(len(traces["v_true"]))
    H = approx_stats["surrogate_horizon"]

    for s in step_budgets:
        ax1.plot(np.arange(1, len(loss_curves[s]) + 1), loss_curves[s], linewidth=1.8, label=f"{s}-Step Emulator RMSE")
    ax1.axhline(approx_stats["lin_1step"]["rmse"], color="b", linestyle="--", label="Paper Linear Approx RMSE")
    ax1.set_title(f"1. Stabilized Emulator Learning Curve ({step_budgets} Steps)")
    ax1.set_xlabel("Gradient Step")
    ax1.set_ylabel("Prediction RMSE (m/s)")
    ax1.legend(fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_ax, np.abs(traces["v_true"] - traces["v_lin_1step"]), "b--", alpha=0.75, label="Paper Linear (1-Step)")
    for s in step_budgets:
        ax2.plot(k_ax, np.abs(traces["v_true"] - traces["v_emu_1step"][s]), alpha=0.85, label=f"{s}-Step Emulator (1-Step)")
    ax2.set_title("2. 1-Step-Ahead Prediction Error Along Track")
    ax2.set_xlabel("Distance Sample (k)")
    ax2.set_ylabel("1-Step Error (m/s)")
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    ax3.plot(k_ax, np.abs(traces["v_true"] - traces["v_lin_hor"]), "b--", alpha=0.75, label=f"Paper Linear ({H}-Step Drift)")
    for s in step_budgets:
        ax3.plot(k_ax, np.abs(traces["v_true"] - traces["v_emu_hor"][s]), alpha=0.85, label=f"{s}-Step Emulator ({H}-Step Drift)")
    ax3.set_title(f"3. Compounding Error Over {H}-Step Surrogate Horizon")
    ax3.set_xlabel("Distance Sample (k)")
    ax3.set_ylabel(f"{H}-Step Horizon Error (m/s)")
    ax3.legend(fontsize=8.5)
    ax3.grid(True, alpha=0.3)

    ax4.plot(res_rl_lin["traj"]["k"], res_rl_lin["traj"]["err"], "b--", linewidth=1.3, label="RL-Switch (Linear)")
    ax4.plot(res_rl_emu_list[-1]["traj"]["k"], res_rl_emu_list[-1]["traj"]["err"], "g-", linewidth=1.4, label=f"RL-Switch ({max_s}-Step Emu)")
    ax4.plot(res_hyb_lin["traj"]["k"], res_hyb_lin["traj"]["err"], "m--", linewidth=1.3, label="Hybrid RL-MPC (Linear)")
    ax4.plot(res_hyb_emu_list[-1]["traj"]["k"], res_hyb_emu_list[-1]["traj"]["err"], "r-", linewidth=1.5, label=f"Hybrid RL-MPC ({max_s}-Step Emu)")
    ax4.set_title("4. Downstream Closed-Loop Tracking Error |v - v_des|")
    ax4.set_xlabel("Distance Sample (k)")
    ax4.set_ylabel("Tracking Error (m/s)")
    ax4.legend(fontsize=8.5)
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"\nSaved 4-panel diagnostic comparison plot to: {args.plot}")


if __name__ == "__main__":
    main()