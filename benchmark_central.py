"""
Fast Focused Benchmark Suite (`benchmark_central.py`)
Focuses directly on:
  - Pure MPC & Adaptive MPC (Idea B)
  - Hardcoded-Switch IPID (Havaei reference)
  - Unshielded RL vs. Shielded RL (1-Step DCBF vs. Multi-Step DMPS Rollout)
  - Unshielded Hybrid RL-MPC vs. Shielded Hybrid RL-MPC (Switch MPS, 1-Step DCBF, Multi-Step DMPS Rollout)
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
from safety_shield import SafetyFilterShield, ShieldConfig, compute_and_print_shield_scissor_band
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
    max_accel: float = 1.85,
    max_decel: float = 1.65,
    dt: float = 0.50,
    service_margin: float = 0.92,
    seed: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Generates random speed profiles across all 6 driving modes of Havaei & Sandidzadeh (2022)
    Eq. 53: (1) Acceleration, (2) Shift-Up, (3) Cruising, (4) Braking, (5) Shift-Down, (6) Stopping.

    Parameters
    ----------
    total_horizon : int
        Number of discrete simulation steps (default: 1000).
    min_phase_len : int
        Minimum duration of a single driving mode phase in steps.
    max_accel : float
        Maximum physical motor traction acceleration in m/s^2 (e.g., 1.85).
    max_decel : float
        Maximum physical service braking deceleration in m/s^2 (e.g., 1.65).
    dt : float
        Sampling interval in seconds (default: 0.50 s).
    service_margin : float
        Safety factor (default: 0.92 = 92% of peak capability) leaving control
        headroom for feedback controllers (PID/MPC) to reject disturbances.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    v_des : np.ndarray
        Array of length `total_horizon + 2` with strictly feasible speed reference.
    schedule : list[dict]
        List of phase segment metadata.
    """
    rng = np.random.default_rng(seed)
    v_des = np.zeros(total_horizon + 2, dtype=np.float64)
    schedule = []

    # Maximum allowable speed delta per step dt
    eff_acc = max_accel * service_margin
    eff_dec = max_decel * service_margin
    max_dv_up = eff_acc * dt
    max_dv_dn = eff_dec * dt

    cursor, curr_v, has_shift = 0, 0.0, False

    while cursor < total_horizon:
        rem_steps = total_horizon - cursor

        # If nearing the end, plan the terminal station stop
        # Min steps needed to stop from curr_v with cosine ramp: (pi/2) * (curr_v / (eff_dec * dt))
        min_stop_steps = int(np.ceil((np.pi / 2.0) * curr_v / max(1e-4, max_dv_dn))) + 5
        if rem_steps <= max(min_phase_len * 2, min_stop_steps):
            dur = rem_steps
            mode = 6 if curr_v > 0.3 else 3
        else:
            max_dur = min(75, rem_steps - min_phase_len)
            dur = int(rng.integers(min_phase_len, max(min_phase_len + 1, max_dur + 1)))

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

        start_k, end_k = cursor, cursor + dur
        t_loc = np.linspace(0.0, 1.0, dur, endpoint=False)

        # ---------------------------------------------------------------------
        # Mode 1: Acceleration
        # ---------------------------------------------------------------------
        if mode == 1:
            # Half-cosine S-curve: peak accel is (pi/2) * Delta_v / (dur * dt) <= eff_acc
            max_possible_jump = (2.0 / np.pi) * eff_acc * (dur * dt)
            target_v = float(min(16.5, curr_v + rng.uniform(1.5, max(1.8, max_possible_jump))))
            v_seg = curr_v + (target_v - curr_v) * 0.5 * (1.0 - np.cos(np.pi * t_loc))
            curr_v = float(v_seg[-1])

        # ---------------------------------------------------------------------
        # Mode 2: Shift-Up
        # ---------------------------------------------------------------------
        elif mode == 2:
            target_v = float(min(16.8, curr_v + rng.uniform(3.0, 4.8)))
            # Compute minimum steps needed so linear slope <= max_dv_up
            min_ramp = int(np.ceil((target_v - curr_v) / max_dv_up))
            ramp_len = min(dur - 5, max(min_ramp, 4))
            # If duration is too tight for the full jump, clamp target_v
            if min_ramp > ramp_len:
                target_v = curr_v + ramp_len * max_dv_up

            v_seg = np.full(dur, target_v)
            v_seg[:ramp_len] = np.linspace(curr_v, target_v, ramp_len + 1)[1:]
            curr_v = target_v

        # ---------------------------------------------------------------------
        # Mode 3: Cruising
        # ---------------------------------------------------------------------
        elif mode == 3:
            # Sine ripple with bounded acceleration derivative
            amp = min(0.12, (eff_acc * dt * dur) / (2.0 * np.pi))
            v_seg = np.full(dur, curr_v) + amp * np.sin(2.0 * np.pi * t_loc)
            curr_v = float(v_seg[-1])

        # ---------------------------------------------------------------------
        # Mode 4: Braking
        # ---------------------------------------------------------------------
        elif mode == 4:
            # Half-cosine S-curve deceleration: peak decel is (pi/2) * Delta_v / (dur * dt) <= eff_dec
            max_possible_drop = (2.0 / np.pi) * eff_dec * (dur * dt)
            target_v = float(max(1.5, curr_v - rng.uniform(2.0, max(2.5, max_possible_drop))))
            v_seg = curr_v + (target_v - curr_v) * 0.5 * (1.0 - np.cos(np.pi * t_loc))
            curr_v = float(v_seg[-1])

        # ---------------------------------------------------------------------
        # Mode 5: Shift-Down
        # ---------------------------------------------------------------------
        elif mode == 5:
            target_v = float(max(3.0, curr_v - rng.uniform(3.0, 4.8)))
            # Compute minimum steps needed so linear slope <= max_dv_dn
            min_ramp = int(np.ceil((curr_v - target_v) / max_dv_dn))
            ramp_len = min(dur - 5, max(min_ramp, 4))
            if min_ramp > ramp_len:
                target_v = curr_v - ramp_len * max_dv_dn

            v_seg = np.full(dur, target_v)
            v_seg[:ramp_len] = np.linspace(curr_v, target_v, ramp_len + 1)[1:]
            curr_v = target_v

        # ---------------------------------------------------------------------
        # Mode 6: Station Stop
        # ---------------------------------------------------------------------
        else:
            if curr_v > 0.3:
                # Dynamic stop ramp length matching service braking capability
                min_stop_steps = int(np.ceil((np.pi / 2.0) * curr_v / max_dv_dn))
                stop_ramp = min(dur, max(min_stop_steps, 8))
                v_seg = np.zeros(dur)
                v_seg[:stop_ramp] = curr_v * 0.5 * (1.0 + np.cos(np.pi * np.linspace(0.0, 1.0, stop_ramp)))
            else:
                v_seg = np.zeros(dur)
            curr_v = 0.0

        v_des[start_k:end_k] = v_seg
        schedule.append({
            "mode": mode,
            "start": start_k,
            "end": end_k - 1,
            "dur": dur,
            "v_target": float(curr_v),
        })
        cursor = end_k

    # -------------------------------------------------------------------------
    # Two-Pass Kinematic Projection (Strict Invariance Guarantee)
    # -------------------------------------------------------------------------
    # Pass 1: Backward pass enforces braking capability at all boundaries & stops
    v_des[total_horizon:] = 0.0
    for k in range(total_horizon - 1, -1, -1):
        v_max_allowable = v_des[k + 1] + max_dv_dn
        if v_des[k] > v_max_allowable:
            v_des[k] = v_max_allowable

    # Pass 2: Forward pass enforces traction capability from rest
    v_des[0] = 0.0
    for k in range(1, total_horizon + 1):
        v_max_allowable = v_des[k - 1] + max_dv_up
        if v_des[k] > v_max_allowable:
            v_des[k] = v_max_allowable

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
    disturbance_profile: np.ndarray | None = None,
) -> dict:
    env = PathAgnosticHavaeiEnv(
        v_des_profile=v_des,
        plant_mode="nonlinear",
        learned_linear_model=hrl.LAST_TRAINED_EMULATOR,
        lookahead_n=lookahead_n,
        shield=shield,
        disturbance_profile=disturbance_profile,
    )
    obs = env.reset()
    if shield is not None:
        shield._get_or_build_emulated_trajectory(env)
    done = False

    hc = HardcodedSwitchController(use_transient_chain=True)
    hc.reset()
    if mpc_ctrl is not None:
        mpc_ctrl.reset()
    if fuzzy_ctrl is not None:
        fuzzy_ctrl.reset()

    k_log, v_log, v_des_log, err_log, os_log, shift_err_log, brake_err_log, alg_log = [], [], [], [], [], [], [], []
    int_log, wind_log, wind_hat_log = [], [], []
    step_latencies_us = []

    while not done:
        t0 = time.perf_counter()

        if controller_type == "MPC":
            assert mpc_ctrl is not None
            k_curr = env.k
            x_sample = min(k_curr + 1, env.max_steps) * env.d
            ref_h = env.v_des[k_curr + 1 : min(k_curr + 1 + mpc_ctrl.n_p, len(env.v_des))]
            slope_iota = (env.physics.grade_resistance(x_sample) + env.physics.curve_resistance(x_sample)) / (env.physics.M * env.physics.g)
            mpc_ctrl.prev_v = float(env.v_prev)
            u_opt = mpc_ctrl.select_action(
                env.v_real,
                ref_h,
                slope_iota=slope_iota,
                dt_step=env.dt,
                u_act_prev=env.u_act,
                disturbance_accel=0.35 * env.d_hat,
            )
            shield_diff = 0.0
            if shield is not None:
                u_opt, shield_diff = shield.filter_mpc_command(env, u_opt)
            t1 = time.perf_counter()
            obs, _, term, trunc, info = env.step_mpc(u_opt, shield_diff=shield_diff, apply_shield=False)

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

        err_log.append(info["abs_err"])
        os_log.append(info["overshoot"])

        if info.get("shift_active", False):
            shift_err_log.append(info["abs_err"])
        if info.get("braking_active", False):
            brake_err_log.append(info["abs_err"])

        alg_log.append(info["alg"])

        if record_traj:
            k_log.append(info["k"])
            v_log.append(info["v_real"])
            v_des_log.append(info["v_des"])
            int_log.append(bool(info.get("shield_intervened", False)))
            wind_log.append(float(info.get("wind_accel", 0.0)))
            wind_hat_log.append(float(info.get("wind_hat", 0.0)))

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

    int_rate = (100.0 * shield.total_interventions / max(1, shield.total_steps)) if shield is not None else 0.0
    alg_counts = {a: int(alg_log.count(a)) for a in HYBRID_ALGORITHMS}
    res = {
        "Method": method_name,
        "Mean_Err": mean_err,
        "Shift_Overshoot": shift_os,
        "Shift_SettleErr": shift_avg_err,
        "Brake_Err": brake_err,
        "Stop_Err_m": stop_err,
        "Paper_Score": paper_score,
        "Shield_Rate_Pct": int_rate,
        "Infer_Mean_us": mean_step_us,
        "Episode_Infer_ms": ep_infer_ms,
        "Passes_Benchmark": "PASS" if passes_realtime else "FAIL",
        "alg_counts": alg_counts,
        "total_steps": len(alg_log),
        "is_disturbed": (disturbance_profile is not None),
    }
    if record_traj:
        res["traj"] = {
            "k": np.array(k_log),
            "v": np.array(v_log),
            "v_des": np.array(v_des_log),
            "err": err_arr,
            "alg": alg_log,
            "shield_intervened": np.array(int_log, dtype=bool),
            "wind": np.array(wind_log, dtype=np.float64),
            "wind_hat": np.array(wind_hat_log, dtype=np.float64),
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
    ap.add_argument("--max-accel", type=float, default=1.85, help="Max traction acceleration (m/s^2)")
    ap.add_argument("--max-decel", type=float, default=1.65, help="Max service braking deceleration (m/s^2)")
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
        total_horizon=args.horizon,
        min_phase_len=25,
        max_accel=args.max_accel,
        max_decel=args.max_decel,
        seed=seed,
    )

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

    mpc_standard = PureLiuMPC(
        n_p=10, n_c=5, q1=160.0, q2=0.05, q3=0.35, adaptive_shift=False,
        max_traction_accel=args.max_accel, max_braking_decel=args.max_decel
    )
    mpc_idea_c = PureLiuMPC(
        n_p=5, n_c=3, q1=160.0, q2=0.05, q3=0.35, adaptive_shift=True,
        max_traction_accel=args.max_accel, max_braking_decel=args.max_decel
    )

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
        ("Adaptive MPC (Idea B: Shift-Q3)", "MPC", None, mpc_idea_c, None, None),
        ("Hardcoded-Switch IPID (Havaei)", "HARDCODED", None, None, None, None),
        ("RL-Switch Way 2 (Unshielded)", "RL_SOFT", rl_way2, None, None, None),
        ("Shielded RL-Switch (1-Step DCBF)", "RL_SOFT", rl_way2, None, None, shield_dcbf),
        ("Shielded RL-Switch (DMPS Rollout)", "RL_SOFT", rl_way2, None, None, shield_dmps),
        ("Hybrid RL-IPID-MPC (Unshielded)", "RL_HYBRID", rl_hybrid, None, None, None),
        ("Shielded Hybrid RL-MPC (Switch MPS)", "RL_HYBRID", rl_hybrid, None, None, shield_switch),
        ("Shielded Hybrid RL-MPC (1-Step DCBF)", "RL_HYBRID", rl_hybrid, None, None, shield_dcbf),
        ("Shielded Hybrid RL-MPC (DMPS Rollout)", "RL_HYBRID", rl_hybrid, None, None, shield_dmps),
        ("Shielded Adaptive MPC (Switch MPS)", "MPC", None, mpc_idea_c, None, shield_switch),
        ("Shielded Adaptive MPC (1-Step DCBF)", "MPC", None, mpc_idea_c, None, shield_dcbf),
        ("Shielded Adaptive MPC (DMPS Rollout)", "MPC", None, mpc_idea_c, None, shield_dmps),
    ])

    agg = {name: [] for name, _, _, _, _, _ in eval_specs}
    sample_trajs = {}
    sample_schedule = None

    for ep in range(args.eval_runs):
        v_rand, sched_rand = generate_truly_random_trajectory(
            total_horizon=args.horizon,
            min_phase_len=25,
            max_accel=args.max_accel,
            max_decel=args.max_decel,
            seed=80000 + ep,
        )

        # Seeded random generator per episode: ep == 0 always gets a storm disturbance for plotting,
        # and remaining episodes have a 60% chance of heavy storm gusts
        ep_rng = np.random.default_rng(90000 + ep)
        if ep == 0 or ep_rng.random() < 0.60:
            d_rand = hrl.generate_wind_disturbance_profile(
                total_horizon=args.horizon,
                seed=90000 + ep,
            )
        else:
            d_rand = None

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
                disturbance_profile=d_rand,
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

    # -------------------------------------------------------------------------
    # Diagnostic Scissor Bands Centered Strictly on the Neural Emulator Rollout
    # -------------------------------------------------------------------------
    v_des_ref = sample_trajs["Shielded Hybrid RL-MPC (DMPS Rollout)"]["v_des"]
    v_des_full_diag = np.concatenate([[0.0], v_des_ref, [v_des_ref[-1]]])

    env_diag = PathAgnosticHavaeiEnv(
        v_des_profile=v_des_full_diag,
        plant_mode="linear_learned",
        learned_linear_model=hrl.LAST_TRAINED_EMULATOR,
        lookahead_n=args.lookahead_n,
    )
    band_info = compute_and_print_shield_scissor_band(
        env_diag,
        sample_trajs["Shielded Hybrid RL-MPC (DMPS Rollout)"],
        shield=shield_dmps,
        learned_policy=rl_hybrid,
    )

    # ---------------- Wind Power & Shield Rejection Diagnostic ----------------
    wind_arr = sample_trajs["Shielded Adaptive MPC (DMPS Rollout)"]["wind"]
    gust_mask = np.abs(wind_arr) > 0.05
    calm_mask = ~gust_mask
    peak_wind_accel = float(np.max(np.abs(wind_arr))) if np.any(gust_mask) else 0.0
    # Equivalent aerodynamic wind speed (km/h) from Davis quadratic drag A*v_wind^2 = M_eff * a_wind
    equiv_wind_kmh = 3.6 * float(np.sqrt(peak_wind_accel * 85000.0 * 1.08 / 14.5))
    unsh_gust_err = float(np.max(sample_trajs["Adaptive MPC (Idea B: Shift-Q3)"]["err"][gust_mask])) if np.any(gust_mask) else 0.0
    sh_gust_err = float(np.max(sample_trajs["Shielded Adaptive MPC (DMPS Rollout)"]["err"][gust_mask])) if np.any(gust_mask) else 0.0
    int_in_gust = int(np.sum(band_info["intervened"][gust_mask]))
    int_in_calm = int(np.sum(band_info["intervened"][calm_mask]))
    print(
        f"WIND DISTURBANCE FEEDBACK | Peak Gust: {peak_wind_accel:.2f} m/s^2 (~{equiv_wind_kmh:.1f} km/h) | "
        f"Max Gust Drift: Unshielded = {unsh_gust_err:.4f} m/s vs Shielded = {sh_gust_err:.4f} m/s "
        f"(-{100.0 * (unsh_gust_err - sh_gust_err) / max(1e-6, unsh_gust_err):.1f}%) | "
        f"Interventions: {int_in_gust} in Gusts, {int_in_calm} in Calm"
    )

    # ---------------- Save 6-Panel Plot ----------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    plot_methods = [
        "Pure MPC (Standard Fixed Q3)",
        "Adaptive MPC (Idea B: Shift-Q3)",
        "Shielded Adaptive MPC (DMPS Rollout)",
        "Hybrid RL-IPID-MPC (Unshielded)",
        "Shielded Hybrid RL-MPC (DMPS Rollout)",
    ]
    k_grid = sample_trajs[plot_methods[0]]["k"]

    # Shade active wind gust intervals in Panel 1
    axes[0, 0].fill_between(
        k_grid, 0.0, 18.0, where=gust_mask,
        color="deepskyblue", alpha=0.14, label=f"Storm Wind Gust (Peak ~{equiv_wind_kmh:.0f} km/h)"
    )
    axes[0, 0].fill_between(
        k_grid, band_info["scissor_lo"], band_info["scissor_hi"],
        color="gold", alpha=0.35, label="99th-Percentile Scissor (Goodall 2026)"
    )
    int_mask = band_info["intervened"]
    if np.any(int_mask):
        axes[0, 0].scatter(
            k_grid[int_mask], sample_trajs["Shielded Hybrid RL-MPC (DMPS Rollout)"]["v"][int_mask],
            color="red", s=12, zorder=5, label="Shield Intervention (Active in Gust)"
        )
    axes[0, 0].plot(k_grid, v_des_ref, "k--", linewidth=2.0, label="Target Profile")
    for m_name in plot_methods:
        axes[0, 0].plot(k_grid, sample_trajs[m_name]["v"], linewidth=1.3, label=m_name)
    axes[0, 0].set_ylim(-0.5, 17.8)
    axes[0, 0].set_title(f"1. Unseen Sample Trajectory [0, {args.horizon}] Tracking & Wind Zones")
    axes[0, 0].set_xlabel("Distance Sample")
    axes[0, 0].set_ylabel("Speed (m/s)")
    axes[0, 0].legend(fontsize=6.2, loc="upper left")
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Instantaneous Tracking Error + Twin-Axis Wind Power Indicator
    ax_wind = axes[0, 1].twinx()
    ax_wind.fill_between(
        k_grid, 0.0, np.abs(wind_arr),
        color="deepskyblue", alpha=0.22, label="Wind Gust Power |d_wind| (m/s²)"
    )
    ax_wind.set_ylabel("Wind Power |d_wind| (m/s²)", color="steelblue", fontsize=8.5)
    ax_wind.tick_params(axis="y", labelcolor="steelblue", labelsize=8)
    ax_wind.set_ylim(0.0, max(1.2, peak_wind_accel * 1.35))

    for m_name in plot_methods:
        axes[0, 1].plot(k_grid, sample_trajs[m_name]["err"], linewidth=1.2, label=m_name)
    axes[0, 1].set_title("2. Tracking Error |v - v_des| vs. Wind Gust Power Indicator")
    axes[0, 1].set_xlabel("Distance Sample")
    axes[0, 1].set_ylabel("Error (m/s)")
    axes[0, 1].legend(fontsize=6.5, loc="upper left")
    axes[0, 1].grid(True, alpha=0.3)

    shift_seg = next((s for s in sample_schedule if s["mode"] in (2, 5)), sample_schedule[1])
    z0 = max(0, shift_seg["start"] - 4)
    z1 = min(len(k_grid), shift_seg["start"] + 16)
    sl = slice(z0, z1)
    axes[0, 2].fill_between(
        k_grid[sl], band_info["v_min"][sl], band_info["v_max"][sl],
        color="mediumseagreen", alpha=0.22, label="100% Max-Error Envelope (Wang 2023)"
    )
    axes[0, 2].fill_between(
        k_grid[sl], band_info["scissor_lo"][sl], band_info["scissor_hi"][sl],
        color="gold", alpha=0.45, label="99th-Percentile Scissor (Goodall 2026)"
    )
    axes[0, 2].plot(k_grid[sl], v_des_ref[sl], "k--", linewidth=2.2, label="Target Shift")
    for m_name in plot_methods:
        axes[0, 2].plot(k_grid[sl], sample_trajs[m_name]["v"][sl], marker="o", markersize=3, label=m_name)
    axes[0, 2].set_title("3. Zoom on Unseen Profile Shift Transition")
    axes[0, 2].set_xlabel("Distance Sample")
    axes[0, 2].legend(fontsize=6.5)
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
        "IdeaB MPC",
        "IPID",
        "RL W2",
        "Hyb RL",
        "Hyb+DMPS",
        "MPC+Sw",
        "MPC+DCBF",
        "MPC+DMPS",
    ]
    bar_keys = [
        "Pure MPC (Standard Fixed Q3)",
        "Adaptive MPC (Idea B: Shift-Q3)",
        "Hardcoded-Switch IPID (Havaei)",
        "RL-Switch Way 2 (Unshielded)",
        "Hybrid RL-IPID-MPC (Unshielded)",
        "Shielded Hybrid RL-MPC (DMPS Rollout)",
        "Shielded Adaptive MPC (Switch MPS)",
        "Shielded Adaptive MPC (1-Step DCBF)",
        "Shielded Adaptive MPC (DMPS Rollout)",
    ]
    scores_bar = [float(np.mean([r["Paper_Score"] for r in agg[k]])) for k in bar_keys]
    axes[1, 1].bar(
        names_bar,
        scores_bar,
        color=[
            "orange",
            "darkorange",
            "steelblue",
            "seagreen",
            "limegreen",
            "mediumseagreen",
            "salmon",
            "indianred",
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
    plt.close(fig)
    print(f"\nSaved 6-panel multi-trajectory benchmark plot to: {args.plot}")

    # ---------------- Save Dedicated 4-Panel Shielded Zoom & Wind Feedback Plot ----------------
    zoom_plot_path = args.plot.replace(".png", "_shielded_zoom.png")
    fig_z, axes_z = plt.subplots(2, 2, figsize=(16, 10))
    shielded_3 = [
        ("Adaptive MPC (Idea B: Shift-Q3)", "darkorange", "--"),
        ("Shielded Adaptive MPC (DMPS Rollout)", "royalblue", "-."),
        ("Shielded Hybrid RL-MPC (DMPS Rollout)", "crimson", "-"),
    ]

    def plot_scissor_panel(ax, sl_idx, title_str, show_wind: bool = False):
        ax.fill_between(
            k_grid[sl_idx], band_info["v_min"][sl_idx], band_info["v_max"][sl_idx],
            color="mediumseagreen", alpha=0.22, label="100% Max-Error & Safe Envelope (Wang 2023)"
        )
        ax.fill_between(
            k_grid[sl_idx], band_info["scissor_lo"][sl_idx], band_info["scissor_hi"][sl_idx],
            color="gold", alpha=0.48, label="99th-Percentile Scissor (Goodall 2026)"
        )
        ax.plot(k_grid[sl_idx], v_des_ref[sl_idx], "k--", linewidth=1.8, label="Target Profile v_des")
        for s_name, col, ls in shielded_3:
            ax.plot(k_grid[sl_idx], sample_trajs[s_name]["v"][sl_idx], color=col, linestyle=ls, linewidth=1.6, label=s_name)
        int_sl = band_info["intervened"][sl_idx]
        if np.any(int_sl):
            ax.scatter(
                k_grid[sl_idx][int_sl],
                sample_trajs["Shielded Hybrid RL-MPC (DMPS Rollout)"]["v"][sl_idx][int_sl],
                color="red", s=18, zorder=6, label="Active Shield Intervention"
            )
        if show_wind:
            ax_w = ax.twinx()
            w_sl = wind_arr[sl_idx]
            wh_sl = sample_trajs["Shielded Adaptive MPC (DMPS Rollout)"]["wind_hat"][sl_idx]
            ax_w.fill_between(k_grid[sl_idx], 0.0, w_sl, color="deepskyblue", alpha=0.20, label="True Wind d_ext (m/s²)")
            ax_w.plot(k_grid[sl_idx], wh_sl, color="teal", linestyle=":", linewidth=1.4, label="Observer d_hat (m/s²)")
            ax_w.set_ylabel("Wind Accel (m/s²)", color="teal", fontsize=8.5)
            ax_w.tick_params(axis="y", labelcolor="teal", labelsize=8)
            ax_w.set_ylim(-1.3, 1.3)
            ax_w.legend(fontsize=6.8, loc="lower right")

        ax.set_title(title_str, fontweight="bold")
        ax.set_xlabel("Distance Sample k")
        ax.set_ylabel("Speed (m/s)")
        ax.legend(fontsize=7.0, loc="upper left")
        ax.grid(True, alpha=0.3)

    plot_scissor_panel(axes_z[0, 0], slice(0, len(k_grid)), f"1. Full Trajectory [0, {args.horizon}]: Unshielded vs Shielded MPC/RL")

    # Find first and last wind gust windows for high-resolution zoomed wind feedback
    gust_indices = np.where(gust_mask)[0]
    if len(gust_indices) > 0:
        g1_start = max(0, int(gust_indices[0]) - 12)
        g1_end = min(len(k_grid), int(gust_indices[0]) + 65)
        g2_start = max(0, int(gust_indices[-1]) - 55)
        g2_end = min(len(k_grid), int(gust_indices[-1]) + 15)
    else:
        cruise_seg = next((s for s in sample_schedule if s["mode"] == 3), sample_schedule[0])
        g1_start, g1_end = max(0, cruise_seg["start"]), min(len(k_grid), cruise_seg["end"])
        g2_start, g2_end = max(0, len(k_grid) - 70), len(k_grid)

    plot_scissor_panel(
        axes_z[0, 1], slice(g1_start, g1_end),
        f"2. Zoom on Storm Gust #1 [{g1_start}, {g1_end}]: Speed vs Wind Power Indicator",
        show_wind=True,
    )

    plot_scissor_panel(axes_z[1, 0], sl, f"3. Zoom on Profile Shift [{z0}, {z1}]: Causal Guard & Scissor Bands")

    plot_scissor_panel(
        axes_z[1, 1], slice(g2_start, g2_end),
        f"4. Zoom on Storm Gust #2 [{g2_start}, {g2_end}]: Active Shielding & Instant Cut-Off",
        show_wind=True,
    )

    fig_z.tight_layout()
    fig_z.savefig(zoom_plot_path, dpi=150)
    plt.close(fig_z)
    print(f"Saved dedicated 4-panel Shielded Hybrid Zoom plot to: {zoom_plot_path}")


    # -------------------------------------------------------------------------
    # Algorithm Selection Breakdown: Undisturbed vs. Disturbed vs. Combined
    # -------------------------------------------------------------------------
    hybrid_check_methods = [
        name for name, ctype, _, _, _, _ in eval_specs if ctype in ("RL_HYBRID", "RL_SOFT")
    ]
    print("\n" + "=" * 132)
    print("HYBRID RL ALGORITHM USAGE STATS (% AND STEP COUNTS): UNDISTURBED vs. DISTURBED vs. COMBINED")
    print("=" * 132)

    for subset_label, filter_fn in [
        ("1. UNDISTURBED TRAJECTORIES ONLY (Clean)", lambda r: not r["is_disturbed"]),
        ("2. DISTURBED TRAJECTORIES ONLY (Storm Wind)", lambda r: r["is_disturbed"]),
        ("3. ALL TRAJECTORIES COMBINED (Clean + Disturbed)", lambda r: True),
    ]:
        sample_runs = [r for r in agg[hybrid_check_methods[0]] if filter_fn(r)]
        n_tracks_sub = len(sample_runs)
        print(f"\n--- {subset_label} ({n_tracks_sub} Tracks) ---")
        print(
            f"{'Controller Method':<37} | {'GA':>14} | {'PSO':>14} | {'DE':>14} | "
            f"{'ICA':>14} | {'MPC':>16} | {'Total Steps':>11}"
        )
        print("-" * 132)
        for m_name in hybrid_check_methods:
            sub_runs = [r for r in agg[m_name] if filter_fn(r)]
            tot_steps = sum(r["total_steps"] for r in sub_runs)
            if tot_steps == 0:
                continue
            cols = []
            for alg_k in HYBRID_ALGORITHMS:
                cnt = sum(r["alg_counts"].get(alg_k, 0) for r in sub_runs)
                pct = 100.0 * cnt / tot_steps
                if alg_k == "MPC":
                    cols.append(f"{pct:6.2f}% ({cnt:5d})")
                else:
                    cols.append(f"{pct:5.2f}% ({cnt:4d})")
            print(
                f"{m_name:<37} | {cols[0]:>14} | {cols[1]:>14} | {cols[2]:>14} | "
                f"{cols[3]:>14} | {cols[4]:>16} | {tot_steps:11d}"
            )
    print("=" * 132)


if __name__ == "__main__":
    main()