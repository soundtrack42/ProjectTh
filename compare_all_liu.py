"""
Complete Benchmark & Comparison Script for LiuTrainEnv:
Compares Unshielded PPO vs. Preemptive Soft Shield vs. Preemptive Hard Shield
across Discrete and/or Continuous action spaces.

Outputs:
  1. Console summary table (matching Brand's Thesis Tables 1-6 & Liu's Table 2)
  2. CSV summary file (`comparison_summary.csv`)
  3. Overlaid comparison plots (`comparison_plots.png`)
"""
from __future__ import annotations

import argparse
import csv
import numpy as np
import torch
import torch.optim as optim

from Env.LiuTrainEnv import LiuTrainEnv
from train_ppo_liu import (
    ActorCritic,
    collect_rollouts,
    normalize_obs,
    ppo_update,
)


def evaluate_policy(env: LiuTrainEnv, model: ActorCritic, shield_mode: str, seed: int = 999):
    """Runs a deterministic evaluation episode and records full trajectory & metrics."""
    obs, info = env.reset(seed=seed)
    done = False

    t_log, v_log, v_ref_log, v_lim_log, p_log, p_ref_log, u_log = [], [], [], [], [], [], []
    total_reward = 0.0
    violations = 0
    modifications = 0
    overspeeds = 0
    steps = 0

    model.eval()
    while not done:
        obs_t = torch.as_tensor(normalize_obs(obs), dtype=torch.float32)
        with torch.no_grad():
            action, _, _, _, violated, modified, *_ = model.act(
                obs_t, info, shield_mode=shield_mode, deterministic=True
            )
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        total_reward += float(reward)
        violations += int(violated)
        modifications += int(modified)
        overspeeds += int(info["overspeed_violation"])
        steps += 1

        t_log.append(info["t"])
        v_log.append(env.vel)
        v_ref_log.append(info["v_ref"])
        v_lim_log.append(info["v_limit"])
        p_log.append(env.pos)
        p_ref_log.append(info["p_ref"])
        u_log.append(info["u"])

    v_arr = np.array(v_log)
    v_ref_arr = np.array(v_ref_log)
    speed_err = v_arr - v_ref_arr

    metrics = {
        "eval_reward": total_reward,
        "viol_rate": violations / steps,
        "mod_rate": modifications / steps,
        "overspeed_rate": overspeeds / steps,
        "max_speed_err": float(np.max(np.abs(speed_err))),
        "mean_speed_err": float(np.mean(np.abs(speed_err))),
        "stop_err_m": float(env.pos - env.station_distance),
        "abs_stop_err_m": float(abs(env.pos - env.station_distance)),
    }
    traj = {
        "t": np.array(t_log),
        "v": v_arr,
        "v_ref": v_ref_arr,
        "v_lim": np.array(v_lim_log),
        "p": np.array(p_log),
        "p_ref": np.array(p_ref_log),
        "u": np.array(u_log),
        "speed_err": speed_err,
    }
    return metrics, traj


def run_single_experiment(
    continuous: bool,
    shield_mode: str,
    n_actions: int,
    epochs: int,
    episodes_per_epoch: int,
    lr: float,
    seed: int,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = LiuTrainEnv(continuous=continuous, n_discrete_actions=n_actions)
    obs_dim = env.observation_space.shape[0]
    action_dim = 1 if continuous else env.n_discrete_actions

    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, continuous=continuous)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_rewards = []
    epoch_viol_rates = []
    epoch_overspeed_rates = []

    for _ in range(1, epochs + 1):
        batch, stats = collect_rollouts(
            env, model, episodes_per_epoch, shield_mode, gamma=0.99, gae_lambda=0.95
        )
        ppo_update(model, optimizer, batch)

        epoch_rewards.append(float(np.mean([s["reward"] for s in stats])))
        epoch_viol_rates.append(float(np.mean([s["viol_rate"] for s in stats])))
        epoch_overspeed_rates.append(float(np.mean([s["overspeed_rate"] for s in stats])))

    eval_metrics, eval_traj = evaluate_policy(env, model, shield_mode, seed=seed + 1000)
    learning_curves = {
        "reward": np.array(epoch_rewards),
        "viol_rate": np.array(epoch_viol_rates),
        "overspeed_rate": np.array(epoch_overspeed_rates),
    }
    return eval_metrics, eval_traj, learning_curves


def main():
    ap = argparse.ArgumentParser(description="Compare all PPO + Shielding configurations on LiuTrainEnv")
    ap.add_argument("--epochs", type=int, default=120, help="Training epochs per run")
    ap.add_argument("--episodes-per-epoch", type=int, default=8)
    ap.add_argument("--n-actions", type=int, default=11)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=1, help="Number of random seeds per config (e.g. 3 or 5)")
    ap.add_argument(
        "--space",
        type=str,
        default="both",
        choices=["discrete", "continuous", "both"],
        help="Which action spaces to include in the comparison",
    )
    ap.add_argument("--plot", type=str, default="comparison_plots.png", help="Output filename for comparison figure")
    ap.add_argument("--csv", type=str, default="comparison_summary.csv", help="Output filename for summary table")
    args = ap.parse_args()

    configs = []
    if args.space in ("discrete", "both"):
        for sh in ["none", "soft", "hard"]:
            configs.append((f"Disc-PPO ({sh.capitalize()})", False, sh))
    if args.space in ("continuous", "both"):
        for sh in ["none", "soft", "hard"]:
            configs.append((f"Cont-PPO ({sh.capitalize()})", True, sh))

    summary_rows = []
    all_curves = {}
    all_trajs = {}

    print(f"=== Running Benchmark: {len(configs)} Configurations x {args.seeds} Seed(s) ({args.epochs} Epochs each) ===\n")

    for label, is_cont, shield_mode in configs:
        print(f"--> Training {label} ...")
        seed_metrics = []
        seed_curves = []
        last_traj = None

        for s in range(args.seeds):
            metrics, traj, curves = run_single_experiment(
                continuous=is_cont,
                shield_mode=shield_mode,
                n_actions=args.n_actions,
                epochs=args.epochs,
                episodes_per_epoch=args.episodes_per_epoch,
                lr=args.lr,
                seed=s * 100,
            )
            seed_metrics.append(metrics)
            seed_curves.append(curves)
            last_traj = traj
            print(
                f"    [Seed {s}] Eval Reward: {metrics['eval_reward']:7.2f} | "
                f"Viol: {metrics['viol_rate']:.3f} | Mod: {metrics['mod_rate']:.3f} | "
                f"Overspd: {metrics['overspeed_rate']:.3f} | "
                f"MaxSpdErr: {metrics['max_speed_err']:.3f}m/s | StopErr: {metrics['stop_err_m']:+.3f}m"
            )

        # Aggregate across seeds
        agg = {"Method": label}
        for k in seed_metrics[0].keys():
            vals = [m[k] for m in seed_metrics]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        summary_rows.append(agg)

        all_curves[label] = {
            k: np.mean([c[k] for c in seed_curves], axis=0) for k in seed_curves[0].keys()
        }
        all_trajs[label] = last_traj

    # -----------------------------------------------------------------------
    # Print Formatted Comparison Table (Brand Sec. 6.5 + Liu Table 2)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 108)
    print(
        f"{'Method':<20} | {'Reward':>14} | {'Viol. Rate':>12} | {'Mod. Rate':>12} | "
        f"{'Overspd Rate':>12} | {'MaxSpdErr(m/s)':>14} | {'StopErr(m)':>11}"
    )
    print("-" * 108)
    for r in summary_rows:
        print(
            f"{r['Method']:<20} | "
            f"{r['eval_reward_mean']:7.2f}±{r['eval_reward_std']:5.2f} | "
            f"{r['viol_rate_mean']:5.3f}±{r['viol_rate_std']:5.3f} | "
            f"{r['mod_rate_mean']:5.3f}±{r['mod_rate_std']:5.3f} | "
            f"{r['overspeed_rate_mean']:5.3f}±{r['overspeed_rate_std']:5.3f} | "
            f"{r['max_speed_err_mean']:6.3f}±{r['max_speed_err_std']:5.3f} | "
            f"{r['stop_err_m_mean']:+5.2f}±{r['stop_err_m_std']:4.2f}"
        )
    print("=" * 108)

    # Save CSV
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSaved summary table to {args.csv}")

    # -----------------------------------------------------------------------
    # Plot All Curves & Trajectories Side-by-Side
    # -----------------------------------------------------------------------
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        epochs_x = np.arange(1, args.epochs + 1)

        # Top Row: Training Curves (Reward, Violation Rate, Overspeed Rate)
        for label, curves in all_curves.items():
            ls = "--" if "Cont" in label else "-"
            axes[0, 0].plot(epochs_x, curves["reward"], linestyle=ls, label=label)
            axes[0, 1].plot(epochs_x, curves["viol_rate"], linestyle=ls, label=label)
            axes[0, 2].plot(epochs_x, curves["overspeed_rate"], linestyle=ls, label=label)

        axes[0, 0].set_title("Training: Episodic Reward -(J1+J2+J3)")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=8)

        axes[0, 1].set_title("Training: Raw Policy Violation Rate (Brand Sec. 6.3)")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Violation Rate")
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].set_title("Training: True Executed Overspeed Rate")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Overspeed Rate")
        axes[0, 2].grid(True, alpha=0.3)

        # Bottom Row: Final Evaluation Rollout (Liu Fig. 3, Fig. 4, Control Input)
        first_traj = next(iter(all_trajs.values()))
        axes[1, 0].plot(first_traj["t"], first_traj["v_lim"], "k-", linewidth=2.0, label="Speed Limit")
        axes[1, 0].plot(first_traj["t"], first_traj["v_ref"], "k--", linewidth=1.8, label="Reference Speed")

        for label, traj in all_trajs.items():
            ls = "--" if "Cont" in label else "-"
            axes[1, 0].plot(traj["t"], traj["v"], linestyle=ls, linewidth=1.3, label=label)
            axes[1, 1].plot(traj["t"], traj["speed_err"], linestyle=ls, linewidth=1.2, label=label)
            axes[1, 2].plot(traj["t"], traj["u"], linestyle=ls, linewidth=1.1, label=label)

        axes[1, 0].set_title("Eval: Speed Tracking Curve (cf. Liu Fig. 3)")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Speed (m/s)")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=7)

        axes[1, 1].axhline(0.0, color="k", linestyle=":", alpha=0.6)
        axes[1, 1].set_title("Eval: Speed Tracking Error (cf. Liu Fig. 4)")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("v - v_ref (m/s)")
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].set_title("Eval: Actuator Control Input u(k)")
        axes[1, 2].set_xlabel("Time (s)")
        axes[1, 2].set_ylabel("u(k) [-1, 1]")
        axes[1, 2].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Saved comparison figure to {args.plot}")


if __name__ == "__main__":
    main()