"""
Epoch-based training of a *stochastic* (softmax) policy on TrainEnv.

Run from the repo root:     python train_trains.py --epochs 200 --episodes-per-epoch 20
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from FirstDiscreteRL.trainsEnv import TrainEnv, cost_fn, make_label_fn


# Stochastic tabular policy:  pi(a|s) = softmax(theta[s]) over VALID actions only
def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.where(mask, logits, -np.inf)         # set -inf invalid actions
    z = z - z.max()                             # make value converge on [1,0) range after the .exp()
    e = np.exp(z)
    return e / e.sum()


class SoftmaxPolicy:
    def __init__(self, obs_masks: np.ndarray, rng: np.random.Generator):
        self.masks = obs_masks
        self.theta = np.zeros(obs_masks.shape)              # logits init (uniform distribution)
        self.rng = rng

    def probs(self, obs: int) -> np.ndarray:
        return masked_softmax(self.theta[obs], self.masks[obs])

    def sample(self, obs: int):
        p = self.probs(obs)
        return int(self.rng.choice(len(p), p=p)), p         # select action based on p distribution

    def greedy(self, obs: int) -> int:
        return int(np.argmax(self.probs(obs)))


# Rollouts
def run_episode(env, policy, label_fn, cost_penalty=0.0):
    obs, _ = env.reset()                                    # obs is current node + visited
    traj, entropies = [], []                                # stats
    raw_return, cost, invalid, done, term = 0.0, 0.0, 0, False, False
    while not done:
        a, p = policy.sample(obs)
        nxt, r, term, trunc, info = env.step(a)
        c = cost_fn(label_fn(nxt))
        traj.append((obs, a, r - cost_penalty * c))   # shaped reward used for learning
        raw_return += r
        cost += c
        invalid += int(info["invalid_move"])
        q = p[p > 0]                                        # removes -inf prob's actions (log(-inf)*-inf = 0*-inf...)
        entropies.append(float(-(q * np.log(q)).sum()))
        obs, done = nxt, term or trunc
    stats = {
        "return": raw_return,                 # env reward, without the cost penalty
        "length": len(traj),
        "success": float(term),
        "cost": cost,
        "invalid": invalid,
        "entropy": float(np.mean(entropies)),
    }
    return traj, stats


# REINFORCE with a learned per-state baseline V(s)
def reinforce_update(policy, V, batch, lr, baseline_lr, gamma):
    grad = np.zeros_like(policy.theta)
    for traj in batch:
        G, returns = 0.0, []
        for (_, _, r) in reversed(traj):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        for (s, a, _), G in zip(traj, returns):
            adv = G - V[s]
            V[s] += baseline_lr * (G - V[s])
            g = -policy.probs(s)                            # gradient of log pi(a|s) = onehot(a) - pi(.|s)  (log cause policy is softmaxed, exponential fn)
            g[a] += 1.0                                     # onehot(a) - pi(.|s)
            grad[s] += adv * g                              # adv * d( log pi(a|s) )
    policy.theta += lr * grad / len(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--cost-penalty", type=float, default=0.0, help="reward shaping: subtract this for hazard steps (0 = ignore costs)")
    ap.add_argument("--episodes-per-epoch", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)           # learning rate of policy update
    ap.add_argument("--baseline-lr", type=float, default=0.1)   # learning rate of optimal reward
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--plot", type=str, default=None)           # needs matplotlib
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)                      # random number generator for variability in learning
    env = TrainEnv(render_mode="ansi")
    env.reset(seed=args.seed)
    label_fn = make_label_fn()
    policy = SoftmaxPolicy(env.obs_masks, rng)
    V = np.zeros(env.observation_space.n)

    keys = ["return", "length", "success", "cost", "invalid", "entropy"]            # stats of episodes
    history = []
    print(f"{'epoch':>5} | {'return':>8} {'length':>7} {'success':>8} {'cost':>6} {'invalid':>8} {'entropy':>8}")
    for epoch in range(1, args.epochs + 1):
        batch, stats = [], []
        for _ in range(args.episodes_per_epoch):
            traj, st = run_episode(env, policy, label_fn, args.cost_penalty)
            batch.append(traj)
            stats.append(st)
        reinforce_update(policy, V, batch, args.lr, args.baseline_lr, args.gamma)

        row = {k: float(np.mean([s[k] for s in stats])) for k in keys}
        row["epoch"] = epoch
        history.append(row)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"{epoch:5d} | {row['return']:8.2f} {row['length']:7.1f} {row['success']:8.2f} "
                  f"{row['cost']:6.2f} {row['invalid']:8.2f} {row['entropy']:8.3f}")

    # most-likely path of the final policy
    obs, _ = env.reset()
    path, done = [env._node], False
    print(env.render(), end="\n\n")
    while not done and len(path) < 50:
        obs, _, term, trunc, _ = env.step(policy.greedy(obs))
        path.append(env._node)
        done = term or trunc
        print(env.render(), end="\n\n")
    print("\nGreedy path of final policy:", path)
    print("pi(.|start) =", np.round(policy.probs(env.reset()[0]), 3))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch"] + keys)
            w.writeheader()
            w.writerows(history)
        print("saved", args.csv)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        for ax, k in zip(axes.ravel(), keys):
            ax.plot([h["epoch"] for h in history], [h[k] for h in history])
            ax.set_title(k)
            ax.set_xlabel("epoch")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print("saved", args.plot)


if __name__ == "__main__":
    main()