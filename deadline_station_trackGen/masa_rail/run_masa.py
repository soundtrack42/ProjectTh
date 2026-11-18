"""
Runs one MASA-Safe-RL method on the RailLeg environment and writes a JSON result.

    python run_masa.py --algo QL --shield none --mode model --frames 300000 --seed 0 --out res.json

--shield none        : the algorithm alone, PCTL constraint (alpha) as in MASA's tabular benchmarks
--shield postposed   : MASA PostposedLTLShield (G !unsafe) around the algorithm (unsafe proposals replaced)
--shield preemptive  : MASA PreemptiveLTLShield; the learner picks only from the safe-action mask
                       (MaskedQL: QL whose action selection is restricted to the mask; learning unchanged)
ProbShieldPPO and CPOShield carry their own probabilistic shield (use --shield none).

Hyperparameters are MASA's `defaults` block of each algorithm's YAML config, with gamma = 0.99 and the
cost limit / safety bound set to alpha (MASA sets both to the env's alpha in its benchmarks).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax.random as jr  # noqa: E402
from ruamel.yaml import YAML  # noqa: E402

import masa  # noqa: E402,F401
import masa.algorithms  # noqa: E402,F401
from masa.common import registry  # noqa: E402
from masa.common.constraints.ltl_safety import LTLSafetyEnv  # noqa: E402
from masa.common.labelled_env import LabelledEnv  # noqa: E402
from masa.common.utils import make_env  # noqa: E402
from masa.common.wrappers import (  # noqa: E402
    ConstraintMonitor, ConstraintPersistentWrapper, RewardMonitor, TimeLimit, get_wrapped,
)
from masa.deterministic_shield import PostposedLTLShield, PreemptiveLTLShield  # noqa: E402

import rail_env as R  # noqa: E402

MAX_STEPS = 300
ALPHA = 0.01
YAML_FILE = {
    "QL": "QL.yaml", "QL_LAMBDA": "QL_LAMBDA.yaml", "RECOVERY_RL": "recovery_rl.yaml", "RECREG": "recreg.yaml",
    "RECREG_EXACT": "RECREG_EXACT.yaml", "RECREG_MODEL_BASED": "RECREG_MODEL_BASED.yaml", "SEM": "sem.yaml",
    "LCRL": "lcrl.yaml", "PPO": "ppo.yaml", "A2C": "a2c.yaml", "TRPO": "TRPO.yaml", "PPOLag": "PPOLag.yaml",
    "TRPOLag": "TRPOLag.yaml", "CPO": "CPO.yaml", "ProbShieldPPO": "ProbShieldPPO.yaml", "CPOShield": "CPO.yaml",
    "MaskedQL": "QL.yaml",
}

registry.ENV_REGISTRY.register("RailLeg", "rail_env:RailLeg")


def algo_kwargs(algo: str) -> dict:
    cfg_dir = Path(masa.__file__).parent / "configs" / "algorithms"
    kw = dict(YAML(typ="safe").load(open(cfg_dir / YAML_FILE[algo]))["defaults"])
    for k in ("gamma", "task_gamma"):
        if k in kw:
            kw[k] = 0.99
    if "cost_limit" in kw:
        kw["cost_limit"] = ALPHA
    if "init_safety_bound" in kw:
        kw["init_safety_bound"] = ALPHA
    if "safety_prob" in kw:
        kw["safety_prob"] = ALPHA
    if algo == "CPOShield":
        kw.update(label_fn=R.label_fn, cost_fn=R.cost_fn, init_safety_bound=ALPHA, theta=1e-10, max_vi_steps=10_000)
    return kw


class MismatchGuard(ConstraintPersistentWrapper):
    """Physics mode only: a deterministic shield raises when the plant leaves its model; end the
    episode there (truncated) and count the event instead of crashing the learner."""

    def __init__(self, env):
        super().__init__(env)
        self.mismatches = 0
        self._last_obs = None

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        try:
            obs, r, term, trunc, info = self.env.step(action)
            self._last_obs = obs
            return obs, r, term, trunc, info
        except RuntimeError as e:
            if "absent from the shield model" in str(e) or "outside the winning region" in str(e):
                self.mismatches += 1
                return self._last_obs, 0.0, False, True, {"model_mismatch": True}
            raise


class Monitors(ConstraintMonitor):
    """ConstraintMonitor that reads the constraint from the LTLSafetyEnv below the shield."""

    def __init__(self, env, constraint_env):
        ConstraintPersistentWrapper.__init__(self, env)
        self._constraint_env = constraint_env


def build_env(shield: str, mode: str):
    if shield == "none":
        return make_env("RailLeg", "PCTL", MAX_STEPS, label_fn=R.label_fn,
                        constraint_kwargs=dict(cost_fn=R.cost_fn, alpha=ALPHA), env_kwargs=dict(mode=mode))
    env = TimeLimit(R.RailLeg(mode=mode), MAX_STEPS)
    env = LabelledEnv(env, R.label_fn)
    ltl = LTLSafetyEnv(env, dfa=R.make_dfa(), obs_type="discrete")
    env = (PostposedLTLShield if shield == "postposed" else PreemptiveLTLShield)(ltl)
    if mode == "physics":
        env = MismatchGuard(env)
    env = Monitors(env, ltl)
    return RewardMonitor(env)


def shield_of(env):
    for cls in (PostposedLTLShield, PreemptiveLTLShield):
        try:
            return get_wrapped(env, cls)
        except Exception:
            pass
    return None


def make_masked_ql():
    from masa.algorithms.tabular import QL

    class MaskedQL(QL):
        """QL for preemptive shielding: actions are chosen only among the shield's safe actions.
        Exploration (Boltzmann / epsilon) and the Q update are the library's."""
        mask_env = None

        def act(self, key, obs, deterministic=False):
            sh = shield_of(self.mask_env if self.mask_env is not None else self.env)
            mask = sh.action_masks()
            q = np.asarray(self.Q[obs], dtype=np.float64)
            allowed = np.flatnonzero(mask)
            if deterministic:
                return int(allowed[np.argmax(q[allowed])])
            rng = np.random.default_rng(int(jr.randint(key, (), 0, 2**31 - 1)))
            if self.exploration == "epsilon_greedy" and rng.random() < self._epsilon:
                return int(rng.choice(allowed))
            z = (q[allowed] - q[allowed].max()) / self.boltzmann_temp
            p = np.exp(z) / np.exp(z).sum()
            return int(rng.choice(allowed, p=p))

    return MaskedQL


def evaluate(model, env, n_episodes: int, seed: int, masked: bool) -> dict:
    base = env.unwrapped
    base.stats.clear()
    base._ep = {}
    guard = None
    try:
        guard = get_wrapped(env, MismatchGuard)
        guard.mismatches = 0
    except Exception:
        pass
    sh = shield_of(env)
    interventions, steps = 0, 0
    key = jr.PRNGKey(seed)
    if masked:
        model.mask_env = env
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            key, sub = jr.split(key)
            a = model.act(sub, obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            interventions += int(bool(info.get("shield_intervened", False)))
            steps += 1
            done = term or trunc
    if masked:
        model.mask_env = None
    st = base.close_stats()
    n = max(1.0, st.get("episodes", 0.0))
    out = dict(
        episodes=int(st.get("episodes", 0)),
        goal_rate=st.get("goal", 0.0) / n,
        unsafe_episode_rate=st.get("unsafe_episodes", 0.0) / n,
        overrun_rate=st.get("overrun", 0.0) / n,
        true_overspeed_episode_rate=st.get("true_overspeed_episodes", 0.0) / n,
        mean_goal_time_s=(st.get("goal_time_s", 0.0) / st["goal"]) if st.get("goal", 0) else None,
        energy_kwh_per_episode=st.get("energy_kwh", 0.0) / n,
        shield_intervention_rate=(interventions / steps) if (sh is not None and steps) else None,
        model_mismatch_episodes=(guard.mismatches if guard is not None else None),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    ap.add_argument("--shield", default="none", choices=["none", "postposed", "preemptive"])
    ap.add_argument("--mode", default="model", choices=["model", "physics"])
    ap.add_argument("--frames", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    masked = args.algo == "MaskedQL"
    if masked:
        algo_cls = make_masked_ql()
        assert args.shield == "preemptive"
    elif args.algo == "CPOShield":
        from masa.prob_shield.cpo_shield import CPOShield as algo_cls
    else:
        algo_cls = registry.get_algorithm(args.algo)

    train_env = build_env(args.shield, args.mode)
    eval_env_fn = lambda: build_env(args.shield, args.mode)
    kw = algo_kwargs(args.algo)
    t0 = time.time()
    model = algo_cls(train_env, tensorboard_logdir=None, seed=args.seed, device="cpu", verbose=0,
                     env_fn=eval_env_fn, **kw)
    model.train(args.frames, num_eval_episodes=0, eval_freq=0, log_freq=0)
    train_time = time.time() - t0

    tr = train_env.unwrapped.close_stats()
    n_tr = max(1.0, tr.get("episodes", 0.0))
    try:
        tr_mism = get_wrapped(train_env, MismatchGuard).mismatches
    except Exception:
        tr_mism = None
    ev_env = model._get_eval_env() if args.algo == "ProbShieldPPO" else eval_env_fn()
    res = dict(
        algo=args.algo, shield=args.shield, mode=args.mode, frames=args.frames, seed=args.seed,
        train_time_s=round(train_time, 1),
        train=dict(episodes=int(tr.get("episodes", 0)),
                   unsafe_episode_rate=tr.get("unsafe_episodes", 0.0) / n_tr,
                   unsafe_episodes=int(tr.get("unsafe_episodes", 0)),
                   true_overspeed_episodes=int(tr.get("true_overspeed_episodes", 0)),
                   goal_rate=tr.get("goal", 0.0) / n_tr,
                   model_mismatch_episodes=tr_mism),
        eval=evaluate(model, ev_env, args.eval_episodes, 100_000 + args.seed * 1000, masked),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
