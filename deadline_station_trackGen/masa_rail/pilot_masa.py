"""
Pilot-learner study: train our cosine pilot with every MASA-Safe-RL algorithm that accepts our pilot
interface (18 continuous features, 3 continuous actions in [-1, 1]) and export the best checkpoint.

Same protocol as `train_pure_ppo_cosine_pilot` (our PPO):
  * training journeys: 400 epochs x 3 generated journeys (seed*1000 + epoch*53 + b*17, b = 2 extreme),
    consumed in the same order; budget = the same 1,200 journeys; seeds 42, 7, 2024 as for our pilots
    (the seed sets the journeys, the validation set and the learner initialisation);
  * reward = env reward / 20, gamma = 0.996;
  * checkpoint selection: every 15 journeys (= 5 epochs), deterministic LossScore on the same 6
    validation journeys (seeds 70000+), lowest kept.
Everything else is MASA's default configuration for the algorithm. The constrained learners (PPOLag,
TRPOLag, CPO) receive a per-step cost = 1 when any 0.5 s substep violates the benchmark's comfort/safety
pressures (rho_1 > 1.02, rho_2 > 1.02 or rho_3 > 1.05), with a budget of `--cost-limit` per leg.

Shield-in-the-loop variant (--shield-train): every training leg runs under UCB-DMPS (the benchmark's
shield, same configuration, one UCB state per journey), the reward loses --interv-penalty for each macro
step the shield overrides, and the checkpoint is the lowest shielded validation LossScore among those whose
validation shield rate is <= --max-val-shield % (default 5, half the 10% target; a barely trained pilot
rescued by the shield must not be selected). All checkpoints are kept in <tag>_ckpts.npz.

    python pilot_masa.py --algo TRPO --out DIR
Writes DIR/<algo>_pilot.npz (actor MLP weights) and DIR/<algo>_history.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ.parent / "core"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402
import jax.random as jr  # noqa: E402
from ruamel.yaml import YAML  # noqa: E402

torch.set_num_threads(1)
os.chdir(PROJ)

import dynamic_railroad_mpc as D  # noqa: E402
from railroad_corridor import RailroadCorridor, generate_multi_station_scenario  # noqa: E402

import masa  # noqa: E402,F401
import masa.algorithms  # noqa: E402,F401
from masa.common import registry  # noqa: E402
from masa.common.constraints.cmdp import CumulativeCostEnv  # noqa: E402
from masa.common.labelled_env import LabelledEnv  # noqa: E402
from masa.common.wrappers import ConstraintMonitor, RewardMonitor, TimeLimit  # noqa: E402
import jax  # noqa: E402

# MASA enables JAX x64 globally when its probabilistic-shield modules are imported (pctl.py,
# interval_bound_vi.py). That breaks the Fisher-vector product of TRPO/TRPOLag/CPO on continuous
# actions (float32 params vs float64 tangents), also on MASA's own ContinuousCartpole. The pilot study
# uses no probabilistic shield, so x64 is switched back off; the algorithms themselves are unchanged.
registry.get_algorithm("PPO")  # trigger the on-policy imports first
jax.config.update("jax_enable_x64", False)

PILOT_ALGOS = ["PPO", "A2C", "TRPO", "PPOLag", "TRPOLag", "CPO"]
YAML_FILE = {"PPO": "ppo.yaml", "A2C": "a2c.yaml", "TRPO": "TRPO.yaml", "PPOLag": "PPOLag.yaml",
             "TRPOLag": "TRPOLag.yaml", "CPO": "CPO.yaml"}
EPOCHS, JOURNEYS_PER_EPOCH, VAL_EVERY_EPOCHS = 400, 3, 5   # our pilots: 400 epochs (1,200 journeys)
TRAIN_SEED = 42                                           # overridden by --seed
GAMMA = 0.996
CHUNK_FRAMES = 2048


def load_emulator():
    emu = D.LearnedLinearizationModel(hidden_dim=64)
    emu.load_state_dict(torch.load("checkpoints/best_emulator.pt", map_location="cpu"))
    emu.eval()
    return emu


class RailPilotEnv(gym.Env):
    """Our MPCTransitionRailroadEnv (cosine shaper, unshielded) as a gymnasium env; one episode = one leg.
    Legs are served from the fixed training-journey stream. A reset while a leg is still running is a
    no-op, so MASA's train() restarts (one per chunk) do not skip journeys."""

    def __init__(self, emu, journeys=None, cosine: bool = True, shield: bool = False, penalty: float = 0.0):
        self._emu, self._cosine, self._shield, self._penalty = emu, cosine, shield, float(penalty)
        self._ucb: dict = {}
        self.interventions = 0
        self.macro_steps = 0
        self.observation_space = spaces.Box(-np.inf, np.inf, (18,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.env = D.MPCTransitionRailroadEnv(corridor=RailroadCorridor(), traj_shield=None, emulator=emu,
                                              use_cosine_shaper=cosine)
        self.env.a_start_center = 0.96
        self.env.coast_enabled = False
        self._journeys = journeys          # None = training stream
        self._j = 0
        self._legs: list = []
        self._in_leg = False
        self._obs = None
        self.journeys_started = 0
        self.violation = False
        self.label_fn = lambda obs: {"unsafe"} if self.violation else set()

    def _next_leg(self):
        if not self._legs:
            if self._journeys is None:
                ep, b = divmod(self._j, JOURNEYS_PER_EPOCH)
                scen = generate_multi_station_scenario(
                    seed=TRAIN_SEED * 1000 + (ep + 1) * 53 + b * 17, n_legs=0, wind_mode="mixed",
                    allow_infeasible_breaks=False, extreme_noise=(b == 2))
            else:
                scen = self._journeys[self._j % len(self._journeys)]
            self._j += 1
            self.journeys_started += 1
            self._ucb = {}                 # UCB contract statistics persist within a journey, as in the benchmark
            self._legs = list(scen["legs"])
        return self._legs.pop(0)

    def reset(self, *, seed=None, options=None):
        if self._in_leg:
            return self._obs, {}
        leg = self._next_leg()
        cor = leg["corridor"]
        if self._shield:
            cor = copy.deepcopy(cor)
            self.env = make_leg_env(self._emu, cor, self._cosine, self._ucb)
        self._obs = self.env.reset(corridor_override=cor, wind_profile=leg["wind_profile"]).astype(np.float32)
        self._in_leg = True
        self.violation = False
        return self._obs, {}

    def step(self, action):
        obs, rew, done, info = self.env.step(np.asarray(action, dtype=np.float32), record_micro=True)
        self.violation = any(m["rho_1"] > 1.02 or m["rho_2"] > 1.02 or m["rho_3"] > 1.05 for m in info["micro_logs"])
        self._obs = obs.astype(np.float32)
        if done:
            self._in_leg = False
        r = float(rew) / 20.0
        if self._shield:
            hit = any(m.get("shield_intervened", False) for m in info["micro_logs"])
            self.interventions += int(hit)
            self.macro_steps += 1
            r -= self._penalty * float(hit)
        return self._obs, r, bool(done), False, {}


def make_leg_env(emu, cor, cosine: bool, ucb_state: dict | None):
    """One leg exactly as evaluate_multi_station_journey builds it for UCB-DMPS (methods 5 / 7)."""
    sh = D.GoodallTrajectoryShield(cor, emu, D.TrajectoryShieldConfig(mode="mps_rollout"), ucb_state=ucb_state) \
        if ucb_state is not None else None
    env = D.MPCTransitionRailroadEnv(corridor=cor, traj_shield=sh, emulator=emu, use_cosine_shaper=cosine)
    env.a_start_center = 0.96
    env.coast_enabled = False
    return env


def build(emu, cost_limit: float, cosine: bool = True, shield: bool = False, penalty: float = 0.0):
    inner = RailPilotEnv(emu, cosine=cosine, shield=shield, penalty=penalty)
    env = TimeLimit(inner, 5000)
    env = LabelledEnv(env, inner.label_fn)
    env = CumulativeCostEnv(env, cost_fn=lambda labels: 1.0 if "unsafe" in labels else 0.0, budget=cost_limit)
    env = RewardMonitor(ConstraintMonitor(env))
    return env, inner


def algo_kwargs(algo: str, cost_limit: float) -> dict:
    cfg_dir = Path(masa.__file__).parent / "configs" / "algorithms"
    kw = dict(YAML(typ="safe").load(open(cfg_dir / YAML_FILE[algo]))["defaults"])
    kw["gamma"] = GAMMA
    if "cost_gamma" in kw:
        kw["cost_gamma"] = GAMMA
    if "cost_limit" in kw:
        kw["cost_limit"] = cost_limit
    return kw


def actor_numpy(model) -> dict:
    p = model.policy.actor_state.params["params"]
    names = sorted(k for k in p if k.startswith("Dense_"))
    out = {}
    for i, n in enumerate(names):
        out[f"W{i}"] = np.asarray(p[n]["kernel"], dtype=np.float32)
        out[f"b{i}"] = np.asarray(p[n]["bias"], dtype=np.float32)
    return out


class NumpyPilot:
    """Deterministic MASA actor (tanh MLP, linear mean, clipped to [-1, 1]) with our pilot interface."""

    def __init__(self, weights: dict):
        n = len([k for k in weights if k.startswith("W")])
        self.layers = [(np.asarray(weights[f"W{i}"]), np.asarray(weights[f"b{i}"])) for i in range(n)]

    def get_action_and_value(self, obs, deterministic: bool = True):
        h = np.asarray(obs, dtype=np.float32)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        return np.clip(h, -1.0, 1.0).astype(np.float32), 0.0, 0.0


def val_loss_score(env: D.MPCTransitionRailroadEnv, pilot, val_scens, shield: bool = False, emu=None,
                   cosine: bool = True) -> dict:
    """Identical to the validation LossScore in train_pure_ppo_cosine_pilot (shield=False). With
    shield=True each leg runs under UCB-DMPS as in the benchmark, and the shield rate is returned too."""
    errs, stops, smooth, kwh, tkm = [], [], [], 0.0, 0.0
    hits, macros = 0, 0
    for sc in val_scens:
        ucb: dict = {}
        for leg in sc["legs"]:
            cor = leg["corridor"]
            if shield:
                cor = copy.deepcopy(cor)
                env = make_leg_env(emu, cor, cosine, ucb)
            obs = env.reset(corridor_override=cor, wind_profile=leg["wind_profile"])
            done = False
            while not done:
                act, _, _ = pilot.get_action_and_value(obs, deterministic=True)
                obs, _, done, info = env.step(act, record_micro=True)
                smooth.extend(m["rho_2"] ** 2 + m["rho_3"] ** 2 for m in info["micro_logs"])
                hits += int(any(m.get("shield_intervened", False) for m in info["micro_logs"]))
                macros += 1
            errs.append(abs(info["schedule_error_s"]))
            stops.append(abs(info["stop_error_m"]))
            kwh += float(info["energy_kwh"])
            tkm += (leg["mass_kg"] / 1000.0) * (leg["corridor"].station_x_m / 1000.0)
    T, St = float(np.mean(errs)), float(np.mean(stops))
    Sm = float(np.mean(smooth)) if smooth else 0.0
    E = kwh * 1000.0 / max(1.0, tkm)
    return dict(loss=T + 15.0 * St + 2.0 * Sm + 0.025 * E, T=T, stop=St, smooth=Sm, E=E,
                shield_pct=100.0 * hits / max(1, macros))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=PILOT_ALGOS)
    ap.add_argument("--free", action="store_true",
                    help="no-cosine pilot (methods 6-7, use_cosine_shaper=False); our seeds 142, 107, 2124")
    ap.add_argument("--cost-limit", type=float, default=1.0, help="violation steps per leg (PPOLag/TRPOLag/CPO)")
    ap.add_argument("--seed", type=int, default=42, help="training seed: journeys, validation set and learner (42, 7, 2024)")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="journeys budget = epochs x 3 (400 = our PPO)")
    ap.add_argument("--shield-train", action="store_true", help="train and validate under UCB-DMPS")
    ap.add_argument("--interv-penalty", type=float, default=0.0, help="reward penalty per shield override (scaled reward units)")
    ap.add_argument("--max-val-shield", type=float, default=5.0, help="selection cap on validation shield %% (--shield-train)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    global TRAIN_SEED
    TRAIN_SEED = int(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    emu = load_emulator()
    cosine = not args.free
    tag = ("free_" if args.free else "") + args.algo + (f"_sh{args.interv_penalty:g}" if args.shield_train else "")
    sh, pen = args.shield_train, args.interv_penalty
    env, inner = build(emu, args.cost_limit, cosine, sh, pen)
    algo_cls = registry.get_algorithm(args.algo)
    model = algo_cls(env, tensorboard_logdir=None, seed=args.seed, device="cpu", verbose=0,
                     env_fn=lambda: build(emu, args.cost_limit, cosine, sh, pen)[0], **algo_kwargs(args.algo, args.cost_limit))

    val_env = D.MPCTransitionRailroadEnv(corridor=RailroadCorridor(), traj_shield=None, emulator=emu, use_cosine_shaper=cosine)
    val_env.a_start_center = 0.96
    val_scens = [generate_multi_station_scenario(seed=70000 + TRAIN_SEED * 13 + v * 31, n_legs=0, wind_mode="mixed",
                                                 allow_infeasible_breaks=False, extreme_noise=(v % 2 == 1))
                 for v in range(6)]

    budget = args.epochs * JOURNEYS_PER_EPOCH
    val_step = VAL_EVERY_EPOCHS * JOURNEYS_PER_EPOCH
    next_val, best, best_w, hist = val_step, float("inf"), None, []
    cap = args.max_val_shield if sh else float("inf")
    all_w, fb, fb_w = {}, float("inf"), None      # all checkpoints; fallback = best ignoring the cap
    frames, t0 = 0, time.time()
    while inner.journeys_started <= budget:
        model.train(CHUNK_FRAMES, num_eval_episodes=0, eval_freq=0, log_freq=0)
        frames += CHUNK_FRAMES
        done_j = min(inner.journeys_started - 1, budget)
        if done_j >= next_val or inner.journeys_started > budget:
            next_val += val_step
            w = actor_numpy(model)
            r = val_loss_score(val_env, NumpyPilot(w), val_scens, shield=sh, emu=emu, cosine=cosine)
            for k, v in w.items():
                all_w[f"c{len(hist)}_{k}"] = v
            if r["loss"] <= fb:
                fb, fb_w = r["loss"], w
            if r["loss"] <= best and r["shield_pct"] <= cap:
                best, best_w = r["loss"], w
            hist.append(dict(journeys=done_j, frames=frames, minutes=(time.time() - t0) / 60, best=best, **r))
            print(f"[{args.algo}] journeys {done_j:4d}/{budget} frames {frames:7d} val LossScore {r['loss']:.4f} "
                  f"(best {best:.4f}) T={r['T']:.2f}s stop={r['stop']:.3f}m shield={r['shield_pct']:.1f}% "
                  f"train-shield={100.0 * inner.interventions / max(1, inner.macro_steps):.1f}%  {(time.time() - t0) / 60:.1f} min", flush=True)
    if best_w is None:
        print(f"[{args.algo}] no checkpoint under the {cap}% shield cap; using the best uncapped one", flush=True)
        best, best_w = fb, fb_w
    np.savez(out / f"{tag}_pilot.npz", **best_w)
    np.savez(out / f"{tag}_ckpts.npz", **all_w)
    json.dump(dict(algo=args.algo, seed=args.seed, cost_limit=args.cost_limit, shield_train=sh, interv_penalty=pen, frames=frames,
                   journeys=budget, best_val_loss=best, history=hist), open(out / f"{tag}_history.json", "w"), indent=1)


if __name__ == "__main__":
    main()
