"""
Standard PPO with CCN+ Projection Layer (ReqL) for LiuTrainEnv.
Training is 100% standard PPO; the shield simply projects actions into the
admissible interval [u_min_safe, u_max_safe] by construction (Brand Sec. 4).
"""
from __future__ import annotations

import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

from Env.LiuTrainEnv import LiuTrainEnv


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """Scales [pos, vel, v_ref, v_lim, speed_err, pos_err, dist_rem, prev_u] to ~[-1, 1]."""
    scale = np.array([1250.0, 30.0, 30.0, 30.0, 10.0, 20.0, 1250.0, 1.0], dtype=np.float32)
    return obs / scale


def apply_preemptive_shield_discrete(
    raw_probs: torch.Tensor,
    safe_mask: np.ndarray,
    mode: str = "none",
    softness: float = 0.05,
) -> torch.Tensor:
    if mode == "none":
        return raw_probs

    mask_t = torch.as_tensor(safe_mask, dtype=torch.bool, device=raw_probs.device)
    if torch.all(mask_t) or not torch.any(mask_t):
        return raw_probs

    if mode == "hard":
        corrected = torch.where(mask_t, raw_probs, torch.zeros_like(raw_probs))
    elif mode == "soft":
        corrected = torch.where(mask_t, raw_probs, raw_probs * softness)
    else:
        raise ValueError(f"Unknown shield mode: {mode}")

    total = corrected.sum(dim=-1, keepdim=True)
    if total.item() <= 1e-12:
        corrected = mask_t.float()
        total = corrected.sum(dim=-1, keepdim=True)
    return corrected / total


def apply_reqL_continuous(
    u_val: torch.Tensor,
    u_min: torch.Tensor,
    u_max: torch.Tensor,
    mode: str = "none",
    softness: float = 0.2,
) -> torch.Tensor:
    """
    CCN+ Requirements Layer (ReqL) from Brand Eq. (1) & Theorem 2:
      ReqL(h_A) = min(max(h_A, b_A^+), b_A^-)
    """
    if mode == "none":
        return torch.clamp(u_val, -1.0, 1.0)
    clamped = torch.minimum(torch.maximum(u_val, u_min), u_max)
    if mode == "hard":
        return clamped
    elif mode == "soft":
        return (1.0 - softness) * clamped + softness * torch.clamp(u_val, -1.0, 1.0)
    else:
        raise ValueError(f"Unknown shield mode: {mode}")


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, continuous: bool = False, hidden_dim: int = 64):
        super().__init__()
        self.continuous = continuous

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        if self.continuous:
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def get_value(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self.critic(obs_t).squeeze(-1)

    def act(
        self,
        obs_t: torch.Tensor,
        info: dict,
        shield_mode: str = "none",
        deterministic: bool = False,
    ):
        value = self.get_value(obs_t)

        if not self.continuous:
            logits = self.actor(obs_t)
            raw_probs = torch.softmax(logits, dim=-1)
            safe_mask = info["action_mask"]

            raw_dist = Categorical(probs=raw_probs)
            a_raw = int(torch.argmax(raw_probs).item() if deterministic else raw_dist.sample().item())
            violated = not bool(safe_mask[a_raw])

            shielded_probs = apply_preemptive_shield_discrete(raw_probs, safe_mask, mode=shield_mode)
            shield_dist = Categorical(probs=shielded_probs)

            if deterministic:
                a_exec = int(torch.argmax(shielded_probs).item())
            else:
                a_exec = a_raw if (shield_mode == "hard" and not violated) else int(shield_dist.sample().item())

            modified = violated and (a_exec != a_raw)
            log_prob = raw_dist.log_prob(torch.tensor(a_exec, device=obs_t.device))
            entropy = raw_dist.entropy()
            return a_exec, log_prob, entropy, value, violated, modified

        else:
            # 1. Standard PPO Actor mean & std (identical to Cont-PPO None!)
            raw_mean = torch.tanh(self.actor(obs_t))
            std = torch.exp(self.log_std.clamp(-3.0, 0.5))

            u_min, u_max = info["safe_u_bounds"]
            b_plus = torch.tensor([u_min], dtype=torch.float32, device=obs_t.device)
            b_minus = torch.tensor([u_max], dtype=torch.float32, device=obs_t.device)

            # 2. Project mean through CCN+ ReqL layer before sampling
            shielded_mean = apply_reqL_continuous(raw_mean, b_plus, b_minus, mode=shield_mode)
            dist = Normal(shielded_mean, std)

            # 3. Sample action from policy distribution
            sampled_u = shielded_mean if deterministic else dist.sample()
            log_prob = dist.log_prob(sampled_u).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            # 4. Check violation on raw mean and clamp final executed action to safe bounds
            raw_u_val = float(raw_mean.item())
            violated = (raw_u_val < u_min - 1e-3) or (raw_u_val > u_max + 1e-3)

            exec_u_t = apply_reqL_continuous(sampled_u, b_plus, b_minus, mode=shield_mode)
            u_exec = float(exec_u_t.item())
            modified = violated and (abs(u_exec - raw_u_val) > 1e-3)

            # Store `sampled_u` as the PPO action for exact log_prob consistency,
            # while sending `u_exec` to the environment!
            return (
                np.array([u_exec], dtype=np.float32),
                log_prob,
                entropy,
                value,
                violated,
                modified,
                float(sampled_u.item()),
            )

    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,
        act_batch: torch.Tensor,
        bounds_batch: torch.Tensor | None = None,
        shield_mode: str = "none",
    ):
        values = self.get_value(obs_batch)
        if not self.continuous:
            logits = self.actor(obs_batch)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(act_batch.long())
            entropy = dist.entropy()
        else:
            raw_mean = torch.tanh(self.actor(obs_batch))
            if shield_mode != "none" and bounds_batch is not None:
                b_plus = bounds_batch[:, 0:1]
                b_minus = bounds_batch[:, 1:2]
                mean = apply_reqL_continuous(raw_mean, b_plus, b_minus, mode=shield_mode)
            else:
                mean = raw_mean
            std = torch.exp(self.log_std.clamp(-3.0, 0.5))
            dist = Normal(mean, std)
            log_probs = dist.log_prob(act_batch).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy, values


def collect_rollouts(env: LiuTrainEnv, model: ActorCritic, n_episodes: int, shield_mode: str, gamma: float, gae_lambda: float):
    obs_list, act_list, logp_list, adv_list, ret_list, bounds_list = [], [], [], [], [], []
    ep_stats = []

    model.eval()
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False

        ep_obs, ep_acts, ep_logps, ep_rews, ep_vals, ep_bounds = [], [], [], [], [], []
        ep_violations, ep_modifications, ep_overspeeds = 0, 0, 0
        speed_errors = []

        while not done:
            obs_norm = normalize_obs(obs)
            obs_t = torch.as_tensor(obs_norm, dtype=torch.float32)
            ep_bounds.append(info["safe_u_bounds"])

            with torch.no_grad():
                out = model.act(obs_t, info, shield_mode=shield_mode, deterministic=False)
                if env.continuous:
                    action, log_prob, _, value, violated, modified, sampled_u = out
                    stored_act = np.array([sampled_u], dtype=np.float32)
                else:
                    action, log_prob, _, value, violated, modified = out
                    stored_act = action

            next_obs, reward, term, trunc, next_info = env.step(action)
            done = term or trunc

            ep_obs.append(obs_norm)
            ep_acts.append(stored_act)
            ep_logps.append(float(log_prob.item()))
            ep_rews.append(float(reward))
            ep_vals.append(float(value.item()))

            ep_violations += int(violated)
            ep_modifications += int(modified)
            ep_overspeeds += int(next_info["overspeed_violation"])
            speed_errors.append(abs(env.vel - next_info["v_ref"]))

            obs, info = next_obs, next_info

        T = len(ep_rews)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = 0.0 if t == T - 1 else ep_vals[t + 1]
            delta = ep_rews[t] + gamma * next_val - ep_vals[t]
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[t] = last_gae
        returns = advantages + np.asarray(ep_vals, dtype=np.float32)

        obs_list.extend(ep_obs)
        act_list.extend(ep_acts)
        logp_list.extend(ep_logps)
        adv_list.extend(advantages)
        ret_list.extend(returns)
        bounds_list.extend(ep_bounds)

        ep_stats.append({
            "reward": float(np.sum(ep_rews)),
            "viol_rate": ep_violations / T,
            "mod_rate": ep_modifications / T,
            "overspeed_rate": ep_overspeeds / T,
            "max_speed_err": float(np.max(speed_errors)),
            "mean_speed_err": float(np.mean(speed_errors)),
            "stop_err_m": float(info["stopping_error_m"]),
        })

    batch = {
        "obs": torch.as_tensor(np.asarray(obs_list, dtype=np.float32)),
        "act": torch.as_tensor(np.asarray(act_list)),
        "old_logp": torch.as_tensor(np.asarray(logp_list, dtype=np.float32)),
        "adv": torch.as_tensor(np.asarray(adv_list, dtype=np.float32)),
        "ret": torch.as_tensor(np.asarray(ret_list, dtype=np.float32)),
        "bounds": torch.as_tensor(np.asarray(bounds_list, dtype=np.float32)),
        "shield_mode": shield_mode,
    }
    return batch, ep_stats


def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    batch: dict,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    ppo_epochs: int = 4,
    minibatch_size: int = 128,
):
    model.train()
    obs_b = batch["obs"]
    act_b = batch["act"]
    old_logp_b = batch["old_logp"]
    adv_b = batch["adv"]
    ret_b = batch["ret"]
    bounds_b = batch.get("bounds")
    shield_mode = batch.get("shield_mode", "none")

    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

    N = len(obs_b)
    indices = np.arange(N)

    for _ in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, N, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]
            mb_bounds = bounds_b[mb_idx] if bounds_b is not None else None

            new_logp, entropy, values = model.evaluate_actions(
                obs_b[mb_idx], act_b[mb_idx], mb_bounds, shield_mode=shield_mode
            )
            ratio = torch.exp(new_logp - old_logp_b[mb_idx])

            surr1 = ratio * adv_b[mb_idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, ret_b[mb_idx])
            entropy_loss = -entropy.mean()

            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()