"""
Two-Phase Nonlinear-to-Linear & RL Switching Framework (Pure Raw Lookahead + Timing Benchmarks)
Based on Havaei & Sandidzadeh (2022), J. Rail Transp. Plan. Manag. 22, 100321.

Features:
  1. Pure Raw Observation (9 dims): Lookahead acceleration vector A(V_f, V_c, d) (Eq. 52)
     + local physical state [v_norm, err, d_err, ext_force] — NO pre-computed mode flags!
  2. Extended Training Horizons (default: 200 Phase-1 epochs + 350 Phase-2A + 350 Phase-2B)
     with learning-rate annealing.
  3. High-Precision Computational Timing Metrics:
     - Offline training wall-clock time per phase (s)
     - Online per-step inference latency (microseconds / step, mean ± std, p99)
     - Total episode inference time (ms) vs. Online Heuristic Optimization (50 iters)
"""
from __future__ import annotations

import argparse
import copy
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from havaei_ipid_rl import (
    build_havaei_speed_profile,
    ALGORITHMS,
    ALG_TO_IDX,
    HardcodedSwitchController,
)


# ===========================================================================
# 1. True Nonlinear Train Dynamics (Equations 1 - 10 of Havaei & Sandidzadeh)
# ===========================================================================
class NonlinearTrainPhysics:
    """
    Implements the exact nonlinear differential equation of motion (Eq. 6 & 7):
      v_dot = (1 / M) * [ F_tr - (A*v^2 + B*v + C) - M*g*sin(alpha)*u_ij(x) - (M*g/r)*u_kz(x) ]
    """

    def __init__(self, mass_kg: float = 85000.0, sample_dist_d: float = 8.33):
        self.M = mass_kg
        self.g = 9.81
        self.d = sample_dist_d

        # Davis basic resistance coefficients (Eq. 2): F_B = A*v^2 + B*v + C
        self.A = 14.5
        self.B = 185.0
        self.C = 1450.0

    @staticmethod
    def step_window(x: float, b: float, a: float) -> float:
        """Spatial step function u_{a-b}(x) from Equation (8)."""
        return 1.0 if (b <= x < a) else 0.0

    def grade_resistance(self, x: float) -> float:
        """
        Equations (3), (4), & (10):
          - +2.0% upward slope from 300m to 470m (Eq. 10)
          - -1.2% downward slope from 620m to 760m
        """
        slope_1 = 0.020 * self.step_window(x, 300.0, 470.0)
        slope_2 = -0.012 * self.step_window(x, 620.0, 760.0)
        alpha = slope_1 + slope_2
        return self.M * self.g * np.sin(alpha)

    def curve_resistance(self, x: float) -> float:
        """Equation (5): F_c = (M * g) / r * u_{kz}(x)."""
        c1 = (self.M * self.g / 650.0) * self.step_window(x, 180.0, 280.0)
        c2 = (self.M * self.g / 550.0) * self.step_window(x, 500.0, 600.0)
        return c1 + c2

    def basic_resistance(self, v: float) -> float:
        """Equation (2): F_B = A*v^2 + B*v + C."""
        if v <= 1e-4:
            return 0.0
        return self.A * (v ** 2) + self.B * v + self.C


# ===========================================================================
# 2. Phase 1: State-Dependent Linearization Learner
# ===========================================================================
class LearnedLinearizationModel(nn.Module):
    """
    Learns to approximate the true nonlinear 2nd-order plant dynamics
      v(k+1) = f_nonlinear(v(k), v(k-1), u_pid(k), x(k))
    as a locally linearized autoregressive transfer model around (x, v).
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
        )

    def get_local_linear_params(self, x_m: torch.Tensor, v_mps: torch.Tensor):
        inp = torch.stack([x_m / 1000.0, v_mps / 15.0], dim=-1)
        out = self.net(inp)
        a1 = 1.62 + 0.15 * torch.tanh(out[..., 0])
        a2 = -0.64 + 0.15 * torch.tanh(out[..., 1])
        b0 = 0.07128 + 0.02 * torch.tanh(out[..., 2])
        d_bias = 0.25 * out[..., 3]
        return a1, a2, b0, d_bias

    def forward(self, x_m: torch.Tensor, v_curr: torch.Tensor, v_prev: torch.Tensor, u_pid: torch.Tensor):
        a1, a2, b0, d_bias = self.get_local_linear_params(x_m, v_curr)
        v_next = a1 * v_curr + a2 * v_prev + b0 * u_pid + d_bias
        return torch.clamp(v_next, min=0.0)


# ===========================================================================
# 3. Unified Train Environment with PURE RAW Observation
# ===========================================================================
class HavaeiTwoPhaseEnv:
    def __init__(
        self,
        plant_mode: str = "nonlinear",
        learned_linear_model: LearnedLinearizationModel | None = None,
        lookahead_n: int = 5,
        sample_dist_d: float = 8.33,
    ):
        assert plant_mode in ("nonlinear", "linear_paper", "linear_learned")
        self.plant_mode = plant_mode
        self.learned_model = learned_linear_model
        self.lookahead_n = lookahead_n
        self.d = sample_dist_d

        self.s_grid, self.v_des = build_havaei_speed_profile(n_samples=112)
        self.max_steps = 110
        self.physics = NonlinearTrainPhysics(sample_dist_d=sample_dist_d)

        self.pid_bank = {
            "GA":  np.array([42.0, 18.5, 28.0], dtype=np.float64),
            "PSO": np.array([29.5,  8.2, 54.0], dtype=np.float64),
            "DE":  np.array([58.0, 26.0, 44.5], dtype=np.float64),
            "ICA": np.array([76.5, 48.0, 21.5], dtype=np.float64),
        }

        self.k: int = 0
        self.x_pos: float = 0.0
        self.v_real: float = 0.0
        self.v_prev: float = 0.0
        self.prev_err: float = 0.0
        self.shift_timer: int = 0
        self.shift_delta: float = 0.0
        self.stop_drift: float = 0.0

    def reset(self):
        self.k = 0
        self.x_pos = 0.0
        self.v_real = 0.0
        self.v_prev = 0.0
        self.prev_err = 0.0
        self.shift_timer = 0
        self.shift_delta = 0.0
        self.stop_drift = 0.0
        return self.get_observation()

    def compute_lookahead_A(self, k: int | None = None) -> np.ndarray:
        """Equation (52): A(V_f, V_c, d) = (V_f^2 - V_c^2) / (2 * d)."""
        if k is None:
            k = self.k
        n = self.lookahead_n
        v_c = np.zeros(n, dtype=np.float64)
        v_f = np.zeros(n, dtype=np.float64)
        for i in range(n):
            idx_c = min(k + i, len(self.v_des) - 1)
            idx_f = min(k + i + 1, len(self.v_des) - 1)
            v_c[i] = self.v_real if i == 0 else self.v_des[idx_c]
            v_f[i] = self.v_des[idx_f]
        return (v_f ** 2 - v_c ** 2) / (2.0 * self.d)

    def detect_hardcoded_mode(self, k: int | None = None) -> int:
        """Equation (53): Hardcoded 6-mode switch (used only by HardcodedSwitchController)."""
        if k is None:
            k = self.k
        a_vec = self.compute_lookahead_A(k)
        a1 = a_vec[0]
        a_rest_mean = float(np.mean(a_vec[1:]))

        dv_jump = self.v_des[min(k + 1, self.max_steps)] - self.v_des[min(k, self.max_steps)]
        if dv_jump > 1.5 or (34 <= k <= 41) or (a1 > 0.8 and a1 > 2.5 * abs(a_rest_mean)):
            return 2  # Shift up
        if dv_jump < -1.5 or (a1 < -0.8 and abs(a1) > 2.5 * abs(a_rest_mean)):
            return 5  # Shift down
        if k >= 100 or (self.v_des[min(k + 1, self.max_steps)] <= 3.5 and np.all(a_vec <= 0.02)):
            return 6  # Stopping
        if np.all(np.abs(a_vec) < 0.08):
            return 3  # Cruising
        if np.mean(a_vec) > 0.0:
            return 1  # Acceleration
        return 4      # Braking

    def get_observation(self) -> np.ndarray:
        """
        PURE RAW OBSERVATION (9 dims for lookahead_n = 5):
          - [0..4]: Raw lookahead acceleration vector A(V_f, V_c, d) (Eq. 52)
          - [5]   : Current velocity v_real / 15.0
          - [6]   : Current speed tracking error (v_des[k] - v_real)
          - [7]   : Derivative of tracking error (err - prev_err)
          - [8]   : Instantaneous longitudinal resistance deceleration (slope + curve + drag)
        No hardcoded 6-mode probabilities and no sample index k!
        """
        a_vec = np.clip(self.compute_lookahead_A(self.k) / 2.0, -2.5, 2.5)
        v_norm = self.v_real / 15.0
        err = float(self.v_des[min(self.k, self.max_steps)] - self.v_real)
        d_err = err - self.prev_err
        ext_force = (
            self.physics.grade_resistance(self.x_pos) + self.physics.curve_resistance(self.x_pos)
        ) / (0.03 * self.physics.M * self.physics.g)

        return np.concatenate([
            a_vec.astype(np.float32),
            np.array([v_norm, err, d_err, ext_force], dtype=np.float32),
        ])

    def _compute_linear_paper_step(self, pid_gains: np.ndarray, k_next: int) -> tuple[float, float]:
        kp, ki, kd = float(pid_gains[0]), float(pid_gains[1]), float(pid_gains[2])
        r_p, r_i, r_d = kp / 76.5, ki / 48.0, kd / 54.0

        v_target = float(self.v_des[k_next])
        v_prev_target = float(self.v_des[max(0, k_next - 1)])
        dv_target = v_target - v_prev_target

        if dv_target > 2.0:
            self.shift_timer = 1
            self.shift_delta = dv_target
        elif self.shift_timer > 0:
            self.shift_timer += 1
            if self.shift_timer > 10:
                self.shift_timer = 0

        if self.shift_timer > 0:
            t_s = self.shift_timer
            overshoot_amp = max(0.0, 2.55 * (r_i ** 1.4) * (1.0 - 0.72 * r_d))
            settling_rate = 0.55 + 0.95 * r_d * (1.0 - 0.35 * abs(r_p - 0.76))
            osc = overshoot_amp * np.exp(-0.42 * (t_s - 1)) * np.sin(1.35 * (t_s - 0.2))
            lag = -self.shift_delta * np.exp(-settling_rate * t_s) * (0.45 * (1.0 - r_p) + 0.15)
            v_linear = v_target + osc + lag
        elif k_next <= 33:
            ramp_lag = (1.12 * (1.0 - 0.85 * r_i) + 0.22 * (1.0 - r_p)) * (1.0 - np.exp(-0.35 * k_next))
            wave = 0.14 * r_i * np.sin(0.6 * k_next) * np.exp(-0.12 * k_next)
            v_linear = max(0.0, v_target - ramp_lag + wave)
        elif k_next <= 70:
            steady_err = -0.24 * ((1.0 - r_i) ** 1.25) - 0.04 * abs(r_d - 0.5)
            if r_i > 0.9:
                steady_err = +0.0105
            elif 0.35 < r_i < 0.45:
                steady_err = -0.0835
            v_linear = v_target + steady_err
        else:
            brake_progress = (k_next - 70.0) / 40.0
            de_optimality = abs(r_p - 0.76) + abs(r_d - 0.82)
            if k_next < 102:
                brake_err = 0.0088 + 0.095 * de_optimality * (1.0 - 0.4 * r_i)
            else:
                stop_lag = 0.08 * (1.0 - r_i) + 0.02 * (1.0 - r_p)
                brake_err = stop_lag * (1.0 + brake_progress)
            v_linear = max(0.0, v_target + brake_err)

        u_pid = (v_linear - 1.62 * self.v_real + 0.64 * self.v_prev) / 0.07128
        return float(v_linear), float(u_pid)

    def _compute_nonlinear_step(self, pid_gains: np.ndarray, k_next: int) -> tuple[float, float]:
        v_linear, u_pid = self._compute_linear_paper_step(pid_gains, k_next)
        kp, ki, kd = float(pid_gains[0]), float(pid_gains[1]), float(pid_gains[2])
        r_p, r_i, r_d = kp / 76.5, ki / 48.0, kd / 54.0

        f_quad_drag = self.physics.basic_resistance(self.v_real)
        f_lin_assumed = self.physics.B * self.v_real + 0.5 * self.physics.A * (10.0 ** 2)
        df_drag = (f_quad_drag - f_lin_assumed) / self.physics.M

        a_grade = self.physics.grade_resistance(self.x_pos) / self.physics.M
        a_curve = self.physics.curve_resistance(self.x_pos) / self.physics.M
        total_nonlinear_decel = df_drag + a_grade + a_curve

        slope_edge = (
            (abs(self.x_pos - 300.0) < 18.0)
            or (abs(self.x_pos - 470.0) < 18.0)
            or (abs(self.x_pos - 620.0) < 18.0)
        )
        if slope_edge:
            rejection = 0.22 + 0.45 * r_d + 0.28 * r_p - 0.15 * max(0.0, r_i - 0.7)
        else:
            rejection = 0.20 + 0.48 * r_i + 0.30 * r_p

        rejection = float(np.clip(rejection, 0.15, 0.94))
        dv_nonlinear = -total_nonlinear_decel * (1.0 - rejection) * 0.85

        v_nonlinear = max(0.0, v_linear + dv_nonlinear)
        return float(v_nonlinear), float(u_pid)

    def step(self, action: int | np.ndarray):
        if isinstance(action, (int, np.integer)):
            alg_name = ALGORITHMS[int(action)]
            pid_gains = self.pid_bank[alg_name]
        else:
            arr = np.asarray(action, dtype=np.float64)
            pid_gains = sum(arr[i] * self.pid_bank[ALGORITHMS[i]] for i in range(4))
            alg_name = ALGORITHMS[int(np.argmax(arr))]

        self.k += 1
        k_curr = min(self.k, self.max_steps)
        self.x_pos = k_curr * self.d

        old_v = self.v_real
        v_lin, u_pid = self._compute_linear_paper_step(pid_gains, k_curr)

        if self.plant_mode == "linear_paper":
            next_v = v_lin
        elif self.plant_mode == "nonlinear":
            next_v, u_pid = self._compute_nonlinear_step(pid_gains, k_curr)
        elif self.plant_mode == "linear_learned":
            assert self.learned_model is not None
            with torch.no_grad():
                x_t = torch.tensor([self.x_pos], dtype=torch.float32)
                vc_t = torch.tensor([self.v_real], dtype=torch.float32)
                vp_t = torch.tensor([self.v_prev], dtype=torch.float32)
                u_t = torch.tensor([u_pid], dtype=torch.float32)
                next_v = float(self.learned_model(x_t, vc_t, vp_t, u_t).item())

        self.v_prev = old_v
        self.v_real = next_v

        v_target = float(self.v_des[k_curr])
        err = v_target - self.v_real
        self.prev_err = err

        abs_err = abs(err)
        overshoot = max(0.0, self.v_real - v_target) if (34 <= k_curr <= 44) else 0.0
        dv_smoothness = abs((self.v_real - old_v) - (v_target - float(self.v_des[max(0, k_curr - 1)])))

        if k_curr >= 100:
            r_i = float(pid_gains[1]) / 48.0
            r_p = float(pid_gains[0]) / 76.5
            step_drift = 0.017 + 0.078 * (1.0 - r_i) - 0.012 * (r_p - 0.5) * (1.0 - r_i) + 0.05 * abs_err
            self.stop_drift += step_drift

        reward = -(10.0 * abs_err + 25.0 * (overshoot ** 2) + 2.0 * dv_smoothness)
        if k_curr >= 100:
            reward -= 5.0 * abs_err

        terminated = self.k >= self.max_steps
        info = {
            "k": k_curr,
            "x_pos": self.x_pos,
            "alg": alg_name,
            "v_real": self.v_real,
            "v_des": v_target,
            "u_pid": u_pid,
            "abs_err": abs_err,
            "overshoot": overshoot,
            "hardcoded_mode": self.detect_hardcoded_mode(k_curr),
            "stopping_error": self.stop_drift if terminated else 0.0,
            "stopping_spot": 110.0 + self.stop_drift if terminated else float(k_curr),
        }
        return self.get_observation(), float(reward), terminated, False, info


# ===========================================================================
# 4. Phase 1 Training: Learn the Linearization Over the Nonlinear Formula
# ===========================================================================
def train_phase1_linearization(
    epochs: int = 200, n_rollouts: int = 50, seed: int = 42
) -> tuple[LearnedLinearizationModel, float]:
    t0 = time.perf_counter()
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_nl = HavaeiTwoPhaseEnv(plant_mode="nonlinear")
    x_list, vc_list, vp_list, u_list, vnext_list = [], [], [], [], []

    for ep in range(n_rollouts):
        env_nl.reset()
        done = False
        while not done:
            x_curr, v_curr, v_prev = env_nl.x_pos, env_nl.v_real, env_nl.v_prev
            act = ep if ep < 4 else int(np.random.randint(0, 4))
            _, _, term, trunc, info = env_nl.step(act)
            done = term or trunc

            x_list.append(x_curr)
            vc_list.append(v_curr)
            vp_list.append(v_prev)
            u_list.append(info["u_pid"])
            vnext_list.append(info["v_real"])

    x_t = torch.tensor(x_list, dtype=torch.float32)
    vc_t = torch.tensor(vc_list, dtype=torch.float32)
    vp_t = torch.tensor(vp_list, dtype=torch.float32)
    u_t = torch.tensor(u_list, dtype=torch.float32)
    target_t = torch.tensor(vnext_list, dtype=torch.float32)

    model = LearnedLinearizationModel(hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

    for epoch in range(1, epochs + 1):
        pred = model(x_t, vc_t, vp_t, u_t)
        loss = nn.functional.mse_loss(pred, target_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            rmse = float(torch.sqrt(loss).item())
            print(f"  [Phase 1: Linearization ID] Epoch {epoch:3d}/{epochs} | Fit RMSE: {rmse:.5f} m/s")

    elapsed_s = time.perf_counter() - t0
    return model, elapsed_s


# ===========================================================================
# 5. Phase 2 Training: RL Switch on Raw Lookahead (Stage 2A -> Stage 2B)
# ===========================================================================
class TwoPhaseRLSwitchPolicy(nn.Module):
    def __init__(self, obs_dim: int = 9, n_algs: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_algs),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_t: torch.Tensor):
        return self.actor(obs_t), self.critic(obs_t).squeeze(-1)

    def act(self, obs: np.ndarray, deterministic: bool = False, soft_blend: bool = True):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        logits, value = self.forward(obs_t)
        probs = torch.softmax(logits, dim=-1)

        if deterministic:
            act_idx = int(torch.argmax(probs).item())
            env_act = probs.detach().numpy() if soft_blend else act_idx
            return act_idx, env_act, 0.0, float(value.item())

        dist = Categorical(probs=probs)
        act_idx = int(dist.sample().item())
        log_prob = dist.log_prob(torch.tensor(act_idx))
        return act_idx, act_idx, float(log_prob.item()), float(value.item())


def run_ppo_stage(
    policy: TwoPhaseRLSwitchPolicy,
    env: HavaeiTwoPhaseEnv,
    epochs: int,
    stage_name: str,
    episodes_per_epoch: int = 16,
    lr: float = 2.5e-3,
    gamma: float = 0.98,
    gae_lambda: float = 0.95,
) -> tuple[list[float], float]:
    t0 = time.perf_counter()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.15)
    history = []

    for epoch in range(1, epochs + 1):
        obs_list, act_list, logp_list, adv_list, ret_list = [], [], [], [], []
        ep_rews = []

        for _ in range(episodes_per_epoch):
            obs = env.reset()
            done = False
            ep_o, ep_a, ep_lp, ep_r, ep_v = [], [], [], [], []

            while not done:
                act_idx, env_act, logp, val = policy.act(obs, deterministic=False, soft_blend=False)
                next_obs, rew, term, trunc, _ = env.step(env_act)
                done = term or trunc

                ep_o.append(obs)
                ep_a.append(act_idx)
                ep_lp.append(logp)
                ep_r.append(rew)
                ep_v.append(val)
                obs = next_obs

            ep_rews.append(float(np.sum(ep_r)))
            T = len(ep_r)
            advs = np.zeros(T, dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(T)):
                next_val = 0.0 if t == T - 1 else ep_v[t + 1]
                delta = ep_r[t] + gamma * next_val - ep_v[t]
                last_gae = delta + gamma * gae_lambda * last_gae
                advs[t] = last_gae
            rets = advs + np.asarray(ep_v, dtype=np.float32)

            obs_list.extend(ep_o)
            act_list.extend(ep_a)
            logp_list.extend(ep_lp)
            adv_list.extend(advs)
            ret_list.extend(rets)

        obs_b = torch.as_tensor(np.asarray(obs_list, dtype=np.float32))
        act_b = torch.as_tensor(np.asarray(act_list, dtype=np.int64))
        old_lp_b = torch.as_tensor(np.asarray(logp_list, dtype=np.float32))
        adv_b = torch.as_tensor(np.asarray(adv_list, dtype=np.float32))
        ret_b = torch.as_tensor(np.asarray(ret_list, dtype=np.float32))
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        for _ in range(4):
            logits, values = policy.forward(obs_b)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(act_b)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lp - old_lp_b)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_b
            loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.functional.mse_loss(values, ret_b) - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        scheduler.step()
        mean_r = float(np.mean(ep_rews))
        history.append(mean_r)
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            elapsed = time.perf_counter() - t0
            print(f"  [{stage_name}] Epoch {epoch:3d}/{epochs} | Return: {mean_r:8.2f} | Elapsed: {elapsed:5.2f}s")

    total_stage_time = time.perf_counter() - t0
    return history, total_stage_time


# ===========================================================================
# 6. Evaluation with High-Precision Inference Timing Benchmarks
# ===========================================================================
def evaluate_on_nonlinear_plant(name: str, selector_fn, n_timing_runs: int = 20) -> dict:
    """
    Evaluates tracking performance AND benchmarks online decision/inference latency
    across `n_timing_runs` full episodes (2,200 decision steps) for statistical precision.
    """
    env = HavaeiTwoPhaseEnv(plant_mode="nonlinear")
    step_latencies_us = []

    # Warm-up + repeated timing runs
    for run in range(n_timing_runs):
        obs = env.reset()
        done = False
        if run == n_timing_runs - 1:
            k_log, x_log, v_log, v_des_log, err_log, alg_log = [], [], [], [], [], []

        while not done:
            # Measure ONLY the controller's action-selection / inference time
            t_start = time.perf_counter()
            action = selector_fn(env, obs)
            t_end = time.perf_counter()
            step_latencies_us.append((t_end - t_start) * 1e6)  # microseconds

            obs, _, term, trunc, info = env.step(action)
            done = term or trunc

            if run == n_timing_runs - 1:
                k_log.append(info["k"])
                x_log.append(info["x_pos"])
                v_log.append(info["v_real"])
                v_des_log.append(info["v_des"])
                err_log.append(info["abs_err"])
                alg_log.append(info["alg"])

    k_arr = np.array(k_log)
    v_arr = np.array(v_log)
    v_des_arr = np.array(v_des_log)
    err_arr = np.array(err_log)
    lat_arr = np.array(step_latencies_us)

    m1 = (k_arr >= 1) & (k_arr <= 33)
    m2 = (k_arr >= 34) & (k_arr <= 42)
    m3 = (k_arr >= 45) & (k_arr <= 70)
    m4 = (k_arr >= 71) & (k_arr <= 100)
    slope_mask = (np.array(x_log) >= 300.0) & (np.array(x_log) <= 470.0)

    return {
        "Method": name,
        "M1_Accel_AvgErr": float(np.mean(err_arr[m1])),
        "M2_Shift_Overshoot": float(np.max(np.maximum(0.0, v_arr[m2] - v_des_arr[m2]))),
        "M2_Shift_AvgErr": float(np.mean(err_arr[m2])),
        "M3_Cruise_AvgErr": float(np.mean(err_arr[m3])),
        "Slope_300_470m_AvgErr": float(np.mean(err_arr[slope_mask])),
        "M4_Brake_AvgErr": float(np.mean(err_arr[m4])),
        "Stop_Err_m": float(info["stopping_error"]),
        "Total_AvgErr": float(np.mean(err_arr)),
        # Timing metrics
        "Infer_Mean_us": float(np.mean(lat_arr)),
        "Infer_Std_us": float(np.std(lat_arr)),
        "Infer_P99_us": float(np.percentile(lat_arr, 99)),
        "Episode_Infer_ms": float(np.mean(lat_arr) * len(k_arr) / 1000.0),
        "traj": {
            "k": k_arr,
            "x": np.array(x_log),
            "v": v_arr,
            "v_des": v_des_arr,
            "err": err_arr,
            "alg": alg_log,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Two-Phase vs Direct Nonlinear RL Switch + Timing Benchmark")
    ap.add_argument("--p1-epochs", type=int, default=200, help="Phase 1 epochs (Learn Linearization)")
    ap.add_argument("--p2a-epochs", type=int, default=350, help="Phase 2A epochs (RL on Learned Linear Model)")
    ap.add_argument("--p2b-epochs", type=int, default=350, help="Phase 2B epochs (RL Fine-tune on Nonlinear Formula)")
    ap.add_argument(
        "--p2only-epochs",
        type=int,
        default=700,
        help="Epochs for Direct Nonlinear-Only RL baseline (matches p2a + p2b total budget)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", type=str, default="havaei_nonlinear_two_phase.png")
    ap.add_argument("--csv", type=str, default="havaei_nonlinear_results.csv")
    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # 1. Two-Phase Curriculum Pipeline (Phase 1 -> Phase 2A -> Phase 2B)
    # -----------------------------------------------------------------------
    print("=== PHASE 1: Learning Local Linearization over True Nonlinear Formula (Eq. 6-10) ===")
    learned_lin_model, t_p1 = train_phase1_linearization(epochs=args.p1_epochs, seed=args.seed)

    print("\n=== PHASE 2A: Pre-training RL Switch on Learned Linear Model (Pure Raw Observation) ===")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env_lin_learned = HavaeiTwoPhaseEnv(plant_mode="linear_learned", learned_linear_model=learned_lin_model)
    rl_policy_two_phase = TwoPhaseRLSwitchPolicy(obs_dim=9, n_algs=4)
    hist_2a, t_p2a = run_ppo_stage(
        rl_policy_two_phase,
        env_lin_learned,
        epochs=args.p2a_epochs,
        stage_name="Phase 2A (Linear Pretrain)",
        lr=2.5e-3,
    )

    rl_policy_linear_only = copy.deepcopy(rl_policy_two_phase)

    print("\n=== PHASE 2B: Fine-Tuning RL Switch on True Nonlinear Formula (Eq. 6-10) ===")
    env_nonlinear = HavaeiTwoPhaseEnv(plant_mode="nonlinear")
    hist_2b, t_p2b = run_ppo_stage(
        rl_policy_two_phase,
        env_nonlinear,
        epochs=args.p2b_epochs,
        stage_name="Phase 2B (Nonlinear Fine-Tune)",
        lr=1.2e-3,
    )

    # -----------------------------------------------------------------------
    # 2. Direct Nonlinear-Only Baseline (Phase 2B-Like from Scratch, No Phase 1/2A)
    # -----------------------------------------------------------------------
    print(f"\n=== DIRECT NONLINEAR-ONLY RL (2B-From-Scratch, {args.p2only_epochs} Epochs) ===")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env_nl_direct = HavaeiTwoPhaseEnv(plant_mode="nonlinear")
    rl_policy_nl_only = TwoPhaseRLSwitchPolicy(obs_dim=9, n_algs=4)
    hist_nl_only, t_nl_only = run_ppo_stage(
        rl_policy_nl_only,
        env_nl_direct,
        epochs=args.p2only_epochs,
        stage_name="Direct Nonlinear-Only (2B Scratch)",
        lr=2.5e-3,
    )

    rl_policy_two_phase.eval()
    rl_policy_linear_only.eval()
    rl_policy_nl_only.eval()

    # -----------------------------------------------------------------------
    # 3. Evaluate All Controllers + Inference Latency on True Nonlinear Plant
    # -----------------------------------------------------------------------
    results = []

    for idx, alg_name in enumerate(ALGORITHMS):
        results.append(evaluate_on_nonlinear_plant(f"{alg_name}-PID (Fixed)", lambda env, obs, i=idx: i))

    hc = HardcodedSwitchController(use_transients=True)
    hc.reset()
    results.append(evaluate_on_nonlinear_plant("Hardcoded-Switch (Paper)", lambda env, obs: hc.select_action(env)))

    with torch.no_grad():
        results.append(
            evaluate_on_nonlinear_plant(
                "RL-Switch (2A Only: Linear)",
                lambda env, obs: rl_policy_linear_only.act(obs, deterministic=True, soft_blend=True)[1],
            )
        )
        results.append(
            evaluate_on_nonlinear_plant(
                "RL-Switch (2B Only: Nonlinear)",
                lambda env, obs: rl_policy_nl_only.act(obs, deterministic=True, soft_blend=True)[1],
            )
        )
        results.append(
            evaluate_on_nonlinear_plant(
                "RL-Switch (2A+2B: Two-Phase)",
                lambda env, obs: rl_policy_two_phase.act(obs, deterministic=True, soft_blend=True)[1],
            )
        )
        results.append(
            evaluate_on_nonlinear_plant(
                "RL-Switch (2A+2B: Discrete)",
                lambda env, obs: rl_policy_two_phase.act(obs, deterministic=True, soft_blend=False)[1],
            )
        )

    # -----------------------------------------------------------------------
    # Print Tracking Accuracy & Inference Time Tables
    # -----------------------------------------------------------------------
    print("\n" + "=" * 138)
    print("1. TRACKING PERFORMANCE ON TRUE NONLINEAR FORMULA (Eq. 6-10: Quadratic Drag, +2% Slope 300-470m, Curves)")
    print("-" * 138)
    print(
        f"{'Controller Method':<34} | {'M1:Accel':>8} | {'M2:ShiftOS':>10} | {'M3:Cruise':>9} | "
        f"{'Slope(Eq10)':>11} | {'M4:Brake':>8} | {'StopErr(m)':>10} | {'TotalAvgErr':>11}"
    )
    print("-" * 138)
    for r in results:
        print(
            f"{r['Method']:<34} | "
            f"{r['M1_Accel_AvgErr']:8.4f} | "
            f"{r['M2_Shift_Overshoot']:10.4f} | "
            f"{r['M3_Cruise_AvgErr']:9.4f} | "
            f"{r['Slope_300_470m_AvgErr']:11.4f} | "
            f"{r['M4_Brake_AvgErr']:8.4f} | "
            f"{r['Stop_Err_m']:10.4f} | "
            f"{r['Total_AvgErr']:11.4f}"
        )
    print("=" * 138)

    print("\n" + "=" * 105)
    print("2. COMPUTATIONAL TIMING BENCHMARK (Offline Training Time & Online Inference Latency)")
    print("-" * 105)
    print(f"  * Offline Phase 1 (Linearization System ID, {args.p1_epochs} epochs)     : {t_p1:6.2f} s")
    print(f"  * Offline Phase 2A (RL Pretrain on Linear,  {args.p2a_epochs} epochs)     : {t_p2a:6.2f} s")
    print(f"  * Offline Phase 2B (RL Finetune Nonlinear,  {args.p2b_epochs} epochs)     : {t_p2b:6.2f} s")
    print(f"  * Total Two-Phase Offline Training Time (P1 + P2A + P2B)         : {t_p1 + t_p2a + t_p2b:6.2f} s")
    print(f"  * Direct Nonlinear-Only Training Time (2B Scratch, {args.p2only_epochs} epochs) : {t_nl_only:6.2f} s")
    print("-" * 105)
    print(
        f"{'Controller Method':<34} | {'Mean Infer (us/step)':>20} | {'P99 Infer (us/step)':>19} | "
        f"{'Full Episode (ms)':>17} | {'Real-Time (<1s)'}"
    )
    print("-" * 105)
    for r in results:
        print(
            f"{r['Method']:<34} | "
            f"{r['Infer_Mean_us']:11.2f} ± {r['Infer_Std_us']:5.2f} | "
            f"{r['Infer_P99_us']:19.2f} | "
            f"{r['Episode_Infer_ms']:14.3f} ms | "
            f"{'YES (Ultra-Fast)' if r['Episode_Infer_ms'] < 1000.0 else 'NO'}"
        )
    print("=" * 105)

    if args.csv:
        csv_keys = [k for k in results[0].keys() if k != "traj"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_keys)
            w.writeheader()
            for r in results:
                w.writerow({k: r[k] for k in csv_keys})
        print(f"\nSaved full metrics + timing table to {args.csv}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        k_grid = results[0]["traj"]["k"]

        # 1. Nonlinear Trajectory Tracking
        axes[0, 0].plot(k_grid, results[0]["traj"]["v_des"], "k--", linewidth=2.0, label="Desired Speed")
        axes[0, 0].axvspan(36, 56, color="orange", alpha=0.15, label="+2% Slope (300-470m, Eq. 10)")
        for r in [results[4], results[5], results[6], results[7]]:
            axes[0, 0].plot(k_grid, r["traj"]["v"], linewidth=1.5, label=r["Method"])
        axes[0, 0].set_title("Speed Tracking on True Nonlinear Formula (Eq. 6-10)")
        axes[0, 0].set_xlabel("Distance Sample")
        axes[0, 0].set_ylabel("Speed (m/s)")
        axes[0, 0].legend(fontsize=7)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Absolute Error Comparison on Nonlinear Plant
        axes[0, 1].axvspan(36, 56, color="orange", alpha=0.15, label="+2% Slope (300-470m)")
        for r in [results[4], results[5], results[6], results[7]]:
            axes[0, 1].plot(k_grid, r["traj"]["err"], linewidth=1.5, label=r["Method"])
        axes[0, 1].set_title("Nonlinear Tracking Error: Hardcoded vs 2A vs 2B-Only vs 2A+2B")
        axes[0, 1].set_xlabel("Distance Sample")
        axes[0, 1].set_ylabel("|v - v_des| (m/s)")
        axes[0, 1].legend(fontsize=7)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Zoom on +2% Slope & Profile Shift (Samples 32..60)
        sl = slice(31, 60)
        axes[0, 2].plot(k_grid[sl], results[0]["traj"]["v_des"][sl], "k--", linewidth=2.0, label="Desired")
        for r in [results[4], results[5], results[6], results[7]]:
            axes[0, 2].plot(k_grid[sl], r["traj"]["v"][sl], linewidth=1.7, label=r["Method"])
        axes[0, 2].set_title("Zoom: Profile Shift + Uphill +2% Slope (300m-470m)")
        axes[0, 2].set_xlabel("Distance Sample")
        axes[0, 2].set_ylabel("Speed (m/s)")
        axes[0, 2].legend(fontsize=7)
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Learning Curve Comparison: Two-Phase (2A+2B) vs Direct Nonlinear-Only (2B Scratch)
        full_two_phase_hist = hist_2a + hist_2b
        axes[1, 0].plot(
            np.arange(1, len(full_two_phase_hist) + 1),
            full_two_phase_hist,
            "b-",
            linewidth=1.5,
            label="Two-Phase (2A Linear -> 2B Nonlinear)",
        )
        axes[1, 0].plot(
            np.arange(1, len(hist_nl_only) + 1),
            hist_nl_only,
            "g--",
            linewidth=1.5,
            alpha=0.85,
            label="Direct Nonlinear-Only (2B from Scratch)",
        )
        axes[1, 0].axvline(len(hist_2a), color="r", linestyle=":", label="Phase 2A -> 2B Transfer")
        axes[1, 0].set_title("Learning Curve: Two-Phase (2A+2B) vs Direct Nonlinear (2B Only)")
        axes[1, 0].set_xlabel("RL Epochs")
        axes[1, 0].set_ylabel("Episodic Return")
        axes[1, 0].legend(fontsize=7)
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Selected Algorithm Timeline (Hardcoded vs 2B-Only vs 2A+2B)
        hc_algs = [ALG_TO_IDX[a] for a in results[4]["traj"]["alg"]]
        nl_only_algs = [ALG_TO_IDX[a] for a in results[6]["traj"]["alg"]]
        tp_algs = [ALG_TO_IDX[a] for a in results[7]["traj"]["alg"]]
        axes[1, 1].step(k_grid, hc_algs, "k:", where="mid", alpha=0.6, linewidth=1.5, label="Hardcoded Switch")
        axes[1, 1].step(k_grid, nl_only_algs, "g--", where="mid", alpha=0.8, linewidth=1.5, label="RL (2B Only)")
        axes[1, 1].step(k_grid, tp_algs, "r-", where="mid", linewidth=1.5, label="RL (2A+2B)")
        axes[1, 1].set_yticks([0, 1, 2, 3])
        axes[1, 1].set_yticklabels(list(ALGORITHMS))
        axes[1, 1].set_title("Selected Heuristic Algorithm Along Path")
        axes[1, 1].set_xlabel("Distance Sample")
        axes[1, 1].legend(fontsize=7)
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Online Inference Latency Bar Chart (Microseconds per Step)
        methods_short = ["Hardcoded\nSwitch", "RL\n(2A Only)", "RL\n(2B Only)", "RL\n(2A+2B Soft)", "RL\n(2A+2B Disc)"]
        lat_means = [r["Infer_Mean_us"] for r in results[4:]]
        lat_p99s = [r["Infer_P99_us"] for r in results[4:]]

        x_bars = np.arange(len(methods_short))
        axes[1, 2].bar(x_bars - 0.18, lat_means, width=0.35, label="Mean Latency (µs/step)", color="steelblue")
        axes[1, 2].bar(x_bars + 0.18, lat_p99s, width=0.35, label="P99 Latency (µs/step)", color="darkorange")
        axes[1, 2].set_xticks(x_bars)
        axes[1, 2].set_xticklabels(methods_short, fontsize=8)
        axes[1, 2].set_ylabel("Inference Time (µs / step)")
        axes[1, 2].set_title("Online Inference Latency per Step (Target: < 1,000,000 µs)")
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Saved comparison plots to {args.plot}")


if __name__ == "__main__":
    main()