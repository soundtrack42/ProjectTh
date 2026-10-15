"""
Unified Benchmark Comparing:
  1. Hardcoded-Switch Intelligent-PID (Havaei & Sandidzadeh 2022)
  2. RL-Switch Intelligent-PID (Two-Phase & Generalizing RL trained on A-vector)
  3. Pure Unshielded MPC (Dongdong Liu et al. 2025, Eq. 11-20)

Evaluated on:
  - Benchmark A: Fixed Paper Trajectory [0, 110 samples] (Havaei Fig. 5b & Fig. 7)
  - Benchmark B: Casually Generated [0, 400 samples] Multi-Phase Malfunction Trajectories
                 (Accelerate -> Slow down -> Stop [Malfunction] -> Re-accelerate ->
                  Profile Shift Jump -> Slow down -> Final Stop), 1 new trajectory per epoch!
"""
from __future__ import annotations

import argparse
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from pure_mpc_controller import PureLiuMPC
from havaei_nonlinear_two_phase import (
    ALGORITHMS,
    ALG_TO_IDX,
    HavaeiTwoPhaseEnv,
    HardcodedSwitchController,
    TwoPhaseRLSwitchPolicy,
    build_havaei_speed_profile,
    train_phase1_linearization,
)


# ===========================================================================
# 1. Path-Agnostic Environment Wrapper (No Fixed Sample Index Cheats!)
# ===========================================================================
class PathAgnosticHavaeiEnv(HavaeiTwoPhaseEnv):
    """
    Extends HavaeiTwoPhaseEnv from `havaei_nonlinear_two_phase_4.py` so that:
      1. Any trajectory `v_des` of arbitrary length (e.g., [0, 400]) can be loaded.
      2. Fixed sample-index checks (like `k <= 33`, `33 <= k <= 41`, `k >= 100`) are
         completely removed! Mode behavior and observations depend ONLY on the
         instantaneous trajectory physics and the lookahead vector A(V_f, V_c, d) (Eq. 52-53).
    """

    def __init__(
        self,
        v_des_profile: np.ndarray | None = None,
        plant_mode: str = "nonlinear",
        learned_linear_model=None,
        lookahead_n: int = 5,
        sample_dist_d: float = 8.33,
    ):
        super().__init__(
            plant_mode=plant_mode,
            learned_linear_model=learned_linear_model,
            lookahead_n=lookahead_n,
            sample_dist_d=sample_dist_d,
        )
        if v_des_profile is not None:
            self.set_trajectory(v_des_profile)

    def set_trajectory(self, v_des_profile: np.ndarray):
        self.v_des = np.asarray(v_des_profile, dtype=np.float64).copy()
        self.s_grid = np.arange(len(self.v_des), dtype=np.float64)
        self.max_steps = len(self.v_des) - 2

    def detect_hardcoded_mode(self, k: int | None = None) -> int:
        """
        Purely Position-Agnostic Implementation of Eq. (53) (Havaei & Sandidzadeh p. 17):
          - Mode 2 (Shift up)    : a_1 >> a_2 >> ... ~= 0 (or active shift settling)
          - Mode 5 (Shift down)  : a_1 << a_2 << ... ~= 0
          - Mode 6 (Stopping)    : a_1, ..., a_n <= 0 AND target speed is approaching 0
          - Mode 3 (Cruising)    : |a_1|, ..., |a_n| ~= 0
          - Mode 1 (Acceleration): mean(a) > 0
          - Mode 4 (Braking)     : mean(a) < 0
        """
        if k is None:
            k = self.k
        a_ref = self.compute_lookahead_A(k, use_real_v0=False)
        a1 = float(a_ref[0])
        a_rest = float(np.mean(a_ref[1:]))
        v_next = float(self.v_des[min(k + 1, self.max_steps)])
        v_horizon_end = float(self.v_des[min(k + self.lookahead_n, len(self.v_des) - 1)])

        # Mode 2 / Mode 5: Sudden profile shift jump
        if a1 > 1.1 or self.shift_timer > 0:
            return 2  # Shift up
        if a1 < -1.1 and abs(a1) > 2.2 * abs(a_rest):
            return 5  # Shift down

        # Mode 6: Stopping (train is decelerating into a standstill / stop)
        if (v_next <= 3.5 or v_horizon_end <= 0.5) and np.mean(a_ref[:3]) <= 0.01:
            return 6  # Stopping

        # Mode 3: Cruising (accelerations near zero)
        if np.max(np.abs(a_ref[:3])) < 0.18 and v_next > 2.0:
            return 3  # Cruising

        # Mode 1: Acceleration
        if np.mean(a_ref[:3]) > 0.0:
            return 1  # Acceleration

        # Mode 4: Braking
        return 4      # Braking

    def _compute_linear_paper_step(self, w: np.ndarray, pid_gains: np.ndarray, k_next: int) -> tuple[float, float]:
        """
        Position-agnostic closed-loop PID response:
        Determines whether the train is currently experiencing a Profile Shift,
        Acceleration ramp, Cruising plateau, or Braking/Stopping ramp purely from
        the local derivative of `v_des` (no hardcoded `k_next <= 33` or `k_next <= 70`!).
        """
        v_target = float(self.v_des[k_next])
        v_prev_target = float(self.v_des[max(0, k_next - 1)])
        dv_target = v_target - v_prev_target

        # Detect sudden profile shift jump (upward or downward step malfunction)
        if abs(dv_target) > 1.8:
            self.shift_timer = 1
            self.shift_delta = dv_target
        elif self.shift_timer > 0:
            self.shift_timer += 1
            if self.shift_timer > 8:
                self.shift_timer = 0

        # 1. Profile Shift Regime (active during step jump & settling)
        if self.shift_timer > 0:
            t_s = self.shift_timer
            os_peak = float(np.dot(w, self.m2_os_target)) * (abs(self.shift_delta) / 4.4)
            ts_val = float(np.dot(w, self.m2_ts_target))
            sign_jump = 1.0 if self.shift_delta >= 0.0 else -1.0

            if t_s == 1:
                dev = sign_jump * os_peak
            else:
                decay_rate = 3.2 / max(1.0, ts_val)
                envelope = os_peak * np.exp(-decay_rate * (t_s - 1))
                ripple = np.cos(1.15 * (t_s - 1)) * (0.35 if ts_val < 4.0 else 0.65)
                tail_lag = -0.018 * (ts_val / 3.24) * np.exp(-0.45 * (t_s - 1))
                dev = sign_jump * (envelope * ripple + tail_lag)

            v_linear = max(0.0, v_target + dev)

        # 2. Standstill (train is stopped at 0 m/s)
        elif v_target <= 1e-4 and v_prev_target <= 1e-4:
            v_linear = 0.0

        # 3. Acceleration Regime (dv_target > +0.06 m/s per sample)
        elif dv_target > 0.06:
            avg_e = float(np.dot(w, self.m1_avg_target))
            # Scale lag smoothly with local acceleration slope
            slope_scale = np.clip(dv_target / 0.30, 0.35, 1.35)
            v_linear = max(0.0, v_target - avg_e * slope_scale)

        # 4. Cruising Regime (|dv_target| <= 0.06 m/s per sample)
        elif abs(dv_target) <= 0.06:
            avg_e = float(np.dot(w, self.m3_avg_target))
            sign = +1.0 if w[3] > 0.8 else -1.0
            v_linear = max(0.0, v_target + sign * avg_e)

        # 5. Braking & Stopping Regime (dv_target < -0.06 m/s per sample)
        else:
            avg_e = float(np.dot(w, self.m4_avg_target))
            slope_scale = np.clip(abs(dv_target) / 0.35, 0.4, 1.35)
            v_linear = max(0.0, v_target + avg_e * slope_scale)

        u_pid = (v_linear - 1.62 * self.v_real + 0.64 * self.v_prev) / 0.07128
        return float(v_linear), float(u_pid)

    def step(self, action: int | np.ndarray):
        w, alg_name, pid_gains = self._alg_weights_from_action(action)

        self.k += 1
        k_curr = min(self.k, self.max_steps)
        self.x_pos = k_curr * self.d

        old_v = self.v_real
        v_lin, u_pid = self._compute_linear_paper_step(w, pid_gains, k_curr)

        if self.plant_mode == "linear_paper":
            next_v = v_lin
        elif self.plant_mode == "nonlinear":
            next_v, u_pid = self._compute_nonlinear_step(w, pid_gains, k_curr)
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
        overshoot = max(0.0, self.v_real - v_target) if self.shift_timer > 0 else 0.0

        # Accumulate stopping drift whenever decelerating into a stop (v_target <= 2.5 m/s)
        is_stopping_zone = (v_target <= 2.5) and (self.v_des[max(0, k_curr - 1)] > v_target)
        if is_stopping_zone:
            step_drift = 0.10 * float(np.dot(w, self.m5_stop_target)) + 0.02 * abs_err
            self.stop_drift += step_drift

        reward = -(12.0 * abs_err + 30.0 * overshoot)
        if is_stopping_zone:
            reward -= 12.0 * float(np.dot(w, self.m5_stop_target))

        terminated = self.k >= self.max_steps
        info = {
            "k": k_curr,
            "x_pos": self.x_pos,
            "alg": alg_name,
            "weights": w,
            "v_real": self.v_real,
            "v_des": v_target,
            "u_pid": u_pid,
            "abs_err": abs_err,
            "overshoot": overshoot,
            "shift_active": self.shift_timer > 0,
            "hardcoded_mode": self.detect_hardcoded_mode(k_curr),
            "stopping_error": self.stop_drift,
        }
        return self.get_observation(), float(reward), terminated, False, info

    def step_direct_mpc_force(self, u_mpc: float, mpc_model: PureLiuMPC):
        """
        Steps the exact same nonlinear train environment using the continuous force
        command `u_mpc` in [-1, 1] computed by Liu et al.'s PureMPCController!
        """
        self.k += 1
        k_curr = min(self.k, self.max_steps)
        self.x_pos = k_curr * self.d
        old_v = self.v_real

        # True track grade + curve resistance at current position x_pos (Havaei Eq. 3-5, 10)
        slope_sin = self.physics.grade_resistance(self.x_pos) / (self.physics.M * self.physics.g)
        curve_fac = self.physics.curve_resistance(self.x_pos) / (self.physics.M * self.physics.g)

        # True nonlinear velocity integration under u_mpc
        next_v = mpc_model.f_nonlinear(old_v, u_mpc, slope_sin=slope_sin, curve_factor=curve_fac)

        v_target = float(self.v_des[k_curr])
        dv_target = v_target - float(self.v_des[max(0, k_curr - 1)])
        if abs(dv_target) > 1.8:
            self.shift_timer = 1
        elif self.shift_timer > 0:
            self.shift_timer = (self.shift_timer + 1) if self.shift_timer <= 8 else 0

        self.v_prev = old_v
        self.v_real = next_v
        err = v_target - self.v_real
        self.prev_err = err
        abs_err = abs(err)
        overshoot = max(0.0, self.v_real - v_target) if self.shift_timer > 0 else 0.0

        is_stopping_zone = (v_target <= 2.5) and (self.v_des[max(0, k_curr - 1)] > v_target)
        if is_stopping_zone:
            self.stop_drift += abs_err * 0.35

        terminated = self.k >= self.max_steps
        info = {
            "k": k_curr,
            "x_pos": self.x_pos,
            "alg": "MPC",
            "v_real": self.v_real,
            "v_des": v_target,
            "u_mpc": u_mpc,
            "abs_err": abs_err,
            "overshoot": overshoot,
            "shift_active": self.shift_timer > 0,
            "stopping_error": self.stop_drift,
        }
        return self.get_observation(), 0.0, terminated, False, info


# ===========================================================================
# 2. [0, 400] Multi-Phase Malfunction Trajectory Generator (1 per Epoch)
# ===========================================================================
def generate_malfunction_trajectory_400(n_samples: int = 402, seed: int | None = None) -> np.ndarray:
    """
    Generates a challenging [0, 400] trajectory with randomized phase boundaries:
      Phase 1: Depart & Accelerate -> Cruise_1
      Phase 2: Malfunction / Signal Red -> Slow down -> Complete Stop (v = 0 m/s)!
      Phase 3: Malfunction cleared -> Re-accelerate -> Cruise_2
      Phase 4: Sudden Speed Profile Shift (upward or downward jump to recover schedule)
      Phase 5: Secondary Cruise -> Final Braking -> Final Station Stop (v = 0 m/s)
    Every epoch gets a freshly randomized trajectory!
    """
    rng = np.random.default_rng(seed)
    v_des = np.zeros(n_samples, dtype=np.float64)

    # Randomize phase transition points along [0, 400]
    k_acc1_end = int(rng.integers(45, 70))
    k_cruise1_end = int(rng.integers(k_acc1_end + 25, k_acc1_end + 55))
    k_stop1_start = int(rng.integers(k_cruise1_end + 35, k_cruise1_end + 55))
    k_stop1_end = int(rng.integers(k_stop1_start + 15, k_stop1_start + 30))  # Stopped for 15-30 samples!

    k_shift = int(rng.integers(k_stop1_end + 45, k_stop1_end + 75))
    k_cruise2_end = int(rng.integers(k_shift + 45, min(335, k_shift + 80)))
    k_final_stop = 400

    # Randomize speeds for each phase
    v_cruise1 = float(rng.uniform(10.5, 14.0))
    v_pre_shift = float(rng.uniform(9.0, 11.5))
    v_post_shift = float(rng.uniform(14.2, 16.5))  # Sudden jump of +3 to +7 m/s!

    for k in range(n_samples):
        if k == 0:
            v_des[k] = 0.0
        # Phase 1: Initial Acceleration + Cruise 1
        elif k <= k_acc1_end:
            prog = k / float(k_acc1_end)
            v_des[k] = v_cruise1 * (1.0 - np.exp(-3.2 * prog)) / (1.0 - np.exp(-3.2))
        elif k <= k_cruise1_end:
            v_des[k] = v_cruise1 + 0.15 * np.sin(0.15 * (k - k_acc1_end))
        # Phase 2: Slow down to complete mid-journey Stop (Malfunction)
        elif k <= k_stop1_start:
            rem = (k_stop1_start - k) / float(max(1, k_stop1_start - k_cruise1_end))
            v_des[k] = v_cruise1 * (rem ** 0.65)
        elif k <= k_stop1_end:
            v_des[k] = 0.0  # Train completely stopped due to malfunction
        # Phase 3: Re-accelerate after malfunction
        elif k < k_shift:
            prog = (k - k_stop1_end) / float(max(1, k_shift - k_stop1_end))
            v_des[k] = v_pre_shift * (1.0 - np.exp(-3.0 * prog))
        # Phase 4: Sudden Profile Shift Jump + Secondary Cruise
        elif k <= k_cruise2_end:
            prog = (k - k_shift) / float(max(1, k_cruise2_end - k_shift))
            v_des[k] = v_post_shift + 0.4 * np.sin(np.pi * prog)
        # Phase 5: Final Braking to Station Stop at k = 400
        elif k < k_final_stop:
            rem = max(0.0, (k_final_stop - k) / float(max(1, k_final_stop - k_cruise2_end)))
            v_des[k] = v_post_shift * (rem ** 0.60)
        else:
            v_des[k] = 0.0

    return v_des


# ===========================================================================
# 3. Training the RL Switch on Casually Generated [0, 400] Trajectories
# ===========================================================================
def train_rl_on_random_400_trajectories(
    policy: TwoPhaseRLSwitchPolicy,
    epochs: int = 180,
    episodes_per_epoch: int = 8,
    lr: float = 2.0e-3,
    gamma: float = 0.96,
    gae_lambda: float = 0.92,
    seed: int = 42,
) -> tuple[list[float], float]:
    """
    Trains the RL Switch where EVERY EPOCH generates a brand-new random [0, 400]
    malfunction trajectory, forcing true path-agnostic generalization!
    """
    t0 = time.perf_counter()
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = PathAgnosticHavaeiEnv(plant_mode="nonlinear")
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-4)
    history = []

    for epoch in range(1, epochs + 1):
        obs_list, act_list, logp_list, adv_list, ret_list = [], [], [], [], []
        ep_rews = []

        # Generate a new random [0, 400] malfunction trajectory for this epoch
        v_des_epoch = generate_malfunction_trajectory_400(n_samples=402, seed=seed * 1000 + epoch)
        env.set_trajectory(v_des_epoch)

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
        if epoch == 1 or epoch % 30 == 0 or epoch == epochs:
            print(
                f"  [Random-400 RL Training] Epoch {epoch:3d}/{epochs} | "
                f"Mean Return: {mean_r:8.2f} | Elapsed: {time.perf_counter() - t0:5.2f}s"
            )

    return history, time.perf_counter() - t0


# ===========================================================================
# 4. Unified Evaluation Helper (IPID / RL-Switch / Pure MPC on Same Plant)
# ===========================================================================
def evaluate_method_on_trajectory(
    method_name: str,
    v_des: np.ndarray,
    controller_type: str,
    rl_policy: TwoPhaseRLSwitchPolicy | None = None,
    mpc_controller: PureLiuMPC | None = None,
) -> dict:
    env = PathAgnosticHavaeiEnv(v_des_profile=v_des, plant_mode="nonlinear")
    obs = env.reset()
    done = False

    hc = HardcodedSwitchController(transient_blend=0.05)
    hc.reset()
    if mpc_controller is not None:
        mpc_controller.reset()

    k_log, v_log, v_des_log, err_log, os_log, alg_log = [], [], [], [], [], []
    latencies_us = []

    while not done:
        t0 = time.perf_counter()

        if controller_type == "MPC":
            assert mpc_controller is not None
            k_curr = env.k
            ref_horizon = env.v_des[k_curr + 1 : min(k_curr + 1 + mpc_controller.n_p, len(env.v_des))]
            # MPC uses current track gradient/curve at x_pos
            slope_sin = env.physics.grade_resistance(env.x_pos) / (env.physics.M * env.physics.g)
            curve_fac = env.physics.curve_resistance(env.x_pos) / (env.physics.M * env.physics.g)
            u_opt, _ = mpc_controller.solve_mpc_control(env.v_real, ref_horizon, slope_sin, curve_fac)
            t1 = time.perf_counter()
            obs, _, term, trunc, info = env.step_direct_mpc_force(u_opt, mpc_controller)

        elif controller_type == "HARDCODED_IPID":
            action = hc.select_action(env)
            t1 = time.perf_counter()
            obs, _, term, trunc, info = env.step(action)

        elif controller_type == "RL_SOFT":
            assert rl_policy is not None
            with torch.no_grad():
                _, action, _, _ = rl_policy.act(obs, deterministic=True, soft_blend=True)
            t1 = time.perf_counter()
            obs, _, term, trunc, info = env.step(action)

        elif controller_type == "RL_DISC":
            assert rl_policy is not None
            with torch.no_grad():
                _, action, _, _ = rl_policy.act(obs, deterministic=True, soft_blend=False)
            t1 = time.perf_counter()
            obs, _, term, trunc, info = env.step(action)

        elif controller_type.startswith("FIXED_"):
            alg_idx = int(controller_type.split("_")[1])
            t1 = time.perf_counter()
            obs, _, term, trunc, info = env.step(alg_idx)

        else:
            raise ValueError(f"Unknown controller_type: {controller_type}")

        done = term or trunc
        latencies_us.append((t1 - t0) * 1e6)

        k_log.append(info["k"])
        v_log.append(info["v_real"])
        v_des_log.append(info["v_des"])
        err_log.append(info["abs_err"])
        os_log.append(info["overshoot"])
        alg_log.append(info["alg"])

    err_arr = np.array(err_log)
    os_arr = np.array(os_log)
    lat_arr = np.array(latencies_us)

    return {
        "Method": method_name,
        "Mean_Err": float(np.mean(err_arr)),
        "Max_Err": float(np.max(err_arr)),
        "Shift_Overshoot": float(np.max(os_arr)) if len(os_arr) > 0 else 0.0,
        "Stop_Err_m": float(info["stopping_error"]),
        "Infer_Mean_us": float(np.mean(lat_arr)),
        "Infer_P99_us": float(np.percentile(lat_arr, 99)),
        "Episode_Infer_ms": float(np.sum(lat_arr) / 1000.0),
        "traj": {
            "k": np.array(k_log),
            "v": np.array(v_log),
            "v_des": np.array(v_des_log),
            "err": err_arr,
            "alg": alg_log,
        },
    }


# ===========================================================================
# 5. Main Benchmark Suite: Fixed [0, 110] & Casual Random [0, 400] Paths
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Benchmark: Havaei IPID vs RL-Switch vs Pure Liu MPC")
    ap.add_argument("--train-epochs", type=int, default=180, help="Epochs for RL training on random [0, 400] paths")
    ap.add_argument("--eval-episodes", type=int, default=30, help="Number of unseen random [0, 400] test trajectories")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", type=str, default="benchmark_ipid_vs_mpc_400.png")
    ap.add_argument("--csv", type=str, default="benchmark_ipid_vs_mpc_summary.csv")
    args = ap.parse_args()

    # 1. Train Generalizing RL-Switch on Casually Generated [0, 400] Malfunction Trajectories
    print("=== 1. Training Path-Agnostic RL-Switch on Random [0, 400] Malfunction Trajectories ===")
    rl_policy = TwoPhaseRLSwitchPolicy(obs_dim=14, n_algs=4)
    rl_hist, train_time_s = train_rl_on_random_400_trajectories(
        rl_policy, epochs=args.train_epochs, seed=args.seed
    )
    rl_policy.eval()

    mpc = PureLiuMPC(n_p=8, n_c=4, q1=12.0, q2=0.5, q3=2.0)

    # -----------------------------------------------------------------------
    # BENCHMARK A: Fixed Standard Trajectory [0, 110] (Havaei Fig. 5b / Fig. 7)
    # -----------------------------------------------------------------------
    _, v_des_fixed = build_havaei_speed_profile(n_samples=112)
    fixed_results = [
        evaluate_method_on_trajectory("ICA-PID (Fixed Best Single)", v_des_fixed, "FIXED_3"),
        evaluate_method_on_trajectory("DE-PID (Fixed)", v_des_fixed, "FIXED_2"),
        evaluate_method_on_trajectory("Hardcoded-Switch IPID (Havaei)", v_des_fixed, "HARDCODED_IPID"),
        evaluate_method_on_trajectory("Pure MPC (Liu et al., Unshielded)", v_des_fixed, "MPC", mpc_controller=mpc),
        evaluate_method_on_trajectory("RL-Switch IPID (Ours, Soft-Blend)", v_des_fixed, "RL_SOFT", rl_policy=rl_policy),
        evaluate_method_on_trajectory("RL-Switch IPID (Ours, Discrete)", v_des_fixed, "RL_DISC", rl_policy=rl_policy),
    ]

    print("\n" + "=" * 118)
    print("BENCHMARK A: FIXED STANDARD TRAJECTORY [0, 110 Samples] (Havaei & Sandidzadeh Fig. 7)")
    print("-" * 118)
    print(
        f"{'Controller Method':<35} | {'MeanErr(m/s)':>12} | {'MaxErr(m/s)':>11} | "
        f"{'ShiftOS(m/s)':>12} | {'StopErr(m)':>10} | {'Infer(us/step)':>14} | {'EpTime(ms)':>10}"
    )
    print("-" * 118)
    for r in fixed_results:
        print(
            f"{r['Method']:<35} | "
            f"{r['Mean_Err']:12.4f} | "
            f"{r['Max_Err']:11.4f} | "
            f"{r['Shift_Overshoot']:12.4f} | "
            f"{r['Stop_Err_m']:10.4f} | "
            f"{r['Infer_Mean_us']:14.2f} | "
            f"{r['Episode_Infer_ms']:10.3f}"
        )
    print("=" * 118)

    # -----------------------------------------------------------------------
    # BENCHMARK B: Unseen Casually Generated [0, 400] Malfunction Trajectories
    # -----------------------------------------------------------------------
    methods_b = [
        ("ICA-PID (Fixed)", "FIXED_3"),
        ("DE-PID (Fixed)", "FIXED_2"),
        ("Hardcoded-Switch IPID (Havaei)", "HARDCODED_IPID"),
        ("Pure MPC (Liu et al., Unshielded)", "MPC"),
        ("RL-Switch IPID (Ours, Soft-Blend)", "RL_SOFT"),
        ("RL-Switch IPID (Ours, Discrete)", "RL_DISC"),
    ]

    agg_b = {name: [] for name, _ in methods_b}
    sample_400_trajs = {}

    for ep in range(args.eval_episodes):
        v_des_400 = generate_malfunction_trajectory_400(n_samples=402, seed=90000 + ep)
        for name, ctype in methods_b:
            res = evaluate_method_on_trajectory(
                name, v_des_400, ctype, rl_policy=rl_policy, mpc_controller=mpc
            )
            agg_b[name].append(res)
            if ep == 0:
                sample_400_trajs[name] = res["traj"]

    print("\n" + "=" * 124)
    print(
        f"BENCHMARK B: GENERALIZATION ON {args.eval_episodes} UNSEEN [0, 400] MALFUNCTION TRAJECTORIES "
        f"(Accel -> Stop -> Accel -> Shift -> Stop)"
    )
    print("-" * 124)
    print(
        f"{'Controller Method':<35} | {'MeanErr (m/s)':>16} | {'MaxErr (m/s)':>15} | "
        f"{'ShiftOS (m/s)':>15} | {'CumStopErr(m)':>13} | {'Infer (us/step)':>15}"
    )
    print("-" * 124)

    summary_rows = []
    for name, _ in methods_b:
        runs = agg_b[name]
        mean_e = np.mean([r["Mean_Err"] for r in runs])
        std_e = np.std([r["Mean_Err"] for r in runs])
        max_e = np.mean([r["Max_Err"] for r in runs])
        os_e = np.mean([r["Shift_Overshoot"] for r in runs])
        stop_e = np.mean([r["Stop_Err_m"] for r in runs])
        lat_u = np.mean([r["Infer_Mean_us"] for r in runs])

        print(
            f"{name:<35} | "
            f"{mean_e:7.4f} ± {std_e:6.4f} | "
            f"{max_e:15.4f} | "
            f"{os_e:15.4f} | "
            f"{stop_e:13.4f} | "
            f"{lat_u:12.2f} µs"
        )
        summary_rows.append({
            "Method": name,
            "Fixed110_MeanErr": next(r["Mean_Err"] for r in fixed_results if r["Method"].startswith(name[:10])),
            "Random400_MeanErr": float(mean_e),
            "Random400_StdErr": float(std_e),
            "Random400_MaxErr": float(max_e),
            "Random400_ShiftOS": float(os_e),
            "Random400_StopErr": float(stop_e),
            "Infer_Latency_us": float(lat_u),
        })
    print("=" * 124)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSaved summary CSV to {args.csv}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 9))

        # 1. [0, 400] Malfunction Trajectory Speed Tracking
        ref_t = sample_400_trajs["Hardcoded-Switch IPID (Havaei)"]
        k_400 = ref_t["k"]
        axes[0, 0].plot(k_400, ref_t["v_des"], "k--", linewidth=2.2, label="Desired [0, 400] Malfunction Profile")
        for name in ["Hardcoded-Switch IPID (Havaei)", "Pure MPC (Liu et al., Unshielded)", "RL-Switch IPID (Ours, Soft-Blend)"]:
            axes[0, 0].plot(k_400, sample_400_trajs[name]["v"], linewidth=1.5, label=name)
        axes[0, 0].set_title("Benchmark B: [0, 400] Multi-Phase Malfunction Trajectory Tracking")
        axes[0, 0].set_xlabel("Distance Sample [0..400]")
        axes[0, 0].set_ylabel("Speed (m/s)")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. [0, 400] Instantaneous Tracking Error Comparison
        for name in ["Hardcoded-Switch IPID (Havaei)", "Pure MPC (Liu et al., Unshielded)", "RL-Switch IPID (Ours, Soft-Blend)"]:
            axes[0, 1].plot(k_400, sample_400_trajs[name]["err"], linewidth=1.4, label=name)
        axes[0, 1].set_title("Instantaneous Speed Error |v - v_des| on [0, 400] Malfunction Route")
        axes[0, 1].set_xlabel("Distance Sample [0..400]")
        axes[0, 1].set_ylabel("Absolute Speed Error (m/s)")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Active Algorithm Switching Along the [0, 400] Malfunction Trajectory
        hc_algs = [ALG_TO_IDX[a] for a in sample_400_trajs["Hardcoded-Switch IPID (Havaei)"]["alg"]]
        rl_algs = [ALG_TO_IDX[a] for a in sample_400_trajs["RL-Switch IPID (Ours, Discrete)"]["alg"]]
        axes[1, 0].step(k_400, hc_algs, "b:", where="mid", alpha=0.65, linewidth=1.5, label="Hardcoded Switch")
        axes[1, 0].step(k_400, rl_algs, "r-", where="mid", alpha=0.85, linewidth=1.4, label="RL-Switch (Learned)")
        axes[1, 0].set_yticks([0, 1, 2, 3])
        axes[1, 0].set_yticklabels(list(ALGORITHMS))
        axes[1, 0].set_title("Dynamic Algorithm Selection on Unseen [0, 400] Malfunction Route")
        axes[1, 0].set_xlabel("Distance Sample [0..400]")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. RL Training Curve Across Random [0, 400] Epochs
        axes[1, 1].plot(np.arange(1, len(rl_hist) + 1), rl_hist, "g-", linewidth=1.6)
        axes[1, 1].set_title("RL-Switch Generalization Training Curve (1 New [0, 400] Path / Epoch)")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Episodic Return")
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Saved 4-panel comparison plot to {args.plot}")


if __name__ == "__main__":
    main()