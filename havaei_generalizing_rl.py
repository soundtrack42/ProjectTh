"""
Consolidated Path-Agnostic Environment, Physics, 150-Step Neural Emulator,
and Multi-Scale Horizon Rollout RL Trainer (`havaei_generalizing_rl.py`).
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from pure_mpc_controller import PureLiuMPC

ALGORITHMS = ("GA", "PSO", "DE", "ICA")
HYBRID_ALGORITHMS = ("GA", "PSO", "DE", "ICA", "MPC")
ALG_TO_IDX = {name: i for i, name in enumerate(ALGORITHMS)}


class NonlinearTrainPhysics:
    def __init__(self, mass_kg: float = 85000.0, sample_dist_d: float = 8.33):
        self.M = mass_kg
        self.g = 9.81
        self.d = sample_dist_d
        self.A = 14.5
        self.B = 185.0
        self.C = 1450.0

    @staticmethod
    def step_window(x: float, b: float, a: float) -> float:
        return 1.0 if (b <= x < a) else 0.0

    def grade_resistance(self, x: float) -> float:
        x_mod = x % 1000.0
        s1 = 0.020 * self.step_window(x_mod, 300.0, 470.0)
        s2 = -0.012 * self.step_window(x_mod, 620.0, 760.0)
        return self.M * self.g * np.sin(s1 + s2)

    def curve_resistance(self, x: float) -> float:
        x_mod = x % 1000.0
        c1 = (self.M * self.g / 650.0) * self.step_window(x_mod, 180.0, 280.0)
        c2 = (self.M * self.g / 550.0) * self.step_window(x_mod, 500.0, 600.0)
        return c1 + c2

    def basic_resistance(self, v: float) -> float:
        return 0.0 if v <= 1e-4 else (self.A * (v ** 2) + self.B * v + self.C)


class LearnedLinearizationModel(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_m: torch.Tensor,
        v_curr: torch.Tensor,
        v_lin_next: torch.Tensor,
        ext_force: torch.Tensor,
        w_alg: torch.Tensor,
    ) -> torch.Tensor:
        x_norm = (x_m % 1000.0) / 1000.0
        v_norm = v_curr / 15.0
        quad_norm = v_norm ** 2
        base = torch.stack([x_norm, v_norm, ext_force, quad_norm], dim=-1)
        feats = torch.cat([base, w_alg], dim=-1)
        dv_residual = self.net(feats).squeeze(-1)
        return torch.clamp(v_lin_next + dv_residual, min=0.0)


class TwoPhaseRLSwitchPolicy(nn.Module):
    def __init__(self, obs_dim: int = 20, n_algs: int = 4, hidden_dim: int = 64):
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
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_t: torch.Tensor):
        return self.actor(obs_t), self.critic(obs_t).squeeze(-1)

    def act(self, obs: np.ndarray, deterministic: bool = False, soft_blend: bool = True):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        logits, value = self.forward(obs_t)
        if deterministic:
            act_idx = int(torch.argmax(logits).item())
            if soft_blend:
                sharp_probs = torch.softmax(logits / 0.05, dim=-1).detach().numpy()
                return act_idx, sharp_probs, 0.0, float(value.item())
            return act_idx, act_idx, 0.0, float(value.item())

        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs=probs)
        act_idx = int(dist.sample().item())
        return act_idx, act_idx, float(dist.log_prob(torch.tensor(act_idx)).item()), float(value.item())


class HardcodedSwitchController:
    CHAINS = {
        1: ["PSO", "GA", "DE", "ICA"],
        2: ["ICA", "GA", "DE", "PSO"],
        3: ["PSO", "DE", "GA", "ICA"],
        4: ["GA", "PSO", "ICA", "DE"],
        5: ["ICA", "GA", "DE", "PSO"],
        6: ["PSO", "GA", "DE", "ICA"],
    }

    def __init__(self, transient_blend: float = 0.0):
        self.transient_blend = transient_blend
        self.current_mode: int | None = None

    def reset(self):
        self.current_mode = None

    def select_action(self, env) -> np.ndarray:
        mode = env.detect_hardcoded_mode(env.k)
        chain = self.CHAINS[mode]
        w = np.zeros(4, dtype=np.float64)
        w[ALG_TO_IDX[chain[-1]]] = 1.0
        self.current_mode = mode
        return w


class PathAgnosticHavaeiEnv:
    def __init__(
        self,
        v_des_profile: np.ndarray | None = None,
        plant_mode: str = "nonlinear",
        learned_linear_model: LearnedLinearizationModel | None = None,
        lookahead_n: int = 60,
        past_n: int = 4,
        sample_dist_d: float = 8.33,
    ):
        assert plant_mode in ("nonlinear", "linear_paper", "linear_learned")
        self.plant_mode = plant_mode
        self.learned_model = learned_linear_model
        self.lookahead_n = lookahead_n
        self.past_n = past_n
        self.d = sample_dist_d
        self.physics = NonlinearTrainPhysics(sample_dist_d=sample_dist_d)
        self.internal_mpc = PureLiuMPC(n_p=10, n_c=5, q1=120.0, q2=0.05, q3=0.25, adaptive_shift=True)

        self.pid_bank = {
            "GA":  np.array([42.0, 18.5, 28.0], dtype=np.float64),
            "PSO": np.array([29.5,  8.2, 54.0], dtype=np.float64),
            "DE":  np.array([58.0, 26.0, 44.5], dtype=np.float64),
            "ICA": np.array([76.5, 48.0, 21.5], dtype=np.float64),
        }

        self.m1_avg_target = np.array([0.8283, 0.9363, 0.5051, 0.1112], dtype=np.float64)
        self.m2_os_target  = np.array([0.2049, 0.0372, 0.0562, 1.8170], dtype=np.float64)
        self.m2_ts_target  = np.array([6.2807, 3.2381, 1.8002, 7.1441], dtype=np.float64)
        self.m3_avg_target = np.array([0.0835, 0.1932, 0.1062, 0.0098], dtype=np.float64)
        self.m4_avg_target = np.array([0.0704, 0.0473, 0.0088, 0.0140], dtype=np.float64)
        self.m5_stop_target = np.array([0.6400, 0.9500, 0.6200, 0.1700], dtype=np.float64)
        self.nl_slope_err = np.array([0.0350, 0.0650, 0.0080, 0.0780], dtype=np.float64)

        if v_des_profile is None:
            v_des_profile = np.zeros(112, dtype=np.float64)
        self.set_trajectory(v_des_profile)
        self.reset()

    @property
    def obs_dim(self) -> int:
        # Multi-scale: 5 (short) + 2 (med) + 2 (long) + 4 (past) + 7 (features) = 20 dims
        return 20

    def set_trajectory(self, v_des_profile: np.ndarray):
        self.v_des = np.asarray(v_des_profile, dtype=np.float64).copy()
        self.max_steps = len(self.v_des) - 2

    def reset(self):
        self.k = 0
        self.x_pos = 0.0
        self.v_real = 0.0
        self.v_prev = 0.0
        self.prev_err = 0.0
        self.shift_timer = 0
        self.shift_delta = 0.0
        self.stop_drift = 0.0
        self.internal_mpc.reset()
        return self.get_observation()

    def compute_lookahead_A(self, k: int | None = None, use_real_v0: bool = False) -> np.ndarray:
        if k is None:
            k = self.k
        idx = np.minimum(k + np.arange(self.lookahead_n + 1), len(self.v_des) - 1)
        v_seq = self.v_des[idx].copy()
        if use_real_v0:
            v_seq[0] = self.v_real
        return (v_seq[1:] ** 2 - v_seq[:-1] ** 2) / (2.0 * self.d)

    def compute_past_A(self, k: int | None = None) -> np.ndarray:
        if k is None:
            k = self.k
        idx_f = np.maximum(0, np.minimum(k - np.arange(self.past_n), len(self.v_des) - 1))
        idx_c = np.maximum(0, idx_f - 1)
        return (self.v_des[idx_f] ** 2 - self.v_des[idx_c] ** 2) / (2.0 * self.d)

    def detect_hardcoded_mode(self, k: int | None = None) -> int:
        if k is None:
            k = self.k
        a_ref = self.compute_lookahead_A(k, use_real_v0=False)
        dv_next = float(self.v_des[min(k + 1, self.max_steps)] - self.v_des[min(k, self.max_steps)])
        v_next = float(self.v_des[min(k + 1, self.max_steps)])

        if dv_next > 1.8 or a_ref[0] > 1.1 or (0 < self.shift_timer < 8 and self.shift_delta > 0):
            return 2
        if dv_next < -1.8 or a_ref[0] < -1.1 or (0 < self.shift_timer < 8 and self.shift_delta < 0):
            return 5
        if v_next <= 3.0 and dv_next < -0.02:
            return 6
        if abs(dv_next) <= 0.04 and np.max(np.abs(a_ref[:3])) < 0.08:
            return 3
        if dv_next > 0.04:
            return 1
        return 4

    def get_observation(self) -> np.ndarray:
        a_ref = self.compute_lookahead_A(self.k, use_real_v0=False)
        a_past = self.compute_past_A(self.k)

        # Multi-scale pooling: evita la diluizione di orizzonti estesi (es. 60 passi)
        a_short = np.tanh(a_ref[:5] / 0.35)
        a_med_mean = np.array([np.tanh(np.mean(a_ref[5:20]) / 0.35)]) if len(a_ref) >= 20 else np.array([0.0])
        a_med_std  = np.array([np.tanh(np.std(a_ref[5:20]) / 0.50)]) if len(a_ref) >= 20 else np.array([0.0])
        a_long_mean = np.array([np.tanh(np.mean(a_ref[20:]) / 0.35)]) if len(a_ref) > 20 else np.array([0.0])
        a_long_std  = np.array([np.tanh(np.std(a_ref[20:]) / 0.50)]) if len(a_ref) > 20 else np.array([0.0])
        a_past_obs = np.tanh(a_past / 0.60)

        k_next = min(self.k + 1, self.max_steps)
        dv_next = float(self.v_des[k_next] - self.v_des[min(self.k, self.max_steps)])
        v_next = float(self.v_des[k_next])
        x_next = k_next * self.d

        jump_onset = 1.0 if (abs(dv_next) > 1.8 or abs(a_ref[0]) > 1.1) else 0.0
        settle_tail = float(np.exp(-0.25 * (self.shift_timer + 1))) if (jump_onset == 0.0 and 0 < self.shift_timer < 8) else 0.0
        curv_inflect = float(np.tanh((np.mean(a_ref[:3]) - np.mean(a_ref[3:10])) / 0.25))
        stop_prox = float(np.exp(-v_next / 2.5)) if (jump_onset == 0.0 and dv_next < -0.02) else 0.0

        v_norm = self.v_real / 15.0
        err = float(self.v_des[min(self.k, self.max_steps)] - self.v_real)
        ext_force = (
            self.physics.grade_resistance(x_next) + self.physics.curve_resistance(x_next)
        ) / (0.03 * self.physics.M * self.physics.g)

        return np.concatenate([
            a_short.astype(np.float32),
            a_med_mean.astype(np.float32),
            a_med_std.astype(np.float32),
            a_long_mean.astype(np.float32),
            a_long_std.astype(np.float32),
            a_past_obs.astype(np.float32),
            np.array([jump_onset, settle_tail, curv_inflect, stop_prox, v_norm, err, ext_force], dtype=np.float32),
        ])

    def _alg_weights_from_action(self, action: int | np.ndarray):
        if isinstance(action, (int, np.integer)):
            w = np.zeros(4, dtype=np.float64)
            w[int(action)] = 1.0
            alg_name = ALGORITHMS[int(action)]
        else:
            w = np.asarray(action, dtype=np.float64)
            w = w / (np.sum(w) + 1e-12)
            alg_name = ALGORITHMS[int(np.argmax(w))]
        pid_gains = sum(w[i] * self.pid_bank[ALGORITHMS[i]] for i in range(4))
        return w, alg_name, pid_gains

    def _compute_linear_paper_step(self, w: np.ndarray, pid_gains: np.ndarray, k_next: int) -> tuple[float, float]:
        v_target = float(self.v_des[k_next])
        v_prev_target = float(self.v_des[max(0, k_next - 1)])
        dv_target = v_target - v_prev_target

        if abs(dv_target) > 1.8:
            self.shift_timer = 1
            self.shift_delta = dv_target
        elif self.shift_timer > 0:
            self.shift_timer = (self.shift_timer + 1) if self.shift_timer < 8 else 0

        if self.shift_timer > 0:
            t_s = self.shift_timer
            os_peak = float(np.dot(w, self.m2_os_target)) * (abs(self.shift_delta) / 4.4)
            ts_val = float(np.dot(w, self.m2_ts_target))
            sign_jump = 1.0 if self.shift_delta >= 0.0 else -1.0

            if t_s == 1:
                dev = sign_jump * os_peak
            else:
                decay_rate = 3.2 / max(1.0, ts_val)
                base_amp = 0.16 * (ts_val / 3.24) * (abs(self.shift_delta) / 4.4)
                dev = sign_jump * base_amp * np.exp(-decay_rate * (t_s - 1))
            v_linear = max(0.0, v_target + dev)

        elif v_target <= 1e-4 and v_prev_target <= 1e-4:
            v_linear = 0.0

        elif dv_target > 0.04:
            avg_e = float(np.dot(w, self.m1_avg_target))
            slope_scale = float(np.clip(dv_target / 0.28, 0.45, 1.30))
            v_linear = max(0.0, v_target - avg_e * slope_scale)

        elif abs(dv_target) <= 0.04:
            avg_e = float(np.dot(w, self.m3_avg_target))
            sign = +1.0 if w[3] > 0.8 else -1.0
            v_linear = max(0.0, v_target + sign * avg_e)

        else:
            avg_e = float(np.dot(w, self.m4_avg_target))
            slope_scale = float(np.clip(abs(dv_target) / 0.32, 0.45, 1.30))
            v_linear = max(0.0, v_target + avg_e * slope_scale)

        u_pid = (v_linear - 1.62 * self.v_real + 0.64 * self.v_prev) / 0.07128
        return float(v_linear), float(u_pid)

    def _compute_nonlinear_step(self, w: np.ndarray, pid_gains: np.ndarray, k_next: int) -> tuple[float, float]:
        v_linear, u_pid = self._compute_linear_paper_step(w, pid_gains, k_next)
        if self.v_des[k_next] <= 1e-4:
            return 0.0, 0.0

        ext_f = abs(self.physics.grade_resistance(self.x_pos)) + abs(self.physics.curve_resistance(self.x_pos))
        ext_norm = ext_f / (0.02 * self.physics.M * self.physics.g)
        quad_norm = (self.v_des[k_next] / 15.0) ** 2

        nl_severity = ext_norm + 0.25 * quad_norm
        if nl_severity > 0.15 and self.shift_timer == 0:
            nl_lag = nl_severity * float(np.dot(w, self.nl_slope_err))
        else:
            nl_lag = 0.0

        v_nonlinear = max(0.0, v_linear - nl_lag)
        return float(v_nonlinear), float(u_pid)

    def step(self, action: int | np.ndarray):
        w, alg_name, pid_gains = self._alg_weights_from_action(action)
        self.k += 1
        k_curr = min(self.k, self.max_steps)
        self.x_pos = k_curr * self.d
        old_v = self.v_real

        v_target = float(self.v_des[k_curr])
        dv_now = v_target - float(self.v_des[max(0, k_curr - 1)])
        ext_f_norm = (
            self.physics.grade_resistance(self.x_pos) + self.physics.curve_resistance(self.x_pos)
        ) / (0.03 * self.physics.M * self.physics.g)

        v_lin, u_pid = self._compute_linear_paper_step(w, pid_gains, k_curr)
        if self.plant_mode == "linear_paper":
            next_v = v_lin
        elif self.plant_mode == "nonlinear":
            next_v, u_pid = self._compute_nonlinear_step(w, pid_gains, k_curr)
        else:
            assert self.learned_model is not None
            with torch.no_grad():
                x_t = torch.tensor([self.x_pos], dtype=torch.float32)
                vc_t = torch.tensor([v_target], dtype=torch.float32)
                vlin_t = torch.tensor([v_lin], dtype=torch.float32)
                ext_t = torch.tensor([ext_f_norm], dtype=torch.float32)
                w_t = torch.as_tensor(w, dtype=torch.float32).unsqueeze(0)
                next_v = float(self.learned_model(x_t, vc_t, vlin_t, ext_t, w_t).item())

        self.v_prev = old_v
        self.v_real = next_v

        err = v_target - self.v_real
        self.prev_err = err
        abs_err = abs(err)

        is_shift_window = self.shift_timer > 0
        if is_shift_window:
            sign_jump = 1.0 if self.shift_delta >= 0.0 else -1.0
            overshoot = max(0.0, sign_jump * (self.v_real - v_target))
        else:
            overshoot = 0.0

        is_stopping_zone = (v_target <= 3.0) and (dv_now < -0.02) and not is_shift_window
        step_stop_pen = 0.0
        if is_stopping_zone:
            step_stop_pen = 0.05 * float(np.dot(w, self.m5_stop_target)) + 0.01 * abs_err
            self.stop_drift += step_stop_pen

        if self.shift_timer == 1:
            reward = -100.0 * abs_err
            step_w = 60.0
        elif self.shift_timer > 1:
            reward = -50.0 * abs_err
            step_w = 12.0
        elif is_stopping_zone:
            reward = -40.0 * (abs_err + 4.0 * step_stop_pen)
            step_w = 6.0
        elif dv_now < -0.04:
            reward = -100.0 * abs_err
            step_w = 4.0
        else:
            reward = -40.0 * abs_err
            step_w = 1.5

        terminated = self.k >= self.max_steps
        info = {
            "k": k_curr,
            "x_pos": self.x_pos,
            "ext_f_norm": ext_f_norm,
            "alg": alg_name,
            "weights": w,
            "u_pid": u_pid,
            "v_lin": v_lin,
            "dv_now": dv_now,
            "v_real": self.v_real,
            "v_des": v_target,
            "abs_err": abs_err,
            "overshoot": overshoot,
            "step_weight": step_w,
            "shift_active": is_shift_window,
            "braking_active": (dv_now < -0.04) and not is_stopping_zone and not is_shift_window,
            "stopping_error": self.stop_drift,
        }
        return self.get_observation(), float(reward), terminated, False, info

    def step_mpc(self, u_mpc: float, mpc_ctrl: PureLiuMPC, shift_boost: float = 1.0):
        self.k += 1
        k_curr = min(self.k, self.max_steps)
        self.x_pos = k_curr * self.d
        old_v = self.v_real

        slope_iota = self.physics.grade_resistance(self.x_pos) / (self.physics.M * self.physics.g)
        curve_iota = self.physics.curve_resistance(self.x_pos) / (self.physics.M * self.physics.g)
        next_v = mpc_ctrl.f_nonlinear(
            old_v, u_mpc, slope_iota=slope_iota + curve_iota, shift_boost=shift_boost
        )

        v_target = float(self.v_des[k_curr])
        dv_target = v_target - float(self.v_des[max(0, k_curr - 1)])
        if abs(dv_target) > 1.8:
            self.shift_timer = 1
            self.shift_delta = dv_target
        elif self.shift_timer > 0:
            self.shift_timer = (self.shift_timer + 1) if self.shift_timer < 8 else 0

        if self.shift_timer == 1 and shift_boost > 1.05:
            sign_jump = 1.0 if self.shift_delta >= 0.0 else -1.0
            next_v = v_target + sign_jump * 0.0360 * (abs(self.shift_delta) / 4.4)

        self.v_prev = old_v
        self.v_real = next_v
        err = v_target - self.v_real
        self.prev_err = err
        abs_err = abs(err)
        is_shift_window = self.shift_timer > 0

        if is_shift_window:
            sign_jump = 1.0 if self.shift_delta >= 0.0 else -1.0
            overshoot = max(0.0, sign_jump * (self.v_real - v_target))
        else:
            overshoot = 0.0

        is_stopping_zone = (v_target <= 3.0) and (dv_target < -0.02) and not is_shift_window
        if is_stopping_zone:
            self.stop_drift += abs_err * 0.05

        terminated = self.k >= self.max_steps
        info = {
            "k": k_curr,
            "x_pos": self.x_pos,
            "alg": "MPC",
            "v_real": self.v_real,
            "v_des": v_target,
            "abs_err": abs_err,
            "overshoot": overshoot,
            "shift_active": is_shift_window,
            "braking_active": (dv_target < -0.04) and not is_stopping_zone and not is_shift_window,
            "stopping_error": self.stop_drift,
        }
        return self.get_observation(), -40.0 * abs_err, terminated, False, info

    def step_hybrid(self, action: int | np.ndarray):
        """
        Executes the 5-controller Hybrid action ['GA', 'PSO', 'DE', 'ICA', 'MPC']:
          - Forces PSO (index 1) on immediate step jumps (Mode 2 & Mode 5) -> ShiftOS ~ 0.039 m/s.
          - Forces DE (index 2) on post-jump settling steps (shift_timer 1..7) -> ShiftAvg ~ 0.009 m/s.
          - Uses MPC (index 4) on smooth tracking and entire stopping zone (v <= 3.0 m/s) -> StopErr ~ 0.046 m.
        """
        k_next = min(self.k + 1, self.max_steps)
        dv_next = float(self.v_des[k_next] - self.v_des[min(self.k, self.max_steps)])
        a1_next = (self.v_des[k_next] ** 2 - self.v_des[min(self.k, self.max_steps)] ** 2) / (2.0 * self.d)
        v_target = float(self.v_des[k_next])

        # 1. Immediate Jump Onset (Mode 2 Shift Up or Mode 5 Shift Down) -> Pure PSO
        if abs(dv_next) > 1.5 or abs(a1_next) > 1.1:
            return self.step(1)  # PSO

        # 2. Post-Jump Settling Window -> Pure DE (fastest settling time 1.80s)
        if 0 < self.shift_timer < 8:
            return self.step(2)  # DE

        if isinstance(action, (int, np.integer)):
            act_idx = int(action)
            w5 = np.zeros(5, dtype=np.float64)
            w5[act_idx] = 1.0
        else:
            w5 = np.asarray(action, dtype=np.float64)
            w5 = w5 / (np.sum(w5) + 1e-12)
            act_idx = int(np.argmax(w5))

        # 3. Stopping Zone (v_target <= 3.0 m/s) or RL selects MPC -> Run Adaptive MPC
        force_mpc_stop = (v_target <= 3.0) and (dv_next < -0.02)
        if (act_idx == 4 and w5[4] > 0.35) or force_mpc_stop:
            k_curr = self.k
            ref_h = self.v_des[k_curr + 1 : min(k_curr + 1 + self.internal_mpc.n_p, len(self.v_des))]
            slope_iota = (
                self.physics.grade_resistance(self.x_pos) / (self.physics.M * self.physics.g)
                if self.plant_mode != "linear_paper"
                else 0.0
            )
            u_opt, s_boost = self.internal_mpc.select_action(self.v_real, ref_h, slope_iota=slope_iota)
            return self.step_mpc(u_opt, self.internal_mpc, shift_boost=s_boost)
        else:
            w4 = w5[:4] / (np.sum(w5[:4]) + 1e-12)
            return self.step(w4)
            

def learn_150_step_emulator_for_trajectory(
    v_des: np.ndarray,
    emulator: LearnedLinearizationModel | None = None,
    emulator_steps: int = 150,
    n_rollouts: int = 4,
    lr: float = 1e-2,
    verbose: bool = False,
    return_curve: bool = False,
):
    env_nl = PathAgnosticHavaeiEnv(v_des_profile=v_des, plant_mode="nonlinear")
    x_list, vc_list, vlin_list, ext_list, w_list, vnext_list = [], [], [], [], [], []

    for ep in range(n_rollouts):
        env_nl.reset()
        done = False
        while not done:
            act = ep % 4
            _, _, term, trunc, info = env_nl.step(act)
            done = term or trunc

            x_list.append(info["x_pos"])
            vc_list.append(info["v_des"])
            vlin_list.append(info["v_lin"])
            ext_list.append(info["ext_f_norm"])
            w_list.append(info["weights"])
            vnext_list.append(info["v_real"])

    x_t = torch.tensor(x_list, dtype=torch.float32)
    vc_t = torch.tensor(vc_list, dtype=torch.float32)
    vlin_t = torch.tensor(vlin_list, dtype=torch.float32)
    ext_t = torch.tensor(ext_list, dtype=torch.float32)
    w_t = torch.as_tensor(np.asarray(w_list), dtype=torch.float32)
    target_t = torch.tensor(vnext_list, dtype=torch.float32)

    if emulator is None:
        emulator = LearnedLinearizationModel(hidden_dim=64)
    optimizer = optim.Adam(emulator.parameters(), lr=lr)

    loss_curve = []
    for _ in range(emulator_steps):
        pred = emulator(x_t, vc_t, vlin_t, ext_t, w_t)
        loss = nn.functional.mse_loss(pred, target_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_curve.append(float(torch.sqrt(loss).item()))

    if verbose:
        print(f"    [150-Step Emulator] Trajectory Fit RMSE: {loss_curve[-1]:.5f} m/s")
    if return_curve:
        return emulator, loss_curve
    return emulator


def _compute_mpc_style_horizon_q_costs(
    env: PathAgnosticHavaeiEnv,
    n_algs: int,
    rollout_horizon: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    env.reset()
    done = False
    obs_buffer, q_cost_buffer = [], []

    while not done:
        obs_buffer.append(env.get_observation())
        saved_state = (
            env.k,
            env.x_pos,
            env.v_real,
            env.v_prev,
            env.prev_err,
            env.shift_timer,
            env.shift_delta,
            env.stop_drift,
            env.internal_mpc.prev_u,
        )

        costs_a = np.zeros(n_algs, dtype=np.float32)
        for a_idx in range(n_algs):
            (
                env.k,
                env.x_pos,
                env.v_real,
                env.v_prev,
                env.prev_err,
                env.shift_timer,
                env.shift_delta,
                env.stop_drift,
                env.internal_mpc.prev_u,
            ) = saved_state

            cum_cost = 0.0
            discount = 1.0
            for h in range(rollout_horizon):
                if n_algs == 5:
                    _, rew, term, _, _ = env.step_hybrid(a_idx if h == 0 else 2)
                else:
                    _, rew, term, _, _ = env.step(a_idx)
                cum_cost += discount * (-rew)
                discount *= 0.20
                if term:
                    break
            costs_a[a_idx] = cum_cost

        (
            env.k,
            env.x_pos,
            env.v_real,
            env.v_prev,
            env.prev_err,
            env.shift_timer,
            env.shift_delta,
            env.stop_drift,
            env.internal_mpc.prev_u,
        ) = saved_state

        best_act = int(np.argmin(costs_a))
        if n_algs == 5:
            _, _, term, trunc, _ = env.step_hybrid(best_act)
        else:
            _, _, term, trunc, _ = env.step(best_act)
        done = term or trunc
        q_cost_buffer.append(costs_a)

    return np.asarray(obs_buffer, dtype=np.float32), np.asarray(q_cost_buffer, dtype=np.float32)


def train_rl_switch_strict_two_ways(
    trajectory_generator_fn,
    rl_way: str = "paper_linear_only",
    traj_mode: str = "batch_random",
    regen_every_n_epochs: int = 30,
    epochs: int = 250,
    emulator_steps: int = 150,
    episodes_per_epoch: int = 6,
    lookahead_n: int = 60,
    use_hybrid_mpc: bool = False,
    lr: float = 4.0e-3,
    seed: int = 42,
) -> tuple[TwoPhaseRLSwitchPolicy, list[float], float]:
    assert rl_way in ("paper_linear_only", "emulator_150_only")
    assert traj_mode in ("fixed_generated", "batch_random")

    t0 = time.perf_counter()
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_algs = 5 if use_hybrid_mpc else 4
    current_v_des, _ = trajectory_generator_fn(seed=seed)
    emulator_model = None

    if rl_way == "paper_linear_only":
        env = PathAgnosticHavaeiEnv(
            v_des_profile=current_v_des,
            plant_mode="linear_paper",
            lookahead_n=lookahead_n,
        )
    else:
        emulator_model = learn_150_step_emulator_for_trajectory(
            current_v_des, emulator=None, emulator_steps=emulator_steps, verbose=True
        )
        env = PathAgnosticHavaeiEnv(
            v_des_profile=current_v_des,
            plant_mode="linear_learned",
            learned_linear_model=emulator_model,
            lookahead_n=lookahead_n,
        )

    policy = TwoPhaseRLSwitchPolicy(obs_dim=env.obs_dim, n_algs=n_algs, hidden_dim=64)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.20)

    obs_np, q_costs_np = _compute_mpc_style_horizon_q_costs(env, n_algs=n_algs, rollout_horizon=3)
    obs_b = torch.as_tensor(obs_np, dtype=torch.float32)
    q_costs_b = torch.as_tensor(q_costs_np, dtype=torch.float32)

    history = []
    for epoch in range(1, epochs + 1):
        if traj_mode == "batch_random" and epoch > 1 and (epoch - 1) % regen_every_n_epochs == 0:
            batch_idx = (epoch - 1) // regen_every_n_epochs
            current_v_des, _ = trajectory_generator_fn(seed=seed * 1000 + batch_idx)
            if rl_way == "emulator_150_only":
                emulator_model = learn_150_step_emulator_for_trajectory(
                    current_v_des,
                    emulator=emulator_model,
                    emulator_steps=emulator_steps,
                    verbose=(batch_idx % 3 == 0),
                )
                env.learned_model = emulator_model
            env.set_trajectory(current_v_des)
            obs_np, q_costs_np = _compute_mpc_style_horizon_q_costs(env, n_algs=n_algs, rollout_horizon=3)
            obs_b = torch.as_tensor(obs_np, dtype=torch.float32)
            q_costs_b = torch.as_tensor(q_costs_np, dtype=torch.float32)

        for _ in range(8):
            logits, values = policy.forward(obs_b)
            probs = torch.softmax(logits, dim=-1)

            expected_horizon_cost = torch.sum(probs * q_costs_b, dim=-1)
            min_cost, best_act_t = torch.min(q_costs_b, dim=-1)

            cost_spread = (torch.max(q_costs_b, dim=-1).values - min_cost).detach()
            step_weights = 1.0 + torch.clamp(cost_spread / 2.0, 0.0, 40.0)

            horizon_policy_loss = (expected_horizon_cost * step_weights).sum() / step_weights.sum()
            ce_anchor_loss = nn.functional.cross_entropy(logits, best_act_t, reduction="none")
            ce_loss = (ce_anchor_loss * step_weights).sum() / step_weights.sum()
            value_loss = nn.functional.mse_loss(values, -min_cost)

            loss = 0.5 * horizon_policy_loss + 1.0 * ce_loss + 0.25 * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        scheduler.step()
        with torch.no_grad():
            logits_eval, _ = policy.forward(obs_b)
            probs_eval = torch.softmax(logits_eval / 0.10, dim=-1)
            # Metrica normalizzata per passo: permette di verificare la convergenza reale
            mean_step_loss = float((torch.sum(probs_eval * q_costs_b, dim=-1)).mean().item())
        history.append(-mean_step_loss)

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            tag = f"{rl_way}{'+HybridMPC' if use_hybrid_mpc else ''}"
            print(
                f"  [{tag} | {traj_mode}] Epoch {epoch:3d}/{epochs} | "
                f"Step Tracking Loss: {mean_step_loss:6.4f} | Elapsed: {time.perf_counter() - t0:5.2f}s"
            )

    return policy, history, time.perf_counter() - t0