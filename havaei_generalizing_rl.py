"""
Unified Forward Closed-Loop PID, MPC & Nonlinear Train Physics (`havaei_generalizing_rl.py`).
Includes atomic hooks for `SafetyFilterShield` (`safety_shield.py`) to support both
Runtime Shielding and Filter-Aware RL Training (Section 3.5.3 of Hsu, Hu, & Fisac, 2024).
"""
from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from pure_mpc_controller import PureLiuMPC
from safety_shield import SafetyFilterShield, ShieldConfig

ALGORITHMS = ("GA", "PSO", "DE", "ICA")
HYBRID_ALGORITHMS = ("GA", "PSO", "DE", "ICA", "MPC")
ALG_TO_IDX = {name: i for i, name in enumerate(ALGORITHMS)}


# ===========================================================================
# 1. Offline Heuristic PID Optimization via Cardano's Formula (Eqs. 13 - 50)
# ===========================================================================
def evaluate_cardano_step_metrics(kp: float, ki: float, kd: float, t_grid: np.ndarray) -> tuple[float, float, float]:
    kd_p = 0.4326 + 0.07128 * kd
    kp_p = 0.0324 + 0.07128 * kp
    ki_p = 0.07128 * ki

    roots = np.roots([1.0, kd_p, kp_p, ki_p])
    if np.any(np.real(roots) >= -1e-4):
        return 10.0, 50.0, 50.0

    num_poly = np.array([0.07128 * kd, 0.07128 * kp, 0.07128 * ki], dtype=np.complex128)
    y_t = np.ones_like(t_grid, dtype=np.complex128)
    for idx in range(3):
        p_i = roots[idx]
        other_roots = [roots[j] for j in range(3) if j != idx]
        denom = p_i * (p_i - other_roots[0]) * (p_i - other_roots[1])
        if abs(denom) < 1e-8:
            return 10.0, 50.0, 50.0
        res_i = np.polyval(num_poly, p_i) / denom
        y_t += res_i * np.exp(p_i * t_grid)

    resp = np.real(y_t)
    mp = float(max(0.0, np.max(resp) - 1.0))

    cross_idx = np.where(resp >= 0.90)[0]
    tr = float(t_grid[cross_idx[0]]) if len(cross_idx) > 0 else float(t_grid[-1])

    outside_idx = np.where(np.abs(resp - 1.0) > 0.02)[0]
    ts = float(t_grid[outside_idx[-1]]) if len(outside_idx) > 0 else 0.1
    return mp, ts, tr


def optimize_heuristic_pid_bank(n_iters: int = 25, pop_size: int = 60, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t_grid = np.linspace(0.0, 12.0, 240)

    def eval_pop(pop: np.ndarray, mode_weights: tuple[float, float, float]) -> np.ndarray:
        m_arr = np.array([evaluate_cardano_step_metrics(p[0], p[1], p[2], t_grid) for p in pop])
        w1 = mode_weights[0] / (np.max(m_arr[:, 0]) + 1e-5)
        w2 = mode_weights[1] / (np.max(m_arr[:, 1]) + 1e-5)
        w3 = mode_weights[2] / (np.max(m_arr[:, 2]) + 1e-5)
        return w1 * m_arr[:, 0] + w2 * m_arr[:, 1] + w3 * m_arr[:, 2]

    pop_ga = rng.uniform([30.0, 12.0, 20.0], [55.0, 28.0, 38.0], size=(pop_size, 3))
    for _ in range(n_iters):
        costs = eval_pop(pop_ga, (1.0, 1.0, 1.0))
        probs = np.exp(-1.5 * (costs - np.min(costs)))
        probs /= np.sum(probs)
        idx = rng.choice(pop_size, size=pop_size, p=probs)
        alpha = rng.uniform(0.0, 1.0, size=(pop_size, 3))
        children = alpha * pop_ga[idx] + (1.0 - alpha) * np.roll(pop_ga[idx], 1, axis=0)
        children += rng.normal(0.0, 1.5, size=(pop_size, 3))
        pop_ga = np.clip(children, [25.0, 8.0, 15.0], [85.0, 55.0, 65.0])
    best_ga = pop_ga[int(np.argmin(eval_pop(pop_ga, (1.0, 1.0, 1.0))))]

    pop_pso = rng.uniform([22.0, 5.0, 42.0], [38.0, 14.0, 62.0], size=(pop_size, 3))
    vel_pso = np.zeros_like(pop_pso)
    pbest = pop_pso.copy()
    pbest_cost = eval_pop(pop_pso, (3.5, 0.8, 0.4))
    gbest = pbest[int(np.argmin(pbest_cost))].copy()
    for _ in range(n_iters):
        r1, r2 = rng.uniform(0, 1, size=(pop_size, 3)), rng.uniform(0, 1, size=(pop_size, 3))
        vel_pso = 0.65 * vel_pso + 1.4 * r1 * (pbest - pop_pso) + 1.6 * r2 * (gbest - pop_pso)
        pop_pso = np.clip(pop_pso + vel_pso, [20.0, 5.0, 40.0], [40.0, 15.0, 65.0])
        c_new = eval_pop(pop_pso, (3.5, 0.8, 0.4))
        improved = c_new < pbest_cost
        pbest[improved] = pop_pso[improved]
        pbest_cost[improved] = c_new[improved]
        if np.min(pbest_cost) < eval_pop(gbest[None, :], (3.5, 0.8, 0.4))[0]:
            gbest = pbest[int(np.argmin(pbest_cost))].copy()
    best_pso = gbest

    pop_de = rng.uniform([48.0, 18.0, 36.0], [66.0, 32.0, 54.0], size=(pop_size, 3))
    cost_de = eval_pop(pop_de, (2.0, 2.5, 0.8))
    for _ in range(n_iters):
        r1 = np.roll(pop_de, 1, axis=0)
        r2 = np.roll(pop_de, 2, axis=0)
        r3 = np.roll(pop_de, 3, axis=0)
        mutant = np.clip(r1 + 0.6 * (r2 - r3), [45.0, 15.0, 35.0], [68.0, 34.0, 56.0])
        mask = rng.uniform(0, 1, size=(pop_size, 3)) < 0.8
        trial = np.where(mask, mutant, pop_de)
        c_trial = eval_pop(trial, (2.0, 2.5, 0.8))
        imp = c_trial < cost_de
        pop_de[imp] = trial[imp]
        cost_de[imp] = c_trial[imp]
    best_de = pop_de[int(np.argmin(cost_de))]

    pop_ica = rng.uniform([68.0, 38.0, 16.0], [86.0, 56.0, 28.0], size=(pop_size, 3))
    for _ in range(n_iters):
        costs = eval_pop(pop_ica, (0.3, 1.0, 3.5))
        imp_idx = np.argsort(costs)[:6]
        imperialists = pop_ica[imp_idx]
        for i in range(pop_size):
            target_imp = imperialists[i % len(imperialists)]
            d = target_imp - pop_ica[i]
            pop_ica[i] = pop_ica[i] + rng.uniform(0.0, 2.0, size=3) * d + rng.uniform(-1.0, 1.0, size=3)
        pop_ica = np.clip(pop_ica, [68.0, 38.0, 16.0], [86.0, 56.0, 28.0])
    best_ica = pop_ica[int(np.argmin(eval_pop(pop_ica, (0.3, 1.0, 3.5))))]

    return {
        "GA": best_ga.astype(np.float64),
        "PSO": best_pso.astype(np.float64),
        "DE": best_de.astype(np.float64),
        "ICA": best_ica.astype(np.float64),
    }


COMPUTED_PID_BANK = optimize_heuristic_pid_bank()


# ===========================================================================
# 2. True Nonlinear Train Physics (Equations 2 - 10 of Havaei & Sandidzadeh)
# ===========================================================================
class NonlinearTrainPhysics:
    def __init__(self, mass_kg: float = 85000.0, sample_dist_d: float = 8.33):
        self.M = mass_kg
        self.gamma = 0.08
        self.M_eff = self.M * (1.0 + self.gamma)
        self.g = 9.81
        self.d = sample_dist_d
        self.A = 14.5
        self.B = 185.0
        self.C = 1450.0
        self.max_traction_accel = 1.85
        self.max_braking_decel = 1.65

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

    def max_propulsion_force(self, v: float, u_cmd: float) -> float:
        if u_cmd >= 0.0:
            scale = 1.0 if v <= 14.0 else (14.0 / max(v, 14.0))
            return self.M_eff * self.max_traction_accel * scale
        return self.M_eff * self.max_braking_decel


# ===========================================================================
# 3. Stabilized Causal Physics Emulator & RL Policy Network
# ===========================================================================
class LearnedLinearizationModel(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
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

        base = torch.stack([zeros_1, v_norm, u_norm, ext_force, quad_norm], dim=-1)
        feats = torch.cat([base, zeros_3], dim=-1)
        dv_residual = self.net(feats).squeeze(-1)
        return torch.clamp(v_lin_next + dv_residual, min=0.0)


class TwoPhaseRLSwitchPolicy(nn.Module):
    def __init__(self, obs_dim: int = 20, n_algs: int = 4, hidden_dim: int = 96):
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
        if deterministic:
            act_idx = int(torch.argmax(logits).item())
            if soft_blend:
                sharp_probs = torch.softmax(logits / 0.12, dim=-1).detach().numpy()
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

    def __init__(self, use_transient_chain: bool = True):
        self.use_transient_chain = use_transient_chain
        self.current_mode: int | None = None
        self.chain_step: int = 3

    def reset(self):
        self.current_mode = None
        self.chain_step = 3

    def select_action(self, env) -> np.ndarray:
        mode = env.detect_hardcoded_mode(env.k)
        if self.current_mode is None or mode == self.current_mode:
            self.chain_step = min(3, self.chain_step + 1)
        else:
            self.chain_step = 1 if self.use_transient_chain else 3
        self.current_mode = mode

        chain = self.CHAINS[mode]
        selected_alg = chain[self.chain_step]
        w = np.zeros(4, dtype=np.float64)
        w[ALG_TO_IDX[selected_alg]] = 1.0
        return w


# ===========================================================================
# 4. Unified Closed-Loop PID + MPC + Nonlinear Train Environment
# ===========================================================================
class PathAgnosticHavaeiEnv:
    def __init__(
        self,
        v_des_profile: np.ndarray | None = None,
        plant_mode: str = "nonlinear",
        learned_linear_model: LearnedLinearizationModel | None = None,
        lookahead_n: int = 60,
        past_n: int = 4,
        sample_dist_d: float = 8.33,
        shield: SafetyFilterShield | None = None,
    ):
        assert plant_mode in ("nonlinear", "linear_paper", "linear_learned")
        self.plant_mode = plant_mode
        self.learned_model = learned_linear_model
        self.lookahead_n = lookahead_n
        self.past_n = past_n
        self.d = sample_dist_d
        self.dt = 0.50
        self.shield = shield
        self.physics = NonlinearTrainPhysics(sample_dist_d=sample_dist_d)
        self.internal_mpc = PureLiuMPC(
            n_p=10,
            n_c=5,
            q1=140.0,
            q2=0.08,
            q3=0.45,
            adaptive_shift=True,
            mass_kg=self.physics.M,
        )

        self.pid_bank = copy.deepcopy(COMPUTED_PID_BANK)
        self.gains_scale = np.max(np.stack(list(self.pid_bank.values())), axis=0)
        self.u_scale = 85.0

        if v_des_profile is None:
            v_des_profile = np.zeros(112, dtype=np.float64)
        self.set_trajectory(v_des_profile)
        self.reset()

    @property
    def obs_dim(self) -> int:
        return 20

    def set_trajectory(self, v_des_profile: np.ndarray):
        self.v_des = np.asarray(v_des_profile, dtype=np.float64).copy()
        self.max_steps = len(self.v_des) - 2
        self.x_ref_profile = np.cumsum(self.v_des * self.dt)

    def reset(self):
        self.k = 0
        self.x_pos = 0.0
        self.v_real = 0.0
        self.v_prev = 0.0
        self.int_err = 0.0
        self.prev_err = 0.0
        self.u_act = 0.0
        self.shift_timer = 0
        self.shift_delta = 0.0
        self.stop_drift = 0.0
        self.internal_mpc.reset()
        if self.shield is not None:
            self.shield.reset_stats()
        return self.get_observation()

    def get_internal_state(self):
        return (
            self.k,
            self.x_pos,
            self.v_real,
            self.v_prev,
            self.int_err,
            self.prev_err,
            self.u_act,
            self.shift_timer,
            self.shift_delta,
            self.stop_drift,
            self.internal_mpc.prev_u,
            self.internal_mpc.prev_v,
        )

    def set_internal_state(self, state_tuple):
        (
            self.k,
            self.x_pos,
            self.v_real,
            self.v_prev,
            self.int_err,
            self.prev_err,
            self.u_act,
            self.shift_timer,
            self.shift_delta,
            self.stop_drift,
            self.internal_mpc.prev_u,
            self.internal_mpc.prev_v,
        ) = state_tuple

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

        if dv_next > 1.5 or a_ref[0] > 1.0 or (0 < self.shift_timer < 8 and self.shift_delta > 0):
            return 2
        if dv_next < -1.5 or a_ref[0] < -1.0 or (0 < self.shift_timer < 8 and self.shift_delta < 0):
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

        a_short = np.tanh(a_ref[:5] / 0.35)
        a_med_mean = np.array([np.tanh(np.mean(a_ref[5:20]) / 0.35)]) if len(a_ref) >= 20 else np.array([0.0])
        a_med_std = np.array([np.tanh(np.std(a_ref[5:20]) / 0.50)]) if len(a_ref) >= 20 else np.array([0.0])
        a_long_mean = np.array([np.tanh(np.mean(a_ref[20:]) / 0.35)]) if len(a_ref) > 20 else np.array([0.0])
        a_long_std = np.array([np.tanh(np.std(a_ref[20:]) / 0.50)]) if len(a_ref) > 20 else np.array([0.0])
        a_past_obs = np.tanh(a_past / 0.60)

        k_next = min(self.k + 1, self.max_steps)
        dv_next = float(self.v_des[k_next] - self.v_des[min(self.k, self.max_steps)])
        v_next = float(self.v_des[k_next])
        x_sample = k_next * self.d

        jump_onset = 1.0 if (abs(dv_next) > 1.5 or abs(a_ref[0]) > 1.0) else 0.0
        settle_tail = (
            float(np.exp(-0.25 * (self.shift_timer + 1)))
            if (jump_onset == 0.0 and 0 < self.shift_timer < 8)
            else 0.0
        )
        dv_real_norm = float(np.tanh((self.v_real - self.v_prev) / 1.5))
        stop_prox = (
            float(np.exp(-v_next / 2.8))
            if (jump_onset == 0.0 and (dv_next < -0.02 or v_next <= 0.5))
            else 0.0
        )

        v_norm = self.v_real / 15.0
        err = float(v_next - self.v_real)
        ext_force = (
            self.physics.grade_resistance(x_sample) + self.physics.curve_resistance(x_sample)
        ) / (0.03 * self.physics.M * self.physics.g)

        return np.concatenate([
            a_short.astype(np.float32),
            a_med_mean.astype(np.float32),
            a_med_std.astype(np.float32),
            a_long_mean.astype(np.float32),
            a_long_std.astype(np.float32),
            a_past_obs.astype(np.float32),
            np.array([jump_onset, settle_tail, dv_real_norm, stop_prox, v_norm, err, ext_force], dtype=np.float32),
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

    def _compute_pid_command(
        self,
        pid_gains: np.ndarray,
        v_target: float,
        dv_target: float,
        update_state: bool = True,
    ) -> float:
        if v_target <= 1e-4 and abs(dv_target) <= 1e-4 and self.v_real < 0.04:
            if update_state:
                self.int_err = 0.0
            return 0.0

        kp, ki, kd = float(pid_gains[0]), float(pid_gains[1]), float(pid_gains[2])
        err_now = v_target - self.v_real
        vel_rate = (self.v_real - self.v_prev) / self.dt
        ref_rate = dv_target / self.dt

        new_int_err = float(np.clip(self.int_err + err_now * self.dt, -3.5, 3.5))
        if update_state:
            self.int_err = new_int_err

        if abs(dv_target) > 1.5:
            d_term = 0.20 * ref_rate - 1.35 * vel_rate
        else:
            d_term = 0.55 * ref_rate - 1.10 * vel_rate

        u_raw = kp * err_now + ki * new_int_err + kd * d_term
        return float(np.clip(u_raw / self.u_scale, -1.0, 1.0))

    def _integrate_plant_physics(self, u_cmd: float, x_sample: float) -> tuple[float, float]:
        u_clamped = float(np.clip(u_cmd, -1.0, 1.0))
        self.u_act = 0.72 * u_clamped + 0.28 * self.u_act

        f_prop_lin = self.u_act * self.physics.M_eff * self.physics.max_traction_accel
        a_lin = (f_prop_lin - self.physics.B * self.v_real) / self.physics.M_eff
        inertia_momentum = 0.14 * (self.v_real - self.v_prev)
        v_linear = float(max(0.0, self.v_real + self.dt * a_lin + inertia_momentum))

        f_prop_nl = self.u_act * self.physics.max_propulsion_force(self.v_real, self.u_act)
        f_drag = self.physics.basic_resistance(self.v_real)
        f_grade = self.physics.grade_resistance(x_sample)
        f_curve = self.physics.curve_resistance(x_sample)

        a_nl = (f_prop_nl - f_drag - f_grade - f_curve) / self.physics.M_eff
        v_nonlinear = float(max(0.0, self.v_real + self.dt * a_nl + inertia_momentum))
        return v_nonlinear, v_linear

    def _finalize_step(
        self,
        next_v: float,
        v_lin: float,
        u_cmd: float,
        alg_name: str,
        w: np.ndarray,
        gains_norm: np.ndarray,
        k_curr: int,
        x_sample: float,
        ext_f_norm: float,
        shield_diff: float = 0.0,
    ):
        v_target = float(self.v_des[k_curr])
        v_prev_target = float(self.v_des[max(0, k_curr - 1)])
        dv_now = v_target - v_prev_target

        if abs(dv_now) > 1.5:
            self.shift_timer = 1
            self.shift_delta = dv_now
        elif self.shift_timer > 0:
            self.shift_timer = (self.shift_timer + 1) if self.shift_timer < 8 else 0

        old_v = self.v_real
        self.v_prev = old_v
        self.v_real = next_v
        self.x_pos += 0.5 * (old_v + self.v_real) * self.dt

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
        is_station_standstill = (v_target <= 1e-3) and (self.v_real > 1e-3) and not is_shift_window
        step_stop_pen = 0.0
        if is_stopping_zone:
            step_stop_pen = abs(self.v_real - v_target) * self.dt
            self.stop_drift += step_stop_pen

        terminated = self.k >= self.max_steps

        if is_shift_window:
            reward = -65.0 * abs_err - 180.0 * overshoot
            step_w = 28.0
        elif is_stopping_zone or is_station_standstill:
            reward = -140.0 * abs_err - 450.0 * step_stop_pen - 60.0 * self.stop_drift
            step_w = 25.0
        elif dv_now < -0.04:
            reward = -90.0 * abs_err
            step_w = 6.0
        else:
            reward = -45.0 * abs_err
            step_w = 1.5

        # Filter-Aware RL penalty (Section 3.5.3 of Hsu, Hu, & Fisac, 2024):
        # Penalizes the task policy when it forces the safety shield to intervene.
        if self.shield is not None and shield_diff > 0.0:
            reward -= self.shield.cfg.intervention_penalty_weight * shield_diff

        if terminated:
            reward -= 250.0 * self.stop_drift + 180.0 * self.v_real

        info = {
            "k": k_curr,
            "x_pos": x_sample,
            "true_integrated_x": self.x_pos,
            "ext_f_norm": ext_f_norm,
            "alg": alg_name,
            "weights": w,
            "gains_norm": gains_norm,
            "u_pid": u_cmd,
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
            "shield_intervened": shield_diff > 1e-5,
            "shield_diff": shield_diff,
        }
        return self.get_observation(), float(reward), terminated, False, info

    def _predict_next_velocity(
        self,
        old_v: float,
        u_cmd: float,
        x_sample: float,
        ext_f_norm: float,
        gains_norm: np.ndarray,
    ) -> tuple[float, float]:
        v_nl, v_lin = self._integrate_plant_physics(u_cmd, x_sample)
        if self.plant_mode == "linear_paper":
            return v_lin, v_lin
        if self.plant_mode == "nonlinear":
            return v_nl, v_lin
        assert self.learned_model is not None
        with torch.no_grad():
            x_t = torch.tensor([x_sample], dtype=torch.float32)
            vc_t = torch.tensor([old_v], dtype=torch.float32)
            vlin_t = torch.tensor([v_lin], dtype=torch.float32)
            u_t = torch.tensor([u_cmd], dtype=torch.float32)
            ext_t = torch.tensor([ext_f_norm], dtype=torch.float32)
            g_t = torch.as_tensor(gains_norm, dtype=torch.float32).unsqueeze(0)
            next_v = float(self.learned_model(x_t, vc_t, vlin_t, u_t, ext_t, g_t).item())
        return next_v, v_lin

    def step(self, action: int | np.ndarray):
        shield_diff = 0.0
        if self.shield is not None:
            action, shield_diff = self.shield.filter_pid_weights(self, action)

        w, alg_name, pid_gains = self._alg_weights_from_action(action)
        self.k += 1
        k_curr = min(self.k, self.max_steps)
        x_sample = k_curr * self.d
        old_v = self.v_real

        v_target = float(self.v_des[k_curr])
        dv_target = v_target - float(self.v_des[max(0, k_curr - 1)])
        ext_f_norm = (
            self.physics.grade_resistance(x_sample) + self.physics.curve_resistance(x_sample)
        ) / (0.03 * self.physics.M * self.physics.g)
        gains_norm = pid_gains / self.gains_scale

        u_cmd = self._compute_pid_command(pid_gains, v_target, dv_target, update_state=True)
        next_v, v_lin = self._predict_next_velocity(old_v, u_cmd, x_sample, ext_f_norm, gains_norm)

        return self._finalize_step(
            next_v, v_lin, u_cmd, alg_name, w, gains_norm, k_curr, x_sample, ext_f_norm, shield_diff=shield_diff
        )

    def step_custom_pid(self, pid_gains: np.ndarray, alg_name: str = "FUZZY"):
        pid_gains = np.asarray(pid_gains, dtype=np.float64)
        self.k += 1
        k_curr = min(self.k, self.max_steps)
        x_sample = k_curr * self.d
        old_v = self.v_real

        v_target = float(self.v_des[k_curr])
        dv_target = v_target - float(self.v_des[max(0, k_curr - 1)])
        ext_f_norm = (
            self.physics.grade_resistance(x_sample) + self.physics.curve_resistance(x_sample)
        ) / (0.03 * self.physics.M * self.physics.g)
        gains_norm = pid_gains / self.gains_scale

        u_cmd = self._compute_pid_command(pid_gains, v_target, dv_target, update_state=True)
        next_v, v_lin = self._predict_next_velocity(old_v, u_cmd, x_sample, ext_f_norm, gains_norm)

        return self._finalize_step(
            next_v, v_lin, u_cmd, alg_name, np.zeros(4, dtype=np.float64), gains_norm, k_curr, x_sample, ext_f_norm
        )

    def step_mpc(self, u_mpc: float, shield_diff: float = 0.0):
        self.k += 1
        k_curr = min(self.k, self.max_steps)
        x_sample = k_curr * self.d
        old_v = self.v_real
        v_target = float(self.v_des[k_curr])
        ext_f_norm = (
            self.physics.grade_resistance(x_sample) + self.physics.curve_resistance(x_sample)
        ) / (0.03 * self.physics.M * self.physics.g)

        u_cmd = float(np.clip(u_mpc, -1.0, 1.0))
        gains_norm = np.ones(3, dtype=np.float64)
        next_v, v_lin = self._predict_next_velocity(old_v, u_cmd, x_sample, ext_f_norm, gains_norm)

        err_now = v_target - next_v
        self.int_err = float(np.clip(0.65 * self.int_err + err_now * self.dt, -3.5, 3.5))

        return self._finalize_step(
            next_v,
            v_lin,
            u_cmd,
            "MPC",
            np.zeros(4, dtype=np.float64),
            gains_norm,
            k_curr,
            x_sample,
            ext_f_norm,
            shield_diff=shield_diff,
        )

    def step_hybrid(self, action: int | np.ndarray):
        k_curr_pre = self.k
        x_sample_next = min(k_curr_pre + 1, self.max_steps) * self.d
        ref_h = self.v_des[k_curr_pre + 1 : min(k_curr_pre + 1 + self.internal_mpc.n_p, len(self.v_des))]
        slope_iota = (
            (self.physics.grade_resistance(x_sample_next) + self.physics.curve_resistance(x_sample_next))
            / (self.physics.M * self.physics.g)
            if self.plant_mode != "linear_paper"
            else 0.0
        )

        # Save MPC internal state in case MPC weight is 0 after filtering
        saved_mpc_u, saved_mpc_v = self.internal_mpc.prev_u, self.internal_mpc.prev_v
        u_mpc_cand = self.internal_mpc.select_action(
            self.v_real, ref_h, slope_iota=slope_iota, dt_step=self.dt
        )

        shield_diff = 0.0
        if self.shield is not None:
            action, shield_diff = self.shield.filter_hybrid_weights(self, action, u_mpc_cand)

        if isinstance(action, (int, np.integer)):
            act_idx = int(action)
            w5 = np.zeros(5, dtype=np.float64)
            w5[act_idx] = 1.0
        else:
            w5 = np.asarray(action, dtype=np.float64)
            w5 = w5 / (np.sum(w5) + 1e-12)
            act_idx = int(np.argmax(w5))

        w_mpc = float(w5[4])
        if w_mpc >= 0.999:
            return self.step_mpc(u_mpc_cand, shield_diff=shield_diff)
        elif w_mpc <= 0.02:
            self.internal_mpc.prev_u, self.internal_mpc.prev_v = saved_mpc_u, saved_mpc_v
            w4 = w5[:4] / (np.sum(w5[:4]) + 1e-12)
            # Call internal PID step without double-filtering
            saved_shield = self.shield
            self.shield = None
            obs, rew, term, trunc, info = self.step(w4)
            self.shield = saved_shield
            if self.shield is not None and shield_diff > 0.0:
                rew -= self.shield.cfg.intervention_penalty_weight * shield_diff
                info["shield_intervened"] = True
                info["shield_diff"] = shield_diff
            return obs, rew, term, trunc, info
        else:
            w4 = w5[:4] / (np.sum(w5[:4]) + 1e-12)
            _, _, pid_gains = self._alg_weights_from_action(w4)

            self.k += 1
            k_curr = min(self.k, self.max_steps)
            x_sample = k_curr * self.d
            old_v = self.v_real
            v_target = float(self.v_des[k_curr])
            dv_target = v_target - float(self.v_des[max(0, k_curr - 1)])
            ext_f_norm = (
                self.physics.grade_resistance(x_sample) + self.physics.curve_resistance(x_sample)
            ) / (0.03 * self.physics.M * self.physics.g)

            u_pid = self._compute_pid_command(pid_gains, v_target, dv_target, update_state=True)
            u_blend = float(np.clip(w_mpc * u_mpc_cand + (1.0 - w_mpc) * u_pid, -1.0, 1.0))
            self.internal_mpc.prev_u = u_blend

            gains_norm = pid_gains / self.gains_scale
            next_v, v_lin = self._predict_next_velocity(old_v, u_blend, x_sample, ext_f_norm, gains_norm)
            alg_name = HYBRID_ALGORITHMS[act_idx]
            return self._finalize_step(
                next_v, v_lin, u_blend, alg_name, w4, gains_norm, k_curr, x_sample, ext_f_norm, shield_diff=shield_diff
            )


# ===========================================================================
# 5. Stabilized Neural Emulator & Cumulative Multi-Track RL Trainers
# ===========================================================================
_EMULATOR_REPLAY_BUFFER: list[dict[str, torch.Tensor]] = []
LAST_TRAINED_EMULATOR: LearnedLinearizationModel | None = None


def learn_150_step_emulator_for_trajectory(
    v_des: np.ndarray,
    emulator: LearnedLinearizationModel | None = None,
    emulator_steps: int = 150,
    n_rollouts: int = 8,
    lr: float = 2.5e-3,
    verbose: bool = False,
    return_curve: bool = False,
):
    global _EMULATOR_REPLAY_BUFFER, LAST_TRAINED_EMULATOR
    env_nl = PathAgnosticHavaeiEnv(v_des_profile=v_des, plant_mode="nonlinear")
    x_list, vc_list, vlin_list, u_list, ext_list, g_list, vnext_list = [], [], [], [], [], [], []
    shift_dir_list = []
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

            if info.get("shift_active", False):
                s_dir = 1.0 if env_nl.shift_delta >= 0.0 else -1.0
            else:
                s_dir = 0.0

            x_list.append(info["x_pos"])
            vc_list.append(v_curr)
            vlin_list.append(info["v_lin"])
            u_list.append(info["u_pid"])
            ext_list.append(info["ext_f_norm"])
            g_list.append(info["gains_norm"])
            vnext_list.append(info["v_real"])
            shift_dir_list.append(s_dir)

    cur_data = {
        "x": torch.tensor(x_list, dtype=torch.float32),
        "vc": torch.tensor(vc_list, dtype=torch.float32),
        "vlin": torch.tensor(vlin_list, dtype=torch.float32),
        "u": torch.tensor(u_list, dtype=torch.float32),
        "ext": torch.tensor(ext_list, dtype=torch.float32),
        "g": torch.as_tensor(np.asarray(g_list), dtype=torch.float32),
        "target": torch.tensor(vnext_list, dtype=torch.float32),
        "s_dir": torch.tensor(shift_dir_list, dtype=torch.float32),
    }

    if emulator is None:
        _EMULATOR_REPLAY_BUFFER.clear()
        emulator = LearnedLinearizationModel(hidden_dim=64)
        step_budget = emulator_steps
        eff_lr = lr
    else:
        step_budget = max(40, emulator_steps // 2)
        eff_lr = lr * 0.45

    _EMULATOR_REPLAY_BUFFER.append(cur_data)
    if len(_EMULATOR_REPLAY_BUFFER) > 5:
        _EMULATOR_REPLAY_BUFFER.pop(0)

    x_t = torch.cat([d["x"] for d in _EMULATOR_REPLAY_BUFFER], dim=0)
    vc_t = torch.cat([d["vc"] for d in _EMULATOR_REPLAY_BUFFER], dim=0)
    vlin_t = torch.cat([d["vlin"] for d in _EMULATOR_REPLAY_BUFFER], dim=0)
    u_t = torch.cat([d["u"] for d in _EMULATOR_REPLAY_BUFFER], dim=0)
    ext_t = torch.cat([d["ext"] for d in _EMULATOR_REPLAY_BUFFER], dim=0)
    g_t = torch.cat([d["g"] for d in _EMULATOR_REPLAY_BUFFER], dim=0)
    target_t = torch.cat([d["target"] for d in _EMULATOR_REPLAY_BUFFER], dim=0)
    s_dir_t = torch.cat([d["s_dir"] for d in _EMULATOR_REPLAY_BUFFER], dim=0)

    optimizer = optim.AdamW(emulator.parameters(), lr=eff_lr, weight_decay=2e-4)
    warmup = max(5, step_budget // 10)

    loss_curve = []
    best_rmse = float("inf")
    best_state = copy.deepcopy(emulator.state_dict())

    for step in range(step_budget):
        if step < warmup:
            lr_mult = float(step + 1) / float(warmup)
        else:
            progress = float(step - warmup) / float(max(1, step_budget - warmup))
            lr_mult = 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = eff_lr * lr_mult

        pred = emulator(x_t, vc_t, vlin_t, u_t, ext_t, g_t)
        diff = pred - target_t

        under_pred_in_shift = (s_dir_t * diff) < 0.0
        asym_weight = torch.where(under_pred_in_shift, 3.0, 1.0)
        loss = torch.mean(asym_weight * (diff ** 2))

        with torch.no_grad():
            pred_cur = emulator(
                cur_data["x"], cur_data["vc"], cur_data["vlin"],
                cur_data["u"], cur_data["ext"], cur_data["g"]
            )
            cur_rmse = float(torch.sqrt(nn.functional.mse_loss(pred_cur, cur_data["target"])).item())
            loss_curve.append(cur_rmse)
            if cur_rmse < best_rmse:
                best_rmse = cur_rmse
                best_state = copy.deepcopy(emulator.state_dict())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(emulator.parameters(), 0.5)
        optimizer.step()

    emulator.load_state_dict(best_state)
    emulator.eval()
    LAST_TRAINED_EMULATOR = emulator

    if verbose:
        print(f"    [Physics Surrogate Emulator ({step_budget} steps)] Fit RMSE: {best_rmse:.5f} m/s")
    if return_curve:
        return emulator, loss_curve
    return emulator


def _compute_surrogate_horizon_q_costs(
    surrogate_env: PathAgnosticHavaeiEnv,
    n_algs: int,
    rollout_horizon: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    assert surrogate_env.plant_mode in ("linear_paper", "linear_learned")
    surrogate_env.reset()
    done = False
    obs_buffer, q_cost_buffer = [], []
    is_learned = surrogate_env.plant_mode == "linear_learned"

    while not done:
        obs_buffer.append(surrogate_env.get_observation())
        saved_state = surrogate_env.get_internal_state()

        costs_a = np.zeros(n_algs, dtype=np.float32)
        for a_idx in range(n_algs):
            surrogate_env.set_internal_state(saved_state)
            cum_cost = 0.0
            discount = 1.0
            for h in range(rollout_horizon):
                if is_learned:
                    act_h = (
                        a_idx
                        if (h <= 1 or not surrogate_env.shift_timer)
                        else (4 if n_algs == 5 else 2)
                    )
                else:
                    act_h = (
                        a_idx
                        if (h == 0 or not surrogate_env.shift_timer)
                        else (4 if n_algs == 5 else 2)
                    )

                if n_algs == 5:
                    _, rew, term, _, info_h = surrogate_env.step_hybrid(act_h)
                else:
                    _, rew, term, _, info_h = surrogate_env.step(act_h)

                step_cost = -rew
                if is_learned and info_h.get("shift_active", False):
                    s_sign = 1.0 if surrogate_env.shift_delta >= 0.0 else -1.0
                    v_rate = surrogate_env.v_real - surrogate_env.v_prev
                    margin_os = max(
                        0.0,
                        s_sign * (surrogate_env.v_real + 0.40 * v_rate - info_h["v_des"]),
                    )
                    step_cost += 180.0 * margin_os

                cum_cost += discount * step_cost
                discount *= 0.72
                if term:
                    break
            costs_a[a_idx] = cum_cost

        surrogate_env.set_internal_state(saved_state)
        best_act = int(np.argmin(costs_a))
        if n_algs == 5:
            _, _, term, trunc, _ = surrogate_env.step_hybrid(best_act)
        else:
            _, _, term, trunc, _ = surrogate_env.step(best_act)
        done = term or trunc
        q_cost_buffer.append(costs_a)

    return np.asarray(obs_buffer, dtype=np.float32), np.asarray(q_cost_buffer, dtype=np.float32)


def train_ppo_policy(
    trajectory_generator_fn,
    epochs: int = 250,
    regen_every_n_epochs: int = 40,
    episodes_per_epoch: int = 4,
    lookahead_n: int = 60,
    use_hybrid_mpc: bool = False,
    lr: float = 2.5e-3,
    gamma: float = 0.96,
    gae_lambda: float = 0.92,
    seed: int = 42,
    shield: SafetyFilterShield | None = None,
) -> tuple[TwoPhaseRLSwitchPolicy, list[float], float]:
    t0 = time.perf_counter()
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_algs = 5 if use_hybrid_mpc else 4
    current_v_des, _ = trajectory_generator_fn(seed=seed)
    env = PathAgnosticHavaeiEnv(
        v_des_profile=current_v_des,
        plant_mode="nonlinear",
        lookahead_n=lookahead_n,
        shield=shield,
    )

    policy = TwoPhaseRLSwitchPolicy(obs_dim=env.obs_dim, n_algs=n_algs, hidden_dim=96)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.15)
    history = []

    for epoch in range(1, epochs + 1):
        if epoch > 1 and (epoch - 1) % regen_every_n_epochs == 0:
            batch_idx = (epoch - 1) // regen_every_n_epochs
            current_v_des, _ = trajectory_generator_fn(seed=seed * 1000 + batch_idx)
            env.set_trajectory(current_v_des)

        obs_list, act_list, logp_list, adv_list, ret_list = [], [], [], [], []
        ep_step_costs = []

        for _ in range(episodes_per_epoch):
            obs = env.reset()
            done = False
            ep_o, ep_a, ep_lp, ep_r, ep_v = [], [], [], [], []

            while not done:
                act_idx, _, logp, val = policy.act(obs, deterministic=False, soft_blend=False)
                if use_hybrid_mpc:
                    next_obs, rew, term, trunc, _ = env.step_hybrid(act_idx)
                else:
                    next_obs, rew, term, trunc, _ = env.step(act_idx)
                done = term or trunc

                ep_o.append(obs)
                ep_a.append(act_idx)
                ep_lp.append(logp)
                ep_r.append(rew / 50.0)
                ep_v.append(val)
                obs = next_obs

            ep_step_costs.append(float(-np.mean(ep_r) * 50.0))
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
            logp_list.extend(logp_list[-1:] if not ep_lp else ep_lp)
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
            loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.functional.mse_loss(values, ret_b) - 0.015 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        scheduler.step()
        mean_cost = float(np.mean(ep_step_costs))
        history.append(-mean_cost)
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(
                f"  [Model-Free PPO{'+HybridMPC' if use_hybrid_mpc else ''}] "
                f"Epoch {epoch:3d}/{epochs} | Mean Step Cost: {mean_cost:6.4f} | Elapsed: {time.perf_counter() - t0:5.2f}s"
            )

    return policy, history, time.perf_counter() - t0


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
    lr: float = 3.0e-3,
    seed: int = 42,
    shield: SafetyFilterShield | None = None,
) -> tuple[TwoPhaseRLSwitchPolicy, list[float], float]:
    assert rl_way in ("paper_linear_only", "emulator_150_only", "ppo_nonlinear")
    assert traj_mode in ("fixed_generated", "batch_random")

    if rl_way == "ppo_nonlinear":
        return train_ppo_policy(
            trajectory_generator_fn,
            epochs=epochs,
            regen_every_n_epochs=regen_every_n_epochs,
            lookahead_n=lookahead_n,
            use_hybrid_mpc=use_hybrid_mpc,
            seed=seed,
            shield=shield,
        )

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
            shield=shield,
        )
    else:
        emulator_model = learn_150_step_emulator_for_trajectory(
            current_v_des, emulator=None, emulator_steps=emulator_steps, verbose=False
        )
        env = PathAgnosticHavaeiEnv(
            v_des_profile=current_v_des,
            plant_mode="linear_learned",
            learned_linear_model=emulator_model,
            lookahead_n=lookahead_n,
            shield=shield,
        )

    policy = TwoPhaseRLSwitchPolicy(obs_dim=env.obs_dim, n_algs=n_algs, hidden_dim=96)
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.10)

    obs_np, q_costs_np = _compute_surrogate_horizon_q_costs(env, n_algs=n_algs, rollout_horizon=6)
    obs_pool = [torch.as_tensor(obs_np, dtype=torch.float32)]
    q_costs_pool = [torch.as_tensor(q_costs_np, dtype=torch.float32)]

    obs_b = obs_pool[0]
    q_costs_b = q_costs_pool[0]

    history = []
    best_regret = float("inf")
    best_policy_state = copy.deepcopy(policy.state_dict())

    for epoch in range(1, epochs + 1):
        if traj_mode == "batch_random" and epoch > 1 and (epoch - 1) % regen_every_n_epochs == 0:
            batch_idx = (epoch - 1) // regen_every_n_epochs
            current_v_des, _ = trajectory_generator_fn(seed=seed * 1000 + batch_idx)
            if rl_way == "emulator_150_only":
                emulator_model = learn_150_step_emulator_for_trajectory(
                    current_v_des,
                    emulator=emulator_model,
                    emulator_steps=emulator_steps,
                    verbose=False,
                )
                env.learned_model = emulator_model
            env.set_trajectory(current_v_des)
            obs_np, q_costs_np = _compute_surrogate_horizon_q_costs(env, n_algs=n_algs, rollout_horizon=6)
            obs_pool.append(torch.as_tensor(obs_np, dtype=torch.float32))
            q_costs_pool.append(torch.as_tensor(q_costs_np, dtype=torch.float32))

            obs_b = torch.cat(obs_pool, dim=0)
            q_costs_b = torch.cat(q_costs_pool, dim=0)

            with torch.no_grad():
                logits_pool, _ = policy.forward(obs_b)
                probs_pool = torch.softmax(logits_pool / 0.12, dim=-1)
                min_c_pool = torch.min(q_costs_b, dim=-1).values
                best_regret = float((torch.sum(probs_pool * q_costs_b, dim=-1) - min_c_pool).mean().item())
                best_policy_state = copy.deepcopy(policy.state_dict())

        min_cost, best_act_t = torch.min(q_costs_b, dim=-1)
        q_regret_b = q_costs_b - min_cost.unsqueeze(-1)
        regret_scale = float(q_regret_b.std().item()) + 1.0
        q_regret_norm = q_regret_b / regret_scale

        cost_spread = (torch.max(q_costs_b, dim=-1).values - min_cost).detach()
        step_weights = 1.0 + torch.clamp(cost_spread / (regret_scale * 0.5), 0.0, 15.0)

        for _ in range(8):
            logits, values = policy.forward(obs_b)
            probs = torch.softmax(logits, dim=-1)

            expected_regret = torch.sum(probs * q_regret_norm, dim=-1)
            horizon_policy_loss = (expected_regret * step_weights).sum() / step_weights.sum()

            ce_anchor_loss = nn.functional.cross_entropy(logits, best_act_t, reduction="none")
            ce_loss = (ce_anchor_loss * step_weights).sum() / step_weights.sum()
            value_loss = nn.functional.mse_loss(values, -min_cost / regret_scale)

            loss = 0.6 * horizon_policy_loss + 1.0 * ce_loss + 0.20 * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        scheduler.step()
        with torch.no_grad():
            logits_eval, _ = policy.forward(obs_b)
            probs_eval = torch.softmax(logits_eval / 0.12, dim=-1)
            mean_step_loss = float((torch.sum(probs_eval * q_costs_b, dim=-1)).mean().item())
            mean_regret = float((torch.sum(probs_eval * q_costs_b, dim=-1) - min_cost).mean().item())

        history.append(-mean_step_loss)
        if mean_regret <= best_regret:
            best_regret = mean_regret
            best_policy_state = copy.deepcopy(policy.state_dict())

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            tag = f"{rl_way}{'+HybridMPC' if use_hybrid_mpc else ''}{'+Shield' if shield is not None else ''}"
            print(
                f"  [{tag} | {traj_mode}] Epoch {epoch:3d}/{epochs} | "
                f"Step Tracking Loss: {mean_step_loss:6.4f} (Regret: {mean_regret:5.3f}) | "
                f"Elapsed: {time.perf_counter() - t0:5.2f}s"
            )

    policy.load_state_dict(best_policy_state)
    policy.eval()
    return policy, history, time.perf_counter() - t0