"""
Literature benchmark baselines (docs/RelatedWork_Survey.tex, [BENCHMARK CANDIDATE] entries).

Every method is re-implemented as a *trajectory pilot*: once per macro step it chooses the cruise setpoint
(and, where the method defines them, the acceleration / landing rates). The pilot runs on the same plant,
departure mass identification, disturbance observer, C^2 shaper and LTV-MPC tracking layer as methods 1-9,
so only the trajectory-generation layer differs. These are re-implementations adapted to this interface,
not the authors' code:

  11. Xiao et al. (2023)   - space-domain convex eco-driving, shrinking-horizon MPC solved by one
                             real-time-iteration QP per macro step (RTI-SQP).
  12. Zhong et al. (2022)  - moving-horizon multi-phase Radau pseudospectral MPC that switches between an
                             energy-optimal and a time-optimal (delay recovery) objective.
  13. Chen et al. (2025)   - CVaR distributional (quantile) Q-learning, CDQ, over space stages.
  14. Zhao et al. (2023)   - Shield SARSA: tabular SARSA with a hard overspeed shield.
  15. Huang et al. (2025)  - EITOP: expert rules + PPO + DMTD coasting.
  16. Li et al. (2023)     - SRL-IETTO: operation-mode allocation (ARTSOM) supervision + DDPG with a binary
                             goal reward.
  17. Zhao et al. (2025)   - SSA-DRL: DDPG + safety shield with a searching tree over candidate actions.

The two convex planners use a small OSQP-style ADMM QP solver (numpy only; scipy is not a dependency).
"""
from __future__ import annotations
import os
import eval_counter as _EC

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from railroad_corridor import generate_multi_station_scenario

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"

# Training / validation protocol shared with the PPO pilots (seeds disjoint from tuning 5000+ and eval 9000+).
VAL_SEEDS = [70000 + 42 * 13 + v_i * 31 for v_i in range(6)]


# ===========================================================================
# Common helpers
# ===========================================================================
def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def naive_pace(env) -> float:
    x_rem = max(0.0, env.corridor.station_x_m - env.x_curr)
    t_rem = max(env.corridor.dt, env.corridor.deadline_s - env.t_curr)
    return x_rem / t_rem


def encode_action(env, v_target: float, a_start: float = 0.96, b_land: float = 0.78) -> np.ndarray:
    """Inverse of MPCTransitionRailroadEnv.decode_rl_action (v_cmd = v_naive + 3.8 a0, clipped to the ceiling)."""
    return np.array([
        _clip((v_target - naive_pace(env)) / 3.80, -1.0, 1.0),
        _clip((a_start - 0.96) / 0.22, -1.0, 1.0),
        _clip((b_land - 0.78) / 0.12, -1.0, 1.0),
    ], dtype=np.float32)


def training_scenario(journeys_seed: int, j: int) -> dict:
    """Training journey j (0..1199) of the PPO pilots' protocol for training seed journeys_seed: the same journeys in
    the same order as train_pure_ppo_cosine_pilot (epoch j // 3 + 1, batch j % 3; the third of each epoch extreme)."""
    epoch, b = j // 3 + 1, j % 3
    return generate_multi_station_scenario(
        seed=journeys_seed * 1000 + epoch * 53 + b * 17, n_legs=0, wind_mode="mixed",
        allow_infeasible_breaks=False, extreme_noise=(b == 2),
    )


# Training budget and checkpoint-selection cadence of the PPO pilots (400 epochs x 3 journeys, validation every
# 5 epochs = 15 journeys), used by every learned baseline.
N_TRAIN_JOURNEYS = 1200
VAL_EVERY_JOURNEYS = 15


def validation_scenarios(journeys_seed: int = 42) -> list[dict]:
    """The 6 validation journeys of training seed journeys_seed, as in train_pure_ppo_cosine_pilot."""
    return [
        generate_multi_station_scenario(seed=70000 + journeys_seed * 13 + i * 31, n_legs=0, wind_mode="mixed",
                                        allow_infeasible_breaks=False, extreme_noise=(i % 2 == 1))
        for i in range(6)
    ]


def validation_loss(pilot, emulator, scens: list[dict]) -> float:
    """Same LossScore as eq. (loss_paper), unshielded, on the 6 validation journeys."""
    from dynamic_railroad_mpc import evaluate_multi_station_journey
    vals = [evaluate_multi_station_journey("val", sc, "LIT_PILOT", lit_pilot=pilot, emulator=emulator)["multi_obj_score"]
            for sc in scens]
    return float(np.mean(vals))


def run_training_leg(env, leg: dict, act_fn, observe_fn=None) -> dict:
    obs = env.reset(corridor_override=leg["corridor"], wind_profile=leg["wind_profile"])
    done = False
    info = {}
    while not done:
        act, ctx = act_fn(env, obs)
        obs2, rew, done, info = env.step(act, record_micro=False)
        if observe_fn is not None:
            observe_fn(env, obs, act, ctx, rew, obs2, done, info)
        obs = obs2
    return info


# ===========================================================================
# OSQP-style ADMM QP solver:  min 0.5 x'Px + q'x  s.t.  l <= A x <= u
# ===========================================================================
def solve_qp_admm(P, q, A, l, u, iters: int = 2000, rho: float = 0.1, sigma: float = 1e-6,
                  alpha: float = 1.6, eps: float = 1e-4):
    n = P.shape[0]
    # Row equilibration and column scaling (cheap Ruiz-like pass):
    r_norm = np.maximum(1e-8, np.abs(A).max(axis=1))
    A = A / r_norm[:, None]
    l = l / r_norm
    u = u / r_norm
    c_norm = np.maximum(1e-8, np.sqrt(np.abs(A).max(axis=0) * np.maximum(1e-8, np.abs(np.diag(P)) + np.abs(A).max(axis=0))))
    D = 1.0 / c_norm
    A = A * D[None, :]
    P = P * D[:, None] * D[None, :]
    q = q * D
    eq = (u - l) < 1e-9
    rho_v = np.where(eq, 1e3 * rho, rho)
    K = P + sigma * np.eye(n) + A.T @ (rho_v[:, None] * A)
    K_inv = np.linalg.inv(K)
    x = np.zeros(n)
    z = np.clip(np.zeros(A.shape[0]), l, u)
    y = np.zeros(A.shape[0])
    for k in range(iters):
        xt = K_inv @ (sigma * x - q + A.T @ (rho_v * z - y))
        zt = A @ xt
        x = alpha * xt + (1.0 - alpha) * x
        zr = alpha * zt + (1.0 - alpha) * z
        z = np.clip(zr + y / rho_v, l, u)
        y = y + rho_v * (zr - z)
        if k % 25 == 24:
            Ax = A @ x
            Px = P @ x
            Aty = A.T @ y
            r_p = np.max(np.abs(Ax - z))
            r_d = np.max(np.abs(Px + q + Aty))
            if r_p < eps * (1.0 + max(np.max(np.abs(Ax)), np.max(np.abs(z)))) and \
                    r_d < eps * (1.0 + max(np.max(np.abs(Px)), np.max(np.abs(Aty)), np.max(np.abs(q)))):
                break
            # OSQP adaptive rho (Stellato et al. 2020, eq. 30):
            num = r_p / max(1e-10, max(np.max(np.abs(Ax)), np.max(np.abs(z))))
            den = r_d / max(1e-10, max(np.max(np.abs(Px)), np.max(np.abs(Aty)), np.max(np.abs(q))))
            ratio = math.sqrt(num / max(1e-10, den))
            if ratio > 5.0 or ratio < 0.2:
                rho_v = rho_v * ratio
                K_inv = np.linalg.inv(P + sigma * np.eye(n) + A.T @ (rho_v[:, None] * A))
    return x * D


def solve_qp_ipm(P, q, C, d, G, h, lb=None, ub=None, iters: int = 30, tol: float = 1e-6,
                 budget_s: float | None = None, deadline: float | None = None):
    """Dense primal-dual interior-point QP solver (Mehrotra predictor-corrector):
        min 0.5 x'Px + q'x   s.t.  C x = d,  G x <= h,  lb <= x <= ub.
    Simple bounds are kept as diagonal barrier terms (never as dense rows), so each iteration costs one
    (n+me) symmetric KKT factorisation reused for predictor and corrector.  With `budget_s` the solver is
    anytime: it stops at the wall-clock budget and returns its last (interior) iterate.  Returns (x, converged)."""
    t_end = deadline if deadline is not None else (None if budget_s is None else time.perf_counter() + budget_s)
    n, me = P.shape[0], C.shape[0]
    lb = np.full(n, -np.inf) if lb is None else lb
    ub = np.full(n, np.inf) if ub is None else ub
    iu = np.flatnonzero(np.isfinite(ub)); il = np.flatnonzero(np.isfinite(lb))
    mg, mu_n, ml = G.shape[0], len(iu), len(il)
    m = mg + mu_n + ml
    hf = np.concatenate([h, ub[iu], -lb[il]])

    def Gx(v):
        return np.concatenate([G @ v, v[iu], -v[il]])

    def GTz(w):
        r = G.T @ w[:mg]
        r[iu] += w[mg:mg + mu_n]                  # bound indices are unique: plain fancy-index update
        r[il] -= w[mg + mu_n:]
        return r

    # Row equilibration of the general inequalities / equalities (unit inf-norm rows):
    gn = np.maximum(1e-12, np.max(np.abs(G), axis=1)) if mg else np.ones(0)
    G = G / gn[:, None]; h = h / gn
    cn = np.maximum(1e-12, np.max(np.abs(C), axis=1)) if me else np.ones(0)
    C = C / cn[:, None]; d = d / cn
    hf = np.concatenate([h, ub[iu], -lb[il]])
    K = np.zeros((n + me, n + me))
    K[n:, :n] = C
    K[:n, n:] = C.T
    # Mehrotra/CVXOPT starting point: least-squares solution of the KKT system with unit barrier weights,
    # then slacks and duals shifted into the strictly positive orthant.
    H0 = P + G.T @ G
    H0[iu, iu] += 1.0
    H0[il, il] += 1.0
    K[:n, :n] = H0
    try:
        sol0 = np.linalg.solve(K + 1e-9 * np.eye(n + me), np.concatenate([-q + GTz(hf), d]))
        x = sol0[:n]
    except np.linalg.LinAlgError:
        x = np.zeros(n)
    r0 = hf - Gx(x)
    s = r0.copy()
    z = -r0.copy()
    a_p = -np.min(s); a_d = -np.min(z)
    s = s + max(0.0, a_p) + 1.0 if a_p >= -1e-8 else s
    z = z + max(0.0, a_d) + 1.0 if a_d >= -1e-8 else z
    s = np.maximum(s, 1e-3); z = np.maximum(z, 1e-3)
    y = np.zeros(me)
    best = x.copy()
    converged = False
    scale = 1.0 + max(np.max(np.abs(q)), 1.0)
    mu0 = max(1.0, float(s @ z) / m)
    for _ in range(iters):
        _EC.N[0] += _EC.QP_NODES[0]
        r_d = P @ x + q + GTz(z) + C.T @ y
        r_e = C @ x - d
        r_i = Gx(x) + s - hf
        mu = float(s @ z) / m
        res = max(np.max(np.abs(r_e), initial=0.0), np.max(np.abs(r_i)), np.max(np.abs(r_d)) / scale)
        if mu < tol * mu0 and res < 1e-6 * (1.0 + np.max(np.abs(hf))):
            converged = True
            break
        if t_end is not None and time.perf_counter() > t_end:
            # Escape: accept the iterate only if it is already converged to a looser tolerance.
            converged = bool(mu < 1e-3 * mu0 and res < 1e-4 * (1.0 + np.max(np.abs(hf))))
            break
        W = z / s
        H = P + G.T @ (W[:mg, None] * G)
        H[iu, iu] += W[mg:mg + mu_n]
        H[il, il] += W[mg + mu_n:]
        K[:n, :n] = H
        K[np.arange(n), np.arange(n)] += 1e-9
        try:
            lu = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            break

        def newton(r_cent):
            rhs = np.concatenate([-r_d + GTz(r_cent / s - W * r_i), -r_e])
            sol = lu @ rhs
            dx = sol[:n]
            ds = -r_i - Gx(dx)
            dz = (-r_cent - z * ds) / s
            return dx, sol[n:], ds, dz

        def max_step(v, dv):
            return min(1.0, float((-v / np.minimum(dv, -1e-12)).min()))

        dx, dy, ds, dz = newton(s * z)
        a_aff = min(max_step(s, ds), max_step(z, dz))
        mu_aff = float((s + a_aff * ds) @ (z + a_aff * dz)) / m
        sigma = (mu_aff / max(1e-16, mu)) ** 3
        dx, dy, ds, dz = newton(s * z + ds * dz - sigma * mu)
        a = 0.99 * min(max_step(s, ds), max_step(z, dz))
        xn = x + a * dx
        if not np.all(np.isfinite(xn)):
            break
        x = xn; y = y + a * dy; s = s + a * ds; z = z + a * dz
        best = x
    return best, converged


class PlannerTimeout(Exception):
    """Raised when a planner call reaches its wall-clock escape deadline before the QP solve starts."""


class _QPBuilder:
    """Collects sparse-ish rows into dense matrices for the small planning QPs."""

    def __init__(self, n: int):
        self.n = n
        self.rows, self.lo, self.hi = [], [], []
        self.P = np.zeros((n, n))
        self.q = np.zeros(n)
        self.lb = np.full(n, -np.inf)
        self.ub = np.full(n, np.inf)
        self.converged = True
        self.dyn = []

    def row(self, coeffs: dict, lo: float, hi: float, dyn: bool = False) -> None:
        self.dyn.append(dyn)
        r = np.zeros(self.n)
        for j, c in coeffs.items():
            r[j] += c
        self.rows.append(r)
        self.lo.append(lo)
        self.hi.append(hi)

    def box(self, j: int, lo: float, hi: float) -> None:
        self.lb[j] = max(self.lb[j], lo)
        self.ub[j] = min(self.ub[j], hi)

    def solve(self, states=None, iters: int = 30, budget_s: float | None = None,
              deadline: float | None = None) -> np.ndarray:
        """Solve the QP.  With `states`, the equality rows flagged `dyn=True` (one per state variable) are used
        to condense the states out (x_S = affine(x_U)), as in condensed MPC: the KKT system then only involves
        the inputs, and state bounds become general inequality rows."""
        _EC.QP_NODES[0] = len(states) if states is not None else self.n
        A = np.array(self.rows)
        lo, hi = np.array(self.lo), np.array(self.hi)
        n = self.n
        P = self.P + 1e-6 * np.eye(n)
        q = self.q
        lb, ub = self.lb, np.maximum(self.ub, self.lb)
        if states is None:
            T, t0 = np.eye(n), np.zeros(n)
            keep = np.ones(len(lo), bool)
            lb_u, ub_u = lb, ub
            extra_G = np.zeros((0, n)); extra_h = np.zeros(0)
        else:
            S = np.asarray(states)
            U = np.setdiff1d(np.arange(n), S)
            dyn = np.array(self.dyn, bool)
            Ad, bd = A[dyn], lo[dyn]
            inv_SS = np.linalg.inv(Ad[:, S])
            T = np.zeros((n, len(U))); t0 = np.zeros(n)
            T[U, np.arange(len(U))] = 1.0
            T[S] = -inv_SS @ Ad[:, U]
            t0[S] = inv_SS @ bd
            keep = ~dyn
            lb_u, ub_u = lb[U], ub[U]
            TS, tS = T[S], t0[S]
            fu, fl = np.isfinite(ub[S]), np.isfinite(lb[S])
            extra_G = np.vstack([TS[fu], -TS[fl]])
            extra_h = np.concatenate([ub[S][fu] - tS[fu], -(lb[S][fl] - tS[fl])])
        A, lo, hi = A[keep] @ T, lo[keep] - A[keep] @ t0, hi[keep] - A[keep] @ t0
        eq = np.abs(hi - lo) < 1e-12
        up = (~eq) & np.isfinite(hi)
        dn = (~eq) & np.isfinite(lo)
        G = np.vstack([A[up], -A[dn], extra_G])
        h = np.concatenate([hi[up], -lo[dn], extra_h])
        Pu = T.T @ P @ T
        qu = T.T @ (P @ t0 + q)
        if deadline is not None and time.perf_counter() >= deadline:
            raise PlannerTimeout()
        u, self.converged = solve_qp_ipm(Pu, qu, A[eq], lo[eq], G, h, lb_u, ub_u, iters=iters, budget_s=budget_s,
                                         deadline=deadline)
        return T @ u + t0


def _plan_physics(env):
    """Controller-side longitudinal model (estimated mass, station temperature, DOB wind)."""
    ph = env.physics
    m_eff = ph.M_eff
    rho_air = 293.15 / max(250.0, 273.15 + env.station_temp_c)
    return ph, m_eff, rho_air


def _resistance_lin(ph, m_eff, rho_air, x: float, v0: float, d_hat: float) -> tuple[float, float]:
    """Linearised specific resistance around v0 in E = v^2/2:  r(E) ~= c0 + c1 E   [m/s^2]."""
    v0 = max(0.8, v0)
    g_spec = (ph.grade_resistance(x) + ph.curve_resistance(x)) / m_eff
    c1 = (2.0 * ph.A * rho_air + ph.B / v0) / m_eff
    c0 = (ph.C * rho_air + 0.5 * ph.B * v0) / m_eff + g_spec - d_hat
    return c0, c1


_TIME_CUT_SPEEDS = (1.5, 4.0, 8.0, 12.0, 16.0)


def _lookahead_target(xs: np.ndarray, vs: np.ndarray, x: float, v: float, horizon_s: float = 4.0) -> float:
    return float(np.interp(x + max(2.0, v) * horizon_s, xs, vs))


# ===========================================================================
# 11. Xiao et al. (2023): space-domain convex shrinking-horizon eco-driving MPC (RTI QP per macro step)
# ===========================================================================
class XiaoConvexSHMPCPilot:
    name = "9. Xiao (2023) Convex SHMPC Eco-Driving"

    def __init__(self, n_seg: int = 30, w_late: float = 60.0, w_jerk: float = 40.0, t_reserve: float = 14.0,
                 n_rti: int = 1, budget_s: float = 2.5e-3):
        self.t_reserve = t_reserve
        self.n_rti = n_rti
        self.n_seg = n_seg
        self.w_late = w_late
        self.w_jerk = w_jerk
        self.budget_s = budget_s
        self.prev = None
        self._last_ok = False
        self.calls = 0
        self.escapes = 0

    def reset_leg(self, env) -> None:
        self.prev = None
        env.pilot_filter = None

    def _solve_once(self, env, s_nodes, v_lin, v0, T_rem, deadline=None):
        cor = env.corridor
        ph, m_eff, rho_air = _plan_physics(env)
        N = self.n_seg
        ds = s_nodes[1] - s_nodes[0]
        # Variables: E_0..E_N | Fp_0..Fp_{N-1} | Fb_0..Fb_{N-1} | late slack
        iE, iP, iB, iS = 0, N + 1, 2 * N + 1, 3 * N + 1
        qp = _QPBuilder(3 * N + 2)
        qp.row({iE: 1.0}, 0.5 * v0 * v0, 0.5 * v0 * v0, dyn=True)
        qp.row({iE + N: 1.0}, 0.0, 0.0)
        for i in range(1, N + 1):
            vl = cor.get_speed_limit(min(cor.station_x_m, s_nodes[i])) - 0.25
            # speed limit + RTI trust region around the linearisation point
            lo_v = max(0.0, v_lin[i] - 2.5)
            hi_v = min(vl, v_lin[i] + 2.5)
            if i < N:                                # E_N is pinned by the terminal equality
                qp.box(iE + i, 0.5 * lo_v * lo_v, 0.5 * max(hi_v, lo_v + 0.1) ** 2)
        time_row = {}
        time_rhs = max(1.0, T_rem - self.t_reserve)
        for i in range(N):
            x_mid = s_nodes[i] + 0.5 * ds
            v_mid = max(1.0, 0.5 * (v_lin[i] + v_lin[i + 1]))
            c0, c1 = _resistance_lin(ph, m_eff, rho_air, x_mid, v_mid, env.d_hat)
            qp.row({iE + i + 1: 1.0 + 0.5 * ds * c1, iE + i: -1.0 + 0.5 * ds * c1, iP + i: -ds, iB + i: ds},
                   -ds * c0, -ds * c0, dyn=True)
            a_tr = ph.max_traction_accel * (1.0 if v_mid <= 14.0 else 14.0 / v_mid)
            qp.box(iP + i, 0.0, min(cor.comfort_accel_max, a_tr))
            # Final approach is executed by the shared cosine landing (mean decel 2/pi * b_land):
            qp.box(iB + i, 0.0, cor.comfort_braking_max if s_nodes[i + 1] < cor.station_x_m - 250.0 else 0.5)
            # RTI linearisation of the running time ds / sqrt(2 Ebar) at the previous plan:
            e0 = 0.5 * v_mid * v_mid
            f0 = ds * (2.0 * e0) ** -0.5
            g0 = -ds * (2.0 * e0) ** -1.5
            time_row[iE + i] = time_row.get(iE + i, 0.0) + 0.5 * g0
            time_row[iE + i + 1] = time_row.get(iE + i + 1, 0.0) + 0.5 * g0
            time_rhs -= f0 - g0 * e0
            qp.q[iP + i] = ds                       # traction work per unit mass (J/kg)
            if i + 1 < N:                           # traction-rate (jerk) regularisation
                w = self.w_jerk
                qp.P[iP + i, iP + i] += w; qp.P[iP + i + 1, iP + i + 1] += w
                qp.P[iP + i, iP + i + 1] -= w; qp.P[iP + i + 1, iP + i] -= w
        time_row[iS] = -1.0
        qp.row(time_row, -np.inf, time_rhs)
        qp.box(iS, 0.0, np.inf)
        qp.q[iS] = self.w_late
        sol = qp.solve(states=np.arange(iE, iE + N + 1), deadline=deadline)
        self._last_ok = qp.converged
        return sol, iE, iP

    def act(self, env, obs) -> np.ndarray:
        t_end = time.perf_counter() + self.budget_s      # hard escape for the whole call (build + solve)
        cor = env.corridor
        x0, v0 = env.x_curr, max(0.0, env.v_meas)
        S = cor.station_x_m - x0
        T_rem = cor.deadline_s - env.t_curr
        if S < 150.0 or env.shaper.in_landing:
            return encode_action(env, naive_pace(env))
        N = self.n_seg
        s_nodes = x0 + (S / N) * np.arange(N + 1)
        if self.prev is not None:
            v_lin = np.interp(s_nodes, self.prev[0], self.prev[1])
            n_iter = self.n_rti                      # real-time iteration: n_rti QPs per sampling instant
        else:
            v_lin = np.minimum([cor.get_speed_limit(s) - 0.3 for s in s_nodes], max(3.0, S / max(1.0, T_rem)) * 1.15)
            n_iter = 4                               # converge the first plan of the leg
        v_lin = np.maximum(1.0, v_lin)
        v_lin[0] = max(1.0, v0)
        self.calls += 1
        sol = None
        for it in range(n_iter):
            if it > 0 and t_end - time.perf_counter() < 5e-4:
                break
            try:
                sol_it, iE, iP = self._solve_once(env, s_nodes, v_lin, v0, T_rem, deadline=t_end)
            except (np.linalg.LinAlgError, PlannerTimeout):
                break
            if not self._last_ok:                    # unfinished solve at the escape deadline: discard it
                break
            sol = sol_it
            v_lin = np.sqrt(2.0 * np.maximum(0.0, sol[iE:iE + N + 1]))
        if sol is None:                              # escape / numerical failure: keep flying the previous plan
            self.escapes += 1
            if self.prev is None:
                return encode_action(env, naive_pace(env))
            return encode_action(env, _lookahead_target(self.prev[0], self.prev[1], x0, v0))
        v_plan = v_lin
        self.prev = (s_nodes, v_plan)
        v_tgt = _lookahead_target(s_nodes, v_plan, x0, v0)
        fp0 = float(np.max(sol[iP:iP + 2]))
        a_start = _clip(fp0, 0.74, 1.18) if v_tgt > v0 + 0.3 else 0.96
        return encode_action(env, v_tgt, a_start, 0.78)


# ===========================================================================
# 12. Zhong et al. (2022): moving-horizon multi-phase Radau pseudospectral MPC (energy / time-optimal switch)
# ===========================================================================
def _flipped_lgr(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flipped Legendre-Gauss-Radau nodes on (-1, 1] (n points), weights, and the n x (n+1) differentiation
    matrix acting on the support points [-1, tau_1..tau_n]."""
    from numpy.polynomial import legendre as L
    c = np.zeros(n + 1); c[n - 1] = 1.0; c[n] = 1.0
    r = np.sort(np.real(L.legroots(c)))           # LGR on [-1, 1)
    w = np.zeros(n)
    pn1 = L.legval(r, np.eye(n)[n - 1])
    w[0] = 2.0 / n ** 2
    w[1:] = (1.0 - r[1:]) / (n ** 2 * pn1[1:] ** 2)
    tau = np.sort(-r)                              # flipped: includes +1
    w = w[::-1]
    sup = np.concatenate([[-1.0], tau])
    m = len(sup)
    bw = np.array([1.0 / np.prod([sup[j] - sup[k] for k in range(m) if k != j]) for j in range(m)])
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i != j:
                D[i, j] = (bw[j] / bw[i]) / (sup[i] - sup[j])
        D[i, i] = -np.sum(D[i])
    return tau, w, D[1:, :]


class ZhongPseudospectralMPCPilot:
    name = "10. Zhong (2022) Pseudospectral MPC (Energy/Time)"

    def __init__(self, horizon_m: float = 1600.0, n_col: int = 5, w_jerk: float = 30.0, t_reserve: float = 12.0,
                 budget_s: float = 2.5e-3):
        self.t_reserve = t_reserve
        self.budget_s = budget_s
        self.horizon_m = horizon_m
        self.n_col = n_col
        self.tau, self.w, self.D = _flipped_lgr(n_col)
        self.w_jerk = w_jerk
        self.prev = None
        self.time_mode_steps = 0
        self.steps = 0
        self.escapes = 0

    def reset_leg(self, env) -> None:
        self.prev = None
        env.pilot_filter = None

    def _phases(self, cor, x0: float, x1: float) -> list[tuple[float, float]]:
        cuts = [x0] + [z.x_start for z in cor.speed_zones if x0 + 60.0 < z.x_start < x1 - 60.0] + [x1]
        return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

    def act(self, env, obs) -> np.ndarray:
        t_end = time.perf_counter() + self.budget_s      # hard escape for the whole call (build + solve)
        cor = env.corridor
        x0, v0 = env.x_curr, max(0.0, env.v_meas)
        x_rem = cor.station_x_m - x0
        T_rem = cor.deadline_s - env.t_curr
        if x_rem < 150.0 or env.shaper.in_landing:
            return encode_action(env, naive_pace(env))
        self.steps += 1
        ph, m_eff, rho_air = _plan_physics(env)
        H = min(x_rem, self.horizon_m)
        reaches_station = H >= x_rem - 1e-6
        # Delay detection (Zhong et al.: switch to the time-optimal objective to recover a delay):
        v_feas, _ = env.compute_feasible_deadline_pace(0.96, 0.78)
        time_mode = env.compute_on_track_slack_s(0.96, 0.78) < -1.0 and v_feas >= cor.get_speed_limit(x0) - 0.3
        self.time_mode_steps += int(time_mode)
        if reaches_station:
            T_alloc = T_rem - self.t_reserve
        else:
            xs = np.linspace(x0, x0 + H, 40)
            T_alloc = float(np.sum((H / 39.0) / np.maximum(1.5, np.minimum(v_feas, [cor.get_speed_limit(s) for s in xs]))))
            T_alloc *= max(0.5, (T_rem - self.t_reserve) / max(1.0, T_rem))

        phases = self._phases(cor, x0, x0 + H)
        nc = self.n_col
        n_sup = 1 + nc * len(phases)                      # shared support nodes (phase continuity)
        n_col_tot = nc * len(phases)
        iE, iP, iB, iT, iS = 0, n_sup, n_sup + n_col_tot, n_sup + 2 * n_col_tot, n_sup + 3 * n_col_tot
        qp = _QPBuilder(iS + 1)
        s_sup = [x0]
        for (a, b) in phases:
            s_sup.extend(list(a + 0.5 * (self.tau + 1.0) * (b - a)))
        s_sup = np.array(s_sup)
        if self.prev is not None:
            v_lin = np.maximum(1.0, np.interp(s_sup, self.prev[0], self.prev[1]))
        else:
            v_lin = np.maximum(1.0, np.minimum([cor.get_speed_limit(s) for s in s_sup], max(3.0, v_feas)))
        qp.row({iE: 1.0}, 0.5 * v0 * v0, 0.5 * v0 * v0, dyn=True)
        for k in range(1, n_sup - (1 if reaches_station else 0)):   # the terminal state is pinned when reached
            vl = cor.get_speed_limit(min(cor.station_x_m, s_sup[k] - 1e-3)) - 0.25
            qp.box(iE + k, 0.0, 0.5 * vl * vl)
        if reaches_station:
            qp.row({iE + n_sup - 1: 1.0}, 0.0, 0.0)
        else:
            v_end = min(v_feas, cor.get_speed_limit(x0 + H)) * 0.9
            qp.row({iE + n_sup - 1: 1.0}, 0.5 * v_end * v_end, np.inf)
        time_row = {}
        for p, (a, b) in enumerate(phases):
            Lp = b - a
            base = p * nc                                  # support index of the phase's first point
            for k in range(nc):
                ci = p * nc + k                            # collocation index
                si = base + 1 + k                          # its support node
                c0, c1 = _resistance_lin(ph, m_eff, rho_air, s_sup[si], v_lin[si], env.d_hat)
                # (2/Lp) * sum_j D_kj E_j = Fp - Fb - c0 - c1 E_k
                coeffs = {iE + base + j: (2.0 / Lp) * self.D[k, j] for j in range(nc + 1)}
                coeffs[iE + si] = coeffs.get(iE + si, 0.0) + c1
                coeffs[iP + ci] = -1.0
                coeffs[iB + ci] = 1.0
                qp.row(coeffs, -c0, -c0, dyn=True)
                a_tr = ph.max_traction_accel * (1.0 if v_lin[si] <= 14.0 else 14.0 / v_lin[si])
                qp.box(iP + ci, 0.0, min(cor.comfort_accel_max, a_tr))
                qp.box(iB + ci, 0.0, cor.comfort_braking_max if s_sup[si] < cor.station_x_m - 250.0 else 0.5)
                for vc in _TIME_CUT_SPEEDS + (max(1.2, v_lin[si]),):
                    ec = 0.5 * vc * vc
                    qp.row({iT + ci: 1.0, iE + si: (2.0 * ec) ** -1.5}, 1.5 * (2.0 * ec) ** -0.5, np.inf)
                wq = 0.5 * Lp * self.w[k]                  # Radau quadrature weight (metres)
                time_row[iT + ci] = wq
                if time_mode:
                    qp.q[iT + ci] += 50.0 * wq             # time-optimal objective
                    qp.q[iP + ci] += 0.05 * wq
                else:
                    qp.q[iP + ci] += wq                    # energy-optimal objective
                if k + 1 < nc:
                    w_j = self.w_jerk
                    qp.P[iP + ci, iP + ci] += w_j; qp.P[iP + ci + 1, iP + ci + 1] += w_j
                    qp.P[iP + ci, iP + ci + 1] -= w_j; qp.P[iP + ci + 1, iP + ci] -= w_j
        qp.row({**time_row, iS: -1.0}, -np.inf, max(1.0, T_alloc))
        qp.box(iS, 0.0, np.inf)
        qp.q[iS] = 60.0
        try:
            sol = qp.solve(states=np.arange(iE, iE + n_sup), deadline=t_end)
            if not qp.converged:                       # unfinished solve at the escape deadline: discard it
                raise PlannerTimeout()
        except (np.linalg.LinAlgError, PlannerTimeout):
            self.escapes += 1                  # numerical failure: keep flying the previous plan
            if self.prev is None:
                return encode_action(env, naive_pace(env))
            return encode_action(env, _lookahead_target(self.prev[0], self.prev[1], x0, v0))
        v_plan = np.sqrt(2.0 * np.maximum(0.0, sol[iE:iE + n_sup]))
        self.prev = (s_sup, v_plan)
        v_tgt = _lookahead_target(s_sup, v_plan, x0, v0)
        a_start = 1.18 if time_mode else (_clip(float(sol[iP]), 0.74, 1.18) if v_tgt > v0 + 0.3 else 0.96)
        return encode_action(env, v_tgt, a_start, 0.78)


# ===========================================================================
# Tabular RL (13. CDQ, 14. Shield SARSA): shared discretisation and discrete speed-level actions
# ===========================================================================
TAB_DV = np.array([-0.5, 0.0, 0.6, 1.2, 1.8, 2.6, 3.4])     # v_cmd = v_naive + dv
TAB_COAST = len(TAB_DV)                                      # extra action: coast (v_cmd = v - 0.4)
N_TAB_ACT = len(TAB_DV) + 1
_V_RATIO_EDGES = np.array([0.30, 0.55, 0.75, 0.88, 0.97])
_DEV_EDGES = np.array([-1.5, -0.6, -0.2, 0.2, 0.6, 1.5])
N_TAB_STATES = 10 * (len(_V_RATIO_EDGES) + 1) * (len(_DEV_EDGES) + 1)


def tab_state(env) -> int:
    cor = env.corridor
    stage = min(9, int(10.0 * env.x_curr / max(1.0, cor.station_x_m)))
    vr = int(np.searchsorted(_V_RATIO_EDGES, env.v_meas / max(1.0, cor.get_speed_limit(env.x_curr))))
    dev = int(np.searchsorted(_DEV_EDGES, env.v_meas - naive_pace(env)))
    return (stage * (len(_V_RATIO_EDGES) + 1) + vr) * (len(_DEV_EDGES) + 1) + dev


def tab_action_target(env, a: int) -> float:
    if a == TAB_COAST:
        return max(1.5, env.v_meas - 0.4)
    return naive_pace(env) + float(TAB_DV[a])


class _TabularPilotBase:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.eps = 0.0

    def reset_leg(self, env) -> None:
        env.pilot_filter = None

    def shield(self, env, a: int) -> int:
        return a

    def greedy(self, s: int) -> int:
        raise NotImplementedError

    def choose(self, env, explore: bool) -> int:
        s = tab_state(env)
        a = int(self.rng.integers(N_TAB_ACT)) if (explore and self.rng.random() < self.eps) else self.greedy(s)
        return self.shield(env, a)

    def act(self, env, obs) -> np.ndarray:
        if env.shaper.in_landing:
            return encode_action(env, naive_pace(env))
        return encode_action(env, tab_action_target(env, self.choose(env, explore=False)))


class CDQPilot(_TabularPilotBase):
    """Chen et al. (2025): distributional (quantile) Q-learning with CVaR_alpha action selection."""
    name = "11. Chen (2025) CVaR Distributional Q (CDQ)"

    def __init__(self, n_q: int = 21, cvar_alpha: float = 0.25, seed: int = 0):
        super().__init__(seed)
        self.n_q = n_q
        self.alpha = cvar_alpha
        self.taus = (np.arange(n_q) + 0.5) / n_q
        self.Z = np.zeros((N_TAB_STATES, N_TAB_ACT, n_q))
        self.k_cvar = max(1, int(round(cvar_alpha * n_q)))

    def cvar(self, s: int) -> np.ndarray:
        return np.sort(self.Z[s], axis=1)[:, :self.k_cvar].mean(axis=1)

    def greedy(self, s: int) -> int:
        return int(np.argmax(self.cvar(s)))

    def update(self, s, a, r, s2, done, lr=0.08, gamma=0.99) -> None:
        if done:
            target = np.full(self.n_q, r)
        else:
            a2 = self.greedy(s2)
            target = r + gamma * self.Z[s2, a2]
        theta = self.Z[s, a]
        # Quantile-regression TD (Dabney et al. 2018), tabular form:
        ind = (target[None, :] < theta[:, None]).astype(float)       # [i, j]
        grad = (self.taus[:, None] - ind).mean(axis=1)
        self.Z[s, a] = theta + lr * grad

    def state_dict(self):
        return {"Z": self.Z.copy()}

    def load(self, d):
        self.Z = d["Z"].copy()


class ShieldSARSAPilot(_TabularPilotBase):
    """Zhao et al. (2023): on-policy SARSA with a hard overspeed shield (action replaced before execution)."""
    name = "12. Zhao (2023) Shield SARSA"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.Q = np.zeros((N_TAB_STATES, N_TAB_ACT))
        self.shield_count = 0
        self.decisions = 0

    def greedy(self, s: int) -> int:
        return int(np.argmax(self.Q[s]))

    def shield(self, env, a: int) -> int:
        self.decisions += 1
        cor = env.corridor
        x, v = env.x_curr, env.v_meas
        # Overspeed shield: the commanded speed must stay under the braking-aware ceiling over the next
        # decision interval (position reached at the higher of current and target speed):
        def safe(act):
            v_t = tab_action_target(env, act)
            x_ahead = min(cor.station_x_m, x + max(v, v_t) * 4.0)
            v_ceil = min(cor.compute_smooth_spatial_ceiling(x, 0.78), cor.compute_smooth_spatial_ceiling(x_ahead, 0.78))
            return v_t <= v_ceil - 0.15
        if safe(a):
            return a
        self.shield_count += 1
        order = sorted(range(N_TAB_ACT), key=lambda k: -tab_action_target(env, k))
        for k in order:
            if safe(k):
                return k
        return TAB_COAST

    def state_dict(self):
        return {"Q": self.Q.copy()}

    def load(self, d):
        self.Q = d["Q"].copy()


def train_tabular(pilot, emulator, n_journeys: int = N_TRAIN_JOURNEYS, seed: int = 11,
                  val_every: int = VAL_EVERY_JOURNEYS, journeys_seed: int = 42, log=print):
    from dynamic_railroad_mpc import MPCTransitionRailroadEnv
    from railroad_corridor import RailroadCorridor
    env = MPCTransitionRailroadEnv(corridor=RailroadCorridor(), traj_shield=None, emulator=emulator)
    val = validation_scenarios(journeys_seed)
    best, best_sd = float("inf"), pilot.state_dict()
    is_sarsa = isinstance(pilot, ShieldSARSAPilot)
    for j in range(n_journeys):
        pilot.eps = max(0.05, 0.6 * (1.0 - j / (0.7 * n_journeys)))
        lr = 0.15 * max(0.15, 1.0 - j / n_journeys)
        for leg in training_scenario(journeys_seed, j)["legs"]:
            obs = env.reset(corridor_override=leg["corridor"], wind_profile=leg["wind_profile"])
            s = tab_state(env)
            a = pilot.choose(env, explore=True)
            done = False
            while not done:
                act = encode_action(env, tab_action_target(env, a)) if not env.shaper.in_landing else encode_action(env, naive_pace(env))
                _, rew, done, _ = env.step(act)
                r = rew / 20.0
                s2 = tab_state(env)
                a2 = pilot.choose(env, explore=True) if not done else 0
                if is_sarsa:
                    target = r if done else r + 0.99 * pilot.Q[s2, a2]
                    pilot.Q[s, a] += lr * (target - pilot.Q[s, a])
                else:
                    pilot.update(s, a, r, s2, done, lr=lr)
                s, a = s2, a2
        if (j + 1) % val_every == 0 or j + 1 == n_journeys:
            vl = validation_loss(pilot, emulator, val)
            if vl <= best:
                best, best_sd = vl, copy.deepcopy(pilot.state_dict())
            log(f"  [{pilot.name}] journey {j + 1}/{n_journeys} val LossScore {vl:.4f} (best {best:.4f})")
    pilot.load(best_sd)
    return pilot, best


# ===========================================================================
# 15. Huang et al. (2025) EITOP: expert rules + PPO + DMTD coasting
# ===========================================================================
def eitop_expert_filter(env, v_cmd: float, a_start: float, b_land: float, v_feas: float):
    """Expert rule system wrapped around the PPO action (applied in training and evaluation)."""
    if env.k_elapsed < 8 or env.shaper.in_landing:
        return v_cmd, a_start, b_land
    v = env.v_meas
    # Rule 1 (traction phase): far below the required pace -> accelerate with strong traction.
    if v < 0.85 * v_feas:
        return max(v_cmd, v_feas), max(a_start, 1.10), b_land
    # Rule 2 (DMTD coasting): coasting is enabled when the minimal running time from here (max-speed
    # profile) plus the time lost by coasting still meets the deadline with a margin.
    t_rem = env.corridor.deadline_s - env.t_curr
    if v >= v_feas and (env.corridor.station_x_m - env.x_curr) > 300.0:
        v_coast = max(v_feas, v - 0.4)
        t_coast = env.estimate_remaining_time_at_speed(v_coast, a_start, b_land)
        if t_coast <= t_rem - 3.0:
            return min(v_cmd, v_coast), a_start, b_land
    return v_cmd, a_start, b_land


class PolicyPilot:
    """Wraps a CosineTrajectoryRLPolicy (or DDPG actor) as a literature pilot, optionally with a rule filter."""

    def __init__(self, name: str, policy, pilot_filter=None, shield_fn=None):
        self.name = name
        self.policy = policy
        self.pilot_filter = pilot_filter
        self.shield_fn = shield_fn
        self.shield_count = 0
        self.decisions = 0

    def reset_leg(self, env) -> None:
        env.pilot_filter = self.pilot_filter

    def act(self, env, obs) -> np.ndarray:
        a = self.policy.get_action_and_value(obs, deterministic=True)[0].copy()
        if self.shield_fn is not None:
            self.decisions += 1
            a2 = self.shield_fn(env, a)
            if np.any(np.abs(a2 - a) > 1e-6):
                self.shield_count += 1
            a = a2
        return a


# ===========================================================================
# DDPG (16. SRL-IETTO, 17. SSA-DRL)
# ===========================================================================
class DDPGActor(nn.Module):
    def __init__(self, obs_dim: int = 18, act_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, act_dim), nn.Tanh())

    def forward(self, o):
        return self.net(o)

    def get_action_and_value(self, obs, deterministic: bool = True):
        with torch.no_grad():
            return self.net(torch.as_tensor(obs, dtype=torch.float32)).numpy(), 0.0, 0.0


class DDPGCritic(nn.Module):
    def __init__(self, obs_dim: int = 18, act_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, o, a):
        return self.net(torch.cat([o, a], dim=-1)).squeeze(-1)


def artsom_action(env, obs) -> np.ndarray:
    """ARTSOM operation-mode allocation (Li et al. 2023): pre-computed mode sequence MT -> CR -> CO -> BR,
    expressed as the supervised target action: accelerate to the feasible pace, cruise, coast when the
    remaining-time margin allows it, and leave braking to the landing."""
    v_feas, _ = env.compute_feasible_deadline_pace(0.96, 0.78)
    v = env.v_meas
    if v < v_feas - 0.3:
        return encode_action(env, v_feas + 0.3, 1.10, 0.78)                     # MT
    t_rem = env.corridor.deadline_s - env.t_curr
    if (env.corridor.station_x_m - env.x_curr) > 400.0:
        t_coast = env.estimate_remaining_time_at_speed(max(v_feas, v - 0.4), 0.96, 0.78)
        if t_coast <= t_rem - 4.0:
            return encode_action(env, max(v_feas, v - 0.4), 0.96, 0.78)        # CO
    return encode_action(env, v_feas, 0.96, 0.78)                               # CR


def ssa_tree_shield(env, a: np.ndarray) -> np.ndarray:
    """SSA-DRL safety layer (Zhao et al. 2025): safety specification G(v <= ceiling) AND F(arrive on time),
    checked by a depth-2 searching tree over candidate actions around the DRL action on a point-mass model.
    The admissible candidate closest to the DRL action is executed."""
    cor = env.corridor
    v_naive = naive_pace(env)
    cands = [a.copy()] + [np.array([_clip(a[0] + d, -1, 1), a[1], a[2]], dtype=np.float32) for d in (-0.15, -0.3, 0.15, -0.6)]
    v_feas, _ = env.compute_feasible_deadline_pace(0.96, 0.78)

    def rollout_ok(c, depth: int) -> tuple[bool, float, float]:
        # Point-mass rollout of the executed pipeline: the decoded command is clamped to the braking-aware
        # ceiling at every step (as in decode_rl_action), then approached at the comfort rates.
        x, v = env.x_curr, env.v_meas
        v_t = _clip(v_naive + 3.8 * float(c[0]), 1.5, 17.8)
        a_up = 0.96 + 0.22 * float(c[1])
        worst = -1e9
        for _ in range(depth):
            for _k in range(8):
                _EC.N[0] += 1
                ceil_x = cor.compute_smooth_spatial_ceiling(min(x, cor.station_x_m), 0.78)
                acc = _clip((min(v_t, ceil_x) - v) / 2.0, -0.78, a_up)
                v = max(0.0, v + 0.5 * acc)
                x += 0.5 * v
                worst = max(worst, v - cor.compute_smooth_spatial_ceiling(min(x, cor.station_x_m), 0.78))
        # safety: G(v <= ceiling + tolerance); liveness: the pace after the tree still meets the deadline
        return worst <= 0.10, max(0.0, v_feas - v), worst

    best, best_d, least_bad, least_w = None, 1e9, cands[0], 1e9
    for c in cands:
        ok, lag, worst = rollout_ok(c, 2)
        if worst < least_w:
            least_bad, least_w = c, worst
        if not ok:
            continue
        d = float(np.abs(c[0] - a[0])) + 0.5 * lag
        if d < best_d:
            best, best_d = c, d
    return best if best is not None else least_bad


def train_ddpg(name: str, emulator, n_journeys: int = N_TRAIN_JOURNEYS, seed: int = 21, supervised: bool = False,
               shield_fn=None, binary_goal: bool = False, val_every: int = VAL_EVERY_JOURNEYS,
               journeys_seed: int = 42, log=print):
    from dynamic_railroad_mpc import MPCTransitionRailroadEnv
    from railroad_corridor import RailroadCorridor
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = MPCTransitionRailroadEnv(corridor=RailroadCorridor(), traj_shield=None, emulator=emulator)
    actor, critic = DDPGActor(), DDPGCritic()
    actor_t, critic_t = copy.deepcopy(actor), copy.deepcopy(critic)
    opt_a = optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = optim.Adam(critic.parameters(), lr=1e-3)
    buf_o, buf_a, buf_r, buf_o2, buf_d, buf_sup = [], [], [], [], [], []
    cap = 60000
    val = validation_scenarios(journeys_seed)
    pilot = PolicyPilot(name, actor, shield_fn=shield_fn)

    if supervised:
        # Step 1-2 of SRL-IETTO: behaviour-clone the ARTSOM operation-mode allocation (on the first 60 of the
        # same training journeys, so the journey budget is unchanged).
        X, Y = [], []
        for j in range(60):
            for leg in training_scenario(journeys_seed, j)["legs"]:
                obs = env.reset(corridor_override=leg["corridor"], wind_profile=leg["wind_profile"])
                done = False
                while not done:
                    a_sup = artsom_action(env, obs)
                    X.append(obs); Y.append(a_sup)
                    obs, _, done, _ = env.step(a_sup)
        X_t, Y_t = torch.as_tensor(np.array(X)), torch.as_tensor(np.array(Y))
        for _ in range(300):
            idx = torch.randint(0, len(X_t), (256,))
            loss = nn.functional.mse_loss(actor(X_t[idx]), Y_t[idx])
            opt_a.zero_grad(); loss.backward(); opt_a.step()
        actor_t.load_state_dict(actor.state_dict())

    best, best_sd = validation_loss(pilot, emulator, val), copy.deepcopy(actor.state_dict())
    log(f"  [{name}] init val LossScore {best:.4f}")
    for j in range(n_journeys):
        sig = 0.25 * max(0.1, 1.0 - j / n_journeys)
        for leg in training_scenario(journeys_seed, j)["legs"]:
            obs = env.reset(corridor_override=leg["corridor"], wind_profile=leg["wind_profile"])
            done = False
            while not done:
                a = actor.get_action_and_value(obs)[0] + rng.normal(0, sig, 3).astype(np.float32)
                a = np.clip(a, -1, 1).astype(np.float32)
                if shield_fn is not None:
                    a = shield_fn(env, a)
                a_sup = artsom_action(env, obs) if supervised else a
                obs2, rew, done, info = env.step(a)
                if binary_goal:
                    # Binary goal-directed reward (+1 on reaching the goal window) with an energy term:
                    r = 0.0
                    if done:
                        r = 1.0 if (abs(info["schedule_error_s"]) <= 2.0 and abs(info["stop_error_m"]) <= 0.3) else 0.0
                        r -= 0.02 * float(info["energy_kwh"])
                else:
                    r = rew / 20.0
                buf_o.append(obs); buf_a.append(a); buf_r.append(r); buf_o2.append(obs2); buf_d.append(float(done)); buf_sup.append(a_sup)
                if len(buf_o) > cap:
                    for b in (buf_o, buf_a, buf_r, buf_o2, buf_d, buf_sup):
                        del b[: len(b) - cap]
                obs = obs2
                if len(buf_o) >= 512:
                    idx = rng.integers(0, len(buf_o), 64)
                    o = torch.as_tensor(np.array([buf_o[i] for i in idx]))
                    aa = torch.as_tensor(np.array([buf_a[i] for i in idx]))
                    rr = torch.as_tensor(np.array([buf_r[i] for i in idx], dtype=np.float32))
                    o2 = torch.as_tensor(np.array([buf_o2[i] for i in idx]))
                    dd = torch.as_tensor(np.array([buf_d[i] for i in idx], dtype=np.float32))
                    with torch.no_grad():
                        y = rr + 0.99 * (1.0 - dd) * critic_t(o2, actor_t(o2))
                    lc = nn.functional.mse_loss(critic(o, aa), y)
                    opt_c.zero_grad(); lc.backward(); opt_c.step()
                    la = -critic(o, actor(o)).mean()
                    if supervised:
                        sup = torch.as_tensor(np.array([buf_sup[i] for i in idx]))
                        la = la + 2.0 * nn.functional.mse_loss(actor(o), sup)
                    opt_a.zero_grad(); la.backward(); opt_a.step()
                    with torch.no_grad():
                        for p, pt in zip(actor.parameters(), actor_t.parameters()):
                            pt.mul_(0.995).add_(0.005 * p)
                        for p, pt in zip(critic.parameters(), critic_t.parameters()):
                            pt.mul_(0.995).add_(0.005 * p)
        if (j + 1) % val_every == 0 or j + 1 == n_journeys:
            vl = validation_loss(pilot, emulator, val)
            if vl <= best:
                best, best_sd = vl, copy.deepcopy(actor.state_dict())
            log(f"  [{name}] journey {j + 1}/{n_journeys} val LossScore {vl:.4f} (best {best:.4f})")
    actor.load_state_dict(best_sd)
    return actor, best


# ===========================================================================
# Construction / loading for the benchmark
# ===========================================================================
LIT_METHOD_NAMES = [
    XiaoConvexSHMPCPilot.name,
    ZhongPseudospectralMPCPilot.name,
    CDQPilot.name,
    ShieldSARSAPilot.name,
    "13. Huang (2025) EITOP (Expert + PPO + DMTD)",
    "14. Li (2023) SRL-IETTO (ARTSOM + DDPG)",
    "15. Zhao (2025) SSA-DRL (Shield + Tree + DRL)",
]


def _ck(fname: str) -> str:
    """Checkpoint name for the training seed selected by LIT_CKPT_SUFFIX (e.g. "_r2"); empty = first seed."""
    suf = os.environ.get("LIT_CKPT_SUFFIX", "")
    return fname[:-3] + suf + ".pt" if suf else fname


def load_literature_pilots(emulator, train_if_missing: bool = True, log=print) -> dict:
    """Returns {method name: pilot factory}. RL baselines are loaded from checkpoints/lit_*.pt (trained once
    with the same journey budget and validation protocol as the PPO pilots if missing)."""
    from dynamic_railroad_mpc import CosineTrajectoryRLPolicy, train_pure_ppo_cosine_pilot
    from railroad_corridor import RailroadCorridor
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    out[XiaoConvexSHMPCPilot.name] = lambda: XiaoConvexSHMPCPilot()
    out[ZhongPseudospectralMPCPilot.name] = lambda: ZhongPseudospectralMPCPilot()

    for cls, fname in ((CDQPilot, _ck("lit_cdq.pt")), (ShieldSARSAPilot, _ck("lit_shield_sarsa.pt"))):
        p = CKPT_DIR / fname
        pil = cls()
        if p.exists():
            pil.load({k: np.asarray(v) for k, v in torch.load(str(p), map_location="cpu", weights_only=False).items()})
        elif train_if_missing:
            pil, _ = train_tabular(pil, emulator, log=log)
            torch.save(pil.state_dict(), str(p))
        out[cls.name] = (lambda pil=pil: pil)

    p = CKPT_DIR / _ck("best_lit_eitop.pt")
    eit = CosineTrajectoryRLPolicy(18, 3, 64)
    if p.exists():
        eit.load_state_dict(torch.load(str(p), map_location="cpu"))
    elif train_if_missing:
        eit, _ = train_pure_ppo_cosine_pilot(RailroadCorridor(), emulator=emulator, epochs=400, wind_mode="mixed",
                                             seed=42, checkpoint_prefix="lit_eitop", pilot_filter=eitop_expert_filter)
    eit.eval()
    out[LIT_METHOD_NAMES[4]] = lambda: PolicyPilot(LIT_METHOD_NAMES[4], eit, pilot_filter=eitop_expert_filter)

    for nm, fname, kw in (
        (LIT_METHOD_NAMES[5], _ck("lit_srl_ietto.pt"), dict(supervised=True, binary_goal=True)),
        (LIT_METHOD_NAMES[6], _ck("lit_ssa_drl.pt"), dict(shield_fn=ssa_tree_shield)),
    ):
        p = CKPT_DIR / fname
        act = DDPGActor()
        if p.exists():
            act.load_state_dict(torch.load(str(p), map_location="cpu"))
        elif train_if_missing:
            act, _ = train_ddpg(nm, emulator, log=log, **kw)
            torch.save(act.state_dict(), str(p))
        act.eval()
        out[nm] = (lambda act=act, nm=nm, kw=kw: PolicyPilot(nm, act, shield_fn=kw.get("shield_fn")))
    return out


# ===========================================================================
# Extra literature baselines (added 2026-11-20 after a literature check; not part of the paper benchmark yet)
#   16. Cheng et al. (2024)  S-PPO: PPO + safe action rechoosing + relaxed dynamic reward.
#   17. Wang et al. (2024)   ES-MEDRL: maximum-entropy DRL (SAC) with Lagrange-multiplier constraint penalties
#                            and a second energy-refinement stage on the speed setpoint.
#   18. Tan et al. (2025)    GA-DDPG: DDPG whose exploration noise is tuned by a genetic algorithm, with an
#                            expert reference system that keeps operation safe and guides training.
# ===========================================================================
def _pm_worst_ceiling_margin(env, v_t: float, a_up: float, n_steps: int = 8) -> float:
    """Point-mass rollout of one macro step toward setpoint v_t at the comfort rates; returns the worst
    (speed - braking-aware ceiling) over the rollout (> 0 means overspeed)."""
    cor = env.corridor
    x, v = env.x_curr, env.v_meas
    worst = -1e9
    for _ in range(n_steps):
        _EC.N[0] += 1
        acc = _clip((v_t - v) / 2.0, -0.78, a_up)
        v = max(0.0, v + 0.5 * acc)
        x += 0.5 * v
        worst = max(worst, v - cor.compute_smooth_spatial_ceiling(min(x, cor.station_x_m), 0.78))
    return worst


def sppo_safe_rechoose(env, v_cmd: float, a_start: float, b_land: float, v_feas: float):
    """S-PPO safe action rechoosing (training and evaluation): an action is safe if the next macro step keeps
    the speed under the speed-limit/braking curve and the train moving; otherwise the largest safe setpoint
    is re-chosen (bisection)."""
    if env.shaper.in_landing:
        return v_cmd, a_start, b_land
    if _pm_worst_ceiling_margin(env, v_cmd, a_start) <= 0.0:
        return v_cmd, a_start, b_land
    lo, hi = 1.5, v_cmd
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        if _pm_worst_ceiling_margin(env, mid, a_start) <= 0.0:
            lo = mid
        else:
            hi = mid
    return lo, a_start, b_land


class SPPOReward:
    """Relaxed dynamic reward of S-PPO: a dense term that trades energy against running time with a weight set
    by the step's initial speed, plus a sparse terminal reward whose trip-time tolerance is relaxed early in
    training (10 s) and tightened to 2 s, so feasible samples are found early."""

    def __init__(self):
        self._key = None
        self._E = 0.0
        self._v0 = 0.0

    def __call__(self, env, rew, done, info) -> float:
        key = id(env.corridor)
        if key != self._key:
            self._key, self._E, self._v0 = key, 0.0, 0.0
        cor = env.corridor
        dE = float(info["energy_kwh"]) - self._E
        self._E = float(info["energy_kwh"])
        c = _clip(self._v0 / 16.0, 0.2, 0.8)              # faster -> energy weighs more
        self._v0 = float(info["v_curr"])
        dur = max(1.0, cor.deadline_s - cor.start_time_s)
        late = (info["t_curr"] - cor.start_time_s) - (info["x_curr"] / max(1.0, cor.station_x_m)) * dur
        r = -(c * dE / 0.3 + (1.0 - c) * abs(late) / 20.0) * 0.1
        if done:
            tau = 10.0 - 8.0 * float(getattr(env, "train_progress", 1.0))
            err = abs(float(info["schedule_error_s"]))
            r += 1.0 if err <= tau else -0.5 * err / tau
            se = abs(float(info["stop_error_m"]))
            r += 0.5 * float(np.exp(-3.0 * se)) - 1.0 * se
            self._key = None
        return r


# ---------------- ES-MEDRL (SAC + Lagrangian penalties + energy refinement) ----------------
class SACActor(nn.Module):
    def __init__(self, obs_dim: int = 18, act_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, o):
        h = self.body(o)
        return self.mu(h), torch.clamp(self.log_std(h), -5.0, 1.0)

    def sample(self, o):
        mu, ls = self(o)
        std = ls.exp()
        eps = torch.randn_like(mu)
        u = mu + std * eps
        a = torch.tanh(u)
        logp = (-0.5 * eps.pow(2) - ls - 0.5 * math.log(2 * math.pi)).sum(-1) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp

    def get_action_and_value(self, obs, deterministic: bool = True):
        with torch.no_grad():
            mu, _ = self(torch.as_tensor(obs, dtype=torch.float32))
            return torch.tanh(mu).numpy(), 0.0, 0.0


def esmedrl_refine(env, v_cmd: float, a_start: float, b_land: float, v_feas: float):
    """ES-MEDRL second stage (speed optimisation for energy): lower the cruise setpoint in 0.2 m/s steps
    (up to 0.6 m/s) while the predicted arrival still meets the deadline with a 2 s margin."""
    if env.shaper.in_landing or env.k_elapsed < 8:
        return v_cmd, a_start, b_land
    t_rem = env.corridor.deadline_s - env.t_curr
    v = v_cmd
    for _ in range(3):
        v_try = v - 0.2
        if v_try < max(1.5, v_feas):
            break
        if env.estimate_remaining_time_at_speed(v_try, a_start, b_land) <= t_rem - 2.0:
            v = v_try
        else:
            break
    return v, a_start, b_land


def train_esmedrl(name: str, emulator, n_journeys: int = N_TRAIN_JOURNEYS, seed: int = 31,
                  val_every: int = VAL_EVERY_JOURNEYS, log=print, lam_max: float = float("inf"), journeys_seed: int = 42):
    """Maximum-entropy actor-critic (SAC, automatic temperature) on the LossScore reward, with Lagrange
    multipliers on the speed, acceleration and jerk violations (dual ascent towards zero violations).
    lam_max caps the multipliers (default: uncapped, as in the method)."""
    from dynamic_railroad_mpc import MPCTransitionRailroadEnv
    from railroad_corridor import RailroadCorridor
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = MPCTransitionRailroadEnv(corridor=RailroadCorridor(), traj_shield=None, emulator=emulator)
    env.pilot_filter = esmedrl_refine
    actor = SACActor()
    q1, q2 = DDPGCritic(), DDPGCritic()
    q1_t, q2_t = copy.deepcopy(q1), copy.deepcopy(q2)
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_a = optim.Adam(actor.parameters(), lr=3e-4)
    opt_q = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=1e-3)
    opt_al = optim.Adam([log_alpha], lr=3e-4)
    target_ent = -3.0
    lam = np.zeros(3)                                     # Lagrange multipliers: speed, accel, jerk
    buf = []
    cap = 60000
    val = validation_scenarios(journeys_seed)
    pilot = PolicyPilot(name, actor, pilot_filter=esmedrl_refine)
    best, best_sd = validation_loss(pilot, emulator, val), copy.deepcopy(actor.state_dict())
    log(f"  [{name}] init val LossScore {best:.4f}")
    for j in range(n_journeys):
        viol_sum, n_steps = np.zeros(3), 0
        legs_j = training_scenario(journeys_seed, j)["legs"]
        for leg in legs_j:
            obs = env.reset(corridor_override=leg["corridor"], wind_profile=leg["wind_profile"])
            done = False
            while not done:
                with torch.no_grad():
                    a = actor.sample(torch.as_tensor(obs, dtype=torch.float32))[0].numpy().astype(np.float32)
                obs2, rew, done, info = env.step(a, record_micro=True)
                ml = info["micro_logs"]
                # Violation indicators with the thresholds of the constrained learners (App. masa, pilot_masa.py):
                v = np.array([
                    float(any(m["rho_1"] > 1.02 for m in ml)),
                    float(any(m["rho_2"] > 1.02 for m in ml)),
                    float(any(m["rho_3"] > 1.05 for m in ml)),
                ])
                viol_sum += v; n_steps += 1
                r = rew / 20.0 - float(lam @ v)
                buf.append((obs, a, r, obs2, float(done)))
                if len(buf) > cap:
                    del buf[: len(buf) - cap]
                obs = obs2
                if len(buf) >= 512:
                    idx = rng.integers(0, len(buf), 64)
                    o = torch.as_tensor(np.array([buf[i][0] for i in idx]))
                    aa = torch.as_tensor(np.array([buf[i][1] for i in idx]))
                    rr = torch.as_tensor(np.array([buf[i][2] for i in idx], dtype=np.float32))
                    o2 = torch.as_tensor(np.array([buf[i][3] for i in idx]))
                    dd = torch.as_tensor(np.array([buf[i][4] for i in idx], dtype=np.float32))
                    alpha = log_alpha.exp().detach()
                    with torch.no_grad():
                        a2, lp2 = actor.sample(o2)
                        y = rr + 0.99 * (1 - dd) * (torch.min(q1_t(o2, a2), q2_t(o2, a2)) - alpha * lp2)
                    lq = nn.functional.mse_loss(q1(o, aa), y) + nn.functional.mse_loss(q2(o, aa), y)
                    opt_q.zero_grad(); lq.backward(); opt_q.step()
                    an, lp = actor.sample(o)
                    la = (alpha * lp - torch.min(q1(o, an), q2(o, an))).mean()
                    opt_a.zero_grad(); la.backward(); opt_a.step()
                    lal = -(log_alpha * (lp.detach() + target_ent)).mean()
                    opt_al.zero_grad(); lal.backward(); opt_al.step()
                    with torch.no_grad():
                        for net, nt in ((q1, q1_t), (q2, q2_t)):
                            for p, pt in zip(net.parameters(), nt.parameters()):
                                pt.mul_(0.995).add_(0.005 * p)
        # Dual ascent on the multipliers (constraint budget as in the constrained learners of App. masa).
        lam = np.maximum(0.0, lam + 0.05 * (viol_sum / max(1, len(legs_j)) - 1.0 / 3.0))   # budget: 1 violating step per leg
        lam = np.minimum(lam, lam_max)
        if (j + 1) % val_every == 0 or j + 1 == n_journeys:
            vl = validation_loss(pilot, emulator, val)
            if vl <= best:
                best, best_sd = vl, copy.deepcopy(actor.state_dict())
            log(f"  [{name}] journey {j + 1}/{n_journeys} val LossScore {vl:.4f} (best {best:.4f}) lambda {np.round(lam, 3)}")
    actor.load_state_dict(best_sd)
    return actor, best


# ---------------- GA-DDPG (Tan et al. 2025) ----------------
def tan_expert_rules(env, v_cmd: float, a_start: float, b_land: float, v_feas: float):
    """Expert reference system (safety rules, training and evaluation): keep a 0.3 m/s margin under the
    braking-aware ceiling ahead, and never command less than the feasible pace when clearly behind."""
    if env.shaper.in_landing:
        return v_cmd, a_start, b_land
    cor = env.corridor
    x_ahead = min(cor.station_x_m, env.x_curr + max(env.v_meas, v_cmd) * 4.0)
    v_ceil = min(cor.compute_smooth_spatial_ceiling(env.x_curr, 0.78), cor.compute_smooth_spatial_ceiling(x_ahead, 0.78))
    if env.v_meas < 0.9 * v_feas:
        v_cmd = max(v_cmd, v_feas)
    return min(v_cmd, v_ceil - 0.3), a_start, b_land


def tan_expert_action(env) -> np.ndarray:
    """Expert reference action used to guide exploration: hold slightly above the feasible pace."""
    v_feas, _ = env.compute_feasible_deadline_pace(0.96, 0.78)
    return encode_action(env, v_feas + 0.4, 0.96, 0.78)


def train_ga_ddpg(name: str, emulator, n_journeys: int = N_TRAIN_JOURNEYS, seed: int = 41,
                  val_every: int = VAL_EVERY_JOURNEYS, pop: int = 4, gen_len: int = 25, journeys_seed: int = 42,
                  log=print):
    from dynamic_railroad_mpc import MPCTransitionRailroadEnv
    from railroad_corridor import RailroadCorridor
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = MPCTransitionRailroadEnv(corridor=RailroadCorridor(), traj_shield=None, emulator=emulator)
    env.pilot_filter = tan_expert_rules
    actor, critic = DDPGActor(), DDPGCritic()
    actor_t, critic_t = copy.deepcopy(actor), copy.deepcopy(critic)
    opt_a = optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = optim.Adam(critic.parameters(), lr=1e-3)
    buf = []
    cap = 60000
    val = validation_scenarios(journeys_seed)
    pilot = PolicyPilot(name, actor, pilot_filter=tan_expert_rules)
    # Genetic algorithm over the exploration-noise scale: each individual explores for gen_len // pop journeys,
    # fitness = mean training return; selection keeps the best half, mutation multiplies by lognormal noise.
    sigmas = list(rng.uniform(0.08, 0.40, pop))
    fitness = [0.0] * pop
    best, best_sd = validation_loss(pilot, emulator, val), copy.deepcopy(actor.state_dict())
    log(f"  [{name}] init val LossScore {best:.4f}")
    for j in range(n_journeys):
        ind = (j % gen_len) * pop // gen_len
        sig = sigmas[ind]
        p_guide = 0.5 * max(0.0, 1.0 - 2.0 * j / n_journeys)   # expert guidance, annealed to 0 by mid-training
        ret = 0.0
        for leg in training_scenario(journeys_seed, j)["legs"]:
            obs = env.reset(corridor_override=leg["corridor"], wind_profile=leg["wind_profile"])
            done = False
            while not done:
                if rng.random() < p_guide:
                    a = tan_expert_action(env)
                else:
                    a = actor.get_action_and_value(obs)[0] + rng.normal(0, sig, 3).astype(np.float32)
                a = np.clip(a, -1, 1).astype(np.float32)
                obs2, rew, done, info = env.step(a)
                r = rew / 20.0
                ret += r
                buf.append((obs, a, r, obs2, float(done)))
                if len(buf) > cap:
                    del buf[: len(buf) - cap]
                obs = obs2
                if len(buf) >= 512:
                    idx = rng.integers(0, len(buf), 64)
                    o = torch.as_tensor(np.array([buf[i][0] for i in idx]))
                    aa = torch.as_tensor(np.array([buf[i][1] for i in idx]))
                    rr = torch.as_tensor(np.array([buf[i][2] for i in idx], dtype=np.float32))
                    o2 = torch.as_tensor(np.array([buf[i][3] for i in idx]))
                    dd = torch.as_tensor(np.array([buf[i][4] for i in idx], dtype=np.float32))
                    with torch.no_grad():
                        y = rr + 0.99 * (1.0 - dd) * critic_t(o2, actor_t(o2))
                    lc = nn.functional.mse_loss(critic(o, aa), y)
                    opt_c.zero_grad(); lc.backward(); opt_c.step()
                    la = -critic(o, actor(o)).mean()
                    opt_a.zero_grad(); la.backward(); opt_a.step()
                    with torch.no_grad():
                        for p, pt in zip(actor.parameters(), actor_t.parameters()):
                            pt.mul_(0.995).add_(0.005 * p)
                        for p, pt in zip(critic.parameters(), critic_t.parameters()):
                            pt.mul_(0.995).add_(0.005 * p)
        fitness[ind] += ret
        if (j + 1) % gen_len == 0:
            order = np.argsort(fitness)[::-1]
            elite = [sigmas[i] for i in order[: pop // 2]]
            sigmas = elite + [float(_clip(s * np.exp(rng.normal(0, 0.3)), 0.03, 0.5)) for s in elite]
            fitness = [0.0] * pop
        if (j + 1) % val_every == 0 or j + 1 == n_journeys:
            vl = validation_loss(pilot, emulator, val)
            if vl <= best:
                best, best_sd = vl, copy.deepcopy(actor.state_dict())
            log(f"  [{name}] journey {j + 1}/{n_journeys} val LossScore {vl:.4f} (best {best:.4f}) sigmas {np.round(sigmas, 3)}")
    actor.load_state_dict(best_sd)
    return actor, best


EXTRA_LIT_NAMES = [
    "16. Cheng (2024) S-PPO (Safe Rechoosing + Relaxed Reward)",
    "17. Wang (2024) ES-MEDRL (Max-Entropy DRL + Lagrange)",
    "18. Tan (2025) GA-DDPG (Expert Rules + GA Noise)",
]


def load_extra_literature_pilots(emulator, train_if_missing: bool = True, log=print) -> dict:
    from dynamic_railroad_mpc import CosineTrajectoryRLPolicy, train_pure_ppo_cosine_pilot
    from railroad_corridor import RailroadCorridor
    out = {}
    p = CKPT_DIR / _ck("best_lit_sppo.pt")
    sp = CosineTrajectoryRLPolicy(18, 3, 64)
    if p.exists():
        sp.load_state_dict(torch.load(str(p), map_location="cpu"))
    elif train_if_missing:
        sp, _ = train_pure_ppo_cosine_pilot(RailroadCorridor(), emulator=emulator, epochs=400, wind_mode="mixed",
                                            seed=42, checkpoint_prefix="lit_sppo", pilot_filter=sppo_safe_rechoose,
                                            reward_fn=SPPOReward())
    sp.eval()
    out[EXTRA_LIT_NAMES[0]] = lambda: PolicyPilot(EXTRA_LIT_NAMES[0], sp, pilot_filter=sppo_safe_rechoose)
    for nm, fname, cls, trainer, filt in (
        (EXTRA_LIT_NAMES[1], _ck("lit_esmedrl.pt"), SACActor, train_esmedrl, esmedrl_refine),
        (EXTRA_LIT_NAMES[2], _ck("lit_ga_ddpg.pt"), DDPGActor, train_ga_ddpg, tan_expert_rules),
    ):
        p = CKPT_DIR / fname
        act = cls()
        if p.exists():
            act.load_state_dict(torch.load(str(p), map_location="cpu"))
        elif train_if_missing:
            act, _ = trainer(nm, emulator, log=log)
            torch.save(act.state_dict(), str(p))
        act.eval()
        out[nm] = (lambda act=act, nm=nm, filt=filt: PolicyPilot(nm, act, pilot_filter=filt))
    return out
