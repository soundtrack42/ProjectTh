"""
Station-to-station railway leg as a MASA tabular environment, built from the same plant equations as
`dynamic_railroad_mpc.py` (Davis drag, periodic grade/curve profile, mass-scaled traction and braking
limits with the 14 m/s power knee, adhesion, wind through relative air speed).

State: (position cell of 25 m, speed bin of 1 m/s) on a 2.4 km leg, plus two absorbing states GOAL
(stopped at the platform) and OVERRUN (passed the stop mark). Actions: notch u in {-1, -0.5, 0, 0.5, 1}
held for a 2 s macro step (four 0.5 s plant substeps).

Two dynamics modes share the same abstraction (transition matrix) that the shields use:
  * mode="model":   the next state is sampled from the abstraction, so shield guarantees hold exactly;
  * mode="physics": the continuous plant is integrated (per-episode mass, adhesion, sustained wind plus
                    gust noise) and the agent observes the quantised cell. This measures how much of
                    the guarantee survives the abstraction gap.

Safety: "unsafe" = overspeed (v > limit + 0.5 m/s, cell-conservative) or OVERRUN. The label of a cell
uses the lowest limit over the cell; true continuous violations are counted separately in physics mode.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from gymnasium import spaces

from masa.common.ltl import DFA, Atom, Neg
from masa.envs.tabular.base import TabularEnv

# ---- Leg geometry -------------------------------------------------------------------------------
LEG_M = 2400.0
DX = 25.0
DV = 1.0
NX = int(LEG_M / DX)                 # 96 position cells
NV = 19                              # speed bins [0,1), ..., [18,19)
GOAL = NX * NV
OVERRUN = GOAL + 1
N_STATES = OVERRUN + 1
ZONES = ((0.0, 500.0, 14.5), (500.0, 1300.0, 17.5), (1300.0, 1800.0, 12.5), (1800.0, 1e9, 14.5))
OVERSPEED_TOL = 0.5
STOP_TOL_M = 5.0                     # overrun beyond the stop mark
GOAL_ZONE_M = 25.0                   # stopping window before the mark
ACTIONS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
N_ACTIONS = len(ACTIONS)
DT_MACRO, N_SUB = 2.0, 4

# ---- Plant (same constants as the benchmark plant) ---------------------------------------------
GAMMA_ROT = 0.08
DAVIS_A, DAVIS_B, DAVIS_C = 14.5, 185.0, 1450.0
A_TRAC_BASE, B_BRAKE_BASE = 1.85, 1.65
M_LO, M_HI = 77_000.0, 98_000.0
MU_LO, MU_HI = 0.80, 1.00
W_MAX = 30.0                         # wind speed cap (m/s), Beaufort 10 gusts
GRADE_SEG = ((300.0, 470.0, 9.81 * 0.02), (620.0, 760.0, -9.81 * 0.012))
CURVE_SEG = ((180.0, 280.0, 9.81 / 650.0), (500.0, 600.0, 9.81 / 550.0))


def speed_limit(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, ZONES[-1][2])
    for a, b, v in ZONES[::-1]:
        out = np.where((x >= a) & (x < b), v, out)
    return out


def _profile(x, segs):
    xm = np.mod(x, 1000.0)
    out = np.zeros_like(x)
    for a, b, val in segs:
        out = out + np.where((xm >= a) & (xm < b), val, 0.0)
    return out


def plant_step(x, v, u, m, mu, w_head, dt=DT_MACRO / N_SUB):
    """One 0.5 s substep, vectorised. Returns (x', v', traction energy kWh, max overspeed margin)."""
    m_eff = m * (1.0 + GAMMA_ROT)
    a_trac = A_TRAC_BASE * (85_000.0 / m) * mu * np.minimum(1.0, 14.0 / np.maximum(v, 1e-3))
    a_brk = B_BRAKE_BASE * (85_000.0 / m) * mu
    f_cmd = np.where(u >= 0.0, u * m_eff * a_trac, u * m_eff * a_brk)
    v_rel = v + w_head
    f_aero = DAVIS_A * v_rel * np.abs(v_rel)             # relative-air-speed drag (includes wind)
    f_mech = np.where(v > 1e-4, DAVIS_B * v + DAVIS_C, 0.0)
    f_track = m * (_profile(x, GRADE_SEG) + _profile(x, CURVE_SEG))
    a = (f_cmd - f_aero * (v > 1e-4) - f_mech - f_track) / m_eff
    v_new = np.maximum(0.0, v + a * dt)
    x_new = x + 0.5 * (v + v_new) * dt
    e_kwh = np.maximum(f_cmd, 0.0) * 0.5 * (v + v_new) * dt / 3.6e6
    return x_new, v_new, e_kwh


def encode(x, v):
    """Continuous (x, v) -> abstract state index (vectorised)."""
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ix = np.clip((x // DX).astype(np.int64), 0, NX - 1)
    iv = np.clip((v // DV).astype(np.int64), 0, NV - 1)
    s = ix * NV + iv
    s = np.where((x >= LEG_M - GOAL_ZONE_M) & (v < 0.5) & (x < LEG_M + STOP_TOL_M), GOAL, s)
    s = np.where(x >= LEG_M + STOP_TOL_M, OVERRUN, s)
    return s


def _build_labels() -> dict[int, set]:
    labels: dict[int, set] = defaultdict(set)
    for ix in range(NX):
        x_lo, x_hi = ix * DX, (ix + 1) * DX
        lim = float(min(speed_limit(np.linspace(x_lo, x_hi - 1e-6, 6))))
        for iv in range(NV):
            if iv * DV >= lim + OVERSPEED_TOL - 1e-9:      # cell lies entirely above limit + tol
                labels[ix * NV + iv] |= {"unsafe", "overspeed"}
    labels[GOAL] = {"target"}
    labels[OVERRUN] = {"unsafe", "overrun"}
    return labels


LABELS = _build_labels()
label_fn = lambda obs: LABELS[int(obs)]
cost_fn = lambda labels: 1.0 if "unsafe" in labels else 0.0


def make_dfa() -> DFA:
    """G !unsafe: accepting state 1 = violation."""
    dfa = DFA([0, 1], 0, [1])
    dfa.add_edge(0, 0, Neg(Atom("unsafe")))
    dfa.add_edge(0, 1, Atom("unsafe"))
    dfa.add_edge(1, 1, Atom("unsafe"))
    dfa.add_edge(1, 1, Neg(Atom("unsafe")))
    return dfa


def _sample_wind(rng, n):
    """Head (+) / tail (-) wind: 70% moderate |w|<=12 m/s, 30% storm |w|<=30 m/s."""
    storm = rng.random(n) < 0.30
    return np.where(storm, rng.uniform(-W_MAX, W_MAX, n), rng.uniform(-12.0, 12.0, n))


_CACHE: dict[int, np.ndarray] = {}


def build_transition_matrix(n_random: int = 192, seed: int = 0) -> np.ndarray:
    """
    T[s', s, a] estimated from the plant: n_random Monte-Carlo samples per (cell, action) give the
    probabilities; 64 worst-case corner samples (cell corners x mass x adhesion x +-30 m/s wind) are
    added with weight 1e-4 each so the support over-approximates every reachable successor.
    """
    if seed in _CACHE:
        return _CACHE[seed]
    rng = np.random.default_rng(seed)
    T = np.zeros((N_STATES, N_STATES, N_ACTIONS))
    cells = np.arange(NX * NV)
    ix, iv = cells // NV, cells % NV
    corner = np.array(np.meshgrid([0.0, 1.0 - 1e-6], [0.0, 1.0 - 1e-6], [M_LO, M_HI], [MU_LO, MU_HI],
                                  [-W_MAX, W_MAX], [0, 1], indexing="ij")).reshape(6, -1)[:, :64]
    for a_i, u in enumerate(ACTIONS):
        for kind, n in (("rand", n_random), ("corner", corner.shape[1])):
            if kind == "rand":
                fx, fv = rng.random((2, len(cells), n))
                m = rng.uniform(M_LO, M_HI, (len(cells), n))
                mu = rng.uniform(MU_LO, MU_HI, (len(cells), n))
                w = _sample_wind(rng, len(cells) * n).reshape(len(cells), n)
                wgt = 1.0 / n
            else:
                fx = np.broadcast_to(corner[0], (len(cells), n))
                fv = np.broadcast_to(corner[1], (len(cells), n))
                m = np.broadcast_to(corner[2], (len(cells), n))
                mu = np.broadcast_to(corner[3], (len(cells), n))
                w = np.broadcast_to(corner[4], (len(cells), n))
                wgt = 1e-4
            x = (ix[:, None] + fx) * DX
            v = (iv[:, None] + fv) * DV
            for _ in range(N_SUB):
                x, v, _ = plant_step(x, v, u, m, mu, w)
            nxt = encode(x, v)
            np.add.at(T, (nxt, np.broadcast_to(cells[:, None], nxt.shape), a_i), wgt)
    T[GOAL, GOAL, :] = 1.0
    T[OVERRUN, OVERRUN, :] = 1.0
    T /= T.sum(axis=0, keepdims=True)
    _CACHE[seed] = T
    return T


class RailLeg(TabularEnv):
    """MASA tabular env (see module docstring). Episode stats are kept on the unwrapped env."""

    metadata = {"render_modes": []}

    def __init__(self, mode: str = "model", energy_w: float = 0.02, render_mode=None):
        super().__init__()
        assert mode in ("model", "physics")
        self.mode = mode
        self.energy_w = float(energy_w)
        self.observation_space = spaces.Discrete(N_STATES)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._transition_matrix = build_transition_matrix()
        self.np_random = None
        self.stats = defaultdict(float)     # cumulative over the env's lifetime
        self._ep: dict[str, float] = {}

    # -- episode bookkeeping --------------------------------------------------------------------
    def _close_episode(self):
        if not self._ep or self._ep.get("closed"):
            return
        e = self._ep
        self.stats["episodes"] += 1
        self.stats["goal"] += float(e["goal"])
        self.stats["overrun"] += float(e["overrun"])
        self.stats["unsafe_episodes"] += float(e["unsafe"] > 0)
        self.stats["true_overspeed_episodes"] += float(e["true_overspeed"] > 0)
        self.stats["steps"] += e["steps"]
        self.stats["energy_kwh"] += e["energy"]
        if e["goal"]:
            self.stats["goal_time_s"] += e["steps"] * DT_MACRO
        e["closed"] = True

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None or self.np_random is None:
            self.np_random = np.random.default_rng(seed)
        self._close_episode()
        self._x, self._v = 0.0, 0.0
        self._state = int(encode(self._x, self._v))
        rng = self.np_random
        self._m = rng.uniform(M_LO, M_HI)
        self._mu = rng.uniform(MU_LO, MU_HI)
        self._w_sus = float(_sample_wind(rng, 1)[0])
        self._ep = dict(goal=False, overrun=False, unsafe=0, true_overspeed=0, steps=0, energy=0.0, closed=False)
        return self._state, {}

    def step(self, action):
        a = int(action)
        u = float(ACTIONS[a])
        e_kwh = 0.0
        true_os = False
        if self.mode == "model":
            p = self._transition_matrix[:, self._state, a]
            s_next = int(self.np_random.choice(N_STATES, p=p))
            # energy proxy from the cell centre (model mode has no continuous state)
            if s_next < GOAL:
                v_c = (s_next % NV + 0.5) * DV
                e_kwh = max(u, 0.0) * 1.08 * 87_500.0 * A_TRAC_BASE * min(1.0, 14.0 / max(v_c, 1e-3)) * v_c * DT_MACRO / 3.6e6
        else:
            x, v = self._x, self._v
            for _ in range(N_SUB):
                w = float(np.clip(self._w_sus + self.np_random.normal(0.0, 3.0), -W_MAX, W_MAX))
                x, v, de = plant_step(np.float64(x), np.float64(v), u, self._m, self._mu, w)
                x, v = float(x), float(v)
                e_kwh += float(de)
                if v > float(speed_limit(x)) + OVERSPEED_TOL:
                    true_os = True
            self._x, self._v = x, v
            s_next = int(encode(x, v))
        self._state = s_next
        ep = self._ep
        ep["steps"] += 1
        ep["energy"] += e_kwh
        ep["unsafe"] += int("unsafe" in LABELS[s_next])
        ep["true_overspeed"] += int(true_os)
        terminated = s_next in (GOAL, OVERRUN)
        ep["goal"] = s_next == GOAL
        ep["overrun"] = s_next == OVERRUN
        reward = -0.05 - self.energy_w * e_kwh + (10.0 if s_next == GOAL else 0.0)
        if terminated:
            self._close_episode()
        return s_next, reward, terminated, False, {}

    def close_stats(self):
        self._close_episode()
        return dict(self.stats)
