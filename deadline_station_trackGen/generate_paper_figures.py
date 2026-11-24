"""
Generate publication-grade diagrams and empirical multi-panel paper figures inside
`deadline_station_trackGen/figures/`:
  1. `pipeline_stages.png`                  - Concise stage-level system pipeline diagram
  2. `train_physical_forces.png`            - Free-body physical force diagram of train + weather
  3. `contract_searching_tree.png`          - Safe Contract Searching Tree with tube pruning & UCB
  4. `testing_disturbance_trajectories.png` - Multi-station trajectories across Baseline vs Extreme disturbances
  5. `appendix_full_benchmark.png`          - Full-page 9-panel comparison across all 18 methods
  6. `appendix_disturbance_vs_clean_trajs.png` - Clean vs. Heavy-Disturbance trajectory comparison (many methods)
"""
from __future__ import annotations

import copy
import csv
import re
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
for _p in (str(SCRIPT_DIR), str(ROOT_DIR), str(ROOT_DIR / "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle
import numpy as np
import torch

from havaei_generalizing_rl import LearnedLinearizationModel
from railroad_corridor import generate_multi_station_scenario
from real_world_rail_weather import fetch_real_world_multi_station_scenario
from dynamic_railroad_mpc import CosineTrajectoryRLPolicy, evaluate_multi_station_journey


def _card(
    ax, x: float, y: float, w: float, h: float, title: str, lines: list[str],
    fc: str, ec: str, hc: str, tc: str = "#FFFFFF", ft: float = 9.5, fb: float = 8.0,
    line_step: float | None = None,
):
    outer = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.14",
        facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=2,
    )
    ax.add_patch(outer)
    hh = min(0.46, 0.34 * h)
    hdr = FancyBboxPatch(
        (x + 0.02, y + h - hh - 0.01), w - 0.04, hh,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=hc, edgecolor="none", zorder=3,
    )
    ax.add_patch(hdr)
    ax.text(x + w / 2, y + h - hh / 2 - 0.01, title, ha="center", va="center",
            fontsize=ft, fontweight="bold", color=tc, zorder=4)
    if lines:
        top = y + h - hh - 0.12
        bot = y + 0.08
        step = line_step if line_step is not None else max(0.22, (top - bot) / len(lines))
        for i, ln in enumerate(lines):
            cy = top - (i + 0.45) * step
            ax.text(x + w / 2, cy, ln, ha="center", va="center",
                    fontsize=fb, color="#1E293B", zorder=4)


def _arr(ax, p1, p2, col="#334155", lw=1.9, ls="-", rad=0.0, ms=13.5, z=5):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        linestyle=ls, color=col, connectionstyle=f"arc3,rad={rad}", zorder=z,
    ))


def _pill(ax, x: float, y: float, txt: str, col="#0F172A", bg="#FFFFFF", ec="#CBD5E1", fs=7.8, bold=False):
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs,
            fontweight=("bold" if bold else "normal"), color=col,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=bg, edgecolor=ec, linewidth=0.95), zorder=7)


def generate_pipeline_stages(out_path: Path) -> None:
    """Control loop as in MPCTransitionRailroadEnv.step(): forces estimated -> PPO plan -> eco governor ->
    shield (contract search tree only when the plan is unsafe) -> shaper + MPC -> back to force estimation.
    The eco governor never coasts or brakes (it holds speed and softens starts when early); coasting is the
    shield's C4 backup. The learned speed-range predictor lives inside the shield (and caps the pilot's speed).
    """
    W, H = 16.0, 3.75
    fig, ax = plt.subplots(figsize=(W, H), dpi=300)
    ax.set_xlim(0, W)
    ax.set_ylim(-0.12, H - 0.25)
    ax.axis("off")

    stages = [
        ("1. Force estimation", ["Mass + wind at departure", "Wind and grip along route"], "#F5F3FF", "#6250D6", "#6250D6"),
        ("2. PPO pilot", ["Picks a speed plan every 4 s"], "#EFF6FF", "#2A78D6", "#2A78D6"),
        ("3. Eco governor", ["When early: hold speed.", "Depart more gently"], "#F0FDF4", "#4D9A3A", "#3F7F30"),
        ("4. Shield", ["Predicts the 99% speed range,", "checks the proposed plan"], "#FFF1F2", "#E34948", "#C73534"),
        ("5. MPC tracker", ["Smooth speed curve,", "traction / brake command"], "#ECFDF5", "#008300", "#006A00"),
    ]
    flow = ["estimated\nforces", "speed\nplan", "capped\nplan", "safe\nplan"]
    x0, w, top_y, top_h = 0.35, 2.45, 2.1, 1.35
    gap = (W - 2 * x0 - len(stages) * w) / (len(stages) - 1)
    xs = [x0 + i * (w + gap) for i in range(len(stages))]
    cx = [x + w / 2 for x in xs]
    for x, (title, lines, fc, ec, hc) in zip(xs, stages):
        _card(ax, x, top_y, w, top_h, title, lines, fc, ec, hc, ft=9.4, fb=8.3)
    ym = top_y + 0.42
    for i in range(len(stages) - 1):
        _arr(ax, (xs[i] + w, ym), (xs[i + 1], ym), col=stages[i][3])
        ax.text(xs[i] + w + gap / 2, ym + 0.12, flow[i], ha="center", va="bottom",
                fontsize=7.6, color="#334155", linespacing=1.05, zorder=6)

    # Contract search: a separate branch, entered only when the shield finds the plan unsafe
    bk_w = 3.4
    bk_x = cx[3] - bk_w / 2
    _card(ax, bk_x, 0.25, bk_w, 1.15, "Contract search tree (only if unsafe)",
          ["Backup contracts C1-C4 x 3 blend strengths;", "prune unsafe leaves, keep the best reward"],
          "#FEF2F2", "#E34948", "#991B1B", ft=8.6, fb=8.0)
    _arr(ax, (cx[3] - 0.45, top_y), (cx[3] - 0.45, 1.40), col="#C73534", lw=1.6, ls="--")
    _pill(ax, cx[3] - 1.20, 1.76, "unsafe", col="#991B1B", bg="#FEF2F2", ec="#F87171", fs=7.4)
    _arr(ax, (cx[3] + 0.45, 1.40), (cx[3] + 0.45, top_y), col="#C73534", lw=1.6)
    _pill(ax, cx[3] + 1.25, 1.76, "best among safe option", col="#991B1B", bg="#FEF2F2", ec="#F87171", fs=7.4)

    # Closed loop: MPC command acts on the train, measurements feed force estimation again
    xa, xb, yb = cx[4], cx[0], -0.04
    ax.plot([xa, xa, xb, xb], [top_y, yb, yb, top_y - 0.2], color="#475569", lw=1.6, zorder=3)
    _arr(ax, (xb, top_y - 0.2), (xb, top_y), col="#475569", lw=1.6)
    _pill(ax, xa, 0.95, "traction / brake\napplied to train", col="#065F46", bg="#ECFDF5", ec="#10B981", fs=7.4)
    _pill(ax, xb, 0.95, "measured speed,\nwind, grip", col="#3B2F99", bg="#F5F3FF", ec="#A78BFA", fs=7.4)

    fig.tight_layout(pad=0.25)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_train_physical_forces(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 5.0), dpi=300)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0.2, 5.2)
    ax.axis("off")

    th = np.deg2rad(8.5)
    ct, st = np.cos(th), np.sin(th)
    u_vec = np.array([ct, st])
    n_vec = np.array([-st, ct])
    p_origin = np.array([1.2, 0.95])
    ax.plot([p_origin[0], p_origin[0] + 10.6], [p_origin[1], p_origin[1]], color="#94A3B8", lw=1.3, ls="--", zorder=1)
    arc_r = 2.6
    angs = np.linspace(0, th, 30)
    ax.plot(p_origin[0] + arc_r * np.cos(angs), p_origin[1] + arc_r * np.sin(angs), color="#475569", lw=1.5, zorder=2)
    ax.text(p_origin[0] + 2.75, p_origin[1] + 0.12, "grade", fontsize=9.2, color="#334155", fontweight="bold")

    p_end = p_origin + 11.1 * u_vec
    ax.plot([p_origin[0], p_end[0]], [p_origin[1], p_end[1]], color="#1E293B", lw=3.2, zorder=3)
    for s in np.linspace(0.3, 10.7, 24):
        sp = p_origin + s * u_vec
        sp2 = sp - 0.18 * n_vec
        ax.plot([sp[0], sp2[0]], [sp[1], sp2[1]], color="#64748B", lw=1.8, zorder=2)

    car_s0, car_len, car_h, wheel_h = 3.2, 5.1, 1.35, 0.24
    base_pts = [np.array([car_s0, wheel_h]), np.array([car_s0 + car_len, wheel_h]),
                np.array([car_s0 + car_len - 0.18, wheel_h + car_h]), np.array([car_s0 + 0.08, wheel_h + car_h])]
    world_pts = [p_origin + pt[0] * u_vec + pt[1] * n_vec for pt in base_pts]
    ax.add_patch(Polygon(world_pts, closed=True, facecolor="#E2E8F0", edgecolor="#1E293B", lw=2.2, zorder=4))
    stripe_l = p_origin + (car_s0 + 0.05) * u_vec + (wheel_h + 0.36) * n_vec
    stripe_r = p_origin + (car_s0 + car_len - 0.06) * u_vec + (wheel_h + 0.36) * n_vec
    ax.plot([stripe_l[0], stripe_r[0]], [stripe_l[1], stripe_r[1]], color="#2A78D6", lw=4.5, zorder=5)
    for ws in np.linspace(car_s0 + 0.45, car_s0 + car_len - 0.95, 5):
        w_pts = [p_origin + ws * u_vec + (wheel_h + 0.62) * n_vec, p_origin + (ws + 0.62) * u_vec + (wheel_h + 0.62) * n_vec,
                 p_origin + (ws + 0.62) * u_vec + (wheel_h + 1.05) * n_vec, p_origin + ws * u_vec + (wheel_h + 1.05) * n_vec]
        ax.add_patch(Polygon(w_pts, closed=True, facecolor="#BAE6FD", edgecolor="#334155", lw=1.0, zorder=5))
    for wx in (car_s0 + 0.75, car_s0 + 1.35, car_s0 + car_len - 1.35, car_s0 + car_len - 0.75):
        wc = p_origin + wx * u_vec + 0.13 * n_vec
        ax.add_patch(Circle((wc[0], wc[1]), 0.13, facecolor="#334155", edgecolor="#0F172A", lw=1.3, zorder=6))
    car_center = p_origin + (car_s0 + 0.5 * car_len) * u_vec + (wheel_h + 0.18) * n_vec
    ax.text(car_center[0], car_center[1], "mass = empty train + passengers (unknown)",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color="#0F172A", rotation=8.5, zorder=6)

    v_p1 = p_origin + (car_s0 + 1.6) * u_vec + (wheel_h + car_h + 0.28) * n_vec
    v_p2 = p_origin + (car_s0 + 3.6) * u_vec + (wheel_h + car_h + 0.28) * n_vec
    _arr(ax, v_p1, v_p2, col="#0F172A", lw=2.2, ms=14)
    ax.text(0.5 * (v_p1[0] + v_p2[0]), 0.5 * (v_p1[1] + v_p2[1]) + 0.24, "direction of travel",
            ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#0F172A", rotation=8.5)

    fp_start = p_origin + (car_s0 + car_len) * u_vec + (wheel_h + 0.45) * n_vec
    fp_end = fp_start + 1.75 * u_vec
    _arr(ax, fp_start, fp_end, col="#008300", lw=3.0, ms=16)
    _pill(ax, fp_end[0] + 0.85, fp_end[1] + 0.25, "Traction / braking\n(drivetrain lag)",
          col="#065F46", bg="#ECFDF5", ec="#10B981", fs=8.4, bold=True)

    fd_tail = p_origin + (car_s0 + car_len + 1.75) * u_vec + (wheel_h + 1.05) * n_vec
    fd_tip = p_origin + (car_s0 + car_len - 0.05) * u_vec + (wheel_h + 1.05) * n_vec
    _arr(ax, fd_tail, fd_tip, col="#2A78D6", lw=2.6, ms=15)
    _pill(ax, fd_tail[0] + 0.55, fd_tail[1] + 0.42, "Air + rolling drag\n(higher in cold, dense air)",
          col="#1E40AF", bg="#EFF6FF", ec="#60A5FA", fs=8.2, bold=True)

    fw_tail = p_origin + (car_s0 - 1.85) * u_vec + (wheel_h + 1.15) * n_vec
    fw_tip = p_origin + (car_s0 + 0.05) * u_vec + (wheel_h + 1.15) * n_vec
    _arr(ax, fw_tail, fw_tip, col="#6250D6", lw=2.6, ls="--", ms=15)
    _arr(ax, fw_tip - 0.25 * u_vec, fw_tail + 0.15 * u_vec, col="#6250D6", lw=2.0, ls=":", ms=13)
    _pill(ax, fw_tail[0] - 0.15, fw_tail[1] + 0.56, "Wind: head / tail,\ngusts up to 30 m/s",
          col="#3B2F99", bg="#F5F3FF", ec="#A78BFA", fs=8.2, bold=True)

    cg = p_origin + (car_s0 + 2.55) * u_vec + (wheel_h + 0.72) * n_vec
    _arr(ax, cg, cg + np.array([0.0, -1.35]), col="#E34948", lw=2.3, ms=14)
    _pill(ax, cg[0] + 0.85, cg[1] - 1.68, "Gravity on the grade", col="#991B1B", bg="#FEF2F2", ec="#F87171", fs=8.2, bold=True)

    fa_start = p_origin + (car_s0 + 0.05) * u_vec + (wheel_h + 0.45) * n_vec
    fa_end = fa_start - 1.55 * u_vec
    _arr(ax, fa_start, fa_end, col="#EB6834", lw=2.6, ms=15)
    _pill(ax, fa_end[0] - 0.30, fa_end[1] - 0.40, "Curve resistance\n(higher with hot rails)",
          col="#92400E", bg="#FFFBEB", ec="#FBBF24", fs=8.2, bold=True)

    w_mark = p_origin + (car_s0 + car_len - 1.05) * u_vec
    ax.plot([w_mark[0], w_mark[0] + 0.75], [w_mark[1], w_mark[1] - 0.58], color="#B45309", lw=1.4, ls=":")
    _pill(ax, w_mark[0] + 1.85, w_mark[1] - 0.80, "Wheel-rail grip limits\nacceleration and braking\n(rain, snow, frost)",
          col="#78350F", bg="#FEF3C7", ec="#F59E0B", fs=8.0, bold=True)

    fig.tight_layout(pad=0.25)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_contract_searching_tree(out_path: Path) -> None:
    """Worked example: a headwind makes the train late, so the deadline margin is violated."""
    fig, ax = plt.subplots(figsize=(13.2, 6.0), dpi=300)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    ax.text(6.6, 5.82, "Example: a headwind makes the train late (deadline margin violated)",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0F172A")

    _card(ax, 0.20, 2.35, 2.10, 1.30, "Pilot command",
          ["Speed, comfort: OK", "Deadline: violated"],
          "#EFF6FF", "#2A78D6", "#2A78D6", ft=9.2, fb=8.0)

    # Safety contracts: each restores one safety property.
    ax.text(4.20, 5.40, "Safety contracts", ha="center", va="center", fontsize=8.6, fontweight="bold",
            color="#334155")
    c_specs = [
        (4.25, "C1 Speed", "#F1F5F9", "#64748B", "#475569"),
        (3.05, "C2 Comfort", "#ECFDF5", "#1BAF7A", "#13895F"),
        (1.85, "C3 Deadline", "#FEF2F2", "#E34948", "#C73534"),
    ]
    for cy, title, fc, ec, hc in c_specs:
        ax.add_patch(FancyBboxPatch((3.25, cy + 0.15), 1.90, 0.65, boxstyle="round,pad=0.04,rounding_size=0.14",
                                    facecolor=hc, edgecolor=ec, linewidth=1.5, zorder=2))
        ax.text(4.20, cy + 0.475, title, ha="center", va="center", fontsize=9.0, fontweight="bold",
                color="#FFFFFF", zorder=4)
        _arr(ax, (2.30, 3.00), (3.25, cy + 0.47), col=hc, lw=1.7)

    # Energy option: not a safety property, kept only if every safety margin holds.
    ax.plot([2.95, 9.05], [1.45, 1.45], color="#94A3B8", lw=1.0, ls="--", zorder=1)
    ax.text(4.20, 1.27, "Energy option (not a safety property)", ha="center", va="center", fontsize=8.0,
            style="italic", color="#475569")
    ax.add_patch(FancyBboxPatch((3.25, 0.325), 1.90, 0.65, boxstyle="round,pad=0.04,rounding_size=0.14",
                                facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.5, linestyle="--", zorder=2))
    ax.text(4.20, 0.65, "C4 Eco coast", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#475569")
    _arr(ax, (2.30, 3.00), (3.25, 0.65), col="#94A3B8", lw=1.3, ls=":")

    GREY = ("#F1F5F9", "#94A3B8", "#475569", False)
    PRUN = ("#FEE2E2", "#F87171", "#991B1B", False)
    FEAS = ("#DBEAFE", "#2A78D6", "#1E3A8A", True)
    WIN = ("#DCFCE7", "#008300", "#14532D", True)
    ax.text(7.05, 5.40, "Blends 55 / 85 / 100%", ha="center", va="center", fontsize=8.6, fontweight="bold",
            color="#334155")
    leaves = [
        (4.25, [("still late", GREY)]),
        (3.05, [("still late", GREY)]),
        (1.85, [("55%: safe", FEAS), ("85%: chosen", WIN), ("100%: over band", PRUN)]),
        (0.30, [("later: pruned", PRUN)]),
    ]
    for cy, items in leaves:
        hgt = 0.95 if cy > 0.5 else 0.70
        n = len(items)
        for i, (lbl, (bg_c, bd_c, txt_c, strong)) in enumerate(items):
            bh = 0.26 if n == 3 else 0.34
            ny = cy + hgt / 2 - bh / 2 + (1 - i) * 0.31 if n == 3 else cy + hgt / 2 - bh / 2
            win = lbl.endswith("chosen")
            ax.add_patch(FancyBboxPatch((6.05, ny), 2.00, bh, boxstyle="round,pad=0.02,rounding_size=0.06",
                                        facecolor=bg_c, edgecolor=bd_c, linewidth=(2.0 if win else 1.1), zorder=3))
            _arr(ax, (5.15, cy + hgt / 2), (6.05, ny + bh / 2),
                 col=("#008300" if win else ("#2A78D6" if strong else "#94A3B8")),
                 lw=(2.2 if win else 1.1), ls=("-" if strong else ":"), ms=10.0)
            ax.text(7.05, ny + bh / 2, ("all blends " + lbl) if n == 1 else lbl, ha="center", va="center",
                    fontsize=7.8, fontweight=("bold" if strong else "normal"), color=txt_c, zorder=4)

    ax.plot([8.15, 8.25, 8.25, 8.15], [0.45, 0.45, 5.10, 5.10], color="#7E22CE", lw=1.4, zorder=2)
    _card(ax, 8.60, 2.35, 2.15, 1.30, "Prune and score",
          ["Drop blends outside", "the 99% band", "Keep the best reward"],
          "#FAF5FF", "#7E22CE", "#7E22CE", ft=9.0, fb=7.9, line_step=0.24)
    _arr(ax, (8.25, 3.00), (8.60, 3.00), col="#7E22CE", lw=2.0, ms=14)

    _card(ax, 11.10, 2.35, 1.90, 1.30, "Verify and hold",
          ["2-step check", "hold 2 steps"],
          "#ECFDF5", "#008300", "#006A00", ft=9.0, fb=7.9, line_step=0.26)
    _arr(ax, (10.75, 3.00), (11.10, 3.00), col="#008300", lw=2.2, ms=14)
    ax.text(12.05, 2.20, "to the shaper", ha="center", va="top", fontsize=7.8, color="#14532D")

    fig.tight_layout(pad=0.25)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _compute_scenario_ceil_grid(scenario: dict) -> tuple[np.ndarray, np.ndarray]:
    xs, vc = [], []
    cum_x = 0.0
    for leg in scenario["legs"]:
        cor = leg["corridor"]
        gx = np.linspace(0.0, cor.station_x_m, 160)
        for xi in gx:
            xs.append(cum_x + float(xi))
            vc.append(float(cor.compute_smooth_spatial_ceiling(float(xi), cor.comfort_braking_decel)))
        cum_x += float(cor.station_x_m)
    return np.asarray(xs, dtype=np.float64), np.asarray(vc, dtype=np.float64)


# Real-journey date of the published figures (the fetcher defaults to "yesterday", which changes the journeys).
FIG_DATE = "2026-11-17"


def generate_empirical_figures(fig_dir: Path) -> None:
    emu = LearnedLinearizationModel(hidden_dim=64)
    emu.load_state_dict(torch.load(SCRIPT_DIR / "checkpoints" / "best_emulator.pt", map_location="cpu", weights_only=True))
    emu.eval()
    pol_cos = CosineTrajectoryRLPolicy(obs_dim=18, act_dim=3, hidden_dim=64)
    pol_cos.load_state_dict(torch.load(SCRIPT_DIR / "checkpoints" / "best_ppo_cosine_pilot.pt", map_location="cpu", weights_only=True))
    pol_cos.eval()
    pol_free = CosineTrajectoryRLPolicy(obs_dim=18, act_dim=3, hidden_dim=64)
    pol_free.load_state_dict(torch.load(SCRIPT_DIR / "checkpoints" / "best_ppo_free_pilot.pt", map_location="cpu", weights_only=True))
    pol_free.eval()

    # Representative Baseline (ep=0: 4 stops, 16.15 km) & Extreme Storm/Slush (ep=11: #8601, 4 stops, 15.60 km, mu=0.62)
    sc_base = fetch_real_world_multi_station_scenario(
        date_str=FIG_DATE, train_number=8311, n_legs_max=0, seed=9000, ep_idx=0, harsh_weather_mode="mixed"
    )
    sc_ext = fetch_real_world_multi_station_scenario(
        date_str=FIG_DATE, train_number=8311, n_legs_max=0, seed=9000 + 11 * 29, ep_idx=11, harsh_weather_mode="mixed"
    )
    sc_clean = generate_multi_station_scenario(seed=9000, n_legs=3, wind_mode="none", allow_infeasible_breaks=False, extreme_noise=False)

    specs = [
        ("1. Greedy", "MAX_SPEED_GREEDY", None, None, "steelblue", "--"),
        ("2. PPO Unsh", "RL_LEARNED", pol_cos, None, "seagreen", "--"),
        ("3. Switch", "RL_LEARNED", pol_cos, "switch", "magenta", ":"),
        ("4. 1-Step DCBF", "RL_LEARNED", pol_cos, "cbf_opt", "saddlebrown", "-."),
        ("5. UCB-DMPS", "RL_LEARNED", pol_cos, "mps_rollout", "crimson", "-"),
        ("6. NonCos Unsh", "RL_FREE_LEARNED", pol_free, None, "slategray", "--"),
        ("7. NonCos UCB", "RL_FREE_LEARNED", pol_free, "mps_rollout", "teal", "-."),
        ("8. Bilbao'25", "BILBAO_CBTC_ATO", None, None, "goldenrod", "-."),
    ]

    trajs_base = {nm: evaluate_multi_station_journey(nm, sc_base, md, rl_policy=pl, shield_mode=sh, emulator=emu, record_traj=True) for nm, md, pl, sh, _, _ in specs}
    trajs_ext = {nm: evaluate_multi_station_journey(nm, sc_ext, md, rl_policy=pl, shield_mode=sh, emulator=emu, record_traj=True) for nm, md, pl, sh, _, _ in specs}
    trajs_clean = {nm: evaluate_multi_station_journey(nm, sc_clean, md, rl_policy=pl, shield_mode=sh, emulator=emu, record_traj=True) for nm, md, pl, sh, _, _ in specs}

    # ------------------------------------------------------------------
    # Figure 4 (Main Paper Testing): Baseline API vs Extreme Disturbance
    # ------------------------------------------------------------------
    C_UCB, C_UNSH, C_BIL, C_CEIL = "#e34948", "#2a78d6", "#eda100", "#52514e"
    fig, axes = plt.subplots(4, 2, figsize=(15.0, 9.0), dpi=220, sharex="col",
                             gridspec_kw={"height_ratios": [3.0, 1.0, 1.0, 1.1]})
    for col_idx, (title_tag, scen_obj, tr_map) in enumerate([
        ("Real-world journey, measured weather (4 stops, 16.2 km)", sc_base, trajs_base),
        ("Real-world journey, injected storm and slush (4 stops, 15.6 km)", sc_ext, trajs_ext),
    ]):
        tr_sh = tr_map["5. UCB-DMPS"]["traj"]
        tr_un = tr_map["2. PPO Unsh"]["traj"]
        tr_bb = tr_map["8. Bilbao'25"]["traj"]
        x_sh = tr_sh["x"] / 1000.0
        xg_c, vc_g = _compute_scenario_ceil_grid(scen_obj)

        ax_t = axes[0, col_idx]
        ax_t.fill_between(x_sh, np.maximum(0.0, tr_sh["v_emu"] - tr_sh["scissor_rad"]),
                          tr_sh["v_emu"] + tr_sh["scissor_rad"], color=C_UCB, alpha=0.18, lw=0, label="99% speed band (shield)")
        ax_t.plot(xg_c / 1000.0, vc_g, color=C_CEIL, ls=":", lw=1.2, label="Speed limit")
        ax_t.plot(tr_bb["x"] / 1000.0, tr_bb["v"], color=C_BIL, ls="-.", lw=1.4, label="8. Bilbao CBTC-ATO")
        ax_t.plot(tr_un["x"] / 1000.0, tr_un["v"], color=C_UNSH, ls="--", lw=1.4, label="2. PPO cosine, unshielded")
        ax_t.plot(x_sh, tr_sh["v"], color=C_UCB, ls="-", lw=2.0, label="5. PPO cosine + UCB-DMPS")
        im = tr_sh["intervened"]
        if np.any(im):
            ax_t.scatter(x_sh[im], tr_sh["v"][im], color=C_UCB, edgecolor="white", s=22, zorder=6, label="Shield intervention")
        ax_t.set_title(title_tag, fontsize=10.5, fontweight="bold")
        ax_t.set_ylabel("Speed (m/s)")

        ax_u = axes[1, col_idx]
        ax_u.axhline(0.0, color="#c3c2b7", lw=0.8)
        ax_u.plot(x_sh, tr_sh["u_act"], color=C_UCB, lw=1.1)
        ax_u.set_ylabel("Traction (+)\nbrake (-)", fontsize=8.5)
        ax_u.set_ylim(-1.05, 1.05)

        ax_w = axes[2, col_idx]
        ax_w.axhline(0.0, color="#c3c2b7", lw=0.8)
        ax_w.plot(x_sh, tr_sh["wind_true"], color="#6250d6", lw=1.3, label="true")
        ax_w.plot(x_sh, tr_sh["wind_hat"], color="#6250d6", ls=":", lw=1.3, label="estimated")
        ax_w.set_ylabel("Wind push\n(m/s$^2$)", fontsize=8.5)
        ax_w.legend(fontsize=7.4, loc="lower left", ncol=2, frameon=False)

        ax_m = axes[3, col_idx]
        ax_m.step(x_sh, tr_sh["mass_kg"] / 1000.0, where="post", color="#1baf7a", lw=2.2, label="true")
        ax_m.plot(x_sh, tr_sh["mass_hat_kg"] / 1000.0, color="#0f6e4b", ls="--", lw=1.3, label="identified at departure")
        ax_m.set_ylabel("Mass (t)", fontsize=8.5)
        ax_m.set_xlabel("Position along the journey (km)")
        ax_m.legend(fontsize=7.4, loc="lower right", ncol=2, frameon=False)
        for a in axes[:, col_idx]:
            a.grid(True, alpha=0.22, lw=0.6)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)

    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=6, fontsize=8.6, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(str(fig_dir / "testing_disturbance_trajectories.png"), dpi=220)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Appendix Full-Page Figure: 9-Panel Confrontation Across All Methods
    # Reads actual exported CSVs in data/ for exact 30-run statistics!
    # ------------------------------------------------------------------
    def _canon(name: str) -> str:
        return re.sub(r"^\d+\.\s*", "", name)

    def _load_csv_means(*csv_paths: Path) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[dict[str, str]]] = {}
        for csv_path in csv_paths:
            with open(csv_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    # Key by the name without its number, so CSVs written before a renumbering still match.
                    grouped.setdefault(_canon(row["method"]), []).append(row)
        out: dict[str, dict[str, float]] = {}
        for m, rows in grouped.items():
            out[m] = {
                "time": float(np.mean([float(r["abs_sched_err_s"]) for r in rows])),
                "stop_cm": float(np.mean([float(r["abs_stop_err_m"]) * 100.0 for r in rows])),
                "smooth": float(np.mean([float(r["step_smooth_cost"]) for r in rows])),
                "energy": float(np.mean([float(r["spec_energy_wh_tkm"]) for r in rows])),
                "loss": float(np.mean([float(r["multi_obj_score"]) for r in rows])),
                "mean_us": float(np.mean([float(r["mean_total_us"]) for r in rows])),
                "p99_us": float(np.mean([float(r["p99_total_us"]) for r in rows])),
                "max_us": float(np.max([float(r["max_total_us"]) for r in rows])),
            }
        return out

    csv_real_b = _load_csv_means(SCRIPT_DIR / "data" / "real_world_api_baseline_metrics.csv")
    csv_real_e = _load_csv_means(SCRIPT_DIR / "data" / "real_world_api_extreme_metrics.csv")
    csv_synth = _load_csv_means(SCRIPT_DIR / "data" / "synthetic_baseline_metrics.csv",
                                SCRIPT_DIR / "data" / "synthetic_extreme_metrics.csv")

    full_method_keys = [
        "1. Max-Speed Greedy Cosine + MPC",
        "2. Multi-Property PPO Pilot (Unshielded)",
        "3. Contract-Shielded PPO (Switch MPS)",
        "4. Contract-Shielded PPO (1-Step DCBF)",
        "5. Contract-Shielded PPO (UCB-DMPS Cosine)",
        "6. Stable-Accel PPO (No Cosine, Unshielded)",
        "7. Stable-Accel PPO + Jerk-Horizon UCB-DMPS",
        "8. Bilbao (2025) CBTC-ATO (RMBTP + PI)",
    ]
    short_lbls = ["Greedy", "PPO Unsh", "Switch", "DCBF", "UCB-Cos", "NonCos", "NonCos+UCB", "Bilbao'25"]
    full_method_keys = [_canon(k) for k in full_method_keys]
    lit_short = [("Xiao (2023)", "Xiao'23"), ("Zhong (2022)", "Zhong'22"), ("Chen (2025)", "CDQ'25"),
                 ("Zhao (2023)", "S-SARSA'23"), ("Huang (2025)", "EITOP'25"), ("Li (2023)", "SRL'23"),
                 ("Zhao (2025)", "SSA-DRL'25"), ("Cheng (2024)", "S-PPO'24"), ("Wang (2024)", "ES-MEDRL'24"),
                 ("Tan (2025)", "GA-DDPG'25")]
    for prefix, lbl in lit_short:
        k_lit = next((k for k in csv_real_b if k.startswith(prefix)), None)
        if k_lit is not None and k_lit in csv_real_e and k_lit in csv_synth:
            full_method_keys.append(k_lit)
            short_lbls.append(lbl)
    n_meth = len(full_method_keys)

    fig_a, ax_a = plt.subplots(3, 3, figsize=(16.5, 19.5), dpi=200)
    # Row 0: Sample closed-loop profiles (all 10 methods in Panel 1!)
    for nm, _, _, _, col, ls in specs:
        r = trajs_base[nm]
        ax_a[0, 0].plot(r["traj"]["x"], r["traj"]["v"], color=col, ls=ls, lw=1.35, label=nm)
    ax_a[0, 0].set_title("1. Speed profiles, methods 1-8 (real journey)", fontweight="bold", fontsize=10.5)
    ax_a[0, 0].set_xlabel("Position (m)")
    ax_a[0, 0].set_ylabel("Speed (m/s)")
    ax_a[0, 0].legend(fontsize=6.3, ncol=2, loc="lower left")
    ax_a[0, 0].grid(True, alpha=0.28)

    tr_dmps = trajs_base["5. UCB-DMPS"]["traj"]
    ax_a[0, 1].plot(tr_dmps["t"], tr_dmps["accel"], color="#e34948", lw=1.3, label="Acceleration")
    ax_a[0, 1].axhline(1.20, color="#52514e", ls="--", lw=1.0, label="Comfort limits (+1.20 / -0.92)")
    ax_a[0, 1].axhline(-0.92, color="#52514e", ls="--", lw=1.0)
    jmax = float(np.max(np.abs(tr_dmps["jerk"])))
    n_over = int(np.sum(np.abs(tr_dmps["jerk"]) > 0.48))
    ax_a[0, 1].text(0.98, 0.04, f"Jerk above 0.48 m/s³ on {n_over} of {len(tr_dmps['jerk'])} steps\n"
                    f"(peak {jmax:.2f} m/s³, at the end of departure ramps)", transform=ax_a[0, 1].transAxes,
                    ha="right", va="bottom", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c3c2b7"))
    ax_a[0, 1].set_title("2. Acceleration within comfort limits (UCB-DMPS)", fontweight="bold", fontsize=10.5)
    ax_a[0, 1].set_xlabel("Time (s)")
    ax_a[0, 1].set_ylabel("Acceleration (m/s²)")
    ax_a[0, 1].set_ylim(-1.4, 1.6)
    ax_a[0, 1].legend(fontsize=7.2, loc="upper right", frameon=False)
    ax_a[0, 1].grid(True, alpha=0.28)

    ax_a[0, 2].step(tr_dmps["t"], tr_dmps["mass_kg"] / 1000.0, where="post", color="#1baf7a", lw=2.2, label="True mass")
    ax_a[0, 2].plot(tr_dmps["t"], tr_dmps["mass_hat_kg"] / 1000.0, color="#0f6e4b", ls="--", lw=1.4, label="Identified at departure")
    ax_a[0, 2].set_title("3. Departure mass identification", fontweight="bold", fontsize=10.5)
    ax_a[0, 2].set_xlabel("Time (s)")
    ax_a[0, 2].set_ylabel("Mass (t)")
    ax_a[0, 2].legend(fontsize=7.2, loc="upper left")
    ax_a[0, 2].grid(True, alpha=0.28)

    xp = np.arange(n_meth)
    bw = 0.26
    metric_panels = [
        (ax_a[1, 0], "time", "4. Mean arrival-time error (s)", "Seconds (lower is better)"),
        (ax_a[1, 1], "stop_cm", "5. Mean stopping error (cm)", "cm (lower is better)"),
        (ax_a[1, 2], "smooth", "6. Smoothness cost", "Lower is better"),
        (ax_a[2, 0], "energy", "7. Energy (Wh/t·km)", "Wh/t·km (lower is better)"),
        (ax_a[2, 1], "loss", "8. LossScore", "Lower is better"),
    ]
    for ax_p, mkey, ttl, ylbl in metric_panels:
        vb = [csv_real_b[k][mkey] for k in full_method_keys]
        ve = [csv_real_e[k][mkey] for k in full_method_keys]
        vs = [csv_synth[k][mkey] for k in full_method_keys]
        ax_p.bar(xp - bw, vb, bw, color="#2a78d6", label="Real, measured weather (15)")
        ax_p.bar(xp, ve, bw, color="#eb6834", label="Real, extreme weather (15)")
        ax_p.bar(xp + bw, vs, bw, color="#1baf7a", label="Synthetic (30)")
        ax_p.set_title(ttl, fontweight="bold", fontsize=10.5)
        ax_p.set_ylabel(ylbl)
        ax_p.set_xticks(xp)
        ax_p.set_xticklabels(short_lbls, rotation=40, ha="right", fontsize=6.8)
        ax_p.set_ylim(0, 1.40 * max(max(vb), max(ve), max(vs)))
        ax_p.legend(fontsize=7.3, loc="upper left", ncol=1, frameon=False)
        ax_p.grid(True, axis="y", alpha=0.28)

    # Panel 9: Latency (Mean, P99, Max on Real Extreme vs 500 us hard budget)
    ax_lat = ax_a[2, 2]
    m_lat = [csv_real_e[k]["mean_us"] for k in full_method_keys]
    p_lat = [csv_real_e[k]["p99_us"] for k in full_method_keys]
    mx_e = [csv_real_e[k]["max_us"] for k in full_method_keys]
    ax_lat.bar(xp - bw, m_lat, bw, color="#2a78d6", label="Mean")
    ax_lat.bar(xp, p_lat, bw, color="#eb6834", label="99th percentile")
    ax_lat.bar(xp + bw, mx_e, bw, color="#1baf7a", label="Worst case")
    ax_lat.axhline(500.0, color="red", ls="--", lw=1.4, label="500 µs budget (methods 2-7)")
    ax_lat.set_yscale("log")
    ax_lat.set_title("9. Step latency, extreme weather (µs, log scale)", fontweight="bold", fontsize=10.5)
    ax_lat.set_ylabel("Microseconds")
    ax_lat.set_xticks(xp)
    ax_lat.set_xticklabels(short_lbls, rotation=40, ha="right", fontsize=6.8)
    ax_lat.set_ylim(top=8000.0)
    ax_lat.legend(fontsize=7.1, loc="upper left", ncol=2, frameon=False)
    ax_lat.grid(True, axis="y", alpha=0.28)

    fig_a.tight_layout()
    fig_a.savefig(str(fig_dir / "appendix_full_benchmark.png"), dpi=200)
    plt.close(fig_a)

    # ------------------------------------------------------------------
    # Appendix Figure 2: Non-Disturbance vs Heavy Disturbance (Many Methods)
    # ------------------------------------------------------------------
    fig_d, ax_d = plt.subplots(2, 2, figsize=(15.2, 8.6), dpi=220, sharex="col")
    plot_9 = [s for s in specs if s[0] != "4. 1-Step DCBF"]
    for c_i, (lbl, sc_obj, tmap) in enumerate([
        ("Synthetic journey without wind (3 stops)", sc_clean, trajs_clean),
        ("Real-world journey, injected storm and slush (4 stops)", sc_ext, trajs_ext),
    ]):
        for nm, _, _, _, col, ls in plot_9:
            ax_d[0, c_i].plot(tmap[nm]["traj"]["x"], tmap[nm]["traj"]["v"], color=col, ls=ls, lw=1.4, label=nm)
        xg_d, vc_d = _compute_scenario_ceil_grid(sc_obj)
        ax_d[0, c_i].plot(xg_d, vc_d, "k:", lw=1.2, label="Speed limit")
        ax_d[0, c_i].set_title(lbl, fontweight="bold", fontsize=10.0)
        ax_d[0, c_i].set_ylabel("Speed (m/s)")
        ax_d[0, c_i].legend(fontsize=6.8, ncol=2, loc="lower left")
        ax_d[0, c_i].grid(True, alpha=0.28)

        tr7 = tmap["5. UCB-DMPS"]["traj"]
        ax_d[1, c_i].plot(tr7["x"], tr7["wind_true"], color="navy", lw=1.4, label="Wind push, true (m/s²)")
        ax_d[1, c_i].plot(tr7["x"], tr7["wind_hat"], color="dodgerblue", ls=":", lw=1.3, label="Wind push, estimated (m/s²)")
        ax_d[1, c_i].plot(tr7["x"], tr7["u_act"], color="crimson", alpha=0.75, lw=1.1, label="Traction/brake command (UCB-DMPS, fraction)")
        ax_d[1, c_i].set_xlabel("Position (m)")
        ax_d[1, c_i].set_ylabel("Wind and command")
        ax_d[1, c_i].set_ylim(-1.1, 1.1)
        ax_d[1, c_i].legend(fontsize=7.5, loc="lower left")
        ax_d[1, c_i].grid(True, alpha=0.28)

    fig_d.tight_layout()
    fig_d.savefig(str(fig_dir / "appendix_disturbance_vs_clean_trajs.png"), dpi=220)
    plt.close(fig_d)


def main():
    fig_dir = SCRIPT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    generate_pipeline_stages(fig_dir / "pipeline_stages.png")
    generate_train_physical_forces(fig_dir / "train_physical_forces.png")
    generate_contract_searching_tree(fig_dir / "contract_searching_tree.png")
    generate_empirical_figures(fig_dir)
    print("Generated all paper & appendix figures inside:", fig_dir)


if __name__ == "__main__":
    main()
