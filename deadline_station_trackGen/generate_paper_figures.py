"""
Generate publication-grade diagrams and empirical multi-panel paper figures inside
`deadline_station_trackGen/figures/`:
  1. `pipeline_stages.png`                  - Concise stage-level system pipeline diagram
  2. `train_physical_forces.png`            - Free-body physical force diagram of train + weather
  3. `contract_searching_tree.png`          - Safe Contract Searching Tree with tube pruning & UCB
  4. `testing_disturbance_trajectories.png` - Multi-station trajectories across Baseline vs Extreme disturbances
  5. `appendix_full_benchmark.png`          - Full-page 9-panel comparison across all 10 methods
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
        step = max(0.22, (top - bot) / len(lines))
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
    fig, ax = plt.subplots(figsize=(14.5, 5.2), dpi=300)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    # Stage Cards (Clean Left-to-Right Pipeline)
    _card(ax, 0.25, 3.00, 2.25, 1.75, "1. Station & Estimation",
          [r"Wayside: $\mu_{\mathrm{adh}}, T_{\mathrm{amb}}, M_0$",
           r"Full-Leg MAP Mass $\hat{M}_k$",
           r"Adaptive Wind DOB $\hat{d}_k$"],
          "#F5F3FF", "#7C3AED", "#6D28D9")

    _card(ax, 2.95, 3.00, 2.35, 1.75, "2. Trajectory Policy",
          ["18-D PPO Macro Pilot",
           "4-Crit. Preemption Gate",
           r"$p_{\mathrm{task}}=(v_{\mathrm{cmd}},a_{\mathrm{st}},b_{\mathrm{ld}})$"],
          "#EFF6FF", "#2563EB", "#1D4ED8")

    _card(ax, 5.75, 3.00, 2.65, 1.75, "3. Contract Shield",
          [r"Pressures $\rho_1,\rho_2,\rho_3,\rho_E,\rho_T$",
           r"4-Contract UCB Tree",
           r"Certified $p_{\mathrm{safe}}$ (+Latch)"],
          "#FFF1F2", "#E11D48", "#BE123C")

    _card(ax, 8.85, 3.00, 2.40, 1.75, r"4. $C^2$ Ref. Shaper",
          ["Half-Cosine S-Curve /",
           "Hermite Soft-Curve",
           r"Smooth $V_{\mathrm{ref}}\in\mathbb{R}^5$"],
          "#F0FDFA", "#0D9488", "#0F766E")

    _card(ax, 11.70, 3.00, 2.55, 1.75, r"5. Lifted 5x3 LTV-MPC",
          [r"Lifted State $[v_k, u_{\mathrm{act}}^-]$",
           r"Closed-Form 3x3 QP",
           r"$u_{\mathrm{cmd}} \in [-1, 1]$"],
          "#ECFDF5", "#059669", "#047857")

    # Bottom layer: Train Plant + Learned Heteroscedastic Emulator
    _card(ax, 9.20, 0.55, 5.05, 1.65, "6. Train & Weather Physical Plant",
          [r"Actuator Lag $0.72u_{\mathrm{cmd}}+0.28u_{\mathrm{act}}^-$ + Adhesion $\mu_{\mathrm{adh}}$",
           r"Davis Drag $\rho_{\mathrm{air}}(T)$, Grade/CWR Curve, Wind $d_{\mathrm{wind}}$"],
          "#F8FAFC", "#475569", "#334155")

    _card(ax, 2.80, 0.55, 5.65, 1.65, "7. Heteroscedastic Physics Emulator & 99% Scissor Tube",
          [r"8-Feature Residual Mean + Variance Net ($\hat{v}_{\mathrm{emu}}, \sigma_\theta^2$)",
           r"Forward Tube $\hat{v}_{\mathrm{emu}} \pm 2.576\,\sigma_{\mathrm{tube}}$ + Regime Pincer"],
          "#FAF5FF", "#9333EA", "#7E22CE")

    # Forward arrows along top pipeline
    _arr(ax, (2.50, 3.88), (2.95, 3.88), col="#475569")
    _arr(ax, (5.30, 3.88), (5.75, 3.88), col="#1D4ED8")
    _arr(ax, (8.40, 3.88), (8.85, 3.88), col="#BE123C")
    _arr(ax, (11.25, 3.88), (11.70, 3.88), col="#0F766E")

    # MPC -> Plant
    _arr(ax, (12.98, 3.00), (12.98, 2.20), col="#047857")
    _pill(ax, 13.78, 2.60, r"$u_{\mathrm{cmd}}(k)$", col="#065F46", bg="#ECFDF5", ec="#10B981", fs=7.8, bold=True)

    # Plant -> Emulator -> Shield & Preemption
    _arr(ax, (9.20, 1.38), (8.45, 1.38), col="#475569")
    _pill(ax, 8.82, 1.78, r"$v_k, x_k$", col="#334155", fs=7.5)

    # Plant -> Estimation feedback routed cleanly below Box 7 along y = 0.16
    ax.plot([11.72, 11.72, 1.38, 1.38], [0.55, 0.16, 0.16, 3.00], color="#6D28D9", lw=1.6, ls="--", zorder=3)
    _arr(ax, (1.38, 2.82), (1.38, 3.00), col="#6D28D9", lw=1.6)
    _pill(ax, 1.38, 1.78, "Telemetry & Obs", col="#4C1D95", bg="#F5F3FF", ec="#A78BFA", fs=7.4)

    # Emulator -> Shield & Preemption
    _arr(ax, (7.05, 2.20), (7.05, 3.00), col="#7E22CE", lw=1.8)
    _arr(ax, (4.12, 2.20), (4.12, 3.00), col="#7E22CE", lw=1.5, ls="--")
    _pill(ax, 5.58, 2.58, r"$99\%$ Scissor $r_{99,k}$", col="#581C87", bg="#FAF5FF", ec="#C084FC", fs=7.6, bold=True)

    fig.tight_layout(pad=0.25)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_train_physical_forces(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 5.4), dpi=300)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    # Track slope angle alpha_vis = 8 deg for clear visual free-body display
    th = np.deg2rad(8.5)
    ct, st = np.cos(th), np.sin(th)
    u_vec = np.array([ct, st])     # along grade (up-right)
    n_vec = np.array([-st, ct])    # normal to rail (up-left)

    p_origin = np.array([1.2, 0.95])
    # Horizontal reference dashed line + angle arc
    ax.plot([p_origin[0], p_origin[0] + 10.6], [p_origin[1], p_origin[1]],
            color="#94A3B8", lw=1.3, ls="--", zorder=1)
    arc_r = 2.6
    angs = np.linspace(0, th, 30)
    ax.plot(p_origin[0] + arc_r * np.cos(angs), p_origin[1] + arc_r * np.sin(angs),
            color="#475569", lw=1.5, zorder=2)
    ax.text(p_origin[0] + 3.0, p_origin[1] + 0.19, r"Grade $\theta(x)$", fontsize=9.2, color="#334155", fontweight="bold")

    # Rail bed + sleepers
    p_end = p_origin + 11.1 * u_vec
    ax.plot([p_origin[0], p_end[0]], [p_origin[1], p_end[1]], color="#1E293B", lw=3.2, zorder=3)
    for s in np.linspace(0.3, 10.7, 24):
        sp = p_origin + s * u_vec
        sp2 = sp - 0.18 * n_vec
        ax.plot([sp[0], sp2[0]], [sp[1], sp2[1]], color="#64748B", lw=1.8, zorder=2)

    # Train Carbody polygon (sloped rectangle with chamfered nose)
    car_s0, car_len, car_h, wheel_h = 3.2, 5.1, 1.35, 0.24
    base_pts = [
        np.array([car_s0, wheel_h]),
        np.array([car_s0 + car_len, wheel_h]),
        np.array([car_s0 + car_len - 0.18, wheel_h + car_h]),
        np.array([car_s0 + 0.08, wheel_h + car_h]),
    ]
    world_pts = [p_origin + pt[0] * u_vec + pt[1] * n_vec for pt in base_pts]
    ax.add_patch(Polygon(world_pts, closed=True, facecolor="#E2E8F0", edgecolor="#1E293B", lw=2.2, zorder=4))

    # Accent livery stripe + Windows + Pantograph
    stripe_l = p_origin + (car_s0 + 0.05) * u_vec + (wheel_h + 0.36) * n_vec
    stripe_r = p_origin + (car_s0 + car_len - 0.06) * u_vec + (wheel_h + 0.36) * n_vec
    ax.plot([stripe_l[0], stripe_r[0]], [stripe_l[1], stripe_r[1]], color="#1D4ED8", lw=4.5, zorder=5)

    for ws in np.linspace(car_s0 + 0.45, car_s0 + car_len - 0.95, 5):
        w_pts = [
            p_origin + ws * u_vec + (wheel_h + 0.62) * n_vec,
            p_origin + (ws + 0.62) * u_vec + (wheel_h + 0.62) * n_vec,
            p_origin + (ws + 0.62) * u_vec + (wheel_h + 1.05) * n_vec,
            p_origin + ws * u_vec + (wheel_h + 1.05) * n_vec,
        ]
        ax.add_patch(Polygon(w_pts, closed=True, facecolor="#BAE6FD", edgecolor="#334155", lw=1.0, zorder=5))

    # Wheels + adhesion patches
    for wx in (car_s0 + 0.75, car_s0 + 1.35, car_s0 + car_len - 1.35, car_s0 + car_len - 0.75):
        wc = p_origin + wx * u_vec + 0.13 * n_vec
        ax.add_patch(Circle((wc[0], wc[1]), 0.13, facecolor="#334155", edgecolor="#0F172A", lw=1.3, zorder=6))

    # Label inside train body bottom stripe area
    car_center = p_origin + (car_s0 + 0.5 * car_len) * u_vec + (wheel_h + 0.18) * n_vec
    ax.text(car_center[0], car_center[1],
            r"$M_{\mathrm{eff}} = (1+\gamma)M,\;\; M = M_{\mathrm{tare}} + M_{\mathrm{pax}}(k)$",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color="#0F172A",
            rotation=8.5, zorder=6)

    # Motion velocity vector above roof
    v_p1 = p_origin + (car_s0 + 1.6) * u_vec + (wheel_h + car_h + 0.28) * n_vec
    v_p2 = p_origin + (car_s0 + 3.6) * u_vec + (wheel_h + car_h + 0.28) * n_vec
    _arr(ax, v_p1, v_p2, col="#0F172A", lw=2.2, ms=14)
    ax.text(0.5 * (v_p1[0] + v_p2[0]), 0.5 * (v_p1[1] + v_p2[1]) + 0.24,
            r"Motion $v(t),\; x(t)$", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#0F172A")

    # 1. Driving / Braking Force F_prop (+x forward)
    fp_start = p_origin + (car_s0 + car_len) * u_vec + (wheel_h + 0.45) * n_vec
    fp_end = fp_start + 1.75 * u_vec
    _arr(ax, fp_start, fp_end, col="#059669", lw=3.0, ms=16)
    _pill(ax, fp_end[0] + 0.95, fp_end[1] + 0.22,
          r"$F_{\mathrm{prop}}(u_{\mathrm{act}})$" + "\n" + r"Lag: $0.72u_{\mathrm{cmd}}+0.28u_{\mathrm{act}}^-$",
          col="#065F46", bg="#ECFDF5", ec="#10B981", fs=8.3, bold=True)

    # 2. Davis Aerodynamic + Rolling Drag (-x opposing front)
    fd_start = p_origin + (car_s0 + car_len - 0.10) * u_vec + (wheel_h + 1.05) * n_vec
    fd_end = fd_start + 1.90 * (-u_vec)  # point opposing left along nose
    # Draw external arrow pushing against front nose from right to left!
    fd_tail = p_origin + (car_s0 + car_len + 1.75) * u_vec + (wheel_h + 1.05) * n_vec
    fd_tip = p_origin + (car_s0 + car_len - 0.05) * u_vec + (wheel_h + 1.05) * n_vec
    _arr(ax, fd_tail, fd_tip, col="#2563EB", lw=2.6, ms=15)
    _pill(ax, fd_tail[0] + 0.72, fd_tail[1] + 0.42,
          r"Davis Drag $F_d(v, T_{\mathrm{amb}})$" + "\n" + r"$(Av^2+Bv+C)\,\rho_{\mathrm{air}}(T_{\mathrm{amb}})$",
          col="#1E40AF", bg="#EFF6FF", ec="#60A5FA", fs=8.1, bold=True)

    # 3. Wind & Atmospheric Gust Force (-x / bidirectional gusts)
    fw_tail = p_origin + (car_s0 - 1.85) * u_vec + (wheel_h + 1.15) * n_vec
    fw_tip = p_origin + (car_s0 + 0.05) * u_vec + (wheel_h + 1.15) * n_vec
    _arr(ax, fw_tail, fw_tip, col="#7C3AED", lw=2.6, ls="--", ms=15)
    _arr(ax, fw_tip - 0.25 * u_vec, fw_tail + 0.15 * u_vec, col="#7C3AED", lw=2.0, ls=":", ms=13)
    _pill(ax, fw_tail[0] - 0.15, fw_tail[1] + 0.56,
          r"Weather & Wind $M_{\mathrm{eff}}\,d_{\mathrm{wind}}(t)$" + "\n" + r"Direction + Gusts + Turbulence",
          col="#5B21B6", bg="#F5F3FF", ec="#A78BFA", fs=8.1, bold=True)

    # 4. Track Grade & Gravity Forces
    cg = p_origin + (car_s0 + 2.55) * u_vec + (wheel_h + 0.72) * n_vec
    _arr(ax, cg, cg + np.array([0.0, -1.35]), col="#DC2626", lw=2.3, ms=14)
    _pill(ax, cg[0] + 0.90, cg[1] - 1.68,
          r"Gravity $Mg$ & Grade $F_{\mathrm{grade}} = Mg\sin\theta(x)$",
          col="#991B1B", bg="#FEF2F2", ec="#F87171", fs=8.0, bold=True)

    # 5. Curve/CWR Resistance & Adhesion Limit under rear wheel
    fa_start = p_origin + (car_s0 + 0.05) * u_vec + (wheel_h + 0.45) * n_vec
    fa_end = fa_start - 1.55 * u_vec
    _arr(ax, fa_start, fa_end, col="#D97706", lw=2.6, ms=15)
    _pill(ax, fa_end[0] - 0.35, fa_end[1] - 0.36,
          r"Curve & CWR Stress $F_{\mathrm{curve}}(x)\,\kappa_{\mathrm{cwr}}(T_{\mathrm{amb}})$",
          col="#92400E", bg="#FFFBEB", ec="#FBBF24", fs=8.0, bold=True)

    # Adhesion callout at wheel contact
    w_mark = p_origin + (car_s0 + car_len - 1.05) * u_vec
    ax.plot([w_mark[0], w_mark[0] + 0.75], [w_mark[1], w_mark[1] - 0.58], color="#B45309", lw=1.4, ls=":")
    _pill(ax, w_mark[0] + 1.85, w_mark[1] - 0.78,
          r"Wheel-Rail Adhesion $\mu_{\mathrm{adh}}(x)$" + "\n" + r"$a_{\max},b_{\max} \propto \mu_{\mathrm{adh}}\,M_0/M$",
          col="#78350F", bg="#FEF3C7", ec="#F59E0B", fs=7.9, bold=True)

    fig.tight_layout(pad=0.25)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_contract_searching_tree(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 6.8), dpi=300)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    # Root Node
    _card(ax, 0.25, 2.55, 2.45, 1.75, "Root: PPO Proposal",
          [r"$p_{\mathrm{task}}=(v_{\mathrm{cmd}},a_{\mathrm{st}},b_{\mathrm{ld}})$",
           r"Evaluate $h_{\mathrm{sp}},h_{\mathrm{sm}},h_{\mathrm{jk}},h_E,h_T$",
           r"Trigger if $\min_i h_i < 0$"],
          "#EFF6FF", "#1D4ED8", "#1E40AF", ft=9.2, fb=7.8)

    # 4 Recovery Contracts (Level 1)
    c_specs = [
        (5.25, r"$\mathcal{C}_1$: Speed Scissor", r"Clamp $v \leq v_{\mathrm{ceil}}-r_{99}$", "#F1F5F9", "#64748B", "#475569", True),
        (3.65, r"$\mathcal{C}_2$: Comfort / Jerk", r"Damp $a_{\mathrm{st}},b_{\mathrm{ld}}$ ramps", "#ECFDF5", "#059669", "#047857", True),
        (2.05, r"$\mathcal{C}_3$: Deadline Rec.", r"Lift $v$ vs headwind", "#FEF2F2", "#DC2626", "#B91C1C", False),
        (0.45, r"$\mathcal{C}_4$: Eco/Regen Coast", r"Coast / $O(Z)\;v_{\mathrm{eco,ub}}$ cap", "#F0FDF4", "#15803D", "#166534", True),
    ]
    for cy, title, sub, fc, ec, hc, active in c_specs:
        _card(ax, 3.45, cy, 2.55, 1.15, title, [sub], fc, ec, hc, ft=8.9, fb=7.7)
        _arr(ax, (2.70, 3.42), (3.45, cy + 0.58), col=hc, lw=1.7)

    # Level 2: Lambda Blends + Reachable Tube Pruning
    lam_y_offsets = [0.92, 0.57, 0.22]
    lam_vals = [r"$\lambda=0.55$", r"$\lambda=0.85$", r"$\lambda=1.00$"]
    # Illustrate concrete example: worst_prop = energy or smoothness -> some pruned, one winner
    node_statuses = [
        # C1
        [("Pruned (margin)", "#E2E8F0", "#94A3B8", False),
         ("Pruned (tube)", "#E2E8F0", "#94A3B8", False),
         ("Feasible", "#DBEAFE", "#3B82F6", True)],
        # C2
        [("Pruned (margin)", "#E2E8F0", "#94A3B8", False),
         ("Feasible", "#DBEAFE", "#3B82F6", True),
         ("Feasible", "#DBEAFE", "#3B82F6", True)],
        # C3 (skipped on eco trigger / fails energy)
        [("Pruned (conflict)", "#FEE2E2", "#F87171", False),
         ("Pruned (conflict)", "#FEE2E2", "#F87171", False),
         ("Pruned (conflict)", "#FEE2E2", "#F87171", False)],
        # C4 (Winner on lambda=1.00)
        [("Pruned (margin)", "#E2E8F0", "#94A3B8", False),
         ("Feasible", "#DBEAFE", "#3B82F6", True),
         ("WINNER (Best UCB)", "#DCFCE7", "#16A34A", True)],
    ]

    for c_idx, (cy, *_rest) in enumerate(c_specs):
        for l_idx, (dy, ltxt) in enumerate(zip(lam_y_offsets, lam_vals)):
            ny = cy + dy - 0.14
            st_lbl, bg_c, bd_c, ok = node_statuses[c_idx][l_idx]
            box = FancyBboxPatch(
                (6.75, ny), 2.35, 0.30, boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=bg_c, edgecolor=bd_c, linewidth=(2.0 if "WINNER" in st_lbl else 1.2), zorder=3,
            )
            ax.add_patch(box)
            _arr(ax, (6.00, cy + 0.58), (6.75, ny + 0.15),
                 col=("#16A34A" if "WINNER" in st_lbl else ("#64748B" if not ok else "#2563EB")),
                 lw=(2.2 if "WINNER" in st_lbl else 1.2), ls=("-" if ok else ":"), ms=10.5)
            txt_c = "#14532D" if "WINNER" in st_lbl else ("#475569" if not ok else "#1E3A8A")
            ax.text(7.92, ny + 0.15, f"{ltxt}: {st_lbl}", ha="center", va="center",
                    fontsize=7.1, fontweight=("bold" if ok else "normal"), color=txt_c, zorder=4)

    # Level 3: Pruning + Tchebycheff + UCB Box
    _card(ax, 9.65, 2.55, 2.35, 2.95, "Pruning & UCB Scoring",
          [r"1. Hierarchy $\mathcal{W}_1,\mathcal{W}_2$ Filter",
           r"2. 8-Step Leaf Tube Check:",
           r"$v_{\mathrm{leaf}} + r_{99,\mathrm{leaf}} \leq v_{\mathrm{ceil}}$",
           r"3. Minimax + Leaf Energy:",
           r"$r_{\mathrm{L1}}^{\mathrm{si}} + 0.35\gamma r_{\mathrm{leaf}}^{\mathrm{si}} + 0.4\,\mathrm{UCB}$"],
          "#FAF5FF", "#9333EA", "#7E22CE", ft=8.9, fb=7.5)

    _arr(ax, (9.10, 0.77), (9.65, 3.20), col="#16A34A", lw=2.3, ms=14)
    _arr(ax, (9.10, 4.24), (9.65, 4.20), col="#3B82F6", lw=1.3, ms=11)
    _arr(ax, (9.10, 5.56), (9.65, 4.85), col="#3B82F6", lw=1.3, ms=11)

    # Level 4: Final 2-Step Verification + Output Latch
    _card(ax, 12.40, 2.75, 1.90, 2.55, "Verification & Latch",
          ["2-Step 8-Feature",
           "Neural Emu Check",
           "2-Step Hysteresis",
           "Latch (Clear on",
           r"$h_{i^*} \geq 0$)",
           r"Output $p_{\mathrm{safe}}$"],
          "#ECFDF5", "#059669", "#047857", ft=8.9, fb=7.6)

    _arr(ax, (12.00, 4.02), (12.40, 4.02), col="#059669", lw=2.2, ms=14)

    ax.text(7.25, 6.55, "Step-Adaptive Safe Contract Searching Tree & Forward 8-Step Tube Pruning",
            ha="center", va="center", fontsize=11.0, fontweight="bold", color="#0F172A")

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
        train_number=8311, n_legs_max=0, seed=9000, ep_idx=0, harsh_weather_mode="mixed"
    )
    sc_ext = fetch_real_world_multi_station_scenario(
        train_number=8311, n_legs_max=0, seed=9000 + 11 * 29, ep_idx=11, harsh_weather_mode="mixed"
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
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.2), dpi=220, sharex="col")
    for col_idx, (title_tag, scen_obj, tr_map) in enumerate([
        ("Raw Open-Meteo Baseline (4 Stops, 16.15 km, Train #8311)", sc_base, trajs_base),
        ("Extreme Noise & Storm Slush (4 Stops, 15.60 km, mu_min=0.62)", sc_ext, trajs_ext),
    ]):
        tr_sh = tr_map["5. UCB-DMPS"]["traj"]
        tr_un = tr_map["2. PPO Unsh"]["traj"]
        tr_bb = tr_map["8. Bilbao'25"]["traj"]
        x_sh = tr_sh["x"]
        xg_c, vc_g = _compute_scenario_ceil_grid(scen_obj)

        ax_t = axes[0, col_idx]
        ax_t.fill_between(x_sh, np.maximum(0.0, tr_sh["v_emu"] - tr_sh["env_rad"]),
                          tr_sh["v_emu"] + tr_sh["env_rad"], color="limegreen", alpha=0.18, label="Safe Envelope")
        ax_t.fill_between(x_sh, np.maximum(0.0, tr_sh["v_emu"] - tr_sh["scissor_rad"]),
                          tr_sh["v_emu"] + tr_sh["scissor_rad"], color="gold", alpha=0.35, label="99% Scissor Band")
        ax_t.plot(xg_c, vc_g, "k:", lw=1.2, alpha=0.8, label="Spatial Ceiling")
        ax_t.plot(tr_bb["x"], tr_bb["v"], color="goldenrod", ls="-.", lw=1.3, alpha=0.85, label="10. Bilbao'25")
        ax_t.plot(tr_un["x"], tr_un["v"], color="seagreen", ls="--", lw=1.4, label="4. PPO Unshielded")
        ax_t.plot(x_sh, tr_sh["v"], color="crimson", ls="-", lw=1.8, label="7. UCB-DMPS Cosine")
        im = tr_sh["intervened"]
        if np.any(im):
            ax_t.scatter(x_sh[im], tr_sh["v"][im], color="crimson", edgecolor="white", s=16, zorder=6, label="Contract Intervention")
        ax_t.set_title(title_tag, fontsize=10.5, fontweight="bold")
        ax_t.set_ylabel("Speed (m/s)")
        ax_t.grid(True, alpha=0.28)
        ax_t.legend(fontsize=7.3, loc="lower left", ncol=2)

        ax_b = axes[1, col_idx]
        ax_b.axhline(0.0, color="gray", lw=0.8)
        ax_b.plot(x_sh, tr_sh["wind_true"], color="navy", lw=1.3, label="True Wind d_wind (m/s^2)")
        ax_b.plot(x_sh, tr_sh["u_act"], color="crimson", lw=1.2, alpha=0.82, label="MPC Effort u_act")
        ax_m = ax_b.twinx()
        ax_m.step(x_sh, tr_sh["mass_kg"] / 1000.0, color="purple", lw=1.5, ls="--", label="True Mass M (t)")
        ax_m.plot(x_sh, tr_sh["mass_hat_kg"] / 1000.0, color="darkorchid", lw=1.3, label="MAP M_hat (t)")
        ax_b.set_xlabel("Cumulative Position x (m)")
        ax_b.set_ylabel("Wind (m/s^2) / Control u_act", color="navy")
        ax_m.set_ylabel("Mass (tonnes)", color="purple")
        ax_b.set_ylim(-1.1, 1.1)
        ax_b.grid(True, alpha=0.28)
        l1, lb1 = ax_b.get_legend_handles_labels()
        l2, lb2 = ax_m.get_legend_handles_labels()
        ax_b.legend(l1 + l2, lb1 + lb2, fontsize=7.2, loc="lower left", ncol=2)

    fig.tight_layout()
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
                 ("Zhao (2025)", "SSA-DRL'25")]
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
    ax_a[0, 0].set_title("1. Speed Trajectories v(x) — All Methods (Real #8311)", fontweight="bold", fontsize=10.5)
    ax_a[0, 0].set_xlabel("Position x (m)")
    ax_a[0, 0].set_ylabel("Speed (m/s)")
    ax_a[0, 0].legend(fontsize=6.3, ncol=2, loc="lower left")
    ax_a[0, 0].grid(True, alpha=0.28)

    tr_dmps = trajs_base["5. UCB-DMPS"]["traj"]
    ax_a[0, 1].plot(tr_dmps["t"], tr_dmps["accel"], color="crimson", lw=1.3, label="Accel a(t) [d=2]")
    ax_a[0, 1].plot(tr_dmps["t"], tr_dmps["jerk"], color="teal", alpha=0.65, lw=1.0, label="Jerk j(t) [d=3]")
    ax_a[0, 1].axhline(-0.92, color="darkorange", ls=":", label="Comfort Brake (-0.92)")
    ax_a[0, 1].axhline(0.48, color="navy", ls="--", alpha=0.6, label="Jerk Bound (0.48)")
    ax_a[0, 1].axhline(-0.48, color="navy", ls="--", alpha=0.6)
    ax_a[0, 1].set_title("2. Smoothness: Accel a(t) & Jerk j(t) (UCB-DMPS)", fontweight="bold", fontsize=10.5)
    ax_a[0, 1].set_xlabel("Clock Time t (s)")
    ax_a[0, 1].set_ylabel("m/s^2  /  m/s^3")
    ax_a[0, 1].legend(fontsize=7.2, loc="lower left")
    ax_a[0, 1].grid(True, alpha=0.28)

    ax_a[0, 2].step(tr_dmps["t"], tr_dmps["mass_kg"] / 1000.0, color="purple", lw=2.0, label="True Mass M(t)")
    ax_a[0, 2].plot(tr_dmps["t"], tr_dmps["mass_hat_kg"] / 1000.0, color="darkorchid", ls="--", lw=1.5, label="Causal MAP M_hat")
    ax_w = ax_a[0, 2].twinx()
    ax_w.plot(tr_dmps["t"], tr_dmps["wind_true"], color="navy", alpha=0.5, lw=1.2, label="True Wind d(t)")
    ax_w.plot(tr_dmps["t"], tr_dmps["wind_hat"], color="dodgerblue", ls=":", lw=1.2, label="DOB d_hat(t)")
    ax_a[0, 2].set_title("3. Causal Mass ID & Weather Observer", fontweight="bold", fontsize=10.5)
    ax_a[0, 2].set_xlabel("Clock Time t (s)")
    ax_a[0, 2].set_ylabel("Mass (t)", color="purple")
    ax_w.set_ylabel("Wind Accel (m/s^2)", color="navy")
    l1, lb1 = ax_a[0, 2].get_legend_handles_labels()
    l2, lb2 = ax_w.get_legend_handles_labels()
    ax_a[0, 2].legend(l1 + l2, lb1 + lb2, fontsize=6.8, loc="upper left")
    ax_a[0, 2].grid(True, alpha=0.28)

    xp = np.arange(n_meth)
    bw = 0.26
    metric_panels = [
        (ax_a[1, 0], "time", "4. Mean |Deadline Error| (s)", "Seconds (Lower = Better)"),
        (ax_a[1, 1], "stop_cm", "5. Mean |Stop Error| (cm)", "Centimeters (Lower = Better)"),
        (ax_a[1, 2], "smooth", "6. Trajectory SmoothLoss E[rho2^2+rho3^2]", "SmoothLoss (Lower = Better)"),
        (ax_a[2, 0], "energy", "7. Specific Traction Energy (Wh/t·km)", "Wh/t·km (Lower = Better)"),
        (ax_a[2, 1], "loss", "8. Composite Multi-Objective LossScore", "LossScore (Lower = Better)"),
    ]
    for ax_p, mkey, ttl, ylbl in metric_panels:
        vb = [csv_real_b[k][mkey] for k in full_method_keys]
        ve = [csv_real_e[k][mkey] for k in full_method_keys]
        vs = [csv_synth[k][mkey] for k in full_method_keys]
        ax_p.bar(xp - bw, vb, bw, color="mediumseagreen", label="Real Raw-API (15)")
        ax_p.bar(xp, ve, bw, color="royalblue", label="Real Extreme (15)")
        ax_p.bar(xp + bw, vs, bw, color="crimson", alpha=0.85, label="Synthetic (30)")
        ax_p.set_title(ttl, fontweight="bold", fontsize=10.5)
        ax_p.set_ylabel(ylbl)
        ax_p.set_xticks(xp)
        ax_p.set_xticklabels(short_lbls, rotation=40, ha="right", fontsize=6.8)
        ax_p.legend(fontsize=7.3)
        ax_p.grid(True, axis="y", alpha=0.28)

    # Panel 9: Latency (Mean, P99, Max on Real Extreme vs 500 us hard budget)
    ax_lat = ax_a[2, 2]
    m_lat = [csv_real_e[k]["mean_us"] for k in full_method_keys]
    p_lat = [csv_real_e[k]["p99_us"] for k in full_method_keys]
    mx_e = [csv_real_e[k]["max_us"] for k in full_method_keys]
    ax_lat.bar(xp - bw, m_lat, bw, color="teal", label="Mean Total (Real Ext)")
    ax_lat.bar(xp, p_lat, bw, color="darkorange", label="P99 Total (Real Ext)")
    ax_lat.bar(xp + bw, mx_e, bw, color="purple", alpha=0.78, label="Max (Real Ext)")
    ax_lat.axhline(400.0, color="red", ls="--", lw=1.4, label="400 us target (methods 2-7)")
    ax_lat.set_yscale("log")
    ax_lat.set_title("9. Step Inference Latency (us, log scale)", fontweight="bold", fontsize=10.5)
    ax_lat.set_ylabel("Microseconds")
    ax_lat.set_xticks(xp)
    ax_lat.set_xticklabels(short_lbls, rotation=40, ha="right", fontsize=6.8)
    ax_lat.legend(fontsize=7.1, loc="upper right")
    ax_lat.grid(True, axis="y", alpha=0.28)

    fig_a.tight_layout()
    fig_a.savefig(str(fig_dir / "appendix_full_benchmark.png"), dpi=200)
    plt.close(fig_a)

    # ------------------------------------------------------------------
    # Appendix Figure 2: Non-Disturbance vs Heavy Disturbance (Many Methods)
    # ------------------------------------------------------------------
    fig_d, ax_d = plt.subplots(2, 2, figsize=(15.2, 8.6), dpi=220, sharex="col")
    plot_9 = [s for s in specs if s[0] != "6. 1-Step DCBF"]
    for c_i, (lbl, sc_obj, tmap) in enumerate([
        ("Undisturbed / Clean Synthetic Scenario (3 Stops, Clear-Dry)", sc_clean, trajs_clean),
        ("Heavy Disturbance Real-World Extreme Scenario (4 Stops, Storm Gales & mu=0.62)", sc_ext, trajs_ext),
    ]):
        for nm, _, _, _, col, ls in plot_9:
            ax_d[0, c_i].plot(tmap[nm]["traj"]["x"], tmap[nm]["traj"]["v"], color=col, ls=ls, lw=1.4, label=nm)
        xg_d, vc_d = _compute_scenario_ceil_grid(sc_obj)
        ax_d[0, c_i].plot(xg_d, vc_d, "k:", lw=1.2, label="Ceiling")
        ax_d[0, c_i].set_title(lbl, fontweight="bold", fontsize=10.0)
        ax_d[0, c_i].set_ylabel("Speed v(x) [m/s]")
        ax_d[0, c_i].legend(fontsize=6.8, ncol=2, loc="lower left")
        ax_d[0, c_i].grid(True, alpha=0.28)

        tr7 = tmap["5. UCB-DMPS"]["traj"]
        ax_d[1, c_i].plot(tr7["x"], tr7["wind_true"], color="navy", lw=1.4, label="Wind d_wind (m/s^2)")
        ax_d[1, c_i].plot(tr7["x"], tr7["wind_hat"], color="dodgerblue", ls=":", lw=1.3, label="DOB d_hat")
        ax_d[1, c_i].plot(tr7["x"], tr7["u_act"], color="crimson", alpha=0.75, lw=1.1, label="u_act (UCB-DMPS)")
        ax_d[1, c_i].set_xlabel("Track Position x (m)")
        ax_d[1, c_i].set_ylabel("Disturbance & Control")
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
