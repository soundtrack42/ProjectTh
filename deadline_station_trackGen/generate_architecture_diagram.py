"""
Generate a publication-grade architecture diagram for the final multi-station
train control architecture:
  - 4-criterion event-triggered mid-horizon preemption gate -> 18-D Macro-Step PPO pilot
  - Hierarchical differential shield + Step-Adaptive Safe Contract Searching Tree
    (Tchebycheff multi-objective scoring + persistent Discounted-UCB + honest hysteresis latch)
  - Dual C^2 shaper (Cosine S-curve / Non-Cosine Hermite Soft-curve + fuzzy softening)
  - Fast 5x3 state-augmented LTV-MPC (closed-form 3x3 Cramer QP + DOB compensation)
  - Nonlinear train & weather-coupled plant (gusts, wet/snow adhesion, CWR/Cu temp drag)
  - Multi-lag differential MAP Mass ID (k<64), Lyapunov-adaptive DOB (k>=8), and
    full 8-feature Heteroscedastic Neural Physics Emulator (99% Gaussian Scissor Tube)

Run with:
  .venv/bin/python3 deadline_station_trackGen/generate_architecture_diagram.py
"""
from __future__ import annotations

import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    facecolor: str,
    edgecolor: str,
    title_color: str,
    header_bg: str,
    fontsize_title: float = 9.4,
    fontsize_body: float = 8.0,
) -> tuple[float, float, float, float]:
    """Draws a crisp rounded module card with a tinted header band and stacked body lines."""
    outer = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.14",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.8,
        zorder=2,
    )
    ax.add_patch(outer)

    header_h = min(0.45, 0.27 * h)
    header = FancyBboxPatch(
        (x + 0.02, y + h - header_h - 0.01), w - 0.04, header_h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=header_bg,
        edgecolor="none",
        zorder=3,
    )
    ax.add_patch(header)

    ax.text(
        x + w / 2.0,
        y + h - header_h / 2.0 - 0.01,
        title,
        ha="center",
        va="center",
        fontsize=fontsize_title,
        fontweight="bold",
        color=title_color,
        zorder=4,
    )

    body_top = y + h - header_h - 0.13
    body_bottom = y + 0.09
    n = len(lines)
    if n > 0:
        span = max(0.2, body_top - body_bottom)
        step = span / max(1, n)
        for idx, line in enumerate(lines):
            cy = body_top - (idx + 0.45) * step
            ax.text(
                x + w / 2.0,
                cy,
                line,
                ha="center",
                va="center",
                fontsize=fontsize_body,
                color="#1E293B",
                linespacing=1.18,
                zorder=4,
            )
    return x, y, w, h


def _arrow(
    ax,
    p1: tuple[float, float],
    p2: tuple[float, float],
    color: str = "#334155",
    lw: float = 1.8,
    style: str = "-|>",
    rad: float = 0.0,
    ls: str = "-",
    zorder: int = 5,
):
    arr = FancyArrowPatch(
        p1, p2,
        arrowstyle=style,
        mutation_scale=14.0,
        linewidth=lw,
        linestyle=ls,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arr)


def _label_pill(
    ax,
    x: float,
    y: float,
    text: str,
    color: str = "#0F172A",
    bg: str = "#FFFFFF",
    edge: str = "#CBD5E1",
    fontsize: float = 7.7,
    bold: bool = False,
):
    ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=fontsize,
        fontweight=("bold" if bold else "normal"),
        color=color,
        bbox=dict(boxstyle="round,pad=0.22", facecolor=bg, edgecolor=edge, linewidth=0.95),
        zorder=7,
    )


def build_diagram(out_dir: str) -> list[str]:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    fig, ax = plt.subplots(figsize=(15.4, 9.2), dpi=300)
    ax.set_xlim(0.0, 15.4)
    ax.set_ylim(0.0, 9.2)
    ax.axis("off")

    # Subtle background grouping bands with generous inter-tier gutters
    tier_super = FancyBboxPatch(
        (0.32, 5.75), 14.76, 2.58,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.1, linestyle="--", zorder=0,
    )
    tier_exec = FancyBboxPatch(
        (0.32, 2.82), 14.76, 2.38,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor="#F0FDF4", edgecolor="#BBF7D0", linewidth=1.1, linestyle="--", zorder=0,
    )
    tier_est = FancyBboxPatch(
        (0.32, 0.25), 14.76, 1.98,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor="#FAF5FF", edgecolor="#E9D5FF", linewidth=1.1, linestyle="--", zorder=0,
    )
    for p in (tier_super, tier_exec, tier_est):
        ax.add_patch(p)

    # Title and subtitle
    ax.text(
        7.70, 8.88,
        "Closed-Loop Multi-Station Train Control: Macro RL, Hierarchical Contract Tree Shielding & Fast Lifted MPC",
        ha="center", va="center", fontsize=12.4, fontweight="bold", color="#0F172A",
    )
    ax.text(
        7.70, 8.56,
        r"Merging ATP Safety Guarantees (Differential Invariance $\mathcal{W}_3 \subset \mathcal{W}_2 \subset \mathcal{W}_1$ + 99% Tube) and ATO Objectives (Punctuality, Smoothness, Energy, Docking)",
        ha="center", va="center", fontsize=8.8, color="#475569",
    )

    # Clean tier headers placed cleanly in open channels (zero crossing arrows!)
    ax.text(
        0.52, 8.18,
        r"SUPERVISORY LAYER (Macro Window $M=8$ / $4.0\,\mathrm{s}$ Cruise, $M=1$ Landing, or Event-Preempted)",
        ha="left", va="center", fontsize=7.9, fontweight="bold", color="#1E3A8A",
    )
    ax.text(
        9.60, 5.02,
        r"MICRO-STEP EXECUTION & PLANT LAYER ($\Delta t = 0.5\,\mathrm{s}$ | Core Shaper+MPC: $11.8$–$13.1\,\mu\mathrm{s}$ | Total Shielded: $27.1$–$28.5\,\mu\mathrm{s}$)",
        ha="center", va="center", fontsize=7.8, fontweight="bold", color="#065F46",
    )
    ax.text(
        10.22, 2.05,
        "ONLINE SYSTEM IDENTIFICATION, DISTURBANCE ESTIMATION & UNCERTAINTY TUBE LAYER",
        ha="center", va="center", fontsize=7.65, fontweight="bold", color="#581C87",
    )

    # =========================================================================
    # TOP ROW: Supervisory Layer (y: 5.92 to 7.92)
    # Dataflow: Preemption Gate (1) -> Macro PPO Pilot (2) -> Contract Tree Shield (3)
    # =========================================================================
    _draw_box(
        ax, 0.52, 5.92, 3.60, 1.98,
        "1. Event-Triggered Preemption Gate",
        [
            r"Interrupts $M=8$ macro hold if any criterion fires:",
            r"1) Gust shift $|\hat{d}_k - \hat{d}_{\mathrm{plan}}| > 0.145\,\mathrm{m/s^2}$",
            r"2) Directional CUSUM drift $|S_{\mathrm{cusum}}| > 0.22\,\mathrm{m/s^2}$",
            r"3) Fuzzy arrival satisfaction $\mu_{\min} < 0.15$",
            r"4) Remaining $M$-step tube breaches $v_{\mathrm{lim}} + 0.02$",
        ],
        facecolor="#FFF7ED", edgecolor="#EA580C", header_bg="#FFEDD5", title_color="#9A3412",
        fontsize_body=7.8,
    )

    _draw_box(
        ax, 4.92, 5.92, 3.60, 1.98,
        r"2. Macro-Step PPO Pilot ($\pi_\theta$)",
        [
            r"18-D observation (forward slope + adhesion aware):",
            r"kinematic pace hint, mass $\hat{M}$, DOB $\hat{d}_k$,",
            r"grade lookahead (+150/350m), speed ceilings, $\mu_{\mathrm{adh}}$",
            r"Outputs $p_{\mathrm{task}} = (v_{\mathrm{cmd}}, a_{\mathrm{start}}, b_{\mathrm{land}})$",
            r"held over $M=8$ cruise ($M=1$ landing) unless preempted",
        ],
        facecolor="#EFF6FF", edgecolor="#1D4ED8", header_bg="#DBEAFE", title_color="#1E3A8A",
        fontsize_body=7.8,
    )

    _draw_box(
        ax, 9.32, 5.92, 5.56, 1.98,
        "3. Hierarchical Differential & Contract Tree Shield (UCB-DMPS)",
        [
            r"Hierarchy $\mathcal{W}_3$ ($j\leq 0.48, a\leq 0.92$) $\subset \mathcal{W}_2$ $\subset \mathcal{W}_1$ ($v_{\mathrm{ceil}} - r_{99}$)",
            r"Evaluates pressures $(\rho_1,\rho_2,\rho_3,\rho_E,\rho_T)$ + forward 'on-track' zone/stop slack",
            r"Safe Contract Searching Tree (depth $M_{\mathrm{rem}}$): evaluates ALL 4 contracts $\{\mathcal{C}_1..\mathcal{C}_4\}$,",
            r"prunes via full 99% leaf tube; causal energy-excess UCB + Tchebycheff + 2-step check",
            r"2-step latch clears immediately on $h\geq 0$ (no post-clear lock / forced $v_{\max}$) $\rightarrow p_{\mathrm{safe}}$",
        ],
        facecolor="#FEF3C7", edgecolor="#B45309", header_bg="#FDE68A", title_color="#78350F",
        fontsize_body=7.65,
    )

    # Arrows inside Supervisory Layer (wide 0.80 gaps!)
    _arrow(ax, (4.12, 6.91), (4.92, 6.91), color="#EA580C", lw=1.8)
    _label_pill(ax, 4.52, 7.22, "Replan\nTrigger", bg="#FFF7ED", edge="#FB923C", color="#9A3412", fontsize=7.3)

    _arrow(ax, (8.52, 6.91), (9.32, 6.91), color="#1D4ED8", lw=1.8)
    _label_pill(ax, 8.92, 7.20, r"$p_{\mathrm{task}}$", bg="#EFF6FF", edge="#60A5FA", color="#1E3A8A", fontsize=7.9, bold=True)

    # =========================================================================
    # MIDDLE ROW: Execution & Plant Layer (y: 3.00 to 4.78)
    # =========================================================================
    _draw_box(
        ax, 0.52, 3.00, 4.20, 1.78,
        r"4. Dual $C^2$ Trajectory Shaper",
        [
            r"Mode A: Phase-tracked $C^2$ Half-Cosine S-Curve",
            r"Mode B: Non-Cosine $C^2$ Hermite Soft-Curve ($j=0$ joins)",
            r"Fuzzy tracking softening $\alpha(\mu_y)$ + dynamic",
            r"braking boundary $x_{\mathrm{brake}} \rightarrow$ micro-ref $v_{\mathrm{ref}}[k:k+5]$",
        ],
        facecolor="#ECFDF5", edgecolor="#047857", header_bg="#D1FAE5", title_color="#064E3B",
        fontsize_body=7.85,
    )

    _draw_box(
        ax, 5.52, 3.00, 4.36, 1.78,
        r"5. Fast $5\times 3$ Lifted LTV-MPC ($3\times 3$ Cramer QP)",
        [
            r"Lifted state $z_k = [v_k, u_{\mathrm{act}}(k-1)]^\top$ absorbs $0.28^4$ lag",
            r"Horizon $N_p=5, N_c=3$ + upper-triangular rollout",
            r"Closed-form $3\times 3$ Cramer QP + Gershgorin clamp",
            r"+ DOB feedforward ($-\hat{d}_k/\eta_u$) $\rightarrow u_{\mathrm{cmd}} \in [-1, 1]$",
        ],
        facecolor="#F0FDFA", edgecolor="#0F766E", header_bg="#CCFBF1", title_color="#134E4A",
        fontsize_body=7.85,
    )

    _draw_box(
        ax, 10.68, 3.00, 4.20, 1.78,
        "6. Nonlinear Train & Weather-Coupled Plant",
        [
            r"First-order lag: $u_{\mathrm{act}}(k) = 0.72\,u_{\mathrm{cmd}}(k) + 0.28\,u_{\mathrm{act}}(k-1)$",
            r"Mass $M \in [75.3, 99.2]\,\mathrm{t}$ + Davis/grade/curve forces",
            "5 Weather/Thermal Regimes: Storm Gusts, Wet/Snow",
            r"Polach adhesion $\mu_{\mathrm{adh}}$, $\rho_{\mathrm{air}}(T)$, CWR/Cu temp drag",
        ],
        facecolor="#FEF2F2", edgecolor="#B91C1C", header_bg="#FEE2E2", title_color="#7F1D1D",
        fontsize_body=7.8,
    )

    # Arrows inside Execution Layer (wide 0.80 gaps!)
    _arrow(ax, (4.72, 3.89), (5.52, 3.89), color="#047857", lw=1.8)
    _label_pill(ax, 5.12, 4.18, r"$v_{\mathrm{ref}}$", bg="#ECFDF5", edge="#10B981", color="#065F46", fontsize=7.9, bold=True)

    _arrow(ax, (9.88, 3.89), (10.68, 3.89), color="#0F766E", lw=1.8)
    _label_pill(ax, 10.28, 4.18, r"$u_{\mathrm{cmd}}$", bg="#F0FDFA", edge="#14B8A6", color="#115E59", fontsize=7.9, bold=True)

    # =========================================================================
    # BOTTOM ROW: Estimation & Tube Layer (y: 0.40 to 1.84)
    # =========================================================================
    _draw_box(
        ax, 0.52, 0.40, 6.85, 1.44,
        r"7. Causal 3-Block 2nd-Diff MAP Mass ID ($k < 64$) & Adaptive DOB ($k \geq 8$)",
        [
            r"Telescoping multi-width $(1,2,4,8)$ / $(4,8,12)$ block IRLS with per-step lag $f_k$ & noise gating",
            r"$k \geq 8$: Adaptive DOB $\hat{d}_k$ tracks storm gusts, Polach adhesion drops, & CWR/Cu thermal shifts",
            r"Cancels constant + linear wind drift; converges to $1.4\,\mathrm{kg}$ MAE clean / $774\,\mathrm{kg}$ extreme",
        ],
        facecolor="#F5F3FF", edgecolor="#6D28D9", header_bg="#EDE9FE", title_color="#4C1D95",
        fontsize_body=7.7,
    )

    _draw_box(
        ax, 7.98, 0.40, 6.90, 1.44,
        r"8. 8-Feature Heteroscedastic Emulator ($\hat{f}_\theta$) & 99% Scissor Tube",
        [
            r"All 8 inputs $(dv, v, u, F_{\mathrm{ext}}, v^2, M, \rho_{\mathrm{air}}, \mu_{\mathrm{adh}})$ active (Gaussian NLL + $2.5\times$ shift weight)",
            r"Propagates contractive variance $\sigma_k^2 = \gamma_\sigma \sigma_{k-1}^2 + \hat{\sigma}_{\mathrm{step}}^2$ ($\gamma_\sigma = 0.65$)",
            r"Full $r_{99,k} = 2.576\,\sigma_k$ tube for Tree Leaf Pruning, 2-Step Check & Inductive Preemption",
        ],
        facecolor="#FAF5FF", edgecolor="#7E22CE", header_bg="#F3E8FF", title_color="#581C87",
        fontsize_body=7.7,
    )

    # =========================================================================
    # INTER-LAYER ROUTING ARROWS (all inside dedicated open horizontal gutters)
    # =========================================================================
    # A. Shield p_safe -> Shaper (gutter y = 5.47, well above Execution tier y=5.20!)
    ax.plot([12.10, 12.10, 2.25, 2.25], [5.92, 5.47, 5.47, 4.78],
            color="#B45309", linewidth=2.0, zorder=4)
    _arrow(ax, (2.25, 4.94), (2.25, 4.78), color="#B45309", lw=2.0)
    _label_pill(
        ax, 7.70, 5.47,
        r"Certified Safe Targets  $p_{\mathrm{safe}} = (v_{\mathrm{safe}}, a_{\mathrm{safe}}, b_{\mathrm{safe}})$  [86.5% Pass-Through / 13.5% Shield-Active]",
        bg="#FFFBEB", edge="#D97706", color="#92400E", bold=True, fontsize=7.9,
    )

    # B. Shaper tracking satisfaction mu_y -> Event Preemption Gate (clean vertical channel x=3.10..3.60)
    ax.plot([3.35, 3.35], [4.78, 5.92], color="#EA580C", linewidth=1.6, linestyle=":", zorder=4)
    _arrow(ax, (3.35, 5.76), (3.35, 5.92), color="#EA580C", lw=1.6)
    _label_pill(
        ax, 3.35, 5.47,
        r"$\mu_y$",
        bg="#FFF7ED", edge="#FB923C", color="#9A3412", fontsize=7.6, bold=True,
    )

    # C. Plant telemetry -> Estimation & Tube Layer (gutter y = 2.52, clean above y=2.23 band)
    ax.plot([14.20, 14.20], [3.00, 2.52], color="#991B1B", linewidth=1.7, zorder=4)
    ax.plot([3.95, 14.20], [2.52, 2.52], color="#991B1B", linewidth=1.7, zorder=4)
    _arrow(ax, (3.95, 2.52), (3.95, 1.84), color="#991B1B", lw=1.7)
    _arrow(ax, (14.20, 2.52), (14.20, 1.84), color="#991B1B", lw=1.7)
    _label_pill(
        ax, 10.22, 2.52,
        r"Plant Telemetry  $(x_k, v_k, a_k, u_{\mathrm{act}}, \mathrm{dwell})$",
        bg="#FEF2F2", edge="#EF4444", color="#7F1D1D", fontsize=7.7,
    )

    # D. DOB d_hat + Mass M_hat feedforward -> Fast MPC (x = 6.25, clean gap left of telemetry pill)
    ax.plot([6.25, 6.25], [1.84, 3.00], color="#6D28D9", linewidth=1.7, linestyle="--", zorder=4)
    _arrow(ax, (6.25, 2.84), (6.25, 3.00), color="#6D28D9", lw=1.7)
    _label_pill(
        ax, 6.25, 2.52,
        r"$\hat{d}_k,\;\hat{M}_{\mathrm{locked}}$",
        bg="#F5F3FF", edge="#8B5CF6", color="#4C1D95", fontsize=7.6, bold=True,
    )

    # E. Mass ID / DOB left spine -> Event Preemption Gate & PPO Pilot (gutter x = 0.16)
    ax.plot([0.52, 0.16, 0.16], [1.12, 1.12, 6.91],
            color="#6D28D9", linewidth=1.6, linestyle="--", zorder=4)
    _arrow(ax, (0.16, 6.91), (0.52, 6.91), color="#6D28D9", lw=1.6)
    # Also route to PPO Pilot along y=5.70 inside supervisory margin? Cleanly tap at y=8.04 top spine to Box 2!
    ax.plot([0.16, 0.16, 6.72, 6.72], [6.91, 8.05, 8.05, 7.90],
            color="#6D28D9", linewidth=1.4, linestyle="--", zorder=4)
    _arrow(ax, (6.72, 8.02), (6.72, 7.90), color="#6D28D9", lw=1.4)
    _label_pill(
        ax, 0.16, 4.10,
        r"$\hat{M},\,\hat{d}_k$",
        bg="#F5F3FF", edge="#8B5CF6", color="#4C1D95", fontsize=7.5, bold=True,
    )

    # F. Scissor Tube right spine -> Contract Shield & Preemption Gate (gutter x = 15.22)
    ax.plot([14.88, 15.22, 15.22], [1.12, 1.12, 8.05],
            color="#7E22CE", linewidth=1.6, linestyle="--", zorder=4)
    _arrow(ax, (15.22, 6.91), (14.88, 6.91), color="#7E22CE", lw=1.6)
    ax.plot([15.22, 2.32, 2.32], [8.05, 8.05, 7.90],
            color="#7E22CE", linewidth=1.4, linestyle="--", zorder=4)
    _arrow(ax, (2.32, 8.02), (2.32, 7.90), color="#7E22CE", lw=1.4)
    _label_pill(
        ax, 15.22, 4.10,
        r"$r_{99,k}$",
        bg="#FAF5FF", edge="#A855F7", color="#581C87", fontsize=7.6, bold=True,
    )
    _label_pill(
        ax, 11.20, 8.05,
        r"Forward 99% Scissor Tube $(\hat{v}_{\mathrm{emu}} \pm 2.576\,\sigma_k)$",
        bg="#FAF5FF", edge="#A855F7", color="#581C87", fontsize=7.5,
    )

    plt.tight_layout(pad=0.25)

    local_out = os.path.join(out_dir, "contracts_hierarchical_shielding_arch.png")
    fig.savefig(local_out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    parent_out = os.path.abspath(os.path.join(out_dir, "..", "contracts_hierarchical_shielding_arch.png"))
    saved = [local_out]
    if os.path.dirname(parent_out) != os.path.abspath(out_dir):
        shutil.copy2(local_out, parent_out)
        saved.append(parent_out)
    return saved


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths = build_diagram(script_dir)
    for p in paths:
        print(f"Generated architecture diagram: {p}")
