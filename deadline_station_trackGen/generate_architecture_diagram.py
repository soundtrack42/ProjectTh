"""
Generates an extremely simple, classic IEEE/TikZ-style serif block diagram of the
ultimate closed-loop control process (Contracts + Hierarchical Shielding) for Report.tex.
"""
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Classic LaTeX / Computer Modern Serif typography (no AI-like sans-serif fonts)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.edgecolor": "none",
})


def draw_box(ax, x, y, w, h, title, lines, edge_color="#1f2937", fill_color="#ffffff", lw=1.5, title_color="#111827"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor=fill_color,
        edgecolor=edge_color,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(box)

    # Title
    n_lines = len(lines)
    if n_lines == 0:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=title_color, zorder=4)
    else:
        title_y = y + h * 0.72 if n_lines <= 2 else y + h * 0.77
        ax.text(x + w / 2, title_y, title, ha="center", va="center",
                fontsize=11.0, fontweight="bold", color=title_color, zorder=4)
        body_text = "\n".join(lines)
        body_y = y + h * 0.34 if n_lines <= 2 else y + h * 0.35
        ax.text(x + w / 2, body_y, body_text, ha="center", va="center",
                fontsize=9.8, color="#1f2937", linespacing=1.35, zorder=4)


def draw_arrow(ax, p1, p2, label="", label_offset=(0.0, 0.18), color="#1f2937", lw=1.4, ls="-", rad=0.0):
    conn = f"arc3,rad={rad}"
    arr = FancyArrowPatch(
        p1, p2,
        connectionstyle=conn,
        arrowstyle="-|>,head_length=6.5,head_width=3.8",
        color=color,
        linewidth=lw,
        linestyle=ls,
        zorder=5,
    )
    ax.add_patch(arr)
    if label:
        mx = 0.5 * (p1[0] + p2[0]) + label_offset[0]
        my = 0.5 * (p1[1] + p2[1]) + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=10.0, color=color, zorder=6)


def main():
    fig, ax = plt.subplots(figsize=(12.5, 4.4), dpi=300)
    ax.set_xlim(0.0, 12.5)
    ax.set_ylim(0.0, 4.4)
    ax.axis("off")

    # Top row: Forward control chain (y = 2.35 to 3.85)
    y_top = 2.30
    h_top = 1.50

    # 1. PPO Pilot
    draw_box(
        ax, x=0.35, y=y_top, w=2.05, h=h_top,
        title="1. PPO Pilot",
        lines=[
            r"Macro-step ($\Delta t_{\mathrm{M}} = 4\mathrm{s}$)",
            r"Outputs $p_{\mathrm{task}} = (v_{\mathrm{cmd}}, a, b)$",
        ],
        edge_color="#1e3a8a", fill_color="#f8fafc", lw=1.5, title_color="#1e3a8a"
    )

    # 2. Hierarchical + Contract Shield (Core Contribution - highlighted subtly)
    draw_box(
        ax, x=3.15, y=y_top - 0.10, w=2.95, h=h_top + 0.20,
        title="2. Hierarchical & Contract Shield",
        lines=[
            r"Orders $d=1..3$: $\rho_1(v), \rho_2(a), \rho_3(j) \leq 1$",
            r"Contracts $\mathcal{C}_1..\mathcal{C}_4$: Priority Latching",
            r"Projects $p_{\mathrm{task}} \mapsto p_{\mathrm{safe}}$ via UCB Tube",
        ],
        edge_color="#991b1b", fill_color="#fffafa", lw=1.8, title_color="#991b1b"
    )

    # 3. C^2 Cosine Shaper
    draw_box(
        ax, x=6.85, y=y_top, w=1.85, h=h_top,
        title=r"3. $C^2$ Cosine Shaper",
        lines=[
            r"S-curve & landing",
            r"horizon $v_{\mathrm{ref}}(k+1:k+5)$",
        ],
        edge_color="#1f2937", fill_color="#ffffff", lw=1.4
    )

    # 4. Fast LTV-MPC
    draw_box(
        ax, x=9.35, y=y_top, w=1.70, h=h_top,
        title="4. Fast LTV-MPC",
        lines=[
            r"Micro-step ($0.5\mathrm{s}$)",
            r"$3\times 3$ QP ($\sim 25\,\mu\mathrm{s}$)",
        ],
        edge_color="#065f46", fill_color="#f8fafc", lw=1.5, title_color="#065f46"
    )

    # 5. Train Plant
    draw_box(
        ax, x=11.45, y=y_top, w=0.85, h=h_top,
        title="Train\nPlant",
        lines=[],
        edge_color="#111827", fill_color="#f3f4f6", lw=1.5
    )

    # Forward arrows
    ym = y_top + h_top / 2.0
    draw_arrow(ax, (2.40, ym), (3.15, ym), label=r"$p_{\mathrm{task}}$", label_offset=(0.0, 0.22))
    draw_arrow(ax, (6.10, ym), (6.85, ym), label=r"$p_{\mathrm{safe}}$", label_offset=(0.0, 0.22), color="#991b1b")
    draw_arrow(ax, (8.70, ym), (9.35, ym), label=r"$v_{\mathrm{ref}}$", label_offset=(0.0, 0.22))
    draw_arrow(ax, (11.05, ym), (11.45, ym), label=r"$u_k$", label_offset=(0.0, 0.22))

    # Bottom row: Online Identification & Uncertainty Tube (Feedback path)
    y_bot = 0.40
    h_bot = 1.15
    draw_box(
        ax, x=3.40, y=y_bot, w=5.60, h=h_bot,
        title="5. Online Station Mass ID & Heteroscedastic Physics Emulator",
        lines=[
            r"Steps $k < 8$: Locks passenger mass $\hat{M}$  |  Steps $k \geq 8$: Tracks weather/wind $\hat{d}_k$",
            r"Propagates 99% UCB Scissor Tube $r_{\mathrm{tube}}(k) = 2.576\,\sigma_{\mathrm{tube}}(k)$",
        ],
        edge_color="#374151", fill_color="#f9fafb", lw=1.4, title_color="#111827"
    )

    # Feedback path from Train Plant -> Bottom Observer Box
    y_bot_mid = y_bot + h_bot / 2.0
    ax.plot([11.88, 11.88, 9.00], [y_top, y_bot_mid, y_bot_mid], color="#374151", linewidth=1.4, zorder=4)
    draw_arrow(ax, (9.20, y_bot_mid), (9.00, y_bot_mid), color="#374151")
    ax.text(10.65, y_bot_mid + 0.20, r"State $(x_k, v_k, u_k)$", ha="center", va="center", fontsize=9.8, color="#374151")

    # Upward arrow from Observer -> Shield (Scissor Tube + Mass/Wind)
    draw_arrow(
        ax, (4.62, y_bot + h_bot), (4.62, y_top - 0.10),
        label=r"$r_{\mathrm{tube}},\,\hat{M},\,\hat{d}_k$",
        label_offset=(0.72, 0.0),
        color="#991b1b", lw=1.4
    )

    # Upward orthogonal arrow from Observer -> Fast LTV-MPC (Model Adaptation)
    ax.plot([8.20, 8.20, 10.20], [y_bot + h_bot, 1.90, 1.90], color="#065f46", linewidth=1.3, linestyle="--", zorder=4)
    draw_arrow(
        ax, (10.20, 1.90), (10.20, y_top),
        label="",
        color="#065f46", lw=1.3, ls="--"
    )
    ax.text(9.20, 1.72, r"$\hat{M},\,\hat{d}_k$", ha="center", va="center", fontsize=9.8, color="#065f46")

    # Feedback path from Observer -> PPO Pilot (15-D State)
    ax.plot([3.40, 1.38], [y_bot_mid, y_bot_mid], color="#1e3a8a", linewidth=1.4, zorder=4)
    draw_arrow(ax, (1.38, y_bot_mid), (1.38, y_top), color="#1e3a8a", lw=1.4)
    ax.text(2.35, y_bot_mid + 0.20, r"15-D Obs $s_t$", ha="center", va="center", fontsize=9.8, color="#1e3a8a")

    plt.tight_layout(pad=0.15)
    out_local = "contracts_hierarchical_shielding_arch.png"
    plt.savefig(out_local, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copyfile(out_local, "../contracts_hierarchical_shielding_arch.png")
    print(f"Saved clean serif architecture diagram to {out_local}")


if __name__ == "__main__":
    main()
