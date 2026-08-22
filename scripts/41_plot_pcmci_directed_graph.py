"""Generate clean, publication-grade graph for PCMCI+ Directed Temporal Discovery.

Creates an expansive, non-overlapping 2-panel publication figure in Times New Roman (300 DPI, Vector PDF + PNG):
- Panel (a): Generously spaced Directed Network Topology Graph with separate clear channels and outer memory loops.
- Panel (b): Full Partial Correlation Matrix Heatmap across all source->target channels and lags 1..12.
"""
from __future__ import annotations

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Set publication style: Strict Times New Roman & STIX math font everywhere, 300 DPI
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
})


def load_pcmci_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads all PCMCI link observations and significant link registry."""
    links_path = PROJECT_ROOT / "outputs/time_series_causality/causality_v001_pcmci_links.csv"
    sig_path = PROJECT_ROOT / "outputs/tables/publication_v001_S03_significant_pcmci_links.csv"

    links = pd.read_csv(links_path)
    sig = pd.read_csv(sig_path)
    return links, sig


def plot_pcmci_directed_network(
    links_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Draws a clean, expansive 2-panel PCMCI+ publication figure with generous spacing and pure Times New Roman."""
    fig = plt.figure(figsize=(18.5, 8.2))

    # Grid: Left = Directed Network Graph (Panel A), Right = Cross-Lag Heatmap (Panel B)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.20)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])

    # ==========================================
    # PANEL A: DIRECTED NETWORK TOPOLOGY GRAPH
    # ==========================================
    ax_net.set_xlim(-3.2, 3.2)
    ax_net.set_ylim(-1.9, 1.9)
    ax_net.axis("off")
    ax_net.set_title(
        r"$\mathbf{(a)\ PCMCI^+\ Directed\ Temporal\ Network\ (\theta=0.82,\ FDR\ q \leq 0.05)}$",
        fontsize=12,
        pad=18,
    )

    # 4 Nodes with wide coordinate spacing
    bw, bh = 0.72, 0.48
    nodes = {
        "msdi_change": {
            "center": (-0.95, 0.40),
            "label": r"$\mathbf{\Delta MSDI_t}$" + "\n(Disruption\nInnovation)",
            "color": "#1f77b4",
            "bg": "#eef6fc",
        },
        "scfi_return": {
            "center": (0.95, 0.40),
            "label": r"$\mathbf{\Delta \ln SCFI_t}$" + "\n(Freight Rate\nReturn)",
            "color": "#d62728",
            "bg": "#fdf2f2",
        },
        "brent_return": {
            "center": (-0.95, -0.90),
            "label": r"$\mathbf{\Delta \ln Brent_t}$" + "\n(Bunker Fuel\nShock)",
            "color": "#ff7f0e",
            "bg": "#fef7ee",
        },
        "news_volume_change": {
            "center": (0.95, -0.90),
            "label": r"$\mathbf{\Delta \ln N_t}$" + "\n(News Media\nVolume)",
            "color": "#2ca02c",
            "bg": "#f2faf2",
        },
    }

    # 1. Top Channel: MSDI -> SCFI (Lead Transmission)
    ax_net.annotate(
        "",
        xy=(0.58, 0.40),
        xytext=(-0.58, 0.40),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#1f77b4",
            lw=3.2,
            mutation_scale=20,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=2,
    )
    # Banner cleanly placed well ABOVE without touching boxes
    ax_net.text(
        0.0,
        0.92,
        r"$\mathbf{MSDI \to SCFI\ (Lead\ 1\text{–}4w)}$"
        + "\n"
        + r"$\mathbf{Local\ Proj\ \beta_{6m}=+11.4\%^*\ (DML\ p<0.001)}$",
        color="#1f77b4",
        fontsize=8.5,
        ha="center",
        va="center",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#1f77b4", lw=1.2),
        zorder=5,
    )

    # 2. Left Vertical Channel: Brent -> MSDI (lag 1w)
    ax_net.annotate(
        "",
        xy=(-0.95, 0.14),
        xytext=(-0.95, -0.64),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#ff7f0e",
            lw=2.2,
            mutation_scale=16,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=2,
    )
    ax_net.text(
        -0.95,
        -0.25,
        r"$\mathbf{\tau = 1w}$" + "\n" + r"$(\hat{\rho}=-0.16^{**})$",
        color="#ff7f0e",
        fontsize=8.0,
        ha="center",
        va="center",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#ff7f0e", lw=0.8),
        zorder=5,
    )

    # 3. Diagonal 1 (Upper Curve): News Volume -> MSDI (lag 2w)
    ax_net.annotate(
        "",
        xy=(-0.58, 0.25),
        xytext=(0.58, -0.70),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#2ca02c",
            lw=2.0,
            mutation_scale=16,
            connectionstyle="arc3,rad=-0.18",
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=2,
    )
    ax_net.text(
        -0.10,
        -0.02,
        r"$\mathbf{\tau=2w}\ (\hat{\rho}=-0.15^*)$",
        color="#2ca02c",
        fontsize=8.0,
        ha="center",
        va="center",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#2ca02c", lw=0.8),
        zorder=5,
    )

    # 4. Diagonal 2 (Lower Curve): MSDI -> News Volume (lag 8w - Long-memory echo)
    ax_net.annotate(
        "",
        xy=(0.58, -0.85),
        xytext=(-0.58, 0.10),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#1f77b4",
            lw=2.2,
            mutation_scale=16,
            linestyle="--",
            connectionstyle="arc3,rad=0.20",
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=2,
    )
    ax_net.text(
        0.10,
        -0.50,
        r"$\mathbf{\tau=8w}\ (\hat{\rho}=-0.16^{**})$",
        color="#1f77b4",
        fontsize=8.0,
        ha="center",
        va="center",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#1f77b4", lw=0.8),
        zorder=5,
    )

    # 5. Autoregressive Memory Loops (Placed with wide, generous clearance)
    # MSDI self-loop (Left outer curve with ample gap)
    arc_msdi = patches.FancyArrowPatch(
        (-1.32, 0.52),
        (-1.32, 0.28),
        connectionstyle="arc3,rad=1.3",
        arrowstyle="-|>",
        color="#1f77b4",
        lw=2.2,
        mutation_scale=14,
        zorder=2,
    )
    ax_net.add_patch(arc_msdi)
    ax_net.text(
        -2.25,
        0.40,
        r"$\mathbf{MSDI\ Memory}$" + "\n" + r"$\tau \in \{1, 2, 4, 8\}\mathrm{w}$" + "\n" + r"$(q<0.05)$",
        color="#1f77b4",
        fontsize=8.0,
        ha="center",
        va="center",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#ffffff", edgecolor="#1f77b4", lw=0.8),
        zorder=5,
    )

    # SCFI self-loop (Right outer curve with ample gap)
    arc_scfi = patches.FancyArrowPatch(
        (1.32, 0.52),
        (1.32, 0.28),
        connectionstyle="arc3,rad=-1.3",
        arrowstyle="-|>",
        color="#d62728",
        lw=2.2,
        mutation_scale=14,
        zorder=2,
    )
    ax_net.add_patch(arc_scfi)
    ax_net.text(
        2.25,
        0.40,
        r"$\mathbf{SCFI\ Persistence}$" + "\n" + r"$\tau \in \{1, 2, 6, 9, 11\}\mathrm{w}$" + "\n" + r"$(q<0.05)$",
        color="#d62728",
        fontsize=8.0,
        ha="center",
        va="center",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#ffffff", edgecolor="#d62728", lw=0.8),
        zorder=5,
    )

    # Draw Nodes as clean rounded boxes on top of background arrows
    for key, n in nodes.items():
        cx, cy = n["center"]
        box = patches.FancyBboxPatch(
            (cx - bw / 2, cy - bh / 2),
            bw,
            bh,
            boxstyle="round,pad=0.06,rounding_size=0.10",
            facecolor=n["bg"],
            edgecolor=n["color"],
            linewidth=2.2,
            zorder=4,
        )
        ax_net.add_patch(box)
        ax_net.text(
            cx,
            cy,
            n["label"],
            ha="center",
            va="center",
            fontsize=9.5,
            color="#111827",
            fontweight="bold",
            zorder=5,
        )

    # Legend for Panel A (cleanly placed at bottom)
    leg_lines = [
        plt.Line2D([0], [0], color="#1f77b4", lw=2.8, label="MSDI Disruption Channel"),
        plt.Line2D([0], [0], color="#ff7f0e", lw=2.2, label="Bunker Fuel Price Channel"),
        plt.Line2D([0], [0], color="#2ca02c", lw=2.0, label="Media Reporting Feedback"),
        plt.Line2D([0], [0], color="#1f77b4", linestyle="--", lw=2.0, label="Long-Lag Feedback (Echo)"),
    ]
    ax_net.legend(
        handles=leg_lines,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="#d1d5db",
        fontsize=8.5,
    )

    # ==========================================
    # PANEL B: CROSS-LAG PARTIAL CORRELATION HEATMAP
    # ==========================================
    channel_order = [
        ("msdi_change", "msdi_change", r"$\Delta \mathrm{MSDI} \to \Delta \mathrm{MSDI}$"),
        ("msdi_change", "scfi_return", r"$\Delta \mathrm{MSDI} \to \Delta \ln \mathrm{SCFI}$"),
        ("msdi_change", "news_volume_change", r"$\Delta \mathrm{MSDI} \to \Delta \ln N$"),
        ("brent_return", "msdi_change", r"$\Delta \ln \mathrm{Brent} \to \Delta \mathrm{MSDI}$"),
        ("brent_return", "scfi_return", r"$\Delta \ln \mathrm{Brent} \to \Delta \ln \mathrm{SCFI}$"),
        ("news_volume_change", "msdi_change", r"$\Delta \ln N \to \Delta \mathrm{MSDI}$"),
        ("scfi_return", "scfi_return", r"$\Delta \ln \mathrm{SCFI} \to \Delta \ln \mathrm{SCFI}$"),
    ]

    w_links = links_df.loc[
        (links_df["frequency"] == "weekly") & (links_df["msdi_path"] == "peak")
    ].copy()

    max_lags = 12
    matrix = np.full((len(channel_order), max_lags), np.nan)
    sig_mask = np.zeros((len(channel_order), max_lags), dtype=bool)

    for r_idx, (src, tgt, _) in enumerate(channel_order):
        for lag in range(1, max_lags + 1):
            match = w_links.loc[
                (w_links["source"] == src)
                & (w_links["target"] == tgt)
                & (w_links["source_lag"] == lag)
            ]
            if not match.empty:
                val = match["partial_correlation"].iloc[0]
                matrix[r_idx, lag - 1] = val
                is_sig = match["significant_fdr"].iloc[0]
                sig_mask[r_idx, lag - 1] = bool(is_sig)

    cmap = plt.cm.RdBu_r
    im = ax_heat.imshow(matrix, cmap=cmap, vmin=-0.45, vmax=0.45, aspect="auto")

    # Add text annotations on cells with asterisks on significant ones
    for r in range(len(channel_order)):
        for c in range(max_lags):
            val = matrix[r, c]
            if not np.isnan(val):
                txt = f"{val:+.2f}"
                if sig_mask[r, c]:
                    txt += r"$^*$"
                    weight = "bold"
                    color = "white" if abs(val) > 0.22 else "black"
                else:
                    weight = "normal"
                    color = "white" if abs(val) > 0.28 else "black"
                ax_heat.text(
                    c,
                    r,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color=color,
                    fontweight=weight,
                )

    ax_heat.set_xticks(range(max_lags))
    ax_heat.set_xticklabels([f"t−{k}" for k in range(1, max_lags + 1)], fontsize=8.5)
    ax_heat.set_yticks(range(len(channel_order)))
    ax_heat.set_yticklabels([label for _, _, label in channel_order], fontsize=9)
    ax_heat.set_xlabel("Time Lag in Weeks (t − k)", fontsize=10.5)
    ax_heat.set_title(
        r"$\mathbf{(b)\ Partial\ Correlation\ Matrix\ \hat{\rho}\ by\ Lag\ (Asterisk\ *\ =\ FDR\ q \leq 0.05)}$",
        fontsize=12,
        pad=18,
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label(r"Partial Correlation $\hat{\rho}$ (Conditioned on All Graph Parents)", fontsize=9.5)

    plt.tight_layout()

    # Save high-res publication formats
    figures_dir = output_dir
    pdf_path = figures_dir / "publication_v001_figure06_temporal_dependence.pdf"
    png_path = figures_dir / "publication_v001_figure06_temporal_dependence.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info("Saved expansive, clean Figure 06 to %s and %s", pdf_path.name, png_path.name)


def main() -> None:
    logging.info("=" * 80)
    logging.info("  GENERATING CLEAN PCMCI+ DIRECTED GRAPH (FIGURE 06)")
    logging.info("=" * 80)

    links_df, sig_df = load_pcmci_data()
    figures_dir = PROJECT_ROOT / "outputs/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_pcmci_directed_network(links_df, sig_df, figures_dir)
    logging.info("=" * 80)
    logging.info("  Figure 06 generated successfully.")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
