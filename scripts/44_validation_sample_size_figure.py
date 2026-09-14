"""Sample-size determination figure for the human validation protocol.

A 50-article dual-annotator pilot fixed each dimension's expected agreement. Those pilot
values were then used to simulate, for candidate sample sizes from 100 to 1,000, how
precisely the quadratic-weighted kappa could be estimated: the expected width of its 95%
confidence interval, and the probability of landing inside the pre-registered precision
target of a half-width no greater than 0.05.

The figure plots both curves so the chosen operating point can be read directly against
the target rather than asserted. It also makes the shortfall visible: the executed sample
of 400 meets the target for the primary dimensions but not for the weakest ones, which is
the honest reading of the pilot.

Source: validation/pilot/results/sample_size_simulation_by_dimension.csv
Output: outputs/figures/validation/validation_figure01_sample_size_convergence.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATION = PROJECT_ROOT / "validation/pilot/results/sample_size_simulation_by_dimension.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures/validation"

TARGET_HALF_WIDTH = 0.05
TARGET_PROBABILITY = 0.90
EXECUTED_N = 400

DIMENSION_LABELS = {
    "global_container_movement_disruption_score": "Container mov.",
    "freight_cost_disruption_score": "Freight cost",
    "port_disruption_score": "Port",
    "supply_disruption_score": "Supply",
    "demand_disruption_score": "Demand",
    "production_disruption_score": "Production",
    "covid19_related_disruption_score": "COVID-19",
}
DIMENSION_COLORS = {
    "global_container_movement_disruption_score": "#1f4e79",
    "freight_cost_disruption_score": "#2e7d32",
    "port_disruption_score": "#0f766e",
    "supply_disruption_score": "#c55a11",
    "demand_disruption_score": "#7c3aed",
    "production_disruption_score": "#b91c1c",
    "covid19_related_disruption_score": "#6b7280",
}

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Times", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10.5
plt.rcParams["axes.titlesize"] = 11.5
plt.rcParams["xtick.labelsize"] = 9.5
plt.rcParams["ytick.labelsize"] = 9.5
plt.rcParams["legend.fontsize"] = 8.5
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def draw_panel(
    ax: plt.Axes, data: pd.DataFrame, value_column: str, reference: float,
    reference_label: str, ylabel: str, title: str,
) -> None:
    for dimension, label in DIMENSION_LABELS.items():
        series = data.loc[data["dimension"] == dimension].sort_values("candidate_N")
        if series.empty:
            raise ValueError(f"Simulation output lacks dimension {dimension}")
        ax.plot(
            series["candidate_N"], series[value_column], color=DIMENSION_COLORS[dimension],
            lw=1.4, label=label,
        )
    ax.axhline(reference, color="#111827", lw=1.0, linestyle="--")
    ax.axvline(EXECUTED_N, color="#b91c1c", lw=1.0, linestyle=":")
    ax.text(
        data["candidate_N"].max(), reference, f" {reference_label}", fontsize=8,
        va="bottom", ha="right", color="#111827",
    )
    ax.text(
        EXECUTED_N, ax.get_ylim()[1], f" executed $N={EXECUTED_N}$ ", fontsize=8,
        va="top", ha="left", color="#b91c1c", rotation=90,
    )
    ax.set_xlabel("Annotated sample size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SIMULATION)

    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    draw_panel(
        axes[0], data, "mean_ci_half_width", TARGET_HALF_WIDTH, "target 0.05",
        "Expected 95% CI half-width (QWK)", "A. Precision against sample size",
    )
    draw_panel(
        axes[1], data, "prob_ci_half_width_leq_target", TARGET_PROBABILITY, "target 0.90",
        "P(half-width $\\leq$ 0.05)", "B. Probability of meeting the precision target",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=7, frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    figure.tight_layout(rect=[0, 0.05, 1, 1])
    for extension in ("pdf", "png"):
        figure.savefig(
            FIGURE_DIR / f"validation_figure01_sample_size_convergence.{extension}",
            format=extension, bbox_inches="tight", dpi=300,
        )
    plt.close(figure)

    executed = data.loc[data["candidate_N"] == EXECUTED_N]
    meeting = executed.loc[executed["mean_ci_half_width"] <= TARGET_HALF_WIDTH, "dimension"]
    print(f"At N={EXECUTED_N}: {len(meeting)} of {data['dimension'].nunique()} dimensions meet the target")
    print(f"Figure written to {FIGURE_DIR / 'validation_figure01_sample_size_convergence.pdf'}")


if __name__ == "__main__":
    main()
