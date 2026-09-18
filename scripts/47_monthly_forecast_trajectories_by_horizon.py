"""Monthly forecast trajectories at every horizon, by disruption path.

The published trajectory figures cover the one-month horizon only, which is the horizon
at which no Diebold-Mariano test rejects. This script plots the same kind of trajectory
at 1, 3 and 6 months so the horizons that carry the paper's results are visible, and it
includes the NNLS forecast stack, which appears nowhere else as a forecast.

Rows are horizons, columns are the two disruption paths. Each panel shows the realised
SCFI level, the AR+Brent+coverage baseline, and every MSDI predictor.

The NNLS stack estimates combination weights from data and so starts later than the
other predictors; at six months it rests on eleven origins. Its line is correspondingly
short, and that is the point rather than a defect.

Source: outputs/llm_scfi_comparison/llm_scfi_monthly_v001_forecast_predictions.csv
Output: outputs/figures/figure22_monthly_forecast_trajectories_by_horizon.{pdf,png}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = (
    PROJECT_ROOT
    / "outputs/llm_scfi_comparison/llm_scfi_monthly_v001_forecast_predictions.csv"
)
FIGURE_DIR = PROJECT_ROOT / "outputs/figures"

PATHS = {"overall_disruption_max": "Peak", "dimension_mean7": "Breadth"}
HORIZONS = (1, 3, 6)

PREDICTORS = {
    "glm_4_9b_0414": ("GLM-4-9B", "#1f77b4", "-"),
    "gemma_2_9b_it": ("Gemma 2 9B", "#2f855a", "-"),
    "qwen_3_8b": ("Qwen3-8B", "#9467bd", "-"),
    "qwen_2_5_7b": ("Qwen2.5-7B", "#7f7f00", "-"),
    "score_median": ("Median-score", "#d95f02", "--"),
    "score_mean": ("Mean-score", "#e7a13d", "--"),
    "forecast_equal_mean": ("Equal-weight stack", "#17becf", "--"),
    "forecast_nnls_stack": ("NNLS stack", "#d62728", "-"),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
})


def load(construction: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PREDICTIONS.is_file():
        raise FileNotFoundError(f"Forecast predictions not found: {PREDICTIONS}")
    frame = pd.read_csv(PREDICTIONS, parse_dates=["target_date"])
    predictors = frame[
        (frame["index_construction"] == construction)
        & (frame["signal_name"].isin(PATHS))
        & (frame["score_source"].isin(PREDICTORS))
    ]
    baseline = frame[frame["source_type"] == "baseline"]
    if predictors.empty:
        raise ValueError(f"No predictor rows for construction {construction!r}")
    return predictors, baseline


def make_figure(predictors: pd.DataFrame, baseline: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(len(HORIZONS), len(PATHS), figsize=(14.0, 11.0),
                             dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.30, wspace=0.16, left=0.07, right=0.98,
                        top=0.90, bottom=0.07)

    for row, horizon in enumerate(HORIZONS):
        for col, (signal, path_label) in enumerate(PATHS.items()):
            ax = axes[row, col]
            panel = predictors[
                (predictors["horizon_months"] == horizon)
                & (predictors["signal_name"] == signal)
            ]

            actual = (
                panel[["target_date", "actual_scfi_level"]]
                .drop_duplicates()
                .sort_values("target_date")
            )
            ax.plot(actual["target_date"], actual["actual_scfi_level"],
                    color="#222222", linewidth=1.5, label="Actual SCFI", zorder=5)

            base = baseline[baseline["horizon_months"] == horizon].sort_values("target_date")
            if not base.empty:
                ax.plot(base["target_date"], base["predicted_scfi_level"],
                        color="#7f7f7f", linewidth=1.4, linestyle=":",
                        label="AR+Brent+coverage baseline", zorder=4)

            for source, (label, color, style) in PREDICTORS.items():
                series = panel[panel["score_source"] == source].sort_values("target_date")
                if series.empty:
                    continue
                # The NNLS stack carries the only six-month result, so it is drawn
                # heavier and above the rest rather than lost among them.
                headline = source == "forecast_nnls_stack"
                ax.plot(series["target_date"], series["predicted_scfi_level"],
                        color=color, linewidth=1.3 if headline else 1.0,
                        linestyle=style, alpha=0.95 if headline else 0.75,
                        zorder=4.5 if headline else 3, label=label)

            ax.set_title(f"{path_label} path $\\cdot$ $h={horizon}$ month"
                         f"{'s' if horizon > 1 else ''}",
                         fontweight="bold", pad=7)
            ax.grid(True, linestyle="--", alpha=0.35)
            if col == 0:
                ax.set_ylabel("SCFI points")
            if row == len(HORIZONS) - 1:
                ax.set_xlabel("Target month")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.975))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction", default="adapted_pdf_sqrt_ema3",
                        help="Index construction to plot")
    args = parser.parse_args()

    predictors, baseline = load(args.construction)
    counts = (
        predictors.groupby(["signal_name", "horizon_months", "score_source"])
        .size()
        .unstack(fill_value=0)
    )
    print(f"Construction: {args.construction}")
    print(f"Origins per predictor:\n{counts.to_string()}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure = make_figure(predictors, baseline)
    base = FIGURE_DIR / "figure22_monthly_forecast_trajectories_by_horizon"
    figure.savefig(f"{base}.pdf", format="pdf", bbox_inches="tight", dpi=300)
    figure.savefig(f"{base}.png", format="png", bbox_inches="tight", dpi=300)
    plt.close(figure)
    print(f"\nWritten to {base}.pdf")


if __name__ == "__main__":
    main()
