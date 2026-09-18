"""Interaction structure of the MSDI predictors against the forecasting target.

Script 31 fits Friedman's H on the contemporaneous SCFI level, which describes
co-movement rather than anything the forecasts claim. This script refits the same
statistic on `target_scfi_monthly_log_change_h6m`, the six-month horizon at which the
Diebold-Mariano tests reject, so the interaction structure is measured on the quantity
the paper actually forecasts.

Two figures are produced. The first reports per-feature interaction strength H_j under
three interaction-capable learners; they are shown together because they disagree, and
that disagreement is the point. The second shows the two-way partial dependence surface
for every one of the fifteen feature pairs, ordered by interaction strength, so the
choice of which surface to discuss is made from the data rather than by inheritance.

Prophet and the GLM are additive by construction, so their H is identically zero, and
the LSTM consumes sequences, for which partial dependence on a fixed feature vector is
undefined. Only tree ensembles can carry a non-trivial H here.

This is descriptive. Partial dependence is an in-sample device: every learner is fit on
the full sample with no held-out evaluation, and no significance is attached to any H
value. The pre-specified heterogeneity analysis is skipped at monthly frequency for
insufficient power, so nothing here supports a claim that an interaction is real.

Source: data/06_model_ready/prediction/llm_specific_scfi_monthly_panel.csv
Output: outputs/figures/figure20_interaction_strength.{pdf,png}
        outputs/figures/figure21_pairwise_partial_dependence.{pdf,png}
        outputs/reports/interaction_structure_h_statistics.csv
        outputs/reports/interaction_structure_pairwise_h.csv
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import partial_dependence
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL = PROJECT_ROOT / "data/06_model_ready/prediction/llm_specific_scfi_monthly_panel.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures"
REPORT_DIR = PROJECT_ROOT / "outputs/reports"

FEATURES = {
    "llm_signal__gemma_2_9b_it__overall_disruption_max": "MSDI Peak",
    "llm_signal__gemma_2_9b_it__dimension_mean7": "MSDI Breadth",
    "brent_monthly_mean": "Brent Crude",
    "cpb_world_trade_volume": "CPB Trade",
    "rwi_isl_container_throughput_index": "RWI Port",
    "cny_usd_exchange_rate": "CNY/USD FX",
}

PALETTE = {
    "Random forest": "#2b5c8f",
    "Gradient boosting": "#d95f02",
    "XGBoost": "#2f855a",
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


def load_design(target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if not PANEL.is_file():
        raise FileNotFoundError(f"Monthly panel not found: {PANEL}")
    frame = pd.read_csv(PANEL)
    missing = [c for c in [target_col, *FEATURES] if c not in frame.columns]
    if missing:
        raise ValueError(f"{PANEL.name} lacks required columns: {missing}")
    subset = frame.dropna(subset=[target_col, *FEATURES])
    if len(subset) < 40:
        raise ValueError(f"Only {len(subset)} complete rows; too few to fit")
    return subset[list(FEATURES)], subset[target_col]


def fit_models(X: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    return {
        "Random forest": RandomForestRegressor(
            n_estimators=100, max_depth=4, random_state=42
        ).fit(X, y),
        "Gradient boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
        ).fit(X, y),
        "XGBoost": XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, verbosity=0
        ).fit(X, y),
    }


def h_matrix(model, X: pd.DataFrame) -> np.ndarray:
    """Friedman's pairwise H: the share of two-way partial-dependence variance
    not explained by the two main effects."""
    p = X.shape[1]
    centered_main = {}
    for i in range(p):
        values = partial_dependence(model, X, [i], grid_resolution=25)["average"][0]
        centered_main[i] = values - np.mean(values)

    matrix = np.zeros((p, p))
    for i, j in combinations(range(p), 2):
        joint = partial_dependence(model, X, [i, j], grid_resolution=25)["average"][0]
        joint_centered = joint - np.mean(joint)
        interaction = joint_centered - (centered_main[i][:, None] + centered_main[j][None, :])
        denominator = np.sum(joint_centered**2)
        value = np.sqrt(np.sum(interaction**2) / denominator) if denominator > 0 else 0.0
        matrix[i, j] = matrix[j, i] = value
    return matrix


def overall_strength(matrix: np.ndarray) -> np.ndarray:
    p = matrix.shape[0]
    return np.sqrt(np.sum(matrix**2, axis=1) / (p - 1))


def make_strength_figure(strengths: dict[str, np.ndarray]) -> plt.Figure:
    labels = list(FEATURES.values())
    p = len(labels)
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=300)

    y = np.arange(p)
    width = 0.78 / len(strengths)
    offsets = (np.arange(len(strengths)) - (len(strengths) - 1) / 2) * width
    for offset, (name, values) in zip(offsets, strengths.items()):
        ax.barh(y + offset, values, height=width, label=name, color=PALETTE[name], alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Interaction statistic ($H_j$)")
    ax.set_xlim(0, max(v.max() for v in strengths.values()) * 1.20)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def make_pdp_grid(
    X: pd.DataFrame, model, matrix: np.ndarray, horizon_label: str
) -> plt.Figure:
    """All fifteen two-way partial-dependence surfaces, ordered by interaction strength
    and drawn on one shared colour scale so the panels are comparable."""
    labels = list(FEATURES.values())
    pairs = sorted(combinations(range(len(labels)), 2), key=lambda ij: -matrix[ij[0], ij[1]])

    surfaces = []
    for i, j in pairs:
        result = partial_dependence(model, X, [i, j], grid_resolution=30)
        grid = result.get("grid_values", result.get("values"))
        surfaces.append((i, j, grid[0], grid[1], result["average"][0].T))

    # One scale across panels so they are comparable, but trimmed to the central 96% of
    # fitted values. Partial dependence at the sparse edges of the Brent range is extreme
    # enough to flatten every other panel if it sets the limits.
    pooled = np.concatenate([z.ravel() for *_, z in surfaces])
    vmin, vmax = np.percentile(pooled, [2.0, 98.0])
    levels = np.linspace(vmin, vmax, 15)

    fig, axes = plt.subplots(4, 4, figsize=(19.0, 16.0), dpi=300)
    plt.subplots_adjust(hspace=0.46, wspace=0.34, left=0.05, right=0.90, top=0.95, bottom=0.05)
    flat = axes.ravel()

    contour = None
    for ax, (i, j, gx, gy, z) in zip(flat, surfaces):
        mesh_x, mesh_y = np.meshgrid(gx, gy)
        contour = ax.contourf(mesh_x, mesh_y, np.clip(z, vmin, vmax), levels=levels,
                              cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{labels[i]} $\\times$ {labels[j]}\n$H_{{jk}}$ = {matrix[i, j]:.3f}",
                     fontweight="bold", pad=8)
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])

    for ax in flat[len(surfaces):]:
        ax.axis("off")

    bar = fig.colorbar(contour, ax=axes, fraction=0.022, pad=0.02)
    bar.set_label(f"Predicted SCFI log change, {horizon_label}")
    return fig


def save(fig: plt.Figure, basename: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = FIGURE_DIR / basename
    fig.savefig(f"{base}.pdf", format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{base}.png", format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", default="h6m", choices=("h1m", "h3m", "h6m"),
                        help="Forecast horizon whose target the interaction is fitted on")
    parser.add_argument("--surface-model", default="Random forest",
                        choices=tuple(PALETTE), help="Learner used for the surface grid")
    args = parser.parse_args()

    target_col = f"target_scfi_monthly_log_change_{args.horizon}"
    horizon_label = {"h1m": "1 month", "h3m": "3 months", "h6m": "6 months"}[args.horizon]

    X, y = load_design(target_col)
    print(f"Target: {target_col}  ({len(X)} complete months, {X.shape[1]} features)")

    models = fit_models(X, y)
    matrices = {name: h_matrix(model, X) for name, model in models.items()}
    strengths = {name: overall_strength(matrix) for name, matrix in matrices.items()}

    labels = list(FEATURES.values())
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    per_feature = pd.DataFrame({"feature": labels})
    for name, values in strengths.items():
        per_feature[f"h_overall_{name.lower().replace(' ', '_')}"] = values
    per_feature["horizon"] = args.horizon
    per_feature["n_complete_months"] = len(X)
    per_feature["inference_status"] = "descriptive_in_sample_no_significance_attached"
    per_feature.to_csv(REPORT_DIR / "interaction_structure_h_statistics.csv", index=False)

    pairwise = pd.DataFrame(
        [{"pair": f"{labels[i]} x {labels[j]}",
          **{name: matrices[name][i, j] for name in models}}
         for i, j in combinations(range(len(labels)), 2)]
    )
    pairwise["mean_across_learners"] = pairwise[list(models)].mean(axis=1)
    pairwise = pairwise.sort_values("mean_across_learners", ascending=False)
    pairwise.to_csv(REPORT_DIR / "interaction_structure_pairwise_h.csv", index=False)

    strength_path = save(make_strength_figure(strengths), "figure20_interaction_strength")
    grid_path = save(
        make_pdp_grid(X, models[args.surface_model], matrices[args.surface_model], horizon_label),
        "figure21_pairwise_partial_dependence",
    )

    print("\nPer-feature interaction strength")
    print(per_feature[["feature", *[c for c in per_feature if c.startswith("h_overall_")]]]
          .round(4).to_string(index=False))
    print("\nPairwise interaction strength, ordered")
    print(pairwise.round(3).to_string(index=False))
    print("\nRank agreement between learners")
    names = list(strengths)
    for a, b in combinations(names, 2):
        rho = pd.Series(strengths[a]).corr(pd.Series(strengths[b]), method="spearman")
        print(f"   {a} vs {b}: rho = {rho:+.3f}")
    print(f"\nSurfaces drawn from the {args.surface_model} fit")
    print(f"Written to {strength_path}.pdf and {grid_path}.pdf")


if __name__ == "__main__":
    main()
