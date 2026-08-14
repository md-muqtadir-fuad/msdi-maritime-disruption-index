"""Point 6: predictor overlap, multicollinearity, and clustering diagnostics."""
from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "eda/outputs/point_06_predictor_redundancy"
MINIMUM_PAIR_OBSERVATIONS = 24
MINIMUM_VIF_SERIES_OBSERVATIONS = 80

CONTROL_VARIABLES = (
    "brent_price",
    "cpb_world_trade_volume",
    "gscpi",
    "unctad_china_core_lane_lsbci",
    "imf_portwatch_global_container_calls",
    "rwi_container_throughput",
    "china_merchandise_exports",
    "cny_per_usd",
    "global_geopolitical_risk",
    "trade_policy_uncertainty",
    "us_china_tension",
    "oil_price_uncertainty",
    "infectious_disease_emv",
)

NEWS_SHOCK_VARIABLES = (
    "global_geopolitical_risk",
    "trade_policy_uncertainty",
    "us_china_tension",
    "oil_price_uncertainty",
    "infectious_disease_emv",
)

TARGETED_PAIRS = (
    ("cpb_world_trade_volume", "china_merchandise_exports", "trade-demand proxies"),
    ("cpb_world_trade_volume", "rwi_container_throughput", "trade and throughput proxies"),
    ("china_merchandise_exports", "rwi_container_throughput", "exports and throughput proxies"),
    ("msdi_peak", "global_geopolitical_risk", "MSDI and news-attention shock"),
    ("msdi_peak", "trade_policy_uncertainty", "MSDI and news-attention shock"),
    ("msdi_peak", "us_china_tension", "MSDI and news-attention shock"),
    ("msdi_peak", "oil_price_uncertainty", "MSDI and news-attention shock"),
    ("msdi_peak", "infectious_disease_emv", "MSDI and news-attention shock"),
    ("msdi_breadth", "global_geopolitical_risk", "MSDI and news-attention shock"),
    ("msdi_breadth", "trade_policy_uncertainty", "MSDI and news-attention shock"),
    ("msdi_breadth", "us_china_tension", "MSDI and news-attention shock"),
    ("msdi_breadth", "oil_price_uncertainty", "MSDI and news-attention shock"),
    ("msdi_breadth", "infectious_disease_emv", "MSDI and news-attention shock"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit redundancy among candidate predictors.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-run", action="store_true", help="Run deterministic diagnostics only.")
    return parser.parse_args()


def load_temporal_module():
    path = ROOT / "eda/03_temporal_properties.py"
    spec = importlib.util.spec_from_file_location("eda_temporal", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pairwise_long(frame: pd.DataFrame, labeler) -> pd.DataFrame:
    rows = []
    variables = list(frame.columns)
    for index, left in enumerate(variables):
        for right in variables[index + 1 :]:
            pair = frame[[left, right]].dropna()
            row = {
                "variable_1": left,
                "label_1": labeler(left),
                "variable_2": right,
                "label_2": labeler(right),
                "observations": len(pair),
                "pearson_correlation": np.nan,
                "spearman_correlation": np.nan,
            }
            if len(pair) >= MINIMUM_PAIR_OBSERVATIONS and pair[left].nunique() >= 3 and pair[right].nunique() >= 3:
                row["pearson_correlation"] = pair[left].corr(pair[right], method="pearson")
                row["spearman_correlation"] = pair[left].corr(pair[right], method="spearman")
            rows.append(row)
    result = pd.DataFrame(rows)
    result["maximum_absolute_correlation"] = result[["pearson_correlation", "spearman_correlation"]].abs().max(axis=1)
    result["near_duplicate_flag_0_80"] = result["maximum_absolute_correlation"].ge(0.80)
    return result


def targeted_overlap(frame: pd.DataFrame, labeler) -> pd.DataFrame:
    rows = []
    for left, right, rationale in TARGETED_PAIRS:
        pair = frame[[left, right]].dropna()
        rows.append({
            "variable_1": left,
            "label_1": labeler(left),
            "variable_2": right,
            "label_2": labeler(right),
            "rationale": rationale,
            "observations": len(pair),
            "pearson_correlation": pair[left].corr(pair[right], method="pearson") if len(pair) >= MINIMUM_PAIR_OBSERVATIONS else np.nan,
            "spearman_correlation": pair[left].corr(pair[right], method="spearman") if len(pair) >= MINIMUM_PAIR_OBSERVATIONS else np.nan,
        })
    return pd.DataFrame(rows)


def vif_table(frame: pd.DataFrame, labeler) -> tuple[pd.DataFrame, pd.DataFrame]:
    availability = pd.DataFrame({"variable": frame.columns, "observations": [frame[c].notna().sum() for c in frame]})
    eligible = availability.loc[availability["observations"].ge(MINIMUM_VIF_SERIES_OBSERVATIONS), "variable"].tolist()
    complete = frame[eligible].dropna()
    audit = pd.DataFrame([{
        "candidate_variables": len(frame.columns),
        "eligible_variables": len(eligible),
        "excluded_for_sparse_support": " | ".join(sorted(set(frame.columns) - set(eligible))),
        "complete_case_rows": len(complete),
        "minimum_required_rows": max(30, len(eligible) + 10),
        "vif_computable": len(eligible) >= 2 and len(complete) >= max(30, len(eligible) + 10),
    }])
    if not bool(audit.loc[0, "vif_computable"]):
        return pd.DataFrame(columns=["variable", "label", "vif", "complete_case_rows", "condition_number"]), audit
    standardized = (complete - complete.mean()) / complete.std(ddof=1)
    standardized = standardized.loc[:, standardized.std(ddof=1).gt(0)]
    matrix = standardized.to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vifs = [variance_inflation_factor(matrix, position) for position in range(matrix.shape[1])]
    condition_number = np.linalg.cond(matrix)
    return pd.DataFrame({
        "variable": standardized.columns,
        "label": [labeler(c) for c in standardized.columns],
        "vif": vifs,
        "complete_case_rows": len(standardized),
        "condition_number": condition_number,
    }), audit


def cluster_plot(frame: pd.DataFrame, labeler, output_dir: Path) -> tuple[Path, pd.DataFrame]:
    eligible = [column for column in frame if frame[column].notna().sum() >= MINIMUM_PAIR_OBSERVATIONS and frame[column].nunique(dropna=True) >= 3]
    correlation = frame[eligible].corr(method="spearman", min_periods=MINIMUM_PAIR_OBSERVATIONS)
    correlation = correlation.fillna(0).clip(-1, 1)
    np.fill_diagonal(correlation.values, 1.0)
    distance = 1 - correlation.abs()
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance.values, 0.0)
    tree = linkage(squareform(distance.to_numpy(), checks=True), method="average")
    fig, ax = plt.subplots(figsize=(11, 7))
    dendrogram(tree, labels=[labeler(c) for c in eligible], orientation="right", leaf_font_size=8, ax=ax)
    ax.set_xlabel("Distance: 1 - absolute Spearman correlation")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path = output_dir / "predictor_hierarchical_clustering.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    order = dendrogram(tree, no_plot=True)["leaves"]
    membership = pd.DataFrame({"display_order": range(1, len(order) + 1), "variable": [eligible[i] for i in order], "label": [labeler(eligible[i]) for i in order]})
    return path, membership


def write_findings(path: Path, pairs: pd.DataFrame, vif: pd.DataFrame, vif_audit: pd.DataFrame, targeted: pd.DataFrame) -> None:
    duplicates = pairs.loc[pairs["near_duplicate_flag_0_80"]].sort_values("maximum_absolute_correlation", ascending=False)
    lines = [
        "# Point 6 - Predictor redundancy findings",
        "",
        f"- Pairwise control relationships audited: {len(pairs)}.",
        f"- Pairs with absolute Pearson or Spearman correlation at least 0.80: {len(duplicates)}.",
        f"- VIF variables: {len(vif)}; complete-case rows: {int(vif_audit.loc[0, 'complete_case_rows'])}.",
        "",
        "High correlation or VIF does not delete a variable. It indicates that variables should not be interpreted as separable mechanisms in the same specification without a pre-declared strategy.",
        "",
        "## Near-duplicate review",
        "",
        "| Variable 1 | Variable 2 | Maximum absolute correlation |",
        "|---|---|---:|",
    ]
    for row in duplicates.head(15).itertuples(index=False):
        lines.append(f"| {row.label_1} | {row.label_2} | {row.maximum_absolute_correlation:.3f} |")
    if duplicates.empty:
        lines.append("| None at the pre-declared threshold | | |")
    lines.extend([
        "",
        "Targeted trade-demand and news-attention overlaps are reported in `targeted_construct_overlap.csv`. They support alternative specifications or grouped interpretation, not outcome-guided pruning.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporal = load_temporal_module()
    panel = temporal.build_panel()
    changes, _, _ = temporal.transformations(panel)
    controls = changes[[column for column in CONTROL_VARIABLES if column in changes]]
    overlap_frame = changes[["msdi_peak", "msdi_breadth", *[column for column in NEWS_SHOCK_VARIABLES if column in changes]]]

    pairs = pairwise_long(controls, temporal.friendly)
    targeted = targeted_overlap(changes, temporal.friendly)
    vif, vif_audit = vif_table(controls, temporal.friendly)
    figure_path, cluster_order = cluster_plot(pd.concat([controls, overlap_frame[["msdi_peak", "msdi_breadth"]]], axis=1), temporal.friendly, args.output_dir)

    controls.corr(method="pearson", min_periods=MINIMUM_PAIR_OBSERVATIONS).to_csv(args.output_dir / "control_pearson_correlation_matrix.csv")
    controls.corr(method="spearman", min_periods=MINIMUM_PAIR_OBSERVATIONS).to_csv(args.output_dir / "control_spearman_correlation_matrix.csv")
    pairs.to_csv(args.output_dir / "control_pairwise_redundancy.csv", index=False)
    targeted.to_csv(args.output_dir / "targeted_construct_overlap.csv", index=False)
    vif.to_csv(args.output_dir / "control_vif.csv", index=False)
    vif_audit.to_csv(args.output_dir / "control_vif_sample_audit.csv", index=False)
    cluster_order.to_csv(args.output_dir / "hierarchical_cluster_order.csv", index=False)
    write_findings(args.output_dir / "point_06_findings.md", pairs, vif, vif_audit, targeted)

    manifest = {
        "protocol_point": 6,
        "run_mode": "test" if args.test_run else "full",
        "status": "complete_descriptive_no_selection",
        "control_variables": len(controls.columns),
        "pairwise_tests": len(pairs),
        "near_duplicate_pairs": int(pairs["near_duplicate_flag_0_80"].sum()),
        "vif_variables": len(vif),
        "figure": figure_path.name,
        "guardrail": "Correlation, VIF, and clustering diagnose overlapping information; they do not provide an outcome-guided deletion rule.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
