#!/usr/bin/env python3
"""
Compute Inter-Rater Reliability for 2 Human Annotators (Pilot N = 50)
====================================================================
This script reads the completed annotation sheets from Annotator 1 and Annotator 2,
merges them on `validation_id`, and calculates:
  1. Exact Agreement (%)
  2. Cohen's Kappa (kappa) for categorical relevance
  3. Quadratic Weighted Kappa (kappa_w) for 0-4 ordinal disruption criteria
  4. Krippendorff's Alpha (alpha) for interval/ordinal consensus
  5. Mean Absolute Error (MAE) and Spearman rank correlation (rho)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = PROJECT_ROOT / "validation"
PILOT_DIR = VAL_DIR / "pilot"

ANNOTATOR_1_PATH = PILOT_DIR / "completed" / "annotator_1_pilot_50_completed.csv"
ANNOTATOR_2_PATH = PILOT_DIR / "completed" / "annotator_2_pilot_50_completed.csv"
OUTPUT_AGREEMENT_PATH = PILOT_DIR / "results" / "pilot_50_inter_rater_agreement.csv"

DISRUPTION_DIMENSIONS = [
    "port_disruption_score",
    "global_container_movement_disruption_score",
    "freight_cost_disruption_score",
    "supply_disruption_score",
    "production_disruption_score",
    "covid19_related_disruption_score",
    "demand_disruption_score",
    "overall_disruption_score",
]


def krippendorff_alpha_interval(ratings_matrix: np.ndarray) -> float:
    """
    Computes Krippendorff's Alpha for interval/ordinal ratings.
    ratings_matrix: shape (n_items, n_raters)
    """
    # Filter rows with at least 2 non-nan values
    valid_mask = np.sum(~np.isnan(ratings_matrix), axis=1) >= 2
    matrix = ratings_matrix[valid_mask]
    if len(matrix) == 0:
        return np.nan

    n_items, n_raters = matrix.shape
    all_vals = matrix[~np.isnan(matrix)]
    if len(np.unique(all_vals)) <= 1:
        return 1.0  # Perfect agreement

    # Total pairwise differences
    n_pairs = 0
    observed_diff = 0.0
    for row in matrix:
        vals = row[~np.isnan(row)]
        m = len(vals)
        for i in range(m):
            for j in range(i + 1, m):
                observed_diff += (vals[i] - vals[j]) ** 2
                n_pairs += 1

    if n_pairs == 0:
        return np.nan

    d_o = observed_diff / n_pairs

    # Expected difference under chance
    n_total = len(all_vals)
    expected_diff = 0.0
    for i in range(n_total):
        for j in range(i + 1, n_total):
            expected_diff += (all_vals[i] - all_vals[j]) ** 2
    d_e = expected_diff / (n_total * (n_total - 1) / 2)

    if d_e == 0:
        return 1.0

    return 1.0 - (d_o / d_e)


def main():
    OUTPUT_AGREEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 75)
    print("  HUMAN INTER-RATER RELIABILITY AUDIT (PILOT N = 50)")
    print("=" * 75)

    if not ANNOTATOR_1_PATH.is_file() or not ANNOTATOR_2_PATH.is_file():
        raise FileNotFoundError("Missing completed annotator CSV files in validation/pilot/completed/.")

    df1 = pd.read_csv(ANNOTATOR_1_PATH)
    df2 = pd.read_csv(ANNOTATOR_2_PATH)

    merged = pd.merge(
        df1, df2,
        on="validation_id",
        suffixes=("_ann1", "_ann2")
    )

    results = []

    # 1. Relevance Agreement
    rel1 = merged["relevance_label_ann1"].astype(str).str.strip().str.lower()
    rel2 = merged["relevance_label_ann2"].astype(str).str.strip().str.lower()
    
    valid_rel = (rel1 != "") & (rel1 != "nan") & (rel2 != "") & (rel2 != "nan")
    if valid_rel.sum() > 0:
        rel1_valid = rel1[valid_rel]
        rel2_valid = rel2[valid_rel]
        exact_rel = (rel1_valid == rel2_valid).mean() * 100
        kappa_rel = cohen_kappa_score(rel1_valid, rel2_valid)
        results.append({
            "variable": "relevance_label",
            "n_annotated": int(valid_rel.sum()),
            "exact_agreement_pct": round(exact_rel, 2),
            "cohen_kappa": round(kappa_rel, 4),
            "weighted_kappa_quadratic": np.nan,
            "krippendorff_alpha": np.nan,
            "mean_absolute_error": np.nan,
            "spearman_rho": np.nan,
        })

    # 2. Ordinal 0-4 Disruption Dimensions
    for dim in DISRUPTION_DIMENSIONS:
        col1 = f"{dim}_ann1"
        col2 = f"{dim}_ann2"
        if col1 in merged.columns and col2 in merged.columns:
            s1 = pd.to_numeric(merged[col1], errors="coerce")
            s2 = pd.to_numeric(merged[col2], errors="coerce")
            valid = s1.notna() & s2.notna()
            n_val = int(valid.sum())
            if n_val > 0:
                v1 = s1[valid].values
                v2 = s2[valid].values
                exact = (v1 == v2).mean() * 100
                mae = np.mean(np.abs(v1 - v2))
                
                # Quadratic weighted kappa
                try:
                    w_kappa = cohen_kappa_score(v1.astype(int), v2.astype(int), weights="quadratic")
                except Exception:
                    w_kappa = np.nan
                    
                # Krippendorff's alpha
                ratings = np.column_stack([v1, v2])
                alpha = krippendorff_alpha_interval(ratings)
                
                # Spearman rank correlation
                rho = pd.Series(v1).corr(pd.Series(v2), method="spearman")

                results.append({
                    "variable": dim,
                    "n_annotated": n_val,
                    "exact_agreement_pct": round(exact, 2),
                    "cohen_kappa": round(cohen_kappa_score(v1.astype(int), v2.astype(int)), 4) if len(np.unique(v1)) > 1 and len(np.unique(v2)) > 1 else np.nan,
                    "weighted_kappa_quadratic": round(w_kappa, 4),
                    "krippendorff_alpha": round(alpha, 4),
                    "mean_absolute_error": round(mae, 4),
                    "spearman_rho": round(rho, 4),
                })

    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        summary_df.to_csv(OUTPUT_AGREEMENT_PATH, index=False, encoding="utf-8-sig")
        print("INTER-RATER RELIABILITY SUMMARY:")
        print(summary_df.to_string(index=False))
        print("-" * 75)
        print(f"Exported agreement metrics to: {OUTPUT_AGREEMENT_PATH.resolve()}")
    else:
        print("Note: The annotation sheets currently contain blank columns.")
        print("Once Annotator 1 and Annotator 2 fill in their scores (0-4), rerun this script.")
    print("=" * 75)


if __name__ == "__main__":
    main()
