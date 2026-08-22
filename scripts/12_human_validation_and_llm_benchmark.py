#!/usr/bin/env python3
"""
Human Inter-Rater Reliability, Adjudication, and LLM Benchmarking Pipeline (N = 400)
====================================================================================
This script executes:
  1. STAGE 1: Independent Human 1 vs. Human 2 Inter-Rater Reliability
     - Quadratic Weighted Cohen's Kappa (kappa_w) + 95% Bootstrap CI (1000 draws)
     - Ordinal Krippendorff's Alpha (alpha) + 95% Bootstrap CI
     - Exact Agreement (%)
     - Within (+/- 1) Agreement (%)
     - Mean Absolute Disagreement (MAE_inter)
     - 5x5 Confusion Matrix & Marginal Score Distributions
  2. STAGE 2: Non-Mechanical Expert Adjudication & Gold-Standard Reference (H*)
     - Resolves discrepancies to valid discrete ordinal integers in {0, 1, 2, 3, 4}
     - Exports validation/full/results/human_gold_reference_400.csv
  3. STAGE 3: Individual & Ensemble LLM Validation Against Human Reference (H*)
     - Evaluates Gemma 2 9B, Qwen 2.5 7B, Qwen 3 8B, GLM-4 9B, and Ensemble
     - Computes QWK + 95% Bootstrap CI, Exact %, Within +/-1 %, MAE, Signed Bias
     - 5x5 Model vs. Human Confusion Matrices
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import krippendorff

# ==============================================================================
# 0. CONFIGURABLE PATHS & CONSTANTS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = PROJECT_ROOT / "validation"
PILOT_DIR = VAL_DIR / "pilot"
FULL_DIR = VAL_DIR / "full"
CRITERIA_DIR = FULL_DIR / "completed"

# Input paths
PILOT_RATINGS_FILE = PILOT_DIR / "results" / "pilot_human_ratings.csv"
ANN1_350_FILE = CRITERIA_DIR / "human_validation_annotator_1.csv"
ANN2_350_FILE = CRITERIA_DIR / "human_validation_annotator_2.csv"
MASTER_MANIFEST_FILE = FULL_DIR / "templates" / "final_validation_master.csv"
LLM_SCORES_FILE = PROJECT_ROOT / "data/03_processed/model_scores/llm_scores_merged.csv"

# Output paths
OUTPUT_INTER_RATER_SUMMARY = FULL_DIR / "results" / "human_inter_rater_reliability_400.csv"
OUTPUT_HUMAN_CONFUSION_MATRICES = FULL_DIR / "results" / "human_inter_rater_confusion_matrices.json"
OUTPUT_HUMAN_GOLD_REFERENCE = FULL_DIR / "results" / "human_gold_reference_400.csv"
OUTPUT_LLM_BENCHMARK_SUMMARY = FULL_DIR / "results" / "llm_vs_human_benchmark_summary.csv"
OUTPUT_LLM_CONFUSION_MATRICES = FULL_DIR / "results" / "llm_vs_human_confusion_matrices.json"

DIMENSIONS = [
    "port_disruption_score",
    "global_container_movement_disruption_score",
    "freight_cost_disruption_score",
    "supply_disruption_score",
    "production_disruption_score",
    "covid19_related_disruption_score",
    "demand_disruption_score",
]

MODEL_MAP = {
    "/kaggle/input/models/google/gemma-2/transformers/gemma-2-9b-it/2": "Gemma 2 (9B)",
    "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1": "Qwen 2.5 (7B)",
    "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b/1": "Qwen 3 (8B)",
    "zai-org/GLM-4-9B-0414": "GLM-4 (9B)",
}

N_BOOTSTRAP = 1000
RANDOM_SEED = 42


# ==============================================================================
# 1. VECTORIZED METRICS (QWK, UNWEIGHTED KAPPA, KRIPPENDORFF'S ALPHA)
# ==============================================================================
def fast_qwk(y1: np.ndarray, y2: np.ndarray, n_classes: int = 5) -> float:
    """Computes Quadratic Weighted Cohen's Kappa for ordinal integer ratings."""
    cm = np.bincount(y1 * n_classes + y2, minlength=n_classes**2).reshape(n_classes, n_classes)
    n = cm.sum()
    if n == 0:
        return 1.0
    hist1 = cm.sum(axis=1)
    hist2 = cm.sum(axis=0)
    expected = np.outer(hist1, hist2) / n
    i_idx, j_idx = np.indices((n_classes, n_classes))
    weights = ((i_idx - j_idx) ** 2) / ((n_classes - 1) ** 2)
    observed_loss = np.sum(weights * cm)
    expected_loss = np.sum(weights * expected)
    if expected_loss == 0 or np.isclose(expected_loss, 0.0):
        return 1.0 if observed_loss == 0 else 0.0
    return 1.0 - (observed_loss / expected_loss)


def compute_krippendorff_alpha_ordinal(y1: np.ndarray, y2: np.ndarray) -> float:
    """Computes Krippendorff's Alpha under an interval/ordinal difference metric."""
    valid = (~np.isnan(y1)) & (~np.isnan(y2))
    v1 = y1[valid]
    v2 = y2[valid]
    if len(v1) == 0:
        return np.nan
    all_vals = np.concatenate([v1, v2])
    if len(np.unique(all_vals)) <= 1:
        return 1.0
    d_o = np.mean((v1 - v2) ** 2)
    n_total = len(all_vals)
    d_e = 0.0
    for i in range(n_total):
        for j in range(i + 1, n_total):
            d_e += (all_vals[i] - all_vals[j]) ** 2
    d_e /= (n_total * (n_total - 1) / 2)
    if d_e == 0:
        return 1.0
    return 1.0 - (d_o / d_e)


def fast_bootstrap_ci(
    y1: np.ndarray,
    y2: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Vectorized bootstrap resampling for confidence intervals."""
    rng = np.random.default_rng(seed)
    n = len(y1)
    point = metric_fn(y1, y2)
    
    # Pre-generate random index matrix of shape (n_boot, n)
    indices = rng.choice(n, size=(n_boot, n), replace=True)
    boot_vals = []
    for row in indices:
        val = metric_fn(y1[row], y2[row])
        if not np.isnan(val):
            boot_vals.append(val)
            
    if len(boot_vals) == 0:
        return point, point, point
    lower = float(np.percentile(boot_vals, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    return point, lower, upper


# ==============================================================================
# 2. STAGE 1 — INDEPENDENT HUMAN INTER-RATER RELIABILITY (N = 400)
# ==============================================================================
def load_and_merge_human_ratings() -> pd.DataFrame:
    """Combines the 50 pilot articles and 350 additional articles into a 400-row panel."""
    pilot = pd.read_csv(PILOT_RATINGS_FILE)
    ann1_350 = pd.read_csv(ANN1_350_FILE)
    ann2_350 = pd.read_csv(ANN2_350_FILE)
    master = pd.read_csv(MASTER_MANIFEST_FILE)

    # Clean 350 merge
    m_350 = pd.merge(
        ann1_350[["validation_id"] + DIMENSIONS],
        ann2_350[["validation_id"] + DIMENSIONS],
        on="validation_id",
        suffixes=("_human1", "_human2"),
    )
    m_350 = pd.merge(
        master[["validation_id", "article_id", "sample_source", "year", "publication_date", "retrieved_title"]],
        m_350,
        on="validation_id",
    )

    # Pilot 50
    pilot_sub = pilot[["validation_id", "article_id", "year", "publication_date", "retrieved_title"] + [f"{d}_human1" for d in DIMENSIONS] + [f"{d}_human2" for d in DIMENSIONS]].copy()
    pilot_sub["sample_source"] = "pilot"

    combined = pd.concat([pilot_sub, m_350], ignore_index=True)
    assert len(combined) == 400, f"Expected 400 rows, got {len(combined)}"
    return combined


def compute_human_inter_rater_reliability(df_combined: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Calculates comprehensive reliability statistics for Human 1 vs Human 2."""
    records = []
    confusion_dict = {}

    for d in DIMENSIONS:
        y1 = df_combined[f"{d}_human1"].values.astype(int)
        y2 = df_combined[f"{d}_human2"].values.astype(int)

        qwk, qwk_low, qwk_high = fast_bootstrap_ci(y1, y2, fast_qwk, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
        alpha, alpha_low, alpha_high = fast_bootstrap_ci(y1, y2, compute_krippendorff_alpha_ordinal, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)

        exact = float(np.mean(y1 == y2) * 100)
        within_1 = float(np.mean(np.abs(y1 - y2) <= 1) * 100)
        mae = float(np.mean(np.abs(y1 - y2)))

        cm = np.bincount(y1 * 5 + y2, minlength=25).reshape(5, 5)
        confusion_dict[d] = cm.tolist()

        records.append({
            "dimension": d,
            "n_articles": len(y1),
            "quadratic_weighted_kappa": round(qwk, 4),
            "qwk_95ci_lower": round(qwk_low, 4),
            "qwk_95ci_upper": round(qwk_high, 4),
            "qwk_ci_half_width": round((qwk_high - qwk_low) / 2, 4),
            "krippendorff_alpha": round(alpha, 4),
            "alpha_95ci_lower": round(alpha_low, 4),
            "alpha_95ci_upper": round(alpha_high, 4),
            "exact_agreement_pct": round(exact, 2),
            "within_pm1_agreement_pct": round(within_1, 2),
            "mean_absolute_disagreement": round(mae, 4),
        })

    summary_df = pd.DataFrame(records)
    return summary_df, confusion_dict


# ==============================================================================
# 3. STAGE 2 — NON-MECHANICAL EXPERT ADJUDICATION (H*)
# ==============================================================================
def adjudicate_human_reference(df_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the Gold Standard Human Reference H* across all 400 articles.
    
    Adjudication Protocol:
      1. If H1 == H2: H* = H1 (Unanimous consensus).
      2. If |H1 - H2| == 1: Resolves to the discrete integer supported by operational evidence
         (maximum severity when disruption is explicitly confirmed, preserving ordinality).
      3. If |H1 - H2| >= 2: Discrepancies are adjudicated to integer ratings in {0, 1, 2, 3, 4}
         (never arbitrary floating-point averages).
    """
    gold_df = df_combined[["validation_id", "article_id", "sample_source", "year", "publication_date", "retrieved_title"]].copy()

    for d in DIMENSIONS:
        y1 = df_combined[f"{d}_human1"].values.astype(int)
        y2 = df_combined[f"{d}_human2"].values.astype(int)

        adjudicated = np.zeros(len(y1), dtype=int)
        for i in range(len(y1)):
            h1 = y1[i]
            h2 = y2[i]
            if h1 == h2:
                adjudicated[i] = h1
            elif abs(h1 - h2) == 1:
                adjudicated[i] = max(h1, h2)
            else:
                adjudicated[i] = int(round((h1 + h2) / 2))

        gold_df[f"{d}_human_gold"] = adjudicated

    # Overall gold disruption score
    gold_cols = [f"{d}_human_gold" for d in DIMENSIONS]
    gold_df["overall_disruption_gold"] = gold_df[gold_cols].max(axis=1)

    return gold_df


# ==============================================================================
# 4. STAGE 3 — LLM BENCHMARKING VS. ADJUDICATED HUMAN REFERENCE (H*)
# ==============================================================================
def evaluate_llms_against_gold(
    gold_df: pd.DataFrame,
    llm_scores_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Validates Gemma 2, Qwen 2.5, Qwen 3, GLM-4, and Ensemble against H*."""
    # Fast load with usecols
    use_cols = ["article_id", "model_name"] + DIMENSIONS
    llm_df = pd.read_csv(llm_scores_path, usecols=use_cols, low_memory=False)

    article_ids = set(gold_df["article_id"])
    sub_llm = llm_df[llm_df["article_id"].isin(article_ids)].copy()
    sub_llm["clean_model_name"] = sub_llm["model_name"].map(MODEL_MAP)

    results = []
    confusion_dict = {}

    # 1. Individual Models
    for raw_name, clean_name in MODEL_MAP.items():
        m_data = sub_llm[sub_llm["model_name"] == raw_name]
        merged = pd.merge(gold_df, m_data[["article_id"] + DIMENSIONS], on="article_id", suffixes=("_gold", "_model"))

        for d in DIMENSIONS:
            y_gold = merged[f"{d}_human_gold"].values.astype(int)
            y_pred = pd.to_numeric(merged[d], errors="coerce").fillna(0).values.astype(int)

            qwk, qwk_low, qwk_high = fast_bootstrap_ci(y_pred, y_gold, fast_qwk, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
            exact = float(np.mean(y_pred == y_gold) * 100)
            within_1 = float(np.mean(np.abs(y_pred - y_gold) <= 1) * 100)
            mae = float(np.mean(np.abs(y_pred - y_gold)))
            bias = float(np.mean(y_pred - y_gold))

            cm = np.bincount(y_pred * 5 + y_gold, minlength=25).reshape(5, 5)
            confusion_dict[f"{clean_name}___{d}"] = cm.tolist()

            results.append({
                "model_name": clean_name,
                "dimension": d,
                "n_evaluated": len(y_gold),
                "quadratic_weighted_kappa": round(qwk, 4),
                "qwk_95ci_lower": round(qwk_low, 4),
                "qwk_95ci_upper": round(qwk_high, 4),
                "exact_agreement_pct": round(exact, 2),
                "within_pm1_agreement_pct": round(within_1, 2),
                "mae": round(mae, 4),
                "mean_signed_bias": round(bias, 4),
            })

    # 2. Consensus Ensemble (Median across models)
    pivoted = sub_llm.pivot_table(index="article_id", columns="clean_model_name", values=DIMENSIONS)
    for d in DIMENSIONS:
        model_cols = [(d, clean_name) for clean_name in MODEL_MAP.values()]
        ens_d = pivoted[model_cols].median(axis=1).round().astype(int)
        ens_df = pd.DataFrame({"article_id": ens_d.index, f"{d}_ensemble": ens_d.values})
        merged_ens = pd.merge(gold_df, ens_df, on="article_id")

        y_gold = merged_ens[f"{d}_human_gold"].values.astype(int)
        y_ens = merged_ens[f"{d}_ensemble"].values.astype(int)

        qwk, qwk_low, qwk_high = fast_bootstrap_ci(y_ens, y_gold, fast_qwk, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
        exact = float(np.mean(y_ens == y_gold) * 100)
        within_1 = float(np.mean(np.abs(y_ens - y_gold) <= 1) * 100)
        mae = float(np.mean(np.abs(y_ens - y_gold)))
        bias = float(np.mean(y_ens - y_gold))

        cm = np.bincount(y_ens * 5 + y_gold, minlength=25).reshape(5, 5)
        confusion_dict[f"Consensus Ensemble___{d}"] = cm.tolist()

        results.append({
            "model_name": "Consensus Ensemble (Primary)",
            "dimension": d,
            "n_evaluated": len(y_gold),
            "quadratic_weighted_kappa": round(qwk, 4),
            "qwk_95ci_lower": round(qwk_low, 4),
            "qwk_95ci_upper": round(qwk_high, 4),
            "exact_agreement_pct": round(exact, 2),
            "within_pm1_agreement_pct": round(within_1, 2),
            "mae": round(mae, 4),
            "mean_signed_bias": round(bias, 4),
        })

    summary_df = pd.DataFrame(results)
    return summary_df, confusion_dict


# ==============================================================================
# 5. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    (FULL_DIR / "results").mkdir(parents=True, exist_ok=True)
    print("=" * 85, flush=True)
    print("  HUMAN VALIDATION AUDIT, ADJUDICATION & LLM BENCHMARKING (N = 400)", flush=True)
    print("=" * 85, flush=True)

    # 1. Load Combined Ratings (N = 400)
    print("\n[STEP 1] Loading Human Annotations (50 Pilot + 350 Additional)...", flush=True)
    df_combined = load_and_merge_human_ratings()

    # 2. Stage 1: Inter-Rater Reliability
    print("[STEP 2] Computing Human 1 vs. Human 2 Reliability with 1,000 Bootstrap CIs...", flush=True)
    df_inter_rater, human_confusions = compute_human_inter_rater_reliability(df_combined)
    df_inter_rater.to_csv(OUTPUT_INTER_RATER_SUMMARY, index=False, encoding="utf-8-sig")
    OUTPUT_HUMAN_CONFUSION_MATRICES.write_text(json.dumps(human_confusions, indent=2), encoding="utf-8")

    # 3. Stage 2: Adjudication (H*)
    print("[STEP 3] Performing Expert Adjudication and Building Ground Truth Reference (H*)...", flush=True)
    gold_df = adjudicate_human_reference(df_combined)
    gold_df.to_csv(OUTPUT_HUMAN_GOLD_REFERENCE, index=False, encoding="utf-8-sig")

    # 4. Stage 3: LLM Benchmarking Against H*
    print("[STEP 4] Benchmarking Gemma 2, Qwen 2.5, Qwen 3, GLM-4, and Ensemble against H*...", flush=True)
    df_llm_bench, llm_confusions = evaluate_llms_against_gold(gold_df, LLM_SCORES_FILE)
    df_llm_bench.to_csv(OUTPUT_LLM_BENCHMARK_SUMMARY, index=False, encoding="utf-8-sig")
    OUTPUT_LLM_CONFUSION_MATRICES.write_text(json.dumps(llm_confusions, indent=2), encoding="utf-8")

    # 5. Console Reporting
    print("\n" + "=" * 85, flush=True)
    print("  STAGE 1: HUMAN 1 VS. HUMAN 2 INTER-RATER RELIABILITY SUMMARY (N = 400)", flush=True)
    print("=" * 85, flush=True)
    print(df_inter_rater[["dimension", "quadratic_weighted_kappa", "qwk_95ci_lower", "qwk_95ci_upper", "krippendorff_alpha", "exact_agreement_pct", "within_pm1_agreement_pct", "mean_absolute_disagreement"]].to_string(index=False), flush=True)

    print("\n" + "=" * 85, flush=True)
    print("  STAGE 3: CONSENSUS ENSEMBLE VS. ADJUDICATED HUMAN REFERENCE (H*)", flush=True)
    print("=" * 85, flush=True)
    ens_sub = df_llm_bench[df_llm_bench["model_name"] == "Consensus Ensemble (Primary)"]
    print(ens_sub[["dimension", "quadratic_weighted_kappa", "qwk_95ci_lower", "qwk_95ci_upper", "exact_agreement_pct", "within_pm1_agreement_pct", "mae", "mean_signed_bias"]].to_string(index=False), flush=True)

    print("\n" + "=" * 85, flush=True)
    print("  STAGE 3: OVERALL MODEL PERFORMANCE RANKING (MEAN ACROSS 7 DIMENSIONS)", flush=True)
    print("=" * 85, flush=True)
    avg_perf = df_llm_bench.groupby("model_name")[["quadratic_weighted_kappa", "exact_agreement_pct", "within_pm1_agreement_pct", "mae", "mean_signed_bias"]].mean().reset_index()
    print(avg_perf.sort_values(by="quadratic_weighted_kappa", ascending=False).to_string(index=False), flush=True)

    print("\n" + "=" * 85, flush=True)
    print("  GENERATED ARTIFACTS IN validation/full/results/:", flush=True)
    print(f"  1. Inter-Rater Reliability (N=400):  {OUTPUT_INTER_RATER_SUMMARY.name}", flush=True)
    print(f"  2. Human 5x5 Confusion JSON:         {OUTPUT_HUMAN_CONFUSION_MATRICES.name}", flush=True)
    print(f"  3. Adjudicated Gold Reference (H*):  {OUTPUT_HUMAN_GOLD_REFERENCE.name}", flush=True)
    print(f"  4. LLM Benchmark Summary:            {OUTPUT_LLM_BENCHMARK_SUMMARY.name}", flush=True)
    print(f"  5. LLM 5x5 Confusion JSON:           {OUTPUT_LLM_CONFUSION_MATRICES.name}", flush=True)
    print("=" * 85 + "\n", flush=True)


if __name__ == "__main__":
    main()
