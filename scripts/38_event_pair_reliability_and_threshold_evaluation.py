#!/usr/bin/env python3
"""
Event Pair Deduplication Inter-Annotator Agreement (IAA) & Threshold Evaluation Pipeline
========================================================================================
This script processes the completed 300-pair human validation package across:
  - Threshold 0.82 (75 pairs)
  - Threshold 0.85 (75 pairs)
  - Threshold 0.88 (75 pairs)
  - Threshold 0.90 (75 pairs)

Outputs:
  1. Part 1: Human 1 vs. Human 2 Inter-Annotator Agreement (Cohen's Kappa, Krippendorff's Alpha, 95% Bootstrap CIs)
  2. Part 2: Expert Adjudication & Ground Truth Construction (H*)
  3. Part 3: Threshold Evaluation Matrix (Precision, Recall, F1, F0.5, False Mergers, Corpus Impact)
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import krippendorff

# ==============================================================================
# 0. CONFIGURABLE PATHS & CONSTANTS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = PROJECT_ROOT / "validation"
EVENT_DIR = VAL_DIR / "event_pairs"
COMPLETED_DIR = EVENT_DIR / "completed"
INPUT_DIR = EVENT_DIR / "inputs"
RESULTS_DIR = EVENT_DIR / "results"

MASTER_AUDIT_FILE = INPUT_DIR / "event_pair_master_audit_300.csv"
CANDIDATE_POOL_FILE = PROJECT_ROOT / "data/03_processed/news_full_text/pre_llm_filter/event_dedup/canonical_event_pair_audit_by_profile.csv"

# Output files
OUTPUT_IAA_OVERALL = RESULTS_DIR / "event_pair_iaa_overall.csv"
OUTPUT_IAA_BY_STRATUM = RESULTS_DIR / "event_pair_iaa_by_stratum.csv"
OUTPUT_GROUND_TRUTH = RESULTS_DIR / "event_pair_adjudicated_ground_truth_300.csv"
OUTPUT_THRESHOLD_MATRIX = RESULTS_DIR / "event_threshold_evaluation_matrix.csv"

STRATA = ["0_82", "0_85", "0_88", "0_90"]
CANDIDATE_THRESHOLDS = [0.80, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.92]
N_BOOTSTRAP = 1000
RANDOM_SEED = 42


# ==============================================================================
# 1. DATA LOADING & MERGING
# ==============================================================================
def load_completed_event_pairs() -> pd.DataFrame:
    """Loads and concatenates completed annotator files across all 4 threshold strata."""
    master_df = pd.read_csv(MASTER_AUDIT_FILE)
    strata_dfs = []

    for stratum in STRATA:
        p1 = COMPLETED_DIR / stratum / "event_pair_annotator_1.csv"
        p2 = COMPLETED_DIR / stratum / "event_pair_annotator_2.csv"

        if not p1.is_file() or not p2.is_file():
            raise FileNotFoundError(f"Missing completed files in {COMPLETED_DIR / stratum}")

        df1 = pd.read_csv(p1)
        df2 = pd.read_csv(p2)

        merged_stratum = pd.merge(
            df1[["pair_id", "same_event", "confidence", "notes"]],
            df2[["pair_id", "same_event", "confidence", "notes"]],
            on="pair_id",
            suffixes=("_ann1", "_ann2"),
        )
        merged_stratum["stratum_folder"] = stratum
        strata_dfs.append(merged_stratum)

    combined_annotated = pd.concat(strata_dfs, ignore_index=True)
    full_df = pd.merge(
        combined_annotated,
        master_df[["pair_id", "cosine_similarity", "target_threshold_band", "stratum_label", "article_id_a", "article_id_b", "title_a", "title_b", "event_day_a", "event_day_b", "day_difference", "source_a", "source_b"]],
        on="pair_id",
    )

    assert len(full_df) == 300, f"Expected 300 pairs, got {len(full_df)}"
    return full_df


# ==============================================================================
# 2. PART 1 — INTER-ANNOTATOR AGREEMENT (IAA)
# ==============================================================================
def compute_bootstrap_ci(y1: np.ndarray, y2: np.ndarray, metric_fn, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Computes point estimate and 95% bootstrap percentile CI."""
    rng = np.random.default_rng(seed)
    n = len(y1)
    point = metric_fn(y1, y2)
    indices = rng.choice(n, size=(n_boot, n), replace=True)
    boot_vals = []
    for idx in indices:
        v = metric_fn(y1[idx], y2[idx])
        if not np.isnan(v):
            boot_vals.append(v)
    if len(boot_vals) == 0:
        return point, point, point
    lower = float(np.percentile(boot_vals, 2.5))
    upper = float(np.percentile(boot_vals, 97.5))
    return point, lower, upper


def evaluate_iaa(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates comprehensive reliability metrics overall and per stratum."""
    y1_all = df["same_event_ann1"].astype(int).values
    y2_all = df["same_event_ann2"].astype(int).values

    # Overall Metrics
    exact_all = float(np.mean(y1_all == y2_all) * 100)
    k_pt, k_low, k_high = compute_bootstrap_ci(y1_all, y2_all, cohen_kappa_score, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)

    def calc_alpha(a, b):
        return krippendorff.alpha(reliability_data=np.vstack([a, b]), level_of_measurement="nominal")

    a_pt, a_low, a_high = compute_bootstrap_ci(y1_all, y2_all, calc_alpha, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)

    cm = confusion_matrix(y1_all, y2_all, labels=[0, 1])

    overall_summary = pd.DataFrame([{
        "sample_size": len(y1_all),
        "exact_agreement_pct": round(exact_all, 2),
        "unanimous_pairs": int(np.sum(y1_all == y2_all)),
        "disagreements": int(np.sum(y1_all != y2_all)),
        "cohen_kappa": round(k_pt, 4),
        "kappa_95ci_lower": round(k_low, 4),
        "kappa_95ci_upper": round(k_high, 4),
        "krippendorff_alpha": round(a_pt, 4),
        "alpha_95ci_lower": round(a_low, 4),
        "alpha_95ci_upper": round(a_high, 4),
        "true_negatives_unanimous_0": int(cm[0, 0]),
        "true_positives_unanimous_1": int(cm[1, 1]),
        "ann1_0_ann2_1": int(cm[0, 1]),
        "ann1_1_ann2_0": int(cm[1, 0]),
    }])

    # Stratum-Level Metrics
    stratum_records = []
    for st in STRATA:
        sub = df[df["stratum_folder"] == st]
        s1 = sub["same_event_ann1"].astype(int).values
        s2 = sub["same_event_ann2"].astype(int).values

        ex = float(np.mean(s1 == s2) * 100)
        k = float(cohen_kappa_score(s1, s2))
        a = float(krippendorff.alpha(reliability_data=np.vstack([s1, s2]), level_of_measurement="nominal"))
        cm_s = confusion_matrix(s1, s2, labels=[0, 1])

        stratum_records.append({
            "stratum_band": st,
            "sample_size": len(s1),
            "exact_agreement_pct": round(ex, 2),
            "cohen_kappa": round(k, 4),
            "krippendorff_alpha": round(a, 4),
            "unanimous_0": int(cm_s[0, 0]),
            "unanimous_1": int(cm_s[1, 1]),
            "disagreements": int(np.sum(s1 != s2)),
        })

    stratum_summary = pd.DataFrame(stratum_records)
    return overall_summary, stratum_summary


# ==============================================================================
# 3. PART 2 — ADJUDICATION & GROUND TRUTH (H*)
# ==============================================================================
def adjudicate_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Creates the Adjudicated Ground Truth Reference H* for all 300 pairs."""
    y1 = df["same_event_ann1"].astype(int).values
    y2 = df["same_event_ann2"].astype(int).values

    # Adjudication logic: Unanimous cases match. Disagreements resolved by positive evidence anchor
    adjudicated = np.where(y1 == y2, y1, np.maximum(y1, y2))

    df_gold = df.copy()
    df_gold["adjudicated_same_event"] = adjudicated
    return df_gold


# ==============================================================================
# 4. PART 3 — THRESHOLD EVALUATION MATRIX
# ==============================================================================
def evaluate_threshold_matrix(df_gold: pd.DataFrame) -> pd.DataFrame:
    """Calculates Sample-Level and Population-Weighted Threshold Metrics."""
    # Load candidate population counts
    raw_candidates = pd.read_csv(CANDIDATE_POOL_FILE)
    monthly_cand = raw_candidates[raw_candidates["event_profile"] == "monthly"]

    # Stratum population weights
    n_082 = len(monthly_cand[(monthly_cand["cosine_similarity"] >= 0.80) & (monthly_cand["cosine_similarity"] < 0.835)])
    n_085 = len(monthly_cand[(monthly_cand["cosine_similarity"] >= 0.835) & (monthly_cand["cosine_similarity"] < 0.865)])
    n_088 = len(monthly_cand[(monthly_cand["cosine_similarity"] >= 0.865) & (monthly_cand["cosine_similarity"] < 0.890)])
    n_090 = len(monthly_cand[monthly_cand["cosine_similarity"] >= 0.890])

    weights_dict = {
        "0_82": n_082 / 75.0,
        "0_85": n_085 / 75.0,
        "0_88": n_088 / 75.0,
        "0_9": n_090 / 75.0,
    }
    df_gold["stratum_weight"] = df_gold["stratum_folder"].map(weights_dict)

    sims = df_gold["cosine_similarity"].values
    gold = df_gold["adjudicated_same_event"].values
    weights = df_gold["stratum_weight"].values

    records = []
    for th in CANDIDATE_THRESHOLDS:
        pred = (sims >= th).astype(int)

        # 1. Sample Metrics (Unweighted N = 300)
        prec = float(precision_score(gold, pred, zero_division=0) * 100)
        rec = float(recall_score(gold, pred, zero_division=0) * 100)
        f1 = float(f1_score(gold, pred, zero_division=0))
        f05 = float((1 + 0.25) * (prec/100 * rec/100) / (0.25 * prec/100 + rec/100)) if (prec + rec) > 0 else 0.0
        acc = float(accuracy_score(gold, pred) * 100)

        fp = int(np.sum((pred == 1) & (gold == 0)))
        fn = int(np.sum((pred == 0) & (gold == 1)))
        tp = int(np.sum((pred == 1) & (gold == 1)))
        tn = int(np.sum((pred == 0) & (gold == 0)))

        # 2. Stratum-Weighted Population Metrics (N_pop = 2,613)
        tp_w = np.sum(weights * (pred == 1) * (gold == 1))
        fp_w = np.sum(weights * (pred == 1) * (gold == 0))
        fn_w = np.sum(weights * (pred == 0) * (gold == 1))

        prec_w = float(tp_w / (tp_w + fp_w) * 100) if (tp_w + fp_w) > 0 else 0.0
        rec_w = float(tp_w / (tp_w + fn_w) * 100) if (tp_w + fn_w) > 0 else 0.0
        f1_w = float(2 * prec_w * rec_w / (prec_w + rec_w) / 100) if (prec_w + rec_w) > 0 else 0.0

        # 3. Corpus Clustered Pairs
        corpus_pairs = len(monthly_cand[monthly_cand["cosine_similarity"] >= th])

        records.append({
            "threshold": round(th, 2),
            "sample_precision_pct": round(prec, 2),
            "sample_recall_pct": round(rec, 2),
            "sample_f1_score": round(f1, 4),
            "sample_f05_score": round(f05, 4),
            "sample_accuracy_pct": round(acc, 2),
            "false_mergers_fp": fp,
            "missed_duplicates_fn": fn,
            "true_duplicates_tp": tp,
            "true_distinct_tn": tn,
            "pop_precision_pct": round(prec_w, 2),
            "pop_recall_pct": round(rec_w, 2),
            "pop_f1_score": round(f1_w, 4),
            "qualified_pairs_in_corpus": corpus_pairs,
        })

    return pd.DataFrame(records)


# ==============================================================================
# 5. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 90, flush=True)
    print("  EVENT PAIR DEDUPLICATION: HUMAN RELIABILITY & THRESHOLD BENCHMARK (N = 300)", flush=True)
    print("=" * 90, flush=True)

    # 1. Load Completed Pairs
    print("\n[STEP 1] Loading completed annotations across 4 threshold strata...", flush=True)
    df_pairs = load_completed_event_pairs()

    # 2. Inter-Annotator Agreement (IAA)
    print("[STEP 2] Computing Human 1 vs. Human 2 Inter-Annotator Agreement...", flush=True)
    df_iaa_overall, df_iaa_stratum = evaluate_iaa(df_pairs)
    df_iaa_overall.to_csv(OUTPUT_IAA_OVERALL, index=False, encoding="utf-8-sig")
    df_iaa_stratum.to_csv(OUTPUT_IAA_BY_STRATUM, index=False, encoding="utf-8-sig")

    # 3. Adjudicate Ground Truth Reference (H*)
    print("[STEP 3] Generating Adjudicated Ground Truth Reference (H*)...", flush=True)
    df_gold = adjudicate_ground_truth(df_pairs)
    df_gold.to_csv(OUTPUT_GROUND_TRUTH, index=False, encoding="utf-8-sig")

    # 4. Threshold Evaluation Matrix
    print("[STEP 4] Calculating Comprehensive Threshold Evaluation Matrix...", flush=True)
    df_matrix = evaluate_threshold_matrix(df_gold)
    df_matrix.to_csv(OUTPUT_THRESHOLD_MATRIX, index=False, encoding="utf-8-sig")

    # 5. Console Reporting
    print("\n" + "=" * 90, flush=True)
    print("  PART 1: HUMAN RELIABILITY CHECK (INTER-ANNOTATOR AGREEMENT)", flush=True)
    print("=" * 90, flush=True)
    print(f"Total Evaluated Pairs:         {df_iaa_overall['sample_size'].values[0]}")
    print(f"Exact Agreement (%):           {df_iaa_overall['exact_agreement_pct'].values[0]}% ({df_iaa_overall['unanimous_pairs'].values[0]} / {df_iaa_overall['sample_size'].values[0]})")
    print(f"Cohen's Kappa (kappa):         {df_iaa_overall['cohen_kappa'].values[0]} [95% CI: {df_iaa_overall['kappa_95ci_lower'].values[0]}, {df_iaa_overall['kappa_95ci_upper'].values[0]}]")
    print(f"Krippendorff's Alpha (alpha):  {df_iaa_overall['krippendorff_alpha'].values[0]} [95% CI: {df_iaa_overall['alpha_95ci_lower'].values[0]}, {df_iaa_overall['alpha_95ci_upper'].values[0]}]")
    print(f"Total Disagreements:           {df_iaa_overall['disagreements'].values[0]}")

    print("\nIAA BY STRATUM BAND:")
    print(df_iaa_stratum.to_string(index=False), flush=True)

    print("\n" + "=" * 90, flush=True)
    print("  PART 2: THRESHOLD EVALUATION MATRIX (SAMPLE & POPULATION METRICS)", flush=True)
    print("=" * 90, flush=True)
    print(df_matrix[["threshold", "sample_precision_pct", "sample_recall_pct", "sample_f1_score", "sample_f05_score", "false_mergers_fp", "missed_duplicates_fn", "pop_precision_pct", "pop_recall_pct", "pop_f1_score", "qualified_pairs_in_corpus"]].to_string(index=False), flush=True)

    print("\n" + "=" * 90, flush=True)
    print("  EXPORTED ARTIFACTS IN validation/event_pairs/results/:", flush=True)
    print(f"  1. Overall IAA Summary:         {OUTPUT_IAA_OVERALL.name}")
    print(f"  2. IAA by Stratum Summary:      {OUTPUT_IAA_BY_STRATUM.name}")
    print(f"  3. Adjudicated Ground Truth:    {OUTPUT_GROUND_TRUTH.name}")
    print(f"  4. Threshold Evaluation Matrix: {OUTPUT_THRESHOLD_MATRIX.name}")
    print("=" * 90 + "\n", flush=True)


if __name__ == "__main__":
    main()
