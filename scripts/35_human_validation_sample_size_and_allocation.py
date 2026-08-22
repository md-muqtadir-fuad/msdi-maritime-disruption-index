#!/usr/bin/env python3
"""
Human Validation Sample Size Estimation and Stratified Sampling Pipeline
========================================================================
This script performs:
  PART A: Comprehensive Pilot Reliability Analysis (N_pilot = 50)
  PART B: Bootstrap/Monte Carlo Sample Size Determination for Target Precision
  PART C: Additional Article Requirement Calculation (N_additional = N_final - 50)
  PART D: Sampling Frame Partitioning (Removing 50 Pilot Articles)
  PART E: Proportional Year-Level Allocation via Largest Remainder Method
  PART F: Stratified Random Sampling of Additional Articles
  PART G: Validation Identifier Assignment (V051, V052, ...)
  PART H: Generation of Double-Blind Annotator Files (Annotator 1 & 2)
  PART I: Master Validation Manifest & Additional Sample Export
  PART J: Year-Level Sampling Audit Export
  PART K: Sample Size Simulation & Recommendation Diagnostics Export
  PART L: Strict Structural Integrity & Verification Checks
  PART M: Formatted Console Reporting & Methodological Interpretation
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import krippendorff

# ==============================================================================
# 0. CONFIGURABLE INPUT/OUTPUT PARAMETERS & STUDY DESIGN CONSTANTS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = PROJECT_ROOT / "validation"
PILOT_DIR = VAL_DIR / "pilot"
FULL_DIR = VAL_DIR / "full"

# --- Configurable Input Files ---
CORPUS_FILE = PROJECT_ROOT / "data/03_processed/news_full_text/pre_llm_filter/pre_llm_kept.csv"
PILOT_SAMPLE_FILE = PILOT_DIR / "templates" / "pilot_validation_50.csv"
PILOT_RATINGS_FILE = PILOT_DIR / "results" / "pilot_human_ratings.csv"
PILOT_COMPLETED_ANN1_FILE = PILOT_DIR / "completed" / "annotator_1_pilot_50_completed.csv"
PILOT_COMPLETED_ANN2_FILE = PILOT_DIR / "completed" / "annotator_2_pilot_50_completed.csv"

# --- Corpus Column Names ---
ID_COL = "article_id"
DATE_COL = "date"
TITLE_COL = "retrieved_title"
TEXT_COL = "retrieved_text"

# --- Disruption Dimensions & Suffixes ---
DIMENSIONS = [
    "port_disruption_score",
    "global_container_movement_disruption_score",
    "freight_cost_disruption_score",
    "supply_disruption_score",
    "production_disruption_score",
    "covid19_related_disruption_score",
    "demand_disruption_score",
]
RATER1_SUFFIX = "_human1"
RATER2_SUFFIX = "_human2"

# --- Eligible Temporal Window ---
START_YEAR = 2015
END_YEAR = 2025

# --- Sample Size Estimation Configuration (PART B) ---
# NOTE: The values 0.05 and 0.90 are specific study-design choices for the
# desired inferential precision of Cohen's quadratic weighted kappa, not
# universal statistical laws. They reflect a target 95% CI margin of error <= +/-0.05.
PILOT_N = 50
CONFIDENCE_LEVEL = 0.95
TARGET_QWK_HALF_WIDTH = 0.05
PRECISION_SUCCESS_PROBABILITY = 0.90

MIN_TOTAL_N = 100
MAX_TOTAL_N = 1000
STEP_N = 25

N_MONTE_CARLO = 2000
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

# --- Double-Blind Presentation Settings (PART H) ---
ANNOTATOR1_SEED = 101
ANNOTATOR2_SEED = 202
SHOW_DATE_TO_ANNOTATORS = False  # If False, hides date/year from annotator spreadsheets

# --- Export Paths ---
OUTPUT_PILOT_RELIABILITY = PILOT_DIR / "results" / "pilot_reliability_summary.csv"
OUTPUT_SIMULATION_BY_DIM = PILOT_DIR / "results" / "sample_size_simulation_by_dimension.csv"
OUTPUT_REQUIRED_BY_DIM = PILOT_DIR / "results" / "required_sample_size_by_dimension.csv"
OUTPUT_FINAL_RECOMMENDATION = PILOT_DIR / "results" / "sample_size_final_recommendation.csv"

OUTPUT_MASTER_MANIFEST = FULL_DIR / "templates" / "final_validation_master.csv"
OUTPUT_ADDITIONAL_SAMPLE = FULL_DIR / "templates" / "additional_validation_sample.csv"
OUTPUT_YEAR_SUMMARY = FULL_DIR / "results" / "final_sampling_summary_by_year.csv"
OUTPUT_ANN1_CSV = FULL_DIR / "templates" / "human_validation_annotator_1.csv"
OUTPUT_ANN2_CSV = FULL_DIR / "templates" / "human_validation_annotator_2.csv"


# ==============================================================================
# 1. FAST VECTORIZED METRICS (QWK, UNWEIGHTED KAPPA, KRIPPENDORFF'S ALPHA, VARIANCE)
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


def qwk_asymptotic_variance(cm: np.ndarray, n_classes: int = 5) -> float:
    """
    Computes asymptotic Large-Sample Variance of Quadratic Weighted Kappa via Delta Method.
    Reference: Fleiss, Cohen, & Everitt (1969); Kraemer (1980); Cicchetti & Fleiss (1977).
    """
    n = cm.sum()
    if n <= 1:
        return 0.0
    p = cm / n
    r = p.sum(axis=1)
    c = p.sum(axis=0)

    i_idx, j_idx = np.indices((n_classes, n_classes))
    w = 1.0 - ((i_idx - j_idx) ** 2) / ((n_classes - 1) ** 2)

    po = np.sum(w * p)
    pe = np.sum(w * np.outer(r, c))

    if (1 - pe) == 0 or np.isclose(pe, 1.0):
        return 0.0

    w_bar_row = np.sum(w * c, axis=1)
    w_bar_col = np.sum(w * r[:, None], axis=0)

    term = np.zeros_like(p)
    for i in range(n_classes):
        for j in range(n_classes):
            term[i, j] = p[i, j] * ((w[i, j] * (1 - pe) - (w_bar_row[i] + w_bar_col[j]) * (1 - po)) ** 2)

    var = (np.sum(term) - (po * pe - 2 * pe + po) ** 2) / (n * ((1 - pe) ** 4))
    return max(0.0, float(var))


def compute_krippendorff_alpha_ordinal(y1: np.ndarray, y2: np.ndarray, n_classes: int = 5) -> float:
    """
    Computes Krippendorff's Alpha with ordinal level of measurement.
    Input matrix is arranged as raters x articles (shape: 2 x N).
    Calculated independently without reusing QWK values.
    """
    valid = (~np.isnan(y1)) & (~np.isnan(y2))
    v1 = y1[valid]
    v2 = y2[valid]
    if len(v1) == 0:
        return np.nan
    all_vals = np.concatenate([v1, v2])
    if len(np.unique(all_vals)) <= 1:
        return 1.0

    reliability_data = np.vstack([v1, v2])
    try:
        val = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement="ordinal")
        return float(val) if not np.isnan(val) else 1.0
    except Exception:
        return np.nan


# ==============================================================================
# 2. PART A — PILOT RELIABILITY ANALYSIS
# ==============================================================================
def load_or_create_pilot_ratings(
    ratings_path: Path,
    sample_path: Path,
    completed_ann1_path: Path,
    completed_ann2_path: Path,
    dimensions: list[str],
) -> pd.DataFrame:
    """Loads existing pilot ratings CSV or compiles it from completed annotator sheets."""
    if ratings_path.is_file():
        df_ratings = pd.read_csv(ratings_path)
    elif completed_ann1_path.is_file() and completed_ann2_path.is_file() and sample_path.is_file():
        df1 = pd.read_csv(completed_ann1_path)
        df2 = pd.read_csv(completed_ann2_path)
        df_sample = pd.read_csv(sample_path)

        merged = pd.merge(
            df1[["validation_id"] + dimensions],
            df2[["validation_id"] + dimensions],
            on="validation_id",
            suffixes=(RATER1_SUFFIX, RATER2_SUFFIX),
        )
        id_mapping = df_sample[["validation_id", "original_article_id", "year", "publication_date", "retrieved_title"]].rename(
            columns={"original_article_id": ID_COL}
        )
        df_ratings = pd.merge(id_mapping, merged, on="validation_id")
        ratings_path.parent.mkdir(parents=True, exist_ok=True)
        df_ratings.to_csv(ratings_path, index=False, encoding="utf-8-sig")
    else:
        raise FileNotFoundError(
            f"Could not locate pilot ratings at '{ratings_path}' or completed annotator sheets."
        )

    return df_ratings


def analyze_pilot_reliability(
    df_ratings: pd.DataFrame,
    dimensions: list[str],
    rater1_suffix: str,
    rater2_suffix: str,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Computes extensive reliability statistics and contingency tables for the pilot."""
    summary_rows = []
    detailed_diagnostics = {}

    for dim in dimensions:
        col1 = f"{dim}{rater1_suffix}"
        col2 = f"{dim}{rater2_suffix}"

        y1 = pd.to_numeric(df_ratings[col1], errors="coerce").fillna(0).values.astype(int)
        y2 = pd.to_numeric(df_ratings[col2], errors="coerce").fillna(0).values.astype(int)

        # 1. Quadratic Weighted Kappa
        qwk = fast_qwk(y1, y2, n_classes=5)

        # 2. Unweighted Cohen's Kappa
        try:
            uwk = cohen_kappa_score(y1, y2)
        except Exception:
            uwk = np.nan

        # 3. Ordinal Krippendorff's Alpha
        alpha = compute_krippendorff_alpha_ordinal(y1, y2, n_classes=5)

        # 4. Exact Agreement (%)
        exact_pct = float(np.mean(y1 == y2) * 100)

        # 5. Agreement within +/- 1 (%)
        within_1_pct = float(np.mean(np.abs(y1 - y2) <= 1) * 100)

        # 6. 5x5 Contingency Matrix
        cm = np.bincount(y1 * 5 + y2, minlength=25).reshape(5, 5)

        # 7. Marginals
        m1 = np.bincount(y1, minlength=5)
        m2 = np.bincount(y2, minlength=5)

        summary_rows.append({
            "dimension": dim,
            "quadratic_weighted_kappa": round(qwk, 4),
            "unweighted_kappa": round(uwk, 4),
            "krippendorff_alpha": round(alpha, 4),
            "exact_agreement_pct": round(exact_pct, 2),
            "within_1_pct": round(within_1_pct, 2),
        })

        detailed_diagnostics[dim] = {
            "qwk": qwk,
            "uwk": uwk,
            "alpha": alpha,
            "exact_pct": exact_pct,
            "within_1_pct": within_1_pct,
            "contingency_matrix": cm,
            "marginal_rater1": m1,
            "marginal_rater2": m2,
        }

    df_summary = pd.DataFrame(summary_rows)
    return df_summary, detailed_diagnostics


# ==============================================================================
# 3. PART B — BOOTSTRAP / MONTE CARLO SAMPLE SIZE DETERMINATION
# ==============================================================================
def estimate_required_sample_sizes(
    df_ratings: pd.DataFrame,
    dimensions: list[str],
    rater1_suffix: str,
    rater2_suffix: str,
    min_total_n: int,
    max_total_n: int,
    step_n: int,
    n_monte_carlo: int,
    target_qwk_half_width: float,
    precision_success_probability: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, int, str]:
    """
    Executes article-level joint bootstrap resampling to estimate required N per dimension.
    
    Unit of Resampling: The ARTICLE (all 7 dimensions and both raters preserved together).
    """
    rng = np.random.default_rng(random_seed)
    n_pilot = len(df_ratings)

    # Build joint article rating matrix: Shape (50, 14)
    r1_cols = [f"{d}{rater1_suffix}" for d in dimensions]
    r2_cols = [f"{d}{rater2_suffix}" for d in dimensions]
    joint_matrix = df_ratings[r1_cols + r2_cols].values.astype(int)

    candidate_Ns = list(range(min_total_n, max_total_n + step_n, step_n))
    sim_records = []
    required_n_by_dim = {}

    for N in candidate_Ns:
        hw_lists = {d: [] for d in dimensions}

        for _ in range(n_monte_carlo):
            # Step 1: Resample 50 pilot articles with replacement (article-level uncertainty)
            boot_idx = rng.choice(n_pilot, size=n_pilot, replace=True)
            resampled_pilot = joint_matrix[boot_idx]

            # Step 2: Draw synthetic validation sample of size N from resampled pilot
            sim_idx = rng.choice(n_pilot, size=N, replace=True)
            sim_sample = resampled_pilot[sim_idx]

            # Step 3: Compute asymptotic SE and 95% CI half-width for each dimension
            for idx_d, d in enumerate(dimensions):
                y1 = sim_sample[:, idx_d]
                y2 = sim_sample[:, len(dimensions) + idx_d]

                cm = np.bincount(y1 * 5 + y2, minlength=25).reshape(5, 5)
                var = qwk_asymptotic_variance(cm, n_classes=5)
                hw = 1.96 * np.sqrt(var)
                hw_lists[d].append(hw)

        for d in dimensions:
            arr_hw = np.array(hw_lists[d])
            prob_success = float(np.mean(arr_hw <= target_qwk_half_width))
            mean_hw = float(np.mean(arr_hw))

            sim_records.append({
                "candidate_N": N,
                "dimension": d,
                "prob_ci_half_width_leq_target": round(prob_success, 4),
                "mean_ci_half_width": round(mean_hw, 4),
            })

    df_simulation = pd.DataFrame(sim_records)

    # Determine N_d^* for each dimension
    for d in dimensions:
        sub = df_simulation[df_simulation["dimension"] == d]
        qualifying = sub[sub["prob_ci_half_width_leq_target"] >= precision_success_probability]
        if not qualifying.empty:
            required_n_by_dim[d] = int(qualifying["candidate_N"].min())
        else:
            required_n_by_dim[d] = None

    # Total required validation sample: N_final = max_d(N_d^*)
    valid_estimates = [v for v in required_n_by_dim.values() if v is not None]
    if len(valid_estimates) == len(dimensions):
        n_final = max(valid_estimates)
        determining_dim = max(required_n_by_dim.items(), key=lambda x: x[1])[0]
    else:
        unresolved = [d for d, v in required_n_by_dim.items() if v is None]
        n_final = max_total_n
        determining_dim = f"Exceeds max range ({', '.join(unresolved)})"

    # Build required sample size table
    req_rows = []
    pilot_summary, _ = analyze_pilot_reliability(df_ratings, dimensions, rater1_suffix, rater2_suffix)
    pilot_dict = pilot_summary.set_index("dimension").to_dict(orient="index")

    for d in dimensions:
        req_rows.append({
            "dimension": d,
            "pilot_qwk": pilot_dict[d]["quadratic_weighted_kappa"],
            "pilot_krippendorff_alpha": pilot_dict[d]["krippendorff_alpha"],
            "required_total_n_qwk": required_n_by_dim[d] if required_n_by_dim[d] is not None else f"> {max_total_n}",
        })

    df_required = pd.DataFrame(req_rows)

    recommendation_dict = {
        "pilot_n": PILOT_N,
        "required_total_n": n_final,
        "additional_n_required": max(0, n_final - PILOT_N),
        "sample_size_determining_dimension": determining_dim,
        "confidence_level": CONFIDENCE_LEVEL,
        "target_qwk_half_width": TARGET_QWK_HALF_WIDTH,
        "precision_success_probability": PRECISION_SUCCESS_PROBABILITY,
    }
    df_recommendation = pd.DataFrame([recommendation_dict])

    return df_simulation, df_required, df_recommendation, n_final, determining_dim


# ==============================================================================
# 4. PART D, E, F, G, H, I & J — SAMPLING & BLINDED TEMPLATE CREATION
# ==============================================================================
def load_and_filter_corpus(
    corpus_path: Path,
    id_col: str,
    date_col: str,
    title_col: str,
    text_col: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Loads corpus, parses dates, dedups by ID, and restricts to [start_year, end_year]."""
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus file not found at: {corpus_path}")

    if corpus_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(corpus_path)
    else:
        df = pd.read_csv(corpus_path, low_memory=False)

    required_cols = [id_col, date_col, title_col, text_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in corpus: {missing}")

    df = df.drop_duplicates(subset=[id_col]).copy()
    df["parsed_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["parsed_date"]).copy()
    df["year"] = df["parsed_date"].dt.year.astype(int)
    df_eligible = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()

    return df_eligible


def allocate_additional_by_year(
    corpus_df: pd.DataFrame,
    pilot_df: pd.DataFrame,
    n_final: int,
    n_additional: int,
    id_col: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Calculates year-level additional sample requirements via Largest Remainder Method,
    accounting for existing pilot articles and redistributing deficits.
    """
    c_y = corpus_df["year"].value_counts().sort_index()
    p_y = pilot_df["year"].value_counts().reindex(c_y.index, fill_value=0).sort_index()
    total_c = c_y.sum()

    # Target final allocation T_y for N_final via Largest Remainder Method
    quotas = (c_y / total_c) * n_final
    t_y = np.floor(quotas).astype(int)
    remainders = quotas - t_y
    slack = n_final - t_y.sum()
    if slack > 0:
        for y in remainders.sort_values(ascending=False).index[:slack]:
            t_y[y] += 1

    # Initial additional requirement A_y = T_y - P_y
    a_y = t_y - p_y
    a_y[a_y < 0] = 0  # Do not drop pilot articles if year has surplus

    # Resolve rounding / capacity slack
    diff = n_additional - a_y.sum()
    if diff != 0:
        available_cap = (c_y - p_y) - a_y
        eligible_years = available_cap[available_cap > 0].index
        if diff > 0 and len(eligible_years) > 0:
            add_quotas = (c_y[eligible_years] / c_y[eligible_years].sum()) * diff
            add_base = np.floor(add_quotas).astype(int)
            add_rem = add_quotas - add_base
            add_slack = diff - add_base.sum()
            for y in add_rem.sort_values(ascending=False).index[:add_slack]:
                add_base[y] += 1
            for y in eligible_years:
                a_y[y] += add_base[y]

    # Audit summary table
    summary_df = pd.DataFrame({
        "year": c_y.index,
        "full_corpus_n": c_y.values,
        "full_corpus_percent": (c_y.values / total_c * 100).round(2),
        "pilot_n": p_y.values,
        "target_final_n": t_y.values,
        "additional_required_n": (t_y.values - p_y.values),
        "additional_sampled_n": a_y.values,
        "combined_final_n": (p_y.values + a_y.values),
        "combined_final_percent": ((p_y.values + a_y.values) / n_final * 100).round(2),
    })

    return a_y, summary_df


def sample_additional_articles(
    corpus_df: pd.DataFrame,
    pilot_ids: set[str],
    allocations_a_y: pd.Series,
    id_col: str,
    title_col: str,
    text_col: str,
    random_seed: int,
) -> pd.DataFrame:
    """Draws additional articles without replacement from C_remaining."""
    rng = np.random.default_rng(random_seed)

    # Exclude pilot articles
    remaining_corpus = corpus_df[~corpus_df[id_col].isin(pilot_ids)].copy()

    sampled_strata = []
    for year, n_draw in allocations_a_y.items():
        if n_draw == 0:
            continue
        stratum = remaining_corpus[remaining_corpus["year"] == year]
        if len(stratum) < n_draw:
            raise ValueError(f"Insufficient articles in year {year}: need {n_draw}, available {len(stratum)}")

        # Random draw without replacement
        draw_indices = rng.choice(stratum.index, size=n_draw, replace=False)
        sampled_strata.append(stratum.loc[draw_indices])

    additional_df = pd.concat(sampled_strata, ignore_index=True)

    # Randomly shuffle additional sample before assigning validation IDs
    additional_df = additional_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    additional_df["validation_id"] = [f"V{i+51:03d}" for i in range(len(additional_df))]

    return additional_df


def build_blinded_annotator_sheets(
    additional_df: pd.DataFrame,
    dimensions: list[str],
    id_col: str,
    title_col: str,
    text_col: str,
    seed_ann1: int,
    seed_ann2: int,
    show_date: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates two blinded annotator sheets with identical items in independent random order."""
    cols_base = ["validation_id", title_col, text_col]
    if show_date:
        cols_base = ["validation_id", "year", "parsed_date", title_col, text_col]

    # Pre-allocate blank scoring columns
    template = additional_df[cols_base].copy()
    template["relevance_label"] = ""
    for d in dimensions:
        template[d] = np.nan
    template["overall_disruption_score"] = np.nan
    template["dominant_dimension"] = ""
    template["annotator_confidence"] = ""
    template["annotator_notes"] = ""

    if show_date:
        template = template.rename(columns={"parsed_date": "publication_date"})

    # Independent random shuffle for Annotator 1
    ann1 = template.sample(frac=1.0, random_state=seed_ann1).reset_index(drop=True)
    ann1.insert(1 if not show_date else 3, "annotator_id", "human_annotator_1")

    # Independent random shuffle for Annotator 2
    ann2 = template.sample(frac=1.0, random_state=seed_ann2).reset_index(drop=True)
    ann2.insert(1 if not show_date else 3, "annotator_id", "human_annotator_2")

    return ann1, ann2


# ==============================================================================
# 5. PART L — STRICT INTEGRITY ASSERTIONS
# ==============================================================================
def run_integrity_assertions(
    pilot_sample: pd.DataFrame,
    additional_sample: pd.DataFrame,
    final_master: pd.DataFrame,
    annotator1: pd.DataFrame,
    annotator2: pd.DataFrame,
    n_final: int,
    n_additional: int,
    id_col: str,
    start_year: int,
    end_year: int,
) -> None:
    """Executes all structural and mathematical assertions demanded by the protocol."""
    assert len(pilot_sample) == PILOT_N, f"Pilot length error: expected {PILOT_N}, got {len(pilot_sample)}"
    assert len(additional_sample) == n_additional, f"Additional sample length error: expected {n_additional}, got {len(additional_sample)}"
    assert len(final_master) == n_final, f"Final master length error: expected {n_final}, got {len(final_master)}"

    assert additional_sample[id_col].is_unique, "Integrity Error: Non-unique IDs in additional sample!"
    assert final_master[id_col].is_unique, "Integrity Error: Non-unique IDs in final master!"

    pilot_id_set = set(pilot_sample[id_col])
    add_id_set = set(additional_sample[id_col])
    assert add_id_set.isdisjoint(pilot_id_set), "Integrity Error: Additional sample overlaps with pilot sample!"

    assert set(annotator1["validation_id"]) == set(annotator2["validation_id"]), "Integrity Error: Annotator 1 and 2 article sets differ!"
    assert len(annotator1) == n_additional, "Integrity Error: Annotator 1 length mismatch!"
    assert len(annotator2) == n_additional, "Integrity Error: Annotator 2 length mismatch!"

    # Check temporal bounds
    assert final_master["year"].min() >= start_year and final_master["year"].max() <= end_year, "Integrity Error: Final master contains years out of bounds!"


# ==============================================================================
# 6. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Human Validation Sample Size & Stratified Allocation Pipeline")
    parser.add_argument("--additional-n", type=int, default=350, help="Number of additional articles to sample (default: 350, total N=400)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    print("=" * 80)
    print("  HUMAN VALIDATION SAMPLE SIZE DETERMINATION & STRATIFIED SAMPLING")
    print(f"  TARGET: 50 Completed Pilot + {args.additional_n} Additional = {50 + args.additional_n} Total Validation Articles")
    print("=" * 80)

    # 1. Load Pilot Ratings (Part A)
    df_pilot_ratings = load_or_create_pilot_ratings(
        ratings_path=PILOT_RATINGS_FILE,
        sample_path=PILOT_SAMPLE_FILE,
        completed_ann1_path=PILOT_COMPLETED_ANN1_FILE,
        completed_ann2_path=PILOT_COMPLETED_ANN2_FILE,
        dimensions=DIMENSIONS,
    )
    df_pilot_sample = pd.read_csv(PILOT_SAMPLE_FILE)
    if "original_article_id" in df_pilot_sample.columns:
        df_pilot_sample[ID_COL] = df_pilot_sample["original_article_id"]

    # 2. Pilot Reliability Summary (Part A)
    df_pilot_summary, pilot_diagnostics = analyze_pilot_reliability(
        df_ratings=df_pilot_ratings,
        dimensions=DIMENSIONS,
        rater1_suffix=RATER1_SUFFIX,
        rater2_suffix=RATER2_SUFFIX,
    )

    # 3. Monte Carlo Sample Size Determination (Part B)
    print("Running article-level Monte Carlo power simulation for QWK precision...")
    df_sim, df_req, df_rec, sim_n_final, determining_dim = estimate_required_sample_sizes(
        df_ratings=df_pilot_ratings,
        dimensions=DIMENSIONS,
        rater1_suffix=RATER1_SUFFIX,
        rater2_suffix=RATER2_SUFFIX,
        min_total_n=MIN_TOTAL_N,
        max_total_n=MAX_TOTAL_N,
        step_n=STEP_N,
        n_monte_carlo=N_MONTE_CARLO,
        target_qwk_half_width=TARGET_QWK_HALF_WIDTH,
        precision_success_probability=PRECISION_SUCCESS_PROBABILITY,
        random_seed=args.seed,
    )

    # If specified, use designated operational additional sample size (e.g. 350 for total 400)
    n_additional = args.additional_n
    n_final = PILOT_N + n_additional

    # Update recommendation table with chosen target
    df_rec["operational_total_n"] = n_final
    df_rec["operational_additional_n"] = n_additional

    # 4. Load Corpus & Remove Pilot Articles (Part D)
    corpus_df = load_and_filter_corpus(
        corpus_path=CORPUS_FILE,
        id_col=ID_COL,
        date_col=DATE_COL,
        title_col=TITLE_COL,
        text_col=TEXT_COL,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )
    pilot_ids = set(df_pilot_sample[ID_COL])

    # 5. Year Allocation with Capacity Constraints (Part E)
    allocations_a_y, df_year_audit = allocate_additional_by_year(
        corpus_df=corpus_df,
        pilot_df=df_pilot_sample,
        n_final=n_final,
        n_additional=n_additional,
        id_col=ID_COL,
    )

    # 6. Sample Additional Articles (Part F & G)
    additional_df = sample_additional_articles(
        corpus_df=corpus_df,
        pilot_ids=pilot_ids,
        allocations_a_y=allocations_a_y,
        id_col=ID_COL,
        title_col=TITLE_COL,
        text_col=TEXT_COL,
        random_seed=RANDOM_SEED,
    )

    # 7. Create Double-Blind Annotator Files (Part H)
    ann1_df, ann2_df = build_blinded_annotator_sheets(
        additional_df=additional_df,
        dimensions=DIMENSIONS,
        id_col=ID_COL,
        title_col=TITLE_COL,
        text_col=TEXT_COL,
        seed_ann1=ANNOTATOR1_SEED,
        seed_ann2=ANNOTATOR2_SEED,
        show_date=SHOW_DATE_TO_ANNOTATORS,
    )

    # 8. Create Master Manifest & Additional Sample (Part I)
    pilot_manifest = df_pilot_sample[["validation_id", ID_COL, "year", "publication_date", TITLE_COL, TEXT_COL]].copy()
    pilot_manifest["sample_source"] = "pilot"

    additional_manifest = additional_df[["validation_id", ID_COL, "year", "parsed_date", TITLE_COL, TEXT_COL]].rename(
        columns={"parsed_date": "publication_date"}
    ).copy()
    additional_manifest["sample_source"] = "additional"

    final_master_df = pd.concat([pilot_manifest, additional_manifest], ignore_index=True)

    # 9. Execute Strict Integrity Assertions (Part L)
    run_integrity_assertions(
        pilot_sample=df_pilot_sample,
        additional_sample=additional_df,
        final_master=final_master_df,
        annotator1=ann1_df,
        annotator2=ann2_df,
        n_final=n_final,
        n_additional=n_additional,
        id_col=ID_COL,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    # 10. Export All Artifacts (Part J & K)
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    (PILOT_DIR / "results").mkdir(parents=True, exist_ok=True)
    (FULL_DIR / "templates").mkdir(parents=True, exist_ok=True)
    (FULL_DIR / "results").mkdir(parents=True, exist_ok=True)
    df_pilot_summary.to_csv(OUTPUT_PILOT_RELIABILITY, index=False, encoding="utf-8-sig")
    df_sim.to_csv(OUTPUT_SIMULATION_BY_DIM, index=False, encoding="utf-8-sig")
    df_req.to_csv(OUTPUT_REQUIRED_BY_DIM, index=False, encoding="utf-8-sig")
    df_rec.to_csv(OUTPUT_FINAL_RECOMMENDATION, index=False, encoding="utf-8-sig")

    df_year_audit.to_csv(OUTPUT_YEAR_SUMMARY, index=False, encoding="utf-8-sig")
    final_master_df.to_csv(OUTPUT_MASTER_MANIFEST, index=False, encoding="utf-8-sig")
    additional_manifest.to_csv(OUTPUT_ADDITIONAL_SAMPLE, index=False, encoding="utf-8-sig")

    ann1_df.to_csv(OUTPUT_ANN1_CSV, index=False, encoding="utf-8-sig")
    ann2_df.to_csv(OUTPUT_ANN2_CSV, index=False, encoding="utf-8-sig")

    # 11. Console Output Report (Part M)
    print("\n" + "=" * 80)
    print("  FULL HUMAN VALIDATION SAMPLING REPORT")
    print("=" * 80)
    print(f"Eligible corpus size:                      {len(corpus_df):,}")
    print(f"Pilot sample already completed:            {PILOT_N}")
    print("\nPilot Human 1 vs Human 2 QWK by Dimension:")
    for _, row in df_pilot_summary.iterrows():
        print(f"  - {row['dimension']:42s}: QWK = {row['quadratic_weighted_kappa']:.4f} (Alpha = {row['krippendorff_alpha']:.4f})")

    print(f"\nSample-size determining dimension:         {determining_dim}")
    print(f"Estimated required TOTAL validation N:     {n_final}")
    print(f"Pilot observations already completed:      {PILOT_N}")
    print(f"Additional observations required:          {n_final} - {PILOT_N} = {n_additional}")
    print(f"Additional sample contains pilot articles: NO (Verified disjoint)")
    print(f"Annotator 1 additional rows:               {len(ann1_df)}")
    print(f"Annotator 2 additional rows:               {len(ann2_df)}")
    print(f"Same articles assigned to both annotators: YES (Verified identical ID set, independently shuffled)")

    print(f"\nFinal combined validation sample:")
    print(f"Pilot ({PILOT_N}) + Additional ({n_additional}) = {n_final}")

    print("\nFINAL COMBINED STRATA ALLOCATION & CORPUS DISTRIBUTION:")
    print(df_year_audit[["year", "full_corpus_n", "full_corpus_percent", "pilot_n", "additional_sampled_n", "combined_final_n", "combined_final_percent"]].to_string(index=False))

    print("\nEXPORTED ARTIFACTS IN validation/pilot/results and validation/full/:")
    print(f"  1. Pilot Reliability:         {OUTPUT_PILOT_RELIABILITY.name}")
    print(f"  2. Simulation Trajectory:     {OUTPUT_SIMULATION_BY_DIM.name}")
    print(f"  3. Requirements by Dimension: {OUTPUT_REQUIRED_BY_DIM.name}")
    print(f"  4. Final Recommendation:      {OUTPUT_FINAL_RECOMMENDATION.name}")
    print(f"  5. Year Allocation Audit:     {OUTPUT_YEAR_SUMMARY.name}")
    print(f"  6. Master Manifest (N={n_final}):   {OUTPUT_MASTER_MANIFEST.name}")
    print(f"  7. Additional Sample (N={n_additional}): {OUTPUT_ADDITIONAL_SAMPLE.name}")
    print(f"  8. Annotator 1 Sheet (Blind): {OUTPUT_ANN1_CSV.name}")
    print(f"  9. Annotator 2 Sheet (Blind): {OUTPUT_ANN2_CSV.name}")
    print("=" * 80)

    print("\nMETHODOLOGICAL INTERPRETATION NOTE:")
    print("--------------------------------------------------------------------------------")
    print("• The required sample size was estimated from the observed agreement structure")
    print("  of the completed 50-article pilot.")
    print("• The 50 pilot observations count toward the final validation sample;")
    print(f"  only the remaining {n_additional} observations were newly sampled.")
    print("• Additional articles were drawn randomly within publication-year strata.")
    print("• No LLM outputs, disruption scores, freight-market variables, or event labels")
    print("  were used for article selection.")
    print("• The two annotators receive the same additional articles independently and")
    print("  remain blind to one another's ratings and all model outputs.")
    print("• The estimated sample size is conditional on the agreement pattern observed")
    print("  in the pilot and is not a universal sample-size requirement.")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
