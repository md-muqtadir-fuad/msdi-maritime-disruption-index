#!/usr/bin/env python3
"""
Reproducible Stratified-Random Pilot Validation Sampler (N = 50)
================================================================
This script extracts a strictly randomized, ex-ante stratified pilot validation
sample of 50 maritime news articles across the 2015-2025 period.

Outputs:
    validation/pilot/templates/pilot_validation_50.csv
    validation/pilot/templates/annotator_1_pilot_50.csv
    validation/pilot/templates/annotator_2_pilot_50.csv
    validation/pilot/results/pilot_sampling_summary.csv
    validation/pilot/annotation_rubric_guide.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ==============================================================================
# 0. CONFIGURABLE PARAMETERS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FILE_PATH = PROJECT_ROOT / "data/03_processed/news_full_text/pre_llm_filter/pre_llm_kept.csv"
ID_COL = "article_id"                 # Unique article identifier column
DATE_COL = "date"                     # Publication date column
TITLE_COL = "retrieved_title"         # Headline/title column
TEXT_COL = "retrieved_text"           # Full article body/text column

START_YEAR = 2015                     # Start of eligible sampling window
END_YEAR = 2025                       # End of eligible sampling window
SAMPLE_SIZE = 50                      # Exact target pilot validation sample size
RANDOM_SEED = 42                      # Master random seed for exact reproducibility

OUTPUT_DIR = PROJECT_ROOT / "validation" / "pilot"
TEMPLATE_DIR = OUTPUT_DIR / "templates"
RESULTS_DIR = OUTPUT_DIR / "results"
OUTPUT_SAMPLE_PATH = TEMPLATE_DIR / "pilot_validation_50.csv"
OUTPUT_ANN1_PATH = TEMPLATE_DIR / "annotator_1_pilot_50.csv"
OUTPUT_ANN2_PATH = TEMPLATE_DIR / "annotator_2_pilot_50.csv"
OUTPUT_SUMMARY_PATH = RESULTS_DIR / "pilot_sampling_summary.csv"
OUTPUT_GUIDE_PATH = OUTPUT_DIR / "annotation_rubric_guide.md"

# The 7 Disruption Taxonomy Criteria + Overall Evaluation
DISRUPTION_DIMENSIONS = [
    "port_disruption_score",                       # Port congestion, dock strikes, berth waiting
    "global_container_movement_disruption_score",  # Vessel rerouting, canal blockage, blank sailings
    "freight_cost_disruption_score",              # Spot rate spikes, surcharges, container lease costs
    "supply_disruption_score",                    # Shortages of raw materials/parts, inventory stockouts
    "production_disruption_score",                # Factory shutdowns, manufacturing halts, labor loss
    "covid19_related_disruption_score",           # Quarantine lockdowns, health protocols
    "demand_disruption_score",                    # Demand destruction, consumer goods surge volatility
]


# ==============================================================================
# 1. DATA LOADING AND CLEANING
# ==============================================================================
def load_and_clean_corpus(
    file_path: str | Path,
    id_col: str,
    date_col: str,
    title_col: str,
    text_col: str,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, int, int, int]:
    """
    Loads raw corpus, drops duplicate article IDs, parses dates, and restricts
    the corpus to the eligible year range [start_year, end_year].
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found at: {path.resolve()}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    n_raw = len(df)

    required_cols = [id_col, date_col, title_col, text_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in dataset: {missing}")

    # Remove duplicate records based on unique article ID
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=[id_col]).copy()
    n_duplicates_dropped = n_before_dedup - len(df)

    # Parse publication dates and extract integer calendar year
    df["parsed_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["parsed_date"]).copy()
    df["year"] = df["parsed_date"].dt.year.astype(int)

    # Filter to eligible sampling window [start_year, end_year]
    df_eligible = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    n_eligible = len(df_eligible)

    return df_eligible, n_raw, n_duplicates_dropped, n_eligible


# ==============================================================================
# 2. PROPORTIONAL INTEGER STRATUM ALLOCATION (LARGEST REMAINDER METHOD)
# ==============================================================================
def compute_strata_allocations(
    df: pd.DataFrame,
    target_sample_size: int,
) -> pd.Series:
    """
    Allocates sample sizes across years using the Largest Remainder (Hare-Niemeyer)
    method, ensuring:
      1. Proportional representation matching the population corpus.
      2. Every represented year receives at least 1 sample.
      3. Sum of allocations equals target_sample_size exactly.
    """
    yearly_counts = df["year"].value_counts().sort_index()
    total_population = len(df)
    n_years = len(yearly_counts)

    if target_sample_size < n_years:
        raise ValueError(
            f"Sample size ({target_sample_size}) must be >= number of represented years ({n_years})."
        )

    quotas = (yearly_counts / total_population) * target_sample_size
    allocations = np.maximum(1, np.floor(quotas)).astype(int)
    remainders = quotas - np.floor(quotas)

    slack = target_sample_size - allocations.sum()

    if slack > 0:
        ranked_years = remainders.sort_values(ascending=False).index
        for y in ranked_years[:slack]:
            allocations[y] += 1
    elif slack < 0:
        ranked_years = remainders.sort_values(ascending=True).index
        for y in ranked_years:
            if allocations[y] > 1 and slack < 0:
                allocations[y] -= 1
                slack += 1

    return allocations


# ==============================================================================
# 3. STRATIFIED RANDOM SAMPLING & ANNOTATION TEMPLATE CREATION
# ==============================================================================
def draw_stratified_pilot_sample(
    df: pd.DataFrame,
    allocations: pd.Series,
    id_col: str,
    title_col: str,
    text_col: str,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Draws randomized samples per stratum without replacement, randomly shuffles,
    and formats the complete 0-4 disruption criteria annotation schema.
    """
    sampled_dfs = []

    for year, n_draw in allocations.items():
        stratum = df[df["year"] == year]
        stratum_sample = stratum.sample(
            n=n_draw,
            replace=False,
            random_state=random_seed + int(year),
        )
        sampled_dfs.append(stratum_sample)

    sample = pd.concat(sampled_dfs, ignore_index=True)

    # Randomly shuffle final combined sample so annotators do not see year ordering
    sample = sample.sample(
        frac=1.0,
        random_state=random_seed,
    ).reset_index(drop=True)

    # Assign anonymized sequential validation ID (P001, P002, ...)
    sample["validation_id"] = [f"P{i+1:03d}" for i in range(len(sample))]

    # Build blank annotation columns for Human Annotators
    sample["annotator_id"] = ""                     # e.g., 'human_annotator_1', 'human_annotator_2'
    sample["relevance_label"] = ""                  # 'relevant', 'irrelevant', 'uncertain'

    # Pre-allocate the 7 criteria (0 to 4 scale)
    for dim in DISRUPTION_DIMENSIONS:
        sample[dim] = np.nan

    sample["overall_disruption_score"] = np.nan     # 0 to 4 overall severity
    sample["dominant_dimension"] = ""               # which of the 7 is primary
    sample["annotator_confidence"] = ""             # 'high', 'medium', 'low'

    # Column ordering: Context first, Scoring Rubric next, Original ID last (easy to hide/mask)
    export_columns = [
        "validation_id",
        "year",
        "parsed_date",
        title_col,
        text_col,
        "annotator_id",
        "relevance_label",
        *DISRUPTION_DIMENSIONS,
        "overall_disruption_score",
        "dominant_dimension",
        "annotator_confidence",
        id_col,  # Original ID preserved at the end for final merging
    ]

    sample_export = sample[export_columns].rename(
        columns={id_col: "original_article_id", "parsed_date": "publication_date"}
    )

    # Build stratum sampling summary table
    yearly_corpus_counts = df["year"].value_counts().sort_index()
    summary_df = pd.DataFrame({
        "year": yearly_corpus_counts.index,
        "eligible_corpus_articles": yearly_corpus_counts.values,
        "corpus_share_pct": (yearly_corpus_counts.values / len(df) * 100).round(2),
        "sampled_articles": [allocations[y] for y in yearly_corpus_counts.index],
        "sample_share_pct": ([allocations[y] for y in yearly_corpus_counts.index] / allocations.sum() * 100).round(2),
    })

    return sample_export, summary_df


# ==============================================================================
# 4. ANNOTATION RUBRIC & PROMPT GUIDE GENERATION
# ==============================================================================
def export_annotation_guide(output_path: Path) -> None:
    """Exports a clean Markdown rubric and prompt guide for annotators."""
    guide_content = """# Maritime Disruption Scoring Rubric & Annotation Guide
================================================================

This document defines the standard double-blind annotation protocol for evaluating maritime news articles on an **ordinal scale from 0 to 4**.

---

## 1. Relevance Label (`relevance_label`)
* `relevant`: The article explicitly discusses container shipping, maritime logistics, ports, vessel movements, freight rates, chokepoints, or supply chain disruptions.
* `irrelevant`: General economics, stock commentary, road trucking only, passenger cruise news, or non-supply-chain stories.
* `uncertain`: Ambiguous context where maritime relevance is tangential.

---

## 2. The Standard 0–4 Disruption Severity Scale

Every disruption dimension must be scored as an integer from **0 to 4**:

| Score | Severity Level | Operational Definition & Criteria |
| :---: | :--- | :--- |
| **0** | **No Disruption / Irrelevant** | Normal market conditions, routine business operations, new service launch, company earnings, or irrelevant text. |
| **1** | **Low / Localized Friction** | Minor delays (1-2 days), localized weather advisories, minor administrative fee changes, slight schedule adjustments with no ripple effect. |
| **2** | **Moderate Operational Disruption** | Measurable operational bottlenecks: localized berth congestion, localized labor disputes, selective port slow-downs, regional blank sailings. |
| **3** | **High Regional / Multi-Actor Disruption** | Widespread severe port congestion across major hubs, significant labor strikes, widespread container equipment shortages, sharp spot rate volatility. |
| **4** | **Severe Systemic / Critical Chokepoint Shock** | Catastrophic disruptions: Suez Canal obstruction, Red Sea missile rerouting around Cape, major multi-month COVID port shutdowns, structural fleet capacity collapse. |

---

## 3. The 7 Disruption Taxonomy Criteria (`K = 7`)

1. **`port_disruption_score`**: Berth congestion, vessel queues at terminals, longshoremen strikes, crane breakdowns, container yard dwell time spikes.
2. **`global_container_movement_disruption_score`**: Chokepoint blockages (Suez/Panama/Bab-el-Mandeb), vessel diversions around Cape, carrier blank sailings, transit delays.
3. **`freight_cost_disruption_score`**: Container spot rate surges (SCFI spikes), war-risk surcharges, emergency bunker surcharges, container lease fee spikes.
4. **`supply_disruption_score`**: Upstream component shortages, raw material delays, inventory stockouts, supply delivery bottlenecks.
5. **`production_disruption_score`**: Factory shutdowns, manufacturing halts, assembly line stoppages caused by missing shipping containers.
6. **`covid19_related_disruption_score`**: Quarantine closures, crew change restrictions, zero-COVID port halts, pandemic vessel delays.
7. **`demand_disruption_score`**: Sudden consumer demand surge/collapse, holiday front-loading, extreme purchasing volatility.

---

## 4. Overall & Confidence Columns

* **`overall_disruption_score` (0-4)**: The overall perceived disruption shock across the entire article.
* **`dominant_dimension`**: The primary dimension among the 7 that is most affected (e.g., `port_disruption_score`).
* **`annotator_confidence`**: `high`, `medium`, or `low`.
"""
    output_path.write_text(guide_content, encoding="utf-8")


# ==============================================================================
# 5. STATISTICAL & INTEGRITY AUDITING
# ==============================================================================
def verify_sample_integrity(
    sample_df: pd.DataFrame,
    expected_size: int,
    start_year: int,
    end_year: int,
) -> None:
    """Runs strict assertions on sample size, uniqueness, and temporal bounds."""
    assert len(sample_df) == expected_size, (
        f"Integrity Error: Expected {expected_size} articles, got {len(sample_df)}"
    )
    assert sample_df["validation_id"].nunique() == expected_size, (
        "Integrity Error: Duplicate validation IDs found!"
    )
    assert sample_df["original_article_id"].nunique() == expected_size, (
        "Integrity Error: An article was sampled more than once!"
    )
    min_year = sample_df["year"].min()
    max_year = sample_df["year"].max()
    assert min_year >= start_year and max_year <= end_year, (
        f"Integrity Error: Sample contains year out of bounds: [{min_year}, {max_year}]"
    )


# ==============================================================================
# 6. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Stratified Pilot Validation Sampler (N = 50)")
    parser.add_argument("--input", type=Path, default=DEFAULT_FILE_PATH, help="Path to news dataset")
    parser.add_argument("--sample-out", type=Path, default=OUTPUT_SAMPLE_PATH, help="Path to master sample CSV output")
    parser.add_argument("--ann1-out", type=Path, default=OUTPUT_ANN1_PATH, help="Path to Annotator 1 CSV")
    parser.add_argument("--ann2-out", type=Path, default=OUTPUT_ANN2_PATH, help="Path to Annotator 2 CSV")
    parser.add_argument("--summary-out", type=Path, default=OUTPUT_SUMMARY_PATH, help="Path to summary CSV output")
    parser.add_argument("--guide-out", type=Path, default=OUTPUT_GUIDE_PATH, help="Path to rubric guide MD output")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, help="Target sample size")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    print("=" * 75)
    print("  MARITIME NEWS PILOT VALIDATION SAMPLER (N = 50, 2 HUMAN ANNOTATORS)")
    print("=" * 75)

    # 1. Load & Clean
    df_eligible, n_raw, n_duplicates, n_eligible = load_and_clean_corpus(
        file_path=args.input,
        id_col=ID_COL,
        date_col=DATE_COL,
        title_col=TITLE_COL,
        text_col=TEXT_COL,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    # 2. Compute Proportional Allocation
    allocations = compute_strata_allocations(
        df=df_eligible,
        target_sample_size=args.sample_size,
    )

    # 3. Sample & Format Annotation Template
    sample_df, summary_df = draw_stratified_pilot_sample(
        df=df_eligible,
        allocations=allocations,
        id_col=ID_COL,
        title_col=TITLE_COL,
        text_col=TEXT_COL,
        random_seed=args.seed,
    )

    # 4. Integrity Validation Check
    verify_sample_integrity(
        sample_df=sample_df,
        expected_size=args.sample_size,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    # 5. Export Master CSV and Dedicated Blinded Human Templates
    for output_path in (
        args.sample_out,
        args.ann1_out,
        args.ann2_out,
        args.summary_out,
        args.guide_out,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(args.sample_out, index=False, encoding="utf-8-sig")

    # Annotator 1 Blinded Template (drop original_article_id for blinding)
    ann1_df = sample_df.drop(columns=["original_article_id"]).copy()
    ann1_df["annotator_id"] = "human_annotator_1"
    ann1_df.to_csv(args.ann1_out, index=False, encoding="utf-8-sig")

    # Annotator 2 Blinded Template
    ann2_df = sample_df.drop(columns=["original_article_id"]).copy()
    ann2_df["annotator_id"] = "human_annotator_2"
    ann2_df.to_csv(args.ann2_out, index=False, encoding="utf-8-sig")

    summary_df.to_csv(args.summary_out, index=False, encoding="utf-8-sig")
    export_annotation_guide(args.guide_out)

    # 6. Console Report
    print(f"Source file path:                  {args.input}")
    print(f"Original dataset rows:             {n_raw:,}")
    print(f"Duplicate article IDs removed:     {n_duplicates:,}")
    print(f"Eligible ({START_YEAR}-{END_YEAR}) articles:       {n_eligible:,}")
    print(f"Final pilot sample size:           {len(sample_df)} (Verified exact)")
    print("-" * 75)
    print("YEARLY STRATA ALLOCATION & CORPUS DISTRIBUTION:")
    print(summary_df.to_string(index=False))
    print("-" * 75)
    print("ASSIGNED VALIDATION COLUMNS IN TEMPLATES:")
    print("  1. validation_id (P001 - P050)")
    print("  2. year & publication_date")
    print("  3. retrieved_title & retrieved_text")
    print("  4. annotator_id ('human_annotator_1' / 'human_annotator_2')")
    print("  5. relevance_label ('relevant', 'irrelevant', 'uncertain')")
    print("  6. port_disruption_score (0-4)")
    print("  7. global_container_movement_disruption_score (0-4)")
    print("  8. freight_cost_disruption_score (0-4)")
    print("  9. supply_disruption_score (0-4)")
    print(" 10. production_disruption_score (0-4)")
    print(" 11. covid19_related_disruption_score (0-4)")
    print(" 12. demand_disruption_score (0-4)")
    print(" 13. overall_disruption_score (0-4)")
    print(" 14. dominant_dimension (name of primary dimension)")
    print(" 15. annotator_confidence ('high', 'medium', 'low')")
    print("-" * 75)
    print(f"Exported Master Sample:            {args.sample_out.resolve()}")
    print(f"Exported Annotator 1 Sheet:        {args.ann1_out.resolve()}")
    print(f"Exported Annotator 2 Sheet:        {args.ann2_out.resolve()}")
    print(f"Exported Sampling Summary:         {args.summary_out.resolve()}")
    print(f"Exported Rubric Guide:             {args.guide_out.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
