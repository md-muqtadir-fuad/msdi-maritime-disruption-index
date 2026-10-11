"""Assemble the standalone MSDI dataset package for Zenodo.

Reads saved pipeline outputs only; nothing is recomputed. Writes
release/msdi_dataset_v<VERSION>/ and a zip of it. Article bodies, SCFI,
Brent and third-party control series are excluded by construction.

    .venv\\Scripts\\python.exe misc\\zenodo_dataset\\build_dataset_package.py
"""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "release" / f"msdi_dataset_v{VERSION}"

ARTICLES_SRC = ROOT / "data/05_indices/article_level/article_consensus_scores.csv"
INDEX_SRC = {
    "msdi_monthly.csv": "data/05_indices/monthly/msdi_monthly.csv",
    "msdi_monthly_model_specific.csv": "data/05_indices/monthly/msdi_monthly_model_specific.csv",
    "msdi_monthly_model_spread.csv": "data/05_indices/uncertainty_bands/msdi_monthly_model_spread.csv",
    "msdi_weekly.csv": "data/05_indices/weekly/msdi_weekly.csv",
    "msdi_weekly_model_specific.csv": "data/05_indices/weekly/msdi_weekly_model_specific.csv",
    "msdi_weekly_model_spread.csv": "data/05_indices/uncertainty_bands/msdi_weekly_model_spread.csv",
}
VALIDATION_SRC = ROOT / "validation"

FORBIDDEN_COLUMNS = {"retrieved_text", "text_excerpt_a", "text_excerpt_b"}
FORBIDDEN_PATTERN = re.compile(r"scfi|brent", re.IGNORECASE)

# Article metadata, in release order. Raw `date`, `retrieved_publish_date` and
# `retrieved_title` are dropped: `publication_date` is the parsed date every
# index stage uses, and `retrieved_title` duplicates `title`.
ARTICLE_COLUMNS = [
    "article_id", "publication_date", "event_day", "headline", "title",
    "source_name", "source_domain", "preferred_url", "quality_score",
    "duplicate_group_id", "duplicate_group_size",
    "event_id", "event_cluster_size", "event_source_count", "event_span_days",
    "event_id_weekly", "event_cluster_size_weekly", "event_source_count_weekly", "event_span_days_weekly",
    "event_id_monthly", "event_cluster_size_monthly", "event_source_count_monthly", "event_span_days_monthly",
]
ARTICLE_RENAME = {"preferred_url": "url"}

MODELS = {
    "gemma_2_9b_it": "Gemma-2-9B-it",
    "qwen_3_8b": "Qwen3-8B",
    "qwen_2_5_7b": "Qwen2.5-7B-Instruct",
    "glm_4_9b_0414": "GLM-4-9B-0414",
}
DIMENSIONS = {
    "overall_disruption_max": "overall disruption (maximum across dimensions)",
    "port_disruption_score": "port disruption",
    "global_container_movement_disruption_score": "global container movement disruption",
    "freight_cost_disruption_score": "freight cost disruption",
    "supply_disruption_score": "supply disruption",
    "production_disruption_score": "production disruption",
    "covid19_related_disruption_score": "COVID-19-related disruption",
    "demand_disruption_score": "demand disruption",
}
STATS = {
    "ensemble_median": "median across the four models",
    "ensemble_mean": "mean across the four models",
    "ensemble_sd": "standard deviation across the four models",
    "ensemble_min": "minimum across the four models",
    "ensemble_max": "maximum across the four models",
    "ensemble_range": "max minus min across the four models",
    "models_available": "number of models that returned a score",
}
FIXED_DESCRIPTIONS = {
    "article_id": "Stable article identifier used throughout the pipeline",
    "publication_date": "Parsed publication date (YYYY-MM-DD); the date used to assign articles to weeks and months",
    "event_day": "Calendar day assigned to the article for event clustering",
    "headline": "Headline as listed by the collection source",
    "title": "Title retrieved from the article page",
    "source_name": "Publisher name",
    "source_domain": "Publisher web domain",
    "url": "Article URL; the article text can be retrieved from here",
    "quality_score": "Retrieval quality score: retrieved text length + 100 x non-empty metadata fields + 50 if canonical URL + 1 if current collection",
    "duplicate_group_id": "Identifier of the near-duplicate group (same story syndicated across outlets)",
    "duplicate_group_size": "Number of articles in the duplicate group",
    "models_successful": "Number of the four LLMs that scored the article successfully",
    "ensemble_eligible": "True if enough models succeeded for the article to enter the index",
    "peak_severity_score": "Primary article score: ensemble median of overall_disruption_max (0-4)",
    "breadth_severity_score": "Secondary article score: mean of the seven dimension ensemble medians (0-4)",
    "breadth_active_dimension_count": "Number of dimensions with ensemble median above 0",
    "non_freight_peak_severity_score": "Maximum ensemble median across the six non-freight-cost dimensions",
    "freight_price_only_flag": "1 if only the freight-cost dimension is above 0",
    "freight_price_dominant_flag": "1 if the freight-cost ensemble median is above 0 and at least 1 point above every other dimension",
    "dimension_disagreement_sd_mean": "Mean across dimensions of the between-model standard deviation",
    "source_duplicate_weight": "Down-weight for syndicated duplicates: 1/sqrt(duplicate_group_size)",
    "novelty_weight": "Final article weight: source_duplicate_weight divided by the summed source weights of the same event in the same period",
    "event_profile_used": "Event map (weekly or monthly) behind the generic weighting columns",
    "event_penalty_method": "Event down-weighting rule",
}


def describe(column: str) -> str:
    if column in FIXED_DESCRIPTIONS:
        return FIXED_DESCRIPTIONS[column]
    for suffix, label in (("_weekly", "weekly"), ("_monthly", "monthly")):
        base = column[: -len(suffix)] if column.endswith(suffix) else None
        if base and base in {"event_id", "event_cluster_size", "event_source_count", "event_span_days",
                             "event_period_article_count", "event_period_source_weight_sum",
                             "novelty_weight", "peak_weighted_contribution",
                             "breadth_weighted_contribution"}:
            return f"{describe(base)} ({label} event map)"
    generic = {
        "event_id": "Semantic event cluster identifier (cosine threshold 0.82, human-validated)",
        "event_cluster_size": "Number of articles in the event cluster",
        "event_source_count": "Number of distinct sources in the event cluster",
        "event_span_days": "Days between first and last article in the event cluster",
        "event_period_article_count": "Articles from the same event in the same period",
        "event_period_source_weight_sum": "Sum of source weights for the same event in the same period",
        "peak_weighted_contribution": "Article contribution to the peak index: peak_severity_score x novelty_weight",
        "breadth_weighted_contribution": "Article contribution to the breadth index: breadth_severity_score x novelty_weight",
        "non_freight_peak_weighted_contribution": "Article contribution to the non-freight peak index",
    }
    if column in generic:
        return generic[column]
    if "__" in column:
        dim, rest = column.split("__", 1)
        dim_label = DIMENSIONS.get(dim, dim)
        if rest in MODELS:
            return f"{MODELS[rest]} score for {dim_label} (0-4)"
        if rest in STATS:
            return f"{dim_label}: {STATS[rest]}"
        if rest.startswith("diff__"):
            a, b = rest[len("diff__"):].split("_minus_")
            return f"{dim_label}: {MODELS.get(a, a)} minus {MODELS.get(b, b)}"
    return ""


def describe_index(column: str, monthly: bool) -> str:
    period = "month" if monthly else "week"
    fixed = {
        "month_end": "Last day of the calendar month",
        "week_ending": "Friday that ends the week (weeks run Saturday to Friday)",
        "model_alias": "LLM the row's series is built from",
        "article_count": f"Index-eligible articles published in the {period}",
        "event_count": f"Distinct events in the {period}",
        "source_count": f"Distinct publisher domains in the {period}",
        "effective_article_count": f"Sum of novelty weights in the {period}",
        "peak_active_article_count": "Articles with peak_severity_score above 0",
        "breadth_active_article_count": "Articles with breadth_severity_score above 0",
        "peak_score_unweighted_mean": "Unweighted mean of peak_severity_score",
        "breadth_score_unweighted_mean": "Unweighted mean of breadth_severity_score",
        "peak_model_disagreement_mean": "Mean between-model SD of overall_disruption_max",
        "dimension_disagreement_mean": "Mean of dimension_disagreement_sd_mean",
        "freight_price_only_share": "Share of articles with freight_price_only_flag = 1",
        "freight_price_dominant_share": "Share of articles with freight_price_dominant_flag = 1",
        "peak_msdi_z_full_sample_descriptive": "Peak MSDI standardized with full-sample mean and SD (descriptive only; uses future data)",
        "breadth_msdi_z_full_sample_descriptive": "Breadth MSDI standardized with full-sample mean and SD (descriptive only; uses future data)",
        "peak_msdi_z_expanding_realtime": "Peak MSDI standardized with the mean and SD of prior periods only (no look-ahead)",
        "breadth_msdi_z_expanding_realtime": "Breadth MSDI standardized with the mean and SD of prior periods only (no look-ahead)",
    }
    if column in fixed:
        return fixed[column]
    stems = {"peak_score": "peak", "breadth_score": "breadth", "non_freight_peak_score": "non-freight peak"}
    stems.update({d.removesuffix("_score"): label for d, label in DIMENSIONS.items() if d != "overall_disruption_max"})
    for stem, label in stems.items():
        if column == f"{stem}_sum":
            return f"Sum of novelty-weighted article {label} scores"
        if column == f"{stem}_sqrt_effective_volume":
            return f"{label}: weighted sum / sqrt(effective_article_count)"
        if column == f"{stem}_sqrt_effective_volume_ema3":
            return f"{label}: causal 3-month EMA of the sqrt-volume series, restarting after months with no articles"
        if column == f"{stem}_media_corrected":
            if monthly:
                return f"Monthly {label} MSDI (equals the _sqrt_effective_volume_ema3 column)"
            return f"Weekly {label} MSDI: weighted sum / log(1 + article_count)"
    if "__" in column:
        path, rest = column.split("__", 1)
        if rest in MODELS:
            return f"{path} MSDI built from {MODELS[rest]} alone"
        if rest.startswith("diff__"):
            m1, m2 = rest[len("diff__"):].split("_minus_")
            return f"{path} MSDI: {MODELS[m1]} series minus {MODELS[m2]} series"
        if rest.startswith("model_"):
            stat = {"median": "median", "mean": "mean", "sd": "SD", "min": "minimum",
                    "max": "maximum", "range": "max minus min"}[rest[len("model_"):]]
            return f"{path} MSDI: {stat} across the four model-specific series"
    return ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_columns(path: Path, columns) -> None:
    bad = [c for c in columns if c in FORBIDDEN_COLUMNS or FORBIDDEN_PATTERN.search(c)]
    if bad:
        raise SystemExit(f"{path}: excluded columns present: {bad}")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "articles").mkdir(parents=True)
    (OUT / "index").mkdir()

    # Articles: metadata and scores split into two files joined on article_id.
    art = pd.read_csv(ARTICLES_SRC, low_memory=False)
    assert art["article_id"].is_unique
    art["publication_date"] = pd.to_datetime(art["publication_date"], format="mixed").dt.strftime("%Y-%m-%d")
    meta = art[ARTICLE_COLUMNS].rename(columns=ARTICLE_RENAME)
    drop = set(ARTICLE_COLUMNS) | {"date", "retrieved_publish_date", "retrieved_title", "publication_date"}
    scores = art[["article_id"] + [c for c in art.columns if c not in drop and c != "article_id"]]
    for frame, name in ((meta, "articles.csv"), (scores, "article_scores.csv")):
        check_columns(Path(name), frame.columns)
        frame.to_csv(OUT / "articles" / name, index=False, encoding="utf-8")

    for name, src in INDEX_SRC.items():
        cols = pd.read_csv(ROOT / src, nrows=0).columns
        check_columns(Path(src), cols)
        shutil.copy2(ROOT / src, OUT / "index" / name)

    # Validation: the git-tracked tree, already stripped of article bodies.
    shutil.copytree(VALIDATION_SRC, OUT / "validation")
    csv.field_size_limit(10**9)
    for path in (OUT / "validation").rglob("*.csv"):
        with path.open(encoding="utf-8-sig", newline="") as fh:
            check_columns(path, next(csv.reader(fh)))

    # Data dictionary for articles/ and index/.
    rows = []
    for path in sorted((OUT / "articles").glob("*.csv")) + sorted((OUT / "index").glob("*.csv")):
        df = pd.read_csv(path, low_memory=False)
        for c in df.columns:
            rows.append({
                "file": path.relative_to(OUT).as_posix(),
                "column": c,
                "dtype": str(df[c].dtype),
                "non_missing": int(df[c].notna().sum()),
                "description": describe_index(c, "monthly" in path.name)
                if path.parent.name == "index" else describe(c),
            })
    pd.DataFrame(rows).to_csv(OUT / "data_dictionary.csv", index=False, encoding="utf-8")

    shutil.copy2(HERE / "DATASET_README.md", OUT / "README.md")
    shutil.copy2(HERE / "LICENSE-CC-BY-4.0.txt", OUT / "LICENSE.txt")

    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    with (OUT / "SHA256SUMS.txt").open("w", encoding="utf-8", newline="\n") as fh:
        for p in files:
            fh.write(f"{sha256(p)}  {p.relative_to(OUT).as_posix()}\n")

    zip_path = OUT.parent / f"{OUT.name}.zip"
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                zf.write(p, Path(OUT.name) / p.relative_to(OUT))

    print(f"articles: {len(meta)} rows; scores: {scores.shape[1] - 1} columns")
    print(f"{len(files) + 1} files -> {OUT}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
