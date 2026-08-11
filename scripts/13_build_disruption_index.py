"""Build article-level consensus scores and monthly/weekly MSDI variants.

Two parallel, pre-specified measurement paths are retained:

* ``peak`` (primary): median ``overall_disruption_max`` across models.
* ``breadth`` (secondary): equal-weight mean of the seven dimension medians.

The script never uses SCFI or another outcome while constructing either index.
It also produces exhaustive inter-model difference diagnostics and a separate
TOPSIS operational ranking. TOPSIS does *not* weight the MSDI ensemble.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = PROJECT_ROOT / "data/03_processed/model_scores/llm_scores_merged.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/05_indices"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs/model_registry.yaml"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs/logs"

SCORE_MIN = 0.0
SCORE_MAX = 4.0
DIMENSION_COLUMNS = [
    "port_disruption_score",
    "global_container_movement_disruption_score",
    "freight_cost_disruption_score",
    "supply_disruption_score",
    "production_disruption_score",
    "covid19_related_disruption_score",
    "demand_disruption_score",
]
PRIMARY_COLUMN = "overall_disruption_max"
ALL_SCORE_COLUMNS = [PRIMARY_COLUMN, *DIMENSION_COLUMNS]
OPERATIONAL_COLUMNS = [
    "model_elapsed_seconds",
    "total_input_tokens",
    "total_generated_tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build primary peak-severity and secondary breadth MSDI series."
    )
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--version", default="msdi_v001")
    parser.add_argument("--frequency", default="W-FRI")
    parser.add_argument(
        "--monthly-frequency",
        default="ME",
        help="Calendar month-end frequency for the publication-primary index.",
    )
    parser.add_argument(
        "--event-profile",
        choices=("auto", "weekly", "monthly"),
        default="auto",
        help="Event map used for weighting; auto selects from --frequency.",
    )
    parser.add_argument("--minimum-models", type=int, default=2)
    parser.add_argument(
        "--duplicate-penalty",
        choices=("sqrt", "inverse", "none"),
        default="sqrt",
        help="Novelty weight applied to the source duplicate-group size.",
    )
    parser.add_argument(
        "--event-penalty",
        choices=("fixed", "sqrt", "none"),
        default="fixed",
        help=(
            "Within-period semantic-event coverage correction. 'fixed' gives every "
            "event one total unit of weight and is the publication-primary setting."
        ),
    )
    parser.add_argument(
        "--media-correction",
        choices=("log", "mean", "none"),
        default="log",
        help="Correction applied to weekly weighted score sums.",
    )
    parser.add_argument("--expanding-min-periods", type=int, default=26)
    parser.add_argument("--monthly-expanding-min-periods", type=int, default=24)
    return parser.parse_args()


def configure_logging(log_dir: Path, version: str) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"05_build_disruption_index_{version}.log"
    logger = logging.getLogger("build_disruption_index")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, path


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value


def load_model_config(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Model registry not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = config.get("active_scoring_models", [])
    criteria = config.get("topsis_operational_ranking", {}).get("criteria", {})
    if len(models) != 4:
        raise ValueError("model_registry.yaml must define exactly four active_scoring_models")
    if not criteria:
        raise ValueError("model_registry.yaml lacks TOPSIS criteria")
    weights = [float(spec["weight"]) for spec in criteria.values()]
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError(f"TOPSIS criterion weights must sum to 1; found {sum(weights)}")
    return models, criteria


def assign_model_alias(model_name: str, registry: list[dict[str, Any]]) -> str:
    matches = [
        str(item["alias"])
        for item in registry
        if str(item["name_contains"]).lower() in str(model_name).lower()
    ]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely map model name to registry alias: {model_name!r}")
    return matches[0]


def resolve_event_profile(frequency: str, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    normalized = frequency.upper()
    if normalized.startswith("W"):
        return "weekly"
    if normalized.startswith("M"):
        return "monthly"
    raise ValueError(
        f"Cannot infer event profile from frequency={frequency!r}; pass --event-profile"
    )


def read_scores(
    path: Path,
    registry: list[dict[str, Any]],
    logger: logging.Logger,
    event_profile: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Merged score file not found: {path}")
    event_columns = [
        f"{column}_{profile}"
        for profile in ("weekly", "monthly")
        for column in (
            "event_id",
            "event_cluster_size",
            "event_source_count",
            "event_span_days",
        )
    ]
    required = [
        "article_id",
        "model_name",
        "scoring_status",
        "date",
        "retrieved_publish_date",
        "headline",
        "retrieved_title",
        "source_name",
        "source_domain",
        "preferred_url",
        "duplicate_group_id",
        "duplicate_group_size",
        "event_day",
        *event_columns,
        "quality_score",
        *ALL_SCORE_COLUMNS,
        *OPERATIONAL_COLUMNS,
    ]
    scores = pd.read_csv(path, usecols=required, low_memory=False)
    for column in (
        "event_id",
        "event_cluster_size",
        "event_source_count",
        "event_span_days",
    ):
        scores[column] = scores[f"{column}_{event_profile}"]
    if scores.empty:
        raise ValueError("Merged score file is empty")
    if scores.duplicated(["model_name", "article_id"]).any():
        raise ValueError("Merged scores contain duplicate (model_name, article_id) keys")
    scores["model_alias"] = scores["model_name"].map(lambda value: assign_model_alias(value, registry))
    if scores["model_alias"].nunique() != 4:
        raise ValueError("Expected exactly four mapped scoring models")

    for column in [
        *ALL_SCORE_COLUMNS,
        *OPERATIONAL_COLUMNS,
        "duplicate_group_size",
        "event_cluster_size",
        "event_source_count",
        "event_span_days",
        "event_cluster_size_weekly",
        "event_source_count_weekly",
        "event_span_days_weekly",
        "event_cluster_size_monthly",
        "event_source_count_monthly",
        "event_span_days_monthly",
        "quality_score",
    ]:
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    successful = scores["scoring_status"].eq("success")
    for column in ALL_SCORE_COLUMNS:
        invalid = successful & scores[column].notna() & ~scores[column].between(SCORE_MIN, SCORE_MAX)
        if invalid.any():
            examples = scores.loc[invalid, ["article_id", "model_alias", column]].head(10)
            raise ValueError(
                f"{column} has {int(invalid.sum())} successful values outside 0-4:\n"
                f"{examples.to_string(index=False)}"
            )
        scores.loc[~successful, column] = np.nan

    logger.info("Loaded %s rows and %s unique articles", f"{len(scores):,}", f"{scores['article_id'].nunique():,}")
    for alias, group in scores.groupby("model_alias", sort=True):
        logger.info(
            "Model %s: rows=%s success=%s failed=%s",
            alias,
            f"{len(group):,}",
            f"{group['scoring_status'].eq('success').sum():,}",
            f"{(~group['scoring_status'].eq('success')).sum():,}",
        )
    return scores


def parse_publication_dates(metadata: pd.DataFrame) -> tuple[pd.Series, dict[str, int]]:
    primary = pd.to_datetime(metadata["date"], format="mixed", errors="coerce", utc=True)
    fallback = pd.to_datetime(
        metadata["retrieved_publish_date"], format="mixed", errors="coerce", utc=True
    )
    combined = primary.fillna(fallback).dt.tz_localize(None)
    report = {
        "parsed_from_dataset_date": int(primary.notna().sum()),
        "parsed_from_retrieved_fallback": int((primary.isna() & fallback.notna()).sum()),
        "unparsed": int(combined.isna().sum()),
    }
    return combined, report


def check_metadata_consistency(scores: pd.DataFrame) -> dict[str, int]:
    report: dict[str, int] = {}
    for column in [
        "date",
        "duplicate_group_id",
        "duplicate_group_size",
        "event_id",
        "event_cluster_size",
        "source_domain",
    ]:
        counts = scores.groupby("article_id")[column].nunique(dropna=True)
        report[f"articles_with_conflicting_{column}"] = int(counts.gt(1).sum())
    return report


def wide_metric(scores: pd.DataFrame, metric: str, aliases: list[str]) -> pd.DataFrame:
    pivot = scores.pivot(index="article_id", columns="model_alias", values=metric).reindex(columns=aliases)
    pivot.columns = [f"{metric}__{alias}" for alias in aliases]
    return pivot


def add_ensemble_diagnostics(
    article: pd.DataFrame, metric: str, aliases: list[str]
) -> list[str]:
    model_columns = [f"{metric}__{alias}" for alias in aliases]
    values = article[model_columns]
    prefix = f"{metric}__"
    article[f"{prefix}ensemble_median"] = values.median(axis=1, skipna=True)
    article[f"{prefix}ensemble_mean"] = values.mean(axis=1, skipna=True)
    article[f"{prefix}ensemble_sd"] = values.std(axis=1, ddof=0, skipna=True)
    article[f"{prefix}ensemble_min"] = values.min(axis=1, skipna=True)
    article[f"{prefix}ensemble_max"] = values.max(axis=1, skipna=True)
    article[f"{prefix}ensemble_range"] = (
        article[f"{prefix}ensemble_max"] - article[f"{prefix}ensemble_min"]
    )
    article[f"{prefix}models_available"] = values.notna().sum(axis=1)
    difference_columns: list[str] = []
    for left, right in combinations(aliases, 2):
        column = f"{metric}__diff__{left}_minus_{right}"
        article[column] = article[f"{metric}__{left}"] - article[f"{metric}__{right}"]
        difference_columns.append(column)
    return difference_columns


def novelty_weight(group_size: pd.Series, method: str) -> pd.Series:
    size = pd.to_numeric(group_size, errors="coerce").fillna(1).clip(lower=1)
    if method == "sqrt":
        return 1.0 / np.sqrt(size)
    if method == "inverse":
        return 1.0 / size
    return pd.Series(1.0, index=size.index)


def select_event_profile(article: pd.DataFrame, profile: str) -> pd.DataFrame:
    """Expose one pre-LLM event map through the generic weighting columns."""
    if profile not in {"weekly", "monthly"}:
        raise ValueError(f"Unsupported event profile: {profile}")
    output = article.copy()
    for column in (
        "event_id",
        "event_cluster_size",
        "event_source_count",
        "event_span_days",
    ):
        source = f"{column}_{profile}"
        if source not in output.columns:
            raise ValueError(f"Article table lacks {source}")
        output[column] = output[source]
    output["event_profile_used"] = profile
    return output


def apply_period_event_weights(
    article: pd.DataFrame,
    frequency: str,
    event_penalty: str,
) -> pd.DataFrame:
    """Create availability-safe event weights within each aggregation period.

    Global event-cluster sizes are descriptive and can include later reports.
    Forecasting features must therefore calculate coverage weights only from
    articles available inside the current week/month.  Existing exact-duplicate
    weights are normalized within an event under the fixed/sqrt specifications.
    """
    if event_penalty not in {"fixed", "sqrt", "none"}:
        raise ValueError(f"Unsupported event penalty: {event_penalty}")
    output = article.copy()
    if "source_duplicate_weight" not in output.columns:
        output["source_duplicate_weight"] = output.get("novelty_weight", 1.0)
    event_id = output["event_id"].astype("string").str.strip()
    missing_event = event_id.isna() | event_id.eq("")
    event_id = event_id.mask(missing_event, "ARTICLE_" + output["article_id"].astype(str))
    output["event_id"] = event_id

    valid_date = output["publication_date"].notna()
    eligible = valid_date & output.get(
        "ensemble_eligible", pd.Series(True, index=output.index)
    ).fillna(False)
    output["event_period_article_count"] = 0
    output["event_period_source_weight_sum"] = 0.0
    if eligible.any():
        period_frequency = "M" if frequency.upper().startswith("M") else frequency
        period = output.loc[eligible, "publication_date"].dt.to_period(period_frequency).astype(str)
        keys = pd.DataFrame(
            {
                "_period": period,
                "_event": output.loc[valid_date, "event_id"].astype(str),
            },
            index=output.index[eligible],
        )
        grouped = output.loc[eligible].assign(
            _period=keys["_period"], _event=keys["_event"]
        ).groupby(["_period", "_event"], sort=False)
        output.loc[eligible, "event_period_article_count"] = grouped[
            "article_id"
        ].transform("nunique").astype(int)
        output.loc[eligible, "event_period_source_weight_sum"] = grouped[
            "source_duplicate_weight"
        ].transform("sum")

    base = pd.to_numeric(output["source_duplicate_weight"], errors="coerce").fillna(1.0)
    denominator = pd.to_numeric(
        output["event_period_source_weight_sum"], errors="coerce"
    ).replace(0, np.nan)
    size = pd.to_numeric(output["event_period_article_count"], errors="coerce").fillna(1).clip(lower=1)
    if event_penalty == "fixed":
        output["novelty_weight"] = base.div(denominator)
    elif event_penalty == "sqrt":
        output["novelty_weight"] = base.div(denominator).mul(np.sqrt(size))
    else:
        output["novelty_weight"] = base.where(eligible)
    output.loc[~eligible, "novelty_weight"] = 0.0
    output["event_penalty_method"] = event_penalty
    output["peak_weighted_contribution"] = (
        output["peak_severity_score"] * output["novelty_weight"]
    )
    output["breadth_weighted_contribution"] = (
        output["breadth_severity_score"] * output["novelty_weight"]
    )
    return output


def build_article_scores(
    scores: pd.DataFrame,
    aliases: list[str],
    minimum_models: int,
    duplicate_penalty: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scores = scores.copy()
    # Backward-compatible for bounded unit fixtures; production ingestion always
    # supplies two independently generated profile maps.
    for profile in ("weekly", "monthly"):
        for column in (
            "event_id",
            "event_cluster_size",
            "event_source_count",
            "event_span_days",
        ):
            profile_column = f"{column}_{profile}"
            if profile_column not in scores.columns and column in scores.columns:
                scores[profile_column] = scores[column]
    metadata_columns = [
        "article_id",
        "date",
        "retrieved_publish_date",
        "headline",
        "retrieved_title",
        "source_name",
        "source_domain",
        "preferred_url",
        "duplicate_group_id",
        "duplicate_group_size",
        "event_day",
        "event_id",
        "event_cluster_size",
        "event_source_count",
        "event_span_days",
        "event_id_weekly",
        "event_cluster_size_weekly",
        "event_source_count_weekly",
        "event_span_days_weekly",
        "event_id_monthly",
        "event_cluster_size_monthly",
        "event_source_count_monthly",
        "event_span_days_monthly",
        "quality_score",
    ]
    metadata = (
        scores.sort_values(["article_id", "model_alias"])
        .groupby("article_id", as_index=False)
        .first()[metadata_columns]
        .set_index("article_id")
    )
    metadata["publication_date"], date_report = parse_publication_dates(metadata)
    metadata["title"] = metadata["retrieved_title"].fillna(metadata["headline"])
    article = metadata.copy()

    all_difference_columns: list[str] = []
    for metric in ALL_SCORE_COLUMNS:
        article = article.join(wide_metric(scores, metric, aliases), how="left")
        all_difference_columns.extend(add_ensemble_diagnostics(article, metric, aliases))

    status = scores.assign(success=scores["scoring_status"].eq("success")).pivot(
        index="article_id", columns="model_alias", values="success"
    ).reindex(columns=aliases)
    article["models_successful"] = status.fillna(False).sum(axis=1).astype(int)
    article["ensemble_eligible"] = article["models_successful"].ge(minimum_models)

    article["peak_severity_score"] = article[f"{PRIMARY_COLUMN}__ensemble_median"]
    dimension_medians = [f"{column}__ensemble_median" for column in DIMENSION_COLUMNS]
    article["breadth_severity_score"] = article[dimension_medians].mean(axis=1, skipna=False)
    article["breadth_active_dimension_count"] = article[dimension_medians].gt(0).sum(axis=1)
    article["dimension_disagreement_sd_mean"] = article[
        [f"{column}__ensemble_sd" for column in DIMENSION_COLUMNS]
    ].mean(axis=1)
    article["source_duplicate_weight"] = novelty_weight(
        article["duplicate_group_size"], duplicate_penalty
    )
    # This is replaced with a period-local semantic-event weight in main().
    article["novelty_weight"] = article["source_duplicate_weight"]
    article["peak_weighted_contribution"] = article["peak_severity_score"] * article["novelty_weight"]
    article["breadth_weighted_contribution"] = (
        article["breadth_severity_score"] * article["novelty_weight"]
    )

    # Consolidate the many diagnostic columns before adding model-specific paths.
    article = article.copy()
    for alias in aliases:
        model_dimensions = [f"{column}__{alias}" for column in DIMENSION_COLUMNS]
        article[f"peak_severity_score__{alias}"] = article[f"{PRIMARY_COLUMN}__{alias}"]
        article[f"breadth_severity_score__{alias}"] = article[model_dimensions].mean(
            axis=1, skipna=False
        )

    eligible = article["ensemble_eligible"] & article["publication_date"].notna()
    diagnostics = {
        "articles_total": len(article),
        "articles_ensemble_eligible": int(article["ensemble_eligible"].sum()),
        "articles_excluded_insufficient_models": int((~article["ensemble_eligible"]).sum()),
        "articles_excluded_unparsed_date": int(
            (article["ensemble_eligible"] & article["publication_date"].isna()).sum()
        ),
        "articles_used_in_weekly_index": int(eligible.sum()),
        "date_parsing": date_report,
        "metadata_consistency": check_metadata_consistency(scores),
        "difference_columns_logged": all_difference_columns,
        "peak_score_summary": article.loc[eligible, "peak_severity_score"].describe().to_dict(),
        "breadth_score_summary": article.loc[eligible, "breadth_severity_score"].describe().to_dict(),
        "peak_model_disagreement_summary": article.loc[
            eligible, f"{PRIMARY_COLUMN}__ensemble_sd"
        ].describe().to_dict(),
    }
    logger.info(
        "Article consensus: eligible=%s excluded_model_count=%s unparsed_dates=%s",
        f"{diagnostics['articles_ensemble_eligible']:,}",
        f"{diagnostics['articles_excluded_insufficient_models']:,}",
        f"{diagnostics['articles_excluded_unparsed_date']:,}",
    )
    return article.copy().reset_index(), diagnostics


def media_correct(raw_sum: pd.Series, article_count: pd.Series, method: str) -> pd.Series:
    count = article_count.astype(float)
    if method == "log":
        denominator = np.log1p(count)
        return raw_sum.div(denominator.where(denominator.gt(0))).fillna(0.0)
    if method == "mean":
        return raw_sum.div(count.where(count.gt(0))).fillna(0.0)
    return raw_sum.astype(float)


def full_sample_zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / sd


def expanding_realtime_zscore(series: pd.Series, minimum_periods: int) -> pd.Series:
    prior_mean = series.expanding(min_periods=minimum_periods).mean().shift(1)
    prior_sd = series.expanding(min_periods=minimum_periods).std(ddof=0).shift(1)
    return (series - prior_mean) / prior_sd.replace(0, np.nan)


def causal_ema_restart_gaps(series: pd.Series, span: int = 3) -> pd.Series:
    """Causal EMA that does not carry state through periods with no articles."""
    if span < 1:
        raise ValueError("EMA span must be positive")
    output = pd.Series(np.nan, index=series.index, dtype=float)
    observed = series.notna()
    block = observed.ne(observed.shift(fill_value=False)).cumsum()
    for _, positions in series[observed].groupby(block[observed]).groups.items():
        output.loc[positions] = series.loc[positions].ewm(span=span, adjust=False).mean()
    return output


def complete_week_index(grouped: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if grouped.empty:
        return grouped
    dates = pd.date_range(grouped.index.min(), grouped.index.max(), freq=frequency)
    result = grouped.reindex(dates)
    result.index.name = "week_ending"
    count_columns = [column for column in result.columns if column.endswith("_count")]
    sum_columns = [column for column in result.columns if column.endswith("_sum")]
    fill_columns = list(dict.fromkeys([*count_columns, *sum_columns]))
    result[fill_columns] = result[fill_columns].fillna(0)
    return result


def aggregate_weekly(
    article: pd.DataFrame,
    frequency: str,
    media_method: str,
    expanding_min_periods: int,
) -> pd.DataFrame:
    work = article[article["ensemble_eligible"] & article["publication_date"].notna()].copy()
    work = work.set_index("publication_date")
    grouped = work.resample(frequency, label="right", closed="right").agg(
        article_count=("article_id", "nunique"),
        event_count=("event_id", "nunique"),
        source_count=("source_domain", "nunique"),
        effective_article_count=("novelty_weight", "sum"),
        peak_active_article_count=("peak_severity_score", lambda value: int(value.gt(0).sum())),
        breadth_active_article_count=("breadth_severity_score", lambda value: int(value.gt(0).sum())),
        peak_score_sum=("peak_weighted_contribution", "sum"),
        breadth_score_sum=("breadth_weighted_contribution", "sum"),
        peak_score_unweighted_mean=("peak_severity_score", "mean"),
        breadth_score_unweighted_mean=("breadth_severity_score", "mean"),
        peak_model_disagreement_mean=(f"{PRIMARY_COLUMN}__ensemble_sd", "mean"),
        dimension_disagreement_mean=("dimension_disagreement_sd_mean", "mean"),
    )

    dimension_sum_columns: list[str] = []
    for dimension in DIMENSION_COLUMNS:
        consensus = f"{dimension}__ensemble_median"
        contribution = f"_weighted_{dimension}"
        work[contribution] = work[consensus] * work["novelty_weight"]
        output = f"{dimension.removesuffix('_score')}_sum"
        grouped[output] = work[contribution].resample(
            frequency, label="right", closed="right"
        ).sum()
        dimension_sum_columns.append(output)

    grouped = complete_week_index(grouped, frequency)
    grouped["peak_score_media_corrected"] = media_correct(
        grouped["peak_score_sum"], grouped["article_count"], media_method
    )
    grouped["breadth_score_media_corrected"] = media_correct(
        grouped["breadth_score_sum"], grouped["article_count"], media_method
    )
    grouped["peak_msdi_z_full_sample_descriptive"] = full_sample_zscore(
        grouped["peak_score_media_corrected"]
    )
    grouped["breadth_msdi_z_full_sample_descriptive"] = full_sample_zscore(
        grouped["breadth_score_media_corrected"]
    )
    grouped["peak_msdi_z_expanding_realtime"] = expanding_realtime_zscore(
        grouped["peak_score_media_corrected"], expanding_min_periods
    )
    grouped["breadth_msdi_z_expanding_realtime"] = expanding_realtime_zscore(
        grouped["breadth_score_media_corrected"], expanding_min_periods
    )
    for column in dimension_sum_columns:
        stem = column.removesuffix("_sum")
        grouped[f"{stem}_media_corrected"] = media_correct(
            grouped[column], grouped["article_count"], media_method
        )
    return grouped.reset_index()


def aggregate_monthly_direct(
    article: pd.DataFrame,
    frequency: str,
    expanding_min_periods: int,
    ema_span: int = 3,
) -> pd.DataFrame:
    """Construct the monthly primary index directly from eligible article rows.

    For each score path, the contemporaneous event-weighted numerator is divided
    by the square root of the period-local effective article count, followed by
    a causal EMA(3). No weekly index value is used in this construction.
    """
    work = article[article["ensemble_eligible"] & article["publication_date"].notna()].copy()
    work = work.set_index("publication_date")
    grouped = work.resample(frequency, label="right", closed="right").agg(
        article_count=("article_id", "nunique"),
        event_count=("event_id", "nunique"),
        source_count=("source_domain", "nunique"),
        effective_article_count=("novelty_weight", "sum"),
        peak_active_article_count=("peak_severity_score", lambda value: int(value.gt(0).sum())),
        breadth_active_article_count=("breadth_severity_score", lambda value: int(value.gt(0).sum())),
        peak_score_sum=("peak_weighted_contribution", "sum"),
        breadth_score_sum=("breadth_weighted_contribution", "sum"),
        peak_score_unweighted_mean=("peak_severity_score", "mean"),
        breadth_score_unweighted_mean=("breadth_severity_score", "mean"),
        peak_model_disagreement_mean=(f"{PRIMARY_COLUMN}__ensemble_sd", "mean"),
        dimension_disagreement_mean=("dimension_disagreement_sd_mean", "mean"),
    )
    grouped.index.name = "month_end"
    denominator = np.sqrt(grouped["effective_article_count"].where(lambda value: value.gt(0)))
    for path in ("peak", "breadth"):
        base_column = f"{path}_score_sqrt_effective_volume"
        ema_column = f"{base_column}_ema{ema_span}"
        grouped[base_column] = grouped[f"{path}_score_sum"].div(denominator)
        grouped[ema_column] = causal_ema_restart_gaps(grouped[base_column], span=ema_span)
        # Generic contract used by panel builders; formula is recorded explicitly.
        grouped[f"{path}_score_media_corrected"] = grouped[ema_column]
        grouped[f"{path}_msdi_z_full_sample_descriptive"] = full_sample_zscore(
            grouped[ema_column]
        )
        grouped[f"{path}_msdi_z_expanding_realtime"] = expanding_realtime_zscore(
            grouped[ema_column], expanding_min_periods
        )

    for dimension in DIMENSION_COLUMNS:
        consensus = f"{dimension}__ensemble_median"
        contribution = work[consensus] * work["novelty_weight"]
        stem = dimension.removesuffix("_score")
        sum_column = f"{stem}_sum"
        base_column = f"{stem}_sqrt_effective_volume"
        ema_column = f"{base_column}_ema{ema_span}"
        grouped[sum_column] = contribution.resample(
            frequency, label="right", closed="right"
        ).sum()
        grouped[base_column] = grouped[sum_column].div(denominator)
        grouped[ema_column] = causal_ema_restart_gaps(grouped[base_column], span=ema_span)
        grouped[f"{stem}_media_corrected"] = grouped[ema_column]
    return grouped.reset_index()


def aggregate_model_monthly_direct(
    article: pd.DataFrame,
    aliases: list[str],
    frequency: str,
    ema_span: int = 3,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for alias in aliases:
        peak = f"peak_severity_score__{alias}"
        breadth = f"breadth_severity_score__{alias}"
        work = article[
            article["ensemble_eligible"]
            & article["publication_date"].notna()
            & article[peak].notna()
        ].copy()
        work["peak_contribution"] = work[peak] * work["novelty_weight"]
        work["breadth_contribution"] = work[breadth] * work["novelty_weight"]
        grouped = work.set_index("publication_date").resample(
            frequency, label="right", closed="right"
        ).agg(
            article_count=("article_id", "nunique"),
            effective_article_count=("novelty_weight", "sum"),
            peak_score_sum=("peak_contribution", "sum"),
            breadth_score_sum=("breadth_contribution", "sum"),
        )
        denominator = np.sqrt(
            grouped["effective_article_count"].where(lambda value: value.gt(0))
        )
        for path in ("peak", "breadth"):
            base = grouped[f"{path}_score_sum"].div(denominator)
            grouped[f"{path}_score_sqrt_effective_volume"] = base
            grouped[f"{path}_score_media_corrected"] = causal_ema_restart_gaps(
                base, span=ema_span
            )
        grouped["model_alias"] = alias
        grouped.index.name = "month_end"
        rows.append(grouped.reset_index())
    return pd.concat(rows, ignore_index=True)


def aggregate_model_weekly(
    article: pd.DataFrame,
    aliases: list[str],
    frequency: str,
    media_method: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for alias in aliases:
        peak = f"peak_severity_score__{alias}"
        breadth = f"breadth_severity_score__{alias}"
        work = article[
            article["ensemble_eligible"]
            & article["publication_date"].notna()
            & article[peak].notna()
        ].copy()
        work["peak_contribution"] = work[peak] * work["novelty_weight"]
        work["breadth_contribution"] = work[breadth] * work["novelty_weight"]
        grouped = work.set_index("publication_date").resample(
            frequency, label="right", closed="right"
        ).agg(
            article_count=("article_id", "nunique"),
            peak_score_sum=("peak_contribution", "sum"),
            breadth_score_sum=("breadth_contribution", "sum"),
        )
        grouped = complete_week_index(grouped, frequency)
        grouped["peak_score_media_corrected"] = media_correct(
            grouped["peak_score_sum"], grouped["article_count"], media_method
        )
        grouped["breadth_score_media_corrected"] = media_correct(
            grouped["breadth_score_sum"], grouped["article_count"], media_method
        )
        grouped["model_alias"] = alias
        rows.append(grouped.reset_index())
    return pd.concat(rows, ignore_index=True)


def build_model_spread(
    model_period: pd.DataFrame, aliases: list[str], date_column: str = "week_ending"
) -> pd.DataFrame:
    base = model_period[[date_column]].drop_duplicates().sort_values(date_column)
    output = base.set_index(date_column)
    for path in ("peak", "breadth"):
        value = f"{path}_score_media_corrected"
        pivot = model_period.pivot(index=date_column, columns="model_alias", values=value).reindex(
            columns=aliases
        )
        for alias in aliases:
            output[f"{path}__{alias}"] = pivot[alias]
        output[f"{path}__model_median"] = pivot.median(axis=1)
        output[f"{path}__model_mean"] = pivot.mean(axis=1)
        output[f"{path}__model_sd"] = pivot.std(axis=1, ddof=0)
        output[f"{path}__model_min"] = pivot.min(axis=1)
        output[f"{path}__model_max"] = pivot.max(axis=1)
        output[f"{path}__model_range"] = pivot.max(axis=1) - pivot.min(axis=1)
        for left, right in combinations(aliases, 2):
            output[f"{path}__diff__{left}_minus_{right}"] = pivot[left] - pivot[right]
    return output.reset_index()


def pairwise_difference_summary(scores: pd.DataFrame, aliases: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in ALL_SCORE_COLUMNS:
        pivot = scores.pivot(index="article_id", columns="model_alias", values=metric).reindex(columns=aliases)
        for left, right in combinations(aliases, 2):
            pair = pivot[[left, right]].dropna()
            difference = pair[left] - pair[right]
            rho = spearmanr(pair[left], pair[right]).statistic if len(pair) > 1 else math.nan
            rows.append(
                {
                    "score": metric,
                    "model_left": left,
                    "model_right": right,
                    "n_complete": len(pair),
                    "mean_signed_difference": difference.mean(),
                    "median_signed_difference": difference.median(),
                    "mean_absolute_difference": difference.abs().mean(),
                    "rmse_difference": np.sqrt(np.mean(difference**2)) if len(pair) else math.nan,
                    "exact_agreement_rate": (difference == 0).mean() if len(pair) else math.nan,
                    "within_one_point_rate": difference.abs().le(1).mean() if len(pair) else math.nan,
                    "spearman_rho": rho,
                }
            )
    return pd.DataFrame(rows)


def model_score_summary(scores: pd.DataFrame, aliases: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for alias in aliases:
        group = scores[scores["model_alias"].eq(alias)]
        for metric in ALL_SCORE_COLUMNS:
            values = group[metric].dropna()
            rows.append(
                {
                    "model_alias": alias,
                    "score": metric,
                    "n": len(values),
                    "missing": int(group[metric].isna().sum()),
                    "mean": values.mean(),
                    "sd": values.std(ddof=0),
                    "minimum": values.min(),
                    "p25": values.quantile(0.25),
                    "median": values.median(),
                    "p75": values.quantile(0.75),
                    "maximum": values.max(),
                    "zero_share": values.eq(0).mean(),
                }
            )
    return pd.DataFrame(rows)


def operational_model_table(
    scores: pd.DataFrame, registry: list[dict[str, Any]], aliases: list[str]
) -> pd.DataFrame:
    size_map = {str(item["alias"]): float(item["parameter_count_billion"]) for item in registry}
    primary_pivot = scores.pivot(
        index="article_id", columns="model_alias", values=PRIMARY_COLUMN
    ).reindex(columns=aliases)
    correlations = primary_pivot.corr(method="spearman")
    rows: list[dict[str, Any]] = []
    for alias in aliases:
        group = scores[scores["model_alias"].eq(alias)]
        other_correlations = correlations.loc[alias, [other for other in aliases if other != alias]]
        rows.append(
            {
                "model_alias": alias,
                "model_name": group["model_name"].iloc[0],
                "parameter_count_billion": size_map[alias],
                "rows": len(group),
                "successful_rows": int(group["scoring_status"].eq("success").sum()),
                "failed_rows": int((~group["scoring_status"].eq("success")).sum()),
                "scoring_success_rate": group["scoring_status"].eq("success").mean(),
                "mean_pairwise_spearman": other_correlations.mean(),
                "median_inference_seconds": group["model_elapsed_seconds"].median(skipna=True),
                "median_input_tokens": group["total_input_tokens"].median(skipna=True),
                "median_generated_tokens": group["total_generated_tokens"].median(skipna=True),
            }
        )
    return pd.DataFrame(rows)


def topsis_rank(
    alternatives: pd.DataFrame, criteria: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [criterion for criterion in criteria if criterion not in alternatives.columns]
    if missing:
        raise ValueError(f"TOPSIS table lacks criteria: {missing}")
    matrix = alternatives[list(criteria)].astype(float)
    if matrix.isna().any().any():
        missing_cells = matrix.isna().sum()
        raise ValueError(f"TOPSIS criteria contain missing values:\n{missing_cells.to_string()}")

    denominators = np.sqrt((matrix**2).sum(axis=0))
    normalized = matrix.div(denominators.replace(0, np.nan)).fillna(0.0)
    weights = pd.Series({name: float(spec["weight"]) for name, spec in criteria.items()})
    weighted = normalized.mul(weights, axis=1)
    ideal_best: dict[str, float] = {}
    ideal_worst: dict[str, float] = {}
    for name, spec in criteria.items():
        direction = str(spec["direction"]).lower()
        if direction not in {"benefit", "cost"}:
            raise ValueError(f"Invalid TOPSIS direction for {name}: {direction}")
        ideal_best[name] = weighted[name].max() if direction == "benefit" else weighted[name].min()
        ideal_worst[name] = weighted[name].min() if direction == "benefit" else weighted[name].max()

    distance_best = np.sqrt(((weighted - pd.Series(ideal_best)) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted - pd.Series(ideal_worst)) ** 2).sum(axis=1))
    denominator = distance_best + distance_worst
    result = alternatives.copy()
    result["topsis_distance_to_ideal"] = distance_best
    result["topsis_distance_to_anti_ideal"] = distance_worst
    result["topsis_score"] = distance_worst.div(denominator.replace(0, np.nan))
    result["topsis_rank"] = result["topsis_score"].rank(method="min", ascending=False).astype("Int64")
    result["ranking_scope"] = "provisional_operational_not_measurement_validity"
    result = result.sort_values(["topsis_rank", "model_alias"]).reset_index(drop=True)

    details: list[dict[str, Any]] = []
    for name, spec in criteria.items():
        for index, row in alternatives.reset_index(drop=True).iterrows():
            details.append(
                {
                    "model_alias": row["model_alias"],
                    "criterion": name,
                    "raw_value": matrix.iloc[index][name],
                    "normalized_value": normalized.iloc[index][name],
                    "weight": float(spec["weight"]),
                    "direction": spec["direction"],
                    "weighted_normalized_value": weighted.iloc[index][name],
                    "ideal_best": ideal_best[name],
                    "ideal_worst": ideal_worst[name],
                }
            )
    return result, pd.DataFrame(details)


def write_outputs(
    args: argparse.Namespace,
    scores_path: Path,
    article: pd.DataFrame,
    weekly: pd.DataFrame,
    model_weekly: pd.DataFrame,
    weekly_spread: pd.DataFrame,
    monthly: pd.DataFrame,
    model_monthly: pd.DataFrame,
    monthly_spread: pd.DataFrame,
    score_summary: pd.DataFrame,
    differences: pd.DataFrame,
    ranking: pd.DataFrame,
    ranking_details: pd.DataFrame,
    diagnostics: dict[str, Any],
    criteria: dict[str, dict[str, Any]],
    log_path: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    article_dir = output_dir / "article_level"
    weekly_dir = output_dir / "weekly"
    monthly_dir = output_dir / "monthly"
    uncertainty_dir = output_dir / "uncertainty_bands"
    version_dir = output_dir / "disruption_index_versions"
    for directory in (article_dir, weekly_dir, monthly_dir, uncertainty_dir, version_dir):
        directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "article_consensus": article_dir / "article_consensus_scores.csv",
        "weekly_indices": weekly_dir / "msdi_weekly.csv",
        "weekly_model_specific": weekly_dir / "msdi_weekly_model_specific.csv",
        "weekly_model_spread": uncertainty_dir / "msdi_weekly_model_spread.csv",
        "monthly_indices": monthly_dir / "msdi_monthly.csv",
        "monthly_model_specific": monthly_dir / "msdi_monthly_model_specific.csv",
        "monthly_model_spread": uncertainty_dir / "msdi_monthly_model_spread.csv",
        "model_score_summary": version_dir / f"{args.version}_model_score_summary.csv",
        "pairwise_score_differences": version_dir / f"{args.version}_pairwise_score_differences.csv",
        "topsis_model_ranking": version_dir / f"{args.version}_topsis_model_ranking.csv",
        "topsis_calculation_details": version_dir / f"{args.version}_topsis_calculation_details.csv",
        "manifest": version_dir / f"{args.version}_manifest.json",
    }
    article.to_csv(paths["article_consensus"], index=False)
    weekly.to_csv(paths["weekly_indices"], index=False)
    model_weekly.to_csv(paths["weekly_model_specific"], index=False)
    weekly_spread.to_csv(paths["weekly_model_spread"], index=False)
    monthly.to_csv(paths["monthly_indices"], index=False)
    model_monthly.to_csv(paths["monthly_model_specific"], index=False)
    monthly_spread.to_csv(paths["monthly_model_spread"], index=False)
    score_summary.to_csv(paths["model_score_summary"], index=False)
    differences.to_csv(paths["pairwise_score_differences"], index=False)
    ranking.to_csv(paths["topsis_model_ranking"], index=False)
    ranking_details.to_csv(paths["topsis_calculation_details"], index=False)

    manifest = {
        "version": args.version,
        "status": "provisional_pending_human_validation",
        "input": {
            "scores_file": str(scores_path),
            "size_bytes": scores_path.stat().st_size,
            "sha256": file_sha256(scores_path),
        },
        "score_scale": {"minimum": SCORE_MIN, "maximum": SCORE_MAX},
        "measurement_paths": {
            "primary_peak": "median overall_disruption_max across available successful models",
            "secondary_breadth": "equal-weight mean of seven dimension-level cross-model medians",
        },
        "index_construction": {
            "primary_frequency": "monthly",
            "weekly_robustness_frequency": args.frequency,
            "monthly_frequency": args.monthly_frequency,
            "minimum_models": args.minimum_models,
            "duplicate_penalty": args.duplicate_penalty,
            "novelty_formula": {
                "sqrt": "1 / sqrt(duplicate_group_size)",
                "inverse": "1 / duplicate_group_size",
                "none": "1",
            }[args.duplicate_penalty],
            "event_penalty": args.event_penalty,
            "event_novelty_formula": {
                "fixed": (
                    "within each aggregation period, source_duplicate_weight divided by "
                    "the event's summed source_duplicate_weight; every event totals one"
                ),
                "sqrt": (
                    "fixed-event weight multiplied by sqrt(within-period event article count)"
                ),
                "none": "source duplicate weight only",
            }[args.event_penalty],
            "event_weight_leakage_guard": (
                "Global event cluster size is descriptive only; model index weights are "
                "recomputed within each contemporaneously available aggregation period."
            ),
            "weekly_media_correction": args.media_correction,
            "media_formula": {
                "log": "weekly weighted sum / log(1 + weekly article count)",
                "mean": "weekly weighted sum / weekly article count",
                "none": "weekly weighted sum",
            }[args.media_correction],
            "monthly_primary_formula": (
                "Direct article-level monthly event-weighted score sum divided by "
                "sqrt(monthly effective article count), followed by causal EMA(3); "
                "event_id_monthly is used and no weekly index value is averaged."
            ),
            "full_sample_zscore_warning": "Descriptive only; do not use in forecasting.",
            "realtime_zscore": (
                f"Current value standardized using prior observations only; minimum "
                f"{args.expanding_min_periods} prior periods."
            ),
            "outcome_leakage": "No SCFI or other outcome was read or used.",
        },
        "scfi_predictor_columns_for_later_scripts": [
            "peak_score_media_corrected",
            "breadth_score_media_corrected",
            "peak_msdi_z_expanding_realtime",
            "breadth_msdi_z_expanding_realtime",
        ],
        "topsis": {
            "scope": "Operational ranking only; never used as an MSDI ensemble weight.",
            "criteria": criteria,
            "human_validation_included": False,
            "warning": (
                "Inter-model agreement is not ground-truth validity. Final quality-aware ranking "
                "must be recomputed after human validation."
            ),
        },
        "diagnostics": diagnostics,
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "log_file": str(log_path),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, default=json_value), encoding="utf-8"
    )
    for name, path in paths.items():
        logger.info("Output %s: %s", name, path)
    return manifest


def main() -> None:
    args = parse_args()
    if args.minimum_models < 1 or args.minimum_models > 3:
        raise ValueError("minimum-models must be between 1 and 3")
    if args.expanding_min_periods < 2:
        raise ValueError("expanding-min-periods must be at least 2")
    if args.monthly_expanding_min_periods < 2:
        raise ValueError("monthly-expanding-min-periods must be at least 2")

    logger, log_path = configure_logging(args.log_dir.resolve(), args.version)
    logger.info("Starting MSDI build version=%s", args.version)
    logger.info("Arguments: %s", vars(args))
    registry, criteria = load_model_config(args.model_config.resolve())
    scores_path = args.scores.resolve()
    event_profile = resolve_event_profile(args.frequency, args.event_profile)
    logger.info("Using %s semantic-event assignment", event_profile)
    scores = read_scores(scores_path, registry, logger, event_profile)
    aliases = [str(item["alias"]) for item in registry]

    article_base, diagnostics = build_article_scores(
        scores,
        aliases,
        args.minimum_models,
        args.duplicate_penalty,
        logger,
    )
    weekly_article = apply_period_event_weights(
        select_event_profile(article_base, "weekly"), args.frequency, args.event_penalty
    )
    monthly_article = apply_period_event_weights(
        select_event_profile(article_base, "monthly"),
        args.monthly_frequency,
        args.event_penalty,
    )
    # The canonical article table exposes the monthly-primary generic fields and
    # retains both frequency-specific audit weights for downstream robustness.
    article = monthly_article.copy()
    for column in (
        "event_period_article_count",
        "event_period_source_weight_sum",
        "novelty_weight",
        "peak_weighted_contribution",
        "breadth_weighted_contribution",
    ):
        article[f"{column}_monthly"] = monthly_article[column]
        article[f"{column}_weekly"] = weekly_article[column]
    diagnostics["event_clustering"] = {}
    for profile, weighted in (("weekly", weekly_article), ("monthly", monthly_article)):
        eligible_weighted = weighted[weighted["ensemble_eligible"]]
        diagnostics["event_clustering"][profile] = {
            "unique_events": int(eligible_weighted["event_id"].nunique()),
            "multi_article_events_global_descriptive": int(
                eligible_weighted.groupby("event_id")["article_id"].nunique().gt(1).sum()
            ),
            "articles_in_multi_article_events_global_descriptive": int(
                eligible_weighted["event_cluster_size"].gt(1).sum()
            ),
            "period_local_weighting": args.event_penalty,
            "maximum_event_period_article_count": int(
                eligible_weighted["event_period_article_count"].max()
            ),
            "ineligible_articles_receive_zero_event_weight": True,
        }
    weekly = aggregate_weekly(
        weekly_article, args.frequency, args.media_correction, args.expanding_min_periods
    )
    model_weekly = aggregate_model_weekly(
        weekly_article, aliases, args.frequency, args.media_correction
    )
    weekly_spread = build_model_spread(model_weekly, aliases, "week_ending")
    monthly = aggregate_monthly_direct(
        monthly_article,
        args.monthly_frequency,
        args.monthly_expanding_min_periods,
        ema_span=3,
    )
    model_monthly = aggregate_model_monthly_direct(
        monthly_article, aliases, args.monthly_frequency, ema_span=3
    )
    monthly_spread = build_model_spread(model_monthly, aliases, "month_end")
    score_summary = model_score_summary(scores, aliases)
    differences = pairwise_difference_summary(scores, aliases)
    operational = operational_model_table(scores, registry, aliases)
    ranking, ranking_details = topsis_rank(operational, criteria)

    logger.info(
        "Weekly coverage: periods=%s first=%s last=%s",
        len(weekly),
        weekly["week_ending"].min(),
        weekly["week_ending"].max(),
    )
    logger.info(
        "Monthly primary coverage: periods=%s first=%s last=%s",
        len(monthly),
        monthly["month_end"].min(),
        monthly["month_end"].max(),
    )
    for _, row in ranking.iterrows():
        logger.info(
            "TOPSIS operational rank %s: %s score=%.6f",
            row["topsis_rank"],
            row["model_alias"],
            row["topsis_score"],
        )

    write_outputs(
        args,
        scores_path,
        article,
        weekly,
        model_weekly,
        weekly_spread,
        monthly,
        model_monthly,
        monthly_spread,
        score_summary,
        differences,
        ranking,
        ranking_details,
        diagnostics,
        criteria,
        log_path,
        logger,
    )
    logger.info("MSDI build completed successfully")


if __name__ == "__main__":
    main()
