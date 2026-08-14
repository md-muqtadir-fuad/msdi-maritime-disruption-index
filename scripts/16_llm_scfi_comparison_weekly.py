"""Compare four LLM score paths and leakage-safe hybrids for SCFI prediction.

This stage answers a measurement-model question that the consensus forecast
benchmark cannot answer: which individual LLM score series contains the most
out-of-sample information for future SCFI?  All individual LLMs use the same
article cohort, forecast dates, controls, estimator, and expanding-window
protocol.  Score-level and forecast-level hybrids are retained as separate
objects so their performance is not conflated.

The target is the forward SCFI log change.  SCFI levels are reconstructed for
interpretability.  Results are predictive, not structural causal estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import sys
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls
from scipy.stats import norm, spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTICLE_SCORES = (
    PROJECT_ROOT / "data/05_indices/article_level/article_consensus_scores.csv"
)
DEFAULT_PANEL = (
    PROJECT_ROOT / "data/06_model_ready/prediction/msdi_scfi_weekly_panel.csv"
)
DEFAULT_PANEL_OUTPUT = (
    PROJECT_ROOT
    / "data/06_model_ready/prediction/llm_specific_scfi_weekly_panel.csv"
)
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs/model_registry.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/llm_scfi_comparison"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs/logs"

HORIZONS = (1, 2, 4, 6)
LAGS = (1, 2, 4, 8, 12)
ROLLING_WINDOWS = (4, 13, 26, 52)
DIMENSION_COLUMNS = (
    "port_disruption_score",
    "global_container_movement_disruption_score",
    "freight_cost_disruption_score",
    "supply_disruption_score",
    "production_disruption_score",
    "covid19_related_disruption_score",
    "demand_disruption_score",
)
PRIMARY_SCORE_COLUMN = "overall_disruption_max"
PRIMARY_SIGNALS = (PRIMARY_SCORE_COLUMN, "dimension_mean7")
SIGNALS = (PRIMARY_SCORE_COLUMN, "dimension_mean7", *DIMENSION_COLUMNS)
SCORE_HYBRID_SOURCES = ("score_median", "score_mean")
PRICE_OVERLAP_SIGNAL = "freight_cost_disruption_score"
EXPECTED_COMMON_COHORT_SIZE = 6480
NOVELTY_WEIGHTING_PROVENANCE = (
    "source duplicate weights are renormalized within each weekly event "
    "using only the frozen common article cohort"
)

AR_FEATURES = (
    "scfi_log",
    "scfi_log_lag_1w",
    "scfi_log_lag_2w",
    "scfi_log_lag_4w",
    "scfi_log_lag_8w",
    "scfi_log_lag_12w",
    "scfi_log_change_1w",
    "scfi_log_change_4w",
    "scfi_log_change_13w",
    "calendar_week_sin",
    "calendar_week_cos",
    "calendar_month_sin",
    "calendar_month_cos",
)
BRENT_FEATURES = (
    "brent_log",
    "brent_log_lag_1w",
    "brent_log_lag_2w",
    "brent_log_lag_4w",
    "brent_log_lag_8w",
    "brent_log_lag_12w",
    "brent_log_change_1w",
    "brent_log_change_4w",
    "brent_log_change_13w",
)
COVERAGE_FEATURES = ("llm_common_article_count", "llm_common_effective_article_count")
EXTERNAL_CONTROL_FEATURES = (
    *[f"cpb_world_trade_volume_lag_{lag}w" for lag in (1, 2, 4, 8, 12)],
    *[f"unctad_china_core_lane_lsbci_lag_{lag}w" for lag in (1, 2, 4, 8, 12)],
)
EXTERNAL_CONTROL_RAW_COLUMNS = ("cpb_world_trade_volume", "unctad_china_core_lane_lsbci")
BASELINE_MODEL_IDS = {
    "naive_zero_change",
    "historical_mean_change",
    "ridge__ar_brent",
    "ridge__ar_brent_coverage",
}


@dataclass(frozen=True)
class ForecastConfiguration:
    model_id: str
    source_type: str
    score_source: str
    signal_name: str
    hybrid_level: str
    feature_set: str

    @property
    def price_overlap_risk(self) -> bool:
        return self.signal_name == PRICE_OVERLAP_SIGNAL

    @property
    def regime_specific_risk(self) -> bool:
        return self.signal_name == "covid19_related_disruption_score"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare individual LLM and hybrid score paths for SCFI prediction."
    )
    parser.add_argument("--article-scores", type=Path, default=DEFAULT_ARTICLE_SCORES)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--panel-output", type=Path, default=DEFAULT_PANEL_OUTPUT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--version", default="llm_scfi_v001")
    parser.add_argument("--minimum-training-rows", type=int, default=156)
    parser.add_argument("--refit-every", type=int, default=13)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--minimum-stacking-history", type=int, default=52)
    parser.add_argument("--maximum-feature-age-days", type=int, default=6)
    parser.add_argument(
        "--expected-common-articles",
        type=int,
        default=EXPECTED_COMMON_COHORT_SIZE,
        help=(
            "Fail closed unless the frozen all-four-LLM complete cohort has this size. "
            "Change only when intentionally versioning a new input dataset."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def configure_logging(log_dir: Path, version: str) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"07b_llm_scfi_comparison_{version}.log"
    logger = logging.getLogger("llm_scfi_comparison")
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


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def load_model_aliases(path: Path) -> list[str]:
    resolved = require_file(path, "Model registry")
    config = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    models = config.get("active_scoring_models", [])
    aliases = [str(item.get("alias", "")).strip() for item in models]
    validate_model_aliases(aliases)
    return aliases


def validate_model_aliases(aliases: Iterable[str]) -> list[str]:
    result = list(aliases)
    if len(result) != 4:
        raise ValueError(f"Exactly four model aliases are required; found {result}")
    if any(not alias for alias in result):
        raise ValueError("Model aliases cannot be blank")
    if len(set(result)) != len(result):
        raise ValueError(f"Model aliases must be unique; found {result}")
    return result


def score_column(signal: str, alias: str) -> str:
    if signal == "dimension_mean7":
        raise ValueError("dimension_mean7 is derived and has no single score column")
    return f"{signal}__{alias}"


def signal_raw_column(score_source: str, signal: str) -> str:
    return f"llm_signal__{score_source}__{signal}"


def signal_feature_columns(raw_column: str) -> list[str]:
    return [
        raw_column,
        *[f"{raw_column}_lag_{lag}w" for lag in LAGS],
        *[
            item
            for window in ROLLING_WINDOWS
            for item in (
                f"{raw_column}_rolling_mean_{window}w",
                f"{raw_column}_rolling_sd_{window}w",
            )
        ],
        f"{raw_column}_acceleration_1w",
    ]


def load_article_scores(
    path: Path,
    aliases: list[str],
    expected_common_articles: int | None = EXPECTED_COMMON_COHORT_SIZE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = require_file(path, "Article consensus scores")
    required_score_columns = [
        f"{signal}__{alias}"
        for alias in aliases
        for signal in (PRIMARY_SCORE_COLUMN, *DIMENSION_COLUMNS)
    ]
    available = set(pd.read_csv(resolved, nrows=0).columns)
    event_weight_columns = [
        "source_duplicate_weight",
        "event_id_weekly",
        "event_id_monthly",
    ]
    has_profile_event_fields = set(event_weight_columns).issubset(available)
    required = [
        "article_id",
        "publication_date",
        *(event_weight_columns if has_profile_event_fields else ["novelty_weight"]),
        *required_score_columns,
    ]
    frame = pd.read_csv(resolved, usecols=required, low_memory=False)
    if frame.empty:
        raise ValueError("Article score file is empty")
    if frame["article_id"].isna().any() or frame["article_id"].duplicated().any():
        raise ValueError("Article score file must have unique, nonmissing article_id values")
    frame["publication_date"] = pd.to_datetime(
        frame["publication_date"], format="mixed", errors="coerce"
    )
    if not has_profile_event_fields:
        # Read-only compatibility for already-generated legacy artifacts. A new
        # Stage 13 run is required for publication analysis.
        frame["source_duplicate_weight"] = frame.pop("novelty_weight")
        frame["event_id_weekly"] = "ARTICLE_" + frame["article_id"].astype(str)
        frame["event_id_monthly"] = frame["event_id_weekly"]
    frame["source_duplicate_weight"] = pd.to_numeric(
        frame["source_duplicate_weight"], errors="coerce"
    )
    if frame["source_duplicate_weight"].isna().any() or frame[
        "source_duplicate_weight"
    ].le(0).any():
        raise ValueError("source_duplicate_weight must be positive and nonmissing")
    for column in required_score_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & ~frame[column].between(0, 4)
        if invalid.any():
            raise ValueError(
                f"{column} contains {int(invalid.sum())} values outside the 0-4 rubric"
            )

    common_mask = frame["publication_date"].notna() & frame[
        required_score_columns
    ].notna().all(axis=1)
    if not common_mask.any():
        raise ValueError("No common complete articles exist across all four LLMs")
    common_count = int(common_mask.sum())
    if expected_common_articles is not None and common_count != expected_common_articles:
        raise ValueError(
            "Frozen common-cohort size mismatch: "
            f"expected {expected_common_articles:,}, found {common_count:,}. "
            "Confirm the input version before changing --expected-common-articles."
        )
    report = {
        "rows": len(frame),
        "unparseable_publication_dates": int(frame["publication_date"].isna().sum()),
        "common_complete_articles": common_count,
        "expected_common_complete_articles": expected_common_articles,
        "common_cohort_size_check_passed": (
            expected_common_articles is None or common_count == expected_common_articles
        ),
        "excluded_from_common_cohort": int((~common_mask).sum()),
        "common_first_date": frame.loc[common_mask, "publication_date"].min(),
        "common_last_date": frame.loc[common_mask, "publication_date"].max(),
        "model_availability": {},
        "profile_specific_event_fields_present": has_profile_event_fields,
    }
    for alias in aliases:
        alias_columns = [
            f"{signal}__{alias}" for signal in (PRIMARY_SCORE_COLUMN, *DIMENSION_COLUMNS)
        ]
        report["model_availability"][alias] = {
            "all_eight_scores": int(frame[alias_columns].notna().all(axis=1).sum()),
            "overall_score": int(frame[f"{PRIMARY_SCORE_COLUMN}__{alias}"].notna().sum()),
        }
    frame["llm_common_complete"] = common_mask
    frame["novelty_weight"] = 0.0
    common = frame.loc[common_mask].copy()
    common["_period"] = common["publication_date"].dt.to_period("W-FRI").astype(str)
    event_id = common["event_id_weekly"].astype("string").str.strip()
    event_id = event_id.mask(
        event_id.isna() | event_id.eq(""), "ARTICLE_" + common["article_id"].astype(str)
    )
    common["_event"] = event_id
    denominator = common.groupby(["_period", "_event"], sort=False)[
        "source_duplicate_weight"
    ].transform("sum")
    common["novelty_weight"] = common["source_duplicate_weight"].div(denominator)
    if common["novelty_weight"].isna().any() or common["novelty_weight"].le(0).any():
        raise AssertionError("Common weekly event weights must be positive and nonmissing")
    frame.loc[common.index, "novelty_weight"] = common["novelty_weight"]
    return frame, report


def article_signal_values(
    frame: pd.DataFrame,
    aliases: list[str],
    score_source: str,
    signal: str,
) -> pd.Series:
    if score_source in aliases:
        if signal == "dimension_mean7":
            columns = [f"{dimension}__{score_source}" for dimension in DIMENSION_COLUMNS]
            return frame[columns].mean(axis=1, skipna=False)
        return frame[score_column(signal, score_source)]

    if score_source not in SCORE_HYBRID_SOURCES:
        raise ValueError(f"Unknown score source: {score_source}")
    operation = "median" if score_source == "score_median" else "mean"
    if signal == "dimension_mean7":
        dimension_ensembles: list[pd.Series] = []
        for dimension in DIMENSION_COLUMNS:
            columns = [f"{dimension}__{alias}" for alias in aliases]
            dimension_ensembles.append(
                frame[columns].agg(operation, axis=1, skipna=False)
            )
        return pd.concat(dimension_ensembles, axis=1).mean(axis=1, skipna=False)
    columns = [score_column(signal, alias) for alias in aliases]
    return frame[columns].agg(operation, axis=1, skipna=False)


def complete_week_index(frame: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(first, last, freq="W-FRI")
    result = frame.reindex(dates)
    result.index.name = "week_ending"
    return result


def build_common_weekly_signals(
    article: pd.DataFrame, aliases: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = article.loc[article["llm_common_complete"]].copy()
    first_week = common["publication_date"].min().to_period("W-FRI").end_time.normalize()
    last_week = common["publication_date"].max().to_period("W-FRI").end_time.normalize()
    indexed = common.set_index("publication_date")
    coverage = indexed.resample("W-FRI", label="right", closed="right").agg(
        llm_common_article_count=("article_id", "nunique"),
        llm_common_effective_article_count=("novelty_weight", "sum"),
    )
    coverage = complete_week_index(coverage, first_week, last_week)
    coverage[[*COVERAGE_FEATURES]] = coverage[[*COVERAGE_FEATURES]].fillna(0.0)

    rows: list[pd.DataFrame] = []
    sources = [*aliases, *SCORE_HYBRID_SOURCES]
    for score_source in sources:
        source_type = "llm" if score_source in aliases else "score_hybrid"
        for signal in SIGNALS:
            work = common[["publication_date", "novelty_weight"]].copy()
            work["signal_value"] = article_signal_values(
                common, aliases, score_source, signal
            ).to_numpy()
            if work["signal_value"].isna().any():
                raise AssertionError(
                    f"Common cohort unexpectedly has missing {score_source}/{signal} values"
                )
            work["weighted_contribution"] = (
                work["signal_value"] * work["novelty_weight"]
            )
            grouped = work.set_index("publication_date").resample(
                "W-FRI", label="right", closed="right"
            ).agg(
                signal_weighted_sum=("weighted_contribution", "sum"),
                signal_unweighted_mean=("signal_value", "mean"),
                signal_min=("signal_value", "min"),
                signal_max=("signal_value", "max"),
            )
            grouped = complete_week_index(grouped, first_week, last_week)
            grouped["signal_weighted_sum"] = grouped["signal_weighted_sum"].fillna(0.0)
            grouped = coverage.join(grouped)
            denominator = np.log1p(grouped["llm_common_article_count"].astype(float))
            grouped["signal_media_corrected"] = (
                grouped["signal_weighted_sum"]
                .div(denominator.where(denominator.gt(0)))
                .fillna(0.0)
            )
            grouped["source_type"] = source_type
            grouped["score_source"] = score_source
            grouped["signal_name"] = signal
            grouped["price_overlap_risk"] = signal == PRICE_OVERLAP_SIGNAL
            grouped["regime_specific_risk"] = (
                signal == "covid19_related_disruption_score"
            )
            rows.append(grouped.reset_index())
    long = pd.concat(rows, ignore_index=True).sort_values(
        ["week_ending", "source_type", "score_source", "signal_name"]
    )
    return long.reset_index(drop=True), coverage.reset_index()


def build_signal_difference_summary(
    article: pd.DataFrame, weekly_long: pd.DataFrame, aliases: list[str]
) -> pd.DataFrame:
    common = article.loc[article["llm_common_complete"]]
    rows: list[dict[str, Any]] = []
    for signal in SIGNALS:
        article_values = {
            alias: article_signal_values(common, aliases, alias, signal) for alias in aliases
        }
        weekly = weekly_long[
            weekly_long["source_type"].eq("llm")
            & weekly_long["signal_name"].eq(signal)
        ].pivot(
            index="week_ending", columns="score_source", values="signal_media_corrected"
        )
        for left, right in combinations(aliases, 2):
            for level, left_values, right_values in (
                ("article", article_values[left], article_values[right]),
                ("weekly_media_corrected", weekly[left], weekly[right]),
            ):
                pair = pd.concat(
                    [left_values.rename("left"), right_values.rename("right")], axis=1
                ).dropna()
                difference = pair["left"] - pair["right"]
                rho = (
                    spearmanr(pair["left"], pair["right"]).statistic
                    if len(pair) > 1
                    else math.nan
                )
                rows.append(
                    {
                        "level": level,
                        "signal_name": signal,
                        "model_left": left,
                        "model_right": right,
                        "n_complete": len(pair),
                        "mean_signed_difference": difference.mean(),
                        "median_signed_difference": difference.median(),
                        "mean_absolute_difference": difference.abs().mean(),
                        "rmse_difference": (
                            np.sqrt(np.mean(difference**2)) if len(pair) else math.nan
                        ),
                        "spearman_correlation": rho,
                        "price_overlap_risk": signal == PRICE_OVERLAP_SIGNAL,
                    }
                )
    return pd.DataFrame(rows)


def pivot_common_weekly_signals(
    weekly_long: pd.DataFrame, aliases: list[str]
) -> pd.DataFrame:
    expected_sources = [*aliases, *SCORE_HYBRID_SOURCES]
    keys = weekly_long[["score_source", "signal_name"]].drop_duplicates()
    expected_keys = pd.MultiIndex.from_product(
        [expected_sources, SIGNALS], names=["score_source", "signal_name"]
    )
    found_keys = pd.MultiIndex.from_frame(keys).sort_values()
    if set(found_keys) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(found_keys))
        extra = sorted(set(found_keys) - set(expected_keys))
        raise AssertionError(f"Unexpected weekly signal keys; missing={missing}, extra={extra}")

    pivot = weekly_long.pivot(
        index="week_ending",
        columns=["score_source", "signal_name"],
        values="signal_media_corrected",
    )
    pivot.columns = [signal_raw_column(source, signal) for source, signal in pivot.columns]
    coverage = weekly_long[
        ["week_ending", *COVERAGE_FEATURES]
    ].drop_duplicates("week_ending")
    if coverage["week_ending"].duplicated().any():
        raise AssertionError("Common-cohort weekly coverage has duplicate weeks")
    result = coverage.set_index("week_ending").join(pivot).reset_index()
    expected_signal_count = len(expected_sources) * len(SIGNALS)
    actual_signal_count = len([column for column in result if column.startswith("llm_signal__")])
    if actual_signal_count != expected_signal_count:
        raise AssertionError(
            f"Expected {expected_signal_count} common signal columns; found {actual_signal_count}"
        )
    return result.sort_values("week_ending").reset_index(drop=True)


def load_base_panel(path: Path) -> pd.DataFrame:
    resolved = require_file(path, "SCFI model-ready panel")
    raw = pd.read_csv(resolved, low_memory=False)
    required = {
        "observation_date",
        "scfi",
        *AR_FEATURES,
        *BRENT_FEATURES,
        *[f"target_scfi_log_change_h{horizon}w" for horizon in HORIZONS],
        *[f"target_scfi_level_h{horizon}w" for horizon in HORIZONS],
        *EXTERNAL_CONTROL_FEATURES,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"SCFI panel lacks required non-MSDI columns: {missing}")
    optional = [
        column
        for column in ("brent_observation_date", "brent_age_days", "brent_price")
        if column in raw.columns
    ]
    selected = [
        "observation_date",
        "scfi",
        *optional,
        *AR_FEATURES,
        *BRENT_FEATURES,
        *[f"target_scfi_log_change_h{horizon}w" for horizon in HORIZONS],
        *[f"target_scfi_level_h{horizon}w" for horizon in HORIZONS],
        *EXTERNAL_CONTROL_FEATURES,
        *EXTERNAL_CONTROL_RAW_COLUMNS,
    ]
    selected = list(dict.fromkeys(selected))
    panel = raw[selected].copy()
    panel["observation_date"] = pd.to_datetime(
        panel["observation_date"], format="mixed", errors="coerce"
    )
    if panel["observation_date"].isna().any():
        raise ValueError("SCFI panel has unparseable observation dates")
    if "brent_observation_date" in panel:
        panel["brent_observation_date"] = pd.to_datetime(
            panel["brent_observation_date"], format="mixed", errors="coerce"
        )
    for column in panel.columns:
        if column in {"observation_date", "brent_observation_date"}:
            continue
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.sort_values("observation_date").reset_index(drop=True)
    if panel["observation_date"].duplicated().any():
        raise ValueError("SCFI panel contains duplicate observation dates")
    gaps = panel["observation_date"].diff().dropna().dt.days
    if not gaps.eq(7).all():
        raise ValueError("SCFI panel must be a regular weekly sequence")
    return panel


def add_past_signal_features(
    panel: pd.DataFrame, raw_signal_columns: list[str]
) -> pd.DataFrame:
    result = panel.copy()
    derived: dict[str, pd.Series] = {}
    for raw_column in raw_signal_columns:
        if raw_column not in result:
            raise ValueError(f"Panel lacks raw signal column: {raw_column}")
        for lag in LAGS:
            derived[f"{raw_column}_lag_{lag}w"] = result[raw_column].shift(lag)
        for window in ROLLING_WINDOWS:
            rolling = result[raw_column].rolling(window, min_periods=window)
            derived[f"{raw_column}_rolling_mean_{window}w"] = rolling.mean()
            derived[f"{raw_column}_rolling_sd_{window}w"] = rolling.std(ddof=0)
        derived[f"{raw_column}_acceleration_1w"] = result[raw_column].diff().diff()
    if derived:
        result = pd.concat([result, pd.DataFrame(derived, index=result.index)], axis=1)
    return result


def build_aligned_panel(
    base_panel: pd.DataFrame,
    weekly_wide: pd.DataFrame,
    aliases: list[str],
    maximum_feature_age_days: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if maximum_feature_age_days < 0:
        raise ValueError("maximum_feature_age_days cannot be negative")
    weekly = weekly_wide.copy()
    weekly["llm_week_ending"] = pd.to_datetime(
        weekly.pop("week_ending"), format="mixed", errors="coerce"
    )
    if weekly["llm_week_ending"].isna().any():
        raise ValueError("Common weekly signals have unparseable dates")
    if weekly["llm_week_ending"].duplicated().any():
        raise ValueError("Common weekly signals have duplicate dates")
    panel = pd.merge_asof(
        base_panel.sort_values("observation_date"),
        weekly.sort_values("llm_week_ending"),
        left_on="observation_date",
        right_on="llm_week_ending",
        direction="backward",
        tolerance=pd.Timedelta(days=maximum_feature_age_days),
        allow_exact_matches=True,
    )
    panel["llm_signal_age_days"] = (
        panel["observation_date"] - panel["llm_week_ending"]
    ).dt.days
    matched = panel["llm_week_ending"].notna()
    future = matched & panel["llm_week_ending"].gt(panel["observation_date"])
    stale = matched & ~panel["llm_signal_age_days"].between(
        0, maximum_feature_age_days
    )
    if future.any() or stale.any():
        raise AssertionError(
            f"Unsafe common-signal alignment: future={int(future.sum())}, stale={int(stale.sum())}"
        )

    score_sources = [*aliases, *SCORE_HYBRID_SOURCES]
    raw_signal_columns = [
        signal_raw_column(source, signal)
        for source in score_sources
        for signal in SIGNALS
    ]
    panel = add_past_signal_features(panel, raw_signal_columns)
    current_requirements = [
        "scfi",
        "brent_log",
        *COVERAGE_FEATURES,
        *raw_signal_columns,
        *EXTERNAL_CONTROL_FEATURES,
    ]
    panel["llm_comparison_complete_case"] = panel[current_requirements].notna().all(axis=1)
    report = {
        "rows": len(panel),
        "matched_common_signal_weeks": int(matched.sum()),
        "unmatched_common_signal_weeks": int((~matched).sum()),
        "future_matches": int(future.sum()),
        "stale_matches": int(stale.sum()),
        "maximum_feature_age_days": maximum_feature_age_days,
        "age_days_counts": panel.loc[
            matched, "llm_signal_age_days"
        ].value_counts().sort_index().to_dict(),
        "comparison_complete_rows": int(panel["llm_comparison_complete_case"].sum()),
        "first_complete_date": panel.loc[
            panel["llm_comparison_complete_case"], "observation_date"
        ].min(),
        "last_complete_date": panel.loc[
            panel["llm_comparison_complete_case"], "observation_date"
        ].max(),
        "raw_signal_columns": len(raw_signal_columns),
    }
    return panel.reset_index(drop=True), report


def build_direct_configurations(aliases: list[str]) -> list[ForecastConfiguration]:
    configurations: list[ForecastConfiguration] = []
    for score_source in [*aliases, *SCORE_HYBRID_SOURCES]:
        source_type = "llm" if score_source in aliases else "score_hybrid"
        for signal in SIGNALS:
            if source_type == "llm":
                model_id = f"ridge__llm__{score_source}__{signal}"
                hybrid_level = "none"
            else:
                model_id = f"ridge__score_hybrid__{score_source}__{signal}"
                hybrid_level = "score"
            configurations.append(
                ForecastConfiguration(
                    model_id=model_id,
                    source_type=source_type,
                    score_source=score_source,
                    signal_name=signal,
                    hybrid_level=hybrid_level,
                    feature_set=f"ar_brent_coverage_plus_{score_source}_{signal}",
                )
            )
    return configurations


def build_feature_sets(
    panel: pd.DataFrame, configurations: list[ForecastConfiguration]
) -> dict[str, list[str]]:
    available = set(panel.columns)
    base = [*AR_FEATURES, *BRENT_FEATURES]
    coverage = [*base, *COVERAGE_FEATURES, *EXTERNAL_CONTROL_FEATURES]
    missing_base = [column for column in coverage if column not in available]
    if missing_base:
        raise ValueError(f"Aligned panel lacks baseline features: {missing_base}")
    feature_sets: dict[str, list[str]] = {
        "ridge__ar_brent": base,
        "ridge__ar_brent_coverage": coverage,
    }
    for config in configurations:
        raw_column = signal_raw_column(config.score_source, config.signal_name)
        features = [*coverage, *signal_feature_columns(raw_column)]
        missing = [column for column in features if column not in available]
        if missing:
            raise ValueError(f"Feature set {config.model_id} lacks columns: {missing}")
        forbidden = [column for column in features if column.startswith("target_")]
        if forbidden:
            raise AssertionError(f"Target leakage in {config.model_id}: {forbidden}")
        feature_sets[config.model_id] = features
    return feature_sets


def training_positions(
    panel: pd.DataFrame,
    origin_position: int,
    horizon: int,
    target_column: str,
) -> np.ndarray:
    maximum_position = origin_position - horizon
    if maximum_position < 0:
        return np.array([], dtype=int)
    candidate = np.arange(maximum_position + 1)
    complete = panel.loc[
        candidate, "llm_comparison_complete_case"
    ].astype(bool).to_numpy()
    target_available = panel.loc[candidate, target_column].notna().to_numpy()
    return candidate[complete & target_available]


def eligible_origins(panel: pd.DataFrame, horizon: int, target_column: str) -> list[int]:
    target_date = panel["observation_date"].shift(-horizon)
    mask = (
        panel["llm_comparison_complete_case"].astype(bool)
        & panel[target_column].notna()
        & target_date.notna()
    )
    return panel.index[mask].tolist()


def make_ridge_estimator(alpha: float) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=alpha)),
            ]
        ),
        transformer=StandardScaler(),
    )


def forecast_row(
    panel: pd.DataFrame,
    position: int,
    horizon: int,
    prediction: float,
    model_id: str,
    feature_set: str,
    source_type: str,
    score_source: str,
    signal_name: str,
    hybrid_level: str,
    training_rows: int,
    training_feature_end_date: pd.Timestamp,
    training_label_end_date: pd.Timestamp,
    model_fit_origin_date: pd.Timestamp,
) -> dict[str, Any]:
    target_column = f"target_scfi_log_change_h{horizon}w"
    level_column = f"target_scfi_level_h{horizon}w"
    actual = float(panel.at[position, target_column])
    current_scfi = float(panel.at[position, "scfi"])
    actual_level = float(panel.at[position, level_column])
    predicted_level = current_scfi * math.exp(prediction)
    return {
        "origin_position": int(position),
        "origin_date": panel.at[position, "observation_date"],
        "target_date": panel.at[position + horizon, "observation_date"],
        "horizon_weeks": int(horizon),
        "model_id": model_id,
        "feature_set": feature_set,
        "source_type": source_type,
        "score_source": score_source,
        "signal_name": signal_name,
        "hybrid_level": hybrid_level,
        "price_overlap_risk": signal_name == PRICE_OVERLAP_SIGNAL,
        "regime_specific_risk": signal_name
        == "covid19_related_disruption_score",
        "actual_log_change": actual,
        "predicted_log_change": float(prediction),
        "forecast_error": actual - float(prediction),
        "current_scfi": current_scfi,
        "actual_scfi_level": actual_level,
        "predicted_scfi_level": predicted_level,
        "level_error": actual_level - predicted_level,
        "training_rows": int(training_rows),
        "training_feature_end_date": training_feature_end_date,
        "training_label_end_date": training_label_end_date,
        "model_fit_origin_date": model_fit_origin_date,
    }


def benchmark_forecasts(
    panel: pd.DataFrame, horizon: int, minimum_training_rows: int
) -> list[dict[str, Any]]:
    target_column = f"target_scfi_log_change_h{horizon}w"
    rows: list[dict[str, Any]] = []
    for position in eligible_origins(panel, horizon, target_column):
        train = training_positions(panel, position, horizon, target_column)
        if len(train) < minimum_training_rows:
            continue
        predictions = {
            "naive_zero_change": 0.0,
            "historical_mean_change": float(panel.loc[train, target_column].mean()),
        }
        for model_id, prediction in predictions.items():
            rows.append(
                forecast_row(
                    panel=panel,
                    position=position,
                    horizon=horizon,
                    prediction=prediction,
                    model_id=model_id,
                    feature_set="benchmark",
                    source_type="benchmark",
                    score_source="none",
                    signal_name="none",
                    hybrid_level="none",
                    training_rows=len(train),
                    training_feature_end_date=panel.at[train[-1], "observation_date"],
                    training_label_end_date=panel.at[
                        train[-1] + horizon, "observation_date"
                    ],
                    model_fit_origin_date=panel.at[position, "observation_date"],
                )
            )
    return rows


def rolling_ridge_forecasts(
    panel: pd.DataFrame,
    horizon: int,
    model_id: str,
    features: list[str],
    source_type: str,
    score_source: str,
    signal_name: str,
    hybrid_level: str,
    minimum_training_rows: int,
    refit_every: int,
    ridge_alpha: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_column = f"target_scfi_log_change_h{horizon}w"
    positions: list[int] = []
    for position in eligible_origins(panel, horizon, target_column):
        train = training_positions(panel, position, horizon, target_column)
        if len(train) >= minimum_training_rows:
            positions.append(position)

    rows: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    for block_start in range(0, len(positions), refit_every):
        block_positions = positions[block_start : block_start + refit_every]
        fit_position = block_positions[0]
        train = training_positions(panel, fit_position, horizon, target_column)
        estimator = make_ridge_estimator(ridge_alpha)
        estimator.fit(panel.loc[train, features], panel.loc[train, target_column])
        feature_end = panel.at[train[-1], "observation_date"]
        label_end = panel.at[train[-1] + horizon, "observation_date"]
        fit_origin = panel.at[fit_position, "observation_date"]
        if label_end > fit_origin:
            raise AssertionError("Training label endpoint occurs after model-fit origin")
        fits.append(
            {
                "horizon_weeks": horizon,
                "model_id": model_id,
                "source_type": source_type,
                "score_source": score_source,
                "signal_name": signal_name,
                "feature_count": len(features),
                "refit_interval_origins": refit_every,
                "training_rows": len(train),
                "training_feature_end_date": feature_end,
                "training_label_end_date": label_end,
                "model_fit_origin_date": fit_origin,
            }
        )
        predictions = estimator.predict(panel.loc[block_positions, features])
        for position, prediction in zip(block_positions, predictions):
            rows.append(
                forecast_row(
                    panel=panel,
                    position=position,
                    horizon=horizon,
                    prediction=float(prediction),
                    model_id=model_id,
                    feature_set=(
                        "ar_brent"
                        if model_id == "ridge__ar_brent"
                        else "ar_brent_coverage"
                        if model_id == "ridge__ar_brent_coverage"
                        else f"ar_brent_coverage_plus_{score_source}_{signal_name}"
                    ),
                    source_type=source_type,
                    score_source=score_source,
                    signal_name=signal_name,
                    hybrid_level=hybrid_level,
                    training_rows=len(train),
                    training_feature_end_date=feature_end,
                    training_label_end_date=label_end,
                    model_fit_origin_date=fit_origin,
                )
            )
    return rows, fits


def derived_forecast_row(
    template: pd.Series,
    prediction: float,
    model_id: str,
    score_source: str,
    signal_name: str,
    training_rows: int,
    training_feature_end_date: pd.Timestamp,
    training_label_end_date: pd.Timestamp,
    model_fit_origin_date: pd.Timestamp,
) -> dict[str, Any]:
    current_scfi = float(template["current_scfi"])
    actual_level = float(template["actual_scfi_level"])
    predicted_level = current_scfi * math.exp(float(prediction))
    origin_date = template.get("origin_date", template.name)
    return {
        "origin_position": int(template["origin_position"]),
        "origin_date": origin_date,
        "target_date": template["target_date"],
        "horizon_weeks": int(template["horizon_weeks"]),
        "model_id": model_id,
        "feature_set": f"forecast_hybrid_from_four_llms_{signal_name}",
        "source_type": "forecast_hybrid",
        "score_source": score_source,
        "signal_name": signal_name,
        "hybrid_level": "forecast",
        "price_overlap_risk": signal_name == PRICE_OVERLAP_SIGNAL,
        "regime_specific_risk": signal_name
        == "covid19_related_disruption_score",
        "actual_log_change": float(template["actual_log_change"]),
        "predicted_log_change": float(prediction),
        "forecast_error": float(template["actual_log_change"]) - float(prediction),
        "current_scfi": current_scfi,
        "actual_scfi_level": actual_level,
        "predicted_scfi_level": predicted_level,
        "level_error": actual_level - predicted_level,
        "training_rows": int(training_rows),
        "training_feature_end_date": training_feature_end_date,
        "training_label_end_date": training_label_end_date,
        "model_fit_origin_date": model_fit_origin_date,
    }


def build_forecast_hybrids(
    direct_forecasts: pd.DataFrame,
    aliases: list[str],
    minimum_stacking_history: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    llm = direct_forecasts[direct_forecasts["source_type"].eq("llm")].copy()
    if llm.empty:
        raise ValueError("No individual LLM forecasts are available for hybridization")
    llm["origin_date"] = pd.to_datetime(llm["origin_date"])
    llm["target_date"] = pd.to_datetime(llm["target_date"])
    for column in (
        "training_feature_end_date",
        "training_label_end_date",
        "model_fit_origin_date",
    ):
        llm[column] = pd.to_datetime(llm[column])

    hybrid_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for (horizon, signal), group in llm.groupby(
        ["horizon_weeks", "signal_name"], sort=True
    ):
        counts = group.groupby("score_source").size().reindex(aliases)
        if counts.isna().any() or counts.nunique() != 1:
            raise AssertionError(
                f"Individual LLM forecasts are unbalanced for h={horizon}, signal={signal}: "
                f"{counts.to_dict()}"
            )
        prediction = group.pivot(
            index="origin_date", columns="score_source", values="predicted_log_change"
        ).reindex(columns=aliases)
        if prediction.isna().any().any():
            raise AssertionError(
                f"Missing constituent forecasts for h={horizon}, signal={signal}"
            )
        metadata = (
            group.sort_values(["origin_date", "score_source"])
            .groupby("origin_date", as_index=True)
            .first()
            .sort_index()
        )
        consistency = group.groupby("origin_date").agg(
            actual_values=("actual_log_change", "nunique"),
            target_dates=("target_date", "nunique"),
        )
        if consistency[["actual_values", "target_dates"]].gt(1).any().any():
            raise AssertionError("Constituent LLM forecasts disagree on targets")

        for origin_date in prediction.index:
            current_group = group[group["origin_date"].eq(origin_date)]
            template = metadata.loc[origin_date]
            equal_prediction = float(prediction.loc[origin_date].mean())
            equal_label_end = current_group["training_label_end_date"].max()
            equal_fit_origin = current_group["model_fit_origin_date"].max()
            if equal_label_end > equal_fit_origin:
                raise AssertionError("Constituent training labels exceed forecast-hybrid fit time")
            equal_row = derived_forecast_row(
                template=template,
                prediction=equal_prediction,
                model_id=f"forecast_equal_mean__{signal}",
                score_source="forecast_equal_mean",
                signal_name=signal,
                training_rows=int(current_group["training_rows"].min()),
                training_feature_end_date=current_group[
                    "training_feature_end_date"
                ].max(),
                training_label_end_date=equal_label_end,
                model_fit_origin_date=equal_fit_origin,
            )
            equal_row["hybrid_training_rows"] = 0
            equal_row["constituent_training_rows_min"] = int(
                current_group["training_rows"].min()
            )
            hybrid_rows.append(equal_row)

            history_mask = metadata["target_date"].le(origin_date)
            history_origins = metadata.index[history_mask]
            history_origins = history_origins.intersection(prediction.index)
            if len(history_origins) < minimum_stacking_history:
                continue
            x_history = prediction.loc[history_origins, aliases].to_numpy(dtype=float)
            y_history = metadata.loc[
                history_origins, "actual_log_change"
            ].to_numpy(dtype=float)
            finite = np.isfinite(x_history).all(axis=1) & np.isfinite(y_history)
            x_history = x_history[finite]
            y_history = y_history[finite]
            used_origins = history_origins[finite]
            if len(used_origins) < minimum_stacking_history:
                continue
            fallback = False
            try:
                raw_weights, _ = nnls(x_history, y_history)
            except (RuntimeError, ValueError):
                raw_weights = np.ones(len(aliases), dtype=float)
                fallback = True
            weight_sum = float(raw_weights.sum())
            if not np.isfinite(weight_sum) or weight_sum <= 0:
                weights = np.full(len(aliases), 1 / len(aliases), dtype=float)
                fallback = True
            else:
                weights = raw_weights / weight_sum
            stacked_prediction = float(
                np.dot(prediction.loc[origin_date, aliases].to_numpy(dtype=float), weights)
            )
            history_target_end = metadata.loc[used_origins, "target_date"].max()
            if history_target_end > origin_date:
                raise AssertionError("Stacker used an outcome unavailable at forecast origin")
            stack_row = derived_forecast_row(
                template=template,
                prediction=stacked_prediction,
                model_id=f"forecast_nnls_stack__{signal}",
                score_source="forecast_nnls_stack",
                signal_name=signal,
                training_rows=len(used_origins),
                training_feature_end_date=used_origins.max(),
                training_label_end_date=history_target_end,
                model_fit_origin_date=origin_date,
            )
            stack_row["hybrid_training_rows"] = len(used_origins)
            stack_row["constituent_training_rows_min"] = int(
                current_group["training_rows"].min()
            )
            hybrid_rows.append(stack_row)
            for alias, weight in zip(aliases, weights):
                weight_rows.append(
                    {
                        "origin_date": origin_date,
                        "target_date": template["target_date"],
                        "horizon_weeks": int(horizon),
                        "signal_name": signal,
                        "model_alias": alias,
                        "weight": float(weight),
                        "stacking_history_rows": len(used_origins),
                        "stacking_history_first_origin": used_origins.min(),
                        "stacking_history_last_origin": used_origins.max(),
                        "stacking_history_label_end": history_target_end,
                        "equal_weight_fallback": fallback,
                    }
                )
    hybrids = pd.DataFrame(hybrid_rows)
    weights = pd.DataFrame(weight_rows)
    return hybrids, weights


def validate_common_origin_sets(
    forecasts: pd.DataFrame, required_model_ids: list[str]
) -> pd.DataFrame:
    duplicate = forecasts.duplicated(
        ["horizon_weeks", "model_id", "origin_date"], keep=False
    )
    if duplicate.any():
        examples = forecasts.loc[
            duplicate, ["horizon_weeks", "model_id", "origin_date"]
        ].head(10)
        raise AssertionError(
            "Duplicate model/horizon/origin forecasts are not allowed:\n"
            f"{examples.to_string(index=False)}"
        )
    rows: list[dict[str, Any]] = []
    required = set(required_model_ids)
    for horizon, group in forecasts.groupby("horizon_weeks"):
        found = set(group["model_id"].unique())
        missing = sorted(required - found)
        if missing:
            raise AssertionError(f"Horizon {horizon} lacks models: {missing}")
        origin_sets = {
            model_id: set(
                pd.to_datetime(
                    group.loc[group["model_id"].eq(model_id), "origin_date"]
                )
            )
            for model_id in required_model_ids
        }
        reference_id = required_model_ids[0]
        reference = origin_sets[reference_id]
        for model_id, origins in origin_sets.items():
            if origins != reference:
                raise AssertionError(
                    f"Forecast origins differ at h={horizon}: {reference_id} versus {model_id}"
                )
        rows.append(
            {
                "horizon_weeks": int(horizon),
                "models_checked": len(required_model_ids),
                "common_forecast_origins": len(reference),
                "first_origin_date": min(reference),
                "last_origin_date": max(reference),
                "origin_sets_identical": True,
            }
        )
    return pd.DataFrame(rows)


def calculate_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, horizon_frame in forecasts.groupby("horizon_weeks"):
        zero = horizon_frame[
            horizon_frame["model_id"].eq("naive_zero_change")
        ].set_index("origin_date")
        historical = horizon_frame[
            horizon_frame["model_id"].eq("historical_mean_change")
        ].set_index("origin_date")
        coverage = horizon_frame[
            horizon_frame["model_id"].eq("ridge__ar_brent_coverage")
        ].set_index("origin_date")
        for model_id, group in horizon_frame.groupby("model_id"):
            group = group.sort_values("origin_date").set_index("origin_date")
            error = group["actual_log_change"] - group["predicted_log_change"]
            level_error = group["actual_scfi_level"] - group["predicted_scfi_level"]
            aligned_zero = zero.reindex(group.index)
            aligned_historical = historical.reindex(group.index)
            aligned_coverage = coverage.reindex(group.index)
            if aligned_zero["predicted_log_change"].isna().any():
                raise AssertionError(f"Zero-change benchmark is not aligned for {model_id}")
            zero_error = (
                aligned_zero["actual_log_change"]
                - aligned_zero["predicted_log_change"]
            )
            historical_error = (
                aligned_historical["actual_log_change"]
                - aligned_historical["predicted_log_change"]
            )
            coverage_error = (
                aligned_coverage["actual_log_change"]
                - aligned_coverage["predicted_log_change"]
            )
            model_sse = float(np.sum(error**2))
            zero_sse = float(np.sum(zero_error**2))
            historical_sse = float(np.sum(historical_error**2))
            coverage_sse = float(np.sum(coverage_error**2))
            correlation = (
                group[["actual_log_change", "predicted_log_change"]]
                .corr()
                .iloc[0, 1]
            )
            metadata = group.iloc[0]
            rows.append(
                {
                    "horizon_weeks": int(horizon),
                    "model_id": model_id,
                    "feature_set": metadata["feature_set"],
                    "source_type": metadata["source_type"],
                    "score_source": metadata["score_source"],
                    "signal_name": metadata["signal_name"],
                    "hybrid_level": metadata["hybrid_level"],
                    "price_overlap_risk": bool(metadata["price_overlap_risk"]),
                    "regime_specific_risk": bool(metadata["regime_specific_risk"]),
                    "n_forecasts": len(group),
                    "first_origin_date": group.index.min(),
                    "last_origin_date": group.index.max(),
                    "mae_log_change": float(np.mean(np.abs(error))),
                    "rmse_log_change": float(np.sqrt(np.mean(error**2))),
                    "bias_log_change": float(
                        np.mean(group["predicted_log_change"] - group["actual_log_change"])
                    ),
                    "directional_accuracy": float(
                        np.mean(
                            np.sign(group["actual_log_change"])
                            == np.sign(group["predicted_log_change"])
                        )
                    ),
                    "forecast_actual_correlation": correlation,
                    "mae_scfi_level": float(np.mean(np.abs(level_error))),
                    "rmse_scfi_level": float(np.sqrt(np.mean(level_error**2))),
                    "oos_r2_vs_zero_change": (
                        math.nan if zero_sse == 0 else 1 - model_sse / zero_sse
                    ),
                    "oos_r2_vs_historical_mean": (
                        math.nan
                        if historical_sse == 0
                        else 1 - model_sse / historical_sse
                    ),
                    "oos_r2_vs_ar_brent_coverage": (
                        math.nan
                        if coverage_sse == 0
                        else 1 - model_sse / coverage_sse
                    ),
                    "relative_rmse_vs_ar_brent_coverage": (
                        math.nan
                        if coverage_sse == 0
                        else math.sqrt(model_sse / coverage_sse)
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon_weeks", "rmse_log_change"])


def hac_mean_test(values: np.ndarray, maximum_lag: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return math.nan, math.nan, math.nan
    centered = values - values.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_variance = gamma0
    for lag in range(1, min(maximum_lag, n - 1) + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1 - lag / (maximum_lag + 1)
        long_run_variance += 2 * weight * covariance
    if long_run_variance <= 0:
        return float(values.mean()), math.nan, math.nan
    standard_error = math.sqrt(long_run_variance / n)
    statistic = float(values.mean() / standard_error)
    return float(values.mean()), statistic, standard_error


def adjust_pvalues(values: Iterable[float], method: str) -> np.ndarray:
    pvalues = np.asarray(list(values), dtype=float)
    result = np.full(len(pvalues), np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(pvalues))
    if len(finite_positions) == 0:
        return result
    finite = np.clip(pvalues[finite_positions], 0, 1)
    order = np.argsort(finite)
    ordered = finite[order]
    m = len(ordered)
    if method == "holm":
        adjusted_ordered = np.maximum.accumulate((m - np.arange(m)) * ordered)
    elif method == "fdr_bh":
        adjusted_ordered = np.minimum.accumulate(
            (ordered * m / (np.arange(m) + 1))[::-1]
        )[::-1]
    else:
        raise ValueError(f"Unknown p-value adjustment method: {method}")
    adjusted_ordered = np.clip(adjusted_ordered, 0, 1)
    restored = np.empty(m, dtype=float)
    restored[order] = adjusted_ordered
    result[finite_positions] = restored
    return result


def loss_test_rows(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
    horizon: int,
    baseline_model: str,
    challenger_model: str,
    source_type: str,
    score_source: str,
    signal_name: str,
    family_root: str,
    include_clark_west: bool,
) -> list[dict[str, Any]]:
    left = baseline.set_index("origin_date")[["actual_log_change", "predicted_log_change"]]
    right = challenger.set_index("origin_date")[["predicted_log_change"]]
    pair = left.join(right, how="inner", lsuffix="_baseline", rsuffix="_challenger")
    if pair.empty:
        return []
    actual = pair["actual_log_change"]
    baseline_prediction = pair["predicted_log_change_baseline"]
    challenger_prediction = pair["predicted_log_change_challenger"]
    baseline_error = actual - baseline_prediction
    challenger_error = actual - challenger_prediction
    comparisons: dict[str, tuple[pd.Series, str]] = {
        "diebold_mariano_squared": (
            baseline_error**2 - challenger_error**2,
            "two_sided",
        ),
        "diebold_mariano_absolute": (
            baseline_error.abs() - challenger_error.abs(),
            "two_sided",
        ),
    }
    if include_clark_west:
        comparisons["clark_west_nested_squared"] = (
            baseline_error**2
            - (
                challenger_error**2
                - (baseline_prediction - challenger_prediction) ** 2
            ),
            "one_sided_positive",
        )
    rows: list[dict[str, Any]] = []
    for test_name, (differential, alternative) in comparisons.items():
        mean_difference, statistic, standard_error = hac_mean_test(
            differential.to_numpy(), max(int(horizon) - 1, 0)
        )
        if pd.isna(statistic):
            p_value = math.nan
        elif alternative == "one_sided_positive":
            p_value = float(1 - norm.cdf(statistic))
        else:
            p_value = float(2 * (1 - norm.cdf(abs(statistic))))
        if family_root.startswith("primary_") and test_name != "diebold_mariano_squared":
            family_id = (
                f"sensitivity_{family_root.removeprefix('primary_')}__{test_name}"
            )
        else:
            family_id = f"{family_root}__{test_name}"
        rows.append(
            {
                "horizon_weeks": int(horizon),
                "baseline_model": baseline_model,
                "challenger_model": challenger_model,
                "source_type": source_type,
                "score_source": score_source,
                "signal_name": signal_name,
                "test": test_name,
                "n": len(pair),
                "hac_lag": max(int(horizon) - 1, 0),
                "mean_loss_improvement": mean_difference,
                "standard_error": standard_error,
                "statistic": statistic,
                "p_value": p_value,
                "alternative": alternative,
                "positive_favors": "challenger",
                "family_id": family_id,
                "price_overlap_risk": signal_name == PRICE_OVERLAP_SIGNAL,
            }
        )
    return rows


def forecast_comparison_tests(
    forecasts: pd.DataFrame, aliases: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, horizon_frame in forecasts.groupby("horizon_weeks"):
        coverage = horizon_frame[
            horizon_frame["model_id"].eq("ridge__ar_brent_coverage")
        ]
        if coverage.empty:
            raise ValueError(f"Missing coverage baseline at horizon {horizon}")
        challengers = horizon_frame[
            ~horizon_frame["model_id"].isin(BASELINE_MODEL_IDS)
        ]
        for challenger_id, challenger in challengers.groupby("model_id"):
            metadata = challenger.iloc[0]
            signal = str(metadata["signal_name"])
            source_type = str(metadata["source_type"])
            if signal in PRIMARY_SIGNALS:
                family_root = (
                    "primary_llm_vs_baseline"
                    if source_type == "llm"
                    else "primary_hybrid_vs_baseline"
                )
            else:
                family_root = "exploratory_dimension_vs_baseline"
            rows.extend(
                loss_test_rows(
                    baseline=coverage,
                    challenger=challenger,
                    horizon=int(horizon),
                    baseline_model="ridge__ar_brent_coverage",
                    challenger_model=challenger_id,
                    source_type=source_type,
                    score_source=str(metadata["score_source"]),
                    signal_name=signal,
                    family_root=family_root,
                    include_clark_west=source_type != "forecast_hybrid",
                )
            )

        individual = horizon_frame[horizon_frame["source_type"].eq("llm")]
        for signal, signal_frame in individual.groupby("signal_name"):
            family_root = (
                "primary_llm_pairwise"
                if signal in PRIMARY_SIGNALS
                else "exploratory_dimension_llm_pairwise"
            )
            by_alias = {
                alias: signal_frame[signal_frame["score_source"].eq(alias)]
                for alias in aliases
            }
            for left_alias, right_alias in combinations(aliases, 2):
                left = by_alias[left_alias]
                right = by_alias[right_alias]
                if left.empty or right.empty:
                    raise AssertionError(
                        f"Missing LLM pair for h={horizon}, signal={signal}"
                    )
                rows.extend(
                    loss_test_rows(
                        baseline=left,
                        challenger=right,
                        horizon=int(horizon),
                        baseline_model=str(left["model_id"].iloc[0]),
                        challenger_model=str(right["model_id"].iloc[0]),
                        source_type="llm_pairwise",
                        score_source=f"{left_alias}_vs_{right_alias}",
                        signal_name=str(signal),
                        family_root=family_root,
                        include_clark_west=False,
                    )
                )

    tests = pd.DataFrame(rows)
    tests["multiple_testing_method"] = np.where(
        tests["family_id"].str.startswith(("primary_", "sensitivity_")),
        "holm",
        "fdr_bh",
    )
    tests["p_value_adjusted"] = np.nan
    for family_id, positions in tests.groupby("family_id").groups.items():
        method = str(tests.loc[positions, "multiple_testing_method"].iloc[0])
        tests.loc[positions, "p_value_adjusted"] = adjust_pvalues(
            tests.loc[positions, "p_value"], method
        )
    tests["significant_raw_0_05"] = tests["p_value"].lt(0.05)
    tests["significant_adjusted_0_05"] = tests["p_value_adjusted"].lt(0.05)
    return tests.sort_values(
        ["family_id", "horizon_weeks", "signal_name", "challenger_model"]
    ).reset_index(drop=True)


def build_llm_rankings(
    metrics: pd.DataFrame, tests: pd.DataFrame, aliases: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = metrics[
        metrics["source_type"].eq("llm")
        & metrics["signal_name"].isin(PRIMARY_SIGNALS)
    ].copy()
    expected_cells = len(aliases) * len(PRIMARY_SIGNALS) * len(HORIZONS)
    if len(cells) != expected_cells:
        raise AssertionError(
            f"Expected {expected_cells} primary LLM ranking cells; found {len(cells)}"
        )
    ranking_keys = ["score_source", "signal_name", "horizon_weeks"]
    if cells.duplicated(ranking_keys).any():
        raise AssertionError("Primary LLM ranking cells contain duplicate Cartesian keys")
    found_keys = set(cells[ranking_keys].itertuples(index=False, name=None))
    expected_keys = set(product(aliases, PRIMARY_SIGNALS, HORIZONS))
    if found_keys != expected_keys:
        raise AssertionError(
            "Primary LLM ranking cells do not form the exact model/signal/horizon product; "
            f"missing={sorted(expected_keys - found_keys)}, "
            f"extra={sorted(found_keys - expected_keys)}"
        )
    cells["rmse_rank_within_horizon_signal"] = cells.groupby(
        ["horizon_weeks", "signal_name"]
    )["rmse_log_change"].rank(method="average", ascending=True)
    cells["mae_rank_within_horizon_signal"] = cells.groupby(
        ["horizon_weeks", "signal_name"]
    )["mae_log_change"].rank(method="average", ascending=True)
    cells["skill_rank_within_horizon_signal"] = cells.groupby(
        ["horizon_weeks", "signal_name"]
    )["oos_r2_vs_ar_brent_coverage"].rank(method="average", ascending=False)

    ranking = cells.groupby("score_source", as_index=False).agg(
        primary_comparison_cells=("model_id", "size"),
        mean_rmse_rank=("rmse_rank_within_horizon_signal", "mean"),
        mean_mae_rank=("mae_rank_within_horizon_signal", "mean"),
        mean_skill_rank=("skill_rank_within_horizon_signal", "mean"),
        mean_rmse_log_change=("rmse_log_change", "mean"),
        mean_relative_rmse_vs_coverage=(
            "relative_rmse_vs_ar_brent_coverage",
            "mean",
        ),
        mean_oos_r2_vs_coverage=("oos_r2_vs_ar_brent_coverage", "mean"),
        mean_directional_accuracy=("directional_accuracy", "mean"),
        rmse_cell_wins=(
            "rmse_rank_within_horizon_signal",
            lambda values: int(np.isclose(values, 1).sum()),
        ),
    )

    ranking["adjusted_dm_squared_pairwise_wins"] = 0
    ranking["adjusted_dm_squared_pairwise_losses"] = 0
    pairwise = tests[
        tests["family_id"].eq(
            "primary_llm_pairwise__diebold_mariano_squared"
        )
        & tests["significant_adjusted_0_05"]
    ]
    for _, row in pairwise.iterrows():
        baseline_alias = next(
            alias for alias in aliases if f"__{alias}__" in row["baseline_model"]
        )
        challenger_alias = next(
            alias for alias in aliases if f"__{alias}__" in row["challenger_model"]
        )
        if row["mean_loss_improvement"] > 0:
            winner, loser = challenger_alias, baseline_alias
        else:
            winner, loser = baseline_alias, challenger_alias
        ranking.loc[
            ranking["score_source"].eq(winner), "adjusted_dm_squared_pairwise_wins"
        ] += 1
        ranking.loc[
            ranking["score_source"].eq(loser), "adjusted_dm_squared_pairwise_losses"
        ] += 1

    ranking = ranking.sort_values(
        [
            "mean_rmse_rank",
            "mean_relative_rmse_vs_coverage",
            "mean_mae_rank",
            "score_source",
        ]
    ).reset_index(drop=True)
    ranking["predictive_point_rank"] = np.arange(1, len(ranking) + 1)
    ranking["ranking_scope"] = (
        "equal-weight overall-max and dimension-mean7 RMSE ranks across four horizons"
    )
    return ranking, cells.sort_values(
        ["horizon_weeks", "signal_name", "rmse_rank_within_horizon_signal"]
    )


def build_predictor_summaries(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_sample_sources = metrics[
        metrics["source_type"].isin({"llm", "score_hybrid", "forecast_hybrid"})
        & metrics["signal_name"].isin(PRIMARY_SIGNALS)
        & ~metrics["score_source"].eq("forecast_nnls_stack")
    ].copy()
    primary = full_sample_sources.groupby(
        ["source_type", "score_source", "hybrid_level"], as_index=False
    ).agg(
        comparison_cells=("model_id", "size"),
        mean_rmse_log_change=("rmse_log_change", "mean"),
        mean_mae_log_change=("mae_log_change", "mean"),
        mean_relative_rmse_vs_coverage=(
            "relative_rmse_vs_ar_brent_coverage",
            "mean",
        ),
        mean_oos_r2_vs_coverage=("oos_r2_vs_ar_brent_coverage", "mean"),
        mean_directional_accuracy=("directional_accuracy", "mean"),
    )
    primary = primary.sort_values(
        ["mean_relative_rmse_vs_coverage", "mean_rmse_log_change"]
    ).reset_index(drop=True)
    primary["full_sample_primary_point_rank"] = np.arange(1, len(primary) + 1)

    dimensions = metrics[
        metrics["source_type"].isin({"llm", "score_hybrid", "forecast_hybrid"})
        & metrics["signal_name"].isin(DIMENSION_COLUMNS)
    ].copy()
    dimension_summary = dimensions.groupby(
        ["source_type", "score_source", "signal_name", "hybrid_level"],
        as_index=False,
    ).agg(
        horizons=("horizon_weeks", "nunique"),
        mean_rmse_log_change=("rmse_log_change", "mean"),
        mean_mae_log_change=("mae_log_change", "mean"),
        mean_relative_rmse_vs_coverage=(
            "relative_rmse_vs_ar_brent_coverage",
            "mean",
        ),
        mean_oos_r2_vs_coverage=("oos_r2_vs_ar_brent_coverage", "mean"),
        mean_directional_accuracy=("directional_accuracy", "mean"),
    )
    dimension_summary["price_overlap_risk"] = dimension_summary["signal_name"].eq(
        PRICE_OVERLAP_SIGNAL
    )
    dimension_summary["exploratory"] = True
    return primary, dimension_summary.sort_values(
        ["signal_name", "mean_relative_rmse_vs_coverage"]
    ).reset_index(drop=True)


def build_common_article_export(
    article: pd.DataFrame, aliases: list[str]
) -> pd.DataFrame:
    common = article.loc[article["llm_common_complete"]].copy()
    base_columns = [
        "article_id",
        "publication_date",
        "novelty_weight",
        "source_duplicate_weight",
        "event_id_weekly",
        "event_id_monthly",
        *[
            f"{signal}__{alias}"
            for alias in aliases
            for signal in (PRIMARY_SCORE_COLUMN, *DIMENSION_COLUMNS)
        ],
    ]
    output = common[base_columns].copy()
    derived: dict[str, pd.Series] = {}
    for source in [*aliases, *SCORE_HYBRID_SOURCES]:
        for signal in SIGNALS:
            derived[f"article_signal__{source}__{signal}"] = article_signal_values(
                common, aliases, source, signal
            ).to_numpy()
    return pd.concat([output.reset_index(drop=True), pd.DataFrame(derived)], axis=1)


def build_common_cohort_audit(
    article_report: dict[str, Any], aliases: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    common_n = int(article_report["common_complete_articles"])
    total = int(article_report["rows"])
    for alias in aliases:
        availability = article_report["model_availability"][alias]
        rows.append(
            {
                "score_source": alias,
                "source_type": "llm",
                "articles_with_all_eight_source_scores": int(
                    availability["all_eight_scores"]
                ),
                "articles_selected_for_comparison": common_n,
                "same_common_cohort_enforced": True,
                "total_article_universe": total,
                "articles_excluded_by_common_cohort_rule": total - common_n,
            }
        )
    for source in SCORE_HYBRID_SOURCES:
        rows.append(
            {
                "score_source": source,
                "source_type": "score_hybrid",
                "articles_with_all_eight_source_scores": common_n,
                "articles_selected_for_comparison": common_n,
                "same_common_cohort_enforced": True,
                "total_article_universe": total,
                "articles_excluded_by_common_cohort_rule": total - common_n,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.minimum_training_rows < 52:
        raise ValueError("minimum-training-rows must be at least 52")
    if args.refit_every < 1:
        raise ValueError("refit-every must be positive")
    if args.ridge_alpha <= 0:
        raise ValueError("ridge-alpha must be positive")
    if args.minimum_stacking_history < 20:
        raise ValueError("minimum-stacking-history must be at least 20")
    if args.maximum_feature_age_days < 0:
        raise ValueError("maximum-feature-age-days cannot be negative")
    if args.expected_common_articles < 1:
        raise ValueError("expected-common-articles must be positive")
    np.random.seed(args.seed)

    logger, log_path = configure_logging(args.log_dir.resolve(), args.version)
    logger.info("Starting common-cohort LLM-to-SCFI comparison version=%s", args.version)
    logger.info("Arguments: %s", vars(args))

    article_path = require_file(args.article_scores, "Article scores")
    panel_path = require_file(args.panel, "SCFI panel")
    model_config_path = require_file(args.model_config, "Model registry")
    aliases = load_model_aliases(model_config_path)
    logger.info("Model aliases in registry order: %s", aliases)

    article, article_report = load_article_scores(
        article_path, aliases, args.expected_common_articles
    )
    common_articles = build_common_article_export(article, aliases)
    cohort_audit = build_common_cohort_audit(article_report, aliases)
    logger.info(
        "Common article cohort=%s of %s; excluded=%s; dates=%s to %s",
        f"{article_report['common_complete_articles']:,}",
        f"{article_report['rows']:,}",
        f"{article_report['excluded_from_common_cohort']:,}",
        article_report["common_first_date"],
        article_report["common_last_date"],
    )
    for alias in aliases:
        availability = article_report["model_availability"][alias]
        logger.info(
            "Availability before common restriction model=%s all_eight=%s overall=%s; "
            "selected_common=%s",
            alias,
            availability["all_eight_scores"],
            availability["overall_score"],
            article_report["common_complete_articles"],
        )

    weekly_long, weekly_coverage = build_common_weekly_signals(article, aliases)
    weekly_count_total = int(
        round(float(weekly_coverage["llm_common_article_count"].sum()))
    )
    if weekly_count_total != args.expected_common_articles:
        raise AssertionError(
            "Weekly common-cohort article counts do not conserve the frozen article cohort: "
            f"expected {args.expected_common_articles:,}, found {weekly_count_total:,}"
        )
    weekly_wide = pivot_common_weekly_signals(weekly_long, aliases)
    signal_differences = build_signal_difference_summary(article, weekly_long, aliases)
    expected_weekly_rows = (
        weekly_long["week_ending"].nunique()
        * (len(aliases) + len(SCORE_HYBRID_SOURCES))
        * len(SIGNALS)
    )
    if len(weekly_long) != expected_weekly_rows:
        raise AssertionError(
            f"Expected {expected_weekly_rows} complete weekly signal rows; found {len(weekly_long)}"
        )
    selected_counts = cohort_audit["articles_selected_for_comparison"]
    if selected_counts.nunique() != 1:
        raise AssertionError("Score sources do not use the same article cohort")
    logger.info(
        "Built %s common-cohort weekly rows across %s weeks, %s sources, %s signals",
        f"{len(weekly_long):,}",
        weekly_long["week_ending"].nunique(),
        weekly_long["score_source"].nunique(),
        weekly_long["signal_name"].nunique(),
    )
    for _, row in signal_differences[
        signal_differences["level"].eq("article")
    ].iterrows():
        logger.info(
            "Score difference signal=%s pair=%s-minus-%s n=%s mean_signed=%.6f "
            "mean_absolute=%.6f rmse=%.6f spearman=%.6f",
            row["signal_name"],
            row["model_left"],
            row["model_right"],
            row["n_complete"],
            row["mean_signed_difference"],
            row["mean_absolute_difference"],
            row["rmse_difference"],
            row["spearman_correlation"],
        )

    base_panel = load_base_panel(panel_path)
    panel, alignment_report = build_aligned_panel(
        base_panel,
        weekly_wide,
        aliases,
        args.maximum_feature_age_days,
    )
    logger.info(
        "Aligned common signals: rows=%s matched=%s complete=%s age_counts=%s",
        alignment_report["rows"],
        alignment_report["matched_common_signal_weeks"],
        alignment_report["comparison_complete_rows"],
        alignment_report["age_days_counts"],
    )

    configurations = build_direct_configurations(aliases)
    feature_sets = build_feature_sets(panel, configurations)
    logger.info(
        "Forecast design: direct configurations=%s baseline_features=%s "
        "challenger_features=%s",
        len(configurations),
        len(feature_sets["ridge__ar_brent_coverage"]),
        sorted(
            {
                len(feature_sets[configuration.model_id])
                for configuration in configurations
            }
        ),
    )

    forecast_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        logger.info("Running common-origin forecasts horizon=%sw", horizon)
        forecast_rows.extend(
            benchmark_forecasts(panel, horizon, args.minimum_training_rows)
        )
        for baseline_id in ("ridge__ar_brent", "ridge__ar_brent_coverage"):
            rows, fits = rolling_ridge_forecasts(
                panel=panel,
                horizon=horizon,
                model_id=baseline_id,
                features=feature_sets[baseline_id],
                source_type="baseline",
                score_source="none",
                signal_name="none",
                hybrid_level="none",
                minimum_training_rows=args.minimum_training_rows,
                refit_every=args.refit_every,
                ridge_alpha=args.ridge_alpha,
            )
            forecast_rows.extend(rows)
            fit_rows.extend(fits)
            logger.info(
                "Completed h=%s model=%s forecasts=%s fits=%s",
                horizon,
                baseline_id,
                len(rows),
                len(fits),
            )
        for configuration in configurations:
            rows, fits = rolling_ridge_forecasts(
                panel=panel,
                horizon=horizon,
                model_id=configuration.model_id,
                features=feature_sets[configuration.model_id],
                source_type=configuration.source_type,
                score_source=configuration.score_source,
                signal_name=configuration.signal_name,
                hybrid_level=configuration.hybrid_level,
                minimum_training_rows=args.minimum_training_rows,
                refit_every=args.refit_every,
                ridge_alpha=args.ridge_alpha,
            )
            forecast_rows.extend(rows)
            fit_rows.extend(fits)
            logger.info(
                "Completed h=%s source=%s signal=%s forecasts=%s fits=%s",
                horizon,
                configuration.score_source,
                configuration.signal_name,
                len(rows),
                len(fits),
            )

    direct_forecasts = pd.DataFrame(forecast_rows)
    fits = pd.DataFrame(fit_rows)
    if direct_forecasts.empty:
        raise ValueError("No direct forecasts were produced")
    date_columns = (
        "origin_date",
        "target_date",
        "training_feature_end_date",
        "training_label_end_date",
        "model_fit_origin_date",
    )
    for column in date_columns:
        direct_forecasts[column] = pd.to_datetime(direct_forecasts[column])
    for column in (
        "training_feature_end_date",
        "training_label_end_date",
        "model_fit_origin_date",
    ):
        fits[column] = pd.to_datetime(fits[column])
    required_direct_ids = [
        "naive_zero_change",
        "historical_mean_change",
        "ridge__ar_brent",
        "ridge__ar_brent_coverage",
        *[configuration.model_id for configuration in configurations],
    ]
    origin_audit = validate_common_origin_sets(direct_forecasts, required_direct_ids)
    logger.info("Identical direct origin counts: %s", origin_audit.to_dict(orient="records"))

    hybrid_forecasts, hybrid_weights = build_forecast_hybrids(
        direct_forecasts, aliases, args.minimum_stacking_history
    )
    if hybrid_forecasts.empty:
        raise ValueError("No forecast-level hybrids were produced")
    for column in date_columns:
        hybrid_forecasts[column] = pd.to_datetime(hybrid_forecasts[column])
    if not hybrid_weights.empty:
        weight_sums = hybrid_weights.groupby(
            ["horizon_weeks", "signal_name", "origin_date"]
        )["weight"].sum()
        if not np.allclose(weight_sums.to_numpy(), 1.0, atol=1e-10):
            raise AssertionError("Past-only stacking weights do not sum to one")
        if (
            pd.to_datetime(hybrid_weights["stacking_history_label_end"])
            > pd.to_datetime(hybrid_weights["origin_date"])
        ).any():
            raise AssertionError("Past-only stacker used a future realized target")

    forecasts = pd.concat(
        [direct_forecasts, hybrid_forecasts], ignore_index=True, sort=False
    ).sort_values(["horizon_weeks", "origin_date", "model_id"])
    if (
        pd.to_datetime(forecasts["training_label_end_date"])
        > pd.to_datetime(forecasts["model_fit_origin_date"])
    ).any():
        raise AssertionError("At least one forecast used a not-yet-observable label")
    metrics = calculate_metrics(forecasts)
    statistical_tests = forecast_comparison_tests(forecasts, aliases)
    llm_ranking, llm_ranking_cells = build_llm_rankings(
        metrics, statistical_tests, aliases
    )
    primary_predictor_ranking, dimension_summary = build_predictor_summaries(metrics)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_output = args.panel_output.resolve()
    panel_output.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "common_article_cohort": output_dir
        / f"{args.version}_common_article_cohort.csv",
        "common_cohort_audit": output_dir / f"{args.version}_common_cohort_audit.csv",
        "weekly_common_signals": output_dir
        / f"{args.version}_weekly_common_signals.csv",
        "weekly_common_coverage": output_dir
        / f"{args.version}_weekly_common_coverage.csv",
        "signal_differences": output_dir
        / f"{args.version}_signal_difference_summary.csv",
        "aligned_panel": panel_output,
        "forecasts": output_dir / f"{args.version}_forecast_predictions.csv",
        "metrics": output_dir / f"{args.version}_forecast_metrics.csv",
        "statistical_tests": output_dir
        / f"{args.version}_forecast_statistical_tests.csv",
        "fit_audit": output_dir / f"{args.version}_model_fit_audit.csv",
        "origin_audit": output_dir / f"{args.version}_common_origin_audit.csv",
        "hybrid_weights": output_dir / f"{args.version}_forecast_hybrid_weights.csv",
        "llm_ranking": output_dir / f"{args.version}_llm_predictive_ranking.csv",
        "llm_ranking_cells": output_dir
        / f"{args.version}_llm_ranking_by_horizon_signal.csv",
        "primary_predictor_ranking": output_dir
        / f"{args.version}_primary_predictor_ranking.csv",
        "dimension_summary": output_dir
        / f"{args.version}_dimension_predictive_summary.csv",
        "feature_sets": output_dir / f"{args.version}_feature_sets.json",
        "manifest": output_dir / f"{args.version}_manifest.json",
    }
    common_articles.to_csv(paths["common_article_cohort"], index=False)
    cohort_audit.to_csv(paths["common_cohort_audit"], index=False)
    weekly_long.to_csv(paths["weekly_common_signals"], index=False)
    weekly_coverage.to_csv(paths["weekly_common_coverage"], index=False)
    signal_differences.to_csv(paths["signal_differences"], index=False)
    panel.to_csv(paths["aligned_panel"], index=False)
    forecasts.to_csv(paths["forecasts"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    statistical_tests.to_csv(paths["statistical_tests"], index=False)
    fits.to_csv(paths["fit_audit"], index=False)
    origin_audit.to_csv(paths["origin_audit"], index=False)
    hybrid_weights.to_csv(paths["hybrid_weights"], index=False)
    llm_ranking.to_csv(paths["llm_ranking"], index=False)
    llm_ranking_cells.to_csv(paths["llm_ranking_cells"], index=False)
    primary_predictor_ranking.to_csv(paths["primary_predictor_ranking"], index=False)
    dimension_summary.to_csv(paths["dimension_summary"], index=False)
    paths["feature_sets"].write_text(
        json.dumps(feature_sets, indent=2), encoding="utf-8"
    )
    output_frames = {
        "common_article_cohort": common_articles,
        "common_cohort_audit": cohort_audit,
        "weekly_common_signals": weekly_long,
        "weekly_common_coverage": weekly_coverage,
        "signal_differences": signal_differences,
        "aligned_panel": panel,
        "forecasts": forecasts,
        "metrics": metrics,
        "statistical_tests": statistical_tests,
        "fit_audit": fits,
        "origin_audit": origin_audit,
        "hybrid_weights": hybrid_weights,
        "llm_ranking": llm_ranking,
        "llm_ranking_cells": llm_ranking_cells,
        "primary_predictor_ranking": primary_predictor_ranking,
        "dimension_summary": dimension_summary,
    }
    output_sha256 = {
        key: sha256_file(path)
        for key, path in paths.items()
        if key != "manifest" and path.is_file()
    }
    output_row_counts = {
        key: int(len(frame)) for key, frame in output_frames.items()
    }

    best_llm = llm_ranking.iloc[0].to_dict()
    significant_primary = statistical_tests[
        statistical_tests["family_id"].str.startswith("primary_")
        & statistical_tests["significant_adjusted_0_05"]
        & statistical_tests["mean_loss_improvement"].gt(0)
    ]
    manifest = {
        "version": args.version,
        "status": "provisional_pending_human_validation",
        "inputs": {
            "article_scores": {
                "path": str(article_path),
                "sha256": sha256_file(article_path),
                "rows": article_report["rows"],
            },
            "scfi_panel_non_msdi_fields_only": {
                "path": str(panel_path),
                "sha256": sha256_file(panel_path),
                "rows": len(base_panel),
            },
            "model_registry": {
                "path": str(model_config_path),
                "sha256": sha256_file(model_config_path),
            },
            "code": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "common_article_cohort": article_report,
        "design": {
            "model_aliases": aliases,
            "common_article_rule": (
                "publication date plus overall and all seven dimension scores "
                "available for every one of the four LLMs"
            ),
            "common_cohort_is_only_score_input": True,
            "expected_common_articles": args.expected_common_articles,
            "weekly_common_article_count_sum": weekly_count_total,
            "broader_consensus_index_used_as_predictor": False,
            "model_specific_unequal_coverage_indices_used": False,
            "score_scale": [0, 4],
            "weekly_frequency": "W-FRI, right-labeled and right-closed",
            "novelty_weighting": NOVELTY_WEIGHTING_PROVENANCE,
            "media_correction": "novelty-weighted sum / log1p(common article count)",
            "signals": list(SIGNALS),
            "primary_signals": list(PRIMARY_SIGNALS),
            "score_hybrids": {
                "score_median": (
                    "article-level median across four LLMs; dimension_mean7 is "
                    "equal mean of seven dimension-specific medians"
                ),
                "score_mean": (
                    "article-level equal mean across four LLMs; dimension_mean7 "
                    "is equal mean of seven dimension-specific means"
                ),
            },
            "forecast_hybrids": {
                "equal_mean": "equal mean of the four individual LLM forecasts",
                "past_only_nnls": (
                    "nonnegative weights estimated only from prior OOS forecasts "
                    "whose target dates are observed by the current origin, normalized to sum one"
                ),
                "minimum_stacking_history": args.minimum_stacking_history,
            },
            "target": "forward SCFI log change with reconstructed future SCFI level",
            "horizons_weeks": list(HORIZONS),
            "forecast_evaluation": "expanding-window rolling-origin",
            "minimum_training_rows": args.minimum_training_rows,
            "refit_every_origins": args.refit_every,
            "training_label_purge": "training row s allowed only when s + horizon <= origin",
            "preprocessing_fit_inside_training_window": True,
            "random_train_test_split": False,
            "same_algorithm_and_controls_for_all_score_sources": True,
            "algorithm": "Ridge",
            "ridge_alpha": args.ridge_alpha,
            "common_forecast_origins_enforced": True,
            "alignment": "backward Friday common signal to Monday SCFI",
            "maximum_feature_age_days": args.maximum_feature_age_days,
            "baseline_for_incremental_tests": "AR + Brent + common news coverage",
            "primary_multiple_testing": (
                "DM squared loss is the sole confirmatory statistic, Holm-adjusted "
                "within named primary families"
            ),
            "sensitivity_forecast_tests": (
                "DM absolute loss and Clark-West are Holm-adjusted sensitivity families"
            ),
            "exploratory_dimension_multiple_testing": "Benjamini-Hochberg FDR within named test families",
            "primary_llm_rank": (
                "equal-weight mean RMSE rank over overall_disruption_max and "
                "dimension_mean7 across 1, 2, 4, 8, and 12 weeks"
            ),
            "seed": args.seed,
        },
        "alignment_audit": alignment_report,
        "origin_audit": origin_audit.to_dict(orient="records"),
        "best_llm_point_rank": best_llm,
        "adjusted_significant_positive_primary_tests": int(len(significant_primary)),
        "interpretation_limits": {
            "best_predictive_llm_is_not_topsis_rank": True,
            "best_predictive_llm_is_not_human_validity_rank": True,
            "causal_claim": False,
            "freight_cost_dimension_price_overlap_risk": True,
            "dimension_paths_are_exploratory": True,
            "past_only_stacker_has_a_shorter_native_evaluation_sample": True,
            "point_ranking_does_not_by_itself_establish_statistical_superiority": True,
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "output_sha256": output_sha256,
        "output_row_counts": output_row_counts,
        "log": str(log_path),
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )

    logger.info(
        "Forecast rows=%s direct=%s hybrids=%s fit events=%s statistical tests=%s",
        f"{len(forecasts):,}",
        f"{len(direct_forecasts):,}",
        f"{len(hybrid_forecasts):,}",
        f"{len(fits):,}",
        f"{len(statistical_tests):,}",
    )
    logger.info(
        "Point-ranked LLM=%s rank=%s mean_RMSE_rank=%.3f mean_relative_RMSE=%.6f "
        "adjusted_pairwise_wins=%s",
        best_llm["score_source"],
        best_llm["predictive_point_rank"],
        best_llm["mean_rmse_rank"],
        best_llm["mean_relative_rmse_vs_coverage"],
        best_llm["adjusted_dm_squared_pairwise_wins"],
    )
    logger.info(
        "Adjusted significant positive primary tests=%s; point rank alone is not superiority",
        len(significant_primary),
    )
    logger.info("LLM ranking: %s", paths["llm_ranking"])
    logger.info("Forecast metrics: %s", paths["metrics"])
    logger.info("Statistical tests: %s", paths["statistical_tests"])
    logger.info("Common-cohort LLM-to-SCFI comparison completed successfully")


if __name__ == "__main__":
    main()
