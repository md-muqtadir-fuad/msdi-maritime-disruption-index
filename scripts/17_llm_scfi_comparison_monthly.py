"""Run a monthly robustness version of the common-cohort LLM forecast comparison.

The weekly 07b analysis is primary.  This script consumes only its frozen,
leakage-safe Monday panel, aggregates raw weekly common-cohort signals to
complete calendar months, and recomputes all monthly lags and SCFI targets.
Friday news is therefore assigned to the month of the following Monday, which
is the date on which it enters the forecasting information set.

Partial months are never treated as complete or filled with zero.  Monthly
results are slower-dynamics robustness evidence, not structural causal claims.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import platform
import sys
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEEKLY_SUPPORT_SCRIPT = PROJECT_ROOT / "scripts/16_llm_scfi_comparison_weekly.py"


def load_weekly_support() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_llm_scfi_weekly_support", WEEKLY_SUPPORT_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load weekly support script: {WEEKLY_SUPPORT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUPPORT = load_weekly_support()

DEFAULT_WEEKLY_PANEL = (
    PROJECT_ROOT
    / "data/06_model_ready/prediction/llm_specific_scfi_weekly_panel.csv"
)
DEFAULT_WEEKLY_MANIFEST = (
    PROJECT_ROOT
    / "outputs/llm_scfi_comparison/llm_scfi_v001_manifest.json"
)
DEFAULT_PANEL_OUTPUT = (
    PROJECT_ROOT
    / "data/06_model_ready/prediction/llm_specific_scfi_monthly_panel.csv"
)
DEFAULT_MODEL_CONFIG = SUPPORT.DEFAULT_MODEL_CONFIG
DEFAULT_OUTPUT_DIR = SUPPORT.DEFAULT_OUTPUT_DIR
DEFAULT_LOG_DIR = SUPPORT.DEFAULT_LOG_DIR

EXPECTED_COMMON_COHORT_SIZE = SUPPORT.EXPECTED_COMMON_COHORT_SIZE
EXPECTED_ALIGNED_WEEKLY_ARTICLE_COUNT = 6075
EXPECTED_COMPLETE_MONTH_ARTICLE_COUNT = 6040
EXPECTED_SCFI_COMPLETE_MONTHS = 119
EXPECTED_LLM_COMPLETE_MONTHS = 118

MEAN_SAFE_WEEKLY = "mean_safe_weekly"
ADAPTED_PDF_SQRT_EMA3 = "adapted_pdf_sqrt_ema3"
PDF_SQRT_EMA3 = ADAPTED_PDF_SQRT_EMA3
MONTHLY_INDEX_METHODS = (MEAN_SAFE_WEEKLY, PDF_SQRT_EMA3)
PDF_EMA_SPAN = 3

DIMENSION_COLUMNS = SUPPORT.DIMENSION_COLUMNS
PRIMARY_SCORE_COLUMN = SUPPORT.PRIMARY_SCORE_COLUMN
PRIMARY_SIGNALS = SUPPORT.PRIMARY_SIGNALS
SIGNALS = SUPPORT.SIGNALS
SCORE_HYBRID_SOURCES = SUPPORT.SCORE_HYBRID_SOURCES
PRICE_OVERLAP_SIGNAL = SUPPORT.PRICE_OVERLAP_SIGNAL
COVERAGE_FEATURES = SUPPORT.COVERAGE_FEATURES
BASELINE_MODEL_IDS = SUPPORT.BASELINE_MODEL_IDS

HORIZONS = (1, 3, 6)
LAGS = (1, 2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)
BACKWARD_CHANGES = (1, 3, 6, 12)
AR_FEATURES = (
    "scfi_monthly_mean_log",
    *[f"scfi_monthly_mean_log_lag_{lag}m" for lag in LAGS],
    *[f"scfi_monthly_mean_log_change_{lag}m" for lag in BACKWARD_CHANGES],
    "calendar_month_sin",
    "calendar_month_cos",
)
BRENT_FEATURES = (
    "brent_monthly_mean_log",
    *[f"brent_monthly_mean_log_lag_{lag}m" for lag in LAGS],
    *[f"brent_monthly_mean_log_change_{lag}m" for lag in BACKWARD_CHANGES],
)
EXTERNAL_CONTROL_RAW_COLUMNS = SUPPORT.EXTERNAL_CONTROL_RAW_COLUMNS
EXTERNAL_CONTROL_FEATURES = (
    "cpb_world_trade_volume_lag_1m",
    "unctad_china_core_lane_lsbci_lag_1m",
)


@dataclass(frozen=True)
class MonthlyForecastConfiguration:
    model_id: str
    index_construction: str
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
        description="Aggregate safe weekly LLM signals and run monthly SCFI robustness forecasts."
    )
    parser.add_argument("--weekly-panel", type=Path, default=DEFAULT_WEEKLY_PANEL)
    parser.add_argument("--weekly-manifest", type=Path, default=DEFAULT_WEEKLY_MANIFEST)
    parser.add_argument("--panel-output", type=Path, default=DEFAULT_PANEL_OUTPUT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--version", default="llm_scfi_monthly_v001")
    parser.add_argument("--minimum-training-rows", type=int, default=60)
    parser.add_argument("--refit-every", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--minimum-stacking-history", type=int, default=24)
    parser.add_argument(
        "--expected-common-articles", type=int, default=EXPECTED_COMMON_COHORT_SIZE
    )
    parser.add_argument(
        "--expected-aligned-weekly-articles",
        type=int,
        default=EXPECTED_ALIGNED_WEEKLY_ARTICLE_COUNT,
    )
    parser.add_argument(
        "--expected-complete-month-articles",
        type=int,
        default=EXPECTED_COMPLETE_MONTH_ARTICLE_COUNT,
    )
    parser.add_argument(
        "--expected-scfi-complete-months",
        type=int,
        default=EXPECTED_SCFI_COMPLETE_MONTHS,
    )
    parser.add_argument(
        "--expected-llm-complete-months",
        type=int,
        default=EXPECTED_LLM_COMPLETE_MONTHS,
    )
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def configure_logging(log_dir: Path, version: str) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"07c_monthly_llm_scfi_comparison_{version}.log"
    logger = logging.getLogger("monthly_llm_scfi_comparison")
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
    return SUPPORT.require_file(path, label)


def sha256_file(path: Path) -> str:
    return SUPPORT.sha256_file(path)


def json_default(value: Any) -> Any:
    return SUPPORT.json_default(value)


def signal_raw_column(score_source: str, signal: str) -> str:
    return SUPPORT.signal_raw_column(score_source, signal)


def monthly_signal_column(
    construction: str, score_source: str, signal: str
) -> str:
    if construction not in MONTHLY_INDEX_METHODS:
        raise ValueError(f"Unknown monthly index construction: {construction}")
    if construction == MEAN_SAFE_WEEKLY:
        # Preserve the original 07c contract for the pre-existing monthly path.
        return signal_raw_column(score_source, signal)
    return f"llm_monthly_signal__{construction}__{score_source}__{signal}"


def pdf_base_column(score_source: str, signal: str) -> str:
    return f"llm_monthly_signal_base__pdf_sqrt__{score_source}__{signal}"


def signal_feature_columns(raw_column: str) -> list[str]:
    return [
        raw_column,
        *[f"{raw_column}_lag_{lag}m" for lag in LAGS],
        *[
            item
            for window in ROLLING_WINDOWS
            for item in (
                f"{raw_column}_rolling_mean_{window}m",
                f"{raw_column}_rolling_sd_{window}m",
            )
        ],
        f"{raw_column}_acceleration_1m",
    ]


def load_and_validate_weekly_manifest(
    path: Path,
    aliases: list[str],
    expected_common_articles: int,
    weekly_panel_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = require_file(path, "Weekly 07b manifest")
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    cohort = manifest.get("common_article_cohort", {})
    design = manifest.get("design", {})
    failures: list[str] = []
    if int(cohort.get("common_complete_articles", -1)) != expected_common_articles:
        failures.append("common_complete_articles mismatch")
    if int(cohort.get("expected_common_complete_articles", -1)) != expected_common_articles:
        failures.append("expected common cohort mismatch")
    if cohort.get("common_cohort_size_check_passed") is not True:
        failures.append("common cohort check not passed")
    if design.get("common_cohort_is_only_score_input") is not True:
        failures.append("common cohort is not the only score input")
    if design.get("broader_consensus_index_used_as_predictor") is not False:
        failures.append("broader consensus predictor was not disabled")
    if design.get("model_specific_unequal_coverage_indices_used") is not False:
        failures.append("unequal-coverage model indices were not disabled")
    if list(design.get("model_aliases", [])) != aliases:
        failures.append("manifest aliases differ from registry order")
    if int(design.get("weekly_common_article_count_sum", -1)) != expected_common_articles:
        failures.append("full weekly coverage does not conserve frozen cohort")
    if design.get("media_correction") != (
        "novelty-weighted sum / log1p(common article count)"
    ):
        failures.append("weekly media-correction formula is incompatible")
    if list(design.get("score_scale", [])) != [0, 4]:
        failures.append("weekly score scale is not the frozen 0-4 scale")
    if design.get("novelty_weighting") != (
        "input novelty_weight combines exact-duplicate correction with "
        "availability-safe within-week canonical-event weighting"
    ):
        failures.append("weekly novelty-weight provenance is incompatible")
    manifest_panel = Path(manifest.get("outputs", {}).get("aligned_panel", "")).resolve()
    if manifest_panel != weekly_panel_path.resolve():
        failures.append("weekly panel path differs from manifest output")
    if failures:
        raise ValueError("Unsafe weekly provenance: " + "; ".join(failures))
    audit = {
        "weekly_manifest_status": manifest.get("status"),
        "weekly_manifest_version": manifest.get("version"),
        "common_complete_articles": int(cohort["common_complete_articles"]),
        "common_cohort_check_passed": True,
        "common_cohort_only": True,
        "broader_consensus_used": False,
        "model_aliases": aliases,
        "weekly_manifest_sha256": sha256_file(resolved),
        "weekly_panel_sha256": sha256_file(weekly_panel_path),
    }
    return manifest, audit


def parse_boolean(series: pd.Series, label: str) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    invalid = normalized.notna() & ~normalized.isin({"true", "false"})
    if invalid.any():
        raise ValueError(f"{label} contains non-boolean values")
    return normalized.map({"true": True, "false": False}).astype("boolean")


def expected_mondays(period: pd.Period) -> pd.DatetimeIndex:
    return pd.date_range(period.start_time.normalize(), period.end_time.normalize(), freq="W-MON")


def causal_ema_restart_on_gap(series: pd.Series, span: int = PDF_EMA_SPAN) -> pd.Series:
    """Return a one-sided EMA and restart its state after every missing month."""
    if span < 1:
        raise ValueError("EMA span must be positive")
    alpha = 2.0 / (span + 1.0)
    result = pd.Series(np.nan, index=series.index, dtype=float)
    state = math.nan
    for position, value in series.items():
        numeric = float(value) if pd.notna(value) else math.nan
        if not math.isfinite(numeric):
            state = math.nan
            continue
        state = numeric if not math.isfinite(state) else alpha * numeric + (1 - alpha) * state
        result.at[position] = state
    return result


def recovered_sum_column(score_source: str, signal: str) -> str:
    return f"llm_monthly_weighted_score_sum__{score_source}__{signal}"


def load_safe_weekly_panel(
    path: Path,
    aliases: list[str],
    expected_aligned_weekly_articles: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = require_file(path, "Weekly common-cohort LLM panel")
    header = pd.read_csv(resolved, nrows=0).columns.tolist()
    score_sources = [*aliases, *SCORE_HYBRID_SOURCES]
    raw_signal_columns = [
        signal_raw_column(source, signal)
        for source in score_sources
        for signal in SIGNALS
    ]
    required = [
        "observation_date",
        "scfi",
        "brent_price",
        "llm_week_ending",
        "llm_signal_age_days",
        "llm_comparison_complete_case",
        *COVERAGE_FEATURES,
        *raw_signal_columns,
    ]
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"Weekly panel lacks safe raw fields: {missing}")
    raw_candidates = {
        column
        for column in header
        if column.startswith("llm_signal__")
        and "_lag_" not in column
        and "_rolling_" not in column
        and "_acceleration_" not in column
    }
    if raw_candidates != set(raw_signal_columns):
        raise ValueError(
            "Weekly raw signal whitelist mismatch; "
            f"missing={sorted(set(raw_signal_columns) - raw_candidates)}, "
            f"extra={sorted(raw_candidates - set(raw_signal_columns))}"
        )
    panel = pd.read_csv(resolved, usecols=required, low_memory=False)
    for column in ("observation_date", "llm_week_ending"):
        panel[column] = pd.to_datetime(panel[column], format="mixed", errors="coerce")
    if panel["observation_date"].isna().any():
        raise ValueError("Weekly panel has unparseable observation dates")
    panel["llm_comparison_complete_case"] = parse_boolean(
        panel["llm_comparison_complete_case"], "llm_comparison_complete_case"
    )
    numeric_columns = [
        "scfi",
        "brent_price",
        "llm_signal_age_days",
        *COVERAGE_FEATURES,
        *raw_signal_columns,
    ]
    for column in numeric_columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.sort_values("observation_date").reset_index(drop=True)
    if panel["observation_date"].duplicated().any():
        raise ValueError("Weekly panel has duplicate observation dates")
    if not panel["observation_date"].dt.dayofweek.eq(0).all():
        raise ValueError("Weekly SCFI observations must be Monday-dated")
    if not panel["observation_date"].diff().dropna().dt.days.eq(7).all():
        raise ValueError("Weekly panel must be a regular seven-day sequence")
    matched = panel["llm_week_ending"].notna()
    future = matched & panel["llm_week_ending"].gt(panel["observation_date"])
    unsafe_age = matched & ~panel["llm_signal_age_days"].between(0, 6)
    if future.any() or unsafe_age.any():
        raise AssertionError(
            f"Unsafe weekly provenance: future={int(future.sum())}, "
            f"age_outside_0_6={int(unsafe_age.sum())}"
        )
    complete = panel["llm_comparison_complete_case"].fillna(False).astype(bool)
    complete_requirements = [
        "scfi",
        "brent_price",
        "llm_week_ending",
        "llm_signal_age_days",
        *COVERAGE_FEATURES,
        *raw_signal_columns,
    ]
    if panel.loc[complete, complete_requirements].isna().any().any():
        raise AssertionError("A weekly complete-case row has missing raw inputs")
    aligned_article_count = int(
        round(float(panel["llm_common_article_count"].fillna(0).sum()))
    )
    if aligned_article_count != expected_aligned_weekly_articles:
        raise ValueError(
            "Aligned weekly article-count mismatch: "
            f"expected {expected_aligned_weekly_articles}, found {aligned_article_count}"
        )
    excluded_targets = [column for column in header if column.startswith("target_")]
    excluded_derived_signals = [
        column
        for column in header
        if column.startswith("llm_signal__") and column not in raw_signal_columns
    ]
    report = {
        "rows": len(panel),
        "first_monday": panel["observation_date"].min(),
        "last_monday": panel["observation_date"].max(),
        "complete_weekly_rows": int(complete.sum()),
        "future_signal_matches": int(future.sum()),
        "unsafe_signal_ages": int(unsafe_age.sum()),
        "signal_age_counts": panel.loc[
            matched, "llm_signal_age_days"
        ].value_counts().sort_index().to_dict(),
        "aligned_weekly_article_count_sum": aligned_article_count,
        "raw_signal_columns_selected": len(raw_signal_columns),
        "target_columns_excluded": len(excluded_targets),
        "weekly_derived_signal_columns_excluded": len(excluded_derived_signals),
        "broader_consensus_predictor_columns_selected": 0,
    }
    return panel, report


def aggregate_complete_months(
    weekly: pd.DataFrame,
    aliases: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    score_sources = [*aliases, *SCORE_HYBRID_SOURCES]
    raw_signal_columns = [
        signal_raw_column(source, signal)
        for source in score_sources
        for signal in SIGNALS
    ]
    work = weekly.copy()
    raw_count = pd.to_numeric(work["llm_common_article_count"], errors="coerce")
    effective_count = pd.to_numeric(
        work["llm_common_effective_article_count"], errors="coerce"
    )
    complete_week = work["llm_comparison_complete_case"].fillna(False).astype(bool)
    invalid_count = complete_week & (
        raw_count.isna()
        | effective_count.isna()
        | raw_count.lt(0)
        | effective_count.lt(0)
        | effective_count.gt(raw_count + 1e-9)
    )
    if invalid_count.any():
        examples = work.loc[
            invalid_count,
            [
                "observation_date",
                "llm_common_article_count",
                "llm_common_effective_article_count",
            ],
        ].head(10)
        raise ValueError(
            "Complete weekly rows contain invalid common/effective counts:\n"
            f"{examples.to_string(index=False)}"
        )
    zero_count = complete_week & raw_count.eq(0)
    if zero_count.any() and work.loc[zero_count, raw_signal_columns].abs().gt(1e-12).any().any():
        raise ValueError("A zero-article week has a nonzero LLM index")
    work["month_period"] = work["observation_date"].dt.to_period("M")
    periods = pd.period_range(
        work["month_period"].min(), work["month_period"].max(), freq="M"
    )
    rows: list[dict[str, Any]] = []
    weekly_date_sets = {
        period: set(group["observation_date"])
        for period, group in work.groupby("month_period")
    }
    for period in periods:
        group = work[work["month_period"].eq(period)].sort_values("observation_date")
        expected_dates = expected_mondays(period)
        dates_complete = weekly_date_sets.get(period, set()) == set(expected_dates)
        expected_count = len(expected_dates)
        scfi_complete = dates_complete and group["scfi"].notna().sum() == expected_count
        brent_complete = (
            dates_complete and group["brent_price"].notna().sum() == expected_count
        )
        llm_complete = (
            dates_complete
            and group["llm_comparison_complete_case"].fillna(False).astype(bool).sum()
            == expected_count
            and group[raw_signal_columns].notna().all(axis=1).sum() == expected_count
        )
        row: dict[str, Any] = {
            "month_ending": period.to_timestamp("M"),
            "expected_mondays": expected_count,
            "observed_mondays": len(group),
            "dates_complete": dates_complete,
            "monthly_scfi_complete": scfi_complete,
            "monthly_brent_complete": brent_complete,
            "monthly_llm_complete": llm_complete,
            "scfi_monthly_mean": group["scfi"].mean() if scfi_complete else math.nan,
            "brent_monthly_mean": (
                group["brent_price"].mean() if brent_complete else math.nan
            ),
            "llm_common_article_count": (
                group["llm_common_article_count"].sum() if llm_complete else math.nan
            ),
            "llm_common_effective_article_count": (
                group["llm_common_effective_article_count"].sum()
                if llm_complete
                else math.nan
            ),
            "llm_weekly_signal_age_min_days": (
                group["llm_signal_age_days"].min() if llm_complete else math.nan
            ),
            "llm_weekly_signal_age_max_days": (
                group["llm_signal_age_days"].max() if llm_complete else math.nan
            ),
        }
        for control in EXTERNAL_CONTROL_RAW_COLUMNS:
            row[control] = group[control].iloc[-1] if dates_complete and control in group else math.nan
        for column in raw_signal_columns:
            source, signal = column.removeprefix("llm_signal__").split("__", maxsplit=1)
            mean_column = monthly_signal_column(MEAN_SAFE_WEEKLY, source, signal)
            base_column = pdf_base_column(source, signal)
            sum_column = recovered_sum_column(source, signal)
            if not llm_complete:
                monthly_mean = math.nan
                weighted_sum = math.nan
                pdf_base = math.nan
            else:
                monthly_mean = float(group[column].mean())
                weighted_sum = float(
                    (
                        group[column].astype(float)
                        * np.log1p(group["llm_common_article_count"].astype(float))
                    ).sum()
                )
                monthly_effective = float(row["llm_common_effective_article_count"])
                pdf_base = (
                    weighted_sum / math.sqrt(monthly_effective)
                    if monthly_effective > 0
                    else math.nan
                )
            # Preserve the original 07c column contract as an audit/backward-compatibility field.
            row[column] = monthly_mean
            row[mean_column] = monthly_mean
            row[base_column] = pdf_base
            row[sum_column] = weighted_sum
        rows.append(row)
    monthly = pd.DataFrame(rows)
    for source in score_sources:
        for signal in SIGNALS:
            base_column = pdf_base_column(source, signal)
            final_column = monthly_signal_column(PDF_SQRT_EMA3, source, signal)
            monthly[final_column] = causal_ema_restart_on_gap(
                monthly[base_column], span=PDF_EMA_SPAN
            )
    monthly["monthly_analysis_complete_case"] = monthly[
        ["monthly_scfi_complete", "monthly_brent_complete", "monthly_llm_complete"]
    ].all(axis=1)

    long_rows: list[pd.DataFrame] = []
    base_columns = [
        "month_ending",
        "expected_mondays",
        "observed_mondays",
        "monthly_analysis_complete_case",
        "llm_common_article_count",
        "llm_common_effective_article_count",
    ]
    for source in score_sources:
        source_type = "llm" if source in aliases else "score_hybrid"
        for signal in SIGNALS:
            mean_column = monthly_signal_column(MEAN_SAFE_WEEKLY, source, signal)
            base_column = pdf_base_column(source, signal)
            sum_column = recovered_sum_column(source, signal)
            for construction in MONTHLY_INDEX_METHODS:
                value_column = monthly_signal_column(construction, source, signal)
                part = monthly[base_columns].copy()
                part["index_construction"] = construction
                part["source_type"] = source_type
                part["score_source"] = source
                part["signal_name"] = signal
                part["monthly_signal_value"] = monthly[value_column]
                part["monthly_unsmoothed_value"] = (
                    monthly[mean_column]
                    if construction == MEAN_SAFE_WEEKLY
                    else monthly[base_column]
                )
                part["monthly_weighted_score_sum"] = monthly[sum_column]
                part["adapted_pdf_minus_mean_safe_weekly"] = (
                    monthly[monthly_signal_column(PDF_SQRT_EMA3, source, signal)]
                    - monthly[mean_column]
                )
                part["absolute_construction_difference"] = part[
                    "adapted_pdf_minus_mean_safe_weekly"
                ].abs()
                part["price_overlap_risk"] = signal == PRICE_OVERLAP_SIGNAL
                long_rows.append(part)
    monthly_long = pd.concat(long_rows, ignore_index=True).sort_values(
        [
            "month_ending",
            "index_construction",
            "source_type",
            "score_source",
            "signal_name",
        ]
    )
    long_keys = ["month_ending", "index_construction", "score_source", "signal_name"]
    if monthly_long.duplicated(long_keys).any():
        raise AssertionError("Monthly long index contains duplicate construction/source/signal keys")
    report = {
        "calendar_months": len(monthly),
        "first_month": monthly["month_ending"].min(),
        "last_month": monthly["month_ending"].max(),
        "scfi_complete_months": int(monthly["monthly_scfi_complete"].sum()),
        "brent_complete_months": int(monthly["monthly_brent_complete"].sum()),
        "llm_complete_months": int(monthly["monthly_llm_complete"].sum()),
        "analysis_complete_months": int(monthly["monthly_analysis_complete_case"].sum()),
        "index_constructions": list(MONTHLY_INDEX_METHODS),
        "mean_safe_weekly_formula": (
            "mean over every expected Monday of the already leakage-safe weekly "
            "novelty-weighted-sum/log1p(common-count) index"
        ),
        "adapted_pdf_formula": (
            "EMA(span=3, alpha=0.5, adjust=False) of monthly recovered novelty-weighted "
            "score sum divided by sqrt(monthly effective article count); EMA restarts after gaps"
        ),
        "adapted_pdf_is_literal_replication": False,
        "zero_effective_volume_months": int(
            monthly["monthly_llm_complete"]
            .astype(bool)
            .mul(monthly["llm_common_effective_article_count"].fillna(0).le(0))
            .sum()
        ),
        "incomplete_months": monthly.loc[
            ~monthly["monthly_analysis_complete_case"],
            [
                "month_ending",
                "expected_mondays",
                "observed_mondays",
                "monthly_scfi_complete",
                "monthly_brent_complete",
                "monthly_llm_complete",
            ],
        ].to_dict(orient="records"),
    }
    return monthly, monthly_long, report


def safe_log(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return np.log(numeric.where(numeric.gt(0)))


def add_past_signal_features(
    panel: pd.DataFrame, raw_signal_columns: list[str]
) -> pd.DataFrame:
    result = panel.copy()
    derived: dict[str, pd.Series] = {}
    for raw_column in raw_signal_columns:
        for lag in LAGS:
            derived[f"{raw_column}_lag_{lag}m"] = result[raw_column].shift(lag)
        for window in ROLLING_WINDOWS:
            rolling = result[raw_column].rolling(window, min_periods=window)
            derived[f"{raw_column}_rolling_mean_{window}m"] = rolling.mean()
            derived[f"{raw_column}_rolling_sd_{window}m"] = rolling.std(ddof=0)
        derived[f"{raw_column}_acceleration_1m"] = result[raw_column].diff().diff()
    return pd.concat([result, pd.DataFrame(derived, index=result.index)], axis=1)


def prepare_monthly_model_panel(
    monthly: pd.DataFrame, aliases: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = monthly.sort_values("month_ending").reset_index(drop=True).copy()
    periods = panel["month_ending"].dt.to_period("M")
    if not pd.Series(periods.astype(int)).diff().dropna().eq(1).all():
        raise ValueError("Monthly model panel must have a contiguous PeriodIndex")
    panel["scfi_monthly_mean_log"] = safe_log(panel["scfi_monthly_mean"])
    panel["brent_monthly_mean_log"] = safe_log(panel["brent_monthly_mean"])
    for lag in LAGS:
        panel[f"scfi_monthly_mean_log_lag_{lag}m"] = panel[
            "scfi_monthly_mean_log"
        ].shift(lag)
        panel[f"brent_monthly_mean_log_lag_{lag}m"] = panel[
            "brent_monthly_mean_log"
        ].shift(lag)
    for lag in BACKWARD_CHANGES:
        panel[f"scfi_monthly_mean_log_change_{lag}m"] = panel[
            "scfi_monthly_mean_log"
        ].diff(lag)
        panel[f"brent_monthly_mean_log_change_{lag}m"] = panel[
            "brent_monthly_mean_log"
        ].diff(lag)
    for control in EXTERNAL_CONTROL_RAW_COLUMNS:
        panel[f"{control}_lag_1m"] = panel[control].shift(1)
    calendar_month = panel["month_ending"].dt.month.astype(float)
    panel["calendar_month_sin"] = np.sin(2 * np.pi * calendar_month / 12)
    panel["calendar_month_cos"] = np.cos(2 * np.pi * calendar_month / 12)

    target_counts: dict[int, int] = {}
    for horizon in HORIZONS:
        future = panel["scfi_monthly_mean"].shift(-horizon)
        panel[f"target_scfi_monthly_mean_h{horizon}m"] = future
        panel[f"target_scfi_monthly_log_change_h{horizon}m"] = (
            safe_log(future) - panel["scfi_monthly_mean_log"]
        )
        target_counts[horizon] = int(
            panel[f"target_scfi_monthly_log_change_h{horizon}m"].notna().sum()
        )

    raw_signal_columns = [
        monthly_signal_column(construction, source, signal)
        for construction in MONTHLY_INDEX_METHODS
        for source in [*aliases, *SCORE_HYBRID_SOURCES]
        for signal in SIGNALS
    ]
    panel = add_past_signal_features(panel, raw_signal_columns)
    current_requirements = [
        "scfi_monthly_mean",
        "brent_monthly_mean_log",
        *COVERAGE_FEATURES,
        *raw_signal_columns,
        *EXTERNAL_CONTROL_FEATURES,
    ]
    panel["llm_monthly_comparison_complete_case"] = (
        panel["monthly_analysis_complete_case"].astype(bool)
        & panel[current_requirements].notna().all(axis=1)
    )
    report = {
        "rows": len(panel),
        "complete_model_origins": int(
            panel["llm_monthly_comparison_complete_case"].sum()
        ),
        "first_complete_month": panel.loc[
            panel["llm_monthly_comparison_complete_case"], "month_ending"
        ].min(),
        "last_complete_month": panel.loc[
            panel["llm_monthly_comparison_complete_case"], "month_ending"
        ].max(),
        "target_nonmissing": target_counts,
        "targets_recomputed_from_complete_monthly_mean_scfi": True,
        "weekly_targets_used": False,
        "monthly_index_constructions": list(MONTHLY_INDEX_METHODS),
        "raw_monthly_signal_columns": len(raw_signal_columns),
    }
    return panel, report


def build_monthly_signal_difference_summary(
    panel: pd.DataFrame, aliases: list[str]
) -> pd.DataFrame:
    complete = panel[panel["llm_monthly_comparison_complete_case"]]
    rows: list[dict[str, Any]] = []
    for construction in MONTHLY_INDEX_METHODS:
        for signal in SIGNALS:
            for left, right in combinations(aliases, 2):
                pair = complete[
                    [
                        monthly_signal_column(construction, left, signal),
                        monthly_signal_column(construction, right, signal),
                    ]
                ].dropna()
                pair.columns = ["left", "right"]
                difference = pair["left"] - pair["right"]
                rho = (
                    SUPPORT.spearmanr(pair["left"], pair["right"]).statistic
                    if len(pair) > 1
                    else math.nan
                )
                rows.append(
                    {
                        "level": "complete_safe_monday_month",
                        "index_construction": construction,
                        "signal_name": signal,
                        "model_left": left,
                        "model_right": right,
                        "n_complete_months": len(pair),
                        "mean_signed_difference": difference.mean(),
                        "median_signed_difference": difference.median(),
                        "mean_absolute_difference": difference.abs().mean(),
                        "rmse_difference": np.sqrt(np.mean(difference**2)),
                        "spearman_correlation": rho,
                        "price_overlap_risk": signal == PRICE_OVERLAP_SIGNAL,
                    }
                )
    return pd.DataFrame(rows)


def build_direct_configurations(aliases: list[str]) -> list[MonthlyForecastConfiguration]:
    configurations: list[MonthlyForecastConfiguration] = []
    for construction in MONTHLY_INDEX_METHODS:
        for score_source in [*aliases, *SCORE_HYBRID_SOURCES]:
            source_type = "llm" if score_source in aliases else "score_hybrid"
            for signal in SIGNALS:
                source_label = "llm" if source_type == "llm" else "score_hybrid"
                model_id = (
                    f"ridge__monthly__{construction}__{source_label}__"
                    f"{score_source}__{signal}"
                )
                hybrid_level = "none" if source_type == "llm" else "score"
                configurations.append(
                    MonthlyForecastConfiguration(
                        model_id=model_id,
                        index_construction=construction,
                        source_type=source_type,
                        score_source=score_source,
                        signal_name=signal,
                        hybrid_level=hybrid_level,
                        feature_set=(
                            f"ar_brent_coverage_plus_{construction}_{score_source}_{signal}"
                        ),
                    )
                )
    expected = len(MONTHLY_INDEX_METHODS) * (
        len(aliases) + len(SCORE_HYBRID_SOURCES)
    ) * len(SIGNALS)
    if len(configurations) != expected or len({item.model_id for item in configurations}) != expected:
        raise AssertionError("Monthly direct configurations are incomplete or non-unique")
    return configurations


def build_feature_sets(
    panel: pd.DataFrame, configurations: list[Any]
) -> dict[str, list[str]]:
    available = set(panel.columns)
    base = [*AR_FEATURES, *BRENT_FEATURES]
    coverage = [*base, *COVERAGE_FEATURES, *EXTERNAL_CONTROL_FEATURES]
    missing_base = [column for column in coverage if column not in available]
    if missing_base:
        raise ValueError(f"Monthly panel lacks baseline features: {missing_base}")
    feature_sets: dict[str, list[str]] = {
        "ridge__ar_brent": base,
        "ridge__ar_brent_coverage": coverage,
    }
    for config in configurations:
        raw_column = monthly_signal_column(
            config.index_construction, config.score_source, config.signal_name
        )
        features = [*coverage, *signal_feature_columns(raw_column)]
        missing = [column for column in features if column not in available]
        if missing:
            raise ValueError(f"Monthly feature set {config.model_id} lacks: {missing}")
        if any(column.startswith("target_") for column in features):
            raise AssertionError(f"Target leakage in {config.model_id}")
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
        candidate, "llm_monthly_comparison_complete_case"
    ].astype(bool).to_numpy()
    target_available = panel.loc[candidate, target_column].notna().to_numpy()
    return candidate[complete & target_available]


def eligible_origins(panel: pd.DataFrame, horizon: int, target_column: str) -> list[int]:
    target_month = panel["month_ending"].shift(-horizon)
    mask = (
        panel["llm_monthly_comparison_complete_case"].astype(bool)
        & panel[target_column].notna()
        & target_month.notna()
    )
    return panel.index[mask].tolist()


def forecast_row(
    panel: pd.DataFrame,
    position: int,
    horizon: int,
    prediction: float,
    model_id: str,
    feature_set: str,
    index_construction: str,
    source_type: str,
    score_source: str,
    signal_name: str,
    hybrid_level: str,
    training_rows: int,
    training_feature_end_date: pd.Timestamp,
    training_label_end_date: pd.Timestamp,
    model_fit_origin_date: pd.Timestamp,
) -> dict[str, Any]:
    target_column = f"target_scfi_monthly_log_change_h{horizon}m"
    level_column = f"target_scfi_monthly_mean_h{horizon}m"
    actual = float(panel.at[position, target_column])
    current_scfi = float(panel.at[position, "scfi_monthly_mean"])
    actual_level = float(panel.at[position, level_column])
    predicted_level = current_scfi * math.exp(float(prediction))
    return {
        "origin_position": int(position),
        "origin_date": panel.at[position, "month_ending"],
        "target_date": panel.at[position + horizon, "month_ending"],
        "horizon_months": int(horizon),
        "model_id": model_id,
        "feature_set": feature_set,
        "index_construction": index_construction,
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
    target_column = f"target_scfi_monthly_log_change_h{horizon}m"
    rows: list[dict[str, Any]] = []
    for position in eligible_origins(panel, horizon, target_column):
        train = training_positions(panel, position, horizon, target_column)
        if len(train) < minimum_training_rows:
            continue
        for model_id, prediction in {
            "naive_zero_change": 0.0,
            "historical_mean_change": float(panel.loc[train, target_column].mean()),
        }.items():
            rows.append(
                forecast_row(
                    panel,
                    position,
                    horizon,
                    prediction,
                    model_id,
                    "benchmark",
                    "shared_baseline",
                    "benchmark",
                    "none",
                    "none",
                    "none",
                    len(train),
                    panel.at[train[-1], "month_ending"],
                    panel.at[train[-1] + horizon, "month_ending"],
                    panel.at[position, "month_ending"],
                )
            )
    return rows


def rolling_ridge_forecasts(
    panel: pd.DataFrame,
    horizon: int,
    model_id: str,
    features: list[str],
    index_construction: str,
    source_type: str,
    score_source: str,
    signal_name: str,
    hybrid_level: str,
    minimum_training_rows: int,
    refit_every: int,
    ridge_alpha: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_column = f"target_scfi_monthly_log_change_h{horizon}m"
    positions = [
        position
        for position in eligible_origins(panel, horizon, target_column)
        if len(training_positions(panel, position, horizon, target_column))
        >= minimum_training_rows
    ]
    rows: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    for block_start in range(0, len(positions), refit_every):
        block_positions = positions[block_start : block_start + refit_every]
        fit_position = block_positions[0]
        train = training_positions(panel, fit_position, horizon, target_column)
        estimator = SUPPORT.make_ridge_estimator(ridge_alpha)
        estimator.fit(panel.loc[train, features], panel.loc[train, target_column])
        feature_end = panel.at[train[-1], "month_ending"]
        label_end = panel.at[train[-1] + horizon, "month_ending"]
        fit_origin = panel.at[fit_position, "month_ending"]
        if label_end > fit_origin:
            raise AssertionError("Monthly training label exceeds model-fit origin")
        fits.append(
            {
                "horizon_months": horizon,
                "model_id": model_id,
                "index_construction": index_construction,
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
            feature_set = (
                "ar_brent"
                if model_id == "ridge__ar_brent"
                else "ar_brent_coverage"
                if model_id == "ridge__ar_brent_coverage"
                else (
                    f"ar_brent_coverage_plus_{index_construction}_"
                    f"{score_source}_{signal_name}"
                )
            )
            rows.append(
                forecast_row(
                    panel,
                    position,
                    horizon,
                    float(prediction),
                    model_id,
                    feature_set,
                    index_construction,
                    source_type,
                    score_source,
                    signal_name,
                    hybrid_level,
                    len(train),
                    feature_end,
                    label_end,
                    fit_origin,
                )
            )
    return rows, fits


def derived_forecast_row(
    template: pd.Series,
    prediction: float,
    model_id: str,
    index_construction: str,
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
        "horizon_months": int(template["horizon_months"]),
        "model_id": model_id,
        "feature_set": (
            f"forecast_hybrid_from_three_llms_{index_construction}_{signal_name}"
        ),
        "index_construction": index_construction,
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
        raise ValueError("No monthly individual-LLM forecasts are available for hybridization")
    date_columns = (
        "origin_date",
        "target_date",
        "training_feature_end_date",
        "training_label_end_date",
        "model_fit_origin_date",
    )
    for column in date_columns:
        llm[column] = pd.to_datetime(llm[column])

    hybrid_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    grouping = ["index_construction", "horizon_months", "signal_name"]
    for (construction, horizon, signal), group in llm.groupby(grouping, sort=True):
        counts = group.groupby("score_source").size().reindex(aliases)
        if counts.isna().any() or counts.nunique() != 1:
            raise AssertionError(
                "Monthly individual-LLM forecasts are unbalanced for "
                f"construction={construction}, h={horizon}, signal={signal}: "
                f"{counts.to_dict()}"
            )
        prediction = group.pivot(
            index="origin_date", columns="score_source", values="predicted_log_change"
        ).reindex(columns=aliases)
        if prediction.isna().any().any():
            raise AssertionError("A monthly forecast hybrid lacks a constituent prediction")
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
            raise AssertionError("Monthly constituent LLM forecasts disagree on targets")

        for origin_date in prediction.index:
            current_group = group[group["origin_date"].eq(origin_date)]
            template = metadata.loc[origin_date]
            equal_label_end = current_group["training_label_end_date"].max()
            equal_fit_origin = current_group["model_fit_origin_date"].max()
            if equal_label_end > equal_fit_origin:
                raise AssertionError("Monthly constituent labels exceed hybrid fit time")
            equal_row = derived_forecast_row(
                template=template,
                prediction=float(prediction.loc[origin_date].mean()),
                model_id=f"forecast_equal_mean__{construction}__{signal}",
                index_construction=str(construction),
                score_source="forecast_equal_mean",
                signal_name=str(signal),
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

            history_origins = metadata.index[metadata["target_date"].le(origin_date)]
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
                raw_weights, _ = SUPPORT.nnls(x_history, y_history)
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
                np.dot(
                    prediction.loc[origin_date, aliases].to_numpy(dtype=float),
                    weights,
                )
            )
            history_target_end = metadata.loc[used_origins, "target_date"].max()
            if history_target_end > origin_date:
                raise AssertionError("Monthly stacker used an unavailable outcome")
            stack_row = derived_forecast_row(
                template=template,
                prediction=stacked_prediction,
                model_id=f"forecast_nnls_stack__{construction}__{signal}",
                index_construction=str(construction),
                score_source="forecast_nnls_stack",
                signal_name=str(signal),
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
                        "horizon_months": int(horizon),
                        "index_construction": construction,
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
    return pd.DataFrame(hybrid_rows), pd.DataFrame(weight_rows)


def validate_common_origin_sets(
    forecasts: pd.DataFrame, required_model_ids: list[str]
) -> pd.DataFrame:
    duplicate = forecasts.duplicated(
        ["horizon_months", "model_id", "origin_date"], keep=False
    )
    if duplicate.any():
        examples = forecasts.loc[
            duplicate, ["horizon_months", "model_id", "origin_date"]
        ].head(10)
        raise AssertionError(
            "Duplicate monthly model/horizon/origin forecasts are not allowed:\n"
            f"{examples.to_string(index=False)}"
        )
    rows: list[dict[str, Any]] = []
    required = set(required_model_ids)
    for horizon, group in forecasts.groupby("horizon_months"):
        missing = sorted(required - set(group["model_id"].unique()))
        if missing:
            raise AssertionError(f"Monthly horizon {horizon} lacks models: {missing}")
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
                    f"Monthly forecast origins differ at h={horizon}: "
                    f"{reference_id} versus {model_id}"
                )
        rows.append(
            {
                "horizon_months": int(horizon),
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
    for horizon, horizon_frame in forecasts.groupby("horizon_months"):
        baselines = {
            model_id: horizon_frame[horizon_frame["model_id"].eq(model_id)].set_index(
                "origin_date"
            )
            for model_id in (
                "naive_zero_change",
                "historical_mean_change",
                "ridge__ar_brent_coverage",
            )
        }
        for model_id, group in horizon_frame.groupby("model_id"):
            group = group.sort_values("origin_date").set_index("origin_date")
            error = group["actual_log_change"] - group["predicted_log_change"]
            level_error = group["actual_scfi_level"] - group["predicted_scfi_level"]
            aligned: dict[str, pd.DataFrame] = {}
            for baseline_id, baseline in baselines.items():
                aligned[baseline_id] = baseline.reindex(group.index)
                if aligned[baseline_id]["predicted_log_change"].isna().any():
                    raise AssertionError(
                        f"Monthly {baseline_id} is not aligned for {model_id}"
                    )
            baseline_sse = {
                baseline_id: float(
                    np.sum(
                        (
                            frame["actual_log_change"]
                            - frame["predicted_log_change"]
                        )
                        ** 2
                    )
                )
                for baseline_id, frame in aligned.items()
            }
            model_sse = float(np.sum(error**2))
            metadata = group.iloc[0]
            correlation = (
                group[["actual_log_change", "predicted_log_change"]].corr().iloc[0, 1]
            )
            rows.append(
                {
                    "horizon_months": int(horizon),
                    "model_id": model_id,
                    "feature_set": metadata["feature_set"],
                    "index_construction": metadata["index_construction"],
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
                        math.nan
                        if baseline_sse["naive_zero_change"] == 0
                        else 1 - model_sse / baseline_sse["naive_zero_change"]
                    ),
                    "oos_r2_vs_historical_mean": (
                        math.nan
                        if baseline_sse["historical_mean_change"] == 0
                        else 1 - model_sse / baseline_sse["historical_mean_change"]
                    ),
                    "oos_r2_vs_ar_brent_coverage": (
                        math.nan
                        if baseline_sse["ridge__ar_brent_coverage"] == 0
                        else 1 - model_sse / baseline_sse["ridge__ar_brent_coverage"]
                    ),
                    "relative_rmse_vs_ar_brent_coverage": (
                        math.nan
                        if baseline_sse["ridge__ar_brent_coverage"] == 0
                        else math.sqrt(
                            model_sse / baseline_sse["ridge__ar_brent_coverage"]
                        )
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon_months", "rmse_log_change"])


def loss_test_rows(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
    horizon: int,
    baseline_model: str,
    challenger_model: str,
    baseline_construction: str,
    challenger_construction: str,
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
        mean_difference, statistic, standard_error = SUPPORT.hac_mean_test(
            differential.to_numpy(), max(int(horizon) - 1, 0)
        )
        if pd.isna(statistic):
            p_value = math.nan
        elif alternative == "one_sided_positive":
            p_value = float(1 - SUPPORT.norm.cdf(statistic))
        else:
            p_value = float(2 * (1 - SUPPORT.norm.cdf(abs(statistic))))
        rows.append(
            {
                "horizon_months": int(horizon),
                "baseline_model": baseline_model,
                "challenger_model": challenger_model,
                "baseline_index_construction": baseline_construction,
                "challenger_index_construction": challenger_construction,
                "analysis_role": (
                    "adapted_equation_sensitivity"
                    if challenger_construction == PDF_SQRT_EMA3
                    else "monthly_frequency_robustness"
                ),
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
                "family_id": f"{family_root}__{test_name}",
                "price_overlap_risk": signal_name == PRICE_OVERLAP_SIGNAL,
            }
        )
    return rows


def forecast_comparison_tests(
    forecasts: pd.DataFrame, aliases: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, horizon_frame in forecasts.groupby("horizon_months"):
        coverage = horizon_frame[
            horizon_frame["model_id"].eq("ridge__ar_brent_coverage")
        ]
        if coverage.empty:
            raise ValueError(f"Missing monthly coverage baseline at horizon {horizon}")
        challengers = horizon_frame[
            ~horizon_frame["model_id"].isin(BASELINE_MODEL_IDS)
        ]
        for challenger_id, challenger in challengers.groupby("model_id"):
            metadata = challenger.iloc[0]
            signal = str(metadata["signal_name"])
            source_type = str(metadata["source_type"])
            construction = str(metadata["index_construction"])
            if signal in PRIMARY_SIGNALS:
                family_root = (
                    "monthly_primary_llm_vs_coverage"
                    if source_type == "llm"
                    else "monthly_primary_hybrid_vs_coverage"
                )
            else:
                family_root = "monthly_exploratory_dimension_vs_coverage"
            rows.extend(
                loss_test_rows(
                    baseline=coverage,
                    challenger=challenger,
                    horizon=int(horizon),
                    baseline_model="ridge__ar_brent_coverage",
                    challenger_model=str(challenger_id),
                    baseline_construction="shared_baseline",
                    challenger_construction=construction,
                    source_type=source_type,
                    score_source=str(metadata["score_source"]),
                    signal_name=signal,
                    family_root=family_root,
                    include_clark_west=source_type != "forecast_hybrid",
                )
            )

        individual = horizon_frame[horizon_frame["source_type"].eq("llm")]
        for (construction, signal), signal_frame in individual.groupby(
            ["index_construction", "signal_name"]
        ):
            family_root = (
                "monthly_primary_llm_pairwise"
                if signal in PRIMARY_SIGNALS
                else "monthly_exploratory_dimension_llm_pairwise"
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
                        "Missing monthly LLM pair for "
                        f"construction={construction}, h={horizon}, signal={signal}"
                    )
                rows.extend(
                    loss_test_rows(
                        baseline=left,
                        challenger=right,
                        horizon=int(horizon),
                        baseline_model=str(left["model_id"].iloc[0]),
                        challenger_model=str(right["model_id"].iloc[0]),
                        baseline_construction=str(construction),
                        challenger_construction=str(construction),
                        source_type="llm_pairwise",
                        score_source=f"{left_alias}_vs_{right_alias}",
                        signal_name=str(signal),
                        family_root=family_root,
                        include_clark_west=False,
                    )
                )

        direct = horizon_frame[
            horizon_frame["source_type"].isin({"llm", "score_hybrid"})
        ]
        method_keys = ["source_type", "score_source", "signal_name"]
        for keys, comparison in direct.groupby(method_keys):
            source_type, score_source, signal = map(str, keys)
            by_method = {
                method: comparison[comparison["index_construction"].eq(method)]
                for method in MONTHLY_INDEX_METHODS
            }
            if any(frame.empty for frame in by_method.values()):
                raise AssertionError(
                    f"Missing monthly construction comparison for {keys}, h={horizon}"
                )
            family_root = (
                "monthly_primary_construction_pairwise"
                if signal in PRIMARY_SIGNALS
                else "monthly_exploratory_dimension_construction_pairwise"
            )
            rows.extend(
                loss_test_rows(
                    baseline=by_method[MEAN_SAFE_WEEKLY],
                    challenger=by_method[PDF_SQRT_EMA3],
                    horizon=int(horizon),
                    baseline_model=str(by_method[MEAN_SAFE_WEEKLY]["model_id"].iloc[0]),
                    challenger_model=str(by_method[PDF_SQRT_EMA3]["model_id"].iloc[0]),
                    baseline_construction=MEAN_SAFE_WEEKLY,
                    challenger_construction=PDF_SQRT_EMA3,
                    source_type=f"{source_type}_construction_pairwise",
                    score_source=score_source,
                    signal_name=signal,
                    family_root=family_root,
                    include_clark_west=False,
                )
            )

    tests = pd.DataFrame(rows)
    if tests.empty:
        raise ValueError("No monthly forecast comparison tests were produced")
    tests["multiple_testing_method"] = np.where(
        tests["family_id"].str.contains("primary"), "holm", "fdr_bh"
    )
    tests["p_value_adjusted"] = np.nan
    for family_id, positions in tests.groupby("family_id").groups.items():
        method = str(tests.loc[positions, "multiple_testing_method"].iloc[0])
        tests.loc[positions, "p_value_adjusted"] = SUPPORT.adjust_pvalues(
            tests.loc[positions, "p_value"], method
        )
    tests["significant_raw_0_05"] = tests["p_value"].lt(0.05)
    tests["significant_adjusted_0_05"] = tests["p_value_adjusted"].lt(0.05)
    return tests.sort_values(
        ["family_id", "horizon_months", "signal_name", "challenger_model"]
    ).reset_index(drop=True)


def build_llm_rankings(
    metrics: pd.DataFrame, tests: pd.DataFrame, aliases: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = metrics[
        metrics["source_type"].eq("llm")
        & metrics["signal_name"].isin(PRIMARY_SIGNALS)
    ].copy()
    expected_cells = (
        len(MONTHLY_INDEX_METHODS)
        * len(aliases)
        * len(PRIMARY_SIGNALS)
        * len(HORIZONS)
    )
    if len(cells) != expected_cells:
        raise AssertionError(
            f"Expected {expected_cells} monthly primary LLM ranking cells; found {len(cells)}"
        )
    ranking_keys = [
        "index_construction",
        "score_source",
        "signal_name",
        "horizon_months",
    ]
    if cells.duplicated(ranking_keys).any():
        raise AssertionError("Monthly LLM ranking cells contain duplicate Cartesian keys")
    expected_keys = set(
        product(MONTHLY_INDEX_METHODS, aliases, PRIMARY_SIGNALS, HORIZONS)
    )
    found_keys = set(cells[ranking_keys].itertuples(index=False, name=None))
    if found_keys != expected_keys:
        raise AssertionError("Monthly ranking cells do not form the expected Cartesian product")
    rank_groups = ["index_construction", "horizon_months", "signal_name"]
    cells["rmse_rank_within_horizon_signal"] = cells.groupby(rank_groups)[
        "rmse_log_change"
    ].rank(method="average", ascending=True)
    cells["mae_rank_within_horizon_signal"] = cells.groupby(rank_groups)[
        "mae_log_change"
    ].rank(method="average", ascending=True)
    cells["skill_rank_within_horizon_signal"] = cells.groupby(rank_groups)[
        "oos_r2_vs_ar_brent_coverage"
    ].rank(method="average", ascending=False)

    aggregation = {
        "primary_comparison_cells": ("model_id", "size"),
        "mean_rmse_rank": ("rmse_rank_within_horizon_signal", "mean"),
        "mean_mae_rank": ("mae_rank_within_horizon_signal", "mean"),
        "mean_skill_rank": ("skill_rank_within_horizon_signal", "mean"),
        "mean_rmse_log_change": ("rmse_log_change", "mean"),
        "mean_relative_rmse_vs_coverage": (
            "relative_rmse_vs_ar_brent_coverage",
            "mean",
        ),
        "mean_oos_r2_vs_coverage": ("oos_r2_vs_ar_brent_coverage", "mean"),
        "mean_directional_accuracy": ("directional_accuracy", "mean"),
        "rmse_cell_wins": (
            "rmse_rank_within_horizon_signal",
            lambda values: int(np.isclose(values, 1).sum()),
        ),
    }
    specific = cells.groupby(
        ["index_construction", "score_source"], as_index=False
    ).agg(**aggregation)
    pooled = cells.groupby("score_source", as_index=False).agg(**aggregation)
    pooled.insert(0, "index_construction", "pooled_monthly_robustness")
    ranking = pd.concat([specific, pooled], ignore_index=True)
    ranking["adjusted_dm_squared_pairwise_wins"] = 0
    ranking["adjusted_dm_squared_pairwise_losses"] = 0
    pairwise = tests[
        tests["family_id"].eq(
            "monthly_primary_llm_pairwise__diebold_mariano_squared"
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
        winner, loser = (
            (challenger_alias, baseline_alias)
            if row["mean_loss_improvement"] > 0
            else (baseline_alias, challenger_alias)
        )
        construction = str(row["challenger_index_construction"])
        for scope in (construction, "pooled_monthly_robustness"):
            ranking.loc[
                ranking["index_construction"].eq(scope)
                & ranking["score_source"].eq(winner),
                "adjusted_dm_squared_pairwise_wins",
            ] += 1
            ranking.loc[
                ranking["index_construction"].eq(scope)
                & ranking["score_source"].eq(loser),
                "adjusted_dm_squared_pairwise_losses",
            ] += 1
    ranking = ranking.sort_values(
        [
            "index_construction",
            "mean_rmse_rank",
            "mean_relative_rmse_vs_coverage",
            "score_source",
        ]
    ).reset_index(drop=True)
    ranking["predictive_point_rank"] = ranking.groupby("index_construction").cumcount() + 1
    ranking["ranking_scope"] = (
        "equal-weight ranks over overall-max and dimension-mean7 at 1, 3, and 6 months; "
        "reported separately by construction plus a pooled robustness summary"
    )
    return ranking, cells.sort_values(
        [
            "index_construction",
            "horizon_months",
            "signal_name",
            "rmse_rank_within_horizon_signal",
        ]
    )


def build_predictor_summaries(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sources = metrics[
        metrics["source_type"].isin({"llm", "score_hybrid", "forecast_hybrid"})
    ].copy()
    primary = sources[
        sources["signal_name"].isin(PRIMARY_SIGNALS)
        & ~sources["score_source"].eq("forecast_nnls_stack")
    ]
    primary_summary = primary.groupby(
        ["index_construction", "source_type", "score_source", "hybrid_level"],
        as_index=False,
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
    primary_summary = primary_summary.sort_values(
        ["index_construction", "mean_relative_rmse_vs_coverage", "mean_rmse_log_change"]
    ).reset_index(drop=True)
    primary_summary["point_rank_within_construction"] = (
        primary_summary.groupby("index_construction").cumcount() + 1
    )

    dimensions = sources[sources["signal_name"].isin(DIMENSION_COLUMNS)].copy()
    dimension_summary = dimensions.groupby(
        [
            "index_construction",
            "source_type",
            "score_source",
            "signal_name",
            "hybrid_level",
        ],
        as_index=False,
    ).agg(
        horizons=("horizon_months", "nunique"),
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

    construction_summary = primary.groupby("index_construction", as_index=False).agg(
        primary_predictor_cells=("model_id", "size"),
        mean_rmse_log_change=("rmse_log_change", "mean"),
        mean_mae_log_change=("mae_log_change", "mean"),
        mean_relative_rmse_vs_coverage=(
            "relative_rmse_vs_ar_brent_coverage",
            "mean",
        ),
        mean_oos_r2_vs_coverage=("oos_r2_vs_ar_brent_coverage", "mean"),
        mean_directional_accuracy=("directional_accuracy", "mean"),
    )
    construction_summary["selection_rule"] = "report_both_no_full_sample_winner_selection"
    construction_summary["analysis_role"] = np.where(
        construction_summary["index_construction"].eq(MEAN_SAFE_WEEKLY),
        "previous_monthly_robustness",
        "adapted_pdf_equation_sensitivity",
    )
    return (
        primary_summary,
        dimension_summary.sort_values(
            ["index_construction", "signal_name", "mean_relative_rmse_vs_coverage"]
        ).reset_index(drop=True),
        construction_summary,
    )


def main() -> None:
    args = parse_args()
    if args.minimum_training_rows < 36:
        raise ValueError("minimum-training-rows must be at least 36 for monthly analysis")
    if args.refit_every < 1:
        raise ValueError("refit-every must be positive")
    if args.ridge_alpha <= 0:
        raise ValueError("ridge-alpha must be positive")
    if args.minimum_stacking_history < 12:
        raise ValueError("minimum-stacking-history must be at least 12 months")
    for name in (
        "expected_common_articles",
        "expected_aligned_weekly_articles",
        "expected_complete_month_articles",
        "expected_scfi_complete_months",
        "expected_llm_complete_months",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    np.random.seed(args.seed)

    logger, log_path = configure_logging(args.log_dir.resolve(), args.version)
    logger.info("Starting monthly common-cohort LLM-to-SCFI comparison version=%s", args.version)
    logger.info("Arguments: %s", vars(args))
    logger.info(
        "Monthly index constructions are frozen ex ante: previous=%s adapted_equation=%s; "
        "neither will be selected by full-sample SCFI correlation",
        MEAN_SAFE_WEEKLY,
        PDF_SQRT_EMA3,
    )

    weekly_panel_path = require_file(args.weekly_panel, "Weekly common-cohort panel")
    weekly_manifest_path = require_file(args.weekly_manifest, "Weekly 07b manifest")
    model_config_path = require_file(args.model_config, "Model registry")
    aliases = SUPPORT.load_model_aliases(model_config_path)
    logger.info("Model aliases in registry order: %s", aliases)
    weekly_manifest, provenance_audit = load_and_validate_weekly_manifest(
        weekly_manifest_path,
        aliases,
        args.expected_common_articles,
        weekly_panel_path,
    )
    weekly, weekly_report = load_safe_weekly_panel(
        weekly_panel_path,
        aliases,
        args.expected_aligned_weekly_articles,
    )
    logger.info(
        "Weekly provenance passed: frozen_common=%s aligned_articles=%s rows=%s "
        "complete_rows=%s age_counts=%s",
        provenance_audit["common_complete_articles"],
        weekly_report["aligned_weekly_article_count_sum"],
        weekly_report["rows"],
        weekly_report["complete_weekly_rows"],
        weekly_report["signal_age_counts"],
    )

    monthly_raw, monthly_long, aggregation_report = aggregate_complete_months(
        weekly, aliases
    )
    complete_month_articles = int(
        round(
            float(
                monthly_raw.loc[
                    monthly_raw["monthly_llm_complete"].astype(bool),
                    "llm_common_article_count",
                ].sum()
            )
        )
    )
    if complete_month_articles != args.expected_complete_month_articles:
        raise ValueError(
            "Complete-month article-count mismatch: "
            f"expected {args.expected_complete_month_articles}, found {complete_month_articles}"
        )
    if aggregation_report["scfi_complete_months"] != args.expected_scfi_complete_months:
        raise ValueError(
            "SCFI complete-month mismatch: "
            f"expected {args.expected_scfi_complete_months}, "
            f"found {aggregation_report['scfi_complete_months']}"
        )
    if aggregation_report["llm_complete_months"] != args.expected_llm_complete_months:
        raise ValueError(
            "LLM complete-month mismatch: "
            f"expected {args.expected_llm_complete_months}, "
            f"found {aggregation_report['llm_complete_months']}"
        )
    expected_long_rows = (
        len(monthly_raw)
        * len(MONTHLY_INDEX_METHODS)
        * (len(aliases) + len(SCORE_HYBRID_SOURCES))
        * len(SIGNALS)
    )
    if len(monthly_long) != expected_long_rows:
        raise AssertionError(
            f"Expected {expected_long_rows} monthly signal-audit rows; found {len(monthly_long)}"
        )
    complete_long = monthly_long[
        monthly_long["monthly_analysis_complete_case"].astype(bool)
    ]
    if complete_long["monthly_signal_value"].isna().any():
        raise AssertionError("A complete monthly construction/source/signal value is missing")
    logger.info(
        "Monthly aggregation passed: calendar_months=%s analysis_complete=%s "
        "SCFI_complete=%s LLM_complete=%s complete_month_articles=%s",
        aggregation_report["calendar_months"],
        aggregation_report["analysis_complete_months"],
        aggregation_report["scfi_complete_months"],
        aggregation_report["llm_complete_months"],
        complete_month_articles,
    )
    difference_log = monthly_long[
        monthly_long["index_construction"].eq(PDF_SQRT_EMA3)
        & monthly_long["monthly_analysis_complete_case"].astype(bool)
    ]
    for (source, signal), group in difference_log.groupby(
        ["score_source", "signal_name"], sort=True
    ):
        difference = group["adapted_pdf_minus_mean_safe_weekly"]
        logger.info(
            "Construction difference source=%s signal=%s n=%s mean_signed=%.6f "
            "mean_absolute=%.6f maximum_absolute=%.6f",
            source,
            signal,
            len(difference),
            difference.mean(),
            difference.abs().mean(),
            difference.abs().max(),
        )

    panel, panel_report = prepare_monthly_model_panel(monthly_raw, aliases)
    signal_differences = build_monthly_signal_difference_summary(panel, aliases)
    configurations = build_direct_configurations(aliases)
    feature_sets = build_feature_sets(panel, configurations)
    logger.info(
        "Forecast design: constructions=%s direct_configurations=%s baseline_features=%s "
        "challenger_feature_counts=%s",
        list(MONTHLY_INDEX_METHODS),
        len(configurations),
        len(feature_sets["ridge__ar_brent_coverage"]),
        sorted({len(feature_sets[item.model_id]) for item in configurations}),
    )

    forecast_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        logger.info("Running common-origin monthly forecasts horizon=%sm", horizon)
        forecast_rows.extend(
            benchmark_forecasts(panel, horizon, args.minimum_training_rows)
        )
        for baseline_id in ("ridge__ar_brent", "ridge__ar_brent_coverage"):
            rows, fits = rolling_ridge_forecasts(
                panel=panel,
                horizon=horizon,
                model_id=baseline_id,
                features=feature_sets[baseline_id],
                index_construction="shared_baseline",
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
                "Completed h=%sm model=%s forecasts=%s fits=%s",
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
                index_construction=configuration.index_construction,
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
                "Completed h=%sm construction=%s source=%s signal=%s forecasts=%s fits=%s",
                horizon,
                configuration.index_construction,
                configuration.score_source,
                configuration.signal_name,
                len(rows),
                len(fits),
            )

    direct_forecasts = pd.DataFrame(forecast_rows)
    fits = pd.DataFrame(fit_rows)
    if direct_forecasts.empty:
        raise ValueError("No monthly direct forecasts were produced")
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
        *[item.model_id for item in configurations],
    ]
    origin_audit = validate_common_origin_sets(direct_forecasts, required_direct_ids)
    logger.info("Identical direct monthly origin counts: %s", origin_audit.to_dict(orient="records"))

    hybrid_forecasts, hybrid_weights = build_forecast_hybrids(
        direct_forecasts, aliases, args.minimum_stacking_history
    )
    if hybrid_forecasts.empty:
        raise ValueError("No monthly forecast-level hybrids were produced")
    for column in date_columns:
        hybrid_forecasts[column] = pd.to_datetime(hybrid_forecasts[column])
    if not hybrid_weights.empty:
        weight_sums = hybrid_weights.groupby(
            ["horizon_months", "index_construction", "signal_name", "origin_date"]
        )["weight"].sum()
        if not np.allclose(weight_sums.to_numpy(), 1.0, atol=1e-10):
            raise AssertionError("Monthly past-only stacking weights do not sum to one")
        if (
            pd.to_datetime(hybrid_weights["stacking_history_label_end"])
            > pd.to_datetime(hybrid_weights["origin_date"])
        ).any():
            raise AssertionError("Monthly stacker used a future realized target")

    forecasts = pd.concat(
        [direct_forecasts, hybrid_forecasts], ignore_index=True, sort=False
    ).sort_values(["horizon_months", "origin_date", "model_id"])
    if (
        pd.to_datetime(forecasts["training_label_end_date"])
        > pd.to_datetime(forecasts["model_fit_origin_date"])
    ).any():
        raise AssertionError("At least one monthly forecast used an unavailable label")
    metrics = calculate_metrics(forecasts)
    statistical_tests = forecast_comparison_tests(forecasts, aliases)
    llm_ranking, llm_ranking_cells = build_llm_rankings(
        metrics, statistical_tests, aliases
    )
    primary_predictor_ranking, dimension_summary, construction_summary = (
        build_predictor_summaries(metrics)
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_output = args.panel_output.resolve()
    panel_output.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "monthly_signal_audit": output_dir / f"{args.version}_monthly_signal_audit.csv",
        "signal_differences": output_dir
        / f"{args.version}_signal_difference_summary.csv",
        "monthly_panel": panel_output,
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
        "construction_summary": output_dir
        / f"{args.version}_index_construction_summary.csv",
        "feature_sets": output_dir / f"{args.version}_feature_sets.json",
        "manifest": output_dir / f"{args.version}_manifest.json",
    }
    monthly_long.to_csv(paths["monthly_signal_audit"], index=False)
    signal_differences.to_csv(paths["signal_differences"], index=False)
    panel.to_csv(paths["monthly_panel"], index=False)
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
    construction_summary.to_csv(paths["construction_summary"], index=False)
    paths["feature_sets"].write_text(
        json.dumps(feature_sets, indent=2), encoding="utf-8"
    )
    output_frames = {
        "monthly_signal_audit": monthly_long,
        "signal_differences": signal_differences,
        "monthly_panel": panel,
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
        "construction_summary": construction_summary,
    }
    output_sha256 = {
        key: sha256_file(path)
        for key, path in paths.items()
        if key != "manifest" and path.is_file()
    }
    output_row_counts = {
        key: int(len(frame)) for key, frame in output_frames.items()
    }

    significant_positive_primary = statistical_tests[
        statistical_tests["family_id"].str.startswith("monthly_primary_")
        & statistical_tests["significant_adjusted_0_05"]
        & statistical_tests["mean_loss_improvement"].gt(0)
    ]
    significant_positive_primary_confirmatory = significant_positive_primary[
        significant_positive_primary["test"].eq("diebold_mariano_squared")
    ]
    significant_positive_primary_sensitivity = significant_positive_primary[
        ~significant_positive_primary["test"].eq("diebold_mariano_squared")
    ]
    construction_tests = statistical_tests[
        statistical_tests["family_id"].str.startswith(
            "monthly_primary_construction_pairwise"
        )
    ]
    best_llms = llm_ranking[
        llm_ranking["predictive_point_rank"].eq(1)
    ].to_dict(orient="records")
    manifest = {
        "version": args.version,
        "status": "provisional_pending_human_validation",
        "inputs": {
            "weekly_common_cohort_panel": {
                "path": str(weekly_panel_path),
                "sha256": sha256_file(weekly_panel_path),
                "rows": len(weekly),
            },
            "weekly_manifest": {
                "path": str(weekly_manifest_path),
                "sha256": sha256_file(weekly_manifest_path),
                "version": weekly_manifest.get("version"),
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
        "provenance_audit": provenance_audit,
        "weekly_input_audit": weekly_report,
        "monthly_aggregation_audit": aggregation_report,
        "monthly_panel_audit": panel_report,
        "design": {
            "weekly_analysis_is_primary": True,
            "monthly_analysis_role": "pre_specified_slower_dynamics_robustness",
            "model_aliases": aliases,
            "common_cohort_is_only_score_input": True,
            "frozen_common_articles": args.expected_common_articles,
            "safe_weekly_overlap_articles": args.expected_aligned_weekly_articles,
            "complete_month_articles": complete_month_articles,
            "score_scale": [0, 4],
            "month_assignment": (
                "calendar month of the Monday observation date; the latest prior Friday signal "
                "therefore enters only when it was in the Monday information set"
            ),
            "partial_month_policy": "missing; never zero-filled",
            "index_constructions": {
                MEAN_SAFE_WEEKLY: {
                    "role": "previous_monthly_concept",
                    "formula": (
                        "complete-month arithmetic mean of leakage-safe weekly indices, where "
                        "weekly index = novelty-weighted score sum / log1p(common article count)"
                    ),
                },
                PDF_SQRT_EMA3: {
                    "role": "pre_specified_adapted_equation_sensitivity",
                    "formula": (
                        "U_t = sum_w[I_w*log1p(n_w)]/sqrt(sum_w effective_count_w); "
                        "B_t = 0.5*U_t + 0.5*B_(t-1)"
                    ),
                    "ema": "one-sided span=3, alpha=0.5, adjust=False, restart after missing month",
                    "literal_pdf_replication": False,
                    "adaptations": (
                        "frozen common cohort, novelty-weighted numerator, effective volume, "
                        "safe Monday-month timing"
                    ),
                },
            },
            "construction_selection": (
                "both paths are always reported; no selection by full-sample SCFI correlation"
            ),
            "signals": list(SIGNALS),
            "primary_signals": list(PRIMARY_SIGNALS),
            "target": "forward change in log complete-month mean SCFI",
            "horizons_months": list(HORIZONS),
            "lags_months": list(LAGS),
            "rolling_windows_months": list(ROLLING_WINDOWS),
            "forecast_evaluation": "expanding-window rolling-origin",
            "minimum_training_rows": args.minimum_training_rows,
            "refit_every_origins": args.refit_every,
            "training_label_purge": "training row s allowed only when s + horizon <= origin",
            "preprocessing_fit_inside_training_window": True,
            "random_train_test_split": False,
            "same_algorithm_controls_and_origins_for_both_constructions": True,
            "algorithm": "Ridge",
            "ridge_alpha": args.ridge_alpha,
            "forecast_hybrids": {
                "equal_mean": "equal mean of three LLM forecasts within each construction",
                "past_only_nnls": (
                    "nonnegative weights use only prior OOS forecasts with targets observed "
                    "by the current origin; constructions are never mixed"
                ),
                "minimum_stacking_history": args.minimum_stacking_history,
            },
            "baseline_for_incremental_tests": "monthly AR + Brent + common news coverage",
            "multiple_testing": (
                "Holm across pooled primary monthly construction families; BH-FDR for "
                "exploratory dimension families"
            ),
            "seed": args.seed,
        },
        "origin_audit": origin_audit.to_dict(orient="records"),
        "best_llm_point_rank_by_construction": best_llms,
        "adjusted_significant_positive_primary_confirmatory_dm_squared_tests": int(
            len(significant_positive_primary_confirmatory)
        ),
        "adjusted_significant_positive_primary_sensitivity_tests": int(
            len(significant_positive_primary_sensitivity)
        ),
        "primary_construction_pairwise_tests": int(len(construction_tests)),
        "interpretation_limits": {
            "causal_claim": False,
            "monthly_results_are_secondary_to_weekly": True,
            "adapted_pdf_path_is_not_literal_replication": True,
            "point_rank_does_not_establish_statistical_superiority": True,
            "freight_cost_dimension_price_overlap_risk": True,
            "dimension_paths_are_exploratory": True,
            "past_only_stacker_has_shorter_native_sample": True,
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
        "Monthly forecast rows=%s direct=%s hybrids=%s fit_events=%s tests=%s",
        f"{len(forecasts):,}",
        f"{len(direct_forecasts):,}",
        f"{len(hybrid_forecasts):,}",
        f"{len(fits):,}",
        f"{len(statistical_tests):,}",
    )
    for best in best_llms:
        logger.info(
            "Point-ranked monthly LLM construction=%s model=%s rank=%s "
            "mean_RMSE_rank=%.3f mean_relative_RMSE=%.6f adjusted_pairwise_wins=%s",
            best["index_construction"],
            best["score_source"],
            best["predictive_point_rank"],
            best["mean_rmse_rank"],
            best["mean_relative_rmse_vs_coverage"],
            best["adjusted_dm_squared_pairwise_wins"],
        )
    logger.info(
        "Adjusted significant positive primary tests: confirmatory_DM_squared=%s "
        "sensitivity=%s; construction paths were not selected using full-sample SCFI association",
        len(significant_positive_primary_confirmatory),
        len(significant_positive_primary_sensitivity),
    )
    logger.info("Monthly construction summary: %s", paths["construction_summary"])
    logger.info("Monthly forecast metrics: %s", paths["metrics"])
    logger.info("Monthly statistical tests: %s", paths["statistical_tests"])
    logger.info("Monthly common-cohort comparison completed successfully")


if __name__ == "__main__":
    main()
