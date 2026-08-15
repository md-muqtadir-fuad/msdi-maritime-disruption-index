"""Build leakage-safe weekly and monthly MSDI–SCFI modeling panels.

The panel keeps both pre-specified news-label paths:
* peak severity: median cross-model ``overall_disruption_max``;
* disruption breadth: equal-weight mean of seven dimension medians;
* non-freight peak severity and price-dominance indicators;
* full set of core controls, benchmark series, and external shock indicators.

SCFI is Monday-dated in the supplied legacy workbook. Friday-ended MSDI,
Brent, and external controls are attached with backward-only as-of joins.
Every matched feature carries its source date and age, and future-dated
matches are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MSDI = PROJECT_ROOT / "data/05_indices/weekly/msdi_weekly.csv"
DEFAULT_MONTHLY_MSDI = PROJECT_ROOT / "data/05_indices/monthly/msdi_monthly.csv"
DEFAULT_SCFI = PROJECT_ROOT / "data/01_raw/freight/scfi/freight_index_data_legacy.xlsx"
DEFAULT_BRENT = PROJECT_ROOT / "data/01_raw/controls/brent/brent_weekly_legacy.csv"
DEFAULT_EXTERNAL_CONTROLS = (
    PROJECT_ROOT / "data/02_interim/controls/external_controls_clean_long.csv"
)

EXTERNAL_CONTROL_COLUMNS = (
    # Core Economic Controls
    "cpb_world_trade_volume",
    "rwi_isl_container_throughput_index",
    "wto_china_merchandise_exports",
    "cny_usd_exchange_rate",
    "unctad_china_core_lane_lsbci",
    # Benchmark / Restricted Controls
    "gscpi",
    "imf_portwatch_global_container_calls",
    # External-Shock Sensitivity
    "gpr_index",
    "tpu_index",
    "uct_index",
    "opu_index",
    "infectious_disease_emv",
)

CORE_ECONOMIC_CONTROLS = (
    "cpb_world_trade_volume",
    "rwi_isl_container_throughput_index",
    "wto_china_merchandise_exports",
    "cny_usd_exchange_rate",
    "unctad_china_core_lane_lsbci",
)

SHOCK_SENSITIVITY_CONTROLS = (
    "gpr_index",
    "tpu_index",
    "uct_index",
    "opu_index",
    "infectious_disease_emv",
)

DEFAULT_WEEKLY_OUTPUT = (
    PROJECT_ROOT / "data/06_model_ready/prediction/msdi_scfi_weekly_panel.csv"
)
DEFAULT_MONTHLY_OUTPUT = (
    PROJECT_ROOT / "data/06_model_ready/causality/msdi_scfi_monthly_panel.csv"
)
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "outputs/model_panel_audit"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs/logs"

FORECAST_HORIZONS_WEEKS = (1, 2, 4, 6)
FORECAST_HORIZONS_MONTHS = (1, 3, 6)
PREDICTOR_LAGS_WEEKS = (1, 2, 4, 8, 12)
PREDICTOR_LAGS_MONTHS = (1, 2, 3, 6)
ROLLING_WINDOWS_WEEKS = (4, 13, 26, 52)

MSDI_BASE_COLUMNS = [
    "article_count",
    "event_count",
    "source_count",
    "effective_article_count",
    "peak_active_article_count",
    "breadth_active_article_count",
    "peak_score_sum",
    "breadth_score_sum",
    "peak_score_unweighted_mean",
    "breadth_score_unweighted_mean",
    "peak_model_disagreement_mean",
    "dimension_disagreement_mean",
    "peak_score_media_corrected",
    "breadth_score_media_corrected",
    "non_freight_peak_score_media_corrected",
    "freight_price_only_share",
    "freight_price_dominant_share",
    "peak_msdi_z_expanding_realtime",
    "breadth_msdi_z_expanding_realtime",
    "port_disruption_media_corrected",
    "global_container_movement_disruption_media_corrected",
    "freight_cost_disruption_media_corrected",
    "supply_disruption_media_corrected",
    "production_disruption_media_corrected",
    "covid19_related_disruption_media_corrected",
    "demand_disruption_media_corrected",
]
PRIMARY_PREDICTORS = ["peak_score_media_corrected", "breadth_score_media_corrected"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build weekly forecasting and monthly causal-analysis panels."
    )
    parser.add_argument("--msdi", type=Path, default=DEFAULT_MSDI)
    parser.add_argument("--monthly-msdi", type=Path, default=DEFAULT_MONTHLY_MSDI)
    parser.add_argument("--scfi", type=Path, default=DEFAULT_SCFI)
    parser.add_argument("--scfi-sheet", default="Sheet1")
    parser.add_argument("--brent", type=Path, default=DEFAULT_BRENT)
    parser.add_argument("--external-controls", type=Path, default=DEFAULT_EXTERNAL_CONTROLS)
    parser.add_argument("--weekly-output", type=Path, default=DEFAULT_WEEKLY_OUTPUT)
    parser.add_argument("--monthly-output", type=Path, default=DEFAULT_MONTHLY_OUTPUT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--version", default="panel_v002")
    parser.add_argument(
        "--maximum-feature-age-days",
        type=int,
        default=6,
        help="Maximum backward gap allowed when attaching Friday features to Monday SCFI.",
    )
    return parser.parse_args()


def configure_logging(log_dir: Path, version: str) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"06_build_model_panels_{version}.log"
    logger = logging.getLogger("build_model_panels")
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


def normalize_column_name(value: object) -> str:
    return "_".join(str(value).strip().lower().replace("-", " ").split())


def find_column(columns: list[str], candidates: list[str], label: str) -> str:
    normalized = {normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        if normalize_column_name(candidate) in normalized:
            return normalized[normalize_column_name(candidate)]
    raise ValueError(f"Could not find {label}; available columns: {columns}")


def ensure_unique_dates(frame: pd.DataFrame, date_column: str, label: str) -> None:
    duplicates = frame[date_column].notna() & frame[date_column].duplicated(keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, [date_column]].head(10)
        raise ValueError(f"{label} has duplicate dates:\n{examples.to_string(index=False)}")


def load_scfi(path: Path, sheet_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = require_file(path, "SCFI workbook")
    raw = pd.read_excel(resolved, sheet_name=sheet_name)
    date_column = find_column(
        raw.columns.astype(str).tolist(), ["weekly date", "date", "observation_date"], "SCFI date"
    )
    value_column = find_column(
        raw.columns.astype(str).tolist(),
        ["containerized freight index", "scfi", "freight_index"],
        "SCFI value",
    )
    frame = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(raw[date_column], format="mixed", errors="coerce"),
            "scfi": pd.to_numeric(raw[value_column], errors="coerce"),
        }
    )
    invalid_date = int(frame["observation_date"].isna().sum())
    invalid_value = int(frame["scfi"].isna().sum())
    frame = frame.dropna(subset=["observation_date"]).sort_values("observation_date").reset_index(drop=True)
    ensure_unique_dates(frame, "observation_date", "SCFI")
    if frame["scfi"].le(0).any():
        raise ValueError("SCFI must be strictly positive for log-return targets")
    report = {
        "input_rows": len(raw),
        "usable_rows": len(frame),
        "invalid_dates": invalid_date,
        "missing_values": invalid_value,
        "first_date": frame["observation_date"].min(),
        "last_date": frame["observation_date"].max(),
        "weekday_counts": frame["observation_date"].dt.day_name().value_counts().to_dict(),
        "gap_days_counts": frame["observation_date"].diff().dt.days.value_counts().sort_index().to_dict(),
    }
    return frame, report


def load_msdi(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = require_file(path, "Weekly MSDI file")
    raw = pd.read_csv(resolved, low_memory=False)
    if "week_ending" not in raw.columns:
        raise ValueError("MSDI file lacks week_ending")
    available_cols = [c for c in MSDI_BASE_COLUMNS if c in raw.columns]
    frame = raw[["week_ending", *available_cols]].copy()
    frame["msdi_week_ending"] = pd.to_datetime(frame.pop("week_ending"), format="mixed", errors="coerce")
    for column in available_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid_date = int(frame["msdi_week_ending"].isna().sum())
    frame = frame.dropna(subset=["msdi_week_ending"]).sort_values("msdi_week_ending").reset_index(drop=True)
    ensure_unique_dates(frame, "msdi_week_ending", "MSDI")
    report = {
        "input_rows": len(raw),
        "usable_rows": len(frame),
        "invalid_dates": invalid_date,
        "first_date": frame["msdi_week_ending"].min(),
        "last_date": frame["msdi_week_ending"].max(),
        "columns_loaded": available_cols,
    }
    return frame, report


def load_monthly_msdi(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = require_file(path, "Monthly MSDI file")
    raw = pd.read_csv(resolved, low_memory=False)
    if "month_end" not in raw.columns:
        raise ValueError("Monthly MSDI file lacks month_end")
    forbidden = [column for column in raw if "z_full_sample" in column]
    keep = [column for column in raw.columns if column == "month_end" or "z_full_sample" not in column]
    frame = raw[keep].copy()
    frame["month_ending"] = pd.to_datetime(frame.pop("month_end"), format="mixed", errors="coerce")
    for column in frame.columns:
        if column != "month_ending":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid_date = int(frame["month_ending"].isna().sum())
    frame = frame.dropna(subset=["month_ending"]).sort_values("month_ending").reset_index(drop=True)
    ensure_unique_dates(frame, "month_ending", "Monthly MSDI")
    report = {
        "input_rows": len(raw),
        "usable_rows": len(frame),
        "invalid_dates": invalid_date,
        "first_date": frame["month_ending"].min(),
        "last_date": frame["month_ending"].max(),
        "excluded_full_sample_standardized_columns": forbidden,
        "construction": "direct_article_monthly_sqrt_effective_volume_causal_ema3",
    }
    return frame, report


def load_brent(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = require_file(path, "Brent file")
    raw = pd.read_csv(resolved, low_memory=False)
    date_column = find_column(
        raw.columns.astype(str).tolist(), ["observation_date", "date"], "Brent date"
    )
    value_column = find_column(
        raw.columns.astype(str).tolist(), ["DCOILBRENTEU", "brent", "brent_price"], "Brent value"
    )
    frame = pd.DataFrame(
        {
            "brent_observation_date": pd.to_datetime(
                raw[date_column], format="mixed", errors="coerce"
            ),
            "brent_price": pd.to_numeric(raw[value_column], errors="coerce"),
        }
    )
    invalid_date = int(frame["brent_observation_date"].isna().sum())
    missing_value = int(frame["brent_price"].isna().sum())
    frame = frame.dropna(subset=["brent_observation_date"]).sort_values(
        "brent_observation_date"
    ).reset_index(drop=True)
    ensure_unique_dates(frame, "brent_observation_date", "Brent")
    report = {
        "input_rows": len(raw),
        "usable_rows": len(frame),
        "invalid_dates": invalid_date,
        "missing_values": missing_value,
        "first_date": frame["brent_observation_date"].min(),
        "last_date": frame["brent_observation_date"].max(),
    }
    return frame, report


def backward_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_date: str,
    right_date: str,
    maximum_age_days: int,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = pd.merge_asof(
        left.sort_values(left_date),
        right.sort_values(right_date),
        left_on=left_date,
        right_on=right_date,
        direction="backward",
        tolerance=pd.Timedelta(days=maximum_age_days),
        allow_exact_matches=True,
    )
    age_column = f"{label}_age_days"
    result[age_column] = (result[left_date] - result[right_date]).dt.days
    future = result[right_date].notna() & result[right_date].gt(result[left_date])
    if future.any():
        raise AssertionError(f"Backward {label} join produced {int(future.sum())} future matches")
    matched = result[right_date].notna()
    report = {
        "rows": len(result),
        "matched": int(matched.sum()),
        "unmatched": int((~matched).sum()),
        "future_matches": int(future.sum()),
        "maximum_allowed_age_days": maximum_age_days,
        "age_days_summary": result.loc[matched, age_column].describe().to_dict(),
        "age_days_counts": result.loc[matched, age_column].value_counts().sort_index().to_dict(),
    }
    return result, report


def safe_log(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return np.log(numeric.where(numeric.gt(0)))


def load_external_controls(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(require_file(path, "Cleaned external controls"), low_memory=False)
    required = {"variable", "availability_date", "value", "source_frequency"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"External controls lack required columns: {sorted(missing)}")
    raw["availability_date"] = pd.to_datetime(raw["availability_date"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.loc[raw["variable"].isin(EXTERNAL_CONTROL_COLUMNS)].dropna(
        subset=["availability_date", "value"]
    )
    if raw.duplicated(["variable", "availability_date"]).any():
        raw = raw.drop_duplicates(["variable", "availability_date"], keep="last")
    return raw.sort_values(["variable", "availability_date"]), {
        "rows": len(raw),
        "variables": sorted(raw.variable.unique().tolist()),
        "first_available": str(raw.availability_date.min().date()),
        "last_available": str(raw.availability_date.max().date()),
    }


def attach_external_controls(
    panel: pd.DataFrame, controls: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach only released values; backward-only joins with conservative age limits."""
    result = panel.sort_values("observation_date").copy()
    audit: dict[str, Any] = {}
    maximum_age = {"daily": 7, "monthly": 75, "quarterly": 180, "annual": 455}
    for variable, group in controls.groupby("variable", sort=True):
        source = group[["availability_date", "value", "source_frequency"]].sort_values(
            "availability_date"
        )
        joined = pd.merge_asof(
            result[["observation_date"]],
            source,
            left_on="observation_date",
            right_on="availability_date",
            direction="backward",
            allow_exact_matches=True,
        )
        age = (result["observation_date"] - joined["availability_date"]).dt.days
        freq = str(group["source_frequency"].iloc[0])
        threshold = int(maximum_age.get(freq, 75))
        stale = joined["availability_date"].notna() & age.gt(threshold)
        if stale.any():
            joined.loc[stale, "value"] = np.nan
        future = joined["availability_date"].notna() & joined["availability_date"].gt(
            result["observation_date"]
        )
        if future.any():
            raise AssertionError(f"Future external-control match: {variable}")
        result[variable] = joined["value"].to_numpy()
        result[f"{variable}_availability_date"] = joined["availability_date"].to_numpy()
        result[f"{variable}_age_days"] = age.to_numpy()
        audit[variable] = {
            "frequency": freq,
            "maximum_age_days": threshold,
            "matched": int(joined["availability_date"].notna().sum()),
            "stale_rejected": int(stale.sum()),
            "future_matches": int(future.sum()),
        }
    for variable in EXTERNAL_CONTROL_COLUMNS:
        if variable not in result.columns:
            result[variable] = np.nan
            result[f"{variable}_availability_date"] = pd.NaT
            result[f"{variable}_age_days"] = np.nan
            audit[variable] = {
                "frequency": None,
                "maximum_age_days": None,
                "matched": 0,
                "stale_rejected": 0,
                "future_matches": 0,
                "status": "not_supplied",
            }
    return result, audit


def add_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    iso_week = result["observation_date"].dt.isocalendar().week.astype(float)
    month = result["observation_date"].dt.month.astype(float)
    result["calendar_year"] = result["observation_date"].dt.year
    result["calendar_month"] = month.astype(int)
    result["calendar_iso_week"] = iso_week.astype(int)
    result["calendar_week_sin"] = np.sin(2 * np.pi * iso_week / 52.1775)
    result["calendar_week_cos"] = np.cos(2 * np.pi * iso_week / 52.1775)
    result["calendar_month_sin"] = np.sin(2 * np.pi * month / 12)
    result["calendar_month_cos"] = np.cos(2 * np.pi * month / 12)
    return result


def add_weekly_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result = panel.copy()
    dictionary: list[dict[str, Any]] = []

    result["scfi_log"] = safe_log(result["scfi"])
    result["brent_log"] = safe_log(result["brent_price"])
    dictionary.extend(
        [
            feature_entry("scfi", "outcome/history", "Raw Monday-dated SCFI level", "raw"),
            feature_entry("scfi_log", "outcome/history", "Natural log of current SCFI", "current"),
            feature_entry("brent_price", "control", "Latest backward-matched Brent price", "current"),
            feature_entry("brent_log", "control", "Natural log of Brent price", "current"),
        ]
    )

    for lag in PREDICTOR_LAGS_WEEKS:
        result[f"scfi_log_lag_{lag}w"] = result["scfi_log"].shift(lag)
        result[f"brent_log_lag_{lag}w"] = result["brent_log"].shift(lag)
        dictionary.append(
            feature_entry(
                f"scfi_log_lag_{lag}w",
                "autoregressive predictor",
                f"SCFI log lagged {lag} rows/weeks",
                f"t-{lag}",
            )
        )
        dictionary.append(
            feature_entry(
                f"brent_log_lag_{lag}w", "control", f"Brent log lagged {lag} rows/weeks", f"t-{lag}"
            )
        )
        for predictor in PRIMARY_PREDICTORS:
            if predictor in result.columns:
                column = f"{predictor}_lag_{lag}w"
                result[column] = result[predictor].shift(lag)
                dictionary.append(
                    feature_entry(column, "MSDI predictor", f"{predictor} lagged {lag} rows/weeks", f"t-{lag}")
                )
        for control in EXTERNAL_CONTROL_COLUMNS:
            column = f"{control}_lag_{lag}w"
            result[column] = result[control].shift(lag)
            dictionary.append(
                feature_entry(column, "external control", f"Released {control} lagged {lag} weeks", f"t-{lag}")
            )

    for horizon in (1, 4, 13, 26, 52):
        result[f"scfi_log_change_{horizon}w"] = result["scfi_log"].diff(horizon)
        result[f"brent_log_change_{horizon}w"] = result["brent_log"].diff(horizon)

    for horizon in FORECAST_HORIZONS_WEEKS:
        future = result["scfi_log"].shift(-horizon)
        target = f"target_scfi_log_change_h{horizon}w"
        result[target] = future - result["scfi_log"]
        description = f"Forward SCFI log return over {horizon} weeks"
        dictionary.append(feature_entry(target, "forecast target", description, f"t+{horizon}"))

        target_level = f"target_scfi_level_h{horizon}w"
        result[target_level] = result["scfi"].shift(-horizon)
        dictionary.append(feature_entry(target_level, "forecast level target", f"Forward SCFI level at t+{horizon} weeks", f"t+{horizon}"))

    result = add_calendar_features(result)
    return result, dictionary


def feature_entry(name: str, role: str, definition: str, timing: str) -> dict[str, Any]:
    return {
        "variable": name,
        "role": role,
        "definition": definition,
        "information_timing": timing,
    }


def build_weekly_panel(
    scfi: pd.DataFrame,
    msdi: pd.DataFrame,
    brent: pd.DataFrame,
    external_controls: pd.DataFrame,
    maximum_age_days: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    panel, msdi_join = backward_join(
        scfi,
        msdi,
        "observation_date",
        "msdi_week_ending",
        maximum_age_days,
        "msdi",
    )
    panel, brent_join = backward_join(
        panel,
        brent,
        "observation_date",
        "brent_observation_date",
        maximum_age_days,
        "brent",
    )
    panel, external_join = attach_external_controls(panel, external_controls)
    panel, dictionary = add_weekly_features(panel)
    panel["has_msdi_predictors"] = panel[PRIMARY_PREDICTORS].notna().all(axis=1)
    panel["has_brent_control"] = panel["brent_price"].notna()
    panel["base_model_complete_case"] = panel[
        ["scfi", "peak_score_media_corrected", "breadth_score_media_corrected", "brent_price"]
    ].notna().all(axis=1)
    report = {
        "msdi_join": msdi_join,
        "brent_join": brent_join,
        "external_control_join": external_join,
        "weekly_rows": len(panel),
        "rows_with_both_msdi_paths": int(panel["has_msdi_predictors"].sum()),
        "base_model_complete_rows": int(panel["base_model_complete_case"].sum()),
        "first_complete_date": panel.loc[
            panel["base_model_complete_case"], "observation_date"
        ].min(),
        "last_complete_date": panel.loc[
            panel["base_model_complete_case"], "observation_date"
        ].max(),
    }
    return panel, dictionary, report


def build_monthly_panel(
    weekly: pd.DataFrame, monthly_msdi: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    work = weekly.copy()
    for control in EXTERNAL_CONTROL_COLUMNS:
        if control not in work.columns:
            work[control] = np.nan
    work["weeks_observed_flag"] = 1
    work["msdi_available_flag"] = work[PRIMARY_PREDICTORS].notna().all(axis=1).astype(int)
    work = work.set_index("observation_date")
    aggregations: dict[str, Any] = {
        "scfi": ["mean", "last"],
        "brent_price": ["mean", "last"],
        "peak_score_media_corrected": ["mean", "max", "sum"],
        "breadth_score_media_corrected": ["mean", "max", "sum"],
        "article_count": "sum",
        "event_count": "sum",
        "source_count": "mean",
        "peak_model_disagreement_mean": "mean",
        "dimension_disagreement_mean": "mean",
        "weeks_observed_flag": "sum",
        "msdi_available_flag": "sum",
    }
    for control in EXTERNAL_CONTROL_COLUMNS:
        aggregations[control] = "last"
    for column in MSDI_BASE_COLUMNS:
        if column.endswith("_media_corrected") and column not in aggregations and column in work.columns:
            aggregations[column] = "mean"

    monthly = work.resample("ME").agg(aggregations)
    monthly.columns = [
        "_".join([str(part) for part in column if str(part)])
        if isinstance(column, tuple)
        else str(column)
        for column in monthly.columns
    ]
    monthly = monthly.rename(
        columns={
            "scfi_mean": "scfi_monthly_mean",
            "scfi_last": "scfi_month_end",
            "brent_price_mean": "brent_monthly_mean",
            "brent_price_last": "brent_month_end",
            "weeks_observed_flag_sum": "weekly_observations_in_month",
            "msdi_available_flag_sum": "msdi_available_weeks",
            **{f"{control}_last": control for control in EXTERNAL_CONTROL_COLUMNS},
        }
    )
    no_msdi = monthly["msdi_available_weeks"].eq(0)
    msdi_monthly_columns = [
        column
        for column in monthly.columns
        if any(column.startswith(f"{source}_") for source in MSDI_BASE_COLUMNS)
    ]
    monthly.loc[no_msdi, msdi_monthly_columns] = np.nan
    monthly = monthly.reset_index().rename(columns={"observation_date": "month_ending"})

    # Core SCFI & Brent transforms
    monthly["scfi_monthly_mean_log"] = safe_log(monthly["scfi_monthly_mean"])
    monthly["scfi_month_end_log"] = safe_log(monthly["scfi_month_end"])
    monthly["brent_monthly_mean_log"] = safe_log(monthly["brent_monthly_mean"])
    monthly["scfi_monthly_mean_log_change_1m"] = monthly["scfi_monthly_mean_log"].diff()
    monthly["scfi_monthly_mean_log_change_12m"] = monthly["scfi_monthly_mean_log"].diff(12)
    monthly["brent_monthly_mean_log_change_1m"] = monthly["brent_monthly_mean_log"].diff()
    monthly["brent_monthly_mean_log_change_12m"] = monthly["brent_monthly_mean_log"].diff(12)

    # Transform all external economic series (Level, Log, Monthly Change, YoY Change, Lags 1, 2, 3, 6)
    for control in EXTERNAL_CONTROL_COLUMNS:
        series = monthly[control]
        # Log level where positive
        if series.dropna().gt(0).all():
            monthly[f"{control}_log"] = safe_log(series)
            monthly[f"{control}_log_change_1m"] = monthly[f"{control}_log"].diff()
            monthly[f"{control}_log_change_12m"] = monthly[f"{control}_log"].diff(12)
        else:
            monthly[f"{control}_change_1m"] = series.diff()
            monthly[f"{control}_change_12m"] = series.diff(12)

        for lag in PREDICTOR_LAGS_MONTHS:
            monthly[f"{control}_lag_{lag}m"] = series.shift(lag)
            if f"{control}_log" in monthly.columns:
                monthly[f"{control}_log_lag_{lag}m"] = monthly[f"{control}_log"].shift(lag)
                monthly[f"{control}_log_change_1m_lag_{lag}m"] = monthly[f"{control}_log_change_1m"].shift(lag)
            elif f"{control}_change_1m" in monthly.columns:
                monthly[f"{control}_change_1m_lag_{lag}m"] = monthly[f"{control}_change_1m"].shift(lag)

    # Merge direct monthly MSDI
    if monthly_msdi is not None:
        legacy_msdi_columns = [
            column
            for column in monthly.columns
            if any(column.startswith(f"{source}_") for source in MSDI_BASE_COLUMNS)
        ]
        monthly = monthly.drop(columns=legacy_msdi_columns, errors="ignore")
        direct = monthly_msdi.copy()
        direct_rename = {
            "article_count": "article_count_sum",
            "event_count": "event_count_sum",
            "source_count": "source_count_mean",
            "effective_article_count": "effective_article_count_sum",
            "peak_model_disagreement_mean": "peak_model_disagreement_mean_mean",
            "dimension_disagreement_mean": "dimension_disagreement_mean_mean",
            "peak_score_media_corrected": "peak_score_media_corrected_mean",
            "breadth_score_media_corrected": "breadth_score_media_corrected_mean",
            "non_freight_peak_score_media_corrected": "non_freight_peak_score_media_corrected_mean",
        }
        for column in direct.columns:
            if column.endswith("_media_corrected") and column not in direct_rename:
                direct_rename[column] = f"{column}_mean"
        direct = direct.rename(columns=direct_rename)
        direct["direct_monthly_msdi_available"] = direct[
            ["peak_score_media_corrected_mean", "breadth_score_media_corrected_mean"]
        ].notna().all(axis=1).astype(int)
        monthly = monthly.merge(direct, on="month_ending", how="left", validate="one_to_one")

        # Compute MSDI monthly changes and lags (1, 2, 3, 6)
        msdi_targets = [
            "peak_score_media_corrected_mean",
            "breadth_score_media_corrected_mean",
            "non_freight_peak_score_media_corrected_mean",
            "article_count_sum",
            "event_count_sum",
            "source_count_mean",
        ]
        for stem in [
            "port_disruption_media_corrected_mean",
            "global_container_movement_disruption_media_corrected_mean",
            "freight_cost_disruption_media_corrected_mean",
            "supply_disruption_media_corrected_mean",
            "production_disruption_media_corrected_mean",
            "covid19_related_disruption_media_corrected_mean",
            "demand_disruption_media_corrected_mean",
        ]:
            if stem in monthly.columns:
                msdi_targets.append(stem)

        for col in msdi_targets:
            if col in monthly.columns:
                monthly[f"{col}_change_1m"] = monthly[col].diff()
                monthly[f"{col}_change_12m"] = monthly[col].diff(12)
                for lag in PREDICTOR_LAGS_MONTHS:
                    monthly[f"{col}_lag_{lag}m"] = monthly[col].shift(lag)
                    monthly[f"{col}_change_1m_lag_{lag}m"] = monthly[f"{col}_change_1m"].shift(lag)

    # Add autoregressive SCFI and Brent lags
    for lag in PREDICTOR_LAGS_MONTHS:
        monthly[f"scfi_monthly_mean_log_lag_{lag}m"] = monthly["scfi_monthly_mean_log"].shift(lag)
        monthly[f"scfi_monthly_mean_log_change_1m_lag_{lag}m"] = monthly["scfi_monthly_mean_log_change_1m"].shift(lag)
        monthly[f"brent_monthly_mean_log_lag_{lag}m"] = monthly["brent_monthly_mean_log"].shift(lag)
        monthly[f"brent_monthly_mean_log_change_1m_lag_{lag}m"] = monthly["brent_monthly_mean_log_change_1m"].shift(lag)

    dictionary: list[dict[str, Any]] = [
        feature_entry("month_ending", "key", "Calendar month end", "t"),
        feature_entry("scfi_monthly_mean", "outcome", "Mean weekly SCFI within month", "t"),
        feature_entry("scfi_month_end", "outcome", "Last observed weekly SCFI within month", "t"),
        feature_entry(
            "peak_score_media_corrected_mean",
            "MSDI treatment",
            "Direct monthly peak MSDI: sqrt effective volume then causal EMA(3)",
            "within t",
        ),
        feature_entry(
            "breadth_score_media_corrected_mean",
            "MSDI treatment",
            "Direct monthly breadth MSDI: sqrt effective volume then causal EMA(3)",
            "within t",
        ),
        feature_entry(
            "non_freight_peak_score_media_corrected_mean",
            "MSDI treatment",
            "Direct monthly non-freight peak MSDI",
            "within t",
        ),
    ]
    for horizon in FORECAST_HORIZONS_MONTHS:
        future = monthly["scfi_monthly_mean"].shift(-horizon)
        target = f"target_scfi_monthly_log_change_h{horizon}m"
        monthly[target] = safe_log(future) - monthly["scfi_monthly_mean_log"]
        dictionary.append(
            feature_entry(
                target,
                "forecast/causal horizon outcome",
                f"Forward mean-SCFI log change over {horizon} months",
                f"t+{horizon}",
            )
        )
        target_mean = f"target_scfi_monthly_mean_h{horizon}m"
        monthly[target_mean] = future
        dictionary.append(
            feature_entry(
                target_mean,
                "forecast level target",
                f"Forward mean-SCFI level at t+{horizon} months",
                f"t+{horizon}",
            )
        )
        target_level = f"target_scfi_monthly_level_h{horizon}m"
        monthly[target_level] = future
        dictionary.append(
            feature_entry(
                target_level,
                "forecast level target",
                f"Forward mean-SCFI level at t+{horizon} months",
                f"t+{horizon}",
            )
        )
    return monthly, dictionary


def missingness_table(frame: pd.DataFrame, panel_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "panel": panel_name,
            "variable": frame.columns,
            "dtype": [str(dtype) for dtype in frame.dtypes],
            "missing_count": frame.isna().sum().to_numpy(),
            "missing_percent": (100 * frame.isna().mean()).to_numpy(),
            "nonmissing_count": frame.notna().sum().to_numpy(),
        }
    )


def write_data_dictionary(
    entries: list[dict[str, Any]], frame: pd.DataFrame, path: Path
) -> None:
    by_name = {entry["variable"]: entry for entry in entries}
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        entry = by_name.get(
            column,
            feature_entry(column, "derived/diagnostic", "Derived panel variable", "current or past only"),
        ).copy()
        entry["dtype"] = str(frame[column].dtype)
        entry["missing_count"] = int(frame[column].isna().sum())
        rows.append(entry)
    pd.DataFrame(rows).to_csv(path, index=False)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def main() -> None:
    args = parse_args()
    logger, log_path = configure_logging(args.log_dir, args.version)
    logger.info("Starting model panel builder version=%s", args.version)

    scfi_df, scfi_audit = load_scfi(args.scfi, args.scfi_sheet)
    msdi_df, msdi_audit = load_msdi(args.msdi)
    brent_df, brent_audit = load_brent(args.brent)
    ext_df, ext_audit = load_external_controls(args.external_controls)
    monthly_msdi_df, monthly_msdi_audit = (
        load_monthly_msdi(args.monthly_msdi) if args.monthly_msdi.is_file() else (None, {})
    )

    weekly_panel, weekly_dict, weekly_report = build_weekly_panel(
        scfi_df, msdi_df, brent_df, ext_df, args.maximum_feature_age_days
    )
    monthly_panel, monthly_dict = build_monthly_panel(weekly_panel, monthly_msdi_df)

    args.weekly_output.parent.mkdir(parents=True, exist_ok=True)
    args.monthly_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    weekly_panel.to_csv(args.weekly_output, index=False)
    monthly_panel.to_csv(args.monthly_output, index=False)

    write_data_dictionary(
        weekly_dict, weekly_panel, args.audit_dir / f"{args.version}_weekly_data_dictionary.csv"
    )
    write_data_dictionary(
        monthly_dict, monthly_panel, args.audit_dir / f"{args.version}_monthly_data_dictionary.csv"
    )

    missing_df = pd.concat(
        [
            missingness_table(weekly_panel, "weekly"),
            missingness_table(monthly_panel, "monthly"),
        ],
        ignore_index=True,
    )
    missing_df.to_csv(args.audit_dir / f"{args.version}_missingness_audit.csv", index=False)

    manifest = {
        "version": args.version,
        "files": {
            "weekly_panel": str(args.weekly_output.resolve()),
            "monthly_panel": str(args.monthly_output.resolve()),
            "weekly_rows": len(weekly_panel),
            "monthly_rows": len(monthly_panel),
        },
        "scfi_audit": scfi_audit,
        "msdi_audit": msdi_audit,
        "brent_audit": brent_audit,
        "external_controls_audit": ext_audit,
        "monthly_msdi_audit": monthly_msdi_audit,
        "weekly_report": weekly_report,
    }
    manifest_path = args.audit_dir / f"{args.version}_panel_build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
    logger.info("Panel build completed successfully; wrote %s and %s", args.weekly_output, args.monthly_output)


if __name__ == "__main__":
    main()
