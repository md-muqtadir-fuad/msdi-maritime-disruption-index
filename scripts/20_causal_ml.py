"""Leakage-safe, time-aware causal-machine-learning analysis.

The stage estimates partially linear Double/Debiased ML (DML) models with
expanding-window cross-fitting and a horizon purge. Weekly estimates are
monthly-primary; weekly estimates are higher-frequency robustness checks. The treatment nuisance is
learned from the raw first difference of each MSDI path, so the resulting
innovation is genuinely out of fold. Full-sample innovations produced by
``19_time_series_causality.py`` are used only for a transparent comparison audit.

These are observational estimates. Unless an independent identification design
is supplied and defended, outputs are labelled orthogonalized predictive
effects rather than structural causal effects.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/causal_ml.yaml"
DEFAULT_INNOVATIONS = (
    PROJECT_ROOT
    / "outputs/time_series_causality/causality_v001_msdi_innovations.csv"
)
DEFAULT_LOCAL_PROJECTIONS = (
    PROJECT_ROOT
    / "outputs/time_series_causality/causality_v001_local_projections.csv"
)
DEFAULT_CAUSALITY_MANIFEST = (
    PROJECT_ROOT / "outputs/time_series_causality/causality_v001_manifest.json"
)
DEFAULT_STATIONARITY_TESTS = (
    PROJECT_ROOT / "outputs/time_series_causality/causality_v001_stationarity_tests.csv"
)
DEFAULT_VAR_LAG_SELECTION = (
    PROJECT_ROOT / "outputs/time_series_causality/causality_v001_var_lag_selection.csv"
)
DEFAULT_VAR_DIAGNOSTICS = (
    PROJECT_ROOT / "outputs/time_series_causality/causality_v001_var_diagnostics.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/causal_ml"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs/logs"
LEARNERS = (
    "histogram_gradient_boosting",
    "elastic_net",
    "random_forest",
)
MSDI_PATHS = ("peak", "breadth")
OPTIONAL_COLUMN_MAP = {
    "weekly": {
        "peak_model_disagreement": "peak_model_disagreement_mean",
        "dimension_disagreement": "dimension_disagreement_mean",
        "port_disruption": "port_disruption_media_corrected",
        "global_container_movement_disruption": (
            "global_container_movement_disruption_media_corrected"
        ),
        "supply_disruption": "supply_disruption_media_corrected",
        "production_disruption": "production_disruption_media_corrected",
        "demand_disruption": "demand_disruption_media_corrected",
        "cpb_world_trade_volume": "cpb_world_trade_volume",
        "unctad_china_core_lane_lsbci": "unctad_china_core_lane_lsbci",
    },
    "monthly": {
        "peak_model_disagreement": "peak_model_disagreement_mean_mean",
        "dimension_disagreement": "dimension_disagreement_mean_mean",
        "port_disruption": "port_disruption_media_corrected_mean",
        "global_container_movement_disruption": (
            "global_container_movement_disruption_media_corrected_mean"
        ),
        "supply_disruption": "supply_disruption_media_corrected_mean",
        "production_disruption": "production_disruption_media_corrected_mean",
        "demand_disruption": "demand_disruption_media_corrected_mean",
        "cpb_world_trade_volume": "cpb_world_trade_volume",
        "unctad_china_core_lane_lsbci": "unctad_china_core_lane_lsbci",
    },
}
HETEROGENEITY_MODIFIERS = (
    "covid_regime",
    "lagged_market_volatility_regime",
    "lagged_news_intensity_regime",
    "lagged_model_disagreement_regime",
)


@dataclass(frozen=True)
class FrequencySpecification:
    frequency: str
    role: str
    panel_path: Path
    date_column: str
    scfi_column: str
    brent_column: str
    article_count_column: str
    source_count_column: str
    msdi_columns: dict[str, str]
    horizons: tuple[int, ...]
    confounder_lags: tuple[int, ...]
    minimum_training_rows: int
    refit_every: int
    volatility_window: int
    minimum_complete_date: str | None


@dataclass(frozen=True)
class ExpandingFold:
    fold: int
    training_positions: np.ndarray
    evaluation_positions: np.ndarray
    evaluation_start: int
    evaluation_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate monthly-primary, time-aware DML effects and diagnostics."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--innovations", type=Path, default=DEFAULT_INNOVATIONS)
    parser.add_argument(
        "--local-projections", type=Path, default=DEFAULT_LOCAL_PROJECTIONS
    )
    parser.add_argument(
        "--causality-manifest", type=Path, default=DEFAULT_CAUSALITY_MANIFEST
    )
    parser.add_argument(
        "--stationarity-tests", type=Path, default=DEFAULT_STATIONARITY_TESTS
    )
    parser.add_argument(
        "--var-lag-selection", type=Path, default=DEFAULT_VAR_LAG_SELECTION
    )
    parser.add_argument(
        "--var-diagnostics", type=Path, default=DEFAULT_VAR_DIAGNOSTICS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--frequency", choices=("both", "weekly", "monthly"), default="both"
    )
    parser.add_argument(
        "--learners",
        default=",".join(LEARNERS),
        help="Comma-separated nuisance learners.",
    )
    parser.add_argument(
        "--paths", default=",".join(MSDI_PATHS), help="Comma-separated MSDI paths."
    )
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=None,
        help="Override the configured moving-block bootstrap repetitions.",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=0,
        help="Zero uses max(horizon, ceil(n^(1/3))).",
    )
    parser.add_argument(
        "--maximum-evaluation-rows",
        type=int,
        default=None,
        help="Debug-only cap per frequency/horizon; makes the run non-publication.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one horizon and the primary learner with small evaluation blocks.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration not found: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {resolved}")
    return payload


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def validate_script08_contract(
    manifest_path: Path,
    innovations_path: Path,
    local_projections_path: Path,
    additional_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = manifest.get("output_sha256", {})
    expected = {
        "msdi_innovations": innovations_path,
        "local_projections": local_projections_path,
    }
    if additional_paths:
        expected.update(additional_paths)
    rows: dict[str, Any] = {
        "script08_version": manifest.get("version"),
        "script08_status": manifest.get("status"),
        "hash_checks": {},
    }
    for name, path in expected.items():
        recorded = hashes.get(name)
        actual = sha256_file(path)
        if recorded is None:
            raise ValueError(f"Script-08 manifest lacks output hash for {name}")
        if recorded != actual:
            raise ValueError(
                f"Script-08 {name} hash mismatch: manifest={recorded} actual={actual}"
            )
        rows["hash_checks"][name] = {
            "recorded": recorded,
            "actual": actual,
            "passed": True,
        }
    return rows


def summarize_script08_triangulation(
    stationarity_path: Path,
    lag_selection_path: Path,
    diagnostics_path: Path,
) -> dict[str, Any]:
    stationarity = pd.read_csv(stationarity_path, low_memory=False)
    lag_selection = pd.read_csv(lag_selection_path, low_memory=False)
    diagnostics = pd.read_csv(diagnostics_path, low_memory=False)
    selected = lag_selection[lag_selection["selected_for_analysis"].astype(bool)]
    selected_records = selected[
        [
            "frequency",
            "msdi_path",
            "lag",
            "passes_residual_whiteness",
            "selection_reason",
        ]
    ].to_dict(orient="records")
    stationary_changes = stationarity[
        stationarity["variable"].isin(
            ["scfi_return", "msdi_change", "brent_return", "news_volume_change"]
        )
    ][["frequency", "msdi_path", "variable", "joint_decision"]].to_dict(
        orient="records"
    )
    whiteness = diagnostics[
        diagnostics["diagnostic"].eq("multivariate_portmanteau_whiteness")
    ][
        ["frequency", "msdi_path", "p_value", "passes_at_alpha"]
    ].to_dict(orient="records")
    return {
        "role": "descriptive_triangulation_only_not_DML_model_selection",
        "stationarity_change_decisions": stationary_changes,
        "selected_VAR_lags": selected_records,
        "selected_VAR_whiteness_diagnostics": whiteness,
    }


def parse_requested(raw: str, allowed: Sequence[str], label: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"At least one {label} must be requested")
    duplicates = sorted({item for item in values if values.count(item) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {label}: {duplicates}")
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; allowed={list(allowed)}")
    return values


def validate_args(args: argparse.Namespace) -> None:
    if args.bootstrap_repetitions is not None and args.bootstrap_repetitions < 0:
        raise ValueError("--bootstrap-repetitions cannot be negative")
    if args.block_length < 0:
        raise ValueError("--block-length cannot be negative")
    if args.maximum_evaluation_rows is not None and args.maximum_evaluation_rows < 1:
        raise ValueError("--maximum-evaluation-rows must be positive")
    parse_requested(args.learners, LEARNERS, "learner")
    parse_requested(args.paths, MSDI_PATHS, "MSDI path")


def frequency_specifications(
    config: dict[str, Any],
) -> dict[str, FrequencySpecification]:
    treatment = config["treatment"]
    result: dict[str, FrequencySpecification] = {}
    for frequency, values in config["frequencies"].items():
        path_columns = (
            treatment["paths"]
            if frequency == "weekly"
            else treatment["monthly_paths"]
        )
        result[frequency] = FrequencySpecification(
            frequency=frequency,
            role=str(values["role"]),
            panel_path=resolve_project_path(values["panel"]),
            date_column=str(values["date_column"]),
            scfi_column=str(values["scfi_column"]),
            brent_column=str(values["brent_column"]),
            article_count_column=str(values["article_count_column"]),
            source_count_column=str(values["source_count_column"]),
            msdi_columns={str(k): str(v) for k, v in path_columns.items()},
            horizons=tuple(int(item) for item in values["horizons"]),
            confounder_lags=tuple(
                int(item) for item in values["confounder_lags"]
            ),
            minimum_training_rows=int(values["minimum_training_rows"]),
            refit_every=int(values["refit_every"]),
            volatility_window=int(values["volatility_window"]),
            minimum_complete_date=(
                None
                if values.get("minimum_complete_date") is None
                else str(values["minimum_complete_date"])
            ),
        )
    return result


def configure_logging(log_dir: Path, version: str) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"08b_causal_ml_{version}.log"
    logger = logging.getLogger("causal_ml")
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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed: int, *parts: Any) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
    )
    temporary.replace(path)


def load_panel(path: Path, spec: FrequencySpecification) -> pd.DataFrame:
    panel = pd.read_csv(path, low_memory=False)
    required = {
        spec.date_column,
        spec.scfi_column,
        spec.brent_column,
        spec.article_count_column,
        spec.source_count_column,
        *spec.msdi_columns.values(),
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"{spec.frequency} panel lacks columns: {missing}")
    panel[spec.date_column] = pd.to_datetime(panel[spec.date_column], errors="coerce")
    if panel[spec.date_column].isna().any():
        raise ValueError(f"{spec.frequency} panel contains unparseable dates")
    if panel[spec.date_column].duplicated().any():
        raise ValueError(f"{spec.frequency} panel contains duplicate dates")
    return panel.sort_values(spec.date_column).reset_index(drop=True)


def apply_panel_completeness_contract(
    panel: pd.DataFrame, spec: FrequencySpecification
) -> tuple[pd.DataFrame, dict[str, Any]]:
    original = panel.copy()
    if spec.frequency == "monthly":
        required = {"msdi_available_weeks", "weekly_observations_in_month"}
        missing = sorted(required.difference(panel.columns))
        if missing:
            raise ValueError(
                "Monthly panel cannot enforce the within-month MSDI completeness "
                f"contract; missing={missing}"
            )
        available = pd.to_numeric(panel["msdi_available_weeks"], errors="coerce")
        expected = pd.to_numeric(
            panel["weekly_observations_in_month"], errors="coerce"
        )
        keep = available.notna() & expected.gt(0) & available.ge(expected)
        if spec.minimum_complete_date is not None:
            keep &= panel[spec.date_column].ge(pd.Timestamp(spec.minimum_complete_date))
        panel = panel.loc[keep].copy()
        rule = (
            "msdi_available_weeks_greater_or_equal_weekly_observations_in_month_"
            "and_pre_specified_complete_sample_start"
        )
    else:
        keep = pd.Series(True, index=panel.index)
        rule = "weekly_tail_missingness_handled_by_complete_analysis_rows"
    if panel.empty:
        raise ValueError(f"Completeness contract removed every {spec.frequency} row")
    audit = {
        "frequency": spec.frequency,
        "rule": rule,
        "rows_before": len(original),
        "rows_after": len(panel),
        "rows_removed": int((~keep).sum()),
        "first_date_after": panel[spec.date_column].min(),
        "last_date_after": panel[spec.date_column].max(),
    }
    return panel.reset_index(drop=True), audit


def expanding_past_median_regime(
    values: pd.Series, minimum_history: int
) -> pd.Series:
    threshold = values.shift(1).expanding(min_periods=minimum_history).median()
    result = pd.Series("unavailable", index=values.index, dtype="object")
    available = values.notna() & threshold.notna()
    result.loc[available & values.le(threshold)] = "low"
    result.loc[available & values.gt(threshold)] = "high"
    return result


def prepare_analysis_frame(
    panel: pd.DataFrame,
    spec: FrequencySpecification,
    path_name: str,
) -> tuple[pd.DataFrame, list[str]]:
    if path_name not in spec.msdi_columns:
        raise ValueError(f"Unknown {spec.frequency} MSDI path: {path_name}")
    frame = pd.DataFrame(
        {
            "position": np.arange(len(panel), dtype=int),
            "date": panel[spec.date_column],
            "scfi_level": pd.to_numeric(panel[spec.scfi_column], errors="coerce"),
            "brent_level": pd.to_numeric(panel[spec.brent_column], errors="coerce"),
            "article_count": pd.to_numeric(
                panel[spec.article_count_column], errors="coerce"
            ),
            "source_count": pd.to_numeric(
                panel[spec.source_count_column], errors="coerce"
            ),
            "msdi_level": pd.to_numeric(
                panel[spec.msdi_columns[path_name]], errors="coerce"
            ),
        }
    )
    if (frame[["scfi_level", "brent_level"]].dropna() <= 0).any().any():
        raise ValueError(f"{spec.frequency} has non-positive SCFI or Brent values")
    if (frame["article_count"].dropna() < 0).any():
        raise ValueError(f"{spec.frequency} has negative article counts")
    frame["scfi_log"] = np.log(frame["scfi_level"])
    frame["brent_log"] = np.log(frame["brent_level"])
    frame["log_news_volume"] = np.log1p(frame["article_count"])
    frame["scfi_return"] = frame["scfi_log"].diff()
    frame["brent_return"] = frame["brent_log"].diff()
    frame["msdi_change"] = frame["msdi_level"].diff()
    frame["news_volume_change"] = frame["log_news_volume"].diff()

    if spec.frequency == "weekly":
        cycle = frame["date"].dt.isocalendar().week.astype(float)
        period = 52.1775
    else:
        cycle = frame["date"].dt.month.astype(float)
        period = 12.0
    frame["season_sin"] = np.sin(2 * np.pi * cycle / period)
    frame["season_cos"] = np.cos(2 * np.pi * cycle / period)
    denominator = max(len(frame) - 1, 1)
    frame["linear_time"] = frame["position"] / denominator

    feature_columns = ["season_sin", "season_cos", "linear_time"]
    for variable in (
        "scfi_return",
        "msdi_change",
        "brent_return",
        "news_volume_change",
    ):
        for lag in spec.confounder_lags:
            name = f"{variable}_lag_{lag}"
            frame[name] = frame[variable].shift(lag)
            feature_columns.append(name)
    for variable in (
        "scfi_log",
        "msdi_level",
        "brent_log",
        "log_news_volume",
        "source_count",
    ):
        name = f"{variable}_lag_1"
        frame[name] = frame[variable].shift(1)
        feature_columns.append(name)

    optional_values: dict[str, pd.Series] = {}
    for logical_name, panel_column in OPTIONAL_COLUMN_MAP[spec.frequency].items():
        if panel_column not in panel.columns:
            continue
        values = pd.to_numeric(panel[panel_column], errors="coerce")
        optional_values[logical_name] = values
        name = f"{logical_name}_lag_1"
        frame[name] = values.shift(1)
        feature_columns.append(name)

    date = frame["date"]
    frame["covid_regime"] = np.select(
        [date.lt("2020-03-01"), date.le("2022-06-30")],
        ["pre_covid", "covid"],
        default="post_covid",
    )
    lagged_volatility = (
        frame["scfi_return"]
        .shift(1)
        .rolling(spec.volatility_window, min_periods=max(3, spec.volatility_window // 2))
        .std()
    )
    frame["lagged_market_volatility"] = lagged_volatility
    frame["lagged_market_volatility_regime"] = expanding_past_median_regime(
        lagged_volatility, max(12, spec.minimum_training_rows // 4)
    )
    lagged_news = frame["log_news_volume"].shift(1)
    frame["lagged_news_intensity_regime"] = expanding_past_median_regime(
        lagged_news, max(12, spec.minimum_training_rows // 4)
    )
    disagreement_parts = [
        optional_values[name]
        for name in ("peak_model_disagreement", "dimension_disagreement")
        if name in optional_values
    ]
    if disagreement_parts:
        disagreement = pd.concat(disagreement_parts, axis=1).mean(axis=1).shift(1)
        frame["lagged_model_disagreement"] = disagreement
        frame["lagged_model_disagreement_regime"] = expanding_past_median_regime(
            disagreement, max(12, spec.minimum_training_rows // 4)
        )
    else:
        frame["lagged_model_disagreement"] = np.nan
        frame["lagged_model_disagreement_regime"] = "unavailable"

    for horizon in spec.horizons:
        frame[f"outcome_h{horizon}"] = (
            frame["scfi_log"].shift(-horizon)
            - frame["scfi_log"].shift(1)
        )
        frame[f"placebo_outcome_h{horizon}"] = (
            frame["scfi_log"].shift(1)
            - frame["scfi_log"].shift(horizon + 1)
        )
    if len(feature_columns) != len(set(feature_columns)):
        raise AssertionError("Duplicate confounder columns were constructed")
    return frame, feature_columns


def build_expanding_folds(
    frame: pd.DataFrame,
    horizon: int,
    feature_columns: Sequence[str],
    minimum_training_rows: int,
    evaluation_block_size: int,
    maximum_evaluation_rows: int | None = None,
) -> list[ExpandingFold]:
    target = f"outcome_h{horizon}"
    placebo = f"placebo_outcome_h{horizon}"
    required = frame[["msdi_change", target, placebo]].notna().all(axis=1)
    # All pre-specified lag features must exist. Allowing calendar/time alone
    # would silently impute the structural warm-up rows before the longest lag.
    feature_available = frame[list(feature_columns)].notna().all(axis=1)
    eligible = frame.loc[required & feature_available, "position"].to_numpy(dtype=int)
    if not len(eligible):
        raise ValueError(f"No eligible observations for horizon={horizon}")

    first_evaluation_index: int | None = None
    for index, position in enumerate(eligible):
        training = eligible[(eligible < position) & (eligible + horizon < position)]
        if len(training) >= minimum_training_rows:
            first_evaluation_index = index
            break
    if first_evaluation_index is None:
        raise ValueError(
            f"Cannot form an expanding fold with {minimum_training_rows} training rows "
            f"for horizon={horizon}"
        )
    evaluation = eligible[first_evaluation_index:]
    if maximum_evaluation_rows is not None:
        evaluation = evaluation[:maximum_evaluation_rows]
    if evaluation_block_size < 1:
        raise ValueError("evaluation_block_size must be positive")
    blocks = [
        evaluation[start : start + evaluation_block_size]
        for start in range(0, len(evaluation), evaluation_block_size)
    ]
    folds: list[ExpandingFold] = []
    for fold_number, block in enumerate(blocks, start=1):
        start = int(block[0])
        training = eligible[(eligible < start) & (eligible + horizon < start)]
        if len(training) < minimum_training_rows:
            raise AssertionError("Expanding fold lost its minimum training history")
        if int(training[-1]) + horizon >= start:
            raise AssertionError("Horizon purge failed")
        if np.intersect1d(training, block).size:
            raise AssertionError("Training/evaluation overlap detected")
        folds.append(
            ExpandingFold(
                fold=fold_number,
                training_positions=training,
                evaluation_positions=block.astype(int),
                evaluation_start=start,
                evaluation_end=int(block[-1]),
            )
        )
    return folds


def make_learner(
    learner: str, parameters: dict[str, Any], seed: int
) -> RegressorMixin:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if learner == "histogram_gradient_boosting":
        estimator: RegressorMixin = HistGradientBoostingRegressor(
            learning_rate=float(parameters["learning_rate"]),
            max_iter=int(parameters["max_iter"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            l2_regularization=float(parameters["l2_regularization"]),
            early_stopping=False,
            random_state=seed,
        )
        return Pipeline([("imputer", imputer), ("model", estimator)])
    if learner == "elastic_net":
        estimator = ElasticNet(
            alpha=float(parameters["alpha"]),
            l1_ratio=float(parameters["l1_ratio"]),
            max_iter=int(parameters["max_iter"]),
            selection="cyclic",
            random_state=seed,
        )
        return Pipeline(
            [
                ("imputer", imputer),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    if learner == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=int(parameters["n_estimators"]),
            max_depth=int(parameters["max_depth"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=float(parameters["max_features"]),
            n_jobs=int(parameters["n_jobs"]),
            random_state=seed,
        )
        return Pipeline([("imputer", imputer), ("model", estimator)])
    raise ValueError(f"Unknown nuisance learner: {learner}")


def calibrate_treatment_scale(
    *,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    cutoff_position_exclusive: int,
    spec: FrequencySpecification,
    path_name: str,
    learner: str,
    learner_parameters: dict[str, Any],
    seed: int,
) -> tuple[float, dict[str, Any]]:
    eligible = (
        frame["position"].lt(cutoff_position_exclusive)
        & frame["msdi_change"].notna()
        & frame[list(feature_columns)].notna().all(axis=1)
    )
    positions = frame.loc[eligible, "position"].to_numpy(dtype=int)
    if len(positions) < 36:
        raise ValueError(
            f"Too few pre-evaluation calibration rows for {spec.frequency}/{path_name}: "
            f"{len(positions)}"
        )
    validation_rows = max(12, len(positions) // 3)
    training_positions = positions[:-validation_rows]
    validation_positions = positions[-validation_rows:]
    if len(training_positions) < 24:
        raise ValueError("Treatment-scale calibration training sample is too small")
    model_seed = stable_seed(seed, spec.frequency, path_name, "scale_calibration")
    model = make_learner(learner, learner_parameters, model_seed)
    model.fit(
        frame.loc[training_positions, list(feature_columns)],
        frame.loc[training_positions, "msdi_change"].to_numpy(dtype=float),
    )
    actual = frame.loc[validation_positions, "msdi_change"].to_numpy(dtype=float)
    predicted = np.asarray(
        model.predict(frame.loc[validation_positions, list(feature_columns)]),
        dtype=float,
    )
    residual = actual - predicted
    scale = float(np.std(residual, ddof=1))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError(
            f"Invalid frozen treatment scale for {spec.frequency}/{path_name}: {scale}"
        )
    audit = {
        "frequency": spec.frequency,
        "frequency_role": spec.role,
        "msdi_path": path_name,
        "calibration_learner": learner,
        "calibration_seed": model_seed,
        "cutoff_position_exclusive": cutoff_position_exclusive,
        "cutoff_date_exclusive": frame.at[cutoff_position_exclusive, "date"],
        "training_rows": len(training_positions),
        "training_start_date": frame.at[int(training_positions[0]), "date"],
        "training_end_date": frame.at[int(training_positions[-1]), "date"],
        "validation_rows": len(validation_positions),
        "validation_start_date": frame.at[int(validation_positions[0]), "date"],
        "validation_end_date": frame.at[int(validation_positions[-1]), "date"],
        "validation_precedes_first_evaluation": bool(
            int(validation_positions[-1]) < cutoff_position_exclusive
        ),
        "frozen_residual_standard_deviation": scale,
        "validation_treatment_rmse": rmse(actual, predicted),
        "validation_treatment_r2": safe_r2(actual, predicted),
        "definition": (
            "single_forward_holdout_before_all_evaluation_dates_constant_across_"
            "horizons_and_learners"
        ),
    }
    return scale, audit


def safe_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2 or float(np.var(actual)) <= 1e-15:
        return math.nan
    return float(r2_score(actual, predicted))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(actual, predicted)))


def cross_fit_configuration(
    *,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    folds: Sequence[ExpandingFold],
    spec: FrequencySpecification,
    path_name: str,
    horizon: int,
    reference_horizon: int,
    learner: str,
    learner_parameters: dict[str, Any],
    seed: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_column = f"outcome_h{horizon}"
    placebo_column = f"placebo_outcome_h{horizon}"
    prediction_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for fold in folds:
        train = fold.training_positions
        evaluation = fold.evaluation_positions
        x_train = frame.loc[train, list(feature_columns)]
        x_evaluation = frame.loc[evaluation, list(feature_columns)]
        d_train = frame.loc[train, "msdi_change"].to_numpy(dtype=float)
        y_train = frame.loc[train, target_column].to_numpy(dtype=float)
        p_train = frame.loc[train, placebo_column].to_numpy(dtype=float)
        d_evaluation = frame.loc[evaluation, "msdi_change"].to_numpy(dtype=float)
        y_evaluation = frame.loc[evaluation, target_column].to_numpy(dtype=float)
        p_evaluation = frame.loc[evaluation, placebo_column].to_numpy(dtype=float)

        treatment_seed = stable_seed(
            seed, spec.frequency, path_name, learner, fold.fold, "treatment"
        )
        outcome_seed = stable_seed(
            seed, spec.frequency, path_name, horizon, learner, fold.fold, "outcome"
        )
        placebo_seed = stable_seed(
            seed, spec.frequency, path_name, horizon, learner, fold.fold, "placebo"
        )
        treatment_model = make_learner(learner, learner_parameters, treatment_seed)
        outcome_model = make_learner(learner, learner_parameters, outcome_seed)
        placebo_model = make_learner(learner, learner_parameters, placebo_seed)
        started = time.perf_counter()
        treatment_model.fit(x_train, d_train)
        treatment_seconds = time.perf_counter() - started
        started = time.perf_counter()
        outcome_model.fit(x_train, y_train)
        outcome_seconds = time.perf_counter() - started
        started = time.perf_counter()
        placebo_model.fit(x_train, p_train)
        placebo_seconds = time.perf_counter() - started
        d_hat = np.asarray(treatment_model.predict(x_evaluation), dtype=float)
        y_hat = np.asarray(outcome_model.predict(x_evaluation), dtype=float)
        p_hat = np.asarray(placebo_model.predict(x_evaluation), dtype=float)

        treatment_rmse = rmse(d_evaluation, d_hat)
        outcome_rmse = rmse(y_evaluation, y_hat)
        placebo_rmse = rmse(p_evaluation, p_hat)
        treatment_r2 = safe_r2(d_evaluation, d_hat)
        outcome_r2 = safe_r2(y_evaluation, y_hat)
        placebo_r2 = safe_r2(p_evaluation, p_hat)
        baseline_outcome_rmse = rmse(
            y_evaluation, np.repeat(float(np.mean(y_train)), len(y_evaluation))
        )
        baseline_treatment_rmse = rmse(
            d_evaluation, np.repeat(float(np.mean(d_train)), len(d_evaluation))
        )
        purge_passed = int(train[-1]) + reference_horizon < int(evaluation[0])
        train_q01 = float(np.quantile(d_train, 0.01))
        train_q99 = float(np.quantile(d_train, 0.99))
        evaluation_outside_support = float(
            np.mean((d_evaluation < train_q01) | (d_evaluation > train_q99))
        )
        audit_rows.append(
            {
                "frequency": spec.frequency,
                "frequency_role": spec.role,
                "msdi_path": path_name,
                "horizon": horizon,
                "fold_reference_maximum_horizon": reference_horizon,
                "learner": learner,
                "fold": fold.fold,
                "training_rows": len(train),
                "training_start_position": int(train[0]),
                "training_end_position": int(train[-1]),
                "training_start_date": frame.at[int(train[0]), "date"],
                "training_end_feature_date": frame.at[int(train[-1]), "date"],
                "training_end_label_position": int(train[-1]) + horizon,
                "training_end_label_date": frame.at[int(train[-1]) + horizon, "date"],
                "strict_embargo_end_position": int(train[-1]) + reference_horizon,
                "strict_embargo_end_date": frame.at[
                    int(train[-1]) + reference_horizon, "date"
                ],
                "evaluation_rows": len(evaluation),
                "evaluation_start_position": int(evaluation[0]),
                "evaluation_end_position": int(evaluation[-1]),
                "evaluation_start_date": frame.at[int(evaluation[0]), "date"],
                "evaluation_end_date": frame.at[int(evaluation[-1]), "date"],
                "horizon_gap": reference_horizon,
                "horizon_purge_passed": purge_passed,
                "no_future_training_passed": int(train[-1]) < int(evaluation[0]),
                "feature_count": len(feature_columns),
                "treatment_seed": treatment_seed,
                "outcome_seed": outcome_seed,
                "placebo_seed": placebo_seed,
                "treatment_fit_runtime_seconds": treatment_seconds,
                "outcome_fit_runtime_seconds": outcome_seconds,
                "placebo_fit_runtime_seconds": placebo_seconds,
                "treatment_oof_rmse": treatment_rmse,
                "treatment_oof_r2": treatment_r2,
                "training_treatment_q01": train_q01,
                "training_treatment_q99": train_q99,
                "evaluation_share_outside_training_q01_q99": (
                    evaluation_outside_support
                ),
                "treatment_baseline_rmse": baseline_treatment_rmse,
                "treatment_rmse_improvement": baseline_treatment_rmse
                - treatment_rmse,
                "outcome_oof_rmse": outcome_rmse,
                "outcome_oof_r2": outcome_r2,
                "outcome_baseline_rmse": baseline_outcome_rmse,
                "outcome_rmse_improvement": baseline_outcome_rmse - outcome_rmse,
                "placebo_oof_rmse": placebo_rmse,
                "placebo_oof_r2": placebo_r2,
            }
        )
        logger.info(
            "crossfit frequency=%s path=%s h=%s learner=%s fold=%s "
            "train=%s eval=%s purge=%s outcome_rmse=%.6f outcome_r2=%.4f "
            "treatment_rmse=%.6f treatment_r2=%.4f diff_rmse=%.6f runtime=%.3fs",
            spec.frequency,
            path_name,
            horizon,
            learner,
            fold.fold,
            len(train),
            len(evaluation),
            purge_passed,
            outcome_rmse,
            outcome_r2,
            treatment_rmse,
            treatment_r2,
            baseline_outcome_rmse - outcome_rmse,
            treatment_seconds + outcome_seconds + placebo_seconds,
        )
        for offset, position in enumerate(evaluation):
            lagged_level = float(frame.at[int(position), "msdi_level"] - d_evaluation[offset])
            predicted_level = lagged_level + float(d_hat[offset])
            level_innovation = float(frame.at[int(position), "msdi_level"] - predicted_level)
            change_innovation = float(d_evaluation[offset] - d_hat[offset])
            row = {
                "frequency": spec.frequency,
                "frequency_role": spec.role,
                "msdi_path": path_name,
                "horizon": horizon,
                "learner": learner,
                "fold": fold.fold,
                "position": int(position),
                "date": frame.at[int(position), "date"],
                "outcome": float(y_evaluation[offset]),
                "outcome_prediction": float(y_hat[offset]),
                "outcome_residual": float(y_evaluation[offset] - y_hat[offset]),
                "placebo_outcome": float(p_evaluation[offset]),
                "placebo_prediction": float(p_hat[offset]),
                "placebo_residual": float(p_evaluation[offset] - p_hat[offset]),
                "treatment_raw_msdi_change": float(d_evaluation[offset]),
                "treatment_prediction": float(d_hat[offset]),
                "treatment_residual_oof": change_innovation,
                "treatment_observed_msdi_level": float(
                    frame.at[int(position), "msdi_level"]
                ),
                "treatment_lagged_msdi_level": lagged_level,
                "treatment_predicted_msdi_level": predicted_level,
                "treatment_level_innovation_oof": level_innovation,
                "innovation_reparameterization_absolute_difference": abs(
                    level_innovation - change_innovation
                ),
            }
            for modifier in HETEROGENEITY_MODIFIERS:
                row[modifier] = frame.at[int(position), modifier]
            prediction_rows.append(row)
    predictions = pd.DataFrame(prediction_rows).sort_values("date").reset_index(drop=True)
    audits = pd.DataFrame(audit_rows)
    return predictions, audits


def enforce_common_samples(
    predictions: pd.DataFrame,
    requested_paths: Sequence[str],
    requested_learners: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    retained: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for frequency, group in predictions.groupby("frequency", sort=True):
        horizons = sorted(int(item) for item in group["horizon"].unique())
        expected = {
            (horizon, path, learner)
            for horizon in horizons
            for path in requested_paths
            for learner in requested_learners
        }
        date_sets = {
            (int(horizon), path, learner): set(part["date"])
            for (horizon, path, learner), part in group.groupby(
                ["horizon", "msdi_path", "learner"]
            )
        }
        missing = sorted(expected.difference(date_sets))
        if missing:
            raise ValueError(
                f"Missing horizon/path/learner cells for {frequency}: {missing}"
            )
        common_dates = set.intersection(*(date_sets[key] for key in sorted(expected)))
        if not common_dates:
            raise ValueError(f"No common DML dates across horizons for {frequency}")
        selected = group[group["date"].isin(common_dates)].copy()
        selected["common_sample"] = True
        retained.append(selected)
        counts = selected.groupby(["horizon", "msdi_path", "learner"]).size()
        for horizon in horizons:
            audit_rows.append(
                {
                    "frequency": frequency,
                    "horizon": int(horizon),
                    "requested_paths": "|".join(requested_paths),
                    "requested_learners": "|".join(requested_learners),
                    "requested_horizons": "|".join(str(item) for item in horizons),
                    "expected_cells_all_horizons": len(expected),
                    "observed_cells_all_horizons": len(counts),
                    "common_dates": len(common_dates),
                    "minimum_cell_dates_before_intersection": min(
                        len(date_sets[key]) for key in expected
                    ),
                    "maximum_cell_dates_before_intersection": max(
                        len(date_sets[key]) for key in expected
                    ),
                    "rows_removed_for_common_sample_all_horizons": (
                        len(group) - len(selected)
                    ),
                    "balanced_common_sample_passed": counts.nunique() == 1,
                    "common_start_date": min(common_dates),
                    "common_end_date": max(common_dates),
                }
            )
    output = pd.concat(retained, ignore_index=True).sort_values(
        ["frequency", "horizon", "msdi_path", "learner", "date"]
    )
    return output.reset_index(drop=True), pd.DataFrame(audit_rows)


def treatment_horizon_invariance_audit(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in predictions.groupby(
        ["frequency", "frequency_role", "msdi_path", "learner"], sort=True
    ):
        per_date = group.groupby("date").agg(
            residual_min=("treatment_residual_oof", "min"),
            residual_max=("treatment_residual_oof", "max"),
            prediction_min=("treatment_prediction", "min"),
            prediction_max=("treatment_prediction", "max"),
            fold_count=("fold", "nunique"),
            horizon_count=("horizon", "nunique"),
        )
        residual_range = per_date["residual_max"] - per_date["residual_min"]
        prediction_range = per_date["prediction_max"] - per_date["prediction_min"]
        maximum_difference = float(max(residual_range.max(), prediction_range.max()))
        fold_invariant = bool(per_date["fold_count"].eq(1).all())
        passed = bool(maximum_difference <= 1e-12 and fold_invariant)
        rows.append(
            {
                "frequency": key[0],
                "frequency_role": key[1],
                "msdi_path": key[2],
                "learner": key[3],
                "dates": len(per_date),
                "horizons": int(group["horizon"].nunique()),
                "maximum_treatment_prediction_or_residual_difference_across_horizons": (
                    maximum_difference
                ),
                "fold_assignment_invariant_across_horizons": fold_invariant,
                "treatment_nuisance_horizon_invariance_passed": passed,
            }
        )
    result = pd.DataFrame(rows)
    if not result["treatment_nuisance_horizon_invariance_passed"].all():
        raise AssertionError(
            "Treatment nuisance predictions changed across outcome horizons"
        )
    return result


def newey_west_long_run_variance(values: np.ndarray, max_lags: int) -> float:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) < 3:
        raise ValueError("HAC input must be a one-dimensional sample of length >= 3")
    centered = vector - vector.mean()
    n = len(centered)
    maximum = min(int(max_lags), n - 1)
    result = float(np.dot(centered, centered) / n)
    for lag in range(1, maximum + 1):
        weight = 1.0 - lag / (maximum + 1.0)
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        result += 2.0 * weight * covariance
    return max(result, 0.0)


def moving_block_indices(
    n_observations: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    if not 1 <= block_length <= n_observations:
        raise ValueError("block length must lie within the sample")
    blocks_needed = int(math.ceil(n_observations / block_length))
    maximum_start = n_observations - block_length
    starts = rng.integers(0, maximum_start + 1, size=blocks_needed)
    indices = np.concatenate(
        [np.arange(start, start + block_length) for start in starts]
    )
    return indices[:n_observations]


def orthogonal_estimate(
    outcome_residual: Iterable[float],
    treatment_residual: Iterable[float],
    *,
    alpha: float,
    hac_lags: int,
    bootstrap_repetitions: int,
    block_length: int,
    seed: int,
    effect_scale: float | None = None,
) -> dict[str, Any]:
    y = np.asarray(list(outcome_residual), dtype=float)
    d = np.asarray(list(treatment_residual), dtype=float)
    finite = np.isfinite(y) & np.isfinite(d)
    y, d = y[finite], d[finite]
    if len(y) < 12:
        raise ValueError("At least 12 residual observations are required")
    denominator = float(np.dot(d, d))
    if denominator <= 1e-12:
        raise ValueError("Treatment residual has insufficient variation")
    theta = float(np.dot(d, y) / denominator)
    hac_fit = sm.OLS(y, d[:, None], missing="raise").fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": min(int(hac_lags), len(d) - 1),
            "use_correction": True,
        },
    )
    if not np.isclose(float(hac_fit.params[0]), theta, rtol=1e-12, atol=1e-14):
        raise AssertionError("Closed-form and statsmodels orthogonal slopes differ")
    standard_error = float(hac_fit.bse[0])
    critical = float(norm.ppf(1 - alpha / 2))
    residual_sd = float(np.std(d, ddof=1))
    scale = float(residual_sd if effect_scale is None else effect_scale)
    effect = theta * scale
    effect_se = standard_error * scale
    lower = effect - critical * effect_se
    upper = effect + critical * effect_se
    z_statistic = effect / effect_se if effect_se > 0 else math.nan
    p_value = float(2 * norm.sf(abs(z_statistic))) if np.isfinite(z_statistic) else math.nan

    bootstrap_values: list[float] = []
    rng = np.random.default_rng(seed)
    if bootstrap_repetitions:
        for _ in range(bootstrap_repetitions):
            indices = moving_block_indices(len(d), block_length, rng)
            sampled_d = d[indices]
            sampled_y = y[indices]
            sampled_denominator = float(np.dot(sampled_d, sampled_d))
            if sampled_denominator <= 1e-12:
                continue
            bootstrap_values.append(
                float(np.dot(sampled_d, sampled_y) / sampled_denominator * scale)
            )
    bootstrap_lower = (
        float(np.quantile(bootstrap_values, alpha / 2))
        if bootstrap_values
        else math.nan
    )
    bootstrap_upper = (
        float(np.quantile(bootstrap_values, 1 - alpha / 2))
        if bootstrap_values
        else math.nan
    )
    correlation = float(np.corrcoef(d, y)[0, 1])
    return {
        "coefficient_per_raw_treatment_unit": theta,
        "raw_standard_error_hac": standard_error,
        "treatment_residual_sd_estimation_sample": residual_sd,
        "effect_scale_frozen_calibration_residual_sd": scale,
        "effect_log_points_per_one_sd_innovation": effect,
        "effect_percent_per_one_sd_innovation": float(100 * np.expm1(effect)),
        "standard_error_hac": effect_se,
        "z_statistic": z_statistic,
        "p_value": p_value,
        "ci_lower_log_points": lower,
        "ci_upper_log_points": upper,
        "ci_lower_percent": float(100 * np.expm1(lower)),
        "ci_upper_percent": float(100 * np.expm1(upper)),
        "hac_max_lags": min(hac_lags, len(d) - 1),
        "bootstrap_repetitions_requested": bootstrap_repetitions,
        "bootstrap_repetitions_successful": len(bootstrap_values),
        "bootstrap_block_length": block_length,
        "bootstrap_ci_lower_log_points": bootstrap_lower,
        "bootstrap_ci_upper_log_points": bootstrap_upper,
        "bootstrap_ci_lower_percent": (
            float(100 * np.expm1(bootstrap_lower))
            if np.isfinite(bootstrap_lower)
            else math.nan
        ),
        "bootstrap_ci_upper_percent": (
            float(100 * np.expm1(bootstrap_upper))
            if np.isfinite(bootstrap_upper)
            else math.nan
        ),
        "orthogonal_residual_correlation": correlation,
        "orthogonal_partial_r_squared": correlation**2,
        "n_observations": len(d),
    }


def configured_block_length(
    n_observations: int, horizon: int, requested: int
) -> int:
    if requested:
        return min(requested, n_observations)
    return min(
        n_observations,
        max(1, horizon, int(math.ceil(n_observations ** (1 / 3)))),
    )


def estimate_effects(
    predictions: pd.DataFrame,
    *,
    treatment_scales: dict[tuple[str, str], float],
    primary_learner: str,
    alpha: float,
    bootstrap_repetitions: int,
    requested_block_length: int,
    seed: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effect_rows: list[dict[str, Any]] = []
    placebo_rows: list[dict[str, Any]] = []
    keys = ["frequency", "frequency_role", "msdi_path", "horizon", "learner"]
    for key, group in predictions.groupby(keys, sort=True):
        frequency, role, path_name, horizon, learner = key
        group = group.sort_values("date")
        n = len(group)
        hac_lags = max(1, int(horizon))
        block_length = configured_block_length(
            n, int(horizon), requested_block_length
        )
        group_seed = stable_seed(seed, *key, "effect")
        frozen_scale = treatment_scales[(str(frequency), str(path_name))]
        result = orthogonal_estimate(
            group["outcome_residual"],
            group["treatment_residual_oof"],
            alpha=alpha,
            hac_lags=hac_lags,
            bootstrap_repetitions=bootstrap_repetitions,
            block_length=block_length,
            seed=group_seed,
            effect_scale=frozen_scale,
        )
        effect_row = {
            "frequency": frequency,
            "frequency_role": role,
            "msdi_path": path_name,
            "horizon": int(horizon),
            "horizon_unit": "weeks" if frequency == "weekly" else "months",
            "learner": learner,
            "primary_learner": learner == primary_learner,
            "outcome": "impact_inclusive_cumulative_scfi_log_change",
            "outcome_window": "log_scfi_t_plus_h_minus_log_scfi_t_minus_1",
            "treatment": (
                "current_msdi_level_innovation_via_stationary_change_"
                "reparameterization"
            ),
            "effect_scale_definition": (
                "one_frozen_past_only_calibration_residual_standard_deviation"
            ),
            "sample_start_date": group["date"].min(),
            "sample_end_date": group["date"].max(),
            "nuisance_outcome_oof_rmse": rmse(
                group["outcome"].to_numpy(),
                group["outcome_prediction"].to_numpy(),
            ),
            "nuisance_outcome_oof_r2": safe_r2(
                group["outcome"].to_numpy(),
                group["outcome_prediction"].to_numpy(),
            ),
            "nuisance_treatment_oof_rmse": rmse(
                group["treatment_raw_msdi_change"].to_numpy(),
                group["treatment_prediction"].to_numpy(),
            ),
            "nuisance_treatment_oof_r2": safe_r2(
                group["treatment_raw_msdi_change"].to_numpy(),
                group["treatment_prediction"].to_numpy(),
            ),
            "confidence_level": 1 - alpha,
            "interpretation": (
                "orthogonalized_predictive_effect_under_unverified_"
                "conditional_exchangeability_not_structural_causality"
            ),
            **result,
        }
        effect_row["negative_outcome_nuisance_r2_warning"] = bool(
            np.isfinite(effect_row["nuisance_outcome_oof_r2"])
            and effect_row["nuisance_outcome_oof_r2"] < 0
        )
        effect_row["negative_treatment_nuisance_r2_warning"] = bool(
            np.isfinite(effect_row["nuisance_treatment_oof_r2"])
            and effect_row["nuisance_treatment_oof_r2"] < 0
        )
        effect_rows.append(effect_row)
        placebo = orthogonal_estimate(
            group["placebo_residual"],
            group["treatment_residual_oof"],
            alpha=alpha,
            hac_lags=hac_lags,
            bootstrap_repetitions=bootstrap_repetitions,
            block_length=block_length,
            seed=stable_seed(seed, *key, "placebo"),
            effect_scale=frozen_scale,
        )
        placebo_rows.append(
            {
                "frequency": frequency,
                "frequency_role": role,
                "msdi_path": path_name,
                "horizon": int(horizon),
                "horizon_unit": "weeks" if frequency == "weekly" else "months",
                "learner": learner,
                "primary_learner": learner == primary_learner,
                "placebo_outcome": (
                    "pre_treatment_scfi_change_t_minus_1_to_t_minus_h_minus_1"
                ),
                "expected_result": "no_association",
                "interpretation": "temporal_falsification_not_proof_of_identification",
                **placebo,
            }
        )
        logger.info(
            "estimate frequency=%s path=%s h=%s learner=%s n=%s "
            "effect_log=%.6f effect_pct=%.3f se=%.6f p=%.6g "
            "placebo_log=%.6f placebo_p=%.6g nuisance_y_r2=%.4f nuisance_d_r2=%.4f",
            frequency,
            path_name,
            horizon,
            learner,
            n,
            result["effect_log_points_per_one_sd_innovation"],
            result["effect_percent_per_one_sd_innovation"],
            result["standard_error_hac"],
            result["p_value"],
            placebo["effect_log_points_per_one_sd_innovation"],
            placebo["p_value"],
            effect_row["nuisance_outcome_oof_r2"],
            effect_row["nuisance_treatment_oof_r2"],
        )
    effects = add_effect_multiplicity(pd.DataFrame(effect_rows), primary_learner)
    placebos = add_effect_multiplicity(pd.DataFrame(placebo_rows), primary_learner)
    placebos["falsification_flag_raw"] = placebos["p_value"].lt(alpha)
    placebos["falsification_flag_fdr"] = placebos["p_value_fdr_all"].lt(alpha)
    placebos["falsification_status"] = np.where(
        placebos["falsification_flag_fdr"], "flagged", "not_flagged"
    )
    return effects, placebos


def add_primary_simultaneous_bands(
    effects: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    primary_learner: str,
    alpha: float,
    bootstrap_repetitions: int,
    requested_block_length: int,
    seed: int,
) -> pd.DataFrame:
    result = effects.copy()
    for column in (
        "simultaneous_lower_log_points",
        "simultaneous_upper_log_points",
        "simultaneous_lower_percent",
        "simultaneous_upper_percent",
        "simultaneous_sup_t_critical_value",
        "simultaneous_bootstrap_repetitions_successful",
    ):
        result[column] = np.nan
    result["simultaneous_family_id"] = ""
    result["simultaneous_band_available"] = False
    result["simultaneous_band_significant"] = False
    family_mask = (
        result["frequency"].eq("monthly")
        & result["msdi_path"].eq("peak")
        & result["learner"].eq(primary_learner)
    )
    family = result.loc[family_mask].sort_values("horizon")
    if family.empty or bootstrap_repetitions <= 0:
        return result
    horizons = [int(item) for item in family["horizon"]]
    groups = {
        horizon: predictions[
            predictions["frequency"].eq("monthly")
            & predictions["msdi_path"].eq("peak")
            & predictions["learner"].eq(primary_learner)
            & predictions["horizon"].eq(horizon)
        ].sort_values("date")
        for horizon in horizons
    }
    date_lists = [tuple(group["date"]) for group in groups.values()]
    if not all(dates == date_lists[0] for dates in date_lists[1:]):
        raise AssertionError("Primary dynamic curve lacks identical dates across horizons")
    n = len(date_lists[0])
    block_length = configured_block_length(
        n, max(horizons), requested_block_length
    )
    point = family.set_index("horizon")[
        "effect_log_points_per_one_sd_innovation"
    ].to_dict()
    standard_error = family.set_index("horizon")["standard_error_hac"].to_dict()
    scale = family.set_index("horizon")[
        "effect_scale_frozen_calibration_residual_sd"
    ].to_dict()
    rng = np.random.default_rng(stable_seed(seed, "monthly", "peak", "joint_band"))
    sup_statistics: list[float] = []
    for _ in range(bootstrap_repetitions):
        indices = moving_block_indices(n, block_length, rng)
        studentized: list[float] = []
        valid = True
        for horizon in horizons:
            group = groups[horizon]
            d = group["treatment_residual_oof"].to_numpy(dtype=float)[indices]
            y = group["outcome_residual"].to_numpy(dtype=float)[indices]
            denominator = float(np.dot(d, d))
            se = float(standard_error[horizon])
            if denominator <= 1e-12 or se <= 0 or not np.isfinite(se):
                valid = False
                break
            bootstrap_effect = float(
                np.dot(d, y) / denominator * float(scale[horizon])
            )
            studentized.append(
                abs((bootstrap_effect - float(point[horizon])) / se)
            )
        if valid:
            sup_statistics.append(max(studentized))
    if not sup_statistics:
        return result
    critical = float(np.quantile(sup_statistics, 1 - alpha))
    family_id = "primary_dml__monthly__peak__all_horizons"
    for index in family.index:
        estimate = float(result.at[index, "effect_log_points_per_one_sd_innovation"])
        se = float(result.at[index, "standard_error_hac"])
        lower = estimate - critical * se
        upper = estimate + critical * se
        result.at[index, "simultaneous_lower_log_points"] = lower
        result.at[index, "simultaneous_upper_log_points"] = upper
        result.at[index, "simultaneous_lower_percent"] = 100 * np.expm1(lower)
        result.at[index, "simultaneous_upper_percent"] = 100 * np.expm1(upper)
        result.at[index, "simultaneous_sup_t_critical_value"] = critical
        result.at[index, "simultaneous_bootstrap_repetitions_successful"] = len(
            sup_statistics
        )
        result.at[index, "simultaneous_family_id"] = family_id
        result.at[index, "simultaneous_band_available"] = True
        result.at[index, "simultaneous_band_significant"] = bool(
            lower > 0 or upper < 0
        )
    return result


def adjust_pvalues(values: pd.Series, method: str) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    finite = values.notna() & np.isfinite(values.astype(float))
    if finite.any():
        result.loc[finite] = multipletests(
            values.loc[finite].astype(float).to_numpy(), method=method
        )[1]
    return result


def add_effect_multiplicity(
    frame: pd.DataFrame, primary_learner: str
) -> pd.DataFrame:
    result = frame.copy()
    result["p_value_fdr_all"] = adjust_pvalues(result["p_value"], "fdr_bh")
    result["p_value_holm_within_frequency_learner"] = np.nan
    for _, indices in result.groupby(["frequency", "learner"]).groups.items():
        result.loc[indices, "p_value_holm_within_frequency_learner"] = adjust_pvalues(
            result.loc[indices, "p_value"], "holm"
        )
    primary = (
        result["frequency"].eq("monthly")
        & result["msdi_path"].eq("peak")
        & result["learner"].eq(primary_learner)
    )
    result["primary_family"] = primary
    result["p_value_holm_primary_family"] = np.nan
    result.loc[primary, "p_value_holm_primary_family"] = adjust_pvalues(
        result.loc[primary, "p_value"], "holm"
    )
    return result


def overlap_diagnostics(
    predictions: pd.DataFrame, nuisance_audit: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["frequency", "frequency_role", "msdi_path", "horizon", "learner"]
    for key, group in predictions.groupby(keys, sort=True):
        residual = group["treatment_residual_oof"].to_numpy(dtype=float)
        raw = group["treatment_raw_msdi_change"].to_numpy(dtype=float)
        squared = residual**2
        total_information = float(squared.sum())
        effective_n = (
            float(total_information**2 / np.square(squared).sum())
            if float(np.square(squared).sum()) > 0
            else 0.0
        )
        max_share = (
            float(squared.max() / total_information) if total_information > 0 else 1.0
        )
        residual_sd = float(np.std(residual, ddof=1))
        raw_sd = float(np.std(raw, ddof=1))
        audit = nuisance_audit[
            nuisance_audit["frequency"].eq(key[0])
            & nuisance_audit["msdi_path"].eq(key[2])
            & nuisance_audit["horizon"].eq(key[3])
            & nuisance_audit["learner"].eq(key[4])
        ]
        support_share = float(
            np.average(
                audit["evaluation_share_outside_training_q01_q99"],
                weights=audit["evaluation_rows"],
            )
        )
        variation_ratio = (
            float(np.var(residual, ddof=1) / np.var(raw, ddof=1))
            if raw_sd > 0
            else math.nan
        )
        weak_variation = bool(np.isfinite(variation_ratio) and variation_ratio < 0.05)
        influence_warning = max_share >= 0.10
        influence_severe = max_share >= 0.25
        low_information = effective_n < max(30, 0.10 * len(group))
        support_warning = support_share > 0.10
        row = {
            "frequency": key[0],
            "frequency_role": key[1],
            "msdi_path": key[2],
            "horizon": int(key[3]),
            "learner": key[4],
            "n_observations": len(group),
            "raw_treatment_sd": raw_sd,
            "residual_treatment_sd": residual_sd,
            "conditional_variation_ratio": variation_ratio,
            "residual_q01": float(np.quantile(residual, 0.01)),
            "residual_q05": float(np.quantile(residual, 0.05)),
            "residual_q25": float(np.quantile(residual, 0.25)),
            "residual_median": float(np.quantile(residual, 0.50)),
            "residual_q75": float(np.quantile(residual, 0.75)),
            "residual_q95": float(np.quantile(residual, 0.95)),
            "residual_q99": float(np.quantile(residual, 0.99)),
            "residual_information_effective_n": effective_n,
            "maximum_single_observation_information_share": max_share,
            "evaluation_share_outside_fold_training_q01_q99": support_share,
            "share_beyond_three_sd": float(
                np.mean(np.abs(residual - residual.mean()) > 3 * residual_sd)
            ),
            "weak_conditional_variation_flag": weak_variation,
            "influence_concentration_warning_flag": influence_warning,
            "influence_concentration_severe_flag": influence_severe,
            "low_information_ess_flag": low_information,
            "training_support_extrapolation_warning_flag": support_warning,
            "continuous_treatment_diagnostic_flag": bool(
                residual_sd <= 1e-6
                or weak_variation
                or influence_warning
                or low_information
                or support_warning
            ),
            "diagnostic_scope": (
                "conditional_treatment_variation_not_binary_propensity_overlap"
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def heterogeneous_group_effects(
    predictions: pd.DataFrame,
    effects: pd.DataFrame,
    *,
    primary_learner: str,
    minimum_group_rows: dict[str, int | None],
    requested_horizons: dict[str, Sequence[int]],
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    learner_all = predictions[predictions["learner"].eq(primary_learner)].copy()
    weekly_robustness = learner_all[learner_all["frequency"].eq("weekly")].copy()
    effect_scale = {
        (row.frequency, row.msdi_path, int(row.horizon)): float(
            row.effect_scale_frozen_calibration_residual_sd
        )
        for row in effects.loc[effects["learner"].eq(primary_learner)].itertuples()
    }
    rows: list[dict[str, Any]] = []
    for key, group in learner_all[learner_all["frequency"].eq("monthly")].groupby(
        ["frequency", "frequency_role", "msdi_path", "horizon"], sort=True
    ):
        rows.append(
            {
                "frequency": key[0],
                "frequency_role": key[1],
                "msdi_path": key[2],
                "horizon": int(key[3]),
                "learner": primary_learner,
                "modifier": "all_pre_specified_modifiers",
                "group": "not_estimated",
                "n_observations": len(group),
                "minimum_group_rows": math.nan,
                "estimation_status": "skipped_monthly_low_power",
                "interpretation": (
                    "monthly_primary_sample_is_too_small_for_"
                    "credible_heterogeneity"
                ),
            }
        )
    keys = ["frequency", "frequency_role", "msdi_path", "horizon"]
    for key, group in weekly_robustness.groupby(keys, sort=True):
        frequency, role, path_name, horizon = key
        scale = effect_scale[(frequency, path_name, int(horizon))]
        if int(horizon) not in {int(item) for item in requested_horizons[frequency]}:
            rows.append(
                {
                    "frequency": frequency,
                    "frequency_role": role,
                    "msdi_path": path_name,
                    "horizon": int(horizon),
                    "learner": primary_learner,
                    "modifier": "all_pre_specified_modifiers",
                    "group": "not_estimated",
                    "n_observations": len(group),
                    "minimum_group_rows": minimum_group_rows[frequency],
                    "estimation_status": "skipped_not_pre_specified_heterogeneity_horizon",
                    "interpretation": "heterogeneity_pre_specified_at_four_week_horizon_only",
                }
            )
            continue
        for modifier in HETEROGENEITY_MODIFIERS:
            configured_minimum = minimum_group_rows[frequency]
            if configured_minimum is None:
                raise AssertionError("Weekly heterogeneity minimum cannot be null")
            minimum = int(configured_minimum)
            counts = group.groupby(modifier, dropna=False).size()
            categories = sorted(
                str(category)
                for category, count in counts.items()
                if str(category) not in {"nan", "unavailable"} and count >= minimum
            )
            for category, count in counts.items():
                category_name = str(category)
                if category_name in categories:
                    continue
                rows.append(
                    {
                        "frequency": frequency,
                        "frequency_role": role,
                        "msdi_path": path_name,
                        "horizon": int(horizon),
                        "learner": primary_learner,
                        "modifier": modifier,
                        "group": category_name,
                        "n_observations": int(count),
                        "minimum_group_rows": minimum,
                        "estimation_status": "skipped_insufficient_or_unavailable_group",
                        "interpretation": "exploratory_R_loss_GATE_BLP_not_causal_forest",
                    }
                )
            if len(categories) < 2:
                for category_name in categories:
                    rows.append(
                        {
                            "frequency": frequency,
                            "frequency_role": role,
                            "msdi_path": path_name,
                            "horizon": int(horizon),
                            "learner": primary_learner,
                            "modifier": modifier,
                            "group": category_name,
                            "n_observations": int(counts[category_name]),
                            "minimum_group_rows": minimum,
                            "estimation_status": "skipped_fewer_than_two_supported_groups",
                            "interpretation": "exploratory_R_loss_GATE_BLP_not_causal_forest",
                        }
                    )
                continue
            work = group[group[modifier].astype(str).isin(categories)].sort_values("date")
            d = work["treatment_residual_oof"].to_numpy(dtype=float)
            y = work["outcome_residual"].to_numpy(dtype=float)
            design = np.column_stack(
                [d * work[modifier].astype(str).eq(category).to_numpy() for category in categories]
            )
            fitted = sm.OLS(y, design, missing="raise").fit(
                cov_type="HAC",
                cov_kwds={"maxlags": max(1, int(horizon)), "use_correction": True},
            )
            contrast_matrix = np.zeros((len(categories) - 1, len(categories)))
            for offset in range(1, len(categories)):
                contrast_matrix[offset - 1, 0] = -1
                contrast_matrix[offset - 1, offset] = 1
            omnibus = fitted.wald_test(contrast_matrix, scalar=True)
            omnibus_statistic = float(np.asarray(omnibus.statistic).squeeze())
            omnibus_p_value = float(np.asarray(omnibus.pvalue).squeeze())
            covariance = np.asarray(fitted.cov_params(), dtype=float)
            critical = float(norm.ppf(1 - alpha / 2))
            reference = categories[0]
            for index, category_name in enumerate(categories):
                coefficient = float(fitted.params[index])
                standard_error = float(fitted.bse[index])
                effect = coefficient * scale
                effect_se = standard_error * scale
                lower = effect - critical * effect_se
                upper = effect + critical * effect_se
                if index == 0:
                    contrast_effect = math.nan
                    contrast_se = math.nan
                    contrast_z = math.nan
                    contrast_p = math.nan
                else:
                    vector = np.zeros(len(categories))
                    vector[0] = -1
                    vector[index] = 1
                    contrast_effect = float(vector @ fitted.params) * scale
                    contrast_se = float(math.sqrt(vector @ covariance @ vector)) * scale
                    contrast_z = contrast_effect / contrast_se if contrast_se > 0 else math.nan
                    contrast_p = (
                        float(2 * norm.sf(abs(contrast_z)))
                        if np.isfinite(contrast_z)
                        else math.nan
                    )
                rows.append(
                    {
                        "frequency": frequency,
                        "frequency_role": role,
                        "msdi_path": path_name,
                        "horizon": int(horizon),
                        "learner": primary_learner,
                        "modifier": modifier,
                        "group": category_name,
                        "reference_group": reference,
                        "n_observations": int(counts[category_name]),
                        "regression_observations": len(work),
                        "minimum_group_rows": minimum,
                        "estimation_status": "estimated_R_loss_GATE_BLP",
                        "method": "residual_on_residual_group_interactions_with_HAC",
                        "threshold_provenance": (
                            "fixed_calendar_or_expanding_past_only_median"
                        ),
                        "coefficient_per_raw_treatment_unit": coefficient,
                        "effect_scale_frozen_calibration_residual_sd": scale,
                        "effect_log_points_per_one_sd_innovation": effect,
                        "effect_percent_per_one_sd_innovation": float(
                            100 * np.expm1(effect)
                        ),
                        "standard_error_hac": effect_se,
                        "z_statistic": effect / effect_se if effect_se > 0 else math.nan,
                        "p_value": float(fitted.pvalues[index]),
                        "ci_lower_log_points": lower,
                        "ci_upper_log_points": upper,
                        "ci_lower_percent": float(100 * np.expm1(lower)),
                        "ci_upper_percent": float(100 * np.expm1(upper)),
                        "contrast_effect_log_points_vs_reference": contrast_effect,
                        "contrast_standard_error_hac": contrast_se,
                        "contrast_z_statistic": contrast_z,
                        "contrast_p_value": contrast_p,
                        "heterogeneity_omnibus_wald_statistic": omnibus_statistic,
                        "heterogeneity_omnibus_p_value": omnibus_p_value,
                        "interpretation": (
                            "exploratory_group_orthogonalized_predictive_effect_and_"
                            "formal_contrast_not_causal_forest"
                        ),
                    }
                )
    output = pd.DataFrame(rows)
    output["p_value_fdr_all"] = np.nan
    output["contrast_p_value_fdr_all"] = np.nan
    output["heterogeneity_omnibus_p_value_fdr"] = np.nan
    estimated = output["estimation_status"].eq("estimated_R_loss_GATE_BLP")
    if estimated.any():
        output.loc[estimated, "p_value_fdr_all"] = adjust_pvalues(
            output.loc[estimated, "p_value"], "fdr_bh"
        )
        contrast = estimated & output["contrast_p_value"].notna()
        output.loc[contrast, "contrast_p_value_fdr_all"] = adjust_pvalues(
            output.loc[contrast, "contrast_p_value"], "fdr_bh"
        )
        omnibus_rows = output.loc[estimated].drop_duplicates(
            ["frequency", "msdi_path", "horizon", "modifier"]
        )
        omnibus_adjusted = adjust_pvalues(
            omnibus_rows["heterogeneity_omnibus_p_value"], "fdr_bh"
        )
        mapping = {
            (
                row.frequency,
                row.msdi_path,
                int(row.horizon),
                row.modifier,
            ): float(omnibus_adjusted.loc[index])
            for index, row in omnibus_rows.iterrows()
        }
        for index in output.index[estimated]:
            key = (
                output.at[index, "frequency"],
                output.at[index, "msdi_path"],
                int(output.at[index, "horizon"]),
                output.at[index, "modifier"],
            )
            output.at[index, "heterogeneity_omnibus_p_value_fdr"] = mapping[key]
    return output


def learner_stability(effects: pd.DataFrame, primary_learner: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["frequency", "frequency_role", "msdi_path", "horizon"]
    value = "effect_log_points_per_one_sd_innovation"
    for key, group in effects.groupby(keys, sort=True):
        values = group.set_index("learner")[value]
        primary_effect = float(values[primary_learner]) if primary_learner in values else math.nan
        signs = np.sign(values.to_numpy(dtype=float))
        rows.append(
            {
                "frequency": key[0],
                "frequency_role": key[1],
                "msdi_path": key[2],
                "horizon": int(key[3]),
                "primary_learner": primary_learner,
                "primary_effect_log_points": primary_effect,
                "learner_count": len(values),
                "minimum_effect_log_points": float(values.min()),
                "maximum_effect_log_points": float(values.max()),
                "effect_range_log_points": float(values.max() - values.min()),
                "maximum_absolute_deviation_from_primary": float(
                    np.max(np.abs(values.to_numpy(dtype=float) - primary_effect))
                ),
                "all_learner_signs_agree": bool(np.all(signs == signs[0])),
                "learner_sign_reversal_model_dependence_flag": bool(
                    not np.all(signs == signs[0])
                ),
                "learner_effects_json": json.dumps(values.to_dict(), sort_keys=True),
                "interpretation": "learner_sensitivity_not_effect_selection",
            }
        )
    return pd.DataFrame(rows)


def innovation_comparison(
    predictions: pd.DataFrame, innovations: pd.DataFrame
) -> pd.DataFrame:
    required = {"date", "frequency", "msdi_path", "msdi_change", "msdi_innovation_std"}
    missing = sorted(required.difference(innovations.columns))
    if missing:
        raise ValueError(f"Script-08 innovation table lacks columns: {missing}")
    reference = innovations[list(required)].copy()
    reference["date"] = pd.to_datetime(reference["date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    keys = ["frequency", "frequency_role", "msdi_path", "horizon", "learner"]
    for key, group in predictions.groupby(keys, sort=True):
        merged = group.merge(
            reference,
            on=["date", "frequency", "msdi_path"],
            how="inner",
            validate="one_to_one",
            suffixes=("_dml", "_script08"),
        )
        raw = merged["treatment_raw_msdi_change"].to_numpy(dtype=float)
        oof = merged["treatment_residual_oof"].to_numpy(dtype=float)
        script08_raw = merged["msdi_change"].to_numpy(dtype=float)
        script08_innovation = merged["msdi_innovation_std"].to_numpy(dtype=float)
        rows.append(
            {
                "frequency": key[0],
                "frequency_role": key[1],
                "msdi_path": key[2],
                "horizon": int(key[3]),
                "learner": key[4],
                "matched_dates": len(merged),
                "raw_change_max_absolute_difference": (
                    float(np.max(np.abs(raw - script08_raw))) if len(merged) else math.nan
                ),
                "correlation_oof_innovation_with_script08_full_sample_innovation": (
                    float(np.corrcoef(oof, script08_innovation)[0, 1])
                    if len(merged) >= 3
                    else math.nan
                ),
                "correlation_raw_change_with_script08_innovation": (
                    float(np.corrcoef(raw, script08_innovation)[0, 1])
                    if len(merged) >= 3
                    else math.nan
                ),
                "script08_innovation_used_as_dml_treatment": False,
                "script08_estimation_scope": "full_sample",
                "role": "comparison_only_due_to_future_information_risk",
            }
        )
    return pd.DataFrame(rows)


def econometric_comparison(
    effects: pd.DataFrame, local_projections: pd.DataFrame
) -> pd.DataFrame:
    required = {
        "frequency",
        "msdi_path",
        "horizon",
        "coefficient_log_points",
        "effect_percent",
        "p_value",
        "outcome_window",
    }
    missing = sorted(required.difference(local_projections.columns))
    if missing:
        raise ValueError(f"Local-projection table lacks columns: {missing}")
    comparison = effects.merge(
        local_projections[list(required)],
        on=["frequency", "msdi_path", "horizon"],
        how="left",
        validate="many_to_one",
        suffixes=("_dml", "_local_projection"),
    )
    comparison["effect_difference_log_points_dml_minus_lp"] = (
        comparison["effect_log_points_per_one_sd_innovation"]
        - comparison["coefficient_log_points"]
    )
    comparison["effect_sign_agreement"] = np.sign(
        comparison["effect_log_points_per_one_sd_innovation"]
    ).eq(np.sign(comparison["coefficient_log_points"]))
    comparison["outcome_window_exactly_aligned"] = True
    comparison["treatment_construction_exactly_aligned"] = False
    comparison["estimand_exactly_aligned"] = False
    comparison["estimand_difference"] = (
        "same_impact_inclusive_outcome_but_DML_uses_past_only_OOF_innovation_"
        "and_frozen_calibration_scale_while_LP_uses_full_sample_ARX_innovation_scale"
    )
    comparison["point_difference_role"] = "descriptive_triangulation_only"
    return comparison


def publication_scope_complete(
    *,
    args: argparse.Namespace,
    requested_frequencies: Sequence[str],
    requested_paths: Sequence[str],
    requested_learners: Sequence[str],
    bootstrap_repetitions: int,
) -> bool:
    return bool(
        not args.smoke_test
        and args.maximum_evaluation_rows is None
        and set(requested_frequencies) == {"weekly", "monthly"}
        and set(requested_paths) == set(MSDI_PATHS)
        and set(requested_learners) == set(LEARNERS)
        and bootstrap_repetitions > 0
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    config_path = require_file(args.config, "causal-ML configuration")
    config = load_yaml(config_path)
    specs = frequency_specifications(config)
    version = args.version or str(config["version"])
    logger, log_path = configure_logging(args.log_dir.resolve(), version)
    requested_frequencies = (
        ["weekly", "monthly"] if args.frequency == "both" else [args.frequency]
    )
    requested_paths = parse_requested(args.paths, MSDI_PATHS, "MSDI path")
    requested_learners = parse_requested(args.learners, LEARNERS, "learner")
    primary_learner = str(config["learners"]["primary"])
    maximum_evaluation_rows = args.maximum_evaluation_rows
    if args.smoke_test:
        requested_learners = [primary_learner]
        maximum_evaluation_rows = maximum_evaluation_rows or 12
    configured_repetitions = int(
        config["inference"]["moving_block_bootstrap_repetitions"]
    )
    bootstrap_repetitions = (
        configured_repetitions
        if args.bootstrap_repetitions is None
        else args.bootstrap_repetitions
    )
    if args.smoke_test:
        bootstrap_repetitions = min(bootstrap_repetitions, 19)
    alpha = float(config["inference"]["alpha"])
    innovation_path = require_file(args.innovations, "script-08 innovations")
    local_projection_path = require_file(
        args.local_projections, "script-08 local projections"
    )
    causality_manifest_path = require_file(
        args.causality_manifest, "script-08 manifest"
    )
    stationarity_path = require_file(
        args.stationarity_tests, "script-08 stationarity tests"
    )
    var_lag_path = require_file(args.var_lag_selection, "script-08 VAR lag selection")
    var_diagnostics_path = require_file(
        args.var_diagnostics, "script-08 VAR diagnostics"
    )
    script08_contract = validate_script08_contract(
        causality_manifest_path,
        innovation_path,
        local_projection_path,
        {
            "stationarity_tests": stationarity_path,
            "var_lag_selection": var_lag_path,
            "var_diagnostics": var_diagnostics_path,
        },
    )
    script08_triangulation = summarize_script08_triangulation(
        stationarity_path, var_lag_path, var_diagnostics_path
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting version=%s frequencies=%s paths=%s learners=%s smoke=%s "
        "bootstrap_repetitions=%s max_evaluation_rows=%s",
        version,
        requested_frequencies,
        requested_paths,
        requested_learners,
        args.smoke_test,
        bootstrap_repetitions,
        maximum_evaluation_rows,
    )

    all_predictions: list[pd.DataFrame] = []
    all_audits: list[pd.DataFrame] = []
    treatment_scales: dict[tuple[str, str], float] = {}
    treatment_scale_audits: list[dict[str, Any]] = []
    sample_audits: list[dict[str, Any]] = []
    feature_sets: dict[str, Any] = {}
    input_paths: dict[str, Path] = {
        "configuration": config_path,
        "script08_manifest": causality_manifest_path,
        "script08_innovations": innovation_path,
        "script08_local_projections": local_projection_path,
        "script08_stationarity_tests": stationarity_path,
        "script08_var_lag_selection": var_lag_path,
        "script08_var_diagnostics": var_diagnostics_path,
    }
    for frequency in requested_frequencies:
        spec = specs[frequency]
        panel_path = require_file(spec.panel_path, f"{frequency} causal panel")
        input_paths[f"{frequency}_panel"] = panel_path
        panel = load_panel(panel_path, spec)
        panel, sample_audit = apply_panel_completeness_contract(panel, spec)
        sample_audits.append(sample_audit)
        logger.info(
            "Loaded frequency=%s rows=%s range=%s to %s",
            frequency,
            len(panel),
            panel[spec.date_column].min(),
            panel[spec.date_column].max(),
        )
        horizons = spec.horizons[:1] if args.smoke_test else spec.horizons
        for path_name in requested_paths:
            frame, feature_columns = prepare_analysis_frame(panel, spec, path_name)
            feature_sets[f"{frequency}__{path_name}"] = {
                "features": feature_columns,
                "feature_count": len(feature_columns),
                "predetermined_only": True,
                "maximum_lag": max(spec.confounder_lags),
                "outcomes": [f"outcome_h{h}" for h in horizons],
                "placebo_outcomes": [f"placebo_outcome_h{h}" for h in horizons],
                "treatment_level_innovation_reparameterization": (
                    "M_t-(M_t_minus_1+E[delta_M_t|X_t])="
                    "delta_M_t-E[delta_M_t|X_t]"
                ),
            }
            reference_horizon = max(horizons)
            evaluation_block_size = spec.refit_every
            if args.smoke_test and maximum_evaluation_rows is not None:
                evaluation_block_size = max(
                    1, int(math.ceil(maximum_evaluation_rows / 2))
                )
            common_folds = build_expanding_folds(
                frame,
                reference_horizon,
                feature_columns,
                spec.minimum_training_rows,
                evaluation_block_size,
                maximum_evaluation_rows,
            )
            logger.info(
                "fold_design frequency=%s path=%s reference_h=%s folds=%s "
                "refit_every=%s first_eval=%s last_eval=%s minimum_train=%s "
                "strict_embargo=%s",
                frequency,
                path_name,
                reference_horizon,
                len(common_folds),
                evaluation_block_size,
                common_folds[0].evaluation_start,
                common_folds[-1].evaluation_end,
                spec.minimum_training_rows,
                reference_horizon,
            )
            scale, scale_audit = calibrate_treatment_scale(
                frame=frame,
                feature_columns=feature_columns,
                cutoff_position_exclusive=common_folds[0].evaluation_start,
                spec=spec,
                path_name=path_name,
                learner=primary_learner,
                learner_parameters=dict(config["learners"][primary_learner]),
                seed=args.seed,
            )
            treatment_scales[(frequency, path_name)] = scale
            treatment_scale_audits.append(scale_audit)
            logger.info(
                "treatment_scale frequency=%s path=%s learner=%s scale=%.6f "
                "calibration_train=%s calibration_validation=%s cutoff=%s",
                frequency,
                path_name,
                primary_learner,
                scale,
                scale_audit["training_rows"],
                scale_audit["validation_rows"],
                scale_audit["cutoff_date_exclusive"],
            )
            for horizon in horizons:
                for learner in requested_learners:
                    predictions, audits = cross_fit_configuration(
                        frame=frame,
                        feature_columns=feature_columns,
                        folds=common_folds,
                        spec=spec,
                        path_name=path_name,
                        horizon=horizon,
                        reference_horizon=reference_horizon,
                        learner=learner,
                        learner_parameters=dict(config["learners"][learner]),
                        seed=args.seed,
                        logger=logger,
                    )
                    all_predictions.append(predictions)
                    all_audits.append(audits)

    predictions = pd.concat(all_predictions, ignore_index=True)
    nuisance_audit = pd.concat(all_audits, ignore_index=True)
    if not nuisance_audit["horizon_purge_passed"].all():
        raise AssertionError("At least one nuisance fold failed its horizon purge")
    if not nuisance_audit["no_future_training_passed"].all():
        raise AssertionError("At least one nuisance fold used future training rows")
    predictions, common_sample_audit = enforce_common_samples(
        predictions, requested_paths, requested_learners
    )
    treatment_invariance_audit = treatment_horizon_invariance_audit(predictions)
    if predictions["innovation_reparameterization_absolute_difference"].max() > 1e-12:
        raise AssertionError(
            "Level-innovation and stationary-change residual parameterizations differ"
        )
    effects, placebos = estimate_effects(
        predictions,
        treatment_scales=treatment_scales,
        primary_learner=primary_learner,
        alpha=alpha,
        bootstrap_repetitions=bootstrap_repetitions,
        requested_block_length=args.block_length,
        seed=args.seed,
        logger=logger,
    )
    effects = add_primary_simultaneous_bands(
        effects,
        predictions,
        primary_learner=primary_learner,
        alpha=alpha,
        bootstrap_repetitions=bootstrap_repetitions,
        requested_block_length=args.block_length,
        seed=args.seed,
    )
    overlap = overlap_diagnostics(predictions, nuisance_audit)
    heterogeneity = heterogeneous_group_effects(
        predictions,
        effects,
        primary_learner=primary_learner,
        minimum_group_rows={
            str(key): (None if value is None else int(value))
            for key, value in config["heterogeneity"]["minimum_group_rows"].items()
        },
        requested_horizons={
            str(key): [int(item) for item in value]
            for key, value in config["heterogeneity"]["horizons"].items()
        },
        alpha=alpha,
        seed=args.seed,
    )
    stability = learner_stability(effects, primary_learner)
    innovations = pd.read_csv(innovation_path, low_memory=False)
    innovation_audit = innovation_comparison(predictions, innovations)
    local_projections = pd.read_csv(local_projection_path, low_memory=False)
    econometric = econometric_comparison(effects, local_projections)
    for row in econometric.itertuples():
        logger.info(
            "econometric_diff frequency=%s path=%s h=%s learner=%s "
            "dml=%.6f lp=%.6f diff=%.6f exact_estimand_match=%s",
            row.frequency,
            row.msdi_path,
            row.horizon,
            row.learner,
            row.effect_log_points_per_one_sd_innovation,
            row.coefficient_log_points,
            row.effect_difference_log_points_dml_minus_lp,
            row.estimand_exactly_aligned,
        )

    prefix = output_dir / version
    paths = {
        "dml_estimates": prefix.with_name(prefix.name + "_dml_estimates.csv"),
        "oof_predictions": prefix.with_name(prefix.name + "_oof_predictions.csv"),
        "nuisance_fold_audit": prefix.with_name(
            prefix.name + "_nuisance_fold_audit.csv"
        ),
        "common_sample_audit": prefix.with_name(
            prefix.name + "_common_sample_audit.csv"
        ),
        "sample_contract_audit": prefix.with_name(
            prefix.name + "_sample_contract_audit.csv"
        ),
        "treatment_scale_audit": prefix.with_name(
            prefix.name + "_treatment_scale_audit.csv"
        ),
        "treatment_invariance_audit": prefix.with_name(
            prefix.name + "_treatment_invariance_audit.csv"
        ),
        "overlap_diagnostics": prefix.with_name(
            prefix.name + "_overlap_diagnostics.csv"
        ),
        "placebo_tests": prefix.with_name(prefix.name + "_placebo_tests.csv"),
        "heterogeneous_group_effects": prefix.with_name(
            prefix.name + "_heterogeneous_group_effects.csv"
        ),
        "learner_stability": prefix.with_name(
            prefix.name + "_learner_stability.csv"
        ),
        "innovation_comparison": prefix.with_name(
            prefix.name + "_innovation_comparison.csv"
        ),
        "econometric_comparison": prefix.with_name(
            prefix.name + "_econometric_comparison.csv"
        ),
        "feature_sets": prefix.with_name(prefix.name + "_feature_sets.json"),
        "manifest": prefix.with_name(prefix.name + "_manifest.json"),
    }
    tables = {
        "dml_estimates": effects,
        "oof_predictions": predictions,
        "nuisance_fold_audit": nuisance_audit,
        "common_sample_audit": common_sample_audit,
        "sample_contract_audit": pd.DataFrame(sample_audits),
        "treatment_scale_audit": pd.DataFrame(treatment_scale_audits),
        "treatment_invariance_audit": treatment_invariance_audit,
        "overlap_diagnostics": overlap,
        "placebo_tests": placebos,
        "heterogeneous_group_effects": heterogeneity,
        "learner_stability": stability,
        "innovation_comparison": innovation_audit,
        "econometric_comparison": econometric,
    }
    for name, table in tables.items():
        write_csv(table, paths[name])
        logger.info("Wrote output=%s rows=%s path=%s", name, len(table), paths[name])
    write_json(feature_sets, paths["feature_sets"])

    complete_scope = publication_scope_complete(
        args=args,
        requested_frequencies=requested_frequencies,
        requested_paths=requested_paths,
        requested_learners=requested_learners,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    status = (
        "provisional_orthogonalized_predictive_effects_only"
        if complete_scope
        else "debug_or_partial_run_not_for_publication"
    )
    manifest = {
        "version": version,
        "status": status,
        "publication_scope_complete": complete_scope,
        "causal_claim_allowed": False,
        "primary_frequency": "monthly",
        "weekly_role": "higher_frequency_robustness",
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in input_paths.items()
        },
        "arguments": vars(args),
        "design": {
            "estimand": (
                "average_partial_effect_per_one_frozen_calibration_sd_of_"
                "current_msdi_level_innovation"
            ),
            "outcome": "log_scfi_t_plus_h_minus_log_scfi_t_minus_1",
            "treatment": (
                "current_msdi_level_innovation_reparameterized_as_raw_msdi_"
                "first_difference_residualized_with_past_only_expanding_window_"
                "nuisance_model"
            ),
            "innovation_reparameterization_identity": (
                "M_t-(M_t_minus_1+m_delta(X_t))="
                "delta_M_t-m_delta(X_t)"
            ),
            "lagged_msdi_level_mandatory_in_confounders": True,
            "cross_fitting": (
                "expanding_window_refits_with_one_maximum_horizon_common_design_"
                "and_strict_embargo"
            ),
            "ordinary_random_kfold_used": False,
            "nuisance_learners": requested_learners,
            "primary_learner": primary_learner,
            "learner_selected_by_treatment_effect": False,
            "inference": (
                "newey_west_hac_pointwise_intervals_and_fixed_nuisance_shared_"
                "moving_block_bootstrap_sup_t_band_for_primary_dynamic_curve"
            ),
            "bootstrap_nuisance_models_refitted": False,
            "common_sample_enforced_across_horizons_paths_and_learners": True,
            "treatment_nuisance_identical_across_outcome_horizons": True,
            "treatment_scale": (
                "single_forward_holdout_residual_sd_before_first_evaluation_"
                "constant_across_horizons_and_learners"
            ),
            "primary_treatment_path": config["treatment"]["primary_path"],
            "script08_full_sample_innovations_used_as_treatment": False,
            "script08_full_sample_innovations_role": "comparison_only",
            "causal_forest_run": False,
            "causal_forest_reason": config["heterogeneity"]["causal_forest_reason"],
            "heterogeneity_method": (
                "pre_specified_R_loss_GATE_BLP_group_interactions_with_HAC_"
                "and_formal_contrasts"
            ),
            "heterogeneity_horizons": config["heterogeneity"]["horizons"],
            "monthly_heterogeneity_run": False,
            "monthly_heterogeneity_reason": "low_power_aggregate_monthly_sample",
            "dml_iv_run": False,
            "dml_iv_reason": "no_defensible_instrument_encoded",
        },
        "identification": {
            **config["identification"],
            "required_assumptions": [
                "consistency",
                "temporal_ordering",
                "conditional_exchangeability_given_recorded_predetermined_controls",
                "continuous_treatment_overlap",
                "correctly_measured_treatment_and_outcome",
            ],
            "unverified_or_unavailable": [
                "conditional_exchangeability",
                "external_instrument",
                "negative_control_article_category",
                "route_exposure_panel",
                "human_validated_final_index",
                "complete_external_confounder_set_including_trade_demand_and_route_connectivity",
            ],
            "permitted_language": (
                "orthogonalized predictive effect or conditional dynamic association"
            ),
            "prohibited_language": "structural causal effect",
        },
        "falsification": {
            "pre_treatment_outcome_placebo_run": True,
            "future_treatment_lead_placebo_run": False,
            "future_treatment_lead_reason": (
                "valid_leads_must_exceed_each_outcome_horizon_and_use_a_separate_"
                "max_horizon_lead_embargo;scheduled_for_09_robustness"
            ),
            "negative_control_run": False,
            "negative_control_reason": "no defensible negative-control series available",
        },
        "scope": {
            "frequencies": requested_frequencies,
            "paths": requested_paths,
            "learners": requested_learners,
            "smoke_test": args.smoke_test,
            "maximum_evaluation_rows": maximum_evaluation_rows,
            "bootstrap_repetitions": bootstrap_repetitions,
        },
        "audit": {
            "all_horizon_purges_passed": bool(
                nuisance_audit["horizon_purge_passed"].all()
            ),
            "all_no_future_training_checks_passed": bool(
                nuisance_audit["no_future_training_passed"].all()
            ),
            "all_common_samples_balanced": bool(
                common_sample_audit["balanced_common_sample_passed"].all()
            ),
            "all_treatment_horizon_invariance_checks_passed": bool(
                treatment_invariance_audit[
                    "treatment_nuisance_horizon_invariance_passed"
                ].all()
            ),
            "maximum_innovation_reparameterization_absolute_difference": float(
                predictions[
                    "innovation_reparameterization_absolute_difference"
                ].max()
            ),
            "continuous_treatment_diagnostic_flags": int(
                overlap["continuous_treatment_diagnostic_flag"].sum()
            ),
            "primary_falsification_flags_raw": int(
                (
                    placebos["primary_family"]
                    & placebos["falsification_flag_raw"]
                ).sum()
            ),
            "primary_falsification_flags_fdr": int(
                (
                    placebos["primary_family"]
                    & placebos["falsification_flag_fdr"]
                ).sum()
            ),
            "script08_contract": script08_contract,
            "script08_triangulation": script08_triangulation,
            "panel_sample_contracts": sample_audits,
        },
        "counts": {name: len(table) for name, table in tables.items()},
        "outputs": {name: str(path) for name, path in paths.items()},
        "output_sha256": {
            name: sha256_file(path)
            for name, path in paths.items()
            if name != "manifest" and path.is_file()
        },
        "log": str(log_path),
        "software": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit_learn": package_version("scikit-learn"),
            "scipy": package_version("scipy"),
            "statsmodels": package_version("statsmodels"),
        },
        "code": {
            "script": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(manifest, paths["manifest"])
    logger.info(
        "Completed status=%s effects=%s oof_rows=%s manifest=%s",
        status,
        len(effects),
        len(predictions),
        paths["manifest"],
    )


if __name__ == "__main__":
    main()
