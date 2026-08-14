"""Run leakage-safe rolling-origin SCFI forecasting benchmarks.

The primary forecast target is the forward log change in SCFI. Forecast levels
are reconstructed for interpretable level errors. Peak-severity and disruption-
breadth MSDI paths are evaluated separately and jointly against autoregressive
and Brent-controlled baselines on identical forecast dates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    configured_loky_cores = int(os.environ.get("LOKY_MAX_CPU_COUNT", "0"))
except ValueError:
    configured_loky_cores = 0
if configured_loky_cores < 1:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(os.cpu_count() or 1, 1))

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge, TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = PROJECT_ROOT / "data/06_model_ready/prediction/msdi_scfi_weekly_panel.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/prediction_benchmarks"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs/logs"
HORIZONS = (1, 2, 4, 6)


@dataclass(frozen=True)
class ModelConfiguration:
    model_name: str
    feature_set: str

    @property
    def model_id(self) -> str:
        return f"{self.model_name}__{self.feature_set}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run expanding-window SCFI prediction benchmarks."
    )
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--version", default="forecast_v001")
    parser.add_argument("--minimum-training-rows", type=int, default=156)
    parser.add_argument(
        "--refit-every",
        type=int,
        default=13,
        help="Refit interval in forecast origins; 13 approximates quarterly updating.",
    )
    parser.add_argument(
        "--nonlinear-refit-every",
        type=int,
        default=52,
        help="Refit interval for gradient boosting; forecasts are still produced every week.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--glm-alpha",
        type=float,
        default=0.01,
        help=(
            "L2 penalty for the Gaussian identity-link GLM; fitted with "
            "training-window-only imputation and standardization."
        ),
    )
    parser.add_argument("--elastic-alpha", type=float, default=0.01)
    parser.add_argument("--elastic-l1-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def configure_logging(log_dir: Path, version: str) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"07_prediction_benchmarks_{version}.log"
    logger = logging.getLogger("prediction_benchmarks")
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


def unique_existing(columns: list[str], available: set[str], label: str) -> list[str]:
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"Panel lacks {label} features: {missing}")
    return list(dict.fromkeys(columns))


def build_feature_sets(columns: list[str]) -> dict[str, list[str]]:
    available = set(columns)
    ar = unique_existing(
        [
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
        ],
        available,
        "autoregressive",
    )
    brent = unique_existing(
        [
            "brent_log",
            "brent_log_lag_1w",
            "brent_log_lag_2w",
            "brent_log_lag_4w",
            "brent_log_lag_8w",
            "brent_log_lag_12w",
            "brent_log_change_1w",
            "brent_log_change_4w",
            "brent_log_change_13w",
        ],
        available,
        "Brent",
    )
    external_controls = unique_existing(
        [
            *[f"cpb_world_trade_volume_lag_{lag}w" for lag in (1, 2, 4, 8, 12)],
            *[f"unctad_china_core_lane_lsbci_lag_{lag}w" for lag in (1, 2, 4, 8, 12)],
        ],
        available,
        "released external controls",
    )
    shared_news = unique_existing(
        [
            "article_count",
            "source_count",
            "effective_article_count",
        ],
        available,
        "news coverage",
    )
    peak = unique_existing(
        [
            "peak_score_media_corrected",
            "peak_msdi_z_expanding_realtime",
            "peak_score_media_corrected_lag_1w",
            "peak_score_media_corrected_lag_2w",
            "peak_score_media_corrected_lag_4w",
            "peak_score_media_corrected_lag_8w",
            "peak_score_media_corrected_lag_12w",
            "peak_score_media_corrected_rolling_mean_4w",
            "peak_score_media_corrected_rolling_sd_4w",
            "peak_score_media_corrected_rolling_mean_13w",
            "peak_score_media_corrected_rolling_sd_13w",
            "peak_score_media_corrected_rolling_mean_26w",
            "peak_score_media_corrected_rolling_sd_26w",
            "peak_score_media_corrected_rolling_mean_52w",
            "peak_score_media_corrected_rolling_sd_52w",
            "peak_score_media_corrected_acceleration_1w",
            "peak_active_article_count",
            "peak_model_disagreement_mean",
            *shared_news,
        ],
        available,
        "peak MSDI",
    )
    breadth = unique_existing(
        [
            "breadth_score_media_corrected",
            "breadth_msdi_z_expanding_realtime",
            "breadth_score_media_corrected_lag_1w",
            "breadth_score_media_corrected_lag_2w",
            "breadth_score_media_corrected_lag_4w",
            "breadth_score_media_corrected_lag_8w",
            "breadth_score_media_corrected_lag_12w",
            "breadth_score_media_corrected_rolling_mean_4w",
            "breadth_score_media_corrected_rolling_sd_4w",
            "breadth_score_media_corrected_rolling_mean_13w",
            "breadth_score_media_corrected_rolling_sd_13w",
            "breadth_score_media_corrected_rolling_mean_26w",
            "breadth_score_media_corrected_rolling_sd_26w",
            "breadth_score_media_corrected_rolling_mean_52w",
            "breadth_score_media_corrected_rolling_sd_52w",
            "breadth_score_media_corrected_acceleration_1w",
            "breadth_active_article_count",
            "dimension_disagreement_mean",
            *shared_news,
        ],
        available,
        "breadth MSDI",
    )
    feature_sets = {
        "ar": ar,
        "ar_brent": [*ar, *brent, *external_controls],
        "ar_brent_peak": list(dict.fromkeys([*ar, *brent, *external_controls, *peak])),
        "ar_brent_breadth": list(dict.fromkeys([*ar, *brent, *external_controls, *breadth])),
        "ar_brent_both": list(dict.fromkeys([*ar, *brent, *external_controls, *peak, *breadth])),
    }
    for name, features in feature_sets.items():
        forbidden = [feature for feature in features if feature.startswith("target_")]
        if forbidden:
            raise AssertionError(f"Target leakage in feature set {name}: {forbidden}")
    return feature_sets


def model_configurations() -> list[ModelConfiguration]:
    configurations = [ModelConfiguration("ridge", "ar")]
    matched_sets = [
        "ar_brent",
        "ar_brent_peak",
        "ar_brent_breadth",
        "ar_brent_both",
    ]
    for model_name in ("ridge", "glm", "elastic_net", "hist_gradient_boosting"):
        configurations.extend(
            ModelConfiguration(model_name, feature_set) for feature_set in matched_sets
        )
    return configurations


def make_estimator(config: ModelConfiguration, args: argparse.Namespace) -> RegressorMixin:
    if config.model_name == "ridge":
        regressor: RegressorMixin = Ridge(alpha=args.ridge_alpha)
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    elif config.model_name == "glm":
        regressor = TweedieRegressor(
            power=0,
            link="identity",
            alpha=args.glm_alpha,
            fit_intercept=True,
            solver="lbfgs",
            max_iter=5_000,
            tol=1e-8,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    elif config.model_name == "elastic_net":
        regressor = ElasticNet(
            alpha=args.elastic_alpha,
            l1_ratio=args.elastic_l1_ratio,
            max_iter=20_000,
            random_state=args.seed,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    elif config.model_name == "hist_gradient_boosting":
        regressor = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=12,
            min_samples_leaf=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=args.seed,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", regressor),
        ]
    else:
        raise ValueError(f"Unknown model: {config.model_name}")
    # The target transformer is fitted only on the current expanding training set.
    return TransformedTargetRegressor(
        regressor=Pipeline(steps),
        transformer=StandardScaler(),
    )


def load_panel(path: Path) -> pd.DataFrame:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Model panel not found: {resolved}")
    panel = pd.read_csv(resolved, low_memory=False)
    if "observation_date" not in panel or "scfi" not in panel:
        raise ValueError("Panel must contain observation_date and scfi")
    panel["observation_date"] = pd.to_datetime(
        panel["observation_date"], format="mixed", errors="coerce"
    )
    if panel["observation_date"].isna().any():
        raise ValueError("Panel contains unparseable observation dates")
    panel = panel.sort_values("observation_date").reset_index(drop=True)
    if panel["observation_date"].duplicated().any():
        raise ValueError("Panel contains duplicate observation dates")
    for column in panel.columns:
        if column in {"observation_date", "msdi_week_ending", "brent_observation_date"}:
            continue
        original = panel[column]
        normalized = original.astype("string").str.strip().str.lower()
        nonempty_values = set(normalized.dropna().unique().tolist())
        if nonempty_values and nonempty_values.issubset({"true", "false"}):
            panel[column] = normalized.map({"true": True, "false": False}).astype("boolean")
            continue
        converted = pd.to_numeric(original, errors="coerce")
        if converted.notna().sum() == original.notna().sum():
            panel[column] = converted
    return panel


def training_positions(
    panel: pd.DataFrame, origin_position: int, horizon: int, target_column: str
) -> np.ndarray:
    # A row s has an h-step label available at origin t only when s+h <= t.
    maximum_position = origin_position - horizon
    if maximum_position < 0:
        return np.array([], dtype=int)
    candidate = np.arange(maximum_position + 1)
    complete = panel.loc[candidate, "base_model_complete_case"].astype(bool).to_numpy()
    target_available = panel.loc[candidate, target_column].notna().to_numpy()
    return candidate[complete & target_available]


def eligible_origins(panel: pd.DataFrame, horizon: int, target_column: str) -> list[int]:
    target_date = panel["observation_date"].shift(-horizon)
    mask = (
        panel["base_model_complete_case"].astype(bool)
        & panel[target_column].notna()
        & target_date.notna()
    )
    return panel.index[mask].tolist()


def baseline_forecasts(
    panel: pd.DataFrame,
    horizon: int,
    minimum_training_rows: int,
) -> list[dict[str, Any]]:
    target_column = f"target_scfi_log_change_h{horizon}w"
    level_column = f"target_scfi_level_h{horizon}w"
    rows: list[dict[str, Any]] = []
    for position in eligible_origins(panel, horizon, target_column):
        train_positions = training_positions(panel, position, horizon, target_column)
        if len(train_positions) < minimum_training_rows:
            continue
        actual = float(panel.at[position, target_column])
        current_scfi = float(panel.at[position, "scfi"])
        target_date = panel.at[position + horizon, "observation_date"]
        predictions = {
            "naive_zero_change": 0.0,
            "historical_mean_change": float(panel.loc[train_positions, target_column].mean()),
        }
        for model_id, prediction in predictions.items():
            rows.append(
                forecast_row(
                    panel,
                    position,
                    horizon,
                    target_date,
                    actual,
                    current_scfi,
                    float(panel.at[position, level_column]),
                    prediction,
                    model_id,
                    "benchmark",
                    len(train_positions),
                    panel.at[train_positions[-1], "observation_date"],
                    panel.at[train_positions[-1] + horizon, "observation_date"],
                    panel.at[position, "observation_date"],
                )
            )
    return rows


def forecast_row(
    panel: pd.DataFrame,
    position: int,
    horizon: int,
    target_date: pd.Timestamp,
    actual: float,
    current_scfi: float,
    actual_level: float,
    prediction: float,
    model_id: str,
    feature_set: str,
    train_n: int,
    training_feature_end_date: pd.Timestamp,
    training_label_end_date: pd.Timestamp,
    model_fit_origin_date: pd.Timestamp,
) -> dict[str, Any]:
    predicted_level = current_scfi * math.exp(prediction)
    return {
        "origin_position": position,
        "origin_date": panel.at[position, "observation_date"],
        "target_date": target_date,
        "horizon_weeks": horizon,
        "model_id": model_id,
        "feature_set": feature_set,
        "actual_log_change": actual,
        "predicted_log_change": prediction,
        "forecast_error": actual - prediction,
        "current_scfi": current_scfi,
        "actual_scfi_level": actual_level,
        "predicted_scfi_level": predicted_level,
        "level_error": actual_level - predicted_level,
        "training_rows": train_n,
        "training_feature_end_date": training_feature_end_date,
        "training_label_end_date": training_label_end_date,
        "model_fit_origin_date": model_fit_origin_date,
    }


def rolling_model_forecasts(
    panel: pd.DataFrame,
    horizon: int,
    config: ModelConfiguration,
    features: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_column = f"target_scfi_log_change_h{horizon}w"
    level_column = f"target_scfi_level_h{horizon}w"
    rows: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    positions: list[int] = []
    for position in eligible_origins(panel, horizon, target_column):
        train_positions = training_positions(panel, position, horizon, target_column)
        if len(train_positions) >= args.minimum_training_rows:
            positions.append(position)

    refit_interval = (
        args.nonlinear_refit_every
        if config.model_name == "hist_gradient_boosting"
        else args.refit_every
    )
    for block_start in range(0, len(positions), refit_interval):
        block_positions = positions[block_start : block_start + refit_interval]
        fit_position = block_positions[0]
        train_positions = training_positions(panel, fit_position, horizon, target_column)
        estimator = make_estimator(config, args)
        estimator.fit(panel.loc[train_positions, features], panel.loc[train_positions, target_column])
        fit_metadata = {
            "training_rows": len(train_positions),
            "training_feature_end_date": panel.at[train_positions[-1], "observation_date"],
            "training_label_end_date": panel.at[train_positions[-1] + horizon, "observation_date"],
            "model_fit_origin_date": panel.at[fit_position, "observation_date"],
        }
        if fit_metadata["training_label_end_date"] > fit_metadata["model_fit_origin_date"]:
            raise AssertionError("Training label endpoint occurs after model-fit origin")
        fits.append(
            {
                "horizon_weeks": horizon,
                "model_id": config.model_id,
                "feature_set": config.feature_set,
                "feature_count": len(features),
                "refit_interval_origins": refit_interval,
                **fit_metadata,
            }
        )
        predictions = estimator.predict(panel.loc[block_positions, features])
        for position, prediction in zip(block_positions, predictions):
            actual = float(panel.at[position, target_column])
            current_scfi = float(panel.at[position, "scfi"])
            rows.append(
                forecast_row(
                    panel,
                    position,
                    horizon,
                    panel.at[position + horizon, "observation_date"],
                    actual,
                    current_scfi,
                    float(panel.at[position, level_column]),
                    float(prediction),
                    config.model_id,
                    config.feature_set,
                    int(fit_metadata["training_rows"]),
                    fit_metadata["training_feature_end_date"],
                    fit_metadata["training_label_end_date"],
                    fit_metadata["model_fit_origin_date"],
                )
            )
    return rows, fits


def calculate_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, horizon_frame in forecasts.groupby("horizon_weeks"):
        zero = horizon_frame[horizon_frame["model_id"].eq("naive_zero_change")].set_index("origin_date")
        historical = horizon_frame[
            horizon_frame["model_id"].eq("historical_mean_change")
        ].set_index("origin_date")
        for model_id, group in horizon_frame.groupby("model_id"):
            group = group.sort_values("origin_date").set_index("origin_date")
            error = group["actual_log_change"] - group["predicted_log_change"]
            level_error = group["actual_scfi_level"] - group["predicted_scfi_level"]
            aligned_zero = zero.reindex(group.index)
            aligned_historical = historical.reindex(group.index)
            zero_sse = np.sum(
                (aligned_zero["actual_log_change"] - aligned_zero["predicted_log_change"]) ** 2
            )
            historical_sse = np.sum(
                (
                    aligned_historical["actual_log_change"]
                    - aligned_historical["predicted_log_change"]
                )
                ** 2
            )
            model_sse = np.sum(error**2)
            correlation = group[["actual_log_change", "predicted_log_change"]].corr().iloc[0, 1]
            rows.append(
                {
                    "horizon_weeks": int(horizon),
                    "model_id": model_id,
                    "feature_set": group["feature_set"].iloc[0],
                    "n_forecasts": len(group),
                    "first_origin_date": group.index.min(),
                    "last_origin_date": group.index.max(),
                    "mae_log_change": np.mean(np.abs(error)),
                    "rmse_log_change": np.sqrt(np.mean(error**2)),
                    "bias_log_change": np.mean(group["predicted_log_change"] - group["actual_log_change"]),
                    "directional_accuracy": np.mean(
                        np.sign(group["actual_log_change"])
                        == np.sign(group["predicted_log_change"])
                    ),
                    "forecast_actual_correlation": correlation,
                    "mae_scfi_level": np.mean(np.abs(level_error)),
                    "rmse_scfi_level": np.sqrt(np.mean(level_error**2)),
                    "oos_r2_vs_zero_change": (
                        math.nan if zero_sse == 0 else 1 - model_sse / zero_sse
                    ),
                    "oos_r2_vs_historical_mean": (
                        math.nan if historical_sse == 0 else 1 - model_sse / historical_sse
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


def forecast_comparison_tests(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, horizon_frame in forecasts.groupby("horizon_weeks"):
        for challenger_id, challenger in horizon_frame.groupby("model_id"):
            if challenger_id in {
                "naive_zero_change",
                "historical_mean_change",
                "ridge__ar",
            } or challenger_id.endswith("__ar_brent"):
                continue
            algorithm = challenger_id.split("__", maxsplit=1)[0]
            baseline_id = f"{algorithm}__ar_brent"
            baseline = horizon_frame[horizon_frame["model_id"].eq(baseline_id)].set_index(
                "origin_date"
            )
            if baseline.empty:
                raise ValueError(f"Missing algorithm-matched baseline: {baseline_id}")
            challenger = challenger.set_index("origin_date")
            pair = baseline[
                ["actual_log_change", "predicted_log_change"]
            ].join(
                challenger[["predicted_log_change"]],
                how="inner",
                lsuffix="_baseline",
                rsuffix="_challenger",
            )
            actual = pair["actual_log_change"]
            base_prediction = pair["predicted_log_change_baseline"]
            challenge_prediction = pair["predicted_log_change_challenger"]
            base_error = actual - base_prediction
            challenge_error = actual - challenge_prediction
            comparisons = {
                "diebold_mariano_squared": base_error**2 - challenge_error**2,
                "diebold_mariano_absolute": base_error.abs() - challenge_error.abs(),
                "clark_west_nested_squared": (
                    base_error**2
                    - (challenge_error**2 - (base_prediction - challenge_prediction) ** 2)
                ),
            }
            for test_name, differential in comparisons.items():
                mean_difference, statistic, standard_error = hac_mean_test(
                    differential.to_numpy(), max(int(horizon) - 1, 0)
                )
                if test_name.startswith("clark_west"):
                    p_value = math.nan if pd.isna(statistic) else float(1 - norm.cdf(statistic))
                    alternative = "challenger improves baseline (one-sided)"
                else:
                    p_value = (
                        math.nan
                        if pd.isna(statistic)
                        else float(2 * (1 - norm.cdf(abs(statistic))))
                    )
                    alternative = "unequal forecast accuracy (two-sided)"
                rows.append(
                    {
                        "horizon_weeks": int(horizon),
                        "baseline_model": baseline_id,
                        "challenger_model": challenger_id,
                        "test": test_name,
                        "n": len(pair),
                        "hac_lag": max(int(horizon) - 1, 0),
                        "mean_loss_improvement": mean_difference,
                        "standard_error": standard_error,
                        "statistic": statistic,
                        "p_value": p_value,
                        "alternative": alternative,
                        "positive_favors": "challenger",
                    }
                )
    return pd.DataFrame(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def main() -> None:
    args = parse_args()
    if args.minimum_training_rows < 52:
        raise ValueError("minimum-training-rows must be at least 52")
    if args.refit_every < 1:
        raise ValueError("refit-every must be positive")
    if args.nonlinear_refit_every < 1:
        raise ValueError("nonlinear-refit-every must be positive")
    logger, log_path = configure_logging(args.log_dir.resolve(), args.version)
    logger.info("Starting forecasting benchmarks version=%s", args.version)
    logger.info("Arguments: %s", vars(args))

    panel_path = args.panel.resolve()
    panel = load_panel(panel_path)
    feature_sets = build_feature_sets(panel.columns.tolist())
    configurations = model_configurations()
    logger.info(
        "Panel rows=%s range=%s to %s",
        len(panel),
        panel["observation_date"].min(),
        panel["observation_date"].max(),
    )
    for name, features in feature_sets.items():
        logger.info("Feature set %s: %s variables", name, len(features))

    forecast_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        logger.info("Running horizon=%s weeks", horizon)
        forecast_rows.extend(baseline_forecasts(panel, horizon, args.minimum_training_rows))
        for config in configurations:
            rows, fits = rolling_model_forecasts(
                panel, horizon, config, feature_sets[config.feature_set], args
            )
            forecast_rows.extend(rows)
            fit_rows.extend(fits)
            logger.info(
                "Completed horizon=%s model=%s forecasts=%s fits=%s",
                horizon,
                config.model_id,
                len(rows),
                len(fits),
            )

    forecasts = pd.DataFrame(forecast_rows).sort_values(
        ["horizon_weeks", "origin_date", "model_id"]
    )
    fits = pd.DataFrame(fit_rows).sort_values(
        ["horizon_weeks", "model_id", "model_fit_origin_date"]
    )
    if forecasts.empty:
        raise ValueError("No forecasts were produced")
    if (
        pd.to_datetime(forecasts["training_label_end_date"])
        > pd.to_datetime(forecasts["model_fit_origin_date"])
    ).any():
        raise AssertionError("At least one forecast used a not-yet-observable training label")

    metrics = calculate_metrics(forecasts)
    statistical_tests = forecast_comparison_tests(forecasts)
    counts = forecasts.groupby(["horizon_weeks", "model_id"]).size().unstack()
    unequal = counts.nunique(axis=1).ne(1)
    if unequal.any():
        raise AssertionError(
            "Models were not evaluated on identical forecast counts:\n"
            f"{counts.loc[unequal].to_string()}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{args.version}_forecast_predictions.csv"
    metrics_path = output_dir / f"{args.version}_forecast_metrics.csv"
    tests_path = output_dir / f"{args.version}_forecast_statistical_tests.csv"
    fits_path = output_dir / f"{args.version}_model_fit_audit.csv"
    features_path = output_dir / f"{args.version}_feature_sets.json"
    manifest_path = output_dir / f"{args.version}_manifest.json"
    forecasts.to_csv(prediction_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    statistical_tests.to_csv(tests_path, index=False)
    fits.to_csv(fits_path, index=False)
    features_path.write_text(json.dumps(feature_sets, indent=2), encoding="utf-8")
    output_paths = {
        "predictions": prediction_path,
        "metrics": metrics_path,
        "statistical_tests": tests_path,
        "fit_audit": fits_path,
        "feature_sets": features_path,
    }
    output_sha256 = {
        key: sha256_file(path) for key, path in output_paths.items()
    }
    output_row_counts = {
        "predictions": int(len(forecasts)),
        "metrics": int(len(metrics)),
        "statistical_tests": int(len(statistical_tests)),
        "fit_audit": int(len(fits)),
        "feature_sets": 1,
    }

    best = (
        metrics.sort_values(["horizon_weeks", "rmse_log_change"])
        .groupby("horizon_weeks", as_index=False)
        .first()[["horizon_weeks", "model_id", "rmse_log_change", "oos_r2_vs_zero_change"]]
    )
    manifest = {
        "version": args.version,
        "status": "provisional_pending_human_validation",
        "input": {
            "path": str(panel_path),
            "sha256": sha256_file(panel_path),
            "rows": len(panel),
        },
        "design": {
            "evaluation": "expanding-window rolling-origin",
            "training_label_purge": "training row s allowed only when s + horizon <= origin",
            "refit_every_origins": args.refit_every,
            "nonlinear_refit_every_origins": args.nonlinear_refit_every,
            "minimum_training_rows": args.minimum_training_rows,
            "horizons_weeks": list(HORIZONS),
            "model_families": [
                "ridge",
                "glm",
                "elastic_net",
                "hist_gradient_boosting",
            ],
            "target": "forward SCFI log change",
            "identical_evaluation_dates_enforced": True,
            "random_train_test_split": False,
            "preprocessing_fit_inside_training_window": True,
            "glm_specification": {
                "family": "Gaussian (Tweedie power=0)",
                "link": "identity",
                "feature_preprocessing": "training-only median imputation and standardization",
                "alpha": args.glm_alpha,
                "max_iter": 5_000,
                "tol": 1e-8,
            },
            "hist_gradient_boosting_internal_random_early_stopping": False,
            "algorithm_matched_incremental_tests": True,
        },
        "measurement_paths": {
            "peak": "primary median-overall-maximum MSDI",
            "breadth": "secondary equal-weight seven-dimension MSDI",
            "both": "union of peak and breadth features",
        },
        "best_rmse_model_by_horizon": best.to_dict(orient="records"),
        "outputs": {
            **{key: str(path) for key, path in output_paths.items()},
            "log": str(log_path),
            "manifest": str(manifest_path),
        },
        "output_sha256": output_sha256,
        "output_row_counts": output_row_counts,
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )
    logger.info("Forecast rows=%s fit events=%s", len(forecasts), len(fits))
    for _, row in best.iterrows():
        logger.info(
            "Best RMSE horizon=%sw model=%s rmse=%.6f OOS_R2_zero=%.6f",
            row["horizon_weeks"],
            row["model_id"],
            row["rmse_log_change"],
            row["oos_r2_vs_zero_change"],
        )
    logger.info("Metrics: %s", metrics_path)
    logger.info("Statistical tests: %s", tests_path)
    logger.info("Forecasting benchmarks completed successfully")


if __name__ == "__main__":
    main()
