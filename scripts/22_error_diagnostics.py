"""Run consolidated forecast, residual, event, and influence diagnostics.

This stage explains where the completed forecasts fail. It never refits a
forecast model, selects a preferred index, or upgrades an observational result
to a causal claim. Direct monthly common-cohort evidence is primary; weekly and
weekly-derived monthly constructions are retained as robustness evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy.stats import norm
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    breaks_cusumolsresid,
    het_arch,
)
from statsmodels.stats.stattools import jarque_bera


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/error_diagnostics.yaml"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "configs/model_registry.yaml"
DEFAULT_ROBUSTNESS_MANIFEST = (
    PROJECT_ROOT / "outputs/robustness/robustness_v001_manifest.json"
)
DEFAULT_PREDICTION_MANIFEST = (
    PROJECT_ROOT / "outputs/prediction_benchmarks/forecast_v001_manifest.json"
)
DEFAULT_NONLINEAR_MANIFEST = (
    PROJECT_ROOT / "outputs/nonlinear_forecasts/nonlinear_scfi_v001_manifest.json"
)
DEFAULT_CAUSALITY_MANIFEST = (
    PROJECT_ROOT / "outputs/time_series_causality/causality_v001_manifest.json"
)
DEFAULT_CAUSAL_ML_MANIFEST = (
    PROJECT_ROOT / "outputs/causal_ml/causal_ml_v001_manifest.json"
)
DEFAULT_ARTICLE_SCORES = (
    PROJECT_ROOT / "data/05_indices/article_level/article_consensus_scores.csv"
)
DEFAULT_WEEKLY_PANEL = (
    PROJECT_ROOT
    / "data/06_model_ready/prediction/msdi_scfi_weekly_panel.csv"
)
DEFAULT_MONTHLY_PANEL = (
    PROJECT_ROOT
    / "data/06_model_ready/prediction/llm_specific_scfi_monthly_panel.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/diagnostics"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs/logs"
CANONICAL_CONFIG_SHA256 = (
    "47da8fef774e794b7bca7633e6b1a0ff8b572af357627f5a6a10163ac58d3230"
)

FROZEN_MODEL_ALIASES = (
    "gemma_2_9b_it",
    "qwen_3_8b",
    "qwen_2_5_7b",
    "glm_4_9b_0414",
)
FROZEN_COMMON_COHORT_SIZE = 6480

PRIMARY_SCORE = "overall_disruption_max"
DIMENSION_COLUMNS = (
    "port_disruption_score",
    "global_container_movement_disruption_score",
    "freight_cost_disruption_score",
    "supply_disruption_score",
    "production_disruption_score",
    "covid19_related_disruption_score",
    "demand_disruption_score",
)
CORE_STAGE09_MODELS = (
    "no_msdi",
    "peak_only",
    "breadth_only",
    "peak_and_breadth",
)
MONTHLY_CONSTRUCTIONS = ("mean_safe_weekly", "adapted_pdf_sqrt_ema3")
OPERATIONAL_START = "2020-03-01"
OPERATIONAL_END = "2022-06-30"
CELL_KEYS = (
    "analysis_source",
    "frequency",
    "index_construction",
    "horizon",
    "model_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run consolidated SCFI forecast-error diagnostics."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    parser.add_argument(
        "--robustness-manifest", type=Path, default=DEFAULT_ROBUSTNESS_MANIFEST
    )
    parser.add_argument(
        "--prediction-manifest", type=Path, default=DEFAULT_PREDICTION_MANIFEST
    )
    parser.add_argument(
        "--nonlinear-manifest", type=Path, default=DEFAULT_NONLINEAR_MANIFEST
    )
    parser.add_argument(
        "--causality-manifest", type=Path, default=DEFAULT_CAUSALITY_MANIFEST
    )
    parser.add_argument(
        "--causal-ml-manifest", type=Path, default=DEFAULT_CAUSAL_ML_MANIFEST
    )
    parser.add_argument("--article-scores", type=Path, default=DEFAULT_ARTICLE_SCORES)
    parser.add_argument("--weekly-panel", type=Path, default=DEFAULT_WEEKLY_PANEL)
    parser.add_argument("--monthly-panel", type=Path, default=DEFAULT_MONTHLY_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--version", default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-error-cases", type=int, default=None)
    parser.add_argument("--top-articles-per-case", type=int, default=None)
    parser.add_argument("--maximum-rows", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--skip-article-trace",
        action="store_true",
        help="Debug-only; makes scope incomplete.",
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


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed: int, *parts: Any) -> int:
    payload = "|".join([str(seed), *[str(part) for part in parts]])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def configure_logging(log_dir: Path, version: str) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"09b_error_diagnostics_{version}.log"
    logger = logging.getLogger(f"error_diagnostics.{version}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger, path


def assert_project_venv() -> Path:
    executable = Path(sys.executable).resolve()
    expected = (PROJECT_ROOT / ".venv").resolve()
    if expected not in executable.parents:
        raise RuntimeError(
            "Run this stage with .venv\\Scripts\\python.exe "
            "scripts\\22_error_diagnostics.py"
        )
    return executable


def robust_zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    median = float(numeric.median())
    mad = float((numeric - median).abs().median())
    if np.isfinite(mad) and mad > 0:
        return (numeric - median) / (1.4826 * mad)
    return pd.Series(0.0, index=numeric.index).where(numeric.notna())


def assign_past_only_volatility_regime(
    values: pd.Series, window: int, minimum_periods: int
) -> pd.DataFrame:
    if window < 2 or minimum_periods < 2 or minimum_periods > window:
        raise ValueError("Invalid past-only volatility window contract")
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    volatility = (
        numeric.shift(1).rolling(window, min_periods=minimum_periods).std(ddof=1)
    )
    threshold = volatility.shift(1).expanding(
        min_periods=minimum_periods
    ).quantile(0.75)
    regime = pd.Series("insufficient_history", index=numeric.index, dtype="str")
    available = volatility.notna() & threshold.notna()
    regime.loc[available & volatility.le(threshold)] = "normal_volatility"
    regime.loc[available & volatility.gt(threshold)] = "high_volatility"
    return pd.DataFrame(
        {
            "volatility": volatility,
            "past_only_threshold": threshold,
            "volatility_regime": regime,
        },
        index=numeric.index,
    )


def normalized_loss(error: Iterable[float], scale: Any) -> Any:
    is_series = isinstance(error, pd.Series)
    values = np.asarray(error, dtype=float)
    scales = np.asarray(scale, dtype=float)
    invalid = np.isfinite(scales) & (scales <= 0)
    if invalid.any():
        raise ValueError("Finite loss-normalization scales must be positive")
    scales = np.where(np.isfinite(scales), scales, np.nan)
    result = (values / scales) ** 2
    return pd.Series(result, index=error.index) if is_series else result


def assert_unique_keys(frame: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    missing = sorted(set(keys).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks unique-key columns: {missing}")
    null_keys = frame[list(keys)].isna().any(axis=1)
    if null_keys.any():
        examples = frame.loc[null_keys, list(keys)].head(10).to_dict("records")
        raise ValueError(f"{label} has null unique-key values: {examples}")
    duplicates = frame.duplicated(list(keys), keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, list(keys)].head(10).to_dict("records")
        raise ValueError(f"{label} has duplicate keys: {examples}")


def validate_common_article_scope(
    article_audit: pd.DataFrame | Mapping[str, Any], expected_n: int = FROZEN_COMMON_COHORT_SIZE
) -> bool:
    if isinstance(article_audit, pd.DataFrame):
        if len(article_audit) != 1:
            raise ValueError("Article scope audit must have exactly one row")
        values: Mapping[str, Any] = article_audit.iloc[0].to_dict()
    else:
        values = article_audit
    observed = values.get(
        "common_complete_rows",
        values.get("common_article_count", values.get("rows")),
    )
    if int(observed or -1) != int(expected_n):
        raise ValueError(
            f"Frozen common-article scope mismatch: expected={expected_n} "
            f"observed={observed}"
        )
    expected = values.get("expected_common_complete_rows", expected_n)
    if int(expected or -1) != int(expected_n):
        raise ValueError(
            f"Frozen expected common-article scope mismatch: expected={expected_n} "
            f"upstream_expected={expected}"
        )
    if not bool(values.get("common_cohort_passed", True)):
        raise ValueError("Upstream common-article cohort contract did not pass")
    return True


def determine_run_status(
    smoke_test: bool,
    maximum_rows: int | None,
    all_contracts_passed: bool,
    full_scope: bool,
) -> str:
    if not all_contracts_passed:
        return "failed_source_contract_not_publication_eligible"
    if smoke_test or maximum_rows is not None:
        return "partial_debug_scope_not_publication_eligible"
    if not full_scope:
        return "complete_requested_subset_scope_not_publication_eligible"
    return "complete_computational_scope_pending_human_validation"


def operational_regime(dates: pd.Series) -> pd.Series:
    values = pd.to_datetime(dates)
    return pd.Series(
        np.select(
            [values.lt(OPERATIONAL_START), values.le(OPERATIONAL_END)],
            ["pre_disruption", "disruption_period"],
            default="post_disruption",
        ),
        index=dates.index,
    )


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "pandas", "scipy", "statsmodels", "PyYAML"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not_installed"
    return result


def read_json(path: Path) -> dict[str, Any]:
    resolved = require_file(path, "JSON source")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON source must be an object: {resolved}")
    return payload


def resolve_recorded_path(value: str | Path, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    project_relative = (PROJECT_ROOT / path).resolve()
    if project_relative.exists():
        return project_relative
    return (manifest_path.resolve().parent / path).resolve()


def audit_manifest_outputs(
    manifest_path: Path,
    stage: str,
    required_outputs: Sequence[str],
    require_output_contract: bool = False,
) -> tuple[dict[str, Any], dict[str, Path], list[dict[str, Any]]]:
    manifest_path = require_file(manifest_path, f"{stage} manifest")
    manifest = read_json(manifest_path)
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError(f"{stage} manifest lacks an outputs mapping")
    paths: dict[str, Path] = {}
    rows: list[dict[str, Any]] = [
        {
            "source_stage": stage,
            "artifact": "manifest",
            "path": str(manifest_path),
            "recorded_sha256": None,
            "actual_sha256": sha256_file(manifest_path),
            "hash_contract_available": False,
            "contract_passed": True,
            "row_count": 1,
            "details": f"version={manifest.get('version')};status={manifest.get('status')}",
        }
    ]
    recorded_hashes = manifest.get("output_sha256", {})
    recorded_counts = manifest.get("output_row_counts", {})
    if require_output_contract and (
        not isinstance(recorded_hashes, dict)
        or not isinstance(recorded_counts, dict)
    ):
        raise ValueError(
            f"{stage} manifest must provide output_sha256 and output_row_counts"
        )
    for key in required_outputs:
        if key not in outputs:
            raise ValueError(f"{stage} manifest lacks required output: {key}")
        path = require_file(
            resolve_recorded_path(outputs[key], manifest_path),
            f"{stage}:{key}",
        )
        paths[key] = path
        actual_hash = sha256_file(path)
        expected_hash = (
            recorded_hashes.get(key) if isinstance(recorded_hashes, dict) else None
        )
        hash_passed = expected_hash is None or str(expected_hash) == actual_hash
        if not hash_passed:
            raise ValueError(
                f"Hash mismatch for {stage}:{key}; expected={expected_hash} "
                f"actual={actual_hash}"
            )
        if path.suffix.lower() == ".csv":
            row_count = sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1
        else:
            row_count = 1
        expected_rows = (
            recorded_counts.get(key) if isinstance(recorded_counts, dict) else None
        )
        if require_output_contract and (
            expected_hash is None or expected_rows is None
        ):
            raise ValueError(
                f"{stage}:{key} lacks a manifest-declared hash or row count"
            )
        count_passed = expected_rows is None or int(expected_rows) == row_count
        if not count_passed:
            raise ValueError(
                f"Row-count mismatch for {stage}:{key}; expected={expected_rows} "
                f"actual={row_count}"
            )
        rows.append(
            {
                "source_stage": stage,
                "artifact": key,
                "path": str(path),
                "recorded_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "hash_contract_available": expected_hash is not None,
                "contract_passed": bool(hash_passed and count_passed),
                "row_count": row_count,
                "details": (
                    "manifest output hash verified"
                    if expected_hash is not None
                    else "upstream manifest has no output hash; hash frozen first in Stage 09b"
                ),
            }
        )
    return manifest, paths, rows


def audit_plain_file(
    path: Path,
    stage: str,
    artifact: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = require_file(path, f"{stage}:{artifact}")
    if path.suffix.lower() == ".csv":
        row_count = sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1
    else:
        row_count = 1
    actual_hash = sha256_file(path)
    passed = expected_sha256 is None or actual_hash == str(expected_sha256)
    if not passed:
        raise ValueError(
            f"Hash mismatch for {stage}:{artifact}; expected={expected_sha256} "
            f"actual={actual_hash}"
        )
    return {
        "source_stage": stage,
        "artifact": artifact,
        "path": str(path),
        "recorded_sha256": expected_sha256,
        "actual_sha256": actual_hash,
        "hash_contract_available": expected_sha256 is not None,
        "contract_passed": passed,
        "row_count": row_count,
        "details": "hash frozen by Stage 09b manifest",
    }


def validate_upstream_contracts(
    config: Mapping[str, Any],
    robustness_manifest: Mapping[str, Any],
    nonlinear_manifest: Mapping[str, Any],
    causal_ml_manifest: Mapping[str, Any],
) -> None:
    allowed = set(
        config["upstream_contracts"]["require_robustness_status"]
    )
    if robustness_manifest.get("status") not in allowed:
        raise ValueError("Stage 09 is not a frozen full-scope run")
    if robustness_manifest.get("causal_claim_allowed") is not False:
        raise ValueError("Stage 09 causal-claim guardrail is not false")
    if not bool(robustness_manifest.get("publication_scope_complete")):
        raise ValueError("Stage 09 publication computational scope is incomplete")
    required_audit = (
        "forecast_horizon_purge_all_passed",
        "reference_reconstruction_all_passed",
        "nonlinear_origin_sets_all_identical",
    )
    audit = robustness_manifest.get("audit", {})
    failed = [key for key in required_audit if not bool(audit.get(key))]
    if failed:
        raise ValueError(f"Stage 09 upstream audit gates failed: {failed}")
    nonlinear_design = nonlinear_manifest.get("design", {})
    expected_algorithms = set(config["analysis"]["nonlinear_algorithms"])
    requested_algorithms = set(nonlinear_design.get("requested_algorithms", []))
    executed_algorithms = set(nonlinear_design.get("executed_algorithms", []))
    if requested_algorithms != expected_algorithms:
        raise ValueError(
            "Nonlinear requested algorithm set changed: "
            f"expected={sorted(expected_algorithms)} "
            f"observed={sorted(requested_algorithms)}"
        )
    if executed_algorithms != expected_algorithms:
        raise ValueError(
            "Nonlinear executed algorithm set changed: "
            f"expected={sorted(expected_algorithms)} "
            f"observed={sorted(executed_algorithms)}"
        )
    if not bool(nonlinear_design.get("full_canonical_scope_complete")):
        raise ValueError("Nonlinear forecast stage is not full canonical scope")
    if not bool(nonlinear_design.get("common_origins_enforced")):
        raise ValueError("Nonlinear forecast common-origin contract failed")
    if causal_ml_manifest.get("causal_claim_allowed") is not False:
        raise ValueError("Causal-ML upstream causal-claim guardrail is not false")
    if not bool(causal_ml_manifest.get("publication_scope_complete")):
        raise ValueError("Causal-ML upstream computational scope is incomplete")
    causal_audit = causal_ml_manifest.get("audit", {})
    for key in (
        "all_horizon_purges_passed",
        "all_no_future_training_checks_passed",
        "all_treatment_horizon_invariance_checks_passed",
    ):
        if not bool(causal_audit.get(key)):
            raise ValueError(f"Causal-ML upstream contract failed: {key}")


def validate_prediction_identities(frame: pd.DataFrame, label: str) -> None:
    required = {
        "actual_log_change",
        "predicted_log_change",
        "forecast_error",
        "absolute_error",
        "squared_error",
        "level_error",
        "actual_scfi_level",
        "predicted_scfi_level",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} predictions lack columns: {missing}")
    checks = {
        "forecast_error": frame["actual_log_change"] - frame["predicted_log_change"],
        "absolute_error": frame["forecast_error"].abs(),
        "squared_error": frame["forecast_error"].pow(2),
        "level_error": frame["actual_scfi_level"] - frame["predicted_scfi_level"],
    }
    for column, rebuilt in checks.items():
        observed = pd.to_numeric(frame[column], errors="coerce")
        valid = observed.notna() & rebuilt.notna()
        if valid.any() and not np.allclose(
            observed.loc[valid], rebuilt.loc[valid], rtol=1.0e-8, atol=1.0e-10
        ):
            difference = float((observed.loc[valid] - rebuilt.loc[valid]).abs().max())
            raise ValueError(f"{label} {column} reconstruction failed: max={difference}")


def load_stage09_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    validate_prediction_identities(frame, "Stage 09")
    frame = frame.rename(columns={"specification_id": "model_id"})
    frame["analysis_source"] = "stage09_common_cohort_ridge"
    frame["model_family"] = "ridge"
    frame["model_tier"] = "primary_robustness_pipeline"
    frame["cohort_scope"] = "frozen_6480_common_articles"
    frame["frequency_role"] = np.where(
        frame["frequency"].eq("monthly"), "primary", "robustness"
    )
    frame["score_source"] = "stage09_specification"
    frame["signal_name"] = frame["signal_variants"].fillna("none")
    frame["is_core_diagnostic"] = frame["model_id"].isin(CORE_STAGE09_MODELS)
    frame["primary_family"] = (
        frame["frequency"].eq("monthly") & frame["is_core_diagnostic"]
    )
    return frame


def load_nonlinear_predictions(path: Path) -> pd.DataFrame:
    usecols = [
        "frequency",
        "algorithm",
        "model_id",
        "is_direct_model",
        "index_construction",
        "source_type",
        "score_source",
        "signal_name",
        "horizon_steps",
        "horizon_unit",
        "origin_date",
        "target_date",
        "actual_log_change",
        "predicted_log_change",
        "forecast_error",
        "absolute_error",
        "squared_error",
        "current_scfi",
        "actual_scfi_level",
        "predicted_scfi_level",
        "level_error",
    ]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    keep = frame["is_direct_model"].astype(bool) & (
        frame["source_type"].eq("baseline")
        | (
            frame["source_type"].eq("score_hybrid")
            & frame["score_source"].eq("score_median")
            & frame["signal_name"].isin(
                ["overall_disruption_max", "dimension_mean7"]
            )
        )
    )
    frame = frame.loc[keep].copy()
    frame = frame.rename(columns={"horizon_steps": "horizon"})
    validate_prediction_identities(frame, "Stage 07d nonlinear")
    frame["analysis_source"] = "stage07d_secondary_nonlinear"
    frame["family_id"] = frame["source_type"]
    frame["model_family"] = frame["algorithm"]
    frame["model_tier"] = "secondary_fixed_parameter_pipeline"
    frame["cohort_scope"] = "frozen_6480_common_articles"
    frame["frequency_role"] = np.where(
        frame["frequency"].eq("monthly"), "primary_frequency_secondary_model", "robustness"
    )
    frame["is_core_diagnostic"] = True
    frame["primary_family"] = False
    return frame


def apply_debug_limit(
    frame: pd.DataFrame, maximum_rows: int | None, smoke_test: bool
) -> pd.DataFrame:
    limit = maximum_rows
    if smoke_test and limit is None:
        limit = 35
    if limit is None:
        return frame
    if limit < 3:
        raise ValueError("--maximum-rows must be at least 3 per diagnostic cell")
    return (
        frame.sort_values(list(CELL_KEYS) + ["origin_date"])
        .groupby(list(CELL_KEYS), sort=False, group_keys=False)
        .head(limit)
        .reset_index(drop=True)
    )


def load_common_articles(path: Path, expected_n: int) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    aliases = FROZEN_MODEL_ALIASES
    score_names = (PRIMARY_SCORE, *DIMENSION_COLUMNS)
    score_columns = [
        f"{score}__{alias}" for alias in aliases for score in score_names
    ]
    missing_scores = sorted(set(score_columns).difference(frame.columns))
    if missing_scores:
        raise ValueError(f"Article file lacks common-cohort scores: {missing_scores}")
    frame["publication_date"] = pd.to_datetime(
        frame["publication_date"], format="mixed", errors="coerce"
    )
    for column in score_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    eligible = frame["publication_date"].notna() & frame[score_columns].notna().all(axis=1)
    frame = frame.loc[eligible].copy()
    if len(frame) != expected_n:
        raise ValueError(
            f"Article file common cohort mismatch: expected={expected_n}; observed={len(frame)}"
        )
    invalid = frame[score_columns].lt(0) | frame[score_columns].gt(4)
    if invalid.any().any():
        raise ValueError("Common-cohort article scores leave the frozen 0--4 scale")
    required = [
        "article_id",
        "publication_date",
        "headline",
        "source_domain",
        "duplicate_group_id",
        "duplicate_group_size",
        "peak_severity_score",
        "breadth_severity_score",
        "dimension_disagreement_sd_mean",
        "event_id_weekly",
        "event_id_monthly",
        "overall_disruption_max__ensemble_sd",
    ]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Common article table lacks fields: {missing}")
    frame["article_week_ending"] = (
        frame["publication_date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    )
    frame["article_week_origin"] = frame["article_week_ending"] + pd.Timedelta(days=3)
    frame["article_month_ending"] = (
        frame["publication_date"].dt.to_period("M").dt.end_time.dt.normalize()
    )
    frame["duplicate_group_size"] = (
        pd.to_numeric(frame["duplicate_group_size"], errors="coerce")
        .fillna(1.0)
        .clip(lower=1.0)
    )
    frame["source_domain"] = (
        frame["source_domain"].fillna("unknown_source").astype(str).str.strip()
    )
    frame["source_duplicate_weight"] = 1.0 / np.sqrt(frame["duplicate_group_size"])
    for profile, date_column in (
        ("weekly", "article_week_ending"),
        ("monthly", "article_month_ending"),
    ):
        event_column = f"event_id_{profile}"
        event = frame[event_column].astype("string").str.strip()
        event = event.mask(
            event.isna() | event.eq(""), "ARTICLE_" + frame["article_id"].astype(str)
        )
        denominator = frame.assign(_event=event).groupby(
            [date_column, "_event"], sort=False
        )["source_duplicate_weight"].transform("sum")
        frame[f"novelty_weight_{profile}"] = frame["source_duplicate_weight"].div(
            denominator
        )
    frame["novelty_weight"] = frame["novelty_weight_monthly"]
    frame["peak_severity_score"] = frame[
        [f"{PRIMARY_SCORE}__{alias}" for alias in aliases]
    ].median(axis=1, skipna=False)
    dimension_values = pd.DataFrame(
        {
            dimension: frame[[f"{dimension}__{alias}" for alias in aliases]].median(
                axis=1, skipna=False
            )
            for dimension in DIMENSION_COLUMNS
        },
        index=frame.index,
    )
    frame["breadth_severity_score"] = dimension_values.mean(axis=1, skipna=False)
    for dimension in DIMENSION_COLUMNS:
        frame[f"{dimension}__consensus_median"] = dimension_values[dimension]
    frame["absolute_novelty_weighted_score_contribution"] = (
        frame["novelty_weight"]
        * frame[["peak_severity_score", "breadth_severity_score"]].abs().max(axis=1)
    )
    return frame


def article_context(articles: pd.DataFrame, period: str) -> pd.DataFrame:
    date_column = "article_week_origin" if period == "weekly" else "article_month_ending"
    work = articles.copy()
    work["novelty_weight"] = work[f"novelty_weight_{period}"]
    work["peak_weighted_sum"] = work["peak_severity_score"] * work["novelty_weight"]
    work["breadth_weighted_sum"] = (
        work["breadth_severity_score"] * work["novelty_weight"]
    )
    dimension_weighted = []
    for dimension in DIMENSION_COLUMNS:
        target = f"{dimension}__weighted_sum"
        work[target] = work[f"{dimension}__consensus_median"] * work["novelty_weight"]
        dimension_weighted.append(target)
    grouped = work.groupby(date_column, sort=True, observed=True)
    aggregation: dict[str, tuple[str, str]] = {
        "article_context_count": ("article_id", "size"),
        "article_context_source_count": ("source_domain", "nunique"),
        "article_context_effective_count": ("novelty_weight", "sum"),
        "peak_model_disagreement": ("overall_disruption_max__ensemble_sd", "mean"),
        "dimension_model_disagreement": ("dimension_disagreement_sd_mean", "mean"),
        "peak_weighted_sum": ("peak_weighted_sum", "sum"),
        "breadth_weighted_sum": ("breadth_weighted_sum", "sum"),
    }
    for dimension, source in zip(DIMENSION_COLUMNS, dimension_weighted):
        aggregation[f"{dimension}__weighted_sum"] = (source, "sum")
    result = grouped.agg(**aggregation).reset_index()
    return result.rename(columns={date_column: "origin_date"})


def build_weekly_context(
    panel_path: Path,
    article_aggregate: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    usecols = [
        "observation_date",
        "scfi_log_change_1w",
        "brent_log_change_1w",
        "cpb_world_trade_volume_lag_1w",
        "unctad_china_core_lane_lsbci_lag_1w",
    ]
    panel = pd.read_csv(panel_path, usecols=usecols)
    panel["origin_date"] = pd.to_datetime(panel.pop("observation_date"))
    panel = panel.rename(
        columns={
            "scfi_log_change_1w": "origin_scfi_return",
            "brent_log_change_1w": "brent_log_change",
        }
    )
    regime_config = config["volatility_regime"]
    volatility = assign_past_only_volatility_regime(
        panel["origin_scfi_return"],
        int(regime_config["weekly_window"]),
        int(regime_config["weekly_minimum_periods"]),
    )
    panel = pd.concat([panel, volatility], axis=1)
    panel["past_only_target_scale"] = panel["volatility"]
    panel["control_missing_count"] = panel[["brent_log_change", "cpb_world_trade_volume_lag_1w", "unctad_china_core_lane_lsbci_lag_1w"]].isna().sum(axis=1)
    panel["frequency"] = "weekly"
    panel["index_construction"] = "weekly_log_volume"
    source_start = article_aggregate["origin_date"].min()
    source_end = article_aggregate["origin_date"].max()
    panel = panel.merge(article_aggregate, on="origin_date", how="left", validate="one_to_one")
    within_source_span = panel["origin_date"].between(source_start, source_end)
    zero_columns = [
        "article_context_count",
        "article_context_source_count",
        "article_context_effective_count",
        "peak_weighted_sum",
        "breadth_weighted_sum",
        *[f"{dimension}__weighted_sum" for dimension in DIMENSION_COLUMNS],
    ]
    panel.loc[within_source_span, zero_columns] = panel.loc[
        within_source_span, zero_columns
    ].fillna(0.0)
    panel["article_count"] = panel["article_context_count"]
    panel["source_count"] = panel["article_context_source_count"]
    panel["effective_article_count"] = panel["article_context_effective_count"]
    denominator = np.log1p(panel["article_count"])
    panel["peak_signal"] = panel["peak_weighted_sum"].div(
        denominator.where(denominator.gt(0))
    ).fillna(0.0)
    panel["breadth_signal"] = panel["breadth_weighted_sum"].div(
        denominator.where(denominator.gt(0))
    ).fillna(0.0)
    panel.loc[~within_source_span, ["peak_signal", "breadth_signal"]] = np.nan
    dimension_signal_columns = []
    for dimension in DIMENSION_COLUMNS:
        target = f"{dimension}__signal"
        panel[target] = panel[f"{dimension}__weighted_sum"].div(
            denominator.where(denominator.gt(0))
        ).fillna(0.0)
        panel.loc[~within_source_span, target] = np.nan
        dimension_signal_columns.append(target)
    panel["dominant_disruption_type"] = "insufficient_history"
    dimension_available = panel[dimension_signal_columns].notna().any(axis=1)
    panel.loc[dimension_available, "dominant_disruption_type"] = (
        panel.loc[dimension_available, dimension_signal_columns]
        .idxmax(axis=1)
        .str.removesuffix("__signal")
    )
    no_active = dimension_available & panel[dimension_signal_columns].fillna(0).eq(0).all(axis=1)
    panel.loc[no_active, "dominant_disruption_type"] = "no_active_disruption"
    panel.loc[~within_source_span, "dominant_disruption_type"] = "insufficient_history"
    panel.loc[within_source_span, "peak_model_disagreement"] = panel.loc[
        within_source_span, "peak_model_disagreement"
    ].fillna(0.0)
    panel["dimension_model_disagreement"] = panel[
        "dimension_model_disagreement"
    ]
    panel.loc[within_source_span, "dimension_model_disagreement"] = panel.loc[
        within_source_span, "dimension_model_disagreement"
    ].fillna(0.0)
    return panel


def build_monthly_context(
    panel_path: Path,
    weekly_context: pd.DataFrame,
    article_aggregate: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    base_columns = [
        "month_ending",
        "expected_mondays",
        "observed_mondays",
        "dates_complete",
        "scfi_monthly_mean_log_change_1m",
        "brent_monthly_mean_log_change_1m",
        "llm_common_article_count",
        "llm_common_effective_article_count",
    ]
    raw = pd.read_csv(panel_path, usecols=base_columns)
    raw["origin_date"] = pd.to_datetime(raw.pop("month_ending"))
    raw = raw.rename(
        columns={
            "scfi_monthly_mean_log_change_1m": "origin_scfi_return",
            "brent_monthly_mean_log_change_1m": "brent_log_change",
            "llm_common_article_count": "article_count",
            "llm_common_effective_article_count": "effective_article_count",
        }
    )
    regime_config = config["volatility_regime"]
    volatility = assign_past_only_volatility_regime(
        raw["origin_scfi_return"],
        int(regime_config["monthly_window"]),
        int(regime_config["monthly_minimum_periods"]),
    )
    raw = pd.concat([raw, volatility], axis=1)
    weekly = weekly_context.copy()
    weekly["month_ending"] = (
        weekly["origin_date"].dt.to_period("M").dt.end_time.dt.normalize()
    )
    monthly_weekly = weekly.groupby("month_ending", sort=True).agg(
        weekly_rows=("origin_date", "size"),
        signal_rows=("peak_signal", lambda values: int(values.notna().sum())),
        mean_safe_peak=("peak_signal", "mean"),
        mean_safe_breadth=("breadth_signal", "mean"),
        article_count_common=("article_count", "sum"),
        effective_article_count_common=("effective_article_count", "sum"),
        source_count_common=("source_count", "mean"),
        peak_weighted_sum=("peak_weighted_sum", "sum"),
        breadth_weighted_sum=("breadth_weighted_sum", "sum"),
    ).reset_index()
    for dimension in DIMENSION_COLUMNS:
        monthly_weekly[f"mean_safe__{dimension}"] = weekly.groupby(
            "month_ending", sort=True
        )[f"{dimension}__signal"].mean().to_numpy()
        monthly_weekly[f"{dimension}__weighted_sum"] = weekly.groupby(
            "month_ending", sort=True
        )[f"{dimension}__weighted_sum"].sum().to_numpy()
    effective = monthly_weekly["effective_article_count_common"]
    sqrt_denominator = np.sqrt(effective).where(effective.gt(0))
    monthly_weekly["adapted_peak_base"] = monthly_weekly["peak_weighted_sum"].div(
        sqrt_denominator
    )
    monthly_weekly["adapted_breadth_base"] = monthly_weekly[
        "breadth_weighted_sum"
    ].div(sqrt_denominator)
    for dimension in DIMENSION_COLUMNS:
        monthly_weekly[f"adapted_base__{dimension}"] = monthly_weekly[
            f"{dimension}__weighted_sum"
        ].div(sqrt_denominator)

    panel_complete = (
        raw["dates_complete"].astype(bool)
        & raw["observed_mondays"].eq(raw["expected_mondays"])
    )
    complete_by_month = pd.Series(
        panel_complete.to_numpy(), index=raw["origin_date"]
    )
    aggregation_complete = (
        monthly_weekly["month_ending"].map(complete_by_month).fillna(False).astype(bool)
        & monthly_weekly["weekly_rows"].eq(
            monthly_weekly["month_ending"].map(
                raw.set_index("origin_date")["expected_mondays"]
            )
        )
        & monthly_weekly["signal_rows"].eq(monthly_weekly["weekly_rows"])
    )
    pre_ema_columns = [
        "mean_safe_peak",
        "mean_safe_breadth",
        "adapted_peak_base",
        "adapted_breadth_base",
        *[f"mean_safe__{dimension}" for dimension in DIMENSION_COLUMNS],
        *[f"adapted_base__{dimension}" for dimension in DIMENSION_COLUMNS],
    ]
    monthly_weekly.loc[~aggregation_complete, pre_ema_columns] = np.nan

    def ema_after_gaps(values: pd.Series) -> pd.Series:
        result = pd.Series(np.nan, index=values.index, dtype=float)
        available = values.notna()
        blocks = available.ne(available.shift(fill_value=False)).cumsum()
        for _, indices in values[available].groupby(blocks[available]).groups.items():
            result.loc[indices] = values.loc[indices].ewm(
                span=3, adjust=False, min_periods=1
            ).mean()
        return result

    for target in [
        "adapted_peak_base",
        "adapted_breadth_base",
        *[f"adapted_base__{dimension}" for dimension in DIMENSION_COLUMNS],
    ]:
        monthly_weekly[target.replace("_base", "_ema3")] = ema_after_gaps(
            monthly_weekly[target]
        )
    raw = raw.merge(
        monthly_weekly, left_on="origin_date", right_on="month_ending", how="left", validate="one_to_one"
    ).drop(columns="month_ending")
    complete = (
        raw["dates_complete"].astype(bool)
        & raw["weekly_rows"].eq(raw["expected_mondays"])
        & raw["observed_mondays"].eq(raw["expected_mondays"])
    )
    monthly_signal_columns = [
        "mean_safe_peak",
        "mean_safe_breadth",
        "adapted_peak_ema3",
        "adapted_breadth_ema3",
        *[f"mean_safe__{dimension}" for dimension in DIMENSION_COLUMNS],
        *[f"adapted_ema3__{dimension}" for dimension in DIMENSION_COLUMNS],
    ]
    raw.loc[~complete, monthly_signal_columns] = np.nan
    direct = article_aggregate.rename(
        columns={
            "article_context_count": "direct_article_count",
            "article_context_source_count": "direct_source_count",
            "article_context_effective_count": "direct_effective_article_count",
            "peak_weighted_sum": "direct_peak_weighted_sum",
            "breadth_weighted_sum": "direct_breadth_weighted_sum",
            **{
                f"{dimension}__weighted_sum": f"direct__{dimension}__weighted_sum"
                for dimension in DIMENSION_COLUMNS
            },
        }
    )
    raw = raw.merge(direct, on="origin_date", how="left", validate="one_to_one")
    direct_denominator = np.sqrt(
        raw["direct_effective_article_count"].where(
            raw["direct_effective_article_count"].gt(0)
        )
    )
    raw["adapted_peak_base"] = raw["direct_peak_weighted_sum"].div(direct_denominator)
    raw["adapted_breadth_base"] = raw["direct_breadth_weighted_sum"].div(
        direct_denominator
    )
    raw["adapted_peak_ema3"] = ema_after_gaps(raw["adapted_peak_base"])
    raw["adapted_breadth_ema3"] = ema_after_gaps(raw["adapted_breadth_base"])
    for dimension in DIMENSION_COLUMNS:
        base = raw[f"direct__{dimension}__weighted_sum"].div(direct_denominator)
        raw[f"adapted_ema3__{dimension}"] = ema_after_gaps(base)
    pieces: list[pd.DataFrame] = []
    for construction in MONTHLY_CONSTRUCTIONS:
        columns = [
            "origin_date",
            "origin_scfi_return",
            "brent_log_change",
            "article_count",
            "effective_article_count",
            "volatility",
            "past_only_threshold",
            "volatility_regime",
        ]
        piece = raw[columns].copy()
        if construction == "mean_safe_weekly":
            piece["peak_signal"] = raw["mean_safe_peak"]
            piece["breadth_signal"] = raw["mean_safe_breadth"]
            dimension_sources = [f"mean_safe__{dimension}" for dimension in DIMENSION_COLUMNS]
        else:
            piece["peak_signal"] = raw["adapted_peak_ema3"]
            piece["breadth_signal"] = raw["adapted_breadth_ema3"]
            dimension_sources = [f"adapted_ema3__{dimension}" for dimension in DIMENSION_COLUMNS]
        dimension_frame = raw[dimension_sources].copy()
        dimension_frame.columns = [dimension.removesuffix("_score") for dimension in DIMENSION_COLUMNS]
        piece["dominant_disruption_type"] = "insufficient_history"
        available_dimensions = dimension_frame.notna().any(axis=1)
        piece.loc[available_dimensions, "dominant_disruption_type"] = (
            dimension_frame.loc[available_dimensions].idxmax(axis=1)
        )
        no_active = available_dimensions & dimension_frame.fillna(0.0).eq(0.0).all(axis=1)
        piece.loc[no_active, "dominant_disruption_type"] = "no_active_disruption"
        piece["frequency"] = "monthly"
        piece["index_construction"] = construction
        piece["past_only_target_scale"] = piece["volatility"]
        piece["control_missing_count"] = piece[["brent_log_change"]].isna().sum(axis=1)
        if construction == "adapted_pdf_sqrt_ema3":
            piece["article_count"] = raw["direct_article_count"]
            piece["effective_article_count"] = raw["direct_effective_article_count"]
            piece["source_count"] = raw["direct_source_count"]
        else:
            piece["article_count"] = raw["article_count_common"]
            piece["effective_article_count"] = raw["effective_article_count_common"]
            piece["source_count"] = raw["source_count_common"]
        piece["peak_model_disagreement"] = raw["peak_model_disagreement"].fillna(0.0)
        piece["dimension_model_disagreement"] = raw[
            "dimension_model_disagreement"
        ].fillna(0.0)
        pieces.append(piece)
    panel = pd.concat(pieces, ignore_index=True)
    return panel


def validate_context_reconstruction(
    weekly_context: pd.DataFrame,
    monthly_context: pd.DataFrame,
    weekly_variant_path: Path,
    monthly_variant_path: Path,
) -> pd.DataFrame:
    reference_ids = {
        "peak": "reference_peak_median",
        "breadth": "reference_breadth_median7",
    }
    rows: list[dict[str, Any]] = []
    weekly_source = pd.read_csv(
        weekly_variant_path,
        usecols=[
            "week_ending",
            "variant_id",
            "signal_value",
            "article_count",
            "source_count",
            "effective_article_count",
        ],
    )
    weekly_source = weekly_source.loc[
        weekly_source["variant_id"].isin(reference_ids.values())
    ].copy()
    weekly_source["origin_date"] = (
        pd.to_datetime(weekly_source["week_ending"]) + pd.Timedelta(days=3)
    )
    for path_name, variant_id in reference_ids.items():
        expected = weekly_source.loc[
            weekly_source["variant_id"].eq(variant_id),
            [
                "origin_date",
                "signal_value",
                "article_count",
                "source_count",
                "effective_article_count",
            ],
        ]
        observed_signal = f"{path_name}_signal"
        comparison = weekly_context.merge(
            expected,
            on="origin_date",
            how="inner",
            validate="one_to_one",
            suffixes=("_observed", "_expected"),
        )
        checks = {observed_signal: "signal_value"}
        if path_name == "peak":
            checks.update(
                {
                    "article_count_observed": "article_count_expected",
                    "source_count_observed": "source_count_expected",
                    "effective_article_count_observed": "effective_article_count_expected",
                }
            )
        for observed, expected_column in checks.items():
            expected_resolved = (
                expected_column
                if expected_column in comparison
                else f"{expected_column}_expected"
            )
            observed_context_column = observed.removesuffix("_observed")
            expected_source_column = expected_column.removesuffix("_expected")
            valid = comparison[observed].notna() & comparison[expected_resolved].notna()
            availability_mismatches = int(
                comparison[observed].notna().ne(comparison[expected_resolved].notna()).sum()
            )
            observed_available_dates = set(
                pd.to_datetime(
                    weekly_context.loc[
                        weekly_context[observed_context_column].notna(), "origin_date"
                    ]
                )
            )
            expected_available_dates_all = set(
                pd.to_datetime(
                    expected.loc[
                        expected[expected_source_column].notna(), "origin_date"
                    ]
                )
            )
            expected_date_domain = set(pd.to_datetime(expected["origin_date"]))
            context_date_domain = set(pd.to_datetime(weekly_context["origin_date"]))
            expected_available_dates = expected_available_dates_all.intersection(
                context_date_domain
            )
            date_set_mismatches = len(
                observed_available_dates.symmetric_difference(expected_available_dates)
            )
            observed_outside_domain = len(
                observed_available_dates.difference(expected_date_domain)
            )
            maximum = float(
                (comparison.loc[valid, observed] - comparison.loc[valid, expected_resolved])
                .abs()
                .max()
            ) if valid.any() else np.nan
            passed = bool(
                valid.any()
                and maximum <= 1.0e-10
                and availability_mismatches == 0
                and date_set_mismatches == 0
                and observed_outside_domain == 0
            )
            rows.append(
                {
                    "frequency": "weekly",
                    "index_construction": "weekly_log_volume",
                    "path": path_name,
                    "field": observed,
                    "matched_rows": int(valid.sum()),
                    "availability_mismatches": availability_mismatches,
                    "date_set_mismatches": date_set_mismatches,
                    "expected_available_outside_context_domain": len(
                        expected_available_dates_all.difference(context_date_domain)
                    ),
                    "observed_available_outside_expected_span": observed_outside_domain,
                    "maximum_absolute_difference": maximum,
                    "reconstruction_passed": passed,
                }
            )
    monthly_source = pd.read_csv(
        monthly_variant_path,
        usecols=[
            "month_ending",
            "variant_id",
            "signal_mean_safe_weekly",
            "signal_adapted_pdf_sqrt_ema3",
        ],
    )
    monthly_source["origin_date"] = pd.to_datetime(monthly_source["month_ending"])
    for construction in MONTHLY_CONSTRUCTIONS:
        expected_column = (
            "signal_mean_safe_weekly"
            if construction == "mean_safe_weekly"
            else "signal_adapted_pdf_sqrt_ema3"
        )
        observed_construction = monthly_context.loc[
            monthly_context["index_construction"].eq(construction)
        ]
        for path_name, variant_id in reference_ids.items():
            expected = monthly_source.loc[
                monthly_source["variant_id"].eq(variant_id),
                ["origin_date", expected_column],
            ]
            comparison = observed_construction.merge(
                expected, on="origin_date", how="inner", validate="one_to_one"
            )
            observed_column = f"{path_name}_signal"
            valid = comparison[observed_column].notna() & comparison[expected_column].notna()
            availability_mismatches = int(
                comparison[observed_column]
                .notna()
                .ne(comparison[expected_column].notna())
                .sum()
            )
            observed_available_dates = set(
                pd.to_datetime(
                    observed_construction.loc[
                        observed_construction[observed_column].notna(), "origin_date"
                    ]
                )
            )
            expected_available_dates_all = set(
                pd.to_datetime(
                    expected.loc[expected[expected_column].notna(), "origin_date"]
                )
            )
            expected_date_domain = set(pd.to_datetime(expected["origin_date"]))
            context_date_domain = set(
                pd.to_datetime(observed_construction["origin_date"])
            )
            expected_available_dates = expected_available_dates_all.intersection(
                context_date_domain
            )
            date_set_mismatches = len(
                observed_available_dates.symmetric_difference(expected_available_dates)
            )
            observed_outside_domain = len(
                observed_available_dates.difference(expected_date_domain)
            )
            maximum = float(
                (
                    comparison.loc[valid, observed_column]
                    - comparison.loc[valid, expected_column]
                ).abs().max()
            ) if valid.any() else np.nan
            passed = bool(
                valid.any()
                and maximum <= 1.0e-10
                and availability_mismatches == 0
                and date_set_mismatches == 0
                and observed_outside_domain == 0
            )
            rows.append(
                {
                    "frequency": "monthly",
                    "index_construction": construction,
                    "path": path_name,
                    "field": observed_column,
                    "matched_rows": int(valid.sum()),
                    "availability_mismatches": availability_mismatches,
                    "date_set_mismatches": date_set_mismatches,
                    "expected_available_outside_context_domain": len(
                        expected_available_dates_all.difference(context_date_domain)
                    ),
                    "observed_available_outside_expected_span": observed_outside_domain,
                    "maximum_absolute_difference": maximum,
                    "reconstruction_passed": passed,
                }
            )
    audit = pd.DataFrame(rows)
    if audit.empty or not audit["reconstruction_passed"].all():
        raise ValueError(
            "Stage 09 reference index reconstruction failed:\n"
            + audit.loc[~audit["reconstruction_passed"]].to_string(index=False)
        )
    return audit


def enrich_predictions(predictions: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    for column in ("origin_date", "target_date"):
        predictions[column] = pd.to_datetime(predictions[column])
    context_columns = [
        "frequency",
        "index_construction",
        "origin_date",
        "origin_scfi_return",
        "brent_log_change",
        "article_count",
        "source_count",
        "effective_article_count",
        "peak_model_disagreement",
        "dimension_model_disagreement",
        "peak_signal",
        "breadth_signal",
        "dominant_disruption_type",
        "volatility",
        "past_only_threshold",
        "volatility_regime",
        "past_only_target_scale",
        "control_missing_count",
    ]
    enriched = predictions.merge(
        context[context_columns],
        on=["frequency", "index_construction", "origin_date"],
        how="left",
        validate="many_to_one",
    )
    unmatched = enriched["past_only_target_scale"].isna()
    # Missing scale can be valid only during initial insufficient-history rows.
    missing_context = enriched.loc[unmatched, "volatility_regime"].isna()
    if missing_context.any():
        examples = enriched.loc[
            unmatched & enriched["volatility_regime"].isna(),
            ["frequency", "index_construction", "origin_date"],
        ].drop_duplicates().head(10)
        raise ValueError(f"Forecast origins lack panel context: {examples.to_dict('records')}")
    enriched["target_operational_regime"] = operational_regime(enriched["target_date"])
    enriched["origin_operational_regime"] = operational_regime(enriched["origin_date"])
    scale = enriched["past_only_target_scale"]
    enriched["normalized_absolute_error"] = enriched["absolute_error"] / scale
    enriched["normalized_squared_error"] = normalized_loss(
        enriched["forecast_error"], scale
    )
    enriched["direction_correct"] = np.sign(enriched["actual_log_change"]).eq(
        np.sign(enriched["predicted_log_change"])
    )
    enriched["underprediction"] = enriched["forecast_error"].gt(0)
    return enriched


def newey_west_lag(n_observations: int, horizon: int) -> int:
    sample_rule = int(math.floor(4 * (max(n_observations, 1) / 100) ** (2 / 9)))
    return max(0, min(max(int(horizon), sample_rule), max(n_observations - 2, 0)))


def hac_mean_test(values: pd.Series, horizon: int) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(clean) < 3:
        return {"estimate": np.nan, "standard_error": np.nan, "p_value": np.nan}
    design = np.ones((len(clean), 1), dtype=float)
    lag = newey_west_lag(len(clean), horizon)
    fitted = sm.OLS(clean.to_numpy(), design).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag}
    )
    return {
        "estimate": float(fitted.params[0]),
        "standard_error": float(fitted.bse[0]),
        "p_value": float(fitted.pvalues[0]),
    }


def moving_block_interval(
    values: np.ndarray,
    statistic: str,
    repetitions: int,
    block_length: int,
    seed: int,
) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or repetitions <= 0:
        return np.nan, np.nan, 0
    block_length = max(1, min(int(block_length), len(values)))
    starts = np.arange(0, len(values) - block_length + 1)
    blocks_needed = int(math.ceil(len(values) / block_length))
    generator = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        sampled_starts = generator.choice(starts, size=blocks_needed, replace=True)
        sampled = np.concatenate(
            [values[start : start + block_length] for start in sampled_starts]
        )[: len(values)]
        if statistic == "bias":
            estimates[repetition] = float(np.mean(sampled))
        elif statistic == "mae":
            estimates[repetition] = float(np.mean(np.abs(sampled)))
        elif statistic == "rmse":
            estimates[repetition] = float(np.sqrt(np.mean(sampled**2)))
        else:
            raise ValueError(f"Unknown bootstrap statistic: {statistic}")
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper), int(len(estimates))


def diagnostic_cells(frame: pd.DataFrame) -> Iterable[tuple[tuple[Any, ...], pd.DataFrame]]:
    return frame.groupby(list(CELL_KEYS), sort=True, dropna=False, observed=True)


def summarize_forecast_errors(
    frame: pd.DataFrame,
    bootstrap_repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in diagnostic_cells(frame):
        group = group.sort_values("origin_date")
        error = pd.to_numeric(group["forecast_error"], errors="coerce").dropna()
        level_error = pd.to_numeric(group["level_error"], errors="coerce").dropna()
        if error.empty:
            continue
        metadata = dict(zip(CELL_KEYS, key))
        horizon = int(metadata["horizon"])
        bias_test = hac_mean_test(error, horizon)
        primary = bool(group["primary_family"].all())
        core = bool(group["is_core_diagnostic"].all())
        bootstrap_this_cell = core and metadata["analysis_source"].startswith("stage09")
        intervals: dict[str, tuple[float, float, int]] = {}
        for statistic in ("bias", "mae", "rmse"):
            intervals[statistic] = (
                moving_block_interval(
                    error.to_numpy(),
                    statistic,
                    bootstrap_repetitions,
                    max(1, horizon),
                    stable_seed(seed, *key, statistic),
                )
                if bootstrap_this_cell
                else (np.nan, np.nan, 0)
            )
        normalized_absolute = pd.to_numeric(
            group["normalized_absolute_error"], errors="coerce"
        )
        normalized_squared = pd.to_numeric(
            group["normalized_squared_error"], errors="coerce"
        )
        rows.append(
            {
                **metadata,
                "model_family": group["model_family"].iloc[0],
                "family_id": group["family_id"].iloc[0],
                "model_tier": group["model_tier"].iloc[0],
                "horizon_unit": group["horizon_unit"].iloc[0],
                "primary_family": primary,
                "is_core_diagnostic": core,
                "n_observations": int(len(error)),
                "sample_start_date": group["origin_date"].min(),
                "sample_end_date": group["target_date"].max(),
                "bias": float(error.mean()),
                "median_error": float(error.median()),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "directional_accuracy": float(group["direction_correct"].mean()),
                "underprediction_share": float(group["underprediction"].mean()),
                "level_mae": float(level_error.abs().mean()),
                "level_rmse": float(np.sqrt(np.mean(level_error**2))),
                "normalized_absolute_error_mean": float(normalized_absolute.mean()),
                "normalized_squared_error_mean": float(normalized_squared.mean()),
                "normalized_loss_available_n": int(normalized_squared.notna().sum()),
                "bias_hac_standard_error": bias_test["standard_error"],
                "bias_raw_p_value": bias_test["p_value"],
                "hac_max_lags": newey_west_lag(len(error), horizon),
                "bias_bootstrap_ci_lower": intervals["bias"][0],
                "bias_bootstrap_ci_upper": intervals["bias"][1],
                "mae_bootstrap_ci_lower": intervals["mae"][0],
                "mae_bootstrap_ci_upper": intervals["mae"][1],
                "rmse_bootstrap_ci_lower": intervals["rmse"][0],
                "rmse_bootstrap_ci_upper": intervals["rmse"][1],
                "bootstrap_repetitions_successful": intervals["rmse"][2],
                "error_sign_convention": "actual_minus_predicted",
            }
        )
    return pd.DataFrame(rows)


def summarize_stratum(
    frame: pd.DataFrame, stratum_column: str, stratum_type: str
) -> pd.DataFrame:
    subset = frame.loc[frame["is_core_diagnostic"]].copy()
    keys = list(CELL_KEYS) + [stratum_column]
    rows: list[dict[str, Any]] = []
    for key, group in subset.groupby(keys, sort=True, dropna=False, observed=True):
        error = pd.to_numeric(group["forecast_error"], errors="coerce").dropna()
        if error.empty:
            continue
        metadata = dict(zip(keys, key))
        stratum = metadata.pop(stratum_column)
        rows.append(
            {
                **metadata,
                "stratum_type": stratum_type,
                "stratum": str(stratum),
                "n_observations": int(len(error)),
                "bias": float(error.mean()),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "directional_accuracy": float(group["direction_correct"].mean()),
                "underprediction_share": float(group["underprediction"].mean()),
                "normalized_absolute_error_mean": float(
                    group["normalized_absolute_error"].mean()
                ),
                "normalized_squared_error_mean": float(
                    group["normalized_squared_error"].mean()
                ),
                "primary_family": bool(group["primary_family"].all()),
            }
        )
    return pd.DataFrame(rows)


def residual_adequacy_tests(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    settings = config["residual_tests"]
    minimum = int(settings["minimum_observations"])
    rows: list[dict[str, Any]] = []
    subset = frame.loc[frame["is_core_diagnostic"]]
    for key, group in diagnostic_cells(subset):
        metadata = dict(zip(CELL_KEYS, key))
        errors = (
            pd.to_numeric(group.sort_values("origin_date")["forecast_error"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(errors) < minimum:
            continue
        frequency = str(metadata["frequency"])
        primary = bool(group["primary_family"].all())
        lag_label = "weekly_lags" if frequency == "weekly" else "monthly_lags"
        for test_name, config_name in (("ljung_box", "ljung_box"), ("arch_lm", "arch_lm")):
            for lag in settings[config_name][lag_label]:
                lag = int(lag)
                if lag >= len(errors):
                    continue
                if test_name == "ljung_box":
                    result = acorr_ljungbox(errors, lags=[lag], return_df=True).iloc[0]
                    statistic = float(result["lb_stat"])
                    p_value = float(result["lb_pvalue"])
                    null = "no_residual_autocorrelation_through_lag"
                else:
                    statistic, p_value, _, _ = het_arch(
                        errors - errors.mean(), nlags=lag
                    )
                    statistic, p_value = float(statistic), float(p_value)
                    null = "no_conditional_heteroskedasticity_through_lag"
                rows.append(
                    {
                        **metadata,
                        "diagnostic_id": "__".join(
                            [test_name, *map(str, key), f"lag{lag}"]
                        ),
                        "hypothesis_family_id": f"forecast_{test_name}__{frequency}",
                        "diagnostic": test_name,
                        "test_lag": lag,
                        "statistic": statistic,
                        "raw_p_value": p_value,
                        "effect_size": statistic,
                        "sample_size": len(errors),
                        "null_hypothesis": null,
                        "reject_raw_at_alpha": p_value < float(settings["alpha"]),
                        "primary_family": primary,
                        "interpretation": "model_adequacy_warning_not_economic_hypothesis",
                    }
                )
        jb_stat, jb_p, skewness, kurtosis = jarque_bera(errors)
        rows.append(
            {
                **metadata,
                "diagnostic_id": "__".join(["jarque_bera", *map(str, key)]),
                "hypothesis_family_id": f"forecast_jarque_bera__{frequency}",
                "diagnostic": "jarque_bera",
                "test_lag": 0,
                "statistic": float(jb_stat),
                "raw_p_value": float(jb_p),
                "effect_size": float(skewness),
                "residual_skewness": float(skewness),
                "residual_kurtosis": float(kurtosis),
                "sample_size": len(errors),
                "null_hypothesis": "residual_normality",
                "reject_raw_at_alpha": float(jb_p) < float(settings["alpha"]),
                "primary_family": primary,
                "interpretation": "model_adequacy_warning_not_economic_hypothesis",
            }
        )
    return pd.DataFrame(rows)


def mincer_zarnowitz_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset = frame.loc[frame["is_core_diagnostic"]]
    for key, group in diagnostic_cells(subset):
        metadata = dict(zip(CELL_KEYS, key))
        data = group[["actual_log_change", "predicted_log_change"]].dropna()
        n = len(data)
        if n < 20 or data["predicted_log_change"].nunique() < 2:
            continue
        x = sm.add_constant(data["predicted_log_change"].astype(float), has_constant="add")
        lag = newey_west_lag(n, int(metadata["horizon"]))
        fitted = sm.OLS(data["actual_log_change"].astype(float), x).fit(
            cov_type="HAC", cov_kwds={"maxlags": lag}
        )
        intercept, slope = map(float, fitted.params.iloc[:2])
        intercept_se, slope_se = map(float, fitted.bse.iloc[:2])
        intercept_p = float(fitted.pvalues.iloc[0])
        slope_z = (slope - 1.0) / slope_se if slope_se > 0 else np.nan
        slope_p = float(2 * norm.sf(abs(slope_z))) if np.isfinite(slope_z) else np.nan
        restriction = np.eye(2)
        joint = fitted.wald_test(
            (restriction, np.array([0.0, 1.0])), scalar=True
        )
        joint_p = float(np.asarray(joint.pvalue).squeeze())
        primary = bool(group["primary_family"].all())
        rows.append(
            {
                **metadata,
                "diagnostic_id": "__".join(["mincer_zarnowitz", *map(str, key)]),
                "hypothesis_family_id": f"forecast_calibration__{metadata['frequency']}",
                "sample_size": n,
                "hac_max_lags": lag,
                "intercept": intercept,
                "intercept_hac_standard_error": intercept_se,
                "intercept_ci_lower": intercept - 1.96 * intercept_se,
                "intercept_ci_upper": intercept + 1.96 * intercept_se,
                "raw_intercept_p_value": intercept_p,
                "slope": slope,
                "slope_hac_standard_error": slope_se,
                "slope_ci_lower": slope - 1.96 * slope_se,
                "slope_ci_upper": slope + 1.96 * slope_se,
                "raw_slope_equals_one_p_value": slope_p,
                "joint_wald_statistic": float(np.asarray(joint.statistic).squeeze()),
                "raw_joint_wald_p_value": joint_p,
                "raw_p_value": joint_p,
                "effect_size": slope - 1.0,
                "primary_family": primary,
                "interpretation": "forecast_calibration_not_structural_identification",
            }
        )
    return pd.DataFrame(rows)


def stability_diagnostics(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset = frame.loc[frame["is_core_diagnostic"]]
    minimum = int(config["stability"]["cusum"]["minimum_observations"])
    boundary = float(config["stability"]["cusum"]["two_sided_critical_value"])
    for key, group in diagnostic_cells(subset):
        metadata = dict(zip(CELL_KEYS, key))
        group = group.sort_values("origin_date")
        data = group[["forecast_error", "target_operational_regime"]].dropna()
        errors = data["forecast_error"].to_numpy(dtype=float)
        if len(errors) < minimum:
            continue
        centered = errors - errors.mean()
        scale = centered.std(ddof=1)
        cusum_path = (
            np.cumsum(centered) / (scale * np.sqrt(len(centered)))
            if scale > 0
            else np.zeros(len(centered))
        )
        max_cusum = float(np.max(np.abs(cusum_path)))
        try:
            cusum_stat, cusum_p, _ = breaks_cusumolsresid(centered, ddof=0)
            cusum_p = float(cusum_p)
        except (ValueError, np.linalg.LinAlgError):
            cusum_stat, cusum_p = max_cusum, np.nan
        frequency = str(metadata["frequency"])
        window = int(
            config["stability"]["rolling_error"][
                "weekly_window" if frequency == "weekly" else "monthly_window"
            ]
        )
        min_periods = int(
            config["stability"]["rolling_error"][
                "weekly_minimum_periods"
                if frequency == "weekly"
                else "monthly_minimum_periods"
            ]
        )
        error_series = pd.Series(errors)
        rolling_bias = error_series.rolling(window, min_periods=min_periods).mean()
        rolling_mae = error_series.abs().rolling(window, min_periods=min_periods).mean()
        rolling_rmse = np.sqrt(
            error_series.pow(2).rolling(window, min_periods=min_periods).mean()
        )
        time = np.linspace(0.0, 1.0, len(errors))
        trend_fit = sm.OLS(errors, sm.add_constant(time)).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": newey_west_lag(len(errors), int(metadata["horizon"]))},
        )
        primary = bool(group["primary_family"].all())
        common = {
            **metadata,
            "sample_size": len(errors),
            "primary_family": primary,
            "rolling_window": window,
            "maximum_absolute_rolling_bias": float(rolling_bias.abs().max()),
            "maximum_rolling_mae": float(rolling_mae.max()),
            "maximum_rolling_rmse": float(rolling_rmse.max()),
            "interpretation": "parameter_and_error_stability_diagnostic",
        }
        rows.append(
            {
                **common,
                "diagnostic_id": "__".join(["cusum", *map(str, key)]),
                "hypothesis_family_id": f"forecast_stability_cusum__{frequency}",
                "diagnostic": "cusum",
                "statistic": float(cusum_stat),
                "raw_p_value": cusum_p,
                "effect_size": max_cusum,
                "critical_boundary": boundary,
                "warning_flag": max_cusum > boundary,
            }
        )
        rows.append(
            {
                **common,
                "diagnostic_id": "__".join(["time_trend", *map(str, key)]),
                "hypothesis_family_id": f"forecast_stability_trend__{frequency}",
                "diagnostic": "hac_time_trend",
                "statistic": float(trend_fit.tvalues[1]),
                "raw_p_value": float(trend_fit.pvalues[1]),
                "effect_size": float(trend_fit.params[1]),
                "critical_boundary": np.nan,
                "warning_flag": float(trend_fit.pvalues[1]) < 0.05,
            }
        )
        regimes = pd.get_dummies(
            data["target_operational_regime"], prefix="regime", dtype=float
        )
        reference = "regime_pre_disruption"
        if regimes.shape[1]:
            drop_column = reference if reference in regimes else sorted(regimes.columns)[0]
            regimes = regimes.drop(columns=drop_column)
        if regimes.shape[1]:
            x = sm.add_constant(regimes, has_constant="add")
            if np.linalg.matrix_rank(x.to_numpy()) < x.shape[1]:
                continue
            fit = sm.OLS(errors, x).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": newey_west_lag(len(errors), int(metadata["horizon"]))},
            )
            restriction = np.zeros((regimes.shape[1], x.shape[1]))
            restriction[:, 1:] = np.eye(regimes.shape[1])
            joint = fit.wald_test(restriction, scalar=True)
            p_value = float(np.asarray(joint.pvalue).squeeze())
            rows.append(
                {
                    **common,
                    "diagnostic_id": "__".join(["operational_regime", *map(str, key)]),
                    "hypothesis_family_id": f"forecast_stability_regime__{frequency}",
                    "diagnostic": "operational_regime_joint_hac",
                    "statistic": float(np.asarray(joint.statistic).squeeze()),
                    "raw_p_value": p_value,
                    "effect_size": float(
                        data.groupby("target_operational_regime")["forecast_error"].mean().max()
                        - data.groupby("target_operational_regime")["forecast_error"].mean().min()
                    ),
                    "critical_boundary": np.nan,
                    "warning_flag": p_value < 0.05,
                }
            )
    return pd.DataFrame(rows)


def influence_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in diagnostic_cells(frame.loc[frame["is_core_diagnostic"]]):
        errors = pd.to_numeric(group["forecast_error"], errors="coerce")
        squared = errors.pow(2).dropna()
        total = float(squared.sum())
        shares = squared / total if total > 0 else pd.Series(0.0, index=squared.index)
        robust = robust_zscore(errors)
        rows.append(
            {
                **dict(zip(CELL_KEYS, key)),
                "n_observations": int(errors.notna().sum()),
                "robust_absolute_z_ge_3_count": int(robust.abs().ge(3).sum()),
                "robust_absolute_z_ge_3_share": float(robust.abs().ge(3).mean()),
                "maximum_single_observation_squared_loss_share": float(shares.max()),
                "top_five_squared_loss_share": float(shares.nlargest(5).sum()),
                "squared_loss_effective_n": float(1 / np.sum(shares**2))
                if np.sum(shares**2) > 0
                else np.nan,
                "influence_concentration_warning_flag": bool(shares.max() > 0.10),
                "primary_family": bool(group["primary_family"].all()),
            }
        )
    return pd.DataFrame(rows)


def build_driver_data(group: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = pd.DataFrame(index=group.index)
    data["past_only_market_volatility"] = group["volatility"]
    data["peak_model_disagreement"] = group["peak_model_disagreement"]
    data["dimension_model_disagreement"] = group["dimension_model_disagreement"]
    data["log1p_article_count"] = np.log1p(group["article_count"])
    data["source_count"] = group["source_count"]
    data["effective_article_count"] = group["effective_article_count"]
    data["peak_signal"] = group["peak_signal"]
    data["breadth_signal"] = group["breadth_signal"]
    data["brent_log_change"] = group["brent_log_change"]
    data["control_missing_count"] = group["control_missing_count"]
    regimes = pd.get_dummies(
        group["origin_operational_regime"], prefix="operational", dtype=float
    )
    for name in ("operational_disruption_period", "operational_post_disruption"):
        data[name] = regimes[name] if name in regimes else 0.0
    requested = list(data.columns)
    data["driver_outcome"] = np.log1p(group["normalized_absolute_error"])
    data["forecast_error_for_fold_label"] = group["forecast_error"]
    data["extreme_absolute_error_flag"] = robust_zscore(
        group["forecast_error"]
    ).abs().ge(3).astype(int)
    data["origin_date"] = group["origin_date"]
    return data, requested


def select_driver_features(
    data: pd.DataFrame,
    requested: Sequence[str],
    minimum_observations_per_parameter: int,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    complete = data.dropna(subset=["driver_outcome"]).copy()
    statuses: dict[str, str] = {}
    for name in requested:
        numeric = pd.to_numeric(complete[name], errors="coerce")
        if numeric.notna().any():
            complete[name] = numeric.fillna(float(numeric.median()))
        else:
            statuses[name] = "unavailable_all_missing"
    if len(complete) < max(20, minimum_observations_per_parameter * 2):
        return complete, [], {name: "insufficient_complete_rows" for name in requested}
    maximum_features = max(
        1, int(len(complete) // minimum_observations_per_parameter) - 1
    )
    selected: list[str] = []
    current = np.ones((len(complete), 1), dtype=float)
    current_rank = 1
    for name in requested:
        if name in statuses:
            continue
        values = pd.to_numeric(complete[name], errors="coerce")
        if values.nunique(dropna=True) < 2:
            statuses[name] = "unavailable_zero_variance"
            continue
        if len(selected) >= maximum_features:
            statuses[name] = "dropped_sample_size_budget"
            continue
        transformed = values if name.startswith("operational_") else robust_zscore(values)
        candidate = np.column_stack([current, transformed.to_numpy(dtype=float)])
        rank = int(np.linalg.matrix_rank(candidate))
        if rank <= current_rank:
            statuses[name] = "dropped_collinear"
            continue
        selected.append(name)
        statuses[name] = "used"
        complete[f"driver_z__{name}"] = transformed
        current = candidate
        current_rank = rank
    return complete, selected, statuses


def logistic_time_ordered_performance(
    driver_data: pd.DataFrame,
    requested: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler

    ordered = driver_data.dropna(
        subset=["driver_outcome", "forecast_error_for_fold_label"]
    ).sort_values("origin_date")
    if len(ordered) < 30:
        return {
            "status": "skipped_insufficient_rows",
            "evaluation_rows": 0,
            "roc_auc": np.nan,
            "brier_score": np.nan,
            "accuracy": np.nan,
        }
    x = ordered[list(requested)].apply(pd.to_numeric, errors="coerce")
    forecast_error = ordered["forecast_error_for_fold_label"].to_numpy(dtype=float)
    boundaries = np.linspace(len(ordered) // 2, len(ordered), 4, dtype=int)
    probabilities: list[float] = []
    observed: list[int] = []
    fold_feature_counts: list[int] = []
    for fold in range(3):
        train_end = int(boundaries[fold])
        test_end = int(boundaries[fold + 1])
        training_error = forecast_error[:train_end]
        center = float(np.median(training_error))
        mad = float(np.median(np.abs(training_error - center)))
        scale = 1.4826 * mad
        if test_end <= train_end or not np.isfinite(scale) or scale <= 0:
            continue
        y_train = (np.abs(training_error - center) / scale >= 3.0).astype(int)
        y_test = (
            np.abs(forecast_error[train_end:test_end] - center) / scale >= 3.0
        ).astype(int)
        if len(np.unique(y_train)) < 2 or int(y_train.sum()) < 2:
            continue
        # Availability and variance screening are learned from the training
        # prefix only. The prespecified predictor list itself is fixed ex ante.
        training_features = [
            name
            for name in requested
            if x.iloc[:train_end][name].notna().any()
            and x.iloc[:train_end][name].nunique(dropna=True) >= 2
        ]
        if not training_features:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RobustScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                random_state=seed,
                max_iter=2000,
            ),
        )
        model.fit(x.iloc[:train_end][training_features], y_train)
        probabilities.extend(
            model.predict_proba(x.iloc[train_end:test_end][training_features])[:, 1]
        )
        observed.extend(y_test)
        fold_feature_counts.append(len(training_features))
    if not probabilities or len(set(observed)) < 2:
        return {
            "status": "skipped_evaluation_has_one_class",
            "evaluation_rows": len(observed),
            "feature_count": max(fold_feature_counts, default=0),
            "roc_auc": np.nan,
            "brier_score": np.nan,
            "accuracy": np.nan,
        }
    probability = np.asarray(probabilities)
    truth = np.asarray(observed)
    return {
        "status": "estimated_time_ordered_expanding_evaluation",
        "evaluation_rows": len(truth),
        "feature_count": max(fold_feature_counts, default=0),
        "roc_auc": float(roc_auc_score(truth, probability)),
        "brier_score": float(brier_score_loss(truth, probability)),
        "accuracy": float(accuracy_score(truth, probability >= 0.5)),
    }


def error_driver_analysis(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    minimum_per_parameter = int(
        config["error_drivers"]["minimum_observations_per_parameter"]
    )
    subset = frame.loc[frame["is_core_diagnostic"]]
    for key, group in diagnostic_cells(subset):
        metadata = dict(zip(CELL_KEYS, key))
        driver_data, requested = build_driver_data(group)
        complete, selected, statuses = select_driver_features(
            driver_data, requested, minimum_per_parameter
        )
        primary = bool(group["primary_family"].all())
        fitted = None
        if selected:
            x_columns = [f"driver_z__{name}" for name in selected]
            x = sm.add_constant(complete[x_columns], has_constant="add")
            fitted = sm.OLS(complete["driver_outcome"], x).fit(
                cov_type="HAC",
                cov_kwds={
                    "maxlags": newey_west_lag(
                        len(complete), int(metadata["horizon"])
                    )
                },
            )
        for name in requested:
            parameter = f"driver_z__{name}"
            used = fitted is not None and parameter in fitted.params.index
            coefficient_rows.append(
                {
                    **metadata,
                    "diagnostic_id": "__".join(
                        ["error_driver_hac", *map(str, key), name]
                    ),
                    "hypothesis_family_id": (
                        f"forecast_error_drivers__{metadata['frequency']}"
                    ),
                    "feature": name,
                    "feature_status": statuses.get(name, "not_evaluated"),
                    "coefficient": float(fitted.params[parameter]) if used else np.nan,
                    "hac_standard_error": float(fitted.bse[parameter]) if used else np.nan,
                    "raw_p_value": float(fitted.pvalues[parameter]) if used else np.nan,
                    "effect_size": float(fitted.params[parameter]) if used else np.nan,
                    "sample_size": int(len(complete)),
                    "primary_family": primary,
                    "interpretation": "descriptive_predictive_association_not_causality",
                }
            )
        if fitted is not None:
            performance_rows.append(
                {
                    **metadata,
                    "model_type": "continuous_ols_hac",
                    "status": "estimated",
                    "sample_size": len(complete),
                    "feature_count": len(selected),
                    "r_squared": float(fitted.rsquared),
                    "adjusted_r_squared": float(fitted.rsquared_adj),
                    "evaluation_rows": len(complete),
                    "roc_auc": np.nan,
                    "brier_score": np.nan,
                    "accuracy": np.nan,
                    "evaluation_contract": "retrospective_hac_association",
                }
            )
            if metadata["analysis_source"] == "stage09_common_cohort_ridge":
                logistic = logistic_time_ordered_performance(
                    driver_data,
                    requested,
                    stable_seed(seed, *key, "logistic"),
                )
            else:
                logistic = {
                    "status": "skipped_secondary_nonlinear_scope",
                    "evaluation_rows": 0,
                    "roc_auc": np.nan,
                    "brier_score": np.nan,
                    "accuracy": np.nan,
                }
            performance_rows.append(
                {
                    **metadata,
                    "model_type": "binary_logistic_l2",
                    "status": logistic["status"],
                    "sample_size": len(complete),
                    "feature_count": logistic.get("feature_count", 0),
                    "r_squared": np.nan,
                    "adjusted_r_squared": np.nan,
                    "evaluation_rows": logistic["evaluation_rows"],
                    "roc_auc": logistic["roc_auc"],
                    "brier_score": logistic["brier_score"],
                    "accuracy": logistic["accuracy"],
                    "evaluation_contract": (
                        "time_ordered_expanding_with_fold_specific_error_thresholds_"
                        "and_training_only_preprocessing"
                    ),
                }
            )
        else:
            performance_rows.append(
                {
                    **metadata,
                    "model_type": "continuous_ols_hac",
                    "status": "skipped_insufficient_identified_features",
                    "sample_size": len(complete),
                    "feature_count": 0,
                    "r_squared": np.nan,
                    "adjusted_r_squared": np.nan,
                    "evaluation_rows": 0,
                    "roc_auc": np.nan,
                    "brier_score": np.nan,
                    "accuracy": np.nan,
                    "evaluation_contract": "retrospective_hac_association",
                }
            )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(performance_rows)


def select_with_ties(
    frame: pd.DataFrame, column: str, count: int, largest: bool
) -> pd.Index:
    candidates = frame.loc[frame[column].notna()].sort_values(
        column, ascending=not largest
    )
    if candidates.empty:
        return candidates.index
    position = min(count, len(candidates)) - 1
    threshold = candidates[column].iloc[position]
    selected = (
        candidates[column].ge(threshold)
        if largest
        else candidates[column].le(threshold)
    )
    return candidates.loc[selected].index


def extreme_forecast_cases(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    subset = frame.loc[
        frame["analysis_source"].eq("stage09_common_cohort_ridge")
        & frame["model_id"].isin(CORE_STAGE09_MODELS)
    ].copy()
    rows: list[pd.DataFrame] = []
    for key, group in diagnostic_cells(subset):
        group = group.copy()
        group["robust_error_z"] = robust_zscore(group["forecast_error"])
        roles: dict[Any, set[str]] = {}
        selections = {
            "largest_underprediction": select_with_ties(
                group.loc[group["forecast_error"].gt(0)],
                "forecast_error",
                count,
                True,
            ),
            "largest_overprediction": select_with_ties(
                group.loc[group["forecast_error"].lt(0)],
                "forecast_error",
                count,
                False,
            ),
            "largest_absolute_error": select_with_ties(
                group.assign(selection_absolute_error=group["forecast_error"].abs()),
                "selection_absolute_error",
                count,
                True,
            ),
        }
        for role, indices in selections.items():
            for index in indices:
                roles.setdefault(index, set()).add(role)
        if not roles:
            continue
        selected = group.loc[list(roles)].copy()
        selected["selection_roles"] = [
            "|".join(sorted(roles[index])) for index in selected.index
        ]
        selected["extreme_robust_z_flag"] = selected["robust_error_z"].abs().ge(3)
        selected["case_id"] = [
            "case_"
            + hashlib.sha256(
                "|".join(
                    [
                        *map(str, key),
                        str(origin),
                        str(target),
                    ]
                ).encode("utf-8")
            ).hexdigest()[:16]
            for origin, target in zip(selected["origin_date"], selected["target_date"])
        ]
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    keep = [
        "case_id",
        *CELL_KEYS,
        "origin_date",
        "target_date",
        "actual_log_change",
        "predicted_log_change",
        "forecast_error",
        "absolute_error",
        "actual_scfi_level",
        "predicted_scfi_level",
        "level_error",
        "normalized_absolute_error",
        "normalized_squared_error",
        "robust_error_z",
        "extreme_robust_z_flag",
        "selection_roles",
        "target_operational_regime",
        "volatility_regime",
        "dominant_disruption_type",
        "article_count",
        "peak_model_disagreement",
        "dimension_model_disagreement",
        "peak_signal",
        "breadth_signal",
    ]
    return result[keep].sort_values(list(CELL_KEYS) + ["origin_date"]).reset_index(drop=True)


def trace_extreme_case_articles(
    cases: pd.DataFrame,
    articles: pd.DataFrame,
    top_articles: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in cases.to_dict("records"):
        origin = pd.Timestamp(case["origin_date"])
        target = pd.Timestamp(case["target_date"])
        if case["frequency"] == "weekly":
            origin_mask = articles["article_week_origin"].eq(origin)
            weight_column = "novelty_weight_weekly"
        else:
            origin_mask = articles["article_month_ending"].eq(origin)
            weight_column = "novelty_weight_monthly"
        outcome_mask = articles["publication_date"].gt(origin) & articles[
            "publication_date"
        ].le(target)
        for role, mask in (
            ("origin_information", origin_mask),
            ("outcome_context", outcome_mask),
        ):
            candidates = articles.loc[mask].copy()
            candidates["_case_contribution"] = candidates[weight_column] * candidates[
                ["peak_severity_score", "breadth_severity_score"]
            ].abs().max(axis=1)
            candidates = candidates.sort_values(
                ["_case_contribution", "article_id"],
                ascending=[False, True],
            )
            if candidates.empty:
                continue
            selected = candidates.head(top_articles)
            for rank, article in enumerate(selected.to_dict("records"), start=1):
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "trace_role": role,
                        "article_rank": rank,
                        "article_id": article["article_id"],
                        "publication_date": article["publication_date"],
                        "headline": article["headline"],
                        "source_domain": article["source_domain"],
                        "duplicate_group_id": article["duplicate_group_id"],
                        "duplicate_group_size": article["duplicate_group_size"],
                        "peak_severity_score": article["peak_severity_score"],
                        "breadth_severity_score": article["breadth_severity_score"],
                        "peak_model_disagreement_sd": article[
                            "overall_disruption_max__ensemble_sd"
                        ],
                        "dimension_disagreement_sd_mean": article[
                            "dimension_disagreement_sd_mean"
                        ],
                        "novelty_weight": article[weight_column],
                        "absolute_novelty_weighted_score_contribution": article[
                            "_case_contribution"
                        ],
                        "known_at_forecast_origin": role == "origin_information",
                        "event_claim_allowed": False,
                    }
                )
    return pd.DataFrame(rows)


def tag_consolidation_table(
    frame: pd.DataFrame,
    diagnostic_source: str,
    source_path: Path,
    default_scope: str,
) -> pd.DataFrame:
    tagged = frame.copy()
    if "diagnostic_source" in tagged:
        tagged = tagged.rename(
            columns={"diagnostic_source": "upstream_diagnostic_source"}
        )
    tagged.insert(0, "diagnostic_source", diagnostic_source)
    tagged.insert(1, "consolidation_source_path", str(source_path))
    tagged.insert(2, "consolidation_source_sha256", sha256_file(source_path))
    if "cohort_scope" not in tagged:
        tagged["cohort_scope"] = default_scope
    tagged["causal_claim_allowed"] = False
    tagged["consolidation_id"] = [
        f"{diagnostic_source}__{index:06d}" for index in range(len(tagged))
    ]
    return tagged


def consolidate_dynamic_diagnostics(
    paths: Mapping[str, Path]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    source_names = (
        "varx_diagnostics_sensitivity",
        "varx_lag_selection_sensitivity",
        "selected_dynamic_lags",
        "conditional_granger_sensitivity",
        "conventional_granger_sensitivity",
        "pcmci_link_stability",
        "pcmci_sensitivity_links",
        "local_projection_sensitivity",
    )
    raw_tables: dict[str, pd.DataFrame] = {}
    tagged: list[pd.DataFrame] = []
    for name in source_names:
        frame = pd.read_csv(paths[name], low_memory=False)
        raw_tables[name] = frame
        tagged.append(
            tag_consolidation_table(
                frame,
                name,
                paths[name],
                "frozen_6480_common_cohort_unless_row_label_overrides",
            )
        )
    result = pd.concat(tagged, ignore_index=True, sort=False)
    return result, raw_tables


def nuisance_oof_summary(oof: pd.DataFrame) -> pd.DataFrame:
    keys = ["frequency", "frequency_role", "msdi_path", "horizon", "learner"]
    rows: list[dict[str, Any]] = []
    for key, group in oof.groupby(keys, sort=True, observed=True):
        outcome = pd.to_numeric(group["outcome"], errors="coerce")
        outcome_prediction = pd.to_numeric(
            group["outcome_prediction"], errors="coerce"
        )
        treatment = pd.to_numeric(group["treatment_raw_msdi_change"], errors="coerce")
        treatment_prediction = pd.to_numeric(
            group["treatment_prediction"], errors="coerce"
        )

        def metrics(actual: pd.Series, predicted: pd.Series) -> tuple[int, float, float]:
            valid = actual.notna() & predicted.notna()
            actual_valid = actual.loc[valid].astype(float)
            predicted_valid = predicted.loc[valid].astype(float)
            if actual_valid.empty:
                return 0, np.nan, np.nan
            residual_sum = float(np.sum((actual_valid - predicted_valid) ** 2))
            total_sum = float(np.sum((actual_valid - actual_valid.mean()) ** 2))
            rmse = float(np.sqrt(residual_sum / len(actual_valid)))
            r_squared = 1 - residual_sum / total_sum if total_sum > 0 else np.nan
            return int(valid.sum()), rmse, float(r_squared)

        def calibration_metrics(
            actual: pd.Series, predicted: pd.Series, horizon: int
        ) -> dict[str, Any]:
            valid = actual.notna() & predicted.notna()
            actual_valid = actual.loc[valid].astype(float)
            predicted_valid = predicted.loc[valid].astype(float)
            if actual_valid.empty:
                return {
                    "residual_bias": np.nan,
                    "calibration_intercept": np.nan,
                    "calibration_slope": np.nan,
                    "calibration_joint_raw_p_value": np.nan,
                    "calibration_status": "skipped_no_joint_actual_prediction_rows",
                }
            bias = float((actual_valid - predicted_valid).mean())
            if len(actual_valid) < 20 or predicted_valid.nunique() < 2:
                return {
                    "residual_bias": bias,
                    "calibration_intercept": np.nan,
                    "calibration_slope": np.nan,
                    "calibration_joint_raw_p_value": np.nan,
                    "calibration_status": "skipped_insufficient_rank_or_rows",
                }
            x = sm.add_constant(predicted_valid, has_constant="add")
            fitted = sm.OLS(actual_valid, x).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": newey_west_lag(len(actual_valid), horizon)},
            )
            joint = fitted.wald_test(
                (np.eye(2), np.array([0.0, 1.0])), scalar=True
            )
            return {
                "residual_bias": bias,
                "calibration_intercept": float(fitted.params.iloc[0]),
                "calibration_slope": float(fitted.params.iloc[1]),
                "calibration_joint_raw_p_value": float(
                    np.asarray(joint.pvalue).squeeze()
                ),
                "calibration_status": "estimated_hac",
            }

        n_outcome, outcome_rmse, outcome_r2 = metrics(outcome, outcome_prediction)
        n_treatment, treatment_rmse, treatment_r2 = metrics(
            treatment, treatment_prediction
        )
        outcome_calibration = calibration_metrics(
            outcome, outcome_prediction, int(dict(zip(keys, key))["horizon"])
        )
        treatment_calibration = calibration_metrics(
            treatment,
            treatment_prediction,
            int(dict(zip(keys, key))["horizon"]),
        )
        joint_valid = (
            outcome.notna()
            & outcome_prediction.notna()
            & treatment.notna()
            & treatment_prediction.notna()
        )
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_observations": int(joint_valid.sum()),
                "n_outcome_observations": n_outcome,
                "n_treatment_observations": n_treatment,
                "nuisance_outcome_oof_rmse_recalculated": outcome_rmse,
                "nuisance_outcome_oof_r2_recalculated": outcome_r2,
                "nuisance_treatment_oof_rmse_recalculated": treatment_rmse,
                "nuisance_treatment_oof_r2_recalculated": treatment_r2,
                **{f"outcome_{name}": value for name, value in outcome_calibration.items()},
                **{
                    f"treatment_{name}": value
                    for name, value in treatment_calibration.items()
                },
                "negative_outcome_nuisance_r2_warning": outcome_r2 < 0,
                "negative_treatment_nuisance_r2_warning": treatment_r2 < 0,
                "cohort_scope": "upstream_stage08b_broader_panel_not_frozen_common_6480",
                "interpretation": "nuisance_predictive_diagnostic_not_identification",
            }
        )
    return pd.DataFrame(rows)


def consolidate_dml_diagnostics(
    causal_paths: Mapping[str, Path],
    robustness_paths: Mapping[str, Path],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    raw_tables: dict[str, pd.DataFrame] = {}
    oof = pd.read_csv(causal_paths["oof_predictions"], low_memory=False)
    nuisance = nuisance_oof_summary(oof)
    raw_tables["nuisance_oof_summary"] = nuisance
    tagged: list[pd.DataFrame] = [
        tag_consolidation_table(
            nuisance,
            "nuisance_oof_summary",
            causal_paths["oof_predictions"],
            "upstream_stage08b_broader_panel_not_frozen_common_6480",
        )
    ]
    causal_names = (
        "nuisance_fold_audit",
        "overlap_diagnostics",
        "treatment_invariance_audit",
        "learner_stability",
        "dml_estimates",
        "placebo_tests",
    )
    for name in causal_names:
        frame = pd.read_csv(causal_paths[name], low_memory=False)
        raw_tables[name] = frame
        tagged.append(
            tag_consolidation_table(
                frame,
                f"stage08b_{name}",
                causal_paths[name],
                "upstream_stage08b_broader_panel_not_frozen_common_6480",
            )
        )
    robustness_names = (
        "causal_ml_sensitivity",
        "causal_ml_source_audit",
        "future_lead_placebos",
        "future_lead_placebo_fold_audit",
    )
    for name in robustness_names:
        frame = pd.read_csv(robustness_paths[name], low_memory=False)
        raw_tables[f"stage09_{name}"] = frame
        tagged.append(
            tag_consolidation_table(
                frame,
                f"stage09_{name}",
                robustness_paths[name],
                "row_specific_common_or_broader_scope",
            )
        )
    return pd.concat(tagged, ignore_index=True, sort=False), raw_tables


def stable_diagnostic_hash(source: str, row: Mapping[str, Any]) -> str:
    identity_fields = (
        "frequency",
        "frequency_role",
        "index_construction",
        "msdi_path",
        "horizon",
        "horizon_unit",
        "lag",
        "lag_count",
        "maximum_lag_specification",
        "selected_lag",
        "sensitivity_lag_count",
        "sensitivity_maximum_lag",
        "sensitivity_pc_alpha",
        "sensitivity_specification_id",
        "learner",
        "model",
        "causing",
        "caused",
        "source",
        "target",
        "source_lag",
        "diagnostic",
        "variable",
        "analysis",
        "future_treatment_lead",
    )
    payload = {field: row.get(field) for field in identity_fields if field in row}
    payload["source_table"] = source
    encoded = json.dumps(payload, sort_keys=True, default=json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def upstream_significance_rows(
    source: str,
    frame: pd.DataFrame,
    p_column: str,
    effect_candidates: Sequence[str],
    default_primary: bool = False,
    force_primary_false: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if p_column not in frame:
        return rows
    for record in frame.to_dict("records"):
        try:
            raw_p_value = float(record.get(p_column))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(raw_p_value) or not 0.0 <= raw_p_value <= 1.0:
            continue
        effect = next(
            (
                record.get(candidate)
                for candidate in effect_candidates
                if candidate in record and pd.notna(record.get(candidate))
            ),
            np.nan,
        )
        primary_value = record.get("primary_family", default_primary)
        if pd.isna(primary_value):
            primary_value = default_primary
        if isinstance(primary_value, str):
            upstream_primary = primary_value.strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
            }
        else:
            upstream_primary = bool(primary_value)
        family = record.get("family_id")
        if pd.isna(family) if family is not None else True:
            family = f"{source}__{record.get('frequency', 'all')}"
        rows.append(
            {
                "diagnostic_id": f"{source}__{stable_diagnostic_hash(source, record)}",
                "hypothesis_family_id": str(family),
                "upstream_primary_family": upstream_primary,
                "primary_family": False if force_primary_false else upstream_primary,
                "raw_p_value": raw_p_value,
                "effect_size": effect,
                "sample_size": record.get(
                    "n_observations", record.get("sample_size", np.nan)
                ),
                "sample_size_unavailable": pd.isna(
                    record.get("n_observations", record.get("sample_size", np.nan))
                ),
                "diagnostic_source": source,
                "frequency": record.get("frequency"),
                "index_construction": record.get("index_construction"),
                "horizon": record.get("horizon"),
                "model_id": record.get("model_id"),
                "interpretation": record.get(
                    "interpretation", "diagnostic_raw_inference_handoff"
                ),
            }
        )
    return rows


def build_raw_significance_handoff(
    summaries: pd.DataFrame,
    residual_tests: pd.DataFrame,
    calibration: pd.DataFrame,
    stability: pd.DataFrame,
    drivers: pd.DataFrame,
    dynamic_raw: Mapping[str, pd.DataFrame],
    dml_raw: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in summaries.to_dict("records"):
        rows.append(
            {
                "diagnostic_id": "__".join(
                    ["forecast_bias", *[str(record[key]) for key in CELL_KEYS]]
                ),
                "hypothesis_family_id": f"forecast_bias__{record['frequency']}",
                "primary_family": bool(record["primary_family"]),
                "raw_p_value": record["bias_raw_p_value"],
                "effect_size": record["bias"],
                "sample_size": record["n_observations"],
                "diagnostic_source": "forecast_bias_hac",
                "frequency": record["frequency"],
                "index_construction": record["index_construction"],
                "horizon": record["horizon"],
                "model_id": record["model_id"],
                "interpretation": "forecast_bias_diagnostic",
            }
        )
    if not residual_tests.empty:
        rows.extend(
            residual_tests[
                [
                    "diagnostic_id",
                    "hypothesis_family_id",
                    "primary_family",
                    "raw_p_value",
                    "effect_size",
                    "sample_size",
                    "frequency",
                    "index_construction",
                    "horizon",
                    "model_id",
                    "interpretation",
                ]
            ]
            .assign(diagnostic_source="forecast_residual_adequacy")
            .to_dict("records")
        )
    for record in calibration.to_dict("records"):
        for test, p_column, effect in (
            ("intercept_zero", "raw_intercept_p_value", record["intercept"]),
            ("slope_one", "raw_slope_equals_one_p_value", record["slope"] - 1.0),
            ("joint", "raw_joint_wald_p_value", record["effect_size"]),
        ):
            rows.append(
                {
                    "diagnostic_id": f"{record['diagnostic_id']}__{test}",
                    "hypothesis_family_id": record["hypothesis_family_id"],
                    "primary_family": bool(record["primary_family"]),
                    "raw_p_value": record[p_column],
                    "effect_size": effect,
                    "sample_size": record["sample_size"],
                    "diagnostic_source": "mincer_zarnowitz_calibration",
                    "frequency": record["frequency"],
                    "index_construction": record["index_construction"],
                    "horizon": record["horizon"],
                    "model_id": record["model_id"],
                    "interpretation": record["interpretation"],
                }
            )
    for frame, source in (
        (stability, "forecast_stability"),
        (drivers.loc[drivers["feature_status"].eq("used")], "error_driver_hac"),
    ):
        if not frame.empty:
            rows.extend(
                frame[
                    [
                        "diagnostic_id",
                        "hypothesis_family_id",
                        "primary_family",
                        "raw_p_value",
                        "effect_size",
                        "sample_size",
                        "frequency",
                        "index_construction",
                        "horizon",
                        "model_id",
                        "interpretation",
                    ]
                ]
                .assign(diagnostic_source=source)
                .to_dict("records")
            )
    dynamic_specs = {
        "varx_diagnostics_sensitivity": ("p_value", ["statistic"]),
        "conditional_granger_sensitivity": ("p_value", ["wald_statistic"]),
        "conventional_granger_sensitivity": ("p_value", ["f_statistic"]),
        "local_projection_sensitivity": ("p_value", ["coefficient_log_points"]),
        "pcmci_sensitivity_links": ("p_value", ["partial_correlation"]),
    }
    for source, (p_column, effects) in dynamic_specs.items():
        rows.extend(
            upstream_significance_rows(source, dynamic_raw[source], p_column, effects)
        )
    dml_specs = {
        "stage09_causal_ml_sensitivity": (
            "stage09_causal_ml_sensitivity",
            "p_value",
            ["effect_log_points_per_one_sd_innovation"],
        ),
        "stage09_future_lead_placebos": (
            "stage09_future_lead_placebos",
            "p_value",
            ["effect_log_points_per_one_sd_innovation"],
        ),
        "stage08b_placebo_tests": (
            "placebo_tests",
            "p_value",
            ["effect_log_points_per_one_sd_innovation"],
        ),
        "stage08b_nuisance_outcome_calibration": (
            "nuisance_oof_summary",
            "outcome_calibration_joint_raw_p_value",
            ["outcome_calibration_slope"],
        ),
        "stage08b_nuisance_treatment_calibration": (
            "nuisance_oof_summary",
            "treatment_calibration_joint_raw_p_value",
            ["treatment_calibration_slope"],
        ),
    }
    for source, (raw_key, p_column, effects) in dml_specs.items():
        rows.extend(
            upstream_significance_rows(
                source,
                dml_raw[raw_key],
                p_column,
                effects,
                force_primary_false=True,
            )
        )
    result = pd.DataFrame(rows)
    result["raw_p_value"] = pd.to_numeric(result["raw_p_value"], errors="coerce")
    result = result.loc[
        result["raw_p_value"].notna()
        & np.isfinite(result["raw_p_value"])
        & result["raw_p_value"].between(0.0, 1.0)
    ].copy()
    if result["diagnostic_id"].duplicated().any():
        duplicated = result.loc[
            result["diagnostic_id"].duplicated(False), "diagnostic_id"
        ].head(10)
        raise ValueError(f"Raw significance IDs collide: {duplicated.tolist()}")
    result["p_value_adjusted_in_stage09b"] = False
    result["causal_claim_allowed"] = False
    if "sample_size_unavailable" not in result:
        result["sample_size_unavailable"] = result["sample_size"].isna()
    else:
        result["sample_size_unavailable"] = result[
            "sample_size_unavailable"
        ].fillna(result["sample_size"].isna()).astype(bool)
    return result.sort_values("diagnostic_id").reset_index(drop=True)


def audit_upstream_manifest_chain(
    upstream_audit_path: Path,
    supplied_manifests: Mapping[str, Path],
) -> list[dict[str, Any]]:
    audit = pd.read_csv(upstream_audit_path)
    rows: list[dict[str, Any]] = []
    for stage, path in supplied_manifests.items():
        match = audit.loc[audit["upstream_stage"].eq(stage)]
        if len(match) != 1:
            raise ValueError(f"Stage 09 upstream audit lacks unique stage: {stage}")
        expected = str(match["manifest_sha256"].iloc[0])
        row = audit_plain_file(
            path,
            "stage09_upstream_manifest_chain",
            f"{stage}_manifest",
            expected,
        )
        row["details"] = "manifest hash matches frozen Stage 09 upstream audit"
        rows.append(row)
    return rows


def validate_output_keys(tables: Mapping[str, pd.DataFrame]) -> None:
    contracts: dict[str, list[str]] = {
        "forecast_error_observations": [
            *CELL_KEYS,
            "origin_date",
            "target_date",
        ],
        "forecast_error_summary": list(CELL_KEYS),
        "forecast_error_by_operational_regime": [*CELL_KEYS, "stratum"],
        "forecast_error_by_volatility_regime": [*CELL_KEYS, "stratum"],
        "forecast_error_by_disruption_type": [*CELL_KEYS, "stratum"],
        "forecast_residual_tests": ["diagnostic_id"],
        "mincer_zarnowitz_calibration": ["diagnostic_id"],
        "forecast_stability": ["diagnostic_id"],
        "forecast_influence_summary": list(CELL_KEYS),
        "error_driver_coefficients": ["diagnostic_id"],
        "error_driver_performance": [*CELL_KEYS, "model_type"],
        "extreme_forecast_cases": ["case_id"],
        "extreme_case_article_trace": ["case_id", "trace_role", "article_id"],
        "dynamic_diagnostic_consolidation": ["consolidation_id"],
        "dml_diagnostic_consolidation": ["consolidation_id"],
        "raw_significance_handoff": ["diagnostic_id"],
        "reference_index_reconstruction_audit": [
            "frequency",
            "index_construction",
            "path",
            "field",
        ],
        "source_contract_audit": ["audit_id"],
    }
    for name, keys in contracts.items():
        frame = tables[name]
        if frame.empty:
            continue
        assert_unique_keys(frame, keys, name)


def validate_full_output_row_counts(
    tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any]
) -> None:
    """Fail closed when a canonical five-algorithm run changes design counts."""
    expected = config["output_contract"]["expected_full_row_counts"]
    for name, expected_rows in expected.items():
        if name not in tables:
            raise ValueError(f"Expected canonical output is missing: {name}")
        actual_rows = len(tables[name])
        if actual_rows != int(expected_rows):
            raise ValueError(
                f"Canonical output row count changed for {name}: "
                f"expected={expected_rows} observed={actual_rows}"
            )


def output_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "analysis_source",
        "frequency",
        "index_construction",
        "horizon",
        "horizon_unit",
        "model_id",
        "model_family",
        "model_tier",
        "cohort_scope",
        "frequency_role",
        "family_id",
        "score_source",
        "signal_name",
        "origin_date",
        "target_date",
        "actual_log_change",
        "predicted_log_change",
        "forecast_error",
        "absolute_error",
        "squared_error",
        "current_scfi",
        "actual_scfi_level",
        "predicted_scfi_level",
        "level_error",
        "normalized_absolute_error",
        "normalized_squared_error",
        "past_only_target_scale",
        "direction_correct",
        "underprediction",
        "target_operational_regime",
        "origin_operational_regime",
        "volatility",
        "past_only_threshold",
        "volatility_regime",
        "dominant_disruption_type",
        "article_count",
        "source_count",
        "effective_article_count",
        "peak_model_disagreement",
        "dimension_model_disagreement",
        "peak_signal",
        "breadth_signal",
        "brent_log_change",
        "control_missing_count",
        "is_core_diagnostic",
        "primary_family",
    ]
    return frame[[column for column in preferred if column in frame]].copy()


def main() -> None:
    started = time.perf_counter()
    arguments = parse_args()
    executable = assert_project_venv()
    config = load_yaml(arguments.config)
    canonical_config = bool(
        arguments.config.resolve() == DEFAULT_CONFIG.resolve()
        and sha256_file(arguments.config.resolve()) == CANONICAL_CONFIG_SHA256
    )
    version = arguments.version or str(config.get("version", "error_diagnostics_v001"))
    if arguments.smoke_test and arguments.version is None:
        version = "error_diagnostics_smoke_v001"
    elif arguments.maximum_rows is not None and arguments.version is None:
        version = "error_diagnostics_debug_v001"
    logger, log_path = configure_logging(arguments.log_dir.resolve(), version)
    try:
        logger.info("stage=09b status=starting version=%s executable=%s", version, executable)
        if float(config["volatility_regime"]["high_quantile"]) != 0.75:
            raise ValueError("The frozen volatility-regime threshold must be q=0.75")
        if config["stability"]["operational_split"]["disruption_period_start"] != OPERATIONAL_START:
            raise ValueError("Operational disruption start differs from frozen contract")
        if config["stability"]["operational_split"]["disruption_period_end"] != OPERATIONAL_END:
            raise ValueError("Operational disruption end differs from frozen contract")
        expected_articles = int(config["analysis"]["frozen_common_article_cohort"])
        bootstrap_repetitions = (
            arguments.bootstrap_repetitions
            if arguments.bootstrap_repetitions is not None
            else (19 if arguments.smoke_test else 999)
        )
        seed = arguments.seed if arguments.seed is not None else 20260803
        top_cases = (
            arguments.top_error_cases
            if arguments.top_error_cases is not None
            else int(config["extreme_cases"]["largest_absolute_errors_per_cell"])
        )
        top_articles = (
            arguments.top_articles_per_case
            if arguments.top_articles_per_case is not None
            else int(config["extreme_cases"]["maximum_articles_per_case_and_role"])
        )
        if bootstrap_repetitions < 1 or top_cases < 1 or top_articles < 1:
            raise ValueError("Bootstrap repetitions and top-case limits must be positive")

        stage09_required = (
            "forecast_predictions",
            "article_audit",
            "upstream_audit",
            "index_variant_weekly",
            "index_variant_monthly",
            "reference_reconstruction_audit",
            "varx_diagnostics_sensitivity",
            "varx_lag_selection_sensitivity",
            "selected_dynamic_lags",
            "conditional_granger_sensitivity",
            "conventional_granger_sensitivity",
            "pcmci_link_stability",
            "pcmci_sensitivity_links",
            "local_projection_sensitivity",
            "causal_ml_sensitivity",
            "causal_ml_source_audit",
            "future_lead_placebos",
            "future_lead_placebo_fold_audit",
            "nonlinear_source_audit",
        )
        robustness_manifest, robustness_paths, source_rows = audit_manifest_outputs(
            arguments.robustness_manifest,
            "stage09_robustness",
            stage09_required,
        )
        nonlinear_required = ("predictions", "origin_audit")
        nonlinear_manifest, nonlinear_paths, nonlinear_rows = audit_manifest_outputs(
            arguments.nonlinear_manifest,
            "stage07d_nonlinear",
            nonlinear_required,
            require_output_contract=True,
        )
        source_rows.extend(nonlinear_rows)
        causal_required = (
            "oof_predictions",
            "nuisance_fold_audit",
            "overlap_diagnostics",
            "treatment_invariance_audit",
            "learner_stability",
            "dml_estimates",
            "placebo_tests",
        )
        causal_ml_manifest, causal_paths, causal_rows = audit_manifest_outputs(
            arguments.causal_ml_manifest,
            "stage08b_causal_ml",
            causal_required,
        )
        source_rows.extend(causal_rows)
        validate_upstream_contracts(
            config, robustness_manifest, nonlinear_manifest, causal_ml_manifest
        )
        nonlinear_source_audit = pd.read_csv(
            robustness_paths["nonlinear_source_audit"]
        )
        origin_source_row = nonlinear_source_audit.loc[
            nonlinear_source_audit["source"].eq("origin_audit")
        ]
        if len(origin_source_row) != 1:
            raise ValueError("Stage 09 nonlinear origin-audit chain is not unique")
        source_rows.append(
            audit_plain_file(
                nonlinear_paths["origin_audit"],
                "stage09_nonlinear_source_chain",
                "origin_audit",
                str(origin_source_row["sha256_runtime"].iloc[0]),
            )
        )
        source_rows.extend(
            audit_upstream_manifest_chain(
                robustness_paths["upstream_audit"],
                {
                    "prediction": arguments.prediction_manifest,
                    "nonlinear": arguments.nonlinear_manifest,
                    "causality": arguments.causality_manifest,
                    "causal_ml": arguments.causal_ml_manifest,
                },
            )
        )

        stage09_inputs = robustness_manifest["inputs"]
        direct_inputs = {
            "article_scores": arguments.article_scores,
            "weekly_panel": arguments.weekly_panel,
            "monthly_panel": arguments.monthly_panel,
            "model_registry": arguments.model_registry,
        }
        for name, path in direct_inputs.items():
            recorded = stage09_inputs[name]["sha256"]
            source_rows.append(
                audit_plain_file(path, "stage09_frozen_input", name, recorded)
            )
        source_rows.append(
            audit_plain_file(arguments.config, "stage09b", "configuration")
        )
        source_rows.append(
            audit_plain_file(Path(__file__), "stage09b", "script")
        )
        registry = load_yaml(arguments.model_registry)
        aliases = tuple(
            item["alias"] for item in registry.get("active_scoring_models", [])
        )
        if aliases != FROZEN_MODEL_ALIASES:
            raise ValueError(f"Frozen model registry alias order changed: {aliases}")

        article_audit = pd.read_csv(robustness_paths["article_audit"])
        validate_common_article_scope(article_audit, expected_articles)
        articles = load_common_articles(arguments.article_scores, expected_articles)
        logger.info(
            "articles common_complete=%s first=%s last=%s",
            len(articles),
            articles["publication_date"].min().date(),
            articles["publication_date"].max().date(),
        )
        weekly_article_context = article_context(articles, "weekly")
        monthly_article_context = article_context(articles, "monthly")
        weekly_context = build_weekly_context(
            arguments.weekly_panel, weekly_article_context, config
        )
        monthly_context = build_monthly_context(
            arguments.monthly_panel,
            weekly_context,
            monthly_article_context,
            config,
        )
        context = pd.concat([weekly_context, monthly_context], ignore_index=True, sort=False)
        reconstruction_audit = validate_context_reconstruction(
            weekly_context,
            monthly_context,
            robustness_paths["index_variant_weekly"],
            robustness_paths["index_variant_monthly"],
        )
        logger.info(
            "context weekly_rows=%s monthly_rows=%s reconstruction_checks=%s",
            len(weekly_context),
            len(monthly_context),
            len(reconstruction_audit),
        )

        stage09_predictions_full = load_stage09_predictions(
            robustness_paths["forecast_predictions"]
        )
        required_reference = set(
            config["analysis"]["required_reference_specifications"]
        )
        observed_reference = set(
            stage09_predictions_full.loc[
                stage09_predictions_full["is_core_diagnostic"], "model_id"
            ].unique()
        )
        if observed_reference != required_reference:
            raise ValueError(
                f"Stage 09 reference specifications changed: {observed_reference}"
            )
        expected_stage09_rows = int(
            robustness_manifest["output_row_counts"]["forecast_predictions"]
        )
        if len(stage09_predictions_full) != expected_stage09_rows:
            raise ValueError("Stage 09 prediction row contract changed after load")
        nonlinear_total_rows = next(
            row["row_count"]
            for row in nonlinear_rows
            if row["artifact"] == "predictions"
        )
        expected_nonlinear_rows = int(nonlinear_manifest["counts"]["total_forecasts"])
        if nonlinear_total_rows != expected_nonlinear_rows:
            raise ValueError(
                "Nonlinear prediction row count differs from its manifest count"
            )
        nonlinear_predictions_full = load_nonlinear_predictions(
            nonlinear_paths["predictions"]
        )
        nonlinear_cell_counts = nonlinear_predictions_full.groupby(
            ["frequency", "index_construction", "horizon", "algorithm"],
            observed=True,
        )["model_id"].nunique()
        expected_algorithms = set(config["analysis"]["nonlinear_algorithms"])
        observed_algorithms = set(nonlinear_predictions_full["algorithm"].unique())
        expected_algorithm_cells = int(
            config["analysis"]["nonlinear_selected_algorithm_cells"]
        )
        expected_models_per_cell = int(
            config["analysis"]["nonlinear_models_per_cell"]
        )
        base_cells = nonlinear_predictions_full[
            ["frequency", "index_construction", "horizon"]
        ].drop_duplicates()
        expected_base_cells = int(
            config["analysis"]["nonlinear_frequency_construction_horizon_cells"]
        )
        if observed_algorithms != expected_algorithms:
            raise ValueError(
                "Secondary nonlinear algorithms changed: "
                f"expected={sorted(expected_algorithms)} "
                f"observed={sorted(observed_algorithms)}"
            )
        if len(base_cells) != expected_base_cells:
            raise ValueError(
                "Secondary nonlinear frequency/construction/horizon scope changed: "
                f"expected={expected_base_cells} observed={len(base_cells)}"
            )
        if (
            len(nonlinear_cell_counts) != expected_algorithm_cells
            or not nonlinear_cell_counts.eq(expected_models_per_cell).all()
        ):
            raise ValueError(
                "Secondary nonlinear diagnostic selection must contain exactly "
                "baseline, median peak, and median breadth in all "
                f"{expected_algorithm_cells} algorithm cells"
            )
        expected_selected_rows = int(
            config["analysis"]["nonlinear_selected_prediction_rows"]
        )
        if len(nonlinear_predictions_full) != expected_selected_rows:
            raise ValueError(
                "Secondary nonlinear selected prediction-row count changed: "
                f"expected={expected_selected_rows} "
                f"observed={len(nonlinear_predictions_full)}"
            )
        origin_audit = pd.read_csv(nonlinear_paths["origin_audit"])
        if not origin_audit["origin_sets_identical"].astype(bool).all():
            raise ValueError("Nonlinear common-origin audit contains failures")
        predictions = pd.concat(
            [stage09_predictions_full, nonlinear_predictions_full],
            ignore_index=True,
            sort=False,
        )
        predictions = apply_debug_limit(
            predictions, arguments.maximum_rows, arguments.smoke_test
        )
        assert_unique_keys(
            predictions,
            [*CELL_KEYS, "origin_date", "target_date"],
            "combined forecast predictions",
        )
        errors = enrich_predictions(predictions, context)
        logger.info(
            "forecast_errors rows=%s cells=%s stage09_full_rows=%s nonlinear_selected_full_rows=%s",
            len(errors),
            errors.groupby(list(CELL_KEYS)).ngroups,
            len(stage09_predictions_full),
            len(nonlinear_predictions_full),
        )

        error_summary = summarize_forecast_errors(
            errors, bootstrap_repetitions, seed
        )
        by_operational = summarize_stratum(
            errors, "target_operational_regime", "target_operational_regime"
        )
        by_volatility = summarize_stratum(
            errors, "volatility_regime", "origin_known_past_only_volatility_regime"
        )
        by_disruption = summarize_stratum(
            errors, "dominant_disruption_type", "origin_dominant_disruption_type"
        )
        residual_tests = residual_adequacy_tests(errors, config)
        calibration = mincer_zarnowitz_calibration(errors)
        stability = stability_diagnostics(errors, config)
        influence = influence_summary(errors)
        driver_coefficients, driver_performance = error_driver_analysis(
            errors, config, seed
        )
        cases = extreme_forecast_cases(errors, top_cases)
        if arguments.skip_article_trace:
            trace = pd.DataFrame(
                columns=[
                    "case_id",
                    "trace_role",
                    "article_rank",
                    "article_id",
                    "publication_date",
                    "headline",
                    "source_domain",
                    "duplicate_group_id",
                    "duplicate_group_size",
                    "peak_severity_score",
                    "breadth_severity_score",
                    "peak_model_disagreement_sd",
                    "dimension_disagreement_sd_mean",
                    "novelty_weight",
                    "absolute_novelty_weighted_score_contribution",
                    "known_at_forecast_origin",
                    "event_claim_allowed",
                ]
            )
        else:
            trace = trace_extreme_case_articles(cases, articles, top_articles)
        logger.info(
            "core_diagnostics summaries=%s residual_tests=%s calibration=%s stability=%s cases=%s trace_rows=%s",
            len(error_summary),
            len(residual_tests),
            len(calibration),
            len(stability),
            len(cases),
            len(trace),
        )

        dynamic, dynamic_raw = consolidate_dynamic_diagnostics(robustness_paths)
        dml, dml_raw = consolidate_dml_diagnostics(causal_paths, robustness_paths)
        significance = build_raw_significance_handoff(
            error_summary,
            residual_tests,
            calibration,
            stability,
            driver_coefficients,
            dynamic_raw,
            dml_raw,
        )
        source_audit = pd.DataFrame(source_rows)
        source_audit["audit_id"] = [
            f"source_{index:04d}" for index in range(len(source_audit))
        ]
        source_audit["all_contracts_passed"] = bool(
            source_audit["contract_passed"].all()
        )
        if significance["raw_p_value"].isna().any() or not significance[
            "raw_p_value"
        ].between(0.0, 1.0).all():
            raise ValueError("Raw significance handoff contains invalid p-values")
        forecast_observations = output_frame_columns(errors)
        tables: dict[str, pd.DataFrame] = {
            "forecast_error_observations": forecast_observations,
            "forecast_error_summary": error_summary,
            "forecast_error_by_operational_regime": by_operational,
            "forecast_error_by_volatility_regime": by_volatility,
            "forecast_error_by_disruption_type": by_disruption,
            "forecast_residual_tests": residual_tests,
            "mincer_zarnowitz_calibration": calibration,
            "forecast_stability": stability,
            "forecast_influence_summary": influence,
            "error_driver_coefficients": driver_coefficients,
            "error_driver_performance": driver_performance,
            "extreme_forecast_cases": cases,
            "extreme_case_article_trace": trace,
            "dynamic_diagnostic_consolidation": dynamic,
            "dml_diagnostic_consolidation": dml,
            "raw_significance_handoff": significance,
            "source_contract_audit": source_audit,
            "reference_index_reconstruction_audit": reconstruction_audit,
        }
        validate_output_keys(tables)
        canonical_output_request = bool(
            not arguments.smoke_test
            and arguments.maximum_rows is None
            and not arguments.skip_article_trace
            and canonical_config
            and version == str(config.get("version", "error_diagnostics_v001"))
        )
        if canonical_output_request:
            validate_full_output_row_counts(tables, config)
        required_table_names = list(config["output_contract"]["tables"])
        required_tables_nonempty = all(
            name in tables and not tables[name].empty for name in required_table_names
        )
        if (
            not arguments.smoke_test
            and arguments.maximum_rows is None
            and not arguments.skip_article_trace
            and not required_tables_nonempty
        ):
            missing_outputs = [
                name
                for name in required_table_names
                if name not in tables or tables[name].empty
            ]
            raise ValueError(
                f"Canonical run produced empty required outputs: {missing_outputs}"
            )

        output_dir = arguments.output_dir.resolve()
        output_paths: dict[str, Path] = {}
        output_hashes: dict[str, str] = {}
        output_counts: dict[str, int] = {}
        for name, frame in tables.items():
            path = output_dir / f"{version}_{name}.csv"
            atomic_write_csv(frame, path)
            output_paths[name] = path
            output_hashes[name] = sha256_file(path)
            output_counts[name] = len(frame)
            logger.info(
                "output table=%s rows=%s sha256=%s path=%s",
                name,
                len(frame),
                output_hashes[name],
                path,
            )

        canonical_top_cases = 10
        canonical_top_articles = 10
        if int(config["extreme_cases"]["largest_absolute_errors_per_cell"]) != 10:
            raise ValueError("Canonical largest-error case count must remain 10")
        if int(config["extreme_cases"]["maximum_articles_per_case_and_role"]) != 10:
            raise ValueError("Canonical article trace count must remain 10")
        full_scope = bool(
            not arguments.smoke_test
            and arguments.maximum_rows is None
            and not arguments.skip_article_trace
            and bootstrap_repetitions == 999
            and seed == 20260803
            and version == str(config.get("version", "error_diagnostics_v001"))
            and canonical_config
            and top_cases == canonical_top_cases
            and top_articles == canonical_top_articles
            and len(stage09_predictions_full) == expected_stage09_rows
            and nonlinear_total_rows == expected_nonlinear_rows
            and required_tables_nonempty
        )
        all_contracts_passed = bool(
            source_audit["contract_passed"].all()
            and reconstruction_audit["reconstruction_passed"].all()
        )
        status = determine_run_status(
            arguments.smoke_test,
            arguments.maximum_rows,
            all_contracts_passed,
            full_scope,
        )
        input_signature = {
            row["audit_id"]: row["actual_sha256"]
            for row in source_audit.to_dict("records")
        }
        run_signature = hashlib.sha256(
            json.dumps(
                {
                    "inputs": input_signature,
                    "version": version,
                    "bootstrap_repetitions": bootstrap_repetitions,
                    "seed": seed,
                    "top_cases": top_cases,
                    "top_articles": top_articles,
                    "maximum_rows": arguments.maximum_rows,
                    "smoke_test": arguments.smoke_test,
                },
                sort_keys=True,
                default=json_default,
            ).encode("utf-8")
        ).hexdigest()
        manifest_path = output_dir / f"{version}_manifest.json"
        manifest = {
            "version": version,
            "created_at_utc": datetime.now(timezone.utc),
            "status": status,
            "run_signature": run_signature,
            "runtime_seconds": time.perf_counter() - started,
            "primary_frequency": "monthly",
            "weekly_role": "higher_frequency_robustness",
            "causal_claim_allowed": False,
            "publication_scope_complete": full_scope and all_contracts_passed,
            "publication_eligible_without_human_validation": False,
            "human_validation_required": True,
            "interpretation": (
                "forecast_error_diagnostics_and_observational_model_checks; "
                "passing diagnostics cannot establish causality"
            ),
            "arguments": vars(arguments),
            "design": {
                "frozen_common_articles": expected_articles,
                "monthly_primary_construction": "adapted_pdf_sqrt_ema3",
                "weekly_role": "higher_frequency_robustness",
                "monthly_constructions": list(MONTHLY_CONSTRUCTIONS),
                "reference_stage09_specifications": list(CORE_STAGE09_MODELS),
                "nonlinear_role": "secondary_fixed_parameter_common_cohort",
                "forecast_error_sign": "actual_minus_predicted",
                "normalized_loss_scale": "origin_known_past_only_trailing_target_sd",
                "volatility_threshold": "past_only_expanding_q75",
                "operational_split_is_official_pandemic_dating": False,
                "raw_p_values_only": True,
                "p_value_adjustment_stage": "09c_significance_analysis",
            },
            "scope": {
                "stage09_prediction_rows_available": len(stage09_predictions_full),
                "nonlinear_prediction_rows_available": nonlinear_total_rows,
                "nonlinear_selected_prediction_rows_available": len(
                    nonlinear_predictions_full
                ),
                "diagnosed_prediction_rows": len(errors),
                "diagnostic_cells": int(errors.groupby(list(CELL_KEYS)).ngroups),
                "bootstrap_repetitions": bootstrap_repetitions,
                "top_error_cases_per_role": top_cases,
                "top_articles_per_case_role": top_articles,
                "debug_maximum_rows_per_cell": arguments.maximum_rows,
                "article_trace_skipped": arguments.skip_article_trace,
            },
            "audit": {
                "all_source_contracts_passed": all_contracts_passed,
                "reference_index_reconstruction_all_passed": bool(
                    reconstruction_audit["reconstruction_passed"].all()
                ),
                "forecast_error_unique_keys_passed": True,
                "p_values_adjusted_in_stage09b": False,
                "future_information_used_in_volatility_regime": False,
                "article_trace_is_event_identification": False,
                "dml_diagnostics_can_upgrade_causal_claim": False,
            },
            "inputs": input_signature,
            "outputs": {name: str(path) for name, path in output_paths.items()},
            "output_sha256": output_hashes,
            "output_row_counts": output_counts,
            "log": str(log_path),
            "code": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "software": package_versions(),
        }
        manifest["outputs"]["manifest"] = str(manifest_path)
        atomic_write_text(
            manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True, default=json_default) + "\n",
        )
        logger.info(
            "stage=09b status=%s runtime_seconds=%.3f manifest=%s",
            status,
            time.perf_counter() - started,
            manifest_path,
        )
    except Exception:
        logger.exception("stage=09b status=failed")
        raise


if __name__ == "__main__":
    main()
