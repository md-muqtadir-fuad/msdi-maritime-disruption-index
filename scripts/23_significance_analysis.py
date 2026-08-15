"""Consolidate effect sizes and multiplicity-aware statistical inference.

Stage 09c never refits an upstream model. It verifies frozen source artifacts,
retains the earliest registered hypothesis-family owner, removes semantic
aliases from canonical discovery counts, and recomputes all adjustments from
raw p-values. Predictive, diagnostic, and observational evidence remain
separate; no result can authorize a causal claim.
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
import yaml
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/significance.yaml"
DEFAULT_DIAGNOSTICS_MANIFEST = (
    PROJECT_ROOT / "outputs/diagnostics/error_diagnostics_v001_manifest.json"
)
DEFAULT_ROBUSTNESS_MANIFEST = (
    PROJECT_ROOT / "outputs/robustness/robustness_v001_manifest.json"
)
DEFAULT_PREDICTION_MANIFEST = (
    PROJECT_ROOT / "outputs/prediction_benchmarks/forecast_v001_manifest.json"
)
DEFAULT_WEEKLY_LLM_MANIFEST = (
    PROJECT_ROOT / "outputs/llm_scfi_comparison/llm_scfi_v001_manifest.json"
)
DEFAULT_MONTHLY_LLM_MANIFEST = (
    PROJECT_ROOT / "outputs/llm_scfi_comparison/llm_scfi_monthly_v001_manifest.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/significance"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs/logs"

# Updated only after configs/significance.yaml is frozen.
CANONICAL_CONFIG_SHA256 = (
    "55bd91437b238fbd39642038c7c213ab27ffca61ad55906cb5fe2f8e2ce655ab"
)

DYNAMIC_IDENTITY_FIELDS = (
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

REGISTRY_DEFAULTS: dict[str, Any] = {
    "hypothesis_id": None,
    "semantic_key": None,
    "source_stage": None,
    "source_table": None,
    "source_row_number": np.nan,
    "evidence_domain": None,
    "inference_role": None,
    "frequency": None,
    "frequency_role": None,
    "index_construction": None,
    "horizon": np.nan,
    "horizon_unit": None,
    "baseline_model": None,
    "challenger_model": None,
    "baseline_index_construction": None,
    "challenger_index_construction": None,
    "model_id": None,
    "learner": None,
    "msdi_path": None,
    "signal_name": None,
    "score_source": None,
    "test_method": None,
    "diagnostic_source": None,
    "null_hypothesis": None,
    "alternative": None,
    "sidedness": None,
    "effect_metric": None,
    "effect_unit": None,
    "effect_size": np.nan,
    "effect_direction": None,
    "effect_interpretation": None,
    "economic_magnitude_metric": None,
    "economic_magnitude_value": np.nan,
    "standard_error": np.nan,
    "ci_lower": np.nan,
    "ci_upper": np.nan,
    "confidence_level": np.nan,
    "uncertainty_method": None,
    "ci_available": False,
    "ci_unavailable_reason": None,
    "sample_size": np.nan,
    "sample_start_date": None,
    "sample_end_date": None,
    "sample_dates_unavailable_reason": None,
    "index_construction_unavailable_reason": None,
    "signed_effect_interpretable": False,
    "rejection_meaning": None,
    "claim_direction": None,
    "raw_p_value": np.nan,
    "upstream_family_id": None,
    "upstream_adjustment_method": None,
    "upstream_adjusted_p_value": np.nan,
    "upstream_reconciliation_family_id": None,
    "upstream_recomputed_adjusted_p_value": np.nan,
    "upstream_adjustment_delta": np.nan,
    "correction_family_id": None,
    "correction_family_size": np.nan,
    "correction_method": None,
    "adjusted_p_value_within_family": np.nan,
    "reject_raw_at_alpha": False,
    "reject_within_family_at_alpha": False,
    "global_role_family_size": np.nan,
    "adjusted_p_value_global_role": np.nan,
    "reject_global_role_at_alpha": False,
    "canonical_inference": True,
    "duplicate_of": None,
    "duplicate_reason": None,
    "maximum_lag_specification": np.nan,
    "selected_lag": np.nan,
    "equation": None,
    "term": None,
    "diagnostic": None,
    "variable": None,
    "supporting_stage09_hypothesis_id": None,
    "supporting_stage09_dm_raw_p_value": np.nan,
    "supporting_stage09_reassigned_adjusted_p_value": np.nan,
    "supporting_block_bootstrap_p_value": np.nan,
    "supporting_block_bootstrap_ci_lower": np.nan,
    "supporting_block_bootstrap_ci_upper": np.nan,
    "supporting_clark_west_one_sided_p_value": np.nan,
    "supporting_rmse_improvement_percent": np.nan,
    "supporting_incremental_oos_r_squared": np.nan,
    "supporting_streams_eligible_for_canonical_multiplicity": False,
    "interpretation": None,
    "causal_claim_allowed": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the consolidated multiplicity-aware inference audit."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--diagnostics-manifest", type=Path, default=DEFAULT_DIAGNOSTICS_MANIFEST
    )
    parser.add_argument(
        "--robustness-manifest", type=Path, default=DEFAULT_ROBUSTNESS_MANIFEST
    )
    parser.add_argument(
        "--prediction-manifest", type=Path, default=DEFAULT_PREDICTION_MANIFEST
    )
    parser.add_argument(
        "--weekly-llm-manifest", type=Path, default=DEFAULT_WEEKLY_LLM_MANIFEST
    )
    parser.add_argument(
        "--monthly-llm-manifest", type=Path, default=DEFAULT_MONTHLY_LLM_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--version", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration not found: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {resolved}")
    return payload


def read_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"JSON source not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON source must be an object: {resolved}")
    return payload


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def read_sha256_lock(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Script SHA-256 lock not found: {resolved}")
    value = resolved.read_text(encoding="utf-8").strip().lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Invalid script SHA-256 lock: {resolved}")
    return value


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


def stable_hash(source: str, values: Mapping[str, Any], length: int = 24) -> str:
    payload = json.dumps(
        {"source": source, **dict(values)},
        sort_keys=True,
        default=json_default,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def stable_dynamic_hash(source: str, row: Mapping[str, Any]) -> str:
    payload = {
        field: row.get(field) for field in DYNAMIC_IDENTITY_FIELDS if field in row
    }
    payload["source_table"] = source
    encoded = json.dumps(payload, sort_keys=True, default=json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def assert_project_venv() -> Path:
    executable = Path(sys.executable).resolve()
    expected = (PROJECT_ROOT / ".venv").resolve()
    if expected not in executable.parents:
        raise RuntimeError(
            "Run this stage with .venv\\Scripts\\python.exe "
            "scripts\\23_significance_analysis.py"
        )
    return executable


def configure_logging(log_dir: Path, version: str) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"09c_significance_analysis_{version}.log"
    logger = logging.getLogger(f"significance.{version}")
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


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def resolve_recorded_path(value: str | Path, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    project_relative = (PROJECT_ROOT / path).resolve()
    if project_relative.exists():
        return project_relative
    return (manifest_path.resolve().parent / path).resolve()


def normalize_bool(value: Any, default: bool = False) -> bool:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y"}:
            return True
        if token in {"0", "false", "no", "n"}:
            return False
        raise ValueError(f"Invalid boolean token: {value!r}")
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if numeric == 1.0:
            return True
        if numeric == 0.0:
            return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def strict_bool_series(series: pd.Series, label: str) -> pd.Series:
    try:
        return series.map(normalize_bool).astype(bool)
    except ValueError as error:
        raise ValueError(f"{label} contains a non-boolean value") from error


def normalize_method(method: Any) -> str:
    value = str(method).strip().lower()
    if value == "holm":
        return "holm"
    if value in {"fdr_bh", "benjamini_hochberg", "benjamini-hochberg"}:
        return "benjamini_hochberg"
    if value in {"none", "nan", ""}:
        return "none"
    raise ValueError(f"Unknown multiplicity method: {method}")


def adjust_pvalues(values: Iterable[float], method: str) -> np.ndarray:
    raw = np.asarray(list(values), dtype=float)
    adjusted = np.full(len(raw), np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(raw))
    if not len(finite_positions):
        return adjusted
    finite = raw[finite_positions]
    if ((finite < 0) | (finite > 1)).any():
        raise ValueError("P-values must be finite values in [0, 1]")
    order = np.argsort(finite, kind="mergesort")
    ordered = finite[order]
    count = len(ordered)
    if method == "holm":
        ordered_adjusted = np.maximum.accumulate(
            (count - np.arange(count)) * ordered
        )
    elif method == "benjamini_hochberg":
        ordered_adjusted = np.minimum.accumulate(
            (ordered * count / (np.arange(count) + 1))[::-1]
        )[::-1]
    else:
        raise ValueError(f"Unknown p-value adjustment method: {method}")
    restored = np.empty(count, dtype=float)
    restored[order] = np.clip(ordered_adjusted, 0.0, 1.0)
    adjusted[finite_positions] = restored
    return adjusted


def normal_interval(
    effect: Any, standard_error: Any, confidence_level: float
) -> tuple[float, float]:
    effect_value = pd.to_numeric(pd.Series([effect]), errors="coerce").iloc[0]
    se_value = pd.to_numeric(pd.Series([standard_error]), errors="coerce").iloc[0]
    if not np.isfinite(effect_value) or not np.isfinite(se_value) or se_value < 0:
        return np.nan, np.nan
    critical = float(norm.ppf(0.5 + confidence_level / 2.0))
    return float(effect_value - critical * se_value), float(
        effect_value + critical * se_value
    )


def direction(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(numeric):
        return "unavailable"
    if numeric > 0:
        return "positive"
    if numeric < 0:
        return "negative"
    return "zero"


def is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null"}
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def make_registry_row(**values: Any) -> dict[str, Any]:
    row = dict(REGISTRY_DEFAULTS)
    row.update(values)
    row["effect_direction"] = values.get(
        "effect_direction", direction(row["effect_size"])
    )
    lower = pd.to_numeric(pd.Series([row["ci_lower"]]), errors="coerce").iloc[0]
    upper = pd.to_numeric(pd.Series([row["ci_upper"]]), errors="coerce").iloc[0]
    row["ci_available"] = bool(np.isfinite(lower) and np.isfinite(upper))
    if row["ci_available"]:
        row["ci_unavailable_reason"] = None
    elif not row.get("ci_unavailable_reason"):
        row["ci_unavailable_reason"] = "compatible_upstream_interval_unavailable"
    if is_missing_scalar(row.get("index_construction")) and is_missing_scalar(
        row.get("index_construction_unavailable_reason")
    ):
        row["index_construction_unavailable_reason"] = (
            "upstream_inference_table_did_not_export_index_construction"
        )
    if (
        is_missing_scalar(row.get("sample_start_date"))
        and is_missing_scalar(row.get("sample_end_date"))
        and is_missing_scalar(row.get("sample_dates_unavailable_reason"))
    ):
        row["sample_dates_unavailable_reason"] = (
            "upstream_inference_table_did_not_export_sample_dates"
        )
    row["causal_claim_allowed"] = False
    return row


def unique_hypothesis_index(registry: pd.DataFrame) -> dict[str, Any]:
    if registry["hypothesis_id"].duplicated().any():
        raise ValueError("Hypothesis IDs must be unique before enrichment")
    return dict(zip(registry["hypothesis_id"].astype(str), registry.index))


def audit_manifest_outputs(
    manifest_path: Path,
    stage: str,
    required_outputs: Sequence[str],
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Path], list[dict[str, Any]]]:
    resolved = manifest_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{stage} manifest not found: {resolved}")
    actual_manifest_hash = sha256_file(resolved)
    if (
        expected_manifest_sha256 is not None
        and actual_manifest_hash != expected_manifest_sha256
    ):
        raise ValueError(f"{stage} manifest SHA-256 contract failed")
    manifest = read_json(resolved)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"{stage} manifest lacks outputs mapping")
    recorded_hashes = manifest.get("output_sha256")
    recorded_counts = manifest.get("output_row_counts")
    if not isinstance(recorded_hashes, dict) or not isinstance(recorded_counts, dict):
        raise ValueError(f"{stage} manifest lacks frozen output hashes/row counts")
    rows: list[dict[str, Any]] = [
        {
            "source_stage": stage,
            "artifact": "manifest",
            "path": str(resolved),
            "expected_sha256": expected_manifest_sha256,
            "actual_sha256": actual_manifest_hash,
            "expected_rows": 1,
            "actual_rows": 1,
            "contract_passed": expected_manifest_sha256 is None
            or actual_manifest_hash == expected_manifest_sha256,
            "details": f"version={manifest.get('version')};status={manifest.get('status')}",
        }
    ]
    paths: dict[str, Path] = {}
    for key in required_outputs:
        if key not in outputs:
            raise ValueError(f"{stage} manifest lacks required output: {key}")
        path = resolve_recorded_path(outputs[key], resolved)
        if not path.is_file():
            raise FileNotFoundError(f"{stage}:{key} not found: {path}")
        if key not in recorded_hashes or key not in recorded_counts:
            raise ValueError(f"{stage}:{key} lacks frozen hash/row count")
        actual_hash = sha256_file(path)
        expected_hash = str(recorded_hashes[key])
        frame = pd.read_csv(path, low_memory=False)
        actual_rows = len(frame)
        expected_rows = int(recorded_counts[key])
        passed = actual_hash == expected_hash and actual_rows == expected_rows
        rows.append(
            {
                "source_stage": stage,
                "artifact": key,
                "path": str(path),
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "expected_rows": expected_rows,
                "actual_rows": actual_rows,
                "contract_passed": passed,
                "details": "manifest hash and row-count contract",
            }
        )
        if not passed:
            raise ValueError(f"{stage}:{key} hash/row-count contract failed")
        paths[key] = path
    return manifest, paths, rows


def audit_legacy_inference_table(
    stage: str,
    manifest_path: Path,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    if manifest.get("version") != contract["manifest_version"]:
        raise ValueError(f"{stage} manifest version changed")
    if manifest.get("status") != contract["manifest_status"]:
        raise ValueError(f"{stage} manifest status changed")
    output_key = str(contract["output_key"])
    path = resolve_recorded_path(manifest["outputs"][output_key], manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"{stage}:{output_key} not found: {path}")
    actual_hash = sha256_file(path)
    frame = pd.read_csv(path, low_memory=False)
    actual_rows = len(frame)
    recorded_hashes = manifest.get("output_sha256")
    recorded_counts = manifest.get("output_row_counts")
    manifest_contract_available = bool(
        isinstance(recorded_hashes, dict)
        and isinstance(recorded_counts, dict)
        and output_key in recorded_hashes
        and output_key in recorded_counts
    )
    if manifest_contract_available:
        expected_hash = str(recorded_hashes[output_key])
        expected_rows = int(recorded_counts[output_key])
        contract_source = "producer_manifest"
    else:
        if bool(contract.get("require_manifest_output_contract", False)):
            raise ValueError(
                f"{stage}:{output_key} requires manifest-declared output hash/count"
            )
        fallback_hash = contract.get("legacy_fallback_sha256")
        fallback_rows = contract.get("legacy_fallback_rows")
        if fallback_hash is None or fallback_rows is None:
            raise ValueError(
                f"{stage}:{output_key} has neither a manifest output contract "
                "nor an explicit legacy fallback"
            )
        expected_hash = str(fallback_hash)
        expected_rows = int(fallback_rows)
        contract_source = "legacy_first_freeze_fallback"
    semantic_expected_raw = contract.get("expected_rows")
    semantic_expected_rows = (
        None if semantic_expected_raw is None else int(semantic_expected_raw)
    )
    passed = (
        actual_hash == expected_hash
        and actual_rows == expected_rows
        and (
            semantic_expected_rows is None
            or actual_rows == semantic_expected_rows
        )
    )
    if not passed:
        raise ValueError(f"{stage} trusted inference table changed")
    audit = {
        "source_stage": stage,
        "artifact": output_key,
        "path": str(path),
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "contract_passed": True,
        "details": (
            f"{contract_source} hash/count contract; "
            f"semantic_expected_rows={semantic_expected_rows}"
        ),
    }
    return manifest, path, audit


def validate_manifest_contracts(
    config: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    robustness: Mapping[str, Any],
) -> None:
    contract = config["upstream_contracts"]
    checks = (
        (
            diagnostics.get("version") == contract["error_diagnostics_version"],
            "Stage 09b version",
        ),
        (
            diagnostics.get("status") == contract["error_diagnostics_status"],
            "Stage 09b full status",
        ),
        (
            robustness.get("version") == contract["robustness_version"],
            "Stage 09 version",
        ),
        (
            robustness.get("status") == contract["robustness_status"],
            "Stage 09 full status",
        ),
        (
            not normalize_bool(diagnostics.get("causal_claim_allowed"), True),
            "Stage 09b no-causal-claim",
        ),
        (
            not normalize_bool(robustness.get("causal_claim_allowed"), True),
            "Stage 09 no-causal-claim",
        ),
    )
    failed = [label for passed, label in checks if not passed]
    if failed:
        raise ValueError("Upstream manifest contract failed: " + ", ".join(failed))


def validate_stage09_manifest_chain(
    upstream_audit: pd.DataFrame,
    supplied: Mapping[str, Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage, path in supplied.items():
        match = upstream_audit.loc[upstream_audit["upstream_stage"].eq(stage)]
        if len(match) != 1:
            raise ValueError(f"Stage 09 upstream audit lacks unique stage: {stage}")
        expected = str(match["manifest_sha256"].iloc[0])
        actual = sha256_file(path.resolve())
        passed = expected == actual and normalize_bool(
            match["contract_passed"].iloc[0]
        )
        rows.append(
            {
                "source_stage": "stage09_upstream_manifest_chain",
                "artifact": f"{stage}_manifest",
                "path": str(path.resolve()),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "expected_rows": 1,
                "actual_rows": 1,
                "contract_passed": passed,
                "details": "exact manifest frozen by Stage 09",
            }
        )
        if not passed:
            raise ValueError(f"Legacy manifest hash mismatch: {stage}")
    return rows


def forecast_effect_unit(test: str) -> str:
    if "squared" in test:
        return "squared_log_change_loss"
    if "absolute" in test:
        return "absolute_log_change_loss"
    return "source_native_loss_units"


def forecast_sidedness(test: str) -> tuple[str, str]:
    if "clark_west" in test:
        return "one_sided", "challenger_has_lower_expected_adjusted_loss"
    return "two_sided", "unequal_expected_forecast_loss"


def forecast_hypothesis_fields(test: str) -> dict[str, str]:
    if "clark_west" in test:
        return {
            "null_hypothesis": "no_positive_clark_west_adjusted_mspe_gain",
            "effect_metric": "clark_west_adjusted_loss_difference",
            "uncertainty_method": "hac_normal_two_sided_interval_for_one_sided_test",
        }
    return {
        "null_hypothesis": "equal_expected_forecast_loss",
        "effect_metric": "mean_loss_improvement",
        "uncertainty_method": "hac_normal_approximation",
    }


def build_legacy_prediction_registry(
    frame: pd.DataFrame, confidence_level: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_number, record in enumerate(frame.to_dict("records")):
        test = str(record["test"])
        sidedness, alternative = forecast_sidedness(test)
        hypothesis_fields = forecast_hypothesis_fields(test)
        lower, upper = normal_interval(
            record["mean_loss_improvement"], record["standard_error"], confidence_level
        )
        identity = {
            key: record.get(key)
            for key in (
                "horizon_weeks",
                "baseline_model",
                "challenger_model",
                "test",
            )
        }
        hypothesis_id = "stage07_prediction__" + stable_hash(
            "stage07_prediction", identity
        )
        rows.append(
            make_registry_row(
                hypothesis_id=hypothesis_id,
                semantic_key="forecast__weekly__{h}__{b}__{c}__{t}".format(
                    h=record["horizon_weeks"],
                    b=record["baseline_model"],
                    c=record["challenger_model"],
                    t=test,
                ),
                source_stage="stage07_prediction",
                source_table="stage07_prediction_statistical_tests",
                source_row_number=row_number,
                evidence_domain="forecast_benchmark",
                inference_role="prespecified_secondary",
                frequency="weekly",
                frequency_role="higher_frequency_robustness_legacy_panel",
                index_construction="legacy_weekly_prediction_panel",
                horizon=int(record["horizon_weeks"]),
                horizon_unit="weeks",
                baseline_model=record["baseline_model"],
                challenger_model=record["challenger_model"],
                test_method=test,
                null_hypothesis=hypothesis_fields["null_hypothesis"],
                alternative=alternative,
                sidedness=sidedness,
                effect_metric=hypothesis_fields["effect_metric"],
                effect_unit=forecast_effect_unit(test),
                effect_size=record["mean_loss_improvement"],
                effect_interpretation="positive_favors_challenger",
                economic_magnitude_metric="mean_loss_improvement",
                economic_magnitude_value=record["mean_loss_improvement"],
                standard_error=record["standard_error"],
                ci_lower=lower,
                ci_upper=upper,
                confidence_level=confidence_level,
                uncertainty_method=hypothesis_fields["uncertainty_method"],
                sample_size=record["n"],
                raw_p_value=record["p_value"],
                upstream_family_id=f"stage07_prediction__{test}",
                correction_family_id=f"stage07_prediction__{test}",
                correction_method="holm",
                interpretation="prespecified_forecast_benchmark_not_causality",
            )
        )
    return pd.DataFrame(rows)


def classify_weekly_llm_family(family_id: str) -> str:
    if family_id.startswith("primary_"):
        return "prespecified_secondary"
    if family_id.startswith("sensitivity_"):
        return "prespecified_secondary"
    if family_id.startswith("exploratory_"):
        return "exploratory"
    raise ValueError(f"Unregistered Stage 07b family: {family_id}")


def classify_monthly_llm_family(family_id: str) -> str:
    if family_id.startswith("monthly_primary_"):
        return (
            "confirmatory_primary"
            if family_id.endswith("__diebold_mariano_squared")
            else "prespecified_secondary"
        )
    if family_id.startswith("monthly_exploratory_"):
        return "exploratory"
    raise ValueError(f"Unregistered Stage 07c family: {family_id}")


def build_llm_registry(
    frame: pd.DataFrame, monthly: bool, confidence_level: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stage = "stage07c_monthly_llm" if monthly else "stage07b_weekly_llm"
    table = (
        "stage07c_monthly_llm_statistical_tests"
        if monthly
        else "stage07b_weekly_llm_statistical_tests"
    )
    horizon_column = "horizon_months" if monthly else "horizon_weeks"
    for row_number, record in enumerate(frame.to_dict("records")):
        family = str(record["family_id"])
        role = (
            classify_monthly_llm_family(family)
            if monthly
            else classify_weekly_llm_family(family)
        )
        method = normalize_method(record["multiple_testing_method"])
        expected_method = (
            "benjamini_hochberg" if role == "exploratory" else "holm"
        )
        if method != expected_method:
            raise ValueError(f"{table}:{family} correction method changed")
        test = str(record["test"])
        sidedness, alternative = forecast_sidedness(test)
        hypothesis_fields = forecast_hypothesis_fields(test)
        lower, upper = normal_interval(
            record["mean_loss_improvement"], record["standard_error"], confidence_level
        )
        base_construction = (
            record.get("baseline_index_construction") if monthly else "weekly_log_volume"
        )
        challenger_construction = (
            record.get("challenger_index_construction")
            if monthly
            else "weekly_log_volume"
        )
        if monthly:
            index_construction = (
                str(base_construction)
                if base_construction == challenger_construction
                else f"{base_construction}_to_{challenger_construction}"
            )
        else:
            index_construction = "weekly_log_volume"
        if monthly and role == "confirmatory_primary":
            direct_primary = (
                str(challenger_construction) == "adapted_pdf_sqrt_ema3"
                and "construction_pairwise" not in family
            )
            if not direct_primary:
                role = "prespecified_secondary"
        identity = {
            key: record.get(key)
            for key in (
                horizon_column,
                "baseline_model",
                "challenger_model",
                "baseline_index_construction",
                "challenger_index_construction",
                "test",
                "family_id",
            )
            if key in record
        }
        hypothesis_id = f"{stage}__" + stable_hash(stage, identity)
        semantic_key = "forecast__{frequency}__{base_c}__{challenger_c}__{h}__{base}__{challenger}__{test}".format(
            frequency="monthly" if monthly else "weekly",
            base_c=base_construction,
            challenger_c=challenger_construction,
            h=record[horizon_column],
            base=record["baseline_model"],
            challenger=record["challenger_model"],
            test=test,
        )
        rows.append(
            make_registry_row(
                hypothesis_id=hypothesis_id,
                semantic_key=semantic_key,
                source_stage=stage,
                source_table=table,
                source_row_number=row_number,
                evidence_domain=(
                    "monthly_llm_forecast_primary" if monthly else "weekly_llm_forecast_robustness"
                ),
                inference_role=role,
                frequency="monthly" if monthly else "weekly",
                frequency_role="primary" if monthly else "robustness",
                index_construction=index_construction,
                horizon=int(record[horizon_column]),
                horizon_unit="months" if monthly else "weeks",
                baseline_model=record["baseline_model"],
                challenger_model=record["challenger_model"],
                baseline_index_construction=base_construction,
                challenger_index_construction=challenger_construction,
                signal_name=record.get("signal_name"),
                score_source=record.get("score_source"),
                test_method=test,
                null_hypothesis=hypothesis_fields["null_hypothesis"],
                alternative=alternative,
                sidedness=sidedness,
                effect_metric=hypothesis_fields["effect_metric"],
                effect_unit=forecast_effect_unit(test),
                effect_size=record["mean_loss_improvement"],
                effect_interpretation="positive_favors_challenger",
                economic_magnitude_metric="mean_loss_improvement",
                economic_magnitude_value=record["mean_loss_improvement"],
                standard_error=record["standard_error"],
                ci_lower=lower,
                ci_upper=upper,
                confidence_level=confidence_level,
                uncertainty_method=hypothesis_fields["uncertainty_method"],
                sample_size=record["n"],
                raw_p_value=record["p_value"],
                upstream_family_id=family,
                upstream_adjustment_method=method,
                upstream_adjusted_p_value=record["p_value_adjusted"],
                upstream_reconciliation_family_id=f"{stage}__{family}",
                correction_family_id=f"{stage}__{role}__{family}",
                correction_method=method,
                interpretation=(
                    "monthly_primary_predictive_comparison_not_causality"
                    if monthly
                    else "weekly_predictive_comparison_not_causality"
                ),
            )
        )
    return pd.DataFrame(rows)


def build_ablation_registry(
    frame: pd.DataFrame, confidence_level: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_number, record in enumerate(frame.to_dict("records")):
        is_primary = normalize_bool(record.get("primary_family"))
        # Stage 21's primary direct-monthly ablations are a separately fitted,
        # prespecified forecast family. They are confirmatory unless an exact
        # semantic duplicate (same paired sample and numerical effect) is
        # validated below.
        role = "confirmatory_primary" if is_primary else "exploratory"
        method = normalize_method(record["multiplicity_method"])
        expected = "holm" if is_primary else "benjamini_hochberg"
        if method != expected:
            raise ValueError("Stage 09 ablation multiplicity method changed")
        loss = str(record["loss"])
        test = (
            "diebold_mariano_squared"
            if loss == "squared_error"
            else "diebold_mariano_absolute"
        )
        lower, upper = normal_interval(
            record["mean_loss_difference"], record["standard_error_hac"], confidence_level
        )
        if is_primary:
            family = f"stage09_ablation__{record['family_id']}"
        else:
            family = "stage09_ablation__{family}__{frequency}__{construction}__{subset}__{loss}".format(
                family=record["family_id"],
                frequency=record["frequency"],
                construction=record["index_construction"],
                subset=record["sample_subset"],
                loss=loss,
            )
        identity = {
            key: record.get(key)
            for key in (
                "frequency",
                "index_construction",
                "comparison_id",
                "baseline_specification_id",
                "variant_specification_id",
                "sample_subset",
                "loss",
                "horizon",
            )
        }
        hypothesis_id = "stage09_ablation__" + stable_hash(
            "stage09_ablation", identity
        )
        rows.append(
            make_registry_row(
                hypothesis_id=hypothesis_id,
                semantic_key="stage09_ablation__{frequency}__{construction}__{baseline}__{variant}__{h}__{loss}__{subset}".format(
                    frequency=record["frequency"],
                    construction=record["index_construction"],
                    baseline=record["baseline_specification_id"],
                    variant=record["variant_specification_id"],
                    h=record["horizon"],
                    loss=loss,
                    subset=record["sample_subset"],
                ),
                source_stage="stage09_robustness",
                source_table="stage09_ablation_tests",
                source_row_number=row_number,
                evidence_domain="forecast_ablation",
                inference_role=role,
                frequency=record["frequency"],
                frequency_role=record["frequency_role"],
                index_construction=record["index_construction"],
                horizon=int(record["horizon"]),
                horizon_unit="weeks" if record["frequency"] == "weekly" else "months",
                baseline_model=record["baseline_specification_id"],
                challenger_model=record["variant_specification_id"],
                test_method=test,
                null_hypothesis="equal_expected_forecast_loss",
                alternative="unequal_expected_forecast_loss",
                sidedness="two_sided",
                effect_metric="mean_loss_improvement",
                effect_unit=forecast_effect_unit(test),
                effect_size=record["mean_loss_difference"],
                effect_interpretation="positive_favors_challenger",
                economic_magnitude_metric="rmse_improvement_percent",
                economic_magnitude_value=record["rmse_improvement_percent"],
                standard_error=record["standard_error_hac"],
                ci_lower=lower,
                ci_upper=upper,
                confidence_level=confidence_level,
                uncertainty_method="hac_normal_approximation",
                sample_size=record["paired_n"],
                sample_start_date=record["first_origin_date"],
                sample_end_date=record["last_origin_date"],
                raw_p_value=record["dm_hac_p_value"],
                upstream_family_id=record["family_id"],
                upstream_adjustment_method=method,
                upstream_adjusted_p_value=record["dm_hac_p_value_adjusted"],
                upstream_reconciliation_family_id=family,
                correction_family_id=f"{family}__{role}",
                correction_method=method,
                supporting_stage09_hypothesis_id=hypothesis_id,
                supporting_stage09_dm_raw_p_value=record["dm_hac_p_value"],
                supporting_stage09_reassigned_adjusted_p_value=record[
                    "dm_hac_p_value_adjusted"
                ],
                supporting_block_bootstrap_p_value=record[
                    "bootstrap_p_value_two_sided"
                ],
                supporting_block_bootstrap_ci_lower=record["bootstrap_ci_lower"],
                supporting_block_bootstrap_ci_upper=record["bootstrap_ci_upper"],
                supporting_clark_west_one_sided_p_value=record[
                    "clark_west_one_sided_p_value"
                ],
                supporting_rmse_improvement_percent=record[
                    "rmse_improvement_percent"
                ],
                supporting_incremental_oos_r_squared=record[
                    "incremental_oos_r_squared"
                ],
                interpretation="ablation_predictive_comparison_not_causality",
            )
        )
    return pd.DataFrame(rows)


def build_nonlinear_registry(
    frame: pd.DataFrame, confidence_level: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_number, record in enumerate(frame.to_dict("records")):
        method = normalize_method(record["adjustment_method"])
        if method != "holm":
            raise ValueError("Frozen Stage 09 nonlinear families must retain Holm")
        test = str(record["test"])
        sidedness, alternative = forecast_sidedness(test)
        hypothesis_fields = forecast_hypothesis_fields(test)
        lower, upper = normal_interval(
            record["mean_loss_improvement"], record["standard_error"], confidence_level
        )
        identity = {
            key: record.get(key)
            for key in (
                "frequency",
                "algorithm",
                "index_construction",
                "horizon_steps",
                "baseline_model",
                "challenger_model",
                "test",
            )
        }
        hypothesis_id = "stage09_nonlinear__" + stable_hash(
            "stage09_nonlinear", identity
        )
        family = f"stage09_nonlinear__{record['family_id']}"
        rows.append(
            make_registry_row(
                hypothesis_id=hypothesis_id,
                semantic_key="nonlinear__{frequency}__{construction}__{algorithm}__{h}__{baseline}__{challenger}__{test}".format(
                    frequency=record["frequency"],
                    construction=record["index_construction"],
                    algorithm=record["algorithm"],
                    h=record["horizon_steps"],
                    baseline=record["baseline_model"],
                    challenger=record["challenger_model"],
                    test=test,
                ),
                source_stage="stage09_robustness",
                source_table="stage09_nonlinear_statistical_tests",
                source_row_number=row_number,
                evidence_domain="nonlinear_forecast_robustness",
                inference_role="prespecified_secondary",
                frequency=record["frequency"],
                frequency_role=(
                    "primary" if record["frequency"] == "monthly" else "robustness"
                ),
                algorithm=record["algorithm"],
                index_construction=record["index_construction"],
                horizon=int(record["horizon_steps"]),
                horizon_unit=record["horizon_unit"],
                baseline_model=record["baseline_model"],
                challenger_model=record["challenger_model"],
                signal_name=record.get("signal_name"),
                score_source=record.get("score_source"),
                test_method=test,
                null_hypothesis=hypothesis_fields["null_hypothesis"],
                alternative=alternative,
                sidedness=sidedness,
                effect_metric=hypothesis_fields["effect_metric"],
                effect_unit=forecast_effect_unit(test),
                effect_size=record["mean_loss_improvement"],
                effect_interpretation="positive_favors_challenger",
                economic_magnitude_metric="mean_loss_improvement",
                economic_magnitude_value=record["mean_loss_improvement"],
                standard_error=record["standard_error"],
                ci_lower=lower,
                ci_upper=upper,
                confidence_level=confidence_level,
                uncertainty_method=hypothesis_fields["uncertainty_method"],
                sample_size=record["n"],
                raw_p_value=record["p_value"],
                upstream_family_id=record["family_id"],
                upstream_adjustment_method=method,
                upstream_adjusted_p_value=record["p_value_adjusted"],
                upstream_reconciliation_family_id=family,
                correction_family_id=f"{family}__prespecified_secondary",
                correction_method=method,
                interpretation="secondary_predictive_algorithm_robustness",
            )
        )
    return pd.DataFrame(rows)


def raw_handoff_domain(source: str) -> str:
    if source.startswith("forecast_") or source in {
        "error_driver_hac",
        "mincer_zarnowitz_calibration",
    }:
        return "forecast_diagnostics"
    if source.startswith("stage08b_") or source.startswith("stage09_causal") or source.startswith("stage09_future"):
        return "causal_ml_diagnostics"
    return "dynamic_time_series_diagnostics"


def raw_handoff_semantics(source: str, diagnostic_id: str) -> dict[str, Any]:
    """Return explicit hypothesis semantics for every Stage 09b handoff source."""
    common = {
        "signed_effect_interpretable": False,
        "rejection_meaning": "source_specific_diagnostic_warning_if_rejected",
        "claim_direction": "diagnostic_not_directional_claim",
    }
    if source == "forecast_bias_hac":
        return {
            **common,
            "null_hypothesis": "mean_forecast_error_equals_zero",
            "alternative": "mean_forecast_error_differs_from_zero",
            "sidedness": "two_sided",
            "effect_metric": "mean_forecast_error_bias",
            "effect_unit": "scfi_log_change",
            "signed_effect_interpretable": True,
            "rejection_meaning": "systematic_forecast_bias_warning_if_rejected",
        }
    if source == "error_driver_hac":
        return {
            **common,
            "null_hypothesis": "standardized_error_driver_coefficient_equals_zero",
            "alternative": "standardized_error_driver_coefficient_differs_from_zero",
            "sidedness": "two_sided",
            "effect_metric": "standardized_forecast_error_driver_coefficient",
            "effect_unit": "scfi_log_change_per_one_sd_driver",
            "signed_effect_interpretable": True,
            "rejection_meaning": "descriptive_error_driver_association_if_rejected",
        }
    if source == "forecast_residual_adequacy":
        if diagnostic_id.startswith("ljung_box__"):
            null = "no_residual_autocorrelation_through_test_lag"
            alternative = "residual_autocorrelation_present_through_test_lag"
            metric = "ljung_box_statistic"
        elif diagnostic_id.startswith("arch_lm__"):
            null = "no_conditional_heteroskedasticity_through_test_lag"
            alternative = "conditional_heteroskedasticity_present_through_test_lag"
            metric = "arch_lm_statistic"
        elif diagnostic_id.startswith("jarque_bera__"):
            null = "forecast_residuals_are_normally_distributed"
            alternative = "forecast_residuals_depart_from_normality"
            metric = "residual_skewness_reported_with_jarque_bera_test"
        else:
            raise ValueError(f"Unknown residual diagnostic ID: {diagnostic_id}")
        return {
            **common,
            "null_hypothesis": null,
            "alternative": alternative,
            "sidedness": "right_tailed_omnibus",
            "effect_metric": metric,
            "effect_unit": "test_specific_statistic",
            "rejection_meaning": "forecast_residual_adequacy_warning_if_rejected",
        }
    if source == "mincer_zarnowitz_calibration":
        if diagnostic_id.endswith("__intercept_zero"):
            null = "calibration_intercept_equals_zero"
            alternative = "calibration_intercept_differs_from_zero"
            metric = "calibration_intercept"
            signed = True
        elif diagnostic_id.endswith("__slope_one"):
            null = "calibration_slope_equals_one"
            alternative = "calibration_slope_differs_from_one"
            metric = "calibration_slope_minus_one"
            signed = True
        elif diagnostic_id.endswith("__joint"):
            null = "calibration_intercept_equals_zero_and_slope_equals_one"
            alternative = "forecast_calibration_restrictions_fail_jointly"
            metric = "calibration_slope_deviation_reported_with_joint_wald_test"
            signed = False
        else:
            raise ValueError(f"Unknown calibration diagnostic ID: {diagnostic_id}")
        return {
            **common,
            "null_hypothesis": null,
            "alternative": alternative,
            "sidedness": "two_sided" if signed else "right_tailed_omnibus",
            "effect_metric": metric,
            "effect_unit": "scfi_log_change" if "intercept" in metric else "slope_units",
            "signed_effect_interpretable": signed,
            "rejection_meaning": "forecast_miscalibration_warning_if_rejected",
        }
    if source == "forecast_stability":
        if diagnostic_id.startswith("cusum__"):
            null = "forecast_errors_are_parameter_stable"
            alternative = "forecast_errors_exhibit_structural_instability"
            metric = "maximum_absolute_cusum_or_cusum_statistic"
        elif diagnostic_id.startswith("time_trend__"):
            null = "forecast_error_time_trend_equals_zero"
            alternative = "forecast_error_time_trend_differs_from_zero"
            metric = "forecast_error_time_trend_coefficient"
        elif diagnostic_id.startswith("operational_regime__"):
            null = "forecast_error_means_are_equal_across_operational_regimes"
            alternative = "forecast_error_mean_differs_across_operational_regimes"
            metric = "forecast_error_regime_mean_range"
        else:
            raise ValueError(f"Unknown stability diagnostic ID: {diagnostic_id}")
        return {
            **common,
            "null_hypothesis": null,
            "alternative": alternative,
            "sidedness": "right_tailed_omnibus",
            "effect_metric": metric,
            "effect_unit": "test_specific_stability_units",
            "rejection_meaning": "forecast_instability_warning_if_rejected",
        }
    if source in {
        "stage08b_nuisance_outcome_calibration",
        "stage08b_nuisance_treatment_calibration",
    }:
        nuisance = "outcome" if "outcome" in source else "treatment"
        return {
            **common,
            "null_hypothesis": f"{nuisance}_nuisance_intercept_zero_and_slope_one",
            "alternative": f"{nuisance}_nuisance_calibration_restrictions_fail_jointly",
            "sidedness": "right_tailed_omnibus",
            "effect_metric": f"{nuisance}_nuisance_calibration_slope",
            "effect_unit": "calibration_slope_units",
            "rejection_meaning": "nuisance_model_miscalibration_warning_if_rejected",
        }
    if source == "varx_diagnostics_sensitivity":
        return {
            **common,
            "null_hypothesis": "varx_residual_adequacy_restriction_holds",
            "alternative": "varx_residual_adequacy_restriction_fails",
            "sidedness": "right_tailed_omnibus",
            "effect_metric": "varx_diagnostic_statistic",
            "effect_unit": "test_specific_statistic",
            "rejection_meaning": "varx_adequacy_warning_if_rejected",
        }
    if source in {"conditional_granger_sensitivity", "conventional_granger_sensitivity"}:
        return {
            **common,
            "null_hypothesis": "no_granger_predictive_precedence",
            "alternative": "granger_predictive_precedence_present",
            "sidedness": "right_tailed_omnibus",
            "effect_metric": "wald_statistic" if source.startswith("conditional") else "f_statistic",
            "effect_unit": "test_statistic",
            "rejection_meaning": "predictive_precedence_evidence_if_rejected_not_causality",
        }
    if source == "local_projection_sensitivity":
        return {
            **common,
            "null_hypothesis": "local_projection_coefficient_equals_zero",
            "alternative": "local_projection_coefficient_differs_from_zero",
            "sidedness": "two_sided",
            "effect_metric": "local_projection_coefficient_log_points",
            "effect_unit": "scfi_log_points_per_one_sd_msdi_innovation",
            "signed_effect_interpretable": True,
            "rejection_meaning": "dynamic_predictive_association_if_rejected_not_causality",
        }
    if source == "pcmci_sensitivity_links":
        return {
            **common,
            "null_hypothesis": "conditional_temporal_partial_correlation_equals_zero",
            "alternative": "conditional_temporal_partial_correlation_differs_from_zero",
            "sidedness": "two_sided",
            "effect_metric": "partial_correlation",
            "effect_unit": "correlation_units",
            "signed_effect_interpretable": True,
            "rejection_meaning": "conditional_temporal_dependence_if_rejected_not_causality",
        }
    if source in {
        "stage08b_placebo_tests",
        "stage09_future_lead_placebos",
    }:
        return {
            **common,
            "null_hypothesis": "placebo_association_equals_zero",
            "alternative": "placebo_association_differs_from_zero",
            "sidedness": "two_sided",
            "effect_metric": "effect_log_points_per_one_sd_innovation",
            "effect_unit": "scfi_log_points_per_one_sd_treatment_innovation",
            "signed_effect_interpretable": True,
            "rejection_meaning": "falsification_warning_if_rejected",
        }
    if source == "stage09_causal_ml_sensitivity":
        return {
            **common,
            "null_hypothesis": "orthogonalized_predictive_effect_equals_zero",
            "alternative": "orthogonalized_predictive_effect_differs_from_zero",
            "sidedness": "two_sided",
            "effect_metric": "effect_log_points_per_one_sd_innovation",
            "effect_unit": "scfi_log_points_per_one_sd_treatment_innovation",
            "signed_effect_interpretable": True,
            "rejection_meaning": "orthogonalized_predictive_association_if_rejected_not_causality",
        }
    raise ValueError(f"Unregistered Stage 09b diagnostic source: {source}")


def build_raw_handoff_registry(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_number, record in enumerate(frame.to_dict("records")):
        if normalize_bool(record.get("p_value_adjusted_in_stage09b")):
            raise ValueError("Stage 09b handoff contains adjusted p-values")
        if normalize_bool(record.get("causal_claim_allowed")):
            raise ValueError("Stage 09b handoff authorizes a causal claim")
        is_primary = normalize_bool(record.get("primary_family"))
        role = "primary_diagnostic" if is_primary else "exploratory"
        method = "holm" if is_primary else "benjamini_hochberg"
        family = f"stage09b__{role}__{record['hypothesis_family_id']}"
        hypothesis_id = f"stage09b__{record['diagnostic_id']}"
        semantics = raw_handoff_semantics(
            str(record["diagnostic_source"]), str(record["diagnostic_id"])
        )
        rows.append(
            make_registry_row(
                hypothesis_id=hypothesis_id,
                semantic_key=f"stage09b__{record['diagnostic_id']}",
                source_stage="stage09b_error_diagnostics",
                source_table="stage09b_raw_significance_handoff",
                source_row_number=row_number,
                evidence_domain=raw_handoff_domain(str(record["diagnostic_source"])),
                inference_role=role,
                frequency=record.get("frequency"),
                frequency_role=(
                    "primary"
                    if record.get("frequency") == "monthly"
                    else "robustness"
                    if record.get("frequency") == "weekly"
                    else "source_defined"
                ),
                index_construction=record.get("index_construction"),
                horizon=record.get("horizon"),
                horizon_unit=(
                    "weeks"
                    if record.get("frequency") == "weekly"
                    else "months"
                    if record.get("frequency") == "monthly"
                    else None
                ),
                model_id=record.get("model_id"),
                diagnostic_source=record.get("diagnostic_source"),
                test_method=record.get("diagnostic_source"),
                null_hypothesis=semantics["null_hypothesis"],
                alternative=semantics["alternative"],
                sidedness=semantics["sidedness"],
                effect_metric=semantics["effect_metric"],
                effect_unit=semantics["effect_unit"],
                effect_size=record.get("effect_size"),
                effect_interpretation="source_specific_diagnostic_effect",
                signed_effect_interpretable=semantics["signed_effect_interpretable"],
                rejection_meaning=semantics["rejection_meaning"],
                claim_direction=semantics["claim_direction"],
                economic_magnitude_metric=record.get("diagnostic_source"),
                economic_magnitude_value=record.get("effect_size"),
                uncertainty_method="upstream_test_specific",
                ci_unavailable_reason="raw_handoff_has_no_compatible_interval",
                sample_size=record.get("sample_size"),
                raw_p_value=record.get("raw_p_value"),
                upstream_family_id=record.get("hypothesis_family_id"),
                correction_family_id=family,
                correction_method=method,
                interpretation=record.get("interpretation"),
            )
        )
    return pd.DataFrame(rows)


def enrich_forecast_handoff_uncertainty(
    registry: pd.DataFrame,
    forecast_summary: pd.DataFrame,
    calibration: pd.DataFrame,
    error_drivers: pd.DataFrame,
    confidence_level: float,
) -> pd.DataFrame:
    """Restore sample windows and compatible forecast-diagnostic uncertainty."""
    result = registry.copy()
    hypothesis_index = unique_hypothesis_index(result)
    cell_keys = ["frequency", "index_construction", "horizon", "model_id"]
    if forecast_summary.duplicated(cell_keys).any():
        raise ValueError("Forecast-summary cell keys are not unique")
    cell_lookup = forecast_summary.set_index(cell_keys)
    forecast_indices = result.index[
        result["source_table"].eq("stage09b_raw_significance_handoff")
        & result["evidence_domain"].eq("forecast_diagnostics")
    ]
    missing_cells: list[str] = []
    for index in forecast_indices:
        key = tuple(result.at[index, column] for column in cell_keys)
        if key not in cell_lookup.index:
            missing_cells.append(str(result.at[index, "hypothesis_id"]))
            continue
        detail = cell_lookup.loc[key]
        result.at[index, "sample_start_date"] = detail["sample_start_date"]
        result.at[index, "sample_end_date"] = detail["sample_end_date"]
        result.at[index, "sample_dates_unavailable_reason"] = None
    if missing_cells:
        raise ValueError(
            "Forecast diagnostic rows lack summary sample windows: "
            + ", ".join(missing_cells[:5])
        )

    bias_count = 0
    for record in forecast_summary.to_dict("records"):
        diagnostic_id = "forecast_bias__" + "__".join(
            str(record[column])
            for column in (
                "analysis_source",
                "frequency",
                "index_construction",
                "horizon",
                "model_id",
            )
        )
        hypothesis_id = f"stage09b__{diagnostic_id}"
        position = hypothesis_index.get(hypothesis_id)
        if position is None:
            raise ValueError(f"Forecast-bias handoff ID mismatch: {diagnostic_id}")
        if not math.isclose(
            float(result.at[position, "effect_size"]),
            float(record["bias"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"Forecast-bias effect changed: {diagnostic_id}")
        result.at[position, "standard_error"] = record["bias_hac_standard_error"]
        result.at[position, "ci_lower"] = record["bias_bootstrap_ci_lower"]
        result.at[position, "ci_upper"] = record["bias_bootstrap_ci_upper"]
        result.at[position, "confidence_level"] = confidence_level
        result.at[position, "uncertainty_method"] = (
            "moving_block_bootstrap_bias_interval_with_hac_standard_error"
        )
        result.at[position, "ci_available"] = True
        result.at[position, "ci_unavailable_reason"] = None
        result.at[position, "bootstrap_repetitions_successful"] = record.get(
            "bootstrap_repetitions_successful"
        )
        bias_count += 1
    if bias_count != len(forecast_summary):
        raise ValueError("Forecast-bias enrichment count changed")

    calibration_count = 0
    for record in calibration.to_dict("records"):
        for suffix, se_column, lower_column, upper_column, shift in (
            (
                "intercept_zero",
                "intercept_hac_standard_error",
                "intercept_ci_lower",
                "intercept_ci_upper",
                0.0,
            ),
            (
                "slope_one",
                "slope_hac_standard_error",
                "slope_ci_lower",
                "slope_ci_upper",
                1.0,
            ),
        ):
            hypothesis_id = f"stage09b__{record['diagnostic_id']}__{suffix}"
            position = hypothesis_index.get(hypothesis_id)
            if position is None:
                raise ValueError(f"Calibration handoff ID mismatch: {hypothesis_id}")
            result.at[position, "standard_error"] = record[se_column]
            result.at[position, "ci_lower"] = float(record[lower_column]) - shift
            result.at[position, "ci_upper"] = float(record[upper_column]) - shift
            result.at[position, "confidence_level"] = confidence_level
            result.at[position, "uncertainty_method"] = (
                "hac_normal_pointwise_upstream_critical_1_96"
            )
            result.at[position, "ci_available"] = True
            result.at[position, "ci_unavailable_reason"] = None
            calibration_count += 1
    expected_calibration = 2 * len(calibration)
    if calibration_count != expected_calibration:
        raise ValueError("Calibration uncertainty enrichment count changed")

    driver_count = 0
    used_drivers = error_drivers.loc[error_drivers["feature_status"].eq("used")]
    for record in used_drivers.to_dict("records"):
        hypothesis_id = f"stage09b__{record['diagnostic_id']}"
        position = hypothesis_index.get(hypothesis_id)
        if position is None:
            raise ValueError(f"Error-driver handoff ID mismatch: {hypothesis_id}")
        lower, upper = normal_interval(
            record["coefficient"], record["hac_standard_error"], confidence_level
        )
        result.at[position, "standard_error"] = record["hac_standard_error"]
        result.at[position, "ci_lower"] = lower
        result.at[position, "ci_upper"] = upper
        result.at[position, "confidence_level"] = confidence_level
        result.at[position, "uncertainty_method"] = (
            "hac_normal_reconstructed_from_exported_standard_error"
        )
        result.at[position, "ci_available"] = True
        result.at[position, "ci_unavailable_reason"] = None
        driver_count += 1
    if driver_count != len(used_drivers):
        raise ValueError("Error-driver uncertainty enrichment count changed")
    return result


def enrich_dml_handoff_uncertainty(
    registry: pd.DataFrame,
    consolidation: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Restore DML uncertainty and the earliest source-global BH families."""
    result = registry.copy()
    for column in (
        "frequency_role",
        "index_construction",
        "index_construction_unavailable_reason",
        "msdi_path",
        "learner",
        "model_id",
        "horizon_unit",
        "effect_metric",
        "effect_unit",
        "effect_interpretation",
        "economic_magnitude_metric",
        "uncertainty_method",
        "sample_start_date",
        "sample_end_date",
        "sample_dates_unavailable_reason",
        "upstream_family_id",
        "upstream_adjustment_method",
        "upstream_reconciliation_family_id",
        "correction_family_id",
        "correction_method",
    ):
        result[column] = result[column].astype("object")
    hypothesis_index = unique_hypothesis_index(result)
    expected_counts = config["registry_sources"][
        "stage09b_dml_handoff_enrichment"
    ]["expected_enriched_rows"]
    date_source = consolidation.loc[
        consolidation["diagnostic_source"].eq("stage08b_dml_estimates")
    ].copy()
    date_keys = ["frequency", "msdi_path", "horizon", "learner"]
    if date_source.duplicated(date_keys).any():
        raise ValueError("Stage 08b DML date lookup is not unique")
    date_lookup = date_source.set_index(date_keys)
    audit_columns = (
        "coefficient_per_raw_treatment_unit",
        "raw_standard_error_hac",
        "effect_scale_frozen_calibration_residual_sd",
        "orthogonal_residual_correlation",
        "orthogonal_partial_r_squared",
        "bootstrap_repetitions_requested",
        "bootstrap_repetitions_successful",
        "bootstrap_block_length",
        "p_value_holm_within_frequency_learner",
        "p_value_holm_primary_family",
        "primary_family",
        "simultaneous_lower_log_points",
        "simultaneous_upper_log_points",
        "simultaneous_family_id",
        "simultaneous_band_available",
        "simultaneous_band_significant",
        "placebo_outcome",
        "expected_result",
        "falsification_flag_raw",
        "falsification_flag_fdr",
        "falsification_status",
        "future_treatment_lead",
        "lead_strictly_beyond_outcome",
        "sensitivity_flag",
        "cohort_scope",
        "interpretation_guardrail",
        "effect_scale_definition",
    )
    for column in audit_columns:
        result[f"dml_audit__{column}"] = pd.Series(
            [None] * len(result), index=result.index, dtype="object"
        )
    for source, expected in expected_counts.items():
        tagged = consolidation.loc[consolidation["diagnostic_source"].eq(source)]
        if len(tagged) != int(expected):
            raise ValueError(
                f"DML consolidation count changed for {source}: {len(tagged)} != {expected}"
            )
        provenance = tagged[
            ["consolidation_source_path", "consolidation_source_sha256"]
        ].drop_duplicates()
        if len(provenance) != 1:
            raise ValueError(f"DML source provenance is not unique: {source}")
        source_path = Path(str(provenance.iloc[0]["consolidation_source_path"])).resolve()
        expected_hash = str(provenance.iloc[0]["consolidation_source_sha256"])
        if not source_path.is_file() or sha256_file(source_path) != expected_hash:
            raise ValueError(f"DML source hash changed: {source}")
        original = pd.read_csv(source_path, low_memory=False)
        if len(original) != int(expected):
            raise ValueError(f"DML source row count changed: {source}")
        expected_ids: set[str] = set()
        for record in original.to_dict("records"):
            hypothesis_id = (
                f"stage09b__{source}__{stable_dynamic_hash(source, record)}"
            )
            expected_ids.add(hypothesis_id)
            position = hypothesis_index.get(hypothesis_id)
            if position is None:
                raise ValueError(f"DML handoff ID mismatch: {hypothesis_id}")
            for source_column, registry_column in (
                ("p_value", "raw_p_value"),
                ("effect_log_points_per_one_sd_innovation", "effect_size"),
                ("n_observations", "sample_size"),
            ):
                left = float(record[source_column])
                right = float(result.at[position, registry_column])
                if not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12):
                    raise ValueError(
                        f"DML source/handoff {source_column} mismatch: {hypothesis_id}"
                    )
            if "causal_claim_allowed" in record and normalize_bool(
                record.get("causal_claim_allowed")
            ):
                raise ValueError(f"DML source authorizes causal claim: {hypothesis_id}")
            frequency = str(record["frequency"])
            result.at[position, "frequency_role"] = record.get("frequency_role")
            result.at[position, "index_construction"] = (
                "weekly_log_volume" if frequency == "weekly" else "adapted_pdf_sqrt_ema3"
            )
            result.at[position, "index_construction_unavailable_reason"] = None
            result.at[position, "msdi_path"] = record.get("msdi_path")
            result.at[position, "learner"] = record.get("learner")
            result.at[position, "model_id"] = record.get("learner")
            result.at[position, "horizon_unit"] = record.get(
                "horizon_unit", "weeks" if frequency == "weekly" else "months"
            )
            result.at[position, "effect_metric"] = (
                "effect_log_points_per_one_sd_innovation"
            )
            result.at[position, "effect_unit"] = (
                "scfi_log_points_per_one_sd_treatment_innovation"
            )
            result.at[position, "effect_interpretation"] = (
                "signed_orthogonalized_predictive_effect_not_causal"
            )
            result.at[position, "economic_magnitude_metric"] = (
                "effect_percent_per_one_sd_innovation"
            )
            result.at[position, "economic_magnitude_value"] = record.get(
                "effect_percent_per_one_sd_innovation"
            )
            result.at[position, "standard_error"] = record.get("standard_error_hac")
            result.at[position, "ci_lower"] = record.get("ci_lower_log_points")
            result.at[position, "ci_upper"] = record.get("ci_upper_log_points")
            result.at[position, "confidence_level"] = record.get(
                "confidence_level", 0.95
            )
            result.at[position, "uncertainty_method"] = "hac_normal_pointwise"
            result.at[position, "ci_available"] = True
            result.at[position, "ci_unavailable_reason"] = None
            result.at[position, "supporting_block_bootstrap_ci_lower"] = record.get(
                "bootstrap_ci_lower_log_points"
            )
            result.at[position, "supporting_block_bootstrap_ci_upper"] = record.get(
                "bootstrap_ci_upper_log_points"
            )
            sample_start = record.get("sample_start_date")
            sample_end = record.get("sample_end_date")
            if source == "stage08b_placebo_tests" and (
                pd.isna(sample_start) or pd.isna(sample_end)
            ):
                date_key = tuple(record[column] for column in date_keys)
                if date_key not in date_lookup.index:
                    raise ValueError(f"DML placebo sample dates unavailable: {hypothesis_id}")
                date_row = date_lookup.loc[date_key]
                sample_start = date_row["sample_start_date"]
                sample_end = date_row["sample_end_date"]
            result.at[position, "sample_start_date"] = sample_start
            result.at[position, "sample_end_date"] = sample_end
            result.at[position, "sample_dates_unavailable_reason"] = None
            family = f"stage09b__exploratory__{source}__all_rows_fdr"
            result.at[position, "upstream_family_id"] = f"{source}__all_rows_fdr"
            result.at[position, "upstream_adjustment_method"] = (
                "benjamini_hochberg"
            )
            result.at[position, "upstream_adjusted_p_value"] = record.get(
                "p_value_fdr_all"
            )
            result.at[position, "upstream_reconciliation_family_id"] = family
            result.at[position, "correction_family_id"] = family
            result.at[position, "correction_method"] = "benjamini_hochberg"
            result.at[position, "stage09b_frequency_family_id"] = (
                f"stage09b__exploratory__{source}__{frequency}"
            )
            for column in audit_columns:
                if column in record:
                    result.at[position, f"dml_audit__{column}"] = record.get(column)
        observed_ids = set(
            result.loc[result["diagnostic_source"].eq(source), "hypothesis_id"]
        )
        if expected_ids != observed_ids:
            raise ValueError(f"DML source/handoff ID sets differ: {source}")
    return result


def audit_dml_source_contracts(
    consolidation: pd.DataFrame, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_counts = config["registry_sources"][
        "stage09b_dml_handoff_enrichment"
    ]["expected_enriched_rows"]
    for source, expected in expected_counts.items():
        tagged = consolidation.loc[consolidation["diagnostic_source"].eq(source)]
        provenance = tagged[
            ["consolidation_source_path", "consolidation_source_sha256"]
        ].drop_duplicates()
        if len(tagged) != int(expected) or len(provenance) != 1:
            raise ValueError(f"DML source provenance contract changed: {source}")
        path = Path(str(provenance.iloc[0]["consolidation_source_path"])).resolve()
        expected_hash = str(provenance.iloc[0]["consolidation_source_sha256"])
        actual_hash = sha256_file(path) if path.is_file() else None
        actual_rows = len(pd.read_csv(path, low_memory=False)) if path.is_file() else 0
        passed = actual_hash == expected_hash and actual_rows == int(expected)
        rows.append(
            {
                "source_stage": "stage09b_dml_source_provenance",
                "artifact": source,
                "path": str(path),
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "expected_rows": int(expected),
                "actual_rows": actual_rows,
                "contract_passed": passed,
                "details": "original DML source frozen by audited consolidation",
            }
        )
        if not passed:
            raise ValueError(f"DML original source contract failed: {source}")
    return rows


def build_peak_breadth_registry(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_number, record in enumerate(frame.to_dict("records")):
        family = f"stage09_peak_breadth_difference__{record['frequency']}"
        identity = {"frequency": record["frequency"], "horizon": record["horizon"]}
        hypothesis_id = "stage09_peak_breadth__" + stable_hash(
            "stage09_peak_breadth", identity
        )
        rows.append(
            make_registry_row(
                hypothesis_id=hypothesis_id,
                semantic_key=f"peak_minus_breadth__{record['frequency']}__h{record['horizon']}",
                source_stage="stage09_robustness",
                source_table="stage09_peak_breadth_difference",
                source_row_number=row_number,
                evidence_domain="dynamic_path_comparison",
                inference_role="exploratory",
                frequency=record["frequency"],
                frequency_role="primary" if record["frequency"] == "monthly" else "robustness",
                index_construction=(
                    "weekly_log_volume" if record["frequency"] == "weekly" else "adapted_pdf_sqrt_ema3"
                ),
                horizon=int(record["horizon"]),
                horizon_unit="weeks" if record["frequency"] == "weekly" else "months",
                msdi_path="peak_minus_breadth",
                test_method="shared_moving_block_bootstrap_two_sided",
                null_hypothesis="peak_and_breadth_dynamic_coefficients_equal",
                alternative="peak_and_breadth_dynamic_coefficients_differ",
                sidedness="two_sided",
                effect_metric="coefficient_difference_peak_minus_breadth",
                effect_unit="scfi_log_points",
                effect_size=record["coefficient_difference_peak_minus_breadth"],
                effect_interpretation="positive_peak_effect_exceeds_breadth_effect",
                economic_magnitude_metric="coefficient_difference_log_points",
                economic_magnitude_value=record[
                    "coefficient_difference_peak_minus_breadth"
                ],
                standard_error=record["shared_bootstrap_standard_error"],
                ci_lower=record["shared_bootstrap_ci_lower"],
                ci_upper=record["shared_bootstrap_ci_upper"],
                confidence_level=0.95,
                uncertainty_method="shared_moving_block_bootstrap",
                sample_size=record["n_common_system_rows"],
                raw_p_value=record["shared_bootstrap_p_value"],
                upstream_family_id=family,
                upstream_adjustment_method="benjamini_hochberg",
                upstream_adjusted_p_value=record["p_value_fdr_bh"],
                upstream_reconciliation_family_id=family,
                correction_family_id=f"{family}__exploratory",
                correction_method="benjamini_hochberg",
                interpretation=record["interpretation"],
            )
        )
    return pd.DataFrame(rows)


def build_ardl_registry(frame: pd.DataFrame) -> pd.DataFrame:
    mappings = {
        "scfi_error_correction_p_value": "scfi_error_correction_coefficient",
        "msdi_lagged_level_p_value": "msdi_lagged_level_coefficient",
    }
    rows: list[dict[str, Any]] = []
    for row_number, record in enumerate(frame.to_dict("records")):
        for p_column, effect_column in mappings.items():
            parameter = p_column.removesuffix("_p_value")
            identity = {
                "frequency": record["frequency"],
                "msdi_path": record["msdi_path"],
                "lag_count": record["lag_count"],
                "deterministic_terms": record["deterministic_terms"],
                "parameter": parameter,
            }
            hypothesis_id = "stage09_ardl__" + stable_hash("stage09_ardl", identity)
            rows.append(
                make_registry_row(
                    hypothesis_id=hypothesis_id,
                    semantic_key="ardl__{frequency}__{path}__lag{lag}__{terms}__{parameter}".format(
                        frequency=record["frequency"],
                        path=record["msdi_path"],
                        lag=record["lag_count"],
                        terms=record["deterministic_terms"],
                        parameter=parameter,
                    ),
                    source_stage="stage09_robustness",
                    source_table="stage09_ardl_ecm_sensitivity",
                    source_row_number=row_number,
                    evidence_domain="dynamic_time_series_sensitivity",
                    inference_role="exploratory",
                    frequency=record["frequency"],
                    frequency_role="primary" if record["frequency"] == "monthly" else "robustness",
                    index_construction=(
                        "weekly_log_volume"
                        if record["frequency"] == "weekly"
                        else "adapted_pdf_sqrt_ema3"
                    ),
                    msdi_path=record["msdi_path"],
                    selected_lag=record["lag_count"],
                    term=parameter,
                    test_method="ardl_ecm_coefficient_test",
                    null_hypothesis=f"{parameter}_equals_zero",
                    alternative=f"{parameter}_differs_from_zero",
                    sidedness="two_sided",
                    effect_metric=parameter,
                    effect_unit="source_native_coefficient_units",
                    effect_size=record[effect_column],
                    effect_interpretation="signed_conditional_levels_association",
                    economic_magnitude_metric=parameter,
                    economic_magnitude_value=record[effect_column],
                    uncertainty_method="upstream_coefficient_test",
                    ci_unavailable_reason="individual_coefficient_standard_error_not_exported",
                    sample_size=record["n_observations"],
                    sample_start_date=record["sample_start_date"],
                    sample_end_date=record["sample_end_date"],
                    raw_p_value=record[p_column],
                    upstream_family_id=f"stage09_ardl__{parameter}",
                    correction_family_id=f"stage09_ardl__exploratory__{parameter}",
                    correction_method="benjamini_hochberg",
                    interpretation=record["interpretation"],
                )
            )
    return pd.DataFrame(rows)


def build_varx_coefficient_registry(
    frame: pd.DataFrame, confidence_level: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_number, record in enumerate(frame.to_dict("records")):
        identity = {
            key: record.get(key)
            for key in (
                "frequency",
                "msdi_path",
                "selected_lag",
                "equation",
                "term",
                "maximum_lag_specification",
            )
        }
        hypothesis_id = "stage09_varx_coefficient__" + stable_hash(
            "stage09_varx_coefficient", identity
        )
        semantic_key = "varx_coefficient__{frequency}__{path}__lag{lag}__{equation}__{term}".format(
            frequency=record["frequency"],
            path=record["msdi_path"],
            lag=record["selected_lag"],
            equation=record["equation"],
            term=record["term"],
        )
        lower, upper = normal_interval(
            record["coefficient"], record["standard_error"], confidence_level
        )
        family = "stage09_varx_coefficient__{frequency}__{path}__lag{lag}".format(
            frequency=record["frequency"],
            path=record["msdi_path"],
            lag=record["selected_lag"],
        )
        rows.append(
            make_registry_row(
                hypothesis_id=hypothesis_id,
                semantic_key=semantic_key,
                source_stage="stage09_robustness",
                source_table="stage09_varx_coefficients_sensitivity",
                source_row_number=row_number,
                evidence_domain="dynamic_time_series_sensitivity",
                inference_role="exploratory",
                frequency=record["frequency"],
                frequency_role="primary" if record["frequency"] == "monthly" else "robustness",
                index_construction=(
                    "weekly_log_volume"
                    if record["frequency"] == "weekly"
                    else "adapted_pdf_sqrt_ema3"
                ),
                msdi_path=record["msdi_path"],
                maximum_lag_specification=record["maximum_lag_specification"],
                selected_lag=record["selected_lag"],
                equation=record["equation"],
                term=record["term"],
                test_method="varx_coefficient_wald_z",
                null_hypothesis="coefficient_equals_zero",
                alternative="coefficient_differs_from_zero",
                sidedness="two_sided",
                effect_metric="varx_coefficient",
                effect_unit="equation_native_units",
                effect_size=record["coefficient"],
                effect_interpretation="signed_dynamic_predictive_association",
                economic_magnitude_metric="varx_coefficient",
                economic_magnitude_value=record["coefficient"],
                standard_error=record["standard_error"],
                ci_lower=lower,
                ci_upper=upper,
                confidence_level=confidence_level,
                uncertainty_method="asymptotic_normal",
                sample_size=record["n_observations"],
                raw_p_value=record["p_value"],
                upstream_family_id=family,
                correction_family_id=f"{family}__exploratory",
                correction_method="benjamini_hochberg",
                interpretation="varx_predictive_coefficient_not_structural_causality",
            )
        )
    return pd.DataFrame(rows)


def enrich_dynamic_handoff_families(
    registry: pd.DataFrame, detail_tables: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    result = registry.copy()
    hypothesis_index = unique_hypothesis_index(result)
    mappings: dict[str, dict[str, Any]] = {}
    for source, frame in detail_tables.items():
        for record in frame.to_dict("records"):
            p_value = pd.to_numeric(pd.Series([record.get("p_value")]), errors="coerce").iloc[0]
            if not np.isfinite(p_value):
                continue
            diagnostic_id = f"{source}__{stable_dynamic_hash(source, record)}"
            full_id = f"stage09b__{diagnostic_id}"
            upstream_adjusted = np.nan
            upstream_method = None
            upstream_reconciliation_family = None
            correction_family = None
            if source == "conditional_granger_sensitivity":
                detail_family = "stage09b__exploratory__conditional_granger__{frequency}__{path}__{causing}_to_{caused}".format(
                    frequency=record["frequency"],
                    path=record["msdi_path"],
                    causing=record["causing"],
                    caused=record["caused"],
                )
                upstream_adjusted = record.get("p_value_fdr_bh")
                upstream_method = "benjamini_hochberg"
                upstream_reconciliation_family = detail_family
                correction_family = detail_family
            elif source == "conventional_granger_sensitivity":
                detail_family = f"stage09b__exploratory__{record['family_id']}"
                upstream_adjusted = record.get("p_value_fdr_bh")
                upstream_method = "benjamini_hochberg"
                upstream_reconciliation_family = detail_family
                correction_family = detail_family
            elif source == "local_projection_sensitivity":
                detail_family = "stage09b__audit_only__local_projection__{frequency}__{path}__lag{lag}".format(
                    frequency=record["frequency"],
                    path=record["msdi_path"],
                    lag=record["sensitivity_lag_count"],
                )
            elif source == "pcmci_sensitivity_links":
                detail_family = f"stage09b__exploratory__pcmci__{record['sensitivity_specification_id']}"
                upstream_adjusted = record.get("p_value_fdr_bh")
                upstream_method = "benjamini_hochberg"
                upstream_reconciliation_family = detail_family
                correction_family = detail_family
            elif source == "varx_diagnostics_sensitivity":
                detail_family = "stage09b__audit_only__varx_diagnostic__{frequency}__{path}__lag{lag}".format(
                    frequency=record["frequency"],
                    path=record["msdi_path"],
                    lag=record["selected_lag"],
                )
            else:
                raise ValueError(f"Unsupported dynamic detail source: {source}")
            mappings[full_id] = {
                "_detail_raw_p_value": p_value,
                "_detail_effect_size": next(
                    record.get(column)
                    for column in (
                        "wald_statistic",
                        "f_statistic",
                        "coefficient_log_points",
                        "partial_correlation",
                        "statistic",
                    )
                    if column in record and not is_missing_scalar(record.get(column))
                ),
                "_detail_sample_size": record.get(
                    "n_observations", record.get("n_observations_before_lagging")
                ),
                "dynamic_detail_family_id": detail_family,
                "sample_size": record.get(
                    "n_observations", record.get("n_observations_before_lagging")
                ),
                "sample_start_date": record.get("sample_start_date"),
                "sample_end_date": record.get("sample_end_date"),
                "msdi_path": record.get("msdi_path"),
                "maximum_lag_specification": record.get(
                    "maximum_lag_specification"
                ),
                "selected_lag": record.get("selected_lag"),
                "diagnostic": record.get("diagnostic"),
                "variable": record.get("variable"),
                "dynamic_details": record.get("details"),
                "standard_error": record.get("standard_error_hac"),
                "ci_lower": record.get("ci_lower_log_points"),
                "ci_upper": record.get("ci_upper_log_points"),
                "uncertainty_method": (
                    "hac_normal_interval"
                    if source == "local_projection_sensitivity"
                    else "upstream_test_specific"
                ),
                "confidence_level": record.get("confidence_level"),
                "upstream_adjusted_p_value": upstream_adjusted,
                "upstream_adjustment_method": upstream_method,
                "upstream_reconciliation_family_id": upstream_reconciliation_family,
                "correction_family_id": correction_family,
                "correction_method": (
                    "benjamini_hochberg" if correction_family else None
                ),
            }
    source_ids = set(mappings)
    registry_ids = set(
        result.loc[
            result["source_table"].eq("stage09b_raw_significance_handoff"),
            "hypothesis_id",
        ]
    )
    missing = sorted(source_ids.difference(registry_ids))
    if missing:
        raise ValueError(f"Dynamic enrichment IDs missing from 09b handoff: {missing[:5]}")
    for hypothesis_id, values in mappings.items():
        position = hypothesis_index.get(hypothesis_id)
        if position is None:
            raise ValueError(f"Dynamic handoff ID is not unique: {hypothesis_id}")
        for registry_column, detail_column in (
            ("raw_p_value", "_detail_raw_p_value"),
            ("effect_size", "_detail_effect_size"),
        ):
            handoff_value = float(result.at[position, registry_column])
            detail_value = float(values[detail_column])
            if not math.isclose(
                handoff_value, detail_value, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError(
                    f"Dynamic detail/handoff {registry_column} mismatch: {hypothesis_id}"
                )
        handoff_sample = pd.to_numeric(
            pd.Series([result.at[position, "sample_size"]]), errors="coerce"
        ).iloc[0]
        detail_sample = pd.to_numeric(
            pd.Series([values["_detail_sample_size"]]), errors="coerce"
        ).iloc[0]
        if np.isfinite(handoff_sample) and (
            not np.isfinite(detail_sample)
            or not math.isclose(
                float(handoff_sample),
                float(detail_sample),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                f"Dynamic detail/handoff sample_size mismatch: {hypothesis_id}"
            )
        for column, value in values.items():
            if column.startswith("_detail_"):
                continue
            if column not in result:
                result[column] = None
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                result.at[position, column] = value
        frequency = str(result.at[position, "frequency"])
        result.at[position, "index_construction"] = (
            "weekly_log_volume" if frequency == "weekly" else "adapted_pdf_sqrt_ema3"
        )
        result.at[position, "index_construction_unavailable_reason"] = None
        if not is_missing_scalar(
            result.at[position, "sample_start_date"]
        ) and not is_missing_scalar(result.at[position, "sample_end_date"]):
            result.at[position, "sample_dates_unavailable_reason"] = None
        lower = pd.to_numeric(pd.Series([result.at[position, "ci_lower"]]), errors="coerce").iloc[0]
        upper = pd.to_numeric(pd.Series([result.at[position, "ci_upper"]]), errors="coerce").iloc[0]
        if np.isfinite(lower) and np.isfinite(upper):
            result.at[position, "ci_available"] = True
            result.at[position, "ci_unavailable_reason"] = None
    expected_counts = {
        "varx_diagnostics_sensitivity": 140,
        "conditional_granger_sensitivity": 28,
        "conventional_granger_sensitivity": 88,
        "local_projection_sensitivity": 186,
        "pcmci_sensitivity_links": 1120,
    }
    for source, expected in expected_counts.items():
        actual = sum(key.startswith(f"stage09b__{source}__") for key in mappings)
        if actual != expected:
            raise ValueError(
                f"Dynamic enrichment count changed for {source}: {actual} != {expected}"
            )
    return result


def finalize_registry_semantics(registry: pd.DataFrame) -> pd.DataFrame:
    """Complete row-level interpretation and missingness provenance fail-closed."""
    result = registry.copy()
    forecast_domains = {
        "forecast_benchmark",
        "weekly_llm_forecast",
        "monthly_llm_forecast_primary",
        "forecast_ablation",
        "nonlinear_forecast_robustness",
    }
    signed_domains = {
        "dynamic_path_comparison",
        "dynamic_time_series_sensitivity",
    }
    for index, row in result.iterrows():
        domain = str(row.get("evidence_domain"))
        source = str(row.get("diagnostic_source"))
        effect = pd.to_numeric(pd.Series([row.get("effect_size")]), errors="coerce").iloc[0]
        effect_direction = direction(effect)
        result.at[index, "effect_direction"] = effect_direction
        if domain in forecast_domains:
            result.at[index, "signed_effect_interpretable"] = True
            result.at[index, "rejection_meaning"] = (
                "forecast_accuracy_difference_if_rejected"
            )
            result.at[index, "claim_direction"] = {
                "positive": "challenger_favored_point_estimate",
                "negative": "baseline_favored_point_estimate",
                "zero": "no_point_estimate_difference",
            }.get(effect_direction, "direction_unavailable")
        elif domain in signed_domains:
            result.at[index, "signed_effect_interpretable"] = True
            if is_missing_scalar(row.get("rejection_meaning")):
                result.at[index, "rejection_meaning"] = (
                    "dynamic_predictive_association_if_rejected_not_causality"
                )
            result.at[index, "claim_direction"] = {
                "positive": "positive_dynamic_association_point_estimate",
                "negative": "negative_dynamic_association_point_estimate",
                "zero": "zero_dynamic_association_point_estimate",
            }.get(effect_direction, "direction_unavailable")
        elif normalize_bool(row.get("signed_effect_interpretable"), False):
            if source in {"stage08b_placebo_tests", "stage09_future_lead_placebos"}:
                label = "placebo_association"
            elif source == "stage09_causal_ml_sensitivity":
                label = "orthogonalized_predictive_association"
            elif source == "forecast_bias_hac":
                label = "forecast_bias"
            elif source == "error_driver_hac":
                label = "descriptive_error_driver_association"
            elif source == "pcmci_sensitivity_links":
                label = "conditional_temporal_association"
            elif source == "local_projection_sensitivity":
                label = "dynamic_predictive_association"
            else:
                label = "signed_effect"
            result.at[index, "claim_direction"] = (
                f"{effect_direction}_{label}_point_estimate"
            )
        elif is_missing_scalar(row.get("claim_direction")):
            result.at[index, "claim_direction"] = "diagnostic_not_directional_claim"
        if is_missing_scalar(row.get("rejection_meaning")):
            result.at[index, "rejection_meaning"] = (
                "test_specific_statistical_difference_if_rejected"
            )

        frequency = str(row.get("frequency"))
        if is_missing_scalar(row.get("index_construction")) and frequency in {
            "weekly",
            "monthly",
        }:
            result.at[index, "index_construction"] = (
                "weekly_log_volume" if frequency == "weekly" else "adapted_pdf_sqrt_ema3"
            )
            result.at[index, "index_construction_unavailable_reason"] = None
        elif is_missing_scalar(row.get("index_construction")):
            if is_missing_scalar(row.get("index_construction_unavailable_reason")):
                result.at[index, "index_construction_unavailable_reason"] = (
                    "source_hypothesis_has_no_index_construction"
                )
        else:
            result.at[index, "index_construction_unavailable_reason"] = None

        start_missing = is_missing_scalar(row.get("sample_start_date"))
        end_missing = is_missing_scalar(row.get("sample_end_date"))
        if not start_missing and not end_missing:
            result.at[index, "sample_dates_unavailable_reason"] = None
        elif is_missing_scalar(row.get("sample_dates_unavailable_reason")):
            result.at[index, "sample_dates_unavailable_reason"] = (
                "upstream_inference_table_did_not_export_sample_dates"
                if start_missing and end_missing
                else "upstream_inference_table_exported_incomplete_sample_dates"
            )

        lower = pd.to_numeric(pd.Series([row.get("ci_lower")]), errors="coerce").iloc[0]
        upper = pd.to_numeric(pd.Series([row.get("ci_upper")]), errors="coerce").iloc[0]
        available = bool(np.isfinite(lower) and np.isfinite(upper))
        result.at[index, "ci_available"] = available
        if available:
            result.at[index, "ci_unavailable_reason"] = None
        elif is_missing_scalar(row.get("ci_unavailable_reason")):
            result.at[index, "ci_unavailable_reason"] = (
                "compatible_upstream_interval_unavailable"
            )
    return result


def reconcile_upstream_adjustments(
    registry: pd.DataFrame, tolerance: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = registry.copy()
    eligible = result["upstream_adjusted_p_value"].notna()
    groups = result.loc[eligible].groupby(
        "upstream_reconciliation_family_id", sort=True, dropna=False
    ).groups
    for family, indices in groups.items():
        methods = set(result.loc[indices, "upstream_adjustment_method"])
        if len(methods) != 1:
            raise ValueError(f"Upstream family mixes methods: {family}")
        method = next(iter(methods))
        recalculated = adjust_pvalues(result.loc[indices, "raw_p_value"], method)
        result.loc[indices, "upstream_recomputed_adjusted_p_value"] = recalculated
    result.loc[eligible, "upstream_adjustment_delta"] = (
        result.loc[eligible, "upstream_recomputed_adjusted_p_value"]
        - result.loc[eligible, "upstream_adjusted_p_value"]
    ).abs()
    reconciliation = result.loc[
        eligible,
        [
            "hypothesis_id",
            "source_stage",
            "source_table",
            "diagnostic_source",
            "upstream_reconciliation_family_id",
            "upstream_adjustment_method",
            "raw_p_value",
            "upstream_adjusted_p_value",
            "upstream_recomputed_adjusted_p_value",
            "upstream_adjustment_delta",
        ],
    ].copy()
    reconciliation["reconciliation_tolerance"] = tolerance
    reconciliation["coverage_scope"] = (
        "rows_with_exported_upstream_adjusted_p_value"
    )
    reconciliation["reconciliation_passed"] = reconciliation[
        "upstream_adjustment_delta"
    ].le(tolerance)
    if reconciliation.empty or not reconciliation["reconciliation_passed"].all():
        bad = reconciliation.loc[~reconciliation["reconciliation_passed"]].head(10)
        raise ValueError("Upstream adjustment reconciliation failed:\n" + bad.to_string(index=False))
    return result, reconciliation


def upstream_adjustment_coverage_audit(
    registry: pd.DataFrame, reconciliation: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = registry.assign(
        diagnostic_source=registry["diagnostic_source"].fillna("not_applicable")
    ).groupby(
        ["source_stage", "source_table", "diagnostic_source"],
        sort=True,
        dropna=False,
    )
    reconciled_ids = set(reconciliation["hypothesis_id"])
    for key, group in grouped:
        available = group["upstream_adjusted_p_value"].notna()
        available_ids = set(group.loc[available, "hypothesis_id"])
        rows.append(
            {
                "source_stage": key[0],
                "source_table": key[1],
                "diagnostic_source": key[2],
                "source_rows": len(group),
                "canonical_hypotheses": int(
                    strict_bool_series(
                        group["canonical_inference"], "canonical_inference"
                    ).sum()
                ),
                "upstream_adjusted_value_available": int(available.sum()),
                "upstream_adjusted_value_unavailable": int((~available).sum()),
                "reconciliation_rows": len(available_ids & reconciled_ids),
                "all_available_values_reconciled": available_ids.issubset(
                    reconciled_ids
                ),
                "coverage_scope": (
                    "reconciliation_covers_only_values_exported_by_upstream_source"
                ),
                "missing_adjusted_values_are_not_reconstructed": True,
            }
        )
    return pd.DataFrame(rows)


def apply_forecast_semantic_deduplication(
    registry: pd.DataFrame,
    ablation_source: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = registry.copy()
    contract = config["semantic_duplicate_contract"]
    tolerance = float(contract["numeric_effect_tolerance"])
    mapping_by_comparison = {
        item["comparison_id"]: item for item in contract["mappings"]
    }
    audits: list[dict[str, Any]] = []
    for source_row, record in ablation_source.iterrows():
        mapping = mapping_by_comparison.get(record["comparison_id"])
        if mapping is None or record["sample_subset"] != "full_sample":
            continue
        if record["baseline_specification_id"] != mapping["baseline_specification_id"]:
            continue
        if record["variant_specification_id"] != mapping["variant_specification_id"]:
            continue
        test = (
            "diebold_mariano_squared"
            if record["loss"] == "squared_error"
            else "diebold_mariano_absolute"
        )
        if record["frequency"] == "weekly":
            earlier_table = "stage07b_weekly_llm_statistical_tests"
            challenger = "ridge__score_hybrid__{score}__{signal}".format(
                score=mapping["earlier_score_source"],
                signal=mapping["earlier_signal_name"],
            )
            construction_mask = result["index_construction"].eq("weekly_log_volume")
        else:
            earlier_table = "stage07c_monthly_llm_statistical_tests"
            challenger = "ridge__monthly__{construction}__score_hybrid__{score}__{signal}".format(
                construction=record["index_construction"],
                score=mapping["earlier_score_source"],
                signal=mapping["earlier_signal_name"],
            )
            construction_mask = result["challenger_index_construction"].eq(
                record["index_construction"]
            )
        candidates = result.loc[
            result["source_table"].eq(earlier_table)
            & result["baseline_model"].eq("ridge__ar_brent_coverage")
            & result["challenger_model"].eq(challenger)
            & result["horizon"].eq(record["horizon"])
            & result["test_method"].eq(test)
            & construction_mask
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Semantic forecast duplicate does not have one earlier owner: "
                f"{record['comparison_id']} {record['frequency']} h={record['horizon']} "
                f"loss={record['loss']} candidates={len(candidates)}"
            )
        owner = candidates.iloc[0]
        if int(owner["sample_size"]) != int(record["paired_n"]):
            raise ValueError("Semantic duplicate paired sample size changed")
        effect_delta = abs(
            float(owner["effect_size"]) - float(record["mean_loss_difference"])
        )
        if effect_delta > tolerance:
            raise ValueError("Semantic duplicate effect validation failed")
        stage09_identity = {
            key: record.get(key)
            for key in (
                "frequency",
                "index_construction",
                "comparison_id",
                "baseline_specification_id",
                "variant_specification_id",
                "sample_subset",
                "loss",
                "horizon",
            )
        }
        duplicate_id = "stage09_ablation__" + stable_hash(
            "stage09_ablation", stage09_identity
        )
        duplicate_index = result.index[result["hypothesis_id"].eq(duplicate_id)]
        if len(duplicate_index) != 1:
            raise ValueError(f"Stage 09 duplicate ID is not unique: {duplicate_id}")
        idx = duplicate_index[0]
        result.at[idx, "canonical_inference"] = False
        result.at[idx, "duplicate_of"] = owner["hypothesis_id"]
        result.at[idx, "duplicate_reason"] = "later_semantic_alias_owned_by_stage07b_or_stage07c"
        owner_index = candidates.index[0]
        support_columns = (
            "supporting_stage09_hypothesis_id",
            "supporting_stage09_dm_raw_p_value",
            "supporting_stage09_reassigned_adjusted_p_value",
            "supporting_block_bootstrap_p_value",
            "supporting_block_bootstrap_ci_lower",
            "supporting_block_bootstrap_ci_upper",
            "supporting_clark_west_one_sided_p_value",
            "supporting_rmse_improvement_percent",
            "supporting_incremental_oos_r_squared",
        )
        for column in support_columns:
            result.at[owner_index, column] = result.at[idx, column]
        audits.append(
            {
                "duplicate_type": "stage09_forecast_semantic_alias",
                "duplicate_hypothesis_id": duplicate_id,
                "canonical_owner_hypothesis_id": owner["hypothesis_id"],
                "frequency": record["frequency"],
                "index_construction": record["index_construction"],
                "comparison_id": record["comparison_id"],
                "horizon": int(record["horizon"]),
                "loss": record["loss"],
                "paired_sample_size": int(record["paired_n"]),
                "effect_absolute_difference": effect_delta,
                "raw_p_value_absolute_difference": abs(
                    float(owner["raw_p_value"]) - float(record["dm_hac_p_value"])
                ),
                "reassigned_primary_family": normalize_bool(record["primary_family"]),
                "canonical_owner_family_id": owner["upstream_family_id"],
                "duplicate_family_id": record["family_id"],
                "canonical_owner_rule": "earliest_frozen_stage07_family",
                "validation_passed": True,
            }
        )
    audit_columns = [
        "duplicate_type",
        "duplicate_hypothesis_id",
        "canonical_owner_hypothesis_id",
        "frequency",
        "index_construction",
        "comparison_id",
        "horizon",
        "loss",
        "paired_sample_size",
        "effect_absolute_difference",
        "raw_p_value_absolute_difference",
        "reassigned_primary_family",
        "canonical_owner_family_id",
        "duplicate_family_id",
        "canonical_owner_rule",
        "validation_passed",
    ]
    audit = pd.DataFrame(audits, columns=audit_columns)
    expected = int(contract["expected_stage09_duplicate_rows"])
    if len(audit) != expected:
        raise ValueError(f"Stage 09 forecast duplicate count changed: {len(audit)} != {expected}")
    frequency_counts = (
        audit.groupby("frequency").size().to_dict() if not audit.empty else {}
    )
    for frequency, count in contract["expected_stage09_duplicate_rows_by_frequency"].items():
        if int(frequency_counts.get(frequency, 0)) != int(count):
            raise ValueError(f"Stage 09 {frequency} duplicate count changed")
    primary_count = int(audit["reassigned_primary_family"].sum()) if not audit.empty else 0
    if primary_count != int(contract["expected_reassigned_primary_rows"]):
        raise ValueError("Stage 09 reassigned primary duplicate count changed")
    return result, audit


def apply_varx_coefficient_deduplication(
    registry: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = registry.copy()
    subset = result.loc[
        result["source_table"].eq("stage09_varx_coefficients_sensitivity")
    ].copy()
    contract = config["semantic_duplicate_contract"]["varx_coefficients"]
    if len(subset) != int(contract["expected_input_rows"]):
        raise ValueError("VARX coefficient input row count changed")
    audits: list[dict[str, Any]] = []
    for semantic_key, group in subset.groupby("semantic_key", sort=True):
        ordered = group.sort_values(
            ["maximum_lag_specification", "hypothesis_id"]
        )
        owner = ordered.iloc[0]
        if not np.allclose(
            ordered["effect_size"], owner["effect_size"], rtol=0.0, atol=0.0
        ) or not np.allclose(
            ordered["raw_p_value"], owner["raw_p_value"], rtol=0.0, atol=0.0
        ) or not np.allclose(
            ordered["standard_error"], owner["standard_error"], rtol=0.0, atol=0.0
        ):
            raise ValueError(f"Repeated VARX coefficient changed: {semantic_key}")
        for idx, duplicate in ordered.iloc[1:].iterrows():
            result.at[idx, "canonical_inference"] = False
            result.at[idx, "duplicate_of"] = owner["hypothesis_id"]
            result.at[idx, "duplicate_reason"] = "same_selected_varx_model_repeated_across_lag_ceilings"
            audits.append(
                {
                    "duplicate_type": "varx_coefficient_repeated_selected_model",
                    "duplicate_hypothesis_id": duplicate["hypothesis_id"],
                    "canonical_owner_hypothesis_id": owner["hypothesis_id"],
                    "frequency": duplicate["frequency"],
                    "index_construction": duplicate["index_construction"],
                    "comparison_id": None,
                    "horizon": np.nan,
                    "loss": None,
                    "paired_sample_size": duplicate["sample_size"],
                    "effect_absolute_difference": 0.0,
                    "raw_p_value_absolute_difference": 0.0,
                    "reassigned_primary_family": False,
                    "canonical_owner_family_id": owner["correction_family_id"],
                    "duplicate_family_id": duplicate["correction_family_id"],
                    "canonical_owner_rule": "smallest_maximum_lag_specification",
                    "validation_passed": True,
                }
            )
    unique_count = int(subset["semantic_key"].nunique())
    if unique_count != int(contract["expected_unique_hypotheses"]):
        raise ValueError("VARX coefficient unique-hypothesis count changed")
    if len(audits) != int(contract["expected_duplicate_rows"]):
        raise ValueError("VARX coefficient duplicate count changed")
    return result, pd.DataFrame(audits)


def apply_varx_diagnostic_deduplication(
    registry: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = registry.copy()
    mask = result["hypothesis_id"].str.startswith(
        "stage09b__varx_diagnostics_sensitivity__", na=False
    )
    subset = result.loc[mask].copy()
    contract = config["semantic_duplicate_contract"]["varx_diagnostics"]
    if len(subset) != int(contract["expected_input_p_value_rows"]):
        raise ValueError("VARX diagnostic p-value row count changed")
    if "dynamic_details" not in subset:
        raise ValueError("VARX diagnostics were not enriched before deduplication")
    subset["semantic_varx_diagnostic"] = subset.apply(
        lambda row: "varx_diagnostic__{frequency}__{path}__lag{lag}__{diagnostic}__{variable}__{details}".format(
            frequency=row["frequency"],
            path=row["msdi_path"],
            lag=row["selected_lag"],
            diagnostic=row["diagnostic"],
            variable=row["variable"],
            details=row["dynamic_details"],
        ),
        axis=1,
    )
    audits: list[dict[str, Any]] = []
    for semantic_key, group in subset.groupby("semantic_varx_diagnostic", sort=True):
        ordered = group.sort_values(
            ["maximum_lag_specification", "hypothesis_id"]
        )
        owner = ordered.iloc[0]
        if not np.allclose(
            ordered["effect_size"], owner["effect_size"], rtol=0.0, atol=0.0
        ) or not np.allclose(
            ordered["raw_p_value"], owner["raw_p_value"], rtol=0.0, atol=0.0
        ):
            raise ValueError(f"Repeated VARX diagnostic changed: {semantic_key}")
        result.at[owner.name, "semantic_key"] = semantic_key
        for idx, duplicate in ordered.iloc[1:].iterrows():
            result.at[idx, "semantic_key"] = semantic_key
            result.at[idx, "canonical_inference"] = False
            result.at[idx, "duplicate_of"] = owner["hypothesis_id"]
            result.at[idx, "duplicate_reason"] = "identical_varx_diagnostic_repeated_across_lag_ceilings"
            audits.append(
                {
                    "duplicate_type": "varx_diagnostic_repeated_selected_model",
                    "duplicate_hypothesis_id": duplicate["hypothesis_id"],
                    "canonical_owner_hypothesis_id": owner["hypothesis_id"],
                    "frequency": duplicate["frequency"],
                    "index_construction": duplicate["index_construction"],
                    "comparison_id": None,
                    "horizon": np.nan,
                    "loss": None,
                    "paired_sample_size": duplicate["sample_size"],
                    "effect_absolute_difference": 0.0,
                    "raw_p_value_absolute_difference": 0.0,
                    "reassigned_primary_family": False,
                    "canonical_owner_family_id": owner["correction_family_id"],
                    "duplicate_family_id": duplicate["correction_family_id"],
                    "canonical_owner_rule": "smallest_maximum_lag_specification",
                    "validation_passed": True,
                }
            )
    unique_count = int(subset["semantic_varx_diagnostic"].nunique())
    if unique_count != int(contract["expected_unique_hypotheses"]):
        raise ValueError(
            f"VARX diagnostic unique count changed: {unique_count}"
        )
    if len(audits) != int(contract["expected_duplicate_rows"]):
        raise ValueError("VARX diagnostic duplicate count changed")
    return result, pd.DataFrame(audits)


def apply_canonical_corrections(
    registry: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    result = registry.copy()
    canonical = result["canonical_inference"].astype(bool)
    if result.loc[canonical, "correction_family_id"].isna().any():
        raise ValueError("Canonical hypotheses lack correction-family IDs")
    for family, indices in result.loc[canonical].groupby(
        "correction_family_id", sort=True
    ).groups.items():
        methods = set(result.loc[indices, "correction_method"])
        roles = set(result.loc[indices, "inference_role"])
        if len(methods) != 1 or len(roles) != 1:
            raise ValueError(f"Correction family mixes methods/roles: {family}")
        method = next(iter(methods))
        ordered_indices = result.loc[indices].sort_values(
            ["raw_p_value", "hypothesis_id"], kind="mergesort"
        ).index
        adjusted = adjust_pvalues(
            result.loc[ordered_indices, "raw_p_value"], method
        )
        result.loc[indices, "correction_family_size"] = len(indices)
        result.loc[ordered_indices, "adjusted_p_value_within_family"] = adjusted
    result["reject_raw_at_alpha"] = result["raw_p_value"].le(alpha)
    result["reject_within_family_at_alpha"] = (
        result["canonical_inference"].astype(bool)
        & result["adjusted_p_value_within_family"].le(alpha)
    )
    role_methods = {
        "confirmatory_primary": "holm",
        "primary_diagnostic": "holm",
        "prespecified_secondary": "holm",
        "exploratory": "benjamini_hochberg",
    }
    for role, method in role_methods.items():
        indices = result.index[canonical & result["inference_role"].eq(role)]
        if not len(indices):
            continue
        ordered_indices = result.loc[indices].sort_values(
            ["raw_p_value", "hypothesis_id"], kind="mergesort"
        ).index
        adjusted = adjust_pvalues(
            result.loc[ordered_indices, "raw_p_value"], method
        )
        result.loc[indices, "global_role_family_size"] = len(indices)
        result.loc[ordered_indices, "adjusted_p_value_global_role"] = adjusted
    result["reject_global_role_at_alpha"] = (
        result["canonical_inference"].astype(bool)
        & result["adjusted_p_value_global_role"].le(alpha)
    )
    return result


def validate_family_contracts(
    registry: pd.DataFrame, config: Mapping[str, Any]
) -> None:
    primary = config["confirmatory_primary_families"]
    expected_sizes = primary["expected_family_sizes"]
    source_table = str(primary["source_table"])
    canonical = registry.loc[registry["canonical_inference"].astype(bool)]
    for family, expected in expected_sizes.items():
        observed = canonical.loc[
            canonical["source_table"].eq(source_table)
            & canonical["upstream_family_id"].eq(family)
            & canonical["inference_role"].eq("confirmatory_primary")
        ]
        if len(observed) != int(expected):
            raise ValueError(f"Confirmatory family size changed: {family}")
        if not observed["inference_role"].eq("confirmatory_primary").all():
            raise ValueError(f"Confirmatory family role changed: {family}")
    diagnostic_expected = config["registry_sources"]["stage09b_raw_handoff"][
        "expected_primary_diagnostic_family_sizes"
    ]
    for family, expected in diagnostic_expected.items():
        observed = canonical.loc[
            canonical["source_table"].eq("stage09b_raw_significance_handoff")
            & canonical["inference_role"].eq("primary_diagnostic")
            & canonical["upstream_family_id"].eq(family)
        ]
        if len(observed) != int(expected):
            raise ValueError(f"Primary diagnostic family size changed: {family}")
    all_diagnostic_expected = config["registry_sources"]["stage09b_raw_handoff"][
        "expected_diagnostic_family_sizes"
    ]
    diagnostic_rows = registry.loc[
        registry["source_table"].eq("stage09b_raw_significance_handoff")
        & registry["upstream_family_id"].astype(str).str.startswith("forecast_")
    ]
    observed_diagnostic_sizes = (
        diagnostic_rows.groupby("upstream_family_id", sort=True).size().to_dict()
    )
    expected_diagnostic_sizes = {
        str(family): int(size)
        for family, size in all_diagnostic_expected.items()
    }
    if expected_diagnostic_sizes and observed_diagnostic_sizes != expected_diagnostic_sizes:
        raise ValueError(
            "Forecast diagnostic family sizes changed: "
            f"expected={expected_diagnostic_sizes} "
            f"observed={observed_diagnostic_sizes}"
        )
    nonlinear = canonical.loc[
        canonical["source_table"].eq("stage09_nonlinear_statistical_tests")
    ]
    nonlinear_contract = config["registry_sources"]["stage09_nonlinear"]
    expected_nonlinear_algorithms = set(
        nonlinear_contract["expected_algorithms"]
    )
    observed_nonlinear_algorithms = set(nonlinear["algorithm"].dropna().astype(str))
    expected_nonlinear_rows = nonlinear_contract.get("expected_rows")
    if (
        expected_nonlinear_rows is not None
        and len(nonlinear) != int(expected_nonlinear_rows)
    ):
        raise ValueError(
            "Stage 09 nonlinear hypothesis count changed: "
            f"expected={nonlinear_contract['expected_rows']} observed={len(nonlinear)}"
        )
    if observed_nonlinear_algorithms != expected_nonlinear_algorithms:
        raise ValueError(
            "Stage 09 nonlinear algorithm set changed: "
            f"expected={sorted(expected_nonlinear_algorithms)} "
            f"observed={sorted(observed_nonlinear_algorithms)}"
        )
    peak = canonical.loc[
        canonical["source_table"].eq("stage09_peak_breadth_difference")
    ]
    peak_contract = config["registry_sources"]["stage09_peak_breadth"]
    if len(peak) != int(peak_contract["expected_rows"]):
        raise ValueError("Peak/breadth family size changed")
    peak_sizes = peak.groupby("frequency").size().to_dict()
    for frequency, expected in peak_contract["expected_family_sizes"].items():
        if int(peak_sizes.get(frequency, 0)) != int(expected):
            raise ValueError(f"Peak/breadth {frequency} family size changed")
    ardl = canonical.loc[
        canonical["source_table"].eq("stage09_ardl_ecm_sensitivity")
    ]
    ardl_contract = config["registry_sources"]["stage09_ardl_ecm"]
    if len(ardl) != int(ardl_contract["expected_rows_after_pivot"]):
        raise ValueError("ARDL pivoted hypothesis count changed")
    if ardl["correction_family_id"].nunique() != int(
        ardl_contract["expected_family_count"]
    ):
        raise ValueError("ARDL correction family count changed")
    if not ardl.groupby("correction_family_id").size().eq(
        int(ardl_contract["expected_family_size"])
    ).all():
        raise ValueError("ARDL correction family size changed")
    varx = canonical.loc[
        canonical["source_table"].eq("stage09_varx_coefficients_sensitivity")
    ]
    varx_contract = config["registry_sources"]["stage09_varx_coefficients"]
    if len(varx) != int(varx_contract["expected_canonical_rows"]):
        raise ValueError("Canonical VARX coefficient count changed")
    if varx["correction_family_id"].nunique() != int(
        varx_contract["expected_family_count"]
    ):
        raise ValueError("VARX coefficient family count changed")


def family_summary(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in registry.groupby("correction_family_id", sort=True, dropna=False):
        canonical = group.loc[group["canonical_inference"].astype(bool)]
        if canonical.empty:
            continue
        rows.append(
            {
                "correction_family_id": family,
                "inference_role": canonical["inference_role"].iloc[0],
                "correction_method": canonical["correction_method"].iloc[0],
                "source_rows": len(group),
                "canonical_hypotheses": len(canonical),
                "semantic_duplicates": int((~group["canonical_inference"].astype(bool)).sum()),
                "raw_rejections_canonical": int(canonical["reject_raw_at_alpha"].sum()),
                "within_family_rejections": int(
                    canonical["reject_within_family_at_alpha"].sum()
                ),
                "global_role_rejections": int(
                    canonical["reject_global_role_at_alpha"].sum()
                ),
                "minimum_raw_p_value": (
                    float(canonical["raw_p_value"].min())
                ),
                "minimum_adjusted_p_value_within_family": (
                    float(canonical["adjusted_p_value_within_family"].min())
                ),
                "maximum_upstream_adjustment_delta": (
                    float(group["upstream_adjustment_delta"].max())
                    if group["upstream_adjustment_delta"].notna().any()
                    else np.nan
                ),
                "causal_claim_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def domain_summary(registry: pd.DataFrame) -> pd.DataFrame:
    canonical = registry.loc[registry["canonical_inference"].astype(bool)]
    rows: list[dict[str, Any]] = []
    for key, group in canonical.groupby(
        ["evidence_domain", "inference_role", "frequency"],
        sort=True,
        dropna=False,
    ):
        signed = group.loc[
            strict_bool_series(
                group["signed_effect_interpretable"],
                "signed_effect_interpretable",
            )
        ]
        rows.append(
            {
                "evidence_domain": key[0],
                "inference_role": key[1],
                "frequency": key[2],
                "hypotheses": len(group),
                "families": group["correction_family_id"].nunique(),
                "raw_rejections": int(group["reject_raw_at_alpha"].sum()),
                "within_family_rejections": int(
                    group["reject_within_family_at_alpha"].sum()
                ),
                "pooled_role_rejections": int(
                    group["reject_global_role_at_alpha"].sum()
                ),
                "signed_effect_hypotheses": len(signed),
                "positive_effects": int(signed["effect_direction"].eq("positive").sum()),
                "negative_effects": int(signed["effect_direction"].eq("negative").sum()),
                "sample_size_unavailable": int(group["sample_size"].isna().sum()),
                "confidence_interval_unavailable": int((~group["ci_available"]).sum()),
                "causal_claim_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def primary_outputs(registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = registry.loc[
        registry["canonical_inference"].astype(bool)
        & registry["inference_role"].eq("confirmatory_primary")
    ].copy()
    primary["adjusted_significant_positive_improvement"] = (
        primary["reject_within_family_at_alpha"]
        & primary["effect_size"].gt(0)
    )
    primary["adjusted_significant_worsening"] = (
        primary["reject_within_family_at_alpha"]
        & primary["effect_size"].lt(0)
    )
    primary["economic_importance_claim_allowed"] = False
    primary["primary_claim_interpretation"] = np.select(
        [
            primary["adjusted_significant_positive_improvement"],
            primary["adjusted_significant_worsening"],
            primary["reject_within_family_at_alpha"],
        ],
        [
            "holm_significant_and_direction_favors_challenger",
            "holm_significant_and_direction_favors_baseline",
            "holm_significant_zero_or_unavailable_direction",
        ],
        default="not_holm_significant",
    )
    summary_rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [
        (str(family), group)
        for family, group in primary.groupby("upstream_family_id", sort=True)
    ]
    groups.append(("all_confirmatory_primary_families", primary))
    for family, group in groups:
        positive = int(group["adjusted_significant_positive_improvement"].sum())
        negative = int(group["adjusted_significant_worsening"].sum())
        if positive and negative:
            conclusion = "mixed_direction_adjusted_significant_results"
        elif positive:
            conclusion = "some_predefined_comparisons_support_challenger_improvement"
        elif negative:
            conclusion = "some_predefined_comparisons_show_challenger_worsening"
        else:
            conclusion = "no_holm_adjusted_directional_difference_in_family"
        summary_rows.append(
            {
                "claim_family": family,
                "hypotheses": len(group),
                "holm_adjusted_rejections": int(
                    group["reject_within_family_at_alpha"].sum()
                ),
                "pooled_role_adjusted_rejections": int(
                    group["reject_global_role_at_alpha"].sum()
                ),
                "adjusted_significant_positive_improvements": positive,
                "adjusted_significant_worsenings": negative,
                "positive_point_estimates": int(group["effect_size"].gt(0).sum()),
                "negative_point_estimates": int(group["effect_size"].lt(0).sum()),
                "minimum_raw_p_value": float(group["raw_p_value"].min()),
                "minimum_holm_adjusted_p_value": float(
                    group["adjusted_p_value_within_family"].min()
                ),
                "conclusion": conclusion,
                "effect_magnitude_reported": bool(
                    group["economic_magnitude_metric"].notna().all()
                    and group["economic_magnitude_value"].notna().all()
                ),
                "economic_importance_evaluated": False,
                "economic_importance_claim_allowed": False,
                "publication_eligible_without_human_validation": False,
                "causal_claim_allowed": False,
            }
        )
    return primary.sort_values(
        ["upstream_family_id", "horizon", "baseline_model", "challenger_model"]
    ).reset_index(drop=True), pd.DataFrame(summary_rows)


def uncertainty_audit(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in registry.groupby(
        ["source_stage", "source_table", "diagnostic_source", "inference_role"],
        sort=True,
        dropna=False,
    ):
        canonical = group.loc[group["canonical_inference"].astype(bool)]
        effects = pd.to_numeric(canonical["effect_size"], errors="coerce")
        standard_errors = pd.to_numeric(canonical["standard_error"], errors="coerce")
        lower = pd.to_numeric(canonical["ci_lower"], errors="coerce")
        upper = pd.to_numeric(canonical["ci_upper"], errors="coerce")
        levels = pd.to_numeric(canonical["confidence_level"], errors="coerce")
        ci_flags = strict_bool_series(canonical["ci_available"], "ci_available")
        finite_bounds = lower.notna() & upper.notna() & np.isfinite(lower) & np.isfinite(upper)
        ci_flag_mismatch = ci_flags.ne(finite_bounds)
        invalid_order = ci_flags & lower.gt(upper)
        invalid_level = ci_flags & (~levels.between(0.0, 1.0, inclusive="neither"))
        missing_method = ci_flags & canonical["uncertainty_method"].map(is_missing_scalar)
        available_with_reason = ci_flags & (~canonical["ci_unavailable_reason"].map(is_missing_scalar))
        unavailable_missing_reason = (~ci_flags) & canonical[
            "ci_unavailable_reason"
        ].map(is_missing_scalar)
        negative_or_nonfinite_se = standard_errors.notna() & (
            ~np.isfinite(standard_errors) | standard_errors.lt(0)
        )
        normal_mismatch = pd.Series(False, index=canonical.index)
        for position in canonical.index[ci_flags]:
            method = str(canonical.at[position, "uncertainty_method"]).lower()
            if "bootstrap" in method or not any(
                token in method for token in ("normal", "asymptotic")
            ):
                continue
            effect = effects.at[position]
            se = standard_errors.at[position]
            level = levels.at[position]
            if not np.isfinite(effect) or not np.isfinite(se) or not 0 < level < 1:
                normal_mismatch.at[position] = True
                continue
            if "critical_1_96" in method:
                expected_lower, expected_upper = (
                    float(effect - 1.96 * se),
                    float(effect + 1.96 * se),
                )
            else:
                expected_lower, expected_upper = normal_interval(effect, se, level)
            if not (
                np.isclose(lower.at[position], expected_lower, rtol=1.0e-5, atol=1.0e-6)
                and np.isclose(upper.at[position], expected_upper, rtol=1.0e-5, atol=1.0e-6)
            ):
                normal_mismatch.at[position] = True
        start_missing = canonical["sample_start_date"].map(is_missing_scalar)
        end_missing = canonical["sample_end_date"].map(is_missing_scalar)
        incomplete_date_pair = start_missing.ne(end_missing)
        dates_missing_without_reason = (
            start_missing
            & end_missing
            & canonical["sample_dates_unavailable_reason"].map(is_missing_scalar)
        )
        construction_missing_without_reason = (
            canonical["index_construction"].map(is_missing_scalar)
            & canonical["index_construction_unavailable_reason"].map(is_missing_scalar)
        )
        semantic_columns = (
            "null_hypothesis",
            "alternative",
            "sidedness",
            "effect_metric",
            "effect_unit",
            "rejection_meaning",
            "claim_direction",
        )
        semantic_missing = pd.Series(False, index=canonical.index)
        for column in semantic_columns:
            semantic_missing = semantic_missing | canonical[column].map(
                is_missing_scalar
            ).astype(bool)
        violation_count = int(
            (~np.isfinite(effects)).sum()
            + negative_or_nonfinite_se.sum()
            + ci_flag_mismatch.sum()
            + invalid_order.sum()
            + invalid_level.sum()
            + missing_method.sum()
            + available_with_reason.sum()
            + unavailable_missing_reason.sum()
            + normal_mismatch.sum()
            + incomplete_date_pair.sum()
            + dates_missing_without_reason.sum()
            + construction_missing_without_reason.sum()
            + semantic_missing.sum()
        )
        rows.append(
            {
                "source_stage": key[0],
                "source_table": key[1],
                "diagnostic_source": key[2],
                "inference_role": key[3],
                "source_rows": len(group),
                "canonical_hypotheses": len(canonical),
                "effect_size_available": int(canonical["effect_size"].notna().sum()),
                "standard_error_available": int(canonical["standard_error"].notna().sum()),
                "confidence_interval_available": int(canonical["ci_available"].sum()),
                "confidence_interval_unavailable": int((~canonical["ci_available"]).sum()),
                "sample_size_available": int(canonical["sample_size"].notna().sum()),
                "sample_size_unavailable": int(canonical["sample_size"].isna().sum()),
                "effect_direction_available": int(
                    canonical["effect_direction"].ne("unavailable").sum()
                ),
                "invalid_effect_size_count": int((~np.isfinite(effects)).sum()),
                "invalid_standard_error_count": int(negative_or_nonfinite_se.sum()),
                "ci_flag_or_bound_mismatch_count": int(ci_flag_mismatch.sum()),
                "ci_bound_order_invalid_count": int(invalid_order.sum()),
                "ci_confidence_level_invalid_count": int(invalid_level.sum()),
                "interval_provenance_invalid_count": int(
                    missing_method.sum() + available_with_reason.sum()
                ),
                "fabricated_interval_count": int(
                    missing_method.sum() + normal_mismatch.sum()
                ),
                "normal_interval_reconstruction_mismatch_count": int(
                    normal_mismatch.sum()
                ),
                "unavailable_interval_reason_missing_count": int(
                    unavailable_missing_reason.sum()
                ),
                "sample_date_pair_invalid_count": int(incomplete_date_pair.sum()),
                "sample_date_reason_missing_count": int(
                    dates_missing_without_reason.sum()
                ),
                "index_construction_reason_missing_count": int(
                    construction_missing_without_reason.sum()
                ),
                "hypothesis_semantics_missing_count": int(semantic_missing.sum()),
                "total_validation_violations": violation_count,
                "audit_passed": violation_count == 0,
            }
        )
    return pd.DataFrame(rows)


def pooled_role_sensitivity(registry: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "hypothesis_id",
        "inference_role",
        "evidence_domain",
        "frequency",
        "correction_family_id",
        "raw_p_value",
        "adjusted_p_value_within_family",
        "reject_within_family_at_alpha",
        "global_role_family_size",
        "adjusted_p_value_global_role",
        "reject_global_role_at_alpha",
        "effect_size",
        "effect_direction",
        "causal_claim_allowed",
    ]
    return registry.loc[registry["canonical_inference"].astype(bool), columns].copy()


def validate_registry(
    registry: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    reconciliation: pd.DataFrame,
    reconciliation_coverage: pd.DataFrame,
    uncertainty: pd.DataFrame,
    families: pd.DataFrame,
    source_audit: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    expected = config["expected_registry_counts"]
    required = config["output_contract"]["required_registry_fields"]
    gates: list[dict[str, Any]] = []

    def add(gate: str, passed: bool, observed: Any, expected_value: Any) -> None:
        gates.append(
            {
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected_value,
            }
        )

    expected_registry_rows = expected.get("main_rows_before_semantic_deduplication")
    add(
        "registry_rows_before_semantic_deduplication",
        len(registry) > 0
        if expected_registry_rows is None
        else len(registry) == int(expected_registry_rows),
        len(registry),
        "manifest_derived_positive_count"
        if expected_registry_rows is None
        else int(expected_registry_rows),
    )
    duplicates = int((~registry["canonical_inference"].astype(bool)).sum())
    expected_duplicates = expected.get("semantic_duplicate_rows")
    add(
        "semantic_duplicate_rows",
        duplicates >= 0
        if expected_duplicates is None
        else duplicates == int(expected_duplicates),
        duplicates,
        "recorded_from_four_model_run"
        if expected_duplicates is None
        else int(expected_duplicates),
    )
    canonical = int(registry["canonical_inference"].astype(bool).sum())
    expected_canonical = expected.get("canonical_main_rows")
    add(
        "canonical_main_rows",
        canonical > 0 and canonical + duplicates == len(registry)
        if expected_canonical is None
        else canonical == int(expected_canonical),
        canonical,
        "positive_and_partitions_registry"
        if expected_canonical is None
        else int(expected_canonical),
    )
    add(
        "duplicate_audit_rows",
        len(duplicate_audit) == duplicates,
        len(duplicate_audit),
        duplicates,
    )
    add(
        "unique_hypothesis_ids",
        not registry["hypothesis_id"].duplicated().any(),
        int(registry["hypothesis_id"].nunique()),
        len(registry),
    )
    canonical_frame = registry.loc[registry["canonical_inference"].astype(bool)]
    duplicate_frame = registry.loc[~registry["canonical_inference"].astype(bool)]
    add(
        "unique_canonical_semantic_keys",
        not canonical_frame["semantic_key"].duplicated().any(),
        int(canonical_frame["semantic_key"].nunique()),
        len(canonical_frame),
    )
    canonical_ids = set(canonical_frame["hypothesis_id"])
    valid_owner = (
        duplicate_frame["duplicate_of"].isin(canonical_ids)
        & duplicate_frame["duplicate_of"].ne(duplicate_frame["hypothesis_id"])
    )
    add(
        "all_duplicates_reference_existing_canonical_owner",
        bool(valid_owner.all()),
        int(valid_owner.sum()),
        duplicates,
    )
    add(
        "canonical_rows_are_not_marked_duplicates",
        not canonical_frame["duplicate_of"].notna().any(),
        int(canonical_frame["duplicate_of"].notna().sum()),
        0,
    )
    add(
        "raw_p_values_valid",
        bool(registry["raw_p_value"].between(0.0, 1.0).all()),
        int(registry["raw_p_value"].between(0.0, 1.0).sum()),
        len(registry),
    )
    add(
        "canonical_adjusted_p_values_complete",
        bool(canonical_frame["adjusted_p_value_within_family"].notna().all()),
        int(canonical_frame["adjusted_p_value_within_family"].notna().sum()),
        len(canonical_frame),
    )
    expected_family_sizes = canonical_frame.groupby("correction_family_id").size()
    family_size_matches = canonical_frame.apply(
        lambda row: int(row["correction_family_size"])
        == int(expected_family_sizes.loc[row["correction_family_id"]]),
        axis=1,
    )
    add(
        "canonical_correction_family_sizes_match_membership",
        bool(family_size_matches.all()),
        int(family_size_matches.sum()),
        len(canonical_frame),
    )
    expected_summary_rows = int(canonical_frame["correction_family_id"].nunique())
    add(
        "family_summary_contains_only_canonical_families",
        len(families) == expected_summary_rows
        and bool(families["canonical_hypotheses"].gt(0).all()),
        len(families),
        expected_summary_rows,
    )
    add(
        "canonical_effect_sizes_complete",
        bool(canonical_frame["effect_size"].notna().all()),
        int(canonical_frame["effect_size"].notna().sum()),
        len(canonical_frame),
    )
    adjusted_rejections = canonical_frame.loc[
        canonical_frame["reject_within_family_at_alpha"]
    ]
    magnitude_complete = (
        adjusted_rejections["economic_magnitude_metric"].notna()
        & pd.to_numeric(
            adjusted_rejections["economic_magnitude_value"], errors="coerce"
        ).notna()
    )
    add(
        "adjusted_rejections_have_effect_magnitude",
        bool(magnitude_complete.all()),
        int(magnitude_complete.sum()),
        len(adjusted_rejections),
    )
    missing_fields = sorted(set(required).difference(registry.columns))
    add("required_registry_fields", not missing_fields, missing_fields, [])
    add(
        "all_source_contracts",
        bool(
            strict_bool_series(
                source_audit["contract_passed"], "source contract_passed"
            ).all()
        ),
        int(
            strict_bool_series(
                source_audit["contract_passed"], "source contract_passed"
            ).sum()
        ),
        len(source_audit),
    )
    add(
        "upstream_adjustment_reconciliation",
        bool(reconciliation["reconciliation_passed"].all()),
        int(reconciliation["reconciliation_passed"].sum()),
        len(reconciliation),
    )
    add(
        "upstream_adjustment_coverage",
        bool(reconciliation_coverage["all_available_values_reconciled"].all()),
        int(reconciliation_coverage["all_available_values_reconciled"].sum()),
        len(reconciliation_coverage),
    )
    add(
        "all_uncertainty_effect_size_gates",
        bool(uncertainty["audit_passed"].all()),
        int(uncertainty["audit_passed"].sum()),
        len(uncertainty),
    )
    add(
        "no_causal_claim_allowed",
        not strict_bool_series(
            registry["causal_claim_allowed"], "causal_claim_allowed"
        ).any(),
        int(
            strict_bool_series(
                registry["causal_claim_allowed"], "causal_claim_allowed"
            ).sum()
        ),
        0,
    )
    add(
        "monthly_primary_present",
        bool(registry.loc[registry["frequency"].eq("monthly")].shape[0] > 0),
        int(registry["frequency"].eq("monthly").sum()),
        ">0",
    )
    audit = pd.DataFrame(gates)
    required_gate_names = set(config["output_contract"]["required_claim_gates"])
    observed_gate_names = set(audit["gate"])
    if observed_gate_names != required_gate_names:
        raise ValueError(
            "Claim-gate contract changed; missing="
            f"{sorted(required_gate_names - observed_gate_names)}; unexpected="
            f"{sorted(observed_gate_names - required_gate_names)}"
        )
    if not audit["passed"].all():
        raise ValueError(
            "Stage 09c claim gates failed:\n"
            + audit.loc[~audit["passed"]].to_string(index=False)
        )
    return audit


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "pandas", "scipy", "PyYAML"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not_installed"
    return result


def output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = list(REGISTRY_DEFAULTS)
    columns = [column for column in preferred if column in frame]
    columns.extend(column for column in frame.columns if column not in columns)
    return frame[columns]


def main() -> None:
    started = time.perf_counter()
    arguments = parse_args()
    executable = assert_project_venv()
    config = load_yaml(arguments.config)
    version = arguments.version or str(config["version"])
    logger, log_path = configure_logging(arguments.log_dir, version)
    logger.info(
        "stage=09c status=starting version=%s executable=%s", version, executable
    )
    try:
        alpha = float(config["analysis"]["alpha"])
        confidence_level = float(config["analysis"]["confidence_level"])
        if not 0 < alpha < 1 or not 0 < confidence_level < 1:
            raise ValueError("Alpha and confidence level must be in (0, 1)")
        script_path = Path(__file__).resolve()
        script_hash = sha256_file(script_path)
        script_lock_path = (
            PROJECT_ROOT / config["output_contract"]["script_sha256_lock"]
        ).resolve()
        expected_script_hash = read_sha256_lock(script_lock_path)
        script_hash_matches_lock = script_hash == expected_script_hash
        if not script_hash_matches_lock:
            raise ValueError(
                "Stage 09c script changed after its SHA-256 lock was frozen; "
                "review the change and regenerate configs/significance_script.sha256"
            )

        diagnostics_required = config["inputs"][
            "required_error_diagnostics_outputs"
        ]
        robustness_required = config["inputs"]["required_robustness_outputs"]
        diagnostics_manifest, diagnostics_paths, source_rows = audit_manifest_outputs(
            arguments.diagnostics_manifest,
            "stage09b_error_diagnostics",
            diagnostics_required,
        )
        robustness_manifest, robustness_paths, robustness_audit = audit_manifest_outputs(
            arguments.robustness_manifest,
            "stage09_robustness",
            robustness_required,
        )
        source_rows.extend(robustness_audit)
        validate_manifest_contracts(config, diagnostics_manifest, robustness_manifest)

        legacy_contracts = config["inputs"]["trusted_legacy_inference_tables"]
        prediction_manifest, prediction_path, audit = audit_legacy_inference_table(
            "stage07_prediction", arguments.prediction_manifest, legacy_contracts["prediction"]
        )
        source_rows.append(audit)
        weekly_manifest, weekly_path, audit = audit_legacy_inference_table(
            "stage07b_weekly_llm", arguments.weekly_llm_manifest, legacy_contracts["weekly_llm"]
        )
        source_rows.append(audit)
        monthly_manifest, monthly_path, audit = audit_legacy_inference_table(
            "stage07c_monthly_llm", arguments.monthly_llm_manifest, legacy_contracts["monthly_llm"]
        )
        source_rows.append(audit)
        source_rows.extend(
            validate_stage09_manifest_chain(
                pd.read_csv(robustness_paths["upstream_audit"]),
                {
                    "prediction": arguments.prediction_manifest,
                    "weekly_llm": arguments.weekly_llm_manifest,
                    "monthly_llm": arguments.monthly_llm_manifest,
                },
            )
        )

        upstream_source_audit = pd.read_csv(diagnostics_paths["source_contract_audit"])
        if not strict_bool_series(
            upstream_source_audit["contract_passed"],
            "Stage 09b source contract_passed",
        ).all():
            raise ValueError("Stage 09b source-contract audit contains failures")
        reconstruction = pd.read_csv(
            diagnostics_paths["reference_index_reconstruction_audit"]
        )
        if not strict_bool_series(
            reconstruction["reconstruction_passed"],
            "Stage 09b reconstruction_passed",
        ).all():
            raise ValueError("Stage 09b reference reconstruction contains failures")

        config_hash = sha256_file(arguments.config.resolve())
        source_rows.extend(
            [
                {
                    "source_stage": "stage09c",
                    "artifact": "configuration",
                    "path": str(arguments.config.resolve()),
                    "expected_sha256": CANONICAL_CONFIG_SHA256,
                    "actual_sha256": config_hash,
                    "expected_rows": 1,
                    "actual_rows": 1,
                    "contract_passed": config_hash == CANONICAL_CONFIG_SHA256,
                    "details": "canonical config required for publication status",
                },
                {
                    "source_stage": "stage09c",
                    "artifact": "script",
                    "path": str(script_path),
                    "expected_sha256": expected_script_hash,
                    "actual_sha256": script_hash,
                    "expected_rows": 1,
                    "actual_rows": 1,
                    "contract_passed": script_hash_matches_lock,
                    "details": "runtime script hash matched external integrity lock",
                },
                {
                    "source_stage": "stage09c",
                    "artifact": "script_sha256_lock",
                    "path": str(script_lock_path),
                    "expected_sha256": None,
                    "actual_sha256": sha256_file(script_lock_path),
                    "expected_rows": 1,
                    "actual_rows": 1,
                    "contract_passed": True,
                    "details": "external integrity lock recorded for version control",
                },
            ]
        )

        prediction = pd.read_csv(prediction_path, low_memory=False)
        weekly_llm = pd.read_csv(weekly_path, low_memory=False)
        monthly_llm = pd.read_csv(monthly_path, low_memory=False)
        handoff = pd.read_csv(
            diagnostics_paths["raw_significance_handoff"], low_memory=False
        )
        expected_handoff_raw = config["registry_sources"][
            "stage09b_raw_handoff"
        ].get("expected_rows")
        expected_handoff_rows = (
            None if expected_handoff_raw is None else int(expected_handoff_raw)
        )
        if expected_handoff_rows is not None and len(handoff) != expected_handoff_rows:
            raise ValueError(
                "Stage 09b raw significance handoff count changed: "
                f"expected={expected_handoff_rows} observed={len(handoff)}"
            )
        ablation = pd.read_csv(robustness_paths["ablation_tests"], low_memory=False)
        nonlinear = pd.read_csv(
            robustness_paths["nonlinear_statistical_tests"], low_memory=False
        )
        peak_breadth = pd.read_csv(
            robustness_paths["peak_breadth_difference"], low_memory=False
        )
        ardl = pd.read_csv(robustness_paths["ardl_ecm_sensitivity"], low_memory=False)
        varx_coefficients = pd.read_csv(
            robustness_paths["varx_coefficients_sensitivity"], low_memory=False
        )
        dml_consolidation = pd.read_csv(
            diagnostics_paths["dml_diagnostic_consolidation"], low_memory=False
        )
        source_rows.extend(audit_dml_source_contracts(dml_consolidation, config))

        registry = pd.concat(
            [
                build_legacy_prediction_registry(prediction, confidence_level),
                build_llm_registry(weekly_llm, False, confidence_level),
                build_llm_registry(monthly_llm, True, confidence_level),
                build_raw_handoff_registry(handoff),
                build_ablation_registry(ablation, confidence_level),
                build_nonlinear_registry(nonlinear, confidence_level),
                build_peak_breadth_registry(peak_breadth),
                build_ardl_registry(ardl),
                build_varx_coefficient_registry(varx_coefficients, confidence_level),
            ],
            ignore_index=True,
            sort=False,
        )
        if registry["hypothesis_id"].duplicated().any():
            raise ValueError("Hypothesis IDs collide before enrichment")
        registry = enrich_dynamic_handoff_families(
            registry,
            {
                key: pd.read_csv(robustness_paths[key], low_memory=False)
                for key in config["registry_sources"][
                    "stage09_dynamic_handoff_enrichment"
                ]["source_tables"]
            },
        )
        registry = enrich_forecast_handoff_uncertainty(
            registry,
            pd.read_csv(diagnostics_paths["forecast_error_summary"], low_memory=False),
            pd.read_csv(
                diagnostics_paths["mincer_zarnowitz_calibration"], low_memory=False
            ),
            pd.read_csv(
                diagnostics_paths["error_driver_coefficients"], low_memory=False
            ),
            confidence_level,
        )
        registry = enrich_dml_handoff_uncertainty(
            registry,
            dml_consolidation,
            config,
        )
        registry = finalize_registry_semantics(registry)
        registry, reconciliation = reconcile_upstream_adjustments(
            registry,
            float(config["corrections"]["upstream_reconciliation_tolerance"]),
        )

        registry, forecast_duplicates = apply_forecast_semantic_deduplication(
            registry, ablation, config
        )
        registry, coefficient_duplicates = apply_varx_coefficient_deduplication(
            registry, config
        )
        registry, diagnostic_duplicates = apply_varx_diagnostic_deduplication(
            registry, config
        )
        duplicate_audit = pd.concat(
            [forecast_duplicates, coefficient_duplicates, diagnostic_duplicates],
            ignore_index=True,
            sort=False,
        )

        registry = apply_canonical_corrections(registry, alpha)
        validate_family_contracts(registry, config)
        source_audit = pd.DataFrame(source_rows)
        source_audit["audit_id"] = source_audit.apply(
            lambda row: stable_hash(
                "source_audit",
                {"source_stage": row["source_stage"], "artifact": row["artifact"]},
            ),
            axis=1,
        )
        if source_audit["audit_id"].duplicated().any():
            raise ValueError("Source audit IDs collide")

        families = family_summary(registry)
        domains = domain_summary(registry)
        primary, primary_claims = primary_outputs(registry)
        uncertainty = uncertainty_audit(registry)
        reconciliation_coverage = upstream_adjustment_coverage_audit(
            registry, reconciliation
        )
        claim_gates = validate_registry(
            registry,
            duplicate_audit,
            reconciliation,
            reconciliation_coverage,
            uncertainty,
            families,
            source_audit,
            config,
        )
        exploratory_rejections = registry.loc[
            registry["canonical_inference"].astype(bool)
            & registry["inference_role"].eq("exploratory")
            & registry["reject_within_family_at_alpha"]
        ].copy()
        pooled = pooled_role_sensitivity(registry)

        tables = {
            "significance_registry": output_columns(
                registry.sort_values(
                    ["inference_role", "correction_family_id", "raw_p_value", "hypothesis_id"]
                ).reset_index(drop=True)
            ),
            "significance_family_summary": families,
            "significance_domain_summary": domains,
            "primary_forecast_inference": output_columns(primary),
            "primary_claim_summary": primary_claims,
            "semantic_duplicate_audit": duplicate_audit,
            "upstream_adjustment_reconciliation": reconciliation,
            "upstream_adjustment_coverage_audit": reconciliation_coverage,
            "uncertainty_effect_size_audit": uncertainty,
            "exploratory_adjusted_rejections": output_columns(
                exploratory_rejections.sort_values(
                    ["adjusted_p_value_within_family", "raw_p_value", "hypothesis_id"]
                ).reset_index(drop=True)
            ),
            "pooled_role_sensitivity": pooled,
            "claim_gate_audit": claim_gates,
            "source_contract_audit": source_audit,
        }
        expected_table_names = set(config["output_contract"]["tables"])
        observed_table_names = set(tables)
        if observed_table_names != expected_table_names:
            raise ValueError(
                "Output-table contract changed; missing="
                f"{sorted(expected_table_names - observed_table_names)}; unexpected="
                f"{sorted(observed_table_names - expected_table_names)}"
            )
        output_dir = arguments.output_dir.resolve()
        output_paths: dict[str, Path] = {}
        output_hashes: dict[str, str] = {}
        output_counts: dict[str, int] = {}
        for name, table in tables.items():
            path = output_dir / f"{version}_{name}.csv"
            atomic_write_csv(table, path)
            output_paths[name] = path
            output_hashes[name] = sha256_file(path)
            output_counts[name] = len(table)
            logger.info(
                "output table=%s rows=%s sha256=%s path=%s",
                name,
                len(table),
                output_hashes[name],
                path,
            )

        canonical_inputs = (
            arguments.config.resolve() == DEFAULT_CONFIG.resolve()
            and arguments.diagnostics_manifest.resolve()
            == DEFAULT_DIAGNOSTICS_MANIFEST.resolve()
            and arguments.robustness_manifest.resolve()
            == DEFAULT_ROBUSTNESS_MANIFEST.resolve()
            and arguments.prediction_manifest.resolve()
            == DEFAULT_PREDICTION_MANIFEST.resolve()
            and arguments.weekly_llm_manifest.resolve()
            == DEFAULT_WEEKLY_LLM_MANIFEST.resolve()
            and arguments.monthly_llm_manifest.resolve()
            == DEFAULT_MONTHLY_LLM_MANIFEST.resolve()
            and config_hash == CANONICAL_CONFIG_SHA256
            and script_hash_matches_lock
            and version == config["version"]
        )
        full_scope = canonical_inputs and not arguments.smoke_test
        status = (
            config["output_contract"]["full_status"]
            if full_scope
            else config["output_contract"]["debug_status"]
        )
        primary_overall = primary_claims.loc[
            primary_claims["claim_family"].eq("all_confirmatory_primary_families")
        ].iloc[0]
        manifest_path = output_dir / f"{version}_manifest.json"
        runtime = time.perf_counter() - started
        manifest = {
            "version": version,
            "stage": "09c_significance_analysis",
            "status": status,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": runtime,
            "publication_scope_complete": bool(full_scope),
            "publication_eligible_without_human_validation": False,
            "human_validation_required": True,
            "causal_claim_allowed": False,
            "semantic_deduplication_applied": True,
            "earliest_frozen_family_ownership_enforced": True,
            "raw_p_values_overwritten": False,
            "runtime_script_sha256": script_hash,
            "expected_script_sha256": expected_script_hash,
            "script_sha256_lock_matched": script_hash_matches_lock,
            "scope": {
                "registry_rows": len(registry),
                "canonical_hypotheses": int(
                    registry["canonical_inference"].astype(bool).sum()
                ),
                "semantic_duplicates": int(
                    (~registry["canonical_inference"].astype(bool)).sum()
                ),
                "correction_families": int(
                    registry.loc[
                        registry["canonical_inference"].astype(bool),
                        "correction_family_id",
                    ].nunique()
                ),
                "confirmatory_primary_hypotheses": len(primary),
                "primary_diagnostic_hypotheses": int(
                    registry["canonical_inference"].astype(bool)
                    .mul(registry["inference_role"].eq("primary_diagnostic"))
                    .sum()
                ),
                "exploratory_hypotheses": int(
                    registry["canonical_inference"].astype(bool)
                    .mul(registry["inference_role"].eq("exploratory"))
                    .sum()
                ),
            },
            "primary_result": {
                "holm_adjusted_rejections": int(
                    primary_overall["holm_adjusted_rejections"]
                ),
                "pooled_role_adjusted_rejections": int(
                    primary_overall["pooled_role_adjusted_rejections"]
                ),
                "adjusted_significant_positive_improvements": int(
                    primary_overall["adjusted_significant_positive_improvements"]
                ),
                "adjusted_significant_worsenings": int(
                    primary_overall["adjusted_significant_worsenings"]
                ),
                "conclusion": primary_overall["conclusion"],
            },
            "inputs": {
                "configuration": {
                    "path": str(arguments.config.resolve()),
                    "sha256": config_hash,
                },
                "script_sha256_lock": {
                    "path": str(script_lock_path),
                    "sha256": sha256_file(script_lock_path),
                },
                "diagnostics_manifest": {
                    "path": str(arguments.diagnostics_manifest.resolve()),
                    "sha256": sha256_file(arguments.diagnostics_manifest.resolve()),
                },
                "robustness_manifest": {
                    "path": str(arguments.robustness_manifest.resolve()),
                    "sha256": sha256_file(arguments.robustness_manifest.resolve()),
                },
                "prediction_manifest": {
                    "path": str(arguments.prediction_manifest.resolve()),
                    "sha256": sha256_file(arguments.prediction_manifest.resolve()),
                },
                "weekly_llm_manifest": {
                    "path": str(arguments.weekly_llm_manifest.resolve()),
                    "sha256": sha256_file(arguments.weekly_llm_manifest.resolve()),
                },
                "monthly_llm_manifest": {
                    "path": str(arguments.monthly_llm_manifest.resolve()),
                    "sha256": sha256_file(arguments.monthly_llm_manifest.resolve()),
                },
            },
            "outputs": {name: str(path) for name, path in output_paths.items()},
            "output_sha256": output_hashes,
            "output_row_counts": output_counts,
            "claim_gates_all_passed": bool(claim_gates["passed"].all()),
            "source_contracts_all_passed": bool(
                source_audit["contract_passed"].all()
            ),
            "software": package_versions(),
            "log": str(log_path.resolve()),
        }
        atomic_write_json(manifest, manifest_path)
        logger.info(
            "stage=09c status=%s runtime_seconds=%.3f manifest=%s",
            status,
            runtime,
            manifest_path,
        )
    except Exception:
        logger.exception("stage=09c status=failed")
        raise


if __name__ == "__main__":
    main()
