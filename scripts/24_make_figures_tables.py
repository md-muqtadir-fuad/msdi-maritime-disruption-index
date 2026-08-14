"""Build the frozen publication tables, figures, report, and result manifest.

Stage 10 is presentation-only. It verifies every upstream manifest and source
artifact before reading results, creates deterministic publication artifacts,
and preserves three hard boundaries: monthly evidence is primary, weekly
evidence is robustness, and observational results do not authorize causality.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/publication_outputs.yaml"
CANONICAL_CONFIG_SHA256 = (
    "c97afd869ba4d6553a1c2532d79cd430bc2d6ec1c78e8a0dfdb1dd73c4b24a27"
)

PUBLICATION_LABELS = {
    # LLMs and ensembles.
    "qwen_3_8b": "Qwen3-8B",
    "qwen_2_5_7b": "Qwen2.5-7B",
    "gemma_2_9b_it": "Gemma 2 9B IT",
    "glm_4_9b_0414": "GLM-4-9B",
    "forecast_equal_mean": "Equal-weight forecast",
    "forecast_nnls_stack": "NNLS forecast stack",
    "score_mean": "Mean-score ensemble",
    "score_median": "Median-score ensemble",
    # Frequencies, paths, model families, and diagnostics.
    "weekly": "Weekly",
    "monthly": "Monthly",
    "peak": "Peak",
    "breadth": "Breadth",
    "lstm": "LSTM",
    "prophet": "Prophet",
    "glm": "Penalized Gaussian GLM",
    "random_forest": "Random forest",
    "ridge": "Ridge",
    "xgboost": "XGBoost",
    "arch_lm": "ARCH LM",
    "jarque_bera": "Jarque–Bera",
    "ljung_box": "Ljung–Box",
    # Dynamic-model identifiers.
    "msdi_change": "MSDI",
    "scfi_return": "SCFI",
    "conditional_varx": "VARX",
    "conditional_ols_hac_prespecified_max_lag": "OLS–HAC",
    # Frozen index constructions used in trajectory panels.
    "weekly_log_volume": "Weekly index",
    "adapted_pdf_sqrt_ema3": "Adapted PDF",
    "mean_safe_weekly": "Mean-safe",
}

ABLATION_LABELS = {
    "value_of_controls__peak_without_coverage_controls": "No coverage controls (peak)",
    "value_of_controls__peak_without_brent_controls": "No Brent controls (peak)",
    "compare__exclude_freight_price_only_peak": "Exclude freight cost (peak)",
    "compare__exclude_freight_price_only_breadth": "Exclude freight cost (breadth)",
    "breadth_vs_no_msdi": "Breadth vs no MSDI",
    "peak_vs_no_msdi": "Peak vs no MSDI",
    "incremental_peak_given_breadth": "Add peak to breadth",
    "incremental_breadth_given_peak": "Add breadth to peak",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create integrity-checked publication tables and figures."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--version", default=None)
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the complete fast presentation layer with a smoke suffix.",
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


def read_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"JSON source not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON source must be an object: {resolved}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sha256_lock(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Script SHA-256 lock not found: {resolved}")
    token = resolved.read_text(encoding="utf-8").strip().split()[0].lower()
    if len(token) != 64 or any(character not in "0123456789abcdef" for character in token):
        raise ValueError(f"Invalid SHA-256 lock: {resolved}")
    return token


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_manifest_output(value: str | Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    if path.is_absolute():
        try:
            relative = path.relative_to(PROJECT_ROOT)
            candidate = PROJECT_ROOT / relative
        except ValueError:
            marker_parts = ("outputs", "data")
            candidate = path
            for marker in marker_parts:
                if marker in path.parts:
                    candidate = PROJECT_ROOT.joinpath(*path.parts[path.parts.index(marker) :])
                    break
        if candidate.is_file():
            return candidate.resolve()
    candidate = PROJECT_ROOT / path
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Manifest output not found: {value}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_workbook(path: Path, tables: Mapping[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        used: set[str] = set()
        for raw_name, frame in tables.items():
            base = raw_name[:31]
            sheet = base
            suffix = 1
            while sheet in used:
                suffix += 1
                sheet = f"{base[: 31 - len(str(suffix)) - 1]}_{suffix}"
            used.add(sheet)
            frame.to_excel(writer, sheet_name=sheet, index=False)
    os.replace(temporary, path)


def strict_bool_series(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }
    normalized = series.map(
        lambda value: None if pd.isna(value) else str(value).strip().lower()
    )
    unknown = sorted({value for value in normalized.dropna() if value not in mapping})
    if unknown:
        raise ValueError(f"Unknown boolean token(s) in {label}: {unknown}")
    return normalized.map(mapping).fillna(False).astype(bool)


def safe_zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = float(numeric.std(ddof=0))
    if not math.isfinite(std) or std <= 0.0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (numeric - float(numeric.mean())) / std


def publication_label(value: Any) -> str:
    """Return a concise human label while preserving registered raw IDs in tables."""
    raw = str(value).strip()
    if raw in PUBLICATION_LABELS:
        return PUBLICATION_LABELS[raw]
    words = raw.replace("__", " ").replace("_", " ").split()
    label = " ".join(words).strip()
    return label[:1].upper() + label[1:] if label else label


def ablation_label(value: Any) -> str:
    raw = str(value).strip()
    if raw in ABLATION_LABELS:
        return ABLATION_LABELS[raw]
    label = publication_label(raw)
    return label if len(label) <= 38 else label[:35].rstrip() + "…"


def granger_label(row: pd.Series) -> str:
    frequency = publication_label(row["frequency"])
    path = publication_label(row["msdi_path"])
    causing = publication_label(row["causing"])
    caused = publication_label(row["caused"])
    model = publication_label(row["model"])
    return f"{frequency} · {path} · {causing}→{caused} · {model}"


def compact(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    if not available:
        raise ValueError("None of the requested publication columns are available")
    return frame.loc[:, available].copy()


def read_csv_output(
    audited: Mapping[str, Mapping[str, Any]], stage: str, output_key: str
) -> pd.DataFrame:
    path = Path(audited[stage]["output_paths"][output_key])
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"Required source is empty: {stage}.{output_key}")
    return frame


def audit_sources(
    config: Mapping[str, Any], logger: logging.Logger
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    sources = config.get("source_manifests")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("source_manifests must be a non-empty mapping")
    audited: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    for stage, contract in sources.items():
        manifest_path = resolve_project_path(contract["path"])
        observed_manifest_hash = sha256_file(manifest_path)
        configured_manifest_hash = contract.get("sha256")
        expected_manifest_hash = (
            str(configured_manifest_hash).strip().lower()
            if configured_manifest_hash is not None
            and str(configured_manifest_hash).strip()
            else None
        )
        if (
            expected_manifest_hash is not None
            and observed_manifest_hash != expected_manifest_hash
        ):
            raise ValueError(
                f"{stage} manifest SHA-256 contract failed: "
                f"expected {expected_manifest_hash}, observed {observed_manifest_hash}"
            )
        manifest = read_json(manifest_path)
        if str(manifest.get("version")) != str(contract["version"]):
            raise ValueError(f"{stage} manifest version contract failed")
        if contract.get("status") and str(manifest.get("status")) != str(
            contract["status"]
        ):
            raise ValueError(f"{stage} manifest status contract failed")
        outputs = manifest.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError(f"{stage} manifest has no outputs mapping")
        manifest_hashes = manifest.get("output_sha256")
        if not isinstance(manifest_hashes, dict):
            manifest_hashes = {}
        manifest_rows = manifest.get("output_row_counts")
        if not isinstance(manifest_rows, dict):
            manifest_rows = {}
        legacy_manifest_rows = manifest.get("counts")
        if not isinstance(legacy_manifest_rows, dict):
            legacy_manifest_rows = {}
        legacy = contract.get("legacy_outputs", {})
        output_paths: dict[str, str] = {}
        for output_key in contract.get("required_outputs", []):
            if output_key not in outputs:
                raise ValueError(f"{stage} manifest lacks required output {output_key}")
            output_path = resolve_manifest_output(outputs[output_key])
            observed_hash = sha256_file(output_path)
            expected_hash = manifest_hashes.get(output_key)
            hash_source = "manifest"
            if expected_hash is None:
                legacy_contract = legacy.get(output_key)
                if not isinstance(legacy_contract, dict) or not legacy_contract.get(
                    "sha256"
                ):
                    raise ValueError(
                        f"{stage}.{output_key} has no frozen hash contract"
                    )
                expected_hash = legacy_contract["sha256"]
                hash_source = "config_legacy_lock"
            if observed_hash != str(expected_hash).lower():
                raise ValueError(f"{stage}.{output_key} SHA-256 contract failed")
            observed_rows: int | None = None
            expected_rows: int | None = None
            row_count_source: str | None = None
            if output_path.suffix.lower() == ".csv":
                observed_rows = int(len(pd.read_csv(output_path, usecols=[0])))
                if output_key in manifest_rows:
                    expected_rows = int(manifest_rows[output_key])
                    row_count_source = "manifest_output_row_counts"
                elif output_key in legacy_manifest_rows:
                    expected_rows = int(legacy_manifest_rows[output_key])
                    row_count_source = "manifest_legacy_counts"
                elif output_key in legacy:
                    legacy_contract = legacy[output_key]
                    if isinstance(legacy_contract, dict) and legacy_contract.get(
                        "rows"
                    ) is not None:
                        expected_rows = int(legacy_contract["rows"])
                        row_count_source = "config_legacy_lock"
                if expected_rows is None:
                    raise ValueError(
                        f"{stage}.{output_key} has no frozen row-count contract"
                    )
                if observed_rows != expected_rows:
                    raise ValueError(f"{stage}.{output_key} row-count contract failed")
            output_paths[output_key] = str(output_path)
            audit_rows.append(
                {
                    "stage": stage,
                    "manifest_version": manifest.get("version"),
                    "manifest_status": manifest.get("status"),
                    "output_key": output_key,
                    "path": str(output_path),
                    "sha256": observed_hash,
                    "hash_contract_source": hash_source,
                    "row_count_contract_source": row_count_source,
                    "expected_rows": expected_rows,
                    "observed_rows": observed_rows,
                    "contract_passed": True,
                }
            )
        audited[stage] = {
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "manifest_sha256": observed_manifest_hash,
            "manifest_hash_contract_source": (
                "config_lock" if expected_manifest_hash is not None else "observed_only"
            ),
            "output_paths": output_paths,
        }
        logger.info(
            "source_stage=%s version=%s status=%s outputs=%d manifest_sha256=%s",
            stage,
            manifest.get("version"),
            manifest.get("status"),
            len(output_paths),
            observed_manifest_hash,
        )
    return audited, pd.DataFrame(audit_rows)


def validate_global_guardrails(
    config: Mapping[str, Any], audited: Mapping[str, Mapping[str, Any]]
) -> None:
    guardrails = config["guardrails"]
    if guardrails.get("causal_claim_allowed") is not False:
        raise ValueError("Stage 10 must freeze causal_claim_allowed=false")
    if guardrails.get("publication_eligible_without_human_validation") is not False:
        raise ValueError("Human validation publication gate must remain closed")
    for stage in ("causal_ml", "robustness", "diagnostics", "significance"):
        manifest = audited[stage]["manifest"]
        if manifest.get("causal_claim_allowed") is not False:
            raise ValueError(f"{stage} unexpectedly authorizes a causal claim")
    significance = audited["significance"]["manifest"]
    if significance.get("claim_gates_all_passed") is not True:
        raise ValueError("Stage 09c claim-gate audit did not pass")
    if significance.get("source_contracts_all_passed") is not True:
        raise ValueError("Stage 09c source contracts did not pass")


def complete_residual_model_family(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain upstream families and fill only genuinely missing legacy values."""

    required = {"analysis_source", "model_id"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "Residual diagnostics lack required model-family keys: "
            f"{sorted(missing_columns)}"
        )
    result = frame.copy()
    if "model_family" in result.columns:
        upstream = result["model_family"].astype("string").str.strip()
        upstream = upstream.mask(upstream.eq(""), pd.NA)
    else:
        upstream = pd.Series(pd.NA, index=result.index, dtype="string")

    derived = (
        result["model_id"]
        .astype("string")
        .str.split("__", n=1)
        .str[0]
        .replace({"rf": "random_forest"})
    )
    stage09_mask = result["analysis_source"].astype(str).eq(
        "stage09_common_cohort_ridge"
    )
    derived = derived.mask(stage09_mask, "ridge")
    result["model_family"] = upstream.fillna(derived)

    unresolved = result["model_family"].isna() | result["model_family"].astype(
        "string"
    ).str.strip().eq("")
    if bool(unresolved.any()):
        examples = (
            result.loc[unresolved, ["analysis_source", "model_id"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(f"Could not derive residual model family: {examples}")
    return result


def any_adjusted_rejection(frame: pd.DataFrame) -> pd.Series:
    candidates = [
        column
        for column in frame.columns
        if (
            column.startswith("reject_")
            and ("adjust" in column or "fdr" in column or "holm" in column)
        )
        or column in {"significant_fdr", "simultaneous_band_significant"}
    ]
    result = pd.Series(False, index=frame.index)
    for column in candidates:
        result |= strict_bool_series(frame[column], column)
    return result


def minimum_adjusted_p(frame: pd.DataFrame) -> float:
    candidates = [
        column
        for column in frame.columns
        if column.startswith("p_value_")
        and any(token in column for token in ("adjust", "fdr", "holm"))
    ]
    values: list[float] = []
    for column in candidates:
        values.extend(pd.to_numeric(frame[column], errors="coerce").dropna().tolist())
    return float(min(values)) if values else float("nan")


def summarize_sensitivity(
    datasets: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for evidence_stream, frame in datasets.items():
        frequencies: Iterable[Any]
        if "frequency" in frame.columns:
            frequencies = sorted(frame["frequency"].dropna().unique())
        else:
            frequencies = ["not_applicable"]
        for frequency in frequencies:
            subset = (
                frame.loc[frame["frequency"] == frequency]
                if "frequency" in frame.columns
                else frame
            )
            rejection = any_adjusted_rejection(subset)
            row: dict[str, Any] = {
                "evidence_stream": evidence_stream,
                "frequency": frequency,
                "frequency_role": (
                    "primary" if frequency == "monthly" else "robustness"
                ),
                "rows_or_tests": int(len(subset)),
                "adjusted_rejections_any_registered_stream": int(rejection.sum()),
                "minimum_adjusted_p_value": minimum_adjusted_p(subset),
                "causal_claim_allowed": False,
            }
            if "rmse_improvement_percent" in subset.columns:
                values = pd.to_numeric(
                    subset["rmse_improvement_percent"], errors="coerce"
                )
                row["median_rmse_improvement_percent"] = float(values.median())
            if "spearman_correlation" in subset.columns:
                values = pd.to_numeric(subset["spearman_correlation"], errors="coerce")
                row["median_spearman_correlation"] = float(values.median())
                row["minimum_spearman_correlation"] = float(values.min())
            rows.append(row)
    return pd.DataFrame(rows)


def select_forecast_trajectories(
    predictions: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Freeze the exact one-step OOS trajectories displayed in Figures 10–11."""
    design = config["forecast_trajectory_figures"]
    required = {
        "frequency",
        "algorithm",
        "index_construction",
        "source_type",
        "score_source",
        "signal_name",
        "horizon_steps",
        "target_date",
        "actual_scfi_level",
        "predicted_scfi_level",
        "model_id",
        "origin_date",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"Nonlinear predictions lack trajectory columns: {sorted(missing)}"
        )
    selected = predictions.loc[
        predictions["horizon_steps"].eq(int(design["horizon_steps"]))
        & predictions["source_type"].eq(design["source_type"])
        & predictions["score_source"].eq(design["score_source"])
        & predictions["algorithm"].isin(design["algorithms"])
        & predictions["signal_name"].isin(design["signals"])
        & (
            (
                predictions["frequency"].eq("weekly")
                & predictions["index_construction"].isin(
                    design["weekly_constructions"]
                )
            )
            | (
                predictions["frequency"].eq("monthly")
                & predictions["index_construction"].isin(
                    design["monthly_constructions"]
                )
            )
        )
    ].copy()
    if selected.empty:
        raise ValueError("No predictions satisfy the frozen trajectory design")
    selected["target_date"] = pd.to_datetime(
        selected["target_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    selected["origin_date"] = pd.to_datetime(
        selected["origin_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    selected["forecast_path"] = selected["signal_name"].map(
        {
            "overall_disruption_max": "peak",
            "dimension_mean7": "breadth",
        }
    )
    key = [
        "frequency",
        "index_construction",
        "algorithm",
        "forecast_path",
        "target_date",
    ]
    if selected.duplicated(key).any():
        examples = selected.loc[selected.duplicated(key, keep=False), key].head(5)
        raise ValueError(
            "Forecast trajectory key is not unique: "
            + examples.to_dict(orient="records").__repr__()
        )
    expected_cells = (
        len(design["algorithms"])
        * len(design["signals"])
        * (
            len(design["weekly_constructions"])
            + len(design["monthly_constructions"])
        )
    )
    observed_cells = selected[
        ["frequency", "index_construction", "algorithm", "forecast_path"]
    ].drop_duplicates()
    if len(observed_cells) != expected_cells:
        raise ValueError(
            f"Expected {expected_cells} forecast trajectory cells, "
            f"observed {len(observed_cells)}"
        )
    actual_consistency = selected.groupby(
        ["frequency", "index_construction", "target_date"], dropna=False
    )["actual_scfi_level"].nunique(dropna=False)
    if not actual_consistency.eq(1).all():
        raise ValueError("Actual SCFI level differs within a trajectory target date")
    selected["frequency_role"] = np.where(
        selected["frequency"].eq("monthly"), "primary", "robustness"
    )
    selected["level_error_observed_minus_forecast"] = (
        pd.to_numeric(selected["actual_scfi_level"], errors="raise")
        - pd.to_numeric(selected["predicted_scfi_level"], errors="raise")
    )
    selected["absolute_level_error"] = selected[
        "level_error_observed_minus_forecast"
    ].abs()
    selected["squared_level_error"] = selected[
        "level_error_observed_minus_forecast"
    ].pow(2)
    selected["forecast_scope"] = design["interpretation"]
    selected["causal_claim_allowed"] = False
    columns = [
        "frequency",
        "frequency_role",
        "index_construction",
        "algorithm",
        "forecast_path",
        "signal_name",
        "horizon_steps",
        "origin_date",
        "target_date",
        "actual_scfi_level",
        "predicted_scfi_level",
        "level_error_observed_minus_forecast",
        "absolute_level_error",
        "squared_level_error",
        "model_id",
        "forecast_scope",
        "causal_claim_allowed",
    ]
    return selected.loc[:, columns].sort_values(key).reset_index(drop=True)


def build_tables(
    config: Mapping[str, Any],
    audited: Mapping[str, Mapping[str, Any]],
    source_audit: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    manifests = {stage: item["manifest"] for stage, item in audited.items()}
    status_rows: list[dict[str, Any]] = []
    for stage, manifest in manifests.items():
        status_rows.append(
            {
                "stage": stage,
                "version": manifest.get("version"),
                "status": manifest.get("status"),
                "publication_scope_complete": manifest.get(
                    "publication_scope_complete", "not_recorded"
                ),
                "human_validation_required": True,
                "publication_eligible_without_human_validation": False,
                "causal_claim_allowed": False,
                "role": (
                    "monthly_primary"
                    if stage == "monthly_llm"
                    else "weekly_higher_frequency_robustness"
                    if stage == "weekly_llm"
                    else "supporting_analysis"
                ),
            }
        )
    status_rows.append(
        {
            "stage": "human_validation",
            "version": "not_yet_completed",
            "status": "pending_completed_dual_annotator_gold_set",
            "publication_scope_complete": False,
            "human_validation_required": True,
            "publication_eligible_without_human_validation": False,
            "causal_claim_allowed": False,
            "role": "mandatory_publication_gate",
        }
    )

    weekly_rank = read_csv_output(audited, "weekly_llm", "llm_ranking")
    weekly_rank["frequency"] = "weekly"
    weekly_rank["frequency_role"] = "robustness"
    weekly_rank["formal_superiority_claim_allowed"] = False
    monthly_rank = read_csv_output(audited, "monthly_llm", "llm_ranking")
    monthly_rank["frequency"] = "monthly"
    monthly_rank["frequency_role"] = "primary"
    monthly_rank["formal_superiority_claim_allowed"] = False
    monthly_primary_rank = monthly_rank.loc[
        monthly_rank["index_construction"].eq("adapted_pdf_sqrt_ema3")
    ].copy()
    if monthly_primary_rank.empty:
        raise ValueError("Monthly LLM ranking lacks adapted_pdf_sqrt_ema3 primary rows")

    weekly_predictor = read_csv_output(
        audited, "weekly_llm", "primary_predictor_ranking"
    )
    weekly_predictor["frequency"] = "weekly"
    weekly_predictor["frequency_role"] = "robustness"
    monthly_predictor = read_csv_output(
        audited, "monthly_llm", "primary_predictor_ranking"
    )
    monthly_predictor["frequency"] = "monthly"
    monthly_predictor["frequency_role"] = "primary"
    predictor_ranking = pd.concat(
        [weekly_predictor, monthly_predictor], ignore_index=True, sort=False
    )

    primary = read_csv_output(audited, "significance", "primary_forecast_inference")
    primary_compact = compact(
        primary,
        [
            "hypothesis_id",
            "frequency",
            "frequency_role",
            "horizon",
            "horizon_unit",
            "baseline_model",
            "challenger_model",
            "signal_name",
            "score_source",
            "test_method",
            "effect_metric",
            "effect_size",
            "standard_error",
            "ci_lower",
            "ci_upper",
            "sample_size",
            "raw_p_value",
            "correction_family_id",
            "correction_method",
            "adjusted_p_value_within_family",
            "reject_within_family_at_alpha",
            "adjusted_significant_positive_improvement",
            "adjusted_significant_worsening",
            "supporting_rmse_improvement_percent",
            "supporting_incremental_oos_r_squared",
            "primary_claim_interpretation",
            "causal_claim_allowed",
        ],
    )

    nonlinear = read_csv_output(audited, "nonlinear_forecasts", "pipeline_ranking")
    nonlinear["frequency_is_primary"] = nonlinear["frequency"].eq("monthly")
    nonlinear["ranking_is_point_estimate_not_significance"] = True
    nonlinear_predictions = read_csv_output(
        audited, "nonlinear_forecasts", "predictions"
    )
    forecast_trajectories = select_forecast_trajectories(
        nonlinear_predictions, config
    )

    local_projection = read_csv_output(
        audited, "time_series_causality", "local_projections"
    )
    local_projection["frequency_role"] = np.where(
        local_projection["frequency"].eq("monthly"), "primary", "robustness"
    )
    local_projection["causal_claim_allowed"] = False

    conditional = read_csv_output(
        audited, "time_series_causality", "conditional_granger_tests"
    )
    conventional = read_csv_output(
        audited, "time_series_causality", "conventional_granger_tests"
    )
    conditional["granger_design"] = "conditional"
    conventional["granger_design"] = "conventional"
    granger = pd.concat([conditional, conventional], ignore_index=True, sort=False)
    granger["frequency_role"] = np.where(
        granger["frequency"].eq("monthly"), "primary", "robustness"
    )
    granger["causal_claim_allowed"] = False

    pcmci = read_csv_output(audited, "time_series_causality", "pcmci_links")
    pcmci["significant_fdr_bool"] = strict_bool_series(
        pcmci["significant_fdr"], "significant_fdr"
    )
    pcmci_summary = (
        pcmci.groupby(["frequency", "msdi_path", "source", "target"], dropna=False)
        .agg(
            lags_tested=("source_lag", "size"),
            fdr_significant_links=("significant_fdr_bool", "sum"),
            minimum_raw_p_value=("p_value", "min"),
            minimum_fdr_p_value=("p_value_fdr_bh", "min"),
            maximum_absolute_partial_correlation=(
                "partial_correlation",
                lambda values: pd.to_numeric(values, errors="coerce").abs().max(),
            ),
        )
        .reset_index()
    )
    pcmci_summary["frequency_role"] = np.where(
        pcmci_summary["frequency"].eq("monthly"), "primary", "robustness"
    )
    pcmci_summary["interpretation"] = "temporal_dependence_not_structural_causality"

    dml = read_csv_output(audited, "causal_ml", "dml_estimates")
    overlap = read_csv_output(audited, "causal_ml", "overlap_diagnostics")
    placebo = read_csv_output(audited, "causal_ml", "placebo_tests")
    keys = ["frequency", "frequency_role", "msdi_path", "horizon", "learner"]
    overlap_columns = keys + [
        column
        for column in (
            "conditional_variation_ratio",
            "residual_information_effective_n",
            "maximum_single_observation_information_share",
            "weak_conditional_variation_flag",
            "influence_concentration_warning_flag",
            "training_support_extrapolation_warning_flag",
        )
        if column in overlap.columns
    ]
    placebo_columns = keys + [
        column
        for column in (
            "p_value_fdr_all",
            "falsification_flag_raw",
            "falsification_flag_fdr",
            "falsification_status",
        )
        if column in placebo.columns
    ]
    placebo_compact = placebo.loc[:, placebo_columns].rename(
        columns={
            column: f"placebo_{column}"
            for column in placebo_columns
            if column not in keys
        }
    )
    dml_diagnostics = dml.merge(
        overlap.loc[:, overlap_columns], on=keys, how="left", validate="one_to_one"
    ).merge(placebo_compact, on=keys, how="left", validate="one_to_one")
    dml_diagnostics["causal_claim_allowed"] = False
    dml_diagnostics["interpretation_guardrail"] = config["guardrails"][
        "dml_interpretation"
    ]

    heterogeneity = read_csv_output(
        audited, "causal_ml", "heterogeneous_group_effects"
    )
    heterogeneity["causal_forest_estimate"] = False
    heterogeneity["causal_claim_allowed"] = False
    heterogeneity["interpretation_guardrail"] = (
        "pre_specified_GATE_BLP_or_guarded_skip_not_causal_forest"
    )

    robustness_frames = {
        "ablation": read_csv_output(audited, "robustness", "ablation_tests"),
        "index_stability": read_csv_output(
            audited, "robustness", "index_stability"
        ),
        "forecast_regime": read_csv_output(
            audited, "robustness", "forecast_regime_sensitivity"
        ),
        "peak_breadth_difference": read_csv_output(
            audited, "robustness", "peak_breadth_difference"
        ),
        "causal_ml_sensitivity": read_csv_output(
            audited, "robustness", "causal_ml_sensitivity"
        ),
        "local_projection_sensitivity": read_csv_output(
            audited, "robustness", "local_projection_sensitivity"
        ),
        "conditional_granger_sensitivity": read_csv_output(
            audited, "robustness", "conditional_granger_sensitivity"
        ),
        "conventional_granger_sensitivity": read_csv_output(
            audited, "robustness", "conventional_granger_sensitivity"
        ),
        "pcmci_link_stability": read_csv_output(
            audited, "robustness", "pcmci_link_stability"
        ),
    }
    robustness_summary = summarize_sensitivity(robustness_frames)

    errors = read_csv_output(audited, "diagnostics", "forecast_error_summary")
    errors["primary_family_bool"] = strict_bool_series(
        errors["primary_family"], "forecast_error_summary.primary_family"
    )
    errors["is_core_diagnostic_bool"] = strict_bool_series(
        errors["is_core_diagnostic"], "forecast_error_summary.is_core_diagnostic"
    )
    error_core = errors.loc[errors["is_core_diagnostic_bool"]].copy()
    error_summary = (
        error_core.groupby(
            ["frequency", "analysis_source", "model_family"], dropna=False
        )
        .agg(
            model_horizon_cells=("model_id", "size"),
            median_bias=("bias", "median"),
            median_mae=("mae", "median"),
            median_rmse=("rmse", "median"),
            median_directional_accuracy=("directional_accuracy", "median"),
        )
        .reset_index()
    )
    residual = read_csv_output(audited, "diagnostics", "forecast_residual_tests")
    residual["primary_family_bool"] = strict_bool_series(
        residual["primary_family"], "forecast_residual_tests.primary_family"
    )
    residual["reject_raw_bool"] = strict_bool_series(
        residual["reject_raw_at_alpha"], "forecast_residual_tests.reject_raw"
    )
    residual = complete_residual_model_family(residual)
    residual_core = residual.copy()
    residual_summary = (
        residual_core.groupby(
            ["frequency", "analysis_source", "model_family", "diagnostic"],
            dropna=False,
        )
        .agg(
            diagnostic_tests=("diagnostic_id", "size"),
            raw_warnings=("reject_raw_bool", "sum"),
            raw_warning_rate=("reject_raw_bool", "mean"),
            minimum_raw_p_value=("raw_p_value", "min"),
        )
        .reset_index()
    )
    residual_rollup = (
        residual_summary.groupby(
            ["frequency", "analysis_source", "model_family"], dropna=False
        )
        .agg(
            residual_tests=("diagnostic_tests", "sum"),
            raw_residual_warnings=("raw_warnings", "sum"),
            raw_residual_warning_rate=("raw_warning_rate", "mean"),
        )
        .reset_index()
    )
    diagnostic_merge_keys = ["frequency", "analysis_source", "model_family"]
    merge_audit = error_summary.loc[:, diagnostic_merge_keys].merge(
        residual_rollup.loc[:, diagnostic_merge_keys],
        on=diagnostic_merge_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merge_audit["_merge"].ne("both")
    if bool(unmatched.any()):
        examples = (
            merge_audit.loc[unmatched, diagnostic_merge_keys + ["_merge"]]
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "Forecast-error and residual diagnostic cells do not match: "
            f"{examples}"
        )
    diagnostic_summary = error_summary.merge(
        residual_rollup,
        on=diagnostic_merge_keys,
        how="left",
        validate="one_to_one",
    )
    diagnostic_summary["warning_interpretation"] = (
        "model_adequacy_warning_not_economic_hypothesis"
    )

    domain = read_csv_output(
        audited, "significance", "significance_domain_summary"
    )
    uncertainty = read_csv_output(
        audited, "significance", "uncertainty_effect_size_audit"
    )
    claims = read_csv_output(audited, "significance", "primary_claim_summary")
    monthly_construction = read_csv_output(
        audited, "monthly_llm", "construction_summary"
    )

    main_tables = {
        "T01_analysis_status": pd.DataFrame(status_rows),
        "T02_monthly_llm_point_ranking": monthly_primary_rank,
        "T03_predictor_ranking": predictor_ranking,
        "T04_primary_forecast_inference": primary_compact,
        "T05_nonlinear_pipeline_ranking": nonlinear,
        "T06_local_projections": local_projection,
        "T07_granger_tests": granger,
        "T08_pcmci_summary": pcmci_summary,
        "T09_dml_effects_diagnostics": dml_diagnostics,
        "T10_guarded_heterogeneity": heterogeneity,
        "T11_robustness_sensitivity": robustness_summary,
        "T12_forecast_diagnostics": diagnostic_summary,
        "T13_significance_domains": domain,
        "T14_uncertainty_audit": uncertainty,
        "T15_primary_claim_summary": claims,
        "T16_monthly_constructions": monthly_construction,
    }
    supplements = {
        "S00_weekly_llm_point_ranking": weekly_rank,
        "S00b_all_monthly_llm_point_rankings": monthly_rank,
        "S01_full_ablation": robustness_frames["ablation"],
        "S02_forecast_regime": robustness_frames["forecast_regime"],
        "S03_significant_pcmci_links": pcmci.loc[
            pcmci["significant_fdr_bool"]
        ].drop(columns=["significant_fdr_bool"]),
        "S04_significance_families": read_csv_output(
            audited, "significance", "significance_family_summary"
        ),
        "S05_claim_gate_audit": read_csv_output(
            audited, "significance", "claim_gate_audit"
        ),
        "S06_source_contract_audit": source_audit,
        "S07_primary_residual_tests": residual_core.drop(
            columns=["primary_family_bool", "reject_raw_bool"]
        ),
        "S08_one_step_forecasts": forecast_trajectories,
    }
    return main_tables, supplements


def configure_plot_style(config: Mapping[str, Any]) -> None:
    style = str(config["figures"].get("style", "default"))
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 13,
            "savefig.bbox": "tight",
        }
    )


def save_figure(
    fig: plt.Figure,
    figures_dir: Path,
    basename: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    paths: list[Path] = []
    figures_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = figures_dir / f"{basename}.{fmt}"
        temporary = path.with_name(path.name + ".tmp")
        fig.savefig(temporary, format=fmt, dpi=dpi, metadata={"Creator": "Stage 10"})
        os.replace(temporary, path)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_timeline(panel: pd.DataFrame, colors: Mapping[str, str]) -> plt.Figure:
    date_column = "month_ending"
    scfi_column = "scfi_monthly_mean_log"
    peak_column = (
        "llm_monthly_signal__adapted_pdf_sqrt_ema3__score_median__"
        "overall_disruption_max"
    )
    breadth_column = (
        "llm_monthly_signal__adapted_pdf_sqrt_ema3__score_median__dimension_mean7"
    )
    required = {date_column, scfi_column, peak_column, breadth_column}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Monthly primary panel lacks timeline columns: {sorted(missing)}")
    data = panel.loc[:, sorted(required)].copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="raise")
    series = {
        "SCFI (log)": scfi_column,
        "Peak MSDI (median overall max)": peak_column,
        "Breadth MSDI (mean of 7 dimensions)": breadth_column,
    }
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    palette = [colors["primary"], colors["secondary"], "#2f855a"]
    for (label, column), color in zip(series.items(), palette):
        ax.plot(data[date_column], safe_zscore(data[column]), label=label, color=color, lw=1.35)
    ax.axhline(0.0, color="#6b7280", lw=0.7)
    ax.set(ylabel="Standardized value (descriptive)", xlabel="")
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    return fig


def plot_llm_ranking(ranking: pd.DataFrame, colors: Mapping[str, str]) -> plt.Figure:
    data = ranking.sort_values("predictive_point_rank", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    values = pd.to_numeric(data["mean_relative_rmse_vs_coverage"], errors="raise")
    labels = data["score_source"].map(publication_label)
    bars = ax.barh(labels, values, color=colors["primary"])
    ax.axvline(1.0, color=colors["warning"], linestyle="--", lw=1.0, label="Coverage-only baseline")
    for bar, rank in zip(bars, data["predictive_point_rank"]):
        ax.text(
            bar.get_width() + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"Rank {int(rank)}",
            va="center",
        )
    ax.set(
        xlabel="Mean RMSE relative to coverage-only baseline (lower is better)",
        ylabel="",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_primary_effects(primary: pd.DataFrame, colors: Mapping[str, str]) -> plt.Figure:
    family = primary.loc[
        primary["correction_family_id"].astype(str).str.contains(
            "monthly_primary_hybrid_vs_coverage__diebold_mariano_squared",
            regex=False,
        )
    ].copy()
    if family.empty:
        raise ValueError("Primary monthly hybrid-versus-coverage family is unavailable")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    palette = [colors["primary"], colors["secondary"], "#2f855a", "#805ad5"]
    for ax, signal in zip(axes, ["overall_disruption_max", "dimension_mean7"]):
        subset = family.loc[family["signal_name"] == signal]
        for color, (source, group) in zip(palette, subset.groupby("score_source")):
            group = group.sort_values("horizon")
            ax.plot(
                group["horizon"],
                pd.to_numeric(group["effect_size"], errors="coerce") * 1000,
                marker="o",
                lw=1.3,
                label=publication_label(source),
                color=color,
            )
            rejected = strict_bool_series(
                group["reject_within_family_at_alpha"], "primary rejection"
            )
            ax.scatter(
                group.loc[rejected, "horizon"],
                pd.to_numeric(group.loc[rejected, "effect_size"], errors="coerce") * 1000,
                marker="x",
                s=42,
                color=color,
                zorder=4,
            )
        ax.axhline(0.0, color="#111827", lw=0.8)
        ax.set_title("Peak path" if signal == "overall_disruption_max" else "Breadth path")
        ax.set_xlabel("Forecast horizon (months)")
    axes[0].set_ylabel("Baseline loss − challenger loss (×1,000)")
    axes[1].legend(loc="best")
    fig.tight_layout()
    return fig


def plot_local_projections(lp: pd.DataFrame, colors: Mapping[str, str]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharey=False)
    for row, frequency in enumerate(["weekly", "monthly"]):
        for column, path in enumerate(["peak", "breadth"]):
            ax = axes[row, column]
            data = lp.loc[(lp["frequency"] == frequency) & (lp["msdi_path"] == path)].sort_values("horizon")
            x = pd.to_numeric(data["horizon"], errors="raise").to_numpy(dtype=float)
            y = pd.to_numeric(data["effect_percent"], errors="raise").to_numpy(dtype=float)
            lower = pd.to_numeric(data["simultaneous_lower_percent"], errors="coerce").to_numpy(dtype=float)
            upper = pd.to_numeric(data["simultaneous_upper_percent"], errors="coerce").to_numpy(dtype=float)
            ax.plot(x, y, color=colors["primary"], marker="o", lw=1.25)
            if np.isfinite(lower).any() and np.isfinite(upper).any():
                ax.fill_between(x, lower, upper, color=colors["primary"], alpha=0.18, label="Simultaneous band")
            ax.axhline(0.0, color="#111827", lw=0.7)
            ax.set_title(
                f"{publication_label(frequency)}: {publication_label(path).lower()}"
            )
            ax.set_xlabel(f"Horizon ({'weeks' if frequency == 'weekly' else 'months'})")
            ax.set_ylabel("SCFI response (%)")
    fig.tight_layout()
    return fig


def plot_dml(dml: pd.DataFrame, colors: Mapping[str, str]) -> plt.Figure:
    primary_learner = strict_bool_series(dml["primary_learner"], "primary_learner")
    data = dml.loc[primary_learner].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharey=False)
    for row, frequency in enumerate(["weekly", "monthly"]):
        for column, path in enumerate(["peak", "breadth"]):
            ax = axes[row, column]
            subset = data.loc[(data["frequency"] == frequency) & (data["msdi_path"] == path)].sort_values("horizon")
            x = pd.to_numeric(subset["horizon"], errors="raise").to_numpy(dtype=float)
            y = pd.to_numeric(subset["effect_percent_per_one_sd_innovation"], errors="raise").to_numpy(dtype=float)
            lower = pd.to_numeric(subset["ci_lower_percent"], errors="coerce").to_numpy(dtype=float)
            upper = pd.to_numeric(subset["ci_upper_percent"], errors="coerce").to_numpy(dtype=float)
            ax.plot(x, y, color=colors["secondary"], marker="o", lw=1.25)
            ax.fill_between(x, lower, upper, color=colors["secondary"], alpha=0.18, label="95% HAC CI")
            ax.axhline(0.0, color="#111827", lw=0.7)
            ax.set_title(
                f"{publication_label(frequency)}: {publication_label(path).lower()}"
            )
            ax.set_xlabel(f"Horizon ({'weeks' if frequency == 'weekly' else 'months'})")
            ax.set_ylabel("SCFI association per 1-SD innovation (%)")
    fig.tight_layout()
    return fig


def plot_temporal_evidence(
    granger: pd.DataFrame, pcmci: pd.DataFrame, colors: Mapping[str, str]
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    conditional = granger.loc[granger["granger_design"] == "conditional"].copy()
    adjusted = pd.to_numeric(conditional["p_value_holm"], errors="coerce").clip(lower=1e-12)
    labels = conditional.apply(granger_label, axis=1)
    order = np.argsort(-np.log10(adjusted.to_numpy()))
    axes[0].barh(np.arange(len(order)), -np.log10(adjusted.iloc[order]), color=colors["primary"])
    axes[0].set_yticks(np.arange(len(order)), labels.iloc[order], fontsize=6)
    axes[0].axvline(-math.log10(0.05), color=colors["warning"], linestyle="--", lw=1)
    axes[0].set(title="Conditional Granger tests", xlabel="−log10(Holm-adjusted p)")

    counts = (
        pcmci.groupby(["frequency", "msdi_path"])["significant_fdr_bool"]
        .sum()
        .unstack(fill_value=0)
    )
    counts = counts.rename(index=publication_label, columns=publication_label)
    counts.plot(kind="bar", ax=axes[1], color=[colors["primary"], colors["secondary"]])
    axes[1].set(title="FDR-significant PCMCI links", xlabel="Frequency", ylabel="Link-lag count")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="MSDI path")
    fig.tight_layout()
    return fig


def plot_ablation(ablation: pd.DataFrame, colors: Mapping[str, str]) -> plt.Figure:
    data = ablation.copy()
    data["rmse_improvement_percent"] = pd.to_numeric(data["rmse_improvement_percent"], errors="coerce")
    comparisons = data["comparison_id"].value_counts().head(8).index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6), sharey=True)
    for ax, frequency in zip(axes, ["weekly", "monthly"]):
        subset = data.loc[(data["frequency"] == frequency) & data["comparison_id"].isin(comparisons)]
        arrays = [
            subset.loc[subset["comparison_id"] == comparison, "rmse_improvement_percent"].dropna().to_numpy()
            for comparison in comparisons
        ]
        ax.boxplot(
            arrays,
            orientation="horizontal",
            tick_labels=[ablation_label(value) for value in comparisons],
            showfliers=False,
        )
        ax.axvline(0.0, color="#111827", lw=0.8)
        ax.set_title(
            f"{publication_label(frequency)} "
            f"({'primary' if frequency == 'monthly' else 'robustness'})"
        )
        ax.set_xlabel("Variant RMSE improvement over baseline (%)")
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    return fig


def plot_diagnostics(
    errors: pd.DataFrame, residual: pd.DataFrame, colors: Mapping[str, str]
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    error_plot = errors.groupby(["frequency", "model_family"], dropna=False)["median_rmse"].median().unstack(0)
    error_plot = error_plot.rename(index=publication_label, columns=publication_label)
    error_plot.plot(kind="bar", ax=axes[0], color=[colors["secondary"], colors["primary"]])
    axes[0].set(title="Median forecast RMSE", xlabel="Model family", ylabel="RMSE")
    axes[0].legend(title="Frequency")
    axes[0].tick_params(axis="x", rotation=30)
    residual_data = residual.copy()
    if "raw_warning_rate" not in residual_data.columns:
        residual_data["raw_warning_rate"] = strict_bool_series(
            residual_data["reject_raw_at_alpha"], "diagnostic raw warning"
        ).astype(float)
    residual_plot = residual_data.groupby(
        ["diagnostic", "frequency"], dropna=False
    )["raw_warning_rate"].mean().unstack(1)
    residual_plot = residual_plot.rename(
        index=publication_label, columns=publication_label
    )
    residual_plot.plot(kind="bar", ax=axes[1], color=[colors["secondary"], colors["primary"]])
    axes[1].set(title="Residual diagnostic warning rate", xlabel="Diagnostic", ylabel="Raw warning share")
    axes[1].legend(title="Frequency")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_var_irf(irf: pd.DataFrame, colors: Mapping[str, str]) -> plt.Figure:
    data = irf.loc[
        (irf["impulse"] == "msdi_change") & (irf["response"] == "scfi_return")
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0), sharey=False)
    for row, frequency in enumerate(["weekly", "monthly"]):
        for column, path in enumerate(["peak", "breadth"]):
            ax = axes[row, column]
            subset = data.loc[(data["frequency"] == frequency) & (data["msdi_path"] == path)].sort_values("horizon")
            diagnostic_pass = strict_bool_series(subset["diagnostics_pass"], "VAR diagnostics").all()
            color = colors["primary"] if diagnostic_pass else colors["robustness"]
            linestyle = "-" if diagnostic_pass else "--"
            ax.plot(subset["horizon"], subset["cumulative_scfi_percent"], color=color, linestyle=linestyle, marker="o", lw=1.2)
            ax.axhline(0.0, color="#111827", lw=0.7)
            status = "passed" if diagnostic_pass else "failed"
            ax.set_title(
                f"{publication_label(frequency)}: "
                f"{publication_label(path).lower()} (diagnostics {status})"
            )
            ax.set_xlabel("Horizon")
            ax.set_ylabel("Cumulative SCFI response (%)")
    fig.tight_layout()
    return fig


def trajectory_panel_data(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["target_date"] = pd.to_datetime(data["target_date"], errors="raise")
    actual = data.groupby("target_date", as_index=True)["actual_scfi_level"].first()
    predictions = data.pivot(
        index="target_date",
        columns="forecast_path",
        values="predicted_scfi_level",
    )
    required_paths = {"peak", "breadth"}
    missing = required_paths.difference(predictions.columns)
    if missing:
        raise ValueError(f"Forecast panel lacks path(s): {sorted(missing)}")
    panel = actual.rename("actual").to_frame().join(predictions, how="inner")
    panel = panel.loc[:, ["actual", "peak", "breadth"]].dropna().sort_index()
    if panel.empty:
        raise ValueError("Forecast panel has no complete actual/predicted dates")
    return panel


def draw_forecast_trajectory_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    colors: Mapping[str, str],
) -> None:
    panel = trajectory_panel_data(frame)
    ax.plot(
        panel.index,
        panel["actual"],
        color="#111827",
        lw=1.7,
        label="Actual SCFI",
        zorder=4,
    )
    ax.plot(
        panel.index,
        panel["peak"],
        color=colors["secondary"],
        lw=1.15,
        alpha=0.9,
        label="Peak-based forecast",
    )
    ax.plot(
        panel.index,
        panel["breadth"],
        color="#2f855a",
        lw=1.15,
        alpha=0.9,
        label="Breadth-based forecast",
    )
    ax.set_title(title)
    ax.set_ylabel("SCFI points")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.margins(x=0.01)


def make_weekly_panel_grid(
    number_of_panels: int,
    *,
    width: float,
    height_per_row: float,
    sharex: bool,
    sharey: bool,
) -> tuple[plt.Figure, np.ndarray, int, int]:
    """Create the smallest two-column grid and leave hiding spare axes to caller."""

    if number_of_panels < 1:
        raise ValueError("A weekly figure requires at least one algorithm panel")
    columns = min(2, number_of_panels)
    rows = int(math.ceil(number_of_panels / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(width, height_per_row * rows),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )
    return figure, axes, rows, columns


def hide_unused_panel_axes(axes: np.ndarray, used: int) -> None:
    for axis in list(axes.flat)[used:]:
        axis.set_visible(False)


def plot_weekly_forecast_trajectories(
    trajectories: pd.DataFrame,
    design: Mapping[str, Any],
    colors: Mapping[str, str],
) -> plt.Figure:
    data = trajectories.loc[trajectories["frequency"].eq("weekly")].copy()
    constructions = list(design["weekly_constructions"])
    if len(constructions) != 1:
        raise ValueError("Weekly trajectory figure requires one frozen construction")
    data = data.loc[data["index_construction"].eq(constructions[0])]
    algorithms = list(design["algorithms"])
    fig, axes, rows, columns = make_weekly_panel_grid(
        len(algorithms),
        width=12.2,
        height_per_row=3.6,
        sharex=True,
        sharey=True,
    )
    panel_letters = "abcdefghijklmnopqrstuvwxyz"
    for position, (ax, algorithm) in enumerate(zip(axes.flat, algorithms)):
        subset = data.loc[data["algorithm"].eq(algorithm)]
        draw_forecast_trajectory_panel(
            ax,
            subset,
            f"({panel_letters[position]}) {publication_label(algorithm)}",
            colors,
        )
        ax.set_xlabel(
            "Target week" if position // columns == rows - 1 else ""
        )
    hide_unused_panel_axes(axes, len(algorithms))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
    )
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_monthly_forecast_trajectories(
    trajectories: pd.DataFrame,
    design: Mapping[str, Any],
    colors: Mapping[str, str],
) -> plt.Figure:
    data = trajectories.loc[trajectories["frequency"].eq("monthly")].copy()
    algorithms = list(design["algorithms"])
    constructions = list(design["monthly_constructions"])
    if len(constructions) != 2:
        raise ValueError("Monthly trajectory figure requires two frozen constructions")
    fig, axes = plt.subplots(
        len(algorithms),
        len(constructions),
        figsize=(13.0, 10.5),
        sharex=True,
        sharey=True,
    )
    panel_letters = "abcdefghijklmnopqrstuvwxyz"
    position = 0
    for row, algorithm in enumerate(algorithms):
        for column, construction in enumerate(constructions):
            ax = axes[row, column]
            subset = data.loc[
                data["algorithm"].eq(algorithm)
                & data["index_construction"].eq(construction)
            ]
            draw_forecast_trajectory_panel(
                ax,
                subset,
                f"({panel_letters[position]}) {publication_label(algorithm)} · "
                f"{publication_label(construction)}",
                colors,
            )
            ax.set_xlabel("Target month" if row == len(algorithms) - 1 else "")
            position += 1
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
    )
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def draw_target_vs_observed_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    colors: Mapping[str, str],
) -> None:
    data = frame.copy()
    actual = pd.to_numeric(data["actual_scfi_level"], errors="raise")
    predicted = pd.to_numeric(data["predicted_scfi_level"], errors="raise")
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))
    padding = max((upper - lower) * 0.04, 1.0)
    bounds = (lower - padding, upper + padding)
    ax.plot(
        bounds,
        bounds,
        color="#6b7280",
        linestyle="--",
        lw=1.0,
        label="45° reference",
        zorder=1,
    )
    path_styles = {
        "peak": (colors["secondary"], "Peak-based forecast"),
        "breadth": ("#2f855a", "Breadth-based forecast"),
    }
    rmse_lines: list[str] = []
    for path, (color, label) in path_styles.items():
        subset = data.loc[data["forecast_path"].eq(path)]
        observed = pd.to_numeric(subset["actual_scfi_level"], errors="raise")
        forecast = pd.to_numeric(subset["predicted_scfi_level"], errors="raise")
        ax.scatter(
            observed,
            forecast,
            s=13,
            alpha=0.50,
            color=color,
            edgecolors="none",
            label=label,
            zorder=2,
        )
        rmse = float(np.sqrt(np.mean(np.square(observed - forecast))))
        rmse_lines.append(f"{publication_label(path)} RMSE: {rmse:,.0f}")
    ax.text(
        0.03,
        0.97,
        "\n".join(rmse_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
    )
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_title(title)
    ax.set_xlabel("Observed SCFI")
    ax.set_ylabel("Forecast SCFI")
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)


def draw_forecast_error_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    colors: Mapping[str, str],
) -> None:
    data = frame.copy()
    data["target_date"] = pd.to_datetime(data["target_date"], errors="raise")
    path_styles = {
        "peak": (colors["secondary"], "Peak-based forecast"),
        "breadth": ("#2f855a", "Breadth-based forecast"),
    }
    for path, (color, label) in path_styles.items():
        subset = data.loc[data["forecast_path"].eq(path)].sort_values(
            "target_date"
        )
        ax.plot(
            subset["target_date"],
            subset["level_error_observed_minus_forecast"],
            color=color,
            lw=1.0,
            alpha=0.88,
            label=label,
        )
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("Error (observed − forecast)")
    ax.margins(x=0.01)


def plot_weekly_target_vs_observed(
    trajectories: pd.DataFrame,
    design: Mapping[str, Any],
    colors: Mapping[str, str],
) -> plt.Figure:
    data = trajectories.loc[trajectories["frequency"].eq("weekly")].copy()
    data = data.loc[
        data["index_construction"].eq(design["weekly_constructions"][0])
    ]
    algorithms = list(design["algorithms"])
    fig, axes, _, _ = make_weekly_panel_grid(
        len(algorithms),
        width=10.8,
        height_per_row=4.5,
        sharex=True,
        sharey=True,
    )
    for position, (ax, algorithm) in enumerate(zip(axes.flat, algorithms)):
        draw_target_vs_observed_panel(
            ax,
            data.loc[data["algorithm"].eq(algorithm)],
            f"({'abcdefghijklmnopqrstuvwxyz'[position]}) {publication_label(algorithm)}",
            colors,
        )
    hide_unused_panel_axes(axes, len(algorithms))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def plot_monthly_target_vs_observed(
    trajectories: pd.DataFrame,
    design: Mapping[str, Any],
    colors: Mapping[str, str],
) -> plt.Figure:
    data = trajectories.loc[trajectories["frequency"].eq("monthly")].copy()
    algorithms = list(design["algorithms"])
    constructions = list(design["monthly_constructions"])
    fig, axes = plt.subplots(
        len(algorithms),
        len(constructions),
        figsize=(11.5, 14.0),
        sharex=True,
        sharey=True,
    )
    position = 0
    for row, algorithm in enumerate(algorithms):
        for column, construction in enumerate(constructions):
            draw_target_vs_observed_panel(
                axes[row, column],
                data.loc[
                    data["algorithm"].eq(algorithm)
                    & data["index_construction"].eq(construction)
                ],
                f"({'abcdefghijklmnopqrstuvwxyz'[position]}) "
                f"{publication_label(algorithm)} · {publication_label(construction)}",
                colors,
            )
            position += 1
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_weekly_forecast_errors(
    trajectories: pd.DataFrame,
    design: Mapping[str, Any],
    colors: Mapping[str, str],
) -> plt.Figure:
    data = trajectories.loc[trajectories["frequency"].eq("weekly")].copy()
    data = data.loc[
        data["index_construction"].eq(design["weekly_constructions"][0])
    ]
    algorithms = list(design["algorithms"])
    fig, axes, rows, columns = make_weekly_panel_grid(
        len(algorithms),
        width=12.2,
        height_per_row=3.6,
        sharex=True,
        sharey=True,
    )
    for position, (ax, algorithm) in enumerate(zip(axes.flat, algorithms)):
        draw_forecast_error_panel(
            ax,
            data.loc[data["algorithm"].eq(algorithm)],
            f"({'abcdefghijklmnopqrstuvwxyz'[position]}) {publication_label(algorithm)}",
            colors,
        )
        ax.set_xlabel(
            "Target week" if position // columns == rows - 1 else ""
        )
    hide_unused_panel_axes(axes, len(algorithms))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
    )
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_monthly_forecast_errors(
    trajectories: pd.DataFrame,
    design: Mapping[str, Any],
    colors: Mapping[str, str],
) -> plt.Figure:
    data = trajectories.loc[trajectories["frequency"].eq("monthly")].copy()
    algorithms = list(design["algorithms"])
    constructions = list(design["monthly_constructions"])
    fig, axes = plt.subplots(
        len(algorithms),
        len(constructions),
        figsize=(13.0, 10.5),
        sharex=True,
        sharey=True,
    )
    position = 0
    for row, algorithm in enumerate(algorithms):
        for column, construction in enumerate(constructions):
            draw_forecast_error_panel(
                axes[row, column],
                data.loc[
                    data["algorithm"].eq(algorithm)
                    & data["index_construction"].eq(construction)
                ],
                f"({'abcdefghijklmnopqrstuvwxyz'[position]}) "
                f"{publication_label(algorithm)} · {publication_label(construction)}",
                colors,
            )
            axes[row, column].set_xlabel(
                "Target month" if row == len(algorithms) - 1 else ""
            )
            position += 1
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
    )
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def build_figures(
    config: Mapping[str, Any],
    audited: Mapping[str, Mapping[str, Any]],
    main_tables: Mapping[str, pd.DataFrame],
    supplements: Mapping[str, pd.DataFrame],
    figures_dir: Path,
    version: str,
) -> list[Path]:
    configure_plot_style(config)
    settings = config["figures"]
    colors = {
        "primary": settings["color_primary"],
        "secondary": settings["color_secondary"],
        "robustness": settings["color_robustness"],
        "warning": settings["color_warning"],
    }
    formats = [str(value).lower() for value in settings["formats"]]
    dpi = int(settings["dpi"])
    panel = read_csv_output(audited, "monthly_llm", "monthly_panel")
    pcmci = read_csv_output(audited, "time_series_causality", "pcmci_links")
    pcmci["significant_fdr_bool"] = strict_bool_series(
        pcmci["significant_fdr"], "significant_fdr"
    )
    irf = read_csv_output(audited, "time_series_causality", "var_impulse_responses")
    figure_specs = [
        ("figure01_monthly_msdi_scfi_timeline", plot_timeline(panel, colors)),
        (
            "figure02_monthly_llm_point_ranking",
            plot_llm_ranking(main_tables["T02_monthly_llm_point_ranking"], colors),
        ),
        (
            "figure03_primary_forecast_effects",
            plot_primary_effects(main_tables["T04_primary_forecast_inference"], colors),
        ),
        (
            "figure04_local_projection_paths",
            plot_local_projections(main_tables["T06_local_projections"], colors),
        ),
        (
            "figure05_dml_predictive_effects",
            plot_dml(main_tables["T09_dml_effects_diagnostics"], colors),
        ),
        (
            "figure06_temporal_dependence",
            plot_temporal_evidence(main_tables["T07_granger_tests"], pcmci, colors),
        ),
        (
            "figure07_ablation_sensitivity",
            plot_ablation(supplements["S01_full_ablation"], colors),
        ),
        (
            "figure08_forecast_diagnostics",
            plot_diagnostics(
                main_tables["T12_forecast_diagnostics"],
                supplements["S07_primary_residual_tests"],
                colors,
            ),
        ),
        ("figure09_var_impulse_responses", plot_var_irf(irf, colors)),
        (
            "figure10_weekly_model_forecasts",
            plot_weekly_forecast_trajectories(
                supplements["S08_one_step_forecasts"],
                config["forecast_trajectory_figures"],
                colors,
            ),
        ),
        (
            "figure11_monthly_model_forecasts",
            plot_monthly_forecast_trajectories(
                supplements["S08_one_step_forecasts"],
                config["forecast_trajectory_figures"],
                colors,
            ),
        ),
        (
            "figure12_weekly_target_vs_observed",
            plot_weekly_target_vs_observed(
                supplements["S08_one_step_forecasts"],
                config["forecast_trajectory_figures"],
                colors,
            ),
        ),
        (
            "figure13_monthly_target_vs_observed",
            plot_monthly_target_vs_observed(
                supplements["S08_one_step_forecasts"],
                config["forecast_trajectory_figures"],
                colors,
            ),
        ),
        (
            "figure14_weekly_forecast_errors",
            plot_weekly_forecast_errors(
                supplements["S08_one_step_forecasts"],
                config["forecast_trajectory_figures"],
                colors,
            ),
        ),
        (
            "figure15_monthly_forecast_errors",
            plot_monthly_forecast_errors(
                supplements["S08_one_step_forecasts"],
                config["forecast_trajectory_figures"],
                colors,
            ),
        ),
    ]
    paths: list[Path] = []
    for label, figure in figure_specs:
        paths.extend(
            save_figure(
                figure,
                figures_dir,
                f"{version}_{label}",
                formats,
                dpi,
            )
        )
    return paths


def make_results_report(
    version: str,
    main: Mapping[str, pd.DataFrame],
    supplements: Mapping[str, pd.DataFrame],
    source_audit: pd.DataFrame,
) -> str:
    ranking = main["T02_monthly_llm_point_ranking"].sort_values(
        "predictive_point_rank"
    )
    best = ranking.iloc[0]
    claims = main["T15_primary_claim_summary"]
    primary_inference = main["T04_primary_forecast_inference"]
    improvements = int(
        strict_bool_series(
            primary_inference["adjusted_significant_positive_improvement"],
            "primary adjusted improvement",
        ).sum()
    )
    worsenings = int(
        strict_bool_series(
            primary_inference["adjusted_significant_worsening"],
            "primary adjusted worsening",
        ).sum()
    )
    primary_tests = int(len(primary_inference))
    pcmci = main["T08_pcmci_summary"]
    pcmci_links = int(pcmci["fdr_significant_links"].sum())
    granger = main["T07_granger_tests"]
    granger_rejections = int((pd.to_numeric(granger["p_value_holm"], errors="coerce") <= 0.05).sum())
    dml = main["T09_dml_effects_diagnostics"]
    dml_fdr = int((pd.to_numeric(dml["p_value_fdr_all"], errors="coerce") <= 0.05).sum())
    placebo_flags = int(
        strict_bool_series(
            dml["placebo_falsification_flag_fdr"], "placebo falsification flag"
        ).sum()
    )
    ablation = supplements["S01_full_ablation"]
    ablation_rejections = int(any_adjusted_rejection(ablation).sum())
    uncertainty = main["T14_uncertainty_audit"]
    uncertainty_pass = int(strict_bool_series(uncertainty["audit_passed"], "uncertainty audit").sum())
    lines = [
        f"# {version}: computational results summary",
        "",
        "> Status: computational publication package complete; human validation is still mandatory before submission.",
        "",
        "## Frozen interpretation",
        "",
        "Monthly results are primary and weekly results are pre-specified higher-frequency robustness. Forecast ranks are descriptive point ranks. Granger, PCMCI, VAR, local projections, and DML are observational/predictive analyses and do not authorize structural causal claims.",
        "",
        "## Main computational findings",
        "",
        f"- The direct monthly LLM point-ranking leader is `{best['score_source']}` (rank {int(best['predictive_point_rank'])}; mean relative RMSE versus coverage-only baseline {float(best['mean_relative_rmse_vs_coverage']):.3f}). This is not formal pairwise superiority.",
        f"- Across {primary_tests} registered primary forecast hypotheses, {improvements} Holm-adjusted result(s) favor the challenger and {worsenings} favor the baseline. No economic-importance claim is permitted.",
        f"- The ablation registry contains {len(ablation):,} comparisons; {ablation_rejections} rows trigger an adjusted rejection in a registered stream. Direction and magnitude must be read from the corresponding effect columns.",
        f"- PCMCI contains {pcmci_links} FDR-significant link-lag entries and the consolidated Granger table contains {granger_rejections} Holm-adjusted rejection(s). These are temporal-dependence findings only.",
        f"- DML contains {len(dml)} orthogonalized predictive estimates, with {dml_fdr} globally FDR-adjusted rejection(s) and {placebo_flags} FDR falsification flag(s). These do not identify causal effects.",
        "- Figures 10–11 show frozen one-step-ahead out-of-sample SCFI levels for the penalized Gaussian GLM, LSTM, Prophet, Random Forest, and XGBoost using the median-ensemble peak and breadth paths; they are forecasts, not in-sample fitted values.",
        "",
        "## Integrity and remaining publication gates",
        "",
        f"- All {len(source_audit)} required source-artifact contracts passed across {source_audit['stage'].nunique()} frozen manifests.",
        f"- {uncertainty_pass}/{len(uncertainty)} uncertainty/effect-size audit groups passed.",
        "- Dual-annotator human validation, LLM-versus-human reliability, and external convergent/discriminant validity tables remain unavailable and must be completed before the paper is submission-ready.",
        "- Causal-forest results are not fabricated: the current heterogeneity table reports only pre-specified GATE/BLP estimates or guarded skips.",
        "",
    ]
    return "\n".join(lines)


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "matplotlib", "openpyxl", "pyyaml"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("publication_outputs")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def output_inventory(paths: Sequence[Path]) -> tuple[dict[str, str], dict[str, int]]:
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for path in paths:
        hashes[path.name] = sha256_file(path)
        sizes[path.name] = int(path.stat().st_size)
    return hashes, sizes


def run(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    config = load_yaml(args.config)
    if args.config.resolve() == DEFAULT_CONFIG.resolve():
        observed = sha256_file(DEFAULT_CONFIG)
        if observed != CANONICAL_CONFIG_SHA256:
            raise ValueError(
                "Canonical publication config hash mismatch: "
                f"expected {CANONICAL_CONFIG_SHA256}, observed {observed}"
            )
    version = str(args.version or config["version"])
    if args.smoke_test and args.version is None:
        version = f"{version}_smoke"
    output_config = config["outputs"]
    script_lock = resolve_project_path(output_config["script_sha256_lock"])
    observed_script_hash = sha256_file(Path(__file__).resolve())
    if read_sha256_lock(script_lock) != observed_script_hash:
        raise ValueError("Stage 10 script SHA-256 integrity lock mismatch")
    tables_dir = (args.tables_dir or resolve_project_path(output_config["tables_directory"])).resolve()
    figures_dir = (args.figures_dir or resolve_project_path(output_config["figures_directory"])).resolve()
    reports_dir = (args.reports_dir or resolve_project_path(output_config["reports_directory"])).resolve()
    log_dir = (args.log_dir or resolve_project_path(output_config["logs_directory"])).resolve()
    for directory in (tables_dir, figures_dir, reports_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{version}_publication_outputs.log"
    logger = configure_logger(log_path)
    logger.info("stage=10 version=%s smoke_test=%s", version, args.smoke_test)

    audited, source_audit = audit_sources(config, logger)
    validate_global_guardrails(config, audited)
    expected_sources = int(config["expected"]["source_manifests"])
    if len(audited) != expected_sources:
        raise ValueError(
            f"Expected {expected_sources} source manifests, audited {len(audited)}"
        )
    main_tables, supplements = build_tables(config, audited, source_audit)
    minimum_tables = int(config["expected"]["minimum_tables"])
    if len(main_tables) < minimum_tables:
        raise ValueError(f"Expected at least {minimum_tables} main tables")

    written: list[Path] = []
    table_rows: dict[str, int] = {}
    all_tables = {**main_tables, **supplements}
    for label, frame in all_tables.items():
        path = tables_dir / f"{version}_{label}.csv"
        atomic_write_csv(path, frame)
        written.append(path)
        table_rows[path.name] = int(len(frame))
        logger.info("table=%s rows=%d columns=%d", path.name, len(frame), len(frame.columns))
    workbook_path = tables_dir / f"{version}_{output_config['workbook_suffix']}"
    atomic_write_workbook(workbook_path, all_tables)
    written.append(workbook_path)

    figure_paths = build_figures(
        config, audited, main_tables, supplements, figures_dir, version
    )
    written.extend(figure_paths)
    figure_basenames = {path.stem for path in figure_paths}
    expected_figures = int(config["expected"]["figure_basenames"])
    # An additional guarded VAR figure is deliberately included beyond the minimum.
    if len(figure_basenames) < expected_figures:
        raise ValueError(
            f"Expected at least {expected_figures} figure basenames, got {len(figure_basenames)}"
        )
    logger.info("figure_files=%d figure_basenames=%d", len(figure_paths), len(figure_basenames))

    report_path = reports_dir / f"{version}_{output_config['report_suffix']}"
    atomic_write_text(
        report_path,
        make_results_report(version, main_tables, supplements, source_audit),
    )
    written.append(report_path)

    source_audit_path = reports_dir / f"{version}_{output_config['source_audit_suffix']}"
    atomic_write_csv(source_audit_path, source_audit)
    written.append(source_audit_path)

    primary_inference = main_tables["T04_primary_forecast_inference"]
    adjusted_improvements = int(
        strict_bool_series(
            primary_inference["adjusted_significant_positive_improvement"],
            "primary adjusted improvement",
        ).sum()
    )
    adjusted_worsenings = int(
        strict_bool_series(
            primary_inference["adjusted_significant_worsening"],
            "primary adjusted worsening",
        ).sum()
    )
    if len(primary_inference) != int(config["expected"]["primary_forecast_hypotheses"]):
        raise ValueError("Primary forecast hypothesis count changed from frozen contract")
    expected_improvements = config["expected"].get("adjusted_primary_improvements")
    if expected_improvements is not None and adjusted_improvements != int(expected_improvements):
        raise ValueError("Primary improvement count changed from frozen contract")
    expected_worsenings = config["expected"].get("adjusted_primary_worsenings")
    if expected_worsenings is not None and adjusted_worsenings != int(expected_worsenings):
        raise ValueError("Primary worsening count changed from frozen contract")
    hashes, sizes = output_inventory(written)
    manifest_path = reports_dir / f"{version}_{output_config['manifest_suffix']}"
    manifest = {
        "version": version,
        "stage": "10_make_figures_tables",
        "status": "complete_computational_publication_package_pending_human_validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "publication_scope_complete": True,
        "publication_eligible_without_human_validation": False,
        "human_validation_required": True,
        "causal_claim_allowed": False,
        "economic_importance_claim_allowed": False,
        "primary_frequency": "monthly",
        "weekly_role": "higher_frequency_robustness",
        "primary_forecast_hypotheses": int(len(primary_inference)),
        "adjusted_primary_improvements": adjusted_improvements,
        "adjusted_primary_worsenings": adjusted_worsenings,
        "source_contracts_all_passed": bool(source_audit["contract_passed"].all()),
        "source_manifest_sha256": {
            stage: item["manifest_sha256"] for stage, item in audited.items()
        },
        "source_artifacts_audited": int(len(source_audit)),
        "main_table_count": int(len(main_tables)),
        "supplementary_table_count": int(len(supplements)),
        "figure_basename_count": int(len(figure_basenames)),
        "figure_file_count": int(len(figure_paths)),
        "table_row_counts": table_rows,
        "outputs": {path.name: str(path.resolve()) for path in written},
        "output_sha256": hashes,
        "output_bytes": sizes,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "script": str(Path(__file__).resolve()),
        "script_sha256": observed_script_hash,
        "script_sha256_lock": str(script_lock),
        "interpretation_guardrails": config["guardrails"],
        "software": package_versions(),
    }
    atomic_write_json(manifest_path, manifest)
    logger.info(
        "complete manifest=%s runtime_seconds=%.3f",
        manifest_path,
        manifest["runtime_seconds"],
    )
    return manifest_path


def main() -> None:
    args = parse_args()
    manifest_path = run(args)
    print(f"Publication package manifest: {manifest_path}")


if __name__ == "__main__":
    main()
