"""Run matched ablation and sensitivity analyses for the MSDI--SCFI pipeline.

This stage deliberately separates three questions:

1. Which measurement components add out-of-sample SCFI forecast information?
2. Are the dynamic time-series conclusions stable across frozen specifications?
3. Are the orthogonalized predictive effects from stage 08b stable to learners,
   inference methods, and strictly future treatment-lead falsification tests?

Weekly forecasts are primary. Monthly forecasts retain both pre-specified index
constructions and are robustness evidence only. All comparisons are evaluated
on matched origin/target dates. Nothing in this script licenses a causal claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
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
from scipy.stats import chi2, norm, spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/robustness.yaml"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "configs/model_registry.yaml"
DEFAULT_ARTICLE_SCORES = (
    PROJECT_ROOT / "data/05_indices/article_level/article_consensus_scores.csv"
)
DEFAULT_WEEKLY_PANEL = (
    PROJECT_ROOT / "data/06_model_ready/prediction/msdi_scfi_weekly_panel.csv"
)
DEFAULT_MONTHLY_PANEL = (
    PROJECT_ROOT
    / "data/06_model_ready/prediction/llm_specific_scfi_monthly_panel.csv"
)
DEFAULT_CAUSALITY_MONTHLY_PANEL = (
    PROJECT_ROOT / "data/06_model_ready/causality/msdi_scfi_monthly_panel.csv"
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
DEFAULT_NONLINEAR_MANIFEST = (
    PROJECT_ROOT / "outputs/nonlinear_forecasts/nonlinear_scfi_v001_manifest.json"
)
DEFAULT_CAUSALITY_MANIFEST = (
    PROJECT_ROOT / "outputs/time_series_causality/causality_v001_manifest.json"
)
DEFAULT_CAUSAL_ML_MANIFEST = (
    PROJECT_ROOT / "outputs/causal_ml/causal_ml_v001_manifest.json"
)
DEFAULT_CAUSAL_ML_CONFIG = PROJECT_ROOT / "configs/causal_ml.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/robustness"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs/logs"

SCRIPT_08 = PROJECT_ROOT / "scripts/19_time_series_causality.py"
SCRIPT_08B = PROJECT_ROOT / "scripts/20_causal_ml.py"

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
PATHS = ("peak", "breadth")
MONTHLY_CONSTRUCTIONS = ("mean_safe_weekly", "adapted_pdf_sqrt_ema3")
OPERATIONAL_DISRUPTION_START = "2020-03-01"
OPERATIONAL_DISRUPTION_END = "2022-06-30"

WEEKLY_AR_FEATURES = (
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
WEEKLY_BRENT_FEATURES = (
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
MONTHLY_LAGS = (1, 2, 3, 6, 12)
MONTHLY_AR_FEATURES = (
    "scfi_monthly_mean_log",
    *[f"scfi_monthly_mean_log_lag_{lag}m" for lag in MONTHLY_LAGS],
    *[f"scfi_monthly_mean_log_change_{lag}m" for lag in (1, 3, 6, 12)],
    "calendar_month_sin",
    "calendar_month_cos",
)
MONTHLY_BRENT_FEATURES = (
    "brent_monthly_mean_log",
    *[f"brent_monthly_mean_log_lag_{lag}m" for lag in MONTHLY_LAGS],
    *[f"brent_monthly_mean_log_change_{lag}m" for lag in (1, 3, 6, 12)],
)
WEEKLY_EXTERNAL_CONTROL_FEATURES = (
    *[f"cpb_world_trade_volume_lag_{lag}w" for lag in (1, 2, 4, 8, 12)],
    *[f"unctad_china_core_lane_lsbci_lag_{lag}w" for lag in (1, 2, 4, 8, 12)],
)
MONTHLY_EXTERNAL_CONTROL_FEATURES = ("cpb_world_trade_volume_lag_1m", "unctad_china_core_lane_lsbci_lag_1m")

PREDICTION_KEY_COLUMNS = (
    "frequency",
    "index_construction",
    "horizon",
    "origin_date",
    "target_date",
)


@dataclass(frozen=True)
class IndexVariant:
    variant_id: str
    path: str
    family_id: str
    score_rule: str = "median"
    included_aliases: tuple[str, ...] = ()
    excluded_dimension: str | None = None
    duplicate_penalty: str = "sqrt"
    media_correction: str = "log"
    filter_rule: str = "all_common_articles"
    excluded_source: str | None = None
    standardization: str = "none"
    permute_dates: bool = False
    comparison_reference: str | None = None
    requested: bool = True
    primary: bool = False
    description: str = ""


@dataclass(frozen=True)
class ForecastSpecification:
    specification_id: str
    family_id: str
    signal_variants: tuple[str, ...]
    coverage_variant: str
    include_brent: bool = True
    include_coverage: bool = True
    signal_lags: tuple[int, ...] = ()
    rolling_windows: tuple[int, ...] = ()
    training_window: int | None = None
    refit_every: int = 1
    description: str = ""


@dataclass(frozen=True)
class PairedComparison:
    comparison_id: str
    family_id: str
    baseline_specification_id: str
    variant_specification_id: str
    nested: bool
    primary: bool
    direction_note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched MSDI ablation and sensitivity checks."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    parser.add_argument("--article-scores", type=Path, default=DEFAULT_ARTICLE_SCORES)
    parser.add_argument("--weekly-panel", type=Path, default=DEFAULT_WEEKLY_PANEL)
    parser.add_argument("--monthly-panel", type=Path, default=DEFAULT_MONTHLY_PANEL)
    parser.add_argument(
        "--causality-monthly-panel", type=Path, default=DEFAULT_CAUSALITY_MONTHLY_PANEL
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
    parser.add_argument(
        "--nonlinear-manifest", type=Path, default=DEFAULT_NONLINEAR_MANIFEST
    )
    parser.add_argument(
        "--causality-manifest", type=Path, default=DEFAULT_CAUSALITY_MANIFEST
    )
    parser.add_argument(
        "--causal-ml-manifest", type=Path, default=DEFAULT_CAUSAL_ML_MANIFEST
    )
    parser.add_argument("--causal-ml-config", type=Path, default=DEFAULT_CAUSAL_ML_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--frequency", choices=("both", "weekly", "monthly"), default="both"
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--maximum-specifications", type=int, default=None)
    parser.add_argument("--maximum-origins", type=int, default=None)
    parser.add_argument(
        "--skip-dynamic-refits",
        action="store_true",
        help="Debug-only; makes scope incomplete.",
    )
    parser.add_argument(
        "--skip-future-lead-placebos",
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
    value = "|".join([str(seed), *[str(part) for part in parts]])
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
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
    path = log_dir / f"09_robustness_checks_{version}.log"
    logger = logging.getLogger(f"robustness.{version}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
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
            "Run this stage with the project environment: "
            ".venv\\Scripts\\python.exe scripts\\21_ablation_sensitivity_checks.py"
        )
    return executable


def import_support(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load support script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_model_aliases(path: Path) -> list[str]:
    payload = load_yaml(path)
    aliases = [
        str(item["alias"]) for item in payload.get("active_scoring_models", [])
    ]
    if len(aliases) != 3 or len(set(aliases)) != 3:
        raise ValueError(f"Expected exactly three unique model aliases; found {aliases}")
    return aliases


def validate_args(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.bootstrap_repetitions is not None and args.bootstrap_repetitions < 0:
        raise ValueError("--bootstrap-repetitions cannot be negative")
    for name in ("maximum_specifications", "maximum_origins"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if config.get("identification", {}).get("causal_claim_allowed") is not False:
        raise ValueError("Robustness configuration must explicitly prohibit causal claims")
    if config.get("analysis", {}).get("primary_frequency") != "weekly":
        raise ValueError("Weekly frequency must remain primary")
    if config.get("analysis", {}).get("monthly_role") != "robustness":
        raise ValueError("Monthly frequency must remain robustness-only")
    analysis = config.get("analysis", {})
    measurement = config.get("measurement_variants", {})
    inference = config.get("inference", {})
    monthly = config.get("forecast", {}).get("monthly", {})
    pcmci = config.get("dynamic_sensitivity", {}).get("pcmci", {})
    future = config.get("causal_sensitivity", {}).get(
        "future_treatment_lead_placebo", {}
    )
    regime = config.get("regime_sensitivity", {})
    contracts = {
        "reference_peak_variant": analysis.get("reference_peak_variant")
        == "reference_peak_median",
        "reference_breadth_variant": analysis.get("reference_breadth_variant")
        == "reference_breadth_median7",
        "monthly_constructions": tuple(monthly.get("constructions", ()))
        == MONTHLY_CONSTRUCTIONS,
        "primary_duplicate_penalty": measurement.get("primary_duplicate_penalty")
        == "sqrt",
        "primary_media_correction": measurement.get("primary_media_correction")
        == "log",
        "paired_loss_convention": inference.get("paired_loss_difference")
        == "baseline_loss_minus_variant_loss",
        "pcmci_system": pcmci.get("variable_sets")
        == ["full_available_four_variable_system"],
        "future_lead_enabled": future.get("enabled") is True,
        "future_lead_formula": future.get("lead_steps_formula")
        == "horizon_plus_one",
        "future_lead_offset": int(future.get("lead_offset_beyond_outcome", -1)) == 1,
        "future_lead_strict": future.get(
            "require_lead_strictly_greater_than_outcome_horizon"
        )
        is True,
        "future_lead_embargo": future.get("use_larger_lead_embargo") is True,
        "future_lead_learner": future.get("nuisance_learner") == "elastic_net",
        "operational_regime_start": regime.get("disruption_period_start")
        == OPERATIONAL_DISRUPTION_START,
        "operational_regime_end": regime.get("disruption_period_end")
        == OPERATIONAL_DISRUPTION_END,
        "operational_regime_not_official": regime.get("official_pandemic_dates")
        is False,
    }
    failed = [name for name, passed in contracts.items() if not passed]
    if failed:
        raise ValueError(f"Frozen robustness contracts failed: {failed}")


def _manifest_output_path(value: Any) -> Path | None:
    if isinstance(value, str):
        return Path(value).resolve()
    if isinstance(value, Mapping) and isinstance(value.get("path"), str):
        return Path(str(value["path"])).resolve()
    return None


def validate_upstream_manifests(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    specifications = [
        ("prediction", args.prediction_manifest, "forecast_v001"),
        ("weekly_llm", args.weekly_llm_manifest, "llm_scfi_v001"),
        ("monthly_llm", args.monthly_llm_manifest, "llm_scfi_monthly_v001"),
        ("nonlinear", args.nonlinear_manifest, "nonlinear_scfi_v001"),
        ("causality", args.causality_manifest, "causality_v001"),
        ("causal_ml", args.causal_ml_manifest, "causal_ml_v001"),
    ]
    rows: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    for name, raw_path, expected_version in specifications:
        path = require_file(raw_path, f"{name} manifest")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[name] = payload
        version = str(payload.get("version", ""))
        status = str(payload.get("status", ""))
        passed = version == expected_version and "smoke" not in version.lower()
        reason = "passed" if passed else "version_or_scope_mismatch"
        if name == "weekly_llm":
            cohort = payload.get("design", {}).get(
                "expected_common_articles",
                payload.get("design", {}).get("frozen_common_articles"),
            )
            passed = passed and int(cohort or -1) == 6481
            reason = "passed" if passed else "weekly_common_cohort_not_6481"
        if name == "monthly_llm":
            methods = set(
                payload.get("design", {}).get("index_constructions", {}).keys()
            )
            passed = passed and methods == set(MONTHLY_CONSTRUCTIONS)
            reason = "passed" if passed else "monthly_construction_contract_failed"
        if name == "nonlinear":
            design = payload.get("design", {})
            passed = passed and bool(design.get("full_canonical_scope_complete"))
            reason = "passed" if passed else "nonlinear_full_scope_incomplete"
        if name == "causal_ml":
            passed = (
                passed
                and bool(payload.get("publication_scope_complete"))
                and payload.get("causal_claim_allowed") is False
            )
            reason = "passed" if passed else "causal_ml_contract_failed"
        rows.append(
            {
                "upstream_stage": name,
                "manifest_path": str(path),
                "manifest_sha256": sha256_file(path),
                "version": version,
                "status": status,
                "contract_passed": passed,
                "contract_note": reason,
            }
        )
    audit = pd.DataFrame(rows)
    if not audit["contract_passed"].all():
        failed = audit.loc[~audit["contract_passed"]]
        raise ValueError("Upstream contract failed:\n" + failed.to_string(index=False))
    return audit, payloads


def determine_run_status(
    smoke_test: bool,
    maximum_specifications: int | None,
    maximum_origins: int | None,
    scope_complete: bool,
) -> str:
    if smoke_test or maximum_specifications is not None or maximum_origins is not None:
        return "partial_debug_scope_not_publication_eligible"
    if not scope_complete:
        return "complete_requested_subset_scope_not_publication_eligible"
    return "complete_computational_scope_pending_human_validation"


def load_common_articles(
    path: Path, aliases: Sequence[str], expected_rows: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = require_file(path, "Article consensus scores")
    header = pd.read_csv(resolved, nrows=0)
    score_columns = [
        f"{score}__{alias}"
        for alias in aliases
        for score in (PRIMARY_SCORE_COLUMN, *DIMENSION_COLUMNS)
    ]
    metadata_columns = [
        "article_id",
        "publication_date",
        "source_domain",
        "source_name",
        "duplicate_group_size",
        "novelty_weight",
    ]
    required = [*metadata_columns, *score_columns]
    missing = sorted(set(required).difference(header.columns))
    if missing:
        raise ValueError(f"Article scores lack required columns: {missing}")
    article = pd.read_csv(resolved, usecols=required, low_memory=False)
    if article["article_id"].duplicated().any():
        raise ValueError("Article consensus scores contain duplicate article IDs")
    article["publication_date"] = pd.to_datetime(
        article["publication_date"], format="mixed", errors="coerce"
    )
    for column in [*score_columns, "duplicate_group_size", "novelty_weight"]:
        article[column] = pd.to_numeric(article[column], errors="coerce")
    common = article.loc[
        article["publication_date"].notna()
        & article[score_columns].notna().all(axis=1)
    ].copy()
    if len(common) != expected_rows:
        raise ValueError(
            f"Frozen all-three-LLM cohort mismatch: expected={expected_rows} "
            f"found={len(common)}"
        )
    invalid = common[score_columns].lt(0) | common[score_columns].gt(4)
    if invalid.any().any():
        locations = invalid.stack()[lambda value: value].index.tolist()[:10]
        raise ValueError(f"Article scores leave the frozen 0--4 scale: {locations}")
    common["duplicate_group_size"] = (
        common["duplicate_group_size"].fillna(1).clip(lower=1)
    )
    common["source_domain"] = (
        common["source_domain"].fillna("unknown_source").astype(str).str.strip()
    )
    reference_dimensions = pd.DataFrame(
        {
            dimension: common[
                [f"{dimension}__{alias}" for alias in aliases]
            ].median(axis=1)
            for dimension in DIMENSION_COLUMNS
        },
        index=common.index,
    )
    non_price = [
        dimension
        for dimension in DIMENSION_COLUMNS
        if dimension != "freight_cost_disruption_score"
    ]
    common["freight_price_only_profile"] = (
        reference_dimensions["freight_cost_disruption_score"].gt(0)
        & reference_dimensions[non_price].max(axis=1).eq(0)
    )
    peak_positive = pd.DataFrame(
        {
            alias: common[f"{PRIMARY_SCORE_COLUMN}__{alias}"].gt(0)
            for alias in aliases
        }
    )
    breadth_positive = pd.DataFrame(
        {
            alias: common[
                [f"{dimension}__{alias}" for dimension in DIMENSION_COLUMNS]
            ].max(axis=1).gt(0)
            for alias in aliases
        }
    )
    common["peak_positive_model_support"] = peak_positive.sum(axis=1)
    common["breadth_positive_model_support"] = breadth_positive.sum(axis=1)
    source_counts = common["source_domain"].value_counts()
    report = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "rows_read": len(article),
        "common_complete_rows": len(common),
        "expected_common_complete_rows": expected_rows,
        "common_cohort_passed": len(common) == expected_rows,
        "first_publication_date": common["publication_date"].min(),
        "last_publication_date": common["publication_date"].max(),
        "freight_price_only_profiles": int(common["freight_price_only_profile"].sum()),
        "peak_low_model_support_positive_articles": int(
            (
                common["peak_positive_model_support"].between(1, 1)
            ).sum()
        ),
        "breadth_low_model_support_positive_articles": int(
            (
                common["breadth_positive_model_support"].between(1, 1)
            ).sum()
        ),
        "top_source_domains": source_counts.head(10).to_dict(),
    }
    return common.sort_values("article_id").reset_index(drop=True), report


def duplicate_weights(group_size: pd.Series, method: str) -> pd.Series:
    size = pd.to_numeric(group_size, errors="coerce").fillna(1).clip(lower=1)
    if method == "sqrt":
        return 1.0 / np.sqrt(size)
    if method == "inverse":
        return 1.0 / size
    if method == "none":
        return pd.Series(1.0, index=size.index, dtype=float)
    raise ValueError(f"Unknown duplicate penalty: {method}")


def media_correct(
    raw_sum: pd.Series,
    count: pd.Series,
    effective_count: pd.Series,
    method: str,
) -> pd.Series:
    raw = pd.to_numeric(raw_sum, errors="coerce").fillna(0.0)
    article_count = pd.to_numeric(count, errors="coerce").fillna(0.0)
    effective = pd.to_numeric(effective_count, errors="coerce").fillna(0.0)
    if method == "log":
        denominator = np.log1p(article_count)
    elif method == "mean":
        denominator = article_count
    elif method == "sqrt":
        denominator = np.sqrt(effective)
    elif method == "none":
        return raw
    else:
        raise ValueError(f"Unknown media correction: {method}")
    return raw.div(denominator.where(denominator.gt(0))).fillna(0.0)


def article_variant_scores(
    article: pd.DataFrame,
    variant: IndexVariant,
    aliases: Sequence[str],
) -> pd.Series:
    if variant.path not in PATHS:
        raise ValueError(f"Unknown index path: {variant.path}")
    included = list(variant.included_aliases or tuple(aliases))
    if not included or not set(included).issubset(set(aliases)):
        raise ValueError(
            f"Variant {variant.variant_id} has invalid included aliases: {included}"
        )
    if variant.score_rule not in {"median", "mean"}:
        raise ValueError(f"Unknown score rule: {variant.score_rule}")

    def combine(columns: Sequence[str]) -> pd.Series:
        values = article[list(columns)]
        if variant.score_rule == "median":
            return values.median(axis=1, skipna=False)
        return values.mean(axis=1, skipna=False)

    if variant.path == "peak":
        result = combine(
            [f"{PRIMARY_SCORE_COLUMN}__{alias}" for alias in included]
        )
    else:
        dimensions = [
            dimension
            for dimension in DIMENSION_COLUMNS
            if dimension != variant.excluded_dimension
        ]
        if not dimensions:
            raise ValueError("Breadth variant cannot exclude all dimensions")
        combined_dimensions = pd.DataFrame(
            {
                dimension: combine(
                    [f"{dimension}__{alias}" for alias in included]
                )
                for dimension in dimensions
            },
            index=article.index,
        )
        result = combined_dimensions.mean(axis=1, skipna=False)
    if result.isna().any():
        raise ValueError(f"Variant {variant.variant_id} produced missing scores")
    if not result.between(0, 4).all():
        raise ValueError(f"Variant {variant.variant_id} leaves the 0--4 score scale")
    return result.astype(float)


def variant_article_mask(article: pd.DataFrame, variant: IndexVariant) -> pd.Series:
    if variant.filter_rule == "all_common_articles":
        mask = pd.Series(True, index=article.index)
    elif variant.filter_rule == "exclude_freight_price_only":
        mask = ~article["freight_price_only_profile"].astype(bool)
    elif variant.filter_rule == "exclude_low_model_support":
        column = f"{variant.path}_positive_model_support"
        # Zero-disruption articles stay in coverage; only a positive signal that
        # appears in exactly one LLM is treated as low cross-model support.
        mask = ~article[column].eq(1)
    elif variant.filter_rule == "exclude_top_ten_sources":
        top = set(article["source_domain"].value_counts().head(10).index)
        mask = ~article["source_domain"].isin(top)
    elif variant.filter_rule == "exclude_one_source":
        if not variant.excluded_source:
            raise ValueError("exclude_one_source requires excluded_source")
        mask = ~article["source_domain"].eq(variant.excluded_source)
    else:
        raise ValueError(f"Unknown article filter: {variant.filter_rule}")
    return mask.astype(bool)


def _standardize_past_only(values: pd.Series, method: str) -> pd.Series:
    if method == "none":
        return values.astype(float)
    history = values.shift(1)
    if method == "expanding":
        mean = history.expanding(min_periods=2).mean()
        scale = history.expanding(min_periods=2).std(ddof=0)
    elif method.startswith("rolling_"):
        window = int(method.split("_", maxsplit=1)[1])
        mean = history.rolling(window, min_periods=2).mean()
        scale = history.rolling(window, min_periods=2).std(ddof=0)
    else:
        raise ValueError(f"Unknown standardization: {method}")
    standardized = (values - mean).div(scale.where(scale.gt(0)))
    return standardized.fillna(0.0)


def aggregate_weekly_variant(
    article: pd.DataFrame,
    variant: IndexVariant,
    aliases: Sequence[str],
    frequency: str = "W-FRI",
    seed: int = 20260803,
) -> pd.DataFrame:
    mask = variant_article_mask(article, variant)
    work = article.loc[mask].copy()
    work["article_score"] = article_variant_scores(work, variant, aliases)
    work["duplicate_weight"] = duplicate_weights(
        work["duplicate_group_size"], variant.duplicate_penalty
    )
    work["weighted_score"] = work["article_score"] * work["duplicate_weight"]
    if variant.permute_dates:
        rng = np.random.default_rng(stable_seed(seed, variant.variant_id, "dates"))
        work["publication_date"] = rng.permutation(
            work["publication_date"].to_numpy()
        )
    grouped = work.set_index("publication_date").resample(
        frequency, label="right", closed="right"
    ).agg(
        article_count=("article_id", "nunique"),
        source_count=("source_domain", "nunique"),
        effective_article_count=("duplicate_weight", "sum"),
        weighted_score_sum=("weighted_score", "sum"),
        unweighted_score_mean=("article_score", "mean"),
        active_article_count=("article_score", lambda value: int(value.gt(0).sum())),
    )
    first = article["publication_date"].min().to_period("W-FRI").end_time.normalize()
    last = article["publication_date"].max().to_period("W-FRI").end_time.normalize()
    grouped = grouped.reindex(pd.date_range(first, last, freq=frequency))
    grouped.index.name = "week_ending"
    fill = [
        "article_count",
        "source_count",
        "effective_article_count",
        "weighted_score_sum",
        "active_article_count",
    ]
    grouped[fill] = grouped[fill].fillna(0.0)
    grouped["signal_value"] = media_correct(
        grouped["weighted_score_sum"],
        grouped["article_count"],
        grouped["effective_article_count"],
        variant.media_correction,
    )
    grouped["signal_value"] = _standardize_past_only(
        grouped["signal_value"], variant.standardization
    )
    grouped["variant_id"] = variant.variant_id
    grouped["path"] = variant.path
    grouped["score_rule"] = variant.score_rule
    grouped["included_model_count"] = len(variant.included_aliases or tuple(aliases))
    grouped["included_article_count_total"] = int(mask.sum())
    return grouped.reset_index()


def build_index_variants(
    article: pd.DataFrame,
    aliases: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[list[IndexVariant], list[dict[str, Any]]]:
    requested = config.get("measurement_variants", {})
    variants: list[IndexVariant] = [
        IndexVariant(
            variant_id="reference_peak_median",
            path="peak",
            family_id="reference",
            included_aliases=tuple(aliases),
            primary=True,
            description="Frozen median overall_disruption_max path",
        ),
        IndexVariant(
            variant_id="reference_breadth_median7",
            path="breadth",
            family_id="reference",
            included_aliases=tuple(aliases),
            description="Frozen equal mean of seven dimension medians",
        ),
    ]
    if requested.get("equal_model_mean", True):
        for path in PATHS:
            variants.append(
                IndexVariant(
                    variant_id=f"score_mean_{path}",
                    path=path,
                    family_id="score_aggregation",
                    score_rule="mean",
                    included_aliases=tuple(aliases),
                    comparison_reference=(
                        "reference_peak_median"
                        if path == "peak"
                        else "reference_breadth_median7"
                    ),
                    description="Equal model mean instead of cross-model median",
                )
            )
    if requested.get("individual_llms", True):
        for alias in aliases:
            for path in PATHS:
                variants.append(
                    IndexVariant(
                        variant_id=f"individual_{alias}_{path}",
                        path=path,
                        family_id="individual_llm",
                        included_aliases=(alias,),
                        comparison_reference=(
                            "reference_peak_median"
                            if path == "peak"
                            else "reference_breadth_median7"
                        ),
                        description=f"Individual score path from {alias}",
                    )
                )
    if requested.get("leave_one_llm_out", True):
        for omitted in aliases:
            included = tuple(alias for alias in aliases if alias != omitted)
            for path in PATHS:
                variants.append(
                    IndexVariant(
                        variant_id=f"leave_{omitted}_out_{path}",
                        path=path,
                        family_id="leave_one_llm_out",
                        included_aliases=included,
                        comparison_reference=(
                            "reference_peak_median"
                            if path == "peak"
                            else "reference_breadth_median7"
                        ),
                        description=f"Two-model median after removing {omitted}",
                    )
                )
    if requested.get("leave_one_dimension_out", True):
        for dimension in DIMENSION_COLUMNS:
            variants.append(
                IndexVariant(
                    variant_id=f"leave_{dimension.removesuffix('_score')}_out_breadth",
                    path="breadth",
                    family_id="leave_one_dimension_out",
                    included_aliases=tuple(aliases),
                    excluded_dimension=dimension,
                    comparison_reference="reference_breadth_median7",
                    description=f"Mean of six dimensions after removing {dimension}",
                )
            )
    for method in requested.get("media_corrections", ["sqrt", "mean", "none"]):
        if method == "log":
            continue
        for path in PATHS:
            variants.append(
                IndexVariant(
                    variant_id=f"media_{method}_{path}",
                    path=path,
                    family_id="media_correction",
                    included_aliases=tuple(aliases),
                    media_correction=str(method),
                    comparison_reference=(
                        "reference_peak_median"
                        if path == "peak"
                        else "reference_breadth_median7"
                    ),
                    description=f"Alternative {method} media-volume correction",
                )
            )
    for method in requested.get("duplicate_penalties", ["inverse", "none"]):
        if method == "sqrt":
            continue
        for path in PATHS:
            variants.append(
                IndexVariant(
                    variant_id=f"duplicate_{method}_{path}",
                    path=path,
                    family_id="duplicate_penalty",
                    included_aliases=tuple(aliases),
                    duplicate_penalty=str(method),
                    comparison_reference=(
                        "reference_peak_median"
                        if path == "peak"
                        else "reference_breadth_median7"
                    ),
                    description=f"Alternative {method} duplicate penalty",
                )
            )
    filters = [
        ("exclude_freight_price_only", "content_exclusion"),
        ("exclude_low_model_support", "evidence_support"),
        ("exclude_top_ten_sources", "source_exclusion"),
    ]
    enabled_filters = set(requested.get("article_filters", [item[0] for item in filters]))
    for filter_rule, family in filters:
        if filter_rule not in enabled_filters:
            continue
        for path in PATHS:
            variants.append(
                IndexVariant(
                    variant_id=f"{filter_rule}_{path}",
                    path=path,
                    family_id=family,
                    included_aliases=tuple(aliases),
                    filter_rule=filter_rule,
                    comparison_reference=(
                        "reference_peak_median"
                        if path == "peak"
                        else "reference_breadth_median7"
                    ),
                    description=f"Frozen article filter: {filter_rule}",
                )
            )
    if requested.get("leave_top_sources_out", True):
        top_n = int(requested.get("top_source_count", 10))
        top_sources = article["source_domain"].value_counts().head(top_n).index
        for rank, source in enumerate(top_sources, start=1):
            slug = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:8]
            for path in PATHS:
                variants.append(
                    IndexVariant(
                        variant_id=f"leave_source_{rank}_{slug}_out_{path}",
                        path=path,
                        family_id="leave_one_source_out",
                        included_aliases=tuple(aliases),
                        filter_rule="exclude_one_source",
                        excluded_source=str(source),
                        comparison_reference=(
                            "reference_peak_median"
                            if path == "peak"
                            else "reference_breadth_median7"
                        ),
                        description=f"Leave source rank {rank} ({source}) out",
                    )
                )
    for method in requested.get("standardization", ["expanding", "rolling_52"]):
        if method == "none":
            continue
        for path in PATHS:
            variants.append(
                IndexVariant(
                    variant_id=f"standardize_{method}_{path}",
                    path=path,
                    family_id="standardization",
                    included_aliases=tuple(aliases),
                    standardization=str(method),
                    comparison_reference=(
                        "reference_peak_median"
                        if path == "peak"
                        else "reference_breadth_median7"
                    ),
                    description=f"Past-only {method} index standardization",
                )
            )
    if requested.get("permuted_article_dates", True):
        for path in PATHS:
            variants.append(
                IndexVariant(
                    variant_id=f"permuted_dates_{path}",
                    path=path,
                    family_id="date_placebo",
                    included_aliases=tuple(aliases),
                    permute_dates=True,
                    comparison_reference=(
                        "reference_peak_median"
                        if path == "peak"
                        else "reference_breadth_median7"
                    ),
                    description="Deterministic article-date permutation placebo",
                )
            )

    ids = [variant.variant_id for variant in variants]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise AssertionError(f"Duplicate index variant IDs: {duplicates}")
    skipped = [
        {
            "specification_id": "event_date_assignment",
            "analysis_layer": "measurement",
            "status": "skipped_unavailable",
            "reason": "no_extracted_event_date_in_frozen_article_panel",
        },
        {
            "specification_id": "revision_real_time_vintage",
            "analysis_layer": "measurement",
            "status": "skipped_unavailable",
            "reason": "no_historical_data_vintages_available",
        },
        {
            "specification_id": "winsorized_scores",
            "analysis_layer": "measurement",
            "status": "skipped_not_applicable",
            "reason": "all_frozen_severity_scores_are_bounded_on_0_to_4_scale",
        },
        {
            "specification_id": "formal_negative_control_category",
            "analysis_layer": "causal_ml",
            "status": "skipped_unavailable",
            "reason": "no_pre_specified_negative_control_article_category",
        },
        {
            "specification_id": "dml_iv",
            "analysis_layer": "causal_ml",
            "status": "skipped_unavailable",
            "reason": "no_defensible_instrument_encoded",
        },
        {
            "specification_id": "causal_forest",
            "analysis_layer": "causal_ml",
            "status": "skipped_identification_gate",
            "reason": "average_effect_identification_and_route_panel_gates_not_met",
        },
    ]
    return variants, skipped


def aggregate_all_weekly_variants(
    article: pd.DataFrame,
    variants: Sequence[IndexVariant],
    aliases: Sequence[str],
    seed: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for number, variant in enumerate(variants, start=1):
        started = time.perf_counter()
        frame = aggregate_weekly_variant(
            article, variant, aliases, frequency="W-FRI", seed=seed
        )
        frames.append(frame)
        logger.info(
            "index_variant %s/%s id=%s path=%s weeks=%s articles=%s "
            "mean=%.6f sd=%.6f runtime=%.3fs",
            number,
            len(variants),
            variant.variant_id,
            variant.path,
            len(frame),
            int(frame["included_article_count_total"].iloc[0]),
            frame["signal_value"].mean(),
            frame["signal_value"].std(ddof=1),
            time.perf_counter() - started,
        )
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["variant_id", "week_ending"]).any():
        raise AssertionError("Weekly index variants contain duplicate keys")
    return result.sort_values(["variant_id", "week_ending"]).reset_index(drop=True)


def load_weekly_base_panel(path: Path) -> pd.DataFrame:
    resolved = require_file(path, "Weekly SCFI panel")
    required = {
        "observation_date",
        "scfi",
        *WEEKLY_AR_FEATURES,
        *WEEKLY_BRENT_FEATURES,
        *WEEKLY_EXTERNAL_CONTROL_FEATURES,
        *[f"target_scfi_log_change_h{h}w" for h in (1, 2, 4, 8, 12)],
        *[f"target_scfi_level_h{h}w" for h in (1, 2, 4, 8, 12)],
    }
    header = pd.read_csv(resolved, nrows=0)
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(f"Weekly panel lacks required columns: {missing}")
    panel = pd.read_csv(resolved, usecols=sorted(required), low_memory=False)
    panel["observation_date"] = pd.to_datetime(
        panel["observation_date"], format="mixed", errors="coerce"
    )
    if panel["observation_date"].isna().any():
        raise ValueError("Weekly panel contains unparseable dates")
    panel = panel.sort_values("observation_date").reset_index(drop=True)
    if panel["observation_date"].duplicated().any():
        raise ValueError("Weekly panel contains duplicate observation dates")
    if not panel["observation_date"].diff().dropna().dt.days.eq(7).all():
        raise ValueError("Weekly panel must have a regular seven-day cadence")
    return panel


def load_monthly_base_panel(path: Path) -> pd.DataFrame:
    resolved = require_file(path, "Monthly SCFI panel")
    required = {
        "month_ending",
        "scfi_monthly_mean",
        *MONTHLY_AR_FEATURES,
        *MONTHLY_BRENT_FEATURES,
        *MONTHLY_EXTERNAL_CONTROL_FEATURES,
        *[f"target_scfi_monthly_log_change_h{h}m" for h in (1, 3, 6)],
        *[f"target_scfi_monthly_mean_h{h}m" for h in (1, 3, 6)],
    }
    header = pd.read_csv(resolved, nrows=0)
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(f"Monthly panel lacks required columns: {missing}")
    panel = pd.read_csv(resolved, usecols=sorted(required), low_memory=False)
    panel["month_ending"] = pd.to_datetime(
        panel["month_ending"], format="mixed", errors="coerce"
    )
    if panel["month_ending"].isna().any():
        raise ValueError("Monthly panel contains unparseable dates")
    panel = panel.sort_values("month_ending").reset_index(drop=True)
    if panel["month_ending"].duplicated().any():
        raise ValueError("Monthly panel contains duplicate month ends")
    periods = panel["month_ending"].dt.to_period("M").astype("int64")
    if not periods.diff().dropna().eq(1).all():
        raise ValueError("Monthly panel must have a regular monthly cadence")
    return panel


def _variant_column(prefix: str, variant_id: str, construction: str | None = None) -> str:
    if construction is None:
        return f"robust_{prefix}__{variant_id}"
    return f"robust_{prefix}__{construction}__{variant_id}"


def align_weekly_variants_to_panel(
    base_panel: pd.DataFrame,
    weekly_long: pd.DataFrame,
    maximum_age_days: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_columns = {
        "signal_value": "signal",
        "article_count": "article_count",
        "source_count": "source_count",
        "effective_article_count": "effective_count",
        "weighted_score_sum": "weighted_sum",
    }
    wide_parts: list[pd.DataFrame] = []
    for source, prefix in value_columns.items():
        pivot = weekly_long.pivot(
            index="week_ending", columns="variant_id", values=source
        )
        pivot.columns = [_variant_column(prefix, str(column)) for column in pivot]
        wide_parts.append(pivot)
    wide = pd.concat(wide_parts, axis=1).reset_index()
    wide = wide.rename(columns={"week_ending": "robust_week_ending"})
    wide["robust_week_ending"] = pd.to_datetime(wide["robust_week_ending"])
    panel = pd.merge_asof(
        base_panel.sort_values("observation_date"),
        wide.sort_values("robust_week_ending"),
        left_on="observation_date",
        right_on="robust_week_ending",
        direction="backward",
        tolerance=pd.Timedelta(days=maximum_age_days),
        allow_exact_matches=True,
    )
    panel["robust_signal_age_days"] = (
        panel["observation_date"] - panel["robust_week_ending"]
    ).dt.days
    matched = panel["robust_week_ending"].notna()
    future = matched & panel["robust_week_ending"].gt(panel["observation_date"])
    stale = matched & ~panel["robust_signal_age_days"].between(0, maximum_age_days)
    if future.any() or stale.any():
        raise AssertionError(
            f"Unsafe weekly signal alignment: future={future.sum()} stale={stale.sum()}"
        )
    audit = pd.DataFrame(
        [
            {
                "frequency": "weekly",
                "rows": len(panel),
                "matched_rows": int(matched.sum()),
                "unmatched_rows": int((~matched).sum()),
                "future_matches": int(future.sum()),
                "stale_matches": int(stale.sum()),
                "maximum_age_days": maximum_age_days,
                "first_matched_date": panel.loc[matched, "observation_date"].min(),
                "last_matched_date": panel.loc[matched, "observation_date"].max(),
            }
        ]
    )
    return panel.reset_index(drop=True), audit


def _ema_restart_after_gaps(values: pd.Series, span: int) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    available = values.notna()
    if not available.any():
        return result
    blocks = available.ne(available.shift(fill_value=False)).cumsum()
    for _, indices in values[available].groupby(blocks[available]).groups.items():
        result.loc[indices] = values.loc[indices].ewm(
            span=span, adjust=False, min_periods=1
        ).mean()
    return result


def build_monthly_variant_panel(
    weekly_aligned: pd.DataFrame,
    monthly_base: pd.DataFrame,
    variants: Sequence[IndexVariant],
    ema_span: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weekly = weekly_aligned.copy()
    weekly["month_ending"] = (
        weekly["observation_date"].dt.to_period("M").dt.to_timestamp("M")
    )
    counts = weekly.groupby("month_ending", sort=True).agg(
        observed_mondays=("observation_date", "size"),
        matched_mondays=("robust_week_ending", lambda value: int(value.notna().sum())),
    )
    counts["expected_mondays"] = [
        len(
            pd.date_range(
                month.to_period("M").start_time.normalize(),
                month.to_period("M").end_time.normalize(),
                freq="W-MON",
            )
        )
        for month in counts.index
    ]
    counts["dates_complete"] = counts["observed_mondays"].eq(
        counts["expected_mondays"]
    )
    monthly = monthly_base.copy().set_index("month_ending")
    monthly = monthly.join(counts, how="left")
    long_rows: list[pd.DataFrame] = []
    for variant in variants:
        signal = _variant_column("signal", variant.variant_id)
        article_count = _variant_column("article_count", variant.variant_id)
        effective = _variant_column("effective_count", variant.variant_id)
        source_count = _variant_column("source_count", variant.variant_id)
        weighted = _variant_column("weighted_sum", variant.variant_id)
        work = weekly[
            [
                "month_ending",
                signal,
                article_count,
                source_count,
                effective,
                weighted,
            ]
        ].copy()
        grouped = work.groupby("month_ending", sort=True).agg(
            observed_rows=(signal, "size"),
            signal_rows=(signal, lambda value: int(value.notna().sum())),
            signal_mean_safe_weekly=(signal, "mean"),
            article_count_sum=(article_count, "sum"),
            source_count_mean=(source_count, "mean"),
            effective_article_count_sum=(effective, "sum"),
            weighted_score_sum=(weighted, "sum"),
        )
        grouped = grouped.join(counts[["expected_mondays", "dates_complete"]])
        complete = (
            grouped["dates_complete"].astype(bool)
            & grouped["observed_rows"].eq(grouped["expected_mondays"])
            & grouped["signal_rows"].eq(grouped["observed_rows"])
        )
        grouped.loc[~complete, "signal_mean_safe_weekly"] = np.nan
        denominator = np.sqrt(grouped["effective_article_count_sum"])
        grouped["adapted_pdf_sqrt_base"] = grouped["weighted_score_sum"].div(
            denominator.where(denominator.gt(0))
        )
        grouped.loc[~complete, "adapted_pdf_sqrt_base"] = np.nan
        grouped["signal_adapted_pdf_sqrt_ema3"] = _ema_restart_after_gaps(
            grouped["adapted_pdf_sqrt_base"], ema_span
        )
        grouped["variant_id"] = variant.variant_id
        grouped["path"] = variant.path
        long = grouped.reset_index()
        long_rows.append(long)
        for construction, source in (
            ("mean_safe_weekly", "signal_mean_safe_weekly"),
            ("adapted_pdf_sqrt_ema3", "signal_adapted_pdf_sqrt_ema3"),
        ):
            monthly[_variant_column("signal", variant.variant_id, construction)] = (
                grouped[source]
            )
            monthly[
                _variant_column("article_count", variant.variant_id, construction)
            ] = grouped["article_count_sum"]
            monthly[
                _variant_column("effective_count", variant.variant_id, construction)
            ] = grouped["effective_article_count_sum"]
            monthly[
                _variant_column("source_count", variant.variant_id, construction)
            ] = grouped["source_count_mean"]
    long_frame = pd.concat(long_rows, ignore_index=True)
    monthly = monthly.reset_index()
    audit = pd.DataFrame(
        [
            {
                "frequency": "monthly",
                "rows": len(monthly),
                "months_with_weekly_rows": int(monthly["observed_mondays"].notna().sum()),
                "months_all_weekly_signals_matched": int(
                    (
                        monthly["dates_complete"].fillna(False).astype(bool)
                        & monthly["observed_mondays"].eq(monthly["matched_mondays"])
                    ).sum()
                ),
                "constructions": "|".join(MONTHLY_CONSTRUCTIONS),
                "partial_months_zero_filled": False,
                "ema_span": ema_span,
                "month_assignment": "calendar_month_of_safe_monday_information_date",
            }
        ]
    )
    return monthly, long_frame, audit


def build_common_cohort_analysis_panels(
    weekly_aligned: pd.DataFrame,
    monthly_aligned: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    peak = "reference_peak_median"
    breadth = "reference_breadth_median7"
    weekly = pd.DataFrame(
        {
            "observation_date": weekly_aligned["observation_date"],
            "scfi": weekly_aligned["scfi"],
            "brent_price": np.exp(weekly_aligned["brent_log"]),
            "article_count": weekly_aligned[_variant_column("article_count", peak)],
            "source_count": weekly_aligned[_variant_column("source_count", peak)],
            "peak_score_media_corrected": weekly_aligned[
                _variant_column("signal", peak)
            ],
            "breadth_score_media_corrected": weekly_aligned[
                _variant_column("signal", breadth)
            ],
        }
    )
    construction = "mean_safe_weekly"
    monthly = pd.DataFrame(
        {
            "month_ending": monthly_aligned["month_ending"],
            "scfi_monthly_mean": monthly_aligned["scfi_monthly_mean"],
            "brent_monthly_mean": np.exp(
                monthly_aligned["brent_monthly_mean_log"]
            ),
            "article_count_sum": monthly_aligned[
                _variant_column("article_count", peak, construction)
            ],
            "source_count_mean": monthly_aligned[
                _variant_column("source_count", peak, construction)
            ],
            "peak_score_media_corrected_mean": monthly_aligned[
                _variant_column("signal", peak, construction)
            ],
            "breadth_score_media_corrected_mean": monthly_aligned[
                _variant_column("signal", breadth, construction)
            ],
            "msdi_available_weeks": monthly_aligned["matched_mondays"],
            "weekly_observations_in_month": monthly_aligned["expected_mondays"],
        }
    )
    return {"weekly": weekly, "monthly": monthly}


def _reconstruction_row(
    reconstructed: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    layer: str,
    frequency: str,
    construction: str,
    path_name: str,
    date_column: str,
    reconstructed_column: str,
    reference_column: str,
    tolerance: float,
) -> dict[str, Any]:
    left = reconstructed[[date_column, reconstructed_column]].rename(
        columns={reconstructed_column: "reconstructed_value"}
    )
    right = reference[[date_column, reference_column]].rename(
        columns={reference_column: "reference_value"}
    )
    if left[date_column].duplicated().any() or right[date_column].duplicated().any():
        raise ValueError(f"Duplicate dates prevent {layer} reconstruction audit")
    matched = left.merge(right, on=date_column, how="outer", indicator=True)
    both = matched.loc[
        matched["reconstructed_value"].notna()
        & matched["reference_value"].notna()
    ].copy()
    if both.empty:
        raise ValueError(f"No matched values for {layer} reconstruction audit")
    difference = (both["reconstructed_value"] - both["reference_value"]).abs()
    maximum = float(difference.max())
    passed = bool(np.isfinite(maximum) and maximum <= tolerance)
    return {
        "audit_layer": layer,
        "frequency": frequency,
        "index_construction": construction,
        "msdi_path": path_name,
        "date_column": date_column,
        "reconstructed_column": reconstructed_column,
        "reference_column": reference_column,
        "matched_nonmissing_rows": len(both),
        "reconstructed_nonmissing_rows": int(
            matched["reconstructed_value"].notna().sum()
        ),
        "reference_nonmissing_rows": int(matched["reference_value"].notna().sum()),
        "reconstructed_only_dates": int(
            (
                matched["reconstructed_value"].notna()
                & matched["reference_value"].isna()
            ).sum()
        ),
        "reference_only_dates": int(
            (
                matched["reference_value"].notna()
                & matched["reconstructed_value"].isna()
            ).sum()
        ),
        "sample_start_date": both[date_column].min(),
        "sample_end_date": both[date_column].max(),
        "maximum_absolute_difference": maximum,
        "mean_absolute_difference": float(difference.mean()),
        "tolerance": tolerance,
        "contract_passed": passed,
    }


def reference_reconstruction_audit(
    article: pd.DataFrame,
    weekly_long: pd.DataFrame,
    monthly_long: pd.DataFrame,
    weekly_reference_panel_path: Path,
    monthly_reference_panel_path: Path,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Fail closed unless stage-09 references reproduce stages 07b/07c."""
    expected_weight = duplicate_weights(article["duplicate_group_size"], "sqrt")
    novelty_difference = (article["novelty_weight"] - expected_weight).abs()
    novelty_maximum = float(novelty_difference.max())
    rows: list[dict[str, Any]] = [
        {
            "audit_layer": "article_novelty_weight",
            "frequency": "article",
            "index_construction": "sqrt_duplicate_weight",
            "msdi_path": "both",
            "date_column": "publication_date",
            "reconstructed_column": "one_over_sqrt_duplicate_group_size",
            "reference_column": "novelty_weight",
            "matched_nonmissing_rows": int(novelty_difference.notna().sum()),
            "reconstructed_nonmissing_rows": int(expected_weight.notna().sum()),
            "reference_nonmissing_rows": int(article["novelty_weight"].notna().sum()),
            "reconstructed_only_dates": 0,
            "reference_only_dates": 0,
            "sample_start_date": article["publication_date"].min(),
            "sample_end_date": article["publication_date"].max(),
            "maximum_absolute_difference": novelty_maximum,
            "mean_absolute_difference": float(novelty_difference.mean()),
            "tolerance": tolerance,
            "contract_passed": bool(novelty_maximum <= tolerance),
        }
    ]

    weekly_columns = {
        "date": "llm_week_ending",
        "peak": "llm_signal__score_median__overall_disruption_max",
        "breadth": "llm_signal__score_median__dimension_mean7",
        "article_count": "llm_common_article_count",
        "effective_count": "llm_common_effective_article_count",
    }
    weekly_reference = pd.read_csv(
        require_file(weekly_reference_panel_path, "Stage-07b aligned weekly panel"),
        usecols=list(weekly_columns.values()),
        low_memory=False,
    ).rename(columns={weekly_columns["date"]: "week_ending"})
    weekly_reference["week_ending"] = pd.to_datetime(
        weekly_reference["week_ending"], format="mixed", errors="coerce"
    )
    weekly_reference = weekly_reference.loc[
        weekly_reference["week_ending"].notna()
    ].drop_duplicates("week_ending")
    weekly_sources = {
        "peak": weekly_long.loc[
            weekly_long["variant_id"].eq("reference_peak_median")
        ],
        "breadth": weekly_long.loc[
            weekly_long["variant_id"].eq("reference_breadth_median7")
        ],
    }
    for path_name, reference_column in (
        ("peak", weekly_columns["peak"]),
        ("breadth", weekly_columns["breadth"]),
    ):
        rows.append(
            _reconstruction_row(
                weekly_sources[path_name],
                weekly_reference,
                layer="reference_weekly_signal",
                frequency="weekly",
                construction="weekly_log_volume",
                path_name=path_name,
                date_column="week_ending",
                reconstructed_column="signal_value",
                reference_column=reference_column,
                tolerance=tolerance,
            )
        )
    for reconstructed_column, reference_column in (
        ("article_count", weekly_columns["article_count"]),
        ("effective_article_count", weekly_columns["effective_count"]),
    ):
        rows.append(
            _reconstruction_row(
                weekly_sources["peak"],
                weekly_reference,
                layer="reference_weekly_coverage",
                frequency="weekly",
                construction="weekly_log_volume",
                path_name="both",
                date_column="week_ending",
                reconstructed_column=reconstructed_column,
                reference_column=reference_column,
                tolerance=tolerance,
            )
        )

    monthly_columns = {
        "date": "month_ending",
        "mean_peak": "llm_signal__score_median__overall_disruption_max",
        "mean_breadth": "llm_signal__score_median__dimension_mean7",
        "adapted_peak": "llm_monthly_signal__adapted_pdf_sqrt_ema3__score_median__overall_disruption_max",
        "adapted_breadth": "llm_monthly_signal__adapted_pdf_sqrt_ema3__score_median__dimension_mean7",
        "article_count": "llm_common_article_count",
        "effective_count": "llm_common_effective_article_count",
    }
    monthly_reference = pd.read_csv(
        require_file(monthly_reference_panel_path, "Stage-07c aligned monthly panel"),
        usecols=list(monthly_columns.values()),
        low_memory=False,
    )
    monthly_reference["month_ending"] = pd.to_datetime(
        monthly_reference["month_ending"], format="mixed", errors="coerce"
    )
    for path_name, variant_id in (
        ("peak", "reference_peak_median"),
        ("breadth", "reference_breadth_median7"),
    ):
        source = monthly_long.loc[monthly_long["variant_id"].eq(variant_id)]
        for construction, reconstructed_column, reference_column in (
            (
                "mean_safe_weekly",
                "signal_mean_safe_weekly",
                monthly_columns[f"mean_{path_name}"],
            ),
            (
                "adapted_pdf_sqrt_ema3",
                "signal_adapted_pdf_sqrt_ema3",
                monthly_columns[f"adapted_{path_name}"],
            ),
        ):
            rows.append(
                _reconstruction_row(
                    source,
                    monthly_reference,
                    layer="reference_monthly_signal",
                    frequency="monthly",
                    construction=construction,
                    path_name=path_name,
                    date_column="month_ending",
                    reconstructed_column=reconstructed_column,
                    reference_column=reference_column,
                    tolerance=tolerance,
                )
            )
    monthly_peak = monthly_long.loc[
        monthly_long["variant_id"].eq("reference_peak_median")
    ]
    for reconstructed_column, reference_column in (
        ("article_count_sum", monthly_columns["article_count"]),
        ("effective_article_count_sum", monthly_columns["effective_count"]),
    ):
        rows.append(
            _reconstruction_row(
                monthly_peak,
                monthly_reference,
                layer="reference_monthly_coverage",
                frequency="monthly",
                construction="both",
                path_name="both",
                date_column="month_ending",
                reconstructed_column=reconstructed_column,
                reference_column=reference_column,
                tolerance=tolerance,
            )
        )
    audit = pd.DataFrame(rows)
    if not audit["contract_passed"].all():
        failed = audit.loc[~audit["contract_passed"]]
        raise AssertionError(
            "Reference reconstruction contract failed:\n"
            + failed.to_string(index=False)
        )
    return audit


def add_signal_features(
    frame: pd.DataFrame,
    raw_columns: Sequence[str],
    lags: Sequence[int],
    windows: Sequence[int],
    unit: str,
) -> pd.DataFrame:
    if unit not in {"w", "m"}:
        raise ValueError("Signal feature unit must be 'w' or 'm'")
    result = frame.copy()
    derived: dict[str, pd.Series] = {}
    for column in raw_columns:
        if column not in result:
            raise ValueError(f"Missing raw signal column: {column}")
        for lag in lags:
            derived[f"{column}_lag_{lag}{unit}"] = result[column].shift(int(lag))
        for window in windows:
            rolling = result[column].rolling(int(window), min_periods=int(window))
            derived[f"{column}_rolling_mean_{window}{unit}"] = rolling.mean()
            derived[f"{column}_rolling_sd_{window}{unit}"] = rolling.std(ddof=0)
        derived[f"{column}_acceleration_1{unit}"] = result[column].diff().diff()
    if derived:
        result = pd.concat([result, pd.DataFrame(derived, index=result.index)], axis=1)
    return result


def _signal_feature_names(
    raw_column: str, lags: Sequence[int], windows: Sequence[int], unit: str
) -> list[str]:
    return [
        raw_column,
        *[f"{raw_column}_lag_{lag}{unit}" for lag in lags],
        *[
            item
            for window in windows
            for item in (
                f"{raw_column}_rolling_mean_{window}{unit}",
                f"{raw_column}_rolling_sd_{window}{unit}",
            )
        ],
        f"{raw_column}_acceleration_1{unit}",
    ]


def prepare_forecast_panels(
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
    variants: Sequence[IndexVariant],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly_config = config["forecast"]["weekly"]
    monthly_config = config["forecast"]["monthly"]
    weekly_lags = sorted(
        {
            int(item)
            for values in weekly_config["lag_sets"].values()
            for item in values
        }
    )
    weekly_windows = sorted(
        {
            int(item)
            for values in weekly_config["rolling_window_sets"].values()
            for item in values
        }
    )
    monthly_lags = sorted(
        {
            int(item)
            for values in monthly_config["lag_sets"].values()
            for item in values
        }
    )
    monthly_windows = sorted(
        {
            int(item)
            for values in monthly_config["rolling_window_sets"].values()
            for item in values
        }
    )
    weekly_raw = [_variant_column("signal", item.variant_id) for item in variants]
    weekly = add_signal_features(
        weekly, weekly_raw, weekly_lags, weekly_windows, "w"
    )
    monthly_raw = [
        _variant_column("signal", item.variant_id, construction)
        for construction in MONTHLY_CONSTRUCTIONS
        for item in variants
    ]
    monthly = add_signal_features(
        monthly, monthly_raw, monthly_lags, monthly_windows, "m"
    )
    return weekly, monthly


def build_forecast_specifications(
    variants: Sequence[IndexVariant],
    frequency: str,
    config: Mapping[str, Any],
) -> tuple[list[ForecastSpecification], list[PairedComparison]]:
    values = config["forecast"][frequency]
    default_lags = tuple(int(item) for item in values["lag_sets"]["primary"])
    default_windows = tuple(
        int(item) for item in values["rolling_window_sets"]["primary"]
    )
    refit_every = int(values["refit_every"])
    coverage = "reference_peak_median"
    specs: list[ForecastSpecification] = [
        ForecastSpecification(
            specification_id="no_msdi",
            family_id="component_ablation",
            signal_variants=(),
            coverage_variant=coverage,
            signal_lags=default_lags,
            rolling_windows=default_windows,
            refit_every=refit_every,
            description="AR, Brent, and frozen common-cohort coverage only",
        ),
        ForecastSpecification(
            specification_id="peak_only",
            family_id="component_ablation",
            signal_variants=("reference_peak_median",),
            coverage_variant=coverage,
            signal_lags=default_lags,
            rolling_windows=default_windows,
            refit_every=refit_every,
            description="Primary median overall-disruption path",
        ),
        ForecastSpecification(
            specification_id="breadth_only",
            family_id="component_ablation",
            signal_variants=("reference_breadth_median7",),
            coverage_variant=coverage,
            signal_lags=default_lags,
            rolling_windows=default_windows,
            refit_every=refit_every,
            description="Secondary seven-dimension breadth path",
        ),
        ForecastSpecification(
            specification_id="peak_and_breadth",
            family_id="component_ablation",
            signal_variants=(
                "reference_peak_median",
                "reference_breadth_median7",
            ),
            coverage_variant=coverage,
            signal_lags=default_lags,
            rolling_windows=default_windows,
            refit_every=refit_every,
            description="Both measurement paths in one Ridge model",
        ),
    ]
    comparisons: list[PairedComparison] = [
        PairedComparison(
            comparison_id="peak_vs_no_msdi",
            family_id="primary_weekly_peak_vs_no_msdi",
            baseline_specification_id="no_msdi",
            variant_specification_id="peak_only",
            nested=True,
            primary=frequency == "weekly",
            direction_note="positive baseline_loss_minus_variant_loss favors peak MSDI",
        ),
        PairedComparison(
            comparison_id="breadth_vs_no_msdi",
            family_id="secondary_breadth_vs_no_msdi",
            baseline_specification_id="no_msdi",
            variant_specification_id="breadth_only",
            nested=True,
            primary=False,
            direction_note="positive baseline_loss_minus_variant_loss favors breadth MSDI",
        ),
        PairedComparison(
            comparison_id="incremental_breadth_given_peak",
            family_id="dual_path_ablation",
            baseline_specification_id="peak_only",
            variant_specification_id="peak_and_breadth",
            nested=True,
            primary=False,
            direction_note="positive baseline_loss_minus_variant_loss favors adding breadth",
        ),
        PairedComparison(
            comparison_id="incremental_peak_given_breadth",
            family_id="dual_path_ablation",
            baseline_specification_id="breadth_only",
            variant_specification_id="peak_and_breadth",
            nested=True,
            primary=False,
            direction_note="positive baseline_loss_minus_variant_loss favors adding peak",
        ),
    ]
    reference_ids = {"reference_peak_median", "reference_breadth_median7"}
    frozen_value_families = {
        "leave_one_llm_out",
        "leave_one_dimension_out",
        "media_correction",
        "duplicate_penalty",
        "content_exclusion",
        "evidence_support",
        "source_exclusion",
        "leave_one_source_out",
        "date_placebo",
    }
    for variant in variants:
        if variant.variant_id in reference_ids:
            continue
        # A whole-sample date permutation can move later news labels into
        # earlier forecast origins. Retain it as an index randomization
        # diagnostic, but never call it a leakage-safe OOS forecast placebo.
        if variant.family_id == "date_placebo":
            continue
        specification_id = f"index_variant__{variant.variant_id}"
        specs.append(
            ForecastSpecification(
                specification_id=specification_id,
                family_id=variant.family_id,
                signal_variants=(variant.variant_id,),
                # Hold coverage regressors fixed. The altered article count is
                # used only inside that variant's index denominator.
                coverage_variant=coverage,
                signal_lags=default_lags,
                rolling_windows=default_windows,
                refit_every=refit_every,
                description=variant.description,
            )
        )
        reference_spec = "peak_only" if variant.path == "peak" else "breadth_only"
        if variant.family_id in frozen_value_families:
            baseline, challenger = specification_id, reference_spec
            direction = "positive difference favors frozen full/reference construction"
        else:
            baseline, challenger = reference_spec, specification_id
            direction = "positive difference favors alternative construction"
        comparisons.append(
            PairedComparison(
                comparison_id=f"compare__{variant.variant_id}",
                family_id=variant.family_id,
                baseline_specification_id=baseline,
                variant_specification_id=challenger,
                nested=False,
                primary=False,
                direction_note=direction,
            )
        )

    no_brent = ForecastSpecification(
        specification_id="peak_without_brent_controls",
        family_id="control_ablation",
        signal_variants=("reference_peak_median",),
        coverage_variant=coverage,
        include_brent=False,
        signal_lags=default_lags,
        rolling_windows=default_windows,
        refit_every=refit_every,
        description="Remove Brent controls only",
    )
    no_coverage = ForecastSpecification(
        specification_id="peak_without_coverage_controls",
        family_id="control_ablation",
        signal_variants=("reference_peak_median",),
        coverage_variant=coverage,
        include_coverage=False,
        signal_lags=default_lags,
        rolling_windows=default_windows,
        refit_every=refit_every,
        description="Remove article-count and effective-count controls only",
    )
    specs.extend([no_brent, no_coverage])
    for ablated in (no_brent, no_coverage):
        comparisons.append(
            PairedComparison(
                comparison_id=f"value_of_controls__{ablated.specification_id}",
                family_id="control_ablation",
                baseline_specification_id=ablated.specification_id,
                variant_specification_id="peak_only",
                nested=True,
                primary=False,
                direction_note="positive difference favors retaining the control group",
            )
        )

    for name, lags in values["lag_sets"].items():
        if name == "primary":
            continue
        spec = ForecastSpecification(
            specification_id=f"peak_lags_{name}",
            family_id="lag_sensitivity",
            signal_variants=("reference_peak_median",),
            coverage_variant=coverage,
            signal_lags=tuple(int(item) for item in lags),
            rolling_windows=default_windows,
            refit_every=refit_every,
            description=f"Pre-specified {name} signal lag set",
        )
        specs.append(spec)
        comparisons.append(
            PairedComparison(
                comparison_id=f"lag_sensitivity__{name}",
                family_id="lag_sensitivity",
                baseline_specification_id="peak_only",
                variant_specification_id=spec.specification_id,
                nested=False,
                primary=False,
                direction_note="positive difference favors alternative lag set",
            )
        )
    for name, windows in values["rolling_window_sets"].items():
        if name == "primary":
            continue
        spec = ForecastSpecification(
            specification_id=f"peak_rolling_features_{name}",
            family_id="rolling_feature_sensitivity",
            signal_variants=("reference_peak_median",),
            coverage_variant=coverage,
            signal_lags=default_lags,
            rolling_windows=tuple(int(item) for item in windows),
            refit_every=refit_every,
            description=f"Pre-specified {name} signal rolling-window set",
        )
        specs.append(spec)
        comparisons.append(
            PairedComparison(
                comparison_id=f"rolling_feature_sensitivity__{name}",
                family_id="rolling_feature_sensitivity",
                baseline_specification_id="peak_only",
                variant_specification_id=spec.specification_id,
                nested=False,
                primary=False,
                direction_note="positive difference favors alternative rolling features",
            )
        )
    for window in values.get("training_windows", []):
        if window in (None, "expanding"):
            continue
        spec = ForecastSpecification(
            specification_id=f"peak_training_window_{int(window)}",
            family_id="training_window_sensitivity",
            signal_variants=("reference_peak_median",),
            coverage_variant=coverage,
            signal_lags=default_lags,
            rolling_windows=default_windows,
            training_window=int(window),
            refit_every=refit_every,
            description=f"Rolling training window of {int(window)} observations",
        )
        specs.append(spec)
        comparisons.append(
            PairedComparison(
                comparison_id=f"training_window_sensitivity__{int(window)}",
                family_id="training_window_sensitivity",
                baseline_specification_id="peak_only",
                variant_specification_id=spec.specification_id,
                nested=False,
                primary=False,
                direction_note="positive difference favors rolling training window",
            )
        )
    for alternative_refit in values.get("alternative_refit_every", []):
        alternative_refit = int(alternative_refit)
        if alternative_refit == refit_every:
            continue
        spec = ForecastSpecification(
            specification_id=f"peak_refit_every_{alternative_refit}",
            family_id="refit_sensitivity",
            signal_variants=("reference_peak_median",),
            coverage_variant=coverage,
            signal_lags=default_lags,
            rolling_windows=default_windows,
            refit_every=alternative_refit,
            description=f"Refit Ridge every {alternative_refit} forecast origins",
        )
        specs.append(spec)
        comparisons.append(
            PairedComparison(
                comparison_id=f"refit_sensitivity__{alternative_refit}",
                family_id="refit_sensitivity",
                baseline_specification_id="peak_only",
                variant_specification_id=spec.specification_id,
                nested=False,
                primary=False,
                direction_note="positive difference favors alternative refit interval",
            )
        )
    ids = [item.specification_id for item in specs]
    if len(ids) != len(set(ids)):
        raise AssertionError("Forecast specification IDs are not unique")
    return specs, comparisons


def forecast_feature_columns(
    specification: ForecastSpecification,
    frequency: str,
    construction: str,
) -> list[str]:
    if frequency == "weekly":
        unit = "w"
        base = list(WEEKLY_AR_FEATURES)
        brent = list(WEEKLY_BRENT_FEATURES)
        external_controls = list(WEEKLY_EXTERNAL_CONTROL_FEATURES)
        raw = [
            _variant_column("signal", variant)
            for variant in specification.signal_variants
        ]
        coverage = [
            _variant_column("article_count", specification.coverage_variant),
            _variant_column("effective_count", specification.coverage_variant),
        ]
    elif frequency == "monthly":
        if construction not in MONTHLY_CONSTRUCTIONS:
            raise ValueError(f"Unknown monthly construction: {construction}")
        unit = "m"
        base = list(MONTHLY_AR_FEATURES)
        brent = list(MONTHLY_BRENT_FEATURES)
        external_controls = list(MONTHLY_EXTERNAL_CONTROL_FEATURES)
        raw = [
            _variant_column("signal", variant, construction)
            for variant in specification.signal_variants
        ]
        coverage = [
            _variant_column(
                "article_count", specification.coverage_variant, construction
            ),
            _variant_column(
                "effective_count", specification.coverage_variant, construction
            ),
        ]
    else:
        raise ValueError(f"Unknown forecast frequency: {frequency}")
    features = list(base)
    if specification.include_brent:
        features.extend(brent)
        features.extend(external_controls)
    if specification.include_coverage:
        features.extend(coverage)
    for column in raw:
        features.extend(
            _signal_feature_names(
                column,
                specification.signal_lags,
                specification.rolling_windows,
                unit,
            )
        )
    features = list(dict.fromkeys(features))
    forbidden = [item for item in features if item.startswith("target_")]
    if forbidden:
        raise AssertionError(f"Target leakage in feature list: {forbidden}")
    return features


def forecast_structural_skip_reason(
    specification: ForecastSpecification,
    construction: str,
    variant_map: Mapping[str, IndexVariant],
) -> str | None:
    if construction != "adapted_pdf_sqrt_ema3":
        return None
    if len(specification.signal_variants) != 1:
        return None
    variant = variant_map.get(specification.signal_variants[0])
    if variant is None:
        return None
    if variant.family_id in {"media_correction", "standardization"}:
        return (
            "adapted_pdf_reconstructs_from_weighted_score_sum_and_effective_count;"
            f"{variant.family_id}_variant_is_algebraically_identical_to_reference"
        )
    return None


def purged_training_positions(
    panel: pd.DataFrame,
    origin_position: int,
    horizon: int,
    target_column: str,
    complete_column: str,
    training_window: int | None,
) -> np.ndarray:
    maximum_position = origin_position - horizon
    if maximum_position < 0:
        return np.array([], dtype=int)
    candidates = np.arange(maximum_position + 1, dtype=int)
    complete = panel.loc[candidates, complete_column].astype(bool).to_numpy()
    target = panel.loc[candidates, target_column].notna().to_numpy()
    selected = candidates[complete & target]
    if training_window is not None:
        selected = selected[-int(training_window) :]
    if len(selected) and int(selected[-1]) + horizon > origin_position:
        raise AssertionError("Forecast training labels are not horizon-purged")
    return selected


def make_ridge_estimator(alpha: float) -> TransformedTargetRegressor:
    regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    return TransformedTargetRegressor(
        regressor=regressor, transformer=StandardScaler()
    )


def _forecast_contract(
    frequency: str, horizon: int
) -> tuple[str, str, str, str, str]:
    if frequency == "weekly":
        return (
            "observation_date",
            "scfi",
            f"target_scfi_log_change_h{horizon}w",
            f"target_scfi_level_h{horizon}w",
            "weeks",
        )
    return (
        "month_ending",
        "scfi_monthly_mean",
        f"target_scfi_monthly_log_change_h{horizon}m",
        f"target_scfi_monthly_mean_h{horizon}m",
        "months",
    )


def run_forecast_cell(
    panel: pd.DataFrame,
    specification: ForecastSpecification,
    frequency: str,
    construction: str,
    horizon: int,
    minimum_training_rows: int,
    ridge_alpha: float,
    maximum_origins: int | None,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_column, current_column, target_column, target_level_column, horizon_unit = (
        _forecast_contract(frequency, horizon)
    )
    features = forecast_feature_columns(specification, frequency, construction)
    missing = sorted(set(features).difference(panel.columns))
    if missing:
        raise ValueError(
            f"Forecast specification {specification.specification_id} lacks features: {missing}"
        )
    current_requirements = [current_column]
    if specification.include_brent:
        current_requirements.append(
            "brent_log" if frequency == "weekly" else "brent_monthly_mean_log"
        )
    current_requirements.extend(
        [
            _variant_column("signal", variant)
            if frequency == "weekly"
            else _variant_column("signal", variant, construction)
            for variant in specification.signal_variants
        ]
    )
    current_requirements.extend(
        [
            _variant_column("article_count", specification.coverage_variant)
            if frequency == "weekly"
            else _variant_column(
                "article_count", specification.coverage_variant, construction
            ),
            _variant_column("effective_count", specification.coverage_variant)
            if frequency == "weekly"
            else _variant_column(
                "effective_count", specification.coverage_variant, construction
            ),
        ]
        if specification.include_coverage
        else []
    )
    work = panel.copy()
    complete_column = "_forecast_complete"
    work[complete_column] = work[current_requirements].notna().all(axis=1)
    target_dates = work[date_column].shift(-horizon)
    eligible = work.index[
        work[complete_column]
        & work[target_column].notna()
        & work[target_level_column].notna()
        & target_dates.notna()
    ].tolist()
    positions: list[int] = []
    for position in eligible:
        train = purged_training_positions(
            work,
            int(position),
            horizon,
            target_column,
            complete_column,
            specification.training_window,
        )
        if len(train) >= minimum_training_rows:
            positions.append(int(position))
    if maximum_origins is not None:
        positions = positions[:maximum_origins]
    if not positions:
        raise ValueError(
            f"No forecast origins for {frequency}/{construction}/h{horizon}/"
            f"{specification.specification_id}"
        )

    prediction_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for start in range(0, len(positions), specification.refit_every):
        block = positions[start : start + specification.refit_every]
        fit_position = int(block[0])
        train = purged_training_positions(
            work,
            fit_position,
            horizon,
            target_column,
            complete_column,
            specification.training_window,
        )
        if len(train) < minimum_training_rows:
            raise AssertionError("Forecast refit lost its minimum training sample")
        estimator = make_ridge_estimator(ridge_alpha)
        started = time.perf_counter()
        estimator.fit(work.loc[train, features], work.loc[train, target_column])
        runtime = time.perf_counter() - started
        predictions = estimator.predict(work.loc[block, features])
        feature_end = work.at[int(train[-1]), date_column]
        label_end_position = int(train[-1]) + horizon
        label_end = work.at[label_end_position, date_column]
        fit_origin = work.at[fit_position, date_column]
        if label_end > fit_origin:
            raise AssertionError("Training label endpoint exceeds forecast origin")
        fit_rows.append(
            {
                "frequency": frequency,
                "index_construction": construction,
                "horizon": horizon,
                "horizon_unit": horizon_unit,
                "specification_id": specification.specification_id,
                "feature_count": len(features),
                "features": "|".join(features),
                "training_rows": len(train),
                "training_window": specification.training_window,
                "refit_every": specification.refit_every,
                "training_first_position": int(train[0]),
                "training_last_position": int(train[-1]),
                "training_feature_end_date": feature_end,
                "training_label_end_date": label_end,
                "model_fit_origin_date": fit_origin,
                "horizon_purge_passed": bool(label_end <= fit_origin),
                "fit_runtime_seconds": runtime,
            }
        )
        for position, prediction in zip(block, predictions):
            actual = float(work.at[position, target_column])
            current = float(work.at[position, current_column])
            actual_level = float(work.at[position, target_level_column])
            predicted = float(prediction)
            predicted_level = float(current * np.exp(predicted))
            prediction_rows.append(
                {
                    "frequency": frequency,
                    "index_construction": construction,
                    "horizon": horizon,
                    "horizon_unit": horizon_unit,
                    "origin_position": int(position),
                    "origin_date": work.at[position, date_column],
                    "target_date": target_dates.at[position],
                    "specification_id": specification.specification_id,
                    "family_id": specification.family_id,
                    "signal_variants": "|".join(specification.signal_variants),
                    "actual_log_change": actual,
                    "predicted_log_change": predicted,
                    "forecast_error": actual - predicted,
                    "absolute_error": abs(actual - predicted),
                    "squared_error": (actual - predicted) ** 2,
                    "current_scfi": current,
                    "actual_scfi_level": actual_level,
                    "predicted_scfi_level": predicted_level,
                    "level_error": actual_level - predicted_level,
                    "training_rows": len(train),
                    "training_feature_end_date": feature_end,
                    "training_label_end_date": label_end,
                    "model_fit_origin_date": fit_origin,
                }
            )
        logger.info(
            "forecast frequency=%s construction=%s h=%s spec=%s train=%s "
            "origins=%s fit_origin=%s runtime=%.3fs",
            frequency,
            construction,
            horizon,
            specification.specification_id,
            len(train),
            len(block),
            fit_origin,
            runtime,
        )
    predictions = pd.DataFrame(prediction_rows)
    fits = pd.DataFrame(fit_rows)
    if predictions.duplicated(
        ["frequency", "index_construction", "horizon", "specification_id", "origin_date"]
    ).any():
        raise AssertionError("Duplicate forecast origins within a specification")
    return predictions, fits


def _checkpoint_cell_path(
    checkpoint_dir: Path,
    frequency: str,
    construction: str,
    horizon: int,
    specification_id: str,
    suffix: str,
) -> Path:
    token = hashlib.sha256(specification_id.encode("utf-8")).hexdigest()[:12]
    return checkpoint_dir / (
        f"{frequency}_{construction}_h{horizon}_{token}_{suffix}.csv"
    )


def run_all_forecasts(
    panels: Mapping[str, pd.DataFrame],
    specifications: Mapping[str, Sequence[ForecastSpecification]],
    selected_frequencies: Sequence[str],
    config: Mapping[str, Any],
    checkpoint_dir: Path,
    resume: bool,
    maximum_origins: int | None,
    variant_map: Mapping[str, IndexVariant],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    fit_frames: list[pd.DataFrame] = []
    skip_rows: list[dict[str, Any]] = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ridge_alpha = float(config["forecast"]["ridge_alpha"])
    for frequency in selected_frequencies:
        frequency_config = config["forecast"][frequency]
        horizons = [int(item) for item in frequency_config["horizons"]]
        constructions = (
            ["weekly_log_volume"]
            if frequency == "weekly"
            else list(MONTHLY_CONSTRUCTIONS)
        )
        for construction in constructions:
            for horizon in horizons:
                for specification in specifications[frequency]:
                    structural_reason = forecast_structural_skip_reason(
                        specification, construction, variant_map
                    )
                    if structural_reason is not None:
                        skip_rows.append(
                            {
                                "analysis_layer": "forecast",
                                "frequency": frequency,
                                "index_construction": construction,
                                "horizon": horizon,
                                "specification_id": specification.specification_id,
                                "status": "skipped_structurally_equivalent",
                                "reason": structural_reason,
                            }
                        )
                        logger.info(
                            "forecast_skipped frequency=%s construction=%s h=%s "
                            "spec=%s reason=%s",
                            frequency,
                            construction,
                            horizon,
                            specification.specification_id,
                            structural_reason,
                        )
                        continue
                    predictions_path = _checkpoint_cell_path(
                        checkpoint_dir,
                        frequency,
                        construction,
                        horizon,
                        specification.specification_id,
                        "predictions",
                    )
                    fits_path = _checkpoint_cell_path(
                        checkpoint_dir,
                        frequency,
                        construction,
                        horizon,
                        specification.specification_id,
                        "fits",
                    )
                    if resume and predictions_path.is_file() and fits_path.is_file():
                        predictions = pd.read_csv(
                            predictions_path,
                            parse_dates=[
                                "origin_date",
                                "target_date",
                                "training_feature_end_date",
                                "training_label_end_date",
                                "model_fit_origin_date",
                            ],
                        )
                        fits = pd.read_csv(
                            fits_path,
                            parse_dates=[
                                "training_feature_end_date",
                                "training_label_end_date",
                                "model_fit_origin_date",
                            ],
                        )
                        logger.info(
                            "checkpoint_loaded frequency=%s construction=%s h=%s spec=%s rows=%s",
                            frequency,
                            construction,
                            horizon,
                            specification.specification_id,
                            len(predictions),
                        )
                    else:
                        predictions, fits = run_forecast_cell(
                            panel=panels[frequency],
                            specification=specification,
                            frequency=frequency,
                            construction=construction,
                            horizon=horizon,
                            minimum_training_rows=int(
                                frequency_config["minimum_training_rows"]
                            ),
                            ridge_alpha=ridge_alpha,
                            maximum_origins=maximum_origins,
                            logger=logger,
                        )
                        atomic_write_csv(predictions, predictions_path)
                        atomic_write_csv(fits, fits_path)
                    prediction_frames.append(predictions)
                    fit_frames.append(fits)
    if not prediction_frames or not fit_frames:
        raise ValueError("No forecast cells were estimated")
    forecasts = pd.concat(prediction_frames, ignore_index=True)
    fits = pd.concat(fit_frames, ignore_index=True)
    return forecasts, fits, pd.DataFrame(skip_rows)


def calculate_forecast_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["frequency", "index_construction", "horizon", "specification_id"]
    for key, group in forecasts.groupby(keys, sort=True):
        group = group.sort_values("origin_date")
        error = group["forecast_error"].to_numpy(dtype=float)
        actual = group["actual_log_change"].to_numpy(dtype=float)
        predicted = group["predicted_log_change"].to_numpy(dtype=float)
        level_error = group["level_error"].to_numpy(dtype=float)
        rows.append(
            {
                "frequency": key[0],
                "index_construction": key[1],
                "horizon": int(key[2]),
                "specification_id": key[3],
                "family_id": group["family_id"].iloc[0],
                "n_forecasts": len(group),
                "first_origin_date": group["origin_date"].min(),
                "last_origin_date": group["origin_date"].max(),
                "mae_log_change": float(np.mean(np.abs(error))),
                "rmse_log_change": float(np.sqrt(np.mean(error**2))),
                "bias_predicted_minus_actual": float(np.mean(predicted - actual)),
                "directional_accuracy": float(
                    np.mean(np.sign(actual) == np.sign(predicted))
                ),
                "forecast_actual_correlation": (
                    float(np.corrcoef(actual, predicted)[0, 1])
                    if len(group) > 1
                    and np.std(actual) > 0
                    and np.std(predicted) > 0
                    else math.nan
                ),
                "mae_scfi_level": float(np.mean(np.abs(level_error))),
                "rmse_scfi_level": float(np.sqrt(np.mean(level_error**2))),
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def match_prediction_pair(
    predictions: pd.DataFrame,
    baseline_spec_id: str,
    variant_spec_id: str,
    frequency: str,
    construction: str,
    horizon: int,
) -> pd.DataFrame:
    subset = predictions.loc[
        predictions["frequency"].eq(frequency)
        & predictions["index_construction"].eq(construction)
        & predictions["horizon"].eq(horizon)
    ].copy()
    baseline = subset.loc[
        subset["specification_id"].eq(baseline_spec_id)
    ].copy()
    variant = subset.loc[subset["specification_id"].eq(variant_spec_id)].copy()
    if baseline.empty or variant.empty:
        return pd.DataFrame()
    key = ["origin_date", "target_date"]
    if baseline.duplicated(key).any() or variant.duplicated(key).any():
        raise ValueError("Duplicate origin/target dates prevent a paired comparison")
    keep = [
        *key,
        "actual_log_change",
        "predicted_log_change",
        "forecast_error",
        "absolute_error",
        "squared_error",
        "actual_scfi_level",
    ]
    matched = baseline[keep].merge(
        variant[keep],
        on=key,
        how="inner",
        suffixes=("_baseline", "_variant"),
        validate="one_to_one",
    )
    if matched.empty:
        return matched
    actual_difference = (
        matched["actual_log_change_baseline"]
        - matched["actual_log_change_variant"]
    ).abs()
    level_difference = (
        matched["actual_scfi_level_baseline"]
        - matched["actual_scfi_level_variant"]
    ).abs()
    if actual_difference.max() > 1e-12 or level_difference.max() > 1e-8:
        raise ValueError(
            "Matched forecast specifications disagree on realized outcomes; "
            f"max_log_difference={actual_difference.max()} "
            f"max_level_difference={level_difference.max()}"
        )
    matched["squared_loss_difference_baseline_minus_variant"] = (
        matched["squared_error_baseline"] - matched["squared_error_variant"]
    )
    matched["absolute_loss_difference_baseline_minus_variant"] = (
        matched["absolute_error_baseline"] - matched["absolute_error_variant"]
    )
    matched["forecast_difference_baseline_minus_variant"] = (
        matched["predicted_log_change_baseline"]
        - matched["predicted_log_change_variant"]
    )
    matched["baseline_specification_id"] = baseline_spec_id
    matched["variant_specification_id"] = variant_spec_id
    matched["baseline_origins_total"] = len(baseline)
    matched["variant_origins_total"] = len(variant)
    matched["matched_origins"] = len(matched)
    matched["baseline_origins_dropped"] = len(baseline) - len(matched)
    matched["variant_origins_dropped"] = len(variant) - len(matched)
    return matched.sort_values("origin_date").reset_index(drop=True)


def hac_mean_inference(
    values: Iterable[float], maximum_lag: int, alternative: str = "two-sided"
) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 3:
        return {
            "mean": math.nan,
            "standard_error_hac": math.nan,
            "statistic": math.nan,
            "p_value": math.nan,
        }
    fitted = sm.OLS(array, np.ones((len(array), 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": min(maximum_lag, len(array) - 1), "use_correction": True},
    )
    mean = float(fitted.params[0])
    standard_error = float(fitted.bse[0])
    statistic = mean / standard_error if standard_error > 0 else math.nan
    if alternative == "two-sided":
        p_value = float(2 * norm.sf(abs(statistic)))
    elif alternative == "greater":
        p_value = float(norm.sf(statistic))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    return {
        "mean": mean,
        "standard_error_hac": standard_error,
        "statistic": statistic,
        "p_value": p_value,
    }


def moving_block_mean_inference(
    values: Iterable[float],
    repetitions: int,
    block_length: int,
    alpha: float,
    seed: int,
) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if repetitions <= 0 or len(array) < 3:
        return {
            "bootstrap_repetitions_requested": repetitions,
            "bootstrap_repetitions_successful": 0,
            "bootstrap_block_length": block_length,
            "bootstrap_ci_lower": math.nan,
            "bootstrap_ci_upper": math.nan,
            "bootstrap_p_value_two_sided": math.nan,
        }
    block_length = min(max(1, int(block_length)), len(array))
    rng = np.random.default_rng(seed)
    blocks_needed = int(np.ceil(len(array) / block_length))
    maximum_start = len(array) - block_length
    draws = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        starts = rng.integers(0, maximum_start + 1, size=blocks_needed)
        indices = np.concatenate(
            [np.arange(start, start + block_length) for start in starts]
        )[: len(array)]
        draws[repetition] = float(np.mean(array[indices]))
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    centered = draws - np.mean(draws)
    observed = abs(float(np.mean(array)))
    p_value = float((1 + np.sum(np.abs(centered) >= observed)) / (repetitions + 1))
    return {
        "bootstrap_repetitions_requested": repetitions,
        "bootstrap_repetitions_successful": repetitions,
        "bootstrap_block_length": block_length,
        "bootstrap_ci_lower": float(lower),
        "bootstrap_ci_upper": float(upper),
        "bootstrap_p_value_two_sided": p_value,
    }


def adjust_pvalues(values: Iterable[float], method: str) -> np.ndarray:
    raw = np.asarray(list(values), dtype=float)
    adjusted = np.full(len(raw), np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(raw))
    if not len(finite_positions):
        return adjusted
    finite = np.clip(raw[finite_positions], 0, 1)
    order = np.argsort(finite)
    ordered = finite[order]
    count = len(ordered)
    if method == "holm":
        ordered_adjusted = np.maximum.accumulate(
            (count - np.arange(count)) * ordered
        )
    elif method in {"fdr_bh", "benjamini_hochberg"}:
        ordered_adjusted = np.minimum.accumulate(
            (ordered * count / (np.arange(count) + 1))[::-1]
        )[::-1]
    else:
        raise ValueError(f"Unknown p-value adjustment method: {method}")
    restored = np.empty(count, dtype=float)
    restored[order] = np.clip(ordered_adjusted, 0, 1)
    adjusted[finite_positions] = restored
    return adjusted


def _operational_disruption_regime(dates: pd.Series) -> pd.Series:
    values = pd.to_datetime(dates)
    return pd.Series(
        np.select(
            [
                values.lt(OPERATIONAL_DISRUPTION_START),
                values.le(OPERATIONAL_DISRUPTION_END),
            ],
            ["pre_operational_disruption", "operational_disruption"],
            default="post_operational_disruption",
        ),
        index=dates.index,
    )


def paired_ablation_tests(
    forecasts: pd.DataFrame,
    comparisons: Mapping[str, Sequence[PairedComparison]],
    selected_frequencies: Sequence[str],
    config: Mapping[str, Any],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha = float(config["inference"]["alpha"])
    rows: list[dict[str, Any]] = []
    differences: list[pd.DataFrame] = []
    for frequency in selected_frequencies:
        constructions = (
            ["weekly_log_volume"]
            if frequency == "weekly"
            else list(MONTHLY_CONSTRUCTIONS)
        )
        horizons = [int(item) for item in config["forecast"][frequency]["horizons"]]
        for construction in constructions:
            for comparison in comparisons[frequency]:
                for horizon in horizons:
                    matched = match_prediction_pair(
                        forecasts,
                        comparison.baseline_specification_id,
                        comparison.variant_specification_id,
                        frequency,
                        construction,
                        horizon,
                    )
                    if matched.empty:
                        continue
                    matched["comparison_id"] = comparison.comparison_id
                    matched["family_id"] = comparison.family_id
                    matched["frequency"] = frequency
                    matched["index_construction"] = construction
                    matched["horizon"] = horizon
                    matched["target_regime"] = _operational_disruption_regime(
                        matched["target_date"]
                    )
                    differences.append(matched)
                    subsets: list[tuple[str, pd.DataFrame]] = [("full_sample", matched)]
                    if comparison.comparison_id in {
                        "peak_vs_no_msdi",
                        "breadth_vs_no_msdi",
                        "incremental_breadth_given_peak",
                        "incremental_peak_given_breadth",
                    }:
                        # Never splice the separated pre- and post-COVID blocks
                        # into one artificial time series for HAC/bootstrap.
                        subsets.extend(
                            [
                                (
                                    "pre_operational_disruption",
                                    matched.loc[
                                        matched["target_regime"].eq(
                                            "pre_operational_disruption"
                                        )
                                    ],
                                ),
                                (
                                    "operational_disruption",
                                    matched.loc[
                                        matched["target_regime"].eq(
                                            "operational_disruption"
                                        )
                                    ],
                                ),
                                (
                                    "post_operational_disruption",
                                    matched.loc[
                                        matched["target_regime"].eq(
                                            "post_operational_disruption"
                                        )
                                    ],
                                ),
                            ]
                        )
                    for sample_subset, sample in subsets:
                        if len(sample) < 3:
                            continue
                        hac_lag = max(0, horizon - 1)
                        block_length = max(
                            horizon, int(np.ceil(len(sample) ** (1 / 3)))
                        )
                        for loss_name, difference_column in (
                            (
                                "squared_error",
                                "squared_loss_difference_baseline_minus_variant",
                            ),
                            (
                                "absolute_error",
                                "absolute_loss_difference_baseline_minus_variant",
                            ),
                        ):
                            inference = hac_mean_inference(
                                sample[difference_column], hac_lag
                            )
                            bootstrap = moving_block_mean_inference(
                                sample[difference_column],
                                repetitions=repetitions,
                                block_length=block_length,
                                alpha=alpha,
                                seed=stable_seed(
                                    seed,
                                    frequency,
                                    construction,
                                    comparison.comparison_id,
                                    horizon,
                                    sample_subset,
                                    loss_name,
                                ),
                            )
                            baseline_rmse = float(
                                np.sqrt(np.mean(sample["squared_error_baseline"]))
                            )
                            variant_rmse = float(
                                np.sqrt(np.mean(sample["squared_error_variant"]))
                            )
                            baseline_mae = float(sample["absolute_error_baseline"].mean())
                            variant_mae = float(sample["absolute_error_variant"].mean())
                            row: dict[str, Any] = {
                                "frequency": frequency,
                                "frequency_role": (
                                    "primary" if frequency == "weekly" else "robustness"
                                ),
                                "index_construction": construction,
                                "comparison_id": comparison.comparison_id,
                                "family_id": comparison.family_id,
                                "baseline_specification_id": comparison.baseline_specification_id,
                                "variant_specification_id": comparison.variant_specification_id,
                                "nested_models": comparison.nested,
                                "primary_family": bool(
                                    comparison.primary
                                    and construction == "weekly_log_volume"
                                    and sample_subset == "full_sample"
                                    and loss_name == "squared_error"
                                ),
                                "sample_subset": sample_subset,
                                "loss": loss_name,
                                "loss_difference_convention": "baseline_loss_minus_variant_loss_positive_favors_variant",
                                "direction_note": comparison.direction_note,
                                "horizon": horizon,
                                "paired_n": len(sample),
                                "first_origin_date": sample["origin_date"].min(),
                                "last_origin_date": sample["origin_date"].max(),
                                "baseline_origins_total": int(
                                    sample["baseline_origins_total"].iloc[0]
                                ),
                                "variant_origins_total": int(
                                    sample["variant_origins_total"].iloc[0]
                                ),
                                "baseline_origins_dropped": int(
                                    sample["baseline_origins_dropped"].iloc[0]
                                ),
                                "variant_origins_dropped": int(
                                    sample["variant_origins_dropped"].iloc[0]
                                ),
                                "mean_loss_difference": inference["mean"],
                                "standard_error_hac": inference["standard_error_hac"],
                                "dm_hac_statistic": inference["statistic"],
                                "dm_hac_p_value": inference["p_value"],
                                "hac_max_lags": hac_lag,
                                "baseline_rmse": baseline_rmse,
                                "variant_rmse": variant_rmse,
                                "rmse_improvement_percent": (
                                    100 * (baseline_rmse - variant_rmse) / baseline_rmse
                                    if baseline_rmse > 0
                                    else math.nan
                                ),
                                "baseline_mae": baseline_mae,
                                "variant_mae": variant_mae,
                                "mae_improvement_percent": (
                                    100 * (baseline_mae - variant_mae) / baseline_mae
                                    if baseline_mae > 0
                                    else math.nan
                                ),
                                "incremental_oos_r_squared": (
                                    1
                                    - float(sample["squared_error_variant"].sum())
                                    / float(sample["squared_error_baseline"].sum())
                                    if float(sample["squared_error_baseline"].sum()) > 0
                                    else math.nan
                                ),
                                **bootstrap,
                            }
                            if loss_name == "squared_error" and comparison.nested:
                                cw = (
                                    sample["squared_error_baseline"]
                                    - (
                                        sample["squared_error_variant"]
                                        - sample[
                                            "forecast_difference_baseline_minus_variant"
                                        ]
                                        ** 2
                                    )
                                )
                                cw_inference = hac_mean_inference(
                                    cw, hac_lag, alternative="greater"
                                )
                                row.update(
                                    {
                                        "clark_west_mean_adjusted_difference": cw_inference[
                                            "mean"
                                        ],
                                        "clark_west_statistic": cw_inference["statistic"],
                                        "clark_west_one_sided_p_value": cw_inference[
                                            "p_value"
                                        ],
                                    }
                                )
                            else:
                                row.update(
                                    {
                                        "clark_west_mean_adjusted_difference": math.nan,
                                        "clark_west_statistic": math.nan,
                                        "clark_west_one_sided_p_value": math.nan,
                                    }
                                )
                            rows.append(row)
    tests = pd.DataFrame(rows)
    if tests.empty:
        return tests, pd.DataFrame()
    tests["dm_hac_p_value_adjusted"] = np.nan
    primary_mask = tests["primary_family"].astype(bool)
    tests.loc[primary_mask, "dm_hac_p_value_adjusted"] = adjust_pvalues(
        tests.loc[primary_mask, "dm_hac_p_value"], "holm"
    )
    exploratory = tests.loc[~primary_mask]
    for _, indices in exploratory.groupby(
        ["family_id", "frequency", "index_construction", "sample_subset", "loss"],
        dropna=False,
    ).groups.items():
        tests.loc[indices, "dm_hac_p_value_adjusted"] = adjust_pvalues(
            tests.loc[indices, "dm_hac_p_value"], "fdr_bh"
        )
    tests["multiplicity_method"] = np.where(primary_mask, "holm", "fdr_bh")
    tests["reject_adjusted_at_alpha"] = tests["dm_hac_p_value_adjusted"].lt(alpha)
    if differences:
        paired = pd.concat(differences, ignore_index=True)
    else:
        paired = pd.DataFrame()
    return tests.sort_values(
        ["frequency", "index_construction", "family_id", "comparison_id", "horizon", "sample_subset", "loss"]
    ).reset_index(drop=True), paired


def forecast_regime_sensitivity(forecasts: pd.DataFrame) -> pd.DataFrame:
    work = forecasts.copy()
    work["target_regime"] = _operational_disruption_regime(work["target_date"])
    rows: list[dict[str, Any]] = []
    keys = [
        "frequency",
        "index_construction",
        "horizon",
        "specification_id",
        "target_regime",
    ]
    for key, group in work.groupby(keys, sort=True):
        rows.append(
            {
                "frequency": key[0],
                "index_construction": key[1],
                "horizon": int(key[2]),
                "specification_id": key[3],
                "target_regime": key[4],
                "n_forecasts": len(group),
                "mae_log_change": float(group["absolute_error"].mean()),
                "rmse_log_change": float(np.sqrt(group["squared_error"].mean())),
                "bias_predicted_minus_actual": float(
                    -group["forecast_error"].mean()
                ),
                "directional_accuracy": float(
                    np.mean(
                        np.sign(group["actual_log_change"])
                        == np.sign(group["predicted_log_change"])
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def index_stability_summary(
    weekly_long: pd.DataFrame,
    monthly_long: pd.DataFrame,
    variants: Sequence[IndexVariant],
) -> pd.DataFrame:
    references = {
        "peak": "reference_peak_median",
        "breadth": "reference_breadth_median7",
    }
    rows: list[dict[str, Any]] = []
    variant_map = {item.variant_id: item for item in variants}
    weekly_pivot = weekly_long.pivot(
        index="week_ending", columns="variant_id", values="signal_value"
    )
    for variant_id, variant in variant_map.items():
        reference = references[variant.path]
        if variant_id == reference:
            continue
        pair = weekly_pivot[[reference, variant_id]].dropna()
        difference = pair[variant_id] - pair[reference]
        rows.append(
            {
                "frequency": "weekly",
                "index_construction": "weekly_log_volume",
                "variant_id": variant_id,
                "reference_variant_id": reference,
                "family_id": variant.family_id,
                "n_matched": len(pair),
                "pearson_correlation": pair[reference].corr(pair[variant_id]),
                "spearman_correlation": (
                    spearmanr(pair[reference], pair[variant_id]).statistic
                    if len(pair) > 1
                    else math.nan
                ),
                "mean_variant_minus_reference": float(difference.mean()),
                "mean_absolute_difference": float(difference.abs().mean()),
                "root_mean_squared_difference": float(
                    np.sqrt(np.mean(difference**2))
                ),
                "standard_deviation_ratio": (
                    pair[variant_id].std(ddof=1) / pair[reference].std(ddof=1)
                    if pair[reference].std(ddof=1) > 0
                    else math.nan
                ),
            }
        )
    for construction, value_column in (
        ("mean_safe_weekly", "signal_mean_safe_weekly"),
        ("adapted_pdf_sqrt_ema3", "signal_adapted_pdf_sqrt_ema3"),
    ):
        pivot = monthly_long.pivot(
            index="month_ending", columns="variant_id", values=value_column
        )
        for variant_id, variant in variant_map.items():
            reference = references[variant.path]
            if variant_id == reference:
                continue
            pair = pivot[[reference, variant_id]].dropna()
            if pair.empty:
                continue
            difference = pair[variant_id] - pair[reference]
            rows.append(
                {
                    "frequency": "monthly",
                    "index_construction": construction,
                    "variant_id": variant_id,
                    "reference_variant_id": reference,
                    "family_id": variant.family_id,
                    "n_matched": len(pair),
                    "pearson_correlation": pair[reference].corr(pair[variant_id]),
                    "spearman_correlation": (
                        spearmanr(pair[reference], pair[variant_id]).statistic
                        if len(pair) > 1
                        else math.nan
                    ),
                    "mean_variant_minus_reference": float(difference.mean()),
                    "mean_absolute_difference": float(difference.abs().mean()),
                    "root_mean_squared_difference": float(
                        np.sqrt(np.mean(difference**2))
                    ),
                    "standard_deviation_ratio": (
                        pair[variant_id].std(ddof=1) / pair[reference].std(ddof=1)
                        if pair[reference].std(ddof=1) > 0
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def prepare_dynamic_frames(
    support08: Any,
    weekly_panel_path: Path,
    monthly_panel_path: Path,
    selected_frequencies: Sequence[str],
    panel_overrides: Mapping[str, pd.DataFrame] | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    specifications = {
        "weekly": support08.FrequencySpecification(
            frequency="weekly",
            date_column="observation_date",
            scfi_column="scfi",
            brent_column="brent_price",
            article_count_column="article_count",
            msdi_columns={
                "peak": "peak_score_media_corrected",
                "breadth": "breadth_score_media_corrected",
            },
            maximum_lag=12,
            lp_horizon=12,
            irf_horizon=12,
        ),
        "monthly": support08.FrequencySpecification(
            frequency="monthly",
            date_column="month_ending",
            scfi_column="scfi_monthly_mean",
            brent_column="brent_monthly_mean",
            article_count_column="article_count_sum",
            msdi_columns={
                "peak": "peak_score_media_corrected_mean",
                "breadth": "breadth_score_media_corrected_mean",
            },
            maximum_lag=6,
            lp_horizon=6,
            irf_horizon=6,
        ),
    }
    paths = {"weekly": weekly_panel_path, "monthly": monthly_panel_path}
    result: dict[tuple[str, str], pd.DataFrame] = {}
    for frequency in selected_frequencies:
        spec = specifications[frequency]
        if panel_overrides is not None and frequency in panel_overrides:
            panel = panel_overrides[frequency].copy()
            required = {
                spec.date_column,
                spec.scfi_column,
                spec.brent_column,
                spec.article_count_column,
                *spec.msdi_columns.values(),
            }
            missing = sorted(required.difference(panel.columns))
            if missing:
                raise ValueError(
                    f"Common-cohort {frequency} dynamic panel lacks {missing}"
                )
        else:
            panel = support08.load_panel(
                require_file(paths[frequency], f"{frequency} dynamic panel"), spec
            )
        for path_name in PATHS:
            prepared = support08.prepare_analysis_frame(panel, spec, path_name)
            frame, _ = support08.select_longest_complete_block(
                prepared,
                [*support08.ENDOGENOUS_VARIABLES, *support08.SEASONAL_CONTROLS],
                frequency,
            )
            result[(frequency, path_name)] = frame
    return result


def varx_lag_sensitivity(
    support08: Any,
    frame: pd.DataFrame,
    frequency: str,
    path_name: str,
    maximum_lags: Sequence[int],
    alpha: float,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit the stationary four-variable VARX over each lag-search ceiling."""
    selections: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    for maximum_lag in maximum_lags:
        maximum_lag = int(maximum_lag)
        started = time.perf_counter()
        fitted, lag_table = support08.fit_varx(frame, maximum_lag, alpha)
        lag_table["frequency"] = frequency
        lag_table["msdi_path"] = path_name
        lag_table["maximum_lag_specification"] = maximum_lag
        lag_table["analysis"] = "stationary_varx_lag_ceiling_sensitivity"
        selections.append(lag_table)
        diagnostic = support08.var_diagnostics(
            fitted, frequency, path_name, alpha, maximum_lag=maximum_lag
        )
        diagnostic["maximum_lag_specification"] = maximum_lag
        diagnostic["selected_lag"] = int(fitted.k_ar)
        diagnostic["analysis"] = "stationary_varx_lag_ceiling_sensitivity"
        diagnostics.append(diagnostic)
        coefficient = support08.extract_var_coefficients(
            fitted, frequency, path_name
        )
        coefficient["maximum_lag_specification"] = maximum_lag
        coefficient["analysis"] = "stationary_varx_lag_ceiling_sensitivity"
        coefficients.append(coefficient)
        logger.info(
            "varx_sensitivity frequency=%s path=%s maximum_lag=%s "
            "selected_lag=%s whiteness=%s runtime=%.3fs",
            frequency,
            path_name,
            maximum_lag,
            int(fitted.k_ar),
            bool(
                lag_table.loc[
                    lag_table["selected_for_analysis"].fillna(False),
                    "passes_residual_whiteness",
                ].iloc[0]
            ),
            time.perf_counter() - started,
        )
    return (
        pd.concat(selections, ignore_index=True),
        pd.concat(diagnostics, ignore_index=True),
        pd.concat(coefficients, ignore_index=True),
    )


def conventional_granger_sensitivity(
    support08: Any,
    frame: pd.DataFrame,
    frequency: str,
    path_name: str,
    maximum_lags: Sequence[int],
) -> pd.DataFrame:
    """Run bivariate conventional Granger tests once through each unique lag."""
    maximum_lag = max(int(value) for value in maximum_lags)
    result = support08.conventional_granger_tests(
        frame, frequency, path_name, maximum_lag
    )
    result["maximum_lag_specification"] = maximum_lag
    result["analysis"] = "conventional_bivariate_granger_lag_sensitivity"
    return result


def conditional_granger_sensitivity(
    frame: pd.DataFrame,
    frequency: str,
    path_name: str,
    lags: Sequence[int],
    alpha: float,
) -> pd.DataFrame:
    variables = (
        "scfi_return",
        "msdi_change",
        "brent_return",
        "news_volume_change",
    )
    directions = (
        ("msdi_change", "scfi_return"),
        ("scfi_return", "msdi_change"),
    )
    rows: list[dict[str, Any]] = []
    for lag_count in lags:
        lag_count = int(lag_count)
        design = pd.DataFrame(index=frame.index)
        for variable in variables:
            for lag in range(1, lag_count + 1):
                design[f"{variable}_lag_{lag}"] = frame[variable].shift(lag)
        design["season_sin"] = frame["season_sin"]
        design["season_cos"] = frame["season_cos"]
        for causing, caused in directions:
            model_frame = pd.concat(
                [frame[["date", caused]], design], axis=1
            ).dropna()
            dates = model_frame.pop("date")
            outcome = model_frame.pop(caused)
            x = sm.add_constant(model_frame, has_constant="add")
            hac_lags = max(1, lag_count)
            fitted = sm.OLS(outcome, x).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": hac_lags, "use_correction": True},
            )
            restriction_names = [f"{causing}_lag_{lag}" for lag in range(1, lag_count + 1)]
            restriction = np.zeros((len(restriction_names), len(x.columns)))
            for row_index, name in enumerate(restriction_names):
                restriction[row_index, x.columns.get_loc(name)] = 1.0
            test = fitted.wald_test(restriction, scalar=True)
            statistic = float(np.asarray(test.statistic).squeeze())
            p_value = float(np.asarray(test.pvalue).squeeze())
            rows.append(
                {
                    "analysis": "conditional_granger_hac",
                    "frequency": frequency,
                    "msdi_path": path_name,
                    "lag_count": lag_count,
                    "causing": causing,
                    "caused": caused,
                    "wald_statistic": statistic,
                    "degrees_freedom": len(restriction_names),
                    "p_value": p_value,
                    "reject_at_alpha": p_value < alpha,
                    "alpha": alpha,
                    "hac_max_lags": hac_lags,
                    "n_observations": int(fitted.nobs),
                    "sample_start_date": dates.min(),
                    "sample_end_date": dates.max(),
                    "dynamic_effective_sample_matched_across_lags": False,
                    "interpretation": "predictive_precedence_not_structural_causality",
                }
            )
    output = pd.DataFrame(rows)
    if not output.empty:
        output["p_value_fdr_bh"] = np.nan
        for _, indices in output.groupby(
            ["frequency", "msdi_path", "causing", "caused"]
        ).groups.items():
            output.loc[indices, "p_value_fdr_bh"] = adjust_pvalues(
                output.loc[indices, "p_value"], "fdr_bh"
            )
    return output


def ardl_ecm_sensitivity(
    frame: pd.DataFrame,
    frequency: str,
    path_name: str,
    lags: Sequence[int],
    deterministic_terms: Sequence[str],
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lag_count in lags:
        lag_count = int(lag_count)
        for deterministic in deterministic_terms:
            design = pd.DataFrame(index=frame.index)
            design["scfi_log_level_lag_1"] = frame["scfi_log"].shift(1)
            design["msdi_level_lag_1"] = frame["msdi_level"].shift(1)
            design["brent_log_level_lag_1"] = frame["brent_log"].shift(1)
            design["log_news_volume_lag_1"] = frame["log_news_volume"].shift(1)
            for variable in (
                "scfi_return",
                "msdi_change",
                "brent_return",
                "news_volume_change",
            ):
                for lag in range(1, lag_count + 1):
                    design[f"{variable}_lag_{lag}"] = frame[variable].shift(lag)
            design["season_sin"] = frame["season_sin"]
            design["season_cos"] = frame["season_cos"]
            if deterministic == "constant_trend":
                design["linear_trend"] = np.arange(len(frame), dtype=float) / max(
                    len(frame) - 1, 1
                )
            elif deterministic != "constant":
                raise ValueError(f"Unknown deterministic term: {deterministic}")
            model_frame = pd.concat(
                [frame[["date", "scfi_return"]], design], axis=1
            ).dropna()
            dates = model_frame.pop("date")
            outcome = model_frame.pop("scfi_return")
            x = sm.add_constant(model_frame, has_constant="add")
            fitted = sm.OLS(outcome, x).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": max(1, lag_count), "use_correction": True},
            )
            adjustment = float(fitted.params["scfi_log_level_lag_1"])
            msdi_level = float(fitted.params["msdi_level_lag_1"])
            long_run = -msdi_level / adjustment if adjustment != 0 else math.nan
            covariance = fitted.cov_params().loc[
                ["msdi_level_lag_1", "scfi_log_level_lag_1"],
                ["msdi_level_lag_1", "scfi_log_level_lag_1"],
            ].to_numpy(dtype=float)
            if adjustment != 0:
                gradient = np.array(
                    [-1 / adjustment, msdi_level / (adjustment**2)], dtype=float
                )
                variance = float(gradient @ covariance @ gradient)
                long_run_se = math.sqrt(max(variance, 0.0))
            else:
                long_run_se = math.nan
            critical = float(norm.ppf(1 - alpha / 2))
            rows.append(
                {
                    "analysis": "conditional_ardl_ecm",
                    "frequency": frequency,
                    "msdi_path": path_name,
                    "lag_count": lag_count,
                    "deterministic_terms": deterministic,
                    "scfi_error_correction_coefficient": adjustment,
                    "scfi_error_correction_p_value": float(
                        fitted.pvalues["scfi_log_level_lag_1"]
                    ),
                    "msdi_lagged_level_coefficient": msdi_level,
                    "msdi_lagged_level_p_value": float(
                        fitted.pvalues["msdi_level_lag_1"]
                    ),
                    "conditional_long_run_msdi_multiplier": long_run,
                    "conditional_long_run_standard_error_delta": long_run_se,
                    "conditional_long_run_ci_lower": long_run - critical * long_run_se,
                    "conditional_long_run_ci_upper": long_run + critical * long_run_se,
                    "n_observations": int(fitted.nobs),
                    "r_squared": float(fitted.rsquared),
                    "sample_start_date": dates.min(),
                    "sample_end_date": dates.max(),
                    "dynamic_effective_sample_matched_across_lags": False,
                    "interpretation": "conditional_levels_sensitivity_not_cointegration_proof_or_causality",
                }
            )
    return pd.DataFrame(rows)


def local_projection_lag_sensitivity(
    support08: Any,
    frame: pd.DataFrame,
    frequency: str,
    path_name: str,
    lags: Sequence[int],
    maximum_horizon: int,
    alpha: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for lag_count in lags:
        innovations, audit = support08.create_msdi_innovation(
            frame, int(lag_count), frequency, path_name
        )
        estimates = support08.local_projections(
            frame,
            innovations,
            int(lag_count),
            frequency,
            path_name,
            maximum_horizon,
            alpha,
        )
        estimates["sensitivity_lag_count"] = int(lag_count)
        estimates["innovation_first_stage_r_squared"] = audit["r_squared"]
        estimates["innovation_residual_standard_deviation"] = audit[
            "residual_standard_deviation"
        ]
        estimates["innovation_first_stage_n"] = audit["n_observations"]
        estimates["treatment_scale_varies_by_lag_specification"] = True
        estimates["dynamic_effective_sample_matched_across_lags"] = False
        estimates["analysis"] = "local_projection_lag_sensitivity"
        rows.append(estimates)
    return pd.concat(rows, ignore_index=True)


def run_pcmci_sensitivity(
    support08: Any,
    frame: pd.DataFrame,
    frequency: str,
    path_name: str,
    maximum_lags: Sequence[int],
    pc_alphas: Sequence[float],
    alpha: float,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for maximum_lag in maximum_lags:
        for pc_alpha in pc_alphas:
            started = time.perf_counter()
            links = support08.pcmci_links(
                frame,
                frequency,
                path_name,
                int(maximum_lag),
                float(pc_alpha),
                alpha,
            )
            links["sensitivity_maximum_lag"] = int(maximum_lag)
            links["sensitivity_pc_alpha"] = float(pc_alpha)
            links["sensitivity_specification_id"] = (
                f"{frequency}_{path_name}_tau{maximum_lag}_pc{pc_alpha:g}"
            )
            rows.append(links)
            logger.info(
                "pcmci_sensitivity frequency=%s path=%s max_lag=%s pc_alpha=%s "
                "links=%s significant=%s runtime=%.3fs",
                frequency,
                path_name,
                maximum_lag,
                pc_alpha,
                len(links),
                int(links["significant_fdr"].sum()),
                time.perf_counter() - started,
            )
    all_links = pd.concat(rows, ignore_index=True)
    stability_rows: list[dict[str, Any]] = []
    keys = ["frequency", "msdi_path", "source", "target", "source_lag"]
    for key, group in all_links.groupby(keys, sort=True):
        values = group["partial_correlation"].to_numpy(dtype=float)
        nonzero = np.sign(values[np.abs(values) > np.finfo(float).eps])
        stability_rows.append(
            {
                "frequency": key[0],
                "msdi_path": key[1],
                "source": key[2],
                "target": key[3],
                "source_lag": int(key[4]),
                "specifications_containing_link": len(group),
                "significant_fdr_count": int(group["significant_fdr"].sum()),
                "significant_fdr_share": float(group["significant_fdr"].mean()),
                "median_partial_correlation": float(np.median(values)),
                "minimum_partial_correlation": float(np.min(values)),
                "maximum_partial_correlation": float(np.max(values)),
                "sign_stable": bool(len(nonzero) == 0 or np.all(nonzero == nonzero[0])),
                "minimum_adjusted_p_value": float(group["p_value_fdr_bh"].min()),
                "maximum_adjusted_p_value": float(group["p_value_fdr_bh"].max()),
                "interpretation": "link_stability_across_pre_specified_pcmci_settings",
            }
        )
    return all_links, pd.DataFrame(stability_rows)


def shared_peak_breadth_bootstrap(
    support08: Any,
    frames: Mapping[tuple[str, str], pd.DataFrame],
    selected_lags: Mapping[tuple[str, str], int],
    frequency: str,
    maximum_horizon: int,
    repetitions: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    peak = frames[(frequency, "peak")]
    breadth = frames[(frequency, "breadth")]
    common_dates = sorted(set(peak["date"]).intersection(set(breadth["date"])))
    aligned = {
        "peak": peak.set_index("date").loc[common_dates].reset_index(),
        "breadth": breadth.set_index("date").loc[common_dates].reset_index(),
    }
    points: dict[str, pd.DataFrame] = {}
    for path_name in PATHS:
        lag = int(selected_lags[(frequency, path_name)])
        innovations, _ = support08.create_msdi_innovation(
            aligned[path_name], lag, frequency, path_name
        )
        points[path_name] = support08.local_projections(
            aligned[path_name],
            innovations,
            lag,
            frequency,
            path_name,
            maximum_horizon,
            alpha,
        ).sort_values("horizon")
    point_difference = (
        points["peak"]["coefficient_log_points"].to_numpy(dtype=float)
        - points["breadth"]["coefficient_log_points"].to_numpy(dtype=float)
    )
    horizons = points["peak"]["horizon"].to_numpy(dtype=int)
    n = len(common_dates)
    block_length = min(max(maximum_horizon, int(np.ceil(n ** (1 / 3)))), n)
    draws: list[np.ndarray] = []
    if repetitions > 0:
        rng = np.random.default_rng(seed)
        original_dates = aligned["peak"]["date"].reset_index(drop=True)
        for _ in range(repetitions):
            indices = support08.moving_block_indices(n, block_length, rng)
            draw_coefficients: dict[str, np.ndarray] = {}
            try:
                for path_name in PATHS:
                    sampled = aligned[path_name].iloc[indices].copy().reset_index(drop=True)
                    sampled["date"] = original_dates
                    sampled["scfi_log"] = sampled["scfi_return"].cumsum()
                    lag = int(selected_lags[(frequency, path_name)])
                    innovations, _ = support08.create_msdi_innovation(
                        sampled, lag, frequency, path_name
                    )
                    estimates = support08.local_projections(
                        sampled,
                        innovations,
                        lag,
                        frequency,
                        path_name,
                        maximum_horizon,
                        alpha,
                    ).sort_values("horizon")
                    draw_coefficients[path_name] = estimates[
                        "coefficient_log_points"
                    ].to_numpy(dtype=float)
                draws.append(draw_coefficients["peak"] - draw_coefficients["breadth"])
            except Exception:
                continue
    draw_array = np.vstack(draws) if draws else np.empty((0, len(horizons)))
    minimum_successes = int(np.ceil(0.8 * repetitions)) if repetitions else 0
    if repetitions and len(draw_array) < minimum_successes:
        raise RuntimeError(
            f"Shared peak/breadth bootstrap succeeded for only "
            f"{len(draw_array)}/{repetitions} draws"
        )
    rows: list[dict[str, Any]] = []
    for index, horizon in enumerate(horizons):
        if len(draw_array):
            values = draw_array[:, index]
            lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
            standard_error = float(np.std(values, ddof=1))
            centered = values - values.mean()
            p_value = float(
                (1 + np.sum(np.abs(centered) >= abs(point_difference[index])))
                / (len(values) + 1)
            )
        else:
            lower = upper = standard_error = p_value = math.nan
        rows.append(
            {
                "frequency": frequency,
                "horizon": int(horizon),
                "coefficient_peak_log_points": float(
                    points["peak"].iloc[index]["coefficient_log_points"]
                ),
                "coefficient_breadth_log_points": float(
                    points["breadth"].iloc[index]["coefficient_log_points"]
                ),
                "coefficient_difference_peak_minus_breadth": float(
                    point_difference[index]
                ),
                "shared_bootstrap_standard_error": standard_error,
                "shared_bootstrap_ci_lower": float(lower),
                "shared_bootstrap_ci_upper": float(upper),
                "shared_bootstrap_p_value": p_value,
                "bootstrap_repetitions_requested": repetitions,
                "bootstrap_repetitions_successful": len(draw_array),
                "bootstrap_block_length": block_length,
                "peak_selected_lag": int(selected_lags[(frequency, "peak")]),
                "breadth_selected_lag": int(selected_lags[(frequency, "breadth")]),
                "n_common_system_rows": n,
                "formal_correlated_estimate_test": True,
                "interpretation": "shared_block_dynamic_path_difference_not_structural_causality",
            }
        )
    output = pd.DataFrame(rows)
    output["p_value_fdr_bh"] = adjust_pvalues(
        output["shared_bootstrap_p_value"], "fdr_bh"
    )
    return output


def causal_ml_sensitivity_from_manifest(
    manifest_path: Path, alpha: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = require_file(manifest_path, "Causal-ML manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("publication_scope_complete"):
        raise ValueError("Causal-ML manifest is not full-scope complete")
    if manifest.get("causal_claim_allowed") is not False:
        raise ValueError("Causal-ML manifest unexpectedly permits a causal claim")
    output_paths: dict[str, Path] = {}
    for key in (
        "dml_estimates",
        "learner_stability",
        "innovation_comparison",
        "econometric_comparison",
        "overlap_diagnostics",
        "placebo_tests",
    ):
        value = manifest.get("outputs", {}).get(key)
        resolved = _manifest_output_path(value)
        if resolved is None:
            raise ValueError(f"Causal-ML manifest lacks output path for {key}")
        require_file(resolved, f"Causal-ML {key}")
        recorded = manifest.get("output_sha256", {}).get(key)
        actual = sha256_file(resolved)
        if recorded is not None and recorded != actual:
            raise ValueError(
                f"Causal-ML output hash mismatch for {key}: "
                f"recorded={recorded} actual={actual}"
            )
        output_paths[key] = resolved
    estimates = pd.read_csv(output_paths["dml_estimates"], low_memory=False)
    stability = pd.read_csv(output_paths["learner_stability"], low_memory=False)
    innovation = pd.read_csv(output_paths["innovation_comparison"], low_memory=False)
    econometric = pd.read_csv(output_paths["econometric_comparison"], low_memory=False)
    overlap = pd.read_csv(output_paths["overlap_diagnostics"], low_memory=False)
    placebos = pd.read_csv(output_paths["placebo_tests"], low_memory=False)
    keys = ["frequency", "msdi_path", "horizon", "learner"]
    result = estimates.copy()
    result["hac_interval_significant"] = (
        result["ci_lower_log_points"].gt(0)
        | result["ci_upper_log_points"].lt(0)
    )
    result["block_bootstrap_interval_significant"] = (
        result["bootstrap_ci_lower_log_points"].gt(0)
        | result["bootstrap_ci_upper_log_points"].lt(0)
    )
    result["hac_bootstrap_significance_agreement"] = result[
        "hac_interval_significant"
    ].eq(result["block_bootstrap_interval_significant"])
    result["hac_interval_width_log_points"] = (
        result["ci_upper_log_points"] - result["ci_lower_log_points"]
    )
    result["bootstrap_interval_width_log_points"] = (
        result["bootstrap_ci_upper_log_points"]
        - result["bootstrap_ci_lower_log_points"]
    )
    result["bootstrap_to_hac_width_ratio"] = result[
        "bootstrap_interval_width_log_points"
    ].div(result["hac_interval_width_log_points"].replace(0, np.nan))
    result = result.merge(
        stability[
            [
                "frequency",
                "msdi_path",
                "horizon",
                "effect_range_log_points",
                "maximum_absolute_deviation_from_primary",
                "all_learner_signs_agree",
                "learner_sign_reversal_model_dependence_flag",
            ]
        ],
        on=["frequency", "msdi_path", "horizon"],
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        innovation[
            [
                *keys,
                "correlation_oof_innovation_with_script08_full_sample_innovation",
                "script08_innovation_used_as_dml_treatment",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    result = result.merge(
        overlap[
            [
                *keys,
                "conditional_variation_ratio",
                "residual_information_effective_n",
                "maximum_single_observation_information_share",
                "weak_conditional_variation_flag",
                "influence_concentration_warning_flag",
                "influence_concentration_severe_flag",
                "low_information_ess_flag",
                "training_support_extrapolation_warning_flag",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    result = result.merge(
        econometric[
            [
                *keys,
                "effect_sign_agreement",
                "estimand_exactly_aligned",
                "effect_difference_log_points_dml_minus_lp",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    placebo_summary = placebos[
        [
            *keys,
            "p_value",
            "p_value_fdr_all",
            "falsification_flag_raw",
            "falsification_flag_fdr",
        ]
    ].rename(
        columns={
            "p_value": "pre_treatment_placebo_p_value",
            "p_value_fdr_all": "pre_treatment_placebo_p_value_fdr",
        }
    )
    result = result.merge(placebo_summary, on=keys, how="left", validate="one_to_one")
    result["causal_claim_allowed"] = False
    result["cohort_scope"] = "upstream_stage08b_broader_panel_not_frozen_common_6481"
    result["evidence_role"] = "broader_cohort_causal_ml_sensitivity_only"
    result["interpretation_guardrail"] = (
        "orthogonalized_predictive_effect_not_structural_causality"
    )
    result["sensitivity_flag"] = (
        ~result["hac_bootstrap_significance_agreement"].astype(bool)
        | result["learner_sign_reversal_model_dependence_flag"].astype(bool)
        | result["influence_concentration_warning_flag"].astype(bool)
        | result["low_information_ess_flag"].astype(bool)
        | result["falsification_flag_fdr"].astype(bool)
    )
    audit = pd.DataFrame(
        [
            {
                "manifest_path": str(path),
                "manifest_sha256": sha256_file(path),
                "rows": len(result),
                "causal_claim_allowed": False,
                "hac_bootstrap_decision_disagreements": int(
                    (~result["hac_bootstrap_significance_agreement"]).sum()
                ),
                "learner_sign_reversal_cells": int(
                    result["learner_sign_reversal_model_dependence_flag"].sum()
                ),
                "influence_warning_cells": int(
                    result["influence_concentration_warning_flag"].sum()
                ),
                "low_information_ess_cells": int(
                    result["low_information_ess_flag"].sum()
                ),
                "pre_treatment_fdr_falsification_flags": int(
                    result["falsification_flag_fdr"].sum()
                ),
                "alpha": alpha,
                "cohort_scope": "upstream_stage08b_broader_panel_not_frozen_common_6481",
                "evidence_role": "broader_cohort_causal_ml_sensitivity_only",
            }
        ]
    )
    return result, audit


def _future_lead_folds(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    horizon: int,
    lead: int,
    minimum_training_rows: int,
    evaluation_block_size: int,
    maximum_evaluation_rows: int | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if lead <= horizon:
        raise ValueError("Future-treatment lead must lie beyond the outcome window")
    future_treatment = frame["msdi_change"].shift(-lead)
    outcome = frame[f"outcome_h{horizon}"]
    complete = (
        frame[list(feature_columns)].notna().all(axis=1)
        & future_treatment.notna()
        & outcome.notna()
    )
    eligible = frame.loc[complete, "position"].to_numpy(dtype=int)
    endpoint_gap = max(horizon, lead)
    first_index: int | None = None
    for index, position in enumerate(eligible):
        training = eligible[(eligible < position) & (eligible + endpoint_gap < position)]
        if len(training) >= minimum_training_rows:
            first_index = index
            break
    if first_index is None:
        raise ValueError(
            f"Cannot form future-lead folds h={horizon} lead={lead} "
            f"with {minimum_training_rows} training rows"
        )
    evaluations = eligible[first_index:]
    if maximum_evaluation_rows is not None:
        evaluations = evaluations[:maximum_evaluation_rows]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, len(evaluations), evaluation_block_size):
        evaluation = evaluations[start : start + evaluation_block_size]
        first = int(evaluation[0])
        training = eligible[
            (eligible < first) & (eligible + endpoint_gap < first)
        ]
        if int(training[-1]) + endpoint_gap >= first:
            raise AssertionError("Future-lead latest-endpoint embargo failed")
        folds.append((training, evaluation))
    return folds


def cross_fit_future_treatment_lead(
    support08b: Any,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    spec: Any,
    path_name: str,
    horizon: int,
    lead: int,
    learner_parameters: Mapping[str, Any],
    seed: int,
    maximum_evaluation_rows: int | None,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = _future_lead_folds(
        frame,
        feature_columns,
        horizon,
        lead,
        spec.minimum_training_rows,
        spec.refit_every,
        maximum_evaluation_rows,
    )
    future_treatment = frame["msdi_change"].shift(-lead)
    outcome = frame[f"outcome_h{horizon}"]
    prediction_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for fold_number, (train, evaluation) in enumerate(folds, start=1):
        treatment_seed = stable_seed(
            seed, spec.frequency, path_name, horizon, lead, fold_number, "future_d"
        )
        outcome_seed = stable_seed(
            seed, spec.frequency, path_name, horizon, lead, fold_number, "outcome"
        )
        treatment_model = support08b.make_learner(
            "elastic_net", dict(learner_parameters), treatment_seed
        )
        outcome_model = support08b.make_learner(
            "elastic_net", dict(learner_parameters), outcome_seed
        )
        x_train = frame.loc[train, list(feature_columns)]
        x_evaluation = frame.loc[evaluation, list(feature_columns)]
        treatment_model.fit(x_train, future_treatment.loc[train])
        outcome_model.fit(x_train, outcome.loc[train])
        d_prediction = np.asarray(
            treatment_model.predict(x_evaluation), dtype=float
        )
        y_prediction = np.asarray(outcome_model.predict(x_evaluation), dtype=float)
        d_actual = future_treatment.loc[evaluation].to_numpy(dtype=float)
        y_actual = outcome.loc[evaluation].to_numpy(dtype=float)
        endpoint_gap = max(horizon, lead)
        purge_passed = int(train[-1]) + endpoint_gap < int(evaluation[0])
        audit_rows.append(
            {
                "frequency": spec.frequency,
                "msdi_path": path_name,
                "horizon": horizon,
                "future_treatment_lead": lead,
                "fold": fold_number,
                "training_rows": len(train),
                "evaluation_rows": len(evaluation),
                "training_start_position": int(train[0]),
                "training_end_position": int(train[-1]),
                "training_latest_endpoint_position": int(train[-1]) + endpoint_gap,
                "evaluation_start_position": int(evaluation[0]),
                "training_start_date": frame.at[int(train[0]), "date"],
                "training_end_feature_date": frame.at[int(train[-1]), "date"],
                "training_latest_endpoint_date": frame.at[
                    int(train[-1]) + endpoint_gap, "date"
                ],
                "evaluation_start_date": frame.at[int(evaluation[0]), "date"],
                "latest_endpoint_embargo_passed": purge_passed,
                "lead_strictly_beyond_outcome_passed": lead > horizon,
                "treatment_seed": treatment_seed,
                "outcome_seed": outcome_seed,
            }
        )
        if not purge_passed:
            raise AssertionError("Future-treatment lead fold failed its embargo")
        for offset, position in enumerate(evaluation):
            prediction_rows.append(
                {
                    "frequency": spec.frequency,
                    "frequency_role": spec.role,
                    "msdi_path": path_name,
                    "horizon": horizon,
                    "future_treatment_lead": lead,
                    "fold": fold_number,
                    "position": int(position),
                    "date": frame.at[int(position), "date"],
                    "outcome": float(y_actual[offset]),
                    "outcome_prediction": float(y_prediction[offset]),
                    "outcome_residual": float(y_actual[offset] - y_prediction[offset]),
                    "future_treatment": float(d_actual[offset]),
                    "future_treatment_prediction": float(d_prediction[offset]),
                    "future_treatment_residual": float(
                        d_actual[offset] - d_prediction[offset]
                    ),
                }
            )
        logger.info(
            "future_lead_crossfit frequency=%s path=%s h=%s lead=%s fold=%s "
            "train=%s eval=%s embargo=%s",
            spec.frequency,
            path_name,
            horizon,
            lead,
            fold_number,
            len(train),
            len(evaluation),
            purge_passed,
        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(audit_rows)


def run_future_treatment_lead_placebos(
    support08b: Any,
    causal_config_path: Path,
    selected_frequencies: Sequence[str],
    lead_offset: int,
    repetitions: int,
    alpha: float,
    seed: int,
    maximum_evaluation_rows: int | None,
    logger: logging.Logger,
    panel_overrides: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    causal_config = support08b.load_yaml(require_file(causal_config_path, "Causal-ML config"))
    specs = support08b.frequency_specifications(causal_config)
    learner_parameters = causal_config["learners"]["elastic_net"]
    prediction_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    for frequency in selected_frequencies:
        spec = specs[frequency]
        if panel_overrides is not None and frequency in panel_overrides:
            panel = panel_overrides[frequency].copy()
        else:
            panel = support08b.load_panel(spec.panel_path, spec)
        panel, _ = support08b.apply_panel_completeness_contract(panel, spec)
        for path_name in PATHS:
            frame, features = support08b.prepare_analysis_frame(panel, spec, path_name)
            for horizon in spec.horizons:
                lead = int(horizon) + int(lead_offset)
                predictions, audit = cross_fit_future_treatment_lead(
                    support08b,
                    frame,
                    features,
                    spec,
                    path_name,
                    int(horizon),
                    lead,
                    learner_parameters,
                    seed,
                    maximum_evaluation_rows,
                    logger,
                )
                prediction_frames.append(predictions)
                audit_frames.append(audit)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    # Freeze a common peak/breadth sample at each frequency/horizon.
    retained: list[pd.DataFrame] = []
    for (frequency, horizon), group in predictions.groupby(["frequency", "horizon"]):
        date_sets = [
            set(part["date"])
            for _, part in group.groupby("msdi_path")
        ]
        common_dates = set.intersection(*date_sets)
        retained.append(group.loc[group["date"].isin(common_dates)])
    predictions = pd.concat(retained, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for key, group in predictions.groupby(
        ["frequency", "frequency_role", "msdi_path", "horizon", "future_treatment_lead"]
    ):
        residual_scale = float(group["future_treatment_residual"].std(ddof=1))
        estimate = support08b.orthogonal_estimate(
            group["outcome_residual"],
            group["future_treatment_residual"],
            alpha=alpha,
            hac_lags=max(int(key[3]), int(key[4])),
            bootstrap_repetitions=repetitions,
            block_length=max(
                int(key[4]), int(np.ceil(len(group) ** (1 / 3)))
            ),
            seed=stable_seed(seed, *key, "future_lead_effect"),
            effect_scale=residual_scale,
        )
        rows.append(
            {
                "frequency": key[0],
                "frequency_role": key[1],
                "msdi_path": key[2],
                "horizon": int(key[3]),
                "future_treatment_lead": int(key[4]),
                "lead_strictly_beyond_outcome": int(key[4]) > int(key[3]),
                "learner": "elastic_net",
                "n_observations": len(group),
                "sample_start_date": group["date"].min(),
                "sample_end_date": group["date"].max(),
                "expected_result": "no_association",
                "interpretation": "future_treatment_lead_falsification_not_identification_proof",
                **estimate,
            }
        )
    estimates = pd.DataFrame(rows)
    estimates["p_value_fdr_all"] = adjust_pvalues(estimates["p_value"], "fdr_bh")
    primary = (
        estimates["frequency"].eq("weekly")
        & estimates["msdi_path"].eq("peak")
    )
    estimates["p_value_holm_primary_family"] = np.nan
    estimates.loc[primary, "p_value_holm_primary_family"] = adjust_pvalues(
        estimates.loc[primary, "p_value"], "holm"
    )
    estimates["falsification_flag_fdr"] = estimates["p_value_fdr_all"].lt(alpha)
    estimates["cohort_scope"] = "frozen_6481_all_three_llm_common_articles"
    predictions["cohort_scope"] = "frozen_6481_all_three_llm_common_articles"
    fold_audit = pd.concat(audit_frames, ignore_index=True)
    fold_audit["cohort_scope"] = "frozen_6481_all_three_llm_common_articles"
    return (
        estimates,
        predictions,
        fold_audit,
    )


def selected_dynamic_lags_from_manifest(
    manifest: Mapping[str, Any], selected_frequencies: Sequence[str]
) -> tuple[dict[tuple[str, str], int], pd.DataFrame]:
    value = manifest.get("outputs", {}).get("var_lag_selection")
    path = _manifest_output_path(value)
    if path is None:
        raise ValueError("Causality manifest lacks var_lag_selection output")
    require_file(path, "Stage-08 VAR lag selection")
    expected_hash = manifest.get("output_sha256", {}).get("var_lag_selection")
    actual_hash = sha256_file(path)
    if expected_hash != actual_hash:
        raise ValueError(
            "Stage-08 VAR lag-selection hash mismatch: "
            f"recorded={expected_hash} actual={actual_hash}"
        )
    table = pd.read_csv(path, low_memory=False)
    selected_flag = (
        table["selected_for_analysis"].astype(str).str.strip().str.lower().eq("true")
    )
    selected = table.loc[
        selected_flag & table["frequency"].isin(selected_frequencies)
    ].copy()
    counts = selected.groupby(["frequency", "msdi_path"]).size()
    expected_keys = {
        (frequency, path_name)
        for frequency in selected_frequencies
        for path_name in PATHS
    }
    if set(counts.index) != expected_keys or not counts.eq(1).all():
        raise ValueError(
            "Stage-08 selected-lag contract requires exactly one row per "
            f"frequency/path; found={counts.to_dict()} expected={expected_keys}"
        )
    mapping = {
        (str(row.frequency), str(row.msdi_path)): int(row.lag)
        for row in selected.itertuples(index=False)
    }
    selected["source_path"] = str(path)
    selected["source_sha256"] = actual_hash
    return mapping, selected.reset_index(drop=True)


def nonlinear_forecast_robustness_from_manifest(
    manifest_path: Path,
    selected_frequencies: Sequence[str],
    expected_algorithm_names: Sequence[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    path = require_file(manifest_path, "Stage-07d nonlinear manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    design = manifest.get("design", {})
    expected_algorithms = set(expected_algorithm_names)
    if len(expected_algorithms) != len(tuple(expected_algorithm_names)):
        raise ValueError("Configured extended algorithms must be unique")
    requested = set(design.get("requested_algorithms", []))
    executed = set(design.get("executed_algorithms", []))
    if (
        requested != expected_algorithms
        or executed != expected_algorithms
        or not bool(design.get("full_requested_model_set_complete"))
        or not bool(design.get("full_canonical_scope_complete"))
        or bool(design.get("smoke_test"))
        or design.get("maximum_configurations_per_construction") is not None
        or design.get("maximum_origins_per_horizon") is not None
    ):
        raise ValueError("Stage-07d nonlinear forecast scope is not canonical/full")
    output_keys = ("pipeline_ranking", "metrics", "statistical_tests", "origin_audit")
    frames: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []
    for key in output_keys:
        output_path = _manifest_output_path(manifest.get("outputs", {}).get(key))
        if output_path is None:
            raise ValueError(f"Stage-07d manifest lacks output path for {key}")
        require_file(output_path, f"Stage-07d {key}")
        frame = pd.read_csv(output_path, low_memory=False)
        if "frequency" in frame:
            frame = frame.loc[frame["frequency"].isin(selected_frequencies)].copy()
        frame["evidence_role"] = "secondary_predictive_algorithm_robustness"
        frame["causal_claim_allowed"] = False
        frames[key] = frame
        audit_rows.append(
            {
                "source": key,
                "path": str(output_path),
                "sha256_runtime": sha256_file(output_path),
                "rows_selected": len(frame),
                "selected_frequencies": "|".join(selected_frequencies),
            }
        )
    origin = frames["origin_audit"]
    origin_passed = (
        origin["origin_sets_identical"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )
    if origin.empty or not origin_passed.all():
        raise ValueError("Stage-07d common-origin audit failed")
    audit = pd.DataFrame(audit_rows)
    audit["manifest_path"] = str(path)
    audit["manifest_sha256"] = sha256_file(path)
    audit["algorithms"] = "|".join(sorted(expected_algorithms))
    audit["all_origin_sets_identical"] = True
    audit["fixed_predefined_parameters_no_tuning"] = not bool(
        design.get("tuning_enabled")
    )
    return frames, audit


def build_specification_registry(
    variants: Sequence[IndexVariant],
    all_specifications: Mapping[str, Sequence[ForecastSpecification]],
    selected_specifications: Mapping[str, Sequence[ForecastSpecification]],
    comparisons: Mapping[str, Sequence[PairedComparison]],
    selected_frequencies: Sequence[str],
    unavailable: Sequence[Mapping[str, Any]],
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    variant_map = {item.variant_id: item for item in variants}
    for variant in variants:
        rows.append(
            {
                "registry_type": "index_variant",
                "frequency": "article_to_weekly_and_monthly",
                "index_construction": "all_feasible",
                "specification_id": variant.variant_id,
                "family_id": variant.family_id,
                "status": (
                    "descriptive_randomization_only_not_forecasted"
                    if variant.family_id == "date_placebo"
                    else "estimated"
                ),
                "primary": variant.primary,
                "comparison_reference": variant.comparison_reference,
                "details_json": json.dumps(asdict(variant), default=json_default),
                "reason": (
                    "whole_sample_date_permutation_can_move_later_news_to_earlier_origins"
                    if variant.family_id == "date_placebo"
                    else ""
                ),
            }
        )
    estimated = set(
        zip(
            forecasts["frequency"],
            forecasts["index_construction"],
            forecasts["specification_id"],
        )
    )
    for frequency in selected_frequencies:
        selected_ids = {
            item.specification_id for item in selected_specifications[frequency]
        }
        constructions = (
            ["weekly_log_volume"]
            if frequency == "weekly"
            else list(MONTHLY_CONSTRUCTIONS)
        )
        for construction in constructions:
            for specification in all_specifications[frequency]:
                reason = forecast_structural_skip_reason(
                    specification, construction, variant_map
                )
                key = (frequency, construction, specification.specification_id)
                if specification.specification_id not in selected_ids:
                    status = "not_selected_debug_limit"
                elif reason is not None:
                    status = "skipped_structurally_equivalent"
                elif key in estimated:
                    status = "estimated"
                else:
                    status = "not_estimated"
                rows.append(
                    {
                        "registry_type": "forecast_specification",
                        "frequency": frequency,
                        "index_construction": construction,
                        "specification_id": specification.specification_id,
                        "family_id": specification.family_id,
                        "status": status,
                        "primary": specification.specification_id == "peak_only"
                        and frequency == "weekly",
                        "comparison_reference": "",
                        "details_json": json.dumps(
                            asdict(specification), default=json_default
                        ),
                        "reason": reason or "",
                    }
                )
        for comparison in comparisons[frequency]:
            rows.append(
                {
                    "registry_type": "paired_comparison",
                    "frequency": frequency,
                    "index_construction": "all_estimable_constructions",
                    "specification_id": comparison.comparison_id,
                    "family_id": comparison.family_id,
                    "status": "estimated_when_both_cells_available",
                    "primary": comparison.primary,
                    "comparison_reference": comparison.baseline_specification_id,
                    "details_json": json.dumps(
                        asdict(comparison), default=json_default
                    ),
                    "reason": comparison.direction_note,
                }
            )
    for item in unavailable:
        rows.append(
            {
                "registry_type": "unavailable_specification",
                "frequency": "all",
                "index_construction": "not_applicable",
                "specification_id": item.get("specification_id", "unknown"),
                "family_id": item.get("analysis_layer", "unknown"),
                "status": item.get("status", "skipped_unavailable"),
                "primary": False,
                "comparison_reference": "",
                "details_json": json.dumps(dict(item), default=json_default),
                "reason": item.get("reason", ""),
            }
        )
    return pd.DataFrame(rows)


def package_versions() -> dict[str, str]:
    values = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "tigramite",
        "PyYAML",
    ):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = "not_installed"
    return values


def _read_checkpoint(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _write_frame_collection(
    frames: Mapping[str, pd.DataFrame], paths: Mapping[str, Path]
) -> None:
    for key, frame in frames.items():
        atomic_write_csv(frame, paths[key])


def _selected_frequencies(value: str) -> list[str]:
    return ["weekly", "monthly"] if value == "both" else [value]


def main() -> int:
    args = parse_args()
    executable = assert_project_venv()
    config = load_yaml(args.config)
    validate_args(args, config)
    version = str(args.version or config.get("version", "robustness_v001"))
    if not version or not version.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Unsafe output version: {version!r}")
    logger, log_path = configure_logging(args.log_dir.resolve(), version)
    started = time.perf_counter()
    try:
        selected_frequencies = _selected_frequencies(args.frequency)
        configured_repetitions = int(config["inference"]["bootstrap_repetitions"])
        repetitions = (
            int(args.bootstrap_repetitions)
            if args.bootstrap_repetitions is not None
            else (0 if args.smoke_test else configured_repetitions)
        )
        seed = (
            int(args.seed)
            if args.seed is not None
            else int(config["inference"]["random_seed"])
        )
        maximum_specifications = args.maximum_specifications
        maximum_origins = args.maximum_origins
        if args.smoke_test:
            maximum_specifications = maximum_specifications or 4
            maximum_origins = maximum_origins or 3
        alpha = float(config["inference"]["alpha"])
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "stage_start version=%s executable=%s frequencies=%s repetitions=%s "
            "seed=%s smoke=%s maximum_specs=%s maximum_origins=%s resume=%s",
            version,
            executable,
            "|".join(selected_frequencies),
            repetitions,
            seed,
            args.smoke_test,
            maximum_specifications,
            maximum_origins,
            args.resume,
        )

        upstream_audit, upstream = validate_upstream_manifests(args)
        aliases = load_model_aliases(args.model_registry)
        expected_articles = int(config["analysis"]["frozen_common_article_cohort"])
        article, article_report = load_common_articles(
            args.article_scores, aliases, expected_articles
        )
        requested_variants, unavailable = build_index_variants(
            article, aliases, config
        )
        variants = requested_variants
        if maximum_specifications is not None:
            references = [item for item in variants if item.family_id == "reference"]
            alternatives = [
                item for item in variants if item.family_id != "reference"
            ]
            alternative_limit = max(int(maximum_specifications) - 4, 0)
            variants = [*references, *alternatives[:alternative_limit]]
            logger.info(
                "debug_measurement_scope variants=%s/%s reason=forecast_specification_limit",
                len(variants),
                len(requested_variants),
            )
        logger.info(
            "article_contract rows=%s aliases=%s variants=%s first=%s last=%s",
            len(article),
            "|".join(aliases),
            len(variants),
            article_report["first_publication_date"],
            article_report["last_publication_date"],
        )
        weekly_long = aggregate_all_weekly_variants(
            article, variants, aliases, seed, logger
        )
        weekly_base = load_weekly_base_panel(args.weekly_panel)
        monthly_base = load_monthly_base_panel(args.monthly_panel)
        weekly_aligned, weekly_alignment_audit = align_weekly_variants_to_panel(
            weekly_base, weekly_long
        )
        monthly_aligned, monthly_long, monthly_alignment_audit = (
            build_monthly_variant_panel(
                weekly_aligned, monthly_base, variants, ema_span=3
            )
        )
        common_analysis_panels = build_common_cohort_analysis_panels(
            weekly_aligned, monthly_aligned
        )
        weekly_reference_path = _manifest_output_path(
            upstream["weekly_llm"].get("outputs", {}).get("aligned_panel")
        )
        monthly_reference_path = _manifest_output_path(
            upstream["monthly_llm"].get("outputs", {}).get("monthly_panel")
        )
        if weekly_reference_path is None or monthly_reference_path is None:
            raise ValueError("Weekly/monthly LLM manifests lack aligned panel paths")
        reconstruction_audit = reference_reconstruction_audit(
            article,
            weekly_long,
            monthly_long,
            weekly_reference_path,
            monthly_reference_path,
        )
        logger.info(
            "reference_reconstruction checks=%s maximum_difference=%.3g passed=%s",
            len(reconstruction_audit),
            reconstruction_audit["maximum_absolute_difference"].max(),
            bool(reconstruction_audit["contract_passed"].all()),
        )
        weekly_forecast_panel, monthly_forecast_panel = prepare_forecast_panels(
            weekly_aligned, monthly_aligned, variants, config
        )

        all_specifications: dict[str, list[ForecastSpecification]] = {}
        all_comparisons: dict[str, list[PairedComparison]] = {}
        selected_specifications: dict[str, list[ForecastSpecification]] = {}
        selected_comparisons: dict[str, list[PairedComparison]] = {}
        for frequency in selected_frequencies:
            specifications, comparisons = build_forecast_specifications(
                variants, frequency, config
            )
            all_specifications[frequency] = specifications
            all_comparisons[frequency] = comparisons
            selected = (
                specifications[:maximum_specifications]
                if maximum_specifications is not None
                else specifications
            )
            selected_specifications[frequency] = selected
            selected_ids = {item.specification_id for item in selected}
            selected_comparisons[frequency] = [
                item
                for item in comparisons
                if item.baseline_specification_id in selected_ids
                and item.variant_specification_id in selected_ids
            ]
            logger.info(
                "forecast_scope frequency=%s specifications=%s/%s comparisons=%s/%s",
                frequency,
                len(selected),
                len(specifications),
                len(selected_comparisons[frequency]),
                len(comparisons),
            )

        input_paths = {
            "config": require_file(args.config, "Robustness config"),
            "model_registry": require_file(args.model_registry, "Model registry"),
            "article_scores": require_file(args.article_scores, "Article scores"),
            "weekly_panel": require_file(args.weekly_panel, "Weekly panel"),
            "monthly_panel": require_file(args.monthly_panel, "Monthly panel"),
            "causality_monthly_panel": require_file(
                args.causality_monthly_panel, "Monthly causality panel"
            ),
            "causal_ml_config": require_file(args.causal_ml_config, "Causal-ML config"),
            "script": Path(__file__).resolve(),
            "stage08_support_script": require_file(
                SCRIPT_08, "Stage-08 support script"
            ),
            "stage08b_support_script": require_file(
                SCRIPT_08B, "Stage-08b support script"
            ),
        }
        input_hashes = {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in input_paths.items()
        }
        software_versions = package_versions()
        signature_payload = {
            "version": version,
            "inputs": input_hashes,
            "upstream_manifest_hashes": upstream_audit.set_index("upstream_stage")[
                "manifest_sha256"
            ].to_dict(),
            "frequencies": selected_frequencies,
            "specifications": {
                frequency: [asdict(item) for item in values]
                for frequency, values in selected_specifications.items()
            },
            "bootstrap_repetitions": repetitions,
            "seed": seed,
            "maximum_origins": maximum_origins,
            "skip_dynamic_refits": args.skip_dynamic_refits,
            "skip_future_lead_placebos": args.skip_future_lead_placebos,
            "software": software_versions,
        }
        signature_json = json.dumps(
            signature_payload, sort_keys=True, default=json_default
        )
        run_signature = hashlib.sha256(signature_json.encode("utf-8")).hexdigest()
        checkpoint_dir = output_dir / "checkpoints" / run_signature[:16]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            checkpoint_dir / "run_signature.json",
            json.dumps(
                {"run_signature": run_signature, **signature_payload},
                indent=2,
                sort_keys=True,
                default=json_default,
            )
            + "\n",
        )
        forecasts, fit_audit, forecast_skips = run_all_forecasts(
            panels={
                "weekly": weekly_forecast_panel,
                "monthly": monthly_forecast_panel,
            },
            specifications=selected_specifications,
            selected_frequencies=selected_frequencies,
            config=config,
            checkpoint_dir=checkpoint_dir / "forecasts",
            resume=args.resume,
            maximum_origins=maximum_origins,
            variant_map={item.variant_id: item for item in variants},
            logger=logger,
        )
        forecast_metrics = calculate_forecast_metrics(forecasts)
        ablation_tests, paired_differences = paired_ablation_tests(
            forecasts,
            selected_comparisons,
            selected_frequencies,
            config,
            repetitions,
            seed,
        )
        regime_sensitivity = forecast_regime_sensitivity(forecasts)
        index_stability = index_stability_summary(
            weekly_long, monthly_long, variants
        )
        logger.info(
            "forecast_complete predictions=%s fits=%s metrics=%s tests=%s paired_rows=%s",
            len(forecasts),
            len(fit_audit),
            len(forecast_metrics),
            len(ablation_tests),
            len(paired_differences),
        )

        _upstream_selected_lags, upstream_selected_lag_audit = (
            selected_dynamic_lags_from_manifest(
            upstream["causality"], selected_frequencies
            )
        )
        upstream_selected_lag_audit["cohort_scope"] = (
            "upstream_stage08_broader_panel_not_frozen_common_6481"
        )
        upstream_selected_lag_audit["selection_role"] = (
            "broader_cohort_reference_only_not_used_for_common_cohort_refits"
        )
        common_selected_lags: dict[tuple[str, str], int] = {}
        common_selected_lag_rows: list[dict[str, Any]] = []
        var_selection_frames: list[pd.DataFrame] = []
        var_diagnostic_frames: list[pd.DataFrame] = []
        var_coefficient_frames: list[pd.DataFrame] = []
        conventional_granger_frames: list[pd.DataFrame] = []
        conditional_granger_frames: list[pd.DataFrame] = []
        ardl_frames: list[pd.DataFrame] = []
        lp_frames: list[pd.DataFrame] = []
        pcmci_frames: list[pd.DataFrame] = []
        pcmci_stability_frames: list[pd.DataFrame] = []
        peak_breadth_frames: list[pd.DataFrame] = []
        if not args.skip_dynamic_refits:
            support08 = import_support(SCRIPT_08, "stage08_support_for_robustness")
            dynamic_frames = prepare_dynamic_frames(
                support08,
                args.weekly_panel,
                args.causality_monthly_panel,
                selected_frequencies,
                panel_overrides=common_analysis_panels,
            )
            dynamic_config = config["dynamic_sensitivity"]
            for frequency in selected_frequencies:
                var_lags = [
                    int(value)
                    for value in dynamic_config["var_granger"][
                        f"{frequency}_max_lag_specs"
                    ]
                ]
                ardl_lags = [
                    int(value)
                    for value in dynamic_config["ardl_ecm"][f"{frequency}_lags"]
                ]
                lp_lags = [
                    int(value)
                    for value in dynamic_config["local_projection"][
                        f"{frequency}_lags"
                    ]
                ]
                pcmci_lags = [
                    int(value)
                    for value in dynamic_config["pcmci"][
                        f"{frequency}_tau_max_specs"
                    ]
                ]
                if args.smoke_test:
                    var_lags = var_lags[:1]
                    ardl_lags = ardl_lags[:1]
                    lp_lags = lp_lags[:1]
                    pcmci_lags = pcmci_lags[:1]
                maximum_horizon = int(
                    dynamic_config["local_projection"][
                        f"{frequency}_maximum_horizon"
                    ]
                )
                for path_name in PATHS:
                    frame = dynamic_frames[(frequency, path_name)]
                    prefix = checkpoint_dir / "dynamic" / f"{frequency}_{path_name}"
                    cache = {
                        "var_selection": Path(str(prefix) + "_var_selection.csv"),
                        "var_diagnostics": Path(str(prefix) + "_var_diagnostics.csv"),
                        "var_coefficients": Path(str(prefix) + "_var_coefficients.csv"),
                        "conventional": Path(str(prefix) + "_conventional_granger.csv"),
                        "conditional": Path(str(prefix) + "_conditional_granger.csv"),
                        "ardl": Path(str(prefix) + "_ardl_ecm.csv"),
                        "lp": Path(str(prefix) + "_local_projection.csv"),
                        "pcmci": Path(str(prefix) + "_pcmci.csv"),
                        "pcmci_stability": Path(str(prefix) + "_pcmci_stability.csv"),
                    }
                    if args.resume and all(path.is_file() for path in cache.values()):
                        cell = {key: _read_checkpoint(path) for key, path in cache.items()}
                        logger.info(
                            "dynamic_checkpoint_loaded frequency=%s path=%s",
                            frequency,
                            path_name,
                        )
                    else:
                        var_selection, var_diagnostics, var_coefficients = (
                            varx_lag_sensitivity(
                                support08,
                                frame,
                                frequency,
                                path_name,
                                var_lags,
                                alpha,
                                logger,
                            )
                        )
                        conventional = conventional_granger_sensitivity(
                            support08, frame, frequency, path_name, var_lags
                        )
                        conditional = conditional_granger_sensitivity(
                            frame, frequency, path_name, var_lags, alpha
                        )
                        ardl = ardl_ecm_sensitivity(
                            frame,
                            frequency,
                            path_name,
                            ardl_lags,
                            dynamic_config["ardl_ecm"]["deterministic_terms"],
                            alpha,
                        )
                        lp = local_projection_lag_sensitivity(
                            support08,
                            frame,
                            frequency,
                            path_name,
                            lp_lags,
                            maximum_horizon,
                            alpha,
                        )
                        pcmci, pcmci_stability = run_pcmci_sensitivity(
                            support08,
                            frame,
                            frequency,
                            path_name,
                            pcmci_lags,
                            [float(dynamic_config["pcmci"]["pc_alpha"])],
                            alpha,
                            logger,
                        )
                        cell = {
                            "var_selection": var_selection,
                            "var_diagnostics": var_diagnostics,
                            "var_coefficients": var_coefficients,
                            "conventional": conventional,
                            "conditional": conditional,
                            "ardl": ardl,
                            "lp": lp,
                            "pcmci": pcmci,
                            "pcmci_stability": pcmci_stability,
                        }
                        for key, path in cache.items():
                            atomic_write_csv(cell[key], path)
                    for key in (
                        "var_selection",
                        "var_diagnostics",
                        "var_coefficients",
                        "conventional",
                        "conditional",
                        "ardl",
                        "lp",
                        "pcmci",
                        "pcmci_stability",
                    ):
                        if "dynamic_effective_sample_matched_across_lags" not in cell[
                            key
                        ].columns:
                            cell[key][
                                "dynamic_effective_sample_matched_across_lags"
                            ] = False
                        cell[key]["cohort_scope"] = (
                            "frozen_6481_all_three_llm_common_articles"
                        )
                    var_selection_frames.append(cell["var_selection"])
                    var_diagnostic_frames.append(cell["var_diagnostics"])
                    var_coefficient_frames.append(cell["var_coefficients"])
                    conventional_granger_frames.append(cell["conventional"])
                    conditional_granger_frames.append(cell["conditional"])
                    ardl_frames.append(cell["ardl"])
                    lp_frames.append(cell["lp"])
                    pcmci_frames.append(cell["pcmci"])
                    pcmci_stability_frames.append(cell["pcmci_stability"])
                    lag_table = cell["var_selection"].copy()
                    selected_flag = (
                        lag_table["selected_for_analysis"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .eq("true")
                    )
                    maximum_grid = pd.to_numeric(
                        lag_table["maximum_lag_specification"], errors="raise"
                    ).max()
                    canonical = lag_table.loc[
                        selected_flag
                        & pd.to_numeric(
                            lag_table["maximum_lag_specification"], errors="raise"
                        ).eq(maximum_grid)
                    ]
                    if len(canonical) != 1:
                        raise ValueError(
                            "Common-cohort VARX lag contract requires one canonical "
                            f"selection for {frequency}/{path_name}; found={len(canonical)}"
                        )
                    canonical_row = canonical.iloc[0]
                    canonical_lag = int(canonical_row["lag"])
                    common_selected_lags[(frequency, path_name)] = canonical_lag
                    common_selected_lag_rows.append(
                        {
                            "frequency": frequency,
                            "msdi_path": path_name,
                            "lag": canonical_lag,
                            "maximum_lag_specification": int(maximum_grid),
                            "passes_residual_whiteness": canonical_row.get(
                                "passes_residual_whiteness"
                            ),
                            "selection_reason": canonical_row.get(
                                "selection_reason"
                            ),
                            "cohort_scope": (
                                "frozen_6481_all_three_llm_common_articles"
                            ),
                            "selection_role": (
                                "canonical_common_cohort_dynamic_lag_used_for_"
                                "shared_peak_breadth_bootstrap"
                            ),
                        }
                    )
                shared_path = (
                    checkpoint_dir / "dynamic" / f"{frequency}_peak_breadth.csv"
                )
                if args.resume and shared_path.is_file():
                    shared = _read_checkpoint(shared_path)
                    logger.info(
                        "dynamic_shared_bootstrap_checkpoint_loaded frequency=%s rows=%s",
                        frequency,
                        len(shared),
                    )
                else:
                    shared = shared_peak_breadth_bootstrap(
                        support08,
                        dynamic_frames,
                        common_selected_lags,
                        frequency,
                        maximum_horizon,
                        repetitions,
                        alpha,
                        stable_seed(seed, frequency, "shared_peak_breadth"),
                    )
                    atomic_write_csv(shared, shared_path)
                peak_breadth_frames.append(shared)
            logger.info(
                "dynamic_complete var_cells=%s conventional=%s conditional=%s "
                "ardl=%s lp=%s pcmci_links=%s",
                sum(len(item) for item in var_selection_frames),
                sum(len(item) for item in conventional_granger_frames),
                sum(len(item) for item in conditional_granger_frames),
                sum(len(item) for item in ardl_frames),
                sum(len(item) for item in lp_frames),
                sum(len(item) for item in pcmci_frames),
            )

        def combine(items: Sequence[pd.DataFrame]) -> pd.DataFrame:
            return pd.concat(items, ignore_index=True) if items else pd.DataFrame()

        dynamic_outputs = {
            "varx_lag_selection": combine(var_selection_frames),
            "varx_diagnostics": combine(var_diagnostic_frames),
            "varx_coefficients": combine(var_coefficient_frames),
            "conventional_granger": combine(conventional_granger_frames),
            "conditional_granger": combine(conditional_granger_frames),
            "ardl_ecm": combine(ardl_frames),
            "local_projections": combine(lp_frames),
            "pcmci_links": combine(pcmci_frames),
            "pcmci_link_stability": combine(pcmci_stability_frames),
            "peak_breadth_difference": combine(peak_breadth_frames),
        }
        if not dynamic_outputs["peak_breadth_difference"].empty:
            dynamic_outputs["peak_breadth_difference"]["cohort_scope"] = (
                "frozen_6481_all_three_llm_common_articles"
            )
        selected_lag_audit = pd.concat(
            [
                upstream_selected_lag_audit,
                pd.DataFrame(common_selected_lag_rows),
            ],
            ignore_index=True,
            sort=False,
        )

        causal_sensitivity, causal_audit = causal_ml_sensitivity_from_manifest(
            args.causal_ml_manifest, alpha
        )
        future_estimates = pd.DataFrame()
        future_predictions = pd.DataFrame()
        future_fold_audit = pd.DataFrame()
        if not args.skip_future_lead_placebos:
            future_cache = {
                "estimates": checkpoint_dir / "future_lead_estimates.csv",
                "predictions": checkpoint_dir / "future_lead_predictions.csv",
                "folds": checkpoint_dir / "future_lead_fold_audit.csv",
            }
            if args.resume and all(path.is_file() for path in future_cache.values()):
                future_estimates = _read_checkpoint(future_cache["estimates"])
                future_predictions = _read_checkpoint(future_cache["predictions"])
                future_fold_audit = _read_checkpoint(future_cache["folds"])
                logger.info(
                    "future_lead_checkpoint_loaded estimates=%s predictions=%s folds=%s",
                    len(future_estimates),
                    len(future_predictions),
                    len(future_fold_audit),
                )
            else:
                support08b = import_support(
                    SCRIPT_08B, "stage08b_support_for_robustness"
                )
                lead_config = config["causal_sensitivity"][
                    "future_treatment_lead_placebo"
                ]
                future_estimates, future_predictions, future_fold_audit = (
                    run_future_treatment_lead_placebos(
                        support08b,
                        args.causal_ml_config,
                        selected_frequencies,
                        int(lead_config["lead_offset_beyond_outcome"]),
                        repetitions,
                        alpha,
                        seed,
                        (
                            max(12, maximum_origins)
                            if args.smoke_test and maximum_origins is not None
                            else maximum_origins
                        ),
                        logger,
                        panel_overrides=common_analysis_panels,
                    )
                )
                atomic_write_csv(future_estimates, future_cache["estimates"])
                atomic_write_csv(future_predictions, future_cache["predictions"])
                atomic_write_csv(future_fold_audit, future_cache["folds"])
        logger.info(
            "causal_sensitivity_complete consolidated=%s future_estimates=%s "
            "future_predictions=%s future_folds=%s",
            len(causal_sensitivity),
            len(future_estimates),
            len(future_predictions),
            len(future_fold_audit),
        )

        nonlinear_outputs, nonlinear_source_audit = (
            nonlinear_forecast_robustness_from_manifest(
                args.nonlinear_manifest,
                selected_frequencies,
                config["forecast"]["extended_algorithms"],
            )
        )
        logger.info(
            "nonlinear_robustness_consolidated pipeline_ranking=%s metrics=%s "
            "tests=%s origin_audit=%s",
            len(nonlinear_outputs["pipeline_ranking"]),
            len(nonlinear_outputs["metrics"]),
            len(nonlinear_outputs["statistical_tests"]),
            len(nonlinear_outputs["origin_audit"]),
        )

        runtime_skips = list(unavailable)
        runtime_skips.append(
            {
                "specification_id": "permuted_article_dates_oos_forecast",
                "analysis_layer": "forecast",
                "status": "skipped_leakage_contract",
                "reason": (
                    "whole_sample_date_permutation_can_move_later_news_labels_into_"
                    "earlier_forecast_origins;retained_only_as_descriptive_index_"
                    "randomization"
                ),
            }
        )
        if not forecast_skips.empty:
            runtime_skips.extend(forecast_skips.to_dict(orient="records"))
        if args.skip_dynamic_refits:
            runtime_skips.append(
                {
                    "specification_id": "dynamic_sensitivity_refits",
                    "analysis_layer": "dynamic",
                    "status": "skipped_by_debug_argument",
                    "reason": "--skip-dynamic-refits",
                }
            )
        if args.skip_future_lead_placebos:
            runtime_skips.append(
                {
                    "specification_id": "future_treatment_lead_placebos",
                    "analysis_layer": "causal_ml",
                    "status": "skipped_by_debug_argument",
                    "reason": "--skip-future-lead-placebos",
                }
            )
        skipped_frame = pd.DataFrame(runtime_skips)
        specification_registry = build_specification_registry(
            variants,
            all_specifications,
            selected_specifications,
            selected_comparisons,
            selected_frequencies,
            runtime_skips,
            forecasts,
        )
        article_audit_row = dict(article_report)
        article_audit_row["top_source_domains"] = json.dumps(
            article_audit_row["top_source_domains"], sort_keys=True
        )
        article_audit = pd.DataFrame([article_audit_row])
        alignment_audit = pd.concat(
            [weekly_alignment_audit, monthly_alignment_audit],
            ignore_index=True,
            sort=False,
        )
        common_sample_columns = [
            "frequency",
            "index_construction",
            "comparison_id",
            "horizon",
            "sample_subset",
            "loss",
            "paired_n",
            "first_origin_date",
            "last_origin_date",
            "baseline_origins_total",
            "variant_origins_total",
            "baseline_origins_dropped",
            "variant_origins_dropped",
        ]
        common_sample_audit = (
            ablation_tests.loc[
                ablation_tests["sample_subset"].eq("full_sample")
                & ablation_tests["loss"].eq("squared_error"),
                common_sample_columns,
            ].copy()
            if not ablation_tests.empty
            else pd.DataFrame(columns=common_sample_columns)
        )

        variant_map = {item.variant_id: item for item in variants}
        feature_sets: dict[str, Any] = {
            "leakage_contract": config["forecast"]["leakage_contract"],
            "frequencies": {},
        }
        for frequency in selected_frequencies:
            constructions = (
                ["weekly_log_volume"]
                if frequency == "weekly"
                else list(MONTHLY_CONSTRUCTIONS)
            )
            feature_sets["frequencies"][frequency] = {}
            for construction in constructions:
                feature_sets["frequencies"][frequency][construction] = {
                    item.specification_id: {
                        "features": forecast_feature_columns(
                            item, frequency, construction
                        ),
                        "structural_skip_reason": forecast_structural_skip_reason(
                            item, construction, variant_map
                        ),
                    }
                    for item in selected_specifications[frequency]
                }

        frames: dict[str, pd.DataFrame] = {
            "article_audit": article_audit,
            "upstream_audit": upstream_audit,
            "alignment_audit": alignment_audit,
            "reference_reconstruction_audit": reconstruction_audit,
            "specification_registry": specification_registry,
            "skipped_specifications": skipped_frame,
            "index_variant_weekly": weekly_long,
            "index_variant_monthly": monthly_long,
            "index_stability": index_stability,
            "forecast_predictions": forecasts,
            "forecast_metrics": forecast_metrics,
            "paired_loss_differences": paired_differences,
            "ablation_tests": ablation_tests,
            "forecast_regime_sensitivity": regime_sensitivity,
            "fit_audit": fit_audit,
            "common_sample_audit": common_sample_audit,
            "selected_dynamic_lags": selected_lag_audit,
            "varx_lag_selection_sensitivity": dynamic_outputs[
                "varx_lag_selection"
            ],
            "varx_diagnostics_sensitivity": dynamic_outputs["varx_diagnostics"],
            "varx_coefficients_sensitivity": dynamic_outputs["varx_coefficients"],
            "conventional_granger_sensitivity": dynamic_outputs[
                "conventional_granger"
            ],
            "conditional_granger_sensitivity": dynamic_outputs[
                "conditional_granger"
            ],
            "ardl_ecm_sensitivity": dynamic_outputs["ardl_ecm"],
            "local_projection_sensitivity": dynamic_outputs[
                "local_projections"
            ],
            "pcmci_sensitivity_links": dynamic_outputs["pcmci_links"],
            "pcmci_link_stability": dynamic_outputs["pcmci_link_stability"],
            "peak_breadth_difference": dynamic_outputs[
                "peak_breadth_difference"
            ],
            "nonlinear_pipeline_ranking": nonlinear_outputs["pipeline_ranking"],
            "nonlinear_metrics": nonlinear_outputs["metrics"],
            "nonlinear_statistical_tests": nonlinear_outputs[
                "statistical_tests"
            ],
            "nonlinear_origin_audit": nonlinear_outputs["origin_audit"],
            "nonlinear_source_audit": nonlinear_source_audit,
            "causal_ml_sensitivity": causal_sensitivity,
            "causal_ml_source_audit": causal_audit,
            "future_lead_placebos": future_estimates,
            "future_lead_placebo_predictions": future_predictions,
            "future_lead_placebo_fold_audit": future_fold_audit,
        }
        suffixes = {
            "article_audit": "article_cohort_audit.csv",
            "upstream_audit": "upstream_manifest_audit.csv",
            "alignment_audit": "index_alignment_audit.csv",
            "reference_reconstruction_audit": "reference_reconstruction_audit.csv",
            "specification_registry": "specification_registry.csv",
            "skipped_specifications": "skipped_specifications.csv",
            "index_variant_weekly": "index_variants_weekly.csv",
            "index_variant_monthly": "index_variants_monthly.csv",
            "index_stability": "index_stability.csv",
            "forecast_predictions": "forecast_predictions.csv",
            "forecast_metrics": "forecast_metrics.csv",
            "paired_loss_differences": "paired_loss_differences.csv",
            "ablation_tests": "ablation_tests.csv",
            "forecast_regime_sensitivity": "forecast_regime_sensitivity.csv",
            "fit_audit": "forecast_fit_audit.csv",
            "common_sample_audit": "common_sample_audit.csv",
            "selected_dynamic_lags": "selected_dynamic_lags.csv",
            "varx_lag_selection_sensitivity": "varx_lag_selection_sensitivity.csv",
            "varx_diagnostics_sensitivity": "varx_diagnostics_sensitivity.csv",
            "varx_coefficients_sensitivity": "varx_coefficients_sensitivity.csv",
            "conventional_granger_sensitivity": "conventional_granger_sensitivity.csv",
            "conditional_granger_sensitivity": "conditional_granger_sensitivity.csv",
            "ardl_ecm_sensitivity": "ardl_ecm_sensitivity.csv",
            "local_projection_sensitivity": "local_projection_sensitivity.csv",
            "pcmci_sensitivity_links": "pcmci_sensitivity_links.csv",
            "pcmci_link_stability": "pcmci_link_stability.csv",
            "peak_breadth_difference": "peak_breadth_difference.csv",
            "nonlinear_pipeline_ranking": "nonlinear_pipeline_ranking.csv",
            "nonlinear_metrics": "nonlinear_metrics.csv",
            "nonlinear_statistical_tests": "nonlinear_statistical_tests.csv",
            "nonlinear_origin_audit": "nonlinear_origin_audit.csv",
            "nonlinear_source_audit": "nonlinear_source_audit.csv",
            "causal_ml_sensitivity": "causal_ml_sensitivity.csv",
            "causal_ml_source_audit": "causal_ml_source_audit.csv",
            "future_lead_placebos": "future_lead_placebos.csv",
            "future_lead_placebo_predictions": "future_lead_placebo_predictions.csv",
            "future_lead_placebo_fold_audit": "future_lead_placebo_fold_audit.csv",
        }
        output_paths = {
            key: output_dir / f"{version}_{suffixes[key]}" for key in frames
        }
        _write_frame_collection(frames, output_paths)
        feature_sets_path = output_dir / f"{version}_feature_sets.json"
        atomic_write_text(
            feature_sets_path,
            json.dumps(feature_sets, indent=2, sort_keys=True, default=json_default)
            + "\n",
        )

        full_scope = bool(
            args.frequency == "both"
            and not args.smoke_test
            and args.maximum_specifications is None
            and args.maximum_origins is None
            and not args.skip_dynamic_refits
            and not args.skip_future_lead_placebos
            and repetitions == configured_repetitions
            and seed == int(config["inference"]["random_seed"])
        )
        status = determine_run_status(
            args.smoke_test,
            args.maximum_specifications,
            args.maximum_origins,
            full_scope,
        )
        manifest_path = output_dir / f"{version}_manifest.json"
        manifest_outputs = {
            key: str(path) for key, path in output_paths.items()
        }
        manifest_outputs["feature_sets"] = str(feature_sets_path)
        manifest_outputs["manifest"] = str(manifest_path)
        output_hashes = {
            key: sha256_file(path) for key, path in output_paths.items()
        }
        output_hashes["feature_sets"] = sha256_file(feature_sets_path)
        elapsed_seconds = time.perf_counter() - started
        manifest = {
            "version": version,
            "status": status,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "publication_scope_complete": full_scope,
            "publication_eligible_without_human_validation": False,
            "causal_claim_allowed": False,
            "primary_frequency": "weekly",
            "monthly_role": "robustness",
            "inputs": input_hashes,
            "upstream_manifests": upstream_audit.to_dict(orient="records"),
            "arguments": {
                "frequency": args.frequency,
                "bootstrap_repetitions": repetitions,
                "seed": seed,
                "resume": args.resume,
                "smoke_test": args.smoke_test,
                "maximum_specifications": maximum_specifications,
                "maximum_origins": maximum_origins,
                "skip_dynamic_refits": args.skip_dynamic_refits,
                "skip_future_lead_placebos": args.skip_future_lead_placebos,
            },
            "run_signature": run_signature,
            "checkpoint_directory": str(checkpoint_dir),
            "design": {
                "frozen_common_article_cohort": expected_articles,
                "model_aliases": aliases,
                "score_scale": [0, 4],
                "reference_peak": "median_over_three_llms_of_overall_disruption_max",
                "reference_breadth": "equal_mean_of_seven_dimension_medians",
                "monthly_constructions": list(MONTHLY_CONSTRUCTIONS),
                "matched_forecast_origins_and_targets": True,
                "loss_difference": "baseline_loss_minus_variant_loss",
                "positive_loss_difference_favors": "variant_challenger",
                "forecast_estimator": "ridge_fixed_alpha",
                "nonlinear_algorithms_consolidated_not_refit": [
                    *config["forecast"]["extended_algorithms"],
                ],
                "nonlinear_evidence_role": "secondary_predictive_algorithm_robustness",
                "dynamic_effective_samples_matched_across_lag_specs": False,
                "dynamic_sample_caveat": (
                    "lag specifications can have different effective starts; "
                    "sample dates and n are retained for every estimate"
                ),
                "pcmci_variable_system": "full_available_four_variable_system",
                "future_treatment_lead": "horizon_plus_one_with_larger_embargo",
                "operational_regime_split": config["regime_sensitivity"],
            },
            "scope": {
                "selected_frequencies": selected_frequencies,
                "full_computational_scope": full_scope,
                "canonical_bootstrap_repetitions_executed": (
                    repetitions == configured_repetitions
                ),
                "canonical_random_seed_used": (
                    seed == int(config["inference"]["random_seed"])
                ),
                "forecast_specification_counts": {
                    frequency: len(selected_specifications[frequency])
                    for frequency in selected_frequencies
                },
                "index_variant_count_executed": len(variants),
                "index_variant_count_requested": len(requested_variants),
                "unavailable_or_ineligible_checks_are_explicit": True,
                "permuted_dates_role": (
                    "descriptive_randomization_only_not_leakage_safe_oos_forecast"
                ),
            },
            "inference": {
                **config["inference"],
                "bootstrap_repetitions_executed": repetitions,
                "raw_and_adjusted_p_values_reported": True,
            },
            "interpretation_guardrails": {
                "forecast": "predictive_association_not_causality",
                "granger": "predictive_precedence_not_structural_causality",
                "pcmci": "conditional_temporal_dependency_not_causal_graph",
                "causal_ml": (
                    "orthogonalized_predictive_effect_under_unverified_"
                    "conditional_exchangeability"
                ),
                "future_lead_placebo": (
                    "falsification_warning_if_significant_not_identification_"
                    "proof_if_null"
                ),
            },
            "audit": {
                "reference_reconstruction_all_passed": bool(
                    reconstruction_audit["contract_passed"].all()
                ),
                "forecast_horizon_purge_all_passed": bool(
                    fit_audit["horizon_purge_passed"].all()
                ),
                "future_lead_embargo_all_passed": (
                    bool(
                        future_fold_audit[
                            "latest_endpoint_embargo_passed"
                        ]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .eq("true")
                        .all()
                    )
                    if not future_fold_audit.empty
                    else None
                ),
                "nonlinear_origin_sets_all_identical": True,
                "adjusted_significant_primary_ablation_tests": int(
                    (
                        ablation_tests["primary_family"].astype(bool)
                        & ablation_tests["reject_adjusted_at_alpha"].astype(bool)
                    ).sum()
                )
                if not ablation_tests.empty
                else 0,
                "adjusted_significant_exploratory_ablation_tests": int(
                    (
                        ~ablation_tests["primary_family"].astype(bool)
                        & ablation_tests["reject_adjusted_at_alpha"].astype(bool)
                    ).sum()
                )
                if not ablation_tests.empty
                else 0,
                "future_lead_fdr_falsification_flags": int(
                    future_estimates["falsification_flag_fdr"].sum()
                )
                if not future_estimates.empty
                else 0,
            },
            "outputs": manifest_outputs,
            "output_sha256": output_hashes,
            "output_row_counts": {
                key: len(frame) for key, frame in frames.items()
            },
            "log": str(log_path),
            "runtime_seconds": elapsed_seconds,
            "software": software_versions,
            "code": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        }
        atomic_write_text(
            manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True, default=json_default)
            + "\n",
        )
        logger.info(
            "stage_complete status=%s full_scope=%s outputs=%s runtime=%.3fs "
            "manifest=%s",
            status,
            full_scope,
            len(frames) + 2,
            elapsed_seconds,
            manifest_path,
        )
        return 0
    except Exception:
        logger.exception("stage_failed version=%s", version)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
