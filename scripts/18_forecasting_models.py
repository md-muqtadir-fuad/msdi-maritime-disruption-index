"""Leakage-safe extended weekly and monthly SCFI forecasts.

This stage extends the frozen common-cohort panels produced by 07b and 07c.
A Gaussian identity-link GLM, random forests, XGBoost, a small PyTorch LSTM,
and Prophet all predict the same direct target: the forward SCFI log change
known only after the forecast origin. Monthly results are primary; weekly
results are higher-frequency robustness evidence.

The normal run uses the fixed, pre-specified parameters in
``configs/nonlinear_forecasting.yaml``.  ``--tune`` is deliberately opt-in and
uses nested expanding-window validation.  At every outer or inner fit, an
endpoint s is admissible only when s + horizon <= fit origin.  Preprocessors
are refit inside that admissible training window.  Nothing in this script is a
structural causal estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/nonlinear_forecasting.yaml"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "configs/model_registry.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/nonlinear_forecasts"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs/logs"
DEFAULT_WEEKLY_PANEL = (
    PROJECT_ROOT
    / "data/06_model_ready/prediction/llm_specific_scfi_weekly_panel.csv"
)
DEFAULT_MONTHLY_PANEL = (
    PROJECT_ROOT
    / "data/06_model_ready/prediction/llm_specific_scfi_monthly_panel.csv"
)
DEFAULT_WEEKLY_MANIFEST = (
    PROJECT_ROOT / "outputs/llm_scfi_comparison/llm_scfi_v001_manifest.json"
)
DEFAULT_MONTHLY_MANIFEST = (
    PROJECT_ROOT
    / "outputs/llm_scfi_comparison/llm_scfi_monthly_v001_manifest.json"
)
DEFAULT_WEEKLY_FEATURES = (
    PROJECT_ROOT / "outputs/llm_scfi_comparison/llm_scfi_v001_feature_sets.json"
)
DEFAULT_MONTHLY_FEATURES = (
    PROJECT_ROOT
    / "outputs/llm_scfi_comparison/llm_scfi_monthly_v001_feature_sets.json"
)

ALGORITHMS = ("glm", "random_forest", "xgboost", "lstm", "prophet")
SHORT_ALGORITHM_NAMES = {
    "glm": "glm",
    "random_forest": "rf",
    "xgboost": "xgboost",
    "lstm": "lstm",
    "prophet": "prophet",
}
DEPENDENCY_PACKAGES = {
    "glm": "scikit-learn",
    "random_forest": "scikit-learn",
    "xgboost": "xgboost",
    "lstm": "torch",
    "prophet": "prophet",
}
PRIMARY_SIGNALS = ("overall_disruption_max", "dimension_mean7")
DIMENSION_SIGNALS = (
    "port_disruption_score",
    "global_container_movement_disruption_score",
    "freight_cost_disruption_score",
    "supply_disruption_score",
    "production_disruption_score",
    "covid19_related_disruption_score",
    "demand_disruption_score",
)
ALL_SIGNALS = (*PRIMARY_SIGNALS, *DIMENSION_SIGNALS)
SCORE_HYBRID_SOURCES = ("score_median", "score_mean")
MONTHLY_CONSTRUCTIONS = ("mean_safe_weekly", "adapted_pdf_sqrt_ema3")
WEEKLY_CONSTRUCTION = "weekly_log_volume"
PRICE_OVERLAP_SIGNAL = "freight_cost_disruption_score"


@dataclass(frozen=True)
class FrequencySpec:
    name: str
    panel_path: Path
    upstream_manifest_path: Path
    upstream_feature_sets_path: Path
    date_column: str
    complete_column: str
    horizons: tuple[int, ...]
    target_template: str
    level_template: str
    current_level_column: str
    horizon_unit: str
    minimum_training_rows: int
    refit_every: int
    sequence_length: int
    primary: bool

    def target_column(self, horizon: int) -> str:
        return self.target_template.format(horizon=horizon)

    def level_column(self, horizon: int) -> str:
        return self.level_template.format(horizon=horizon)

    @property
    def constructions(self) -> tuple[str, ...]:
        if self.name == "weekly":
            return (WEEKLY_CONSTRUCTION,)
        return MONTHLY_CONSTRUCTIONS


@dataclass(frozen=True)
class ForecastConfiguration:
    frequency: str
    index_construction: str
    source_type: str
    score_source: str
    signal_name: str
    hybrid_level: str
    upstream_feature_key: str

    @property
    def is_baseline(self) -> bool:
        return self.source_type == "baseline"

    def model_id(self, algorithm: str) -> str:
        short = SHORT_ALGORITHM_NAMES[algorithm]
        if self.is_baseline:
            return (
                f"{short}__{self.frequency}__{self.index_construction}__"
                "ar_brent_coverage"
            )
        source_label = "llm" if self.source_type == "llm" else "score_hybrid"
        return (
            f"{short}__{self.frequency}__{self.index_construction}__"
            f"{source_label}__{self.score_source}__{self.signal_name}"
        )


@dataclass
class TabularArtifact:
    estimator: Any

    def predict(
        self,
        panel: pd.DataFrame,
        positions: Sequence[int],
        features: Sequence[str],
        sequence_length: int,
    ) -> np.ndarray:
        del sequence_length
        return np.asarray(
            self.estimator.predict(panel.loc[list(positions), list(features)]),
            dtype=float,
        )


@dataclass
class LstmArtifact:
    model: Any
    imputer: SimpleImputer
    scaler: StandardScaler
    target_mean: float
    target_scale: float

    def predict(
        self,
        panel: pd.DataFrame,
        positions: Sequence[int],
        features: Sequence[str],
        sequence_length: int,
    ) -> np.ndarray:
        import torch

        raw = build_sequence_array(panel, positions, features, sequence_length)
        rows, steps, columns = raw.shape
        flat = raw.reshape(-1, columns)
        transformed = self.scaler.transform(self.imputer.transform(flat))
        tensor = torch.as_tensor(
            transformed.reshape(rows, steps, columns), dtype=torch.float32
        )
        self.model.eval()
        with torch.no_grad():
            scaled = self.model(tensor).cpu().numpy().astype(float)
        return scaled * self.target_scale + self.target_mean


@dataclass
class ProphetArtifact:
    model: Any
    imputer: SimpleImputer
    active_features: list[str]

    def predict(
        self,
        panel: pd.DataFrame,
        positions: Sequence[int],
        features: Sequence[str],
        sequence_length: int,
    ) -> np.ndarray:
        del sequence_length
        positions = list(positions)
        values = self.imputer.transform(panel.loc[positions, list(features)])
        transformed = pd.DataFrame(values, columns=list(features))
        future = pd.DataFrame({"ds": panel.loc[positions, "__forecast_date"].to_numpy()})
        for feature in self.active_features:
            future[feature] = transformed[feature].to_numpy()
        return self.model.predict(future)["yhat"].to_numpy(dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe Gaussian GLM, RF, XGBoost, LSTM, and Prophet forecasts "
            "for weekly and monthly SCFI. Fixed parameters are used unless "
            "--tune is set."
        )
    )
    parser.add_argument(
        "--frequency", choices=("weekly", "monthly", "both"), default="both"
    )
    parser.add_argument(
        "--models",
        default=",".join(ALGORITHMS),
        help="Comma-separated subset of glm,random_forest,xgboost,lstm,prophet.",
    )
    parser.add_argument("--signals", choices=("primary", "all"), default="primary")
    parser.add_argument(
        "--tune",
        action="store_true",
        help=(
            "Enable expensive nested purged expanding-window tuning. Off by "
            "default; the fixed pre-specified parameters are otherwise used."
        ),
    )
    parser.add_argument(
        "--missing-dependency",
        choices=("error", "skip"),
        default="error",
        help=(
            "Fail closed by default. 'skip' is an explicit debug convenience "
            "and makes the run ineligible for publication."
        ),
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse matching per-cell checkpoints from an interrupted run.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Checkpoint root (default: <output-dir>/<version>_checkpoints).",
    )
    parser.add_argument("--max-configurations", type=int)
    parser.add_argument("--max-origins-per-horizon", type=int)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    parser.add_argument("--weekly-panel", type=Path, default=DEFAULT_WEEKLY_PANEL)
    parser.add_argument("--monthly-panel", type=Path, default=DEFAULT_MONTHLY_PANEL)
    parser.add_argument(
        "--weekly-manifest", type=Path, default=DEFAULT_WEEKLY_MANIFEST
    )
    parser.add_argument(
        "--monthly-manifest", type=Path, default=DEFAULT_MONTHLY_MANIFEST
    )
    parser.add_argument(
        "--weekly-feature-sets", type=Path, default=DEFAULT_WEEKLY_FEATURES
    )
    parser.add_argument(
        "--monthly-feature-sets", type=Path, default=DEFAULT_MONTHLY_FEATURES
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--version", default="nonlinear_scfi_v001")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


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


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def json_default(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def configure_logging(log_dir: Path, version: str) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"07d_nonlinear_forecasts_{version}.log"
    logger = logging.getLogger("nonlinear_forecasts")
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


def assert_project_venv() -> Path:
    expected = (PROJECT_ROOT / ".venv").resolve()
    actual = Path(sys.prefix).resolve()
    if actual != expected:
        raise RuntimeError(
            "This project must run from its .venv. Use "
            f"{PROJECT_ROOT / '.venv/Scripts/python.exe'} "
            "scripts/18_forecasting_models.py ...; "
            f"current sys.prefix is {actual}."
        )
    return Path(sys.executable).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    resolved = require_file(path, "Configuration")
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {resolved}")
    return data


def frequency_specs(
    config: dict[str, Any],
    *,
    weekly_panel: Path = DEFAULT_WEEKLY_PANEL,
    monthly_panel: Path = DEFAULT_MONTHLY_PANEL,
    weekly_manifest: Path = DEFAULT_WEEKLY_MANIFEST,
    monthly_manifest: Path = DEFAULT_MONTHLY_MANIFEST,
    weekly_feature_sets: Path = DEFAULT_WEEKLY_FEATURES,
    monthly_feature_sets: Path = DEFAULT_MONTHLY_FEATURES,
) -> dict[str, FrequencySpec]:
    design = config.get("design", {}).get("frequency", {})
    try:
        weekly = design["weekly"]
        monthly = design["monthly"]
    except KeyError as exc:
        raise ValueError(f"Missing frequency design in configuration: {exc}") from exc
    return {
        "weekly": FrequencySpec(
            name="weekly",
            panel_path=weekly_panel,
            upstream_manifest_path=weekly_manifest,
            upstream_feature_sets_path=weekly_feature_sets,
            date_column="observation_date",
            complete_column="llm_comparison_complete_case",
            horizons=(1, 2, 4, 8, 12),
            target_template="target_scfi_log_change_h{horizon}w",
            level_template="target_scfi_level_h{horizon}w",
            current_level_column="scfi",
            horizon_unit="weeks",
            minimum_training_rows=int(weekly["minimum_training_rows"]),
            refit_every=int(weekly["refit_every"]),
            sequence_length=int(weekly["sequence_length"]),
            primary=str(weekly.get("role", "")).lower() == "primary",
        ),
        "monthly": FrequencySpec(
            name="monthly",
            panel_path=monthly_panel,
            upstream_manifest_path=monthly_manifest,
            upstream_feature_sets_path=monthly_feature_sets,
            date_column="month_ending",
            complete_column="llm_monthly_comparison_complete_case",
            horizons=(1, 3, 6),
            target_template="target_scfi_monthly_log_change_h{horizon}m",
            level_template="target_scfi_monthly_mean_h{horizon}m",
            current_level_column="scfi_monthly_mean",
            horizon_unit="months",
            minimum_training_rows=int(monthly["minimum_training_rows"]),
            refit_every=int(monthly["refit_every"]),
            sequence_length=int(monthly["sequence_length"]),
            primary=str(monthly.get("role", "")).lower() == "primary",
        ),
    }


def parse_model_selection(value: str) -> list[str]:
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not requested:
        raise ValueError("At least one forecasting model must be selected")
    unknown = sorted(set(requested) - set(ALGORITHMS))
    if unknown:
        raise ValueError(f"Unknown forecasting model(s): {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError("Forecasting model names must be unique")
    return requested


def dependency_audit(
    requested: Sequence[str], missing_policy: str
) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    selected: list[str] = []
    missing: list[str] = []
    for algorithm in requested:
        package = DEPENDENCY_PACKAGES[algorithm]
        import_name = "sklearn" if package == "scikit-learn" else package
        discoverable = importlib.util.find_spec(import_name) is not None
        import_error: str | None = None
        if discoverable:
            try:
                probe_algorithm_dependency(algorithm)
            except Exception as exc:  # pragma: no cover - environment-specific failure
                import_error = f"{type(exc).__name__}: {exc}"
        available = discoverable and import_error is None
        version: str | None = None
        if available:
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = "installed_version_unknown"
            selected.append(algorithm)
            action = "selected"
        else:
            missing.append(algorithm)
            action = "skipped_explicitly" if missing_policy == "skip" else "error"
        rows.append(
            {
                "algorithm": algorithm,
                "package": package,
                "discoverable": discoverable,
                "available": available,
                "version": version,
                "import_probe_error": import_error,
                "missing_dependency_policy": missing_policy,
                "action": action,
            }
        )
    audit = pd.DataFrame(rows)
    if missing and missing_policy == "error":
        packages = sorted({DEPENDENCY_PACKAGES[item] for item in missing})
        raise RuntimeError(
            f"Missing requested forecasting dependencies for {missing}: {packages}. "
            "Install them into the project environment with "
            ".venv\\Scripts\\python.exe -m pip install -r "
            "requirements-forecasting.txt, then rerun."
        )
    if not selected:
        raise RuntimeError("No requested forecasting model is available")
    return selected, audit


def probe_algorithm_dependency(algorithm: str) -> None:
    """Import the actual adapter and initialize lightweight backend state."""

    if algorithm == "glm":
        TweedieRegressor(
            power=0,
            link="identity",
            alpha=0.01,
            solver="lbfgs",
            max_iter=5_000,
            tol=1e-8,
        )
    elif algorithm == "random_forest":
        RandomForestRegressor(n_estimators=1, n_jobs=1)
    elif algorithm == "xgboost":
        from xgboost import XGBRegressor

        XGBRegressor(n_estimators=1, n_jobs=1)
    elif algorithm == "lstm":
        import torch

        torch.nn.LSTM(input_size=1, hidden_size=1, batch_first=True)
    elif algorithm == "prophet":
        from prophet import Prophet

        # Constructor initialization verifies that Prophet can find a Stan
        # backend; merely finding the Python package is not a sufficient probe.
        Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            uncertainty_samples=0,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def load_model_aliases(path: Path) -> list[str]:
    registry = load_yaml(path)
    rows = registry.get("active_scoring_models", [])
    aliases = [str(row.get("alias", "")).strip() for row in rows]
    if len(aliases) != 4 or any(not alias for alias in aliases):
        raise ValueError("model_registry.yaml must define exactly four active aliases")
    if len(set(aliases)) != 4:
        raise ValueError("Active scoring aliases must be unique")
    return aliases


def resolve_parameter_candidates(
    config: dict[str, Any], algorithm: str, tune: bool
) -> list[dict[str, Any]]:
    model_config = config.get("models", {}).get(algorithm)
    if not isinstance(model_config, dict):
        raise ValueError(f"No configuration found for {algorithm}")
    key = "candidates" if tune else "fixed"
    values = model_config.get(key)
    if tune:
        if not isinstance(values, list) or not values:
            raise ValueError(f"{algorithm} tuning candidates must be a nonempty list")
        if not all(isinstance(item, dict) for item in values):
            raise ValueError(f"Every {algorithm} tuning candidate must be a mapping")
        return [dict(item) for item in values]
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{algorithm} fixed parameters must be a nonempty mapping")
    return [dict(values)]


def smoke_parameters(algorithm: str, parameters: dict[str, Any]) -> dict[str, Any]:
    result = dict(parameters)
    if algorithm == "glm":
        # The Gaussian GLM is already cheap; retain the exact fixed specification.
        pass
    elif algorithm == "random_forest":
        result.update(n_estimators=20, n_jobs=1)
    elif algorithm == "xgboost":
        result.update(n_estimators=20, n_jobs=1)
    elif algorithm == "lstm":
        result.update(epochs=2, hidden_size=min(int(result["hidden_size"]), 8))
    elif algorithm == "prophet":
        result.update(uncertainty_samples=0)
    return result


def parse_boolean(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    unexpected = sorted(set(normalized.dropna()) - set(mapping))
    if unexpected:
        raise ValueError(f"{label} has non-boolean values: {unexpected[:10]}")
    return normalized.map(mapping).fillna(False).astype(bool)


def validate_upstream_manifest(
    spec: FrequencySpec, aliases: Sequence[str]
) -> dict[str, Any]:
    path = require_file(spec.upstream_manifest_path, f"{spec.name} upstream manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    status = str(manifest.get("status", ""))
    if not status:
        raise ValueError(f"{path} has no status")
    ineligible_markers = ("debug", "not_for_publication", "incomplete", "failed")
    if any(marker in status.lower() for marker in ineligible_markers):
        raise ValueError(
            f"{spec.name} upstream manifest is not publication-eligible: "
            f"status={status!r}"
        )
    design_aliases = manifest.get("design", {}).get("model_aliases")
    if list(design_aliases or []) != list(aliases):
        raise ValueError(
            f"{spec.name} upstream model aliases {design_aliases} do not match "
            f"the registry {list(aliases)}"
        )
    cohort = manifest.get("design", {}).get(
        "expected_common_articles",
        manifest.get("design", {}).get("frozen_common_articles"),
    )
    if cohort is not None and int(cohort) != 6480:
        raise ValueError(
            f"{spec.name} manifest is not the frozen 6,480-article cohort: {cohort}"
        )
    output_key = "aligned_panel" if spec.name == "weekly" else "monthly_panel"
    recorded_panel = manifest.get("outputs", {}).get(output_key)
    if not recorded_panel:
        raise ValueError(f"{path} does not record outputs.{output_key}")
    if Path(recorded_panel).resolve() != spec.panel_path.resolve():
        raise ValueError(
            f"{spec.name} panel does not match its upstream manifest: "
            f"{spec.panel_path.resolve()} != {Path(recorded_panel).resolve()}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": status,
        "frozen_common_articles": 6480,
        "panel_path_verified": True,
    }


def validate_date_cadence(dates: pd.Series, frequency: str) -> None:
    if dates.isna().any():
        raise ValueError(f"{frequency} forecast dates contain missing values")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"{frequency} forecast dates must be unique and increasing")
    if frequency == "weekly":
        differences = dates.diff().dropna().dt.days
        if not differences.eq(7).all():
            raise ValueError("Weekly panel must have an uninterrupted seven-day cadence")
    else:
        periods = dates.dt.to_period("M").astype(int)
        if not periods.diff().dropna().eq(1).all():
            raise ValueError("Monthly panel must contain consecutive calendar months")
        if not dates.dt.is_month_end.all():
            raise ValueError("Monthly panel dates must be calendar month ends")


def load_panel(spec: FrequencySpec) -> pd.DataFrame:
    path = require_file(spec.panel_path, f"{spec.name} model panel")
    panel = pd.read_csv(path, low_memory=False)
    required = {
        spec.date_column,
        spec.complete_column,
        spec.current_level_column,
        *[spec.target_column(horizon) for horizon in spec.horizons],
        *[spec.level_column(horizon) for horizon in spec.horizons],
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"{spec.name} panel lacks required columns: {missing}")
    panel[spec.date_column] = pd.to_datetime(panel[spec.date_column], errors="raise")
    panel = panel.sort_values(spec.date_column).reset_index(drop=True)
    validate_date_cadence(panel[spec.date_column], spec.name)
    panel[spec.complete_column] = parse_boolean(
        panel[spec.complete_column], spec.complete_column
    )
    panel = panel.copy()
    panel["__forecast_date"] = panel[spec.date_column]
    numeric = [
        spec.current_level_column,
        *[spec.target_column(horizon) for horizon in spec.horizons],
        *[spec.level_column(horizon) for horizon in spec.horizons],
    ]
    for column in numeric:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    if int(panel[spec.complete_column].sum()) < spec.minimum_training_rows:
        raise ValueError(
            f"{spec.name} panel has too few complete rows for its design: "
            f"{int(panel[spec.complete_column].sum())}"
        )
    return panel


def load_feature_sets(spec: FrequencySpec) -> dict[str, list[str]]:
    path = require_file(spec.upstream_feature_sets_path, f"{spec.name} feature sets")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a feature-set mapping in {path}")
    normalized: dict[str, list[str]] = {}
    for key, values in data.items():
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"Feature set {key} is not a list of column names")
        forbidden = [
            item
            for item in values
            if item.startswith("target_") or "_lead_" in item.lower()
        ]
        if forbidden:
            raise AssertionError(f"Target/lead leakage in feature set {key}: {forbidden}")
        normalized[str(key)] = list(values)
    if "ridge__ar_brent_coverage" not in normalized:
        raise ValueError(f"{path} lacks ridge__ar_brent_coverage")
    return normalized


def upstream_feature_key(
    frequency: str,
    construction: str,
    source_type: str,
    score_source: str,
    signal_name: str,
) -> str:
    source_label = "llm" if source_type == "llm" else "score_hybrid"
    if frequency == "weekly":
        return f"ridge__{source_label}__{score_source}__{signal_name}"
    return (
        f"ridge__monthly__{construction}__{source_label}__"
        f"{score_source}__{signal_name}"
    )


def build_configurations(
    spec: FrequencySpec, aliases: Sequence[str], signal_scope: str
) -> list[ForecastConfiguration]:
    signals = PRIMARY_SIGNALS if signal_scope == "primary" else ALL_SIGNALS
    configurations: list[ForecastConfiguration] = []
    for construction in spec.constructions:
        for source in [*aliases, *SCORE_HYBRID_SOURCES]:
            source_type = "llm" if source in aliases else "score_hybrid"
            for signal in signals:
                configurations.append(
                    ForecastConfiguration(
                        frequency=spec.name,
                        index_construction=construction,
                        source_type=source_type,
                        score_source=source,
                        signal_name=signal,
                        hybrid_level="none" if source_type == "llm" else "score",
                        upstream_feature_key=upstream_feature_key(
                            spec.name,
                            construction,
                            source_type,
                            source,
                            signal,
                        ),
                    )
                )
    if len({(item.index_construction, item.upstream_feature_key) for item in configurations}) != len(
        configurations
    ):
        raise AssertionError(f"{spec.name} forecasting configurations are not unique")
    return configurations


def baseline_configuration(spec: FrequencySpec, construction: str) -> ForecastConfiguration:
    return ForecastConfiguration(
        frequency=spec.name,
        index_construction=construction,
        source_type="baseline",
        score_source="none",
        signal_name="none",
        hybrid_level="none",
        upstream_feature_key="ridge__ar_brent_coverage",
    )


def algorithm_features(algorithm: str, upstream_features: Sequence[str]) -> list[str]:
    features = list(upstream_features)
    if algorithm in {"lstm", "prophet"}:
        # Temporal dependence is supplied by the sequence for LSTM and by the
        # trend/seasonality for Prophet.  Retain contemporaneous, backward-change,
        # coverage, and raw score fields; remove duplicated engineered lag paths.
        features = [
            feature
            for feature in features
            if "_lag_" not in feature
            and "_rolling_" not in feature
            and "_acceleration_" not in feature
        ]
    if algorithm == "prophet":
        # Prophet supplies calendar seasonality directly.  Duplicating the sine/
        # cosine fields creates unnecessary collinearity in the small monthly fit.
        features = [feature for feature in features if not feature.startswith("calendar_")]
    forbidden = [
        feature
        for feature in features
        if feature.startswith("target_") or "_lead_" in feature.lower()
    ]
    if forbidden:
        raise AssertionError(f"Target/lead leakage in {algorithm}: {forbidden}")
    if not features:
        raise ValueError(f"No usable features remain for {algorithm}")
    return features


def validate_selected_features(
    panel: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    configurations: Sequence[ForecastConfiguration],
    algorithms: Sequence[str],
) -> dict[str, dict[str, list[str]]]:
    available = set(panel.columns)
    result: dict[str, dict[str, list[str]]] = {}
    for algorithm in algorithms:
        algorithm_map: dict[str, list[str]] = {}
        for configuration in configurations:
            key = configuration.upstream_feature_key
            if key not in feature_sets:
                raise ValueError(f"Upstream feature set is missing {key}")
            features = algorithm_features(algorithm, feature_sets[key])
            missing = sorted(set(features) - available)
            if missing:
                raise ValueError(
                    f"{algorithm}/{configuration.model_id(algorithm)} lacks panel "
                    f"columns: {missing}"
                )
            for feature in features:
                panel[feature] = pd.to_numeric(panel[feature], errors="coerce")
            algorithm_map[configuration.model_id(algorithm)] = features
        result[algorithm] = algorithm_map
    return result


def training_positions(
    panel: pd.DataFrame,
    origin_position: int,
    horizon: int,
    target_column: str,
    complete_column: str,
) -> np.ndarray:
    maximum_position = int(origin_position) - int(horizon)
    if maximum_position < 0:
        return np.array([], dtype=int)
    candidates = np.arange(maximum_position + 1, dtype=int)
    complete = panel.loc[candidates, complete_column].astype(bool).to_numpy()
    target_available = panel.loc[candidates, target_column].notna().to_numpy()
    selected = candidates[complete & target_available]
    if len(selected) and int(selected[-1]) + int(horizon) > int(origin_position):
        raise AssertionError("Training-label purge failed")
    return selected


def eligible_origins(
    panel: pd.DataFrame,
    spec: FrequencySpec,
    horizon: int,
    maximum_origins: int | None = None,
) -> list[int]:
    target_column = spec.target_column(horizon)
    positions: list[int] = []
    for position in panel.index:
        if position + horizon >= len(panel):
            continue
        if not bool(panel.at[position, spec.complete_column]):
            continue
        if pd.isna(panel.at[position, target_column]):
            continue
        train = training_positions(
            panel, position, horizon, target_column, spec.complete_column
        )
        if len(train) >= spec.minimum_training_rows:
            positions.append(int(position))
    if maximum_origins is not None:
        positions = positions[:maximum_origins]
    return positions


def build_inner_folds(
    outer_training_positions: Sequence[int],
    outer_fit_position: int,
    horizon: int,
    minimum_training_rows: int,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic expanding validation folds inside one outer fit.

    The validation endpoints are drawn from the latter half of the already
    admissible outer endpoints.  For every fold beginning at v, the inner train
    is further purged to s + horizon <= v.  This helper deliberately operates
    on positions instead of row counts so non-contiguous complete cases remain
    leakage-safe.
    """

    outer = np.asarray(sorted(set(int(item) for item in outer_training_positions)), dtype=int)
    if len(outer) == 0 or n_splits < 1:
        return []
    if int(outer[-1]) + int(horizon) > int(outer_fit_position):
        raise AssertionError("Outer training positions are not horizon-purged")
    first_validation_index = max(int(minimum_training_rows), len(outer) // 2)
    validation_pool = outer[first_validation_index:]
    if len(validation_pool) < n_splits:
        return []
    validation_blocks = [block for block in np.array_split(validation_pool, n_splits) if len(block)]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for validation in validation_blocks:
        validation_start = int(validation[0])
        inner_train = outer[outer + int(horizon) <= validation_start]
        if len(inner_train) < minimum_training_rows:
            continue
        if int(inner_train[-1]) + int(horizon) > validation_start:
            raise AssertionError("Inner training-label purge failed")
        if int(validation[-1]) + int(horizon) > int(outer_fit_position):
            raise AssertionError("Inner validation target was unavailable at outer fit")
        folds.append((inner_train.copy(), validation.copy()))
    return folds


def build_sequence_array(
    panel: pd.DataFrame,
    positions: Sequence[int],
    features: Sequence[str],
    sequence_length: int,
) -> np.ndarray:
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    arrays: list[np.ndarray] = []
    for position in positions:
        position = int(position)
        start = position - sequence_length + 1
        if start < 0:
            raise ValueError(
                f"Position {position} has no full {sequence_length}-row history"
            )
        window = panel.loc[start:position, list(features)].to_numpy(dtype=float)
        if window.shape != (sequence_length, len(features)):
            raise AssertionError("LSTM sequence did not end exactly at its origin")
        arrays.append(window)
    if not arrays:
        return np.empty((0, sequence_length, len(features)), dtype=float)
    return np.stack(arrays)


def parameter_json(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"), default=json_default)


def parameter_hash(parameters: dict[str, Any]) -> str:
    return hashlib.sha256(parameter_json(parameters).encode("utf-8")).hexdigest()[:16]


def fit_tabular_predictor(
    algorithm: str,
    panel: pd.DataFrame,
    training: np.ndarray,
    features: Sequence[str],
    target_column: str,
    parameters: dict[str, Any],
    fit_seed: int,
) -> tuple[TabularArtifact, dict[str, Any]]:
    estimator_parameters = dict(parameters)
    if algorithm == "glm":
        # TweedieRegressor is deterministic and does not accept random_state.
        estimator = TweedieRegressor(**estimator_parameters)
    elif algorithm == "random_forest":
        estimator_parameters["random_state"] = int(fit_seed)
        estimator = RandomForestRegressor(**estimator_parameters)
    elif algorithm == "xgboost":
        from xgboost import XGBRegressor

        estimator_parameters["random_state"] = int(fit_seed)
        estimator = XGBRegressor(**estimator_parameters)
    else:
        raise ValueError(f"Unsupported tabular algorithm: {algorithm}")
    steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(strategy="median", keep_empty_features=True),
        )
    ]
    if algorithm == "glm":
        steps.append(("scaler", StandardScaler()))
    steps.append(("estimator", estimator))
    pipeline = Pipeline(steps)
    pipeline.fit(
        panel.loc[training, list(features)],
        panel.loc[training, target_column].to_numpy(dtype=float),
    )
    return TabularArtifact(pipeline), {
        "effective_training_rows": int(len(training)),
        "effective_training_first_position": int(training[0]),
        "effective_training_last_position": int(training[-1]),
        "training_loss": math.nan,
        "epochs_completed": math.nan,
    }


def fit_lstm_predictor(
    panel: pd.DataFrame,
    training: np.ndarray,
    features: Sequence[str],
    target_column: str,
    sequence_length: int,
    parameters: dict[str, Any],
    fit_seed: int,
) -> tuple[LstmArtifact, dict[str, Any]]:
    import torch
    from torch import nn
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(fit_seed)
    np.random.seed(fit_seed)
    torch.manual_seed(fit_seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)

    effective_training = training[training >= sequence_length - 1]
    if len(effective_training) < 12:
        raise ValueError(
            "Too few full LSTM sequences remain after enforcing the lookback: "
            f"{len(effective_training)}"
        )
    sequences = build_sequence_array(
        panel, effective_training, features, sequence_length
    )
    rows, steps, columns = sequences.shape
    flattened = sequences.reshape(-1, columns)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputed = imputer.fit_transform(flattened)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed).reshape(rows, steps, columns)
    target = panel.loc[effective_training, target_column].to_numpy(dtype=float)
    target_mean = float(target.mean())
    target_scale = float(target.std(ddof=0))
    if not np.isfinite(target_scale) or target_scale <= 1e-12:
        target_scale = 1.0
    scaled_target = (target - target_mean) / target_scale

    class DirectLstm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            num_layers = int(parameters["num_layers"])
            dropout = float(parameters.get("dropout", 0.0)) if num_layers > 1 else 0.0
            self.recurrent = nn.LSTM(
                input_size=columns,
                hidden_size=int(parameters["hidden_size"]),
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
            )
            self.output = nn.Linear(int(parameters["hidden_size"]), 1)

        def forward(self, values: Any) -> Any:
            recurrent, _ = self.recurrent(values)
            return self.output(recurrent[:, -1, :]).squeeze(-1)

    model = DirectLstm()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters.get("weight_decay", 0.0)),
    )
    loss_function = nn.MSELoss()
    dataset = TensorDataset(
        torch.as_tensor(scaled, dtype=torch.float32),
        torch.as_tensor(scaled_target, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(int(parameters["batch_size"]), len(dataset)),
        shuffle=False,
        num_workers=0,
    )
    epochs = int(parameters["epochs"])
    final_loss = math.nan
    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        total_rows = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = loss_function(prediction, batch_y)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_x)
            total_rows += len(batch_x)
        final_loss = total_loss / max(total_rows, 1)
    artifact = LstmArtifact(
        model=model,
        imputer=imputer,
        scaler=scaler,
        target_mean=target_mean,
        target_scale=target_scale,
    )
    return artifact, {
        "effective_training_rows": int(rows),
        "effective_training_first_position": int(effective_training[0]),
        "effective_training_last_position": int(effective_training[-1]),
        "training_loss": final_loss,
        "epochs_completed": epochs,
    }


def fit_prophet_predictor(
    panel: pd.DataFrame,
    training: np.ndarray,
    features: Sequence[str],
    target_column: str,
    parameters: dict[str, Any],
) -> tuple[ProphetArtifact, dict[str, Any]]:
    from prophet import Prophet

    parameters = dict(parameters)
    regressor_prior_scale = float(parameters.pop("regressor_prior_scale", 0.1))
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    values = imputer.fit_transform(panel.loc[training, list(features)])
    transformed = pd.DataFrame(values, columns=list(features))
    active_features = [
        feature
        for feature in features
        if float(np.nanstd(transformed[feature].to_numpy(dtype=float))) > 1e-12
    ]
    model = Prophet(**parameters)
    for feature in active_features:
        model.add_regressor(
            feature,
            prior_scale=regressor_prior_scale,
            standardize=True,
        )
    fit_frame = pd.DataFrame(
        {
            "ds": panel.loc[training, "__forecast_date"].to_numpy(),
            "y": panel.loc[training, target_column].to_numpy(dtype=float),
        }
    )
    for feature in active_features:
        fit_frame[feature] = transformed[feature].to_numpy()
    model.fit(fit_frame)
    return ProphetArtifact(model, imputer, active_features), {
        "effective_training_rows": int(len(training)),
        "effective_training_first_position": int(training[0]),
        "effective_training_last_position": int(training[-1]),
        "training_loss": math.nan,
        "epochs_completed": math.nan,
        "active_prophet_regressors": len(active_features),
    }


def fit_predictor(
    algorithm: str,
    panel: pd.DataFrame,
    training: np.ndarray,
    features: Sequence[str],
    target_column: str,
    sequence_length: int,
    parameters: dict[str, Any],
    fit_seed: int,
) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    if algorithm in {"glm", "random_forest", "xgboost"}:
        artifact, metadata = fit_tabular_predictor(
            algorithm,
            panel,
            training,
            features,
            target_column,
            parameters,
            fit_seed,
        )
    elif algorithm == "lstm":
        artifact, metadata = fit_lstm_predictor(
            panel,
            training,
            features,
            target_column,
            sequence_length,
            parameters,
            fit_seed,
        )
    elif algorithm == "prophet":
        artifact, metadata = fit_prophet_predictor(
            panel, training, features, target_column, parameters
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    metadata = dict(metadata)
    metadata["fit_runtime_seconds"] = time.perf_counter() - started
    return artifact, metadata


def select_parameters_nested(
    *,
    algorithm: str,
    candidates: Sequence[dict[str, Any]],
    panel: pd.DataFrame,
    spec: FrequencySpec,
    horizon: int,
    outer_training: np.ndarray,
    outer_fit_position: int,
    features: Sequence[str],
    target_column: str,
    seed: int,
    n_splits: int,
    smoke_test: bool,
    fit_id: str,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    minimum_inner = 52 if spec.name == "weekly" else 24
    folds = build_inner_folds(
        outer_training,
        outer_fit_position,
        horizon,
        minimum_inner,
        n_splits,
    )
    if len(folds) < n_splits:
        raise ValueError(
            f"Cannot form {n_splits} purged inner folds for {fit_id}; "
            f"formed {len(folds)}. Increase the outer training history."
        )
    audit_rows: list[dict[str, Any]] = []
    candidate_scores: list[float] = []
    for candidate_index, original_parameters in enumerate(candidates):
        parameters = (
            smoke_parameters(algorithm, original_parameters)
            if smoke_test
            else dict(original_parameters)
        )
        fold_squared_errors: list[np.ndarray] = []
        for fold_index, (inner_training, validation) in enumerate(folds):
            inner_seed = stable_seed(
                seed,
                spec.name,
                algorithm,
                horizon,
                outer_fit_position,
                "candidate",
                candidate_index,
                "fold",
                fold_index,
            )
            artifact, fit_metadata = fit_predictor(
                algorithm,
                panel,
                inner_training,
                features,
                target_column,
                spec.sequence_length,
                parameters,
                inner_seed,
            )
            predictions = artifact.predict(
                panel, validation, features, spec.sequence_length
            )
            actual = panel.loc[validation, target_column].to_numpy(dtype=float)
            squared_errors = (actual - predictions) ** 2
            fold_squared_errors.append(squared_errors)
            training_label_end_position = int(inner_training[-1]) + horizon
            audit_rows.append(
                {
                    "fit_id": fit_id,
                    "frequency": spec.name,
                    "algorithm": algorithm,
                    "horizon_steps": horizon,
                    "outer_fit_position": outer_fit_position,
                    "outer_fit_origin_date": panel.at[
                        outer_fit_position, spec.date_column
                    ],
                    "candidate_id": f"candidate_{candidate_index:02d}",
                    "candidate_index": candidate_index,
                    "parameters_json": parameter_json(parameters),
                    "parameters_hash": parameter_hash(parameters),
                    "fold": fold_index + 1,
                    "inner_training_rows": len(inner_training),
                    "inner_train_first_position": int(inner_training[0]),
                    "inner_train_last_position": int(inner_training[-1]),
                    "inner_train_first_date": panel.at[
                        int(inner_training[0]), spec.date_column
                    ],
                    "inner_train_last_feature_date": panel.at[
                        int(inner_training[-1]), spec.date_column
                    ],
                    "inner_train_last_label_date": panel.at[
                        training_label_end_position, spec.date_column
                    ],
                    "validation_rows": len(validation),
                    "validation_first_position": int(validation[0]),
                    "validation_last_position": int(validation[-1]),
                    "validation_first_date": panel.at[int(validation[0]), spec.date_column],
                    "validation_last_date": panel.at[int(validation[-1]), spec.date_column],
                    "purge_check_passed": bool(
                        int(inner_training[-1]) + horizon <= int(validation[0])
                    ),
                    "fold_rmse": float(np.sqrt(np.mean(squared_errors))),
                    "fit_runtime_seconds": fit_metadata["fit_runtime_seconds"],
                    "selected": False,
                }
            )
        candidate_scores.append(
            float(np.sqrt(np.mean(np.concatenate(fold_squared_errors))))
        )
    selected_index = min(range(len(candidate_scores)), key=lambda index: (candidate_scores[index], index))
    selected_parameters = (
        smoke_parameters(algorithm, candidates[selected_index])
        if smoke_test
        else dict(candidates[selected_index])
    )
    selected_id = f"candidate_{selected_index:02d}"
    for row in audit_rows:
        row["candidate_pooled_rmse"] = candidate_scores[int(row["candidate_index"])]
        row["selected"] = row["candidate_index"] == selected_index
    return selected_parameters, selected_id, audit_rows


def inference_tier(
    spec: FrequencySpec, algorithm: str, configuration: ForecastConfiguration
) -> str:
    if configuration.is_baseline:
        return "algorithm_matched_baseline"
    if configuration.signal_name not in PRIMARY_SIGNALS:
        return "exploratory_dimension"
    # The frozen Q1 blueprint pre-specifies the Ridge comparison as primary.
    # All estimators in this extended forecasting stage remain secondary
    # robustness paths, including the higher-frequency weekly specification.
    return "secondary_predictive_robustness"


def direct_forecast_row(
    *,
    panel: pd.DataFrame,
    spec: FrequencySpec,
    position: int,
    horizon: int,
    prediction: float,
    algorithm: str,
    configuration: ForecastConfiguration,
    feature_set: str,
    feature_count: int,
    training: np.ndarray,
    fit_position: int,
    parameters: dict[str, Any],
    candidate_id: str,
    fit_seed: int,
    tuning_enabled: bool,
) -> dict[str, Any]:
    target_column = spec.target_column(horizon)
    level_column = spec.level_column(horizon)
    actual = float(panel.at[position, target_column])
    current_level = float(panel.at[position, spec.current_level_column])
    actual_level = float(panel.at[position, level_column])
    predicted_level = current_level * math.exp(float(prediction))
    training_last = int(training[-1])
    label_end_position = training_last + horizon
    fit_origin = pd.Timestamp(panel.at[fit_position, spec.date_column])
    training_label_end = pd.Timestamp(panel.at[label_end_position, spec.date_column])
    if training_label_end > fit_origin:
        raise AssertionError("A forecast fit used a not-yet-observable target label")
    return {
        "frequency": spec.name,
        "frequency_is_primary": spec.primary,
        "algorithm": algorithm,
        "model_id": configuration.model_id(algorithm),
        "is_direct_model": True,
        "index_construction": configuration.index_construction,
        "source_type": configuration.source_type,
        "score_source": configuration.score_source,
        "signal_name": configuration.signal_name,
        "hybrid_level": configuration.hybrid_level,
        "inference_tier": inference_tier(spec, algorithm, configuration),
        "price_overlap_risk": configuration.signal_name == PRICE_OVERLAP_SIGNAL,
        "regime_specific_risk": configuration.signal_name
        == "covid19_related_disruption_score",
        "horizon_steps": int(horizon),
        "horizon_unit": spec.horizon_unit,
        "horizon_weeks": int(horizon) if spec.name == "weekly" else math.nan,
        "horizon_months": int(horizon) if spec.name == "monthly" else math.nan,
        "origin_position": int(position),
        "origin_date": panel.at[position, spec.date_column],
        "target_date": panel.at[position + horizon, spec.date_column],
        "actual_log_change": actual,
        "predicted_log_change": float(prediction),
        "forecast_error": actual - float(prediction),
        "absolute_error": abs(actual - float(prediction)),
        "squared_error": (actual - float(prediction)) ** 2,
        "current_scfi": current_level,
        "actual_scfi_level": actual_level,
        "predicted_scfi_level": predicted_level,
        "level_error": actual_level - predicted_level,
        "feature_set": feature_set,
        "feature_count": int(feature_count),
        "sequence_length": spec.sequence_length if algorithm == "lstm" else 1,
        "training_rows": int(len(training)),
        "training_first_position": int(training[0]),
        "training_last_position": training_last,
        "training_feature_end_date": panel.at[training_last, spec.date_column],
        "training_label_end_date": training_label_end,
        "model_fit_origin_position": int(fit_position),
        "model_fit_origin_date": fit_origin,
        "purge_gap_positions": int(fit_position - training_last),
        "tuning_enabled": bool(tuning_enabled),
        "candidate_id": candidate_id,
        "parameters_json": parameter_json(parameters),
        "parameters_hash": parameter_hash(parameters),
        "fit_seed": int(fit_seed),
        "causal_claim": False,
    }


def rolling_forecasts(
    *,
    panel: pd.DataFrame,
    spec: FrequencySpec,
    horizon: int,
    origins: Sequence[int],
    algorithm: str,
    configuration: ForecastConfiguration,
    features: Sequence[str],
    candidates: Sequence[dict[str, Any]],
    tune: bool,
    inner_splits: int,
    seed: int,
    smoke_test: bool,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target_column = spec.target_column(horizon)
    forecast_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    model_id = configuration.model_id(algorithm)
    for block_start in range(0, len(origins), spec.refit_every):
        block = list(origins[block_start : block_start + spec.refit_every])
        fit_position = int(block[0])
        training = training_positions(
            panel,
            fit_position,
            horizon,
            target_column,
            spec.complete_column,
        )
        if len(training) < spec.minimum_training_rows:
            raise AssertionError("An eligible origin has too little outer training data")
        fit_id = (
            f"{model_id}__h{horizon}__origin{fit_position}__"
            f"{pd.Timestamp(panel.at[fit_position, spec.date_column]).date()}"
        )
        if tune:
            parameters, candidate_id, nested_rows = select_parameters_nested(
                algorithm=algorithm,
                candidates=candidates,
                panel=panel,
                spec=spec,
                horizon=horizon,
                outer_training=training,
                outer_fit_position=fit_position,
                features=features,
                target_column=target_column,
                seed=seed,
                n_splits=inner_splits,
                smoke_test=smoke_test,
                fit_id=fit_id,
            )
            tuning_rows.extend(nested_rows)
        else:
            parameters = (
                smoke_parameters(algorithm, candidates[0])
                if smoke_test
                else dict(candidates[0])
            )
            candidate_id = "fixed"
        # Every score path in a matched comparison receives the same stochastic
        # seed.  The model/source/construction ID is intentionally excluded.
        fit_seed = stable_seed(
            seed,
            spec.name,
            algorithm,
            horizon,
            fit_position,
            candidate_id,
        )
        artifact, metadata = fit_predictor(
            algorithm,
            panel,
            training,
            features,
            target_column,
            spec.sequence_length,
            parameters,
            fit_seed,
        )
        predictions = artifact.predict(panel, block, features, spec.sequence_length)
        if len(predictions) != len(block) or not np.isfinite(predictions).all():
            raise ValueError(f"{fit_id} returned invalid predictions")
        training_last = int(training[-1])
        label_end_position = training_last + horizon
        label_end = pd.Timestamp(panel.at[label_end_position, spec.date_column])
        fit_origin = pd.Timestamp(panel.at[fit_position, spec.date_column])
        if label_end > fit_origin:
            raise AssertionError("Outer fit used a future target label")
        fit_rows.append(
            {
                "fit_id": fit_id,
                "frequency": spec.name,
                "frequency_is_primary": spec.primary,
                "algorithm": algorithm,
                "model_id": model_id,
                "index_construction": configuration.index_construction,
                "source_type": configuration.source_type,
                "score_source": configuration.score_source,
                "signal_name": configuration.signal_name,
                "inference_tier": inference_tier(spec, algorithm, configuration),
                "horizon_steps": horizon,
                "horizon_unit": spec.horizon_unit,
                "forecast_block_rows": len(block),
                "forecast_block_first_position": int(block[0]),
                "forecast_block_last_position": int(block[-1]),
                "forecast_block_first_date": panel.at[block[0], spec.date_column],
                "forecast_block_last_date": panel.at[block[-1], spec.date_column],
                "feature_set": configuration.upstream_feature_key,
                "feature_count": len(features),
                "feature_columns_json": json.dumps(list(features)),
                "sequence_length": spec.sequence_length if algorithm == "lstm" else 1,
                "training_rows": len(training),
                "effective_training_rows": metadata["effective_training_rows"],
                "effective_training_first_position": metadata[
                    "effective_training_first_position"
                ],
                "effective_training_last_position": metadata[
                    "effective_training_last_position"
                ],
                "training_first_position": int(training[0]),
                "training_last_position": training_last,
                "training_first_feature_date": panel.at[
                    int(training[0]), spec.date_column
                ],
                "training_feature_end_date": panel.at[training_last, spec.date_column],
                "training_label_end_position": label_end_position,
                "training_label_end_date": label_end,
                "model_fit_origin_position": fit_position,
                "model_fit_origin_date": fit_origin,
                "purge_gap_positions": fit_position - training_last,
                "training_label_observable_at_fit": bool(label_end <= fit_origin),
                "preprocessing_scope": "outer_training_only",
                "random_train_test_split": False,
                "tuning_enabled": tune,
                "tuning_protocol": (
                    "nested_purged_expanding_window" if tune else "fixed_prespecified"
                ),
                "candidate_id": candidate_id,
                "parameters_json": parameter_json(parameters),
                "parameters_hash": parameter_hash(parameters),
                "fit_seed": fit_seed,
                "fit_seed_scope": (
                    "shared_frequency_algorithm_horizon_fit_origin_candidate"
                ),
                "training_loss": metadata.get("training_loss", math.nan),
                "epochs_completed": metadata.get("epochs_completed", math.nan),
                "active_prophet_regressors": metadata.get(
                    "active_prophet_regressors", math.nan
                ),
                "fit_runtime_seconds": metadata["fit_runtime_seconds"],
                "causal_claim": False,
            }
        )
        for position, prediction in zip(block, predictions):
            forecast_rows.append(
                direct_forecast_row(
                    panel=panel,
                    spec=spec,
                    position=int(position),
                    horizon=horizon,
                    prediction=float(prediction),
                    algorithm=algorithm,
                    configuration=configuration,
                    feature_set=configuration.upstream_feature_key,
                    feature_count=len(features),
                    training=training,
                    fit_position=fit_position,
                    parameters=parameters,
                    candidate_id=candidate_id,
                    fit_seed=fit_seed,
                    tuning_enabled=tune,
                )
            )
        logger.info(
            "fit=%s train=%s label_end=%s block=%s params=%s runtime=%.3fs",
            fit_id,
            len(training),
            label_end.date(),
            len(block),
            parameter_hash(parameters),
            metadata["fit_runtime_seconds"],
        )
    return forecast_rows, fit_rows, tuning_rows


def derived_hybrid_row(
    template: pd.Series,
    prediction: float,
    model_id: str,
    score_source: str,
    training_rows: int,
    training_label_end_date: pd.Timestamp,
    fit_origin_date: pd.Timestamp,
) -> dict[str, Any]:
    row = template.to_dict()
    row["origin_date"] = template.name
    actual = float(template["actual_log_change"])
    current_level = float(template["current_scfi"])
    actual_level = float(template["actual_scfi_level"])
    predicted_level = current_level * math.exp(float(prediction))
    row.update(
        {
            "model_id": model_id,
            "is_direct_model": False,
            "source_type": "forecast_hybrid",
            "score_source": score_source,
            "hybrid_level": "forecast",
            "inference_tier": (
                "exploratory_dimension"
                if str(template["signal_name"]) not in PRIMARY_SIGNALS
                else "secondary_predictive_robustness"
            ),
            "predicted_log_change": float(prediction),
            "forecast_error": actual - float(prediction),
            "absolute_error": abs(actual - float(prediction)),
            "squared_error": (actual - float(prediction)) ** 2,
            "predicted_scfi_level": predicted_level,
            "level_error": actual_level - predicted_level,
            "feature_set": "forecast_hybrid_from_four_llm_predictions",
            "feature_count": 3,
            "training_rows": int(training_rows),
            "training_label_end_date": training_label_end_date,
            "model_fit_origin_position": int(template["origin_position"]),
            "model_fit_origin_date": fit_origin_date,
            "purge_gap_positions": math.nan,
            "tuning_enabled": False,
            "candidate_id": "derived",
            "parameters_json": "{}",
            "parameters_hash": parameter_hash({}),
            "fit_seed": math.nan,
            "causal_claim": False,
        }
    )
    return row


def build_forecast_hybrids(
    direct: pd.DataFrame,
    aliases: Sequence[str],
    minimum_weekly_history: int = 52,
    minimum_monthly_history: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    individual = direct[direct["source_type"].eq("llm")].copy()
    if individual.empty:
        return pd.DataFrame(), pd.DataFrame()
    hybrid_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    group_columns = [
        "frequency",
        "algorithm",
        "index_construction",
        "horizon_steps",
        "signal_name",
    ]
    for keys, group in individual.groupby(group_columns, sort=True):
        frequency, algorithm, construction, horizon, signal = keys
        if set(group["score_source"].unique()) != set(aliases):
            # Deliberately limited/debug runs may omit some LLM configurations.
            # A hybrid is meaningful only when all four frozen paths are present.
            continue
        wide = group.pivot(
            index="origin_date", columns="score_source", values="predicted_log_change"
        ).sort_index()
        wide = wide.reindex(columns=list(aliases))
        if wide.isna().any().any():
            raise AssertionError(f"Unbalanced individual-LLM forecasts for {keys}")
        templates = (
            group.sort_values("score_source")
            .drop_duplicates("origin_date")
            .set_index("origin_date")
            .reindex(wide.index)
        )
        minimum_history = (
            minimum_weekly_history if frequency == "weekly" else minimum_monthly_history
        )
        short = SHORT_ALGORITHM_NAMES[str(algorithm)]
        prefix = f"{short}__{frequency}__{construction}__forecast_hybrid"
        for origin_date, predictions in wide.iterrows():
            template = templates.loc[origin_date]
            equal_prediction = float(predictions.mean())
            hybrid_rows.append(
                derived_hybrid_row(
                    template,
                    equal_prediction,
                    f"{prefix}__forecast_equal_mean__{signal}",
                    "forecast_equal_mean",
                    int(template["training_rows"]),
                    pd.Timestamp(template["training_label_end_date"]),
                    pd.Timestamp(origin_date),
                )
            )
            history = templates[
                pd.to_datetime(templates["target_date"]) <= pd.Timestamp(origin_date)
            ]
            history_dates = history.index.intersection(wide.index)
            fallback = len(history_dates) < minimum_history
            if fallback:
                weights = np.repeat(1.0 / len(aliases), len(aliases))
                history_label_end = pd.Timestamp(template["training_label_end_date"])
            else:
                matrix = wide.loc[history_dates, list(aliases)].to_numpy(dtype=float)
                outcomes = history.loc[history_dates, "actual_log_change"].to_numpy(
                    dtype=float
                )
                weights, _ = nnls(matrix, outcomes)
                if float(weights.sum()) <= 1e-12:
                    weights = np.repeat(1.0 / len(aliases), len(aliases))
                    fallback = True
                else:
                    weights = weights / weights.sum()
                history_label_end = pd.to_datetime(
                    history.loc[history_dates, "target_date"]
                ).max()
            stacked_prediction = float(np.dot(predictions.to_numpy(dtype=float), weights))
            hybrid_rows.append(
                derived_hybrid_row(
                    template,
                    stacked_prediction,
                    f"{prefix}__forecast_nnls_stack__{signal}",
                    "forecast_nnls_stack",
                    len(history_dates),
                    pd.Timestamp(history_label_end),
                    pd.Timestamp(origin_date),
                )
            )
            if pd.Timestamp(history_label_end) > pd.Timestamp(origin_date):
                raise AssertionError("Past-only NNLS used a future realized target")
            for alias, weight in zip(aliases, weights):
                weight_rows.append(
                    {
                        "frequency": frequency,
                        "algorithm": algorithm,
                        "index_construction": construction,
                        "horizon_steps": int(horizon),
                        "signal_name": signal,
                        "origin_date": origin_date,
                        "llm_score_source": alias,
                        "weight": float(weight),
                        "stacking_history_rows": len(history_dates),
                        "stacking_history_label_end": history_label_end,
                        "minimum_stacking_history": minimum_history,
                        "equal_weight_fallback": bool(fallback),
                    }
                )
    hybrids = pd.DataFrame(hybrid_rows)
    weights = pd.DataFrame(weight_rows)
    if not weights.empty:
        sums = weights.groupby(
            [
                "frequency",
                "algorithm",
                "index_construction",
                "horizon_steps",
                "signal_name",
                "origin_date",
            ]
        )["weight"].sum()
        if not np.allclose(sums.to_numpy(), 1.0, atol=1e-10):
            raise AssertionError("Past-only NNLS weights do not sum to one")
    return hybrids, weights


def attach_algorithm_baseline_differences(forecasts: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "frequency",
        "algorithm",
        "index_construction",
        "horizon_steps",
        "origin_date",
    ]
    baseline = forecasts[forecasts["source_type"].eq("baseline")].copy()
    if baseline.duplicated(keys).any():
        raise AssertionError("Algorithm-matched baseline is not unique")
    baseline = baseline[
        [*keys, "model_id", "predicted_log_change", "absolute_error", "squared_error"]
    ].rename(
        columns={
            "model_id": "baseline_model_id",
            "predicted_log_change": "baseline_predicted_log_change",
            "absolute_error": "baseline_absolute_error",
            "squared_error": "baseline_squared_error",
        }
    )
    result = forecasts.merge(baseline, on=keys, how="left", validate="many_to_one")
    if result["baseline_model_id"].isna().any():
        raise AssertionError("At least one forecast lacks its same-algorithm baseline")
    result["prediction_difference_vs_baseline"] = (
        result["predicted_log_change"] - result["baseline_predicted_log_change"]
    )
    result["absolute_loss_improvement_vs_baseline"] = (
        result["baseline_absolute_error"] - result["absolute_error"]
    )
    result["squared_loss_improvement_vs_baseline"] = (
        result["baseline_squared_error"] - result["squared_error"]
    )
    return result


def validate_common_origin_sets(forecasts: pd.DataFrame) -> pd.DataFrame:
    duplicate_keys = [
        "frequency",
        "algorithm",
        "index_construction",
        "horizon_steps",
        "model_id",
        "origin_date",
    ]
    duplicate = forecasts.duplicated(duplicate_keys, keep=False)
    if duplicate.any():
        examples = forecasts.loc[duplicate, duplicate_keys].head(10)
        raise AssertionError(
            "Duplicate extended-forecast origins are not allowed:\n"
            f"{examples.to_string(index=False)}"
        )
    rows: list[dict[str, Any]] = []
    group_columns = [
        "frequency",
        "algorithm",
        "index_construction",
        "horizon_steps",
    ]
    for keys, group in forecasts.groupby(group_columns, sort=True):
        origin_sets = {
            model_id: set(pd.to_datetime(model_frame["origin_date"]))
            for model_id, model_frame in group.groupby("model_id")
        }
        reference_id = sorted(origin_sets)[0]
        reference = origin_sets[reference_id]
        mismatches = [
            model_id
            for model_id, origins in origin_sets.items()
            if origins != reference
        ]
        if mismatches:
            raise AssertionError(
                f"Forecast origins differ for {keys}: reference={reference_id}, "
                f"mismatches={mismatches}"
            )
        rows.append(
            {
                "audit_level": "within_algorithm",
                "frequency": keys[0],
                "algorithm": keys[1],
                "index_construction": keys[2],
                "horizon_steps": int(keys[3]),
                "models_checked": len(origin_sets),
                "common_forecast_origins": len(reference),
                "first_origin_date": min(reference),
                "last_origin_date": max(reference),
                "origin_sets_identical": True,
            }
        )
    return pd.DataFrame(rows)


def validate_cross_algorithm_origin_sets(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in forecasts.groupby(
        ["frequency", "index_construction", "horizon_steps"], sort=True
    ):
        origin_sets = {
            algorithm: set(pd.to_datetime(frame["origin_date"]))
            for algorithm, frame in group.groupby("algorithm")
        }
        reference_algorithm = sorted(origin_sets)[0]
        reference = origin_sets[reference_algorithm]
        mismatches = [
            algorithm
            for algorithm, origins in origin_sets.items()
            if origins != reference
        ]
        if mismatches:
            raise AssertionError(
                f"Cross-algorithm origins differ for {keys}: "
                f"reference={reference_algorithm}, mismatches={mismatches}"
            )
        rows.append(
            {
                "audit_level": "cross_algorithm",
                "frequency": keys[0],
                "algorithm": "all_selected_algorithms",
                "index_construction": keys[1],
                "horizon_steps": int(keys[2]),
                "models_checked": len(origin_sets),
                "common_forecast_origins": len(reference),
                "first_origin_date": min(reference),
                "last_origin_date": max(reference),
                "origin_sets_identical": True,
            }
        )
    return pd.DataFrame(rows)


def calculate_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping = [
        "frequency",
        "algorithm",
        "index_construction",
        "horizon_steps",
        "model_id",
    ]
    for keys, group in forecasts.groupby(grouping, sort=True):
        group = group.sort_values("origin_date")
        metadata = group.iloc[0]
        error = group["forecast_error"].to_numpy(dtype=float)
        level_error = group["level_error"].to_numpy(dtype=float)
        baseline_squared = group["baseline_squared_error"].to_numpy(dtype=float)
        model_squared = group["squared_error"].to_numpy(dtype=float)
        baseline_sse = float(baseline_squared.sum())
        model_sse = float(model_squared.sum())
        actual = group["actual_log_change"].to_numpy(dtype=float)
        predicted = group["predicted_log_change"].to_numpy(dtype=float)
        correlation = (
            float(np.corrcoef(actual, predicted)[0, 1])
            if len(group) > 1
            and float(np.std(actual)) > 0
            and float(np.std(predicted)) > 0
            else math.nan
        )
        rows.append(
            {
                "frequency": keys[0],
                "frequency_is_primary": bool(metadata["frequency_is_primary"]),
                "algorithm": keys[1],
                "index_construction": keys[2],
                "horizon_steps": int(keys[3]),
                "horizon_unit": metadata["horizon_unit"],
                "model_id": keys[4],
                "baseline_model_id": metadata["baseline_model_id"],
                "source_type": metadata["source_type"],
                "score_source": metadata["score_source"],
                "signal_name": metadata["signal_name"],
                "hybrid_level": metadata["hybrid_level"],
                "inference_tier": metadata["inference_tier"],
                "price_overlap_risk": bool(metadata["price_overlap_risk"]),
                "regime_specific_risk": bool(metadata["regime_specific_risk"]),
                "n_forecasts": len(group),
                "first_origin_date": group["origin_date"].min(),
                "last_origin_date": group["origin_date"].max(),
                "mae_log_change": float(np.mean(np.abs(error))),
                "rmse_log_change": float(np.sqrt(np.mean(error**2))),
                "bias_log_change": float(np.mean(predicted - actual)),
                "directional_accuracy": float(
                    np.mean(np.sign(actual) == np.sign(predicted))
                ),
                "forecast_actual_correlation": correlation,
                "mae_scfi_level": float(np.mean(np.abs(level_error))),
                "rmse_scfi_level": float(np.sqrt(np.mean(level_error**2))),
                "oos_r2_vs_algorithm_baseline": (
                    math.nan if baseline_sse <= 0 else 1 - model_sse / baseline_sse
                ),
                "relative_rmse_vs_algorithm_baseline": (
                    math.nan
                    if baseline_sse <= 0
                    else math.sqrt(model_sse / baseline_sse)
                ),
                "mean_absolute_loss_improvement_vs_baseline": float(
                    group["absolute_loss_improvement_vs_baseline"].mean()
                ),
                "mean_squared_loss_improvement_vs_baseline": float(
                    group["squared_loss_improvement_vs_baseline"].mean()
                ),
                "causal_claim": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["frequency", "algorithm", "index_construction", "horizon_steps", "rmse_log_change"]
    )


def hac_mean_test(values: np.ndarray, maximum_lag: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return math.nan, math.nan, math.nan
    centered = values - values.mean()
    gamma_zero = float(np.dot(centered, centered) / n)
    long_run_variance = gamma_zero
    for lag in range(1, min(int(maximum_lag), n - 1) + 1):
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
    adjusted = np.full(len(pvalues), np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(pvalues))
    if not len(finite_positions):
        return adjusted
    finite = np.clip(pvalues[finite_positions], 0, 1)
    order = np.argsort(finite)
    ordered = finite[order]
    count = len(ordered)
    if method == "holm":
        corrected = np.maximum.accumulate((count - np.arange(count)) * ordered)
    elif method == "fdr_bh":
        corrected = np.minimum.accumulate(
            (ordered * count / (np.arange(count) + 1))[::-1]
        )[::-1]
    else:
        raise ValueError(f"Unknown p-value adjustment: {method}")
    restored = np.empty(count, dtype=float)
    restored[order] = np.clip(corrected, 0, 1)
    adjusted[finite_positions] = restored
    return adjusted


def statistical_tests(forecasts: pd.DataFrame) -> pd.DataFrame:
    challengers = forecasts[~forecasts["source_type"].eq("baseline")]
    rows: list[dict[str, Any]] = []
    grouping = [
        "frequency",
        "algorithm",
        "index_construction",
        "horizon_steps",
        "model_id",
    ]
    for keys, group in challengers.groupby(grouping, sort=True):
        metadata = group.iloc[0]
        horizon = int(keys[3])
        comparisons = {
            "diebold_mariano_squared": group[
                "squared_loss_improvement_vs_baseline"
            ].to_numpy(dtype=float),
            "diebold_mariano_absolute": group[
                "absolute_loss_improvement_vs_baseline"
            ].to_numpy(dtype=float),
        }
        primary_signal = metadata["signal_name"] in PRIMARY_SIGNALS
        for test_name, differential in comparisons.items():
            mean_difference, statistic, standard_error = hac_mean_test(
                differential, max(horizon - 1, 0)
            )
            p_value = (
                math.nan
                if not np.isfinite(statistic)
                else float(2 * (1 - norm.cdf(abs(statistic))))
            )
            if primary_signal:
                family_id = f"sensitivity_{keys[0]}_{keys[1]}_{test_name}"
                adjustment = "holm"
                tier = "secondary_robustness"
            else:
                family_id = f"exploratory_dimensions_{keys[0]}_{keys[1]}_{test_name}"
                adjustment = "fdr_bh"
                tier = "exploratory"
            rows.append(
                {
                    "frequency": keys[0],
                    "algorithm": keys[1],
                    "index_construction": keys[2],
                    "horizon_steps": horizon,
                    "horizon_unit": metadata["horizon_unit"],
                    "baseline_model": metadata["baseline_model_id"],
                    "challenger_model": keys[4],
                    "source_type": metadata["source_type"],
                    "score_source": metadata["score_source"],
                    "signal_name": metadata["signal_name"],
                    "test": test_name,
                    "n": len(group),
                    "hac_lag": max(horizon - 1, 0),
                    "mean_loss_improvement": mean_difference,
                    "standard_error": standard_error,
                    "statistic": statistic,
                    "p_value": p_value,
                    "alternative": "two_sided",
                    "positive_favors": "challenger",
                    "family_id": family_id,
                    "adjustment_method": adjustment,
                    "inference_tier": tier,
                    "price_overlap_risk": bool(metadata["price_overlap_risk"]),
                    "causal_claim": False,
                }
            )
    tests = pd.DataFrame(rows)
    if tests.empty:
        return tests
    tests["p_value_adjusted"] = math.nan
    for (_, method), positions in tests.groupby(
        ["family_id", "adjustment_method"], sort=False
    ).groups.items():
        tests.loc[positions, "p_value_adjusted"] = adjust_pvalues(
            tests.loc[positions, "p_value"], method
        )
    tests["significant_raw_0_05"] = tests["p_value"].lt(0.05)
    tests["significant_adjusted_0_05"] = tests["p_value_adjusted"].lt(0.05)
    return tests.sort_values(
        ["family_id", "p_value_adjusted", "frequency", "algorithm", "horizon_steps"]
    )


def build_llm_ranking(metrics: pd.DataFrame, aliases: Sequence[str]) -> pd.DataFrame:
    selected = metrics[
        metrics["source_type"].eq("llm")
        & metrics["score_source"].isin(aliases)
        & metrics["signal_name"].isin(PRIMARY_SIGNALS)
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    cell = [
        "frequency",
        "algorithm",
        "index_construction",
        "horizon_steps",
        "signal_name",
    ]
    selected["rmse_rank_in_cell"] = selected.groupby(cell)["rmse_log_change"].rank(
        method="average", ascending=True
    )
    ranking = (
        selected.groupby(
            ["frequency", "algorithm", "index_construction", "score_source"],
            as_index=False,
        )
        .agg(
            evaluation_cells=("model_id", "size"),
            mean_rmse_rank=("rmse_rank_in_cell", "mean"),
            mean_rmse=("rmse_log_change", "mean"),
            mean_relative_rmse=("relative_rmse_vs_algorithm_baseline", "mean"),
            mean_oos_r2=("oos_r2_vs_algorithm_baseline", "mean"),
        )
    )
    ranking["predictive_point_rank"] = ranking.groupby(
        ["frequency", "algorithm", "index_construction"]
    )["mean_rmse_rank"].rank(method="min", ascending=True)
    ranking["point_rank_is_not_significance"] = True
    return ranking.sort_values(
        ["frequency", "algorithm", "index_construction", "predictive_point_rank"]
    )


def build_pipeline_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[
        metrics["source_type"].eq("llm")
        & metrics["signal_name"].isin(PRIMARY_SIGNALS)
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    cell = [
        "frequency",
        "index_construction",
        "horizon_steps",
        "score_source",
        "signal_name",
    ]
    selected["algorithm_rmse_rank_in_cell"] = selected.groupby(cell)[
        "rmse_log_change"
    ].rank(method="average", ascending=True)
    ranking = (
        selected.groupby(
            ["frequency", "index_construction", "algorithm"], as_index=False
        )
        .agg(
            evaluation_cells=("model_id", "size"),
            mean_algorithm_rmse_rank=("algorithm_rmse_rank_in_cell", "mean"),
            mean_rmse=("rmse_log_change", "mean"),
            mean_relative_rmse=("relative_rmse_vs_algorithm_baseline", "mean"),
        )
    )
    ranking["pipeline_point_rank"] = ranking.groupby(
        ["frequency", "index_construction"]
    )["mean_algorithm_rmse_rank"].rank(method="min", ascending=True)
    representations = {
        "glm": "full_upstream_tabular_lag_features",
        "random_forest": "full_upstream_tabular_lag_features",
        "xgboost": "full_upstream_tabular_lag_features",
        "lstm": "reduced_origin_features_plus_sequence",
        "prophet": "reduced_origin_regressors_plus_trend_seasonality",
    }
    ranking["feature_representation"] = ranking["algorithm"].map(representations)
    ranking["ranking_unit"] = "end_to_end_pipeline_not_algorithm_only"
    ranking["point_rank_is_not_significance"] = True
    return ranking.sort_values(
        ["frequency", "index_construction", "pipeline_point_rank"]
    )


TUNING_AUDIT_COLUMNS = [
    "fit_id",
    "frequency",
    "algorithm",
    "horizon_steps",
    "outer_fit_position",
    "outer_fit_origin_date",
    "candidate_id",
    "candidate_index",
    "parameters_json",
    "parameters_hash",
    "fold",
    "inner_training_rows",
    "inner_train_first_position",
    "inner_train_last_position",
    "inner_train_first_date",
    "inner_train_last_feature_date",
    "inner_train_last_label_date",
    "validation_rows",
    "validation_first_position",
    "validation_last_position",
    "validation_first_date",
    "validation_last_date",
    "purge_check_passed",
    "fold_rmse",
    "fit_runtime_seconds",
    "candidate_pooled_rmse",
    "selected",
]


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def checkpoint_cell_paths(
    root: Path,
    frequency: str,
    horizon: int,
    algorithm: str,
    model_id: str,
) -> dict[str, Path]:
    identity = parameter_hash(
        {
            "frequency": frequency,
            "horizon": horizon,
            "algorithm": algorithm,
            "model_id": model_id,
        }
    )
    stem = f"{frequency}_h{horizon}_{algorithm}_{identity}"
    return {
        "forecasts": root / f"{stem}_forecasts.csv",
        "fits": root / f"{stem}_fits.csv",
        "tuning": root / f"{stem}_tuning.csv",
    }


def save_cell_checkpoint(
    paths: dict[str, Path],
    forecasts: Sequence[dict[str, Any]],
    fits: Sequence[dict[str, Any]],
    tuning: Sequence[dict[str, Any]],
) -> None:
    atomic_write_csv(pd.DataFrame(forecasts), paths["forecasts"])
    atomic_write_csv(pd.DataFrame(fits), paths["fits"])
    atomic_write_csv(
        pd.DataFrame(tuning, columns=TUNING_AUDIT_COLUMNS), paths["tuning"]
    )


def load_cell_checkpoint(
    paths: dict[str, Path]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    if not all(path.is_file() for path in paths.values()):
        return None
    forecasts = pd.read_csv(paths["forecasts"], low_memory=False).to_dict(
        orient="records"
    )
    fits = pd.read_csv(paths["fits"], low_memory=False).to_dict(orient="records")
    tuning = pd.read_csv(paths["tuning"], low_memory=False).to_dict(orient="records")
    if not forecasts or not fits:
        raise ValueError(f"Checkpoint cell is empty or corrupt: {paths}")
    return forecasts, fits, tuning


def limited_configurations(
    configurations: Sequence[ForecastConfiguration], maximum: int | None
) -> list[ForecastConfiguration]:
    if maximum is None:
        return list(configurations)
    if maximum < 1:
        raise ValueError("max-configurations must be positive")
    selected: list[ForecastConfiguration] = []
    for construction in dict.fromkeys(
        configuration.index_construction for configuration in configurations
    ):
        selected.extend(
            [
                configuration
                for configuration in configurations
                if configuration.index_construction == construction
            ][:maximum]
        )
    return selected


def determine_run_status(
    *,
    smoke_test: bool,
    skipped_algorithms: Sequence[str],
    selected_frequencies: Sequence[str],
    requested_algorithms: Sequence[str],
    executed_algorithms: Sequence[str],
    signals_scope: str,
    maximum_configurations: int | None,
    maximum_origins: int | None,
) -> tuple[str, bool]:
    canonical_scope = (
        list(selected_frequencies) == ["weekly", "monthly"]
        and set(requested_algorithms) == set(ALGORITHMS)
        and set(executed_algorithms) == set(ALGORITHMS)
        and not skipped_algorithms
        and signals_scope == "primary"
        and maximum_configurations is None
        and maximum_origins is None
    )
    if smoke_test:
        return "debug_smoke_not_for_publication", False
    if skipped_algorithms:
        return "incomplete_models_not_for_publication", False
    if not canonical_scope:
        return "partial_scope_not_for_publication", False
    return "provisional_pending_human_validation", True


def main() -> None:
    args = parse_args()
    executable = assert_project_venv()
    if args.max_origins_per_horizon is not None and args.max_origins_per_horizon < 1:
        raise ValueError("max-origins-per-horizon must be positive")
    config_path = require_file(args.config, "Nonlinear forecast configuration")
    config = load_yaml(config_path)
    seed = int(
        args.seed
        if args.seed is not None
        else config.get("design", {}).get("seed", 20260712)
    )
    random.seed(seed)
    np.random.seed(seed)

    logger, log_path = configure_logging(args.log_dir.resolve(), args.version)
    logger.info("Starting extended SCFI forecasts version=%s", args.version)
    logger.info("Python executable=%s sys.prefix=%s", executable, Path(sys.prefix).resolve())
    logger.info("Arguments=%s", vars(args))
    requested_algorithms = parse_model_selection(args.models)
    selected_algorithms, dependencies = dependency_audit(
        requested_algorithms, args.missing_dependency
    )
    skipped_algorithms = [
        algorithm
        for algorithm in requested_algorithms
        if algorithm not in selected_algorithms
    ]
    logger.info(
        "Dependency preflight requested=%s selected=%s skipped=%s",
        requested_algorithms,
        selected_algorithms,
        skipped_algorithms,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": output_dir / f"{args.version}_forecast_predictions.csv",
        "metrics": output_dir / f"{args.version}_forecast_metrics.csv",
        "statistical_tests": output_dir
        / f"{args.version}_forecast_statistical_tests.csv",
        "fit_audit": output_dir / f"{args.version}_model_fit_audit.csv",
        "tuning_audit": output_dir / f"{args.version}_tuning_audit.csv",
        "dependency_audit": output_dir / f"{args.version}_dependency_audit.csv",
        "origin_audit": output_dir / f"{args.version}_common_origin_audit.csv",
        "loss_differences": output_dir / f"{args.version}_loss_differences.csv",
        "hybrid_weights": output_dir / f"{args.version}_forecast_hybrid_weights.csv",
        "llm_ranking": output_dir / f"{args.version}_llm_predictive_ranking.csv",
        "pipeline_ranking": output_dir / f"{args.version}_pipeline_ranking.csv",
        "feature_sets": output_dir / f"{args.version}_feature_sets.json",
        "manifest": output_dir / f"{args.version}_manifest.json",
    }
    dependencies.to_csv(paths["dependency_audit"], index=False)

    registry_path = require_file(args.model_registry, "Model registry")
    aliases = load_model_aliases(registry_path)
    specifications = frequency_specs(
        config,
        weekly_panel=args.weekly_panel,
        monthly_panel=args.monthly_panel,
        weekly_manifest=args.weekly_manifest,
        monthly_manifest=args.monthly_manifest,
        weekly_feature_sets=args.weekly_feature_sets,
        monthly_feature_sets=args.monthly_feature_sets,
    )
    selected_frequencies = (
        ["weekly", "monthly"]
        if args.frequency == "both"
        else [args.frequency]
    )
    maximum_configurations = args.max_configurations
    maximum_origins = args.max_origins_per_horizon
    if args.smoke_test:
        if maximum_configurations is None:
            maximum_configurations = 1
        if maximum_origins is None:
            maximum_origins = 2
        logger.warning(
            "Smoke-test limits active: challenger configurations per construction=%s "
            "origins per horizon=%s; outputs are not for publication",
            maximum_configurations,
            maximum_origins,
        )

    signature_inputs: dict[str, Any] = {}
    for frequency in selected_frequencies:
        signature_spec = specifications[frequency]
        signature_inputs[frequency] = {
            "panel_sha256": sha256_file(
                require_file(signature_spec.panel_path, f"{frequency} panel")
            ),
            "manifest_sha256": sha256_file(
                require_file(
                    signature_spec.upstream_manifest_path,
                    f"{frequency} upstream manifest",
                )
            ),
            "feature_sets_sha256": sha256_file(
                require_file(
                    signature_spec.upstream_feature_sets_path,
                    f"{frequency} feature sets",
                )
            ),
        }
    run_signature = {
        "version": args.version,
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "configuration_sha256": sha256_file(config_path),
        "model_registry_sha256": sha256_file(registry_path),
        "input_hashes": signature_inputs,
        "frequencies": selected_frequencies,
        "requested_algorithms": requested_algorithms,
        "executed_algorithms": selected_algorithms,
        "signals": args.signals,
        "tune": args.tune,
        "inner_splits": int(config.get("design", {}).get("inner_splits", 3)),
        "seed": seed,
        "smoke_test": args.smoke_test,
        "maximum_configurations_per_construction": maximum_configurations,
        "maximum_origins_per_horizon": maximum_origins,
    }
    signature_json = json.dumps(
        run_signature, sort_keys=True, separators=(",", ":"), default=json_default
    )
    signature_hash = hashlib.sha256(signature_json.encode("utf-8")).hexdigest()
    checkpoint_base = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir is not None
        else output_dir / f"{args.version}_checkpoints"
    )
    checkpoint_root = checkpoint_base / signature_hash[:16]
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    signature_path = checkpoint_root / "run_signature.json"
    if signature_path.is_file():
        recorded_signature = json.loads(signature_path.read_text(encoding="utf-8"))
        if recorded_signature != run_signature:
            raise AssertionError("Checkpoint run signature collision")
    else:
        atomic_write_text(
            signature_path,
            json.dumps(run_signature, indent=2, default=json_default),
        )
    paths["checkpoint_directory"] = checkpoint_root
    logger.info(
        "Checkpoint directory=%s resume=%s signature=%s",
        checkpoint_root,
        args.resume,
        signature_hash[:16],
    )

    all_forecast_rows: list[dict[str, Any]] = []
    all_fit_rows: list[dict[str, Any]] = []
    all_tuning_rows: list[dict[str, Any]] = []
    feature_audit: dict[str, Any] = {}
    input_audit: dict[str, Any] = {
        "configuration": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "model_registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
        },
    }
    configuration_counts: dict[str, Any] = {}
    origin_design: list[dict[str, Any]] = []
    inner_splits = int(config.get("design", {}).get("inner_splits", 3))
    if inner_splits < 1:
        raise ValueError("design.inner_splits must be positive")

    for frequency in selected_frequencies:
        spec = specifications[frequency]
        spec = FrequencySpec(
            **{
                **spec.__dict__,
                "panel_path": require_file(spec.panel_path, f"{frequency} panel"),
                "upstream_manifest_path": require_file(
                    spec.upstream_manifest_path, f"{frequency} manifest"
                ),
                "upstream_feature_sets_path": require_file(
                    spec.upstream_feature_sets_path, f"{frequency} feature sets"
                ),
            }
        )
        upstream = validate_upstream_manifest(spec, aliases)
        panel = load_panel(spec)
        upstream_feature_sets = load_feature_sets(spec)
        challenger_configurations = limited_configurations(
            build_configurations(spec, aliases, args.signals),
            maximum_configurations,
        )
        configurations: list[ForecastConfiguration] = []
        for construction in spec.constructions:
            configurations.append(baseline_configuration(spec, construction))
            configurations.extend(
                configuration
                for configuration in challenger_configurations
                if configuration.index_construction == construction
            )
        selected_features = validate_selected_features(
            panel,
            upstream_feature_sets,
            configurations,
            selected_algorithms,
        )
        feature_audit[frequency] = {
            algorithm: {
                model_id: {
                    "feature_count": len(features),
                    "features": features,
                    "preprocessing": (
                        "training-only median imputer and standard scaler"
                        if algorithm == "glm"
                        else "training-only median imputer"
                        if algorithm in {"random_forest", "xgboost"}
                        else "training-only sequence median imputer and standard scaler"
                        if algorithm == "lstm"
                        else "training-only median imputer; Prophet internal regressor scaling"
                    ),
                }
                for model_id, features in mapping.items()
            }
            for algorithm, mapping in selected_features.items()
        }
        input_audit[frequency] = {
            "panel": {
                "path": str(spec.panel_path),
                "sha256": sha256_file(spec.panel_path),
                "rows": len(panel),
                "complete_rows": int(panel[spec.complete_column].sum()),
                "first_date": panel[spec.date_column].min(),
                "last_date": panel[spec.date_column].max(),
            },
            "feature_sets": {
                "path": str(spec.upstream_feature_sets_path),
                "sha256": sha256_file(spec.upstream_feature_sets_path),
            },
            "upstream_manifest": upstream,
        }
        configuration_counts[frequency] = {
            "challengers": len(challenger_configurations),
            "baselines": len(spec.constructions),
            "constructions": list(spec.constructions),
        }
        logger.info(
            "Loaded %s panel rows=%s complete=%s challengers=%s constructions=%s",
            frequency,
            len(panel),
            int(panel[spec.complete_column].sum()),
            len(challenger_configurations),
            list(spec.constructions),
        )

        for horizon in spec.horizons:
            origins = eligible_origins(panel, spec, horizon, maximum_origins)
            if not origins:
                raise ValueError(f"No eligible {frequency} origins for h={horizon}")
            origin_design.append(
                {
                    "frequency": frequency,
                    "horizon_steps": horizon,
                    "eligible_origins": len(origins),
                    "first_origin_date": panel.at[origins[0], spec.date_column],
                    "last_origin_date": panel.at[origins[-1], spec.date_column],
                    "minimum_training_rows": spec.minimum_training_rows,
                    "refit_every": spec.refit_every,
                }
            )
            logger.info(
                "Running %s horizon=%s%s common_origins=%s first=%s last=%s",
                frequency,
                horizon,
                spec.horizon_unit,
                len(origins),
                panel.at[origins[0], spec.date_column],
                panel.at[origins[-1], spec.date_column],
            )
            for algorithm in selected_algorithms:
                candidates = resolve_parameter_candidates(config, algorithm, args.tune)
                for configuration in configurations:
                    model_id = configuration.model_id(algorithm)
                    features = selected_features[algorithm][model_id]
                    cell_paths = checkpoint_cell_paths(
                        checkpoint_root,
                        frequency,
                        horizon,
                        algorithm,
                        model_id,
                    )
                    checkpoint = (
                        load_cell_checkpoint(cell_paths) if args.resume else None
                    )
                    if checkpoint is None:
                        rows, cell_fits, tuning = rolling_forecasts(
                            panel=panel,
                            spec=spec,
                            horizon=horizon,
                            origins=origins,
                            algorithm=algorithm,
                            configuration=configuration,
                            features=features,
                            candidates=candidates,
                            tune=args.tune,
                            inner_splits=inner_splits,
                            seed=seed,
                            smoke_test=args.smoke_test,
                            logger=logger,
                        )
                        save_cell_checkpoint(cell_paths, rows, cell_fits, tuning)
                    else:
                        rows, cell_fits, tuning = checkpoint
                        logger.info(
                            "Resumed checkpoint frequency=%s h=%s algorithm=%s "
                            "model=%s forecasts=%s fits=%s",
                            frequency,
                            horizon,
                            algorithm,
                            model_id,
                            len(rows),
                            len(cell_fits),
                        )
                    all_forecast_rows.extend(rows)
                    all_fit_rows.extend(cell_fits)
                    all_tuning_rows.extend(tuning)
                    logger.info(
                        "Completed frequency=%s h=%s algorithm=%s model=%s "
                        "forecasts=%s fits=%s",
                        frequency,
                        horizon,
                        algorithm,
                        model_id,
                        len(rows),
                        len(cell_fits),
                    )

    direct = pd.DataFrame(all_forecast_rows)
    fits = pd.DataFrame(all_fit_rows)
    tuning_audit = pd.DataFrame(all_tuning_rows, columns=TUNING_AUDIT_COLUMNS)
    if direct.empty or fits.empty:
        raise ValueError("No extended forecasts or fits were produced")
    date_columns = [
        "origin_date",
        "target_date",
        "training_feature_end_date",
        "training_label_end_date",
        "model_fit_origin_date",
    ]
    for column in date_columns:
        direct[column] = pd.to_datetime(direct[column])
    fit_date_columns = [
        "forecast_block_first_date",
        "forecast_block_last_date",
        "training_first_feature_date",
        "training_feature_end_date",
        "training_label_end_date",
        "model_fit_origin_date",
    ]
    for column in fit_date_columns:
        fits[column] = pd.to_datetime(fits[column])
    if (fits["training_label_end_date"] > fits["model_fit_origin_date"]).any():
        raise AssertionError("Fit audit found future training labels")

    hybrids, hybrid_weights = build_forecast_hybrids(direct, aliases)
    if not hybrids.empty:
        for column in date_columns:
            hybrids[column] = pd.to_datetime(hybrids[column])
        forecasts = pd.concat([direct, hybrids], ignore_index=True, sort=False)
    else:
        forecasts = direct.copy()
    forecasts = attach_algorithm_baseline_differences(forecasts)
    forecasts = forecasts.sort_values(
        [
            "frequency",
            "algorithm",
            "index_construction",
            "horizon_steps",
            "origin_date",
            "model_id",
        ]
    ).reset_index(drop=True)
    origin_audit = pd.concat(
        [
            validate_common_origin_sets(forecasts),
            validate_cross_algorithm_origin_sets(forecasts),
        ],
        ignore_index=True,
    )
    metrics = calculate_metrics(forecasts)
    tests = statistical_tests(forecasts)
    llm_ranking = build_llm_ranking(metrics, aliases)
    pipeline_ranking = build_pipeline_ranking(metrics)
    loss_differences = forecasts[
        ~forecasts["source_type"].eq("baseline")
    ][
        [
            "frequency",
            "algorithm",
            "index_construction",
            "horizon_steps",
            "origin_date",
            "target_date",
            "model_id",
            "baseline_model_id",
            "source_type",
            "score_source",
            "signal_name",
            "actual_log_change",
            "predicted_log_change",
            "baseline_predicted_log_change",
            "prediction_difference_vs_baseline",
            "absolute_error",
            "baseline_absolute_error",
            "absolute_loss_improvement_vs_baseline",
            "squared_error",
            "baseline_squared_error",
            "squared_loss_improvement_vs_baseline",
        ]
    ].copy()

    forecasts.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    tests.to_csv(paths["statistical_tests"], index=False)
    fits.to_csv(paths["fit_audit"], index=False)
    tuning_audit.to_csv(paths["tuning_audit"], index=False)
    origin_audit.to_csv(paths["origin_audit"], index=False)
    loss_differences.to_csv(paths["loss_differences"], index=False)
    hybrid_weights.to_csv(paths["hybrid_weights"], index=False)
    llm_ranking.to_csv(paths["llm_ranking"], index=False)
    pipeline_ranking.to_csv(paths["pipeline_ranking"], index=False)
    paths["feature_sets"].write_text(
        json.dumps(feature_audit, indent=2, default=json_default), encoding="utf-8"
    )
    output_frames = {
        "predictions": forecasts,
        "metrics": metrics,
        "statistical_tests": tests,
        "fit_audit": fits,
        "tuning_audit": tuning_audit,
        "dependency_audit": dependencies,
        "origin_audit": origin_audit,
        "loss_differences": loss_differences,
        "hybrid_weights": hybrid_weights,
        "llm_ranking": llm_ranking,
        "pipeline_ranking": pipeline_ranking,
    }
    contract_output_keys = [*output_frames, "feature_sets"]
    output_sha256 = {
        key: sha256_file(paths[key]) for key in contract_output_keys
    }
    output_row_counts = {
        **{key: int(len(frame)) for key, frame in output_frames.items()},
        "feature_sets": 1,
    }

    status, canonical_scope = determine_run_status(
        smoke_test=args.smoke_test,
        skipped_algorithms=skipped_algorithms,
        selected_frequencies=selected_frequencies,
        requested_algorithms=requested_algorithms,
        executed_algorithms=selected_algorithms,
        signals_scope=args.signals,
        maximum_configurations=maximum_configurations,
        maximum_origins=maximum_origins,
    )
    significant_monthly_primary = (
        tests[
            tests.get("frequency", pd.Series(dtype=str)).eq("monthly")
            & tests.get("signal_name", pd.Series(dtype=str)).isin(PRIMARY_SIGNALS)
            & tests.get("test", pd.Series(dtype=str)).eq(
                "diebold_mariano_squared"
            )
            & tests.get("significant_adjusted_0_05", pd.Series(dtype=bool))
            & tests.get("mean_loss_improvement", pd.Series(dtype=float)).gt(0)
        ]
        if not tests.empty
        else tests
    )
    manifest = {
        "version": args.version,
        "status": status,
        "inputs": input_audit,
        "design": {
            "primary_frequency": "monthly",
            "weekly_analysis_role": "higher-frequency robustness",
            "selected_frequencies": selected_frequencies,
            "requested_algorithms": requested_algorithms,
            "executed_algorithms": selected_algorithms,
            "skipped_algorithms": skipped_algorithms,
            "full_requested_model_set_complete": not skipped_algorithms,
            "full_canonical_scope_complete": canonical_scope,
            "signals_scope": args.signals,
            "primary_signals": list(PRIMARY_SIGNALS),
            "dimension_signals_are_exploratory": True,
            "score_sources": [*aliases, *SCORE_HYBRID_SOURCES],
            "configuration_counts": configuration_counts,
            "target": "direct forward SCFI log change; future level reconstructed",
            "same_algorithm_baseline": "AR + Brent + common news coverage",
            "training_label_purge": "endpoint s allowed only when s + horizon <= fit origin",
            "outer_evaluation": "purged expanding-window rolling origin",
            "preprocessing_fit_inside_training_window": True,
            "random_train_test_split": False,
            "common_origins_enforced": True,
            "tuning_enabled": args.tune,
            "tuning_protocol": (
                "nested purged expanding-window validation at every outer refit"
                if args.tune
                else "fixed pre-specified parameters; no search"
            ),
            "predefined_parameter_source": str(config_path),
            "inner_splits": inner_splits,
            "forecast_hybrids": {
                "equal_mean": "equal mean of four individual-LLM forecasts",
                "past_only_nnls": (
                    "nonnegative weights learned only from prior OOS forecasts "
                    "whose target dates are observed by the current origin"
                ),
            },
            "seed": seed,
            "smoke_test": args.smoke_test,
            "maximum_configurations_per_construction": maximum_configurations,
            "maximum_origins_per_horizon": maximum_origins,
            "per_cell_atomic_checkpoints": True,
            "resume_requested": args.resume,
            "checkpoint_run_signature": signature_hash,
        },
        "origin_design": origin_design,
        "common_origin_audit": origin_audit.to_dict(orient="records"),
        "counts": {
            "direct_forecasts": len(direct),
            "forecast_hybrids": len(hybrids),
            "total_forecasts": len(forecasts),
            "outer_fits": len(fits),
            "nested_tuning_rows": len(tuning_audit),
            "statistical_tests": len(tests),
            "confirmatory_tests": 0,
            "adjusted_significant_positive_monthly_primary_tests": len(
                significant_monthly_primary
            ),
        },
        "interpretation_limits": {
            "causal_claim": False,
            "best_predictive_llm_is_separate_from_algorithm_rank": True,
            "point_rank_does_not_establish_statistical_superiority": True,
            "monthly_is_primary": True,
            "monthly_small_sample_overfit_risk": True,
            "lstm_and_prophet_are_sensitivity_models": True,
            "glm_is_gaussian_identity_link_with_l2_regularization": True,
            "all_07d_extended_models_are_secondary_to_frozen_ridge_design": True,
            "pipeline_rank_compares_different_feature_representations": True,
            "overlapping_horizons_use_hac_inference": True,
            "explicit_missing_model_skip_invalidates_publication_run": bool(
                skipped_algorithms
            ),
            "any_partial_frequency_model_configuration_or_origin_scope_invalidates_publication_run": True,
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "output_sha256": output_sha256,
        "output_row_counts": output_row_counts,
        "log": str(log_path),
        "software": {
            "python": platform.python_version(),
            "python_executable": str(executable),
            "sys_prefix": str(Path(sys.prefix).resolve()),
            "project_venv_enforced": True,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "dependencies": dependencies.to_dict(orient="records"),
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )
    logger.info(
        "Completed forecasts=%s direct=%s hybrids=%s fits=%s tests=%s status=%s",
        len(forecasts),
        len(direct),
        len(hybrids),
        len(fits),
        len(tests),
        status,
    )
    logger.info("Predictions=%s", paths["predictions"])
    logger.info("Metrics=%s", paths["metrics"])
    logger.info("Manifest=%s", paths["manifest"])


if __name__ == "__main__":
    main()
