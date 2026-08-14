"""Validate or execute the complete SC-Disrupt research pipeline.

``smoke`` is deliberately read-only: it compiles every stage, validates the
stage registry, imports required packages, parses every YAML configuration,
and audits existing manifests/artifacts.  It never invokes a stage-local
``--smoke-test`` option because those options write partial research outputs.

``full`` performs a fail-fast, sequential run with the publication settings.
The forecasting-model stage enables one initial, horizon-purged calibration per
algorithm and frequency, then freezes parameters for rolling evaluation.
Manual LLM scoring and human annotation remain explicit gates rather
than being represented as completed automated work.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PIPELINE_RUNS_DIR = PROJECT_ROOT / "outputs/pipeline_runs"
EXPECTED_FORECAST_ALGORITHMS = (
    "glm",
    "random_forest",
    "xgboost",
    "lstm",
    "prophet",
)
# Optional scripts are checked by ``smoke`` but are deliberately outside the
# publication pipeline until their external raw-data prerequisites exist.
OPTIONAL_EXTENSION_SCRIPTS = (
    "25_clean_external_controls.py",
    "26_external_controls_eda.py",
    "27_external_controls_statistical_analysis.py",
    "28_create_pseudo_annotations.py",
    "29_inter_annotator_agreement.py",
    "30_portwatch_event_study.py",
)


@dataclass(frozen=True)
class Stage:
    number: int
    script: str
    purpose: str
    mode: str = "automatic"
    arguments: tuple[str, ...] = ()
    completion_artifacts: tuple[str, ...] = ()

    @property
    def stage_id(self) -> str:
        return f"{self.number:02d}"

    @property
    def script_path(self) -> Path:
        return SCRIPTS_DIR / self.script

    def command(self, python: Path) -> list[str]:
        return [str(python), str(self.script_path), *self.arguments]


STAGES: tuple[Stage, ...] = (
    Stage(0, "00_audit_legacy_project.py", "Audit legacy inputs", "structural"),
    Stage(1, "01_ingest_sources.py", "Register source inputs", "structural"),
    Stage(
        2,
        "02_normalize_legacy_news.py",
        "Normalize legacy news",
        completion_artifacts=(
            "data/02_interim/news_cleaned/legacy_news_unique_for_retrieval.csv",
        ),
    ),
    Stage(
        3,
        "03_normalize_llm_news.py",
        "Normalize LLM-search news",
        completion_artifacts=(
            "data/02_interim/news_cleaned/llm_search_results_cleaned.csv",
        ),
    ),
    Stage(4, "04_enrich_full_text.py", "Full-text umbrella stage", "structural"),
    Stage(
        5,
        "05_retrieve_legacy_full_text.py",
        "Retrieve legacy full text",
        arguments=("--resume",),
        completion_artifacts=(
            "data/02_interim/article_enrichment/full_text_news4k/full_text_news4k_results.csv",
        ),
    ),
    Stage(
        6,
        "06_retrieve_llm_full_text.py",
        "Retrieve LLM-search full text",
        arguments=("--resume",),
        completion_artifacts=(
            "data/02_interim/article_enrichment/full_text_llm_news4k/llm_full_text_news4k_succeeded.csv",
        ),
    ),
    Stage(
        7,
        "07_merge_full_text_deduplicate.py",
        "Merge and deduplicate full text",
        completion_artifacts=(
            "data/03_processed/news_full_text/full_text_merge_summary.json",
        ),
    ),
    Stage(
        8,
        "08_filter_irrelevant_news.py",
        "Filter irrelevant news and assign canonical events",
        "external_kaggle",
        completion_artifacts=(
            "data/03_processed/news_full_text/pre_llm_filter/pre_llm_filter_summary.json",
            "data/03_processed/news_full_text/pre_llm_filter/event_dedup/canonical_event_manifest.json",
        ),
    ),
    Stage(
        9,
        "09_analyze_pre_llm_corpus.py",
        "Describe the retained pre-LLM corpus",
        completion_artifacts=("outputs/pre_llm_kept_analysis/analysis_summary.json",),
    ),
    Stage(
        10,
        "10_run_llm_scoring.py",
        "Run external LLM scoring",
        "manual_scoring",
        completion_artifacts=("data/03_processed/model_scores/scores_zips",),
    ),
    Stage(
        11,
        "11_merge_llm_scores.py",
        "Merge LLM score archives",
        completion_artifacts=(
            "data/03_processed/model_scores/llm_scores_merged_audit.json",
        ),
    ),
    Stage(
        12,
        "12_human_validation_metrics.py",
        "Create or evaluate the human gold set",
        "human_validation",
        completion_artifacts=(
            "data/04_validation/annotation_samples/goldset_sampling_manifest.json",
        ),
    ),
    Stage(
        13,
        "13_build_disruption_index.py",
        "Build article, monthly-primary, and weekly-robustness disruption indices",
        completion_artifacts=(
            "data/05_indices/disruption_index_versions/msdi_v001_manifest.json",
            "data/05_indices/monthly/msdi_monthly.csv",
            "data/05_indices/weekly/msdi_weekly.csv",
        ),
    ),
    Stage(
        14,
        "14_build_model_panels.py",
        "Build leakage-safe model panels",
        completion_artifacts=("outputs/model_panel_audit/panel_v001_merge_audit.json",),
    ),
    Stage(
        15,
        "15_prediction_benchmarks.py",
        "Run statistical and ML benchmarks",
        completion_artifacts=(
            "outputs/prediction_benchmarks/forecast_v001_manifest.json",
        ),
    ),
    Stage(
        16,
        "16_llm_scfi_comparison_weekly.py",
        "Compare weekly LLM signals",
        completion_artifacts=(
            "outputs/llm_scfi_comparison/llm_scfi_v001_manifest.json",
        ),
    ),
    Stage(
        17,
        "17_llm_scfi_comparison_monthly.py",
        "Compare monthly LLM signals",
        completion_artifacts=(
            "outputs/llm_scfi_comparison/llm_scfi_monthly_v001_manifest.json",
        ),
    ),
    Stage(
        18,
        "18_forecasting_models.py",
        "Calibrate and evaluate GLM, RF, XGBoost, LSTM, and Prophet forecasts",
        arguments=(
            "--frequency",
            "both",
            "--models",
            ",".join(EXPECTED_FORECAST_ALGORITHMS),
            "--signals",
            "primary",
            "--tune",
            "--missing-dependency",
            "error",
            "--resume",
        ),
        completion_artifacts=(
            "outputs/nonlinear_forecasts/nonlinear_scfi_v001_manifest.json",
        ),
    ),
    Stage(
        19,
        "19_time_series_causality.py",
        "Run VAR/VARX, Granger, PCMCI, and local projections",
        completion_artifacts=(
            "outputs/time_series_causality/causality_v001_manifest.json",
        ),
    ),
    Stage(
        20,
        "20_causal_ml.py",
        "Run time-aware causal-ML diagnostics",
        arguments=("--frequency", "both"),
        completion_artifacts=("outputs/causal_ml/causal_ml_v001_manifest.json",),
    ),
    Stage(
        21,
        "21_ablation_sensitivity_checks.py",
        "Run ablation and sensitivity checks",
        arguments=("--frequency", "both", "--resume"),
        completion_artifacts=("outputs/robustness/robustness_v001_manifest.json",),
    ),
    Stage(
        22,
        "22_error_diagnostics.py",
        "Run forecast and model diagnostics",
        completion_artifacts=(
            "outputs/diagnostics/error_diagnostics_v001_manifest.json",
        ),
    ),
    Stage(
        23,
        "23_significance_analysis.py",
        "Consolidate multiplicity-aware inference",
        completion_artifacts=(
            "outputs/significance/significance_v001_manifest.json",
        ),
    ),
    Stage(
        24,
        "24_make_figures_tables.py",
        "Create publication tables and figures",
        completion_artifacts=("outputs/reports/publication_v001_manifest.json",),
    ),
)

STAGE_DEPENDENCIES: dict[int, tuple[int, ...]] = {
    0: (),
    1: (),
    2: (1,),
    3: (1,),
    4: (2, 3),
    5: (2,),
    6: (3,),
    7: (5, 6),
    8: (7,),
    9: (8,),
    10: (8,),
    11: (10,),
    12: (11,),
    13: (11,),
    14: (13,),
    15: (14,),
    16: (13, 14),
    17: (16,),
    18: (16, 17),
    19: (14,),
    20: (19,),
    21: (15, 16, 17, 18, 19, 20),
    22: (15, 18, 19, 20, 21),
    23: (15, 16, 17, 21, 22),
    24: (16, 17, 18, 19, 20, 21, 22, 23),
}

CONFIG_PATHS = (
    "configs/causal_ml.yaml",
    "configs/error_diagnostics.yaml",
    "configs/event_clustering.yaml",
    "configs/external_controls.yaml",
    "configs/portwatch_event_study.yaml",
    "configs/model_registry.yaml",
    "configs/nonlinear_forecasting.yaml",
    "configs/paths.yaml",
    "configs/publication_outputs.yaml",
    "configs/research_config.yaml",
    "configs/robustness.yaml",
    "configs/scoring_rubric.yaml",
    "configs/significance.yaml",
    "configs/taxonomy.yaml",
)

REQUIRED_IMPORTS = {
    "matplotlib": "matplotlib",
    "newspaper4k": "newspaper",
    "numpy": "numpy",
    "openpyxl": "openpyxl",
    "pandas": "pandas",
    "Prophet": "prophet",
    "PyTorch": "torch",
    "PyYAML": "yaml",
    "SciPy": "scipy",
    "scikit-learn": "sklearn",
    "sentence-transformers": "sentence_transformers",
    "statsmodels": "statsmodels",
    "Tigramite": "tigramite",
    "XGBoost": "xgboost",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def expected_venv() -> Path:
    return (PROJECT_ROOT / ".venv").resolve()


def interpreter_check() -> dict[str, Any]:
    actual_prefix = Path(sys.prefix).resolve()
    expected_prefix = expected_venv()
    return {
        "check": "project_virtual_environment",
        "passed": actual_prefix == expected_prefix,
        "expected": str(expected_prefix),
        "observed": str(actual_prefix),
        "python": str(Path(sys.executable).resolve()),
    }


def script_registry_checks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    numbers = [stage.number for stage in STAGES]
    rows.append(
        {
            "check": "registry_unique_numbers",
            "passed": len(numbers) == len(set(numbers)),
            "details": numbers,
        }
    )
    rows.append(
        {
            "check": "registry_contiguous_00_to_24",
            "passed": numbers == list(range(25)),
            "details": numbers,
        }
    )
    dependency_keys = sorted(STAGE_DEPENDENCIES)
    forward_edges = {
        stage: list(dependencies)
        for stage, dependencies in STAGE_DEPENDENCIES.items()
        if any(dependency >= stage for dependency in dependencies)
    }
    rows.append(
        {
            "check": "dependency_dag_is_complete_and_acyclic",
            "passed": dependency_keys == list(range(25)) and not forward_edges,
            "dependency_keys": dependency_keys,
            "invalid_forward_edges": forward_edges,
        }
    )
    registered = {stage.script for stage in STAGES} | set(OPTIONAL_EXTENSION_SCRIPTS) | {Path(__file__).name}
    observed = {path.name for path in SCRIPTS_DIR.glob("[0-9][0-9]_*.py")}
    rows.append(
        {
            "check": "registered_script_set",
            "passed": observed == registered,
            "missing": sorted(registered - observed),
            "unexpected": sorted(observed - registered),
        }
    )
    for stage in STAGES:
        match = re.fullmatch(r"(\d{2})_[a-z0-9_]+\.py", stage.script)
        rows.append(
            {
                "check": f"stage_{stage.stage_id}_filename",
                "passed": bool(match and int(match.group(1)) == stage.number),
                "path": str(stage.script_path),
            }
        )
    return rows


def compile_script(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "check": f"compile:{path.name}",
        "path": str(path.resolve()),
        "passed": False,
        "todo_placeholder": False,
    }
    if not path.is_file():
        row["error"] = "missing"
        return row
    try:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
        row["passed"] = True
        row["sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
        row["todo_placeholder"] = path.name != Path(__file__).name and "TODO: implement" in source
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def dependency_checks() -> list[dict[str, Any]]:
    return [
        {
            "check": f"dependency:{display_name}",
            "passed": importlib.util.find_spec(import_name) is not None,
            "import_name": import_name,
        }
        for display_name, import_name in REQUIRED_IMPORTS.items()
    ]


def config_checks() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    for relative in CONFIG_PATHS:
        path = resolve_project_path(relative)
        row: dict[str, Any] = {
            "check": f"yaml:{relative}",
            "path": str(path),
            "passed": False,
        }
        if not path.is_file():
            row["error"] = "missing"
            rows.append(row)
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("top level is not a mapping")
            payloads[relative] = payload
            row.update(passed=True, sha256=sha256_file(path))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows, payloads


def integrity_lock_checks() -> list[dict[str, Any]]:
    specifications = (
        (
            "configs/significance_script.sha256",
            "scripts/23_significance_analysis.py",
        ),
        (
            "configs/publication_outputs_script.sha256",
            "scripts/24_make_figures_tables.py",
        ),
    )
    rows: list[dict[str, Any]] = []
    for lock_relative, script_relative in specifications:
        lock = resolve_project_path(lock_relative)
        script = resolve_project_path(script_relative)
        row: dict[str, Any] = {
            "check": f"script_lock:{script_relative}",
            "lock": str(lock),
            "script": str(script),
            "passed": False,
        }
        try:
            fields = lock.read_text(encoding="utf-8").strip().split()
            expected = fields[0].lower()
            if len(expected) != 64 or any(
                character not in "0123456789abcdef" for character in expected
            ):
                raise ValueError("invalid SHA-256 token")
            if len(fields) > 1 and fields[1].replace("\\", "/") != script_relative:
                raise ValueError(
                    f"lock path {fields[1]!r} does not match {script_relative!r}"
                )
            observed = sha256_file(script)
            row.update(
                passed=expected == observed,
                expected_sha256=expected,
                observed_sha256=observed,
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def canonical_config_hash_checks() -> list[dict[str, Any]]:
    specifications = (
        ("configs/error_diagnostics.yaml", "scripts/22_error_diagnostics.py"),
        ("configs/significance.yaml", "scripts/23_significance_analysis.py"),
        ("configs/publication_outputs.yaml", "scripts/24_make_figures_tables.py"),
    )
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r'CANONICAL_CONFIG_SHA256\s*=\s*\(\s*"([0-9a-f]{64})"', re.MULTILINE
    )
    for config_relative, script_relative in specifications:
        config = resolve_project_path(config_relative)
        script = resolve_project_path(script_relative)
        observed = sha256_file(config) if config.is_file() else None
        match = pattern.search(script.read_text(encoding="utf-8")) if script.is_file() else None
        expected = match.group(1) if match else None
        rows.append(
            {
                "check": f"canonical_config_hash:{config_relative}",
                "passed": expected is not None and observed == expected,
                "config": str(config),
                "script": str(script),
                "expected_sha256": expected,
                "observed_sha256": observed,
            }
        )
    return rows


def glm_contract_checks(configs: dict[str, Any]) -> list[dict[str, Any]]:
    nonlinear = configs.get("configs/nonlinear_forecasting.yaml", {})
    glm = nonlinear.get("models", {}).get("glm", {})
    fixed = glm.get("fixed", {}) if isinstance(glm, dict) else {}
    candidates = glm.get("candidates", []) if isinstance(glm, dict) else []
    candidate_alphas = [item.get("alpha") for item in candidates if isinstance(item, dict)]
    publication = configs.get("configs/publication_outputs.yaml", {})
    publication_algorithms = publication.get("forecast_trajectory_figures", {}).get(
        "algorithms", []
    )
    return [
        {
            "check": "glm_gaussian_identity_contract",
            "passed": fixed.get("power") == 0
            and fixed.get("link") == "identity"
            and float(fixed.get("alpha", -1)) >= 0,
            "observed": fixed,
        },
        {
            "check": "glm_tuning_grid",
            "passed": candidate_alphas == [0.001, 0.01, 0.1, 1.0],
            "observed": candidate_alphas,
        },
        {
            "check": "publication_includes_all_forecast_algorithms",
            "passed": set(publication_algorithms)
            == set(EXPECTED_FORECAST_ALGORITHMS),
            "observed": publication_algorithms,
            "expected": list(EXPECTED_FORECAST_ALGORITHMS),
        },
    ]


def manifest_audit(path: Path, deep_hash: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "passed": False,
        "declared_outputs": 0,
        "missing_outputs": [],
        "hash_mismatches": [],
    }
    if not path.is_file():
        return row
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("manifest is not a JSON object")
        outputs = payload.get("outputs", {})
        if not isinstance(outputs, dict):
            outputs = {}
        recorded_hashes = payload.get("output_sha256", {})
        if not isinstance(recorded_hashes, dict):
            recorded_hashes = {}
        missing: list[str] = []
        mismatches: list[str] = []
        for key, raw_output in outputs.items():
            if key in {"manifest", "checkpoint_directory"} or not raw_output:
                continue
            output = resolve_project_path(str(raw_output))
            # External Kaggle manifests retain their original absolute paths.
            # Once the artifact bundle is copied into this project, resolve an
            # unavailable external path to the same-named file beside the
            # imported manifest instead of requiring the Kaggle filesystem.
            if not output.exists():
                imported_output = path.parent / Path(str(raw_output)).name
                if imported_output.exists():
                    output = imported_output.resolve()
            if not output.exists():
                missing.append(f"{key}:{output}")
                continue
            if deep_hash and output.is_file() and key in recorded_hashes:
                if sha256_file(output) != str(recorded_hashes[key]):
                    mismatches.append(f"{key}:{output}")
        row.update(
            passed=not missing and not mismatches,
            version=payload.get("version"),
            status=payload.get("status"),
            publication_scope_complete=payload.get("publication_scope_complete"),
            sha256=sha256_file(path) if deep_hash else None,
            declared_outputs=len(outputs),
            missing_outputs=missing,
            hash_mismatches=mismatches,
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def artifact_checks(deep_hash: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in STAGES:
        for relative in stage.completion_artifacts:
            path = resolve_project_path(relative)
            if path.suffix.lower() == ".json" and "manifest" in path.name:
                audit = manifest_audit(path, deep_hash)
            else:
                audit = {
                    "path": str(path),
                    "exists": path.exists(),
                    "passed": path.exists(),
                }
            audit.update(stage=stage.stage_id, purpose=stage.purpose)
            rows.append(audit)
    return rows


def build_smoke_report(deep_hash: bool = False) -> dict[str, Any]:
    config_rows, configs = config_checks()
    checks = [
        interpreter_check(),
        *script_registry_checks(),
        *(compile_script(stage.script_path) for stage in STAGES),
        *(compile_script(SCRIPTS_DIR / script) for script in OPTIONAL_EXTENSION_SCRIPTS),
        compile_script(Path(__file__).resolve()),
        *dependency_checks(),
        *config_rows,
        *canonical_config_hash_checks(),
        *integrity_lock_checks(),
        *glm_contract_checks(configs),
    ]
    errors = [row for row in checks if not bool(row.get("passed"))]
    placeholders = [row for row in checks if row.get("todo_placeholder")]
    artifacts = artifact_checks(deep_hash)
    return {
        "mode": "smoke",
        "read_only": True,
        "created_at_utc": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "checks_passed": not errors,
        "check_count": len(checks),
        "error_count": len(errors),
        "placeholder_count": len(placeholders),
        "checks": checks,
        "artifact_inventory": artifacts,
        "artifact_inventory_complete": all(row["passed"] for row in artifacts),
        "planned_full_commands": [
            {
                "stage": stage.stage_id,
                "mode": stage.mode,
                "command": stage.command(Path(sys.executable))
                if stage.mode == "automatic"
                else None,
            }
            for stage in STAGES
        ],
    }


def render_smoke_report(report: dict[str, Any]) -> None:
    print(
        "Pipeline smoke check: "
        f"{report['check_count'] - report['error_count']}/{report['check_count']} "
        "structural checks passed"
    )
    for row in report["checks"]:
        if not row.get("passed"):
            print(f"ERROR  {row['check']}: {row.get('error', row.get('observed', 'failed'))}")
        elif row.get("todo_placeholder"):
            print(f"NOTICE {row['check']}: registered structural/manual placeholder")
    present = sum(row["passed"] for row in report["artifact_inventory"])
    total = len(report["artifact_inventory"])
    print(f"Existing completion artifacts: {present}/{total}")
    missing = [row for row in report["artifact_inventory"] if not row["passed"]]
    for row in missing:
        print(f"INFO   stage {row['stage']} output not currently complete: {row['path']}")
    print("Smoke mode did not execute a pipeline stage or write an output.")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    # On Windows, antivirus scanners and file indexers can briefly open the
    # destination without delete sharing.  ``os.replace`` then raises WinError
    # 5 even though the directory ACL permits writes.  Retrying preserves the
    # atomic contract without turning a transient lock into a pipeline failure.
    maximum_attempts = 10
    for attempt in range(maximum_attempts):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt + 1 == maximum_attempts:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(min(0.05 * (2**attempt), 1.0))


def selected_stages(first: int, last: int) -> list[Stage]:
    if first < 0 or last > 24 or first > last:
        raise ValueError("Stage range must satisfy 00 <= from-stage <= to-stage <= 24")
    return [stage for stage in STAGES if first <= stage.number <= last]


def scoring_archives() -> list[Path]:
    root = PROJECT_ROOT / "data/03_processed/model_scores/scores_zips"
    return sorted(root.rglob("*.zip")) if root.is_dir() else []


def validate_annotations(paths: Sequence[Path]) -> list[Path]:
    resolved = [path.resolve() for path in paths]
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Annotation file(s) not found: {missing}")
    return resolved


def full_preflight(args: argparse.Namespace, stages: Sequence[Stage]) -> dict[str, Any]:
    smoke = build_smoke_report(deep_hash=False)
    errors = [row for row in smoke["checks"] if not bool(row.get("passed"))]
    if errors:
        names = [str(row["check"]) for row in errors]
        raise RuntimeError(f"Structural preflight failed: {names}")
    selected_numbers = {stage.number for stage in stages}
    stage_by_number = {stage.number: stage for stage in STAGES}
    missing_prior_artifacts: list[str] = []
    for stage in stages:
        for dependency in STAGE_DEPENDENCIES[stage.number]:
            if dependency in selected_numbers:
                continue
            dependency_stage = stage_by_number[dependency]
            for relative in dependency_stage.completion_artifacts:
                path = resolve_project_path(relative)
                if not path.exists():
                    missing_prior_artifacts.append(
                        f"stage {stage.stage_id} requires stage "
                        f"{dependency_stage.stage_id}: {path}"
                    )
    if missing_prior_artifacts:
        raise RuntimeError(
            "Selected range has incomplete prior-stage dependencies: "
            + "; ".join(missing_prior_artifacts)
        )
    if 10 in selected_numbers and not scoring_archives():
        raise RuntimeError(
            "Stage 10 is an external/manual LLM-scoring gate and no score ZIPs "
            "were found under data/03_processed/model_scores/scores_zips"
        )
    annotations = validate_annotations(args.annotations or [])
    if 12 in selected_numbers and not annotations and not args.allow_pending_human_validation:
        raise RuntimeError(
            "Human validation is not complete. Supply --annotations FILE [FILE ...] "
            "or explicitly use --allow-pending-human-validation to run the "
            "computational pipeline with publication status still pending."
        )
    if args.adjudicated:
        adjudicated = args.adjudicated.resolve()
        if not adjudicated.is_file():
            raise FileNotFoundError(f"Adjudicated labels not found: {adjudicated}")
        if not annotations:
            raise ValueError("--adjudicated requires --annotations")
    return {
        "structural_checks_passed": True,
        "selected_stage_numbers": sorted(selected_numbers),
        "score_archives": len(scoring_archives()),
        "annotations": [str(path) for path in annotations],
        "pending_human_validation_allowed": bool(
            args.allow_pending_human_validation and not annotations
        ),
    }


def stage_command(
    stage: Stage, args: argparse.Namespace, python: Path
) -> tuple[list[str] | None, str]:
    if stage.mode == "structural":
        return None, "covered_by_granular_registered_stages"
    if stage.mode == "manual_scoring":
        return None, f"external_scores_present:{len(scoring_archives())}_archives"
    if stage.mode == "external_kaggle":
        return None, "external_kaggle_artifacts_present"
    if stage.mode == "human_validation":
        if args.annotations:
            command = [
                str(python),
                str(stage.script_path),
                "evaluate",
                "--annotations",
                *(str(path.resolve()) for path in args.annotations),
            ]
            if args.adjudicated:
                command.extend(("--adjudicated", str(args.adjudicated.resolve())))
            return command, "human_validation_evaluation"
        return [str(python), str(stage.script_path), "sample"], "sampling_only_pending_labels"
    return stage.command(python), "automatic"


def validate_stage_completion(stage: Stage, args: argparse.Namespace) -> dict[str, Any]:
    expected = list(stage.completion_artifacts)
    if stage.number == 12 and args.annotations:
        expected = ["data/04_validation/validation_metrics_summary.json"]
    rows = []
    for relative in expected:
        path = resolve_project_path(relative)
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    passed = all(row["exists"] for row in rows)
    if not passed:
        raise RuntimeError(
            f"Stage {stage.stage_id} exited successfully but its completion "
            f"artifact is missing: {[row['path'] for row in rows if not row['exists']]}"
        )
    manifest_rows = [
        manifest_audit(Path(row["path"]), deep_hash=True)
        for row in rows
        if Path(row["path"]).suffix.lower() == ".json"
        and "manifest" in Path(row["path"]).name
    ]
    failed_manifests = [row for row in manifest_rows if not row["passed"]]
    if failed_manifests:
        raise RuntimeError(
            f"Stage {stage.stage_id} manifest audit failed: {failed_manifests}"
        )
    return {"passed": True, "artifacts": rows, "manifest_audits": manifest_rows}


def stream_subprocess(command: Sequence[str], log_handle: Any) -> int:
    process = subprocess.Popen(
        list(command),
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        shell=False,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log_handle.write(line)
        log_handle.flush()
    return int(process.wait())


def run_full(args: argparse.Namespace) -> int:
    stages = selected_stages(args.from_stage, args.to_stage)
    preflight = full_preflight(args, stages)
    python = Path(sys.executable).resolve()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("full_%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("--run-id may contain only letters, numbers, dot, underscore, and dash")
    run_dir = PIPELINE_RUNS_DIR / run_id
    run_manifest_path = run_dir / "pipeline_run.json"
    log_path = run_dir / "pipeline_run.log"
    if run_manifest_path.exists() and not args.resume_pipeline:
        raise FileExistsError(
            f"Pipeline run already exists: {run_manifest_path}. Use --resume-pipeline "
            "or choose another --run-id."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_pipeline:
        if not run_manifest_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume; pipeline run manifest not found: {run_manifest_path}"
            )
        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_id") != run_id or manifest.get("mode") != "full":
            raise ValueError("Existing pipeline manifest does not match this full run")
        completed_ids = {
            str(item.get("stage"))
            for item in manifest.get("stages", [])
            if item.get("status")
            in {"completed", "satisfied_external", "skipped_structural"}
        }
        manifest["status"] = "running"
        manifest["resumed_at_utc"] = utc_now()
        manifest["updated_at_utc"] = utc_now()
    else:
        manifest = {
            "run_id": run_id,
            "mode": "full",
            "status": "running",
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "project_root": str(PROJECT_ROOT),
            "python": str(python),
            "frozen_initial_tuning": True,
            "forecast_algorithms": list(EXPECTED_FORECAST_ALGORITHMS),
            "preflight": preflight,
            "arguments": {
                "from_stage": args.from_stage,
                "to_stage": args.to_stage,
                "annotations": [
                    str(path.resolve()) for path in (args.annotations or [])
                ],
                "adjudicated": (
                    str(args.adjudicated.resolve()) if args.adjudicated else None
                ),
                "allow_pending_human_validation": (
                    args.allow_pending_human_validation
                ),
            },
            "stages": [],
            "log": str(log_path),
        }
        completed_ids = set()
    atomic_write_json(run_manifest_path, manifest)
    with log_path.open("a", encoding="utf-8") as log_handle:
        for stage in stages:
            if stage.stage_id in completed_ids:
                print(f"Stage {stage.stage_id} already completed; resuming after it.")
                continue
            command, execution_role = stage_command(stage, args, python)
            record: dict[str, Any] = {
                "stage": stage.stage_id,
                "script": stage.script,
                "purpose": stage.purpose,
                "execution_role": execution_role,
                "started_at_utc": utc_now(),
                "command": command,
            }
            manifest["stages"].append(record)
            manifest["updated_at_utc"] = utc_now()
            atomic_write_json(run_manifest_path, manifest)
            if command is None:
                record["status"] = (
                    "skipped_structural"
                    if stage.mode == "structural"
                    else "satisfied_external"
                )
                record["completion"] = validate_stage_completion(stage, args) if stage.completion_artifacts else {"passed": True}
                record["finished_at_utc"] = utc_now()
                atomic_write_json(run_manifest_path, manifest)
                print(f"Stage {stage.stage_id}: {execution_role}")
                continue
            print(f"Stage {stage.stage_id}: {stage.purpose}")
            print(subprocess.list2cmdline(command))
            log_handle.write(f"\n[{utc_now()}] stage={stage.stage_id} command={subprocess.list2cmdline(command)}\n")
            started = time.perf_counter()
            try:
                return_code = stream_subprocess(command, log_handle)
            except KeyboardInterrupt:
                record.update(
                    status="interrupted",
                    runtime_seconds=time.perf_counter() - started,
                    finished_at_utc=utc_now(),
                )
                manifest["status"] = "interrupted"
                manifest["updated_at_utc"] = utc_now()
                atomic_write_json(run_manifest_path, manifest)
                raise
            record["return_code"] = return_code
            record["runtime_seconds"] = time.perf_counter() - started
            record["finished_at_utc"] = utc_now()
            if return_code != 0:
                record["status"] = "failed"
                manifest["status"] = "failed"
                manifest["failed_stage"] = stage.stage_id
                manifest["updated_at_utc"] = utc_now()
                atomic_write_json(run_manifest_path, manifest)
                print(f"Pipeline stopped: stage {stage.stage_id} returned {return_code}.")
                return return_code or 1
            try:
                record["completion"] = validate_stage_completion(stage, args)
                record["status"] = "completed"
            except Exception as exc:
                record["status"] = "failed_completion_contract"
                record["error"] = f"{type(exc).__name__}: {exc}"
                manifest["status"] = "failed"
                manifest["failed_stage"] = stage.stage_id
                manifest["updated_at_utc"] = utc_now()
                atomic_write_json(run_manifest_path, manifest)
                raise
            manifest["updated_at_utc"] = utc_now()
            atomic_write_json(run_manifest_path, manifest)
    pending = bool(args.allow_pending_human_validation and not args.annotations)
    manifest["status"] = (
        "computational_complete_pending_human_validation" if pending else "complete"
    )
    manifest["completed_at_utc"] = utc_now()
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(run_manifest_path, manifest)
    print(f"Pipeline status: {manifest['status']}")
    print(f"Run manifest: {run_manifest_path}")
    return 0


def add_shared_full_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--from-stage", type=int, default=0)
    parser.add_argument("--to-stage", type=int, default=24)
    parser.add_argument("--run-id")
    parser.add_argument("--annotations", type=Path, nargs="+")
    parser.add_argument("--adjudicated", type=Path)
    parser.add_argument(
        "--allow-pending-human-validation",
        action="store_true",
        help=(
            "Create/retain the annotation sample and continue computational stages; "
            "the run remains explicitly ineligible for publication."
        ),
    )
    parser.add_argument("--resume-pipeline", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit or full tuned SC-Disrupt pipeline run."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    smoke = subparsers.add_parser("smoke", help="Read-only structural and artifact checks.")
    smoke.add_argument(
        "--deep-hash",
        action="store_true",
        help="Recompute hashes for all manifest-declared outputs (can be slow).",
    )
    full = subparsers.add_parser("full", help="Run every selected stage with full tuning.")
    add_shared_full_arguments(full)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "smoke":
        report = build_smoke_report(deep_hash=args.deep_hash)
        render_smoke_report(report)
        return 0 if report["checks_passed"] else 2
    try:
        return run_full(args)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"Pipeline preflight/error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
