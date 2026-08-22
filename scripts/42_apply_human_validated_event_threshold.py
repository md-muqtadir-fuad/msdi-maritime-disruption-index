#!/usr/bin/env python3
"""Apply a human-validated event threshold without recomputing embeddings.

The Stage-08 pair audit stores every within-window cosine similarity at or
above 0.80.  Therefore a threshold at or above 0.80 can be replayed exactly.
Before applying a new threshold, this script replays the currently active
threshold and requires an exact match with the existing Kaggle event map.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msdi.preprocessing.event_clustering import (  # noqa: E402
    _event_identifier,
    greedy_complete_linkage_from_pair_audit,
    threshold_flag_column,
)


DEFAULT_EVENT_DIR = (
    PROJECT_ROOT
    / "data/03_processed/news_full_text/pre_llm_filter/event_dedup"
)
DEFAULT_ARTICLES = (
    PROJECT_ROOT / "data/03_processed/news_full_text/pre_llm_filter/pre_llm_kept.csv"
)
DEFAULT_THRESHOLD_MATRIX = (
    PROJECT_ROOT / "validation/event_pairs/results/event_threshold_evaluation_matrix.csv"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs/event_clustering.yaml"
PROFILE_DEFAULTS = {
    "weekly": (5, "weeks"),
    "monthly": (3, "months"),
}
ARTIFACT_NAMES = (
    "canonical_event_article_map.csv",
    "canonical_event_catalog_by_profile.csv",
    "canonical_event_pair_audit_by_profile.csv",
    "canonical_event_manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--event-dir", type=Path, default=DEFAULT_EVENT_DIR)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--threshold-matrix", type=Path, default=DEFAULT_THRESHOLD_MATRIX)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate replay and construct results in memory without writing files.",
    )
    return parser.parse_args()


def event_identifier_for_undated(article_id: str) -> str:
    digest = sha256(article_id.encode("utf-8")).hexdigest()[:12]
    return f"EVT_UNDATED_{digest}"


def load_threshold_evidence(path: Path, threshold: float) -> dict[str, Any]:
    matrix = pd.read_csv(path)
    required = {"threshold", "sample_f1_score"}
    missing = sorted(required.difference(matrix.columns))
    if missing:
        raise ValueError(f"Threshold evaluation matrix lacks columns: {missing}")
    matrix["threshold"] = pd.to_numeric(matrix["threshold"], errors="raise")
    selected = matrix[np.isclose(matrix["threshold"], threshold, rtol=0.0, atol=1e-12)]
    if len(selected) != 1:
        raise ValueError(f"Expected one human-validation row for threshold {threshold}")
    row = selected.iloc[0]
    maximum_f1 = float(pd.to_numeric(matrix["sample_f1_score"], errors="raise").max())
    if not np.isclose(float(row["sample_f1_score"]), maximum_f1, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Threshold {threshold} is not the sample-F1 selection in {path}"
        )
    return {
        column: (value.item() if isinstance(value, np.generic) else value)
        for column, value in row.to_dict().items()
    }


def profile_settings(manifest: dict[str, Any], profile: str) -> tuple[int, str]:
    configured = manifest.get("profiles", {}).get(profile, {}).get("configuration", {})
    default_value, default_unit = PROFILE_DEFAULTS[profile]
    return (
        int(configured.get("window_value", default_value)),
        str(configured.get("window_unit", default_unit)),
    )


def replay_profile(
    work: pd.DataFrame,
    pair_audit: pd.DataFrame,
    profile: str,
    threshold: float,
    window_value: int,
    window_unit: str,
    audit_floor: float,
) -> tuple[dict[str, str], list[list[int]]]:
    profile_pairs = pair_audit.loc[pair_audit["event_profile"].eq(profile)].copy()
    clusters = greedy_complete_linkage_from_pair_audit(
        work,
        profile_pairs,
        threshold,
        window_value,
        window_unit,
        audit_floor,
    )
    assignments: dict[str, str] = {}
    for members in clusters:
        event_id = _event_identifier(members, work)
        for member in members:
            assignments[str(work.at[member, "article_id"])] = event_id
    return assignments, clusters


def exact_replay_mismatches(
    mapping: pd.DataFrame,
    assignments: dict[str, str],
    profile: str,
) -> int:
    observed = mapping.set_index("article_id")[f"event_id_{profile}"].astype(str)
    mismatches = 0
    for article_id, observed_event in observed.items():
        expected = assignments.get(str(article_id))
        if expected is None:
            expected = event_identifier_for_undated(str(article_id))
        mismatches += int(expected != observed_event)
    return mismatches


def build_profile_outputs(
    mapping: pd.DataFrame,
    articles: pd.DataFrame,
    assignments: dict[str, str],
    profile: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    frame = mapping[["article_id", "event_day"]].copy()
    frame["article_id"] = frame["article_id"].astype(str)
    frame["event_day"] = pd.to_datetime(frame["event_day"], errors="coerce")
    frame["event_id"] = frame["article_id"].map(assignments)
    undated = frame["event_day"].isna()
    frame.loc[undated, "event_id"] = frame.loc[undated, "article_id"].map(
        event_identifier_for_undated
    )
    if frame["event_id"].isna().any():
        raise ValueError(f"{profile} reconstruction left blank event IDs")

    metadata_columns = ["article_id", "retrieved_title", "source_domain"]
    missing_metadata = sorted(set(metadata_columns).difference(articles.columns))
    if missing_metadata:
        raise ValueError(f"Article file lacks catalog metadata: {missing_metadata}")
    metadata = articles[metadata_columns].copy()
    metadata["article_id"] = metadata["article_id"].astype(str)
    if metadata["article_id"].duplicated().any():
        raise ValueError("Article metadata contains duplicate article IDs")
    frame = frame.merge(metadata, on="article_id", how="left", validate="one_to_one")
    if frame["retrieved_title"].isna().all() or frame["source_domain"].isna().all():
        raise ValueError("Article metadata did not align with the event map")

    grouped = frame.groupby("event_id", sort=False)
    frame["event_cluster_size"] = grouped["article_id"].transform("nunique").astype(int)
    frame["event_source_count"] = grouped["source_domain"].transform("nunique").astype(int)
    frame["event_span_days"] = grouped["event_day"].transform(
        lambda values: int((values.max() - values.min()).days)
        if values.notna().any()
        else 0
    ).astype(int)
    frame["event_weight_fixed_total_descriptive"] = 1.0 / frame[
        "event_cluster_size"
    ].clip(lower=1)
    frame["event_weight_sqrt_descriptive"] = 1.0 / np.sqrt(
        frame["event_cluster_size"].clip(lower=1)
    )

    output_map = frame[
        [
            "article_id",
            "event_day",
            "event_id",
            "event_cluster_size",
            "event_source_count",
            "event_span_days",
            "event_weight_fixed_total_descriptive",
            "event_weight_sqrt_descriptive",
        ]
    ].copy()
    output_map["event_day"] = output_map["event_day"].dt.strftime("%Y-%m-%d")
    output_map = output_map.rename(
        columns={
            column: f"{column}_{profile}"
            for column in output_map.columns
            if column not in {"article_id", "event_day"}
        }
    )

    catalog_rows: list[dict[str, Any]] = []
    for event_id, group in frame.groupby("event_id", sort=True):
        dates = group["event_day"].dropna()
        titles = group["retrieved_title"].fillna("").astype(str).str.strip()
        titles = titles[titles.ne("")]
        catalog_rows.append(
            {
                "event_id": event_id,
                "start_day": dates.min().date().isoformat() if not dates.empty else "",
                "end_day": dates.max().date().isoformat() if not dates.empty else "",
                "event_span_days": int(group["event_span_days"].max()),
                "article_count": int(group["article_id"].nunique()),
                "source_count": int(group["source_domain"].nunique()),
                "representative_title": titles.iloc[0] if not titles.empty else "",
                "article_ids": " | ".join(group["article_id"].astype(str)),
                "event_profile": profile,
            }
        )
    catalog = pd.DataFrame(catalog_rows)
    counts = {
        "articles": int(len(frame)),
        "dated_articles": int(frame["event_day"].notna().sum()),
        "undated_singletons": int(frame["event_day"].isna().sum()),
        "events": int(frame["event_id"].nunique()),
        "multi_article_events": int(catalog["article_count"].gt(1).sum()),
        "articles_in_multi_article_events": int(
            frame["event_cluster_size"].gt(1).sum()
        ),
    }
    return output_map, catalog, counts


def update_pair_audit(
    pair_audit: pd.DataFrame,
    profile_assignments: dict[str, dict[str, str]],
    threshold: float,
) -> pd.DataFrame:
    output = pair_audit.copy()
    for profile, assignments in profile_assignments.items():
        mask = output["event_profile"].eq(profile)
        output.loc[mask, "event_id_a"] = output.loc[mask, "article_id_a"].astype(str).map(
            assignments
        )
        output.loc[mask, "event_id_b"] = output.loc[mask, "article_id_b"].astype(str).map(
            assignments
        )
    if output[["event_id_a", "event_id_b"]].isna().any().any():
        raise ValueError("Reconstructed pair audit contains blank event IDs")
    output["same_event_cluster"] = output["event_id_a"].eq(output["event_id_b"]).astype(int)
    output["passes_similarity_threshold"] = output["cosine_similarity"].ge(threshold).astype(int)
    output[threshold_flag_column(threshold)] = output["cosine_similarity"].ge(threshold).astype(int)
    return output


def archive_inputs(event_dir: Path, current_threshold: float) -> Path:
    label = str(current_threshold).replace(".", "_")
    archive_dir = event_dir / "archive" / f"threshold_{label}_before_human_calibration"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_NAMES:
        source = event_dir / name
        target = archive_dir / name
        if not target.exists():
            shutil.copy2(source, target)
    return archive_dir


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".recluster.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".recluster.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    event_dir = args.event_dir.resolve()
    artifact_paths = {name: event_dir / name for name in ARTIFACT_NAMES}
    required_paths = [*artifact_paths.values(), args.articles, args.threshold_matrix, args.config]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required reconstruction inputs are missing: {missing}")

    evidence = load_threshold_evidence(args.threshold_matrix, args.threshold)
    config_payload = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    manifest = json.loads(artifact_paths["canonical_event_manifest.json"].read_text(encoding="utf-8"))
    current_threshold = float(manifest["selected_threshold_provisional"])
    audit_floor = min(
        float(manifest["profiles"][profile]["configuration"]["pair_audit_min_similarity"])
        for profile in PROFILE_DEFAULTS
    )
    if args.threshold < audit_floor:
        raise ValueError(
            f"Threshold {args.threshold} is below saved pair-audit floor {audit_floor}; "
            "embeddings would be required."
        )

    mapping = pd.read_csv(artifact_paths["canonical_event_article_map.csv"], low_memory=False)
    pair_audit = pd.read_csv(
        artifact_paths["canonical_event_pair_audit_by_profile.csv"], low_memory=False
    )
    articles = pd.read_csv(args.articles, low_memory=False)
    if len(mapping) != mapping["article_id"].nunique():
        raise ValueError("Active event map is not one-to-one by article ID")
    if set(mapping["article_id"].astype(str)) != set(articles["article_id"].astype(str)):
        raise ValueError("Active event map and retained article cohort differ")

    work = mapping.loc[mapping["event_day"].notna(), ["article_id", "event_day"]].copy()
    work["article_id"] = work["article_id"].astype(str)
    work["event_day"] = pd.to_datetime(work["event_day"], errors="raise")
    work = work.sort_values(["event_day", "article_id"], kind="stable").reset_index(drop=True)

    profile_assignments: dict[str, dict[str, str]] = {}
    profile_maps: dict[str, pd.DataFrame] = {}
    catalogs: list[pd.DataFrame] = []
    replay_mismatches: dict[str, int] = {}
    new_counts: dict[str, dict[str, int]] = {}
    for profile in PROFILE_DEFAULTS:
        window_value, window_unit = profile_settings(manifest, profile)
        current_assignments, _ = replay_profile(
            work,
            pair_audit,
            profile,
            current_threshold,
            window_value,
            window_unit,
            audit_floor,
        )
        mismatches = exact_replay_mismatches(mapping, current_assignments, profile)
        replay_mismatches[profile] = mismatches
        if mismatches:
            raise ValueError(
                f"Saved pair audit failed exact {current_threshold} replay for {profile}: "
                f"{mismatches} mismatches"
            )

        assignments, _ = replay_profile(
            work,
            pair_audit,
            profile,
            args.threshold,
            window_value,
            window_unit,
            audit_floor,
        )
        profile_assignments[profile] = assignments
        profile_map, catalog, counts = build_profile_outputs(
            mapping, articles, assignments, profile
        )
        profile_maps[profile] = profile_map
        catalogs.append(catalog)
        counts["candidate_pairs"] = int(pair_audit["event_profile"].eq(profile).sum())
        counts["pair_validation_sample"] = int(
            manifest["profiles"][profile]["counts"].get("pair_validation_sample", 0)
        )
        new_counts[profile] = counts

    new_map = profile_maps["weekly"].merge(
        profile_maps["monthly"].drop(columns="event_day"),
        on="article_id",
        how="outer",
        validate="one_to_one",
    )
    order = {article_id: index for index, article_id in enumerate(mapping["article_id"].astype(str))}
    new_map["_order"] = new_map["article_id"].map(order)
    new_map = new_map.sort_values("_order", kind="stable").drop(columns="_order")
    new_catalog = pd.concat(catalogs, ignore_index=True, sort=False)
    new_pairs = update_pair_audit(pair_audit, profile_assignments, args.threshold)

    candidates = sorted(
        set(float(value) for value in manifest.get("threshold_candidates_for_human_validation", []))
        | {float(args.threshold)}
    )
    manifest["selected_threshold_provisional"] = float(args.threshold)
    manifest["selected_threshold_final"] = float(args.threshold)
    manifest["version"] = str(config_payload.get("version", "event_clustering_v001"))
    manifest["status"] = "complete_human_validated"
    manifest["publication_scope_complete"] = True
    manifest["threshold_candidates_for_human_validation"] = candidates
    manifest["threshold_selection"] = {
        "status": "completed_human_validated",
        "criterion": "maximum_sample_f1_on_adjudicated_event_pairs",
        "evidence_path": str(args.threshold_matrix.resolve()),
        "adjudicated_pairs": 300,
        "selected_row": evidence,
    }
    for profile, counts in new_counts.items():
        configuration = manifest["profiles"][profile]["configuration"]
        configuration["similarity_threshold"] = float(args.threshold)
        configuration["threshold_candidates"] = candidates
        manifest["profiles"][profile]["human_validation_threshold_candidates"] = candidates
        manifest["profiles"][profile]["counts"] = counts
        guardrails = manifest["profiles"][profile].setdefault("guardrails", {})
        guardrails["threshold_human_calibration_completed"] = True
        guardrails["embeddings_recomputed_for_threshold_application"] = False
    manifest["reconstruction"] = {
        "method": "exact_pair_audit_replay_of_date_ordered_greedy_complete_linkage",
        "previous_threshold": current_threshold,
        "applied_threshold": float(args.threshold),
        "pair_audit_similarity_floor": audit_floor,
        "previous_threshold_replay_mismatches": replay_mismatches,
        "embeddings_recomputed": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
    }
    manifest["guardrail"] = "final threshold selected only from adjudicated human event-pair labels"
    manifest["configuration_file"] = str(args.config.resolve())
    manifest["outputs"] = {
        "article_event_map": str(artifact_paths["canonical_event_article_map.csv"].resolve()),
        "event_catalog": str(artifact_paths["canonical_event_catalog_by_profile.csv"].resolve()),
        "pair_audit": str(artifact_paths["canonical_event_pair_audit_by_profile.csv"].resolve()),
        "pair_validation_template": str(
            (event_dir / "canonical_event_pair_validation_template.csv").resolve()
        ),
        "pair_validation_annotator_1": str(
            (event_dir / "canonical_event_pair_validation_annotator_1.csv").resolve()
        ),
        "pair_validation_annotator_2": str(
            (event_dir / "canonical_event_pair_validation_annotator_2.csv").resolve()
        ),
        "human_threshold_evaluation": str(args.threshold_matrix.resolve()),
        "reconstruction_audit": str(
            (event_dir / "event_threshold_reconstruction_audit.json").resolve()
        ),
        "manifest": str(artifact_paths["canonical_event_manifest.json"].resolve()),
    }

    reconstruction_audit = {
        "status": "check_only" if args.check_only else "applied",
        "active_threshold_before": current_threshold,
        "selected_threshold": float(args.threshold),
        "human_validation_evidence": evidence,
        "pair_audit_similarity_floor": audit_floor,
        "exact_replay_mismatches": replay_mismatches,
        "counts": new_counts,
        "embeddings_recomputed": False,
    }
    print(json.dumps(reconstruction_audit, indent=2, ensure_ascii=False))
    if args.check_only:
        return

    archive_dir = archive_inputs(event_dir, current_threshold)
    atomic_write_csv(new_map, artifact_paths["canonical_event_article_map.csv"])
    atomic_write_csv(new_catalog, artifact_paths["canonical_event_catalog_by_profile.csv"])
    atomic_write_csv(new_pairs, artifact_paths["canonical_event_pair_audit_by_profile.csv"])
    atomic_write_json(manifest, artifact_paths["canonical_event_manifest.json"])
    atomic_write_json(reconstruction_audit, event_dir / "event_threshold_reconstruction_audit.json")
    print(f"Archived previous artifacts: {archive_dir}")
    print(f"Applied human-validated threshold {args.threshold} without embedding inference.")


if __name__ == "__main__":
    main()
