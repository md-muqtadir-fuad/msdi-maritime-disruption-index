#!/usr/bin/env python3
"""Remove totally irrelevant articles before LLM classification.

The filter is intentionally permissive:
- KEEP an article if its title or full text contains at least one maritime,
  shipping, port, vessel, freight, cargo, or supply-chain keyword.
- DISCARD it only if the text is empty or no keyword is found.

The script preserves every original column and adds:
- matched_maritime_keywords
- pre_llm_filter_decision
- pre_llm_filter_reason

By default, retained articles are then assigned model-independent weekly and
monthly semantic event IDs with the stronger BAAI/bge-m3 encoder. Event
matching does not read LLM scores or SCFI and does not delete independently
written coverage.  It exports a blinded pair-validation sample because the
semantic threshold must be human-calibrated before publication.

Default input:
    data/03_processed/news_full_text/merged_full_text_unique.csv

Default outputs:
    data/03_processed/news_full_text/pre_llm_filter/pre_llm_kept.csv
    data/03_processed/news_full_text/pre_llm_filter/pre_llm_discarded.csv
    data/03_processed/news_full_text/pre_llm_filter/pre_llm_filter_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from msdi.preprocessing.event_clustering import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_THRESHOLD_CANDIDATES,
    EventClusteringConfig,
    cluster_event_articles,
    embed_event_articles,
)


DEFAULT_INPUT = Path(
    "data/03_processed/news_full_text/merged_full_text_unique.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/03_processed/news_full_text/pre_llm_filter"
)
DEFAULT_EVENT_CONFIG = PROJECT_ROOT / "configs/event_clustering.yaml"

# This list is deliberately broad. The purpose is only to remove articles that
# are clearly unrelated before the stricter LLM relevance classification.
MARITIME_KEYWORDS = [
    # Core shipping and maritime terms
    "ship",
    "ships",
    "shipping",
    "shipowner",
    "shipowners",
    "shipbuilding",
    "shipbuilder",
    "shipbuilders",
    "shipyard",
    "shipyards",
    "vessel",
    "vessels",
    "maritime",
    "marine transport",
    "merchant marine",
    "merchant fleet",
    "seaborne",
    "ocean transport",
    "ocean shipping",
    "sea transport",
    "shipping industry",
    "shipping company",
    "shipping companies",
    "shipping line",
    "shipping lines",
    "ocean carrier",
    "ocean carriers",

    # Vessel types and operations
    "container ship",
    "container ships",
    "containership",
    "containerships",
    "cargo ship",
    "cargo ships",
    "freighter",
    "freighters",
    "tanker",
    "tankers",
    "oil tanker",
    "oil tankers",
    "chemical tanker",
    "chemical tankers",
    "gas carrier",
    "gas carriers",
    "lng carrier",
    "lng carriers",
    "lpg carrier",
    "lpg carriers",
    "bulk carrier",
    "bulk carriers",
    "dry bulk",
    "reefer vessel",
    "reefer vessels",
    "ro ro",
    "roll on roll off",
    "ferry",
    "ferries",
    "cruise ship",
    "cruise ships",
    "barge",
    "barges",
    "tugboat",
    "tugboats",
    "dredger",
    "dredgers",
    "fleet",
    "fleets",
    "carrier",
    "carriers",

    # Ports and port operations
    "port",
    "ports",
    "seaport",
    "seaports",
    "harbor",
    "harbors",
    "harbour",
    "harbours",
    "terminal",
    "terminals",
    "container terminal",
    "container terminals",
    "port terminal",
    "port terminals",
    "dock",
    "docks",
    "dockyard",
    "dockyards",
    "berth",
    "berths",
    "quay",
    "quays",
    "wharf",
    "wharves",
    "anchorage",
    "anchorages",
    "port authority",
    "port authorities",
    "port operator",
    "port operators",
    "dockworker",
    "dockworkers",
    "longshoreman",
    "longshoremen",
    "stevedore",
    "stevedores",

    # Cargo, containers, freight, and logistics
    "cargo",
    "cargoes",
    "freight",
    "ocean freight",
    "sea freight",
    "freight rate",
    "freight rates",
    "shipping rate",
    "shipping rates",
    "charter rate",
    "charter rates",
    "container",
    "containers",
    "containerized",
    "containerised",
    "container traffic",
    "container trade",
    "teu",
    "feu",
    "bill of lading",
    "bills of lading",
    "shipment",
    "shipments",
    "liner",
    "liners",
    "logistics",
    "supply chain",
    "supply chains",

    # Routes, waterways, and maritime geography
    "shipping route",
    "shipping routes",
    "sea route",
    "sea routes",
    "shipping lane",
    "shipping lanes",
    "sea lane",
    "sea lanes",
    "waterway",
    "waterways",
    "canal",
    "canals",
    "strait",
    "straits",
    "maritime chokepoint",
    "maritime chokepoints",
    "suez canal",
    "panama canal",
    "red sea",
    "bab el mandeb",
    "strait of hormuz",
    "hormuz strait",
    "strait of malacca",
    "malacca strait",
    "bosporus",
    "bosphorus",
    "dardanelles",
    "black sea shipping",

    # Maritime personnel, security, and disruption terms
    "seafarer",
    "seafarers",
    "mariner",
    "mariners",
    "coast guard",
    "naval blockade",
    "maritime security",
    "maritime trade",
    "piracy",
    "pirate attack",
    "pirate attacks",
    "vessel attack",
    "vessel attacks",
    "ship attack",
    "ship attacks",
    "port congestion",
    "port closure",
    "port closures",
    "port strike",
    "port strikes",
    "dock strike",
    "dock strikes",
    "blank sailing",
    "blank sailings",
    "container shortage",
    "container shortages",
    "shipping disruption",
    "shipping disruptions",
    "maritime disruption",
    "maritime disruptions",
    "supply chain disruption",
    "supply chain disruptions",

    # Common maritime market indicators
    "baltic dry index",
    "baltic freight index",
    "shanghai containerized freight index",
    "shanghai containerised freight index",
    "container freight index",
    "scfi",
]

TITLE_COLUMNS = [
    "retrieved_title",
    "headline",
    "title",
    "headline_clean",
    "article_title",
]

TEXT_COLUMNS = [
    "retrieved_text",
    "full_text",
    "article_text",
    "text",
    "content",
    "body",
]


def normalize_for_matching(value: object) -> str:
    """Normalize punctuation and whitespace while preserving word boundaries."""
    if value is None or pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_keywords(keywords: Iterable[str]) -> list[str]:
    """Normalize and deduplicate keywords, retaining longest phrases first."""
    normalized = {
        normalize_for_matching(keyword)
        for keyword in keywords
        if normalize_for_matching(keyword)
    }
    return sorted(normalized, key=lambda item: (-len(item.split()), -len(item), item))


def compile_keyword_pattern(keywords: Iterable[str]) -> re.Pattern[str]:
    """Compile one whole-word pattern for all normalized words and phrases."""
    alternatives = [re.escape(keyword) for keyword in keywords]
    if not alternatives:
        raise ValueError("MARITIME_KEYWORDS cannot be empty.")

    # Text and keywords are normalized to letters/numbers/underscores plus spaces.
    # These boundaries prevent 'ship' from matching 'leadership', for example.
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)")


def available_columns(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [column for column in candidates if column in df.columns]


def combine_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Combine selected columns row by row without converting missing values to 'nan'."""
    if not columns:
        return pd.Series("", index=df.index, dtype="object")

    cleaned = [df[column].fillna("").astype(str).str.strip() for column in columns]
    combined = cleaned[0]
    for series in cleaned[1:]:
        combined = combined.str.cat(series, sep=" ")
    return combined.str.strip()


def find_matches(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Return unique matched terms in their first-occurrence order."""
    if not text:
        return []
    return list(dict.fromkeys(match.group(0) for match in pattern.finditer(text)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep articles containing at least one maritime/shipping keyword and "
            "discard only empty or clearly unrelated articles."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input merged CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--kept-name",
        default="pre_llm_kept.csv",
        help="Filename for retained articles.",
    )
    parser.add_argument(
        "--discarded-name",
        default="pre_llm_discarded.csv",
        help="Filename for discarded articles.",
    )
    parser.add_argument(
        "--skip-event-clustering",
        action="store_true",
        help="Run only keyword filtering (not valid for the publication-final index).",
    )
    parser.add_argument("--event-config", type=Path, default=DEFAULT_EVENT_CONFIG)
    parser.add_argument("--event-model", default=None)
    parser.add_argument(
        "--event-model-revision",
        default=None,
        help="Optional Hugging Face commit/tag used to pin the encoder.",
    )
    parser.add_argument(
        "--event-similarity-threshold",
        type=float,
        default=None,
    )
    parser.add_argument("--event-weekly-window-weeks", type=int, default=None)
    parser.add_argument("--event-monthly-window-months", type=int, default=None)
    parser.add_argument("--event-batch-size", type=int, default=None)
    parser.add_argument("--event-device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument(
        "--event-torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default=None,
    )
    parser.add_argument(
        "--event-local-files-only", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--event-max-sequence-length", type=int, default=None)
    parser.add_argument("--event-maximum-body-characters", type=int, default=None)
    parser.add_argument("--event-validation-sample-size", type=int, default=None)
    parser.add_argument("--event-seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}\n"
            "Run the merge script first or pass the correct path with --input."
        )

    df = pd.read_csv(args.input, low_memory=False)
    if df.empty:
        raise ValueError(f"Input file contains no rows: {args.input}")

    title_columns = available_columns(df, TITLE_COLUMNS)
    text_columns = available_columns(df, TEXT_COLUMNS)

    if not title_columns and not text_columns:
        raise ValueError(
            "No usable title or text column was found. Expected one of: "
            f"{TITLE_COLUMNS + TEXT_COLUMNS}"
        )

    title_text = combine_columns(df, title_columns)
    body_text = combine_columns(df, text_columns)
    combined_raw = title_text.str.cat(body_text, sep=" ").str.strip()
    combined_normalized = combined_raw.map(normalize_for_matching)

    normalized_keywords = normalize_keywords(MARITIME_KEYWORDS)
    keyword_pattern = compile_keyword_pattern(normalized_keywords)

    matches = combined_normalized.map(lambda text: find_matches(text, keyword_pattern))
    is_empty = combined_normalized.eq("")
    has_keyword = matches.map(bool)
    keep_mask = (~is_empty) & has_keyword

    result = df.copy()
    result["matched_maritime_keywords"] = matches.map(lambda items: " | ".join(items))
    result["pre_llm_filter_decision"] = keep_mask.map({True: "keep", False: "discard"})
    result["pre_llm_filter_reason"] = ""
    result.loc[is_empty, "pre_llm_filter_reason"] = "empty_title_and_text"
    result.loc[(~is_empty) & (~has_keyword), "pre_llm_filter_reason"] = (
        "no_maritime_keyword_found"
    )
    result.loc[keep_mask, "pre_llm_filter_reason"] = "maritime_keyword_found"

    kept = result.loc[keep_mask].copy()
    discarded = result.loc[~keep_mask].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    kept_path = args.output_dir / args.kept_name
    discarded_path = args.output_dir / args.discarded_name
    summary_path = args.output_dir / "pre_llm_filter_summary.json"

    event_outputs: dict[str, str] = {}
    event_audit: dict[str, object] | None = None
    if not args.skip_event_clustering:
        if not args.event_config.is_file():
            raise FileNotFoundError(f"Event-clustering config not found: {args.event_config}")
        raw_event_config = yaml.safe_load(args.event_config.read_text(encoding="utf-8")) or {}
        settings = raw_event_config.get("event_clustering", raw_event_config)
        model_settings = settings.get("model", {})
        matching_settings = settings.get("matching", {})
        validation_settings = settings.get("validation", {})

        def chosen(cli_value: object, configured: object, fallback: object) -> object:
            return cli_value if cli_value is not None else (
                configured if configured is not None else fallback
            )

        profile_settings = matching_settings.get("profiles", {})
        weekly_settings = profile_settings.get("weekly", {})
        monthly_settings = profile_settings.get("monthly", {})
        threshold_candidates = tuple(
            float(value)
            for value in validation_settings.get(
                "threshold_candidates", DEFAULT_THRESHOLD_CANDIDATES
            )
        )
        selected_threshold = float(
            chosen(
                args.event_similarity_threshold,
                matching_settings.get("selected_threshold"),
                DEFAULT_SIMILARITY_THRESHOLD,
            )
        )
        total_validation_size = int(
            chosen(
                args.event_validation_sample_size,
                validation_settings.get("pair_sample_size_total"),
                400,
            )
        )

        common_config = {
            "model_name": str(chosen(args.event_model, model_settings.get("name"), DEFAULT_MODEL)),
            "model_revision": chosen(
                args.event_model_revision, model_settings.get("revision"), None
            ),
            "similarity_threshold": selected_threshold,
            "threshold_candidates": threshold_candidates,
            "batch_size": int(chosen(args.event_batch_size, model_settings.get("batch_size"), 16)),
            "device": str(chosen(args.event_device, model_settings.get("device"), "auto")),
            "torch_dtype": str(
                chosen(args.event_torch_dtype, model_settings.get("torch_dtype"), "auto")
            ),
            "local_files_only": bool(
                chosen(args.event_local_files_only, model_settings.get("local_files_only"), False)
            ),
            "max_sequence_length": int(
                chosen(args.event_max_sequence_length, model_settings.get("max_sequence_length"), 2048)
            ),
            "maximum_body_characters": int(
                chosen(
                    args.event_maximum_body_characters,
                    model_settings.get("maximum_body_characters"),
                    12000,
                )
            ),
            "pair_audit_min_similarity": float(
                validation_settings.get("pair_audit_min_similarity", 0.80)
            ),
            "random_seed": int(chosen(args.event_seed, validation_settings.get("seed"), 20260810)),
        }
        profile_sample_size = int((total_validation_size + 1) // 2)
        weekly_config = EventClusteringConfig(
            **common_config,
            window_value=int(
                chosen(
                    args.event_weekly_window_weeks,
                    weekly_settings.get("window_value"),
                    5,
                )
            ),
            window_unit=str(weekly_settings.get("window_unit", "weeks")),
            validation_sample_size=profile_sample_size,
        )
        monthly_config = EventClusteringConfig(
            **common_config,
            window_value=int(
                chosen(
                    args.event_monthly_window_months,
                    monthly_settings.get("window_value"),
                    3,
                )
            ),
            window_unit=str(monthly_settings.get("window_unit", "months")),
            validation_sample_size=profile_sample_size,
        )

        # Model inference is the expensive step. Embed once, then apply both
        # temporal profiles to the same frozen vectors.
        event_embeddings, event_model = embed_event_articles(kept, weekly_config)
        profile_results: dict[str, tuple] = {}
        for profile_name, profile_config in (
            ("weekly", weekly_config),
            ("monthly", monthly_config),
        ):
            profile_results[profile_name] = cluster_event_articles(
                kept,
                profile_config,
                model=event_model,
                embeddings=event_embeddings,
            )

        profile_maps: list[pd.DataFrame] = []
        catalogs: list[pd.DataFrame] = []
        pair_audits: list[pd.DataFrame] = []
        pair_validations: list[pd.DataFrame] = []
        profile_audits: dict[str, object] = {}
        for profile_name, result_tuple in profile_results.items():
            _, raw_map, raw_catalog, raw_pairs, raw_validation, raw_audit = result_tuple
            renamed_map = raw_map.rename(
                columns={
                    column: f"{column}_{profile_name}"
                    for column in raw_map.columns
                    if column not in {"article_id", "event_day"}
                }
            )
            profile_maps.append(renamed_map)
            catalogs.append(raw_catalog.assign(event_profile=profile_name))
            pair_audits.append(raw_pairs.assign(event_profile=profile_name))
            pair_validations.append(raw_validation.assign(event_profile=profile_name))
            profile_audits[profile_name] = raw_audit

        event_map = profile_maps[0].merge(
            profile_maps[1].drop(columns="event_day"),
            on="article_id",
            how="outer",
            validate="one_to_one",
        )
        event_columns = [
            column
            for column in event_map.columns
            if column != "article_id" and column in kept.columns
        ]
        kept = kept.drop(columns=event_columns, errors="ignore").merge(
            event_map, on="article_id", how="left", validate="one_to_one"
        )
        event_catalog = pd.concat(catalogs, ignore_index=True, sort=False)
        pair_audit = pd.concat(pair_audits, ignore_index=True, sort=False)
        pair_validation = (
            pd.concat(pair_validations, ignore_index=True, sort=False)
            .drop_duplicates(["article_id_a", "article_id_b", "event_profile"])
            .head(total_validation_size)
            .reset_index(drop=True)
        )
        event_audit = {
            "selected_threshold_provisional": selected_threshold,
            "threshold_candidates_for_human_validation": list(threshold_candidates),
            "profiles": profile_audits,
            "embedding_reused_across_profiles": True,
            "guardrail": "final threshold must be selected from human labels only",
        }
        event_dir = args.output_dir / "event_dedup"
        event_dir.mkdir(parents=True, exist_ok=True)
        event_paths = {
            "article_event_map": event_dir / "canonical_event_article_map.csv",
            "event_catalog": event_dir / "canonical_event_catalog_by_profile.csv",
            "pair_audit": event_dir / "canonical_event_pair_audit_by_profile.csv",
            "pair_validation_template": event_dir / "canonical_event_pair_validation_template.csv",
            "pair_validation_annotator_1": event_dir / "canonical_event_pair_validation_annotator_1.csv",
            "pair_validation_annotator_2": event_dir / "canonical_event_pair_validation_annotator_2.csv",
            "manifest": event_dir / "canonical_event_manifest.json",
        }
        event_map.to_csv(event_paths["article_event_map"], index=False, encoding="utf-8-sig")
        event_catalog.to_csv(event_paths["event_catalog"], index=False, encoding="utf-8-sig")
        pair_audit.to_csv(event_paths["pair_audit"], index=False, encoding="utf-8-sig")
        pair_validation.to_csv(
            event_paths["pair_validation_template"], index=False, encoding="utf-8-sig"
        )
        blinded_drop = {
            "cosine_similarity",
            "event_id_a",
            "event_id_b",
            "same_event_cluster",
            "passes_similarity_threshold",
            "closest_candidate_threshold",
        }
        blinded_drop.update(
            column for column in pair_validation.columns
            if column.startswith("passes_threshold_")
        )
        blinded_validation = pair_validation.drop(
            columns=sorted(blinded_drop), errors="ignore"
        )
        for annotator_number in (1, 2):
            annotator_template = blinded_validation.sample(
                frac=1.0,
                random_state=int(common_config["random_seed"]) + annotator_number,
            ).reset_index(drop=True)
            annotator_template["reviewer_id"] = f"event_annotator_{annotator_number}"
            annotator_template.to_csv(
                event_paths[f"pair_validation_annotator_{annotator_number}"],
                index=False,
                encoding="utf-8-sig",
            )
        event_audit["outputs"] = {key: str(path) for key, path in event_paths.items()}
        event_audit["configuration_file"] = str(args.event_config.resolve())
        event_paths["manifest"].write_text(
            json.dumps(event_audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        event_outputs = {key: str(path) for key, path in event_paths.items()}

    kept.to_csv(kept_path, index=False)
    discarded.to_csv(discarded_path, index=False)

    keyword_counts: Counter[str] = Counter()
    for article_matches in matches.loc[keep_mask]:
        keyword_counts.update(article_matches)

    summary = {
        "input_file": str(args.input),
        "output_directory": str(args.output_dir),
        "input_rows": int(len(result)),
        "kept_rows": int(len(kept)),
        "discarded_rows": int(len(discarded)),
        "empty_title_and_text_rows": int(is_empty.sum()),
        "no_keyword_rows": int(((~is_empty) & (~has_keyword)).sum()),
        "kept_percentage": round((len(kept) / len(result)) * 100, 2),
        "discarded_percentage": round((len(discarded) / len(result)) * 100, 2),
        "title_columns_used": title_columns,
        "text_columns_used": text_columns,
        "keyword_count": len(normalized_keywords),
        "most_common_matches": dict(keyword_counts.most_common(30)),
        "event_clustering_enabled": not args.skip_event_clustering,
        "event_clustering": event_audit,
        "output_files": {
            "kept": str(kept_path),
            "discarded": str(discarded_path),
            "summary": str(summary_path),
            **event_outputs,
        },
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("Pre-LLM relevance filter completed.")
    print(f"Input rows:     {len(result):,}")
    print(f"Kept rows:      {len(kept):,}")
    print(f"Discarded rows: {len(discarded):,}")
    print(f"Kept file:      {kept_path}")
    print(f"Discarded file: {discarded_path}")
    if event_audit is not None:
        weekly_events = event_audit["profiles"]["weekly"]["counts"]["events"]
        monthly_events = event_audit["profiles"]["monthly"]["counts"]["events"]
        print(f"Canonical events: weekly={weekly_events:,} monthly={monthly_events:,}")
        print(f"Event map:        {event_outputs['article_event_map']}")
    print(f"Summary file:   {summary_path}")


if __name__ == "__main__":
    main()
