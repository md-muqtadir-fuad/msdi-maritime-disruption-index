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
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_INPUT = Path(
    "data/03_processed/news_full_text/merged_full_text_unique.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/03_processed/news_full_text/pre_llm_filter"
)

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
        "output_files": {
            "kept": str(kept_path),
            "discarded": str(discarded_path),
            "summary": str(summary_path),
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
    print(f"Summary file:   {summary_path}")


if __name__ == "__main__":
    main()
