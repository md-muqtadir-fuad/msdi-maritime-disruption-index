#!/usr/bin/env python3
"""Merge legacy and current full-text retrieval datasets with duplicate auditing.

Duplicate rules (conservative):
1. Same normalized article text -> duplicate.
2. Same canonical article URL -> duplicate.

The script keeps one best-quality row from every duplicate group and writes:
- merged_full_text_unique.csv
- full_text_duplicates.csv
- full_text_merge_summary.json

Run this script from the project root, or pass explicit paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd


DEFAULT_LEGACY = Path(
    "data/02_interim/article_enrichment/full_text_news4k/"
    "full_text_news4k_results.csv"
)
DEFAULT_CURRENT = Path(
    "data/02_interim/article_enrichment/full_text_llm_news4k/"
    "llm_full_text_news4k_succeeded.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/03_processed/news_full_text")

TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "dclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "igshid",
    "vero_id",
    "ref_src",
    "ref_url",
}


class UnionFind:
    """Small disjoint-set implementation for transitive duplicate grouping."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return

        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def normalize_text(value: object) -> str:
    """Normalize text enough to detect exact-content duplicates safely."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def sha256_text(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonicalize_url(value: object) -> str:
    """Create a conservative canonical URL while preserving meaningful query data."""
    if pd.isna(value):
        return ""

    raw = str(value).strip()
    if not raw:
        return ""
    if not re.match(r"^https?://", raw, flags=re.IGNORECASE):
        raw = "https://" + raw

    try:
        parsed = urlsplit(raw)
        hostname = (parsed.hostname or "").lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        if not hostname:
            return ""

        port = parsed.port
        if port and not (
            (parsed.scheme.lower() == "http" and port == 80)
            or (parsed.scheme.lower() == "https" and port == 443)
        ):
            hostname = f"{hostname}:{port}"

        path = re.sub(r"/{2,}", "/", parsed.path or "/")
        if path != "/":
            path = path.rstrip("/")

        filtered_query = []
        for key, val in parse_qsl(parsed.query, keep_blank_values=True):
            key_lower = key.lower()
            if key_lower.startswith("utm_") or key_lower in TRACKING_PARAMETERS:
                continue
            filtered_query.append((key, val))

        query = urlencode(sorted(filtered_query), doseq=True)
        # Treat HTTP and HTTPS versions of the same article as one URL.
        return urlunsplit(("https", hostname, path, query, ""))
    except (ValueError, TypeError):
        return raw.casefold()


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    """Return the first available candidate column, or an empty string Series."""
    for name in candidates:
        if name in df.columns:
            return df[name]
    return pd.Series("", index=df.index, dtype="object")


def load_and_standardize(path: Path, collection: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    original_rows = len(df)

    if "retrieved_text" not in df.columns:
        raise ValueError(f"{path} has no 'retrieved_text' column.")

    # Keep only genuinely successful full-text records.
    status = first_existing_column(df, ["retrieval_status"]).astype(str).str.casefold()
    has_text = df["retrieved_text"].fillna("").astype(str).str.strip().ne("")
    if "retrieval_status" in df.columns:
        df = df.loc[status.eq("success") & has_text].copy()
    else:
        df = df.loc[has_text].copy()

    df["collection"] = collection
    df["input_file"] = str(path)
    df["input_row_number"] = df.index + 2  # CSV header is row 1.
    df["source_url"] = first_existing_column(df, ["url", "link", "normalized_link"])

    final_url = first_existing_column(df, ["final_url"])
    final_url = final_url.fillna("").astype(str).str.strip()
    source_url = df["source_url"].fillna("").astype(str).str.strip()
    df["preferred_url"] = final_url.where(final_url.ne(""), source_url)
    df["canonical_url"] = df["preferred_url"].map(canonicalize_url)

    df["normalized_text_hash"] = (
        df["retrieved_text"].map(normalize_text).map(sha256_text)
    )

    # Recalculate objective quality measures instead of trusting stale values.
    df["calculated_text_length"] = df["retrieved_text"].fillna("").astype(str).str.len()
    df["calculated_word_count"] = (
        df["retrieved_text"].fillna("").astype(str).str.findall(r"\b\w+\b").str.len()
    )
    df["input_rows_before_filter"] = original_rows
    return df.reset_index(drop=True)


def nonempty_flag(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0, index=df.index, dtype="int64")
    return df[column].fillna("").astype(str).str.strip().ne("").astype("int64")


def add_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """Score records so the most complete row survives within each duplicate group."""
    df = df.copy()
    metadata_columns = [
        "retrieved_title",
        "headline",
        "date",
        "retrieved_publish_date",
        "retrieved_authors",
        "source_name",
        "final_url",
        "retrieved_meta_description",
    ]
    metadata_score = sum((nonempty_flag(df, col) for col in metadata_columns))

    # Text length dominates; metadata and successful current retrieval break ties.
    df["quality_score"] = (
        df["calculated_text_length"].fillna(0).astype(float)
        + metadata_score.astype(float) * 100.0
        + nonempty_flag(df, "canonical_url") * 50.0
        + (df["collection"].eq("current")).astype(float) * 1.0
    )
    return df


def union_same_nonempty_key(df: pd.DataFrame, key: str, uf: UnionFind) -> int:
    """Union rows sharing a non-empty key; return duplicate-group count for that key."""
    duplicate_group_count = 0
    for value, indices in df.groupby(key, sort=False).groups.items():
        if not value or len(indices) < 2:
            continue
        duplicate_group_count += 1
        indices = list(indices)
        anchor = int(indices[0])
        for idx in indices[1:]:
            uf.union(anchor, int(idx))
    return duplicate_group_count


def build_duplicate_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    uf = UnionFind(len(df))
    text_groups = union_same_nonempty_key(df, "normalized_text_hash", uf)
    url_groups = union_same_nonempty_key(df, "canonical_url", uf)

    roots = [uf.find(i) for i in range(len(df))]
    root_to_group: dict[int, str] = {}
    next_number = 1
    group_ids = []
    for root in roots:
        if root not in root_to_group:
            root_to_group[root] = f"dup_group_{next_number:06d}"
            next_number += 1
        group_ids.append(root_to_group[root])

    out = df.copy()
    out["duplicate_group_id"] = group_ids
    out["duplicate_group_size"] = out.groupby("duplicate_group_id")[
        "duplicate_group_id"
    ].transform("size")

    stats = {
        "normalized_text_duplicate_key_groups": text_groups,
        "canonical_url_duplicate_key_groups": url_groups,
        "connected_duplicate_groups": int(
            out.loc[out["duplicate_group_size"] > 1, "duplicate_group_id"].nunique()
        ),
    }
    return out, stats


def select_winners(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = df.sort_values(
        by=[
            "duplicate_group_id",
            "quality_score",
            "calculated_text_length",
            "retrieved_at_utc" if "retrieved_at_utc" in df.columns else "collection",
            "article_id" if "article_id" in df.columns else "collection",
        ],
        ascending=[True, False, False, False, True],
        kind="stable",
        na_position="last",
    ).copy()

    winners = ranked.groupby("duplicate_group_id", sort=False).head(1).copy()
    winners["deduplication_status"] = "kept"

    winner_lookup = winners.set_index("duplicate_group_id")
    duplicates = ranked.loc[ranked["duplicate_group_size"] > 1].copy()
    duplicates["kept_article_id"] = duplicates["duplicate_group_id"].map(
        winner_lookup["article_id"]
        if "article_id" in winner_lookup.columns
        else pd.Series("", index=winner_lookup.index)
    )
    duplicates["kept_collection"] = duplicates["duplicate_group_id"].map(
        winner_lookup["collection"]
    )
    duplicates["kept_normalized_text_hash"] = duplicates["duplicate_group_id"].map(
        winner_lookup["normalized_text_hash"]
    )
    duplicates["kept_canonical_url"] = duplicates["duplicate_group_id"].map(
        winner_lookup["canonical_url"]
    )

    same_text = duplicates["normalized_text_hash"].eq(
        duplicates["kept_normalized_text_hash"]
    ) & duplicates["normalized_text_hash"].ne("")
    same_url = duplicates["canonical_url"].eq(
        duplicates["kept_canonical_url"]
    ) & duplicates["canonical_url"].ne("")

    duplicates["duplicate_reason"] = "transitive_match"
    duplicates.loc[same_url, "duplicate_reason"] = "same_canonical_url"
    duplicates.loc[same_text, "duplicate_reason"] = "same_normalized_text"
    duplicates.loc[same_text & same_url, "duplicate_reason"] = (
        "same_normalized_text_and_canonical_url"
    )

    winner_index = set(winners.index)
    duplicates["deduplication_status"] = [
        "kept" if idx in winner_index else "removed_duplicate" for idx in duplicates.index
    ]

    return winners, duplicates


def order_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "article_id",
        "collection",
        "headline",
        "retrieved_title",
        "date",
        "retrieved_publish_date",
        "source_name",
        "source_domain",
        "source_url",
        "final_url",
        "preferred_url",
        "canonical_url",
        "retrieved_text",
        "calculated_text_length",
        "calculated_word_count",
        "retrieved_text_hash",
        "normalized_text_hash",
        "retrieval_method",
        "retrieved_at_utc",
        "duplicate_group_id",
        "duplicate_group_size",
        "quality_score",
        "deduplication_status",
        "input_file",
        "input_row_number",
    ]
    first = [col for col in preferred if col in df.columns]
    rest = [col for col in df.columns if col not in first]
    return df[first + rest]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge legacy and current full-text CSVs and remove duplicates."
    )
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    legacy = load_and_standardize(args.legacy, "legacy")
    current = load_and_standardize(args.current, "current")
    combined = pd.concat([legacy, current], ignore_index=True, sort=False)
    combined = add_quality_score(combined)
    combined, duplicate_stats = build_duplicate_groups(combined)
    winners, duplicate_audit = select_winners(combined)

    # Stable final ordering for research reproducibility.
    sort_columns = [col for col in ["date", "article_id"] if col in winners.columns]
    if sort_columns:
        winners = winners.sort_values(sort_columns, kind="stable", na_position="last")

    merged_path = args.output_dir / "merged_full_text_unique.csv"
    duplicates_path = args.output_dir / "full_text_duplicates.csv"
    summary_path = args.output_dir / "full_text_merge_summary.json"

    order_output_columns(winners).to_csv(merged_path, index=False, encoding="utf-8-sig")
    order_output_columns(duplicate_audit).to_csv(
        duplicates_path, index=False, encoding="utf-8-sig"
    )

    duplicate_rows_removed = int(len(combined) - len(winners))
    summary = {
        "legacy_input_file": str(args.legacy),
        "current_input_file": str(args.current),
        "legacy_successful_full_text_rows": int(len(legacy)),
        "current_successful_full_text_rows": int(len(current)),
        "combined_successful_full_text_rows_before_deduplication": int(len(combined)),
        "unique_rows_after_deduplication": int(len(winners)),
        "duplicate_rows_removed": duplicate_rows_removed,
        "duplicate_rows_in_audit_including_kept_representatives": int(
            len(duplicate_audit)
        ),
        **duplicate_stats,
        "duplicate_rules": [
            "same normalized full text SHA-256",
            "same canonical URL after removing common tracking parameters",
        ],
        "merged_output": str(merged_path),
        "duplicates_output": str(duplicates_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
