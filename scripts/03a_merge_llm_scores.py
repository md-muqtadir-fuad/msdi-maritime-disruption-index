"""Merge batched LLM score files stored in ZIP archives.

By default, archives are read from
``data/03_processed/model_scores/scores_zips`` without extracting them. The
merged dataset and a JSON audit report are written beside that directory.
CSV, JSON Lines, and JSON members are supported.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/03_processed/model_scores/scores_zips"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/03_processed/model_scores/llm_scores_merged.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LLM score files from ZIP archives.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        dest="input_dirs",
        help="Directory containing ZIPs; repeat for multiple models. Searched recursively.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--id-column",
        default="article_id",
        help="Identifier used to detect conflicting repeated records.",
    )
    return parser.parse_args()


def read_member(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    suffix = Path(member).suffix.lower()
    raw = archive.read(member)
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(raw))
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(io.BytesIO(raw), lines=True)
    if suffix == ".json":
        value = json.loads(raw.decode("utf-8-sig"))
        if isinstance(value, dict):
            for key in ("results", "records", "data", "items"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list):
            raise ValueError("JSON must be a record list or contain results/records/data/items")
        return pd.DataFrame(value)
    raise ValueError(f"Unsupported file type: {suffix}")


def is_score_member(member: str) -> bool:
    """Return true only for final article-level score tables, not ZIP diagnostics."""
    name = Path(member).name.lower()
    return (
        (name.startswith("scored_chunk_") and name.endswith(".csv"))
        or ("article_level_scores_valid" in name and name.endswith(".csv"))
    )


def merge_archives(input_dirs: list[Path], id_column: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    archives = sorted(
        {archive.resolve() for directory in input_dirs for archive in directory.rglob("*.zip")},
        key=lambda path: str(path).lower(),
    )
    if not archives:
        searched = ", ".join(str(path) for path in input_dirs)
        raise FileNotFoundError(f"No ZIP files found under: {searched}")

    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    members_read = 0
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            for member in sorted(archive.namelist()):
                if not is_score_member(member):
                    continue
                try:
                    frame = read_member(archive, member)
                    frame["_source_zip"] = archive_path.name
                    frame["_source_file"] = member
                    frame["_source_row"] = range(1, len(frame) + 1)
                    frames.append(frame)
                    members_read += 1
                except Exception as exc:  # continue so one bad member does not lose a batch
                    errors.append({"zip": archive_path.name, "member": member, "error": str(exc)})

    if not frames:
        raise ValueError("ZIP files contained no recognized final score tables")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    source_columns = ["_source_zip", "_source_file", "_source_row"]
    value_columns = [column for column in combined.columns if column not in source_columns]
    rows_before = len(combined)
    combined = combined.drop_duplicates(subset=value_columns, keep="first").reset_index(drop=True)

    conflicts = pd.DataFrame()
    key_columns = [id_column]
    if "model_name" in combined.columns:
        key_columns.insert(0, "model_name")
    if id_column in combined.columns:
        ids = combined[id_column].astype("string").str.strip()
        valid_id = ids.notna() & ids.ne("")
        repeated = valid_id & combined.duplicated(subset=key_columns, keep=False)
        conflicts = combined.loc[repeated].sort_values(key_columns, kind="stable")

    audit = {
        "input_directories": [str(path) for path in input_dirs],
        "archives_found": len(archives),
        "members_read": members_read,
        "rows_before_exact_deduplication": rows_before,
        "exact_duplicate_rows_removed": rows_before - len(combined),
        "merged_rows": len(combined),
        "columns": list(combined.columns),
        "id_column": id_column,
        "record_key": key_columns,
        "rows_with_conflicting_repeated_ids": len(conflicts),
        "read_errors": errors,
    }
    return combined, conflicts, audit


def main() -> None:
    args = parse_args()
    input_dirs = args.input_dirs or [DEFAULT_INPUT]
    input_dirs = [path.resolve() for path in input_dirs]
    merged, conflicts, audit = merge_archives(input_dirs, args.id_column)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)

    conflict_path = output.with_name(f"{output.stem}_conflicts.csv")
    audit_path = output.with_name(f"{output.stem}_audit.json")
    if not conflicts.empty:
        conflicts.to_csv(conflict_path, index=False)
    audit["output_file"] = str(output)
    audit["conflicts_file"] = str(conflict_path) if not conflicts.empty else None
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"Merged {audit['merged_rows']:,} rows into {output}")
    print(f"Audit: {audit_path}")
    if not conflicts.empty:
        print(f"Warning: {len(conflicts):,} rows have repeated {args.id_column}; see {conflict_path}")


if __name__ == "__main__":
    main()
