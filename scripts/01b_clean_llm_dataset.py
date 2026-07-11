#!/usr/bin/env python3
"""Clean and combine only the CSV files in the LLM dataset folder.

Input folder by default:
    data/llm_dataset/

Output file by default:
    data/02_interim/news_cleaned/llm_search_results_cleaned.csv

The output contains exactly these columns:
    article_id, link, headline, source_name, date

Article IDs are assigned after cleaning and duplicate removal using:
    llm_news_0000001
    llm_news_0000002
    ...

No earlier/legacy dataset is read or merged.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


# Accept both the original Google-scrape headers and already-renamed headers.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "link": ("zReHs href", "link"),
    "headline": ("LC20lb", "headline"),
    "source_name": ("VuuXrf", "source_name"),
    "date": ("YrbPuc 2", "date"),
}

# Columns read from the input files and used for duplicate detection.
DATA_COLUMNS = ["link", "headline", "source_name", "date"]

# Exact schema written to the final CSV.
OUTPUT_COLUMNS = ["article_id", *DATA_COLUMNS]

NULL_TEXT = {"", "nan", "none", "null", "n/a", "na"}


def project_root() -> Path:
    """Return the repository root when this script is inside scripts/."""
    return Path(__file__).resolve().parents[1]


def normalize_header(header: object) -> str:
    """Normalize a column header for reliable matching."""
    return re.sub(r"[^a-z0-9]", "", str(header).lower())


def find_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    """Find a source column using its possible header names."""
    available = {normalize_header(column): str(column) for column in df.columns}

    for alias in aliases:
        matched = available.get(normalize_header(alias))
        if matched is not None:
            return matched

    return None


def clean_cell(value: object) -> str:
    """Trim whitespace and convert textual null values to an empty string."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if text.lower() in NULL_TEXT:
        return ""

    return text


def read_csv_safely(csv_path: Path) -> pd.DataFrame:
    """Read a CSV while handling common text encodings."""
    last_error: Exception | None = None

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(
                csv_path,
                dtype="string",
                keep_default_na=False,
                encoding=encoding,
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            last_error = error

    raise RuntimeError(f"Could not read {csv_path}: {last_error}")


def select_required_columns(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Keep and rename only link, headline, source_name, and date."""
    selected: dict[str, pd.Series] = {}
    missing: list[str] = []

    for output_column, aliases in COLUMN_ALIASES.items():
        source_column = find_column(df, aliases)

        if source_column is None:
            missing.append(output_column)
        else:
            selected[output_column] = df[source_column]

    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return pd.DataFrame(selected, columns=DATA_COLUMNS)


def assign_article_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Sort records reproducibly and assign sequential LLM article IDs."""
    result = df.copy()

    # Parse dates only for stable sorting. The original cleaned date text is retained.
    result["_sort_date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
        format="mixed",
    )

    result = result.sort_values(
        by=["_sort_date", "headline", "link", "source_name"],
        ascending=[True, True, True, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    result.insert(
        0,
        "article_id",
        [f"llm_news_{number:07d}" for number in range(1, len(result) + 1)],
    )

    return result.drop(columns=["_sort_date"])


def clean_llm_dataset(input_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read, combine, clean, deduplicate, and identify all LLM CSV records."""
    csv_files = sorted(input_dir.glob("*.csv"), key=lambda path: path.name.lower())

    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found in: {input_dir}")

    frames: list[pd.DataFrame] = []
    total_loaded = 0

    for csv_path in csv_files:
        raw_df = read_csv_safely(csv_path)
        selected_df = select_required_columns(raw_df, csv_path)
        frames.append(selected_df)
        total_loaded += len(selected_df)
        print(f"[READ] {csv_path.name}: {len(selected_df):,} rows")

    combined = pd.concat(frames, ignore_index=True)

    # Clean whitespace and textual null values.
    for column in DATA_COLUMNS:
        combined[column] = combined[column].map(clean_cell)

    # Remove rows where all four retained data columns are blank.
    before_blank_removal = len(combined)
    combined = combined.loc[~combined.eq("").all(axis=1)].copy()
    blank_rows_removed = before_blank_removal - len(combined)

    # Remove exact duplicate rows before assigning IDs.
    before_duplicate_removal = len(combined)
    combined = combined.drop_duplicates(subset=DATA_COLUMNS, keep="first")
    duplicate_rows_removed = before_duplicate_removal - len(combined)

    # Assign IDs only to the final clean, unique records.
    combined = assign_article_ids(combined)

    # Enforce the exact final schema and order.
    combined = combined.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)

    stats = {
        "files_read": len(csv_files),
        "rows_loaded": total_loaded,
        "blank_rows_removed": blank_rows_removed,
        "duplicate_rows_removed": duplicate_rows_removed,
        "rows_written": len(combined),
    }

    return combined, stats


def parse_arguments() -> argparse.Namespace:
    root = project_root()

    parser = argparse.ArgumentParser(
        description="Clean and combine only the LLM dataset CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "data" / "llm_dataset",
        help="Folder containing the LLM CSV files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=(
            root
            / "data"
            / "02_interim"
            / "news_cleaned"
            / "llm_search_results_cleaned.csv"
        ),
        help="Path for the cleaned combined CSV.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    input_dir = args.input_dir.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()

    cleaned_df, stats = clean_llm_dataset(input_dir)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    # Verify that no unwanted columns were written.
    written_columns = pd.read_csv(
        output_file,
        nrows=0,
        encoding="utf-8-sig",
    ).columns.tolist()

    if written_columns != OUTPUT_COLUMNS:
        raise RuntimeError(
            f"Unexpected output columns: {written_columns}. "
            f"Expected: {OUTPUT_COLUMNS}"
        )

    if cleaned_df["article_id"].duplicated().any():
        raise RuntimeError("Duplicate article_id values were generated.")

    print("\n[DONE]")
    print(f"Input folder: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Files read: {stats['files_read']:,}")
    print(f"Rows loaded: {stats['rows_loaded']:,}")
    print(f"Blank rows removed: {stats['blank_rows_removed']:,}")
    print(f"Duplicate rows removed: {stats['duplicate_rows_removed']:,}")
    print(f"Rows written: {stats['rows_written']:,}")
    print(f"Verified columns: {', '.join(OUTPUT_COLUMNS)}")
    print(
        "Article ID range: "
        f"{cleaned_df['article_id'].iloc[0] if not cleaned_df.empty else 'none'} "
        "to "
        f"{cleaned_df['article_id'].iloc[-1] if not cleaned_df.empty else 'none'}"
    )


if __name__ == "__main__":
    main()