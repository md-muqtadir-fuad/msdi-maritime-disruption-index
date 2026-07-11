#!/usr/bin/env python3
"""Normalize legacy headline/news datasets into one clean table.

Run from project root:
    python scripts/01_normalize_legacy_news.py

Optional:
    python scripts/01_normalize_legacy_news.py --project-root . --include-backup

Inputs expected, if present:
    data/01_raw/news/headlines/final_dataset_with_links_legacy.xlsx
    data/01_raw/news/headlines/final_dataset_headline_only_backup.xlsx
    data/01_raw/news/headlines/manually_collected/*.xlsx
    data/03_processed/model_scores/classified_headlines_legacy.csv

Outputs:
    data/02_interim/news_cleaned/legacy_news_normalized.csv
    data/02_interim/news_cleaned/legacy_news_unique_for_retrieval.csv
    data/02_interim/news_cleaned/legacy_news_duplicates.csv
    data/02_interim/news_cleaned/legacy_news_missing_report.csv
    data/02_interim/news_cleaned/legacy_news_source_summary.csv
    outputs/reports/legacy_news_normalization_report.md
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# -----------------------------
# Path helpers
# -----------------------------

def default_project_root() -> Path:
    # scripts/01_normalize_legacy_news.py -> project root
    return Path(__file__).resolve().parents[1]


def existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]


# -----------------------------
# Cleaning helpers
# -----------------------------

def normalize_column_name(col: object) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^a-z0-9_]+", "", col)
    return col.strip("_")


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {normalize_column_name(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_column_name(cand)
        if key in normalized:
            return normalized[key]
    return None


def clean_headline(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"^[\s\-–—•·:]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def headline_key(value: object) -> str:
    text = clean_headline(value).lower()
    text = re.sub(r"[’']", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sha1_short(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def normalize_url(value: object, headline: object = "") -> str:
    if pd.isna(value):
        return ""
    url = str(value).strip()
    url = re.sub(r"\s+", "", url)
    if not url or url.lower() in {"nan", "none", "null", "-"}:
        return ""
    # Manual files sometimes put the headline text inside Link/link column.
    if headline_key(url) and headline_key(url) == headline_key(headline):
        return ""
    if not re.match(r"^https?://", url, flags=re.I):
        return ""
    return url


def parse_legacy_date(value: object) -> pd.Timestamp | pd.NaT:
    if value is None or pd.isna(value):
        return pd.NaT

    # Already datetime-like
    if isinstance(value, pd.Timestamp):
        return value.normalize()

    # Excel/pandas may read YYYYMMDD as int/float, e.g. 20200901.0
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        number = int(value)
        s = str(number)
        if len(s) == 8 and 19000101 <= number <= 21001231:
            parsed = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
            return parsed.normalize() if pd.notna(parsed) else pd.NaT
        # Excel serial fallback
        if 20000 <= number <= 60000:
            parsed = pd.to_datetime(number, unit="D", origin="1899-12-30", errors="coerce")
            return parsed.normalize() if pd.notna(parsed) else pd.NaT

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "-"}:
        return pd.NaT
    s = re.sub(r"^[\s\-–—•·]+", "", s).strip()
    s = re.sub(r"\.0$", "", s)

    if re.fullmatch(r"\d{8}", s):
        parsed = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        return parsed.normalize() if pd.notna(parsed) else pd.NaT

    # Try broad pandas parsing for strings such as "September 1, 2020" or "2021-01-03".
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.notna(parsed):
        return parsed.normalize()

    parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return parsed.normalize() if pd.notna(parsed) else pd.NaT


# -----------------------------
# Loading helpers
# -----------------------------

def normalize_raw_frame(
    df: pd.DataFrame,
    source_file: str,
    source_sheet: str = "",
    source_type: str = "legacy_raw",
    default_source_dataset: str = "",
) -> pd.DataFrame:
    """Convert one raw file/sheet to the project news schema."""
    if df.empty:
        return pd.DataFrame()

    headline_col = first_existing_column(df, [
        "headline", "headlines", "news_headlines", "news headline", "News headlines",
        "title", "news_title", "text",
    ])
    date_col = first_existing_column(df, ["date", "Date", "published_date", "publication_date", "time"])
    url_col = first_existing_column(df, ["link", "Link", "url", "URL", "article_url", "source_url"])
    source_dataset_col = first_existing_column(df, ["source_dataset", "dataset", "source", "source_name"])

    if headline_col is None:
        return pd.DataFrame()

    # Preserve the input index so scalar metadata fills all rows correctly.
    out = pd.DataFrame(index=df.index)
    out["source_file"] = source_file
    out["source_sheet"] = source_sheet
    out["source_type"] = source_type
    out["source_dataset"] = (
        df[source_dataset_col].astype(str).replace({"nan": ""})
        if source_dataset_col else default_source_dataset
    )
    out["headline"] = df[headline_col].map(clean_headline)
    out["date_raw"] = df[date_col] if date_col else ""
    out["date"] = out["date_raw"].map(parse_legacy_date)
    raw_url = df[url_col] if url_col else pd.Series([""] * len(df), index=df.index)
    out["url"] = [normalize_url(u, h) for u, h in zip(raw_url, out["headline"])]

    # Remove empty/junk headline rows.
    out = out[out["headline"].str.len() > 0].copy()
    out = out[~out["headline"].str.fullmatch(r"(?i)nan|none|null")].copy()
    return out.reset_index(drop=True)


def load_excel_all_sheets(path: Path, source_type: str, default_source_dataset: str = "") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    try:
        xls = pd.ExcelFile(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not open Excel file: {path} ({exc})")
        return pd.DataFrame()

    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not read sheet {sheet!r} in {path}: {exc}")
            continue
        norm = normalize_raw_frame(
            df,
            source_file=str(path),
            source_sheet=sheet,
            source_type=source_type,
            default_source_dataset=default_source_dataset or path.stem,
        )
        if not norm.empty:
            frames.append(norm)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_sources(project_root: Path, include_backup: bool) -> tuple[pd.DataFrame, dict[str, int]]:
    headline_dir = project_root / "data/01_raw/news/headlines"
    manual_dir = headline_dir / "manually_collected"

    main_candidates = existing([
        headline_dir / "final_dataset_with_links_legacy.xlsx",
        headline_dir / "FinalDataset.xlsx",
        headline_dir / "final_dataset_legacy.xlsx",
    ])
    backup_candidates = existing([
        headline_dir / "final_dataset_headline_only_backup.xlsx",
        headline_dir / "final_dataset_headline_only.xlsx",
    ])
    manual_candidates = sorted(manual_dir.glob("*.xlsx")) if manual_dir.exists() else []

    frames: list[pd.DataFrame] = []
    source_counts: dict[str, int] = {}

    for path in main_candidates:
        frame = load_excel_all_sheets(path, source_type="main_headline_with_links", default_source_dataset="main_legacy")
        if not frame.empty:
            frames.append(frame)
            source_counts[str(path)] = len(frame)

    # Backup is intentionally optional because it duplicates the main file but lacks URLs.
    if include_backup or not frames:
        for path in backup_candidates:
            frame = load_excel_all_sheets(path, source_type="headline_only_backup", default_source_dataset="backup_legacy")
            if not frame.empty:
                frames.append(frame)
                source_counts[str(path)] = len(frame)

    for path in manual_candidates:
        frame = load_excel_all_sheets(path, source_type="manual_collected", default_source_dataset=path.stem)
        if not frame.empty:
            frames.append(frame)
            source_counts[str(path)] = len(frame)

    if not frames:
        raise FileNotFoundError(
            "No usable headline Excel files found. Expected files under "
            f"{headline_dir} and {manual_dir}."
        )
    return pd.concat(frames, ignore_index=True), source_counts


def load_legacy_scores(project_root: Path) -> pd.DataFrame:
    candidates = existing([
        project_root / "data/03_processed/model_scores/classified_headlines_legacy.csv",
        project_root / "data/03_processed/model_scores/classified_headlines.csv",
    ])
    if not candidates:
        return pd.DataFrame()

    path = candidates[0]
    scores = pd.read_csv(path)
    headline_col = first_existing_column(scores, ["headline", "title"])
    date_col = first_existing_column(scores, ["date", "published_date", "publication_date"])
    if headline_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["headline_key"] = scores[headline_col].map(headline_key)
    out["date"] = scores[date_col].map(parse_legacy_date) if date_col else pd.NaT

    rename = {
        "score": "legacy_score",
        "themes": "legacy_themes",
        "dimension": "legacy_dimension",
        "reasoning": "legacy_reasoning",
    }
    for src, dst in rename.items():
        if src in scores.columns:
            out[dst] = scores[src]
        else:
            out[dst] = np.nan

    out = out.drop_duplicates(subset=["headline_key", "date"], keep="first")
    return out


# -----------------------------
# Main normalization
# -----------------------------

def assign_priority(row: pd.Series) -> int:
    """Lower is better. Used to choose one row per duplicate group for full-text retrieval."""
    if bool(row["has_url"]) and row["source_type"] == "main_headline_with_links":
        return 1
    if bool(row["has_url"]) and row["source_type"] == "manual_collected":
        return 2
    if bool(row["has_url"]):
        return 3
    if row["source_type"] == "main_headline_with_links":
        return 4
    if row["source_type"] == "manual_collected":
        return 5
    return 9


def build_normalized(project_root: Path, include_backup: bool, max_valid_date: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw, source_counts = load_sources(project_root, include_backup=include_backup)

    raw["headline_clean"] = raw["headline"].map(clean_headline)
    raw["headline_key"] = raw["headline_clean"].map(headline_key)
    raw["headline_hash"] = raw["headline_key"].map(lambda x: sha1_short(x) if x else "")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    max_date_ts = pd.to_datetime(max_valid_date, errors="coerce") if max_valid_date else pd.NaT
    raw["is_suspicious_future_date"] = False
    if pd.notna(max_date_ts):
        raw["is_suspicious_future_date"] = raw["date"].notna() & (raw["date"] > max_date_ts)
    raw["date_str"] = raw["date"].dt.strftime("%Y-%m-%d")
    raw["url"] = [normalize_url(u, h) for u, h in zip(raw["url"], raw["headline_clean"])]
    raw["has_url"] = raw["url"].str.len() > 0
    raw["source_name"] = ""

    # Duplicate logic
    raw["duplicate_key_headline"] = raw["headline_hash"]
    raw["duplicate_key_headline_date"] = raw["headline_hash"] + "__" + raw["date_str"].fillna("missing_date")
    raw["is_duplicate_headline"] = raw.duplicated("duplicate_key_headline", keep=False)
    raw["is_duplicate_headline_date"] = raw.duplicated("duplicate_key_headline_date", keep=False)
    raw["record_priority"] = raw.apply(assign_priority, axis=1)

    # Stable article_id after sorting.
    raw = raw.sort_values(
        by=["record_priority", "date", "headline_key", "source_file", "source_sheet"],
        ascending=[True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    raw["article_id"] = [f"legacy_news_{i:07d}" for i in range(1, len(raw) + 1)]

    # Merge old LLM scores where available.
    scores = load_legacy_scores(project_root)
    if not scores.empty:
        raw = raw.merge(scores, on=["headline_key", "date"], how="left")
    else:
        raw["legacy_score"] = np.nan
        raw["legacy_themes"] = np.nan
        raw["legacy_dimension"] = np.nan
        raw["legacy_reasoning"] = np.nan

    # Keep one best row per headline+date for retrieval.
    unique = raw.sort_values(["duplicate_key_headline_date", "record_priority", "has_url"], ascending=[True, True, False])
    unique = unique.drop_duplicates("duplicate_key_headline_date", keep="first").copy()
    unique["recommended_for_fulltext_retrieval"] = unique["has_url"]

    duplicates = raw[raw["is_duplicate_headline_date"] | raw["is_duplicate_headline"]].copy()

    missing_rows = []
    checks = {
        "missing_headline": raw["headline_clean"].eq(""),
        "missing_date": raw["date"].isna(),
        "missing_url": ~raw["has_url"],
        "duplicate_headline": raw["is_duplicate_headline"],
        "duplicate_headline_date": raw["is_duplicate_headline_date"],
        "missing_legacy_score": raw["legacy_score"].isna(),
        "suspicious_future_date": raw["is_suspicious_future_date"],
    }
    for issue, mask in checks.items():
        missing_rows.append({
            "issue": issue,
            "count": int(mask.sum()),
            "share": float(mask.mean()) if len(raw) else 0.0,
        })
    missing_report = pd.DataFrame(missing_rows)

    source_summary = (
        raw.groupby(["source_type", "source_file"], dropna=False)
        .agg(
            rows=("article_id", "size"),
            unique_headline_date=("duplicate_key_headline_date", "nunique"),
            rows_with_url=("has_url", "sum"),
            missing_date=("date", lambda x: int(x.isna().sum())),
            suspicious_future_date=("is_suspicious_future_date", "sum"),
        )
        .reset_index()
    )
    source_summary["loaded_rows_before_cleaning"] = source_summary["source_file"].map(source_counts).fillna(source_summary["rows"])

    # Final column order
    ordered_cols = [
        "article_id", "source_file", "source_sheet", "source_type", "source_dataset",
        "headline", "headline_clean", "headline_key", "headline_hash",
        "date_raw", "date", "date_str", "url", "has_url", "source_name",
        "duplicate_key_headline", "duplicate_key_headline_date",
        "is_duplicate_headline", "is_duplicate_headline_date", "is_suspicious_future_date", "record_priority",
        "legacy_score", "legacy_themes", "legacy_dimension", "legacy_reasoning",
    ]
    raw = raw[[c for c in ordered_cols if c in raw.columns]]
    unique = unique[[c for c in ordered_cols + ["recommended_for_fulltext_retrieval"] if c in unique.columns]]
    return raw, unique, duplicates, missing_report, source_summary


def write_report(
    project_root: Path,
    normalized: pd.DataFrame,
    unique: pd.DataFrame,
    duplicates: pd.DataFrame,
    missing_report: pd.DataFrame,
    source_summary: pd.DataFrame,
) -> Path:
    report_dir = project_root / "outputs/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "legacy_news_normalization_report.md"

    date_min = normalized["date"].min()
    date_max = normalized["date"].max()
    with_url = int(normalized["has_url"].sum())
    unique_with_url = int(unique["has_url"].sum())
    legacy_matched = int(normalized["legacy_score"].notna().sum()) if "legacy_score" in normalized.columns else 0

    lines = [
        "# Legacy News Normalization Report",
        "",
        "## Summary",
        f"- Total normalized rows: **{len(normalized):,}**",
        f"- Unique headline-date rows: **{len(unique):,}**",
        f"- Rows with usable URLs: **{with_url:,}**",
        f"- Unique rows with usable URLs for full-text retrieval: **{unique_with_url:,}**",
        f"- Duplicate rows captured: **{len(duplicates):,}**",
        f"- Rows matched with legacy LLM score: **{legacy_matched:,}**",
        f"- Date range: **{date_min.date() if pd.notna(date_min) else 'missing'}** to **{date_max.date() if pd.notna(date_max) else 'missing'}**",
        "",
        "## Missing/quality checks",
        missing_report.to_markdown(index=False),
        "",
        "## Source summary",
        source_summary.to_markdown(index=False),
        "",
        "## Output files",
        "- `data/02_interim/news_cleaned/legacy_news_normalized.csv`",
        "- `data/02_interim/news_cleaned/legacy_news_unique_for_retrieval.csv`",
        "- `data/02_interim/news_cleaned/legacy_news_duplicates.csv`",
        "- `data/02_interim/news_cleaned/legacy_news_missing_report.csv`",
        "- `data/02_interim/news_cleaned/legacy_news_source_summary.csv`",
        "",
        "## Next step",
        "Use `legacy_news_unique_for_retrieval.csv` as the input for full-text retrieval.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(default_project_root()), help="Project root folder")
    parser.add_argument("--include-backup", action="store_true", help="Also include headline-only backup file if present")
    parser.add_argument("--max-valid-date", default="2026-07-02", help="Dates after this are flagged as suspicious. Use empty string to disable.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = project_root / "data/02_interim/news_cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized, unique, duplicates, missing_report, source_summary = build_normalized(
        project_root=project_root,
        include_backup=args.include_backup,
        max_valid_date=args.max_valid_date or None,
    )

    normalized.to_csv(out_dir / "legacy_news_normalized.csv", index=False)
    unique.to_csv(out_dir / "legacy_news_unique_for_retrieval.csv", index=False)
    duplicates.to_csv(out_dir / "legacy_news_duplicates.csv", index=False)
    missing_report.to_csv(out_dir / "legacy_news_missing_report.csv", index=False)
    source_summary.to_csv(out_dir / "legacy_news_source_summary.csv", index=False)

    report_path = write_report(project_root, normalized, unique, duplicates, missing_report, source_summary)

    print("Normalization complete.")
    print(f"Project root: {project_root}")
    print(f"Normalized rows: {len(normalized):,}")
    print(f"Unique headline-date rows: {len(unique):,}")
    print(f"Rows with usable URLs: {int(normalized['has_url'].sum()):,}")
    print(f"Unique URLs for retrieval: {int(unique['has_url'].sum()):,}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
