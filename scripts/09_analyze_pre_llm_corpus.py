#!/usr/bin/env python3
"""Analyze the maritime-related pre-LLM kept-news dataset.

The script creates descriptive tables, data-quality checks, and PNG figures for:
- full-text character and word counts
- publisher/source names, domains, and base URLs
- total unique sources and source concentration
- annual and monthly article distributions
- month-of-year distribution
- language, collection, retrieval method, and matched-keyword distributions
- missing values, date coverage, duplicate warnings, short/long text, and likely index/topic pages

Default input:
    data/03_processed/news_full_text/pre_llm_filter/pre_llm_kept.csv

Default output:
    outputs/pre_llm_kept_analysis/
        analysis_summary.json
        analysis_report.md
        tables/*.csv
        figures/*.png

Run from the project root:
    python scripts/09_analyze_pre_llm_corpus.py

Or specify paths:
    python scripts/09_analyze_pre_llm_corpus.py \
        --input path/to/pre_llm_kept.csv \
        --output-dir outputs/pre_llm_kept_analysis
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Times", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


DEFAULT_INPUT = Path(
    "data/03_processed/news_full_text/pre_llm_filter/pre_llm_kept.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/pre_llm_kept_analysis")

TEXT_COLUMN_CANDIDATES = [
    "retrieved_text",
    "full_text",
    "article_text",
    "text",
    "content",
    "body",
]
TITLE_COLUMN_CANDIDATES = [
    "retrieved_title",
    "headline",
    "title",
    "headline_clean",
    "article_title",
]
DATE_COLUMN_CANDIDATES = [
    # Prefer the dataset's original date. Extracted publication timestamps can
    # sometimes be page-update times rather than the article's actual date.
    "date",
    "retrieved_publish_date",
    "published_date",
    "publish_date",
    "publication_date",
]
URL_COLUMN_CANDIDATES = [
    "preferred_url",
    "canonical_url",
    "final_url",
    "source_url",
    "url",
    "link",
    "normalized_link",
]
SOURCE_NAME_CANDIDATES = ["source_name", "publisher", "newspaper", "source"]
SOURCE_DOMAIN_CANDIDATES = ["source_domain", "domain", "publisher_domain"]
LANGUAGE_COLUMN_CANDIDATES = [
    "retrieved_meta_lang",
    "language",
    "lang",
]
COLLECTION_COLUMN_CANDIDATES = ["collection", "source_dataset", "source_file"]
RETRIEVAL_COLUMN_CANDIDATES = ["retrieval_method", "retrieval_status"]

SHORT_TEXT_CHARACTER_THRESHOLD = 500
VERY_SHORT_TEXT_CHARACTER_THRESHOLD = 250
LONG_TEXT_CHARACTER_THRESHOLD = 25_000
EXTREME_TEXT_CHARACTER_THRESHOLD = 50_000
TOP_N_SOURCES = 20
TOP_N_KEYWORDS = 25

MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

INDEX_PAGE_PATTERNS = [
    r"/(?:topic|topics)/",
    r"/(?:tag|tags)/",
    r"/(?:search)/",
    r"/(?:category|categories)/",
    r"/(?:archive|archives)/",
    r"/(?:definition|definitions)/",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create source, text-length, time-distribution, and data-quality "
            "statistics for the pre-LLM kept-news dataset."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input kept CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--top-sources",
        type=int,
        default=TOP_N_SOURCES,
        help=f"Number of sources displayed in the source graph (default: {TOP_N_SOURCES}).",
    )
    parser.add_argument(
        "--top-keywords",
        type=int,
        default=TOP_N_KEYWORDS,
        help=f"Number of matched keywords displayed in the keyword graph (default: {TOP_N_KEYWORDS}).",
    )
    return parser.parse_args()


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def clean_string_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def combine_columns(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    columns = [column for column in candidates if column in df.columns]
    if not columns:
        return pd.Series("", index=df.index, dtype="object")

    combined = clean_string_series(df[columns[0]])
    for column in columns[1:]:
        additional = clean_string_series(df[column])
        combined = combined.where(combined.ne(""), additional)
    return combined


def parse_publication_dates(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Parse date candidates in priority order, using later columns as fallbacks."""
    parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    source_column = pd.Series("", index=df.index, dtype="object")

    for column in DATE_COLUMN_CANDIDATES:
        if column not in df.columns:
            continue

        raw = clean_string_series(df[column])
        candidate = pd.to_datetime(raw, format="mixed", errors="coerce", utc=True)
        fill_mask = parsed.isna() & candidate.notna()
        parsed.loc[fill_mask] = candidate.loc[fill_mask]
        source_column.loc[fill_mask] = column

    return parsed, source_column


def normalize_domain(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().lower()
    if not text:
        return ""

    if "://" not in text:
        text = f"https://{text}"

    try:
        domain = urlsplit(text).netloc.lower()
    except ValueError:
        return ""

    if "@" in domain:
        domain = domain.rsplit("@", 1)[-1]
    if ":" in domain:
        domain = domain.split(":", 1)[0]
    if domain.startswith("www."):
        domain = domain[4:]
    return domain.strip(".")


def derive_domain_and_base_url(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    selected_url = combine_columns(df, URL_COLUMN_CANDIDATES)

    existing_domain_column = first_existing_column(df, SOURCE_DOMAIN_CANDIDATES)
    if existing_domain_column:
        domain = clean_string_series(df[existing_domain_column]).map(normalize_domain)
    else:
        domain = pd.Series("", index=df.index, dtype="object")

    parsed_from_url = selected_url.map(normalize_domain)
    domain = domain.where(domain.ne(""), parsed_from_url)
    base_url = domain.map(lambda item: f"https://{item}" if item else "")
    return domain, base_url, selected_url


def duplicate_nonempty_row_count(series: pd.Series) -> int:
    """Count rows belonging to duplicate groups, ignoring empty values."""
    cleaned = clean_string_series(series)
    nonempty = cleaned[cleaned.ne("")]
    return int(nonempty.duplicated(keep=False).sum())


def safe_percentage(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator) * 100, 2)


def json_value(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def make_text_length_summary(character_count: pd.Series, word_count: pd.Series) -> pd.DataFrame:
    quantiles = [0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
    rows: list[dict[str, object]] = []

    for metric_name, values in [
        ("character_count", character_count),
        ("word_count", word_count),
    ]:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            continue

        rows.extend(
            [
                {"metric": metric_name, "statistic": "count", "value": int(numeric.count())},
                {"metric": metric_name, "statistic": "mean", "value": round(float(numeric.mean()), 2)},
                {"metric": metric_name, "statistic": "std", "value": round(float(numeric.std()), 2)},
                {"metric": metric_name, "statistic": "minimum", "value": int(numeric.min())},
                {"metric": metric_name, "statistic": "maximum", "value": int(numeric.max())},
            ]
        )
        for quantile in quantiles:
            rows.append(
                {
                    "metric": metric_name,
                    "statistic": f"p{int(quantile * 100):02d}",
                    "value": round(float(numeric.quantile(quantile)), 2),
                }
            )

    return pd.DataFrame(rows)


def make_text_length_buckets(character_count: pd.Series) -> pd.DataFrame:
    bins = [-1, 249, 499, 999, 1_999, 4_999, 9_999, 24_999, 49_999, np.inf]
    labels = [
        "0-249",
        "250-499",
        "500-999",
        "1,000-1,999",
        "2,000-4,999",
        "5,000-9,999",
        "10,000-24,999",
        "25,000-49,999",
        "50,000+",
    ]
    bucket = pd.cut(character_count, bins=bins, labels=labels, include_lowest=True)
    counts = bucket.value_counts(sort=False).reindex(labels, fill_value=0)
    result = counts.rename_axis("character_count_bucket").reset_index(name="article_count")
    result["share_percent"] = result["article_count"].map(
        lambda value: safe_percentage(value, len(character_count))
    )
    return result


def build_source_statistics(
    df: pd.DataFrame,
    source_name: pd.Series,
    domain: pd.Series,
    base_url: pd.Series,
    character_count: pd.Series,
    word_count: pd.Series,
    publication_date: pd.Series,
) -> pd.DataFrame:
    working = pd.DataFrame(
        {
            "source_name": source_name,
            "source_domain": domain,
            "base_url": base_url,
            "character_count": character_count,
            "word_count": word_count,
            "publication_date": publication_date,
        }
    )
    working["source_name"] = working["source_name"].where(
        working["source_name"].ne(""), working["source_domain"]
    )
    working["source_name"] = working["source_name"].replace("", "Unknown source")
    working["source_domain"] = working["source_domain"].replace("", "unknown")
    working["base_url"] = working["base_url"].replace("", "unknown")

    grouped = (
        working.groupby(["source_name", "source_domain", "base_url"], dropna=False)
        .agg(
            article_count=("source_name", "size"),
            mean_character_count=("character_count", "mean"),
            median_character_count=("character_count", "median"),
            mean_word_count=("word_count", "mean"),
            first_publication_date=("publication_date", "min"),
            last_publication_date=("publication_date", "max"),
        )
        .reset_index()
    )
    grouped["article_share_percent"] = grouped["article_count"].map(
        lambda value: safe_percentage(value, len(df))
    )
    grouped["mean_character_count"] = grouped["mean_character_count"].round(2)
    grouped["median_character_count"] = grouped["median_character_count"].round(2)
    grouped["mean_word_count"] = grouped["mean_word_count"].round(2)
    grouped["first_publication_date"] = grouped["first_publication_date"].dt.strftime(
        "%Y-%m-%d"
    )
    grouped["last_publication_date"] = grouped["last_publication_date"].dt.strftime(
        "%Y-%m-%d"
    )
    return grouped.sort_values(
        ["article_count", "source_name"], ascending=[False, True]
    ).reset_index(drop=True)


def build_domain_statistics(
    df: pd.DataFrame,
    source_name: pd.Series,
    domain: pd.Series,
    base_url: pd.Series,
    character_count: pd.Series,
    word_count: pd.Series,
    publication_date: pd.Series,
) -> pd.DataFrame:
    """Aggregate all publisher-name variants at the base-domain level."""
    working = pd.DataFrame(
        {
            "source_name": source_name.replace("", "Unknown source"),
            "source_domain": domain.replace("", "unknown"),
            "base_url": base_url.replace("", "unknown"),
            "character_count": character_count,
            "word_count": word_count,
            "publication_date": publication_date,
        }
    )

    def join_names(values: pd.Series) -> str:
        names = sorted({str(value).strip() for value in values if str(value).strip() and str(value).strip() != "Unknown source"})
        return " | ".join(names) if names else "Unknown source"

    grouped = (
        working.groupby(["source_domain", "base_url"], dropna=False)
        .agg(
            source_name_variants=("source_name", join_names),
            article_count=("source_domain", "size"),
            mean_character_count=("character_count", "mean"),
            median_character_count=("character_count", "median"),
            mean_word_count=("word_count", "mean"),
            first_publication_date=("publication_date", "min"),
            last_publication_date=("publication_date", "max"),
        )
        .reset_index()
    )
    grouped["article_share_percent"] = grouped["article_count"].map(
        lambda value: safe_percentage(value, len(df))
    )
    grouped["mean_character_count"] = grouped["mean_character_count"].round(2)
    grouped["median_character_count"] = grouped["median_character_count"].round(2)
    grouped["mean_word_count"] = grouped["mean_word_count"].round(2)
    grouped["first_publication_date"] = grouped["first_publication_date"].dt.strftime("%Y-%m-%d")
    grouped["last_publication_date"] = grouped["last_publication_date"].dt.strftime("%Y-%m-%d")
    return grouped.sort_values(
        ["article_count", "source_domain"], ascending=[False, True]
    ).reset_index(drop=True)


def build_column_missingness(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in df.columns:
        series = df[column]
        missing_mask = series.isna() | series.astype(str).str.strip().eq("")
        missing_count = int(missing_mask.sum())
        rows.append(
            {
                "column": column,
                "nonmissing_count": int(len(df) - missing_count),
                "missing_count": missing_count,
                "missing_percent": safe_percentage(missing_count, len(df)),
                "dtype": str(series.dtype),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["missing_percent", "column"], ascending=[False, True]
    ).reset_index(drop=True)


def build_frequency_table(series: pd.Series, column_name: str) -> pd.DataFrame:
    cleaned = clean_string_series(series).replace("", "Unknown")
    result = cleaned.value_counts(dropna=False).rename_axis(column_name).reset_index(name="article_count")
    result["share_percent"] = result["article_count"].map(
        lambda value: safe_percentage(value, len(series))
    )
    return result


def parse_keyword_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "matched_maritime_keywords" not in df.columns:
        return pd.DataFrame(columns=["matched_keyword", "article_count", "share_percent"])

    counter: Counter[str] = Counter()
    for value in clean_string_series(df["matched_maritime_keywords"]):
        if not value:
            continue
        terms = [term.strip() for term in value.split("|") if term.strip()]
        counter.update(set(terms))

    result = pd.DataFrame(counter.most_common(), columns=["matched_keyword", "article_count"])
    if result.empty:
        result["share_percent"] = pd.Series(dtype="float64")
    else:
        result["share_percent"] = result["article_count"].map(
            lambda value: safe_percentage(value, len(df))
        )
    return result


def make_quality_checks(
    df: pd.DataFrame,
    title: pd.Series,
    text: pd.Series,
    source_name: pd.Series,
    domain: pd.Series,
    selected_url: pd.Series,
    publication_date: pd.Series,
    character_count: pd.Series,
    normalized_text: pd.Series,
) -> pd.DataFrame:
    normalized_url = selected_url.str.strip().str.lower()
    article_id = clean_string_series(df["article_id"]) if "article_id" in df.columns else pd.Series("", index=df.index)

    index_page_pattern = re.compile("|".join(INDEX_PAGE_PATTERNS), flags=re.IGNORECASE)
    suspected_index_page = selected_url.map(lambda value: bool(index_page_pattern.search(value)))

    checks = [
        ("total_rows", len(df)),
        ("missing_or_empty_full_text", int(text.eq("").sum())),
        ("missing_or_empty_title", int(title.eq("").sum())),
        ("missing_source_name", int(source_name.eq("").sum())),
        ("missing_source_domain", int(domain.eq("").sum())),
        ("missing_url", int(selected_url.eq("").sum())),
        ("unparseable_or_missing_publication_date", int(publication_date.isna().sum())),
        (f"very_short_text_below_{VERY_SHORT_TEXT_CHARACTER_THRESHOLD}_characters", int((character_count < VERY_SHORT_TEXT_CHARACTER_THRESHOLD).sum())),
        (f"short_text_below_{SHORT_TEXT_CHARACTER_THRESHOLD}_characters", int((character_count < SHORT_TEXT_CHARACTER_THRESHOLD).sum())),
        (f"long_text_at_least_{LONG_TEXT_CHARACTER_THRESHOLD}_characters", int((character_count >= LONG_TEXT_CHARACTER_THRESHOLD).sum())),
        (f"extreme_text_at_least_{EXTREME_TEXT_CHARACTER_THRESHOLD}_characters", int((character_count >= EXTREME_TEXT_CHARACTER_THRESHOLD).sum())),
        ("duplicate_article_id_rows", duplicate_nonempty_row_count(article_id)),
        ("duplicate_nonempty_url_rows", duplicate_nonempty_row_count(normalized_url)),
        ("duplicate_nonempty_normalized_text_rows", duplicate_nonempty_row_count(normalized_text)),
        ("suspected_topic_tag_search_or_index_pages", int(suspected_index_page.sum())),
    ]

    quality = pd.DataFrame(checks, columns=["check", "affected_rows"])
    quality["share_percent"] = quality["affected_rows"].map(
        lambda value: safe_percentage(value, len(df))
    )
    return quality


def plot_character_distribution(character_count: pd.Series, output_path: Path) -> None:
    values = character_count[character_count > 0]
    if values.empty:
        return

    upper_limit = max(float(values.quantile(0.99)), 1.0)
    clipped = values.clip(upper=upper_limit)

    plt.figure(figsize=(10, 6))
    plt.hist(clipped, bins=40)
    plt.xlabel("Full-text character count (values above p99 clipped)")
    plt.ylabel("Number of articles")
    plt.title("Distribution of full-text character counts")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_sources(source_statistics: pd.DataFrame, output_path: Path, top_n: int) -> None:
    top = source_statistics.head(max(top_n, 1)).sort_values("article_count")
    if top.empty:
        return

    labels = top["source_name"].where(
        top["source_name"].ne("Unknown source"), top["source_domain"]
    )

    plt.figure(figsize=(11, max(6, len(top) * 0.36)))
    plt.barh(labels, top["article_count"])
    plt.xlabel("Number of articles")
    plt.ylabel("Publisher/source")
    plt.title(f"Top {len(top)} publishers in the pre-LLM kept dataset")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_year_distribution(year_statistics: pd.DataFrame, output_path: Path) -> None:
    valid = year_statistics[year_statistics["year"].ne("Unknown")].copy()
    if valid.empty:
        return

    valid["year_numeric"] = pd.to_numeric(valid["year"], errors="coerce")
    valid = valid.dropna(subset=["year_numeric"]).sort_values("year_numeric")

    plt.figure(figsize=(10, 6))
    plt.bar(valid["year_numeric"].astype(int).astype(str), valid["article_count"])
    plt.xlabel("Publication year")
    plt.ylabel("Number of articles")
    plt.title("Articles by publication year")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_monthly_timeline(month_statistics: pd.DataFrame, output_path: Path) -> None:
    if month_statistics.empty:
        return

    dates = pd.to_datetime(month_statistics["year_month"] + "-01", errors="coerce")
    valid = month_statistics.assign(plot_date=dates).dropna(subset=["plot_date"])
    if valid.empty:
        return

    plt.figure(figsize=(13, 6))
    plt.plot(valid["plot_date"], valid["article_count"], marker="o", markersize=2)
    plt.xlabel("Publication month")
    plt.ylabel("Number of articles")
    plt.title("Monthly article count over time")
    axis = plt.gca()
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_month_of_year(month_of_year: pd.DataFrame, output_path: Path) -> None:
    if month_of_year.empty:
        return

    ordered = month_of_year.set_index("month_name").reindex(MONTH_ORDER).fillna(0).reset_index()
    plt.figure(figsize=(11, 6))
    plt.bar(ordered["month_name"].str[:3], ordered["article_count"])
    plt.xlabel("Calendar month")
    plt.ylabel("Number of articles")
    plt.title("Article distribution by calendar month")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_keywords(keyword_statistics: pd.DataFrame, output_path: Path, top_n: int) -> None:
    top = keyword_statistics.head(max(top_n, 1)).sort_values("article_count")
    if top.empty:
        return

    plt.figure(figsize=(11, max(6, len(top) * 0.34)))
    plt.barh(top["matched_keyword"], top["article_count"])
    plt.xlabel("Number of articles")
    plt.ylabel("Matched maritime keyword")
    plt.title(f"Top {len(top)} maritime filter keywords")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def write_markdown_report(
    path: Path,
    summary: dict[str, object],
    quality_checks: pd.DataFrame,
    source_statistics: pd.DataFrame,
    year_statistics: pd.DataFrame,
    text_length_summary: pd.DataFrame,
) -> None:
    def metric(metric_name: str, statistic: str) -> str:
        match = text_length_summary[
            (text_length_summary["metric"] == metric_name)
            & (text_length_summary["statistic"] == statistic)
        ]
        if match.empty:
            return "N/A"
        value = match.iloc[0]["value"]
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        try:
            return f"{float(value):,.2f}"
        except (TypeError, ValueError):
            return str(value)

    top_source_rows = source_statistics.head(10)
    top_sources_md = "\n".join(
        f"| {row.source_name} | {row.source_domain} | {int(row.article_count):,} | {float(row.article_share_percent):.2f}% |"
        for row in top_source_rows.itertuples(index=False)
    )
    if not top_sources_md:
        top_sources_md = "| N/A | N/A | 0 | 0.00% |"

    yearly_rows = year_statistics[year_statistics["year"].ne("Unknown")]
    year_md = "\n".join(
        f"| {row.year} | {int(row.article_count):,} | {float(row.share_percent):.2f}% |"
        for row in yearly_rows.itertuples(index=False)
    )
    if not year_md:
        year_md = "| N/A | 0 | 0.00% |"

    quality_md = "\n".join(
        f"| {row.check} | {int(row.affected_rows):,} | {float(row.share_percent):.2f}% |"
        for row in quality_checks.itertuples(index=False)
    )

    lines = [
        "# Pre-LLM Kept Dataset Analysis",
        "",
        "## Dataset overview",
        "",
        f"- Input rows: **{int(summary['row_count']):,}**",
        f"- Unique publisher names: **{int(summary['unique_source_names']):,}**",
        f"- Unique source domains: **{int(summary['unique_source_domains']):,}**",
        f"- Parsed publication-date coverage: **{float(summary['date_parse_success_percent']):.2f}%**",
        f"- Publication range: **{summary.get('first_publication_date') or 'N/A'}** to **{summary.get('last_publication_date') or 'N/A'}**",
        f"- Mean full-text length: **{metric('character_count', 'mean')} characters**",
        f"- Median full-text length: **{metric('character_count', 'p50')} characters**",
        f"- Mean word count: **{metric('word_count', 'mean')} words**",
        f"- Median word count: **{metric('word_count', 'p50')} words**",
        "",
        "## Top publishers",
        "",
        "| Source | Domain | Articles | Share |",
        "|---|---|---:|---:|",
        top_sources_md,
        "",
        "## Annual distribution",
        "",
        "| Year | Articles | Share |",
        "|---:|---:|---:|",
        year_md,
        "",
        "## Data-quality checks",
        "",
        "These checks are warnings for review; the script does not delete records.",
        "",
        "| Check | Affected rows | Share |",
        "|---|---:|---:|",
        quality_md,
        "",
        "## Generated files",
        "",
        "See `tables/` for CSV statistics and `figures/` for PNG charts.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}\n"
            "Run the pre-LLM filter first or pass the correct file using --input."
        )

    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    if df.empty:
        raise ValueError(f"Input CSV contains no rows: {args.input}")

    text_column = first_existing_column(df, TEXT_COLUMN_CANDIDATES)
    if text_column is None:
        raise ValueError(
            "No full-text column was found. Expected one of: "
            + ", ".join(TEXT_COLUMN_CANDIDATES)
        )

    title = combine_columns(df, TITLE_COLUMN_CANDIDATES)
    text = clean_string_series(df[text_column])
    source_name = combine_columns(df, SOURCE_NAME_CANDIDATES)
    domain, base_url, selected_url = derive_domain_and_base_url(df)
    publication_date, publication_date_source = parse_publication_dates(df)

    character_count = text.str.len().astype("int64")
    word_count = text.str.split().str.len().astype("int64")
    normalized_text = (
        text.str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    article_level = pd.DataFrame(
        {
            "article_id": clean_string_series(df["article_id"]) if "article_id" in df.columns else df.index.astype(str),
            "headline": title,
            "source_name": source_name,
            "source_domain": domain,
            "base_url": base_url,
            "selected_url": selected_url,
            "publication_date": publication_date.dt.strftime("%Y-%m-%d"),
            "publication_date_source": publication_date_source,
            "publication_year": publication_date.dt.year.astype("Int64"),
            "publication_month": publication_date.dt.tz_convert(None).dt.to_period("M").astype(str).replace("NaT", ""),
            "character_count": character_count,
            "word_count": word_count,
            "characters_per_word": np.where(word_count > 0, character_count / word_count, np.nan),
            "very_short_text_flag": character_count < VERY_SHORT_TEXT_CHARACTER_THRESHOLD,
            "short_text_flag": character_count < SHORT_TEXT_CHARACTER_THRESHOLD,
            "long_text_flag": character_count >= LONG_TEXT_CHARACTER_THRESHOLD,
            "extreme_text_flag": character_count >= EXTREME_TEXT_CHARACTER_THRESHOLD,
        }
    )
    article_level["characters_per_word"] = article_level["characters_per_word"].round(2)

    source_statistics = build_source_statistics(
        df,
        source_name,
        domain,
        base_url,
        character_count,
        word_count,
        publication_date,
    )
    domain_statistics = build_domain_statistics(
        df,
        source_name,
        domain,
        base_url,
        character_count,
        word_count,
        publication_date,
    )
    column_missingness = build_column_missingness(df)
    date_source_statistics = build_frequency_table(
        publication_date_source, "publication_date_source"
    )

    valid_dates = publication_date.dropna()
    if valid_dates.empty:
        year_statistics = pd.DataFrame(columns=["year", "article_count", "share_percent"])
        month_statistics = pd.DataFrame(columns=["year_month", "article_count", "share_percent"])
        month_of_year_statistics = pd.DataFrame(columns=["month_number", "month_name", "article_count", "share_percent"])
        year_month_matrix = pd.DataFrame()
    else:
        year_counts = publication_date.dt.year.value_counts().sort_index()
        year_statistics = year_counts.rename_axis("year").reset_index(name="article_count")
        year_statistics["year"] = year_statistics["year"].astype("Int64").astype(str)
        year_statistics["share_percent"] = year_statistics["article_count"].map(
            lambda value: safe_percentage(value, len(df))
        )

        month_period = publication_date.dt.tz_convert(None).dt.to_period("M")
        first_month = month_period.dropna().min()
        last_month = month_period.dropna().max()
        full_month_range = pd.period_range(first_month, last_month, freq="M")
        month_counts = month_period.value_counts().reindex(full_month_range, fill_value=0).sort_index()
        month_statistics = month_counts.rename_axis("year_month").reset_index(name="article_count")
        month_statistics["year_month"] = month_statistics["year_month"].astype(str)
        month_statistics["share_percent"] = month_statistics["article_count"].map(
            lambda value: safe_percentage(value, len(df))
        )

        month_number_counts = publication_date.dt.month.value_counts().reindex(range(1, 13), fill_value=0)
        month_of_year_statistics = month_number_counts.rename_axis("month_number").reset_index(name="article_count")
        month_of_year_statistics["month_name"] = month_of_year_statistics["month_number"].map(
            dict(enumerate(MONTH_ORDER, start=1))
        )
        month_of_year_statistics["share_percent"] = month_of_year_statistics["article_count"].map(
            lambda value: safe_percentage(value, len(df))
        )

        matrix_source = pd.DataFrame(
            {
                "year": publication_date.dt.year,
                "month_number": publication_date.dt.month,
            }
        ).dropna()
        matrix_source["year"] = matrix_source["year"].astype(int)
        matrix_source["month_number"] = matrix_source["month_number"].astype(int)
        year_month_matrix = pd.crosstab(matrix_source["year"], matrix_source["month_number"])
        year_month_matrix = year_month_matrix.reindex(columns=range(1, 13), fill_value=0)
        year_month_matrix.columns = [MONTH_ORDER[index - 1] for index in year_month_matrix.columns]
        year_month_matrix = year_month_matrix.reset_index()

    text_length_summary = make_text_length_summary(character_count, word_count)
    text_length_buckets = make_text_length_buckets(character_count)
    keyword_statistics = parse_keyword_counts(df)

    language_column = first_existing_column(df, LANGUAGE_COLUMN_CANDIDATES)
    language_statistics = (
        build_frequency_table(df[language_column], "language")
        if language_column
        else pd.DataFrame(columns=["language", "article_count", "share_percent"])
    )

    collection_statistics_frames: list[pd.DataFrame] = []
    for column in COLLECTION_COLUMN_CANDIDATES:
        if column in df.columns:
            frequency = build_frequency_table(df[column], "value")
            frequency.insert(0, "field", column)
            collection_statistics_frames.append(frequency)
    collection_statistics = (
        pd.concat(collection_statistics_frames, ignore_index=True)
        if collection_statistics_frames
        else pd.DataFrame(columns=["field", "value", "article_count", "share_percent"])
    )

    retrieval_statistics_frames: list[pd.DataFrame] = []
    for column in RETRIEVAL_COLUMN_CANDIDATES:
        if column in df.columns:
            frequency = build_frequency_table(df[column], "value")
            frequency.insert(0, "field", column)
            retrieval_statistics_frames.append(frequency)
    retrieval_statistics = (
        pd.concat(retrieval_statistics_frames, ignore_index=True)
        if retrieval_statistics_frames
        else pd.DataFrame(columns=["field", "value", "article_count", "share_percent"])
    )

    quality_checks = make_quality_checks(
        df,
        title,
        text,
        source_name,
        domain,
        selected_url,
        publication_date,
        character_count,
        normalized_text,
    )

    # Additional warning tables for direct inspection.
    suspected_index_pattern = re.compile("|".join(INDEX_PAGE_PATTERNS), flags=re.IGNORECASE)
    suspected_index_mask = selected_url.map(lambda value: bool(suspected_index_pattern.search(value)))
    suspected_index_pages = article_level.loc[suspected_index_mask].copy()
    shortest_articles = article_level.sort_values("character_count").head(100).copy()
    longest_articles = article_level.sort_values("character_count", ascending=False).head(100).copy()

    # Save tables.
    save_dataframe(article_level, tables_dir / "article_level_text_and_source_statistics.csv")
    save_dataframe(source_statistics, tables_dir / "source_statistics.csv")
    save_dataframe(domain_statistics, tables_dir / "domain_statistics.csv")
    save_dataframe(column_missingness, tables_dir / "column_missingness.csv")
    save_dataframe(date_source_statistics, tables_dir / "publication_date_source_statistics.csv")
    save_dataframe(year_statistics, tables_dir / "year_statistics.csv")
    save_dataframe(month_statistics, tables_dir / "monthly_timeline_statistics.csv")
    save_dataframe(month_of_year_statistics, tables_dir / "month_of_year_statistics.csv")
    save_dataframe(year_month_matrix, tables_dir / "year_month_matrix.csv")
    save_dataframe(text_length_summary, tables_dir / "text_length_summary.csv")
    save_dataframe(text_length_buckets, tables_dir / "text_length_buckets.csv")
    save_dataframe(keyword_statistics, tables_dir / "matched_keyword_statistics.csv")
    save_dataframe(language_statistics, tables_dir / "language_statistics.csv")
    save_dataframe(collection_statistics, tables_dir / "collection_and_dataset_statistics.csv")
    save_dataframe(retrieval_statistics, tables_dir / "retrieval_statistics.csv")
    save_dataframe(quality_checks, tables_dir / "data_quality_checks.csv")
    save_dataframe(suspected_index_pages, tables_dir / "suspected_index_topic_tag_pages.csv")
    save_dataframe(shortest_articles, tables_dir / "shortest_100_articles.csv")
    save_dataframe(longest_articles, tables_dir / "longest_100_articles.csv")

    # Save figures.
    plot_character_distribution(character_count, figures_dir / "01_text_character_distribution.png")
    plot_top_sources(source_statistics, figures_dir / "02_top_sources.png", args.top_sources)
    plot_year_distribution(year_statistics, figures_dir / "03_articles_by_year.png")
    plot_monthly_timeline(month_statistics, figures_dir / "04_articles_by_month_timeline.png")
    plot_month_of_year(month_of_year_statistics, figures_dir / "05_month_of_year_distribution.png")
    plot_top_keywords(keyword_statistics, figures_dir / "06_top_matched_keywords.png", args.top_keywords)

    unique_source_names = source_name[source_name.ne("")].nunique()
    unique_domains = domain[domain.ne("")].nunique()
    top_source_share = float(source_statistics.iloc[0]["article_share_percent"]) if not source_statistics.empty else 0.0
    top_10_source_share = round(float(source_statistics.head(10)["article_count"].sum()) / len(df) * 100, 2)

    first_publication_date = valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else None
    last_publication_date = valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else None

    summary = {
        "input_file": str(args.input),
        "output_directory": str(output_dir),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "text_column_used": text_column,
        "date_columns_considered": [column for column in DATE_COLUMN_CANDIDATES if column in df.columns],
        "source_name_columns_considered": [column for column in SOURCE_NAME_CANDIDATES if column in df.columns],
        "url_columns_considered": [column for column in URL_COLUMN_CANDIDATES if column in df.columns],
        "unique_source_names": int(unique_source_names),
        "unique_source_domains": int(unique_domains),
        "top_source_share_percent": top_source_share,
        "top_10_source_share_percent": top_10_source_share,
        "first_publication_date": first_publication_date,
        "last_publication_date": last_publication_date,
        "parsed_publication_dates": int(publication_date.notna().sum()),
        "unparsed_publication_dates": int(publication_date.isna().sum()),
        "date_parse_success_percent": safe_percentage(publication_date.notna().sum(), len(df)),
        "character_count": {
            "total": int(character_count.sum()),
            "mean": round(float(character_count.mean()), 2),
            "median": round(float(character_count.median()), 2),
            "minimum": int(character_count.min()),
            "maximum": int(character_count.max()),
            "p95": round(float(character_count.quantile(0.95)), 2),
            "p99": round(float(character_count.quantile(0.99)), 2),
        },
        "word_count": {
            "total": int(word_count.sum()),
            "mean": round(float(word_count.mean()), 2),
            "median": round(float(word_count.median()), 2),
            "minimum": int(word_count.min()),
            "maximum": int(word_count.max()),
            "p95": round(float(word_count.quantile(0.95)), 2),
            "p99": round(float(word_count.quantile(0.99)), 2),
        },
        "quality_checks": {
            row.check: int(row.affected_rows)
            for row in quality_checks.itertuples(index=False)
        },
        "generated_tables": sorted(path.name for path in tables_dir.glob("*.csv")),
        "generated_figures": sorted(path.name for path in figures_dir.glob("*.png")),
    }

    summary_path = output_dir / "analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, default=json_value)

    write_markdown_report(
        output_dir / "analysis_report.md",
        summary,
        quality_checks,
        source_statistics,
        year_statistics,
        text_length_summary,
    )

    print("Pre-LLM kept-dataset analysis completed.")
    print(f"Input rows:             {len(df):,}")
    print(f"Unique source names:    {unique_source_names:,}")
    print(f"Unique source domains:  {unique_domains:,}")
    print(f"Date parse success:     {summary['date_parse_success_percent']:.2f}%")
    print(f"Mean text characters:   {summary['character_count']['mean']:,.2f}")
    print(f"Median text characters: {summary['character_count']['median']:,.2f}")
    print(f"Tables directory:       {tables_dir}")
    print(f"Figures directory:      {figures_dir}")
    print(f"Summary:                {summary_path}")
    print(f"Report:                 {output_dir / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
