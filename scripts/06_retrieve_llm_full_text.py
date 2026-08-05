#!/usr/bin/env python3
"""Retrieve full article text for the cleaned LLM news dataset using Newspaper4k.

This script writes successful and unsuccessful retrievals to separate CSV files.

Default input:
    data/02_interim/news_cleaned/llm_search_results_cleaned.csv

Required input columns:
    link, headline, source_name, date

Optional input column:
    article_id

Default outputs:
    data/02_interim/article_enrichment/full_text_llm_news4k/
        llm_full_text_news4k_succeeded.csv
        llm_full_text_news4k_unsucceeded.csv

    outputs/reports/llm_full_text_news4k_report.md

A row is written to the succeeded file only when:
    retrieval_status == "success"

The unsuccessful file contains:
    invalid_url, failed, empty_text, and too_short rows

Install Newspaper4k:
    pip install newspaper4k

Run from the project root:
    python scripts/06_retrieve_llm_full_text.py

Small test run:
    python scripts/06_retrieve_llm_full_text.py --max-rows 20 --sleep 1 --overwrite

Resume after interruption:
    python scripts/06_retrieve_llm_full_text.py --resume
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

try:
    from newspaper import Article, Config  # provided by newspaper4k
except Exception as exc:  # pragma: no cover - depends on local environment
    Article = None  # type: ignore[assignment]
    Config = None  # type: ignore[assignment]
    NEWSPAPER_IMPORT_ERROR = exc
else:
    NEWSPAPER_IMPORT_ERROR = None


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

REQUIRED_INPUT_COLUMNS = ["link", "headline", "source_name", "date"]

RESULT_COLUMNS = [
    "article_id",
    "link",
    "normalized_link",
    "headline",
    "source_name",
    "date",
    "final_url",
    "source_domain",
    "retrieval_status",
    "error_type",
    "error_message",
    "retrieved_title",
    "retrieved_text",
    "retrieved_text_len",
    "retrieved_word_count",
    "retrieved_text_hash",
    "retrieved_publish_date",
    "retrieved_authors",
    "retrieved_top_image",
    "retrieved_meta_description",
    "retrieved_meta_lang",
    "retrieved_keywords",
    "retrieved_summary",
    "retrieval_method",
    "retrieved_at_utc",
]


@dataclass
class ExtractionResult:
    article_id: str
    link: str
    normalized_link: str
    headline: str
    source_name: str
    date: str
    final_url: str
    source_domain: str
    retrieval_status: str
    error_type: str
    error_message: str
    retrieved_title: str
    retrieved_text: str
    retrieved_text_len: int
    retrieved_word_count: int
    retrieved_text_hash: str
    retrieved_publish_date: str
    retrieved_authors: str
    retrieved_top_image: str
    retrieved_meta_description: str
    retrieved_meta_lang: str
    retrieved_keywords: str
    retrieved_summary: str
    retrieval_method: str
    retrieved_at_utc: str


def project_root() -> Path:
    """Return the repository root when this file is stored in scripts/."""
    return Path(__file__).resolve().parents[1]


def clean_text(value: Any) -> str:
    """Normalize whitespace without flattening meaningful paragraph breaks."""
    if value is None or pd.isna(value):
        return ""

    text = html.unescape(str(value)).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_url(value: Any) -> str:
    """Normalize a URL and remove only its fragment component."""
    if value is None or pd.isna(value):
        return ""

    url = html.unescape(str(value)).strip()
    if not url or url.lower() in {"nan", "none", "null", "n/a"}:
        return ""

    if url.startswith("//"):
        url = "https:" + url
    elif url.startswith("www."):
        url = "https://" + url

    try:
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        netloc = parts.netloc.lower()
        path = parts.path or "/"

        normalized = urlunsplit((scheme, netloc, path, parts.query, ""))
        return normalized.rstrip("/") if path != "/" else normalized
    except Exception:
        return url


def valid_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
        return parts.scheme in {"http", "https"} and bool(parts.netloc)
    except Exception:
        return False


def domain_of(url: str) -> str:
    try:
        return urlsplit(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def article_id_for(url: str, row_number: int) -> str:
    """Create a stable fallback ID from the normalized URL."""
    if url:
        digest = hashlib.sha256(url.encode("utf-8", errors="ignore")).hexdigest()[:20]
        return f"llm_{digest}"
    return f"llm_row_{row_number:07d}"


def sha256_text(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def safe_join(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "; ".join(clean_text(item) for item in value if clean_text(item))
    return clean_text(value)


def date_to_iso(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        if hasattr(value, "isoformat"):
            return value.isoformat()
        parsed = pd.to_datetime(value, errors="coerce")
        return "" if pd.isna(parsed) else parsed.isoformat()
    except Exception:
        return clean_text(value)


def get_attr(article: Any, name: str, default: Any = "") -> Any:
    try:
        return getattr(article, name, default)
    except Exception:
        return default


def read_input(path: Path) -> pd.DataFrame:
    """Read and validate the cleaned LLM CSV."""
    last_error: Exception | None = None

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(
                path,
                dtype="string",
                keep_default_na=False,
                encoding=encoding,
            )
            break
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    else:
        raise RuntimeError(f"Could not read {path}: {last_error}")

    missing = [
        column for column in REQUIRED_INPUT_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Input file is missing columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    selected_columns = REQUIRED_INPUT_COLUMNS.copy()
    if "article_id" in df.columns:
        selected_columns.insert(0, "article_id")

    df = df.loc[:, selected_columns].copy()

    for column in selected_columns:
        df[column] = df[column].map(clean_text)

    df["normalized_link"] = df["link"].map(normalize_url)
    df = df.loc[
        ~df[REQUIRED_INPUT_COLUMNS].eq("").all(axis=1)
    ].copy()

    # Keep only the first row for each nonblank normalized URL.
    nonblank = df["normalized_link"].ne("")
    duplicates = nonblank & df.duplicated(
        subset=["normalized_link"],
        keep="first",
    )
    duplicate_links_removed = int(duplicates.sum())
    df = df.loc[~duplicates].reset_index(drop=True)
    df.attrs["duplicate_links_removed"] = duplicate_links_removed

    if "article_id" not in df.columns:
        df["article_id"] = ""

    generated_ids = [
        article_id_for(url, index)
        for index, url in enumerate(df["normalized_link"], start=1)
    ]
    df["article_id"] = [
        existing_id if existing_id else generated_id
        for existing_id, generated_id in zip(df["article_id"], generated_ids)
    ]

    return df


def extract_article(
    *,
    article_id: str,
    link: str,
    normalized_link: str,
    headline: str,
    source_name: str,
    date: str,
    language: str,
    request_timeout: int,
    run_nlp: bool,
    min_words: int,
) -> ExtractionResult:
    """Download and parse one article with Newspaper4k."""
    if Article is None or Config is None:
        raise RuntimeError(
            "Newspaper4k could not be imported. Install it with: "
            "pip install newspaper4k. "
            f"Original error: {NEWSPAPER_IMPORT_ERROR}"
        )

    retrieved_at = datetime.now(timezone.utc).isoformat()

    config = Config()
    config.browser_user_agent = USER_AGENT
    config.request_timeout = request_timeout

    article = Article(
        normalized_link,
        language=language or "en",
        config=config,
    )
    article.download()
    article.parse()

    if run_nlp:
        try:
            article.nlp()
        except Exception:
            pass

    text = clean_text(get_attr(article, "text", ""))
    word_count = len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))

    if not text:
        status = "empty_text"
    elif word_count < min_words:
        status = "too_short"
    else:
        status = "success"

    final_url = clean_text(get_attr(article, "url", "")) or normalized_link

    return ExtractionResult(
        article_id=article_id,
        link=link,
        normalized_link=normalized_link,
        headline=headline,
        source_name=source_name,
        date=date,
        final_url=final_url,
        source_domain=domain_of(final_url),
        retrieval_status=status,
        error_type="",
        error_message="",
        retrieved_title=clean_text(get_attr(article, "title", "")),
        retrieved_text=text,
        retrieved_text_len=len(text),
        retrieved_word_count=word_count,
        retrieved_text_hash=sha256_text(text),
        retrieved_publish_date=date_to_iso(
            get_attr(article, "publish_date", "")
        ),
        retrieved_authors=safe_join(get_attr(article, "authors", [])),
        retrieved_top_image=clean_text(
            get_attr(article, "top_image", "")
        ),
        retrieved_meta_description=clean_text(
            get_attr(article, "meta_description", "")
        ),
        retrieved_meta_lang=clean_text(
            get_attr(article, "meta_lang", "")
        ),
        retrieved_keywords=safe_join(
            get_attr(article, "keywords", [])
        ),
        retrieved_summary=clean_text(
            get_attr(article, "summary", "")
        ),
        retrieval_method="newspaper4k.Article",
        retrieved_at_utc=retrieved_at,
    )


def failure_result(
    *,
    article_id: str,
    link: str,
    normalized_link: str,
    headline: str,
    source_name: str,
    date: str,
    status: str,
    error_type: str,
    error_message: str,
) -> ExtractionResult:
    return ExtractionResult(
        article_id=article_id,
        link=link,
        normalized_link=normalized_link,
        headline=headline,
        source_name=source_name,
        date=date,
        final_url="",
        source_domain=domain_of(normalized_link),
        retrieval_status=status,
        error_type=clean_text(error_type),
        error_message=clean_text(error_message)[:1000],
        retrieved_title="",
        retrieved_text="",
        retrieved_text_len=0,
        retrieved_word_count=0,
        retrieved_text_hash="",
        retrieved_publish_date="",
        retrieved_authors="",
        retrieved_top_image="",
        retrieved_meta_description="",
        retrieved_meta_lang="",
        retrieved_keywords="",
        retrieved_summary="",
        retrieval_method="newspaper4k.Article",
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def append_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Append rows to a CSV, creating the header when needed."""
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    frame = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    frame.to_csv(
        path,
        mode="a",
        index=False,
        header=write_header,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )


def completed_ids(*paths: Path) -> set[str]:
    """Return article IDs already written to any output file."""
    completed: set[str] = set()

    for path in paths:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(
                path,
                usecols=["article_id"],
                dtype="string",
            )
            completed.update(
                frame["article_id"].dropna().astype(str)
            )
        except Exception:
            continue

    return completed


def load_results(path: Path) -> pd.DataFrame:
    """Read one output CSV or return an empty result frame."""
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)

    try:
        return pd.read_csv(
            path,
            dtype="string",
            keep_default_na=False,
        )
    except Exception:
        return pd.DataFrame(columns=RESULT_COLUMNS)


def write_report(
    *,
    success_path: Path,
    unsuccessful_path: Path,
    report_path: Path,
    input_path: Path,
    input_rows: int,
    unique_rows: int,
    duplicate_links_removed: int,
    args: argparse.Namespace,
) -> None:
    successes = load_results(success_path)
    unsuccessful = load_results(unsuccessful_path)
    results = pd.concat(
        [successes, unsuccessful],
        ignore_index=True,
    )

    status_counts = (
        results["retrieval_status"]
        .value_counts(dropna=False)
        .to_dict()
    )

    total = len(results)
    success_count = len(successes)
    unsuccessful_count = len(unsuccessful)
    success_rate = (
        success_count / total * 100
        if total
        else 0.0
    )

    report = [
        "# LLM News Full-text Retrieval Report: Newspaper4k",
        "",
        f"Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Input",
        "",
        f"- File: `{input_path}`",
        f"- Rows loaded: **{input_rows:,}**",
        f"- Unique normalized article links: **{unique_rows:,}**",
        (
            "- Duplicate links removed before retrieval: "
            f"**{duplicate_links_removed:,}**"
        ),
        "",
        "## Run settings",
        "",
        f"- Language: `{args.language}`",
        f"- Minimum words for success: `{args.min_words}`",
        f"- Request timeout: `{args.timeout}` seconds",
        f"- Base sleep: `{args.sleep}` seconds",
        f"- Random jitter: `{args.jitter}` seconds",
        f"- NLP enabled: `{args.run_nlp}`",
        f"- Resume mode: `{args.resume}`",
        "",
        "## Results",
        "",
        f"- Total attempted rows saved: **{total:,}**",
        f"- Succeeded rows: **{success_count:,}**",
        f"- Unsuccessful rows: **{unsuccessful_count:,}**",
        f"- Success rate: **{success_rate:.2f}%**",
        f"- Succeeded file: `{success_path}`",
        f"- Unsuccessful file: `{unsuccessful_path}`",
        "",
        "### Retrieval status counts",
        "",
        "```json",
        json.dumps(
            status_counts,
            indent=2,
            ensure_ascii=False,
        ),
        "```",
        "",
        "## Status definitions",
        "",
        "- `success`: extracted text met the minimum word threshold.",
        "- `too_short`: extracted text was below the threshold.",
        "- `empty_text`: no article body was extracted.",
        "- `failed`: downloading or parsing raised an exception.",
        "- `invalid_url`: the input URL was missing or malformed.",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    root = project_root()
    output_dir = (
        root
        / "data"
        / "02_interim"
        / "article_enrichment"
        / "full_text_llm_news4k"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Retrieve full text and save succeeded and "
            "unsuccessful rows in separate CSV files."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            root
            / "data"
            / "02_interim"
            / "news_cleaned"
            / "llm_search_results_cleaned.csv"
        ),
    )
    parser.add_argument(
        "--success-output",
        type=Path,
        default=(
            output_dir
            / "llm_full_text_news4k_succeeded.csv"
        ),
    )
    parser.add_argument(
        "--unsuccessful-output",
        type=Path,
        default=(
            output_dir
            / "llm_full_text_news4k_unsucceeded.csv"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            root
            / "outputs"
            / "reports"
            / "llm_full_text_news4k_report.md"
        ),
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--jitter", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--run-nlp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if Article is None or Config is None:
        print(
            "ERROR: Newspaper4k is not installed "
            "or could not be imported.",
            file=sys.stderr,
        )
        print(
            "Install it with: pip install newspaper4k",
            file=sys.stderr,
        )
        print(
            f"Original import error: {NEWSPAPER_IMPORT_ERROR}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    args.input = args.input.expanduser().resolve()
    args.success_output = (
        args.success_output.expanduser().resolve()
    )
    args.unsuccessful_output = (
        args.unsuccessful_output.expanduser().resolve()
    )
    args.report = args.report.expanduser().resolve()

    if not args.input.exists():
        print(
            f"ERROR: input file not found: {args.input}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    output_files = [
        args.success_output,
        args.unsuccessful_output,
    ]

    if args.overwrite:
        for path in output_files:
            if path.exists():
                path.unlink()
    elif any(path.exists() for path in output_files) and not args.resume:
        existing = [
            str(path)
            for path in output_files
            if path.exists()
        ]
        print(
            "ERROR: one or more output files already exist:\n"
            + "\n".join(existing)
            + "\nUse --resume to continue or --overwrite to start again.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    raw_input_rows = len(
        pd.read_csv(
            args.input,
            dtype="string",
            keep_default_na=False,
        )
    )

    data = read_input(args.input)
    duplicate_links_removed = int(
        data.attrs.get("duplicate_links_removed", 0)
    )
    unique_rows = len(data)

    data = data.iloc[max(args.start_row, 0):].copy()

    if args.max_rows is not None:
        data = data.head(max(args.max_rows, 0)).copy()

    done_ids = (
        completed_ids(
            args.success_output,
            args.unsuccessful_output,
        )
        if args.resume
        else set()
    )

    success_buffer: list[dict[str, Any]] = []
    unsuccessful_buffer: list[dict[str, Any]] = []

    attempted = 0
    skipped = 0
    succeeded_this_run = 0
    unsuccessful_this_run = 0

    print(f"Input file: {args.input}")
    print(f"Rows loaded: {raw_input_rows:,}")
    print(f"Unique links available: {unique_rows:,}")
    print(f"Rows selected this run: {len(data):,}")
    print(f"Already completed IDs: {len(done_ids):,}")
    print(f"Succeeded output: {args.success_output}")
    print(f"Unsuccessful output: {args.unsuccessful_output}")

    try:
        for selected_number, (_, row) in enumerate(
            data.iterrows(),
            start=1,
        ):
            article_id = clean_text(row["article_id"])
            link = clean_text(row["link"])
            normalized_link = clean_text(row["normalized_link"])
            headline = clean_text(row["headline"])
            source_name = clean_text(row["source_name"])
            date = clean_text(row["date"])

            if args.resume and article_id in done_ids:
                skipped += 1
                continue

            if not valid_url(normalized_link):
                result = failure_result(
                    article_id=article_id,
                    link=link,
                    normalized_link=normalized_link,
                    headline=headline,
                    source_name=source_name,
                    date=date,
                    status="invalid_url",
                    error_type="InvalidURL",
                    error_message=(
                        "Missing or malformed HTTP/HTTPS URL"
                    ),
                )
            else:
                try:
                    result = extract_article(
                        article_id=article_id,
                        link=link,
                        normalized_link=normalized_link,
                        headline=headline,
                        source_name=source_name,
                        date=date,
                        language=args.language,
                        request_timeout=args.timeout,
                        run_nlp=args.run_nlp,
                        min_words=args.min_words,
                    )
                except Exception as exc:
                    result = failure_result(
                        article_id=article_id,
                        link=link,
                        normalized_link=normalized_link,
                        headline=headline,
                        source_name=source_name,
                        date=date,
                        status="failed",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )

            record = asdict(result)

            if result.retrieval_status == "success":
                success_buffer.append(record)
                succeeded_this_run += 1
            else:
                unsuccessful_buffer.append(record)
                unsuccessful_this_run += 1

            attempted += 1

            flush_every = max(args.checkpoint_every, 1)
            buffered_count = (
                len(success_buffer)
                + len(unsuccessful_buffer)
            )

            if buffered_count >= flush_every:
                append_rows(
                    success_buffer,
                    args.success_output,
                )
                append_rows(
                    unsuccessful_buffer,
                    args.unsuccessful_output,
                )
                success_buffer.clear()
                unsuccessful_buffer.clear()

            if selected_number % flush_every == 0:
                print(
                    f"Selected {selected_number:,}/{len(data):,} | "
                    f"attempted={attempted:,} | "
                    f"succeeded={succeeded_this_run:,} | "
                    f"unsuccessful={unsuccessful_this_run:,} | "
                    f"skipped={skipped:,} | "
                    f"last_status={result.retrieval_status}"
                )

            if args.sleep > 0:
                time.sleep(
                    args.sleep
                    + random.random() * max(args.jitter, 0.0)
                )

    except KeyboardInterrupt:
        print(
            "\nInterrupted. Saving current buffers before exit..."
        )

    finally:
        append_rows(
            success_buffer,
            args.success_output,
        )
        append_rows(
            unsuccessful_buffer,
            args.unsuccessful_output,
        )

    write_report(
        success_path=args.success_output,
        unsuccessful_path=args.unsuccessful_output,
        report_path=args.report,
        input_path=args.input,
        input_rows=raw_input_rows,
        unique_rows=unique_rows,
        duplicate_links_removed=duplicate_links_removed,
        args=args,
    )

    successes = load_results(args.success_output)
    unsuccessful = load_results(args.unsuccessful_output)

    print("\n[DONE]")
    print(f"Attempted in this run: {attempted:,}")
    print(f"Skipped in resume mode: {skipped:,}")
    print(f"Succeeded in this run: {succeeded_this_run:,}")
    print(
        f"Unsuccessful in this run: "
        f"{unsuccessful_this_run:,}"
    )
    print(f"Total succeeded rows: {len(successes):,}")
    print(
        f"Total unsuccessful rows: "
        f"{len(unsuccessful):,}"
    )
    print(f"Succeeded file: {args.success_output}")
    print(
        f"Unsuccessful file: "
        f"{args.unsuccessful_output}"
    )
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
