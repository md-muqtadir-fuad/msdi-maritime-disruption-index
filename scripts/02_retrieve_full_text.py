"""Retrieve full article text with Newspaper4k.

Input default:
    data/02_interim/news_cleaned/legacy_news_unique_for_retrieval.csv

Outputs default:
    data/02_interim/article_enrichment/full_text_news4k/full_text_news4k_results.csv
    data/02_interim/article_enrichment/full_text_news4k/full_text_news4k_failed.csv
    data/02_interim/article_enrichment/full_text_news4k/full_text_news4k_checkpoint.csv
    outputs/reports/full_text_news4k_report.md

Install dependency:
    pip install newspaper4k

Run from project root:
    python scripts/02_retrieve_full_text_news4k.py

Useful test run:
    python scripts/02_retrieve_full_text_news4k.py --max-rows 50 --sleep 1

Resume after interruption:
    python scripts/02_retrieve_full_text_news4k.py --resume
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

try:
    import newspaper  # installed by: pip install newspaper4k
except Exception as exc:  # pragma: no cover
    newspaper = None
    NEWSPAPER_IMPORT_ERROR = exc
else:
    NEWSPAPER_IMPORT_ERROR = None


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


@dataclass
class ExtractionResult:
    article_id: str
    url: str
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


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_url(url: Any) -> str:
    if pd.isna(url):
        return ""
    url = str(url).strip()
    if not url or url.lower() in {"nan", "none", "null"}:
        return ""
    if url.startswith("www."):
        url = "https://" + url
    return url


def valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sha256_text(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def safe_join_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(x) for x in value if str(x).strip())
    return str(value)


def date_to_iso(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        if hasattr(value, "isoformat"):
            return value.isoformat()
        parsed = pd.to_datetime(value, errors="coerce")
        return "" if pd.isna(parsed) else parsed.isoformat()
    except Exception:
        return str(value)


def get_attr(article: Any, attr: str, default: Any = "") -> Any:
    try:
        return getattr(article, attr, default)
    except Exception:
        return default


def extract_with_newspaper4k(url: str, language: str, run_nlp: bool) -> ExtractionResult:
    if newspaper is None:
        raise RuntimeError(
            f"Could not import newspaper4k. Install with: pip install newspaper4k. Original error: {NEWSPAPER_IMPORT_ERROR}"
        )

    retrieved_at = datetime.now(timezone.utc).isoformat()
    article = None
    method = "newspaper.article"

    # Newspaper4k convenience API: downloads and parses in one call.
    try:
        kwargs: dict[str, Any] = {}
        if language:
            kwargs["language"] = language
        # Some newspaper4k versions accept browser_user_agent; older newspaper3k-style APIs may not.
        try:
            article = newspaper.article(url, browser_user_agent=USER_AGENT, **kwargs)
        except TypeError:
            article = newspaper.article(url, **kwargs)
    except Exception:
        # Fallback to newspaper3k-compatible Article API retained by newspaper4k.
        method = "newspaper.Article_fallback"
        try:
            from newspaper import Article, Config

            config = Config()
            config.browser_user_agent = USER_AGENT
            config.request_timeout = 20
            article = Article(url, language=language or "en", config=config)
            article.download()
            article.parse()
        except Exception:
            raise

    if run_nlp:
        try:
            article.nlp()
        except Exception:
            # NLP is optional; extraction should not fail only because keyword/summary generation fails.
            pass

    text = clean_text(get_attr(article, "text", ""))
    title = clean_text(get_attr(article, "title", ""))
    final_url = str(get_attr(article, "url", url) or url)
    words = re.findall(r"\b\w+\b", text)

    return ExtractionResult(
        article_id="",  # filled by caller
        url=url,
        final_url=final_url,
        source_domain=domain_of(final_url or url),
        retrieval_status="success" if text else "empty_text",
        error_type="",
        error_message="",
        retrieved_title=title,
        retrieved_text=text,
        retrieved_text_len=len(text),
        retrieved_word_count=len(words),
        retrieved_text_hash=sha256_text(text),
        retrieved_publish_date=date_to_iso(get_attr(article, "publish_date", "")),
        retrieved_authors=safe_join_list(get_attr(article, "authors", [])),
        retrieved_top_image=str(get_attr(article, "top_image", "") or ""),
        retrieved_meta_description=clean_text(get_attr(article, "meta_description", "")),
        retrieved_meta_lang=str(get_attr(article, "meta_lang", "") or ""),
        retrieved_keywords=safe_join_list(get_attr(article, "keywords", [])),
        retrieved_summary=clean_text(get_attr(article, "summary", "")),
        retrieval_method=method,
        retrieved_at_utc=retrieved_at,
    )


def make_failure(article_id: str, url: str, status: str, error_type: str, error_message: str) -> ExtractionResult:
    now = datetime.now(timezone.utc).isoformat()
    return ExtractionResult(
        article_id=article_id,
        url=url,
        final_url="",
        source_domain=domain_of(url),
        retrieval_status=status,
        error_type=error_type,
        error_message=str(error_message)[:1000],
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
        retrieval_method="newspaper4k",
        retrieved_at_utc=now,
    )


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        done = pd.read_csv(path, usecols=["article_id"], dtype=str)
        return set(done["article_id"].dropna().astype(str))
    except Exception:
        return set()


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    tmp.replace(path)


def append_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def build_report(results: pd.DataFrame, out_report: Path, args: argparse.Namespace, input_rows: int) -> None:
    status_counts = results["retrieval_status"].value_counts(dropna=False).to_dict() if not results.empty else {}
    total = len(results)
    success = int((results["retrieval_status"] == "success").sum()) if not results.empty else 0
    empty = int((results["retrieval_status"] == "empty_text").sum()) if not results.empty else 0
    failed = total - success - empty
    success_rate = (success / total * 100) if total else 0

    top_domains = {}
    if not results.empty and "source_domain" in results.columns:
        top_domains = results["source_domain"].value_counts().head(20).to_dict()

    md = [
        "# Full-text Retrieval Report: Newspaper4k\n",
        f"Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`\n",
        "## Inputs\n",
        f"- Input file: `{args.input}`",
        f"- Input rows loaded: **{input_rows:,}**",
        f"- Max rows argument: `{args.max_rows}`",
        f"- Resume mode: `{args.resume}`",
        f"- Sleep seconds: `{args.sleep}`",
        f"- Language: `{args.language}`",
        f"- NLP enabled: `{args.run_nlp}`\n",
        "## Outputs\n",
        f"- Results: `{args.output}`",
        f"- Failures: `{args.failed_output}`",
        f"- Checkpoint: `{args.checkpoint}`\n",
        "## Summary\n",
        f"- Total attempted/written rows: **{total:,}**",
        f"- Successful full-text rows: **{success:,}**",
        f"- Empty-text rows: **{empty:,}**",
        f"- Failed/invalid/skipped rows: **{failed:,}**",
        f"- Success rate among written rows: **{success_rate:.2f}%**\n",
        "## Status counts\n",
        "```json",
        json.dumps(status_counts, indent=2, ensure_ascii=False),
        "```\n",
        "## Top source domains\n",
        "```json",
        json.dumps(top_domains, indent=2, ensure_ascii=False),
        "```\n",
        "## Notes\n",
        "- `success` means Newspaper4k returned non-empty article text.",
        "- `empty_text` usually means paywall, blocked page, JavaScript-heavy page, non-article URL, or parser failure.",
        "- Use `full_text_news4k_failed.csv` for later fallback extraction using requests/trafilatura/Playwright/GDELT.",
    ]
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(md), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = project_root_from_script()
    default_input = root / "data/02_interim/news_cleaned/legacy_news_unique_for_retrieval.csv"
    default_dir = root / "data/02_interim/article_enrichment/full_text_news4k"
    default_report = root / "outputs/reports/full_text_news4k_report.md"

    parser = argparse.ArgumentParser(description="Retrieve full news text using Newspaper4k.")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_dir / "full_text_news4k_results.csv")
    parser.add_argument("--failed-output", type=Path, default=default_dir / "full_text_news4k_failed.csv")
    parser.add_argument("--checkpoint", type=Path, default=default_dir / "full_text_news4k_checkpoint.csv")
    parser.add_argument("--report", type=Path, default=default_report)
    parser.add_argument("--url-col", default="url")
    parser.add_argument("--id-col", default="article_id")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.5, help="Base sleep between requests.")
    parser.add_argument("--jitter", type=float, default=0.5, help="Random extra sleep between 0 and jitter.")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--min-words", type=int, default=80, help="Mark extracted text below this word count as too_short.")
    parser.add_argument("--resume", action="store_true", help="Skip article_ids already present in output/checkpoint.")
    parser.add_argument("--run-nlp", action="store_true", help="Run Newspaper4k NLP keyword/summary extraction. Slower.")
    parser.add_argument("--overwrite", action="store_true", help="Delete previous output files before running.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if newspaper is None:
        print("ERROR: newspaper4k is not installed or could not be imported.", file=sys.stderr)
        print("Install it with: pip install newspaper4k", file=sys.stderr)
        print(f"Original import error: {NEWSPAPER_IMPORT_ERROR}", file=sys.stderr)
        sys.exit(1)

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.overwrite:
        for p in [args.output, args.failed_output, args.checkpoint]:
            if p.exists():
                p.unlink()

    df = pd.read_csv(args.input, dtype=str).fillna("")
    input_rows = len(df)

    if args.url_col not in df.columns:
        print(f"ERROR: URL column `{args.url_col}` not found. Available: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    if args.id_col not in df.columns:
        df[args.id_col] = [f"row_{i:07d}" for i in range(len(df))]

    df[args.url_col] = df[args.url_col].map(normalize_url)
    df = df.iloc[args.start_row :].copy()
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    done_ids: set[str] = set()
    if args.resume:
        done_ids = load_done_ids(args.output) | load_done_ids(args.checkpoint)

    rows_buffer: list[dict[str, Any]] = []
    fail_buffer: list[dict[str, Any]] = []
    all_written_now: list[dict[str, Any]] = []

    total_to_process = len(df)
    print(f"Input rows loaded: {input_rows:,}")
    print(f"Rows selected this run: {total_to_process:,}")
    print(f"Resume skip ids: {len(done_ids):,}")
    print(f"Output: {args.output}")

    for n, (_, row) in enumerate(df.iterrows(), start=1):
        article_id = str(row.get(args.id_col, "")).strip() or f"row_{n:07d}"
        url = normalize_url(row.get(args.url_col, ""))

        if args.resume and article_id in done_ids:
            continue

        if not valid_url(url):
            result = make_failure(article_id, url, "invalid_url", "InvalidURL", "Missing or malformed URL")
        else:
            try:
                result = extract_with_newspaper4k(url, args.language, args.run_nlp)
                result.article_id = article_id
                if result.retrieval_status == "success" and result.retrieved_word_count < args.min_words:
                    result.retrieval_status = "too_short"
            except KeyboardInterrupt:
                print("Interrupted. Writing buffers before exit...")
                break
            except Exception as exc:  # noqa: BLE001
                result = make_failure(article_id, url, "failed", type(exc).__name__, str(exc))

        out = asdict(result)
        # Carry useful normalized metadata forward for joins and auditing.
        for col in [
            "headline",
            "date",
            "source_file",
            "source_dataset",
            "source_name",
            "headline_clean",
            "headline_hash",
            "legacy_score",
            "legacy_themes",
            "legacy_dimension",
        ]:
            if col in row.index:
                out[col] = row.get(col, "")

        rows_buffer.append(out)
        all_written_now.append(out)
        if result.retrieval_status not in {"success"}:
            fail_buffer.append(out)

        if len(rows_buffer) >= args.checkpoint_every:
            append_csv(rows_buffer, args.output)
            append_csv(rows_buffer, args.checkpoint)
            append_csv(fail_buffer, args.failed_output)
            rows_buffer.clear()
            fail_buffer.clear()

        if n % args.checkpoint_every == 0:
            print(f"Processed selected row {n:,}/{total_to_process:,} | last status={result.retrieval_status} | url={url[:80]}")

        if args.sleep > 0:
            time.sleep(args.sleep + random.random() * max(args.jitter, 0))

    append_csv(rows_buffer, args.output)
    append_csv(rows_buffer, args.checkpoint)
    append_csv(fail_buffer, args.failed_output)

    # Build report from complete output file, not just this run, when available.
    try:
        results_df = pd.read_csv(args.output, dtype=str).fillna("") if args.output.exists() else pd.DataFrame(all_written_now)
    except Exception:
        results_df = pd.DataFrame(all_written_now)
    build_report(results_df, args.report, args, input_rows)

    success = int((results_df["retrieval_status"] == "success").sum()) if not results_df.empty else 0
    print("Full-text retrieval complete.")
    print(f"Rows in output: {len(results_df):,}")
    print(f"Success rows: {success:,}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
