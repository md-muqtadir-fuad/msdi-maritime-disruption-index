"""Standardize raw CPB, RWI/ISL, WTO, DEXCHUS, UNCTAD, GSCPI, PortWatch, and Shock-Sensitivity controls.

Raw files remain untouched under ``data/01_raw/controls``. This script writes
an auditable long-format dataset with source dates and conservative
availability dates; later panel builders must use backward-only joins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/external_controls.yaml"
DEFAULT_OUTPUT = ROOT / "data/02_interim/controls/external_controls_clean_long.csv"
DEFAULT_MANIFEST = ROOT / "data/02_interim/controls/external_controls_clean_manifest.json"
DEFAULT_LOG_DIR = ROOT / "outputs/logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean external control source files.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--version", default="external_controls_v002")
    parser.add_argument(
        "--allow-missing-sources",
        action="store_true",
        help="Write available sources and record missing ones.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate configuration only; do not read or write data.",
    )
    return parser.parse_args()


def norm(name: object) -> str:
    return "_".join(str(name).strip().lower().replace("-", " ").replace("/", " ").split())


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_matching_column(frame: pd.DataFrame, candidates: list[str], label: str) -> str:
    lookup = {norm(column): str(column) for column in frame.columns}
    for candidate in candidates:
        if norm(candidate) in lookup:
            return lookup[norm(candidate)]
    raise ValueError(f"No {label} column found. Available columns: {list(frame.columns)}")


def discover_file(spec: dict[str, Any]) -> Path | None:
    directory = ROOT / spec["raw_directory"]
    if not directory.exists():
        return None
    matches: list[Path] = []
    for pattern in spec["file_patterns"]:
        matches.extend(sorted(path for path in directory.glob(pattern) if path.is_file()))
    if not matches:
        return None
    # Remove duplicates while preserving order
    unique_matches: list[Path] = []
    for m in matches:
        if m not in unique_matches:
            unique_matches.append(m)
    return unique_matches[0]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin1", low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported raw control format: {path}")


def clean_cpb(spec: dict[str, Any], path: Path) -> tuple[pd.DataFrame, str, str, int]:
    """Parse CPB's wide World Trade Monitor workbook without altering it."""
    raw = pd.read_excel(path, header=None)
    date_row = next(
        index
        for index, row in raw.iterrows()
        if row.astype(str).str.fullmatch(r"\d{4}m\d{2}", case=False).any()
    )
    value_row = next(
        index
        for index, row in raw.iterrows()
        if row.astype(str).str.strip().str.casefold().eq("world imports").any()
    )
    dates = raw.iloc[date_row]
    values = raw.iloc[value_row]
    records = []
    for date, value in zip(dates, values):
        if isinstance(date, str) and len(date) == 7 and date[4].lower() == "m":
            records.append((pd.Timestamp(f"{date[:4]}-{date[5:]}-01"), value))
    frame = pd.DataFrame(records, columns=["observation_date", "value"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(), "wide_month_codes", "World imports", len(raw)


def clean_lsbci(spec: dict[str, Any], path: Path) -> tuple[pd.DataFrame, str, str, int]:
    raw = read_table(path)
    origin = spec.get("origin_economy")
    partners = set(spec.get("focus_partners", []))
    if origin and partners:
        raw = raw.loc[
            raw["Economy Label"].eq(origin) & raw["Partner Label"].isin(partners)
        ].copy()
        if raw.empty:
            raise ValueError(
                "UNCTAD LSBCI route filter produced no rows; check origin_economy/focus_partners."
            )
    date_candidates = spec.get("date_columns", ["Quarter", "quarter", "date", "month", "period", "time_period", "observation_date"])
    value_candidates = spec.get("value_columns", ["Index", "index", "lsbci", "value"])
    date_column = first_matching_column(raw, list(date_candidates), "date")
    value_column = first_matching_column(raw, list(value_candidates), "value")
    frame = pd.DataFrame(
        {
            "observation_date": pd.PeriodIndex(
                raw[date_column].astype(str).str.replace("Q0", "Q", regex=False),
                freq="Q",
            )
            .to_timestamp(how="end")
            .normalize(),
            "value": pd.to_numeric(raw[value_column], errors="coerce"),
        }
    ).dropna()
    return frame, date_column, value_column, len(raw)


def clean_portwatch(spec: dict[str, Any], path: Path) -> tuple[pd.DataFrame, str, str, int]:
    """Aggregate the large daily PortWatch extract in chunks to limit memory use."""
    pieces, total = [], 0
    date_candidates = spec.get("date_columns", ["date", "day", "period", "time_period", "observation_date"])
    value_candidates = spec.get("value_columns", ["portcalls_container", "port_calls", "value"])
    for chunk in pd.read_csv(
        path,
        usecols=lambda column: norm(column) in {"date", "portcalls_container"},
        chunksize=250_000,
    ):
        total += len(chunk)
        date_column = first_matching_column(chunk, list(date_candidates), "date")
        value_column = first_matching_column(chunk, list(value_candidates), "value")
        part = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(chunk[date_column], errors="coerce"),
                "value": pd.to_numeric(chunk[value_column], errors="coerce"),
            }
        ).dropna()
        pieces.append(part.groupby("observation_date", as_index=False).value.sum())
    frame = pd.concat(pieces, ignore_index=True).groupby("observation_date", as_index=False).value.sum()
    return frame, "date", "portcalls_container", total


def clean_rwi_isl(spec: dict[str, Any], path: Path) -> tuple[pd.DataFrame, str, str, int]:
    """Extract seasonally and working day adjusted headline container throughput index."""
    raw = pd.read_excel(path, sheet_name="Output", header=None)
    dates = pd.to_datetime(raw.iloc[6:, 0], errors="coerce")
    vals = pd.to_numeric(raw.iloc[6:, 2], errors="coerce")
    frame = pd.DataFrame({"observation_date": dates, "value": vals}).dropna()
    return frame, "Col 0 (Date)", "Col 2 (Total SA Index)", len(raw)


def clean_wto(spec: dict[str, Any], path: Path) -> tuple[pd.DataFrame, str, str, int]:
    """Parse WTO monthly merchandise exports for China to World."""
    raw = pd.read_csv(path, encoding="latin1", low_memory=False)
    indicator_code = spec.get("indicator_code", "ITS_MTV_MX")
    reporting_economy = spec.get("reporting_economy", "China")
    partner_economy = spec.get("partner_economy", "World")
    sub = raw[
        (raw["Indicator Code"] == indicator_code)
        & (raw["Reporting Economy"] == reporting_economy)
        & (raw["Partner Economy"] == partner_economy)
    ].copy()
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }
    sub["month_num"] = sub["Period"].map(month_map)
    sub = sub.dropna(subset=["month_num"]).copy()
    sub["observation_date"] = pd.to_datetime(
        sub["Year"].astype(int).astype(str)
        + "-"
        + sub["month_num"].astype(int).astype(str).str.zfill(2)
        + "-01"
    )
    sub["value"] = pd.to_numeric(sub["Value"], errors="coerce")
    frame = sub[["observation_date", "value"]].dropna().sort_values("observation_date")
    return frame, "Year + Period", "Value (Monthly USD M)", len(raw)


def clean_gpr(spec: dict[str, Any], path: Path) -> tuple[pd.DataFrame, str, str, int]:
    """Parse Geopolitical Risk index."""
    raw = read_table(path)
    date_col = None
    val_col = None
    for c in raw.columns:
        if norm(c) in ["month", "date", "day", "observation_date"]:
            date_col = c
            break
    for c in raw.columns:
        if str(c).upper() in ["GPR", "GPRD", "GPR_INDEX", "VALUE"]:
            val_col = c
            break
    if date_col is None:
        date_col = first_matching_column(raw, list(spec.get("date_columns", ["month", "date", "day"])), "date")
    if val_col is None:
        val_col = first_matching_column(raw, list(spec.get("value_columns", ["GPR", "gpr", "GPRD", "gprd"])), "value")
    dates = pd.to_datetime(raw[date_col], errors="coerce")
    vals = pd.to_numeric(raw[val_col], errors="coerce")
    frame = pd.DataFrame({"observation_date": dates, "value": vals}).dropna()
    return frame, str(date_col), str(val_col), len(raw)


def clean_uct(spec: dict[str, Any], path: Path) -> tuple[pd.DataFrame, str, str, int]:
    """Parse US-China Tension index with monthly '1993m1' dates."""
    raw = pd.read_csv(path)
    d_clean = raw["date"].astype(str).str.strip().str.lower().str.replace("m", "-")
    dates = pd.to_datetime(
        d_clean.apply(lambda x: f"{x}-01" if "-" in x and len(x.split("-")[1]) in (1, 2) else np.nan),
        errors="coerce",
    )
    vals = pd.to_numeric(raw["UCT"], errors="coerce")
    frame = pd.DataFrame({"observation_date": dates, "value": vals}).dropna()
    return frame, "date", "UCT", len(raw)


def clean_emv(spec: dict[str, Any], path: Path) -> tuple[pd.DataFrame, str, str, int]:
    """Parse daily Infectious Disease Equity Market Volatility index."""
    raw = pd.read_csv(path)
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    raw["month"] = pd.to_numeric(raw["month"], errors="coerce")
    raw["day"] = pd.to_numeric(raw["day"], errors="coerce")
    valid = raw.dropna(subset=["year", "month", "day"]).copy()
    valid["observation_date"] = pd.to_datetime(
        valid["year"].astype(int).astype(str)
        + "-"
        + valid["month"].astype(int).astype(str).str.zfill(2)
        + "-"
        + valid["day"].astype(int).astype(str).str.zfill(2),
        errors="coerce",
    )
    valid["value"] = pd.to_numeric(valid["daily_infect_emv_index"], errors="coerce")
    frame = valid[["observation_date", "value"]].dropna()
    return frame, "year/month/day", "daily_infect_emv_index", len(raw)


def clean_gscpi(spec: dict[str, Any], path: Path) -> tuple[pd.DataFrame, str, str, int]:
    """Parse NY Fed GSCPI monthly spreadsheet."""
    raw = pd.read_excel(path, sheet_name="GSCPI Monthly Data")
    header_idx = None
    for idx, row in raw.iterrows():
        if row.astype(str).str.contains("Date", case=False).any() and row.astype(str).str.contains("GSCPI", case=False).any():
            header_idx = idx
            break
    if header_idx is not None:
        raw.columns = raw.iloc[header_idx]
        raw = raw.iloc[header_idx + 1 :]
    date_col = [c for c in raw.columns if "date" in str(c).lower()][0]
    val_col = [c for c in raw.columns if "gscpi" in str(c).lower()][0]
    dates = pd.to_datetime(raw[date_col], errors="coerce")
    vals = pd.to_numeric(raw[val_col], errors="coerce")
    frame = pd.DataFrame({"observation_date": dates, "value": vals}).dropna()
    return frame, str(date_col), str(val_col), len(raw)


def clean_tpu(spec: dict[str, Any], path: Path) -> tuple[pd.DataFrame, str, str, int]:
    """Parse Trade Policy Uncertainty index."""
    raw = pd.read_excel(path, sheet_name="TPU_MONTHLY")
    dates = pd.to_datetime(raw["DATE"], errors="coerce")
    vals = pd.to_numeric(raw["TPU"], errors="coerce")
    frame = pd.DataFrame({"observation_date": dates, "value": vals}).dropna()
    return frame, "DATE", "TPU", len(raw)


def clean_one(source_id: str, spec: dict[str, Any], path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if source_id == "cpb_world_trade_monitor":
        frame, date_column, value_column, raw_rows = clean_cpb(spec, path)
    elif source_id == "unctad_lsbci":
        frame, date_column, value_column, raw_rows = clean_lsbci(spec, path)
    elif source_id == "imf_portwatch":
        frame, date_column, value_column, raw_rows = clean_portwatch(spec, path)
    elif source_id == "rwi_isl_container_throughput":
        frame, date_column, value_column, raw_rows = clean_rwi_isl(spec, path)
    elif source_id == "wto_china_merchandise_exports":
        frame, date_column, value_column, raw_rows = clean_wto(spec, path)
    elif source_id == "gpr_risk":
        frame, date_column, value_column, raw_rows = clean_gpr(spec, path)
    elif source_id == "uct_index":
        frame, date_column, value_column, raw_rows = clean_uct(spec, path)
    elif source_id == "emv_infectious":
        frame, date_column, value_column, raw_rows = clean_emv(spec, path)
    elif source_id == "gscpi":
        frame, date_column, value_column, raw_rows = clean_gscpi(spec, path)
    elif source_id == "tpu":
        frame, date_column, value_column, raw_rows = clean_tpu(spec, path)
    else:
        raw = read_table(path)
        raw_rows = len(raw)
        date_column = first_matching_column(raw, list(spec["date_columns"]), "date")
        value_column = first_matching_column(raw, list(spec["value_columns"]), "value")
        frame = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(raw[date_column], errors="coerce"),
                "value": pd.to_numeric(raw[value_column], errors="coerce"),
            }
        ).dropna(subset=["observation_date", "value"])

    aggregation = spec.get("aggregation", "last")
    grouped = frame.groupby("observation_date", as_index=False)["value"]
    frame = (
        grouped.mean()
        if aggregation == "mean"
        else grouped.sum()
        if aggregation == "sum"
        else grouped.last()
    )
    frame["source_id"] = source_id
    frame["variable"] = spec["output_variable"]
    frame["source_frequency"] = spec["frequency"]
    frame["role"] = spec.get("role", "core_control")
    frame["availability_date"] = frame["observation_date"] + pd.Timedelta(
        days=int(spec["release_lag_days"])
    )
    frame["raw_file"] = str(path.relative_to(ROOT))
    frame = frame[
        [
            "source_id",
            "variable",
            "observation_date",
            "availability_date",
            "value",
            "source_frequency",
            "role",
            "raw_file",
        ]
    ]
    report = {
        "raw_file": str(path.relative_to(ROOT)),
        "sha256": file_hash(path),
        "rows_raw": raw_rows,
        "rows_usable": len(frame),
        "date_column": date_column,
        "value_column": value_column,
        "role": spec.get("role", "core_control"),
        "first_date": str(frame.observation_date.min().date()),
        "last_date": str(frame.observation_date.max().date()),
    }
    return frame, report


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config.get("sources"), dict):
        raise ValueError("Configuration must define sources.")
    if args.smoke_test:
        print(f"Configuration valid: {len(config['sources'])} source specifications")
        return
    logger = logging.getLogger("clean_external_controls")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(
        args.log_dir / f"25_clean_external_controls_{args.version}.log",
        encoding="utf-8",
    )
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    pieces, audits, missing = [], {}, []
    for source_id, spec in config["sources"].items():
        path = discover_file(spec)
        if path is None:
            missing.append(source_id)
            logger.warning("Missing raw source: %s", source_id)
            continue
        cleaned, report = clean_one(source_id, spec, path)
        pieces.append(cleaned)
        audits[source_id] = report
        logger.info(
            "Cleaned %s (%s): %s rows [%s to %s]",
            source_id,
            spec["output_variable"],
            len(cleaned),
            report["first_date"],
            report["last_date"],
        )

    if missing and not args.allow_missing_sources:
        raise FileNotFoundError(
            f"Missing raw source files: {missing}. Use --allow-missing-sources only for a partial run."
        )
    if not pieces:
        raise ValueError("No external-control source files were cleaned.")

    result = pd.concat(pieces, ignore_index=True).sort_values(["variable", "observation_date"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    manifest = {
        "version": args.version,
        "config": str(args.config.relative_to(ROOT)),
        "output": str(args.output.relative_to(ROOT)),
        "rows": len(result),
        "variables": sorted(result.variable.unique().tolist()),
        "missing_sources": missing,
        "sources": audits,
        "availability_rule": "availability_date = observation_date + configured conservative release lag; downstream joins must be backward-only",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%s rows) and %s", args.output, len(result), args.manifest)


if __name__ == "__main__":
    main()
