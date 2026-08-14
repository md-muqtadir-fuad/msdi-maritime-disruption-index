"""Point 1 of the EDA protocol: raw-data integrity and coverage audit."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/01_raw"
DEFAULT_OUTPUT = ROOT / "eda/outputs/point_01_raw_integrity"
STUDY_START = pd.Timestamp("2016-02-29")


SOURCE_CONTRACTS: dict[str, dict[str, Any]] = {
    "scfi": {
        "source": "Shanghai Shipping Exchange",
        "unit": "composite index points",
        "native_frequency": "weekly",
        "release_schedule": "generally Friday",
        "revision_policy": "requires source verification",
    },
    "rwi_container_throughput": {
        "source": "RWI/ISL Container Throughput Index",
        "unit": "seasonally and working-day-adjusted index, 2015=100",
        "native_frequency": "monthly",
        "release_schedule": "monthly flash estimate",
        "revision_policy": "revised as additional port observations arrive",
    },
    "china_merchandise_exports": {
        "source": "WTO Stats",
        "unit": "million US dollars",
        "native_frequency": "monthly",
        "release_schedule": "requires source verification",
        "revision_policy": "requires source verification",
    },
    "cny_per_usd": {
        "source": "Federal Reserve H.10 via FRED",
        "unit": "Chinese yuan per US dollar",
        "native_frequency": "daily business days",
        "release_schedule": "requires source verification",
        "revision_policy": "requires source verification",
    },
    "global_geopolitical_risk": {
        "source": "Caldara-Iacoviello Geopolitical Risk Index",
        "unit": "index, 1985-2019=100",
        "native_frequency": "monthly",
        "release_schedule": "updated monthly",
        "revision_policy": "requires source verification",
    },
    "trade_policy_uncertainty": {
        "source": "Caldara et al. Trade Policy Uncertainty Index",
        "unit": "index; 100 means 1% article share",
        "native_frequency": "monthly",
        "release_schedule": "updated monthly",
        "revision_policy": "requires source verification",
    },
    "us_china_tension": {
        "source": "Rogers-Sun-Sun US-China Tension Index",
        "unit": "index",
        "native_frequency": "monthly",
        "release_schedule": "requires source verification",
        "revision_policy": "requires source verification",
    },
    "oil_price_uncertainty": {
        "source": "Oil Price Uncertainty Index",
        "unit": "index",
        "native_frequency": "monthly",
        "release_schedule": "requires source verification",
        "revision_policy": "requires source verification",
    },
    "infectious_disease_emv": {
        "source": "Infectious Disease Equity Market Volatility Tracker",
        "unit": "daily news-based index",
        "native_frequency": "daily",
        "release_schedule": "requires source verification",
        "revision_policy": "requires source verification",
    },
    "brent_price": {
        "source": "legacy Brent extract",
        "unit": "US dollars per barrel",
        "native_frequency": "weekly in local raw file",
        "release_schedule": "requires source verification",
        "revision_policy": "requires source verification",
    },
    "cpb_world_trade_volume": {
        "source": "CPB World Trade Monitor",
        "unit": "fixed-base volume index, 2021=100",
        "native_frequency": "monthly",
        "release_schedule": "monthly; conservative pipeline lag 45 days",
        "revision_policy": "revised releases",
    },
    "gscpi": {
        "source": "Federal Reserve Bank of New York",
        "unit": "standard deviations from historical mean",
        "native_frequency": "monthly",
        "release_schedule": "monthly; conservative pipeline lag 5 days",
        "revision_policy": "requires source verification",
    },
    "unctad_china_core_lane_lsbci": {
        "source": "UNCTADstat bilateral liner shipping connectivity",
        "unit": "connectivity index",
        "native_frequency": "annual empirical cadence in attached extract",
        "release_schedule": "conservative pipeline lag 45 days",
        "revision_policy": "requires source verification",
    },
    "imf_portwatch_global_container_calls": {
        "source": "IMF PortWatch Daily Ports Data",
        "unit": "global daily container-vessel calls",
        "native_frequency": "daily",
        "release_schedule": "conservative pipeline lag 2 days",
        "revision_policy": "experimental data; requires verification",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit integrity of all raw project datasets.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-run", action="store_true", help="Avoid deep row scans for files larger than the configured threshold.")
    parser.add_argument("--deep-scan-limit-mb", type=float, default=30.0)
    return parser.parse_args()


def load_prototype_module():
    path = ROOT / "eda/00_raw_controls_eda.py"
    spec = importlib.util.spec_from_file_location("raw_controls_eda", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s]+", text)


def source_sidecar_audit() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW.rglob("source.txt")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        urls = extract_urls(text)
        malformed = [url for url in urls if url.lower().endswith(("xlsxex", "xlsxx", "csvv"))]
        rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "has_source_description": bool(text),
                "url_count": len(urls),
                "urls": " | ".join(urls),
                "malformed_urls": " | ".join(malformed),
                "requires_manual_source_completion": not urls or bool(malformed),
            }
        )
    return pd.DataFrame(rows)


def read_csv_header(path: Path) -> tuple[list[str], str]:
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            frame = pd.read_csv(path, nrows=0, encoding=encoding)
            return [str(column) for column in frame.columns], encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Unable to identify CSV encoding: {path}")


def inspect_tabular_file(path: Path, deep_scan: bool) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    relative = path.relative_to(ROOT).as_posix()
    rows: list[dict[str, Any]] = []
    if suffix == ".csv":
        columns, encoding = read_csv_header(path)
        record: dict[str, Any] = {
            "relative_path": relative,
            "table": "csv",
            "encoding": encoding,
            "column_count": len(columns),
            "columns": " | ".join(columns),
            "row_count": np.nan,
            "row_count_complete": False,
            "full_row_duplicates": np.nan,
            "empty_rows": np.nan,
            "read_error": "",
        }
        if deep_scan:
            try:
                frame = pd.read_csv(path, encoding=encoding, low_memory=False)
                record.update(
                    {
                        "row_count": len(frame),
                        "row_count_complete": True,
                        "full_row_duplicates": int(frame.duplicated().sum()),
                        "empty_rows": int(frame.isna().all(axis=1).sum()),
                    }
                )
            except Exception as exc:
                record["read_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(record)
    elif suffix in {".xls", ".xlsx"}:
        try:
            workbook = pd.ExcelFile(path)
            for sheet in workbook.sheet_names:
                record = {
                    "relative_path": relative,
                    "table": sheet,
                    "encoding": "binary_excel",
                    "column_count": np.nan,
                    "columns": "",
                    "row_count": np.nan,
                    "row_count_complete": False,
                    "full_row_duplicates": np.nan,
                    "empty_rows": np.nan,
                    "read_error": "",
                }
                if deep_scan:
                    try:
                        frame = pd.read_excel(path, sheet_name=sheet)
                        record.update(
                            {
                                "column_count": len(frame.columns),
                                "columns": " | ".join(str(column) for column in frame.columns),
                                "row_count": len(frame),
                                "row_count_complete": True,
                                "full_row_duplicates": int(frame.duplicated().sum()),
                                "empty_rows": int(frame.isna().all(axis=1).sum()),
                            }
                        )
                    except Exception as exc:
                        record["read_error"] = f"{type(exc).__name__}: {exc}"
                rows.append(record)
        except Exception as exc:
            rows.append(
                {
                    "relative_path": relative,
                    "table": "unreadable_workbook",
                    "encoding": "binary_excel",
                    "column_count": np.nan,
                    "columns": "",
                    "row_count": np.nan,
                    "row_count_complete": False,
                    "full_row_duplicates": np.nan,
                    "empty_rows": np.nan,
                    "read_error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def file_and_table_inventory(test_run: bool, deep_scan_limit_mb: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_rows = []
    table_rows: list[dict[str, Any]] = []
    sidecar_directories = {path.parent: path for path in RAW.rglob("source.txt")}
    for path in sorted(item for item in RAW.rglob("*") if item.is_file()):
        size_mb = path.stat().st_size / 1_000_000
        source_sidecar = sidecar_directories.get(path.parent)
        file_rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "extension": path.suffix.lower(),
                "size_mb": size_mb,
                "modified_utc": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat(),
                "has_source_sidecar": source_sidecar is not None,
                "source_sidecar": source_sidecar.relative_to(ROOT).as_posix() if source_sidecar else "",
            }
        )
        if path.suffix.lower() in {".csv", ".xls", ".xlsx"}:
            deep_scan = (not test_run) or size_mb <= deep_scan_limit_mb
            table_rows.extend(inspect_tabular_file(path, deep_scan))
    return pd.DataFrame(file_rows), pd.DataFrame(table_rows)


def load_all_series(module) -> tuple[pd.DataFrame, dict[str, str]]:
    loaders = [
        ("scfi", module.load_scfi),
        ("rwi", module.load_rwi),
        ("cny", module.load_cny),
        ("gpr", module.load_gpr),
        ("tpu", module.load_tpu),
        ("uct", module.load_uct),
        ("opu", module.load_opu),
        ("emv", module.load_emv),
        ("brent", module.load_brent),
        ("wto", module.load_wto_exports),
    ]
    pieces = []
    failures: dict[str, str] = {}
    for name, loader in loaders:
        try:
            pieces.append(loader(module.FILES[name]))
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    pieces.extend(module.load_existing_controls(module.EXISTING_CLEAN))
    return pd.concat(pieces, ignore_index=True), failures


def maximum_consecutive_missing(expected: pd.DatetimeIndex, present: pd.DatetimeIndex) -> int:
    missing = expected.isin(present) == 0
    best = current = 0
    for value in missing:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def integrity_reports(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scfi_end = long.loc[long["variable"].eq("scfi"), "observation_date"].max()
    expected_study = pd.date_range(STUDY_START, scfi_end, freq="ME")
    summary_rows = []
    gap_rows = []
    extreme_rows = []
    for variable, group in long.groupby("variable", sort=True):
        group = group.sort_values("observation_date").copy()
        group["value"] = pd.to_numeric(group["value"], errors="coerce")
        study = group[group["observation_date"].between(expected_study.min(), expected_study.max())]
        present = pd.DatetimeIndex(study.loc[study["value"].notna(), "observation_date"].unique())
        missing = expected_study.difference(present)
        for date in missing:
            gap_rows.append({"variable": variable, "missing_month": date.date().isoformat()})
        values = study["value"].replace([np.inf, -np.inf], np.nan).dropna()
        q1, q3 = values.quantile([0.25, 0.75]) if len(values) else (np.nan, np.nan)
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        extremes = study.loc[(study["value"] < lower) | (study["value"] > upper)] if np.isfinite(iqr) and iqr > 0 else study.iloc[0:0]
        for row in extremes.itertuples(index=False):
            extreme_rows.append(
                {
                    "variable": variable,
                    "observation_date": row.observation_date.date().isoformat(),
                    "value": row.value,
                    "rule": "outside_q1_minus_or_q3_plus_3_iqr",
                    "action": "review_against_real_event_do_not_auto_delete",
                }
            )
        coverage = len(present) / len(expected_study)
        if coverage >= 0.95:
            coverage_class = "full_or_near_full"
        elif coverage >= 0.70:
            coverage_class = "restricted_sample_required"
        else:
            coverage_class = "sparse_or_low_frequency"
        summary_rows.append(
            {
                "variable": variable,
                **SOURCE_CONTRACTS.get(variable, {}),
                "rows_all_monthly": len(group),
                "first_date": group["observation_date"].min().date().isoformat(),
                "last_date": group["observation_date"].max().date().isoformat(),
                "study_months_expected": len(expected_study),
                "study_months_present": len(present),
                "study_coverage_share": coverage,
                "coverage_class": coverage_class,
                "missing_study_months": len(missing),
                "maximum_consecutive_missing_months": maximum_consecutive_missing(expected_study, present),
                "duplicate_months_after_parsing": int(group.duplicated("observation_date").sum()),
                "nonfinite_values": int((~np.isfinite(pd.to_numeric(group["value"], errors="coerce"))).sum()),
                "negative_values": int((values < 0).sum()),
                "zero_values": int((values == 0).sum()),
                "extreme_values_3iqr": len(extremes),
                "minimum_study_value": values.min() if len(values) else np.nan,
                "maximum_study_value": values.max() if len(values) else np.nan,
                "point_1_disposition": "retain_for_next_eda_point",
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(gap_rows), pd.DataFrame(extreme_rows)


def write_findings(
    output: Path,
    series: pd.DataFrame,
    tables: pd.DataFrame,
    sidecars: pd.DataFrame,
    failures: dict[str, str],
    test_run: bool,
) -> None:
    lines = [
        "# Point 1 — Raw-data integrity findings",
        "",
        f"Run mode: `{'test' if test_run else 'deep'}`.",
        "",
        "No variable was discarded. All parsed series advance to Point 2 with the restrictions recorded below.",
        "",
        "## Series coverage",
        "",
        "| Variable | Native frequency | Study coverage | Missing months | Coverage class | First | Last |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in series.sort_values("variable").itertuples(index=False):
        lines.append(
            f"| `{row.variable}` | {row.native_frequency} | {row.study_coverage_share:.1%} | "
            f"{row.missing_study_months} | `{row.coverage_class}` | {row.first_date} | {row.last_date} |"
        )
    lines.extend(["", "## Integrity restrictions", ""])
    for row in series.loc[~series["coverage_class"].eq("full_or_near_full")].itertuples(index=False):
        lines.append(
            f"- `{row.variable}`: {row.study_coverage_share:.1%} coverage; `{row.coverage_class}`. Retained, but unrestricted full-period use is not supported."
        )
    unknown_release = series[series["release_schedule"].astype(str).str.contains("requires", case=False, na=False)]
    if not unknown_release.empty:
        lines.append(
            "- Release timing still requires primary-source verification for: "
            + ", ".join(f"`{value}`" for value in unknown_release["variable"])
            + "."
        )
    incomplete_sidecars = sidecars[sidecars["requires_manual_source_completion"]]
    for row in incomplete_sidecars.itertuples(index=False):
        lines.append(f"- Source sidecar requires correction/completion: `{row.relative_path}`.")
    deferred = tables.loc[~tables["row_count_complete"]]
    if not deferred.empty:
        lines.append(
            f"- {len(deferred)} large table(s) received structural/header checks only in test mode; deep row scans remain pending."
        )
    if failures:
        lines.extend(["", "## Parser failures", ""])
        lines.extend(f"- `{name}`: {message}" for name, message in sorted(failures.items()))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    module = load_prototype_module()

    files, tables = file_and_table_inventory(args.test_run, args.deep_scan_limit_mb)
    sidecars = source_sidecar_audit()
    long, failures = load_all_series(module)
    series, gaps, extremes = integrity_reports(long)

    files.to_csv(args.output_dir / "raw_file_inventory.csv", index=False)
    tables.to_csv(args.output_dir / "raw_table_structure_audit.csv", index=False)
    sidecars.to_csv(args.output_dir / "source_sidecar_audit.csv", index=False)
    series.to_csv(args.output_dir / "series_integrity_summary.csv", index=False)
    gaps.to_csv(args.output_dir / "missing_months.csv", index=False)
    extremes.to_csv(args.output_dir / "extreme_value_review.csv", index=False)
    write_findings(args.output_dir / "point_01_findings.md", series, tables, sidecars, failures, args.test_run)

    manifest = {
        "protocol_point": 1,
        "run_mode": "test" if args.test_run else "deep",
        "status": "complete_test_scope" if args.test_run else "complete_deep_scope",
        "raw_files": len(files),
        "raw_tables": len(tables),
        "series": len(series),
        "parser_failures": failures,
        "variables_retained": sorted(series["variable"].tolist()),
        "guardrail": "No variable excluded at Point 1.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

