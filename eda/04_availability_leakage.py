"""Point 4 of the EDA protocol: availability, release lag, and leakage audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "eda/outputs/point_04_availability_leakage"
EXTERNAL_LONG = ROOT / "data/02_interim/controls/external_controls_clean_long.csv"
MONTHLY_PANEL = ROOT / "data/06_model_ready/causality/msdi_scfi_monthly_panel.csv"
WEEKLY_PANEL = ROOT / "data/06_model_ready/prediction/msdi_scfi_weekly_panel.csv"
STUDY_START = pd.Timestamp("2016-02-29")
STUDY_END = pd.Timestamp("2026-02-28")

MAXIMUM_AGE_DAYS = {
    "daily": 7,
    "weekly": 14,
    "monthly": 75,
    "quarterly": 180,
    "annual": 455,
}

# A configured lag is not labelled source-verified. Unknown lags remain unknown.
SOURCE_CONTRACTS = [
    ("scfi", "SCFI", "weekly", None, "target", "needs_source_verification", "Raw workbook has observation dates; first-publication timestamps and revision policy are not encoded."),
    ("msdi", "MSDI and seven dimensions", "article/monthly", 0, "exposure", "retrospective_latency_unresolved", "Article publication date is used, but historical LLM scoring and ingestion latency were not observed in real time."),
    ("brent_price", "Brent price", "weekly", None, "control", "needs_source_verification", "Backward weekly join is implemented; the legacy file does not encode release timestamps."),
    ("cpb_world_trade_volume", "CPB world trade volume", "monthly", 45, "control", "configured_not_source_verified", "Cleaner applies a conservative 45-day lag; vintage/revision data are not stored."),
    ("gscpi", "GSCPI", "monthly", 5, "benchmark/control", "configured_not_source_verified", "Cleaner applies a five-day lag; construct overlap with freight costs is a separate issue."),
    ("unctad_china_core_lane_lsbci", "China core-lane LSBCI", "annual", 45, "control", "configured_not_source_verified", "Cleaner applies a 45-day lag to the empirical annual extract; no vintage data are stored."),
    ("imf_portwatch_global_container_calls", "IMF PortWatch container calls", "daily", 2, "control", "configured_not_source_verified", "Cleaner applies a two-day lag; coverage begins in 2019."),
    ("rwi_container_throughput", "RWI container throughput", "monthly", None, "candidate control", "contract_missing", "Loaded for EDA only; publication lag and vintage policy are not encoded."),
    ("china_merchandise_exports", "China merchandise exports", "monthly", None, "candidate control", "contract_missing", "Loaded for EDA only; WTO release lag and revision policy are not encoded."),
    ("cny_per_usd", "CNY per USD", "daily/monthly", None, "candidate control", "contract_missing", "Loaded for EDA only; a reproducible market-data availability contract is not encoded."),
    ("global_geopolitical_risk", "Global geopolitical risk", "monthly", None, "candidate shock", "contract_missing", "Loaded for EDA only; first-release timing and revisions are not encoded."),
    ("trade_policy_uncertainty", "Trade policy uncertainty", "monthly", None, "candidate shock", "contract_missing", "Loaded for EDA only; first-release timing and revisions are not encoded."),
    ("us_china_tension", "US-China tension", "monthly", None, "candidate shock", "contract_missing", "Loaded for EDA only; first-release timing and revisions are not encoded."),
    ("oil_price_uncertainty", "Oil-price uncertainty", "monthly", None, "candidate shock", "contract_missing", "Loaded for EDA only; first-release timing and revisions are not encoded."),
    ("infectious_disease_emv", "Infectious-disease EMV", "daily/monthly", None, "candidate shock", "contract_missing", "Loaded for EDA only; first-release timing and revisions are not encoded."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit release timing and leakage-safe alignment.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-run", action="store_true", help="Run deterministic audits only; no model fitting.")
    return parser.parse_args()


def contract_registry() -> pd.DataFrame:
    columns = [
        "variable",
        "label",
        "native_frequency",
        "configured_release_lag_days",
        "role",
        "contract_status",
        "evidence_and_limitation",
    ]
    return pd.DataFrame(SOURCE_CONTRACTS, columns=columns)


def load_external() -> pd.DataFrame:
    required = {
        "source_id",
        "variable",
        "observation_date",
        "availability_date",
        "value",
        "source_frequency",
        "raw_file",
    }
    raw = pd.read_csv(EXTERNAL_LONG, low_memory=False)
    missing = required - set(raw)
    if missing:
        raise ValueError(f"External-control file lacks fields: {sorted(missing)}")
    raw["observation_date"] = pd.to_datetime(raw["observation_date"], errors="coerce")
    raw["availability_date"] = pd.to_datetime(raw["availability_date"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    return raw


def release_lag_audit(raw: pd.DataFrame) -> pd.DataFrame:
    work = raw.copy()
    work["release_lag_days"] = (
        work["availability_date"] - work["observation_date"]
    ).dt.days
    rows = []
    for variable, group in work.groupby("variable", sort=True):
        lag = group["release_lag_days"]
        rows.append(
            {
                "variable": variable,
                "source_id": group["source_id"].iloc[0],
                "source_frequency": group["source_frequency"].iloc[0],
                "rows": len(group),
                "first_observation": group["observation_date"].min().date().isoformat(),
                "last_observation": group["observation_date"].max().date().isoformat(),
                "minimum_release_lag_days": lag.min(),
                "median_release_lag_days": lag.median(),
                "maximum_release_lag_days": lag.max(),
                "availability_before_observation_rows": int(
                    group["availability_date"].lt(group["observation_date"]).sum()
                ),
                "missing_availability_rows": int(group["availability_date"].isna().sum()),
                "duplicate_availability_dates": int(group["availability_date"].duplicated().sum()),
                "vintage_values_available": False,
            }
        )
    return pd.DataFrame(rows)


def monthly_asof_audit(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = pd.DataFrame(
        {"forecast_origin": pd.date_range(STUDY_START, STUDY_END, freq="ME")}
    )
    joined_frames = []
    summary = []
    for variable, group in raw.groupby("variable", sort=True):
        group = group.sort_values("observation_date")
        source = group[
            ["observation_date", "availability_date", "value", "source_frequency"]
        ].dropna(subset=["availability_date", "value"]).sort_values("availability_date")
        if source.empty:
            continue
        frequency = str(source["source_frequency"].iloc[0])
        maximum_age = MAXIMUM_AGE_DAYS.get(frequency)
        safe = pd.merge_asof(
            origins,
            source,
            left_on="forecast_origin",
            right_on="availability_date",
            direction="backward",
            allow_exact_matches=True,
        )
        safe["age_days"] = (
            safe["forecast_origin"] - safe["availability_date"]
        ).dt.days
        safe["stale"] = (
            safe["age_days"].gt(maximum_age) if maximum_age is not None else False
        )
        safe.loc[safe["stale"], "value"] = np.nan
        safe["variable"] = variable
        safe["future_match"] = safe["availability_date"].gt(safe["forecast_origin"])

        naive = pd.merge_asof(
            origins,
            group[["observation_date", "value"]].dropna().sort_values("observation_date"),
            left_on="forecast_origin",
            right_on="observation_date",
            direction="backward",
            allow_exact_matches=True,
        )
        safe["naive_value"] = naive["value"].to_numpy()
        safe["naive_observation_date"] = naive["observation_date"].to_numpy()
        safe["naive_uses_not_yet_released_observation"] = (
            safe["naive_observation_date"].notna()
            & safe["naive_observation_date"].gt(
                safe["availability_date"].fillna(pd.Timestamp.min)
            )
        )
        safe["value_difference"] = safe["naive_value"] - safe["value"]
        joined_frames.append(safe)
        study = safe.loc[safe["forecast_origin"].between(STUDY_START, STUDY_END)]
        available = study["value"].notna()
        summary.append(
            {
                "variable": variable,
                "source_frequency": frequency,
                "maximum_age_days": maximum_age,
                "study_origins": len(study),
                "safe_available_origins": int(available.sum()),
                "safe_coverage_share": available.mean(),
                "first_safe_origin": study.loc[available, "forecast_origin"].min(),
                "last_safe_origin": study.loc[available, "forecast_origin"].max(),
                "future_matches": int(study["future_match"].sum()),
                "naive_origins_using_not_yet_released_observation": int(
                    study["naive_uses_not_yet_released_observation"].sum()
                ),
                "naive_safe_value_mismatches": int(
                    (study["value_difference"].abs() > 1e-12).sum()
                ),
            }
        )
    return pd.DataFrame(summary), pd.concat(joined_frames, ignore_index=True)


def panel_audit() -> pd.DataFrame:
    rows = []
    weekly = pd.read_csv(WEEKLY_PANEL, low_memory=False)
    weekly["observation_date"] = pd.to_datetime(
        weekly["observation_date"], errors="coerce"
    )
    variables = (
        "cpb_world_trade_volume",
        "gscpi",
        "unctad_china_core_lane_lsbci",
        "imf_portwatch_global_container_calls",
    )
    for variable in variables:
        date_column = f"{variable}_availability_date"
        age_column = f"{variable}_age_days"
        available = pd.to_datetime(weekly[date_column], errors="coerce")
        future = available.notna() & available.gt(weekly["observation_date"])
        rows.append(
            {
                "panel": "weekly",
                "variable": variable,
                "rows": len(weekly),
                "nonmissing_values": int(
                    pd.to_numeric(weekly[variable], errors="coerce").notna().sum()
                ),
                "future_availability_matches": int(future.sum()),
                "negative_age_rows": int(
                    pd.to_numeric(weekly[age_column], errors="coerce").lt(0).sum()
                ),
                "availability_date_retained": True,
                "assessment": "pass" if not future.any() else "fail",
            }
        )
    monthly = pd.read_csv(MONTHLY_PANEL, low_memory=False)
    rows.append(
        {
            "panel": "monthly",
            "variable": "all_external_controls",
            "rows": len(monthly),
            "nonmissing_values": np.nan,
            "future_availability_matches": np.nan,
            "negative_age_rows": np.nan,
            "availability_date_retained": False,
            "assessment": "provenance_not_row_level",
        }
    )
    return pd.DataFrame(rows)


def write_findings(
    path: Path,
    contracts: pd.DataFrame,
    lags: pd.DataFrame,
    coverage: pd.DataFrame,
    panels: pd.DataFrame,
) -> None:
    missing = contracts.loc[
        contracts["contract_status"].eq("contract_missing"), "label"
    ].tolist()
    configured = contracts.loc[
        contracts["contract_status"].eq("configured_not_source_verified"), "label"
    ].tolist()
    lines = [
        "# Point 4 - Availability and leakage findings",
        "",
        "## What passes",
        "",
        f"- Encoded availability-before-observation rows: {int(lags['availability_before_observation_rows'].sum())}; zero is required.",
        f"- Weekly panel future matches: {int(panels['future_availability_matches'].fillna(0).sum())}; zero is required.",
        "- The existing external-control join is backward-only and its monthly model columns are additionally lagged one month.",
        "",
        "## What remains unresolved",
        "",
        "- Configured conservative lags are not the same as verified historical first-release timestamps.",
        f"- Configured but not source-verified: {', '.join(configured)}.",
        f"- No encoded release contract yet: {', '.join(missing)}.",
        "- CPB, GSCPI and similar macro series may be revised; the repository stores current-vintage downloads, not real-time vintages.",
        "- Monthly panels do not retain row-level external-control availability dates, so provenance must be traced to the weekly panel and cleaner manifest.",
        "- MSDI is indexed by article publication date but was produced retrospectively. A real-time forecast interpretation needs a stated ingestion/scoring latency or a latency sensitivity analysis.",
        "",
        "## Decision",
        "",
        "All variables remain in EDA. Variables without a verified release contract must not enter leakage-safe forecasting or causal specifications until the contract is implemented. This is an eligibility rule, not selection by statistical performance.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contracts = contract_registry()
    external = load_external()
    lags = release_lag_audit(external)
    coverage, asof_rows = monthly_asof_audit(external)
    panels = panel_audit()

    contracts.to_csv(args.output_dir / "source_contract_registry.csv", index=False)
    lags.to_csv(args.output_dir / "encoded_release_lag_audit.csv", index=False)
    coverage.to_csv(args.output_dir / "monthly_asof_coverage.csv", index=False)
    asof_rows.to_csv(args.output_dir / "monthly_asof_matches.csv", index=False)
    panels.to_csv(args.output_dir / "panel_leakage_checks.csv", index=False)
    write_findings(
        args.output_dir / "point_04_findings.md", contracts, lags, coverage, panels
    )

    manifest = {
        "protocol_point": 4,
        "run_mode": "test" if args.test_run else "full",
        "status": "complete_with_release_contract_gaps",
        "source_contracts": len(contracts),
        "contracts_missing": int(
            contracts["contract_status"].eq("contract_missing").sum()
        ),
        "configured_not_source_verified": int(
            contracts["contract_status"].eq("configured_not_source_verified").sum()
        ),
        "encoded_external_variables": int(external["variable"].nunique()),
        "encoded_availability_before_observation_rows": int(
            lags["availability_before_observation_rows"].sum()
        ),
        "weekly_panel_future_matches": int(
            panels["future_availability_matches"].fillna(0).sum()
        ),
        "guardrail": "Unknown release timing is not imputed; unresolved variables are EDA-only until a source contract is encoded.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
