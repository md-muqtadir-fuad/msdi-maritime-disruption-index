"""Audit raw external controls before cleaning or pipeline integration.

The script is intentionally read-only with respect to ``data/01_raw``. It
creates comparable monthly series for exploratory coverage and distribution
checks, while leaving release-lag handling to the production cleaner.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Times", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/01_raw"
DEFAULT_OUTPUT = ROOT / "eda/outputs"
STUDY_START = pd.Timestamp("2016-02-01")

FILES = {
    "scfi": RAW / "freight/scfi/freight_index_data_legacy.xlsx",
    "rwi": RAW / "controls/container_throughput_index/containerumschlag-Index_260629.xlsx",
    "cny": RAW / "controls/exchange_rate/DEXCHUS.csv",
    "gpr": RAW / "controls/gpr_risk/data_gpr_export.xls",
    "tpu": RAW / "controls/TPU/tpu_web_latest.xlsx",
    "wto": RAW / "controls/WTO/WtoData_20260814111914.csv",
    "uct": RAW / "controls/uct_index/UCT.csv",
    "opu": RAW / "controls/OPU_Index/OPU_index_monthly.csv",
    "emv": RAW / "controls/EMV_data/All_Infectious_EMV_Data.csv",
    "brent": RAW / "controls/brent/brent_weekly_legacy.csv",
}
EXISTING_CLEAN = ROOT / "data/02_interim/controls/external_controls_clean_long.csv"

LABELS = {
    "scfi": "SCFI",
    "rwi_container_throughput": "Container throughput",
    "china_merchandise_exports": "China merchandise exports",
    "cny_per_usd": "CNY per USD",
    "global_geopolitical_risk": "Geopolitical risk",
    "trade_policy_uncertainty": "Trade policy uncertainty",
    "us_china_tension": "US–China tension",
    "oil_price_uncertainty": "Oil-price uncertainty",
    "infectious_disease_emv": "Infectious-disease EMV",
    "brent_price": "Brent price",
    "cpb_world_trade_volume": "World trade volume",
    "gscpi": "Global supply-chain pressure",
    "unctad_china_core_lane_lsbci": "China route connectivity",
    "imf_portwatch_global_container_calls": "Global container vessel calls",
}

ROLES = {
    "rwi_container_throughput": "candidate_core_shipping_demand",
    "china_merchandise_exports": "candidate_core_china_demand",
    "cny_per_usd": "candidate_core_currency_cost",
    "global_geopolitical_risk": "candidate_sensitivity_external_shock",
    "trade_policy_uncertainty": "candidate_sensitivity_policy_shock",
    "us_china_tension": "candidate_sensitivity_bilateral_shock",
    "oil_price_uncertainty": "candidate_sensitivity_energy_uncertainty",
    "infectious_disease_emv": "candidate_sensitivity_pandemic_attention",
    "brent_price": "existing_core_energy_cost",
    "cpb_world_trade_volume": "existing_core_trade_demand",
    "gscpi": "existing_benchmark_not_core_control",
    "unctad_china_core_lane_lsbci": "existing_connectivity_control",
    "imf_portwatch_global_container_calls": "existing_2019_plus_robustness",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA of raw candidate SCFI controls.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--study-end", type=str, default=None)
    parser.add_argument("--test-run", action="store_true", help="Run the complete data audit without any model fitting or tuning.")
    parser.add_argument("--skip-wto", action="store_true", help="Skip the large WTO extract during a quick diagnostic run.")
    return parser.parse_args()


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def month_end(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        dates = pd.to_datetime(values, errors="coerce")
    else:
        dates = pd.to_datetime(values, errors="coerce", format="mixed")
    return dates.dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()


def finish(frame: pd.DataFrame, variable: str) -> pd.DataFrame:
    result = frame[["observation_date", "value"]].copy()
    result["observation_date"] = month_end(result["observation_date"])
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result = result.dropna().groupby("observation_date", as_index=False)["value"].mean()
    result["variable"] = variable
    return result[["variable", "observation_date", "value"]]


def load_scfi(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(require(path))
    return finish(
        pd.DataFrame({"observation_date": raw.iloc[:, 0], "value": raw.iloc[:, 1]}),
        "scfi",
    )


def load_rwi(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(require(path), sheet_name="Output", header=None)
    # Column 2 is the published total index adjusted for seasonality and
    # working days. Trend-cycle is deliberately not used as an input series.
    return finish(
        pd.DataFrame({"observation_date": raw.iloc[:, 0], "value": raw.iloc[:, 2]}),
        "rwi_container_throughput",
    )


def load_cny(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(require(path), low_memory=False)
    return finish(
        pd.DataFrame({"observation_date": raw["observation_date"], "value": raw["DEXCHUS"]}),
        "cny_per_usd",
    )


def load_gpr(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(require(path), sheet_name="Sheet1")
    return finish(
        pd.DataFrame({"observation_date": raw["month"], "value": raw["GPR"]}),
        "global_geopolitical_risk",
    )


def load_tpu(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(require(path), sheet_name="TPU_MONTHLY")
    return finish(
        pd.DataFrame({"observation_date": raw["DATE"], "value": raw["TPU"]}),
        "trade_policy_uncertainty",
    )


def load_wto_exports(path: Path) -> pd.DataFrame:
    keep = [
        "Indicator Code",
        "Reporting Economy ISO3A Code",
        "Partner Economy",
        "Product/Sector",
        "Period Code",
        "Year",
        "Value",
    ]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        require(path),
        usecols=keep,
        chunksize=200_000,
        encoding="cp1252",
        low_memory=False,
    ):
        selected = chunk.loc[
            chunk["Indicator Code"].eq("ITS_MTV_MX")
            & chunk["Reporting Economy ISO3A Code"].eq("CHN")
            & chunk["Partner Economy"].eq("World")
            & chunk["Product/Sector"].eq("Total merchandise")
        ].copy()
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        raise ValueError("The WTO extract has no China-to-world monthly merchandise export series.")
    raw = pd.concat(pieces, ignore_index=True)
    years = pd.to_numeric(raw["Year"], errors="coerce").astype("Int64").astype(str)
    months = raw["Period Code"].astype(str).str.extract(r"M(\d{2})", expand=False)
    dates = pd.to_datetime(years + "-" + months + "-01", errors="coerce")
    return finish(
        pd.DataFrame({"observation_date": dates, "value": raw["Value"]}),
        "china_merchandise_exports",
    )


def load_uct(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(require(path), low_memory=False)
    dates = pd.to_datetime(raw["date"].astype(str).str.replace("m", "-", regex=False) + "-01", errors="coerce")
    return finish(pd.DataFrame({"observation_date": dates, "value": raw["UCT"]}), "us_china_tension")


def load_opu(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(require(path), low_memory=False)
    return finish(
        pd.DataFrame({"observation_date": raw["date"], "value": raw["OPU_index"]}),
        "oil_price_uncertainty",
    )


def load_emv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(require(path), low_memory=False)
    dates = pd.to_datetime(dict(year=raw["year"], month=raw["month"], day=raw["day"]), errors="coerce")
    return finish(
        pd.DataFrame({"observation_date": dates, "value": raw["daily_infect_emv_index"]}),
        "infectious_disease_emv",
    )


def load_brent(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(require(path), low_memory=False)
    return finish(
        pd.DataFrame({"observation_date": raw["observation_date"], "value": raw["DCOILBRENTEU"]}),
        "brent_price",
    )


def load_existing_controls(path: Path) -> list[pd.DataFrame]:
    if not path.is_file():
        return []
    raw = pd.read_csv(path, low_memory=False)
    selected = raw.loc[
        raw["variable"].isin(
            {
                "cpb_world_trade_volume",
                "gscpi",
                "unctad_china_core_lane_lsbci",
                "imf_portwatch_global_container_calls",
            }
        ),
        ["variable", "observation_date", "value"],
    ].copy()
    selected["observation_date"] = month_end(selected["observation_date"])
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    selected = selected.dropna()
    return [
        group.groupby("observation_date", as_index=False)["value"].mean().assign(variable=variable)[
            ["variable", "observation_date", "value"]
        ]
        for variable, group in selected.groupby("variable", sort=True)
    ]


def file_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": path.relative_to(ROOT).as_posix(),
                    "extension": path.suffix.lower(),
                    "size_mb": path.stat().st_size / 1_000_000,
                    "modified_utc": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat(),
                }
            )
    return pd.DataFrame(rows)


def series_summary(long: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    expected = pd.date_range(start.to_period("M").to_timestamp(how="end").normalize(), end, freq="ME")
    rows = []
    for variable, group in long.groupby("variable", sort=True):
        group = group.sort_values("observation_date")
        study = group[group["observation_date"].between(expected.min(), expected.max())]
        present = pd.DatetimeIndex(study["observation_date"].unique())
        missing = expected.difference(present)
        values = study["value"]
        rows.append(
            {
                "variable": variable,
                "label": LABELS.get(variable, variable),
                "role": ROLES.get(variable, "outcome" if variable == "scfi" else "unclassified"),
                "rows_all": len(group),
                "first_date": group["observation_date"].min().date().isoformat(),
                "last_date": group["observation_date"].max().date().isoformat(),
                "study_months_expected": len(expected),
                "study_months_present": len(present),
                "study_coverage_share": len(present) / len(expected),
                "missing_study_months": len(missing),
                "first_missing_study_month": missing.min().date().isoformat() if len(missing) else "",
                "duplicate_months": int(group.duplicated("observation_date").sum()),
                "minimum": values.min(),
                "maximum": values.max(),
                "mean": values.mean(),
                "standard_deviation": values.std(),
            }
        )
    return pd.DataFrame(rows)


def transformed_changes(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = pd.DataFrame(index=panel.index)
    records = []
    for column in panel.columns:
        values = panel[column]
        usable = values.dropna()
        if not usable.empty and (usable > 0).all():
            result[column] = np.log(values).diff()
            transform = "first_difference_of_log"
        else:
            result[column] = values.diff()
            transform = "first_difference"
        records.append({"relation": "first_changes", "variable": column, "transform": transform})
    return result, pd.DataFrame(records)


def transformed_year_over_year(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = pd.DataFrame(index=panel.index)
    records = []
    for column in panel.columns:
        values = panel[column]
        usable = values.dropna()
        if not usable.empty and (usable > 0).all():
            result[column] = np.log(values).diff(12)
            transform = "12_month_difference_of_log"
        else:
            result[column] = values.diff(12)
            transform = "12_month_difference"
        records.append({"relation": "year_over_year_changes", "variable": column, "transform": transform})
    return result, pd.DataFrame(records)


def adjust_fdr(frame: pd.DataFrame, p_columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in p_columns:
        q_column = column.replace("_p_value", "_fdr_q_value")
        result[q_column] = np.nan
        mask = result[column].notna()
        if mask.any():
            result.loc[mask, q_column] = multipletests(result.loc[mask, column], method="fdr_bh")[1]
    return result


def association_table(panel: pd.DataFrame, changes: pd.DataFrame, year_over_year: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for relation, frame in (
        ("levels_descriptive_nonstationary", panel),
        ("first_changes", changes),
        ("year_over_year_changes", year_over_year),
    ):
        for variable in frame.columns:
            if variable == "scfi":
                continue
            pair = pd.concat(
                [frame["scfi"].rename("scfi"), frame[variable].rename("predictor")], axis=1
            ).dropna()
            if len(pair) >= 3 and pair.nunique().min() > 1:
                pearson = pearsonr(pair["scfi"], pair["predictor"])
                spearman = spearmanr(pair["scfi"], pair["predictor"])
                pearson_stat, pearson_p = pearson.statistic, pearson.pvalue
                spearman_stat, spearman_p = spearman.statistic, spearman.pvalue
            else:
                pearson_stat = pearson_p = spearman_stat = spearman_p = np.nan
            rows.append(
                {
                    "relation": relation,
                    "variable": variable,
                    "label": LABELS.get(variable, variable),
                    "paired_months": len(pair),
                    "pearson_correlation": pearson_stat,
                    "pearson_p_value": pearson_p,
                    "spearman_correlation": spearman_stat,
                    "spearman_p_value": spearman_p,
                    "guardrail": "descriptive_association_not_causality_or_selection_rule",
                }
            )
    return adjust_fdr(pd.DataFrame(rows), ["pearson_p_value", "spearman_p_value"])


def lagged_correlations(changes: pd.DataFrame, relation: str, maximum_lag: int = 6) -> pd.DataFrame:
    rows = []
    target = changes["scfi"]
    for variable in changes.columns:
        if variable == "scfi":
            continue
        for lag in range(-maximum_lag, maximum_lag + 1):
            pair = pd.concat([target.rename("target"), changes[variable].shift(lag).rename("predictor")], axis=1).dropna()
            if len(pair) >= 3 and pair.nunique().min() > 1:
                pearson = pearsonr(pair["target"], pair["predictor"])
                spearman = spearmanr(pair["target"], pair["predictor"])
                pearson_stat, pearson_p = pearson.statistic, pearson.pvalue
                spearman_stat, spearman_p = spearman.statistic, spearman.pvalue
            else:
                pearson_stat = pearson_p = spearman_stat = spearman_p = np.nan
            rows.append(
                {
                    "relation": relation,
                    "variable": variable,
                    "label": LABELS.get(variable, variable),
                    "predictor_lag_months": lag,
                    "lag_definition": "positive means predictor precedes SCFI; negative means predictor follows SCFI",
                    "paired_months": len(pair),
                    "pearson_correlation": pearson_stat,
                    "pearson_p_value": pearson_p,
                    "spearman_correlation": spearman_stat,
                    "spearman_p_value": spearman_p,
                    "interpretation": "descriptive_only_not_for_variable_or_lag_selection",
                }
            )
    return adjust_fdr(pd.DataFrame(rows), ["pearson_p_value", "spearman_p_value"])


def plot_coverage(summary: pd.DataFrame, output: Path) -> None:
    ordered = summary.sort_values("first_date").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, max(5, len(ordered) * 0.42)))
    for index, row in ordered.iterrows():
        ax.plot(pd.to_datetime([row["first_date"], row["last_date"]]), [index, index], linewidth=5, solid_capstyle="butt")
    ax.axvspan(STUDY_START, pd.Timestamp("2026-02-28"), color="grey", alpha=0.08)
    ax.set_yticks(range(len(ordered)), ordered["label"])
    ax.set_xlabel("Observation date")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_series(panel: pd.DataFrame, output: Path) -> None:
    candidates = [column for column in panel.columns if column != "scfi"]
    ncols = 2
    nrows = math.ceil(len(candidates) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4, nrows * 2.7)), squeeze=False, sharex=True)
    for ax, column in zip(axes.flat, candidates):
        ax.plot(panel.index, panel[column], linewidth=1.15)
        ax.set_title(LABELS.get(column, column), loc="left", fontsize=10)
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes.flat[len(candidates) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_change_relationships(changes: pd.DataFrame, output: Path) -> None:
    candidates = [column for column in changes.columns if column != "scfi"]
    ncols = 3
    nrows = math.ceil(len(candidates) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, max(4, nrows * 3)), squeeze=False)
    for ax, column in zip(axes.flat, candidates):
        pair = pd.concat([changes[column].rename("x"), changes["scfi"].rename("y")], axis=1).dropna()
        ax.scatter(pair["x"], pair["y"], s=14, alpha=0.65)
        if len(pair) >= 3 and pair["x"].nunique() > 1:
            slope, intercept = np.polyfit(pair["x"], pair["y"], 1)
            grid = np.linspace(pair["x"].min(), pair["x"].max(), 100)
            ax.plot(grid, intercept + slope * grid, linewidth=1, color="#b44b3a")
        ax.set_title(LABELS.get(column, column), loc="left", fontsize=10)
        ax.set_xlabel("Predictor change")
        ax.set_ylabel("SCFI log change")
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(candidates) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_findings(
    summary: pd.DataFrame,
    associations: pd.DataFrame,
    lagged: pd.DataFrame,
    output: Path,
    failures: dict[str, str],
) -> None:
    study = summary[summary["variable"] != "scfi"].sort_values(["role", "variable"])
    lines = [
        "# Raw controls EDA findings",
        "",
        "This report evaluates data availability and integrity before cleaning. It does not select variables from SCFI correlation or significance.",
        "Every successfully loaded variable is retained in the association audit; this EDA accepts or rejects none of them.",
        "",
        "## Coverage in the SCFI study window",
        "",
        "| Variable | Intended role | Months present | Coverage | First | Last |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in study.itertuples(index=False):
        lines.append(
            f"| {row.label} | `{row.role}` | {row.study_months_present}/{row.study_months_expected} | "
            f"{row.study_coverage_share:.1%} | {row.first_date} | {row.last_date} |"
        )
    lines.extend(
        [
            "",
            "## Contemporaneous SCFI associations",
            "",
            "The table uses transformed monthly changes and reports FDR-adjusted q-values across the full contemporaneous comparison family.",
            "",
            "| Variable | Months | Pearson r | Pearson q | Spearman rho | Spearman q |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    contemporaneous = associations.loc[associations["relation"].eq("first_changes")].sort_values("variable")
    for row in contemporaneous.itertuples(index=False):
        lines.append(
            f"| {row.label} | {row.paired_months} | {row.pearson_correlation:.3f} | "
            f"{row.pearson_fdr_q_value:.3g} | {row.spearman_correlation:.3f} | {row.spearman_fdr_q_value:.3g} |"
        )
    lines.extend(
        [
            "",
            "## Strongest observed leading association for each variable",
            "",
            "Positive lag means the predictor precedes SCFI. This is an exhaustive descriptive screen, not a lag-selection rule.",
            "",
            "| Variable | Lag (months) | Months | Pearson r | Pearson q | Spearman rho | Spearman q |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    leading = lagged.loc[
        lagged["relation"].eq("first_changes") & lagged["predictor_lag_months"].between(1, 6)
    ].copy()
    leading["maximum_absolute_correlation"] = leading[["pearson_correlation", "spearman_correlation"]].abs().max(axis=1)
    leading = leading.sort_values("maximum_absolute_correlation", ascending=False).drop_duplicates("variable").sort_values("variable")
    for row in leading.itertuples(index=False):
        lines.append(
            f"| {row.label} | {row.predictor_lag_months} | {row.paired_months} | "
            f"{row.pearson_correlation:.3f} | {row.pearson_fdr_q_value:.3g} | "
            f"{row.spearman_correlation:.3f} | {row.spearman_fdr_q_value:.3g} |"
        )
    lines.extend(
        [
            "",
            "## Processing rule",
            "",
            "Only variables with a defensible economic role, adequate temporal coverage, documented units, and an explicit release-lag rule are eligible for `data/02_interim/controls/`. News-derived uncertainty measures remain sensitivity controls because their measurement mechanism can overlap with the MSDI news signal.",
        ]
    )
    if failures:
        lines.extend(["", "## Loader failures", ""])
        lines.extend(f"- `{name}`: {message}" for name, message in sorted(failures.items()))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaders: list[tuple[str, Callable[[Path], pd.DataFrame]]] = [
        ("scfi", load_scfi),
        ("rwi", load_rwi),
        ("cny", load_cny),
        ("gpr", load_gpr),
        ("tpu", load_tpu),
        ("uct", load_uct),
        ("opu", load_opu),
        ("emv", load_emv),
        ("brent", load_brent),
    ]
    if not args.skip_wto:
        loaders.append(("wto", load_wto_exports))

    pieces: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    for name, loader in loaders:
        try:
            piece = loader(FILES[name])
            pieces.append(piece)
            print(f"Loaded {name}: {len(piece)} monthly observations")
        except Exception as exc:  # preserve the remaining audit when one optional source is malformed
            failures[name] = f"{type(exc).__name__}: {exc}"
            print(f"FAILED {name}: {failures[name]}")
    pieces.extend(load_existing_controls(EXISTING_CLEAN))

    long = pd.concat(pieces, ignore_index=True).sort_values(["variable", "observation_date"])
    if "scfi" not in set(long["variable"]):
        raise ValueError("SCFI could not be loaded; study-window EDA is impossible.")
    scfi_end = long.loc[long["variable"].eq("scfi"), "observation_date"].max()
    study_end = pd.Timestamp(args.study_end) if args.study_end else scfi_end
    study_end = study_end.to_period("M").to_timestamp(how="end").normalize()

    inventory = file_inventory()
    summary = series_summary(long, STUDY_START, study_end)
    panel = long.pivot(index="observation_date", columns="variable", values="value").sort_index()
    panel = panel.loc[panel.index.to_series().between(STUDY_START, study_end)]
    changes, monthly_transformations = transformed_changes(panel)
    year_over_year, yearly_transformations = transformed_year_over_year(panel)
    transformations = pd.concat([monthly_transformations, yearly_transformations], ignore_index=True)
    associations = association_table(panel, changes, year_over_year)
    correlations = pd.concat(
        [
            lagged_correlations(changes, "first_changes"),
            lagged_correlations(year_over_year, "year_over_year_changes"),
        ],
        ignore_index=True,
    )
    missingness = pd.DataFrame(
        {
            "variable": panel.columns,
            "label": [LABELS.get(column, column) for column in panel.columns],
            "missing_months": panel.isna().sum().to_numpy(),
            "missing_share": panel.isna().mean().to_numpy(),
        }
    )

    inventory.to_csv(args.output_dir / "raw_file_inventory.csv", index=False)
    summary.to_csv(args.output_dir / "candidate_series_summary.csv", index=False)
    long.to_csv(args.output_dir / "candidate_series_monthly_long.csv", index=False)
    panel.rename_axis("month_end").to_csv(args.output_dir / "candidate_series_monthly_panel.csv")
    transformations.to_csv(args.output_dir / "change_transformations.csv", index=False)
    associations.to_csv(args.output_dir / "scfi_contemporaneous_associations.csv", index=False)
    correlations.to_csv(args.output_dir / "lagged_descriptive_correlations.csv", index=False)
    missingness.to_csv(args.output_dir / "study_window_missingness.csv", index=False)
    plot_coverage(summary, args.output_dir / "series_coverage.png")
    plot_candidate_series(panel, args.output_dir / "candidate_series.png")
    plot_change_relationships(changes, args.output_dir / "scfi_change_relationships.png")
    write_findings(summary, associations, correlations, args.output_dir / "eda_findings.md", failures)

    manifest = {
        "status": "complete" if not failures else "complete_with_optional_loader_failures",
        "run_mode": "test" if args.test_run else "eda",
        "study_start": STUDY_START.date().isoformat(),
        "study_end": study_end.date().isoformat(),
        "variables": sorted(long["variable"].unique().tolist()),
        "failures": failures,
        "outputs": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
        "guardrail": "Descriptive EDA only; release-aware production cleaning and backward joins remain mandatory.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"EDA complete: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
