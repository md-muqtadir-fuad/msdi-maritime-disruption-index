"""Point 3 of the EDA protocol: monthly temporal properties and event windows."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "eda/outputs/point_03_temporal_properties"
MONTHLY_PANEL = ROOT / "data/06_model_ready/causality/msdi_scfi_monthly_panel.csv"
MODEL_MONTHLY = ROOT / "data/05_indices/monthly/msdi_monthly_model_specific.csv"
STUDY_START = pd.Timestamp("2016-02-29")
STUDY_END = pd.Timestamp("2026-02-28")

EVENTS = {
    "COVID-19 onset": pd.Timestamp("2020-03-31"),
    "Ever Given blockage": pd.Timestamp("2021-03-31"),
    "Russia-Ukraine invasion": pd.Timestamp("2022-02-28"),
    "Red Sea disruption": pd.Timestamp("2023-12-31"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze monthly temporal properties for all EDA series.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-run", action="store_true", help="Use the full small monthly panel but no resampling or model tuning.")
    return parser.parse_args()


def load_prototype_module():
    path = ROOT / "eda/00_raw_controls_eda.py"
    spec = importlib.util.spec_from_file_location("raw_controls_eda", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def friendly(name: str) -> str:
    replacements = {
        "scfi": "SCFI",
        "msdi": "MSDI",
        "gscpi": "GSCPI",
        "cpb": "CPB",
        "cny": "CNY",
        "usd": "USD",
        "glm": "GLM",
        "qwen": "Qwen",
        "gemma": "Gemma",
        "covid19": "COVID-19",
    }
    words = name.replace("__", " ").replace("_", " ").split()
    return " ".join(replacements.get(word.lower(), word) for word in words).capitalize().replace("Scfi", "SCFI").replace("Msdi", "MSDI").replace("Gscpi", "GSCPI").replace("Cpb", "CPB").replace("Cny", "CNY").replace("Usd", "USD").replace("Glm", "GLM")


def load_controls(module) -> pd.DataFrame:
    loaders = [
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
    pieces = [loader(module.FILES[name]) for name, loader in loaders]
    pieces.extend(module.load_existing_controls(module.EXISTING_CLEAN))
    long = pd.concat(pieces, ignore_index=True)
    return long.pivot(index="observation_date", columns="variable", values="value").sort_index()


def load_msdi_panel() -> pd.DataFrame:
    raw = pd.read_csv(MONTHLY_PANEL, low_memory=False)
    raw["month_ending"] = pd.to_datetime(raw["month_ending"], errors="coerce")
    columns = {
        "scfi_monthly_mean": "scfi",
        "peak_score_media_corrected_mean": "msdi_peak",
        "breadth_score_media_corrected_mean": "msdi_breadth",
        "article_count_sum": "news_article_count",
        "event_count_sum": "news_event_count",
        "source_count_mean": "news_source_count",
        "effective_article_count_sum": "effective_article_count",
        "peak_model_disagreement_mean_mean": "peak_model_disagreement",
        "dimension_disagreement_mean_mean": "dimension_model_disagreement",
    }
    for dimension in (
        "port_disruption",
        "global_container_movement_disruption",
        "freight_cost_disruption",
        "supply_disruption",
        "production_disruption",
        "covid19_related_disruption",
        "demand_disruption",
    ):
        source = f"{dimension}_media_corrected_mean"
        if source in raw.columns:
            columns[source] = f"msdi_{dimension}"
    selected = raw[["month_ending", *[column for column in columns if column in raw.columns]]].rename(columns=columns)
    return selected.set_index("month_ending").sort_index()


def load_model_paths() -> pd.DataFrame:
    raw = pd.read_csv(MODEL_MONTHLY, low_memory=False)
    raw["month_end"] = pd.to_datetime(raw["month_end"], errors="coerce")
    pieces = []
    for path in ("peak_score_media_corrected", "breadth_score_media_corrected"):
        pivot = raw.pivot(index="month_end", columns="model_alias", values=path)
        prefix = "model_peak" if path.startswith("peak") else "model_breadth"
        pivot.columns = [f"{prefix}_{column}" for column in pivot.columns]
        pieces.append(pivot)
    return pd.concat(pieces, axis=1).sort_index()


def build_panel() -> pd.DataFrame:
    module = load_prototype_module()
    controls = load_controls(module)
    msdi = load_msdi_panel()
    models = load_model_paths()
    panel = pd.concat([msdi, controls, models], axis=1)
    panel = panel.loc[:, ~panel.columns.duplicated(keep="first")]
    return panel.loc[panel.index.to_series().between(STUDY_START, STUDY_END)].sort_index()


def transformations(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    changes = pd.DataFrame(index=panel.index)
    yoy = pd.DataFrame(index=panel.index)
    records = []
    for column in panel.columns:
        values = pd.to_numeric(panel[column], errors="coerce")
        usable = values.dropna()
        if len(usable) and (usable > 0).all():
            changes[column] = np.log(values).diff()
            yoy[column] = np.log(values).diff(12)
            change_name, yoy_name = "first log difference", "12-month log difference"
        elif len(usable) and (usable >= 0).all():
            changes[column] = np.log1p(values).diff()
            yoy[column] = np.log1p(values).diff(12)
            change_name, yoy_name = "first log1p difference", "12-month log1p difference"
        else:
            changes[column] = values.diff()
            yoy[column] = values.diff(12)
            change_name, yoy_name = "first difference", "12-month difference"
        records.append({"variable": column, "change_transform": change_name, "year_over_year_transform": yoy_name})
    return changes, yoy, pd.DataFrame(records)


def distribution_summary(panel: pd.DataFrame, changes: pd.DataFrame, yoy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for representation, frame in (("level", panel), ("monthly_change", changes), ("year_over_year_change", yoy)):
        for column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "representation": representation,
                    "variable": column,
                    "label": friendly(column),
                    "observations": len(values),
                    "missing_share": 1 - len(values) / len(frame),
                    "mean": values.mean(),
                    "standard_deviation": values.std(),
                    "minimum": values.min(),
                    "p01": values.quantile(0.01),
                    "p25": values.quantile(0.25),
                    "median": values.median(),
                    "p75": values.quantile(0.75),
                    "p99": values.quantile(0.99),
                    "maximum": values.max(),
                    "zero_share": values.eq(0).mean() if len(values) else np.nan,
                    "skewness": skew(values, bias=False) if len(values) >= 3 else np.nan,
                    "excess_kurtosis": kurtosis(values, fisher=True, bias=False) if len(values) >= 4 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def one_stationarity(values: pd.Series) -> dict[str, object]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    result: dict[str, object] = {"observations": len(clean), "adf_statistic": np.nan, "adf_p_value": np.nan, "kpss_statistic": np.nan, "kpss_p_value": np.nan, "decision": "insufficient_or_constant"}
    if len(clean) < 24 or clean.nunique() < 3:
        return result
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            adf = adfuller(clean, autolag="AIC")
            result.update({"adf_statistic": adf[0], "adf_p_value": adf[1]})
        except Exception:
            pass
        try:
            kp = kpss(clean, regression="c", nlags="auto")
            result.update({"kpss_statistic": kp[0], "kpss_p_value": kp[1]})
        except Exception:
            pass
    adf_p, kpss_p = result["adf_p_value"], result["kpss_p_value"]
    if pd.notna(adf_p) and pd.notna(kpss_p):
        if adf_p < 0.05 and kpss_p >= 0.05:
            result["decision"] = "stationary_by_both"
        elif adf_p >= 0.05 and kpss_p < 0.05:
            result["decision"] = "nonstationary_by_both"
        else:
            result["decision"] = "tests_disagree"
    return result


def stationarity_table(panel: pd.DataFrame, changes: pd.DataFrame, yoy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for representation, frame in (("level", panel), ("monthly_change", changes), ("year_over_year_change", yoy)):
        for column in frame.columns:
            rows.append({"representation": representation, "variable": column, "label": friendly(column), **one_stationarity(frame[column])})
    return pd.DataFrame(rows)


def dependence_table(changes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in changes.columns:
        values = pd.to_numeric(changes[column], errors="coerce").dropna()
        if len(values) < 24 or values.nunique() < 3:
            rows.append({"variable": column, "label": friendly(column), "observations": len(values), "lag": np.nan, "acf": np.nan, "pacf": np.nan})
            continue
        maximum = min(12, max(1, len(values) // 2 - 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acf_values = acf(values, nlags=maximum, fft=False, missing="drop")
            pacf_values = pacf(values, nlags=maximum, method="ywm")
        rows.extend(
            {"variable": column, "label": friendly(column), "observations": len(values), "lag": lag, "acf": acf_values[lag], "pacf": pacf_values[lag]}
            for lag in range(1, maximum + 1)
        )
    return pd.DataFrame(rows)


def seasonal_table(changes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in changes.columns:
        frame = pd.DataFrame({"value": changes[column], "calendar_month": changes.index.month}).dropna()
        for month, group in frame.groupby("calendar_month"):
            rows.append({"variable": column, "label": friendly(column), "calendar_month": month, "observations": len(group), "mean_change": group["value"].mean(), "median_change": group["value"].median()})
    return pd.DataFrame(rows)


def outlier_table(changes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in changes.columns:
        values = pd.to_numeric(changes[column], errors="coerce")
        if not values.notna().any():
            continue
        median = values.median(skipna=True)
        mad = (values - median).abs().median(skipna=True)
        if not pd.notna(mad) or mad <= 0:
            continue
        robust_z = 0.67448975 * (values - median) / mad
        for date, value in values.loc[robust_z.abs().ge(4)].items():
            nearby = [name for name, event_date in EVENTS.items() if abs((date.to_period("M") - event_date.to_period("M")).n) <= 2]
            rows.append({"variable": column, "label": friendly(column), "observation_date": date.date().isoformat(), "transformed_value": value, "robust_z": robust_z.loc[date], "near_prespecified_event": " | ".join(nearby), "action": "review_do_not_auto_delete"})
    return pd.DataFrame(rows)


def event_windows(changes: pd.DataFrame, width: int = 6) -> pd.DataFrame:
    rows = []
    for event, date in EVENTS.items():
        before_start = date - pd.offsets.MonthEnd(width)
        after_end = date + pd.offsets.MonthEnd(width)
        for column in changes.columns:
            before = changes.loc[(changes.index >= before_start) & (changes.index < date), column].dropna()
            after = changes.loc[(changes.index >= date) & (changes.index <= after_end), column].dropna()
            pooled = pd.concat([before, after]).std()
            rows.append({"event": event, "event_month": date.date().isoformat(), "variable": column, "label": friendly(column), "months_before": len(before), "months_after": len(after), "mean_before": before.mean(), "mean_after": after.mean(), "mean_shift": after.mean() - before.mean(), "standardized_mean_shift": (after.mean() - before.mean()) / pooled if pd.notna(pooled) and pooled > 0 else np.nan, "interpretation": "descriptive_window_not_causal_effect"})
    return pd.DataFrame(rows)


def plot_panels(panel: pd.DataFrame, output_dir: Path) -> list[str]:
    families = {
        "external_controls": [column for column in panel if not column.startswith(("msdi_", "model_", "news_", "effective_", "peak_model", "dimension_model")) and column != "scfi"],
        "msdi_and_coverage": [column for column in panel if column.startswith(("msdi_", "news_", "effective_", "peak_model", "dimension_model"))],
        "model_specific_paths": [column for column in panel if column.startswith("model_")],
    }
    outputs = []
    for family, columns in families.items():
        if not columns:
            continue
        columns = sorted(set(columns))
        ncols = 3
        nrows = math.ceil(len(columns) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(4, 2.7 * nrows)), squeeze=False, sharex=True)
        for ax, column in zip(axes.flat, columns):
            ax.plot(panel.index, panel[column], linewidth=1.1)
            ax.set_title(friendly(column), loc="left", fontsize=9)
            ax.grid(alpha=0.2)
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for ax in axes.flat[len(columns):]:
            ax.set_visible(False)
        fig.tight_layout()
        path = output_dir / f"{family}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path.name)
    return outputs


def write_findings(output: Path, panel: pd.DataFrame, stationarity: pd.DataFrame, outliers: pd.DataFrame, windows: pd.DataFrame) -> None:
    changed = stationarity[stationarity["representation"].eq("monthly_change")]
    counts = changed["decision"].value_counts()
    largest = windows.dropna(subset=["standardized_mean_shift"]).assign(abs_shift=lambda x: x["standardized_mean_shift"].abs()).nlargest(12, "abs_shift")
    lines = [
        "# Point 3 - Temporal-property findings",
        "",
        f"Monthly variables analyzed: {len(panel.columns)} over {len(panel)} calendar months.",
        "",
        "## Stationarity after the declared monthly transformation",
        "",
    ]
    lines.extend(f"- `{name}`: {count} series." for name, count in counts.items())
    lines.extend(["", "## Extreme observations", "", f"- Robust monthly-change flags: {len(outliers)}. These are review flags and were not deleted.", "", "## Largest pre-specified event-window shifts", "", "| Event | Variable | Standardized shift |", "|---|---|---:|"])
    for row in largest.itertuples(index=False):
        lines.append(f"| {row.event} | {row.label} | {row.standardized_mean_shift:.3f} |")
    lines.extend(["", "Event-window shifts are descriptive and do not identify causal effects."])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    changes, yoy, transform_table = transformations(panel)
    distributions = distribution_summary(panel, changes, yoy)
    stationarity = stationarity_table(panel, changes, yoy)
    dependence = dependence_table(changes)
    seasonality = seasonal_table(changes)
    outliers = outlier_table(changes)
    windows = event_windows(changes)
    figures = plot_panels(panel, args.output_dir)

    panel.rename_axis("month_end").to_csv(args.output_dir / "monthly_temporal_panel.csv")
    transform_table.to_csv(args.output_dir / "transformation_registry.csv", index=False)
    distributions.to_csv(args.output_dir / "distribution_summary.csv", index=False)
    stationarity.to_csv(args.output_dir / "stationarity_tests.csv", index=False)
    dependence.to_csv(args.output_dir / "acf_pacf.csv", index=False)
    seasonality.to_csv(args.output_dir / "seasonal_month_profiles.csv", index=False)
    outliers.to_csv(args.output_dir / "extreme_month_review.csv", index=False)
    windows.to_csv(args.output_dir / "prespecified_event_windows.csv", index=False)
    write_findings(args.output_dir / "point_03_findings.md", panel, stationarity, outliers, windows)

    manifest = {
        "protocol_point": 3,
        "run_mode": "test" if args.test_run else "full",
        "status": "complete",
        "months": len(panel),
        "variables": len(panel.columns),
        "first_month": panel.index.min().date().isoformat(),
        "last_month": panel.index.max().date().isoformat(),
        "figures": figures,
        "guardrail": "Outliers retained; event windows and stationarity diagnostics are descriptive, not causal identification.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
