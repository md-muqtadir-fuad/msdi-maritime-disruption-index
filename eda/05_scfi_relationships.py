"""Point 5: descriptive, symmetric relationships between every series and SCFI."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "eda/outputs/point_05_scfi_relationships"
MINIMUM_OBSERVATIONS = 24
LAGS = tuple(range(-6, 7))
ROLLING_WINDOW = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze all monthly relationships with SCFI.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-run", action="store_true", help="No tuning; deterministic monthly diagnostics only.")
    return parser.parse_args()


def load_temporal_module():
    path = ROOT / "eda/03_temporal_properties.py"
    spec = importlib.util.spec_from_file_location("eda_temporal", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_correlation(x: pd.Series, y: pd.Series, method: str) -> dict[str, float | int]:
    frame = pd.concat([x.rename("x"), y.rename("y")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    result: dict[str, float | int] = {"observations": len(frame), "coefficient": np.nan, "p_value": np.nan}
    if len(frame) < MINIMUM_OBSERVATIONS or frame["x"].nunique() < 3 or frame["y"].nunique() < 3:
        return result
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        statistic, p_value = (pearsonr(frame["x"], frame["y"]) if method == "pearson" else spearmanr(frame["x"], frame["y"]))
    result.update({"coefficient": float(statistic), "p_value": float(p_value)})
    return result


def bh_by_family(table: pd.DataFrame, family_columns: list[str]) -> pd.DataFrame:
    result = table.copy()
    result["fdr_q_value"] = np.nan
    result["fdr_reject_05"] = False
    for _, indices in result.groupby(family_columns, dropna=False).groups.items():
        valid = result.loc[indices, "p_value"].notna()
        selected = result.loc[indices].index[valid]
        if len(selected):
            reject, adjusted, _, _ = multipletests(result.loc[selected, "p_value"], alpha=0.05, method="fdr_bh")
            result.loc[selected, "fdr_q_value"] = adjusted
            result.loc[selected, "fdr_reject_05"] = reject
    return result


def relationship_table(representations: dict[str, pd.DataFrame], labeler) -> pd.DataFrame:
    rows = []
    for representation, frame in representations.items():
        target = frame["scfi"]
        for variable in frame.columns:
            if variable == "scfi":
                continue
            for lag in LAGS:
                direction = "predictor_leads_scfi" if lag > 0 else ("scfi_leads_predictor" if lag < 0 else "contemporaneous")
                for method in ("pearson", "spearman"):
                    rows.append({
                        "representation": representation,
                        "variable": variable,
                        "label": labeler(variable),
                        "lag_months": lag,
                        "direction": direction,
                        "method": method,
                        **safe_correlation(frame[variable].shift(lag), target, method),
                    })
    return bh_by_family(pd.DataFrame(rows), ["representation", "method", "direction"])


def ar1_residual(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    frame = pd.concat([values.rename("current"), values.shift(1).rename("lag")], axis=1).dropna()
    residual = pd.Series(np.nan, index=values.index, dtype=float)
    if len(frame) < MINIMUM_OBSERVATIONS or frame["lag"].nunique() < 3:
        return residual
    design = np.column_stack([np.ones(len(frame)), frame["lag"].to_numpy()])
    coefficients = np.linalg.lstsq(design, frame["current"].to_numpy(), rcond=None)[0]
    residual.loc[frame.index] = frame["current"].to_numpy() - design @ coefficients
    return residual


def prewhitened_table(changes: pd.DataFrame, labeler) -> pd.DataFrame:
    target = ar1_residual(changes["scfi"])
    rows = []
    for variable in changes.columns:
        if variable == "scfi":
            continue
        residual = ar1_residual(changes[variable])
        for lag in LAGS:
            rows.append({
                "variable": variable,
                "label": labeler(variable),
                "lag_months": lag,
                "direction": "predictor_leads_scfi" if lag > 0 else ("scfi_leads_predictor" if lag < 0 else "contemporaneous"),
                "method": "pearson_ar1_residuals",
                **safe_correlation(residual.shift(lag), target, "pearson"),
            })
    return bh_by_family(pd.DataFrame(rows), ["method", "direction"])


def rolling_table(changes: pd.DataFrame, labeler) -> pd.DataFrame:
    rows = []
    for variable in changes.columns:
        if variable == "scfi":
            continue
        correlations = changes[variable].rolling(ROLLING_WINDOW, min_periods=18).corr(changes["scfi"])
        for date, value in correlations.dropna().items():
            rows.append({"variable": variable, "label": labeler(variable), "window_months": ROLLING_WINDOW, "window_end": date.date().isoformat(), "pearson_correlation": value})
    return pd.DataFrame(rows)


def regime_table(changes: pd.DataFrame, labeler) -> pd.DataFrame:
    regimes = {
        "pre_covid": (pd.Timestamp("2016-02-29"), pd.Timestamp("2020-02-29")),
        "covid_and_immediate_aftermath": (pd.Timestamp("2020-03-31"), pd.Timestamp("2022-12-31")),
        "post_covid": (pd.Timestamp("2023-01-31"), pd.Timestamp("2026-02-28")),
    }
    rows = []
    for regime, (start, end) in regimes.items():
        frame = changes.loc[start:end]
        for variable in changes.columns:
            if variable == "scfi":
                continue
            for method in ("pearson", "spearman"):
                rows.append({"regime": regime, "start": start.date().isoformat(), "end": end.date().isoformat(), "variable": variable, "label": labeler(variable), "method": method, **safe_correlation(frame[variable], frame["scfi"], method)})
    return bh_by_family(pd.DataFrame(rows), ["regime", "method"])


def nonlinear_table(changes: pd.DataFrame, labeler) -> pd.DataFrame:
    rows = []
    for variable in changes.columns:
        if variable == "scfi":
            continue
        frame = changes[[variable, "scfi"]].dropna()
        record = {"variable": variable, "label": labeler(variable), "observations": len(frame), "linear_r_squared": np.nan, "quadratic_r_squared": np.nan, "quadratic_incremental_r_squared": np.nan}
        if len(frame) >= MINIMUM_OBSERVATIONS and frame[variable].nunique() >= 4:
            x = frame[variable].to_numpy(dtype=float)
            y = frame["scfi"].to_numpy(dtype=float)
            standard_deviation = x.std(ddof=1)
            total = np.square(y - y.mean()).sum()
            if standard_deviation > 0 and total > 0:
                x = (x - x.mean()) / standard_deviation
                linear = np.column_stack([np.ones(len(x)), x])
                quadratic = np.column_stack([np.ones(len(x)), x, np.square(x)])
                linear_residual = y - linear @ np.linalg.lstsq(linear, y, rcond=None)[0]
                quadratic_residual = y - quadratic @ np.linalg.lstsq(quadratic, y, rcond=None)[0]
                linear_r2 = 1 - np.square(linear_residual).sum() / total
                quadratic_r2 = 1 - np.square(quadratic_residual).sum() / total
                record.update({"linear_r_squared": linear_r2, "quadratic_r_squared": quadratic_r2, "quadratic_incremental_r_squared": quadratic_r2 - linear_r2})
        rows.append(record)
    return pd.DataFrame(rows)


def plot_scatter_families(changes: pd.DataFrame, labeler, output_dir: Path) -> list[str]:
    candidates = [column for column in changes if column != "scfi"]
    prefixes = ("msdi_", "model_", "news_", "effective_", "peak_model", "dimension_model")
    families = {
        "control_change_scatter": [c for c in candidates if not c.startswith(prefixes)],
        "msdi_change_scatter": [c for c in candidates if c.startswith(prefixes) and not c.startswith("model_")],
        "model_change_scatter": [c for c in candidates if c.startswith("model_")],
    }
    outputs = []
    for name, columns in families.items():
        if not columns:
            continue
        ncols = 3
        nrows = math.ceil(len(columns) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, max(4, 3.0 * nrows)), squeeze=False)
        for ax, variable in zip(axes.flat, columns):
            frame = changes[[variable, "scfi"]].dropna()
            ax.scatter(frame[variable], frame["scfi"], s=11, alpha=0.45, edgecolors="none")
            if len(frame) >= MINIMUM_OBSERVATIONS and frame[variable].nunique() >= 4:
                smooth = lowess(frame["scfi"], frame[variable], frac=0.5, return_sorted=True)
                ax.plot(smooth[:, 0], smooth[:, 1], linewidth=1.3, color="#b24a3a")
            ax.axhline(0, linewidth=0.6, color="0.55")
            ax.set_title(labeler(variable), loc="left", fontsize=9)
            ax.set_xlabel("Monthly transformed change", fontsize=8)
            ax.set_ylabel("SCFI monthly log change", fontsize=8)
            ax.grid(alpha=0.18)
        for ax in axes.flat[len(columns):]:
            ax.set_visible(False)
        fig.tight_layout()
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path.name)
    return outputs


def write_findings(path: Path, relations: pd.DataFrame, prewhitened: pd.DataFrame, regimes: pd.DataFrame) -> None:
    monthly = relations.loc[relations["representation"].eq("monthly_change")]
    lead = monthly.loc[monthly["lag_months"].between(1, 6)]
    reverse = monthly.loc[monthly["lag_months"].between(-6, -1)]
    concurrent = monthly.loc[monthly["lag_months"].eq(0)]
    lines = [
        "# Point 5 - Descriptive relationships with SCFI",
        "",
        f"- Variables analyzed: {relations['variable'].nunique()}.",
        f"- FDR-significant monthly-change predictor-leading tests: {int(lead['fdr_reject_05'].sum())} of {len(lead)}.",
        f"- FDR-significant monthly-change reverse-leading tests: {int(reverse['fdr_reject_05'].sum())} of {len(reverse)}.",
        f"- FDR-significant contemporaneous monthly-change tests: {int(concurrent['fdr_reject_05'].sum())} of {len(concurrent)}.",
        f"- FDR-significant AR(1)-residual cross-correlations: {int(prewhitened['fdr_reject_05'].sum())} of {len(prewhitened)}.",
        f"- FDR-significant regime-specific contemporaneous tests: {int(regimes['fdr_reject_05'].sum())} of {len(regimes)}.",
        "",
        "These counts audit dependence; they are not a variable-selection rule. Level correlations can be spurious, lead/lag tests are symmetric, and repeated event regimes can drive apparent relationships. No variable was accepted or rejected.",
        "",
        "Quadratic incremental R-squared is descriptive and does not justify a nonlinear forecasting model without out-of-sample validation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporal = load_temporal_module()
    panel = temporal.build_panel()
    changes, yoy, _ = temporal.transformations(panel)
    representations = {"level": panel, "monthly_change": changes, "year_over_year_change": yoy}

    relations = relationship_table(representations, temporal.friendly)
    prewhitened = prewhitened_table(changes, temporal.friendly)
    rolling = rolling_table(changes, temporal.friendly)
    regimes = regime_table(changes, temporal.friendly)
    nonlinear = nonlinear_table(changes, temporal.friendly)
    figures = plot_scatter_families(changes, temporal.friendly, args.output_dir)

    relations.to_csv(args.output_dir / "scfi_lead_lag_correlations.csv", index=False)
    prewhitened.to_csv(args.output_dir / "scfi_prewhitened_cross_correlations.csv", index=False)
    rolling.to_csv(args.output_dir / "scfi_rolling_correlations.csv", index=False)
    regimes.to_csv(args.output_dir / "scfi_regime_correlations.csv", index=False)
    nonlinear.to_csv(args.output_dir / "scfi_nonlinear_diagnostics.csv", index=False)
    write_findings(args.output_dir / "point_05_findings.md", relations, prewhitened, regimes)

    manifest = {
        "protocol_point": 5,
        "run_mode": "test" if args.test_run else "full",
        "status": "complete_descriptive_no_selection",
        "variables": int(relations["variable"].nunique()),
        "relationship_tests": len(relations),
        "prewhitened_tests": len(prewhitened),
        "rolling_rows": len(rolling),
        "figures": figures,
        "guardrail": "No variable accepted or rejected from correlation, FDR significance, regime behavior, or in-sample nonlinearity.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
