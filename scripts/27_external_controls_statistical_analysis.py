"""Run descriptive pairwise, lagged-association, and multicollinearity checks.

This script does not estimate structural causality. It uses only already
available controls and reports FDR-adjusted pairwise tests and VIF diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EDA = ROOT / "outputs/external_controls_eda"
DEFAULT_OUTPUT = ROOT / "outputs/external_controls_statistics"
TARGETS = ("scfi_return", "scfi_log_change_1w", "peak_score_media_corrected", "breadth_score_media_corrected", "scfi_monthly_mean_log_change_1m", "peak_score_media_corrected_mean", "breadth_score_media_corrected_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistical checks for external controls.")
    parser.add_argument("--eda-dir", type=Path, default=DEFAULT_EDA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frequency", choices=("weekly", "monthly", "both"), default="both")
    parser.add_argument("--maximum-lag", type=int, default=12)
    parser.add_argument("--minimum-observations", type=int, default=40)
    parser.add_argument("--version", default="external_controls_statistics_v001")
    return parser.parse_args()


def controls(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(("cpb_", "gscpi", "unctad_", "imf_")) and not column.endswith("_source_date")]


def hac_association(pair: pd.DataFrame, maximum_lag: int) -> tuple[float, float, float]:
    standardized = (pair - pair.mean()) / pair.std(ddof=0)
    fit = sm.OLS(standardized.iloc[:, 0], sm.add_constant(standardized.iloc[:, 1])).fit(
        cov_type="HAC", cov_kwds={"maxlags": min(maximum_lag, max(1, len(pair) // 5))}
    )
    return float(fit.params.iloc[1]), float(fit.bse.iloc[1]), float(fit.pvalues.iloc[1])


def correlation_rows(frame: pd.DataFrame, candidates: list[str], minimum: int, maximum_lag: int) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        if target not in frame: continue
        for variable in candidates:
            pair = frame[[target, variable]].dropna()
            if len(pair) < minimum or pair[target].nunique() < 2 or pair[variable].nunique() < 2: continue
            pr, pp = pearsonr(pair[target], pair[variable]); sr, sp = spearmanr(pair[target], pair[variable]); beta, se, hp = hac_association(pair, maximum_lag)
            rows.extend(({"target": target, "variable": variable, "method": "pearson", "n": len(pair), "correlation": pr, "p_value_iid": pp, "hac_standardized_beta": beta, "hac_standard_error": se, "p_value_hac": hp}, {"target": target, "variable": variable, "method": "spearman", "n": len(pair), "correlation": sr, "p_value_iid": sp, "hac_standardized_beta": beta, "hac_standard_error": se, "p_value_hac": hp}))
    output = pd.DataFrame(rows)
    if not output.empty: output["p_value_hac_fdr_bh"] = multipletests(output.p_value_hac, method="fdr_bh")[1]
    return output


def lagged_rows(frame: pd.DataFrame, candidates: list[str], maximum_lag: int, minimum: int, hac_lag: int) -> pd.DataFrame:
    target = next((column for column in ("scfi_return", "scfi_log_change_1w", "scfi_monthly_mean_log_change_1m") if column in frame), None)
    if target is None: return pd.DataFrame()
    rows = []
    for variable in candidates:
        for lag in range(maximum_lag + 1):
            pair = pd.DataFrame({"target": frame[target], "control": frame[variable].shift(lag)}).dropna()
            if len(pair) < minimum or pair.control.nunique() < 2: continue
            r, p = pearsonr(pair.target, pair.control); beta, se, hp = hac_association(pair, hac_lag)
            rows.append({"target": target, "variable": variable, "control_lag_rows": lag, "n": len(pair), "pearson_correlation": r, "p_value_iid": p, "hac_standardized_beta": beta, "hac_standard_error": se, "p_value_hac": hp})
    output = pd.DataFrame(rows)
    if not output.empty: output["p_value_hac_fdr_bh"] = multipletests(output.p_value_hac, method="fdr_bh")[1]
    return output


def vif_rows(frame: pd.DataFrame, candidates: list[str], minimum: int) -> pd.DataFrame:
    usable = frame[candidates].dropna()
    if len(usable) < minimum or len(candidates) < 2: return pd.DataFrame(columns=["variable", "vif", "n_complete"])
    design = sm.add_constant(usable.astype(float), has_constant="add")
    return pd.DataFrame({"variable": candidates, "vif": [variance_inflation_factor(design.values, index + 1) for index in range(len(candidates))], "n_complete": len(usable)})


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); reports = {}
    frequencies = ("weekly", "monthly") if args.frequency == "both" else (args.frequency,)
    for frequency in frequencies:
        path = args.eda_dir / f"{frequency}_external_controls_panel.csv"
        if not path.is_file(): raise FileNotFoundError(f"Run Stage 26 first; missing {path}")
        frame = pd.read_csv(path, low_memory=False); candidates = controls(frame)
        if not candidates: raise ValueError(f"No cleaned external-control columns in {path}")
        hac_lag = 12 if frequency == "weekly" else 6
        corr = correlation_rows(frame, candidates, args.minimum_observations, hac_lag)
        lagged = lagged_rows(frame, candidates, args.maximum_lag, args.minimum_observations, hac_lag)
        vif = vif_rows(frame, candidates, args.minimum_observations)
        corr.to_csv(args.output_dir / f"{frequency}_control_pairwise_correlations.csv", index=False)
        lagged.to_csv(args.output_dir / f"{frequency}_control_lagged_scfi_associations.csv", index=False)
        vif.to_csv(args.output_dir / f"{frequency}_control_vif.csv", index=False)
        reports[frequency] = {"control_variables": candidates, "pairwise_tests": len(corr), "lagged_tests": len(lagged), "vif_rows": len(vif), "interpretation": "descriptive associations only; use selected lagged controls in the later leakage-safe forecasting and causal-ML reruns"}
    (args.output_dir / f"{args.version}_manifest.json").write_text(json.dumps({"version": args.version, "reports": reports, "inference": "HAC OLS p-values are FDR-adjusted within each exported test family; IID Pearson/Spearman p-values are descriptive only", "causal_claim_allowed": False}, indent=2), encoding="utf-8")
    print(f"Statistical analysis complete: {args.output_dir}")


if __name__ == "__main__": main()
