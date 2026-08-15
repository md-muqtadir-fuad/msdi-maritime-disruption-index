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

TARGETS = (
    "scfi_return",
    "scfi_log_change_1w",
    "peak_score_media_corrected",
    "breadth_score_media_corrected",
    "scfi_monthly_mean_log_change_1m",
    "peak_score_media_corrected_mean",
    "breadth_score_media_corrected_mean",
    "non_freight_peak_score_media_corrected_mean",
)

PRIMARY_CONTROLS = (
    "cpb_world_trade_volume",
    "rwi_isl_container_throughput_index",
    "wto_china_merchandise_exports",
    "cny_usd_exchange_rate",
    "unctad_china_core_lane_lsbci",
    "gscpi",
    "imf_portwatch_global_container_calls",
    "gpr_index",
    "tpu_index",
    "uct_index",
    "opu_index",
    "infectious_disease_emv",
    "brent_price",
    "brent_monthly_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistical checks for external controls.")
    parser.add_argument("--eda-dir", type=Path, default=DEFAULT_EDA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frequency", choices=("weekly", "monthly", "both"), default="both")
    parser.add_argument("--maximum-lag", type=int, default=6)
    parser.add_argument("--minimum-observations", type=int, default=30)
    parser.add_argument("--version", default="external_controls_statistics_v002")
    return parser.parse_args()


def controls(frame: pd.DataFrame) -> list[str]:
    # Select unlagged distinct control series for clean pairwise and lag checks
    return [col for col in PRIMARY_CONTROLS if col in frame.columns]


def hac_association(pair: pd.DataFrame, maximum_lag: int) -> tuple[float, float, float]:
    standardized = (pair - pair.mean()) / pair.std(ddof=0).replace(0, np.nan)
    standardized = standardized.dropna()
    if len(standardized) < 5:
        return 0.0, 1.0, 1.0
    fit = sm.OLS(standardized.iloc[:, 0], sm.add_constant(standardized.iloc[:, 1])).fit(
        cov_type="HAC", cov_kwds={"maxlags": min(maximum_lag, max(1, len(pair) // 5))}
    )
    return float(fit.params.iloc[1]), float(fit.bse.iloc[1]), float(fit.pvalues.iloc[1])


def correlation_rows(
    frame: pd.DataFrame, candidates: list[str], minimum: int, maximum_lag: int
) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        if target not in frame:
            continue
        for variable in candidates:
            if target == variable:
                continue
            pair = frame[[target, variable]].dropna()
            if (
                len(pair) < minimum
                or pair[target].nunique() < 2
                or pair[variable].nunique() < 2
            ):
                continue
            pr, pp = pearsonr(pair[target], pair[variable])
            sr, sp = spearmanr(pair[target], pair[variable])
            beta, se, hp = hac_association(pair, maximum_lag)
            rows.extend(
                (
                    {
                        "target": target,
                        "variable": variable,
                        "method": "pearson",
                        "n": len(pair),
                        "correlation": pr,
                        "p_value_iid": pp,
                        "hac_standardized_beta": beta,
                        "hac_standard_error": se,
                        "p_value_hac": hp,
                    },
                    {
                        "target": target,
                        "variable": variable,
                        "method": "spearman",
                        "n": len(pair),
                        "correlation": sr,
                        "p_value_iid": sp,
                        "hac_standardized_beta": beta,
                        "hac_standard_error": se,
                        "p_value_hac": hp,
                    },
                )
            )
    output = pd.DataFrame(rows)
    if not output.empty and "p_value_hac" in output.columns:
        output["p_value_hac_fdr_bh"] = multipletests(output.p_value_hac, method="fdr_bh")[1]
    return output


def lagged_rows(
    frame: pd.DataFrame, candidates: list[str], maximum_lag: int, minimum: int, hac_lag: int
) -> pd.DataFrame:
    target = next(
        (
            column
            for column in (
                "scfi_monthly_mean_log_change_1m",
                "scfi_log_change_1w",
                "scfi_return",
            )
            if column in frame
        ),
        None,
    )
    if target is None:
        return pd.DataFrame()
    rows = []
    for variable in candidates:
        for lag in range(maximum_lag + 1):
            pair = pd.DataFrame(
                {"target": frame[target], "control": frame[variable].shift(lag)}
            ).dropna()
            if len(pair) < minimum or pair.control.nunique() < 2:
                continue
            r, p = pearsonr(pair.target, pair.control)
            beta, se, hp = hac_association(pair, hac_lag)
            rows.append(
                {
                    "target": target,
                    "variable": variable,
                    "control_lag_rows": lag,
                    "n": len(pair),
                    "pearson_correlation": r,
                    "p_value_iid": p,
                    "hac_standardized_beta": beta,
                    "hac_standard_error": se,
                    "p_value_hac": hp,
                }
            )
    output = pd.DataFrame(rows)
    if not output.empty and "p_value_hac" in output.columns:
        output["p_value_hac_fdr_bh"] = multipletests(output.p_value_hac, method="fdr_bh")[1]
    return output


def vif_rows(frame: pd.DataFrame, candidates: list[str], minimum: int) -> pd.DataFrame:
    # Select non-constant candidate columns that have complete cases
    clean_candidates = [c for c in candidates if c in frame.columns and frame[c].notna().sum() >= minimum]
    usable = frame[clean_candidates].dropna()
    if len(usable) < minimum or len(clean_candidates) < 2 or len(clean_candidates) >= len(usable):
        # Fallback to pairwise or core subset
        core_sub = [c for c in clean_candidates if c in PRIMARY_CONTROLS[:6]]
        usable = frame[core_sub].dropna()
        if len(usable) < minimum or len(core_sub) < 2:
            return pd.DataFrame(columns=["variable", "vif", "n_complete"])
        clean_candidates = core_sub

    design = sm.add_constant(usable.astype(float), has_constant="add")
    vif_vals = []
    for index in range(len(clean_candidates)):
        try:
            vif_val = variance_inflation_factor(design.values, index + 1)
        except Exception:
            vif_val = np.nan
        vif_vals.append(vif_val)
    return pd.DataFrame(
        {"variable": clean_candidates, "vif": vif_vals, "n_complete": len(usable)}
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    frequencies = ("weekly", "monthly") if args.frequency == "both" else (args.frequency,)

    for frequency in frequencies:
        path = args.eda_dir / f"{frequency}_external_controls_panel.csv"
        if not path.is_file():
            # Fallback to model-ready panel
            path = (
                ROOT / "data/06_model_ready/prediction/msdi_scfi_weekly_panel.csv"
                if frequency == "weekly"
                else ROOT / "data/06_model_ready/causality/msdi_scfi_monthly_panel.csv"
            )
        frame = pd.read_csv(path, low_memory=False)
        candidate_controls = controls(frame)
        corr_df = correlation_rows(
            frame, candidate_controls, args.minimum_observations, args.maximum_lag
        )
        lag_df = lagged_rows(
            frame, candidate_controls, args.maximum_lag, args.minimum_observations, args.maximum_lag
        )
        vif_df = vif_rows(frame, candidate_controls, args.minimum_observations)

        corr_df.to_csv(
            args.output_dir / f"{frequency}_external_controls_correlations.csv", index=False
        )
        lag_df.to_csv(
            args.output_dir / f"{frequency}_external_controls_lagged_associations.csv", index=False
        )
        vif_df.to_csv(args.output_dir / f"{frequency}_external_controls_vif.csv", index=False)
        reports[frequency] = {
            "rows": len(frame),
            "controls_checked": candidate_controls,
            "correlations_tested": len(corr_df),
            "lag_pairs_tested": len(lag_df),
            "vif_variables": len(vif_df),
        }

    manifest = {
        "version": args.version,
        "reports": reports,
        "warning": "Statistical associations are descriptive; do not claim causal identification.",
    }
    (args.output_dir / f"{args.version}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Statistical checks complete: {args.output_dir}")


if __name__ == "__main__":
    main()
