"""Point 8: low-cost, leakage-safe forecasting readiness diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "eda/outputs/point_08_forecasting_readiness"
MONTHLY_PANEL = ROOT / "data/06_model_ready/causality/msdi_scfi_monthly_panel.csv"
POINT_05_NONLINEAR = ROOT / "eda/outputs/point_05/scfi_nonlinear_diagnostics.csv"
HORIZONS = (1, 3, 6)
MINIMUM_TRAINING_ROWS = 48

PREDICTOR_SETS = {
    "ar_baseline": ["scfi_return_lag_0", "scfi_return_lag_1", "scfi_return_lag_2"],
    "ar_plus_peak_msdi": ["scfi_return_lag_0", "scfi_return_lag_1", "scfi_return_lag_2", "peak_score_media_corrected_mean"],
    "ar_plus_breadth_msdi": ["scfi_return_lag_0", "scfi_return_lag_1", "scfi_return_lag_2", "breadth_score_media_corrected_mean"],
    "ar_plus_both_msdi": ["scfi_return_lag_0", "scfi_return_lag_1", "scfi_return_lag_2", "peak_score_media_corrected_mean", "breadth_score_media_corrected_mean"],
    "ar_plus_existing_controls": ["scfi_return_lag_0", "scfi_return_lag_1", "scfi_return_lag_2", "brent_monthly_mean_log_change_1m", "cpb_world_trade_volume_lag_1m", "gscpi_lag_1m", "imf_portwatch_global_container_calls_lag_1m"],
    "ar_plus_msdi_and_controls": ["scfi_return_lag_0", "scfi_return_lag_1", "scfi_return_lag_2", "peak_score_media_corrected_mean", "breadth_score_media_corrected_mean", "brent_monthly_mean_log_change_1m", "cpb_world_trade_volume_lag_1m", "gscpi_lag_1m", "imf_portwatch_global_container_calls_lag_1m"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check monthly forecasting readiness without tuning.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-run", action="store_true", help="Run expanding OLS and naive baselines only.")
    return parser.parse_args()


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(MONTHLY_PANEL, low_memory=False)
    panel["month_ending"] = pd.to_datetime(panel["month_ending"], errors="coerce")
    panel = panel.sort_values("month_ending").reset_index(drop=True)
    current = pd.to_numeric(panel["scfi_monthly_mean_log_change_1m"], errors="coerce")
    panel["scfi_return_lag_0"] = current
    panel["scfi_return_lag_1"] = current.shift(1)
    panel["scfi_return_lag_2"] = current.shift(2)
    return panel


def target_audit(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        column = f"target_scfi_monthly_log_change_h{horizon}m"
        values = pd.to_numeric(panel[column], errors="coerce")
        available = values.notna()
        rows.append({
            "horizon_months": horizon,
            "target": column,
            "observations": int(available.sum()),
            "first_origin": panel.loc[available, "month_ending"].min(),
            "last_origin": panel.loc[available, "month_ending"].max(),
            "mean": values.mean(),
            "standard_deviation": values.std(),
            "minimum": values.min(),
            "median": values.median(),
            "maximum": values.max(),
            "positive_share": values.gt(0).mean(),
            "adjacent_targets_share_underlying_months": (horizon - 1) / horizon,
            "error_dependence_note": "overlapping target; use HAC/block methods" if horizon > 1 else "nonoverlapping at monthly origins",
        })
    return pd.DataFrame(rows)


def predictor_support(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, columns in PREDICTOR_SETS.items():
        complete = panel[columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        rows.append({"predictor_set": name, "predictors": " | ".join(columns), "rows_complete": int(complete.sum()), "first_complete_origin": panel.loc[complete, "month_ending"].min(), "last_complete_origin": panel.loc[complete, "month_ending"].max(), "operational_status": "publication_date_simulation" if "msdi" in name else "encoded_existing_controls_only"})
    unresolved = ("rwi_container_throughput", "china_merchandise_exports", "cny_per_usd", "global_geopolitical_risk", "trade_policy_uncertainty", "us_china_tension", "oil_price_uncertainty", "infectious_disease_emv")
    for variable in unresolved:
        rows.append({"predictor_set": f"candidate_{variable}", "predictors": variable, "rows_complete": np.nan, "first_complete_origin": pd.NaT, "last_complete_origin": pd.NaT, "operational_status": "eda_only_release_contract_missing"})
    return pd.DataFrame(rows)


def fit_ols(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.Series) -> float:
    mean = train_x.mean()
    scale = train_x.std(ddof=1).replace(0, 1)
    x = ((train_x - mean) / scale).to_numpy(dtype=float)
    z = ((test_x - mean) / scale).to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(design, train_y.to_numpy(dtype=float), rcond=None)[0]
    return float(np.r_[1.0, z] @ coefficients)


def expanding_forecasts(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    for horizon in HORIZONS:
        target = f"target_scfi_monthly_log_change_h{horizon}m"
        for origin_index in range(len(panel)):
            observed = pd.to_numeric(pd.Series([panel.loc[origin_index, target]]), errors="coerce").iloc[0]
            if pd.isna(observed):
                continue
            origin_date = panel.loc[origin_index, "month_ending"]
            # A training label formed at row t is known only after t+h.
            eligible_train_end = origin_index - horizon
            if eligible_train_end < 0:
                continue
            predictions.append({"origin": origin_date, "horizon_months": horizon, "model": "random_walk_zero_change", "observed": observed, "prediction": 0.0, "training_rows": eligible_train_end + 1, "label_end_index": eligible_train_end})
            historic_targets = pd.to_numeric(panel.loc[:eligible_train_end, target], errors="coerce").dropna()
            if len(historic_targets) >= MINIMUM_TRAINING_ROWS:
                predictions.append({"origin": origin_date, "horizon_months": horizon, "model": "expanding_historical_mean", "observed": observed, "prediction": historic_targets.mean(), "training_rows": len(historic_targets), "label_end_index": eligible_train_end})
            for name, columns in PREDICTOR_SETS.items():
                train = panel.loc[:eligible_train_end, [*columns, target]].copy()
                train[columns] = train[columns].apply(pd.to_numeric, errors="coerce")
                train[target] = pd.to_numeric(train[target], errors="coerce")
                train = train.dropna()
                test_x = pd.to_numeric(panel.loc[origin_index, columns], errors="coerce")
                if len(train) < max(MINIMUM_TRAINING_ROWS, len(columns) * 5) or test_x.isna().any():
                    continue
                prediction = fit_ols(train[columns], train[target], test_x)
                predictions.append({"origin": origin_date, "horizon_months": horizon, "model": name, "observed": observed, "prediction": prediction, "training_rows": len(train), "label_end_index": eligible_train_end})
    prediction_frame = pd.DataFrame(predictions)
    prediction_frame["error"] = prediction_frame["observed"] - prediction_frame["prediction"]
    prediction_frame["absolute_error"] = prediction_frame["error"].abs()
    prediction_frame["squared_error"] = prediction_frame["error"].pow(2)
    prediction_frame["direction_correct"] = np.sign(prediction_frame["observed"]) == np.sign(prediction_frame["prediction"])
    metrics = prediction_frame.groupby(["horizon_months", "model"], as_index=False).agg(forecasts=("observed", "size"), mae=("absolute_error", "mean"), rmse=("squared_error", lambda s: np.sqrt(s.mean())), bias=("error", "mean"), direction_accuracy=("direction_correct", "mean"), first_origin=("origin", "min"), last_origin=("origin", "max"))
    return prediction_frame, metrics


def comparison_table(predictions: pd.DataFrame) -> pd.DataFrame:
    baseline = predictions.loc[predictions["model"].eq("ar_baseline"), ["origin", "horizon_months", "absolute_error"]].rename(columns={"absolute_error": "baseline_absolute_error"})
    merged = predictions.merge(baseline, on=["origin", "horizon_months"], how="inner")
    merged = merged.loc[~merged["model"].eq("ar_baseline")]
    merged["absolute_error_improvement_vs_ar"] = merged["baseline_absolute_error"] - merged["absolute_error"]
    return merged.groupby(["horizon_months", "model"], as_index=False).agg(paired_forecasts=("origin", "size"), mean_absolute_error_improvement_vs_ar=("absolute_error_improvement_vs_ar", "mean"), improvement_share=("absolute_error_improvement_vs_ar", lambda s: s.gt(0).mean()))


def nonlinear_readiness() -> pd.DataFrame:
    if not POINT_05_NONLINEAR.exists():
        return pd.DataFrame(columns=["variable", "label", "observations", "quadratic_incremental_r_squared", "nonlinearity_review_flag"])
    raw = pd.read_csv(POINT_05_NONLINEAR)
    raw["nonlinearity_review_flag"] = pd.to_numeric(raw["quadratic_incremental_r_squared"], errors="coerce").ge(0.05)
    return raw


def write_findings(path: Path, target: pd.DataFrame, support: pd.DataFrame, metrics: pd.DataFrame, comparison: pd.DataFrame, nonlinear: pd.DataFrame) -> None:
    best = metrics.sort_values(["horizon_months", "rmse"]).groupby("horizon_months", as_index=False).first()
    lines = [
        "# Point 8 - Forecasting-readiness findings",
        "",
        "This is a small expanding-window OLS audit, not the final model comparison. Hyperparameter tuning is not run.",
        "",
        "| Horizon | Lowest test RMSE model | Forecasts | RMSE |",
        "|---:|---|---:|---:|",
    ]
    for row in best.itertuples(index=False): lines.append(f"| {row.horizon_months} | {row.model} | {row.forecasts} | {row.rmse:.4f} |")
    lines.extend([
        "",
        "Different models can have different forecast-origin counts because of control availability. Final comparisons must use identical paired origins.",
        "",
        "For 3- and 6-month horizons, adjacent targets overlap. Standard iid tests are invalid; use HAC or block-based inference.",
        "",
        "MSDI forecasts simulate availability at article publication month. Real-time operational claims remain conditional on an ingestion/scoring-latency sensitivity analysis.",
        "",
        f"Variables with quadratic incremental R-squared at least 0.05: {int(nonlinear.get('nonlinearity_review_flag', pd.Series(dtype=bool)).sum())}. This is only a flag for later out-of-sample testing.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    target = target_audit(panel)
    support = predictor_support(panel)
    predictions, metrics = expanding_forecasts(panel)
    comparisons = comparison_table(predictions)
    nonlinear = nonlinear_readiness()
    target.to_csv(args.output_dir / "target_distribution_and_overlap.csv", index=False)
    support.to_csv(args.output_dir / "predictor_set_support.csv", index=False)
    predictions.to_csv(args.output_dir / "eda_expanding_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "eda_expanding_metrics.csv", index=False)
    comparisons.to_csv(args.output_dir / "eda_incremental_comparison.csv", index=False)
    nonlinear.to_csv(args.output_dir / "nonlinearity_readiness_flags.csv", index=False)
    write_findings(args.output_dir / "point_08_findings.md", target, support, metrics, comparisons, nonlinear)
    manifest = {
        "protocol_point": 8,
        "run_mode": "test" if args.test_run else "full",
        "status": "complete_low_cost_readiness_only",
        "horizons": list(HORIZONS),
        "predictions": len(predictions),
        "models_including_baselines": int(predictions["model"].nunique()),
        "tuning_performed": False,
        "guardrail": "Expanding training uses only labels realized by each forecast origin; results are readiness diagnostics, not final model rankings.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
