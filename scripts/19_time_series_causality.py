"""Run dynamic time-series analyses for MSDI and SCFI.

Weekly estimates are primary and monthly estimates are robustness checks. The
script tests stationarity and cointegration, estimates stationary VARX systems,
runs conventional and conditional Granger predictive-precedence tests, applies
PCMCI with FDR control, and estimates HAC local projections using innovations
in the change of each MSDI path.

These are observational dynamic-association analyses. VAR, Granger, PCMCI, and
local projections do not by themselves identify structural causal effects.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEEKLY_PANEL = (
    PROJECT_ROOT / "data/06_model_ready/prediction/msdi_scfi_weekly_panel.csv"
)
DEFAULT_MONTHLY_PANEL = (
    PROJECT_ROOT / "data/06_model_ready/causality/msdi_scfi_monthly_panel.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/time_series_causality"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs/logs"
ENDOGENOUS_VARIABLES = (
    "scfi_return",
    "msdi_change",
    "brent_return",
    "news_volume_change",
)
SEASONAL_CONTROLS = ("season_sin", "season_cos")
EXTERNAL_CONTROL_COLUMNS = ("cpb_world_trade_volume", "unctad_china_core_lane_lsbci")


def exogenous_controls(frame: pd.DataFrame) -> list[str]:
    return [*SEASONAL_CONTROLS, *[f"{column}_lag_1" for column in EXTERNAL_CONTROL_COLUMNS if f"{column}_lag_1" in frame]]


@dataclass(frozen=True)
class FrequencySpecification:
    frequency: str
    date_column: str
    scfi_column: str
    brent_column: str
    article_count_column: str
    msdi_columns: dict[str, str]
    maximum_lag: int
    lp_horizon: int
    irf_horizon: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate weekly-primary VAR, Granger, PCMCI, and local projections."
    )
    parser.add_argument("--weekly-panel", type=Path, default=DEFAULT_WEEKLY_PANEL)
    parser.add_argument("--monthly-panel", type=Path, default=DEFAULT_MONTHLY_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--version", default="causality_v001")
    parser.add_argument("--weekly-maximum-lag", type=int, default=12)
    parser.add_argument("--monthly-maximum-lag", type=int, default=6)
    parser.add_argument("--weekly-lp-horizon", type=int, default=12)
    parser.add_argument("--monthly-lp-horizon", type=int, default=6)
    parser.add_argument("--weekly-irf-horizon", type=int, default=12)
    parser.add_argument("--monthly-irf-horizon", type=int, default=6)
    parser.add_argument("--minimum-observations", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--pcmci-pc-alpha", type=float, default=0.20)
    parser.add_argument("--irf-monte-carlo-repetitions", type=int, default=500)
    parser.add_argument("--lp-bootstrap-repetitions", type=int, default=500)
    parser.add_argument(
        "--lp-block-length",
        type=int,
        default=0,
        help="Moving-block length; zero selects ceil(n^(1/3)) separately by sample.",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--skip-pcmci",
        action="store_true",
        help="Skip PCMCI only for dependency/debug checks; not for final research runs.",
    )
    return parser.parse_args()


def configure_logging(log_dir: Path, version: str) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"08_causality_analysis_{version}.log"
    logger = logging.getLogger("causality_analysis")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def frequency_specifications(args: argparse.Namespace) -> list[FrequencySpecification]:
    return [
        FrequencySpecification(
            frequency="weekly",
            date_column="observation_date",
            scfi_column="scfi",
            brent_column="brent_price",
            article_count_column="article_count",
            msdi_columns={
                "peak": "peak_score_media_corrected",
                "breadth": "breadth_score_media_corrected",
            },
            maximum_lag=args.weekly_maximum_lag,
            lp_horizon=args.weekly_lp_horizon,
            irf_horizon=args.weekly_irf_horizon,
        ),
        FrequencySpecification(
            frequency="monthly",
            date_column="month_ending",
            scfi_column="scfi_monthly_mean",
            brent_column="brent_monthly_mean",
            article_count_column="article_count_sum",
            msdi_columns={
                "peak": "peak_score_media_corrected_mean",
                "breadth": "breadth_score_media_corrected_mean",
            },
            maximum_lag=args.monthly_maximum_lag,
            lp_horizon=args.monthly_lp_horizon,
            irf_horizon=args.monthly_irf_horizon,
        ),
    ]


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "weekly_maximum_lag",
        "monthly_maximum_lag",
        "weekly_lp_horizon",
        "monthly_lp_horizon",
        "weekly_irf_horizon",
        "monthly_irf_horizon",
        "minimum_observations",
        "irf_monte_carlo_repetitions",
        "lp_bootstrap_repetitions",
    )
    for name in positive_integer_names:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.lp_block_length < 0:
        raise ValueError("--lp-block-length cannot be negative")
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must lie strictly between zero and one")
    if not 0 < args.pcmci_pc_alpha < 1:
        raise ValueError("--pcmci-pc-alpha must lie strictly between zero and one")


def load_panel(path: Path, spec: FrequencySpecification) -> pd.DataFrame:
    panel = pd.read_csv(path, low_memory=False)
    required = {
        spec.date_column,
        spec.scfi_column,
        spec.brent_column,
        spec.article_count_column,
        *spec.msdi_columns.values(),
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"{spec.frequency} panel lacks required columns: {missing}")
    panel[spec.date_column] = pd.to_datetime(panel[spec.date_column], errors="coerce")
    if panel[spec.date_column].isna().any():
        raise ValueError(f"{spec.frequency} panel contains unparseable dates")
    if panel[spec.date_column].duplicated().any():
        raise ValueError(f"{spec.frequency} panel contains duplicate dates")
    return panel.sort_values(spec.date_column).reset_index(drop=True)


def prepare_analysis_frame(
    panel: pd.DataFrame,
    spec: FrequencySpecification,
    path_name: str,
) -> pd.DataFrame:
    msdi_column = spec.msdi_columns[path_name]
    frame = pd.DataFrame(
        {
            "date": panel[spec.date_column],
            "scfi_level": pd.to_numeric(panel[spec.scfi_column], errors="coerce"),
            "brent_level": pd.to_numeric(panel[spec.brent_column], errors="coerce"),
            "msdi_level": pd.to_numeric(panel[msdi_column], errors="coerce"),
            "article_count": pd.to_numeric(
                panel[spec.article_count_column], errors="coerce"
            ),
        }
    )
    if (frame[["scfi_level", "brent_level"]] <= 0).any().any():
        raise ValueError(f"{spec.frequency} contains non-positive SCFI or Brent levels")
    if (frame["article_count"].dropna() < 0).any():
        raise ValueError(f"{spec.frequency} contains negative article counts")

    frame["scfi_log"] = np.log(frame["scfi_level"])
    frame["brent_log"] = np.log(frame["brent_level"])
    frame["log_news_volume"] = np.log1p(frame["article_count"])
    frame["scfi_return"] = frame["scfi_log"].diff()
    frame["brent_return"] = frame["brent_log"].diff()
    frame["msdi_change"] = frame["msdi_level"].diff()
    frame["news_volume_change"] = frame["log_news_volume"].diff()
    for column in EXTERNAL_CONTROL_COLUMNS:
        if column not in panel.columns:
            raise ValueError(f"{spec.frequency} panel lacks required released external control: {column}")
        frame[f"{column}_lag_1"] = pd.to_numeric(panel[column], errors="coerce").shift(1)

    if spec.frequency == "weekly":
        cycle = frame["date"].dt.isocalendar().week.astype(float)
        period = 52.1775
    else:
        cycle = frame["date"].dt.month.astype(float)
        period = 12.0
    frame["season_sin"] = np.sin(2 * np.pi * cycle / period)
    frame["season_cos"] = np.cos(2 * np.pi * cycle / period)
    return frame


def select_longest_complete_block(
    frame: pd.DataFrame,
    columns: list[str],
    frequency: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = frame.dropna(subset=["date", *columns]).copy()
    if work.empty:
        raise ValueError(f"No complete {frequency} observations for {columns}")
    if frequency == "weekly":
        ordinal = work["date"].dt.to_period("W-SUN").astype("int64")
    else:
        ordinal = work["date"].dt.to_period("M").astype("int64")
    new_block = ordinal.diff().fillna(1).ne(1)
    work["_block"] = new_block.cumsum()
    sizes = work.groupby("_block").size()
    selected_block = sizes.idxmax()
    selected = work.loc[work["_block"].eq(selected_block)].drop(columns="_block")
    audit = {
        "complete_rows_before_contiguity_filter": int(len(work)),
        "selected_rows": int(len(selected)),
        "excluded_noncontiguous_rows": int(len(work) - len(selected)),
        "block_count": int(len(sizes)),
        "start_date": selected["date"].min(),
        "end_date": selected["date"].max(),
    }
    return selected.reset_index(drop=True), audit


def stationarity_test(
    series: pd.Series,
    frequency: str,
    path_name: str,
    variable: str,
    transformation: str,
    alpha: float,
) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    row: dict[str, Any] = {
        "frequency": frequency,
        "msdi_path": path_name,
        "variable": variable,
        "transformation": transformation,
        "n_observations": int(len(values)),
    }
    warning_messages: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adf = adfuller(values, regression="c", autolag="AIC")
        warning_messages.extend(str(item.message) for item in caught)
        row.update(
            {
                "adf_statistic": float(adf[0]),
                "adf_p_value": float(adf[1]),
                "adf_used_lag": int(adf[2]),
                "adf_n_observations": int(adf[3]),
                "adf_critical_5pct": float(adf[4]["5%"]),
            }
        )
    except Exception as error:  # pragma: no cover - defensive for degenerate data
        row["adf_error"] = f"{type(error).__name__}: {error}"

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stationarity = kpss(values, regression="c", nlags="auto")
        warning_messages.extend(str(item.message) for item in caught)
        row.update(
            {
                "kpss_statistic": float(stationarity[0]),
                "kpss_p_value": float(stationarity[1]),
                "kpss_used_lag": int(stationarity[2]),
                "kpss_critical_5pct": float(stationarity[3]["5%"]),
            }
        )
    except Exception as error:  # pragma: no cover - defensive for degenerate data
        row["kpss_error"] = f"{type(error).__name__}: {error}"

    adf_p = row.get("adf_p_value", np.nan)
    kpss_p = row.get("kpss_p_value", np.nan)
    if np.isfinite(adf_p) and np.isfinite(kpss_p):
        if adf_p < alpha and kpss_p >= alpha:
            decision = "stationary_by_both"
        elif adf_p >= alpha and kpss_p < alpha:
            decision = "nonstationary_by_both"
        else:
            decision = "mixed_test_evidence"
    else:
        decision = "test_failure"
    row["joint_decision"] = decision
    row["warnings"] = " | ".join(dict.fromkeys(warning_messages))
    return row


def run_stationarity_tests(
    frame: pd.DataFrame,
    frequency: str,
    path_name: str,
    alpha: float,
) -> pd.DataFrame:
    definitions = [
        ("scfi_log", "log_level"),
        ("scfi_return", "first_difference_of_log"),
        ("brent_log", "log_level"),
        ("brent_return", "first_difference_of_log"),
        ("msdi_level", "media_corrected_level"),
        ("msdi_change", "first_difference"),
        ("log_news_volume", "log1p_level"),
        ("news_volume_change", "first_difference_of_log1p"),
    ]
    return pd.DataFrame(
        [
            stationarity_test(
                frame[variable], frequency, path_name, variable, transformation, alpha
            )
            for variable, transformation in definitions
        ]
    )


def run_cointegration_tests(
    frame: pd.DataFrame,
    frequency: str,
    path_name: str,
) -> pd.DataFrame:
    level_columns = ["scfi_log", "msdi_level", "brent_log", "log_news_volume"]
    complete, _ = select_longest_complete_block(frame, level_columns, frequency)
    rows: list[dict[str, Any]] = []
    try:
        johansen = coint_johansen(complete[level_columns], det_order=0, k_ar_diff=1)
        for rank in range(len(level_columns)):
            rows.append(
                {
                    "frequency": frequency,
                    "msdi_path": path_name,
                    "test": "johansen_trace",
                    "variables": "|".join(level_columns),
                    "null_hypothesis": f"cointegration_rank_at_most_{rank}",
                    "statistic": float(johansen.lr1[rank]),
                    "critical_value_90pct": float(johansen.cvt[rank, 0]),
                    "critical_value_95pct": float(johansen.cvt[rank, 1]),
                    "critical_value_99pct": float(johansen.cvt[rank, 2]),
                    "reject_at_5pct": bool(johansen.lr1[rank] > johansen.cvt[rank, 1]),
                    "n_observations": int(len(complete)),
                    "deterministic_terms": "constant",
                    "k_ar_diff": 1,
                }
            )
    except Exception as error:
        rows.append(
            {
                "frequency": frequency,
                "msdi_path": path_name,
                "test": "johansen_trace",
                "variables": "|".join(level_columns),
                "error": f"{type(error).__name__}: {error}",
                "n_observations": int(len(complete)),
            }
        )

    for comparison in ("msdi_level", "brent_log"):
        pair = complete[["scfi_log", comparison]].dropna()
        try:
            statistic, p_value, critical = coint(
                pair["scfi_log"], pair[comparison], trend="ct", autolag="aic"
            )
            rows.append(
                {
                    "frequency": frequency,
                    "msdi_path": path_name,
                    "test": "engle_granger",
                    "variables": f"scfi_log|{comparison}",
                    "null_hypothesis": "no_cointegration",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "critical_value_90pct": float(critical[2]),
                    "critical_value_95pct": float(critical[1]),
                    "critical_value_99pct": float(critical[0]),
                    "reject_at_5pct": bool(p_value < 0.05),
                    "n_observations": int(len(pair)),
                    "deterministic_terms": "constant_and_trend",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "frequency": frequency,
                    "msdi_path": path_name,
                    "test": "engle_granger",
                    "variables": f"scfi_log|{comparison}",
                    "error": f"{type(error).__name__}: {error}",
                    "n_observations": int(len(pair)),
                }
            )
    return pd.DataFrame(rows)


def fit_varx(
    frame: pd.DataFrame,
    maximum_lag: int,
    alpha: float = 0.05,
) -> tuple[Any, pd.DataFrame]:
    endog = frame[list(ENDOGENOUS_VARIABLES)].astype(float)
    exog = frame[exogenous_controls(frame)].astype(float)
    model = VAR(endog=endog, exog=exog)
    rows: list[dict[str, Any]] = []
    fitted: dict[int, Any] = {}
    for lag in range(1, maximum_lag + 1):
        try:
            result = model.fit(lag, trend="c")
            fitted[lag] = result
            diagnostic_horizon = maximum_lag + 6
            whiteness = result.test_whiteness(
                nlags=diagnostic_horizon, adjusted=True, signif=alpha
            )
            rows.append(
                {
                    "lag": lag,
                    "aic": float(result.aic),
                    "bic": float(result.bic),
                    "hqic": float(result.hqic),
                    "fpe": float(result.fpe),
                    "n_observations": int(result.nobs),
                    "n_equations": int(result.neqs),
                    "parameters_per_equation": int(result.df_model),
                    "residual_whiteness_horizon": diagnostic_horizon,
                    "residual_whiteness_statistic": float(
                        whiteness.test_statistic
                    ),
                    "residual_whiteness_p_value": float(whiteness.pvalue),
                    "passes_residual_whiteness": bool(whiteness.pvalue >= alpha),
                    "fit_status": "success",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "lag": lag,
                    "fit_status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    if not fitted:
        raise RuntimeError("Every candidate VARX lag specification failed")
    successful = [row for row in rows if row["fit_status"] == "success"]
    bic_lag = int(min(successful, key=lambda row: row["bic"])["lag"])
    diagnostic_candidates = [
        row for row in successful if row["passes_residual_whiteness"]
    ]
    if diagnostic_candidates:
        selected_lag = int(
            min(diagnostic_candidates, key=lambda row: row["bic"])["lag"]
        )
        selection_reason = "minimum_bic_among_residual_white_candidates"
    else:
        selected_lag = bic_lag
        selection_reason = "bic_fallback_no_candidate_passed_whiteness"
    lag_table = pd.DataFrame(rows)
    lag_table["selected_by_bic"] = lag_table["lag"].eq(bic_lag)
    lag_table["selected_for_analysis"] = lag_table["lag"].eq(selected_lag)
    lag_table["selection_reason"] = selection_reason
    return fitted[selected_lag], lag_table


def extract_var_coefficients(
    result: Any,
    frequency: str,
    path_name: str,
) -> pd.DataFrame:
    # Statsmodels 0.14.6 may expose stale model-level xnames after fitting
    # multiple lag candidates with exogenous regressors. The underlying result
    # contains the correctly aligned names and arrays, so use those explicitly.
    raw_result = getattr(result, "_results", result)
    terms = list(raw_result.exog_names)
    equations = list(raw_result.names)
    parameters = np.asarray(raw_result.params)
    standard_errors = np.asarray(raw_result.stderr)
    t_values = np.asarray(raw_result.tvalues)
    p_values = np.asarray(raw_result.pvalues)
    if parameters.shape != (len(terms), len(equations)):
        raise AssertionError(
            "VAR coefficient labels do not match the fitted parameter matrix: "
            f"shape={parameters.shape}, terms={len(terms)}, equations={len(equations)}"
        )
    rows: list[dict[str, Any]] = []
    for term_index, term in enumerate(terms):
        for equation_index, equation in enumerate(equations):
            rows.append(
                {
                    "frequency": frequency,
                    "msdi_path": path_name,
                    "selected_lag": int(result.k_ar),
                    "equation": equation,
                    "term": term,
                    "coefficient": float(parameters[term_index, equation_index]),
                    "standard_error": float(
                        standard_errors[term_index, equation_index]
                    ),
                    "t_statistic": float(t_values[term_index, equation_index]),
                    "p_value": float(p_values[term_index, equation_index]),
                    "n_observations": int(result.nobs),
                }
            )
    return pd.DataFrame(rows)


def conditional_granger_tests(
    result: Any,
    frequency: str,
    path_name: str,
    alpha: float,
    diagnostics_pass: bool,
) -> pd.DataFrame:
    directions = [
        ("msdi_change", "scfi_return"),
        ("scfi_return", "msdi_change"),
    ]
    rows: list[dict[str, Any]] = []
    for causing, caused in directions:
        test = result.test_causality(caused=caused, causing=[causing], kind="f", signif=alpha)
        rows.append(
            {
                "frequency": frequency,
                "msdi_path": path_name,
                "model": "conditional_varx",
                "causing": causing,
                "caused": caused,
                "selected_lag": int(result.k_ar),
                "test_statistic": float(test.test_statistic),
                "test_distribution": "F",
                "p_value": float(test.pvalue),
                "degrees_freedom": str(test.df),
                "reject_at_alpha": bool(test.pvalue < alpha),
                "alpha": alpha,
                "diagnostics_pass": diagnostics_pass,
                "exploratory_only": not diagnostics_pass,
                "diagnostic_note": (
                    "selected VAR passed residual-whiteness gate"
                    if diagnostics_pass
                    else "selected VAR failed residual-whiteness gate; ordinary F test is exploratory"
                ),
                "interpretation": "predictive_precedence_not_structural_causality",
            }
        )
    return pd.DataFrame(rows)


def hac_conditional_granger_tests(
    frame: pd.DataFrame,
    frequency: str,
    path_name: str,
    maximum_lag: int,
    alpha: float,
) -> pd.DataFrame:
    """Test all causing-variable lags jointly with HAC equation covariance."""
    design = pd.DataFrame(index=frame.index)
    for variable in ENDOGENOUS_VARIABLES:
        for lag in range(1, maximum_lag + 1):
            design[f"{variable}_lag_{lag}"] = frame[variable].shift(lag)
    for control in exogenous_controls(frame):
        design[control] = frame[control]
    directions = [
        ("msdi_change", "scfi_return"),
        ("scfi_return", "msdi_change"),
    ]
    rows: list[dict[str, Any]] = []
    for causing, caused in directions:
        model_frame = pd.concat(
            [frame[caused].rename("outcome"), design], axis=1
        ).dropna()
        regressors = sm.add_constant(
            model_frame.drop(columns="outcome"), has_constant="add"
        )
        fitted = sm.OLS(model_frame["outcome"], regressors, missing="raise").fit(
            cov_type="HAC",
            cov_kwds={"maxlags": maximum_lag, "use_correction": True},
        )
        tested_terms = [f"{causing}_lag_{lag}" for lag in range(1, maximum_lag + 1)]
        restrictions = np.zeros((len(tested_terms), len(regressors.columns)))
        for row_index, term in enumerate(tested_terms):
            restrictions[row_index, regressors.columns.get_loc(term)] = 1.0
        test = fitted.wald_test(restrictions, use_f=False, scalar=True)
        statistic = float(np.asarray(test.statistic).squeeze())
        p_value = float(np.asarray(test.pvalue).squeeze())
        rows.append(
            {
                "frequency": frequency,
                "msdi_path": path_name,
                "model": "conditional_ols_hac_prespecified_max_lag",
                "causing": causing,
                "caused": caused,
                "selected_lag": maximum_lag,
                "test_statistic": statistic,
                "test_distribution": "chi_square",
                "p_value": p_value,
                "degrees_freedom": len(tested_terms),
                "reject_at_alpha": bool(p_value < alpha),
                "alpha": alpha,
                "n_observations": int(fitted.nobs),
                "covariance": f"HAC_maxlags_{maximum_lag}",
                "diagnostics_pass": np.nan,
                "exploratory_only": False,
                "diagnostic_note": (
                    "dependence-robust equation test at pre-specified maximum lag"
                ),
                "interpretation": "dependence_robust_predictive_precedence_not_structural_causality",
            }
        )
    return pd.DataFrame(rows)


def conventional_granger_tests(
    frame: pd.DataFrame,
    frequency: str,
    path_name: str,
    maximum_lag: int,
) -> pd.DataFrame:
    directions = [
        ("msdi_change", "scfi_return"),
        ("scfi_return", "msdi_change"),
    ]
    rows: list[dict[str, Any]] = []
    for causing, caused in directions:
        pair = frame[[caused, causing]].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            results = grangercausalitytests(pair, maxlag=maximum_lag, verbose=False)
        for lag, test_bundle in results.items():
            statistic, p_value, denominator_df, numerator_df = test_bundle[0]["ssr_ftest"]
            rows.append(
                {
                    "frequency": frequency,
                    "msdi_path": path_name,
                    "model": "bivariate_ols",
                    "causing": causing,
                    "caused": caused,
                    "lag": int(lag),
                    "f_statistic": float(statistic),
                    "p_value": float(p_value),
                    "denominator_df": float(denominator_df),
                    "numerator_df": float(numerator_df),
                    "n_observations_before_lagging": int(len(pair)),
                    "sample_start_date": frame["date"].min(),
                    "sample_end_date": frame["date"].max(),
                    "family_id": (
                        f"conventional_granger__{frequency}__{path_name}__"
                        f"{causing}_to_{caused}__candidate_lags"
                    ),
                    "interpretation": "predictive_precedence_not_structural_causality",
                }
            )
    table = pd.DataFrame(rows)
    return add_grouped_pvalue_adjustments(
        table,
        ["frequency", "msdi_path", "causing", "caused"],
        "p_value",
    )


def add_grouped_pvalue_adjustments(
    frame: pd.DataFrame,
    group_columns: list[str],
    p_column: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["p_value_holm"] = np.nan
    result["p_value_fdr_bh"] = np.nan
    if result.empty:
        return result
    group_key: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else group_columns
    )
    for _, index in result.groupby(group_key, dropna=False).groups.items():
        index = list(index)
        valid_index = [i for i in index if pd.notna(result.at[i, p_column])]
        if not valid_index:
            continue
        p_values = result.loc[valid_index, p_column].astype(float).to_numpy()
        result.loc[valid_index, "p_value_holm"] = multipletests(
            p_values, method="holm"
        )[1]
        result.loc[valid_index, "p_value_fdr_bh"] = multipletests(
            p_values, method="fdr_bh"
        )[1]
    return result


def var_diagnostics(
    result: Any,
    frequency: str,
    path_name: str,
    alpha: float,
    maximum_lag: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root_moduli = np.abs(result.roots)
    rows.extend(
        [
            {
                "frequency": frequency,
                "msdi_path": path_name,
                "diagnostic": "var_stability",
                "variable": "system",
                "statistic": float(np.min(root_moduli)),
                "p_value": np.nan,
                "passes_at_alpha": bool(result.is_stable(verbose=False)),
                "details": "statistic is minimum inverse-root modulus; stable when > 1",
            },
            {
                "frequency": frequency,
                "msdi_path": path_name,
                "diagnostic": "var_max_inverse_root_modulus",
                "variable": "system",
                "statistic": float(np.max(root_moduli)),
                "p_value": np.nan,
                "passes_at_alpha": bool(result.is_stable(verbose=False)),
                "details": "reported for root audit",
            },
        ]
    )
    whiteness_lag = (
        maximum_lag + 6
        if maximum_lag is not None
        else max(int(result.k_ar) + 1, min(int(result.k_ar) + 6, 24))
    )
    try:
        test = result.test_whiteness(nlags=whiteness_lag, signif=alpha, adjusted=True)
        rows.append(
            {
                "frequency": frequency,
                "msdi_path": path_name,
                "diagnostic": "multivariate_portmanteau_whiteness",
                "variable": "system",
                "statistic": float(test.test_statistic),
                "p_value": float(test.pvalue),
                "degrees_freedom": test.df,
                "passes_at_alpha": bool(test.pvalue >= alpha),
                "details": f"adjusted test through lag {whiteness_lag}",
            }
        )
    except Exception as error:
        rows.append(
            {
                "frequency": frequency,
                "msdi_path": path_name,
                "diagnostic": "multivariate_portmanteau_whiteness",
                "variable": "system",
                "details": f"failed: {type(error).__name__}: {error}",
            }
        )
    try:
        test = result.test_normality(signif=alpha)
        rows.append(
            {
                "frequency": frequency,
                "msdi_path": path_name,
                "diagnostic": "multivariate_jarque_bera",
                "variable": "system",
                "statistic": float(test.test_statistic),
                "p_value": float(test.pvalue),
                "degrees_freedom": test.df,
                "passes_at_alpha": bool(test.pvalue >= alpha),
                "details": "normality is diagnostic, not required for HAC inference",
            }
        )
    except Exception as error:
        rows.append(
            {
                "frequency": frequency,
                "msdi_path": path_name,
                "diagnostic": "multivariate_jarque_bera",
                "variable": "system",
                "details": f"failed: {type(error).__name__}: {error}",
            }
        )

    residuals = result.resid
    univariate_lag = max(1, min(12, int(result.nobs // 5)))
    for variable in residuals.columns:
        residual = residuals[variable].dropna()
        try:
            ljung = acorr_ljungbox(
                residual,
                lags=[univariate_lag],
                model_df=int(result.k_ar),
                return_df=True,
            ).iloc[0]
            rows.append(
                {
                    "frequency": frequency,
                    "msdi_path": path_name,
                    "diagnostic": "ljung_box_autocorrelation",
                    "variable": variable,
                    "statistic": float(ljung["lb_stat"]),
                    "p_value": float(ljung["lb_pvalue"]),
                    "passes_at_alpha": bool(ljung["lb_pvalue"] >= alpha),
                    "details": f"lag={univariate_lag}; model_df={result.k_ar}",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "frequency": frequency,
                    "msdi_path": path_name,
                    "diagnostic": "ljung_box_autocorrelation",
                    "variable": variable,
                    "details": f"failed: {type(error).__name__}: {error}",
                }
            )
        try:
            arch_lags = max(1, min(12, len(residual) // 10))
            lm_stat, lm_p, f_stat, f_p = het_arch(residual, nlags=arch_lags)
            rows.append(
                {
                    "frequency": frequency,
                    "msdi_path": path_name,
                    "diagnostic": "arch_lm_heteroskedasticity",
                    "variable": variable,
                    "statistic": float(lm_stat),
                    "p_value": float(lm_p),
                    "secondary_statistic": float(f_stat),
                    "secondary_p_value": float(f_p),
                    "passes_at_alpha": bool(lm_p >= alpha),
                    "details": f"lag={arch_lags}",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "frequency": frequency,
                    "msdi_path": path_name,
                    "diagnostic": "arch_lm_heteroskedasticity",
                    "variable": variable,
                    "details": f"failed: {type(error).__name__}: {error}",
                }
            )
    return pd.DataFrame(rows)


def impulse_responses(
    result: Any,
    frame: pd.DataFrame,
    frequency: str,
    path_name: str,
    horizon: int,
    monte_carlo_repetitions: int,
    alpha: float,
    seed: int,
    diagnostics_pass: bool,
) -> pd.DataFrame:
    analysis = result.irf(horizon)
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    cumulative_lower: np.ndarray | None = None
    cumulative_upper: np.ndarray | None = None
    error = ""
    try:
        lower, upper = analysis.errband_mc(
            orth=False,
            repl=monte_carlo_repetitions,
            signif=alpha,
            seed=seed,
            burn=max(100, result.k_ar * 20),
        )
        cumulative_lower, cumulative_upper = analysis.cum_errband_mc(
            orth=False,
            repl=monte_carlo_repetitions,
            signif=alpha,
            seed=seed,
            burn=max(100, result.k_ar * 20),
        )
    except Exception as exception:  # retain point estimates if MC bands fail
        error = f"{type(exception).__name__}: {exception}"

    names = list(result.names)
    directions = [
        ("msdi_change", "scfi_return"),
        ("scfi_return", "msdi_change"),
    ]
    rows: list[dict[str, Any]] = []
    raw_result = getattr(result, "_results", result)
    residual_covariance = np.asarray(raw_result.sigma_u)
    for impulse_name, response_name in directions:
        impulse_index = names.index(impulse_name)
        response_index = names.index(response_name)
        shock_scale = float(
            np.sqrt(residual_covariance[impulse_index, impulse_index])
        )
        point_values = analysis.irfs[:, response_index, impulse_index] * shock_scale
        low_values = (
            lower[:, response_index, impulse_index] * shock_scale
            if lower is not None
            else np.repeat(np.nan, horizon + 1)
        )
        high_values = (
            upper[:, response_index, impulse_index] * shock_scale
            if upper is not None
            else np.repeat(np.nan, horizon + 1)
        )
        cumulative_point = (
            analysis.cum_effects[:, response_index, impulse_index] * shock_scale
        )
        cumulative_low = (
            cumulative_lower[:, response_index, impulse_index] * shock_scale
            if cumulative_lower is not None
            else np.repeat(np.nan, horizon + 1)
        )
        cumulative_high = (
            cumulative_upper[:, response_index, impulse_index] * shock_scale
            if cumulative_upper is not None
            else np.repeat(np.nan, horizon + 1)
        )
        for step in range(horizon + 1):
            is_scfi_response = response_name == "scfi_return"
            rows.append(
                {
                    "frequency": frequency,
                    "msdi_path": path_name,
                    "selected_lag": int(result.k_ar),
                    "impulse": impulse_name,
                    "response": response_name,
                    "horizon": step,
                    "shock_scale": shock_scale,
                    "shock_definition": "one_VAR_residual_standard_deviation_reduced_form_innovation",
                    "response_change": float(point_values[step]),
                    "lower_pointwise": float(low_values[step]),
                    "upper_pointwise": float(high_values[step]),
                    "cumulative_response": float(cumulative_point[step]),
                    "cumulative_lower_pointwise": float(cumulative_low[step]),
                    "cumulative_upper_pointwise": float(cumulative_high[step]),
                    "cumulative_scfi_percent": (
                        float(100 * np.expm1(cumulative_point[step]))
                        if is_scfi_response
                        else np.nan
                    ),
                    "monte_carlo_repetitions": monte_carlo_repetitions,
                    "interval_method": "statsmodels_VAR_Monte_Carlo_pointwise",
                    "confidence_level": 1 - alpha,
                    "band_error": error,
                    "identification": "reduced_form_not_structural",
                    "sample_start_date": frame["date"].min(),
                    "sample_end_date": frame["date"].max(),
                    "diagnostics_pass": diagnostics_pass,
                    "exploratory_only": not diagnostics_pass,
                    "diagnostic_note": (
                        "selected VAR passed residual-whiteness gate"
                        if diagnostics_pass
                        else "selected VAR failed residual-whiteness gate; do not use as main inference"
                    ),
                }
            )
    return pd.DataFrame(rows)


def forecast_error_variance_decomposition(
    result: Any,
    frequency: str,
    path_name: str,
    periods: int,
    diagnostics_pass: bool,
) -> pd.DataFrame:
    decomposition = result.fevd(periods)
    values = np.asarray(decomposition.decomp)
    names = list(result.names)
    if values.shape != (len(names), periods, len(names)):
        raise AssertionError(
            f"Unexpected FEVD shape {values.shape}; expected {(len(names), periods, len(names))}"
        )
    rows: list[dict[str, Any]] = []
    for response_index, response in enumerate(names):
        for horizon_index in range(periods):
            for shock_index, shock in enumerate(names):
                rows.append(
                    {
                        "frequency": frequency,
                        "msdi_path": path_name,
                        "selected_lag": int(result.k_ar),
                        "forecast_horizon": horizon_index + 1,
                        "response": response,
                        "shock": shock,
                        "variance_share": float(
                            values[response_index, horizon_index, shock_index]
                        ),
                        "identification": "Cholesky_order_dependent_descriptive_FEVD",
                        "variable_order": "|".join(names),
                        "diagnostics_pass": diagnostics_pass,
                        "exploratory_only": not diagnostics_pass,
                        "diagnostic_note": (
                            "selected VAR passed residual-whiteness gate"
                            if diagnostics_pass
                            else "selected VAR failed residual-whiteness gate; do not use as main inference"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def residualize_seasonality(frame: pd.DataFrame) -> pd.DataFrame:
    controls = sm.add_constant(frame[list(SEASONAL_CONTROLS)], has_constant="add")
    result = pd.DataFrame(index=frame.index)
    for variable in ENDOGENOUS_VARIABLES:
        fitted = sm.OLS(frame[variable], controls, missing="raise").fit()
        residual = fitted.resid
        scale = float(residual.std(ddof=1))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Cannot standardize PCMCI variable {variable}")
        result[variable] = (residual - residual.mean()) / scale
    return result


def pcmci_links(
    frame: pd.DataFrame,
    frequency: str,
    path_name: str,
    maximum_lag: int,
    pc_alpha: float,
    alpha: float,
) -> pd.DataFrame:
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ImportError as error:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PCMCI requires Tigramite. Install with: "
            ".venv\\Scripts\\python.exe -m pip install tigramite==5.2.10.1"
        ) from error

    standardized = residualize_seasonality(frame)
    variable_names = list(standardized.columns)
    tigramite_frame = pp.DataFrame(
        standardized.to_numpy(dtype=float), var_names=variable_names
    )
    parcorr = ParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=tigramite_frame, cond_ind_test=parcorr, verbosity=0)
    result = pcmci.run_pcmci(
        tau_min=1,
        tau_max=maximum_lag,
        pc_alpha=pc_alpha,
        alpha_level=alpha,
        fdr_method="none",
    )
    p_matrix = result["p_matrix"]
    adjusted = pcmci.get_corrected_pvalues(
        p_matrix=p_matrix,
        fdr_method="fdr_bh",
        exclude_contemporaneous=True,
        tau_min=1,
        tau_max=maximum_lag,
    )
    fdr_graph = pcmci.get_graph_from_pmatrix(
        p_matrix=adjusted,
        alpha_level=alpha,
        tau_min=1,
        tau_max=maximum_lag,
    )
    value_matrix = result["val_matrix"]
    graph = result.get("graph")
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(variable_names):
        for target_index, target in enumerate(variable_names):
            for lag in range(1, maximum_lag + 1):
                raw_p = float(p_matrix[source_index, target_index, lag])
                adjusted_p = float(adjusted[source_index, target_index, lag])
                rows.append(
                    {
                        "frequency": frequency,
                        "msdi_path": path_name,
                        "source": source,
                        "target": target,
                        "source_lag": lag,
                        "partial_correlation": float(
                            value_matrix[source_index, target_index, lag]
                        ),
                        "p_value": raw_p,
                        "p_value_fdr_bh": adjusted_p,
                        "significant_fdr": bool(adjusted_p < alpha),
                        "graph_mark_raw_alpha": (
                            str(graph[source_index, target_index, lag])
                            if graph is not None
                            else ""
                        ),
                        "graph_mark_fdr": str(
                            fdr_graph[source_index, target_index, lag]
                        ),
                        "pc_alpha": pc_alpha,
                        "alpha": alpha,
                        "conditional_independence_test": "ParCorr_analytic",
                        "preprocessing": "first_differences_then_seasonal_residualization_and_zscore",
                        "n_observations": int(len(standardized)),
                        "sample_start_date": frame["date"].min(),
                        "sample_end_date": frame["date"].max(),
                        "family_id": f"pcmci__{frequency}__{path_name}__all_lagged_links",
                        "interpretation": "conditional_temporal_dependency_not_structural_causality",
                    }
                )
    return pd.DataFrame(rows)


def create_msdi_innovation(
    frame: pd.DataFrame,
    selected_lag: int,
    frequency: str,
    path_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    design = pd.DataFrame(index=frame.index)
    for variable in ENDOGENOUS_VARIABLES:
        for lag in range(1, selected_lag + 1):
            design[f"{variable}_lag_{lag}"] = frame[variable].shift(lag)
    for control in SEASONAL_CONTROLS:
        design[control] = frame[control]
    design = sm.add_constant(design, has_constant="add")
    model_frame = pd.concat(
        [frame[["date", "msdi_change"]], design], axis=1
    ).dropna()
    target = model_frame.pop("msdi_change")
    dates = model_frame.pop("date")
    fitted = sm.OLS(target, model_frame, missing="raise").fit()
    residual = fitted.resid
    residual_sd = float(residual.std(ddof=1))
    if residual_sd <= 0 or not np.isfinite(residual_sd):
        raise ValueError("MSDI innovation residual has invalid standard deviation")
    output = pd.DataFrame(
        {
            "date": dates,
            "frequency": frequency,
            "msdi_path": path_name,
            "msdi_change": target,
            "predicted_msdi_change": fitted.fittedvalues,
            "msdi_innovation": residual,
            "msdi_innovation_std": residual / residual_sd,
        }
    ).reset_index(drop=True)
    audit = {
        "frequency": frequency,
        "msdi_path": path_name,
        "selected_lag": selected_lag,
        "n_observations": int(fitted.nobs),
        "r_squared": float(fitted.rsquared),
        "adjusted_r_squared": float(fitted.rsquared_adj),
        "residual_standard_deviation": residual_sd,
        "first_date": output["date"].min(),
        "last_date": output["date"].max(),
        "regressors": "|".join(model_frame.columns),
        "interpretation": "innovation_is_not_assumed_exogenous",
    }
    return output, audit


def local_projections(
    frame: pd.DataFrame,
    innovations: pd.DataFrame,
    selected_lag: int,
    frequency: str,
    path_name: str,
    maximum_horizon: int,
    alpha: float,
) -> pd.DataFrame:
    work = frame.merge(
        innovations[["date", "msdi_innovation_std"]], on="date", how="left", validate="one_to_one"
    )
    base_design = pd.DataFrame(index=work.index)
    base_design["msdi_innovation_std"] = work["msdi_innovation_std"]
    for variable in ENDOGENOUS_VARIABLES:
        for lag in range(1, selected_lag + 1):
            base_design[f"{variable}_lag_{lag}"] = work[variable].shift(lag)
    for control in SEASONAL_CONTROLS:
        base_design[control] = work[control]

    rows: list[dict[str, Any]] = []
    for horizon in range(0, maximum_horizon + 1):
        # Impact-inclusive cumulative response: h=0 is the t return and h>0
        # cumulates from t through t+h. This aligns the LP horizon convention
        # with the VAR impulse-response horizon convention.
        outcome = work["scfi_log"].shift(-horizon) - work["scfi_log"].shift(1)
        model_frame = pd.concat(
            [work["date"], outcome.rename("outcome"), base_design], axis=1
        ).dropna()
        model_dates = model_frame.pop("date")
        design = sm.add_constant(model_frame.drop(columns="outcome"), has_constant="add")
        hac_lags = max(1, horizon, selected_lag)
        fitted = sm.OLS(model_frame["outcome"], design, missing="raise").fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_lags, "use_correction": True}
        )
        coefficient = float(fitted.params["msdi_innovation_std"])
        standard_error = float(fitted.bse["msdi_innovation_std"])
        p_value = float(fitted.pvalues["msdi_innovation_std"])
        critical = float(norm.ppf(1 - alpha / 2))
        rows.append(
            {
                "frequency": frequency,
                "msdi_path": path_name,
                "horizon": horizon,
                "outcome": "impact_inclusive_cumulative_scfi_log_change",
                "outcome_window": "log_scfi_t_plus_h_minus_log_scfi_t_minus_1",
                "treatment": "one_sd_innovation_in_msdi_change",
                "coefficient_log_points": coefficient,
                "effect_percent": float(100 * np.expm1(coefficient)),
                "standard_error_hac": standard_error,
                "z_statistic": coefficient / standard_error,
                "p_value": p_value,
                "ci_lower_log_points": coefficient - critical * standard_error,
                "ci_upper_log_points": coefficient + critical * standard_error,
                "ci_lower_percent": float(
                    100 * np.expm1(coefficient - critical * standard_error)
                ),
                "ci_upper_percent": float(
                    100 * np.expm1(coefficient + critical * standard_error)
                ),
                "n_observations": int(fitted.nobs),
                "sample_start_date": model_dates.min(),
                "sample_end_date": model_dates.max(),
                "r_squared": float(fitted.rsquared),
                "selected_lag": selected_lag,
                "hac_max_lags": hac_lags,
                "confidence_level": 1 - alpha,
                "pointwise_interval": True,
                "interpretation": "orthogonalized_dynamic_association_not_structural_causality",
            }
        )
    return pd.DataFrame(rows)


def moving_block_indices(
    n_observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not 1 <= block_length <= n_observations:
        raise ValueError("block_length must lie between one and sample size")
    blocks_needed = int(np.ceil(n_observations / block_length))
    maximum_start = n_observations - block_length
    starts = rng.integers(0, maximum_start + 1, size=blocks_needed)
    indices = np.concatenate(
        [np.arange(start, start + block_length) for start in starts]
    )
    return indices[:n_observations]


def bootstrap_local_projection_bands(
    frame: pd.DataFrame,
    point_estimates: pd.DataFrame,
    selected_lag: int,
    frequency: str,
    path_name: str,
    maximum_horizon: int,
    alpha: float,
    repetitions: int,
    requested_block_length: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add moving-block-bootstrap percentile and simultaneous sup-t bands.

    Stationary system rows are resampled in contiguous blocks. Each draw
    reconstructs the SCFI log path and re-estimates both the MSDI innovation
    model and every local projection, propagating first-stage uncertainty.
    """
    n_observations = len(frame)
    block_length = (
        requested_block_length
        if requested_block_length > 0
        else int(np.ceil(n_observations ** (1 / 3)))
    )
    rng = np.random.default_rng(seed)
    expected_horizons = np.arange(0, maximum_horizon + 1)
    point = point_estimates.sort_values("horizon").reset_index(drop=True)
    if not np.array_equal(point["horizon"].to_numpy(), expected_horizons):
        raise AssertionError("Point estimates do not cover every requested LP horizon")
    bootstrap_coefficients: list[np.ndarray] = []
    bootstrap_standard_errors: list[np.ndarray] = []
    failed_draws: list[str] = []
    original_dates = frame["date"].reset_index(drop=True)

    for repetition in range(repetitions):
        try:
            indices = moving_block_indices(n_observations, block_length, rng)
            sampled = frame.iloc[indices].copy().reset_index(drop=True)
            sampled["date"] = original_dates
            sampled["scfi_log"] = sampled["scfi_return"].cumsum()
            innovations, _ = create_msdi_innovation(
                sampled, selected_lag, frequency, path_name
            )
            estimates = local_projections(
                sampled,
                innovations,
                selected_lag,
                frequency,
                path_name,
                maximum_horizon,
                alpha,
            ).sort_values("horizon")
            if not np.array_equal(
                estimates["horizon"].to_numpy(), expected_horizons
            ):
                raise AssertionError("Bootstrap draw omitted an LP horizon")
            bootstrap_coefficients.append(
                estimates["coefficient_log_points"].to_numpy(dtype=float)
            )
            bootstrap_standard_errors.append(
                estimates["standard_error_hac"].to_numpy(dtype=float)
            )
        except Exception as error:  # retain an auditable count of failed draws
            failed_draws.append(
                f"draw={repetition}:{type(error).__name__}:{error}"
            )

    minimum_successes = max(20, int(np.ceil(0.8 * repetitions)))
    if len(bootstrap_coefficients) < minimum_successes:
        raise RuntimeError(
            "Too few successful local-projection bootstrap draws: "
            f"{len(bootstrap_coefficients)}/{repetitions}; first failures={failed_draws[:3]}"
        )
    coefficient_draws = np.vstack(bootstrap_coefficients)
    standard_error_draws = np.vstack(bootstrap_standard_errors)
    point_coefficients = point["coefficient_log_points"].to_numpy(dtype=float)
    valid_standard_errors = np.where(
        standard_error_draws > np.finfo(float).eps,
        standard_error_draws,
        np.nan,
    )
    studentized = np.abs(
        (coefficient_draws - point_coefficients[None, :]) / valid_standard_errors
    )
    sup_t = np.nanmax(studentized, axis=1)
    sup_t_critical = float(np.nanquantile(sup_t, 1 - alpha))
    point_standard_errors = point["standard_error_hac"].to_numpy(dtype=float)
    joint_lower = point_coefficients - sup_t_critical * point_standard_errors
    joint_upper = point_coefficients + sup_t_critical * point_standard_errors
    percentile_lower = np.nanquantile(coefficient_draws, alpha / 2, axis=0)
    percentile_upper = np.nanquantile(coefficient_draws, 1 - alpha / 2, axis=0)

    point["bootstrap_percentile_lower_log_points"] = percentile_lower
    point["bootstrap_percentile_upper_log_points"] = percentile_upper
    point["simultaneous_lower_log_points"] = joint_lower
    point["simultaneous_upper_log_points"] = joint_upper
    point["simultaneous_lower_percent"] = 100 * np.expm1(joint_lower)
    point["simultaneous_upper_percent"] = 100 * np.expm1(joint_upper)
    point["simultaneous_band_significant"] = (joint_lower > 0) | (joint_upper < 0)
    point["sup_t_critical_value"] = sup_t_critical
    point["lp_bootstrap_method"] = "moving_block_bootstrap_studentized_sup_t"
    point["lp_bootstrap_repetitions_requested"] = repetitions
    point["lp_bootstrap_repetitions_successful"] = len(bootstrap_coefficients)
    point["lp_bootstrap_block_length"] = block_length
    point["bootstrap_first_stage_reestimated"] = True

    audit = {
        "frequency": frequency,
        "msdi_path": path_name,
        "method": "moving_block_bootstrap_studentized_sup_t",
        "requested_repetitions": repetitions,
        "successful_repetitions": len(bootstrap_coefficients),
        "failed_repetitions": len(failed_draws),
        "first_failure_messages": failed_draws[:10],
        "block_length": block_length,
        "block_length_rule": (
            "user_supplied"
            if requested_block_length > 0
            else "ceil(n_observations^(1/3))"
        ),
        "sup_t_critical_value": sup_t_critical,
        "first_stage_reestimated_each_draw": True,
        "seed": seed,
    }
    return point, audit


def dual_path_comparison(local_projection_results: pd.DataFrame) -> pd.DataFrame:
    peak = local_projection_results.loc[
        local_projection_results["msdi_path"].eq("peak")
    ].copy()
    breadth = local_projection_results.loc[
        local_projection_results["msdi_path"].eq("breadth")
    ].copy()
    columns = [
        "frequency",
        "horizon",
        "coefficient_log_points",
        "effect_percent",
        "p_value",
        "p_value_holm",
        "p_value_fdr_bh",
        "n_observations",
    ]
    comparison = peak[columns].merge(
        breadth[columns],
        on=["frequency", "horizon"],
        how="inner",
        suffixes=("_peak", "_breadth"),
        validate="one_to_one",
    )
    comparison["coefficient_difference_peak_minus_breadth"] = (
        comparison["coefficient_log_points_peak"]
        - comparison["coefficient_log_points_breadth"]
    )
    comparison["effect_percent_difference_peak_minus_breadth"] = (
        comparison["effect_percent_peak"] - comparison["effect_percent_breadth"]
    )
    comparison["comparison_note"] = (
        "descriptive paired-horizon difference; formal correlated-estimate test deferred to sensitivity stage"
    )
    return comparison


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, log_path = configure_logging(args.log_dir.resolve(), args.version)
    logger.info("Starting dynamic time-series analysis version=%s", args.version)
    logger.info("Arguments: %s", vars(args))
    logger.warning(
        "Interpretation guardrail: VAR, Granger, PCMCI, and local projections are observational dynamic analyses, not automatic causal identification."
    )

    weekly_path = require_file(args.weekly_panel, "weekly model panel")
    monthly_path = require_file(args.monthly_panel, "monthly model panel")
    input_paths = {"weekly": weekly_path, "monthly": monthly_path}
    all_stationarity: list[pd.DataFrame] = []
    all_cointegration: list[pd.DataFrame] = []
    all_lag_selection: list[pd.DataFrame] = []
    all_coefficients: list[pd.DataFrame] = []
    all_conditional_granger: list[pd.DataFrame] = []
    all_conventional_granger: list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    all_irf: list[pd.DataFrame] = []
    all_fevd: list[pd.DataFrame] = []
    all_pcmci: list[pd.DataFrame] = []
    all_innovations: list[pd.DataFrame] = []
    all_lp: list[pd.DataFrame] = []
    innovation_audits: list[dict[str, Any]] = []
    lp_bootstrap_audits: list[dict[str, Any]] = []
    sample_audits: list[dict[str, Any]] = []

    for spec in frequency_specifications(args):
        panel = load_panel(input_paths[spec.frequency], spec)
        logger.info(
            "Loaded %s panel rows=%s range=%s to %s",
            spec.frequency,
            len(panel),
            panel[spec.date_column].min(),
            panel[spec.date_column].max(),
        )
        for path_index, path_name in enumerate(("peak", "breadth")):
            prepared = prepare_analysis_frame(panel, spec, path_name)
            stationarity = run_stationarity_tests(
                prepared, spec.frequency, path_name, args.alpha
            )
            all_stationarity.append(stationarity)
            all_cointegration.append(
                run_cointegration_tests(prepared, spec.frequency, path_name)
            )
            analysis_frame, sample_audit = select_longest_complete_block(
                prepared,
                [*ENDOGENOUS_VARIABLES, *exogenous_controls(analysis_frame), "scfi_log"],
                spec.frequency,
            )
            sample_audit.update(
                {
                    "frequency": spec.frequency,
                    "msdi_path": path_name,
                    "role": "primary" if spec.frequency == "weekly" else "robustness",
                }
            )
            sample_audits.append(sample_audit)
            if len(analysis_frame) < args.minimum_observations:
                raise ValueError(
                    f"{spec.frequency}/{path_name} has {len(analysis_frame)} contiguous complete rows; "
                    f"minimum is {args.minimum_observations}"
                )
            logger.info(
                "%s/%s stationary-change sample rows=%s range=%s to %s excluded_noncontiguous=%s",
                spec.frequency,
                path_name,
                len(analysis_frame),
                analysis_frame.date.min(),
                analysis_frame.date.max(),
                sample_audit["excluded_noncontiguous_rows"],
            )
            for _, row in stationarity.iterrows():
                logger.info(
                    "Stationarity %s/%s variable=%s transform=%s ADF_p=%s KPSS_p=%s decision=%s",
                    spec.frequency,
                    path_name,
                    row["variable"],
                    row["transformation"],
                    row.get("adf_p_value"),
                    row.get("kpss_p_value"),
                    row["joint_decision"],
                )

            result, lag_table = fit_varx(
                analysis_frame, spec.maximum_lag, args.alpha
            )
            lag_table.insert(0, "msdi_path", path_name)
            lag_table.insert(0, "frequency", spec.frequency)
            all_lag_selection.append(lag_table)
            selected = lag_table.loc[lag_table["selected_for_analysis"]].iloc[0]
            diagnostics_pass = bool(selected["passes_residual_whiteness"])
            coefficients = extract_var_coefficients(
                result, spec.frequency, path_name
            )
            coefficients["diagnostics_pass"] = diagnostics_pass
            coefficients["exploratory_only"] = not diagnostics_pass
            all_coefficients.append(coefficients)
            conditional = conditional_granger_tests(
                result,
                spec.frequency,
                path_name,
                args.alpha,
                diagnostics_pass,
            )
            conditional = pd.concat(
                [
                    conditional,
                    hac_conditional_granger_tests(
                        analysis_frame,
                        spec.frequency,
                        path_name,
                        spec.maximum_lag,
                        args.alpha,
                    ),
                ],
                ignore_index=True,
            )
            conditional["sample_start_date"] = analysis_frame["date"].min()
            conditional["sample_end_date"] = analysis_frame["date"].max()
            conditional["innovation_definition_version"] = (
                "first_differenced_msdi_v001"
            )
            all_conditional_granger.append(conditional)
            conventional = conventional_granger_tests(
                analysis_frame,
                spec.frequency,
                path_name,
                spec.maximum_lag,
            )
            all_conventional_granger.append(conventional)
            diagnostics = var_diagnostics(
                result,
                spec.frequency,
                path_name,
                args.alpha,
                spec.maximum_lag,
            )
            all_diagnostics.append(diagnostics)
            all_irf.append(
                impulse_responses(
                    result,
                    analysis_frame,
                    spec.frequency,
                    path_name,
                    spec.irf_horizon,
                    args.irf_monte_carlo_repetitions,
                    args.alpha,
                    args.seed + path_index,
                    diagnostics_pass,
                )
            )
            all_fevd.append(
                forecast_error_variance_decomposition(
                    result,
                    spec.frequency,
                    path_name,
                    spec.irf_horizon,
                    diagnostics_pass,
                )
            )
            if not args.skip_pcmci:
                pcmci = pcmci_links(
                    analysis_frame,
                    spec.frequency,
                    path_name,
                    spec.maximum_lag,
                    args.pcmci_pc_alpha,
                    args.alpha,
                )
                all_pcmci.append(pcmci)
                scfi_links = pcmci.loc[
                    pcmci["target"].eq("scfi_return") & pcmci["significant_fdr"]
                ]
                logger.info(
                    "PCMCI %s/%s FDR-significant links into SCFI return=%s",
                    spec.frequency,
                    path_name,
                    len(scfi_links),
                )
                for _, link in scfi_links.iterrows():
                    logger.info(
                        "PCMCI link %s/%s %s(t-%s)->SCFI partial_r=%.6f raw_p=%.6g fdr_p=%.6g",
                        spec.frequency,
                        path_name,
                        link["source"],
                        link["source_lag"],
                        link["partial_correlation"],
                        link["p_value"],
                        link["p_value_fdr_bh"],
                    )

            innovations, innovation_audit = create_msdi_innovation(
                analysis_frame,
                int(result.k_ar),
                spec.frequency,
                path_name,
            )
            all_innovations.append(innovations)
            innovation_audits.append(innovation_audit)
            lp_point = local_projections(
                analysis_frame,
                innovations,
                int(result.k_ar),
                spec.frequency,
                path_name,
                spec.lp_horizon,
                args.alpha,
            )
            frequency_seed_offset = 0 if spec.frequency == "weekly" else 100_000
            lp, lp_bootstrap_audit = bootstrap_local_projection_bands(
                analysis_frame,
                lp_point,
                int(result.k_ar),
                spec.frequency,
                path_name,
                spec.lp_horizon,
                args.alpha,
                args.lp_bootstrap_repetitions,
                args.lp_block_length,
                args.seed + frequency_seed_offset + path_index * 10_000,
            )
            all_lp.append(lp)
            lp_bootstrap_audits.append(lp_bootstrap_audit)

            logger.info(
                "VARX %s/%s selected_lag=%s BIC=%.6f stable=%s residual_whiteness_p=%.6g selection_reason=%s",
                spec.frequency,
                path_name,
                int(result.k_ar),
                selected["bic"],
                result.is_stable(verbose=False),
                selected["residual_whiteness_p_value"],
                selected["selection_reason"],
            )
            if not selected["passes_residual_whiteness"]:
                logger.warning(
                    "VARX %s/%s failed residual-whiteness diagnostics at all candidate lags; use HAC local projections and HAC Granger as primary inference, and treat VAR IRFs as exploratory.",
                    spec.frequency,
                    path_name,
                )
            for _, test in conditional.iterrows():
                logger.info(
                    "Conditional Granger %s/%s model=%s %s->%s statistic=%.6f distribution=%s p=%.6g",
                    spec.frequency,
                    path_name,
                    test["model"],
                    test["causing"],
                    test["caused"],
                    test["test_statistic"],
                    test["test_distribution"],
                    test["p_value"],
                )
            logger.info(
                "LP bootstrap %s/%s successful=%s/%s block_length=%s sup_t_critical=%.6f",
                spec.frequency,
                path_name,
                lp_bootstrap_audit["successful_repetitions"],
                lp_bootstrap_audit["requested_repetitions"],
                lp_bootstrap_audit["block_length"],
                lp_bootstrap_audit["sup_t_critical_value"],
            )

    stationarity_table = pd.concat(all_stationarity, ignore_index=True)
    cointegration_table = pd.concat(all_cointegration, ignore_index=True)
    lag_selection_table = pd.concat(all_lag_selection, ignore_index=True)
    coefficient_table = pd.concat(all_coefficients, ignore_index=True)
    conditional_granger_table = pd.concat(all_conditional_granger, ignore_index=True)
    conditional_granger_table["family_id"] = (
        "conditional_granger__"
        + conditional_granger_table["frequency"].astype(str)
        + "__"
        + conditional_granger_table["model"].astype(str)
        + "__both_paths_both_directions"
    )
    conditional_granger_table = add_grouped_pvalue_adjustments(
        conditional_granger_table, ["family_id"], "p_value"
    )
    conventional_granger_table = pd.concat(all_conventional_granger, ignore_index=True)
    diagnostic_table = pd.concat(all_diagnostics, ignore_index=True)
    irf_table = pd.concat(all_irf, ignore_index=True)
    fevd_table = pd.concat(all_fevd, ignore_index=True)
    pcmci_table = (
        pd.concat(all_pcmci, ignore_index=True) if all_pcmci else pd.DataFrame()
    )
    innovation_table = pd.concat(all_innovations, ignore_index=True)
    local_projection_table = pd.concat(all_lp, ignore_index=True)
    local_projection_table["family_id"] = (
        "local_projection__"
        + local_projection_table["frequency"].astype(str)
        + "__both_paths_all_horizons"
    )
    local_projection_table["innovation_definition_version"] = (
        "msdi_change_arx_residual_v001"
    )
    local_projection_table = add_grouped_pvalue_adjustments(
        local_projection_table, ["family_id"], "p_value"
    )
    path_comparison_table = dual_path_comparison(local_projection_table)

    for _, estimate in local_projection_table.iterrows():
        logger.info(
            "Local projection %s/%s h=%s beta_log=%.6f effect_pct=%.4f HAC_se=%.6f raw_p=%.6g Holm_p=%.6g FDR_p=%.6g simultaneous=[%.6f,%.6f] joint_significant=%s",
            estimate["frequency"],
            estimate["msdi_path"],
            int(estimate["horizon"]),
            estimate["coefficient_log_points"],
            estimate["effect_percent"],
            estimate["standard_error_hac"],
            estimate["p_value"],
            estimate["p_value_holm"],
            estimate["p_value_fdr_bh"],
            estimate["simultaneous_lower_log_points"],
            estimate["simultaneous_upper_log_points"],
            estimate["simultaneous_band_significant"],
        )

    for _, comparison in path_comparison_table.iterrows():
        logger.info(
            "Dual-path difference %s h=%s peak_beta=%.6f breadth_beta=%.6f peak_minus_breadth=%.6f",
            comparison["frequency"],
            int(comparison["horizon"]),
            comparison["coefficient_log_points_peak"],
            comparison["coefficient_log_points_breadth"],
            comparison["coefficient_difference_peak_minus_breadth"],
        )

    output_frames = {
        "stationarity_tests": stationarity_table,
        "cointegration_tests": cointegration_table,
        "var_lag_selection": lag_selection_table,
        "var_coefficients": coefficient_table,
        "conditional_granger_tests": conditional_granger_table,
        "conventional_granger_tests": conventional_granger_table,
        "var_diagnostics": diagnostic_table,
        "var_impulse_responses": irf_table,
        "var_fevd": fevd_table,
        "pcmci_links": pcmci_table,
        "msdi_innovations": innovation_table,
        "local_projections": local_projection_table,
        "dual_path_comparison": path_comparison_table,
    }
    output_paths: dict[str, str] = {}
    for label, frame in output_frames.items():
        path = output_dir / f"{args.version}_{label}.csv"
        write_csv(frame, path)
        output_paths[label] = str(path)
        logger.info("Wrote %s rows=%s path=%s", label, len(frame), path)
    output_hashes = {
        label: sha256_file(Path(path)) for label, path in output_paths.items()
    }

    manifest_path = output_dir / f"{args.version}_manifest.json"
    manifest = {
        "version": args.version,
        "status": "provisional_pending_human_validation_and_causal_identification",
        "primary_frequency": "weekly",
        "monthly_role": "robustness_and_slower_dynamics",
        "inputs": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in input_paths.items()
        },
        "arguments": vars(args),
        "transformations": {
            "scfi": "first difference of log level",
            "brent": "first difference of log level",
            "msdi": "first difference of media-corrected level",
            "news_volume": "first difference of log1p article count",
            "reason": "stationary short-run system; level tests are reported separately",
        },
        "models": {
            "varx": {
                "endogenous": list(ENDOGENOUS_VARIABLES),
                "exogenous": list(SEASONAL_CONTROLS),
                "lag_selection": (
                    "minimum BIC among candidates passing adjusted residual-whiteness test; "
                    "BIC fallback with explicit warning if none pass"
                ),
            },
            "conventional_granger": "bivariate SSR F tests at every candidate lag",
            "conditional_granger": (
                "VARX joint F test at selected lag plus primary HAC Wald test at the "
                "pre-specified maximum lag"
            ),
            "pcmci": {
                "enabled": not args.skip_pcmci,
                "test": "ParCorr analytic",
                "pc_alpha": args.pcmci_pc_alpha,
                "fdr": "Benjamini-Hochberg across all lagged links within frequency and path",
                "graph_for_reporting": "graph_mark_fdr only; graph_mark_raw_alpha is diagnostic",
                "assumptions": [
                    "causal stationarity",
                    "no hidden confounding for causal interpretation",
                    "linear Gaussian conditional relations for ParCorr",
                ],
            },
            "local_projections": {
                "treatment": "one-SD residual innovation in first-differenced MSDI",
                "horizon_definition": "impact inclusive: log(SCFI[t+h]) - log(SCFI[t-1]), h=0..H",
                "controls": "lags of all stationary system variables plus seasonal Fourier terms",
                "inference": (
                    "HAC pointwise intervals; Holm and BH adjustment across both paths and all "
                    "horizons within frequency; moving-block-bootstrap studentized sup-t bands"
                ),
                "bootstrap_first_stage": "MSDI innovation model re-estimated in every draw",
            },
        },
        "sample_audit": sample_audits,
        "innovation_model_audit": innovation_audits,
        "local_projection_bootstrap_audit": lp_bootstrap_audits,
        "outputs": {**output_paths, "log": str(log_path)},
        "output_sha256": output_hashes,
        "code": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "output_row_counts": {
            label: int(len(frame)) for label, frame in output_frames.items()
        },
        "interpretation_guardrails": [
            "Granger means predictive precedence, not structural causality.",
            "PCMCI links are conditional temporal dependencies under stated assumptions.",
            "Local-projection innovations are not assumed exogenous.",
            "Impulse responses are reduced-form and not structurally identified.",
            "FEVD uses Cholesky ordering and is descriptive/order-dependent.",
            "VAR results failing residual-whiteness diagnostics are exploratory; HAC tests are primary.",
            "Cointegration tests are diagnostic only; VECM/ARDL and deterministic-term sensitivity are deferred to script 09.",
            "PCMCI raw-alpha graph marks must not be used for claims; graph_mark_fdr is authoritative.",
            "Causal claims require script 08b plus a defensible identification design.",
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "tigramite": package_version("tigramite"),
        },
    }
    manifest["outputs"]["manifest"] = str(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )
    logger.info("Wrote manifest path=%s", manifest_path)
    logger.info("Dynamic time-series analysis completed")


if __name__ == "__main__":
    main()
