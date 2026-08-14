"""Point 9: causal-role, temporal-support, and identification-readiness audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "eda/outputs/point_09_causal_readiness"
MONTHLY_PANEL = ROOT / "data/06_model_ready/causality/msdi_scfi_monthly_panel.csv"
CONTRACTS = ROOT / "eda/outputs/point_04/source_contract_registry.csv"
HORIZONS = (1, 3, 6)

ROLE_ROWS = [
    ("peak_score_media_corrected_mean", "Peak MSDI", "exposure", "monthly t", "Primary measured disruption intensity; measurement is retrospective and news-selected."),
    ("breadth_score_media_corrected_mean", "Breadth MSDI", "exposure", "monthly t", "Alternative measured disruption breadth; analyze separately and jointly only as pre-specified."),
    ("seven_dimension_paths", "Seven MSDI dimensions", "exposure components", "monthly t", "Potentially correlated components; dimension-wise causal claims multiply hypotheses and require construct validity."),
    ("target_scfi_monthly_log_change_h1m", "Future SCFI change, 1 month", "outcome", "t to t+1", "Forward outcome; overlapping issue is minimal at monthly-spaced origins."),
    ("target_scfi_monthly_log_change_h3m", "Future SCFI change, 3 months", "outcome", "t to t+3", "Forward overlapping outcome; inference requires HAC or blocks."),
    ("target_scfi_monthly_log_change_h6m", "Future SCFI change, 6 months", "outcome", "t to t+6", "Forward overlapping outcome; inference requires HAC or blocks."),
    ("brent_price", "Brent price", "candidate pre-exposure common cause", "released at or before t", "Energy costs may affect both disruption reporting and freight rates; contract needs source verification."),
    ("cpb_world_trade_volume", "CPB world trade volume", "candidate demand confounder/proxy", "released before t", "Global demand proxy; current-vintage revisions and release timing limit causal use."),
    ("china_merchandise_exports", "China merchandise exports", "candidate demand confounder/proxy", "must be released before t", "Highly relevant to Shanghai-origin demand, but its release contract is not encoded."),
    ("rwi_container_throughput", "RWI container throughput", "candidate demand/flow proxy", "must be released before t", "May be a common cause or downstream flow measure depending timing; DAG decision required."),
    ("cny_per_usd", "CNY per USD", "candidate macro confounder", "must be released before t", "Can affect exports and freight demand; release contract is not encoded."),
    ("gscpi", "GSCPI", "competing proxy/benchmark", "released before t", "Contains transportation-cost information; conditioning can absorb the channel being studied and create construct overlap."),
    ("imf_portwatch_global_container_calls", "IMF PortWatch container calls", "possible mediator/operational proxy", "daily before or within t", "Port activity may lie downstream of disruption. Do not automatically treat it as a confounder."),
    ("unctad_china_core_lane_lsbci", "China core-lane LSBCI", "slow-moving capacity/connectivity proxy", "annual release before t", "Sparse and potentially jointly determined by capacity and demand."),
    ("global_geopolitical_risk", "Global geopolitical risk", "candidate external shock/common cause", "must be released before t", "Can drive disruption and demand simultaneously; not an instrument without exclusion evidence."),
    ("trade_policy_uncertainty", "Trade policy uncertainty", "candidate external shock/common cause", "must be released before t", "News-based and potentially shares media attention with MSDI."),
    ("us_china_tension", "US-China tension", "candidate external shock/common cause", "must be released before t", "Can affect exports, routes, and news; not automatically exogenous to economic conditions."),
    ("oil_price_uncertainty", "Oil-price uncertainty", "candidate external shock/common cause", "must be released before t", "May affect both news and freight costs; not the realized bunker cost."),
    ("infectious_disease_emv", "Infectious-disease EMV", "candidate external shock/common cause", "must be released before t", "News-based measure can share attention and COVID content with MSDI."),
    ("article_count", "News article count", "measurement-process variable", "within t", "Affects precision and media correction; not automatically a causal control."),
    ("model_disagreement", "LLM disagreement", "measurement diagnostic", "within t", "Use for uncertainty or sensitivity, not as a substantive confounder."),
]

ESTIMANDS = [
    ("incremental_prediction", "Difference in out-of-sample SCFI forecast loss after adding MSDI", "identifiable_with_leakage_safe_backtest", "Predictive, not causal; identical origins and fixed model-selection protocol required."),
    ("observational_dynamic_association", "Conditional association of MSDI(t) with SCFI change(t,t+h)", "estimable_with_assumptions", "Use HAC/block inference and cautious language; VAR, Granger, PCMCI, and local projections do not create identification."),
    ("continuous_treatment_ate", "Average effect of a one-unit or one-SD MSDI intervention", "not_currently_identified", "MSDI is a measurement index, not a manipulable treatment; sequential ignorability and overlap are unverified."),
    ("causal_ml_heterogeneity", "Conditional treatment-effect heterogeneity by market state", "not_currently_identified", "Causal forests/DoubleML cannot repair unmeasured confounding, post-treatment controls, or retrospective measurement."),
    ("event_specific_effect", "SCFI response to a pre-defined external disruption event", "potential_separate_design", "Requires a defensible counterfactual/control group or instrument and must be separated from outcome-guided event selection."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit causal readiness without estimating causal effects.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-run", action="store_true", help="Run support and identification audits only.")
    return parser.parse_args()


def role_registry() -> pd.DataFrame:
    return pd.DataFrame(ROLE_ROWS, columns=["variable", "label", "causal_role", "required_timing", "interpretation_risk"])


def estimand_registry() -> pd.DataFrame:
    return pd.DataFrame(ESTIMANDS, columns=["estimand_id", "estimand", "current_status", "requirements_and_limits"])


def temporal_support(panel: pd.DataFrame) -> pd.DataFrame:
    exposures = ("peak_score_media_corrected_mean", "breadth_score_media_corrected_mean")
    controls = ("brent_monthly_mean_log_change_1m", "cpb_world_trade_volume_lag_1m", "gscpi_lag_1m", "unctad_china_core_lane_lsbci_lag_1m", "imf_portwatch_global_container_calls_lag_1m")
    rows = []
    for horizon in HORIZONS:
        outcome = f"target_scfi_monthly_log_change_h{horizon}m"
        for exposure in exposures:
            for specification, variables in {
                "exposure_only": [exposure, outcome],
                "exposure_plus_brent": [exposure, "brent_monthly_mean_log_change_1m", outcome],
                "exposure_plus_released_controls": [exposure, *controls, outcome],
            }.items():
                complete = panel[variables].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
                rows.append({"horizon_months": horizon, "exposure": exposure, "specification": specification, "variables": " | ".join(variables), "complete_months": int(complete.sum()), "first_complete_month": panel.loc[complete, "month_ending"].min(), "last_complete_month": panel.loc[complete, "month_ending"].max(), "adequate_for_simple_low_dimensional_model": int(complete.sum()) >= 60, "adequate_for_high_dimensional_causal_ml": int(complete.sum()) >= 200})
    return pd.DataFrame(rows)


def overlap_audit(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    exposures = ("peak_score_media_corrected_mean", "breadth_score_media_corrected_mean")
    modifiers = ("gscpi_lag_1m", "cpb_world_trade_volume_lag_1m", "imf_portwatch_global_container_calls_lag_1m", "brent_monthly_mean_log_change_1m")
    for exposure in exposures:
        for modifier in modifiers:
            pair = panel[[exposure, modifier]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 20 or pair[modifier].nunique() < 4:
                rows.append({"exposure": exposure, "modifier": modifier, "observations": len(pair), "bins": 0, "minimum_bin_rows": np.nan, "maximum_bin_rows": np.nan, "minimum_within_bin_exposure_sd": np.nan, "overlap_flag": "insufficient_support"})
                continue
            bins = pd.qcut(pair[modifier], q=min(4, pair[modifier].nunique()), duplicates="drop")
            grouped = pair.groupby(bins, observed=True)[exposure].agg(["size", "std"])
            minimum_sd = grouped["std"].min()
            rows.append({"exposure": exposure, "modifier": modifier, "observations": len(pair), "bins": len(grouped), "minimum_bin_rows": grouped["size"].min(), "maximum_bin_rows": grouped["size"].max(), "minimum_within_bin_exposure_sd": minimum_sd, "overlap_flag": "review" if grouped["size"].min() < 15 or pd.isna(minimum_sd) or minimum_sd <= 0 else "descriptive_support_present"})
    return pd.DataFrame(rows)


def identification_gate(contracts: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("clear_manipulable_treatment", False, "MSDI measures news-coded disruption; it is not itself a manipulable intervention."),
        ("pre_exposure_confounders_prespecified_in_dag", False, "Roles remain ambiguous for demand, port-flow, and benchmark variables."),
        ("all_model_controls_release_safe", False, f"{int(contracts['contract_status'].eq('contract_missing').sum())} candidate variables lack encoded release contracts."),
        ("historical_first_release_vintages_available", False, "Current-vintage macro downloads are stored; revision histories are not."),
        ("exposure_measured_in_real_time", False, "Article dates are historical, but operational ingestion/scoring latency was not observed."),
        ("human_validated_exposure_construction", False, "Event threshold and LLM labels remain pending human validation."),
        ("no_unmeasured_confounding_defensible", False, "Capacity, utilization, schedule reliability, bunker prices, and other market states are not fully observed."),
        ("positivity_for_high_dimensional_causal_ml", False, "Maximum monthly sample is about 120, below the pre-declared 200-row readiness floor."),
        ("outcome_temporally_after_exposure", True, "Forward 1-, 3-, and 6-month SCFI outcomes are constructed after month-t exposure."),
        ("post_treatment_controls_excluded", False, "Port calls, GSCPI, and throughput require DAG-specific timing decisions before conditioning."),
        ("causal_claim_allowed", False, "Identification prerequisites fail; only predictive and observational dynamic language is allowed."),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "evidence"])


def write_findings(path: Path, gate: pd.DataFrame, support: pd.DataFrame, overlap: pd.DataFrame) -> None:
    failures = gate.loc[~gate["passed"], "check"].tolist()
    lines = [
        "# Point 9 - Causal-readiness findings",
        "",
        "## Claim gate",
        "",
        "A causal claim is not currently supported. MSDI can be evaluated as a predictive index and as an observational dynamic signal.",
        "",
        f"Failed prerequisites ({len(failures)}): {', '.join(failures)}.",
        "",
        "## Why changing the causal model is insufficient",
        "",
        "DoubleML, causal forests, PCMCI, Granger tests, VAR/VARX, and local projections estimate different objects, but none creates exogeneity. More flexible models can reduce functional-form bias while leaving unmeasured confounding, post-treatment conditioning, measurement error, and current-vintage leakage untouched.",
        "",
        "## Defensible structure now",
        "",
        "1. Primary claim: leakage-safe out-of-sample predictive evaluation of MSDI for future SCFI changes.",
        "2. Secondary claim: multiplicity-aware observational lead-lag and dynamic association.",
        "3. Separate future causal design: pre-specify an external event/intervention and a defensible counterfactual, then use causal ML only for nuisance estimation or heterogeneity.",
        "",
        "No control is automatically included because it correlates with SCFI. Port calls, throughput, and GSCPI need DAG-specific decisions because they may be mediators or competing proxies.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(MONTHLY_PANEL, low_memory=False)
    panel["month_ending"] = pd.to_datetime(panel["month_ending"], errors="coerce")
    contracts = pd.read_csv(CONTRACTS)
    roles = role_registry()
    estimands = estimand_registry()
    support = temporal_support(panel)
    overlap = overlap_audit(panel)
    gate = identification_gate(contracts, support)
    roles.to_csv(args.output_dir / "causal_role_registry.csv", index=False)
    estimands.to_csv(args.output_dir / "candidate_estimand_registry.csv", index=False)
    support.to_csv(args.output_dir / "causal_temporal_support.csv", index=False)
    overlap.to_csv(args.output_dir / "continuous_exposure_overlap_audit.csv", index=False)
    gate.to_csv(args.output_dir / "causal_identification_gate.csv", index=False)
    write_findings(args.output_dir / "point_09_findings.md", gate, support, overlap)
    manifest = {
        "protocol_point": 9,
        "run_mode": "test" if args.test_run else "full",
        "status": "complete_no_causal_claim_allowed",
        "causal_claim_allowed": False,
        "failed_identification_checks": int((~gate["passed"]).sum()),
        "estimands_audited": len(estimands),
        "guardrail": "Causal ML may be used only after an estimand, DAG, timing contract, and identification strategy are defensible; algorithm choice cannot substitute for identification.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
