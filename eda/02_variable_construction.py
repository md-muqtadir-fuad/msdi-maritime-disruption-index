"""Point 2 of the EDA protocol: variable definitions and construction audit."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "eda/outputs/point_02_variable_construction"
MODEL_REGISTRY = ROOT / "configs/model_registry.yaml"

DATASETS = {
    "llm_scores": ROOT / "data/03_processed/model_scores/llm_scores_merged.csv",
    "article_consensus": ROOT / "data/05_indices/article_level/article_consensus_scores.csv",
    "weekly_index": ROOT / "data/05_indices/weekly/msdi_weekly.csv",
    "monthly_index": ROOT / "data/05_indices/monthly/msdi_monthly.csv",
    "weekly_model_specific": ROOT / "data/05_indices/weekly/msdi_weekly_model_specific.csv",
    "monthly_model_specific": ROOT / "data/05_indices/monthly/msdi_monthly_model_specific.csv",
    "weekly_panel": ROOT / "data/06_model_ready/prediction/msdi_scfi_weekly_panel.csv",
    "monthly_panel": ROOT / "data/06_model_ready/causality/msdi_scfi_monthly_panel.csv",
}

DIMENSIONS = (
    "port_disruption",
    "global_container_movement_disruption",
    "freight_cost_disruption",
    "supply_disruption",
    "production_disruption",
    "covid19_related_disruption",
    "demand_disruption",
)

EXTERNAL_DEFINITIONS = {
    "brent_price": (
        "USD per barrel",
        "energy input-cost proxy",
        "positive_hypothesis",
        "Higher fuel costs can increase carrier operating costs; pass-through is not guaranteed.",
    ),
    "brent_monthly_mean": (
        "USD per barrel",
        "monthly mean energy input-cost proxy",
        "positive_hypothesis",
        "Higher fuel costs can increase carrier operating costs; pass-through is not guaranteed.",
    ),
    "brent_month_end": (
        "USD per barrel",
        "month-end energy input-cost proxy",
        "positive_hypothesis",
        "Higher fuel costs can increase carrier operating costs; pass-through is not guaranteed.",
    ),
    "cpb_world_trade_volume": (
        "fixed-base volume index, 2021=100",
        "global merchandise-demand proxy",
        "positive_hypothesis",
        "Higher traded volume can increase container demand, conditional on available capacity.",
    ),
    "gscpi": (
        "standard deviations from historical mean",
        "supply-chain-pressure benchmark",
        "positive_but_construct_overlap",
        "Higher pressure should align with freight stress, but GSCPI contains transportation-cost information and is not a clean causal control.",
    ),
    "unctad_china_core_lane_lsbci": (
        "connectivity index",
        "China core-lane connectivity proxy",
        "ambiguous",
        "Connectivity can proxy capacity supply and demand simultaneously.",
    ),
    "imf_portwatch_global_container_calls": (
        "container-vessel calls",
        "global container activity proxy",
        "ambiguous",
        "More calls can indicate demand, while disruption or vessel upsizing can reduce calls without reducing cargo.",
    ),
    "rwi_container_throughput": (
        "seasonally adjusted index, 2015=100",
        "global container-throughput proxy",
        "positive_hypothesis",
        "Higher throughput can increase container-shipping demand.",
    ),
    "china_merchandise_exports": (
        "million USD",
        "China export-demand proxy",
        "positive_hypothesis",
        "Higher Chinese exports can increase demand for Shanghai-origin container capacity.",
    ),
    "cny_per_usd": (
        "CNY per USD",
        "exchange-rate proxy",
        "ambiguous",
        "Depreciation can support exports but can also alter USD prices and macroeconomic demand.",
    ),
    "global_geopolitical_risk": (
        "index, 1985-2019=100",
        "geopolitical-risk shock proxy",
        "ambiguous_predominantly_positive_cost_channel",
        "Risk can raise routing and insurance costs while also reducing trade demand.",
    ),
    "trade_policy_uncertainty": (
        "news-based index",
        "trade-policy uncertainty proxy",
        "ambiguous",
        "Uncertainty can suppress trade or induce temporary front-loading.",
    ),
    "us_china_tension": (
        "news-based index",
        "bilateral geopolitical/economic tension proxy",
        "ambiguous",
        "Tension can reroute, front-load, or reduce trade.",
    ),
    "oil_price_uncertainty": (
        "news-based index",
        "energy-price uncertainty proxy",
        "ambiguous",
        "Uncertainty has no mechanically signed effect on realized bunker costs.",
    ),
    "infectious_disease_emv": (
        "daily news-based index",
        "pandemic attention and equity-volatility proxy",
        "ambiguous",
        "Pandemics can simultaneously restrict supply and reduce or rotate demand.",
    ),
}


FORMULAS = [
    {
        "construction": "article_peak_score",
        "formula": "P_i = median_m(overall_disruption_max_{i,m})",
        "inputs": "successful model scores on common 0-4 scale",
        "frequency": "article",
        "information_rule": "minimum-model eligibility; no SCFI input",
        "status": "implemented",
    },
    {
        "construction": "article_dimension_consensus",
        "formula": "S_tilde_{i,d} = median_m(S_{i,d,m})",
        "inputs": "successful model dimension scores on common 0-4 scale",
        "frequency": "article",
        "information_rule": "no SCFI input",
        "status": "implemented",
    },
    {
        "construction": "article_breadth_score",
        "formula": "B_i = (1/7) * sum_d(S_tilde_{i,d})",
        "inputs": "seven dimension-level cross-model medians",
        "frequency": "article",
        "information_rule": "all seven dimensions required",
        "status": "implemented",
    },
    {
        "construction": "source_duplicate_weight",
        "formula": "d_i = 1 / sqrt(k_i)",
        "inputs": "duplicate_group_size k_i",
        "frequency": "article",
        "information_rule": "reference specification; inverse and none are sensitivities",
        "status": "implemented",
    },
    {
        "construction": "fixed_event_period_weight",
        "formula": "w_{i,t} = d_i / sum_{j in A_{e,t}} d_j",
        "inputs": "eligible articles in event e and period t",
        "frequency": "period-local",
        "information_rule": "calculated within period so later articles cannot revise earlier weights",
        "status": "implemented",
    },
    {
        "construction": "weighted_period_numerator",
        "formula": "C_{t,j} = sum_{i in t}(w_{i,t} * S_{i,j})",
        "inputs": "period-local weights and severity path j",
        "frequency": "weekly or monthly",
        "information_rule": "eligible articles only",
        "status": "implemented",
    },
    {
        "construction": "effective_period_volume",
        "formula": "n_eff_t = sum_{i in t} w_{i,t}",
        "inputs": "period-local event weights",
        "frequency": "weekly or monthly",
        "information_rule": "equals active event count under fixed event-total weighting when fully eligible",
        "status": "implemented",
    },
    {
        "construction": "monthly_sqrt_effective_volume",
        "formula": "U_{t,j} = C_{t,j} / sqrt(n_eff_t)",
        "inputs": "direct monthly numerator and effective volume",
        "frequency": "monthly primary",
        "information_rule": "constructed directly from monthly event assignments",
        "status": "implemented",
    },
    {
        "construction": "monthly_causal_ema3",
        "formula": "M_{t,j} = 0.5*U_{t,j} + 0.5*M_{t-1,j}",
        "inputs": "monthly sqrt-effective-volume path",
        "frequency": "monthly primary",
        "information_rule": "one-sided; restarts after a missing month",
        "status": "implemented",
    },
    {
        "construction": "weekly_log_volume_index",
        "formula": "I_{w,j} = C_{w,j} / log(1+n_w)",
        "inputs": "weekly numerator and distinct eligible article count",
        "frequency": "weekly robustness",
        "information_rule": "weekly event profile; effective-event denominator is sensitivity",
        "status": "implemented",
    },
    {
        "construction": "realtime_z_score",
        "formula": "Z_t = (M_t - mean(M_1...M_{t-1})) / sd(M_1...M_{t-1})",
        "inputs": "past index observations",
        "frequency": "weekly or monthly",
        "information_rule": "excludes current and future values",
        "status": "implemented",
    },
    {
        "construction": "monthly_forward_scfi_target",
        "formula": "y_{t,h} = log(SCFI_{t+h}) - log(SCFI_t), h in {1,3,6}",
        "inputs": "SCFI monthly series",
        "frequency": "monthly",
        "information_rule": "outcome only; never available as a predictor at origin t",
        "status": "implemented",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an audited variable-construction dictionary.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-run", action="store_true", help="Profile a bounded row sample while auditing every column.")
    return parser.parse_args()


def describe_variable(name: str) -> dict[str, str]:
    lower = name.lower()
    if lower == "legacy_score":
        return {
            "unit": "legacy project score",
            "measure": "legacy annotation retained for provenance only",
            "expected_scfi_direction": "not_directly_modelled",
            "direction_justification": "The current MSDI is rebuilt from the four active LLM outputs.",
            "construction_family": "legacy_annotation",
        }
    for base, (unit, measure, direction, justification) in EXTERNAL_DEFINITIONS.items():
        if lower == base or lower.startswith(base + "_") or re.fullmatch(re.escape(base) + r"_lag_\d+[wm]", lower):
            return {
                "unit": unit,
                "measure": measure,
                "expected_scfi_direction": direction,
                "direction_justification": justification,
                "construction_family": "external_control",
            }
    if lower in {"scfi", "scfi_monthly_mean", "scfi_month_end"}:
        return {
            "unit": "SCFI index points",
            "measure": "freight-rate outcome level",
            "expected_scfi_direction": "outcome_not_predictor_direction",
            "direction_justification": "Observed SCFI outcome.",
            "construction_family": "outcome",
        }
    if lower.startswith("target_scfi"):
        return {
            "unit": "index points, log change, or proportion as named",
            "measure": "future forecast outcome",
            "expected_scfi_direction": "outcome_not_predictor_direction",
            "direction_justification": "Future target; prohibited from predictors at forecast origin.",
            "construction_family": "forecast_target",
        }
    if any(dimension in lower for dimension in DIMENSIONS):
        return {
            "unit": "0-4 at article level; aggregated index units after weighting",
            "measure": "LLM disruption-severity dimension",
            "expected_scfi_direction": "ambiguous_non_directional_severity",
            "direction_justification": "Severity does not encode whether demand or supply moves freight rates upward or downward.",
            "construction_family": "msdi_dimension",
        }
    if "peak" in lower and any(token in lower for token in ("score", "msdi", "severity")):
        return {
            "unit": "0-4 article severity or constructed index units",
            "measure": "maximum/peak disruption-severity path",
            "expected_scfi_direction": "predominantly_positive_but_not_sign_definite",
            "direction_justification": "Cost-raising supply disruptions may increase SCFI, while demand-collapse events may decrease it.",
            "construction_family": "msdi_peak",
        }
    if "breadth" in lower and any(token in lower for token in ("score", "msdi", "severity")):
        return {
            "unit": "0-4 article breadth or constructed index units",
            "measure": "mean disruption breadth across seven dimensions",
            "expected_scfi_direction": "predominantly_positive_but_not_sign_definite",
            "direction_justification": "Broad disruption pressure can raise rates, but the score is severity rather than signed economic direction.",
            "construction_family": "msdi_breadth",
        }
    if lower == "overall_disruption_max":
        return {
            "unit": "ordinal severity score, 0-4",
            "measure": "maximum disruption score within an article/model output",
            "expected_scfi_direction": "predominantly_positive_but_not_sign_definite",
            "direction_justification": "The score is not a signed price-change forecast.",
            "construction_family": "llm_article_score",
        }
    if lower.startswith("overall_disruption_max__"):
        if any(token in lower for token in ("__diff__", "ensemble_sd", "ensemble_range")):
            measure = "cross-model peak-score difference or dispersion"
            family = "measurement_uncertainty"
            direction = "no_structural_sign"
            justification = "Model difference is a measurement diagnostic rather than an economic shock."
        elif lower.endswith("models_available"):
            measure = "number of successful LLM scores available"
            family = "measurement_eligibility"
            direction = "not_a_substantive_predictor"
            justification = "Availability count determines measurement support."
        else:
            measure = "model-specific or ensemble peak disruption severity"
            family = "llm_article_score"
            direction = "predominantly_positive_but_not_sign_definite"
            justification = "Peak severity is not a signed SCFI forecast."
        return {
            "unit": "ordinal severity score, 0-4, or score difference",
            "measure": measure,
            "expected_scfi_direction": direction,
            "direction_justification": justification,
            "construction_family": family,
        }
    if lower.endswith("_disruption_score"):
        return {
            "unit": "ordinal severity score, 0-4",
            "measure": "LLM article-level disruption score",
            "expected_scfi_direction": "ambiguous_non_directional_severity",
            "direction_justification": "The taxonomy scores severity, not a signed SCFI effect.",
            "construction_family": "llm_article_score",
        }
    if any(token in lower for token in ("shipping_dimension_", "economic_dimension_", "overall_disruption_mean", "overall_disruption_sum")):
        return {
            "unit": "derived 0-4 score summary or sum",
            "measure": "within-output disruption score summary",
            "expected_scfi_direction": "ambiguous_non_directional_severity",
            "direction_justification": "Derived severity summaries do not encode a signed SCFI effect.",
            "construction_family": "llm_article_score_summary",
        }
    if any(token in lower for token in ("article_count", "event_count", "source_count", "news_volume")):
        return {
            "unit": "count",
            "measure": "coverage or media-volume diagnostic",
            "expected_scfi_direction": "no_structural_sign",
            "direction_justification": "Coverage intensity is not itself a signed economic shock.",
            "construction_family": "coverage_diagnostic",
        }
    if any(token in lower for token in ("disagreement", "ensemble_sd", "ensemble_range")):
        return {
            "unit": "score dispersion",
            "measure": "cross-model disagreement diagnostic",
            "expected_scfi_direction": "no_structural_sign",
            "direction_justification": "Model disagreement measures uncertainty in measurement, not economic pressure.",
            "construction_family": "measurement_uncertainty",
        }
    if any(token in lower for token in ("novelty_weight", "duplicate_weight", "event_weight")):
        return {
            "unit": "nonnegative weight",
            "measure": "deduplication/event contribution weight",
            "expected_scfi_direction": "not_a_substantive_predictor",
            "direction_justification": "Measurement weight, not an economic variable.",
            "construction_family": "measurement_weight",
        }
    if any(token in lower for token in ("weighted_contribution", "event_period_source_weight_sum")):
        return {
            "unit": "weighted score contribution or weight sum",
            "measure": "intermediate MSDI aggregation field",
            "expected_scfi_direction": "not_an_independent_predictor",
            "direction_justification": "This field is an input to the declared index construction.",
            "construction_family": "msdi_aggregation_intermediate",
        }
    if any(token in lower for token in ("event_id", "event_cluster_size", "event_span_days", "event_profile", "event_penalty")):
        return {
            "unit": "identifier, count, days, or method label",
            "measure": "semantic-event assignment or provenance",
            "expected_scfi_direction": "not_a_substantive_predictor",
            "direction_justification": "Event metadata controls measurement and deduplication.",
            "construction_family": "event_measurement_metadata",
        }
    if any(token in lower for token in ("duplicate_group", "deduplication_status")):
        return {
            "unit": "identifier, count, or status",
            "measure": "exact-duplicate measurement metadata",
            "expected_scfi_direction": "not_a_substantive_predictor",
            "direction_justification": "Deduplication metadata controls measurement rather than representing economic pressure.",
            "construction_family": "duplicate_measurement_metadata",
        }
    if lower in {"models_successful", "ensemble_eligible", "breadth_active_dimension_count"} or lower.endswith("_available"):
        return {
            "unit": "count or Boolean eligibility flag",
            "measure": "score/index eligibility diagnostic",
            "expected_scfi_direction": "not_a_substantive_predictor",
            "direction_justification": "Eligibility describes measurement support.",
            "construction_family": "measurement_eligibility",
        }
    if any(token in lower for token in ("token", "elapsed_seconds", "model_attempt", "text_window_count", "run_chunk", "fallback_window_count")):
        return {
            "unit": "count or seconds",
            "measure": "LLM operational-cost/runtime diagnostic",
            "expected_scfi_direction": "not_a_substantive_predictor",
            "direction_justification": "Operational metrics are used for TOPSIS/reporting and never weight MSDI.",
            "construction_family": "llm_operational_diagnostic",
        }
    if any(token in lower for token in ("scoring_status", "scoring_error", "retrieval_status", "error_type", "error_message", "text_encoding_repaired")):
        return {
            "unit": "status, error label, message, or Boolean",
            "measure": "processing-quality diagnostic",
            "expected_scfi_direction": "not_a_substantive_predictor",
            "direction_justification": "Processing status is retained for audit and eligibility.",
            "construction_family": "processing_quality_metadata",
        }
    if any(token in lower for token in ("quality_score", "calculated_text_length", "calculated_word_count", "retrieved_text_len", "retrieved_word_count", "model_text_char_count", "model_text_word_count", "input_rows_before_filter", "active_theme_count")):
        return {
            "unit": "score or count",
            "measure": "article/text quality or coverage diagnostic",
            "expected_scfi_direction": "not_a_substantive_predictor",
            "direction_justification": "Quality and length fields support measurement auditing.",
            "construction_family": "article_quality_diagnostic",
        }
    if any(token in lower for token in ("reasoning", "evidence", "legacy_theme", "legacy_dimension", "matched_maritime_keywords", "pre_llm_filter")):
        return {
            "unit": "text, label, or Boolean",
            "measure": "LLM/filter explanation or legacy annotation",
            "expected_scfi_direction": "not_directly_modelled",
            "direction_justification": "Textual explanations and legacy annotations require separate content validation.",
            "construction_family": "annotation_or_explanation",
        }
    if lower.endswith("_log") or re.search(r"_log_lag_\d+[wm]$", lower):
        return {
            "unit": "natural-log units",
            "measure": "log-transformed level or backward lag",
            "expected_scfi_direction": "inherits_source_variable_hypothesis",
            "direction_justification": "Transformation does not change the conceptual direction of its source variable.",
            "construction_family": "time_series_transformation",
        }
    if "_log_change_" in lower or "_pct_change_" in lower or lower.endswith("_return"):
        return {
            "unit": "log change or proportional change",
            "measure": "backward or forward change as indicated by field name",
            "expected_scfi_direction": "inherits_source_or_is_target",
            "direction_justification": "Timing and target status are determined from the explicit lag/horizon suffix.",
            "construction_family": "time_series_transformation",
        }
    if re.search(r"_lag_\d+[wm]$", lower) or any(token in lower for token in ("rolling_mean", "rolling_sd", "acceleration")):
        return {
            "unit": "inherits source variable",
            "measure": "backward-only lag or rolling feature",
            "expected_scfi_direction": "inherits_source_variable_hypothesis",
            "direction_justification": "Backward transformation preserves the conceptual role of its source variable.",
            "construction_family": "time_series_transformation",
        }
    if lower.startswith("calendar_"):
        return {
            "unit": "calendar value or sine/cosine encoding",
            "measure": "deterministic seasonal control",
            "expected_scfi_direction": "no_single_linear_sign",
            "direction_justification": "Calendar controls capture recurring seasonality.",
            "construction_family": "calendar_control",
        }
    if lower.startswith("has_") or lower.endswith("complete_case") or lower.endswith("_available"):
        return {
            "unit": "Boolean flag",
            "measure": "availability/completeness diagnostic",
            "expected_scfi_direction": "not_a_substantive_predictor",
            "direction_justification": "Availability flags define auditable samples.",
            "construction_family": "availability_diagnostic",
        }
    if lower.endswith("_age_days"):
        return {
            "unit": "days",
            "measure": "source-data age at panel observation",
            "expected_scfi_direction": "not_a_substantive_predictor",
            "direction_justification": "Age verifies as-of alignment.",
            "construction_family": "availability_provenance",
        }
    if lower in {"weekly_observations_in_month", "msdi_available_weeks"}:
        return {
            "unit": "count",
            "measure": "monthly coverage diagnostic",
            "expected_scfi_direction": "not_a_substantive_predictor",
            "direction_justification": "Coverage count defines measurement completeness.",
            "construction_family": "availability_diagnostic",
        }
    if any(token in lower for token in ("article_id", "row_uid", "source_row", "input_row", "model_name", "model_alias", "taxonomy_version", "severity_scale")):
        return {
            "unit": "identifier or version label",
            "measure": "record/model/taxonomy provenance",
            "expected_scfi_direction": "not_applicable",
            "direction_justification": "Identifier/provenance field.",
            "construction_family": "key_or_provenance",
        }
    if any(token in lower for token in ("headline", "title", "retrieved_text", "summary", "author", "keyword", "meta_", "source_name", "source_domain", "source_dataset", "source_file", "input_file", "_source_", "collection", "retrieval_method")):
        return {
            "unit": "text or category",
            "measure": "article content or provenance metadata",
            "expected_scfi_direction": "not_directly_modelled",
            "direction_justification": "Content/provenance is audited separately from numeric predictors.",
            "construction_family": "article_content_or_provenance",
        }
    if any(token in lower for token in ("url", "link", "hash", "top_image")):
        return {
            "unit": "identifier or URL",
            "measure": "retrieval/deduplication provenance",
            "expected_scfi_direction": "not_applicable",
            "direction_justification": "Identifier/provenance field.",
            "construction_family": "key_or_provenance",
        }
    if lower.endswith(("_date", "_ending", "_at_utc")) or lower in {"date", "event_day", "month_end", "week_ending", "observation_date"}:
        return {
            "unit": "date/time",
            "measure": "time or provenance key",
            "expected_scfi_direction": "not_applicable",
            "direction_justification": "Identifier/provenance field.",
            "construction_family": "key_or_provenance",
        }
    return {
        "unit": "requires field-level verification",
        "measure": "requires field-level verification",
        "expected_scfi_direction": "not_prespecified",
        "direction_justification": "No direction will be inferred from observed SCFI associations.",
        "construction_family": "unclassified_for_manual_review",
    }


def profile_datasets(test_run: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    dictionary_rows = []
    schema_rows = []
    for dataset, path in DATASETS.items():
        if not path.is_file():
            schema_rows.append({"dataset": dataset, "path": str(path.relative_to(ROOT)), "status": "missing"})
            continue
        frame = pd.read_csv(path, low_memory=False, nrows=500 if test_run else None)
        schema_rows.append(
            {
                "dataset": dataset,
                "path": str(path.relative_to(ROOT)),
                "status": "sampled" if test_run else "fully_profiled",
                "rows_profiled": len(frame),
                "columns": len(frame.columns),
            }
        )
        for column in frame.columns:
            description = describe_variable(str(column))
            dictionary_rows.append(
                {
                    "dataset": dataset,
                    "variable": str(column),
                    "dtype_observed": str(frame[column].dtype),
                    "nonmissing_profiled": int(frame[column].notna().sum()),
                    "unique_profiled": int(frame[column].nunique(dropna=True)),
                    **description,
                }
            )
    return pd.DataFrame(dictionary_rows), pd.DataFrame(schema_rows)


def llm_registry_audit(test_run: bool) -> pd.DataFrame:
    registry = yaml.safe_load(MODEL_REGISTRY.read_text(encoding="utf-8"))
    configured = registry.get("active_scoring_models", [])
    scores_path = DATASETS["llm_scores"]
    # Reading one narrow column is inexpensive and ensures test mode still
    # validates every observed LLM rather than only the first file block.
    names = pd.read_csv(scores_path, usecols=["model_name"])["model_name"].dropna().astype(str).unique()
    rows = []
    for name in sorted(names):
        matches = [model for model in configured if str(model.get("name_contains", "")).casefold() in name.casefold()]
        rows.append(
            {
                "observed_model_name": name,
                "registry_matches": len(matches),
                "model_alias": matches[0]["alias"] if len(matches) == 1 else "",
                "parameter_count_billion": matches[0].get("parameter_count_billion") if len(matches) == 1 else None,
                "mapping_status": "unique" if len(matches) == 1 else "unmapped" if not matches else "ambiguous",
            }
        )
    return pd.DataFrame(rows)


def write_findings(output: Path, dictionary: pd.DataFrame, schemas: pd.DataFrame, models: pd.DataFrame) -> None:
    unresolved = dictionary[dictionary["construction_family"].eq("unclassified_for_manual_review")]
    direction_counts = dictionary["expected_scfi_direction"].value_counts()
    lines = [
        "# Point 2 — Variable construction findings",
        "",
        "All observed fields remain documented. Expected signs are hypotheses recorded before modelling and are not fitted to SCFI results.",
        "",
        "## Coverage",
        "",
        f"- Datasets audited: {len(schemas)}.",
        f"- Dataset-variable records: {len(dictionary)}.",
        f"- Unique variable names: {dictionary['variable'].nunique()}.",
        f"- Fields requiring manual definition: {len(unresolved)}.",
        "",
        "## Expected-direction classifications",
        "",
        "| Classification | Dataset-variable records |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | {count} |" for name, count in direction_counts.items())
    lines.extend(
        [
            "",
            "## Critical interpretation",
            "",
            "- The seven LLM dimensions and overall disruption scores are severity measures, not signed SCFI forecasts.",
            "- Demand disruption can increase rates through demand surges or decrease rates through demand collapse; its sign cannot be assumed from a 0-4 severity score.",
            "- GSCPI is retained but labelled as a benchmark with freight-cost construct overlap, not an ordinary causal confounder.",
            "- Article counts, event counts, source counts, weights, and model disagreement are measurement diagnostics unless a later specification explicitly justifies another role.",
            "- TOPSIS ranks operational LLM efficiency and never weights the median MSDI.",
            "",
            "## LLM registry mapping",
            "",
            "| Observed model | Alias | Mapping |",
            "|---|---|---|",
        ]
    )
    for row in models.itertuples(index=False):
        lines.append(f"| `{row.observed_model_name}` | `{row.model_alias}` | `{row.mapping_status}` |")
    if not unresolved.empty:
        lines.extend(
            [
                "",
                "## Manual definition queue",
                "",
                "Unclassified fields are retained and listed in `manual_definition_queue.csv`; they are not silently assigned a meaning.",
            ]
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dictionary, schemas = profile_datasets(args.test_run)
    models = llm_registry_audit(args.test_run)
    formulas = pd.DataFrame(FORMULAS)
    manual = dictionary[dictionary["construction_family"].eq("unclassified_for_manual_review")].copy()

    dictionary.to_csv(args.output_dir / "variable_dictionary.csv", index=False)
    schemas.to_csv(args.output_dir / "dataset_schema_summary.csv", index=False)
    formulas.to_csv(args.output_dir / "construction_formula_registry.csv", index=False)
    models.to_csv(args.output_dir / "llm_registry_audit.csv", index=False)
    manual.to_csv(args.output_dir / "manual_definition_queue.csv", index=False)
    write_findings(args.output_dir / "point_02_findings.md", dictionary, schemas, models)

    failed_models = models.loc[~models["mapping_status"].eq("unique")]
    manifest = {
        "protocol_point": 2,
        "run_mode": "test" if args.test_run else "full_profile",
        "status": "complete_with_manual_definition_queue" if len(manual) else "complete",
        "dataset_variable_records": len(dictionary),
        "unique_variables": int(dictionary["variable"].nunique()),
        "manual_definition_records": len(manual),
        "llm_mapping_failures": failed_models["observed_model_name"].tolist(),
        "guardrail": "Expected directions are prespecified hypotheses, not estimates from SCFI outcomes.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return_code = 1 if len(failed_models) else 0
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
