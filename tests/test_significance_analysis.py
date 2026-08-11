from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/23_significance_analysis.py"
SPEC = importlib.util.spec_from_file_location("significance_analysis", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SignificanceAnalysisTests(unittest.TestCase):
    def _valid_registry_row(self, **overrides: object) -> dict[str, object]:
        effect = float(overrides.pop("effect_size", 0.10))
        standard_error = float(overrides.pop("standard_error", 0.05))
        confidence_level = float(overrides.pop("confidence_level", 0.95))
        ci_lower, ci_upper = MODULE.normal_interval(
            effect, standard_error, confidence_level
        )
        values: dict[str, object] = {
            "hypothesis_id": "valid_hypothesis",
            "semantic_key": "valid_semantic_key",
            "source_stage": "unit_test_stage",
            "source_table": "unit_test_table",
            "diagnostic_source": "unit_test_diagnostic",
            "evidence_domain": "unit_test_domain",
            "inference_role": "exploratory",
            "frequency": "monthly",
            "frequency_role": "robustness",
            "index_construction": "mean_safe_weekly",
            "test_method": "unit_test_statistic",
            "null_hypothesis": "effect_equals_zero",
            "alternative": "effect_differs_from_zero",
            "sidedness": "two_sided",
            "effect_metric": "unit_test_effect",
            "effect_unit": "unit_test_units",
            "effect_size": effect,
            "effect_interpretation": "signed_unit_test_effect",
            "standard_error": standard_error,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence_level": confidence_level,
            "uncertainty_method": "hac_normal_approximation",
            "sample_size": 100,
            "sample_start_date": "2020-01-01",
            "sample_end_date": "2021-01-01",
            "signed_effect_interpretable": True,
            "rejection_meaning": "signed_association",
            "claim_direction": "neutral_association",
            "raw_p_value": 0.20,
            "correction_family_id": "unit_test_family",
            "correction_method": "benjamini_hochberg",
            "canonical_inference": True,
            "interpretation": "unit_test_not_causality",
            "causal_claim_allowed": False,
        }
        values.update(overrides)
        return MODULE.make_registry_row(**values)

    def _validation_config(
        self, registry_rows: int, canonical_rows: int, duplicate_rows: int
    ) -> dict[str, object]:
        config = copy.deepcopy(MODULE.load_yaml(MODULE.DEFAULT_CONFIG))
        config["expected_registry_counts"] = {
            "main_rows_before_semantic_deduplication": registry_rows,
            "semantic_duplicate_rows": duplicate_rows,
            "canonical_main_rows": canonical_rows,
            "supporting_bootstrap_and_clark_west_are_not_main_rows": True,
        }
        return config

    def test_canonical_config_hash_and_no_causal_claim_are_frozen(self) -> None:
        config = MODULE.load_yaml(MODULE.DEFAULT_CONFIG)
        self.assertEqual(
            MODULE.sha256_file(MODULE.DEFAULT_CONFIG),
            MODULE.CANONICAL_CONFIG_SHA256,
        )
        self.assertFalse(config["identification"]["causal_claim_allowed"])
        self.assertFalse(config["identification"]["publication_claim_ready"])
        self.assertTrue(config["identification"]["human_validation_required"])

    def test_integrity_lock_and_manifest_native_upstream_outputs_are_frozen(self) -> None:
        config = MODULE.load_yaml(MODULE.DEFAULT_CONFIG)
        lock = PROJECT_ROOT / config["output_contract"]["script_sha256_lock"]
        self.assertEqual(MODULE.read_sha256_lock(lock), MODULE.sha256_file(SCRIPT))
        upstream = config["upstream_contracts"]
        self.assertNotIn("error_diagnostics_manifest_sha256", upstream)
        self.assertNotIn("robustness_manifest_sha256", upstream)
        _, _, diagnostics_audit = MODULE.audit_manifest_outputs(
            MODULE.DEFAULT_DIAGNOSTICS_MANIFEST,
            "stage09b",
            ["raw_significance_handoff"],
        )
        self.assertTrue(all(row["contract_passed"] for row in diagnostics_audit))
        with self.assertRaisesRegex(ValueError, "manifest SHA-256 contract failed"):
            MODULE.audit_manifest_outputs(
                MODULE.DEFAULT_DIAGNOSTICS_MANIFEST,
                "tampered_stage09b",
                [],
                "0" * 64,
            )

    def test_four_llm_scorer_dependent_counts_are_manifest_native(self) -> None:
        config = MODULE.load_yaml(MODULE.DEFAULT_CONFIG)
        prediction = config["inputs"]["trusted_legacy_inference_tables"][
            "prediction"
        ]
        self.assertTrue(prediction["require_manifest_output_contract"])
        self.assertEqual(prediction["expected_rows"], 180)
        handoff = config["registry_sources"]["stage09b_raw_handoff"]
        self.assertIsNone(handoff["expected_rows"])
        nonlinear = config["registry_sources"]["stage09_nonlinear"]
        self.assertIsNone(nonlinear["expected_rows"])
        self.assertEqual(
            nonlinear["expected_algorithms"],
            ["glm", "lstm", "prophet", "random_forest", "xgboost"],
        )
        self.assertEqual(
            config["expected_registry_counts"],
            {
                "main_rows_before_semantic_deduplication": None,
                "semantic_duplicate_rows": None,
                "canonical_main_rows": None,
                "supporting_bootstrap_and_clark_west_are_not_main_rows": True,
            },
        )

    def test_holm_and_bh_adjustments_match_closed_form_examples(self) -> None:
        raw = [0.01, 0.04, 0.03]
        np.testing.assert_allclose(
            MODULE.adjust_pvalues(raw, "holm"),
            [0.03, 0.06, 0.06],
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            MODULE.adjust_pvalues(raw, "benjamini_hochberg"),
            [0.03, 0.04, 0.04],
            rtol=0.0,
            atol=1.0e-15,
        )
        with self.assertRaises(ValueError):
            MODULE.adjust_pvalues([0.1, 1.1], "holm")

    def test_boolean_normalization_rejects_unknown_tokens(self) -> None:
        for value in (True, np.bool_(True), 1, 1.0, "true", "YES", "y"):
            self.assertTrue(MODULE.normalize_bool(value))
        for value in (False, np.bool_(False), 0, 0.0, "false", "NO", "n"):
            self.assertFalse(MODULE.normalize_bool(value, default=True))
        self.assertTrue(MODULE.normalize_bool(None, default=True))
        with self.assertRaises(ValueError):
            MODULE.normalize_bool("not-a-boolean")
        with self.assertRaises(ValueError):
            MODULE.normalize_bool(2)
        with self.assertRaisesRegex(ValueError, "non-boolean"):
            MODULE.strict_bool_series(
                pd.Series([True, "False", "unknown"]), "test boolean field"
            )

    def test_exact_alpha_is_rejected_for_all_canonical_streams(self) -> None:
        row = self._valid_registry_row(
            raw_p_value=0.05,
            correction_family_id="single_hypothesis_family",
        )
        observed = MODULE.apply_canonical_corrections(pd.DataFrame([row]), 0.05)
        result = observed.iloc[0]
        self.assertEqual(result["adjusted_p_value_within_family"], 0.05)
        self.assertEqual(result["adjusted_p_value_global_role"], 0.05)
        self.assertTrue(result["reject_raw_at_alpha"])
        self.assertTrue(result["reject_within_family_at_alpha"])
        self.assertTrue(result["reject_global_role_at_alpha"])

    def test_family_summary_excludes_alias_only_families(self) -> None:
        canonical = self._valid_registry_row(
            hypothesis_id="canonical",
            semantic_key="canonical_key",
            correction_family_id="canonical_family",
        )
        alias = self._valid_registry_row(
            hypothesis_id="alias",
            semantic_key="alias_key",
            correction_family_id="alias_only_family",
            canonical_inference=False,
            duplicate_of="canonical",
        )
        corrected = MODULE.apply_canonical_corrections(
            pd.DataFrame([canonical, alias]), 0.05
        )
        summary = MODULE.family_summary(corrected)
        self.assertEqual(summary["correction_family_id"].tolist(), ["canonical_family"])
        self.assertTrue(summary["canonical_hypotheses"].gt(0).all())

    def test_uncertainty_failure_is_a_fail_closed_claim_gate(self) -> None:
        invalid = self._valid_registry_row(
            ci_lower=0.30,
            ci_upper=-0.10,
        )
        registry = MODULE.apply_canonical_corrections(
            pd.DataFrame([invalid]), 0.05
        )
        uncertainty = MODULE.uncertainty_audit(registry)
        self.assertFalse(uncertainty["audit_passed"].all())
        self.assertGreater(
            int(uncertainty["total_validation_violations"].sum()), 0
        )
        with self.assertRaisesRegex(ValueError, "all_uncertainty_effect_size_gates"):
            MODULE.validate_registry(
                registry,
                pd.DataFrame(),
                pd.DataFrame(columns=["reconciliation_passed"]),
                pd.DataFrame(columns=["all_available_values_reconciled"]),
                uncertainty,
                MODULE.family_summary(registry),
                pd.DataFrame([{"contract_passed": True}]),
                self._validation_config(1, 1, 0),
            )

    def test_duplicate_owner_must_reference_an_existing_canonical_row(self) -> None:
        canonical = self._valid_registry_row(
            hypothesis_id="canonical",
            semantic_key="canonical_key",
        )
        alias = self._valid_registry_row(
            hypothesis_id="alias",
            semantic_key="alias_key",
            canonical_inference=False,
            duplicate_of="missing_owner",
        )
        registry = MODULE.apply_canonical_corrections(
            pd.DataFrame([canonical, alias]), 0.05
        )
        uncertainty = MODULE.uncertainty_audit(registry)
        with self.assertRaisesRegex(
            ValueError, "all_duplicates_reference_existing_canonical_owner"
        ):
            MODULE.validate_registry(
                registry,
                pd.DataFrame([{"duplicate_hypothesis_id": "alias"}]),
                pd.DataFrame(columns=["reconciliation_passed"]),
                pd.DataFrame(columns=["all_available_values_reconciled"]),
                uncertainty,
                MODULE.family_summary(registry),
                pd.DataFrame([{"contract_passed": True}]),
                self._validation_config(2, 1, 1),
            )

    def test_clark_west_registry_uses_one_sided_adjusted_mspe_semantics(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "horizon_weeks": 1,
                    "baseline_model": "nested_baseline",
                    "challenger_model": "larger_challenger",
                    "test": "clark_west_nested_squared",
                    "n": 100,
                    "mean_loss_improvement": 0.01,
                    "standard_error": 0.005,
                    "p_value": 0.025,
                }
            ]
        )
        result = MODULE.build_legacy_prediction_registry(source, 0.95).iloc[0]
        self.assertEqual(result["sidedness"], "one_sided")
        self.assertEqual(
            result["alternative"],
            "challenger_has_lower_expected_adjusted_loss",
        )
        self.assertEqual(
            result["null_hypothesis"],
            "no_positive_clark_west_adjusted_mspe_gain",
        )
        self.assertEqual(
            result["effect_metric"], "clark_west_adjusted_loss_difference"
        )
        self.assertEqual(
            result["uncertainty_method"],
            "hac_normal_two_sided_interval_for_one_sided_test",
        )

    def test_family_tiers_preserve_earliest_registration(self) -> None:
        self.assertEqual(
            MODULE.classify_weekly_llm_family(
                "primary_hybrid_vs_baseline__diebold_mariano_squared"
            ),
            "prespecified_secondary",
        )
        self.assertEqual(
            MODULE.classify_monthly_llm_family(
                "monthly_primary_llm_vs_coverage__diebold_mariano_squared"
            ),
            "confirmatory_primary",
        )
        self.assertEqual(
            MODULE.classify_weekly_llm_family(
                "sensitivity_hybrid_vs_baseline__diebold_mariano_absolute"
            ),
            "prespecified_secondary",
        )
        self.assertEqual(
            MODULE.classify_weekly_llm_family(
                "exploratory_dimension_vs_baseline__diebold_mariano_squared"
            ),
            "exploratory",
        )
    def test_semantic_forecast_alias_uses_models_then_validates_numbers(self) -> None:
        ablation = pd.DataFrame(
            [
                {
                    "frequency": "monthly",
                    "frequency_role": "primary",
                    "index_construction": "adapted_pdf_sqrt_ema3",
                    "comparison_id": "peak_vs_no_msdi",
                    "family_id": "primary_monthly_peak_vs_no_msdi",
                    "baseline_specification_id": "no_msdi",
                    "variant_specification_id": "peak_only",
                    "primary_family": True,
                    "sample_subset": "full_sample",
                    "loss": "squared_error",
                    "horizon": 1,
                    "paired_n": 100,
                    "first_origin_date": "2020-01-01",
                    "last_origin_date": "2021-01-01",
                    "mean_loss_difference": -0.01,
                    "standard_error_hac": 0.002,
                    "dm_hac_p_value": 0.01,
                    "dm_hac_p_value_adjusted": 0.05,
                    "multiplicity_method": "holm",
                    "bootstrap_p_value_two_sided": 0.02,
                    "bootstrap_ci_lower": -0.02,
                    "bootstrap_ci_upper": -0.001,
                    "clark_west_one_sided_p_value": 0.4,
                    "rmse_improvement_percent": -5.0,
                    "incremental_oos_r_squared": -0.1,
                }
            ]
        )
        later = MODULE.build_ablation_registry(ablation, 0.95)
        owner = pd.DataFrame(
            [
                MODULE.make_registry_row(
                    hypothesis_id="earliest_owner",
                    semantic_key="owner",
                    source_stage="stage07c_monthly_llm",
                    source_table="stage07c_monthly_llm_statistical_tests",
                    inference_role="confirmatory_primary",
                    index_construction="adapted_pdf_sqrt_ema3",
                    challenger_index_construction="adapted_pdf_sqrt_ema3",
                    baseline_model="ridge__ar_brent_coverage",
                    challenger_model="ridge__monthly__adapted_pdf_sqrt_ema3__score_hybrid__score_median__overall_disruption_max",
                    horizon=1,
                    test_method="diebold_mariano_squared",
                    sample_size=100,
                    effect_size=-0.01,
                    raw_p_value=0.009,
                    upstream_family_id="original_40_row_family",
                    correction_family_id="original_40_row_family",
                    correction_method="holm",
                )
            ]
        )
        config = {
            "semantic_duplicate_contract": {
                "numeric_effect_tolerance": 1.0e-12,
                "mappings": [
                    {
                        "comparison_id": "peak_vs_no_msdi",
                        "baseline_specification_id": "no_msdi",
                        "variant_specification_id": "peak_only",
                        "earlier_score_source": "score_median",
                        "earlier_signal_name": "overall_disruption_max",
                    }
                ],
                "expected_stage09_duplicate_rows": 1,
                "expected_stage09_duplicate_rows_by_frequency": {"monthly": 1},
                "expected_reassigned_primary_rows": 1,
            }
        }
        observed, audit = MODULE.apply_forecast_semantic_deduplication(
            pd.concat([owner, later], ignore_index=True, sort=False),
            ablation,
            config,
        )
        duplicate = observed.loc[observed["hypothesis_id"].ne("earliest_owner")].iloc[0]
        canonical = observed.loc[observed["hypothesis_id"].eq("earliest_owner")].iloc[0]
        self.assertFalse(duplicate["canonical_inference"])
        self.assertEqual(duplicate["duplicate_of"], "earliest_owner")
        self.assertEqual(canonical["supporting_block_bootstrap_p_value"], 0.02)
        self.assertEqual(len(audit), 1)

    def test_varx_repeats_choose_smallest_lag_ceiling(self) -> None:
        rows = []
        for maximum_lag in (4, 8):
            rows.append(
                MODULE.make_registry_row(
                    hypothesis_id=f"varx_{maximum_lag}",
                    semantic_key="same_fitted_coefficient",
                    source_table="stage09_varx_coefficients_sensitivity",
                    inference_role="exploratory",
                    frequency="weekly",
                    maximum_lag_specification=maximum_lag,
                    selected_lag=1,
                    effect_size=0.2,
                    standard_error=0.1,
                    raw_p_value=0.04,
                    correction_family_id="varx_family",
                    correction_method="benjamini_hochberg",
                )
            )
        config = {
            "semantic_duplicate_contract": {
                "varx_coefficients": {
                    "expected_input_rows": 2,
                    "expected_unique_hypotheses": 1,
                    "expected_duplicate_rows": 1,
                }
            }
        }
        observed, audit = MODULE.apply_varx_coefficient_deduplication(
            pd.DataFrame(rows), config
        )
        owner = observed.loc[observed["maximum_lag_specification"].eq(4)].iloc[0]
        duplicate = observed.loc[observed["maximum_lag_specification"].eq(8)].iloc[0]
        self.assertTrue(owner["canonical_inference"])
        self.assertFalse(duplicate["canonical_inference"])
        self.assertEqual(duplicate["duplicate_of"], owner["hypothesis_id"])
        self.assertEqual(len(audit), 1)

    def test_canonical_corrections_separate_roles_and_ignore_aliases(self) -> None:
        rows = [
            MODULE.make_registry_row(
                hypothesis_id="primary_a",
                canonical_inference=True,
                inference_role="confirmatory_primary",
                correction_family_id="primary_family",
                correction_method="holm",
                raw_p_value=0.01,
            ),
            MODULE.make_registry_row(
                hypothesis_id="primary_b",
                canonical_inference=True,
                inference_role="confirmatory_primary",
                correction_family_id="primary_family",
                correction_method="holm",
                raw_p_value=0.04,
            ),
            MODULE.make_registry_row(
                hypothesis_id="diagnostic",
                canonical_inference=True,
                inference_role="primary_diagnostic",
                correction_family_id="diagnostic_family",
                correction_method="holm",
                raw_p_value=0.01,
            ),
            MODULE.make_registry_row(
                hypothesis_id="alias",
                canonical_inference=False,
                inference_role="confirmatory_primary",
                correction_family_id="primary_family",
                correction_method="holm",
                raw_p_value=0.000001,
            ),
        ]
        observed = MODULE.apply_canonical_corrections(pd.DataFrame(rows), 0.05)
        primary = observed.loc[observed["hypothesis_id"].eq("primary_a")].iloc[0]
        diagnostic = observed.loc[observed["hypothesis_id"].eq("diagnostic")].iloc[0]
        alias = observed.loc[observed["hypothesis_id"].eq("alias")].iloc[0]
        self.assertAlmostEqual(primary["adjusted_p_value_within_family"], 0.02)
        self.assertEqual(primary["global_role_family_size"], 2)
        self.assertEqual(diagnostic["global_role_family_size"], 1)
        self.assertTrue(pd.isna(alias["adjusted_p_value_within_family"]))

    def test_upstream_reconciliation_recomputes_from_raw_values(self) -> None:
        raw = [0.01, 0.04, 0.03]
        adjusted = MODULE.adjust_pvalues(raw, "holm")
        rows = []
        for index, (p_value, adjusted_value) in enumerate(zip(raw, adjusted)):
            rows.append(
                MODULE.make_registry_row(
                    hypothesis_id=f"h{index}",
                    raw_p_value=p_value,
                    upstream_adjusted_p_value=adjusted_value,
                    upstream_adjustment_method="holm",
                    upstream_reconciliation_family_id="frozen_family",
                )
            )
        observed, audit = MODULE.reconcile_upstream_adjustments(
            pd.DataFrame(rows), 1.0e-12
        )
        self.assertTrue(audit["reconciliation_passed"].all())
        np.testing.assert_allclose(
            observed["upstream_adjustment_delta"], np.zeros(3), atol=1.0e-15
        )


if __name__ == "__main__":
    unittest.main()
