from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/09b_error_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("error_diagnostics", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ErrorDiagnosticsTests(unittest.TestCase):
    def test_canonical_config_and_legacy_nonlinear_freeze_are_pinned(self) -> None:
        config = MODULE.load_yaml(MODULE.DEFAULT_CONFIG)
        self.assertEqual(
            MODULE.sha256_file(MODULE.DEFAULT_CONFIG),
            MODULE.CANONICAL_CONFIG_SHA256,
        )
        contracts = config["upstream_contracts"]
        self.assertEqual(
            contracts["trusted_stage07d_nonlinear_predictions_rows"], 125760
        )
        self.assertEqual(
            len(contracts["trusted_stage07d_nonlinear_predictions_sha256"]), 64
        )

    def test_config_freezes_frequency_cohort_constructions_and_regime(self) -> None:
        config = MODULE.load_yaml(MODULE.DEFAULT_CONFIG)
        self.assertEqual(config["analysis"]["primary_frequency"], "weekly")
        self.assertEqual(config["analysis"]["monthly_role"], "robustness")
        self.assertEqual(config["analysis"]["frozen_common_article_cohort"], 6481)
        self.assertTrue(config["analysis"]["common_cohort_required"])
        self.assertEqual(
            config["analysis"]["monthly_index_constructions"],
            ["mean_safe_weekly", "adapted_pdf_sqrt_ema3"],
        )
        split = config["stability"]["operational_split"]
        self.assertEqual(split["disruption_period_start"], "2020-03-01")
        self.assertEqual(split["disruption_period_end"], "2022-06-30")
        self.assertFalse(split["official_pandemic_dates"])

    def test_config_freezes_diagnostics_handoff_and_no_causal_claim(self) -> None:
        config = MODULE.load_yaml(MODULE.DEFAULT_CONFIG)
        self.assertFalse(config["identification"]["causal_claim_allowed"])
        self.assertTrue(
            config["identification"]
            ["diagnostic_pass_does_not_validate_identification"]
        )
        self.assertTrue(config["residual_tests"]["ljung_box"]["enabled"])
        self.assertTrue(config["residual_tests"]["arch_lm"]["enabled"])
        self.assertTrue(config["residual_tests"]["jarque_bera"]["enabled"])
        self.assertEqual(config["calibration"]["method"], "mincer_zarnowitz")
        self.assertEqual(config["calibration"]["covariance"], "hac")
        self.assertTrue(config["stability"]["cusum"]["enabled"])
        self.assertTrue(config["volatility_regime"]["future_mutation_invariance_required"])
        self.assertFalse(config["inference_handoff"]["adjust_p_values_in_this_stage"])
        self.assertTrue(config["inference_handoff"]["emit_raw_p_values"])
        self.assertTrue(config["dml_consolidation"]["no_diagnostic_can_upgrade_causal_claim"])

    def test_robust_zscore_matches_median_mad_and_preserves_missing(self) -> None:
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, np.nan])
        observed = MODULE.robust_zscore(values)
        expected = (values - 3.0) / 1.4826
        np.testing.assert_allclose(
            observed.iloc[:5].to_numpy(dtype=float),
            expected.iloc[:5].to_numpy(dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertTrue(pd.isna(observed.iloc[5]))

        constant = MODULE.robust_zscore(pd.Series([5.0, 5.0, np.nan]))
        np.testing.assert_allclose(constant.iloc[:2], [0.0, 0.0])
        self.assertTrue(pd.isna(constant.iloc[2]))

    def test_past_only_volatility_regime_matches_formula(self) -> None:
        values = pd.Series(
            [0.01, 0.02, -0.01, 0.04, -0.03, 0.08, 0.02, -0.05, 0.03, 0.09],
            dtype=float,
        )
        observed = MODULE.assign_past_only_volatility_regime(
            values, window=3, minimum_periods=3
        )
        self.assertEqual(
            list(observed.columns),
            ["volatility", "past_only_threshold", "volatility_regime"],
        )
        expected_volatility = values.shift(1).rolling(3, min_periods=3).std()
        expected_threshold = (
            expected_volatility.shift(1)
            .expanding(min_periods=3)
            .quantile(0.75)
        )
        np.testing.assert_allclose(
            observed["volatility"], expected_volatility, equal_nan=True
        )
        np.testing.assert_allclose(
            observed["past_only_threshold"], expected_threshold, equal_nan=True
        )
        usable = expected_volatility.notna() & expected_threshold.notna()
        expected_labels = pd.Series("insufficient_history", index=values.index)
        expected_labels.loc[usable] = np.where(
            expected_volatility.loc[usable] > expected_threshold.loc[usable],
            "high_volatility",
            "normal_volatility",
        )
        pd.testing.assert_series_equal(
            observed["volatility_regime"],
            expected_labels.rename("volatility_regime"),
        )

    def test_past_only_volatility_regime_is_future_mutation_invariant(self) -> None:
        values = pd.Series(np.linspace(-0.08, 0.12, 30), dtype=float)
        original = MODULE.assign_past_only_volatility_regime(
            values, window=5, minimum_periods=4
        )
        mutated_values = values.copy()
        mutated_values.loc[20:] = [1000.0, -1000.0] * 5
        mutated = MODULE.assign_past_only_volatility_regime(
            mutated_values, window=5, minimum_periods=4
        )
        # Row 20 may use values only through row 19, so it is invariant too.
        pd.testing.assert_frame_equal(original.loc[:20], mutated.loc[:20])

    def test_normalized_loss_is_squared_scaled_error_and_rejects_bad_scale(self) -> None:
        error = pd.Series([2.0, -4.0, np.nan])
        scale = pd.Series([2.0, 2.0, 1.0])
        observed = MODULE.normalized_loss(error, scale)
        np.testing.assert_allclose(observed.iloc[:2], [1.0, 4.0])
        self.assertTrue(pd.isna(observed.iloc[2]))
        with self.assertRaises(ValueError):
            MODULE.normalized_loss(pd.Series([1.0]), pd.Series([0.0]))
        with self.assertRaises(ValueError):
            MODULE.normalized_loss(pd.Series([1.0]), pd.Series([-1.0]))

    def test_status_gates_never_upgrade_debug_failed_or_subset_scope(self) -> None:
        debug = "partial_debug_scope_not_publication_eligible"
        failed = "failed_source_contract_not_publication_eligible"
        subset = "complete_requested_subset_scope_not_publication_eligible"
        complete = "complete_computational_scope_pending_human_validation"
        self.assertEqual(MODULE.determine_run_status(True, None, True, True), debug)
        self.assertEqual(MODULE.determine_run_status(False, 10, True, True), debug)
        self.assertEqual(MODULE.determine_run_status(False, None, False, True), failed)
        self.assertEqual(MODULE.determine_run_status(False, None, True, False), subset)
        self.assertEqual(MODULE.determine_run_status(False, None, True, True), complete)

    def test_common_article_scope_accepts_only_the_frozen_stage09_contract(self) -> None:
        good = pd.DataFrame(
            [
                {
                    "rows_read": 6542,
                    "common_complete_rows": 6481,
                    "expected_common_complete_rows": 6481,
                    "common_cohort_passed": True,
                }
            ]
        )
        MODULE.validate_common_article_scope(good, expected_n=6481)

        wrong_count = good.copy()
        wrong_count.loc[0, "common_complete_rows"] = 6480
        with self.assertRaises((AssertionError, ValueError)):
            MODULE.validate_common_article_scope(wrong_count, expected_n=6481)

        upstream_failed = good.copy()
        upstream_failed.loc[0, "common_cohort_passed"] = False
        with self.assertRaises((AssertionError, ValueError)):
            MODULE.validate_common_article_scope(upstream_failed, expected_n=6481)

        with self.assertRaises((AssertionError, ValueError)):
            MODULE.validate_common_article_scope(
                pd.concat([good, good], ignore_index=True), expected_n=6481
            )

    def test_unique_key_contract_rejects_duplicates_and_missing_columns(self) -> None:
        unique = pd.DataFrame(
            {
                "frequency": ["weekly", "weekly"],
                "horizon": [1, 2],
                "diagnostic_id": ["a", "b"],
            }
        )
        MODULE.assert_unique_keys(
            unique, ["frequency", "horizon", "diagnostic_id"], "diagnostics"
        )

        duplicate = pd.concat([unique, unique.iloc[[0]]], ignore_index=True)
        with self.assertRaises((AssertionError, ValueError)):
            MODULE.assert_unique_keys(
                duplicate,
                ["frequency", "horizon", "diagnostic_id"],
                "diagnostics",
            )

        with self.assertRaises((AssertionError, ValueError)):
            MODULE.assert_unique_keys(
                unique, ["frequency", "missing_key"], "diagnostics"
            )

        null_key = unique.copy()
        null_key.loc[0, "diagnostic_id"] = np.nan
        with self.assertRaises((AssertionError, ValueError)):
            MODULE.assert_unique_keys(
                null_key,
                ["frequency", "horizon", "diagnostic_id"],
                "diagnostics",
            )

    def test_logistic_evaluation_uses_training_prefix_feature_availability(self) -> None:
        rows = 60
        errors = np.sin(np.arange(rows, dtype=float)) * 0.2
        errors[[5, 15, 25, 35, 45, 55]] = 5.0
        future_only = np.full(rows, np.nan)
        future_only[50:] = np.arange(10, dtype=float)
        frame = pd.DataFrame(
            {
                "origin_date": pd.date_range("2020-01-01", periods=rows, freq="W"),
                "driver_outcome": np.log1p(np.abs(errors)),
                "forecast_error_for_fold_label": errors,
                "available_from_start": np.linspace(-1.0, 1.0, rows),
                "future_only": future_only,
            }
        )
        observed = MODULE.logistic_time_ordered_performance(
            frame,
            ["available_from_start", "future_only"],
            seed=123,
        )
        self.assertEqual(
            observed["status"], "estimated_time_ordered_expanding_evaluation"
        )
        self.assertEqual(observed["feature_count"], 1)
        self.assertGreater(observed["evaluation_rows"], 0)

    def test_nuisance_summary_handles_empty_and_uses_joint_support(self) -> None:
        frame = pd.DataFrame(
            {
                "frequency": ["weekly", "weekly"],
                "frequency_role": ["primary", "primary"],
                "msdi_path": ["peak", "peak"],
                "horizon": [1, 1],
                "learner": ["rf", "rf"],
                "outcome": [1.0, np.nan],
                "outcome_prediction": [0.9, np.nan],
                "treatment_raw_msdi_change": [np.nan, 2.0],
                "treatment_prediction": [np.nan, 1.8],
            }
        )
        observed = MODULE.nuisance_oof_summary(frame).iloc[0]
        self.assertEqual(observed["n_observations"], 0)
        self.assertEqual(observed["n_outcome_observations"], 1)
        self.assertEqual(observed["n_treatment_observations"], 1)

        all_missing = frame.copy()
        for column in (
            "outcome",
            "outcome_prediction",
            "treatment_raw_msdi_change",
            "treatment_prediction",
        ):
            all_missing[column] = np.nan
        missing_summary = MODULE.nuisance_oof_summary(all_missing).iloc[0]
        self.assertEqual(missing_summary["n_observations"], 0)
        self.assertTrue(pd.isna(missing_summary["nuisance_outcome_oof_rmse_recalculated"]))
        self.assertEqual(
            missing_summary["outcome_calibration_status"],
            "skipped_no_joint_actual_prediction_rows",
        )

    def test_broader_dml_significance_is_forced_exploratory(self) -> None:
        rows = MODULE.upstream_significance_rows(
            "stage08b_test",
            pd.DataFrame(
                [{"p_value": 0.01, "effect": 0.2, "primary_family": "True"}]
            ),
            "p_value",
            ["effect"],
            force_primary_false=True,
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["upstream_primary_family"])
        self.assertFalse(rows[0]["primary_family"])


if __name__ == "__main__":
    unittest.main()
