from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/21_ablation_sensitivity_checks.py"
SPEC = importlib.util.spec_from_file_location("robustness_checks", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ALIASES = ("model_a", "model_b", "model_c")


def scored_article_frame() -> pd.DataFrame:
    values = {
        MODULE.PRIMARY_SCORE_COLUMN: (1.0, 3.0, 4.0),
        "port_disruption_score": (0.0, 2.0, 4.0),
        "global_container_movement_disruption_score": (1.0, 1.0, 3.0),
        "freight_cost_disruption_score": (2.0, 2.0, 4.0),
        "supply_disruption_score": (3.0, 1.0, 3.0),
        "production_disruption_score": (4.0, 0.0, 2.0),
        "covid19_related_disruption_score": (0.0, 0.0, 3.0),
        "demand_disruption_score": (1.0, 3.0, 4.0),
    }
    row: dict[str, object] = {
        "article_id": "a1",
        "publication_date": pd.Timestamp("2024-01-03"),
        "source_domain": "example.test",
        "duplicate_group_size": 1.0,
        "event_id": "event-a1",
        "event_cluster_size": 1.0,
    }
    for score, model_values in values.items():
        for alias, value in zip(ALIASES, model_values):
            row[f"{score}__{alias}"] = value
    return pd.DataFrame([row])


class RobustnessChecksTests(unittest.TestCase):
    def test_config_freezes_publication_scope_and_inference(self) -> None:
        config = MODULE.load_yaml(MODULE.DEFAULT_CONFIG)
        self.assertFalse(config["identification"]["causal_claim_allowed"])
        self.assertEqual(config["analysis"]["primary_frequency"], "weekly")
        self.assertEqual(config["analysis"]["monthly_role"], "robustness")
        self.assertEqual(config["analysis"]["frozen_common_article_cohort"], 6481)
        self.assertTrue(config["analysis"]["common_cohort_required"])
        self.assertEqual(config["analysis"]["score_scale"], [0, 4])
        self.assertEqual(config["forecast"]["ridge_alpha"], 1.0)
        self.assertEqual(
            config["forecast"]["extended_algorithms"],
            ["glm", "lstm", "prophet", "random_forest", "xgboost"],
        )
        self.assertEqual(config["inference"]["bootstrap_repetitions"], 999)
        self.assertEqual(
            config["causal_sensitivity"]["future_treatment_lead_placebo"]
            ["lead_offset_beyond_outcome"],
            1,
        )

    def test_config_freezes_frequency_lags_pcmci_and_explicit_skips(self) -> None:
        config = MODULE.load_yaml(MODULE.DEFAULT_CONFIG)
        self.assertEqual(config["forecast"]["weekly"]["role"], "primary")
        self.assertEqual(config["forecast"]["monthly"]["role"], "robustness")
        self.assertEqual(
            config["forecast"]["weekly"]["lag_sets"]["primary"],
            [1, 2, 4, 8, 12],
        )
        self.assertEqual(
            config["forecast"]["monthly"]["lag_sets"]["primary"],
            [1, 2, 3, 6, 12],
        )
        self.assertEqual(
            config["dynamic_sensitivity"]["pcmci"]["weekly_tau_max_specs"],
            [4, 8, 12],
        )
        self.assertEqual(
            config["dynamic_sensitivity"]["pcmci"]["monthly_tau_max_specs"],
            [2, 3, 6],
        )
        self.assertIn("dml_iv", config["skipped_specifications"])
        self.assertIn("causal_forest", config["skipped_specifications"])
        self.assertIn("event_date_assignment", config["skipped_specifications"])

    def test_duplicate_weights_match_hand_calculation(self) -> None:
        size = pd.Series([1.0, 4.0, 9.0, np.nan, 0.0])
        np.testing.assert_allclose(
            MODULE.duplicate_weights(size, "sqrt"),
            [1.0, 0.5, 1.0 / 3.0, 1.0, 1.0],
        )
        np.testing.assert_allclose(
            MODULE.duplicate_weights(size, "inverse"),
            [1.0, 0.25, 1.0 / 9.0, 1.0, 1.0],
        )
        np.testing.assert_allclose(MODULE.duplicate_weights(size, "none"), 1.0)
        with self.assertRaises(ValueError):
            MODULE.duplicate_weights(size, "unknown")

    def test_media_corrections_match_hand_calculation_and_zero_rule(self) -> None:
        raw = pd.Series([12.0, 0.0])
        count = pd.Series([3.0, 0.0])
        effective = pd.Series([4.0, 0.0])
        np.testing.assert_allclose(
            MODULE.media_correct(raw, count, effective, "log"),
            [12.0 / np.log(4.0), 0.0],
        )
        np.testing.assert_allclose(
            MODULE.media_correct(raw, count, effective, "mean"), [4.0, 0.0]
        )
        np.testing.assert_allclose(
            MODULE.media_correct(raw, count, effective, "sqrt"), [6.0, 0.0]
        )
        np.testing.assert_allclose(
            MODULE.media_correct(raw, count, effective, "none"), raw
        )
        with self.assertRaises(ValueError):
            MODULE.media_correct(raw, count, effective, "unknown")

    def test_fixed_event_weights_sum_to_one_per_week_event(self) -> None:
        frame = pd.DataFrame(
            {
                "article_id": ["a", "b", "c"],
                "publication_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"]
                ),
                "event_id": ["same", "same", "other"],
            }
        )
        weights = MODULE.period_event_weights(
            frame, pd.Series([0.5, 1.0, 1.0]), "W-FRI", "fixed"
        )
        observed = frame.assign(weight=weights).groupby("event_id")["weight"].sum()
        self.assertAlmostEqual(observed["same"], 1.0)
        self.assertAlmostEqual(observed["other"], 1.0)

    def test_article_peak_breadth_mean_and_dimension_ablation_by_hand(self) -> None:
        article = scored_article_frame()
        peak = MODULE.IndexVariant(
            variant_id="peak", path="peak", family_id="test",
            included_aliases=ALIASES,
        )
        breadth = MODULE.IndexVariant(
            variant_id="breadth", path="breadth", family_id="test",
            included_aliases=ALIASES,
        )
        peak_mean = MODULE.IndexVariant(
            variant_id="peak_mean", path="peak", family_id="test",
            score_rule="mean", included_aliases=ALIASES,
        )
        without_demand = MODULE.IndexVariant(
            variant_id="without_demand", path="breadth", family_id="test",
            included_aliases=ALIASES,
            excluded_dimension="demand_disruption_score",
        )
        self.assertAlmostEqual(MODULE.article_variant_scores(article, peak, ALIASES).iat[0], 3.0)
        self.assertAlmostEqual(
            MODULE.article_variant_scores(article, breadth, ALIASES).iat[0],
            13.0 / 7.0,
        )
        self.assertAlmostEqual(
            MODULE.article_variant_scores(article, peak_mean, ALIASES).iat[0],
            8.0 / 3.0,
        )
        self.assertAlmostEqual(
            MODULE.article_variant_scores(article, without_demand, ALIASES).iat[0],
            10.0 / 6.0,
        )

    def test_weekly_aggregation_uses_variant_duplicate_and_media_rules(self) -> None:
        first = scored_article_frame()
        second = scored_article_frame()
        first.loc[0, "article_id"] = "a1"
        second.loc[0, "article_id"] = "a2"
        second.loc[0, "publication_date"] = pd.Timestamp("2024-01-05")
        second.loc[0, "source_domain"] = "other.test"
        second.loc[0, "event_id"] = "event-a2"
        first.loc[0, "duplicate_group_size"] = 1.0
        second.loc[0, "duplicate_group_size"] = 4.0
        # Both article peak medians are 3, hence raw = 3*1 + 3*(1/2).
        article = pd.concat([first, second], ignore_index=True)
        variant = MODULE.IndexVariant(
            variant_id="reference", path="peak", family_id="test",
            included_aliases=ALIASES, duplicate_penalty="sqrt",
            media_correction="log",
        )
        weekly = MODULE.aggregate_weekly_variant(article, variant, ALIASES, "W-FRI")
        self.assertEqual(len(weekly), 1)
        self.assertEqual(int(weekly.at[0, "article_count"]), 2)
        self.assertEqual(int(weekly.at[0, "source_count"]), 2)
        self.assertAlmostEqual(weekly.at[0, "effective_article_count"], 1.5)
        self.assertAlmostEqual(weekly.at[0, "weighted_score_sum"], 4.5)
        self.assertAlmostEqual(weekly.at[0, "signal_value"], 4.5 / np.log(3.0))

    def test_signal_features_are_future_mutation_invariant(self) -> None:
        frame = pd.DataFrame({"signal": np.arange(30, dtype=float)})
        original = MODULE.add_signal_features(frame, ["signal"], [1, 2], [3, 5], "w")
        changed = frame.copy()
        changed.loc[20:, "signal"] = 10000.0
        mutated = MODULE.add_signal_features(changed, ["signal"], [1, 2], [3, 5], "w")
        pd.testing.assert_frame_equal(original.loc[:19], mutated.loc[:19])
        self.assertEqual(original.at[4, "signal_lag_2w"], 2.0)
        self.assertEqual(original.at[4, "signal_rolling_mean_3w"], 3.0)
        self.assertEqual(original.at[4, "signal_acceleration_1w"], 0.0)
        with self.assertRaises(ValueError):
            MODULE.add_signal_features(frame, ["signal"], [1], [3], "days")

    def test_training_positions_are_complete_horizon_purged_and_windowed(self) -> None:
        panel = pd.DataFrame(
            {
                "complete": [True, True, False, True, True, True, True, True, True, True, True],
                "target": [0.0, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            }
        )
        expanding = MODULE.purged_training_positions(
            panel, 10, 4, "target", "complete", None
        )
        np.testing.assert_array_equal(expanding, [0, 1, 4, 5, 6])
        self.assertLessEqual(int(expanding[-1]) + 4, 10)
        rolling = MODULE.purged_training_positions(
            panel, 10, 4, "target", "complete", 3
        )
        np.testing.assert_array_equal(rolling, [4, 5, 6])

    def test_matched_prediction_loss_is_baseline_minus_variant(self) -> None:
        predictions = self._matched_predictions()
        pair = MODULE.match_prediction_pair(
            predictions, "baseline", "variant", "weekly", "weekly_log_volume", 1
        ).sort_values("origin_date")
        np.testing.assert_allclose(
            pair["absolute_loss_difference_baseline_minus_variant"], [1.0, -2.0]
        )
        np.testing.assert_allclose(
            pair["squared_loss_difference_baseline_minus_variant"], [1.0, -4.0]
        )
        self.assertGreater(
            pair.iloc[0]["squared_loss_difference_baseline_minus_variant"], 0.0
        )

    def test_prediction_pair_rejects_actual_outcome_mismatch(self) -> None:
        predictions = self._matched_predictions()
        mask = predictions["specification_id"].eq("variant")
        predictions.loc[mask.idxmax(), "actual_log_change"] = 999.0
        with self.assertRaises((AssertionError, ValueError)):
            MODULE.match_prediction_pair(
                predictions, "baseline", "variant", "weekly", "weekly_log_volume", 1
            )

    def test_pvalue_adjustment_preserves_nan_and_matches_hand_values(self) -> None:
        values = np.array([0.01, 0.04, np.nan, 0.03])
        holm = MODULE.adjust_pvalues(values, "holm")
        bh = MODULE.adjust_pvalues(values, "benjamini_hochberg")
        np.testing.assert_allclose(holm[[0, 1, 3]], [0.03, 0.06, 0.06])
        np.testing.assert_allclose(bh[[0, 1, 3]], [0.03, 0.04, 0.04])
        self.assertTrue(np.isnan(holm[2]))
        self.assertTrue(np.isnan(bh[2]))

    def test_status_never_upgrades_debug_or_incomplete_scope(self) -> None:
        partial = "partial_debug_scope_not_publication_eligible"
        subset = "complete_requested_subset_scope_not_publication_eligible"
        complete = "complete_computational_scope_pending_human_validation"
        self.assertEqual(MODULE.determine_run_status(True, None, None, True), partial)
        self.assertEqual(MODULE.determine_run_status(False, 2, None, True), partial)
        self.assertEqual(MODULE.determine_run_status(False, None, 5, True), partial)
        self.assertEqual(MODULE.determine_run_status(False, None, None, False), subset)
        self.assertEqual(MODULE.determine_run_status(False, None, None, True), complete)

    def test_adapted_pdf_skips_only_algebraically_redundant_variants(self) -> None:
        media = MODULE.IndexVariant(
            variant_id="media_mean_peak",
            path="peak",
            family_id="media_correction",
        )
        duplicate = MODULE.IndexVariant(
            variant_id="duplicate_inverse_peak",
            path="peak",
            family_id="duplicate_penalty",
        )
        media_spec = MODULE.ForecastSpecification(
            specification_id="media",
            family_id="media_correction",
            signal_variants=(media.variant_id,),
            coverage_variant="reference_peak_median",
        )
        duplicate_spec = MODULE.ForecastSpecification(
            specification_id="duplicate",
            family_id="duplicate_penalty",
            signal_variants=(duplicate.variant_id,),
            coverage_variant="reference_peak_median",
        )
        variants = {media.variant_id: media, duplicate.variant_id: duplicate}
        self.assertIsNotNone(
            MODULE.forecast_structural_skip_reason(
                media_spec, "adapted_pdf_sqrt_ema3", variants
            )
        )
        self.assertIsNone(
            MODULE.forecast_structural_skip_reason(
                media_spec, "mean_safe_weekly", variants
            )
        )
        self.assertIsNone(
            MODULE.forecast_structural_skip_reason(
                duplicate_spec, "adapted_pdf_sqrt_ema3", variants
            )
        )

    def test_future_lead_folds_reject_overlap_and_enforce_endpoint_embargo(self) -> None:
        frame = pd.DataFrame(
            {
                "position": np.arange(60, dtype=int),
                "feature": np.arange(60, dtype=float),
                "msdi_change": np.linspace(-1.0, 1.0, 60),
                "outcome_h4": np.linspace(1.0, -1.0, 60),
            }
        )
        with self.assertRaises(ValueError):
            MODULE._future_lead_folds(
                frame, ["feature"], 4, 4, 12, 5, None
            )
        folds = MODULE._future_lead_folds(
            frame, ["feature"], 4, 5, 12, 5, 15
        )
        self.assertGreater(len(folds), 0)
        for training, evaluation in folds:
            self.assertGreaterEqual(len(training), 12)
            self.assertLess(int(training[-1]) + 5, int(evaluation[0]))

    @staticmethod
    def _matched_predictions() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        origins = pd.to_datetime(["2024-01-01", "2024-01-08"])
        targets = pd.to_datetime(["2024-01-08", "2024-01-15"])
        actuals = [1.0, 2.0]
        for specification, fitted in (("baseline", [0.0, 2.0]), ("variant", [1.0, 0.0])):
            for origin, target, actual, prediction in zip(origins, targets, actuals, fitted):
                error = actual - prediction
                rows.append(
                    {
                        "frequency": "weekly",
                        "index_construction": "weekly_log_volume",
                        "horizon": 1,
                        "origin_date": origin,
                        "target_date": target,
                        "specification_id": specification,
                        "actual_log_change": actual,
                        "actual_scfi_level": 100.0 + actual,
                        "predicted_log_change": prediction,
                        "forecast_error": error,
                        "absolute_error": abs(error),
                        "squared_error": error**2,
                    }
                )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
