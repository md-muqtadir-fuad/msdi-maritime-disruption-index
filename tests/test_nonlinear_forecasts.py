from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/18_forecasting_models.py"
SPEC = importlib.util.spec_from_file_location("nonlinear_forecasts", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


ALIASES = ["gemma_2_9b_it", "qwen_3_8b", "qwen_2_5_7b", "glm_4_9b_0414"]


class NonlinearForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = MODULE.load_yaml(MODULE.DEFAULT_CONFIG)
        cls.specs = MODULE.frequency_specs(cls.config)

    def test_frequency_specs_match_frozen_panel_contracts(self) -> None:
        weekly = self.specs["weekly"]
        monthly = self.specs["monthly"]
        self.assertEqual(weekly.date_column, "observation_date")
        self.assertEqual(weekly.complete_column, "llm_comparison_complete_case")
        self.assertEqual(weekly.horizons, (1, 2, 4, 8, 12))
        self.assertEqual(weekly.minimum_training_rows, 156)
        self.assertFalse(weekly.primary)
        self.assertEqual(monthly.date_column, "month_ending")
        self.assertEqual(
            monthly.complete_column, "llm_monthly_comparison_complete_case"
        )
        self.assertEqual(monthly.horizons, (1, 3, 6))
        self.assertEqual(monthly.minimum_training_rows, 60)
        self.assertTrue(monthly.primary)
        for spec in (weekly, monthly):
            columns = set(pd.read_csv(spec.panel_path, nrows=0).columns)
            for horizon in spec.horizons:
                self.assertIn(spec.target_column(horizon), columns)
                self.assertIn(spec.level_column(horizon), columns)

    def test_primary_configuration_counts_and_monthly_separation(self) -> None:
        weekly = MODULE.build_configurations(
            self.specs["weekly"], ALIASES, "primary"
        )
        monthly = MODULE.build_configurations(
            self.specs["monthly"], ALIASES, "primary"
        )
        self.assertEqual(len(weekly), 12)
        self.assertEqual(len(monthly), 24)
        self.assertEqual(
            {item.index_construction for item in monthly},
            set(MODULE.MONTHLY_CONSTRUCTIONS),
        )
        self.assertEqual(
            len({item.model_id("random_forest") for item in monthly}), 24
        )
        for spec, configurations in (
            (self.specs["weekly"], weekly),
            (self.specs["monthly"], monthly),
        ):
            feature_sets = MODULE.load_feature_sets(spec)
            missing = [
                configuration.upstream_feature_key
                for configuration in configurations
                if configuration.upstream_feature_key not in feature_sets
            ]
            if missing:
                self.assertTrue(all("__glm_4_9b_0414__" in key for key in missing))
                self.skipTest("requires Stage 16-17 to regenerate four-LLM manifests")
            for configuration in configurations:
                features = feature_sets[configuration.upstream_feature_key]
                self.assertFalse(any(item.startswith("target_") for item in features))

    def test_training_positions_are_horizon_purged(self) -> None:
        panel = pd.DataFrame(
            {
                "complete": [True] * 20,
                "target": np.arange(20, dtype=float),
            }
        )
        panel.loc[2, "complete"] = False
        panel.loc[4, "target"] = np.nan
        weekly = MODULE.training_positions(panel, 10, 4, "target", "complete")
        monthly = MODULE.training_positions(panel, 10, 3, "target", "complete")
        np.testing.assert_array_equal(weekly, [0, 1, 3, 5, 6])
        np.testing.assert_array_equal(monthly, [0, 1, 3, 5, 6, 7])
        self.assertLessEqual(int(weekly[-1]) + 4, 10)
        self.assertLessEqual(int(monthly[-1]) + 3, 10)
        self.assertEqual(
            len(MODULE.training_positions(panel, 2, 4, "target", "complete")), 0
        )

    def test_inner_tuning_folds_expand_and_are_horizon_purged(self) -> None:
        outer = np.array([*range(0, 50), *range(52, 80)], dtype=int)
        first = MODULE.build_inner_folds(
            outer,
            outer_fit_position=84,
            horizon=4,
            minimum_training_rows=20,
            n_splits=3,
        )
        second = MODULE.build_inner_folds(
            outer,
            outer_fit_position=84,
            horizon=4,
            minimum_training_rows=20,
            n_splits=3,
        )
        self.assertEqual(len(first), 3)
        for (train, validation), (train_again, validation_again) in zip(first, second):
            np.testing.assert_array_equal(train, train_again)
            np.testing.assert_array_equal(validation, validation_again)
            self.assertLessEqual(int(train[-1]) + 4, int(validation[0]))
            self.assertLessEqual(int(validation[-1]) + 4, 84)
            self.assertTrue(np.all(np.diff(train) > 0))
            self.assertTrue(np.all(np.diff(validation) > 0))
            self.assertFalse(set(train).intersection(validation))
        self.assertTrue(set(first[0][0]).issubset(set(first[-1][0])))

    def test_tuning_is_opt_in_and_uses_explicit_candidate_list(self) -> None:
        fixed = MODULE.resolve_parameter_candidates(
            self.config, "random_forest", tune=False
        )
        tuned = MODULE.resolve_parameter_candidates(
            self.config, "random_forest", tune=True
        )
        self.assertEqual(fixed, [self.config["models"]["random_forest"]["fixed"]])
        self.assertEqual(tuned, self.config["models"]["random_forest"]["candidates"])
        self.assertGreater(len(tuned), 1)

    def test_glm_is_canonical_deterministic_and_leakage_safe(self) -> None:
        self.assertEqual(MODULE.ALGORITHMS[0], "glm")
        fixed = MODULE.resolve_parameter_candidates(self.config, "glm", tune=False)
        tuned = MODULE.resolve_parameter_candidates(self.config, "glm", tune=True)
        self.assertEqual(fixed[0]["power"], 0)
        self.assertEqual(fixed[0]["link"], "identity")
        self.assertEqual(fixed[0]["alpha"], 0.01)
        self.assertEqual([item["alpha"] for item in tuned], [0.001, 0.01, 0.1, 1.0])

        panel = pd.DataFrame(
            {
                "x1": np.linspace(-1.0, 1.0, 24),
                "x2": np.linspace(2.0, 4.0, 24),
                "target": np.linspace(-0.2, 0.2, 24),
            }
        )
        panel.loc[3, "x1"] = np.nan
        training = np.arange(20, dtype=int)
        artifact, metadata = MODULE.fit_tabular_predictor(
            "glm",
            panel,
            training,
            ["x1", "x2"],
            "target",
            fixed[0],
            fit_seed=12345,
        )
        self.assertEqual(
            list(artifact.estimator.named_steps), ["imputer", "scaler", "estimator"]
        )
        glm = artifact.estimator.named_steps["estimator"]
        self.assertNotIn("random_state", glm.get_params())
        self.assertEqual(metadata["effective_training_rows"], len(training))
        predictions = artifact.predict(panel, [20, 21], ["x1", "x2"], 1)
        self.assertTrue(np.isfinite(predictions).all())

    def test_missing_prophet_has_explicit_skip_and_error_policies(self) -> None:
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name: str):
            if name == "prophet":
                return None
            return real_find_spec(name)

        with patch.object(MODULE.importlib.util, "find_spec", side_effect=fake_find_spec):
            selected, audit = MODULE.dependency_audit(
                ["random_forest", "prophet"], "skip"
            )
            self.assertEqual(selected, ["random_forest"])
            prophet = audit[audit["algorithm"].eq("prophet")].iloc[0]
            self.assertFalse(bool(prophet["available"]))
            self.assertEqual(prophet["action"], "skipped_explicitly")
            with self.assertRaisesRegex(RuntimeError, r"\.venv\\Scripts\\python.exe"):
                MODULE.dependency_audit(["random_forest", "prophet"], "error")

        def broken_probe(algorithm: str) -> None:
            if algorithm == "prophet":
                raise RuntimeError("Stan backend is broken")

        with patch.object(MODULE, "probe_algorithm_dependency", side_effect=broken_probe):
            selected, audit = MODULE.dependency_audit(
                ["random_forest", "prophet"], "skip"
            )
            self.assertEqual(selected, ["random_forest"])
            prophet = audit[audit["algorithm"].eq("prophet")].iloc[0]
            self.assertIn("Stan backend is broken", prophet["import_probe_error"])

    def test_lstm_sequences_end_at_the_forecast_origin(self) -> None:
        panel = pd.DataFrame({"x": np.arange(20, dtype=float)})
        arrays = MODULE.build_sequence_array(panel, [5, 8], ["x"], 4)
        np.testing.assert_array_equal(arrays[0, :, 0], [2, 3, 4, 5])
        np.testing.assert_array_equal(arrays[1, :, 0], [5, 6, 7, 8])
        with self.assertRaises(ValueError):
            MODULE.build_sequence_array(panel, [2], ["x"], 4)

    def test_partial_scopes_cannot_receive_publication_status(self) -> None:
        canonical = dict(
            smoke_test=False,
            skipped_algorithms=[],
            selected_frequencies=["weekly", "monthly"],
            requested_algorithms=list(MODULE.ALGORITHMS),
            executed_algorithms=list(MODULE.ALGORITHMS),
            signals_scope="primary",
            maximum_configurations=None,
            maximum_origins=None,
        )
        status, eligible = MODULE.determine_run_status(**canonical)
        self.assertEqual(status, "provisional_pending_human_validation")
        self.assertTrue(eligible)
        for changes, expected in (
            ({"selected_frequencies": ["weekly"]}, "partial_scope_not_for_publication"),
            ({"requested_algorithms": ["random_forest"], "executed_algorithms": ["random_forest"]}, "partial_scope_not_for_publication"),
            ({"maximum_origins": 2}, "partial_scope_not_for_publication"),
            ({"smoke_test": True}, "debug_smoke_not_for_publication"),
            ({"skipped_algorithms": ["prophet"]}, "incomplete_models_not_for_publication"),
        ):
            arguments = {**canonical, **changes}
            status, eligible = MODULE.determine_run_status(**arguments)
            self.assertEqual(status, expected)
            self.assertFalse(eligible)

    def test_forecast_hybrids_are_past_only_and_construction_specific(self) -> None:
        dates = pd.date_range("2020-01-31", periods=4, freq="ME")
        rows: list[dict[str, object]] = []
        for construction_index, construction in enumerate(
            MODULE.MONTHLY_CONSTRUCTIONS
        ):
            for alias_index, alias in enumerate(ALIASES):
                for position, origin in enumerate(dates):
                    actual = 0.01 * (position + 1)
                    target = (
                        dates[position + 1]
                        if position + 1 < len(dates)
                        else origin + pd.offsets.MonthEnd(1)
                    )
                    prediction = actual + (alias_index - 1) * 0.002 + construction_index * 0.0001
                    rows.append(
                        {
                            "frequency": "monthly",
                            "frequency_is_primary": False,
                            "algorithm": "random_forest",
                            "model_id": f"synthetic__{construction}__{alias}",
                            "is_direct_model": True,
                            "index_construction": construction,
                            "source_type": "llm",
                            "score_source": alias,
                            "signal_name": "overall_disruption_max",
                            "hybrid_level": "none",
                            "inference_tier": "secondary_predictive_robustness",
                            "price_overlap_risk": False,
                            "regime_specific_risk": False,
                            "horizon_steps": 1,
                            "horizon_unit": "months",
                            "origin_position": position,
                            "origin_date": origin,
                            "target_date": target,
                            "actual_log_change": actual,
                            "predicted_log_change": prediction,
                            "forecast_error": actual - prediction,
                            "absolute_error": abs(actual - prediction),
                            "squared_error": (actual - prediction) ** 2,
                            "current_scfi": 100.0,
                            "actual_scfi_level": 100.0 * np.exp(actual),
                            "predicted_scfi_level": 100.0 * np.exp(prediction),
                            "level_error": 100.0 * (np.exp(actual) - np.exp(prediction)),
                            "training_rows": 60,
                            "training_label_end_date": origin,
                            "training_feature_end_date": origin,
                            "model_fit_origin_position": position,
                            "model_fit_origin_date": origin,
                            "purge_gap_positions": 1,
                        }
                    )
        hybrids, weights = MODULE.build_forecast_hybrids(
            pd.DataFrame(rows),
            ALIASES,
            minimum_weekly_history=1,
            minimum_monthly_history=1,
        )
        self.assertEqual(len(hybrids), 2 * len(dates) * 2)
        self.assertEqual(
            set(hybrids["index_construction"]), set(MODULE.MONTHLY_CONSTRUCTIONS)
        )
        self.assertTrue(
            hybrids.apply(
                lambda row: str(row["index_construction"]) in str(row["model_id"]),
                axis=1,
            ).all()
        )
        sums = weights.groupby(
            [
                "frequency",
                "algorithm",
                "index_construction",
                "horizon_steps",
                "signal_name",
                "origin_date",
            ]
        )["weight"].sum()
        np.testing.assert_allclose(sums, 1.0)
        self.assertTrue(
            (
                pd.to_datetime(weights["stacking_history_label_end"])
                <= pd.to_datetime(weights["origin_date"])
            ).all()
        )

    def test_both_frequency_random_forest_real_panel_smoke(self) -> None:
        self.assertEqual(MODULE.assert_project_venv(), Path(sys.executable).resolve())
        logger = logging.getLogger("nonlinear_forecast_unittest")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        rows: list[dict[str, object]] = []
        fit_rows: list[dict[str, object]] = []
        candidates = MODULE.resolve_parameter_candidates(
            self.config, "random_forest", tune=False
        )
        for frequency in ("weekly", "monthly"):
            spec = self.specs[frequency]
            panel = MODULE.load_panel(spec)
            feature_sets = MODULE.load_feature_sets(spec)
            challengers = MODULE.limited_configurations(
                MODULE.build_configurations(spec, ALIASES, "primary"), 1
            )
            configurations = []
            for construction in spec.constructions:
                configurations.append(MODULE.baseline_configuration(spec, construction))
                configurations.extend(
                    item
                    for item in challengers
                    if item.index_construction == construction
                )
            selected_features = MODULE.validate_selected_features(
                panel, feature_sets, configurations, ["random_forest"]
            )
            horizon = spec.horizons[0]
            origins = MODULE.eligible_origins(panel, spec, horizon, maximum_origins=2)
            for configuration in configurations:
                model_id = configuration.model_id("random_forest")
                forecasts, fits, tuning = MODULE.rolling_forecasts(
                    panel=panel,
                    spec=spec,
                    horizon=horizon,
                    origins=origins,
                    algorithm="random_forest",
                    configuration=configuration,
                    features=selected_features["random_forest"][model_id],
                    candidates=candidates,
                    tune=False,
                    inner_splits=3,
                    seed=20260712,
                    smoke_test=True,
                    logger=logger,
                )
                rows.extend(forecasts)
                fit_rows.extend(fits)
                self.assertFalse(tuning)
        predictions = MODULE.attach_algorithm_baseline_differences(pd.DataFrame(rows))
        fits = pd.DataFrame(fit_rows)
        audit = MODULE.validate_common_origin_sets(predictions)
        self.assertEqual(set(predictions["frequency"]), {"weekly", "monthly"})
        self.assertTrue(audit["origin_sets_identical"].all())
        self.assertFalse(
            predictions.duplicated(
                ["frequency", "horizon_steps", "model_id", "origin_date"]
            ).any()
        )
        self.assertTrue(
            (
                pd.to_datetime(fits["training_label_end_date"])
                <= pd.to_datetime(fits["model_fit_origin_date"])
            ).all()
        )
        shared_seed_counts = fits.groupby(
            ["frequency", "algorithm", "horizon_steps", "model_fit_origin_position"]
        )["fit_seed"].nunique()
        self.assertTrue(shared_seed_counts.eq(1).all())


if __name__ == "__main__":
    unittest.main()
