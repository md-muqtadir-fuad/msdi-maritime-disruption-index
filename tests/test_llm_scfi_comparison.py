from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/16_llm_scfi_comparison_weekly.py"
SPEC = importlib.util.spec_from_file_location("llm_scfi_comparison", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


ALIASES = ["model_a", "model_b", "model_c", "model_d"]


def synthetic_article_frame(rows: int = 3) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "article_id": [f"a{index}" for index in range(rows)],
            "publication_date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "novelty_weight": np.ones(rows),
            "source_duplicate_weight": np.ones(rows),
            "event_id_weekly": [f"weekly-{index}" for index in range(rows)],
            "event_id_monthly": [f"monthly-{index}" for index in range(rows)],
            "llm_common_complete": np.ones(rows, dtype=bool),
        }
    )
    for alias_index, alias in enumerate(ALIASES):
        frame[f"overall_disruption_max__{alias}"] = alias_index + np.arange(rows)
        for dimension_index, dimension in enumerate(MODULE.DIMENSION_COLUMNS):
            frame[f"{dimension}__{alias}"] = (
                alias_index + dimension_index / 10 + np.arange(rows)
            )
    return frame


def synthetic_direct_forecasts(periods: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2020-01-06", periods=periods, freq="W-MON")
    rows: list[dict[str, object]] = []
    for alias_index, alias in enumerate(ALIASES):
        for position, origin in enumerate(dates):
            actual = 0.01 * np.sin(position / 3)
            prediction = actual + (alias_index - 1) * 0.001
            target = origin + pd.Timedelta(days=7)
            rows.append(
                {
                    "origin_position": position,
                    "origin_date": origin,
                    "target_date": target,
                    "horizon_weeks": 1,
                    "model_id": f"ridge__llm__{alias}__overall_disruption_max",
                    "feature_set": "synthetic",
                    "source_type": "llm",
                    "score_source": alias,
                    "signal_name": "overall_disruption_max",
                    "hybrid_level": "none",
                    "price_overlap_risk": False,
                    "regime_specific_risk": False,
                    "actual_log_change": actual,
                    "predicted_log_change": prediction,
                    "forecast_error": actual - prediction,
                    "current_scfi": 100.0,
                    "actual_scfi_level": 100.0 * np.exp(actual),
                    "predicted_scfi_level": 100.0 * np.exp(prediction),
                    "level_error": 100.0 * (np.exp(actual) - np.exp(prediction)),
                    "training_rows": 156 + position,
                    "training_feature_end_date": origin - pd.Timedelta(days=14),
                    "training_label_end_date": origin - pd.Timedelta(days=7),
                    "model_fit_origin_date": origin,
                }
            )
    return pd.DataFrame(rows)


class LlmScfiComparisonTests(unittest.TestCase):
    def test_alias_validation_requires_exactly_four_unique_values(self) -> None:
        self.assertEqual(MODULE.validate_model_aliases(ALIASES), ALIASES)
        with self.assertRaises(ValueError):
            MODULE.validate_model_aliases(["a", "b", "c"])
        with self.assertRaises(ValueError):
            MODULE.validate_model_aliases(["a", "a", "b", "c"])

    def test_score_hybrids_are_computed_at_article_level(self) -> None:
        frame = synthetic_article_frame(2)
        frame.loc[0, "overall_disruption_max__model_a"] = 0.0
        frame.loc[0, "overall_disruption_max__model_b"] = 0.0
        frame.loc[0, "overall_disruption_max__model_c"] = 4.0
        frame.loc[0, "overall_disruption_max__model_d"] = 4.0
        median = MODULE.article_signal_values(
            frame, ALIASES, "score_median", "overall_disruption_max"
        )
        mean = MODULE.article_signal_values(
            frame, ALIASES, "score_mean", "overall_disruption_max"
        )
        np.testing.assert_allclose(median, [2.0, 2.5])
        np.testing.assert_allclose(mean, [2.0, 2.5])

        expected_dimension_mean = np.mean(
            [1.5 + index / 10 for index in range(len(MODULE.DIMENSION_COLUMNS))]
        )
        breadth = MODULE.article_signal_values(
            frame, ALIASES, "score_median", "dimension_mean7"
        )
        self.assertAlmostEqual(float(breadth.iloc[0]), expected_dimension_mean)

    def test_common_article_export_excludes_any_incomplete_article(self) -> None:
        frame = synthetic_article_frame(3)
        frame.loc[1, f"{MODULE.DIMENSION_COLUMNS[0]}__model_c"] = np.nan
        required = [
            f"{signal}__{alias}"
            for alias in ALIASES
            for signal in (MODULE.PRIMARY_SCORE_COLUMN, *MODULE.DIMENSION_COLUMNS)
        ]
        frame["llm_common_complete"] = frame[required].notna().all(axis=1)
        exported = MODULE.build_common_article_export(frame, ALIASES)
        self.assertEqual(exported["article_id"].tolist(), ["a0", "a2"])
        self.assertFalse(exported.isna().any().any())

    def test_frozen_repository_input_has_exact_common_cohort(self) -> None:
        aliases = MODULE.load_model_aliases(MODULE.DEFAULT_MODEL_CONFIG)
        header = pd.read_csv(MODULE.DEFAULT_ARTICLE_SCORES, nrows=0).columns
        if f"{MODULE.PRIMARY_SCORE_COLUMN}__glm_4_9b_0414" not in header:
            self.skipTest("requires Stage 13 to regenerate four-LLM article scores")
        _, report = MODULE.load_article_scores(
            MODULE.DEFAULT_ARTICLE_SCORES,
            aliases,
            MODULE.EXPECTED_COMMON_COHORT_SIZE,
        )
        self.assertEqual(
            report["common_complete_articles"], MODULE.EXPECTED_COMMON_COHORT_SIZE
        )
        self.assertTrue(report["common_cohort_size_check_passed"])

    def test_weekly_common_counts_conserve_synthetic_cohort(self) -> None:
        frame = synthetic_article_frame(3)
        _, coverage = MODULE.build_common_weekly_signals(frame, ALIASES)
        self.assertEqual(int(coverage["llm_common_article_count"].sum()), 3)

    def test_trailing_features_cannot_change_when_only_future_value_changes(self) -> None:
        raw = "llm_signal__model_a__overall_disruption_max"
        panel = pd.DataFrame({raw: np.arange(80, dtype=float)})
        original = MODULE.add_past_signal_features(panel, [raw])
        mutated_panel = panel.copy()
        mutated_panel.loc[70:, raw] = 10000.0
        mutated = MODULE.add_past_signal_features(mutated_panel, [raw])
        pd.testing.assert_frame_equal(
            original.loc[:69], mutated.loc[:69], check_dtype=False
        )

    def test_training_positions_are_horizon_purged(self) -> None:
        panel = pd.DataFrame(
            {
                "llm_comparison_complete_case": [True] * 20,
                "target": np.arange(20, dtype=float),
            }
        )
        positions = MODULE.training_positions(panel, 10, 4, "target")
        self.assertEqual(int(positions[-1]), 6)
        self.assertLessEqual(int(positions[-1]) + 4, 10)

    def test_backward_alignment_rejects_future_and_stale_signal_weeks(self) -> None:
        base = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(["2024-01-08", "2024-01-15"]),
                "scfi": [100.0, 101.0],
                "brent_log": [4.0, 4.1],
            }
        )
        for feature in MODULE.EXTERNAL_CONTROL_FEATURES:
            base[feature] = [1.0, 1.0]
        weekly = pd.DataFrame(
            {
                "week_ending": pd.to_datetime(["2024-01-05", "2024-01-19"]),
                "llm_common_article_count": [2.0, 3.0],
                "llm_common_effective_article_count": [2.0, 3.0],
            }
        )
        for source in [*ALIASES, *MODULE.SCORE_HYBRID_SOURCES]:
            for signal in MODULE.SIGNALS:
                weekly[MODULE.signal_raw_column(source, signal)] = [1.0, 2.0]
        aligned, report = MODULE.build_aligned_panel(base, weekly, ALIASES, 6)
        self.assertEqual(report["future_matches"], 0)
        self.assertEqual(report["stale_matches"], 0)
        self.assertEqual(report["matched_common_signal_weeks"], 1)
        self.assertEqual(aligned.loc[0, "llm_signal_age_days"], 3)
        self.assertTrue(pd.isna(aligned.loc[1, "llm_week_ending"]))

    def test_past_only_stacking_never_uses_unrealized_target(self) -> None:
        direct = synthetic_direct_forecasts(30)
        hybrids, weights = MODULE.build_forecast_hybrids(
            direct, ALIASES, minimum_stacking_history=20
        )
        stack = hybrids[hybrids["score_source"].eq("forecast_nnls_stack")]
        self.assertFalse(stack.empty)
        self.assertTrue(
            (
                pd.to_datetime(stack["training_label_end_date"])
                <= pd.to_datetime(stack["model_fit_origin_date"])
            ).all()
        )
        self.assertTrue(
            (
                pd.to_datetime(weights["stacking_history_label_end"])
                <= pd.to_datetime(weights["origin_date"])
            ).all()
        )
        sums = weights.groupby(["horizon_weeks", "signal_name", "origin_date"])[
            "weight"
        ].sum()
        np.testing.assert_allclose(sums, 1.0)

    def test_holm_and_fdr_adjustments_are_bounded(self) -> None:
        raw = np.array([0.001, 0.02, 0.2, np.nan])
        holm = MODULE.adjust_pvalues(raw, "holm")
        fdr = MODULE.adjust_pvalues(raw, "fdr_bh")
        self.assertTrue(np.isnan(holm[-1]))
        self.assertTrue(np.isnan(fdr[-1]))
        self.assertTrue(np.all((holm[:3] >= raw[:3]) & (holm[:3] <= 1)))
        self.assertTrue(np.all((fdr[:3] >= raw[:3]) & (fdr[:3] <= 1)))

    def test_duplicate_forecast_origins_are_rejected(self) -> None:
        forecasts = pd.DataFrame(
            {
                "horizon_weeks": [1, 1],
                "model_id": ["model", "model"],
                "origin_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            }
        )
        with self.assertRaises(AssertionError):
            MODULE.validate_common_origin_sets(forecasts, ["model"])


if __name__ == "__main__":
    unittest.main()
