from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/07_prediction_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("prediction_benchmarks", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PredictionBenchmarkTests(unittest.TestCase):
    def test_training_positions_are_horizon_purged(self) -> None:
        panel = pd.DataFrame(
            {
                "base_model_complete_case": [True] * 20,
                "target": np.arange(20, dtype=float),
            }
        )
        positions = MODULE.training_positions(panel, 10, 4, "target")
        self.assertEqual(positions[-1], 6)
        self.assertLessEqual(positions[-1] + 4, 10)

    def test_feature_sets_contain_no_forward_targets(self) -> None:
        panel_path = (
            Path(__file__).resolve().parents[1]
            / "data/06_model_ready/prediction/msdi_scfi_weekly_panel.csv"
        )
        columns = pd.read_csv(panel_path, nrows=0).columns.tolist()
        feature_sets = MODULE.build_feature_sets(columns)
        self.assertIn("peak_score_media_corrected", feature_sets["ar_brent_peak"])
        self.assertIn("breadth_score_media_corrected", feature_sets["ar_brent_breadth"])
        self.assertFalse(
            any(
                feature.startswith("target_")
                for features in feature_sets.values()
                for feature in features
            )
        )

    def test_hac_mean_test_detects_positive_average(self) -> None:
        mean, statistic, standard_error = MODULE.hac_mean_test(
            np.array([1.0, 1.2, 0.8, 1.1, 0.9]), 1
        )
        self.assertGreater(mean, 0)
        self.assertGreater(statistic, 0)
        self.assertGreater(standard_error, 0)

    def test_comparison_uses_algorithm_matched_baseline(self) -> None:
        dates = pd.date_range("2024-01-01", periods=8, freq="W-MON")
        rows = []
        actual = np.linspace(-0.1, 0.1, len(dates))
        for model_id, offset in [
            ("ridge__ar_brent", 0.05),
            ("ridge__ar_brent_peak", 0.01),
        ]:
            for date, value in zip(dates, actual):
                rows.append(
                    {
                        "horizon_weeks": 1,
                        "origin_date": date,
                        "model_id": model_id,
                        "actual_log_change": value,
                        "predicted_log_change": value + offset,
                    }
                )
        result = MODULE.forecast_comparison_tests(pd.DataFrame(rows))
        self.assertFalse(result.empty)
        self.assertTrue(result["baseline_model"].eq("ridge__ar_brent").all())
        dm = result[result["test"].eq("diebold_mariano_squared")].iloc[0]
        self.assertGreater(dm["mean_loss_improvement"], 0)


if __name__ == "__main__":
    unittest.main()
