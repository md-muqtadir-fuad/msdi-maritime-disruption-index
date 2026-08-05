from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/19_time_series_causality.py"
SPEC = importlib.util.spec_from_file_location("causality_analysis", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CausalityAnalysisTests(unittest.TestCase):
    def test_prepare_analysis_frame_uses_stationary_changes(self) -> None:
        dates = pd.date_range("2024-01-01", periods=4, freq="W-MON")
        panel = pd.DataFrame(
            {
                "observation_date": dates,
                "scfi": [100.0, 110.0, 121.0, 133.1],
                "brent_price": [50.0, 55.0, 60.5, 66.55],
                "article_count": [1.0, 3.0, 7.0, 15.0],
                "peak_score_media_corrected": [0.0, 1.0, 3.0, 2.0],
                "breadth_score_media_corrected": [0.0, 0.5, 1.0, 1.5],
            }
        )
        spec = MODULE.FrequencySpecification(
            frequency="weekly",
            date_column="observation_date",
            scfi_column="scfi",
            brent_column="brent_price",
            article_count_column="article_count",
            msdi_columns={
                "peak": "peak_score_media_corrected",
                "breadth": "breadth_score_media_corrected",
            },
            maximum_lag=2,
            lp_horizon=2,
            irf_horizon=2,
        )
        result = MODULE.prepare_analysis_frame(panel, spec, "peak")
        self.assertAlmostEqual(result.loc[1, "scfi_return"], np.log(1.1))
        self.assertEqual(result["msdi_change"].iloc[1:].tolist(), [1.0, 2.0, -1.0])
        self.assertAlmostEqual(
            result.loc[1, "news_volume_change"], np.log1p(3) - np.log1p(1)
        )

    def test_longest_complete_block_does_not_bridge_time_gaps(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-01", "2024-01-08", "2024-01-29", "2024-02-05", "2024-02-12"]
                ),
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        selected, audit = MODULE.select_longest_complete_block(
            frame, ["x"], "weekly"
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected["date"].min(), pd.Timestamp("2024-01-29"))
        self.assertEqual(audit["block_count"], 2)

    def test_pvalue_adjustments_are_bounded_and_not_below_raw_holm(self) -> None:
        frame = pd.DataFrame(
            {
                "group": ["a", "a", "a"],
                "p_value": [0.01, 0.03, 0.5],
            }
        )
        result = MODULE.add_grouped_pvalue_adjustments(
            frame, ["group"], "p_value"
        )
        self.assertTrue(result["p_value_holm"].between(0, 1).all())
        self.assertTrue(result["p_value_fdr_bh"].between(0, 1).all())
        self.assertTrue((result["p_value_holm"] >= result["p_value"]).all())

    def test_hac_granger_detects_known_lagged_signal(self) -> None:
        rng = np.random.default_rng(20260801)
        n = 350
        disruption = np.zeros(n)
        freight = np.zeros(n)
        for index in range(1, n):
            disruption[index] = 0.35 * disruption[index - 1] + rng.normal(scale=0.8)
            freight[index] = (
                0.25 * freight[index - 1]
                + 0.70 * disruption[index - 1]
                + rng.normal(scale=0.8)
            )
        frame = pd.DataFrame(
            {
                "scfi_return": freight,
                "msdi_change": disruption,
                "brent_return": rng.normal(size=n),
                "news_volume_change": rng.normal(size=n),
                "season_sin": np.sin(2 * np.pi * np.arange(n) / 52),
                "season_cos": np.cos(2 * np.pi * np.arange(n) / 52),
            }
        )
        result = MODULE.hac_conditional_granger_tests(
            frame, "weekly", "peak", maximum_lag=2, alpha=0.05
        )
        forward = result.loc[
            result["causing"].eq("msdi_change")
            & result["caused"].eq("scfi_return")
        ].iloc[0]
        self.assertLess(forward["p_value"], 0.01)

    def test_local_projection_includes_impact_horizon_zero(self) -> None:
        rng = np.random.default_rng(17)
        n = 180
        disruption = rng.normal(size=n)
        freight_return = 0.55 * disruption + rng.normal(scale=0.25, size=n)
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-06", periods=n, freq="W-MON"),
                "scfi_log": np.cumsum(freight_return),
                "scfi_return": freight_return,
                "msdi_change": disruption,
                "brent_return": rng.normal(size=n),
                "news_volume_change": rng.normal(size=n),
                "season_sin": np.sin(2 * np.pi * np.arange(n) / 52),
                "season_cos": np.cos(2 * np.pi * np.arange(n) / 52),
            }
        )
        innovations = pd.DataFrame(
            {"date": frame["date"], "msdi_innovation_std": disruption}
        )
        result = MODULE.local_projections(
            frame,
            innovations,
            selected_lag=1,
            frequency="weekly",
            path_name="peak",
            maximum_horizon=1,
            alpha=0.05,
        )
        self.assertEqual(result["horizon"].tolist(), [0, 1])
        self.assertTrue(
            result["outcome_window"].eq(
                "log_scfi_t_plus_h_minus_log_scfi_t_minus_1"
            ).all()
        )
        self.assertGreater(result.loc[result["horizon"].eq(0), "coefficient_log_points"].iloc[0], 0.4)

    def test_pcmci_fdr_graph_marks_match_adjusted_significance(self) -> None:
        rng = np.random.default_rng(31)
        n = 140
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2021-01-04", periods=n, freq="W-MON"),
                "scfi_return": rng.normal(size=n),
                "msdi_change": rng.normal(size=n),
                "brent_return": rng.normal(size=n),
                "news_volume_change": rng.normal(size=n),
                "season_sin": np.sin(2 * np.pi * np.arange(n) / 52),
                "season_cos": np.cos(2 * np.pi * np.arange(n) / 52),
            }
        )
        result = MODULE.pcmci_links(
            frame,
            frequency="weekly",
            path_name="peak",
            maximum_lag=1,
            pc_alpha=0.2,
            alpha=0.05,
        )
        self.assertTrue(
            ((result["graph_mark_fdr"] != "") == result["significant_fdr"]).all()
        )


if __name__ == "__main__":
    unittest.main()
