from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/14_build_model_panels.py"
SPEC = importlib.util.spec_from_file_location("build_model_panels", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def msdi_frame(dates: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"msdi_week_ending": pd.to_datetime(dates)})
    for column in MODULE.MSDI_BASE_COLUMNS:
        frame[column] = 1.0
    frame["peak_score_media_corrected"] = np.arange(1, len(frame) + 1, dtype=float)
    frame["breadth_score_media_corrected"] = frame["peak_score_media_corrected"] / 2
    return frame


class BuildModelPanelsTests(unittest.TestCase):
    def test_obsolete_generic_unctad_control_is_not_silently_substituted(self) -> None:
        frame = pd.DataFrame(
            {
                "variable": [
                    "cpb_world_trade_volume",
                    "gscpi",
                    "unctad_lsbci",
                    "imf_portwatch_global_container_calls",
                ],
                "availability_date": ["2024-01-01"] * 4,
                "value": [1.0] * 4,
                "source_frequency": ["monthly", "monthly", "annual", "daily"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controls.csv"
            frame.to_csv(path, index=False)
            controls, audit = MODULE.load_external_controls(path)
        # The generic global LSBCI is not the China core-lane series the study uses,
        # so it must be dropped by the allowlist rather than standing in for it.
        self.assertNotIn("unctad_lsbci", set(controls["variable"]))
        self.assertNotIn("unctad_lsbci", audit["variables"])
        self.assertNotIn("unctad_china_core_lane_lsbci", set(controls["variable"]))

    def test_annual_control_is_backward_joined_with_annual_validity(self) -> None:
        panel = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(
                    ["2025-05-01", "2025-09-01"]
                )
            }
        )
        controls = pd.DataFrame(
            {
                "variable": ["unctad_china_core_lane_lsbci"],
                "availability_date": pd.to_datetime(["2024-05-15"]),
                "value": [0.7],
                "source_frequency": ["annual"],
            }
        )
        result, audit = MODULE.attach_external_controls(panel, controls)
        self.assertEqual(result.loc[0, "unctad_china_core_lane_lsbci"], 0.7)
        self.assertTrue(pd.isna(result.loc[1, "unctad_china_core_lane_lsbci"]))
        self.assertEqual(
            audit["unctad_china_core_lane_lsbci"]["maximum_age_days"], 455
        )

    def test_backward_join_never_uses_future_or_stale_features(self) -> None:
        left = pd.DataFrame(
            {"observation_date": pd.to_datetime(["2024-01-08", "2024-01-15"])}
        )
        right = pd.DataFrame(
            {
                "feature_date": pd.to_datetime(["2024-01-05", "2024-01-19"]),
                "value": [1.0, 2.0],
            }
        )
        result, report = MODULE.backward_join(
            left, right, "observation_date", "feature_date", 6, "feature"
        )
        self.assertEqual(result.loc[0, "value"], 1.0)
        self.assertTrue(np.isnan(result.loc[1, "value"]))
        self.assertEqual(report["future_matches"], 0)

    def test_weekly_targets_are_forward_and_features_are_backward(self) -> None:
        scfi = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(
                    ["2024-01-08", "2024-01-15", "2024-01-22"]
                ),
                "scfi": [100.0, 110.0, 121.0],
            }
        )
        msdi = msdi_frame(["2024-01-05", "2024-01-12", "2024-01-19"])
        brent = pd.DataFrame(
            {
                "brent_observation_date": pd.to_datetime(
                    ["2024-01-05", "2024-01-12", "2024-01-19"]
                ),
                "brent_price": [70.0, 71.0, 72.0],
            }
        )
        external_controls = pd.DataFrame(
            columns=["variable", "availability_date", "value", "source_frequency"]
        )
        panel, _, report = MODULE.build_weekly_panel(
            scfi, msdi, brent, external_controls, 6
        )
        self.assertEqual(panel.loc[0, "msdi_age_days"], 3)
        self.assertEqual(panel.loc[0, "target_scfi_level_h1w"], 110.0)
        self.assertAlmostEqual(panel.loc[0, "target_scfi_log_change_h1w"], np.log(1.1))
        self.assertEqual(report["msdi_join"]["future_matches"], 0)

    def test_monthly_absence_is_nan_not_zero_disruption(self) -> None:
        weekly = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(["2024-01-08", "2024-02-05"]),
                "scfi": [100.0, 110.0],
                "brent_price": [70.0, 71.0],
            }
        )
        for column in MODULE.MSDI_BASE_COLUMNS:
            weekly[column] = [1.0, np.nan]
        monthly, _ = MODULE.build_monthly_panel(weekly)
        february = monthly[monthly["month_ending"].dt.month.eq(2)].iloc[0]
        self.assertEqual(february["msdi_available_weeks"], 0)
        self.assertTrue(np.isnan(february["peak_score_media_corrected_sum"]))
        self.assertTrue(np.isnan(february["article_count_sum"]))

    def test_direct_monthly_msdi_replaces_weekly_monthly_average(self) -> None:
        weekly = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(["2024-01-08", "2024-01-15"]),
                "scfi": [100.0, 101.0],
                "brent_price": [70.0, 71.0],
            }
        )
        for column in MODULE.MSDI_BASE_COLUMNS:
            weekly[column] = 99.0
        direct = pd.DataFrame(
            {
                "month_ending": pd.to_datetime(["2024-01-31"]),
                "article_count": [10],
                "event_count": [8],
                "source_count": [4],
                "effective_article_count": [8.0],
                "peak_score_media_corrected": [2.5],
                "breadth_score_media_corrected": [1.5],
            }
        )
        monthly, _ = MODULE.build_monthly_panel(weekly, direct)
        self.assertEqual(monthly.loc[0, "peak_score_media_corrected_mean"], 2.5)
        self.assertEqual(monthly.loc[0, "breadth_score_media_corrected_mean"], 1.5)
        self.assertEqual(monthly.loc[0, "article_count_sum"], 10)
        self.assertEqual(monthly.loc[0, "msdi_available_weeks"], 2)
        self.assertEqual(monthly.loc[0, "weekly_observations_in_month"], 2)
        self.assertEqual(monthly.loc[0, "direct_monthly_msdi_available"], 1)


if __name__ == "__main__":
    unittest.main()
