from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/17_llm_scfi_comparison_monthly.py"
)
SPEC = importlib.util.spec_from_file_location("monthly_llm_scfi_comparison", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


ALIASES = ["model_a", "model_b", "model_c", "model_d"]
SOURCES = [*ALIASES, *MODULE.SCORE_HYBRID_SOURCES]


def _weekly_rows(
    month: str,
    article_counts: list[float],
    effective_counts: list[float],
    weighted_sums: list[float],
    *,
    omit_positions: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Create one synthetic month in the safe Monday-aligned weekly schema."""

    period = pd.Period(month, freq="M")
    mondays = MODULE.expected_mondays(period)
    expected_length = len(mondays)
    if not (
        len(article_counts) == len(effective_counts) == len(weighted_sums)
        == expected_length
    ):
        raise ValueError(f"{month} has {expected_length} Mondays")

    rows: list[dict[str, object]] = []
    omitted = set(omit_positions)
    for position, observation_date in enumerate(mondays):
        if position in omitted:
            continue
        article_count = float(article_counts[position])
        effective_count = float(effective_counts[position])
        weighted_sum = float(weighted_sums[position])
        weekly_index = (
            weighted_sum / math.log1p(article_count)
            if article_count > 0
            else 0.0
        )
        row: dict[str, object] = {
            "observation_date": observation_date,
            # A Friday signal is available to the following Monday.  In
            # particular, 2024-01-01 carries a 2023-12-29 signal date.
            "llm_week_ending": observation_date - pd.Timedelta(days=3),
            "llm_signal_age_days": 3.0,
            "llm_comparison_complete_case": True,
            "scfi": 1000.0 + position,
            "brent_price": 80.0 + position,
            "llm_common_article_count": article_count,
            "llm_common_effective_article_count": effective_count,
        }
        for source in SOURCES:
            for signal in MODULE.SIGNALS:
                row[MODULE.signal_raw_column(source, signal)] = weekly_index
        rows.append(row)
    return pd.DataFrame(rows)


def _month_with_pdf_base(
    month: str,
    base_value: float,
    *,
    omit_positions: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Build a positive-volume month whose PDF square-root base is exact."""

    monday_count = len(MODULE.expected_mondays(pd.Period(month, freq="M")))
    contribution = base_value / math.sqrt(monday_count)
    return _weekly_rows(
        month,
        [1.0] * monday_count,
        [1.0] * monday_count,
        [contribution] * monday_count,
        omit_positions=omit_positions,
    )


def _synthetic_direct_monthly_forecasts(periods: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=periods, freq="ME")
    rows: list[dict[str, object]] = []
    for construction_index, construction in enumerate(MODULE.MONTHLY_INDEX_METHODS):
        for alias_index, alias in enumerate(ALIASES):
            for position, origin in enumerate(dates):
                actual = 0.01 * math.sin(position / 3)
                prediction = actual + (alias_index - 1) * 0.001 + construction_index * 0.0001
                target = dates[position + 1] if position + 1 < periods else origin + pd.offsets.MonthEnd(1)
                rows.append(
                    {
                        "origin_position": position,
                        "origin_date": origin,
                        "target_date": target,
                        "horizon_months": 1,
                        "model_id": f"ridge__monthly__{construction}__llm__{alias}__overall_disruption_max",
                        "feature_set": "synthetic",
                        "index_construction": construction,
                        "source_type": "llm",
                        "score_source": alias,
                        "signal_name": MODULE.PRIMARY_SCORE_COLUMN,
                        "hybrid_level": "none",
                        "price_overlap_risk": False,
                        "regime_specific_risk": False,
                        "actual_log_change": actual,
                        "predicted_log_change": prediction,
                        "forecast_error": actual - prediction,
                        "current_scfi": 100.0,
                        "actual_scfi_level": 100.0 * math.exp(actual),
                        "predicted_scfi_level": 100.0 * math.exp(prediction),
                        "level_error": 100.0 * (math.exp(actual) - math.exp(prediction)),
                        "training_rows": 60 + position,
                        "training_feature_end_date": origin - pd.offsets.MonthEnd(2),
                        "training_label_end_date": origin - pd.offsets.MonthEnd(1),
                        "model_fit_origin_date": origin,
                    }
                )
    return pd.DataFrame(rows)


class MonthlyLlmScfiComparisonTests(unittest.TestCase):
    def test_weekly_manifest_accepts_current_event_weight_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel = root / "weekly_panel.csv"
            panel.write_text("observation_date\n", encoding="utf-8")
            manifest = root / "weekly_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "version": "test",
                        "common_article_cohort": {
                            "common_complete_articles": 4,
                            "expected_common_complete_articles": 4,
                            "common_cohort_size_check_passed": True,
                        },
                        "design": {
                            "common_cohort_is_only_score_input": True,
                            "broader_consensus_index_used_as_predictor": False,
                            "model_specific_unequal_coverage_indices_used": False,
                            "model_aliases": ALIASES,
                            "weekly_common_article_count_sum": 4,
                            "media_correction": (
                                "novelty-weighted sum / log1p(common article count)"
                            ),
                            "score_scale": [0, 4],
                            "novelty_weighting": (
                                MODULE.SUPPORT.NOVELTY_WEIGHTING_PROVENANCE
                            ),
                        },
                        "outputs": {"aligned_panel": str(panel.resolve())},
                    }
                ),
                encoding="utf-8",
            )
            _, audit = MODULE.load_and_validate_weekly_manifest(
                manifest, ALIASES, 4, panel
            )
            self.assertTrue(audit["common_cohort_check_passed"])

    def test_direct_configuration_product_is_complete_and_unique(self) -> None:
        configurations = MODULE.build_direct_configurations(ALIASES)
        expected = (
            len(MODULE.MONTHLY_INDEX_METHODS)
            * len(SOURCES)
            * len(MODULE.SIGNALS)
        )
        self.assertEqual(len(configurations), expected)
        self.assertEqual(len({item.model_id for item in configurations}), expected)
        self.assertEqual(
            {item.index_construction for item in configurations},
            set(MODULE.MONTHLY_INDEX_METHODS),
        )

    def test_monthly_training_positions_are_horizon_purged(self) -> None:
        panel = pd.DataFrame(
            {
                "llm_monthly_comparison_complete_case": [True] * 20,
                "target": np.arange(20, dtype=float),
            }
        )
        positions = MODULE.training_positions(panel, 10, 3, "target")
        self.assertEqual(int(positions[-1]), 7)
        self.assertLessEqual(int(positions[-1]) + 3, 10)

    def test_forecast_hybrids_never_mix_monthly_constructions(self) -> None:
        direct = _synthetic_direct_monthly_forecasts(30)
        hybrids, weights = MODULE.build_forecast_hybrids(
            direct, ALIASES, minimum_stacking_history=12
        )
        self.assertEqual(
            set(hybrids["index_construction"]), set(MODULE.MONTHLY_INDEX_METHODS)
        )
        self.assertTrue(
            hybrids.apply(
                lambda row: str(row["index_construction"]) in str(row["model_id"]),
                axis=1,
            ).all()
        )
        self.assertFalse(
            hybrids.duplicated(
                ["horizon_months", "index_construction", "model_id", "origin_date"]
            ).any()
        )
        sums = weights.groupby(
            ["horizon_months", "index_construction", "signal_name", "origin_date"]
        )["weight"].sum()
        np.testing.assert_allclose(sums, 1.0)
        self.assertTrue(
            (
                pd.to_datetime(weights["stacking_history_label_end"])
                <= pd.to_datetime(weights["origin_date"])
            ).all()
        )

    def test_duplicate_monthly_forecast_origins_are_rejected(self) -> None:
        forecasts = pd.DataFrame(
            {
                "horizon_months": [1, 1],
                "model_id": ["model", "model"],
                "origin_date": pd.to_datetime(["2024-01-31", "2024-01-31"]),
            }
        )
        with self.assertRaises(AssertionError):
            MODULE.validate_common_origin_sets(forecasts, ["model"])

    def test_pdf_path_reconstructs_weighted_numerator_and_effective_volume(self) -> None:
        weekly = _weekly_rows(
            "2024-01",
            article_counts=[1.0, 4.0, 0.0, 9.0, 1.0],
            effective_counts=[1.0, 2.0, 0.0, 3.0, 1.0],
            weighted_sums=[2.0, 6.0, 0.0, 3.0, 1.0],
        )
        monthly, _, _ = MODULE.aggregate_complete_months(weekly, ALIASES)

        source = ALIASES[0]
        signal = MODULE.PRIMARY_SCORE_COLUMN
        mean_column = MODULE.monthly_signal_column(
            MODULE.MEAN_SAFE_WEEKLY, source, signal
        )
        base_column = MODULE.pdf_base_column(source, signal)
        pdf_column = MODULE.monthly_signal_column(
            MODULE.PDF_SQRT_EMA3, source, signal
        )

        # Backward compatibility: the established monthly path keeps its
        # original raw signal name and remains the mean of safe weekly indices.
        self.assertEqual(mean_column, MODULE.signal_raw_column(source, signal))
        expected_weekly_mean = np.mean(
            [
                2.0 / math.log(2.0),
                6.0 / math.log(5.0),
                0.0,
                3.0 / math.log(10.0),
                1.0 / math.log(2.0),
            ]
        )
        self.assertAlmostEqual(float(monthly.at[0, mean_column]), expected_weekly_mean)

        # The weekly media correction is reversed before monthly aggregation:
        # sum_w[I_w log(1 + N_w)] / sqrt(sum_w effective_N_w).
        self.assertEqual(float(monthly.at[0, "llm_common_article_count"]), 15.0)
        self.assertEqual(
            float(monthly.at[0, "llm_common_effective_article_count"]), 7.0
        )
        expected_pdf_base = 12.0 / math.sqrt(7.0)
        self.assertAlmostEqual(float(monthly.at[0, base_column]), expected_pdf_base)
        # The first observation initializes adjust=False EMA at the base value.
        self.assertAlmostEqual(float(monthly.at[0, pdf_column]), expected_pdf_base)

    def test_causal_ema_uses_alpha_half_restarts_and_ignores_future_changes(self) -> None:
        values = pd.Series([1.0, 2.0, np.nan, 8.0, 10.0])
        actual = MODULE.causal_ema_restart_on_gap(values, span=3)
        expected = pd.Series([1.0, 1.5, np.nan, 8.0, 9.0])
        pd.testing.assert_series_equal(actual, expected)

        changed_future = values.copy()
        changed_future.iloc[-1] = 10_000.0
        mutated = MODULE.causal_ema_restart_on_gap(changed_future, span=3)
        pd.testing.assert_series_equal(actual.iloc[:-1], mutated.iloc[:-1])
        self.assertAlmostEqual(float(mutated.iloc[-1]), 5004.0)

    def test_incomplete_month_is_missing_and_pdf_ema_restarts_after_gap(self) -> None:
        weekly = pd.concat(
            [
                _month_with_pdf_base("2024-01", 1.0),
                _month_with_pdf_base("2024-02", 2.0, omit_positions=(1,)),
                _month_with_pdf_base("2024-03", 3.0),
            ],
            ignore_index=True,
        )
        monthly, _, _ = MODULE.aggregate_complete_months(weekly, ALIASES)
        source = ALIASES[0]
        signal = MODULE.PRIMARY_SCORE_COLUMN
        mean_column = MODULE.monthly_signal_column(
            MODULE.MEAN_SAFE_WEEKLY, source, signal
        )
        base_column = MODULE.pdf_base_column(source, signal)
        pdf_column = MODULE.monthly_signal_column(
            MODULE.PDF_SQRT_EMA3, source, signal
        )

        self.assertEqual(
            monthly["monthly_llm_complete"].tolist(), [True, False, True]
        )
        self.assertTrue(pd.isna(monthly.at[1, mean_column]))
        self.assertTrue(pd.isna(monthly.at[1, base_column]))
        self.assertTrue(pd.isna(monthly.at[1, pdf_column]))
        np.testing.assert_allclose(
            monthly[base_column].to_numpy(), [1.0, np.nan, 3.0], equal_nan=True
        )
        # March starts a new causal EMA run; January is not bridged across the
        # unobserved February measurement.
        np.testing.assert_allclose(
            monthly[pdf_column].to_numpy(), [1.0, np.nan, 3.0], equal_nan=True
        )

    def test_zero_effective_volume_is_missing_for_pdf_but_not_old_path(self) -> None:
        april_mondays = len(
            MODULE.expected_mondays(pd.Period("2024-04", freq="M"))
        )
        weekly = pd.concat(
            [
                _weekly_rows(
                    "2024-04",
                    [0.0] * april_mondays,
                    [0.0] * april_mondays,
                    [0.0] * april_mondays,
                ),
                _month_with_pdf_base("2024-05", 2.0),
            ],
            ignore_index=True,
        )
        monthly, _, _ = MODULE.aggregate_complete_months(weekly, ALIASES)
        source = ALIASES[0]
        signal = MODULE.PRIMARY_SCORE_COLUMN
        mean_column = MODULE.monthly_signal_column(
            MODULE.MEAN_SAFE_WEEKLY, source, signal
        )
        base_column = MODULE.pdf_base_column(source, signal)
        pdf_column = MODULE.monthly_signal_column(
            MODULE.PDF_SQRT_EMA3, source, signal
        )

        self.assertTrue(bool(monthly.at[0, "monthly_llm_complete"]))
        self.assertEqual(float(monthly.at[0, mean_column]), 0.0)
        self.assertTrue(pd.isna(monthly.at[0, base_column]))
        self.assertTrue(pd.isna(monthly.at[0, pdf_column]))
        # Missing zero-volume April is a gap, so May initializes at its base.
        self.assertAlmostEqual(float(monthly.at[1, base_column]), 2.0)
        self.assertAlmostEqual(float(monthly.at[1, pdf_column]), 2.0)

    def test_availability_monday_determines_month_assignment(self) -> None:
        monday_count = len(
            MODULE.expected_mondays(pd.Period("2024-01", freq="M"))
        )
        weekly = _weekly_rows(
            "2024-01",
            [1.0] * monday_count,
            [1.0] * monday_count,
            [4.0, *([0.0] * (monday_count - 1))],
        )
        self.assertEqual(
            weekly.loc[0, "llm_week_ending"], pd.Timestamp("2023-12-29")
        )

        monthly, _, report = MODULE.aggregate_complete_months(weekly, ALIASES)
        source = ALIASES[0]
        signal = MODULE.PRIMARY_SCORE_COLUMN
        mean_column = MODULE.monthly_signal_column(
            MODULE.MEAN_SAFE_WEEKLY, source, signal
        )
        base_column = MODULE.pdf_base_column(source, signal)

        self.assertEqual(report["calendar_months"], 1)
        self.assertEqual(monthly.at[0, "month_ending"], pd.Timestamp("2024-01-31"))
        self.assertAlmostEqual(
            float(monthly.at[0, mean_column]), 4.0 / math.log(2.0) / monday_count
        )
        self.assertAlmostEqual(
            float(monthly.at[0, base_column]), 4.0 / math.sqrt(monday_count)
        )

    def test_monthly_long_has_one_unique_row_per_construction_and_signal(self) -> None:
        weekly = _month_with_pdf_base("2024-01", 1.0)
        monthly, monthly_long, _ = MODULE.aggregate_complete_months(weekly, ALIASES)

        key = [
            "month_ending",
            "index_construction",
            "source_type",
            "score_source",
            "signal_name",
        ]
        self.assertTrue(set(key).issubset(monthly_long.columns))
        self.assertFalse(monthly_long.duplicated(key).any())
        self.assertEqual(
            set(monthly_long["index_construction"]),
            {MODULE.MEAN_SAFE_WEEKLY, MODULE.PDF_SQRT_EMA3},
        )
        expected_rows = (
            len(monthly)
            * len(SOURCES)
            * len(MODULE.SIGNALS)
            * 2
        )
        self.assertEqual(len(monthly_long), expected_rows)

        wide_signal_columns: list[str] = []
        for construction in (MODULE.MEAN_SAFE_WEEKLY, MODULE.PDF_SQRT_EMA3):
            for source in SOURCES:
                for signal in MODULE.SIGNALS:
                    column = MODULE.monthly_signal_column(
                        construction, source, signal
                    )
                    self.assertIn(column, monthly.columns)
                    wide_signal_columns.append(column)
        self.assertEqual(len(wide_signal_columns), len(set(wide_signal_columns)))

    def test_repository_monthly_invariants_and_formula_reconstruction(self) -> None:
        aliases = MODULE.SUPPORT.load_model_aliases(MODULE.DEFAULT_MODEL_CONFIG)
        header = pd.read_csv(MODULE.DEFAULT_WEEKLY_PANEL, nrows=0).columns
        required = MODULE.signal_raw_column("glm_4_9b_0414", MODULE.PRIMARY_SCORE_COLUMN)
        if required not in header:
            self.skipTest("requires Stages 16–17 to regenerate four-LLM panels")
        weekly, weekly_report = MODULE.load_safe_weekly_panel(
            MODULE.DEFAULT_WEEKLY_PANEL,
            aliases,
            (
                MODULE.EXPECTED_ALIGNED_WEEKLY_ARTICLE_COUNT
                if MODULE.EXPECTED_ALIGNED_WEEKLY_ARTICLE_COUNT > 0
                else None
            ),
        )
        monthly, monthly_long, report = MODULE.aggregate_complete_months(weekly, aliases)

        if MODULE.EXPECTED_ALIGNED_WEEKLY_ARTICLE_COUNT > 0:
            self.assertEqual(
                weekly_report["aligned_weekly_article_count_sum"],
                MODULE.EXPECTED_ALIGNED_WEEKLY_ARTICLE_COUNT,
            )
        else:
            self.assertGreater(weekly_report["aligned_weekly_article_count_sum"], 0)
        if MODULE.EXPECTED_LLM_COMPLETE_MONTHS > 0:
            self.assertEqual(
                report["llm_complete_months"], MODULE.EXPECTED_LLM_COMPLETE_MONTHS
            )
        complete = monthly[monthly["monthly_llm_complete"].astype(bool)]
        if MODULE.EXPECTED_COMPLETE_MONTH_ARTICLE_COUNT > 0:
            self.assertEqual(
                int(round(float(complete["llm_common_article_count"].sum()))),
                MODULE.EXPECTED_COMPLETE_MONTH_ARTICLE_COUNT,
            )
        else:
            self.assertGreater(
                int(round(float(complete["llm_common_article_count"].sum()))), 0
            )
        complete_long = monthly_long[
            monthly_long["monthly_analysis_complete_case"].astype(bool)
        ]
        self.assertTrue(np.isfinite(complete_long["monthly_signal_value"]).all())

        row = complete.iloc[0]
        period = row["month_ending"].to_period("M")
        weekly_month = weekly[weekly["observation_date"].dt.to_period("M").eq(period)]
        source = aliases[0]
        signal = MODULE.PRIMARY_SCORE_COLUMN
        weekly_column = MODULE.signal_raw_column(source, signal)
        expected = float(
            (
                weekly_month[weekly_column]
                * np.log1p(weekly_month["llm_common_article_count"])
            ).sum()
            / math.sqrt(weekly_month["llm_common_effective_article_count"].sum())
        )
        actual = float(row[MODULE.pdf_base_column(source, signal)])
        self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
