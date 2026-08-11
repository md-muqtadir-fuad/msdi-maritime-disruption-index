from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/13_build_disruption_index.py"
SPEC = importlib.util.spec_from_file_location("build_disruption_index", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


ALIASES = ["model_a", "model_b", "model_c", "model_d"]


def synthetic_scores() -> pd.DataFrame:
    rows = []
    for alias, overall, dimension in zip(ALIASES, [4, 2, 3, 1], [1, 2, 3, 4]):
        row = {
            "article_id": "article-1",
            "model_name": alias,
            "model_alias": alias,
            "scoring_status": "success",
            "date": "2024-01-01",
            "retrieved_publish_date": "2024-01-01T00:00:00Z",
            "headline": "Test",
            "retrieved_title": "Test article",
            "source_name": "Source",
            "source_domain": "example.com",
            "preferred_url": "https://example.com/a",
            "duplicate_group_id": "group-1",
            "duplicate_group_size": 4,
            "event_day": "2024-01-01",
            "event_id": "EVT_20240101_test",
            "event_cluster_size": 1,
            "event_source_count": 1,
            "event_span_days": 0,
            "quality_score": 1,
            MODULE.PRIMARY_COLUMN: overall,
        }
        row.update({column: dimension for column in MODULE.DIMENSION_COLUMNS})
        row.update({column: 1 for column in MODULE.OPERATIONAL_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


class BuildDisruptionIndexTests(unittest.TestCase):
    def test_event_profile_follows_aggregation_frequency(self) -> None:
        self.assertEqual(MODULE.resolve_event_profile("W-FRI"), "weekly")
        self.assertEqual(MODULE.resolve_event_profile("ME"), "monthly")
        self.assertEqual(MODULE.resolve_event_profile("W-FRI", "monthly"), "monthly")
        with self.assertRaises(ValueError):
            MODULE.resolve_event_profile("D")

    def test_two_measurement_paths_and_differences(self) -> None:
        logger = logging.getLogger("test_index")
        article, diagnostics = MODULE.build_article_scores(
            synthetic_scores(), ALIASES, minimum_models=2, duplicate_penalty="sqrt", logger=logger
        )
        row = article.iloc[0]
        self.assertEqual(row["peak_severity_score"], 2.5)
        self.assertEqual(row["breadth_severity_score"], 2.5)
        self.assertEqual(row["novelty_weight"], 0.5)
        self.assertEqual(
            row[f"{MODULE.PRIMARY_COLUMN}__diff__model_a_minus_model_b"], 2
        )
        self.assertEqual(diagnostics["articles_ensemble_eligible"], 1)

    def test_fixed_event_weight_totals_one_within_period(self) -> None:
        article = pd.DataFrame(
            {
                "article_id": ["a", "b", "c"],
                "publication_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"]
                ),
                "event_id": ["event-1", "event-1", "event-2"],
                "source_duplicate_weight": [0.5, 1.0, 1.0],
                "peak_severity_score": [2.0, 4.0, 1.0],
                "breadth_severity_score": [1.0, 3.0, 1.0],
            }
        )
        observed = MODULE.apply_period_event_weights(article, "W-FRI", "fixed")
        totals = observed.groupby("event_id")["novelty_weight"].sum()
        self.assertAlmostEqual(totals["event-1"], 1.0)
        self.assertAlmostEqual(totals["event-2"], 1.0)
        self.assertAlmostEqual(observed.loc[0, "novelty_weight"], 1 / 3)
        self.assertAlmostEqual(observed.loc[1, "novelty_weight"], 2 / 3)

    def test_ineligible_articles_do_not_dilute_event_weights(self) -> None:
        article = pd.DataFrame(
            {
                "article_id": ["eligible", "excluded"],
                "publication_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "event_id": ["event-1", "event-1"],
                "ensemble_eligible": [True, False],
                "source_duplicate_weight": [1.0, 1.0],
                "peak_severity_score": [2.0, 4.0],
                "breadth_severity_score": [1.0, 3.0],
            }
        )
        observed = MODULE.apply_period_event_weights(article, "W-FRI", "fixed")
        self.assertAlmostEqual(observed.loc[0, "novelty_weight"], 1.0)
        self.assertAlmostEqual(observed.loc[1, "novelty_weight"], 0.0)

    def test_monthly_primary_is_direct_sqrt_volume_then_causal_ema(self) -> None:
        article = pd.DataFrame(
            {
                "article_id": ["a", "b", "c"],
                "publication_date": pd.to_datetime(
                    ["2024-01-05", "2024-01-06", "2024-02-05"]
                ),
                "event_id": ["e1", "e2", "e3"],
                "source_domain": ["a.com", "b.com", "c.com"],
                "ensemble_eligible": [True, True, True],
                "novelty_weight": [1.0, 1.0, 1.0],
                "peak_severity_score": [2.0, 4.0, 2.0],
                "breadth_severity_score": [1.0, 3.0, 2.0],
                "peak_weighted_contribution": [2.0, 4.0, 2.0],
                "breadth_weighted_contribution": [1.0, 3.0, 2.0],
                f"{MODULE.PRIMARY_COLUMN}__ensemble_sd": [0.1, 0.2, 0.1],
                "dimension_disagreement_sd_mean": [0.1, 0.2, 0.1],
            }
        )
        for dimension in MODULE.DIMENSION_COLUMNS:
            article[f"{dimension}__ensemble_median"] = [1.0, 3.0, 2.0]
        observed = MODULE.aggregate_monthly_direct(article, "ME", 2, ema_span=3)
        january = observed.iloc[0]
        february = observed.iloc[1]
        january_base = 6.0 / np.sqrt(2.0)
        self.assertAlmostEqual(january["peak_score_sqrt_effective_volume"], january_base)
        self.assertAlmostEqual(january["peak_score_media_corrected"], january_base)
        self.assertAlmostEqual(
            february["peak_score_media_corrected"],
            0.5 * 2.0 + 0.5 * january_base,
        )

    def test_media_correction(self) -> None:
        raw = pd.Series([0.0, 10.0])
        count = pd.Series([0, 4])
        corrected = MODULE.media_correct(raw, count, "log")
        self.assertEqual(corrected.iloc[0], 0)
        self.assertAlmostEqual(corrected.iloc[1], 10 / np.log1p(4))

    def test_expanding_zscore_uses_only_prior_values(self) -> None:
        result = MODULE.expanding_realtime_zscore(pd.Series([1.0, 2.0, 3.0]), 2)
        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertTrue(np.isnan(result.iloc[1]))
        self.assertAlmostEqual(result.iloc[2], 3.0)

    def test_topsis_respects_benefit_and_cost_directions(self) -> None:
        alternatives = pd.DataFrame(
            {
                "model_alias": ["better", "worse"],
                "benefit": [10.0, 5.0],
                "cost": [1.0, 5.0],
            }
        )
        criteria = {
            "benefit": {"direction": "benefit", "weight": 0.5},
            "cost": {"direction": "cost", "weight": 0.5},
        }
        ranking, details = MODULE.topsis_rank(alternatives, criteria)
        self.assertEqual(ranking.iloc[0]["model_alias"], "better")
        self.assertEqual(ranking.iloc[0]["topsis_rank"], 1)
        self.assertEqual(len(details), 4)


if __name__ == "__main__":
    unittest.main()
