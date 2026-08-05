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


ALIASES = ["model_a", "model_b", "model_c"]


def synthetic_scores() -> pd.DataFrame:
    rows = []
    for alias, overall, dimension in zip(ALIASES, [4, 2, 3], [1, 2, 3]):
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
            "quality_score": 1,
            MODULE.PRIMARY_COLUMN: overall,
        }
        row.update({column: dimension for column in MODULE.DIMENSION_COLUMNS})
        row.update({column: 1 for column in MODULE.OPERATIONAL_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


class BuildDisruptionIndexTests(unittest.TestCase):
    def test_two_measurement_paths_and_differences(self) -> None:
        logger = logging.getLogger("test_index")
        article, diagnostics = MODULE.build_article_scores(
            synthetic_scores(), ALIASES, minimum_models=2, duplicate_penalty="sqrt", logger=logger
        )
        row = article.iloc[0]
        self.assertEqual(row["peak_severity_score"], 3)
        self.assertEqual(row["breadth_severity_score"], 2)
        self.assertEqual(row["novelty_weight"], 0.5)
        self.assertEqual(
            row[f"{MODULE.PRIMARY_COLUMN}__diff__model_a_minus_model_b"], 2
        )
        self.assertEqual(diagnostics["articles_ensemble_eligible"], 1)

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
