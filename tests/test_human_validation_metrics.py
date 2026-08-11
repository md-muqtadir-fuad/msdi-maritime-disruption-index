from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/12_human_validation_metrics.py"
SPEC = importlib.util.spec_from_file_location("human_validation_metrics", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def annotations() -> pd.DataFrame:
    rows = []
    labels = {
        "a1": [("relevant", 3), ("irrelevant", 0), ("relevant", 2)],
        "a2": [("relevant", 3), ("irrelevant", 0), ("uncertain", 1)],
    }
    for annotator, values in labels.items():
        for index, (relevance, score) in enumerate(values, start=1):
            row = {
                "article_id": f"id-{index}",
                "annotator_id": annotator,
                "relevance_label": relevance,
                MODULE.OVERALL_HUMAN_COLUMN: score,
            }
            row.update({column: score for column in MODULE.HUMAN_SCORE_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


class HumanValidationMetricsTests(unittest.TestCase):
    def test_consensus_is_deterministic_and_uses_median(self) -> None:
        result = MODULE.consensus_labels(annotations()).set_index("article_id")
        self.assertEqual(result.loc["id-1", "relevance_label"], "relevant")
        self.assertEqual(result.loc["id-3", "relevance_label"], "uncertain")
        self.assertEqual(result.loc["id-3", MODULE.OVERALL_HUMAN_COLUMN], 1.5)

    def test_perfect_interval_agreement_has_alpha_one(self) -> None:
        values = pd.DataFrame({"a": [0, 1, 4], "b": [0, 1, 4]})
        self.assertAlmostEqual(MODULE.krippendorff_alpha(values, "interval"), 1.0)

    def test_pairwise_metrics_cover_relevance_and_scores(self) -> None:
        result = MODULE.pairwise_human_metrics(annotations())
        expected = 1 + len(MODULE.HUMAN_SCORE_COLUMNS) + 1
        self.assertEqual(len(result), expected)
        relevance = result[result["variable"].eq("relevance_label")].iloc[0]
        self.assertEqual(relevance["n_complete"], 3)
        self.assertAlmostEqual(relevance["exact_agreement"], 2 / 3)

    def test_icc_requires_at_least_two_complete_items_and_raters(self) -> None:
        self.assertTrue(
            np.isnan(MODULE.icc_two_way_random_single(pd.DataFrame({"a": [1.0]})))
        )

    def test_event_threshold_metrics_use_cosine_and_human_labels(self) -> None:
        master = pd.DataFrame(
            {
                "event_profile": ["weekly"] * 4,
                "article_id_a": ["a", "b", "c", "d"],
                "article_id_b": ["e", "f", "g", "h"],
                "cosine_similarity": [0.95, 0.89, 0.87, 0.80],
            }
        )
        reference = master[MODULE.EVENT_PAIR_KEYS].copy()
        reference["human_same_event"] = [1, 1, 0, 0]
        result = MODULE.event_threshold_metrics(master, reference, [0.88])
        weekly = result[result["event_profile"].eq("weekly")].iloc[0]
        self.assertEqual(weekly["true_positive"], 2)
        self.assertEqual(weekly["true_negative"], 2)
        self.assertEqual(weekly["f1"], 1.0)
        self.assertTrue(weekly["is_interim_operational_threshold"])


if __name__ == "__main__":
    unittest.main()
