from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/11_merge_llm_scores.py"
SPEC = importlib.util.spec_from_file_location("merge_llm_scores", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MergeLlmScoresEventTests(unittest.TestCase):
    def test_authoritative_event_map_is_attached_to_every_model(self) -> None:
        scores = pd.DataFrame(
            {
                "article_id": ["a", "a", "b", "b"],
                "model_name": ["m1", "m2", "m1", "m2"],
                "event_id": ["stale"] * 4,
            }
        )
        event_map = pd.DataFrame(
            {
                "article_id": ["a", "b"],
                "event_day": ["2024-01-01", "2024-01-02"],
                **{
                    f"{column}_{profile}": values
                    for profile in ("weekly", "monthly")
                    for column, values in {
                        "event_id": ["event-1", "event-1"],
                        "event_cluster_size": [2, 2],
                        "event_source_count": [2, 2],
                        "event_span_days": [1, 1],
                        "event_weight_fixed_total_descriptive": [0.5, 0.5],
                        "event_weight_sqrt_descriptive": [2 ** -0.5, 2 ** -0.5],
                    }.items()
                },
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event_map.csv"
            event_map.to_csv(path, index=False)
            observed, audit = MODULE.enrich_with_event_map(scores, path, "article_id")
        self.assertEqual(set(observed["event_id_weekly"]), {"event-1"})
        self.assertEqual(set(observed["event_id_monthly"]), {"event-1"})
        self.assertEqual(audit["score_rows_enriched"], 4)
        self.assertEqual(audit["mapping_events_weekly"], 1)
        self.assertEqual(audit["mapping_events_monthly"], 1)

    def test_missing_article_assignment_fails(self) -> None:
        scores = pd.DataFrame({"article_id": ["a", "missing"], "model_name": ["m1", "m1"]})
        event_map = pd.DataFrame(
            {
                "article_id": ["a"],
                "event_day": ["2024-01-01"],
                **{
                    f"{column}_{profile}": values
                    for profile in ("weekly", "monthly")
                    for column, values in {
                        "event_id": ["event-1"],
                        "event_cluster_size": [1],
                        "event_source_count": [1],
                        "event_span_days": [0],
                        "event_weight_fixed_total_descriptive": [1.0],
                        "event_weight_sqrt_descriptive": [1.0],
                    }.items()
                },
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event_map.csv"
            event_map.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                MODULE.enrich_with_event_map(scores, path, "article_id")


if __name__ == "__main__":
    unittest.main()
