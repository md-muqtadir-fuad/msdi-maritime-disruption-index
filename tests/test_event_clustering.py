from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msdi.preprocessing.event_clustering import (  # noqa: E402
    EventClusteringConfig,
    cluster_event_articles,
    within_event_window,
)


class EventClusteringTests(unittest.TestCase):
    def test_week_and_calendar_month_windows_are_exact(self) -> None:
        self.assertTrue(
            within_event_window(
                pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-05"), 5, "weeks"
            )
        )
        self.assertFalse(
            within_event_window(
                pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-06"), 5, "weeks"
            )
        )
        self.assertTrue(
            within_event_window(
                pd.Timestamp("2024-01-31"), pd.Timestamp("2024-04-30"), 3, "months"
            )
        )
        self.assertFalse(
            within_event_window(
                pd.Timestamp("2024-01-31"), pd.Timestamp("2024-05-01"), 3, "months"
            )
        )

    def test_date_bound_and_complete_linkage_prevent_chaining(self) -> None:
        frame = pd.DataFrame(
            {
                "article_id": ["a", "b", "c", "d", "undated"],
                "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-20", None],
                "retrieved_title": ["A", "B", "C", "D", "Unknown"],
                "retrieved_text": ["event"] * 5,
                "source_domain": ["one", "two", "three", "four", "five"],
            }
        )
        angles = np.deg2rad([0.0, 15.0, 30.0, 0.0])
        embeddings = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
        config = EventClusteringConfig(
            similarity_threshold=0.95,
            day_window=3,
            validation_sample_size=4,
        )
        enriched, mapping, catalog, pairs, validation, audit = cluster_event_articles(
            frame, config, embeddings=embeddings
        )

        ids = enriched.set_index("article_id")["event_id"]
        self.assertEqual(ids["a"], ids["b"])
        self.assertNotEqual(ids["b"], ids["c"])
        self.assertNotEqual(ids["a"], ids["d"])
        self.assertTrue(ids["undated"].startswith("EVT_UNDATED_"))
        self.assertEqual(len(mapping), 5)
        self.assertEqual(int(catalog["article_count"].sum()), 5)
        self.assertEqual(audit["counts"]["multi_article_events"], 1)
        self.assertIn("human_same_event", validation.columns)
        self.assertIn("passes_threshold_0_85", validation.columns)
        self.assertIn("passes_threshold_0_92", validation.columns)
        self.assertTrue((pairs["day_difference"] <= 3).all())

    def test_event_ids_are_deterministic(self) -> None:
        frame = pd.DataFrame(
            {
                "article_id": ["x", "y"],
                "date": ["2024-02-01", "2024-02-01"],
                "headline": ["Same event one", "Same event two"],
                "retrieved_text": ["report one", "report two"],
                "source_domain": ["a.example", "b.example"],
            }
        )
        embeddings = np.array([[1.0, 0.0], [0.999, 0.001]], dtype=np.float32)
        config = EventClusteringConfig(similarity_threshold=0.9, validation_sample_size=1)
        first = cluster_event_articles(frame, config, embeddings=embeddings)[1]
        second = cluster_event_articles(frame, config, embeddings=embeddings)[1]
        self.assertEqual(first["event_id"].tolist(), second["event_id"].tolist())


if __name__ == "__main__":
    unittest.main()
