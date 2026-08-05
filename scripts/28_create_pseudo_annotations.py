"""Create deterministic synthetic annotation files for pipeline testing only.

These files are explicitly marked pseudo and are rejected by the human
validation stage unless its explicit testing flag is supplied. They are never
evidence of human or construct validity.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "data/04_validation/human_goldset"
OUTPUT = ROOT / "data/04_validation/pseudo_test_only"
DIMENSIONS = ["port_disruption_score", "global_container_movement_disruption_score", "freight_cost_disruption_score", "supply_disruption_score", "production_disruption_score", "covid19_related_disruption_score", "demand_disruption_score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create pseudo labels for plumbing tests only.")
    parser.add_argument("--template-dir", type=Path, default=TEMPLATES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def seed_for(article_id: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{article_id}".encode()).hexdigest()[:16], 16)


def label(frame: pd.DataFrame, annotator: int, seed: int) -> pd.DataFrame:
    output = frame.copy()
    for column in ("relevance_label", "dominant_dimension", "evidence_span", "notes"):
        output[column] = output[column].astype("object")
    for index, row in output.iterrows():
        base = np.random.default_rng(seed_for(str(row.article_id), seed))
        relevance = base.choice(["relevant", "irrelevant", "uncertain"], p=[.55, .30, .15])
        scores = base.integers(0, 5, size=len(DIMENSIONS)) if relevance == "relevant" else np.zeros(len(DIMENSIONS), dtype=int)
        noise = np.random.default_rng(seed_for(f"{row.article_id}:{annotator}", seed))
        if noise.random() < .12: scores[noise.integers(0, len(scores))] = noise.integers(0, 5)
        if noise.random() < .08: relevance = noise.choice(["relevant", "irrelevant", "uncertain"])
        output.at[index, "relevance_label"] = relevance
        for column, score in zip(DIMENSIONS, scores): output.at[index, f"human_{column}"] = int(score)
        output.at[index, "human_overall_disruption_score"] = int(scores.max())
        output.at[index, "dominant_dimension"] = DIMENSIONS[int(scores.argmax())] if scores.max() else ""
        output.at[index, "evidence_span"] = "SYNTHETIC TEST LABEL — not human evidence."
        output.at[index, "annotator_confidence"] = round(float(noise.uniform(.55, .95)), 2)
        output.at[index, "notes"] = "Pseudo annotation for pipeline testing only."
    output["annotation_provenance"] = "pseudo_test_only"
    return output


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    for annotator in (1, 2):
        template = args.template_dir / f"annotator_{annotator}_template.csv"
        if not template.is_file(): raise FileNotFoundError(template)
        result = label(pd.read_csv(template, low_memory=False), annotator, args.seed)
        result.to_csv(args.output_dir / f"annotator_{annotator}_pseudo.csv", index=False)
    (args.output_dir / "README.txt").write_text("SYNTHETIC PIPELINE-TEST DATA ONLY. These are not human annotations and cannot support a publication claim.\n", encoding="utf-8")
    print(f"Created pseudo annotations: {args.output_dir}")


if __name__ == "__main__": main()
