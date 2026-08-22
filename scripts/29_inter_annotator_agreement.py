"""Export standalone inter-annotator agreement metrics from completed labels."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SCRIPT = ROOT / "scripts/12_human_validation_and_llm_benchmark.py"


def load_validation_module():
    spec = importlib.util.spec_from_file_location("human_validation_metrics", VALIDATION_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute inter-annotator agreement only.")
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/inter_annotator_agreement")
    parser.add_argument("--allow-pseudo-annotations", action="store_true")
    args = parser.parse_args(); module = load_validation_module(); annotations = module.load_annotations(args.annotations)
    pseudo = "annotation_provenance" in annotations and annotations.annotation_provenance.astype(str).str.contains("pseudo", case=False, na=False).any()
    if pseudo and not args.allow_pseudo_annotations: raise ValueError("Pseudo annotations require --allow-pseudo-annotations and remain test-only.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairwise = module.pairwise_human_metrics(annotations); overall = module.overall_human_metrics(annotations)
    pairwise.to_csv(args.output_dir / "pairwise_agreement.csv", index=False); overall.to_csv(args.output_dir / "overall_agreement.csv", index=False)
    (args.output_dir / "agreement_manifest.json").write_text(json.dumps({"articles": int(annotations.article_id.nunique()), "annotators": sorted(annotations.annotator_id.astype(str).unique()), "status": "synthetic_test_only" if pseudo else "human_annotation_agreement", "publication_eligible": not pseudo}, indent=2), encoding="utf-8")
    print(f"Agreement metrics written: {args.output_dir}")


if __name__ == "__main__": main()
