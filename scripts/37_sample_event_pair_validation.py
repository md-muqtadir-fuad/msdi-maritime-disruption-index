#!/usr/bin/env python3
"""
Event Pair Deduplication Validation Sampling Pipeline (N = 300 Pairs)
====================================================================
Generates double-blind human annotation packages for the 4 candidate threshold scores:
  - 0.82 (Loose clustering regime, 75 pairs)
  - 0.85 (Standard coreference regime, 75 pairs)
  - 0.88 (Strict coreference regime, 75 pairs)
  - 0.90 (Verbatim syndication regime, 75 pairs)

Exports:
  1. Per-threshold subfolders in validation/event_pairs/inputs/:
     - by_threshold/0_82/event_pair_annotator_1.csv & event_pair_annotator_2.csv
     - by_threshold/0_85/event_pair_annotator_1.csv & event_pair_annotator_2.csv
     - by_threshold/0_88/event_pair_annotator_1.csv & event_pair_annotator_2.csv
     - by_threshold/0_90/event_pair_annotator_1.csv & event_pair_annotator_2.csv
  2. Master 300-pair combined sheets:
     - all_300_pairs_annotator_1.csv
     - all_300_pairs_annotator_2.csv
     - event_pair_master_audit_300.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = PROJECT_ROOT / "validation"
EVENT_DIR = VAL_DIR / "event_pairs"
INPUT_DIR = EVENT_DIR / "inputs"

INPUT_PAIR_AUDIT = PROJECT_ROOT / "data/03_processed/news_full_text/pre_llm_filter/event_dedup/canonical_event_pair_audit_by_profile.csv"

THRESHOLDS = [0.82, 0.85, 0.88, 0.90]
PAIRS_PER_THRESHOLD = 75
RANDOM_SEED = 42


def sample_threshold_pairs(df_pairs: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Samples 75 candidate pairs per threshold band across the 4 regimes."""
    rng = np.random.default_rng(seed)

    # Use monthly profile pairs (standard 3-month clustering window)
    monthly = df_pairs[df_pairs["event_profile"] == "monthly"].copy()

    # Define mutually exclusive similarity bands centered on target thresholds
    strata = {
        0.82: monthly[(monthly["cosine_similarity"] >= 0.80) & (monthly["cosine_similarity"] < 0.835)],
        0.85: monthly[(monthly["cosine_similarity"] >= 0.835) & (monthly["cosine_similarity"] < 0.865)],
        0.88: monthly[(monthly["cosine_similarity"] >= 0.865) & (monthly["cosine_similarity"] < 0.890)],
        0.90: monthly[(monthly["cosine_similarity"] >= 0.890)],
    }

    sampled_list = []
    for th, pool in strata.items():
        n_avail = len(pool)
        n_sample = min(PAIRS_PER_THRESHOLD, n_avail)
        sample = pool.sample(n=n_sample, random_state=seed).copy()
        sample["target_threshold_band"] = th
        sample["stratum_label"] = f"Band_{th:.2f}"
        sampled_list.append(sample)
        print(f"Stratum {th:.2f}: Pool = {n_avail:4d} | Sampled = {n_sample:2d}")

    master_df = pd.concat(sampled_list, ignore_index=True)
    master_df["pair_id"] = [f"PAIR_{i+1:03d}" for i in range(len(master_df))]
    return master_df


def export_annotator_sheets(master_df: pd.DataFrame, base_out_dir: Path):
    """Exports clean, double-blinded spreadsheets for Human 1 and Human 2."""
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Core metadata for annotator review
    blinded_cols = [
        "pair_id",
        "title_a",
        "title_b",
        "text_excerpt_a",
        "text_excerpt_b",
        "event_day_a",
        "event_day_b",
        "day_difference",
        "source_a",
        "source_b",
    ]

    # Save Master Audit (with true cosine_similarity and article IDs)
    master_path = base_out_dir / "event_pair_master_audit_300.csv"
    master_df.to_csv(master_path, index=False, encoding="utf-8-sig")
    print(f"\n[EXPORT] Master Manifest: {master_path.relative_to(PROJECT_ROOT)} (N = {len(master_df)})")

    # 1. Export Per-Threshold Folders
    for th in THRESHOLDS:
        th_folder = base_out_dir / "by_threshold" / f"{th:.2f}".replace(".", "_")
        th_folder.mkdir(parents=True, exist_ok=True)

        th_sub = master_df[master_df["target_threshold_band"] == th].copy()

        # Annotator 1 (shuffled with seed 101)
        ann1_th = th_sub[blinded_cols].sample(frac=1.0, random_state=101).copy()
        ann1_th["annotator_id"] = "human_annotator_1"
        ann1_th["same_event"] = ""  # 1 = Same Event, 0 = Different Event
        ann1_th["confidence"] = ""  # High, Medium, Low
        ann1_th["notes"] = ""
        ann1_path = th_folder / "event_pair_annotator_1.csv"
        ann1_th.to_csv(ann1_path, index=False, encoding="utf-8-sig")

        # Annotator 2 (shuffled with seed 202)
        ann2_th = th_sub[blinded_cols].sample(frac=1.0, random_state=202).copy()
        ann2_th["annotator_id"] = "human_annotator_2"
        ann2_th["same_event"] = ""
        ann2_th["confidence"] = ""
        ann2_th["notes"] = ""
        ann2_path = th_folder / "event_pair_annotator_2.csv"
        ann2_th.to_csv(ann2_path, index=False, encoding="utf-8-sig")

        print(f"  - [{th:.2f} Folder] -> {th_folder.name}/ (Annotator 1: {len(ann1_th)} rows | Annotator 2: {len(ann2_th)} rows)")

    # 2. Export All-In-One 300 Pairs Combined Files
    all_ann1 = master_df[blinded_cols].sample(frac=1.0, random_state=101).copy()
    all_ann1["annotator_id"] = "human_annotator_1"
    all_ann1["same_event"] = ""
    all_ann1["confidence"] = ""
    all_ann1["notes"] = ""
    all_ann1_path = base_out_dir / "all_300_pairs_annotator_1.csv"
    all_ann1.to_csv(all_ann1_path, index=False, encoding="utf-8-sig")

    all_ann2 = master_df[blinded_cols].sample(frac=1.0, random_state=202).copy()
    all_ann2["annotator_id"] = "human_annotator_2"
    all_ann2["same_event"] = ""
    all_ann2["confidence"] = ""
    all_ann2["notes"] = ""
    all_ann2_path = base_out_dir / "all_300_pairs_annotator_2.csv"
    all_ann2.to_csv(all_ann2_path, index=False, encoding="utf-8-sig")

    print(f"  - [Combined All 300] -> {all_ann1_path.name} & {all_ann2_path.name}")


def main():
    print("=" * 80)
    print("  EVENT PAIR DEDUPLICATION VALIDATION SAMPLING (N = 300 PAIRS)")
    print("=" * 80)

    df_pairs = pd.read_csv(INPUT_PAIR_AUDIT)
    master_sample = sample_threshold_pairs(df_pairs, seed=RANDOM_SEED)
    export_annotator_sheets(master_sample, INPUT_DIR)

    print("\n" + "=" * 80)
    print("  SAMPLING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
