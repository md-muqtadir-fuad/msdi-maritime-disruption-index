"""Create a blinded human-validation sample and evaluate completed annotations.

The script has two modes:

1. ``sample`` creates a reproducible 1,000-article gold-set template from the
   merged outputs of the three aligned LLMs.
2. ``evaluate`` computes human-human reliability, adjudicated/consensus labels,
   and model-human agreement from completed annotation files.

Examples
--------
python scripts/12_human_validation_metrics.py sample
python scripts/12_human_validation_metrics.py evaluate \
    --annotations annotator_1_completed.csv annotator_2_completed.csv
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = PROJECT_ROOT / "data/03_processed/model_scores/llm_scores_merged.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/04_validation"

MODEL_SCORE_COLUMNS = [
    "port_disruption_score",
    "global_container_movement_disruption_score",
    "freight_cost_disruption_score",
    "supply_disruption_score",
    "production_disruption_score",
    "covid19_related_disruption_score",
    "demand_disruption_score",
]
HUMAN_SCORE_COLUMNS = [f"human_{column}" for column in MODEL_SCORE_COLUMNS]
OVERALL_HUMAN_COLUMN = "human_overall_disruption_score"
SCORE_MIN = 0
SCORE_MAX = 4
RELEVANCE_LABELS = {"relevant", "irrelevant", "uncertain"}

ANNOTATION_COLUMNS = [
    "annotator_id",
    "relevance_label",
    *HUMAN_SCORE_COLUMNS,
    OVERALL_HUMAN_COLUMN,
    "dominant_dimension",
    "evidence_span",
    "annotator_confidence",
    "notes",
]

EVENT_WINDOWS = {
    "covid_initial_shock": ("2020-02-01", "2020-06-30"),
    "suez_ever_given": ("2021-03-20", "2021-04-15"),
    "russia_ukraine_initial": ("2022-02-20", "2022-05-31"),
    "shanghai_lockdown": ("2022-03-25", "2022-06-15"),
    "panama_canal_drought": ("2023-07-01", "2024-05-31"),
    "red_sea_rerouting": ("2023-11-01", "2024-06-30"),
    "us_port_labor_disruption": ("2024-09-01", "2024-10-31"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create human annotation samples or evaluate human/LLM agreement."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="Create blinded annotation templates.")
    sample.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    sample.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    sample.add_argument("--sample-size", type=int, default=1000)
    sample.add_argument("--random-size", type=int, default=600)
    sample.add_argument("--event-size", type=int, default=200)
    sample.add_argument("--disagreement-size", type=int, default=200)
    sample.add_argument("--annotators", type=int, default=2)
    sample.add_argument("--seed", type=int, default=20260712)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate completed annotations.")
    evaluate.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    evaluate.add_argument("--annotations", type=Path, nargs="+", required=True)
    evaluate.add_argument(
        "--adjudicated",
        type=Path,
        help="Optional final labels. Without this, deterministic consensus labels are used.",
    )
    evaluate.add_argument(
        "--allow-pseudo-annotations",
        action="store_true",
        help="Permit explicitly synthetic test labels; outputs remain non-human and non-publication-eligible.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def load_scores_for_sampling(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "article_id",
        "headline",
        "retrieved_title",
        "date",
        "retrieved_publish_date",
        "source_name",
        "source_domain",
        "preferred_url",
        "retrieved_text",
        "model_name",
        "scoring_status",
        "overall_disruption_mean",
        *MODEL_SCORE_COLUMNS,
    ]
    scores = pd.read_csv(require_file(path, "Merged score file"), usecols=columns, low_memory=False)
    if scores["model_name"].nunique(dropna=True) != 3:
        raise ValueError(
            "Expected exactly three aligned models; found "
            f"{scores['model_name'].nunique(dropna=True)}"
        )
    if scores.duplicated(["model_name", "article_id"]).any():
        raise ValueError("Duplicate (model_name, article_id) keys found in merged scores")

    successful = scores[scores["scoring_status"].eq("success")].copy()
    successful["overall_disruption_mean"] = pd.to_numeric(
        successful["overall_disruption_mean"], errors="coerce"
    )
    pivot = successful.pivot(
        index="article_id", columns="model_name", values="overall_disruption_mean"
    )
    agreement = pd.DataFrame(
        {
            "article_id": pivot.index,
            "models_successful": pivot.notna().sum(axis=1),
            "model_score_mean": pivot.mean(axis=1),
            "model_score_median": pivot.median(axis=1),
            "model_disagreement_sd": pivot.std(axis=1, ddof=0),
            "model_score_range": pivot.max(axis=1) - pivot.min(axis=1),
        }
    ).reset_index(drop=True)

    metadata = (
        scores.sort_values(["article_id", "model_name"])
        .groupby("article_id", as_index=False)
        .first()[
            [
                "article_id",
                "headline",
                "retrieved_title",
                "date",
                "retrieved_publish_date",
                "source_name",
                "source_domain",
                "preferred_url",
                "retrieved_text",
            ]
        ]
    )
    articles = metadata.merge(agreement, on="article_id", how="left", validate="one_to_one")
    articles = articles[articles["models_successful"].ge(2)].copy()
    primary_dates = pd.to_datetime(
        articles["date"], format="mixed", errors="coerce", utc=True
    )
    fallback_dates = pd.to_datetime(
        articles["retrieved_publish_date"], format="mixed", errors="coerce", utc=True
    )
    articles["publication_date"] = primary_dates.fillna(fallback_dates).dt.tz_localize(None)
    articles["title"] = articles["retrieved_title"].fillna(articles["headline"])
    articles["year"] = articles["publication_date"].dt.year.astype("Int64")
    return scores, articles


def assign_event_window(dates: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=dates.index, dtype="string")
    for name, (start, end) in EVENT_WINDOWS.items():
        mask = dates.between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both") & result.isna()
        result.loc[mask] = name
    return result


def sample_balanced_groups(
    frame: pd.DataFrame, group_column: str, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    n = min(n, len(frame))
    remaining = frame.copy()
    selected: list[pd.DataFrame] = []
    groups = [value for value in sorted(remaining[group_column].dropna().unique().tolist())]
    if not groups:
        return remaining.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1)))

    while sum(len(part) for part in selected) < n and not remaining.empty:
        progressed = False
        for group in groups:
            candidates = remaining[remaining[group_column].eq(group)]
            if candidates.empty:
                continue
            row = candidates.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1)))
            selected.append(row)
            remaining = remaining.drop(row.index)
            progressed = True
            if sum(len(part) for part in selected) >= n:
                break
        if not progressed:
            break
    return pd.concat(selected, ignore_index=False).head(n)


def create_annotation_sample(args: argparse.Namespace) -> dict:
    if args.sample_size <= 0 or args.annotators < 1:
        raise ValueError("sample-size and annotators must be positive")
    requested_parts = args.random_size + args.event_size + args.disagreement_size
    if requested_parts != args.sample_size:
        raise ValueError(
            "random-size + event-size + disagreement-size must equal sample-size "
            f"({requested_parts} != {args.sample_size})"
        )

    _, articles = load_scores_for_sampling(args.scores)
    if args.sample_size > len(articles):
        raise ValueError(f"Requested {args.sample_size} articles, but only {len(articles)} are eligible")
    rng = np.random.default_rng(args.seed)

    disagreement = (
        articles.sort_values(
            ["model_disagreement_sd", "model_score_range", "article_id"],
            ascending=[False, False, True],
        )
        .head(args.disagreement_size)
        .copy()
    )
    disagreement["sampling_stratum"] = "model_disagreement"
    selected_ids = set(disagreement["article_id"])

    remaining = articles[~articles["article_id"].isin(selected_ids)].copy()
    remaining["event_window"] = assign_event_window(remaining["publication_date"])
    event_candidates = remaining[remaining["event_window"].notna()].copy()
    event = sample_balanced_groups(event_candidates, "event_window", args.event_size, rng).copy()
    event["sampling_stratum"] = "known_event"
    selected_ids.update(event["article_id"])

    remaining = articles[~articles["article_id"].isin(selected_ids)].copy()
    source_counts = remaining["source_domain"].value_counts()
    remaining["source_tier"] = np.where(
        remaining["source_domain"].map(source_counts).fillna(0).ge(50), "frequent_source", "other_source"
    )
    remaining["random_stratum"] = (
        remaining["year"].astype("string").fillna("unknown") + "|" + remaining["source_tier"]
    )
    random_n = min(args.random_size, len(remaining))
    random_sample = sample_balanced_groups(
        remaining, "random_stratum", random_n, rng
    ).copy()
    random_sample["sampling_stratum"] = "probability_core"
    selected_ids.update(random_sample["article_id"])

    selected = pd.concat([random_sample, event, disagreement], ignore_index=True)
    if len(selected) < args.sample_size:
        fill = articles[~articles["article_id"].isin(selected_ids)].sample(
            n=args.sample_size - len(selected), random_state=args.seed + 1
        )
        fill["sampling_stratum"] = "probability_core_fill"
        selected = pd.concat([selected, fill], ignore_index=True)

    selected = selected.drop_duplicates("article_id", keep="first").head(args.sample_size)
    selected["annotation_scale"] = "0-4"
    selected["selection_seed"] = args.seed

    output_dir = args.output_dir.resolve()
    sample_dir = output_dir / "human_goldset"
    registry_dir = output_dir / "annotation_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    registry_columns = [
        "article_id",
        "publication_date",
        "source_name",
        "source_domain",
        "sampling_stratum",
        "event_window",
        "year",
        "models_successful",
        "model_score_mean",
        "model_score_median",
        "model_disagreement_sd",
        "model_score_range",
        "annotation_scale",
        "selection_seed",
    ]
    for column in registry_columns:
        if column not in selected.columns:
            selected[column] = pd.NA
    registry_path = registry_dir / "goldset_sampling_registry.csv"
    selected[registry_columns].to_csv(registry_path, index=False)

    blind_columns = [
        "article_id",
        "publication_date",
        "source_name",
        "source_domain",
        "preferred_url",
        "title",
        "retrieved_text",
    ]
    written_templates: list[str] = []
    for annotator_number in range(1, args.annotators + 1):
        template = selected[blind_columns].sample(
            frac=1, random_state=args.seed + annotator_number
        ).reset_index(drop=True)
        template.insert(0, "sample_order", np.arange(1, len(template) + 1))
        template["annotator_id"] = f"annotator_{annotator_number}"
        for column in ANNOTATION_COLUMNS[1:]:
            template[column] = pd.NA
        path = sample_dir / f"annotator_{annotator_number}_template.csv"
        template.to_csv(path, index=False)
        written_templates.append(str(path))

    manifest = {
        "scores_file": str(args.scores.resolve()),
        "eligible_articles": len(articles),
        "sample_size": len(selected),
        "requested_composition": {
            "probability_core": args.random_size,
            "known_event": args.event_size,
            "model_disagreement": args.disagreement_size,
        },
        "actual_composition": selected["sampling_stratum"].value_counts().to_dict(),
        "annotators": args.annotators,
        "seed": args.seed,
        "score_scale": {"minimum": SCORE_MIN, "maximum": SCORE_MAX},
        "registry_file": str(registry_path),
        "templates": written_templates,
        "blinding": "Model names, model scores, disagreement, and sampling strata are excluded from templates.",
    }
    manifest_path = registry_dir / "goldset_sampling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created {len(selected):,}-article validation sample")
    print(f"Registry: {registry_path}")
    for template in written_templates:
        print(f"Template: {template}")
    return manifest


def normalize_relevance(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    aliases = {
        "1": "relevant",
        "true": "relevant",
        "yes": "relevant",
        "0": "irrelevant",
        "false": "irrelevant",
        "no": "irrelevant",
        "unsure": "uncertain",
    }
    return normalized.replace(aliases)


def load_annotations(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        resolved = require_file(path, "Annotation file")
        frame = pd.read_csv(resolved, low_memory=False)
        if "article_id" not in frame.columns:
            raise ValueError(f"Annotation file lacks article_id: {resolved}")
        if "annotator_id" not in frame.columns:
            frame["annotator_id"] = resolved.stem
        else:
            frame["annotator_id"] = frame["annotator_id"].fillna(resolved.stem)
        frame["_annotation_file"] = resolved.name
        frames.append(frame)

    annotations = pd.concat(frames, ignore_index=True, sort=False)
    if annotations.duplicated(["article_id", "annotator_id"]).any():
        examples = annotations.loc[
            annotations.duplicated(["article_id", "annotator_id"], keep=False),
            ["article_id", "annotator_id"],
        ].head(10)
        raise ValueError(f"Duplicate article/annotator labels found:\n{examples.to_string(index=False)}")
    if "relevance_label" in annotations.columns:
        annotations["relevance_label"] = normalize_relevance(annotations["relevance_label"])
        invalid = annotations.loc[
            annotations["relevance_label"].notna()
            & ~annotations["relevance_label"].isin(RELEVANCE_LABELS),
            "relevance_label",
        ].unique()
        if len(invalid):
            raise ValueError(f"Invalid relevance labels: {invalid.tolist()}")

    for column in [*HUMAN_SCORE_COLUMNS, OVERALL_HUMAN_COLUMN, "annotator_confidence"]:
        if column not in annotations.columns:
            annotations[column] = np.nan
        annotations[column] = pd.to_numeric(annotations[column], errors="coerce")
    for column in [*HUMAN_SCORE_COLUMNS, OVERALL_HUMAN_COLUMN]:
        invalid = annotations[column].notna() & ~annotations[column].between(SCORE_MIN, SCORE_MAX)
        if invalid.any():
            raise ValueError(f"{column} contains values outside {SCORE_MIN}-{SCORE_MAX}")
    return annotations


def krippendorff_alpha(values: pd.DataFrame, level: str) -> float:
    """Compute nominal or interval alpha from an item × annotator matrix."""
    observed_num = 0.0
    observed_den = 0
    pooled: list[object] = []
    for _, row in values.iterrows():
        labels = row.dropna().tolist()
        pooled.extend(labels)
        for left, right in combinations(labels, 2):
            observed_num += float(left != right) if level == "nominal" else (float(left) - float(right)) ** 2
            observed_den += 1
    if observed_den == 0 or len(pooled) < 2:
        return math.nan
    observed = observed_num / observed_den
    expected_pairs = list(combinations(pooled, 2))
    expected = np.mean(
        [float(left != right) if level == "nominal" else (float(left) - float(right)) ** 2
         for left, right in expected_pairs]
    )
    return math.nan if expected == 0 else float(1 - observed / expected)


def icc_two_way_random_single(values: pd.DataFrame) -> float:
    """ICC(2,1) on complete cases using the Shrout-Fleiss ANOVA form."""
    matrix = values.dropna().to_numpy(dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return math.nan
    n, k = matrix.shape
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_columns = n * np.sum((column_means - grand) ** 2) / (k - 1)
    residual = matrix - row_means[:, None] - column_means[None, :] + grand
    ms_error = np.sum(residual**2) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return math.nan if denominator == 0 else float((ms_rows - ms_error) / denominator)


def pairwise_human_metrics(annotations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    annotators = sorted(annotations["annotator_id"].dropna().astype(str).unique())
    variables = [("relevance_label", "categorical"), *[(c, "ordinal") for c in [*HUMAN_SCORE_COLUMNS, OVERALL_HUMAN_COLUMN]]]
    for left_id, right_id in combinations(annotators, 2):
        left = annotations[annotations["annotator_id"].astype(str).eq(left_id)]
        right = annotations[annotations["annotator_id"].astype(str).eq(right_id)]
        for variable, kind in variables:
            pair = left[["article_id", variable]].merge(
                right[["article_id", variable]], on="article_id", suffixes=("_left", "_right")
            ).dropna()
            row = {
                "annotator_1": left_id,
                "annotator_2": right_id,
                "variable": variable,
                "n_complete": len(pair),
                "exact_agreement": math.nan,
                "cohen_kappa": math.nan,
                "weighted_kappa_quadratic": math.nan,
                "mae": math.nan,
            }
            if len(pair):
                a, b = pair[f"{variable}_left"], pair[f"{variable}_right"]
                row["exact_agreement"] = float((a == b).mean())
                if kind == "categorical":
                    row["cohen_kappa"] = float(cohen_kappa_score(a, b))
                else:
                    row["weighted_kappa_quadratic"] = float(
                        cohen_kappa_score(a.astype(int), b.astype(int), weights="quadratic")
                    )
                    row["mae"] = float(np.mean(np.abs(a.astype(float) - b.astype(float))))
            rows.append(row)
    return pd.DataFrame(rows)


def overall_human_metrics(annotations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for variable in ["relevance_label", *HUMAN_SCORE_COLUMNS, OVERALL_HUMAN_COLUMN]:
        matrix = annotations.pivot(index="article_id", columns="annotator_id", values=variable)
        complete = matrix.dropna()
        level = "nominal" if variable == "relevance_label" else "interval"
        rows.append(
            {
                "variable": variable,
                "items_with_two_or_more_labels": int((matrix.notna().sum(axis=1) >= 2).sum()),
                "complete_items": len(complete),
                "krippendorff_alpha": krippendorff_alpha(matrix, level),
                "icc_2_1": math.nan if level == "nominal" else icc_two_way_random_single(matrix),
            }
        )
    return pd.DataFrame(rows)


def consensus_labels(annotations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for article_id, group in annotations.groupby("article_id", sort=True):
        row: dict[str, object] = {"article_id": article_id}
        relevance = group["relevance_label"].dropna()
        if relevance.empty:
            row["relevance_label"] = pd.NA
        else:
            counts = relevance.value_counts()
            winners = sorted(counts[counts.eq(counts.max())].index.tolist())
            row["relevance_label"] = winners[0] if len(winners) == 1 else "uncertain"
        for column in [*HUMAN_SCORE_COLUMNS, OVERALL_HUMAN_COLUMN]:
            row[column] = group[column].median(skipna=True)
        row["human_label_count"] = group["annotator_id"].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def load_reference_labels(path: Path | None, annotations: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if path is None:
        return consensus_labels(annotations), "deterministic_human_consensus"
    reference = pd.read_csv(require_file(path, "Adjudicated label file"), low_memory=False)
    if "article_id" not in reference.columns:
        raise ValueError("Adjudicated label file lacks article_id")
    if reference.duplicated("article_id").any():
        raise ValueError("Adjudicated label file has duplicate article_id values")
    if "relevance_label" in reference.columns:
        reference["relevance_label"] = normalize_relevance(reference["relevance_label"])
    for column in [*HUMAN_SCORE_COLUMNS, OVERALL_HUMAN_COLUMN]:
        if column in reference.columns:
            reference[column] = pd.to_numeric(reference[column], errors="coerce")
    return reference, "adjudicated_labels"


def model_human_metrics(scores_path: Path, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_columns = ["article_id", "model_name", "scoring_status", "overall_disruption_max", *MODEL_SCORE_COLUMNS]
    scores = pd.read_csv(require_file(scores_path, "Merged score file"), usecols=score_columns, low_memory=False)
    scores = scores[scores["scoring_status"].eq("success")].copy()
    merged = scores.merge(reference, on="article_id", how="inner", validate="many_to_one")
    rows: list[dict] = []

    if "relevance_label" in merged.columns:
        relevance = merged[merged["relevance_label"].isin(["relevant", "irrelevant"])].copy()
        relevance["human_relevant"] = relevance["relevance_label"].eq("relevant").astype(int)
        relevance["model_relevant"] = pd.to_numeric(
            relevance["overall_disruption_max"], errors="coerce"
        ).gt(0).astype(int)
        for model, group in relevance.groupby("model_name"):
            rows.append(
                {
                    "model_name": model,
                    "variable": "relevance_label",
                    "n": len(group),
                    "mae": math.nan,
                    "rmse": math.nan,
                    "spearman_rho": math.nan,
                    "weighted_kappa_quadratic": math.nan,
                    "macro_f1": float(f1_score(group["human_relevant"], group["model_relevant"], average="macro")),
                    "accuracy": float((group["human_relevant"] == group["model_relevant"]).mean()),
                }
            )

    score_pairs = [
        *zip(MODEL_SCORE_COLUMNS, HUMAN_SCORE_COLUMNS),
        ("overall_disruption_max", OVERALL_HUMAN_COLUMN),
    ]
    for model_column, human_column in score_pairs:
        if human_column not in merged.columns:
            continue
        for model, group in merged.groupby("model_name"):
            pair = group[[model_column, human_column]].apply(pd.to_numeric, errors="coerce").dropna()
            if pair.empty:
                continue
            errors = pair[model_column] - pair[human_column]
            rho = spearmanr(pair[model_column], pair[human_column]).statistic if len(pair) > 1 else math.nan
            rows.append(
                {
                    "model_name": model,
                    "variable": model_column,
                    "n": len(pair),
                    "mae": float(np.mean(np.abs(errors))),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "spearman_rho": float(rho),
                    "weighted_kappa_quadratic": float(
                        cohen_kappa_score(
                            pair[human_column].astype(int), pair[model_column].astype(int), weights="quadratic"
                        )
                    ),
                    "macro_f1": math.nan,
                    "accuracy": float((pair[model_column] == pair[human_column]).mean()),
                }
            )

    confusion_rows: list[dict] = []
    if "relevance_label" in merged.columns:
        for model, group in merged[merged["relevance_label"].isin(["relevant", "irrelevant"])].groupby("model_name"):
            actual = group["relevance_label"].eq("relevant").astype(int)
            predicted = pd.to_numeric(group["overall_disruption_max"], errors="coerce").gt(0).astype(int)
            matrix = confusion_matrix(actual, predicted, labels=[0, 1])
            for actual_label, row_values in zip(["irrelevant", "relevant"], matrix):
                for predicted_label, count in zip(["irrelevant", "relevant"], row_values):
                    confusion_rows.append(
                        {
                            "model_name": model,
                            "actual": actual_label,
                            "predicted": predicted_label,
                            "count": int(count),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(confusion_rows)


def evaluate_annotations(args: argparse.Namespace) -> dict:
    annotations = load_annotations(args.annotations)
    pseudo = "annotation_provenance" in annotations.columns and annotations["annotation_provenance"].astype(str).str.contains("pseudo", case=False, na=False).any()
    if pseudo and not args.allow_pseudo_annotations:
        raise ValueError("Pseudo annotations detected. Supply real human labels or explicitly use --allow-pseudo-annotations for pipeline testing only.")
    if pseudo and args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError("Pseudo annotations require a separate --output-dir under data/04_validation/pseudo_test_only/ to avoid overwriting human-validation artifacts.")
    reference, reference_type = load_reference_labels(args.adjudicated, annotations)
    pairwise = pairwise_human_metrics(annotations)
    overall = overall_human_metrics(annotations)
    model_metrics, confusion = model_human_metrics(args.scores, reference)

    output_dir = args.output_dir.resolve()
    reliability_dir = output_dir / "inter_rater_reliability"
    calibration_dir = output_dir / "calibration"
    adjudicated_dir = output_dir / "adjudicated_labels"
    for directory in (reliability_dir, calibration_dir, adjudicated_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pairwise_path = reliability_dir / "pairwise_human_reliability.csv"
    overall_path = reliability_dir / "overall_human_reliability.csv"
    reference_path = adjudicated_dir / "validation_reference_labels.csv"
    model_path = calibration_dir / "model_human_metrics.csv"
    confusion_path = calibration_dir / "relevance_confusion_matrices.csv"
    pairwise.to_csv(pairwise_path, index=False)
    overall.to_csv(overall_path, index=False)
    reference.to_csv(reference_path, index=False)
    model_metrics.to_csv(model_path, index=False)
    confusion.to_csv(confusion_path, index=False)

    summary = {
        "annotation_files": [str(path.resolve()) for path in args.annotations],
        "annotators": sorted(annotations["annotator_id"].dropna().astype(str).unique().tolist()),
        "unique_articles": int(annotations["article_id"].nunique()),
        "reference_type": reference_type,
        "reference_articles": len(reference),
        "score_scale": {"minimum": SCORE_MIN, "maximum": SCORE_MAX},
        "status": "synthetic_test_only_not_human_validation" if pseudo else "human_annotation_evaluation",
        "publication_eligible": False if pseudo else None,
        "outputs": {
            "pairwise_human_reliability": str(pairwise_path),
            "overall_human_reliability": str(overall_path),
            "reference_labels": str(reference_path),
            "model_human_metrics": str(model_path),
            "relevance_confusion_matrices": str(confusion_path),
        },
    }
    summary_path = output_dir / "validation_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Evaluated {summary['unique_articles']:,} annotated articles")
    print(f"Summary: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    if args.command == "sample":
        create_annotation_sample(args)
    else:
        evaluate_annotations(args)


if __name__ == "__main__":
    main()
