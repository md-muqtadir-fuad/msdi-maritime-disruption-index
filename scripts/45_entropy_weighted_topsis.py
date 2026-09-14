"""Entropy-weighted TOPSIS for the operational model ranking.

Script 13 ranks the four scoring models with criterion weights asserted in
`configs/model_registry.yaml`. Those weights are a judgement call and a reviewer can
reasonably ask where they came from. This script derives them from the decision matrix
instead, using Shannon entropy: a criterion that separates the alternatives carries
information and earns weight, a criterion on which they are near-identical does not.

Both weightings are reported side by side so the ranking can be shown to be a property
of the data rather than of the weights. Because entropy weights are estimated from only
four alternatives, a leave-one-model-out pass reports how far they move when any single
model is withdrawn.

Direction (benefit/cost) plays no part in the entropy step, which measures dispersion
only; it enters at the TOPSIS ideal/anti-ideal step exactly as in script 13. The
normalization used for the entropy step is switchable, since the choice is a modelling
decision rather than a settled one.

This script is the reporting path for any TOPSIS figure that reaches the manuscript.
Script 13 is left as it is, so it still computes a score from the asserted weights and
still writes the two CSVs this script reads. Those `topsis_score` and `topsis_rank`
columns are a superseded intermediate and should not be quoted; the published score is
`entropy_score` in `outputs/topsis/entropy_weighted_topsis_ranking.csv`. The ordering is
the same under either weighting, so the distinction affects the reported score and not
the conclusion.

This ranking is operational. It does not weight the MSDI ensemble and it is not a
measurement-validity ranking; see `validation/full/results/llm_vs_human_benchmark_summary.csv`.

Source: data/05_indices/disruption_index_versions/<version>_topsis_model_ranking.csv
        configs/model_registry.yaml
Output: outputs/topsis/*.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = PROJECT_ROOT / "configs/model_registry.yaml"
INDEX_VERSIONS = PROJECT_ROOT / "data/05_indices/disruption_index_versions"
OUTPUT_DIR = PROJECT_ROOT / "outputs/topsis"

ALTERNATIVE_KEY = "model_alias"


def load_criteria(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Model registry not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    criteria = config.get("topsis_operational_ranking", {}).get("criteria", {})
    if not criteria:
        raise ValueError("model_registry.yaml lacks topsis_operational_ranking.criteria")
    for name, spec in criteria.items():
        direction = str(spec.get("direction", "")).lower()
        if direction not in {"benefit", "cost"}:
            raise ValueError(f"Invalid TOPSIS direction for {name}: {spec.get('direction')!r}")
    return criteria


def load_decision_matrix(version: str, criteria: dict[str, dict[str, Any]]) -> pd.DataFrame:
    path = INDEX_VERSIONS / f"{version}_topsis_model_ranking.csv"
    if not path.is_file():
        raise FileNotFoundError(f"TOPSIS ranking not found: {path}. Run script 13 first.")
    frame = pd.read_csv(path)
    missing = [name for name in [ALTERNATIVE_KEY, *criteria] if name not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} lacks required columns: {missing}")
    matrix = frame[[ALTERNATIVE_KEY, *criteria]].copy()
    values = matrix[list(criteria)].astype(float)
    if values.isna().any().any():
        raise ValueError(f"Criteria contain missing values:\n{values.isna().sum().to_string()}")
    if (values < 0).any().any():
        raise ValueError("Entropy weighting requires non-negative criterion values")
    if len(matrix) < 2:
        raise ValueError("Entropy weighting requires at least two alternatives")
    return matrix.set_index(ALTERNATIVE_KEY)


def entropy_weights(
    matrix: pd.DataFrame, criteria: dict[str, dict[str, Any]], normalization: str
) -> pd.DataFrame:
    """Shannon entropy weights.

    `proportion` scales each column by its own total, the textbook form. `minmax` rescales
    by direction first, which makes the worst alternative on every criterion contribute
    zero probability mass and therefore reacts more sharply to the extremes.
    """
    values = matrix.astype(float)
    if normalization == "proportion":
        totals = values.sum(axis=0)
        if (totals <= 0).any():
            raise ValueError("Proportion normalization requires strictly positive column sums")
        scaled = values.div(totals, axis=1)
    elif normalization == "minmax":
        scaled = pd.DataFrame(index=values.index, columns=values.columns, dtype=float)
        for name, spec in criteria.items():
            column = values[name]
            spread = column.max() - column.min()
            if spread == 0:
                scaled[name] = 0.0
                continue
            rescaled = (column - column.min()) / spread
            scaled[name] = rescaled if spec["direction"] == "benefit" else 1.0 - rescaled
        totals = scaled.sum(axis=0)
        scaled = scaled.div(totals.replace(0, np.nan), axis=1).fillna(0.0)
    else:
        raise ValueError(f"Unknown entropy normalization: {normalization!r}")

    count = len(values)
    coefficient = 1.0 / np.log(count)
    proportions = scaled.to_numpy(dtype=float)
    # 0 log 0 is taken as 0, the standard convention for the limit.
    terms = np.zeros_like(proportions)
    positive = proportions > 0
    terms[positive] = proportions[positive] * np.log(proportions[positive])
    entropy = pd.Series(-coefficient * terms.sum(axis=0), index=scaled.columns)
    divergence = 1.0 - entropy
    total_divergence = divergence.sum()
    if total_divergence <= 0:
        raise ValueError("All criteria are constant; entropy weights are undefined")

    return pd.DataFrame(
        {
            "criterion": list(values.columns),
            "direction": [criteria[name]["direction"] for name in values.columns],
            "registry_weight": [float(criteria[name]["weight"]) for name in values.columns],
            "entropy": entropy,
            "divergence": divergence,
            "entropy_weight": divergence / total_divergence,
        }
    ).set_index("criterion")


def topsis(
    matrix: pd.DataFrame, criteria: dict[str, dict[str, Any]], weights: pd.Series
) -> pd.DataFrame:
    """Vector-normalized TOPSIS, matching `topsis_rank` in script 13."""
    values = matrix.astype(float)
    denominators = np.sqrt((values**2).sum(axis=0))
    normalized = values.div(denominators.replace(0, np.nan), axis=1).fillna(0.0)
    weighted = normalized.mul(weights, axis=1)

    ideal_best = {}
    ideal_worst = {}
    for name, spec in criteria.items():
        benefit = spec["direction"] == "benefit"
        ideal_best[name] = weighted[name].max() if benefit else weighted[name].min()
        ideal_worst[name] = weighted[name].min() if benefit else weighted[name].max()

    distance_best = np.sqrt(((weighted - pd.Series(ideal_best)) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted - pd.Series(ideal_worst)) ** 2).sum(axis=1))
    denominator = (distance_best + distance_worst).replace(0, np.nan)

    score = distance_worst.div(denominator)
    return pd.DataFrame(
        {
            "distance_to_ideal": distance_best,
            "distance_to_anti_ideal": distance_worst,
            "score": score,
            "rank": score.rank(method="min", ascending=False).astype("Int64"),
        }
    )


def leave_one_out_weights(
    matrix: pd.DataFrame, criteria: dict[str, dict[str, Any]], normalization: str
) -> pd.DataFrame:
    """How far the entropy weights move when one alternative is withdrawn."""
    rows = []
    for dropped in matrix.index:
        subset = matrix.drop(index=dropped)
        weights = entropy_weights(subset, criteria, normalization)["entropy_weight"]
        for criterion, weight in weights.items():
            rows.append(
                {"dropped_model": dropped, "criterion": criterion, "entropy_weight": weight}
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="msdi_v001", help="Index version prefix")
    parser.add_argument(
        "--entropy-normalization",
        default="proportion",
        choices=("proportion", "minmax"),
        help="Normalization applied before the entropy calculation",
    )
    args = parser.parse_args()

    criteria = load_criteria(REGISTRY)
    matrix = load_decision_matrix(args.version, criteria)
    weights = entropy_weights(matrix, criteria, args.entropy_normalization)

    entropy_ranking = topsis(matrix, criteria, weights["entropy_weight"])
    registry_ranking = topsis(matrix, criteria, weights["registry_weight"])

    comparison = matrix.join(
        entropy_ranking.add_prefix("entropy_").join(registry_ranking.add_prefix("registry_"))
    )
    comparison["rank_agreement"] = comparison["entropy_rank"] == comparison["registry_rank"]
    comparison["entropy_normalization"] = args.entropy_normalization
    comparison["ranking_scope"] = "provisional_operational_not_measurement_validity"
    comparison = comparison.sort_values("entropy_rank")

    stability = leave_one_out_weights(matrix, criteria, args.entropy_normalization)
    stability_summary = (
        stability.groupby("criterion")["entropy_weight"]
        .agg(loo_min="min", loo_max="max")
        .join(weights["entropy_weight"].rename("full_sample_weight"))
    )
    stability_summary["loo_range"] = stability_summary["loo_max"] - stability_summary["loo_min"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights.join(stability_summary[["loo_min", "loo_max", "loo_range"]]).to_csv(
        OUTPUT_DIR / "entropy_criterion_weights.csv"
    )
    comparison.to_csv(OUTPUT_DIR / "entropy_weighted_topsis_ranking.csv")
    stability.to_csv(OUTPUT_DIR / "entropy_weight_stability_loo.csv", index=False)

    ordered = weights.sort_values("entropy_weight", ascending=False)
    print(f"Entropy normalization: {args.entropy_normalization}  (m={len(matrix)} alternatives)")
    print("\nCriterion weights")
    print(
        ordered[["direction", "registry_weight", "entropy", "entropy_weight"]]
        .round(4)
        .to_string()
    )
    print("\nRanking")
    print(
        comparison[["entropy_score", "entropy_rank", "registry_score", "registry_rank"]]
        .round(4)
        .to_string()
    )

    concordant = bool(comparison["rank_agreement"].all())
    print(f"\nOrderings agree on every alternative: {concordant}")
    widest = stability_summary["loo_range"].idxmax()
    print(
        f"Largest leave-one-out weight movement: {widest} "
        f"({stability_summary.loc[widest, 'loo_range']:.4f} across "
        f"{len(matrix)} single-model withdrawals)"
    )
    print(f"\nWritten to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
