"""Point 7: internal validation and coverage diagnostics for the MSDI."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "eda/outputs/point_07_msdi_validation"
ARTICLE_FILE = ROOT / "data/05_indices/article_level/article_consensus_scores.csv"
MONTHLY_FILE = ROOT / "data/05_indices/monthly/msdi_monthly.csv"
MODEL_MONTHLY_FILE = ROOT / "data/05_indices/monthly/msdi_monthly_model_specific.csv"
EVENT_ROOT = ROOT / "data/03_processed/news_full_text/pre_llm_filter/event_dedup"
EVENT_CATALOG = EVENT_ROOT / "canonical_event_catalog_by_profile.csv"
PAIR_AUDIT = EVENT_ROOT / "canonical_event_pair_audit_by_profile.csv"
EVENT_MANIFEST = EVENT_ROOT / "canonical_event_manifest.json"
ANNOTATORS = (EVENT_ROOT / "canonical_event_pair_validation_annotator_1.csv", EVENT_ROOT / "canonical_event_pair_validation_annotator_2.csv")

MODELS = ("gemma_2_9b_it", "qwen_3_8b", "qwen_2_5_7b", "glm_4_9b_0414")
DIMENSIONS = (
    "port_disruption_score",
    "global_container_movement_disruption_score",
    "freight_cost_disruption_score",
    "supply_disruption_score",
    "production_disruption_score",
    "covid19_related_disruption_score",
    "demand_disruption_score",
)
SCORES = ("overall_disruption_max", *DIMENSIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MSDI coverage, weights, thresholds, and LLM agreement.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-run", action="store_true", help="Use complete compact score tables; no embedding or LLM run.")
    return parser.parse_args()


def friendly(value: str) -> str:
    replacements = {"covid19": "COVID-19", "msdi": "MSDI", "glm": "GLM", "qwen": "Qwen", "gemma": "Gemma"}
    words = value.replace("_score", "").replace("_", " ").split()
    label = " ".join(replacements.get(word.lower(), word) for word in words).capitalize()
    return label.replace("Covid-19", "COVID-19").replace("Msdi", "MSDI").replace("Glm", "GLM")


def load_articles() -> pd.DataFrame:
    raw = pd.read_csv(ARTICLE_FILE, low_memory=False)
    raw["publication_date"] = pd.to_datetime(raw["publication_date"], errors="coerce")
    raw["month_end"] = raw["publication_date"].dt.to_period("M").dt.to_timestamp("M")
    return raw


def monthly_coverage(article: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = article.loc[article["ensemble_eligible"].fillna(False).astype(bool)].copy()
    rows = []
    source_rows = []
    for month, group in eligible.groupby("month_end", dropna=False):
        source_counts = group["source_domain"].fillna("unknown").value_counts()
        shares = source_counts / source_counts.sum()
        rows.append({
            "month_end": month,
            "articles": len(group),
            "events_monthly": group["event_id_monthly"].nunique(),
            "sources": group["source_domain"].nunique(),
            "source_hhi": np.square(shares).sum(),
            "largest_source_share": shares.max(),
            "effective_articles": group["novelty_weight_monthly"].sum(),
            "mean_peak": group["peak_severity_score"].mean(),
            "mean_breadth": group["breadth_severity_score"].mean(),
            "mean_model_disagreement": group["overall_disruption_max__ensemble_sd"].mean(),
        })
        for source, count in source_counts.items():
            source_rows.append({"month_end": month, "source_domain": source, "articles": count, "share": count / len(group)})
    return pd.DataFrame(rows).sort_values("month_end"), pd.DataFrame(source_rows)


def event_profile_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog = pd.read_csv(EVENT_CATALOG, low_memory=False)
    summary = catalog.groupby("event_profile", as_index=False).agg(
        events=("event_id", "nunique"),
        represented_articles=("article_count", "sum"),
        multi_article_events=("article_count", lambda s: int(s.gt(1).sum())),
        articles_in_multi_article_events=("article_count", lambda s: int(s.loc[s.gt(1)].sum())),
        mean_cluster_size=("article_count", "mean"),
        p95_cluster_size=("article_count", lambda s: s.quantile(0.95)),
        maximum_cluster_size=("article_count", "max"),
        mean_source_count=("source_count", "mean"),
        maximum_span_days=("event_span_days", "max"),
    )
    distributions = catalog.groupby(["event_profile", "article_count"], as_index=False).size().rename(columns={"size": "events"})
    return summary, distributions


def threshold_audit() -> pd.DataFrame:
    columns = ["event_profile", "cosine_similarity", "passes_threshold_0_85", "passes_threshold_0_88", "passes_threshold_0_90", "passes_threshold_0_92"]
    pairs = pd.read_csv(PAIR_AUDIT, usecols=columns, low_memory=False)
    rows = []
    for profile, group in pairs.groupby("event_profile"):
        for threshold in (0.85, 0.88, 0.90, 0.92):
            passed = pd.to_numeric(group[f"passes_threshold_{threshold:.2f}".replace(".", "_")], errors="coerce").eq(1)
            rows.append({"event_profile": profile, "threshold": threshold, "candidate_pairs": len(group), "pairs_passing_threshold": int(passed.sum()), "passing_share": passed.mean(), "human_calibration_status": "pending"})
    return pd.DataFrame(rows)


def weight_audit(article: pd.DataFrame) -> pd.DataFrame:
    columns = ("source_duplicate_weight", "novelty_weight_monthly", "novelty_weight_weekly", "event_period_article_count_monthly", "event_period_article_count_weekly")
    rows = []
    for column in columns:
        values = pd.to_numeric(article[column], errors="coerce").dropna()
        rows.append({"variable": column, "label": friendly(column), "observations": len(values), "minimum": values.min(), "p01": values.quantile(0.01), "median": values.median(), "mean": values.mean(), "p99": values.quantile(0.99), "maximum": values.max(), "zero_share": values.eq(0).mean(), "above_one_share": values.gt(1).mean()})
    return pd.DataFrame(rows)


def dimension_audit(article: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in DIMENSIONS:
        values = pd.to_numeric(article[f"{dimension}__ensemble_median"], errors="coerce").dropna()
        rows.append({"dimension": dimension, "label": friendly(dimension), "observations": len(values), "mean": values.mean(), "median": values.median(), "zero_share": values.eq(0).mean(), "severe_3_or_4_share": values.ge(3).mean(), "maximum_4_share": values.eq(4).mean()})
    return pd.DataFrame(rows)


def model_agreement(article: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for score in SCORES:
        for left, right in itertools.combinations(MODELS, 2):
            pair = article[[f"{score}__{left}", f"{score}__{right}"]].apply(pd.to_numeric, errors="coerce").dropna()
            x, y = pair.iloc[:, 0], pair.iloc[:, 1]
            rows.append({
                "score": score,
                "label": friendly(score),
                "model_1": left,
                "model_2": right,
                "observations": len(pair),
                "exact_agreement": x.eq(y).mean(),
                "within_one_point": x.sub(y).abs().le(1).mean(),
                "mean_absolute_difference": x.sub(y).abs().mean(),
                "spearman_correlation": x.corr(y, method="spearman"),
                "quadratic_weighted_kappa": cohen_kappa_score(x.astype(int), y.astype(int), weights="quadratic") if len(pair) else np.nan,
            })
    long = pd.DataFrame(rows)
    summary = long.groupby(["model_1", "model_2"], as_index=False).agg(mean_exact_agreement=("exact_agreement", "mean"), mean_within_one_point=("within_one_point", "mean"), mean_absolute_difference=("mean_absolute_difference", "mean"), mean_spearman=("spearman_correlation", "mean"), mean_quadratic_weighted_kappa=("quadratic_weighted_kappa", "mean"))
    return long, summary


def model_temporal_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    model = pd.read_csv(MODEL_MONTHLY_FILE, low_memory=False)
    model["month_end"] = pd.to_datetime(model["month_end"], errors="coerce")
    rows = []
    for path in ("peak_score_media_corrected", "breadth_score_media_corrected"):
        pivot = model.pivot(index="month_end", columns="model_alias", values=path)
        correlations = pivot.corr(method="spearman")
        for left, right in itertools.combinations(correlations.columns, 2):
            pair = pivot[[left, right]].dropna()
            rows.append({"path": path, "model_1": left, "model_2": right, "observations": len(pair), "spearman_correlation": pair[left].corr(pair[right], method="spearman"), "mean_absolute_difference": pair[left].sub(pair[right]).abs().mean()})
    monthly = pd.read_csv(MONTHLY_FILE, low_memory=False)
    monthly["month_end"] = pd.to_datetime(monthly["month_end"], errors="coerce")
    monthly["peak_minus_breadth"] = monthly["peak_score_media_corrected"] - monthly["breadth_score_media_corrected"]
    monthly["covid_dimension_share"] = monthly["covid19_related_disruption_media_corrected"] / monthly[[f"{d.replace('_score', '')}_media_corrected" for d in DIMENSIONS]].sum(axis=1).replace(0, np.nan)
    monthly["news_volume_peak_pearson_24m"] = monthly["article_count"].rolling(24, min_periods=18).corr(monthly["peak_score_media_corrected"])
    return pd.DataFrame(rows), monthly


def human_validation_audit() -> pd.DataFrame:
    rows = []
    for path in ANNOTATORS:
        raw = pd.read_csv(path, low_memory=False)
        labelled = pd.to_numeric(raw["human_same_event"], errors="coerce").notna()
        rows.append({"file": str(path.relative_to(ROOT)), "rows": len(raw), "labelled_rows": int(labelled.sum()), "unlabelled_rows": int((~labelled).sum()), "complete": bool(labelled.all()), "usable_for_threshold_selection": bool(labelled.all())})
    pseudo = ROOT / "data/04_validation/pseudo_test_only/evaluation"
    rows.append({"file": str(pseudo.relative_to(ROOT)), "rows": np.nan, "labelled_rows": np.nan, "unlabelled_rows": np.nan, "complete": False, "usable_for_threshold_selection": False})
    return pd.DataFrame(rows)


def plots(coverage: pd.DataFrame, dimensions: pd.DataFrame, monthly: pd.DataFrame, output_dir: Path) -> list[str]:
    outputs = []
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(coverage["month_end"], coverage["articles"], linewidth=1.2)
    axes[0].set_title("Articles per month", loc="left")
    axes[1].plot(coverage["month_end"], coverage["events_monthly"], linewidth=1.2)
    axes[1].set_title("Canonical events per month", loc="left")
    axes[2].plot(coverage["month_end"], coverage["largest_source_share"], linewidth=1.2)
    axes[2].set_title("Largest source share", loc="left")
    for ax in axes: ax.grid(alpha=0.2)
    fig.tight_layout(); path = output_dir / "monthly_coverage_and_concentration.png"; fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); outputs.append(path.name)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(dimensions))
    ax.bar(positions - 0.18, dimensions["zero_share"], width=0.36, label="Zero")
    ax.bar(positions + 0.18, dimensions["severe_3_or_4_share"], width=0.36, label="Score 3 or 4")
    ax.set_xticks(positions, [label.replace(" disruption", "") for label in dimensions["label"]], rotation=30, ha="right")
    ax.set_ylabel("Article share"); ax.legend(frameon=False); ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); path = output_dir / "dimension_score_distribution.png"; fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); outputs.append(path.name)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    axes[0].plot(monthly["month_end"], monthly["peak_minus_breadth"], linewidth=1.2)
    axes[0].axhline(0, color="0.5", linewidth=0.7); axes[0].set_title("Peak minus breadth MSDI", loc="left")
    axes[1].plot(monthly["month_end"], monthly["covid_dimension_share"], linewidth=1.2)
    axes[1].set_title("COVID-19 dimension share", loc="left")
    for ax in axes: ax.grid(alpha=0.2)
    fig.tight_layout(); path = output_dir / "peak_breadth_and_covid_share.png"; fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); outputs.append(path.name)
    return outputs


def write_findings(path: Path, article: pd.DataFrame, coverage: pd.DataFrame, events: pd.DataFrame, thresholds: pd.DataFrame, agreement: pd.DataFrame, human: pd.DataFrame) -> None:
    dates = article["publication_date"].dropna()
    lines = [
        "# Point 7 - MSDI validation findings",
        "",
        f"- Article rows: {len(article)}; ensemble eligible: {int(article['ensemble_eligible'].fillna(False).astype(bool).sum())}.",
        f"- Article coverage: {dates.min().date()} to {dates.max().date()}. The model panel extends beyond the last article and must not treat carried/smoothed values as new news evidence.",
        f"- Unique source domains: {article['source_domain'].nunique()}.",
        f"- Monthly median article count: {coverage['articles'].median():.1f}; minimum: {coverage['articles'].min()}; maximum: {coverage['articles'].max()}.",
        f"- Mean pairwise quadratic-weighted kappa across eight scores: {agreement['mean_quadratic_weighted_kappa'].mean():.3f}.",
        f"- Human event-pair labels completed: {int(human['labelled_rows'].fillna(0).sum())}; threshold selection remains pending.",
        "",
        "The selected 0.88 threshold remains explicitly provisional. Counts at 0.85, 0.88, 0.90, and 0.92 are descriptive; no threshold is chosen from SCFI association or forecasting performance.",
        "",
        "Pseudo-test validation artifacts are excluded from substantive human-validation claims.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    article = load_articles()
    coverage, sources = monthly_coverage(article)
    events, cluster_sizes = event_profile_audit()
    thresholds = threshold_audit()
    weights = weight_audit(article)
    dimensions = dimension_audit(article)
    agreement_long, agreement_summary = model_agreement(article)
    temporal_agreement, monthly = model_temporal_audit()
    human = human_validation_audit()
    figure_files = plots(coverage, dimensions, monthly, args.output_dir)

    outputs = {
        "monthly_article_coverage.csv": coverage,
        "monthly_source_composition.csv": sources,
        "event_profile_summary.csv": events,
        "event_cluster_size_distribution.csv": cluster_sizes,
        "threshold_pair_counts.csv": thresholds,
        "weight_distribution.csv": weights,
        "dimension_distribution.csv": dimensions,
        "article_model_agreement.csv": agreement_long,
        "model_pair_agreement_summary.csv": agreement_summary,
        "monthly_model_path_agreement.csv": temporal_agreement,
        "monthly_peak_breadth_covid_diagnostics.csv": monthly,
        "human_validation_completion_audit.csv": human,
    }
    for name, frame in outputs.items(): frame.to_csv(args.output_dir / name, index=False)
    write_findings(args.output_dir / "point_07_findings.md", article, coverage, events, thresholds, agreement_summary, human)
    manifest = {
        "protocol_point": 7,
        "run_mode": "test" if args.test_run else "full",
        "status": "computational_checks_complete_human_validation_pending",
        "articles": len(article),
        "ensemble_eligible": int(article["ensemble_eligible"].fillna(False).astype(bool).sum()),
        "article_last_date": article["publication_date"].max().date().isoformat(),
        "provisional_threshold": json.loads(EVENT_MANIFEST.read_text(encoding="utf-8"))["selected_threshold_provisional"],
        "human_labels_completed": int(human["labelled_rows"].fillna(0).sum()),
        "figures": figure_files,
        "guardrail": "The 0.88 event threshold remains provisional until independent human labels are completed; pseudo labels are excluded.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
