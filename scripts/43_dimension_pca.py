"""Principal component analysis of the seven MSDI disruption dimensions.

The index scores every article on seven disruption criteria and then aggregates them
into the peak (maximum) and breadth (mean) paths. This script asks whether those seven
criteria carry seven independent signals or collapse onto a smaller latent structure,
which is the measurement-model question behind the peak/breadth construction.

The analysis runs at the article level (n = 6,542 scored articles, ensemble-median score
per dimension), which is where the measurement question lives: do the seven criteria
measure fewer underlying constructs? These scores are ordinal (0-4) and heavily
zero-inflated (70-91% zeros per dimension), so component extraction uses the Spearman
rank correlation matrix rather than Pearson.

Component retention uses permutation parallel analysis (Buja & Eyuboglu): each column is
independently shuffled to destroy cross-variable association while preserving that
variable's own marginal distribution, which matters here because the article-level
marginals are far from normal. A component is retained when its observed eigenvalue
exceeds the 95th percentile of the permuted eigenvalues at the same position. The Kaiser
eigenvalue-of-one rule is reported alongside for reference but is not the criterion.

Outputs
-------
outputs/pca/pca_article_adequacy.csv          Bartlett sphericity and KMO/MSA
outputs/pca/pca_article_eigenvalues.csv       Eigenvalues, variance shares, retention
outputs/pca/pca_article_loadings.csv          Unrotated and varimax-rotated loadings
outputs/pca/pca_article_component_scores.csv  Article scores on retained components
outputs/figures/eda/eda_figure05_dimension_pca.{pdf,png}       Scree and biplot
outputs/figures/eda/eda_figure06_dimension_pairplot.{pdf,png}  Scatter matrix
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTICLE_SCORES = PROJECT_ROOT / "data/05_indices/article_level/article_consensus_scores.csv"
TABLE_DIR = PROJECT_ROOT / "outputs/pca"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures/eda"

DIMENSIONS = [
    "port",
    "global_container_movement",
    "freight_cost",
    "supply",
    "production",
    "covid19_related",
    "demand",
]
DIMENSION_LABELS = {
    "port": "Port",
    "global_container_movement": "Global container movement",
    "freight_cost": "Freight cost",
    "supply": "Supply",
    "production": "Production",
    "covid19_related": "COVID-19 related",
    "demand": "Demand",
}
SHORT_LABELS = {
    "port": "Port",
    "global_container_movement": "Container mov.",
    "freight_cost": "Freight cost",
    "supply": "Supply",
    "production": "Production",
    "covid19_related": "COVID-19",
    "demand": "Demand",
}
BREADTH_CLASSES = ["Narrow (1-2)", "Moderate (3-4)", "Broad (5-7)"]
CLASS_COLORS = {
    "Narrow (1-2)": "#1f4e79",
    "Moderate (3-4)": "#c55a11",
    "Broad (5-7)": "#2e7d32",
}
PERMUTATION_DRAWS = 1000
PERMUTATION_PERCENTILE = 95.0
RANDOM_SEED = 20260801

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Times", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10.5
plt.rcParams["axes.titlesize"] = 11.5
plt.rcParams["xtick.labelsize"] = 9.5
plt.rcParams["ytick.labelsize"] = 9.5
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def bartlett_sphericity(correlation: np.ndarray, n_observations: int) -> tuple[float, int, float]:
    """Test the null that the correlation matrix is an identity matrix."""
    variables = correlation.shape[0]
    sign, log_determinant = np.linalg.slogdet(correlation)
    if sign <= 0:
        raise ValueError("Correlation matrix is not positive definite; Bartlett is undefined")
    statistic = -((n_observations - 1) - (2 * variables + 5) / 6) * log_determinant
    degrees_freedom = variables * (variables - 1) // 2
    return float(statistic), int(degrees_freedom), float(stats.chi2.sf(statistic, degrees_freedom))


def kaiser_meyer_olkin(correlation: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the overall KMO statistic and each variable's measure of sampling adequacy."""
    inverse = np.linalg.inv(correlation)
    diagonal = np.sqrt(np.diag(inverse))
    partial = -inverse / np.outer(diagonal, diagonal)
    np.fill_diagonal(partial, 0.0)
    observed = correlation.copy()
    np.fill_diagonal(observed, 0.0)
    observed_squares = observed**2
    partial_squares = partial**2
    overall = observed_squares.sum() / (observed_squares.sum() + partial_squares.sum())
    per_variable = observed_squares.sum(axis=0) / (
        observed_squares.sum(axis=0) + partial_squares.sum(axis=0)
    )
    return float(overall), per_variable


def parallel_analysis_thresholds(
    data: pd.DataFrame, method: str, draws: int, percentile: float, seed: int
) -> np.ndarray:
    """Eigenvalue thresholds from column-wise permutation of the observed data.

    Shuffling each column independently preserves that variable's marginal distribution
    (critical for zero-inflated ordinal scores) while removing any association between
    variables, so the resulting eigenvalues describe what this data would produce under
    pure noise.
    """
    generator = np.random.default_rng(seed)
    values = data.to_numpy(dtype=float)
    rows, columns = values.shape
    simulated = np.empty((draws, columns), dtype=float)
    for draw in range(draws):
        permuted = np.column_stack(
            [values[generator.permutation(rows), column] for column in range(columns)]
        )
        correlation = pd.DataFrame(permuted).corr(method=method).to_numpy(dtype=float)
        simulated[draw] = np.sort(np.linalg.eigvalsh(correlation))[::-1]
    return np.percentile(simulated, percentile, axis=0)


def varimax(loadings: np.ndarray, tolerance: float = 1e-6, max_iterations: int = 100) -> np.ndarray:
    """Kaiser varimax rotation; returns the rotated loading matrix."""
    variables, components = loadings.shape
    if components < 2:
        return loadings.copy()
    rotation = np.eye(components)
    previous = 0.0
    for _ in range(max_iterations):
        rotated = loadings @ rotation
        cubed = rotated**3
        column_means = np.diag((rotated**2).sum(axis=0)) / variables
        gradient = loadings.T @ (cubed - rotated @ column_means)
        u, singular_values, vt = np.linalg.svd(gradient)
        rotation = u @ vt
        current = singular_values.sum()
        if previous != 0.0 and current / previous < 1 + tolerance:
            break
        previous = current
    return loadings @ rotation


def run_pca(data: pd.DataFrame, level: str, method: str) -> dict[str, object]:
    """Extract components from the correlation matrix and assess retention."""
    correlation_frame = data.corr(method=method)
    correlation = correlation_frame.to_numpy(dtype=float)
    observations = len(data)

    statistic, degrees_freedom, p_value = bartlett_sphericity(correlation, observations)
    kmo_overall, kmo_per_variable = kaiser_meyer_olkin(correlation)

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    thresholds = parallel_analysis_thresholds(
        data, method, PERMUTATION_DRAWS, PERMUTATION_PERCENTILE, RANDOM_SEED
    )
    retained = int((eigenvalues > thresholds).sum())
    # retention must be a leading block: stop at the first component that fails
    failures = np.where(eigenvalues <= thresholds)[0]
    if failures.size:
        retained = int(failures[0])
    retained = max(retained, 1)

    loadings = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    # Component signs are arbitrary in PCA. Orient each component so its largest-magnitude
    # loading is positive, so a higher score reads as more disruption rather than less.
    dominant = np.argmax(np.abs(loadings), axis=0)
    signs = np.where(loadings[dominant, np.arange(loadings.shape[1])] < 0, -1.0, 1.0)
    loadings = loadings * signs
    eigenvectors = eigenvectors * signs
    rotated = varimax(loadings[:, :retained])

    labels = [DIMENSION_LABELS[name] for name in DIMENSIONS]
    component_names = [f"PC{index + 1}" for index in range(len(eigenvalues))]

    adequacy = pd.DataFrame(
        {
            "level": level,
            "correlation_method": method,
            "n_observations": observations,
            "n_variables": len(DIMENSIONS),
            "bartlett_chi_square": statistic,
            "bartlett_degrees_freedom": degrees_freedom,
            "bartlett_p_value": p_value,
            "kmo_overall": kmo_overall,
            "dimension": labels,
            "kmo_measure_of_sampling_adequacy": kmo_per_variable,
        }
    )
    eigen_table = pd.DataFrame(
        {
            "level": level,
            "component": component_names,
            "eigenvalue": eigenvalues,
            "variance_explained_percent": 100 * eigenvalues / eigenvalues.sum(),
            "cumulative_variance_percent": 100 * np.cumsum(eigenvalues) / eigenvalues.sum(),
            "parallel_analysis_threshold": thresholds,
            "exceeds_parallel_threshold": eigenvalues > thresholds,
            "exceeds_kaiser_unity": eigenvalues > 1.0,
            "retained": [index < retained for index in range(len(eigenvalues))],
        }
    )
    loading_table = pd.DataFrame(loadings, index=labels, columns=component_names)
    loading_table = loading_table.reset_index(names="dimension").assign(level=level, rotation="none")
    rotated_table = pd.DataFrame(
        rotated, index=labels, columns=component_names[:retained]
    ).reset_index(names="dimension").assign(level=level, rotation="varimax")

    return {
        "level": level,
        "correlation": correlation_frame,
        "eigenvalues": eigenvalues,
        "thresholds": thresholds,
        "retained": retained,
        "loadings": loadings,
        "rotated": rotated,
        "labels": labels,
        "eigenvectors": eigenvectors,
        "adequacy_table": adequacy,
        "eigen_table": eigen_table,
        "loading_table": pd.concat([loading_table, rotated_table], ignore_index=True),
    }


def plot_scree(ax: plt.Axes, result: dict[str, object], title: str) -> None:
    eigenvalues = np.asarray(result["eigenvalues"])
    thresholds = np.asarray(result["thresholds"])
    positions = np.arange(1, len(eigenvalues) + 1)
    ax.plot(positions, eigenvalues, marker="o", color="#1f4e79", lw=1.4, label="Observed eigenvalue")
    ax.plot(
        positions, thresholds, marker="^", color="#b91c1c", lw=1.2, linestyle="--", markersize=5,
        label="Permutation 95th percentile",
    )
    ax.axhline(1.0, color="#6b7280", lw=0.9, linestyle=":", label="Kaiser criterion")
    retained = int(result["retained"])
    ax.axvspan(0.5, retained + 0.5, color="#1f4e79", alpha=0.07, zorder=0)
    ax.set_xticks(positions)
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)


def breadth_class(active_dimensions: pd.Series) -> pd.Series:
    """Group articles by how many disruption dimensions they activate."""
    return pd.cut(
        active_dimensions,
        bins=[0, 2, 4, 7],
        labels=BREADTH_CLASSES,
        include_lowest=False,
    )


def plot_biplot(
    ax: plt.Axes, result: dict[str, object], scores: np.ndarray, classes: pd.Series, title: str
) -> None:
    """PC1-PC2 scores with variable loading vectors overlaid."""
    generator = np.random.default_rng(RANDOM_SEED)
    jitter = generator.normal(0.0, 0.03, scores.shape)
    for name in BREADTH_CLASSES:
        mask = (classes == name).to_numpy()
        ax.scatter(
            scores[mask, 0] + jitter[mask, 0], scores[mask, 1] + jitter[mask, 1],
            s=5, color=CLASS_COLORS[name], alpha=0.35, linewidths=0, label=name, zorder=2,
        )
    loadings = np.asarray(result["loadings"])
    scale = 0.9 * np.abs(scores[:, :2]).max()
    for index, label in enumerate(SHORT_LABELS[name] for name in DIMENSIONS):
        x, y = loadings[index, 0] * scale, loadings[index, 1] * scale
        ax.annotate(
            "", xy=(x, y), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#111827", lw=1.2, mutation_scale=11), zorder=4,
        )
        ax.text(x * 1.08, y * 1.08, label, fontsize=8, color="#111827", ha="center", va="center", zorder=5)
    ax.axhline(0.0, color="#6b7280", lw=0.7, linestyle=":")
    ax.axvline(0.0, color="#6b7280", lw=0.7, linestyle=":")
    eigen = result["eigen_table"]
    ax.set_xlabel(f"PC1 ({eigen['variance_explained_percent'].iloc[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({eigen['variance_explained_percent'].iloc[1]:.1f}% variance)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    legend = ax.legend(frameon=False, fontsize=8, title="Disruption breadth", loc="upper left")
    legend.get_title().set_fontsize(8)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
        handle.set_sizes([26])


def plot_pairplot(data: pd.DataFrame, classes: pd.Series) -> plt.Figure:
    """Scatter-matrix of the seven dimension scores, split by disruption breadth."""
    generator = np.random.default_rng(RANDOM_SEED)
    count = len(DIMENSIONS)
    figure, axes = plt.subplots(count, count, figsize=(13.5, 13.5))
    grid = np.linspace(-0.4, 4.4, 200)
    for row, y_name in enumerate(DIMENSIONS):
        for column, x_name in enumerate(DIMENSIONS):
            ax = axes[row, column]
            if row == column:
                for name in BREADTH_CLASSES:
                    values = data.loc[classes == name, x_name].to_numpy(dtype=float)
                    if values.std() < 1e-9:
                        continue
                    density = stats.gaussian_kde(values)(grid)
                    ax.fill_between(grid, density, color=CLASS_COLORS[name], alpha=0.35, lw=0)
                    ax.plot(grid, density, color=CLASS_COLORS[name], lw=1.0)
                ax.set_yticks([])
            else:
                for name in BREADTH_CLASSES:
                    subset = data.loc[classes == name]
                    x = subset[x_name].to_numpy(dtype=float) + generator.normal(0, 0.07, len(subset))
                    y = subset[y_name].to_numpy(dtype=float) + generator.normal(0, 0.07, len(subset))
                    ax.scatter(x, y, s=4, color=CLASS_COLORS[name], alpha=0.35, linewidths=0)
                ax.set_ylim(-0.45, 4.45)
            ax.set_xlim(-0.45, 4.45)
            if row < count - 1:
                ax.set_xticklabels([])
            if column > 0 and row != column:
                ax.set_yticklabels([])
            if row == count - 1:
                ax.set_xlabel(SHORT_LABELS[x_name], fontsize=8.5)
            if column == 0:
                ax.set_ylabel(SHORT_LABELS[y_name], fontsize=8.5)
            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle="--", alpha=0.25)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", color=CLASS_COLORS[name],
                   markersize=6, label=name)
        for name in BREADTH_CLASSES
    ]
    figure.legend(
        handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=10,
        bbox_to_anchor=(0.5, 1.004), title="Disruption breadth (active dimensions)",
        title_fontsize=10,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.975])
    return figure


def save_figure(figure: plt.Figure, basename: str) -> None:
    for extension in ("pdf", "png"):
        figure.savefig(
            FIGURE_DIR / f"{basename}.{extension}", format=extension, bbox_inches="tight", dpi=300
        )
    plt.close(figure)
    print(f"Figure written to {FIGURE_DIR / f'{basename}.pdf'}")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    articles = pd.read_csv(ARTICLE_SCORES, low_memory=False)
    score_columns = [f"{name}_disruption_score__ensemble_median" for name in DIMENSIONS]
    missing = [column for column in score_columns if column not in articles.columns]
    if missing:
        raise ValueError(f"Article score file lacks dimension column(s): {missing}")
    if "breadth_active_dimension_count" not in articles.columns:
        raise ValueError("Article score file lacks breadth_active_dimension_count")
    frame = articles[score_columns + ["breadth_active_dimension_count"]].dropna()
    data = frame[score_columns]
    data.columns = DIMENSIONS

    result = run_pca(data, "article", "spearman")
    result["adequacy_table"].to_csv(TABLE_DIR / "pca_article_adequacy.csv", index=False)
    result["eigen_table"].to_csv(TABLE_DIR / "pca_article_eigenvalues.csv", index=False)
    result["loading_table"].to_csv(TABLE_DIR / "pca_article_loadings.csv", index=False)
    print(
        f"article: n={len(data)}, retained={result['retained']}, "
        f"PC1 variance={result['eigen_table']['variance_explained_percent'].iloc[0]:.1f}%, "
        f"KMO={result['adequacy_table']['kmo_overall'].iloc[0]:.3f}"
    )

    standardized = (data - data.mean()) / data.std(ddof=1)
    eigenvectors = np.asarray(result["eigenvectors"])
    all_scores = standardized.to_numpy() @ eigenvectors
    retained = int(result["retained"])
    pd.DataFrame(
        all_scores[:, :retained], columns=[f"PC{index + 1}" for index in range(retained)]
    ).to_csv(TABLE_DIR / "pca_article_component_scores.csv", index=False)

    # Components are estimated on every scored article. The visual panels show only
    # articles activating at least one dimension: the rest carry an all-zero profile and
    # would collapse into a single uninformative point at the origin.
    classes = breadth_class(frame["breadth_active_dimension_count"])
    active = classes.notna().to_numpy()
    active_data = data.loc[active]
    active_classes = classes[active]
    print(f"active articles plotted: {int(active.sum())} of {len(data)}")

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    plot_scree(axes[0], result, "A. Scree and parallel analysis")
    plot_biplot(
        axes[1], result, all_scores[active][:, :2], active_classes,
        "B. Component scores and dimension loadings",
    )
    figure.tight_layout()
    save_figure(figure, "eda_figure05_dimension_pca")

    save_figure(plot_pairplot(active_data, active_classes), "eda_figure06_dimension_pairplot")


if __name__ == "__main__":
    main()
