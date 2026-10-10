"""Scripts 47 (forecast trajectories), 48 (PCMCI comparison), 46 (interaction figures)."""
import sys
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from common import (REPO, load, restyle, resize, save, fit_save, measure, slim, shared_labels,
                    legend_bottom, constrained, capture, WIDTH_IN, TICK, LABEL, TITLE)


def monthly_model_forecasts():
    s47 = load("47_monthly_forecast_trajectories_by_horizon.py", "s47")
    predictors, baseline = s47.load("adapted_pdf_sqrt_ema3")
    with capture():
        fig = s47.make_figure(predictors, baseline)
    resize(fig, 7.6)
    slim(fig, marker=0.7, line=0.75)
    for ax in fig.axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    legend_bottom(fig, ncol=4, loc="outside upper center")
    constrained(fig)
    restyle(fig)
    return fig


def pcmci_links():
    """Same panels, data and drawing routine as script 48's plot_comparison(),
    laid out 2x2 (legend in the fourth cell) so node text fits the circles."""
    s48 = load("48_pcmci_gpdc_nonlinear_robustness.py", "s48")
    published = pd.read_csv(s48.PUBLISHED_LINKS)
    gpdc = pd.read_csv(s48.OUTPUT_DIR / "causality_gpdc_monthly_pcmci_links.csv")
    gpdc["partial_correlation"] = gpdc["dependence_measure"]
    panels = [
        (published[published["frequency"].eq("weekly")], "PCMCI+ links: weekly (peak), ParCorr"),
        (published[published["frequency"].eq("monthly")], "PCMCI+ links: monthly (peak), ParCorr"),
        (gpdc, "PCMCI+ links: monthly (peak), GPDC"),
    ]
    stage24 = s48.load_stage24()
    colors = s48.PUBLICATION_COLORS
    import pcmci_draw
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "STIXGeneral"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 5.2))
    constrained(fig, wspace=0.0, hspace=0.0)
    for ax in axes.ravel():
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-1.95, 1.95); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("x")
    fig.canvas.draw()
    unit_pt = (axes[0, 0].transData.transform((1, 0)) - axes[0, 0].transData.transform((0, 0)))[0] * 72 / fig.dpi
    print(f"PCMCI data unit = {unit_pt:.1f} pt")
    for ax, (frame, title) in zip(axes.ravel(), panels):
        subset = frame[frame["msdi_path"].eq("peak") & frame["significant_fdr"]]
        pcmci_draw.draw(ax, subset, colors, title, stage24, unit_pt)
    legend_ax = axes[1, 1]
    legend_ax.set_title("")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#eef2f7",
               markeredgecolor="#111827", markersize=12, label="Variable (node)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor=colors["robustness"], markersize=8,
               label="Self-loop: own-lag autocorrelation"),
        Line2D([0], [0], marker=r"$\rightarrow$", linestyle="none",
               color=colors["primary"], markersize=12, label="Arrow: FDR-significant cross-link"),
    ]
    restyle(fig)
    legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=TICK)
    return fig


def _s46():
    return load("46_interaction_structure_forecast_target.py", "s46")


def interaction_strength():
    s46 = _s46()
    stats = pd.read_csv(REPO / "outputs/reports/interaction_structure_h_statistics.csv")
    assert list(stats["feature"]) == list(s46.FEATURES.values())
    strengths = {name: stats[f"h_overall_{name.lower().replace(' ', '_')}"].to_numpy()
                 for name in s46.PALETTE}
    with capture():
        fig = s46.make_strength_figure(strengths)
    resize(fig, 3.6)
    slim(fig, marker=0.8, line=0.85)
    legend_bottom(fig, ncol=3)
    constrained(fig)
    restyle(fig)
    return fig


def pairwise_pdp_grid():
    """Script 46's make_pdp_grid() logic (same RF, PDPs, colour limits, ordering),
    laid out 5x3 at text width. The seeded RF refit is checked against the saved H values."""
    from sklearn.inspection import partial_dependence
    s46 = _s46()
    X, y = s46.load_design("target_scfi_monthly_log_change_h6m")
    model = s46.fit_models(X, y)["Random forest"]
    matrix = s46.h_matrix(model, X)

    labels = list(s46.FEATURES.values())
    saved = pd.read_csv(REPO / "outputs/reports/interaction_structure_pairwise_h.csv")
    for i, j in combinations(range(len(labels)), 2):
        want = saved.loc[saved["pair"] == f"{labels[i]} x {labels[j]}", "Random forest"].item()
        assert abs(matrix[i, j] - want) < 1e-12, (labels[i], labels[j], matrix[i, j], want)
    print("RF refit reproduces all 15 saved pairwise H values (|diff| < 1e-12)")

    pairs = sorted(combinations(range(len(labels)), 2), key=lambda ij: -matrix[ij[0], ij[1]])
    surfaces = []
    for i, j in pairs:
        result = partial_dependence(model, X, [i, j], grid_resolution=30)
        grid = result.get("grid_values", result.get("values"))
        surfaces.append((i, j, grid[0], grid[1], result["average"][0].T))
    pooled = np.concatenate([z.ravel() for *_, z in surfaces])
    vmin, vmax = np.percentile(pooled, [2.0, 98.0])
    levels = np.linspace(vmin, vmax, 15)

    fig, axes = plt.subplots(5, 3, figsize=(WIDTH_IN, 7.9))  # figure + caption must fit 9 in text height
    contour = None
    for ax, (i, j, gx, gy, z) in zip(axes.ravel(), surfaces):
        mesh_x, mesh_y = np.meshgrid(gx, gy)
        contour = ax.contourf(mesh_x, mesh_y, np.clip(z, vmin, vmax), levels=levels,
                              cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{labels[i]} $\\times$ {labels[j]}\n$H_{{jk}}$ = {matrix[i, j]:.3f}",
                     fontweight="bold", pad=3)
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.locator_params(axis="both", nbins=4)
    bar = fig.colorbar(contour, ax=axes, fraction=0.03, pad=0.02, aspect=40)
    bar.set_label("Predicted SCFI log change, 6 months")
    restyle(fig)
    constrained(fig, hspace=0.03)
    return fig


BUILD = {f.__name__: f for f in (monthly_model_forecasts, pcmci_links,
                                  interaction_strength, pairwise_pdp_grid)}

if __name__ == "__main__":
    for name in sys.argv[1:] or BUILD:
        measure(fit_save(BUILD[name](), name))
