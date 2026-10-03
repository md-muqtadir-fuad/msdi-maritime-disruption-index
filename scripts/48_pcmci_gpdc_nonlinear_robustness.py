"""PCMCI+ conditional dependence under a nonlinear independence test, monthly.

Stage 19 runs PCMCI+ with ParCorr, a linear partial-correlation test, so its null result
bounds linear conditional dependence only. A reviewer can reasonably ask whether a
nonlinear dependence was missed. This script repeats the monthly discovery with GPDC
(Gaussian-process distance correlation), which detects nonlinear conditional dependence,
and leaves stage 19 untouched.

Design, preprocessing, lag order, pc_alpha, alpha and FDR handling are taken from stage 19
by importing its own functions, so the only thing that changes is the independence test.
Running with --ci-test parcorr reproduces the published links and is the parity check that
the reconstruction is faithful; --ci-test gpdc is the robustness run.

Monthly only. The weekly system has n=490 and tau_max=12, where GPDC's Gaussian-process
fits are roughly two orders of magnitude more costly per test; it is not attempted here.

GPDC requires `dcor`. On Windows, numba's on-disk cache can fail to write inside
site-packages, so set NUMBA_DISABLE_JIT=1 if the import raises FileNotFoundError.

Source: data/06_model_ready/causality/msdi_scfi_monthly_panel.csv
Output: outputs/time_series_causality/causality_gpdc_monthly_pcmci_links.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE19 = PROJECT_ROOT / "scripts/19_time_series_causality.py"
MONTHLY_PANEL = PROJECT_ROOT / "data/06_model_ready/causality/msdi_scfi_monthly_panel.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/time_series_causality"
PUBLISHED_LINKS = OUTPUT_DIR / "causality_v001_pcmci_links.csv"


def load_stage19():
    """Import stage 19 so the frame construction cannot drift from the published run."""
    if not STAGE19.is_file():
        raise FileNotFoundError(f"Stage 19 not found: {STAGE19}")
    spec = importlib.util.spec_from_file_location("stage19_causality", STAGE19)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage19_causality"] = module
    spec.loader.exec_module(module)
    return module


def monthly_specification(stage19, maximum_lag: int):
    return stage19.FrequencySpecification(
        frequency="monthly",
        date_column="month_ending",
        scfi_column="scfi_monthly_mean",
        brent_column="brent_monthly_mean",
        article_count_column="article_count_sum",
        msdi_columns={
            "peak": "peak_score_media_corrected_mean",
            "breadth": "breadth_score_media_corrected_mean",
            "non_freight_peak": "non_freight_peak_score_media_corrected_mean",
        },
        maximum_lag=maximum_lag,
        lp_horizon=6,
        irf_horizon=6,
    )


def build_test(ci_test: str):
    if ci_test == "parcorr":
        from tigramite.independence_tests.parcorr import ParCorr

        return ParCorr(significance="analytic"), "ParCorr_analytic"
    from tigramite.independence_tests.gpdc import GPDC

    return GPDC(significance="analytic"), "GPDC_analytic"


def discover(
    stage19,
    frame: pd.DataFrame,
    path_name: str,
    maximum_lag: int,
    pc_alpha: float,
    alpha: float,
    ci_test: str,
) -> pd.DataFrame:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI

    standardized = stage19.residualize_seasonality(frame)
    names = list(standardized.columns)
    tigramite_frame = pp.DataFrame(standardized.to_numpy(dtype=float), var_names=names)

    test, test_label = build_test(ci_test)
    pcmci = PCMCI(dataframe=tigramite_frame, cond_ind_test=test, verbosity=0)
    result = pcmci.run_pcmci(
        tau_min=1,
        tau_max=maximum_lag,
        pc_alpha=pc_alpha,
        alpha_level=alpha,
        fdr_method="none",
    )
    p_matrix = result["p_matrix"]
    adjusted = pcmci.get_corrected_pvalues(
        p_matrix=p_matrix,
        fdr_method="fdr_bh",
        exclude_contemporaneous=True,
        tau_min=1,
        tau_max=maximum_lag,
    )
    values = result["val_matrix"]

    rows: list[dict[str, Any]] = []
    for i, source in enumerate(names):
        for j, target in enumerate(names):
            for lag in range(1, maximum_lag + 1):
                rows.append(
                    {
                        "frequency": "monthly",
                        "msdi_path": path_name,
                        "source": source,
                        "target": target,
                        "source_lag": lag,
                        "dependence_measure": float(values[i, j, lag]),
                        "p_value": float(p_matrix[i, j, lag]),
                        "p_value_fdr_bh": float(adjusted[i, j, lag]),
                        "significant_fdr": bool(adjusted[i, j, lag] < alpha),
                        "conditional_independence_test": test_label,
                        "pc_alpha": pc_alpha,
                        "alpha": alpha,
                        "n_observations": int(len(standardized)),
                        "interpretation": (
                            "conditional_temporal_dependency_not_structural_causality"
                        ),
                    }
                )
    return pd.DataFrame(rows)


STAGE24 = PROJECT_ROOT / "scripts/24_make_figures_tables.py"
PUBLICATION_COLORS = {
    "primary": "#1f4e79",
    "secondary": "#c55a11",
    "robustness": "#6b7280",
    "warning": "#b91c1c",
}


def load_stage24():
    """Import stage 24 so the graph is drawn by the same routine as the published
    PCMCI+ figure. Stage 24 carries a SHA-256 self-lock; it is imported, never edited."""
    if not STAGE24.is_file():
        raise FileNotFoundError(f"Stage 24 not found: {STAGE24}")
    spec = importlib.util.spec_from_file_location("stage24_publication", STAGE24)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage24_publication"] = module
    spec.loader.exec_module(module)
    return module


def plot_comparison() -> Path | None:
    """One figure: weekly and monthly under the linear test, monthly under the nonlinear
    test. Peak path throughout, matching the published figure's convention. Drawn by
    stage 24's own routine so the style is identical rather than imitated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    gpdc_path = OUTPUT_DIR / "causality_gpdc_monthly_pcmci_links.csv"
    if not (gpdc_path.is_file() and PUBLISHED_LINKS.is_file()):
        print("Published ParCorr links and the GPDC run are both needed; figure skipped.")
        return None

    published = pd.read_csv(PUBLISHED_LINKS)
    gpdc = pd.read_csv(gpdc_path)
    gpdc["partial_correlation"] = gpdc["dependence_measure"]

    panels = [
        (published[published["frequency"].eq("weekly")],
         "PCMCI+ links: weekly (peak), ParCorr"),
        (published[published["frequency"].eq("monthly")],
         "PCMCI+ links: monthly (peak), ParCorr"),
        (gpdc, "PCMCI+ links: monthly (peak), GPDC"),
    ]

    stage24 = load_stage24()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), dpi=300)
    for ax, (frame, title) in zip(axes, panels):
        subset = frame[frame["msdi_path"].eq("peak") & frame["significant_fdr"]]
        stage24.draw_pcmci_causal_graph(ax, subset, PUBLICATION_COLORS, title)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#eef2f7",
               markeredgecolor="#111827", markersize=15, label="Variable (node)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor=PUBLICATION_COLORS["robustness"], markersize=9,
               label="Self-loop: own-lag autocorrelation"),
        Line2D([0], [0], marker=r"$\rightarrow$", linestyle="none",
               color=PUBLICATION_COLORS["primary"], markersize=16,
               label="Arrow: FDR-significant cross-link"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(wspace=0.02, top=0.93, bottom=0.12, left=0.02, right=0.98)

    figure_dir = PROJECT_ROOT / "outputs/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    base = figure_dir / "figure23_pcmci_linear_vs_nonlinear"
    fig.savefig(f"{base}.pdf", format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{base}.png", format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return base


def parity_report(found: pd.DataFrame) -> None:
    """Compare a ParCorr rerun against the published links."""
    if not PUBLISHED_LINKS.is_file():
        print("Published links not found; parity check skipped.")
        return
    published = pd.read_csv(PUBLISHED_LINKS)
    published = published[published["frequency"] == "monthly"]
    key = ["msdi_path", "source", "target", "source_lag"]
    merged = found.merge(
        published[key + ["p_value", "significant_fdr"]],
        on=key,
        suffixes=("_new", "_published"),
    )
    if merged.empty:
        print("Parity check: no overlapping rows.")
        return
    gap = (merged["p_value_new"] - merged["p_value_published"]).abs().max()
    agree = int((merged["significant_fdr_new"] == merged["significant_fdr_published"]).sum())
    print(f"Parity check on {len(merged)} links: max |p difference| = {gap:.2e}, "
          f"significance agrees on {agree}/{len(merged)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-test", default="gpdc", choices=("gpdc", "parcorr"))
    parser.add_argument("--maximum-lag", type=int, default=6)
    parser.add_argument("--pc-alpha", type=float, default=0.20)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    stage19 = load_stage19()
    spec = monthly_specification(stage19, args.maximum_lag)
    panel = pd.read_csv(MONTHLY_PANEL)
    panel[spec.date_column] = pd.to_datetime(panel[spec.date_column])

    frames = []
    for path_name in ("peak", "breadth"):
        prepared = stage19.prepare_analysis_frame(panel, spec, path_name)
        analysis_frame, _ = stage19.select_longest_complete_block(
            prepared, stage19.analysis_columns(prepared), spec.frequency
        )
        print(f"{path_name}: {len(analysis_frame)} contiguous complete months")
        frames.append(
            discover(
                stage19,
                analysis_frame,
                path_name,
                args.maximum_lag,
                args.pc_alpha,
                args.alpha,
                args.ci_test,
            )
        )

    links = pd.concat(frames, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "parcorr" if args.ci_test == "parcorr" else "gpdc"
    destination = OUTPUT_DIR / f"causality_{suffix}_monthly_pcmci_links.csv"
    links.to_csv(destination, index=False)

    if args.ci_test == "parcorr":
        parity_report(links)

    significant = links[links["significant_fdr"]]
    print(f"\nFDR-significant links ({args.ci_test}): {len(significant)} of {len(links)}")
    if not significant.empty:
        print(
            significant[
                ["msdi_path", "source", "target", "source_lag",
                 "dependence_measure", "p_value", "p_value_fdr_bh"]
            ].round(4).to_string(index=False)
        )

    into_scfi = significant[
        significant["target"].eq("scfi_return") & significant["source"].ne("scfi_return")
    ]
    print(f"\nCross-variable links into SCFI return: {len(into_scfi)}")
    print(f"Written to {destination}")

    figure_base = plot_comparison()
    if figure_base is not None:
        print(f"Comparison figure written to {figure_base}.pdf")


if __name__ == "__main__":
    main()
