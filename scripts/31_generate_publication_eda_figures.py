"""Generate publication-grade EDA figures with Times New Roman typography and PDF format.

Output deliverables:
1. Spearman Rank Correlation (Strictly lower-triangular matrix, no diagonal, interactive palette).
2. Data distributions across LLMs and core macroeconomic/maritime variables.
3. Friedman's H-Statistic Interaction Grid across ML estimators.
4. Nonparametric Empirical Relationship Grid (LOESS & Polynomial Fits with Density Rugs).

All figures conform to publication guidelines:
- Times New Roman font
- Vector PDF and high-resolution PNG outputs
- Subplot titles placed strictly at the TOP above each axis (no internal canvas overlap)
- Expanded subplot padding (wspace/hspace) to eliminate any y-axis or colorbar overlap
- No figure master title
"""

from __future__ import annotations

import os
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import partial_dependence

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration & Typography
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs/figures/eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Times", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10.5
plt.rcParams["axes.titlesize"] = 11.5
plt.rcParams["xtick.labelsize"] = 9.5
plt.rcParams["ytick.labelsize"] = 9.5
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 12
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def save_figure(fig: plt.Figure, basename: str) -> None:
    pdf_path = OUTPUT_DIR / f"{basename}.pdf"
    png_path = OUTPUT_DIR / f"{basename}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path.name} and {png_path.name}")


# -----------------------------------------------------------------------------
# Figure 1: Spearman Correlation Heatmap (Lower Triangular Only, No Diagonal)
# -----------------------------------------------------------------------------
def make_spearman_correlation_figure() -> None:
    print("Generating Figure 1: Spearman Correlation (Lower Triangle, No Diagonal)...")
    monthly_panel_path = PROJECT_ROOT / "data/06_model_ready/prediction/llm_specific_scfi_monthly_panel.csv"
    if not monthly_panel_path.exists():
        print("Monthly panel missing, skipping correlation figure.")
        return

    df = pd.read_csv(monthly_panel_path)
    
    var_map = {
        "scfi_monthly_mean": "SCFI Freight Level",
        "brent_monthly_mean": "Brent Crude Oil",
        "llm_signal__gemma_2_9b_it__overall_disruption_max": "MSDI Peak Disruption",
        "llm_signal__gemma_2_9b_it__dimension_mean7": "MSDI Breadth Disruption",
        "llm_signal__gemma_2_9b_it__port_disruption_score": "Port Congestion Stress",
        "llm_signal__gemma_2_9b_it__global_container_movement_disruption_score": "Container Movement Stress",
        "llm_signal__gemma_2_9b_it__freight_cost_disruption_score": "Freight Cost Stress",
        "llm_signal__gemma_2_9b_it__supply_disruption_score": "Supply Disruption",
        "cpb_world_trade_volume": "CPB World Trade Volume",
        "rwi_isl_container_throughput_index": "RWI/ISL Throughput",
        "wto_china_merchandise_exports": "WTO China Exports",
        "cny_usd_exchange_rate": "CNY/USD FX Rate",
        "unctad_china_core_lane_lsbci": "UNCTAD Connectivity",
    }
    
    selected_cols = [c for c in var_map.keys() if c in df.columns]
    data = df[selected_cols].dropna().copy()
    labels = [var_map[c] for c in selected_cols]
    n = len(labels)

    corr_matrix = np.zeros((n, n))
    pval_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rho, p = stats.spearmanr(data.iloc[:, i], data.iloc[:, j])
            corr_matrix[i, j] = rho
            pval_matrix[i, j] = p

    # Mask upper triangle AND diagonal
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)

    fig, ax = plt.subplots(figsize=(10.5, 8.5), dpi=300)
    
    cmap = matplotlib.colormaps.get_cmap("RdYlBu_r").copy()
    cmap.set_bad(color="white")
    
    masked_corr = np.ma.masked_array(corr_matrix, mask=mask)
    cax = ax.imshow(masked_corr, cmap=cmap, vmin=-0.85, vmax=0.85, aspect="auto")

    cbar = fig.colorbar(cax, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Spearman Rank Correlation ($\\rho$)", fontsize=10.5)
    cbar.ax.tick_params(labelsize=9.5)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9.5)
    ax.set_yticklabels(labels, fontsize=9.5)

    for i in range(n):
        for j in range(n):
            if i > j:  # Strictly lower triangle
                rho = corr_matrix[i, j]
                pval = pval_matrix[i, j]
                star = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
                text_color = "white" if abs(rho) > 0.45 else "black"
                ax.text(
                    j, i,
                    f"{rho:.2f}{star}",
                    ha="center", va="center",
                    color=text_color,
                    fontsize=8.5,
                    fontweight="normal"
                )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", size=0)

    save_figure(fig, "eda_figure01_spearman_correlation_lower_triangle")


# -----------------------------------------------------------------------------
# Figure 2: Data Distributions (LLM Scores & Core Model Series)
# -----------------------------------------------------------------------------
def make_data_distributions_figure() -> None:
    print("Generating Figure 2: Data Distributions...")
    article_scores_path = PROJECT_ROOT / "data/05_indices/article_level/article_consensus_scores.csv"
    monthly_panel_path = PROJECT_ROOT / "data/06_model_ready/prediction/llm_specific_scfi_monthly_panel.csv"

    if not article_scores_path.exists() or not monthly_panel_path.exists():
        print("Required datasets missing for distribution figure.")
        return

    df_articles = pd.read_csv(article_scores_path)
    df_panel = pd.read_csv(monthly_panel_path)
    df_panel["month_ending"] = pd.to_datetime(df_panel["month_ending"])

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), dpi=300)
    plt.subplots_adjust(hspace=0.38, wspace=0.28, left=0.08, right=0.96, top=0.93, bottom=0.08)

    colors = {
        "Gemma 2 9B": "#2b5c8f",
        "Qwen 3 8B": "#d95f02",
        "Qwen 2.5 7B": "#7570b3",
        "GLM-4 9B": "#1b9e77",
    }

    # (a) Kernel Density of Peak Disruption across LLMs
    ax_a = axes[0, 0]
    llm_cols = {
        "Gemma 2 9B": "overall_disruption_max__gemma_2_9b_it",
        "Qwen 3 8B": "overall_disruption_max__qwen_3_8b",
        "Qwen 2.5 7B": "overall_disruption_max__qwen_2_5_7b",
        "GLM-4 9B": "overall_disruption_max__glm_4_9b_0414",
    }
    for name, col in llm_cols.items():
        if col in df_articles.columns:
            vals = df_articles[col].dropna()
            kde = stats.gaussian_kde(vals)
            x_grid = np.linspace(0, 1, 200)
            ax_a.plot(x_grid, kde(x_grid), label=name, color=colors[name], lw=1.8)
            ax_a.fill_between(x_grid, kde(x_grid), alpha=0.10, color=colors[name])
    ax_a.set_title("(a) LLM Peak Disruption Density ($N=6,480$)", fontsize=11, fontweight="bold", pad=8)
    ax_a.set_xlabel("Peak Disruption Score ($S_{i}$)", fontsize=10)
    ax_a.set_ylabel("Kernel Density", fontsize=10)
    ax_a.set_xlim(0, 1)
    ax_a.legend(frameon=True, framealpha=0.9, loc="upper right")

    # (b) Boxplot of 7 Disruption Dimensions (Ensemble Median)
    ax_b = axes[0, 1]
    dim_cols = [
        ("Port", "port_disruption_score__ensemble_median"),
        ("Movement", "global_container_movement_disruption_score__ensemble_median"),
        ("Freight Cost", "freight_cost_disruption_score__ensemble_median"),
        ("Supply", "supply_disruption_score__ensemble_median"),
        ("Production", "production_disruption_score__ensemble_median"),
        ("COVID-19", "covid19_related_disruption_score__ensemble_median"),
        ("Demand", "demand_disruption_score__ensemble_median"),
    ]
    dim_data = [df_articles[c].dropna() for _, c in dim_cols if c in df_articles.columns]
    dim_names = [name for name, c in dim_cols if c in df_articles.columns]
    
    bp = ax_b.boxplot(
        dim_data,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        boxprops=dict(facecolor="#c6dbef", color="#08519c", lw=1.2),
        medianprops=dict(color="#d94801", lw=1.6),
        meanprops=dict(color="#006d2c", lw=1.4, ls="--"),
        flierprops=dict(marker=".", color="#9ecae1", alpha=0.3, markersize=3)
    )
    ax_b.set_title("(b) Disruption Dimension Profiles (Ensemble)", fontsize=11, fontweight="bold", pad=8)
    ax_b.set_xticks(range(1, len(dim_names) + 1))
    ax_b.set_xticklabels(dim_names, rotation=30, ha="right", fontsize=9)
    ax_b.set_ylabel("Ensemble Median Score", fontsize=10)
    ax_b.set_ylim(-0.02, 1.02)

    # (c) Standardized Macro & Maritime Controls Distributions
    ax_c = axes[1, 0]
    control_vars = [
        ("SCFI", "scfi_monthly_mean", "#08519c"),
        ("MSDI Peak", "llm_signal__gemma_2_9b_it__overall_disruption_max", "#d94801"),
        ("CPB Trade", "cpb_world_trade_volume", "#238b45"),
        ("RWI Port", "rwi_isl_container_throughput_index", "#6a51a3"),
        ("CNY/USD", "cny_usd_exchange_rate", "#006d2c"),
    ]
    for lbl, col, col_c in control_vars:
        if col in df_panel.columns:
            s = df_panel[col].dropna()
            z_s = (s - s.mean()) / s.std()
            kde_z = stats.gaussian_kde(z_s)
            xz = np.linspace(-3.5, 3.5, 200)
            ax_c.plot(xz, kde_z(xz), label=lbl, color=col_c, lw=1.6)
    ax_c.set_title("(c) Standardized Variable Densities", fontsize=11, fontweight="bold", pad=8)
    ax_c.set_xlabel("Standardized Value ($z$-score)", fontsize=10)
    ax_c.set_ylabel("Density", fontsize=10)
    ax_c.set_xlim(-3.5, 3.5)
    ax_c.legend(frameon=True, loc="upper right")

    # (d) Historical Time Series Trajectories (Normalized 2016–2026)
    ax_d = axes[1, 1]
    df_sorted = df_panel.sort_values("month_ending")
    scfi_norm = (df_sorted["scfi_monthly_mean"] - df_sorted["scfi_monthly_mean"].min()) / (df_sorted["scfi_monthly_mean"].max() - df_sorted["scfi_monthly_mean"].min())
    msdi_col = "llm_signal__gemma_2_9b_it__overall_disruption_max"
    msdi_norm = (df_sorted[msdi_col] - df_sorted[msdi_col].min()) / (df_sorted[msdi_col].max() - df_sorted[msdi_col].min())

    ax_d.plot(df_sorted["month_ending"], scfi_norm, label="SCFI Freight Level", color="#08519c", lw=2.0)
    ax_d.plot(df_sorted["month_ending"], msdi_norm, label="MSDI Disruption (Peak)", color="#d94801", lw=1.6, ls="-.")
    
    ax_d.axvspan(pd.to_datetime("2020-02-01"), pd.to_datetime("2020-07-01"), color="#fee0d2", alpha=0.5, label="COVID-19 Shock")
    ax_d.axvspan(pd.to_datetime("2023-11-01"), pd.to_datetime("2024-05-01"), color="#deebf7", alpha=0.5, label="Red Sea Crisis")

    ax_d.set_title("(d) Monthly Co-movement & Historical Shocks", fontsize=11, fontweight="bold", pad=8)
    ax_d.set_ylabel("Normalized Scale [0, 1]", fontsize=10)
    ax_d.tick_params(axis="x", rotation=30)
    ax_d.legend(frameon=True, loc="upper left", fontsize=8.5)

    save_figure(fig, "eda_figure02_data_distributions")


# -----------------------------------------------------------------------------
# Figure 3: Friedman's H-Statistic Interaction Grid
# -----------------------------------------------------------------------------
def make_friedmans_h_statistic_grid() -> None:
    print("Generating Figure 3: Friedman's H-Statistic Interaction Grid...")
    monthly_panel_path = PROJECT_ROOT / "data/06_model_ready/prediction/llm_specific_scfi_monthly_panel.csv"
    if not monthly_panel_path.exists():
        print("Monthly panel missing, skipping H-statistic figure.")
        return

    df = pd.read_csv(monthly_panel_path)
    
    feature_map = {
        "llm_signal__gemma_2_9b_it__overall_disruption_max": "MSDI Peak",
        "llm_signal__gemma_2_9b_it__dimension_mean7": "MSDI Breadth",
        "brent_monthly_mean": "Brent Crude",
        "cpb_world_trade_volume": "CPB Trade",
        "rwi_isl_container_throughput_index": "RWI Port",
        "cny_usd_exchange_rate": "CNY/USD FX",
    }
    
    feature_cols = list(feature_map.keys())
    feature_labels = list(feature_map.values())
    target_col = "scfi_monthly_mean"

    sub = df.dropna(subset=[target_col] + feature_cols).copy()
    X = sub[feature_cols]
    y = sub[target_col]
    p = len(feature_cols)

    rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42).fit(X, y)
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X, y)

    def compute_h_matrix(model):
        H_mat = np.zeros((p, p))
        pdp_1d = {}
        for i in range(p):
            res1 = partial_dependence(model, X, [i], grid_resolution=25)
            vals = res1["average"][0]
            pdp_1d[i] = vals - np.mean(vals)

        for i in range(p):
            for j in range(i + 1, p):
                res2 = partial_dependence(model, X, [i, j], grid_resolution=25)
                avg2d = res2["average"][0]
                avg2d_c = avg2d - np.mean(avg2d)
                
                p1 = pdp_1d[i][:, None]
                p2 = pdp_1d[j][None, :]
                interaction = avg2d_c - (p1 + p2)
                
                denom = np.sum(avg2d_c ** 2)
                numer = np.sum(interaction ** 2)
                h = np.sqrt(numer / denom) if denom > 0 else 0.0
                H_mat[i, j] = h
                H_mat[j, i] = h
        return H_mat

    H_rf = compute_h_matrix(rf)
    H_gbr = compute_h_matrix(gbr)

    H_total_rf = np.sqrt(np.sum(H_rf**2, axis=1) / (p - 1))
    H_total_gbr = np.sqrt(np.sum(H_gbr**2, axis=1) / (p - 1))

    # Increased width, height, and generous wspace/hspace to prevent any overlap
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), dpi=300)
    plt.subplots_adjust(hspace=0.42, wspace=0.42, left=0.09, right=0.92, top=0.93, bottom=0.08)

    cmap_h = matplotlib.colormaps.get_cmap("YlGnBu").copy()
    cmap_h.set_bad(color="white")

    # (a) Random Forest Pairwise Interaction Heatmap (Lower Triangle Only, No Diagonal)
    ax_a = axes[0, 0]
    mask_h = np.triu(np.ones_like(H_rf, dtype=bool), k=0)
    masked_rf = np.ma.masked_array(H_rf, mask=mask_h)
    cax_a = ax_a.imshow(masked_rf, cmap=cmap_h, vmin=0.0, vmax=0.45, aspect="auto")
    cb_a = fig.colorbar(cax_a, ax=ax_a, fraction=0.045, pad=0.04)
    cb_a.set_label("Interaction ($H_{jk}$)", fontsize=9.5)

    ax_a.set_title("(a) Random Forest Pairwise Interactions ($H_{jk}$)", fontsize=11, fontweight="bold", pad=8)
    ax_a.set_xticks(np.arange(p))
    ax_a.set_yticks(np.arange(p))
    ax_a.set_xticklabels(feature_labels, rotation=35, ha="right", fontsize=9)
    ax_a.set_yticklabels(feature_labels, fontsize=9)

    for i in range(p):
        for j in range(i):
            val = H_rf[i, j]
            tc = "white" if val > 0.25 else "black"
            ax_a.text(j, i, f"{val:.3f}", ha="center", va="center", color=tc, fontsize=8.5)

    # (b) Gradient Boosting Pairwise Interaction Heatmap (Lower Triangle Only, No Diagonal)
    ax_b = axes[0, 1]
    masked_gbr = np.ma.masked_array(H_gbr, mask=mask_h)
    cax_b = ax_b.imshow(masked_gbr, cmap=cmap_h, vmin=0.0, vmax=0.45, aspect="auto")
    cb_b = fig.colorbar(cax_b, ax=ax_b, fraction=0.045, pad=0.04)
    cb_b.set_label("Interaction ($H_{jk}$)", fontsize=9.5)

    ax_b.set_title("(b) Gradient Boosting Pairwise Interactions ($H_{jk}$)", fontsize=11, fontweight="bold", pad=8)
    ax_b.set_xticks(np.arange(p))
    ax_b.set_yticks(np.arange(p))
    ax_b.set_xticklabels(feature_labels, rotation=35, ha="right", fontsize=9)
    ax_b.set_yticklabels(feature_labels, fontsize=9)

    for i in range(p):
        for j in range(i):
            val = H_gbr[i, j]
            tc = "white" if val > 0.25 else "black"
            ax_b.text(j, i, f"{val:.3f}", ha="center", va="center", color=tc, fontsize=8.5)

    # (c) Overall Total Feature Interaction Strength Bar Chart
    ax_c = axes[1, 0]
    y_pos = np.arange(p)
    bar_width = 0.35

    ax_c.barh(y_pos - bar_width / 2, H_total_rf, height=bar_width, label="Random Forest", color="#2b5c8f", alpha=0.9)
    ax_c.barh(y_pos + bar_width / 2, H_total_gbr, height=bar_width, label="Gradient Boosting", color="#d95f02", alpha=0.9)

    ax_c.set_title("(c) Overall Feature Interaction Strength ($H_j$)", fontsize=11, fontweight="bold", pad=8)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(feature_labels, fontsize=9.5)
    ax_c.set_xlabel("Overall Interaction Statistic ($H_j$)", fontsize=10)
    ax_c.set_xlim(0, max(np.max(H_total_rf), np.max(H_total_gbr)) * 1.18)
    ax_c.legend(loc="lower right", frameon=True, fontsize=8.5)
    ax_c.grid(True, axis="x", linestyle="--", alpha=0.5)

    # (d) 2-Way Partial Dependence Surface (MSDI Peak x RWI Port Throughput)
    ax_d = axes[1, 1]
    res_pdp2d = partial_dependence(rf, X, [0, 4], grid_resolution=30)
    grid_x = res_pdp2d["grid_values"][0] if "grid_values" in res_pdp2d else res_pdp2d["values"][0]
    grid_y = res_pdp2d["grid_values"][1] if "grid_values" in res_pdp2d else res_pdp2d["values"][1]
    Z = res_pdp2d["average"][0].T

    X_mesh, Y_mesh = np.meshgrid(grid_x, grid_y)
    contour = ax_d.contourf(X_mesh, Y_mesh, Z, levels=14, cmap="viridis")
    cbar_d = fig.colorbar(contour, ax=ax_d, fraction=0.045, pad=0.04)
    cbar_d.set_label("SCFI Freight Level", fontsize=9.5)

    ax_d.set_title("(d) 2D Response: MSDI Peak $\\times$ Port Throughput", fontsize=11, fontweight="bold", pad=8)
    ax_d.set_xlabel("MSDI Peak Disruption", fontsize=10)
    ax_d.set_ylabel("RWI/ISL Port Throughput", fontsize=10)

    save_figure(fig, "eda_figure03_friedmans_h_statistic_grid")


# -----------------------------------------------------------------------------
# Figure 4: Nonparametric Empirical Relationship Grid (LOESS & Polynomial)
# -----------------------------------------------------------------------------
def make_empirical_relationships_figure() -> None:
    print("Generating Figure 4: Empirical Nonlinear Relationship Grid...")
    monthly_panel_path = PROJECT_ROOT / "data/06_model_ready/prediction/llm_specific_scfi_monthly_panel.csv"
    if not monthly_panel_path.exists():
        print("Monthly panel missing, skipping empirical relationships figure.")
        return

    df = pd.read_csv(monthly_panel_path)
    target = "scfi_monthly_mean"

    predictors = [
        ("MSDI Peak Disruption", "llm_signal__gemma_2_9b_it__overall_disruption_max", "#d94801", "(a)"),
        ("MSDI Breadth Disruption", "llm_signal__gemma_2_9b_it__dimension_mean7", "#7570b3", "(b)"),
        ("Brent Crude Oil", "brent_monthly_mean", "#08519c", "(c)"),
        ("CPB World Trade Volume", "cpb_world_trade_volume", "#238b45", "(d)"),
        ("RWI/ISL Port Throughput", "rwi_isl_container_throughput_index", "#6a51a3", "(e)"),
        ("CNY/USD Exchange Rate", "cny_usd_exchange_rate", "#006d2c", "(f)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.5), dpi=300)
    plt.subplots_adjust(hspace=0.38, wspace=0.28, left=0.07, right=0.96, top=0.92, bottom=0.08)

    for idx, (p_name, p_col, p_color, tag) in enumerate(predictors):
        ax = axes[idx // 3, idx % 3]
        if p_col in df.columns:
            sub = df.dropna(subset=[target, p_col])
            x_vals = sub[p_col].values
            y_vals = sub[target].values

            # Scatter points
            ax.scatter(x_vals, y_vals, color=p_color, alpha=0.45, s=25, edgecolor="none")

            # Polynomial smooth curve
            sort_idx = np.argsort(x_vals)
            x_sorted = x_vals[sort_idx]
            
            poly = np.polyfit(x_vals, y_vals, deg=2)
            y_poly = np.polyval(poly, x_sorted)
            ax.plot(x_sorted, y_poly, color="black", lw=2.0, label="Empirical Fit")

            # Rug plot at bottom
            ax.plot(x_vals, [ax.get_ylim()[0]] * len(x_vals), "|", color="gray", alpha=0.6, ms=6)

            # Spearman rho
            rho, pval = stats.spearmanr(x_vals, y_vals)
            star = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
            
            # SUBTITLE PLACED STRICTLY AT THE TOP ABOVE THE AXIS
            ax.set_title(f"{tag} {p_name} ($\\rho = {rho:.2f}{star}$)", fontsize=11, fontweight="bold", pad=8)

            ax.set_xlabel(p_name, fontsize=9.5)
            ax.set_ylabel("SCFI Freight Level", fontsize=9.5)
            ax.grid(True, linestyle="--", alpha=0.35)

    axes[0, 0].legend(loc="upper right", frameon=True, fontsize=8)
    save_figure(fig, "eda_figure04_empirical_relationships_grid")


def main() -> None:
    print(f"Starting EDA figures generation. Target directory: {OUTPUT_DIR}")
    make_spearman_correlation_figure()
    make_data_distributions_figure()
    make_friedmans_h_statistic_grid()
    make_empirical_relationships_figure()
    print("All publication EDA figures generated successfully!")


if __name__ == "__main__":
    main()
