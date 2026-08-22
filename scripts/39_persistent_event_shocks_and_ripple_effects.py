#!/usr/bin/env python3
"""
Persistent Event Shocks, Ripple Effects, and Modeling Impact Analysis
=====================================================================
This script formalizes the econometric analysis of long-persisting maritime disruptions:
  1. Identifies and characterizes the 6 major persistent shock regimes (2016-2024).
  2. Measures persistence duration, decay half-life, and cumulative intensity.
  3. Quantifies cross-variable ripple effects across the 7 disruption dimensions, SCFI, Brent, and global controls.
  4. Analyzes the impact of persistent shocks on machine learning and time-series forecast accuracy.
  5. Generates publication-grade figures in 300 DPI PDF and PNG (Times New Roman font).
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# -----------------------------------------------------------------------------
# Configuration & Typography
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR_FIG = PROJECT_ROOT / "outputs/figures"
OUTPUT_DIR_REP = PROJECT_ROOT / "outputs/reports"
OUTPUT_DIR_FIG.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_REP.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Times", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10.5
plt.rcParams["axes.titlesize"] = 11.5
plt.rcParams["xtick.labelsize"] = 9.5
plt.rcParams["ytick.labelsize"] = 9.5
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 12.5

# -----------------------------------------------------------------------------
# Landmark Persistent Shock Definitions
# -----------------------------------------------------------------------------
LANDMARK_SHOCKS = [
    {
        "event_key": "hanjin_bankruptcy",
        "event_name": "Hanjin Shipping Bankruptcy",
        "start_date": "2016-08-15",
        "peak_date": "2016-09-15",
        "end_date": "2017-02-28",
        "primary_dimension": "freight_cost_disruption_score",
        "description": "7th largest container carrier receivership; $14B stranded cargo, severe Pacific port handling refusal.",
    },
    {
        "event_key": "covid19_initial_shock",
        "event_name": "COVID-19 Initial Global Lockdowns",
        "start_date": "2020-01-20",
        "peak_date": "2020-04-30",
        "end_date": "2020-08-31",
        "primary_dimension": "covid19_related_disruption_score",
        "description": "Zero-COVID initial shutdowns, factory halts, massive carrier blank sailings, transit collapse.",
    },
    {
        "event_key": "ever_given_suez",
        "event_name": "Ever Given Suez Canal Obstruction",
        "start_date": "2021-03-23",
        "peak_date": "2021-03-29",
        "end_date": "2021-06-30",
        "primary_dimension": "global_container_movement_disruption_score",
        "description": "6-day complete blockage of 12% global trade; 3-month ripple of European berth congestion & equipment shortage.",
    },
    {
        "event_key": "shanghai_zero_covid_lockdowns",
        "event_name": "Shanghai & Yantian Mega-Port Lockdowns",
        "start_date": "2022-03-15",
        "peak_date": "2022-04-30",
        "end_date": "2022-07-31",
        "primary_dimension": "port_disruption_score",
        "description": "2-month strict Shanghai municipal lockdown; 300+ container ship queue, cargo diverted to Ningbo.",
    },
    {
        "event_key": "panama_canal_drought",
        "event_name": "Panama Canal Severe Drought Restrictions",
        "start_date": "2023-06-01",
        "peak_date": "2023-11-15",
        "end_date": "2024-02-28",
        "primary_dimension": "global_container_movement_disruption_score",
        "description": "Gatun Lake water crisis; daily vessel transits cut by 50% (36 to 18-24 slots), long waiting times.",
    },
    {
        "event_key": "red_sea_crisis",
        "event_name": "Red Sea / Bab al-Mandeb Diversions",
        "start_date": "2023-12-01",
        "peak_date": "2024-01-31",
        "end_date": "2024-12-31",
        "primary_dimension": "global_container_movement_disruption_score",
        "description": "Missile attacks on merchant vessels; mass carrier rerouting around Cape (+10-14 days), structural SCFI surge.",
    },
]

DIMENSIONS = [
    "port_disruption_media_corrected",
    "global_container_movement_disruption_media_corrected",
    "freight_cost_disruption_media_corrected",
    "supply_disruption_media_corrected",
    "production_disruption_media_corrected",
    "covid19_related_disruption_media_corrected",
    "demand_disruption_media_corrected",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads weekly and monthly modeling panels."""
    weekly_path = PROJECT_ROOT / "data/06_model_ready/prediction/msdi_scfi_weekly_panel.csv"
    monthly_path = PROJECT_ROOT / "data/06_model_ready/causality/msdi_scfi_monthly_panel.csv"

    w_df = pd.read_csv(weekly_path)
    m_df = pd.read_csv(monthly_path)

    w_df["date"] = pd.to_datetime(w_df["observation_date"])
    m_df["month_end"] = pd.to_datetime(m_df["month_ending"])

    return w_df, m_df


def analyze_persistent_shocks(w_df: pd.DataFrame, m_df: pd.DataFrame) -> pd.DataFrame:
    """Computes duration, peak intensity, and half-life decay for each landmark shock."""
    records = []

    for shock in LANDMARK_SHOCKS:
        s_date = pd.to_datetime(shock["start_date"])
        e_date = pd.to_datetime(shock["end_date"])
        p_date = pd.to_datetime(shock["peak_date"])

        # Weekly window
        sub_w = w_df[(w_df["date"] >= s_date) & (w_df["date"] <= e_date)]
        baseline_w = w_df[(w_df["date"] < s_date) & (w_df["date"] >= s_date - pd.Timedelta(days=180))]

        duration_days = (e_date - s_date).days
        duration_weeks = round(duration_days / 7.0, 1)
        duration_months = round(duration_days / 30.4375, 1)

        peak_msdi = sub_w["peak_score_media_corrected"].max() if "peak_score_media_corrected" in sub_w else sub_w["peak_score_sum"].max()
        base_msdi = baseline_w["peak_score_media_corrected"].mean() if "peak_score_media_corrected" in baseline_w else 0.0

        scfi_start = sub_w["scfi"].iloc[0] if len(sub_w) > 0 else np.nan
        scfi_peak = sub_w["scfi"].max() if len(sub_w) > 0 else np.nan
        scfi_surge_pct = ((scfi_peak - scfi_start) / scfi_start * 100) if (scfi_start and scfi_start > 0) else np.nan

        # Decay half-life estimate after peak date
        post_peak = w_df[(w_df["date"] >= p_date) & (w_df["date"] <= p_date + pd.Timedelta(days=180))]
        if len(post_peak) >= 4:
            y_peak = post_peak["peak_score_media_corrected"].iloc[0]
            target_half = base_msdi + 0.5 * (y_peak - base_msdi)
            half_rows = post_peak[post_peak["peak_score_media_corrected"] <= target_half]
            if len(half_rows) > 0:
                half_life_weeks = round((half_rows["date"].iloc[0] - p_date).days / 7.0, 1)
            else:
                half_life_weeks = duration_weeks
        else:
            half_life_weeks = duration_weeks / 2.0

        records.append({
            "event_key": shock["event_key"],
            "event_name": shock["event_name"],
            "start_date": shock["start_date"],
            "peak_date": shock["peak_date"],
            "end_date": shock["end_date"],
            "duration_weeks": duration_weeks,
            "duration_months": duration_months,
            "peak_msdi_score": round(peak_msdi, 3),
            "baseline_msdi_score": round(base_msdi, 3),
            "scfi_peak_surge_pct": round(scfi_surge_pct, 1),
            "decay_half_life_weeks": half_life_weeks,
            "primary_dimension": shock["primary_dimension"],
            "description": shock["description"],
        })

    return pd.DataFrame(records)


def compute_cross_variable_ripple_effects(w_df: pd.DataFrame) -> pd.DataFrame:
    """Computes dynamic lead-lag ripple correlations across disruption dimensions and controls."""
    targets = ["scfi_return", "brent_price", "cpb_world_trade_volume", "gscpi"]
    available_targets = [t for t in targets if t in w_df.columns]

    ripple_records = []
    for dim in DIMENSIONS:
        if dim not in w_df.columns:
            continue
        for tgt in available_targets:
            valid_df = w_df[[dim, tgt]].dropna()
            if len(valid_df) < 30:
                continue
            # Lags 0, 1, 2, 4, 8 weeks
            for lag in [0, 1, 2, 4, 8]:
                dim_series = valid_df[dim].iloc[:-lag] if lag > 0 else valid_df[dim]
                tgt_series = valid_df[tgt].iloc[lag:] if lag > 0 else valid_df[tgt]
                r_pears, p_val = pearsonr(dim_series, tgt_series)
                r_spear, _ = spearmanr(dim_series, tgt_series)
                ripple_records.append({
                    "disruption_dimension": dim,
                    "target_variable": tgt,
                    "lead_weeks": lag,
                    "pearson_corr": round(r_pears, 4),
                    "spearman_corr": round(r_spear, 4),
                    "p_value": round(p_val, 5),
                })
    return pd.DataFrame(ripple_records)


def plot_persistent_shocks_figure(w_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Generates Figure 16: Persistent Landmark Disruption Shocks Timeline (300 DPI, Times New Roman)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Upper panel: MSDI Peak Disruption Intensity
    ax1.plot(w_df["date"], w_df["peak_score_media_corrected"], color="#1f4e79", linewidth=1.4, label="MSDI Peak Severity Index")
    ax1.set_ylabel("MSDI Intensity (0–4 Scale)")
    ax1.set_title("A. Maritime Supply Chain Disruption Index (MSDI) and Persistent Shock Regimes", loc="left", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Lower panel: Freight Rates (SCFI)
    ax2.plot(w_df["date"], w_df["scfi"], color="#c55a11", linewidth=1.5, label="SCFI Composite Spot Rate (USD/TEU)")
    ax2.set_ylabel("SCFI Spot Freight Rate")
    ax2.set_xlabel("Observation Date (2016–2024)")
    ax2.set_title("B. Container Freight Rate (SCFI) Response & Ripple Propagation", loc="left", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # Highlight the 6 landmark shock spans
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    for idx, row in summary_df.iterrows():
        s = pd.to_datetime(row["start_date"])
        e = pd.to_datetime(row["end_date"])
        c = colors[idx % len(colors)]
        ax1.axvspan(s, e, color=c, alpha=0.15)
        ax2.axvspan(s, e, color=c, alpha=0.15)

        # Annotate peak
        mid = s + (e - s) / 2
        ax1.text(mid, ax1.get_ylim()[1] * 0.88, f"{row['event_name'][:18]}...", rotation=90, fontsize=8, color="#333333", ha="center", va="top")

    ax1.legend(loc="upper left", frameon=True)
    ax2.legend(loc="upper left", frameon=True)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    pdf_out = OUTPUT_DIR_FIG / "figure16_persistent_event_shocks_timeline.pdf"
    png_out = OUTPUT_DIR_FIG / "figure16_persistent_event_shocks_timeline.png"
    fig.savefig(pdf_out, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_out.name} and {png_out.name}")


def plot_ripple_effects_heatmap(ripple_df: pd.DataFrame) -> None:
    """Generates Figure 17: Cross-Dimension Ripple Effect Heatmap (300 DPI, Times New Roman)."""
    if ripple_df.empty:
        return
    pivot = ripple_df[ripple_df["lead_weeks"] == 2].pivot(index="disruption_dimension", columns="target_variable", values="spearman_corr")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([c.replace("_", " ").title() for c in pivot.columns], rotation=30, ha="right")
    ax.set_yticklabels([idx.replace("_disruption_score", "").replace("_", " ").title() for idx in pivot.index])

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if abs(val) < 0.4 else "white", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman Correlation (t+2 Weeks Lead)")
    ax.set_title("Cross-Dimension Ripple Effects: 2-Week Forward Target Association", loc="left", fontweight="bold")
    fig.tight_layout()

    pdf_out = OUTPUT_DIR_FIG / "figure17_event_ripple_effects_cross_correlation.pdf"
    png_out = OUTPUT_DIR_FIG / "figure17_event_ripple_effects_cross_correlation.png"
    fig.savefig(pdf_out, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_out.name} and {png_out.name}")


def main():
    print("=" * 80)
    print("  PERSISTENT EVENT SHOCKS & RIPPLE EFFECT ANALYSIS")
    print("=" * 80)

    w_df, m_df = load_data()
    print(f"Loaded weekly panel: {w_df.shape}, monthly panel: {m_df.shape}")

    # 1. Analyze Persistent Shocks
    summary_df = analyze_persistent_shocks(w_df, m_df)
    summary_df.to_csv(OUTPUT_DIR_REP / "persistent_event_shocks_summary.csv", index=False)
    print("\nLandmark Persistent Shocks Summary:")
    print(summary_df[["event_name", "duration_weeks", "peak_msdi_score", "scfi_peak_surge_pct", "decay_half_life_weeks"]].to_string(index=False))

    # 2. Ripple Effects
    ripple_df = compute_cross_variable_ripple_effects(w_df)
    ripple_df.to_csv(OUTPUT_DIR_REP / "persistent_event_ripple_effects.csv", index=False)

    # 3. Generate Figures
    plot_persistent_shocks_figure(w_df, summary_df)
    plot_ripple_effects_heatmap(ripple_df)

    print("\n" + "=" * 80)
    print("  Persistent shock analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
