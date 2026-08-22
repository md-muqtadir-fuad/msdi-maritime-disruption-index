"""Create leakage-aware EDA for cleaned external controls and modeling panels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Times", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROLS = ROOT / "data/02_interim/controls/external_controls_clean_long.csv"
DEFAULT_WEEKLY = ROOT / "data/06_model_ready/prediction/msdi_scfi_weekly_panel.csv"
DEFAULT_MONTHLY = ROOT / "data/06_model_ready/causality/msdi_scfi_monthly_panel.csv"
DEFAULT_OUTPUT = ROOT / "outputs/external_controls_eda"

CORE_COLUMNS = (
    "scfi",
    "scfi_log",
    "scfi_return",
    "peak_score_media_corrected",
    "breadth_score_media_corrected",
    "non_freight_peak_score_media_corrected",
    "brent_price",
    "article_count",
    "source_count",
    "scfi_monthly_mean",
    "scfi_monthly_mean_log",
    "scfi_monthly_mean_log_change_1m",
    "peak_score_media_corrected_mean",
    "breadth_score_media_corrected_mean",
    "non_freight_peak_score_media_corrected_mean",
    "brent_monthly_mean",
    "brent_monthly_mean_log_change_1m",
    "article_count_sum",
    "source_count_mean",
)

PRIMARY_CONTROL_COLUMNS = (
    "cpb_world_trade_volume",
    "rwi_isl_container_throughput_index",
    "wto_china_merchandise_exports",
    "cny_usd_exchange_rate",
    "unctad_china_core_lane_lsbci",
    "gscpi",
    "imf_portwatch_global_container_calls",
    "gpr_index",
    "tpu_index",
    "uct_index",
    "opu_index",
    "infectious_disease_emv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA for external controls, MSDI, and SCFI.")
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--weekly-panel", type=Path, default=DEFAULT_WEEKLY)
    parser.add_argument("--monthly-panel", type=Path, default=DEFAULT_MONTHLY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frequency", choices=("weekly", "monthly", "both"), default="both")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Restrict matrices to core outcome/MSDI/Brent/coverage/control variables.",
    )
    parser.add_argument("--version", default="external_controls_eda_v002")
    return parser.parse_args()


def require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path.resolve()}")
    return path


def attach_controls(panel: pd.DataFrame, date_column: str, controls: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy().sort_values(date_column)
    for variable, group in controls.groupby("variable", sort=True):
        source = (
            group[["availability_date", "value"]]
            .sort_values("availability_date")
            .drop_duplicates("availability_date", keep="last")
        )
        attached = pd.merge_asof(
            result[[date_column]],
            source,
            left_on=date_column,
            right_on="availability_date",
            direction="backward",
        )
        result[variable] = attached["value"].to_numpy()
        result[f"{variable}_source_date"] = attached["availability_date"].to_numpy()
    return result


def numeric_columns(frame: pd.DataFrame, core_only: bool) -> list[str]:
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    if core_only:
        allowed = set(CORE_COLUMNS) | set(PRIMARY_CONTROL_COLUMNS)
        return [col for col in numeric if col in allowed]
    # For full analysis, include core, controls, and unlagged transformations
    return [
        col
        for col in numeric
        if not col.endswith(("_age_days", "_source_date", "_lag_12w", "_lag_8w", "_lag_6m"))
    ]


def run_frequency(
    name: str,
    path: Path,
    date_column: str,
    controls: pd.DataFrame,
    output: Path,
    core_only: bool,
) -> dict:
    panel = pd.read_csv(require(path, f"{name} panel"), low_memory=False)
    panel[date_column] = pd.to_datetime(panel[date_column], errors="coerce")
    panel = panel.dropna(subset=[date_column]).sort_values(date_column)
    merged = attach_controls(panel, date_column, controls)
    columns = numeric_columns(merged, core_only)
    summary = merged[columns].describe().T.rename_axis("variable").reset_index()
    summary["missing_count"] = merged[columns].isna().sum().to_numpy()
    summary["missing_share"] = summary["missing_count"] / len(merged)
    missing = pd.DataFrame(
        {
            "variable": columns,
            "missing_count": merged[columns].isna().sum().values,
            "missing_share": merged[columns].isna().mean().values,
        }
    )
    pearson = merged[columns].corr("pearson")
    spearman = merged[columns].corr("spearman")

    prefix = output / name
    merged.to_csv(prefix.with_name(f"{name}_external_controls_panel.csv"), index=False)
    summary.to_csv(prefix.with_name(f"{name}_summary.csv"), index=False)
    missing.to_csv(prefix.with_name(f"{name}_missingness.csv"), index=False)
    pearson.to_csv(prefix.with_name(f"{name}_pearson_correlation_matrix.csv"))
    spearman.to_csv(prefix.with_name(f"{name}_spearman_correlation_matrix.csv"))

    # Plot top primary columns for clean heatmap
    plot_cols = [c for c in [*CORE_COLUMNS, *PRIMARY_CONTROL_COLUMNS] if c in columns][:25]
    if len(plot_cols) >= 2:
        p_sub = merged[plot_cols].corr("pearson")
        s_sub = merged[plot_cols].corr("spearman")
        fig, axes = plt.subplots(
            1, 2, figsize=(max(10, len(plot_cols) * 0.45), max(5, len(plot_cols) * 0.4))
        )
        for axis, matrix, label in zip(
            axes, (p_sub, s_sub), ("Pearson correlation", "Spearman correlation")
        ):
            image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
            axis.set_title(label, fontsize=10)
            axis.set_xticks(range(len(plot_cols)))
            axis.set_xticklabels(plot_cols, rotation=70, ha="right", fontsize=7)
            axis.set_yticks(range(len(plot_cols)))
            axis.set_yticklabels(plot_cols, fontsize=7)
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out_png = prefix.with_name(f"{name}_correlation_matrices.png")
        out_pdf = prefix.with_name(f"{name}_correlation_matrices.pdf")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {
        "rows": len(merged),
        "columns": columns,
        "date_start": str(merged[date_column].min().date()),
        "date_end": str(merged[date_column].max().date()),
    }


def main() -> None:
    args = parse_args()
    controls = pd.read_csv(require(args.controls, "Cleaned controls"), low_memory=False)
    required = {"variable", "availability_date", "value"}
    if not required.issubset(controls.columns):
        raise ValueError(f"Controls missing required columns: {required - set(controls.columns)}")
    controls["availability_date"] = pd.to_datetime(controls["availability_date"], errors="coerce")
    controls = controls.dropna(subset=["availability_date", "value"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    if args.frequency in {"weekly", "both"}:
        reports["weekly"] = run_frequency(
            "weekly",
            args.weekly_panel,
            "observation_date",
            controls,
            args.output_dir,
            args.core_only,
        )
    if args.frequency in {"monthly", "both"}:
        reports["monthly"] = run_frequency(
            "monthly",
            args.monthly_panel,
            "month_ending",
            controls,
            args.output_dir,
            args.core_only,
        )
    (args.output_dir / f"{args.version}_manifest.json").write_text(
        json.dumps(
            {
                "version": args.version,
                "reports": reports,
                "warning": "EDA correlations are descriptive; no causal interpretation.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"EDA complete: {args.output_dir}")


if __name__ == "__main__":
    main()
