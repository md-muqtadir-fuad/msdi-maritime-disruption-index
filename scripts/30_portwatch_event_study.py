"""Describe 2019+ PortWatch/SCFI behavior around pre-specified disruption dates.

This is a robustness/event-study descriptive module. It neither enters primary
forecast models nor identifies a causal effect of news or PortWatch activity.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
def main() -> None:
    p = argparse.ArgumentParser(description="Run a descriptive PortWatch event study.")
    p.add_argument("--panel", type=Path, default=ROOT / "data/06_model_ready/prediction/msdi_scfi_weekly_panel.csv")
    p.add_argument("--config", type=Path, default=ROOT / "configs/portwatch_event_study.yaml")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/portwatch_event_study")
    args = p.parse_args(); config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    panel = pd.read_csv(args.panel, low_memory=False); panel["observation_date"] = pd.to_datetime(panel["observation_date"], errors="coerce")
    variable = config["portwatch_column"]
    required = {"observation_date", "scfi", variable}
    if missing := required - set(panel.columns): raise ValueError(f"Panel missing {sorted(missing)}")
    panel = panel.loc[panel["observation_date"].ge("2019-01-01")].sort_values("observation_date").reset_index(drop=True)
    rows = []
    for name, event_date in config["events"].items():
        event = pd.Timestamp(event_date); distance = ((panel["observation_date"] - event).dt.days / 7).round().astype("Int64")
        window = panel.loc[distance.between(-int(config["pre_weeks"]), int(config["post_weeks"]))].copy(); window["event_time_weeks"] = distance.loc[window.index]
        baseline = window.loc[window.event_time_weeks.lt(0), ["scfi", variable]].mean()
        for _, row in window.iterrows(): rows.append({"event": name, "event_date": event.date().isoformat(), "observation_date": row.observation_date.date().isoformat(), "event_time_weeks": int(row.event_time_weeks), "scfi": row.scfi, variable: row[variable], "scfi_minus_pre_event_mean": row.scfi - baseline.scfi, "portwatch_minus_pre_event_mean": row[variable] - baseline[variable]})
    args.output_dir.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(args.output_dir / "portwatch_event_windows.csv", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps({"role": config["role"], "events": config["events"], "causal_claim_allowed": False}, indent=2), encoding="utf-8")
    print(f"PortWatch event study written: {args.output_dir}")
if __name__ == "__main__": main()
