# Exploratory data analysis

This directory contains reproducible, read-only exploratory analysis. Raw files
under `data/01_raw/` are never edited by EDA code.

The controlling questions, decision rules, and completion status are defined in
[`EDA_ANALYSIS_PROTOCOL.md`](EDA_ANALYSIS_PROTOCOL.md). All EDA scripts and
interpretations must follow that protocol.

## Order of work

`00_raw_controls_eda.py` is the original prototype. The reproducible protocol
is implemented by `01_raw_data_integrity.py` through
`09_causal_readiness.py`, in numerical order. `99_run_eda.py` is the only
start-to-finish entry point.

Correlations produced here are descriptive diagnostics. They must not be used
to choose variables, transformations, lags, or samples because they maximize
statistical significance or forecasting performance.

## Run

```powershell
.\.venv\Scripts\python.exe eda\00_raw_controls_eda.py
```

Low-cost validation mode (still audits every candidate, but performs no model
fitting or tuning):

```powershell
.\.venv\Scripts\python.exe eda\00_raw_controls_eda.py --test-run --output-dir eda\outputs\test_run
```

Generated CSV, Markdown, JSON, and PNG files are written to `eda/outputs/`.

## Complete EDA runner

List the Point 1-9 stages:

```powershell
.\.venv\Scripts\python.exe eda\99_run_eda.py --list-stages
```

Run the lightweight test workflow:

```powershell
.\.venv\Scripts\python.exe eda\99_run_eda.py --mode test
```

Run the full EDA workflow:

```powershell
.\.venv\Scripts\python.exe eda\99_run_eda.py --mode full
```

All nine scripts now exist. A normal run fails if any protocol stage is missing
so an incomplete EDA cannot be mistaken for a complete one.
