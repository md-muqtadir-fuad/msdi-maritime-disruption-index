# SC-Disrupt

Reproducible research pipeline for building a human-validated, LLM-derived
maritime supply-chain disruption index; forecasting SCFI; and running dynamic
time-series, causal-ML, significance, ablation, sensitivity, and diagnostic
analyses.

## Environment

All project commands must use the repository virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-forecasting.txt
```

The forecasting requirements add PyTorch and Prophet. The Gaussian GLM,
Random Forest, and preprocessing components use scikit-learn from the base
environment.

## Pipeline entry point

Run the read-only pipeline check first. It compiles every numbered script,
checks dependencies/configurations/the stage DAG, and inventories artifacts;
it does not fit models or write partial smoke outputs.

```powershell
.venv\Scripts\python.exe scripts\99_run_full_pipeline.py smoke
```

The full run executes stages `00`–`24` sequentially and enables complete
nested, horizon-purged tuning for the weekly-primary/monthly-robustness
forecasting stage:

```powershell
.venv\Scripts\python.exe scripts\99_run_full_pipeline.py full --allow-pending-human-validation
```

For a publication-eligible run, omit the pending-validation flag and supply
completed annotation files with `--annotations`. External LLM score ZIPs must
already be present under `data/03_processed/model_scores/scores_zips`.

Stage 18 evaluates a penalized Gaussian identity-link GLM, Random Forest,
XGBoost, LSTM, and Prophet at both frequencies. Weekly evidence is primary;
monthly evidence is robustness. Fixed parameters live in
`configs/nonlinear_forecasting.yaml`; `99` uses the predefined full tuning grid
and resumable per-cell checkpoints.

The weekly and monthly LLM-comparison stages remain Ridge-only by design so the
algorithm is held constant while comparing labels. The extended forecasting
stage then compares the five end-to-end algorithm pipelines. Granger, PCMCI,
local projections, and observational causal-ML outputs do not by themselves
support structural causal claims.

See [the full runner guide](docs/reproducibility/full_pipeline_runner.md) and
[the forecasting method](docs/methodology/nonlinear_forecasting.md).

## Testing

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

## Repository notes

- Scripts use strict sequential names from `00_...` through `24_...`; `99` is
  the orchestrator.
- Large or sensitive data under `data/` are intentionally not tracked.
- Method/configuration files live under `configs/` and `docs/`.
- Generated audits, forecasts, inference tables, figures, and run manifests
  live under `outputs/`.
- Historical manifest identifiers such as `stage07d` remain unchanged for
  backward-compatible provenance even though the executable is now Stage 18.
