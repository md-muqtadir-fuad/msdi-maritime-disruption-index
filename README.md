# SC-Disrupt

Reproducible pipelines for building an LLM-derived maritime supply-chain
disruption index, validating it against human judgments, forecasting SCFI, and
running time-series and causal-ML analyses.

## Environment

All project commands must use the repository virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-forecasting.txt
```

The second requirements file adds PyTorch and Prophet for the nonlinear
forecasting stage. The Make equivalents are `make setup` and
`make setup-forecasting`; the Makefile also resolves Python through `.venv`.

## Main pipeline

Populate `data/` and then run the required stages in order. The common-cohort
forecast comparison is weekly first, monthly second, followed by the nonlinear
extension.

```powershell
.venv\Scripts\python.exe scripts/01_ingest_sources.py
.venv\Scripts\python.exe scripts/02_enrich_full_text.py
.venv\Scripts\python.exe scripts/03_run_llm_scoring.py
.venv\Scripts\python.exe scripts/07b_llm_scfi_comparison.py
.venv\Scripts\python.exe scripts/07c_monthly_llm_scfi_comparison.py
.venv\Scripts\python.exe scripts/07d_nonlinear_forecasts.py --frequency both
```

`07d` runs leakage-safe Random Forest, XGBoost, LSTM, and Prophet forecasts at
both frequencies. Weekly is primary; monthly is robustness. Fixed parameters
from `configs/nonlinear_forecasting.yaml` are used by default. The expensive
predefined nested search is explicitly enabled for the final run:

```powershell
.venv\Scripts\python.exe scripts/07d_nonlinear_forecasts.py --frequency both --tune
```

Long runs save atomic per-cell checkpoints. Re-run the identical command with
`--resume` after an interruption. The nonlinear stage is intentionally not part
of the ordinary `make analysis` target because the complete four-pipeline run
is computationally expensive.

See [the nonlinear forecasting method](docs/methodology/nonlinear_forecasting.md)
for targets, purge rules, score paths, outputs, and interpretation limits.

## Testing

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

## Repository notes

- Large or sensitive data under `data/` are intentionally not tracked.
- Configuration files live under `configs/`.
- Generated audits, forecasts, tests, rankings, figures, and manifests live
  under `outputs/`.
- Predictive precedence, Granger results, and causal-ML estimates have distinct
  interpretations; forecasting results alone do not support causal claims.
