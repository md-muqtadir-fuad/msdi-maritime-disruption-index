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

## External control data (planned extension)

Keep downloaded source files in the folders below, preserving the original
download alongside its retrieval date and source metadata. These controls are
not automatically used until the panel-building stage is extended; when added,
use only publication-available lagged values and retain the current
Brent-only run as the baseline.

| Source | Role in this study | Raw-data location |
|---|---|---|
| [IMF PortWatch daily ports data](https://portwatch.imf.org/datasets/daily-ports-data/explore) | **2019+ robustness-only** port-activity / physical-shipping evidence; never a primary 2016–2026 control. | `data/01_raw/controls/portwatch/` |
| [CPB World Trade Monitor](https://www.cpb.nl/en/world-trade-monitor/cpb-wereldhandelsmonitor-may-2026) | Global import-demand control. | `data/01_raw/controls/macro/` |
| [UNCTAD Liner Shipping Bilateral Connectivity Index](https://unctadstat.unctad.org/datacentre/dataviewer/US.LSBCI) | Shipping-network supply and route-connectivity control. | `data/01_raw/controls/connectivity/` |
| [New York Fed GSCPI](https://www.newyorkfed.org/research/policy/gscpi#/interactive) | External validity benchmark and optional lagged forecasting sensitivity feature; not a primary causal control because it includes transport-cost indicators. | `data/01_raw/controls/macro/` |

Do not use a contemporaneous external series when forecasting or causal
analysis. Record each release date and apply an availability lag before merging
it into the weekly or monthly panel.

The current UNCTAD control is the unweighted mean of pre-specified China-origin
core lanes (United States, Germany, Netherlands, United Kingdom, Japan,
Republic of Korea, and Singapore), not an all-pairs mean. Correlation and
lagged-association outputs report HAC/FDR inference; ordinary IID correlation
p-values are descriptive only.

After placing the four downloads in those folders, run the optional control
extension in order:

```powershell
.venv\Scripts\python.exe scripts\25_clean_external_controls.py
.venv\Scripts\python.exe scripts\26_external_controls_eda.py
.venv\Scripts\python.exe scripts\27_external_controls_statistical_analysis.py
```

Stage 25 leaves raw files unchanged and writes the standardized, availability-
dated long dataset to `data/02_interim/controls/`. Stage 26 produces complete
numeric correlation matrices, missingness summaries, and EDA figures. Stage 27
exports FDR-adjusted Pearson/Spearman tests, lagged one-to-one SCFI
associations, and VIF multicollinearity diagnostics. These optional scripts do
not alter the existing `00`â€“`24` publication pipeline until the controls are
explicitly incorporated into Stage 14 and the downstream models are rerun.

## Temporary pseudo annotations

For pipeline plumbing only, create explicitly synthetic annotation files and
their inter-annotator-agreement outputs with:

```powershell
.venv\Scripts\python.exe scripts\28_create_pseudo_annotations.py
.venv\Scripts\python.exe scripts\29_inter_annotator_agreement.py --annotations data\04_validation\pseudo_test_only\annotator_1_pseudo.csv data\04_validation\pseudo_test_only\annotator_2_pseudo.csv --allow-pseudo-annotations
```

Pseudo labels are stored in `data/04_validation/pseudo_test_only/` and are
blocked by Stage 12 unless `--allow-pseudo-annotations` is explicit. They are
never human validation and never publication eligible.

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
