# LLM-derived Maritime Supply Chain Disruption Index (MSDI) from news text, with human validation and SCFI freight-rate forecasting: code and data.

Reproducible pipeline for an LLM-derived, human-validated Maritime Supply Chain
Disruption Index (MSDI), used to forecast Shanghai Containerized Freight Index
(SCFI) movements and to run time-series causality, causal-ML, multiplicity, and
sensitivity analyses.

Monthly is the primary frequency; weekly is robustness. No result in this
repository supports a structural causal claim.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` covers everything, including `torch`, `prophet` and
`xgboost`. There is no separate forecasting requirements file.

## Running

Read-only structural check (compiles every script, validates the stage registry,
configs, and artifact contracts; runs no fits and writes nothing):

```powershell
.venv\Scripts\python.exe scripts\99_run_full_pipeline.py smoke
```

Expected: `111/111 structural checks passed`.

Full sequential run, stages 00–24:

```powershell
.venv\Scripts\python.exe scripts\99_run_full_pipeline.py full `
  --annotations validation\full\completed\human_validation_annotator_1.csv validation\full\completed\human_validation_annotator_2.csv `
  --adjudicated validation\event_pairs\results\event_pair_adjudicated_ground_truth_300.csv
```

Human validation is complete (400 articles, 300 event pairs), so
`--allow-pending-human-validation` is not the current research state; it exists
only for development reruns and marks the run publication-ineligible.

Stage 10 (LLM scoring) is an external manual gate: score archives must already
exist under `data/03_processed/model_scores/scores_zips`.

Partial reruns use `--run-id`, `--resume-pipeline`, `--from-stage`, `--to-stage`.

## Pipeline

```text
news ingestion, full-text retrieval, dedup, relevance gate     scripts 00-09
    └── four-model LLM scoring and merge                              10-11
        └── human validation (400 articles, 300 event pairs)             12
            └── MSDI index: monthly primary, weekly robustness           13
                ├── leakage-safe panels and forecasting               14-18
                ├── VAR/VARX, Granger, PCMCI, local projections          19
                ├── double/debiased machine learning                     20
                └── ablation, diagnostics, significance, outputs      21-24
```

Scripts `25`–`45` are supporting analyses (external controls, validation
sampling, EDA, PCA, event studies, TOPSIS). They are compiled by `smoke` but sit
outside the sequential run. Full listing in [TREE.md](TREE.md).

## Data

Outcome: SCFI. Energy control: Brent crude. Five external controls, merged
strictly backward by release date:

| Column | Series |
|---|---|
| `cpb_world_trade_volume` | CPB World Trade Monitor |
| `rwi_isl_container_throughput_index` | RWI/ISL Container Throughput Index |
| `wto_china_merchandise_exports` | WTO China merchandise exports |
| `cny_usd_exchange_rate` | CNY/USD exchange rate |
| `unctad_china_core_lane_lsbci` | UNCTAD LSBCI, China-origin core lanes |

Sources, portal URLs and availability-lag rules: `.manu/DATA_SOURCES.md`.

```powershell
.venv\Scripts\python.exe scripts\25_clean_external_controls.py
.venv\Scripts\python.exe scripts\26_external_controls_eda.py
.venv\Scripts\python.exe scripts\27_external_controls_statistical_analysis.py
```

## Tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

150 tests.

## Outputs

| Location | Contents |
|---|---|
| `outputs/tables/` | 26 CSVs — `T01`–`T16` main, `S00`, `S00b`, `S01`–`S08` supplementary — plus `publication_v001_publication_tables.xlsx` |
| `outputs/figures/` | `figure01`–`figure19`, each as PDF and PNG |
| `outputs/figures/eda/` | `eda_figure01`–`eda_figure06`, PDF and PNG |
| `outputs/significance/` | Hypothesis registry (14,072 rows) with Holm and BH-FDR adjustments |
| `outputs/reports/` | `publication_v001_results_summary.md` and run manifests |
| `outputs/topsis/` | Entropy-weighted operational model ranking |
