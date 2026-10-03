# Maritime Supply Chain Disruption Index (MSDI)

Reproducible pipeline for an LLM-derived, human-validated Maritime Supply Chain
Disruption Index (MSDI), used to forecast Shanghai Containerized Freight Index
(SCFI) movements and to run time-series causality, causal-ML, multiplicity, and
sensitivity analyses.

Monthly is the primary frequency; weekly is robustness. No result in this repository supports a structural causal claim.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Running

Read-only structural check:

```powershell
.venv\Scripts\python.exe scripts\99_run_full_pipeline.py smoke
```

Full sequential run, stages 00–24:

```powershell
.venv\Scripts\python.exe scripts\99_run_full_pipeline.py full `
  --annotations validation\full\completed\human_validation_annotator_1.csv validation\full\completed\human_validation_annotator_2.csv `
  --adjudicated validation\event_pairs\results\event_pair_adjudicated_ground_truth_300.csv
```

Human validation is complete (400 articles, 300 event pairs).
`--allow-pending-human-validation` exists only for development reruns and marks the run publication-ineligible.

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

Scripts `25`–`49` are supporting analyses (external controls, validation
sampling and agreement, EDA, PCA, event studies, PCMCI robustness, TOPSIS) and
run outside the sequential pipeline. Full listing in [TREE.md](TREE.md).

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

Raw-file locations and release lags: [configs/external_controls.yaml](configs/external_controls.yaml).

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

Written by the pipeline; not tracked in git.

| Location | Contents |
|---|---|
| `outputs/tables/` | 26 CSVs — `T01`–`T16` main, `S00`, `S00b`, `S01`–`S08` supplementary — plus `publication_v001_publication_tables.xlsx` |
| `outputs/figures/` | `publication_v001_figure01`–`figure19`, each as PDF and PNG |
| `outputs/figures/eda/` | `eda_figure01`–`eda_figure06`, PDF and PNG |
| `outputs/significance/` | Hypothesis registry (14,072 rows) with Holm and BH-FDR adjustments |
| `outputs/reports/` | `publication_v001_results_summary.md` and run manifests |
| `outputs/topsis/` | Entropy-weighted operational model ranking |

## License

Code is released under the MIT License ([LICENSE](LICENSE)).
