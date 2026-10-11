# Maritime Supply Chain Disruption Index (MSDI)

[![DOI](https://zenodo.org/badge/1284364848.svg)](https://doi.org/10.5281/zenodo.23300977)

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

`requirements.txt` lists direct dependencies. `requirements-lock.txt` pins the
exact versions (Python 3.11.9) that produced the archived results.

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

`data/` and most of `outputs/` are not tracked in git. The pipeline writes
them, and several inputs (the news corpus, SCFI, Brent) cannot be
redistributed. A subset of `outputs/` is tracked as reference results:
forecast benchmarks, LLM-vs-SCFI comparisons, causality tests, panel audits,
and the pre-LLM corpus profile. Their run manifests record absolute paths
from the original machine. They are kept byte-identical because downstream
stages verify them by SHA-256.

The full pipeline writes:

| Location | Contents |
|---|---|
| `outputs/tables/` | 26 CSVs — `T01`–`T16` main, `S00`, `S00b`, `S01`–`S08` supplementary — plus `publication_v001_publication_tables.xlsx` |
| `outputs/figures/` | `publication_v001_figure01`–`figure19`, each as PDF and PNG |
| `outputs/figures/eda/` | `eda_figure01`–`eda_figure06`, PDF and PNG |
| `outputs/significance/` | Hypothesis registry (14,072 rows) with Holm and BH-FDR adjustments |
| `outputs/reports/` | `publication_v001_results_summary.md` and run manifests |
| `outputs/topsis/` | Entropy-weighted operational model ranking |

## Dataset package

The derived data are archived as a second zip in the same Zenodo record:
article metadata and URLs, LLM scores, MSDI series, and human validation. It excludes article
text and third-party price series. To build `release/msdi_dataset_v1.0.0/`
and its zip from saved outputs:

```powershell
.venv\Scripts\python.exe misc\zenodo_dataset\build_dataset_package.py
```

## Citation

The Zenodo record holds the code and the dataset package.

- All versions (resolves to the latest): [10.5281/zenodo.23300977](https://doi.org/10.5281/zenodo.23300977)
- Version 1.0.0: [10.5281/zenodo.23301240](https://doi.org/10.5281/zenodo.23301240)

Metadata is in [CITATION.cff](CITATION.cff).

## License

Code is released under the MIT License ([LICENSE](LICENSE)).
