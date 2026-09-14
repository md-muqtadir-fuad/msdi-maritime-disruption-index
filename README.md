# Maritime Supply Chain Disruption Code and Dataset

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

## Stages

| # | Script | Purpose |
|---|---|---|
| 00 | `00_audit_legacy_project.py` | Audit legacy inputs |
| 01 | `01_ingest_sources.py` | Register source inputs |
| 02 | `02_normalize_legacy_news.py` | Normalize legacy news |
| 03 | `03_normalize_llm_news.py` | Normalize LLM-search news |
| 04 | `04_enrich_full_text.py` | Full-text umbrella stage |
| 05 | `05_retrieve_legacy_full_text.py` | Retrieve legacy full text |
| 06 | `06_retrieve_llm_full_text.py` | Retrieve LLM-search full text |
| 07 | `07_merge_full_text_deduplicate.py` | Merge and deduplicate full text |
| 08 | `08_filter_irrelevant_news.py` | Relevance gate; assign canonical events |
| 09 | `09_analyze_pre_llm_corpus.py` | Describe the retained pre-LLM corpus |
| 10 | `10_run_llm_scoring.py` | External LLM scoring (manual gate) |
| 11 | `11_merge_llm_scores.py` | Merge LLM score archives |
| 12 | `12_human_validation_and_llm_benchmark.py` | Inter-rater reliability, gold reference, LLM benchmark |
| 13 | `13_build_disruption_index.py` | Article, monthly-primary and weekly-robustness indices |
| 14 | `14_build_model_panels.py` | Leakage-safe model panels |
| 15 | `15_prediction_benchmarks.py` | Statistical and ML benchmarks |
| 16 | `16_llm_scfi_comparison_weekly.py` | Weekly LLM signal comparison |
| 17 | `17_llm_scfi_comparison_monthly.py` | Monthly LLM signal comparison |
| 18 | `18_forecasting_models.py` | GLM, RF, XGBoost, LSTM, Prophet forecasts |
| 19 | `19_time_series_causality.py` | VAR/VARX, Granger, PCMCI, local projections |
| 20 | `20_causal_ml.py` | Time-aware causal-ML diagnostics |
| 21 | `21_ablation_sensitivity_checks.py` | Ablation and sensitivity checks |
| 22 | `22_error_diagnostics.py` | Forecast and model diagnostics |
| 23 | `23_significance_analysis.py` | Multiplicity-aware inference |
| 24 | `24_make_figures_tables.py` | Publication tables and figures |

### Outside the sequential run

Registered in `OPTIONAL_EXTENSION_SCRIPTS` and compiled by `smoke`, but not part
of stages 00–24:

- `25`–`27` external control cleaning, EDA, statistical association tests
- `28`–`29` pseudo-annotation plumbing, inter-annotator agreement
- `30` PortWatch event study
- `31` publication EDA figures
- `33`–`35` pilot validation sampling, pilot agreement, sample-size allocation
- `37`–`38` event-pair sampling, reliability and threshold evaluation
- `39` persistent event shocks and ripple effects
- `40` external disruption benchmark
- `41` PCMCI directed graph
- `42` apply human-validated event threshold
- `43` article-level dimension PCA
- `44` validation sample-size figure
- `45` entropy-weighted TOPSIS (reporting path for the operational ranking)

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
