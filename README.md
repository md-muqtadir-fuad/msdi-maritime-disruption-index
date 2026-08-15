# SC-Disrupt

Reproducible research pipeline for building an LLM-derived, human-validated Maritime Supply Chain Disruption Index (MSDI), forecasting container freight rates (SCFI), and conducting econometric time-series causality, causal machine learning (Double/Debiased ML), statistical significance audits, and sensitivity analyses.

---

## 1. Environment & Setup

All commands must be executed using the project virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-forecasting.txt
```

> **Requirements note**: `requirements-forecasting.txt` provides PyTorch, Prophet, and specialized time-series dependencies. Scikit-learn, statsmodels, linearmodels, and core scientific libraries are included in `requirements.txt`.

---

## 2. Pipeline Execution

### Smoke Verification (Fast Architecture & DAG Check)
Run the read-only verification check first. It compiles every script, checks dependencies, validates configuration hashes, and verifies artifact contracts without executing long fits:

```powershell
.venv\Scripts\python.exe scripts\99_run_full_pipeline.py smoke
```

### Full End-to-End Pipeline
To run the full sequential pipeline across Stages `00` through `24`:

```powershell
.venv\Scripts\python.exe scripts\99_run_full_pipeline.py full --allow-pending-human-validation
```

For publication-ready runs requiring completed dual-human annotations:
```powershell
.venv\Scripts\python.exe scripts\99_run_full_pipeline.py full --annotations data/04_validation/human_annotations/
```

---

## 3. Pipeline Stages & Architecture

| Stage | Script | Description |
| :--- | :--- | :--- |
| **00–04** | `scripts/00_...` to `04_...` | Ingestion, news normalization, deduplication, full-text enrichment. |
| **05–08** | `scripts/05_...` to `08_...` | Embedding generation, BGE-M3 event clustering, text-quality filtering. |
| **09–13** | `scripts/09_...` to `13_...` | LLM scoring prompts, dual-prompt evaluation, MSDI index aggregation (Peak & Breadth). |
| **14–15** | `scripts/14_...` to `15_...` | Multi-horizon panel construction (weekly/monthly) and EDA. |
| **16–17** | `scripts/16_...` & `17_...` | Out-of-sample LLM comparison benchmarks (Weekly & Monthly). |
| **18** | `scripts/18_nonlinear_forecasting.py` | Multi-model forecasting benchmarks (GLM, Random Forest, XGBoost, LSTM, Prophet). |
| **19** | `scripts/19_time_series_causality.py` | Econometric time-series causality (VARX, Granger causality, PCMCI+ graphical discovery, Jordà local projections). |
| **20** | `scripts/20_causal_ml.py` | Double/Debiased Machine Learning (Partially Linear DML, Causal Forest GATE/BLP heterogeneity, placebo falsifications). |
| **21** | `scripts/21_ablation_sensitivity_checks.py` | Comprehensive ablation matrix (69 index variants, media controls, prompt architectures). |
| **22** | `scripts/22_error_diagnostics.py` | Forecast error diagnostics, Mincer–Zarnowitz calibration, extreme-case article traces. |
| **23** | `scripts/23_significance_analysis.py` | Multiplicity-aware statistical audit (Holm FWER & Benjamini–Hochberg FDR adjustments across 14,000+ hypotheses). |
| **24** | `scripts/24_make_figures_tables.py` | Publication package generator (26 publication tables, 30 vector/raster figures, summary reports). |
| **25–27** | `scripts/25_...` to `27_...` | External control data ingestion, cleaning, EDA, and statistical association tests. |
| **28–30** | `scripts/28_...` to `30_...` | Synthetic validation plumbing, inter-annotator agreement, and PortWatch event studies. |

---

## 4. Integrated External Controls

The core model panels integrate five macroeconomic and maritime trade controls merged strictly by backward availability date:

1. **CPB World Trade Volume** (`cpb_world_trade_volume`): Global import/export merchandise demand.
2. **RWI/ISL Container Throughput Index** (`rwi_isl_container_throughput_index`): Global physical container throughput across 90+ international ports.
3. **WTO China Merchandise Exports** (`wto_china_merchandise_exports`): Export origin volume benchmark.
4. **CNY/USD Exchange Rate** (`cny_usd_exchange_rate`): Bilateral monetary control.
5. **UNCTAD Liner Shipping Bilateral Connectivity Index** (`unctad_china_core_lane_lsbci`): Bilateral container network connectivity for China-origin core trade lanes.

For complete details on data sources, portal URLs, and ingestion protocols, see [DATA_SOURCES.md](DATA_SOURCES.md).

To clean and analyze external control series:
```powershell
.venv\Scripts\python.exe scripts\25_clean_external_controls.py
.venv\Scripts\python.exe scripts\26_external_controls_eda.py
.venv\Scripts\python.exe scripts\27_external_controls_statistical_analysis.py
```

---

## 5. Testing & Validation

Execute unit and integration tests across all modules:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

---

## 6. Output Artifacts

- **Publication Tables**: [`outputs/tables/`](file:///c:/Users/DELL/Desktop/venv-python/sc-disrupt/sc-disrupt/outputs/tables) (Tables T01–T16, Supplementary Tables S00–S08, and combined Excel workbook).
- **Publication Figures**: [`outputs/figures/`](file:///c:/Users/DELL/Desktop/venv-python/sc-disrupt/sc-disrupt/outputs/figures) (High-resolution PDF and PNG figures F01–F15).
- **Significance Registry**: [`outputs/significance/`](file:///c:/Users/DELL/Desktop/venv-python/sc-disrupt/sc-disrupt/outputs/significance) (Full hypothesis registry with FDR and Holm adjustments).
- **Executive Reports**: [`outputs/reports/`](file:///c:/Users/DELL/Desktop/venv-python/sc-disrupt/sc-disrupt/outputs/reports) (`publication_v001_results_summary.md` and machine-readable run manifests).
