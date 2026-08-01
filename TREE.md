# Repository hierarchy

```text
sc-disrupt/
├── configs/
│   ├── paths.yaml
│   ├── research_config.yaml
│   ├── taxonomy.yaml
│   ├── scoring_rubric.yaml
│   ├── model_registry.yaml
│   └── nonlinear_forecasting.yaml      # fixed parameters + opt-in tuning candidates
├── data/
│   ├── 00_legacy/
│   ├── 01_raw/
│   │   ├── news/
│   │   ├── freight/
│   │   ├── controls/
│   │   └── exogenous/
│   ├── 02_interim/
│   │   ├── news_cleaned/
│   │   └── article_enrichment/
│   ├── 03_processed/
│   │   ├── news_full_text/
│   │   └── model_scores/
│   │       ├── scores_zips/
│   │       ├── classified_headlines_legacy.csv
│   │       ├── llm_scores_merged.csv
│   │       └── llm_scores_merged_audit.json
│   ├── 04_validation/
│   │   ├── annotation_samples/          # planned human gold-set samples
│   │   ├── raw_annotations/             # planned blinded annotator labels
│   │   └── adjudicated_labels/          # planned final human reference labels
│   ├── 05_indices/
│   │   ├── article_level/               # planned consensus scores and uncertainty
│   │   ├── event_clusters/              # planned novelty/deduplication weights
│   │   └── weekly/                      # planned MSDI and sub-indices
│   └── 06_model_ready/
│       ├── prediction/
│       ├── causality/
│       ├── causal_ml/                   # planned DML/causal-forest panels
│       └── robustness/
├── prompts/
│   ├── 00_relevance/
│   ├── 01_event_extraction/
│   ├── 02_taxonomy_classification/
│   ├── 03_severity_scoring/
│   └── 04_confidence_evidence/
├── src/msdi/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── enrichment/
│   ├── taxonomy/
│   ├── llm_scoring/
│   ├── validation/
│   ├── indexing/
│   ├── external_data/
│   ├── modeling/
│   │   ├── prediction/
│   │   ├── causality/
│   │   └── causal_ml/                   # planned DML and heterogeneous effects
│   ├── robustness/
│   └── visualization/
├── docs/
│   ├── annotation_guidelines/
│   ├── codebook/
│   ├── methodology/
│   │   ├── methodology_overview.md
│   │   └── q1_research_blueprint.md
│   └── reproducibility/
├── annotation/
├── experiments/
├── notebooks/
├── outputs/
│   ├── prediction_benchmarks/           # rolling forecasts and DM/Clark-West tests
│   ├── llm_scfi_comparison/             # common-cohort LLM and hybrid forecast comparison
│   ├── nonlinear_forecasts/             # RF/XGBoost/LSTM/Prophet weekly + monthly outputs
│   ├── time_series_causality/           # VAR/VARX, Granger, PCMCI, local projections
│   ├── causal_ml/                       # DML and heterogeneous-effect outputs
│   ├── robustness/                      # ablation and sensitivity outputs
│   ├── diagnostics/                     # forecast/residual/error diagnostics
│   ├── significance/                    # consolidated multiplicity-aware inference
│   ├── tables/
│   ├── figures/
│   ├── logs/
│   └── reports/
├── paper/
├── scripts/
│   ├── 00_audit_legacy_project.py
│   ├── 01_ingest_sources.py
│   ├── 01_normalize_legacy_news.py
│   ├── 01b_clean_llm_dataset.py
│   ├── 02_retrieve_full_text.py
│   ├── 02b_retrieve_llm_full_text.py
│   ├── 02c_merge_full_text_deduplicate.py
│   ├── 02d_filter_irrelevant_news.py
│   ├── 02e_analyze_pre_llm_kept.py
│   ├── 03_run_llm_scoring.py
│   ├── 03a_merge_llm_scores.py
│   ├── 04_human_validation_metrics.py
│   ├── 05_build_disruption_index.py
│   ├── 06_build_model_panels.py
│   ├── 07_prediction_benchmarks.py      # statistical + time-series ML forecasts
│   ├── 07b_llm_scfi_comparison.py       # common-cohort LLM/dimension/hybrid forecasts
│   ├── 07c_monthly_llm_scfi_comparison.py # dual-construction monthly robustness forecasts
│   ├── 07d_nonlinear_forecasts.py       # leakage-safe RF/XGBoost/LSTM/Prophet forecasts
│   ├── 08_causality_analysis.py         # VAR/VARX, Granger, PCMCI, local projections
│   ├── 08b_causal_ml.py                 # time-aware DML/causal forests
│   ├── 09_robustness_checks.py          # ablation and sensitivity
│   ├── 09b_error_diagnostics.py         # residual, regime, event, influence errors
│   ├── 09c_significance_analysis.py     # CIs, effect sizes, p-value correction
│   └── 10_make_figures_tables.py
├── tests/
├── README.md
├── TREE.md
├── requirements.txt
├── requirements-forecasting.txt         # optional heavy forecasting dependencies
├── environment.yml
├── pyproject.toml
└── Makefile
```

## Research-analysis layers

```text
News ingestion and full-text enrichment
    └── Three-model LLM scoring and score merging
        └── Human annotation and measurement validation
            └── Article consensus, event clustering, and weekly MSDI
                ├── External and known-event validation
                ├── Out-of-sample freight-rate forecasting
                │   ├── Statistical time-series benchmarks
                │   ├── Lag-feature time-series ML
                │   ├── Common-cohort individual-LLM and hybrid comparison
                │   └── DM / Clark-West forecast significance
                ├── Econometric dynamic analysis
                │   ├── VAR / VARX
                │   ├── Conventional Granger predictive precedence
                │   ├── PCMCI conditional temporal dependencies
                │   ├── Local projections
                │   └── Event-study or IV identification
                ├── Causal machine learning
                │   ├── Time-aware Double/Debiased ML
                │   ├── DML-IV when an instrument is defensible
                │   └── Causal forests and heterogeneous effects
                └── Publication-validity audit
                    ├── Component ablation
                    ├── Frequency/specification sensitivity
                    ├── Error and residual diagnostics
                    └── Multiplicity-aware significance analysis
```

Items marked `planned` describe the Q1 research target architecture and are not yet implemented.
