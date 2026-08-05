# Repository hierarchy

```text
sc-disrupt/
├── configs/
│   ├── paths.yaml
│   ├── research_config.yaml
│   ├── taxonomy.yaml
│   ├── scoring_rubric.yaml
│   ├── model_registry.yaml
│   ├── nonlinear_forecasting.yaml      # fixed parameters + opt-in tuning candidates
│   └── causal_ml.yaml                  # frozen DML estimand, folds, learners, inference
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
│       ├── causal_ml/                   # reserved for future route/IV panels
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
│   │   └── causal_ml/                   # reusable causal-ML package namespace
│   ├── robustness/
│   └── visualization/
├── docs/
│   ├── annotation_guidelines/
│   ├── codebook/
│   ├── methodology/
│   │   ├── methodology_overview.md
│   │   ├── q1_research_blueprint.md
│   │   └── causal_ml.md
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
│   ├── 02_normalize_legacy_news.py
│   ├── 03_normalize_llm_news.py
│   ├── 04_enrich_full_text.py           # umbrella marker
│   ├── 05_retrieve_legacy_full_text.py
│   ├── 06_retrieve_llm_full_text.py
│   ├── 07_merge_full_text_deduplicate.py
│   ├── 08_filter_irrelevant_news.py
│   ├── 09_analyze_pre_llm_corpus.py
│   ├── 10_run_llm_scoring.py            # external/manual gate
│   ├── 11_merge_llm_scores.py
│   ├── 12_human_validation_metrics.py
│   ├── 13_build_disruption_index.py
│   ├── 14_build_model_panels.py
│   ├── 15_prediction_benchmarks.py      # statistical + ML forecasts, including GLM
│   ├── 16_llm_scfi_comparison_weekly.py # common-cohort LLM/dimension/hybrid forecasts
│   ├── 17_llm_scfi_comparison_monthly.py # dual-construction monthly robustness
│   ├── 18_forecasting_models.py         # GLM/RF/XGBoost/LSTM/Prophet
│   ├── 19_time_series_causality.py       # VAR/VARX, Granger, PCMCI, local projections
│   ├── 20_causal_ml.py                   # time-aware DML + guarded heterogeneity
│   ├── 21_ablation_sensitivity_checks.py
│   ├── 22_error_diagnostics.py
│   ├── 23_significance_analysis.py
│   ├── 24_make_figures_tables.py
│   └── 99_run_full_pipeline.py         # read-only smoke / full tuned orchestration
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
                │   ├── Pre-specified GATE/BLP heterogeneous effects
                │   ├── DML-IV when an instrument is defensible
                │   └── Causal forests only after identification/data gates
                └── Publication-validity audit
                    ├── Component ablation
                    ├── Frequency/specification sensitivity
                    ├── Error and residual diagnostics
                    └── Multiplicity-aware significance analysis
```

Items marked `planned` describe the Q1 research target architecture and are not yet implemented.
