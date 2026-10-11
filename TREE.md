# Repository hierarchy

Directories and files below exist in the working tree. Nothing here is aspirational.
Folders marked *local only* are git-ignored and not part of the archived release.

```text
sc-disrupt/
├── configs/
│   ├── causal_ml.yaml                     # frozen DML estimand, folds, learners, inference
│   ├── error_diagnostics.yaml
│   ├── event_clustering.yaml
│   ├── external_controls.yaml
│   ├── model_registry.yaml                # 4 scoring models + TOPSIS criteria
│   ├── nonlinear_forecasting.yaml         # fixed parameters + opt-in tuning candidates
│   ├── paths.yaml
│   ├── portwatch_event_study.yaml
│   ├── publication_outputs.yaml
│   ├── publication_outputs_script.sha256  # stage 24 self-integrity lock
│   ├── research_config.yaml
│   ├── robustness.yaml
│   ├── scoring_rubric.yaml
│   ├── significance.yaml
│   ├── significance_script.sha256
│   └── taxonomy.yaml
├── data/                                  # local only
│   ├── 00_legacy/
│   ├── 01_raw/
│   │   ├── news/
│   │   ├── freight/                       # SCFI
│   │   └── controls/                      # brent/, macro/, connectivity/, portwatch/
│   ├── 02_interim/
│   │   ├── news_cleaned/
│   │   └── article_enrichment/
│   ├── 03_processed/
│   │   ├── news_full_text/                # incl. pre_llm_filter/event_dedup/
│   │   └── model_scores/
│   │       └── scores_zips/               # external LLM scoring archives
│   ├── 05_indices/
│   │   ├── article_level/
│   │   ├── disruption_index_versions/
│   │   ├── monthly/                       # primary
│   │   ├── weekly/                        # robustness
│   │   └── uncertainty_bands/
│   ├── 06_model_ready/
│   │   ├── prediction/
│   │   ├── causality/
│   │   └── robustness/
│   └── llm_dataset/
├── prompts/
│   ├── 00_relevance/
│   ├── 01_event_extraction/
│   ├── 02_taxonomy_classification/
│   └── 03_severity_scoring/
├── src/msdi/
│   └── preprocessing/                     # event_clustering.py
├── validation/                            # executed human validation
│   ├── full/                              # N=400 gold standard
│   ├── pilot/                             # N=50 pilot
│   └── event_pairs/                       # N=300 adjudicated pairs
├── eda/                                   # standalone EDA scripts 00–09
├── latex/                                 # local only: manuscript sources and figures
├── misc/
│   ├── fig_readability_and _dry_change/   # figure redraw toolkit (saved results only)
│   └── zenodo_dataset/                    # builds the standalone dataset package
├── notebooks/                             # Kaggle LLM scoring and event embedding
├── outputs/                               # mostly local only; reference results tracked
│   ├── prediction_benchmarks/             # rolling forecasts, DM / Clark-West
│   ├── llm_scfi_comparison/
│   ├── nonlinear_forecasts/               # GLM/RF/XGBoost/LSTM/Prophet
│   ├── time_series_causality/             # VAR/VARX, Granger, PCMCI, local projections
│   ├── causal_ml/                         # DML outputs
│   ├── robustness/                        # ablation and sensitivity
│   ├── diagnostics/
│   ├── significance/
│   ├── external_controls_eda/
│   ├── external_controls_statistics/
│   ├── model_panel_audit/
│   ├── pre_llm_kept_analysis/
│   ├── pca/
│   ├── topsis/
│   ├── pipeline_runs/
│   ├── presentations/
│   ├── tables/
│   ├── figures/                           # incl. eda/ subdirectory
│   ├── logs/
│   └── reports/
├── scripts/
│   ├── 00_audit_legacy_project.py
│   ├── 01_ingest_sources.py
│   ├── 02_normalize_legacy_news.py
│   ├── 03_normalize_llm_news.py
│   ├── 04_enrich_full_text.py             # umbrella marker
│   ├── 05_retrieve_legacy_full_text.py
│   ├── 06_retrieve_llm_full_text.py
│   ├── 07_merge_full_text_deduplicate.py
│   ├── 08_filter_irrelevant_news.py
│   ├── 09_analyze_pre_llm_corpus.py
│   ├── 10_run_llm_scoring.py              # external/manual gate
│   ├── 11_merge_llm_scores.py
│   ├── 12_human_validation_and_llm_benchmark.py
│   ├── 13_build_disruption_index.py
│   ├── 14_build_model_panels.py
│   ├── 15_prediction_benchmarks.py
│   ├── 16_llm_scfi_comparison_weekly.py
│   ├── 17_llm_scfi_comparison_monthly.py
│   ├── 18_forecasting_models.py
│   ├── 19_time_series_causality.py
│   ├── 20_causal_ml.py
│   ├── 21_ablation_sensitivity_checks.py
│   ├── 22_error_diagnostics.py
│   ├── 23_significance_analysis.py
│   ├── 24_make_figures_tables.py          # self SHA-256 integrity lock
│   ├── 25_clean_external_controls.py
│   ├── 26_external_controls_eda.py
│   ├── 27_external_controls_statistical_analysis.py
│   ├── 29_inter_annotator_agreement.py
│   ├── 30_portwatch_event_study.py
│   ├── 31_generate_publication_eda_figures.py
│   ├── 33_sample_pilot_validation.py
│   ├── 34_compute_pilot_agreement.py
│   ├── 35_human_validation_sample_size_and_allocation.py
│   ├── 37_sample_event_pair_validation.py
│   ├── 38_event_pair_reliability_and_threshold_evaluation.py
│   ├── 39_persistent_event_shocks_and_ripple_effects.py
│   ├── 40_external_disruption_benchmark.py
│   ├── 41_plot_pcmci_directed_graph.py
│   ├── 42_apply_human_validated_event_threshold.py
│   ├── 43_dimension_pca.py
│   ├── 44_validation_sample_size_figure.py
│   ├── 45_entropy_weighted_topsis.py
│   ├── 46_interaction_structure_forecast_target.py
│   ├── 47_monthly_forecast_trajectories_by_horizon.py
│   ├── 48_pcmci_gpdc_nonlinear_robustness.py
│   ├── 49_shanghai_vs_other_ports.py
│   └── 99_run_full_pipeline.py            # smoke / full orchestration
├── tests/                                 # 150 tests
├── .manu/                                 # local only: research notes
├── README.md
│   ├── pipeline_variables_and_mathematical_formulation.md
│   ├── event_clustering_threshold_calibration_and_event_counts.md
│   └── docs/
│       ├── annotation_guidelines/
│       ├── codebook/
│       ├── methodology/
│       ├── misc/
│       └── reproducibility/
├── release/                               # local only: built dataset package and zip
├── README.md
├── TREE.md
├── CITATION.cff
├── .zenodo.json                           # Zenodo release metadata
├── LICENSE
├── requirements.txt                       # direct dependencies
├── requirements-lock.txt                  # exact versions used for the results
├── environment.yml
├── pyproject.toml
└── Makefile
```

Scripts 28, 32 and 36 do not exist; the numbering is not contiguous.

## Analysis layers

```text
News ingestion and full-text enrichment
    └── Four-model LLM scoring and score merging
        └── Human validation (400 articles, 300 event pairs)
            └── Article consensus, semantic event clustering (θ=0.82), MSDI
                │   monthly primary, weekly robustness
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
                │   └── Local projections
                ├── Causal machine learning
                │   ├── Time-aware Double/Debiased ML
                │   └── Exploratory R-loss GATE/BLP heterogeneity
                └── Publication-validity audit
                    ├── Component ablation
                    ├── Frequency/specification sensitivity
                    ├── Error and residual diagnostics
                    └── Multiplicity-aware significance analysis
```

Not implemented, and deliberately so: causal forests, DML-IV, event-study or IV
identification, and route/exposure heterogeneity panels. Stage 20 records
`causal_forest_run: False` with its skip reason. No structural causal claim is
authorized anywhere in the pipeline.
