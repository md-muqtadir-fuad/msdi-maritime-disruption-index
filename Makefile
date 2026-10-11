.PHONY: setup audit ingest normalize-legacy normalize-llm retrieve-legacy retrieve-llm merge-text filter-news analyze-news score merge-scores validation index panel predict compare-llms-weekly compare-llms-monthly forecast-models forecast-models-tuned causality causal-ml robustness diagnostics significance figures pipeline-smoke pipeline-full analysis all

ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
else
PYTHON ?= .venv/bin/python
endif

PIPELINE_ARGS ?=

setup:
	$(PYTHON) -m pip install -r requirements.txt

audit:
	$(PYTHON) scripts/00_audit_legacy_project.py

ingest:
	$(PYTHON) scripts/01_ingest_sources.py

normalize-legacy:
	$(PYTHON) scripts/02_normalize_legacy_news.py

normalize-llm:
	$(PYTHON) scripts/03_normalize_llm_news.py

retrieve-legacy:
	$(PYTHON) scripts/05_retrieve_legacy_full_text.py --resume

retrieve-llm:
	$(PYTHON) scripts/06_retrieve_llm_full_text.py --resume

merge-text:
	$(PYTHON) scripts/07_merge_full_text_deduplicate.py

filter-news:
	$(PYTHON) scripts/08_filter_irrelevant_news.py

analyze-news:
	$(PYTHON) scripts/09_analyze_pre_llm_corpus.py

score:
	$(PYTHON) scripts/10_run_llm_scoring.py

merge-scores:
	$(PYTHON) scripts/11_merge_llm_scores.py

validation:
	$(PYTHON) scripts/12_human_validation_and_llm_benchmark.py

index:
	$(PYTHON) scripts/13_build_disruption_index.py

panel:
	$(PYTHON) scripts/14_build_model_panels.py

predict:
	$(PYTHON) scripts/15_prediction_benchmarks.py

compare-llms-weekly:
	$(PYTHON) scripts/16_llm_scfi_comparison_weekly.py

compare-llms-monthly: compare-llms-weekly
	$(PYTHON) scripts/17_llm_scfi_comparison_monthly.py

forecast-models: compare-llms-monthly
	$(PYTHON) scripts/18_forecasting_models.py --frequency both

forecast-models-tuned: compare-llms-monthly
	$(PYTHON) scripts/18_forecasting_models.py --frequency both --tune --resume

causality:
	$(PYTHON) scripts/19_time_series_causality.py

causal-ml:
	$(PYTHON) scripts/20_causal_ml.py

robustness:
	$(PYTHON) scripts/21_ablation_sensitivity_checks.py

diagnostics:
	$(PYTHON) scripts/22_error_diagnostics.py

significance:
	$(PYTHON) scripts/23_significance_analysis.py

figures:
	$(PYTHON) scripts/24_make_figures_tables.py

pipeline-smoke:
	$(PYTHON) scripts/99_run_full_pipeline.py smoke

pipeline-full:
	$(PYTHON) scripts/99_run_full_pipeline.py full $(PIPELINE_ARGS)

analysis: predict compare-llms-monthly forecast-models causality causal-ml robustness diagnostics significance figures

all: pipeline-smoke
