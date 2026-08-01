.PHONY: setup audit ingest enrich score validate index panel predict compare-llms-weekly compare-llms-monthly causality causal-ml robustness diagnostics significance figures analysis all

ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
else
PYTHON ?= .venv/bin/python
endif

setup:
	$(PYTHON) -m pip install -r requirements.txt

audit:
	$(PYTHON) scripts/00_audit_legacy_project.py

ingest:
	$(PYTHON) scripts/01_ingest_sources.py

enrich:
	$(PYTHON) scripts/02_enrich_full_text.py

score:
	$(PYTHON) scripts/03_run_llm_scoring.py

validate:
	$(PYTHON) scripts/04_human_validation_metrics.py

index:
	$(PYTHON) scripts/05_build_disruption_index.py

panel:
	$(PYTHON) scripts/06_build_model_panels.py

predict:
	$(PYTHON) scripts/07_prediction_benchmarks.py

compare-llms-weekly:
	$(PYTHON) scripts/07b_llm_scfi_comparison.py

compare-llms-monthly: compare-llms-weekly
	$(PYTHON) scripts/07c_monthly_llm_scfi_comparison.py

causality:
	$(PYTHON) scripts/08_causality_analysis.py

causal-ml:
	$(PYTHON) scripts/08b_causal_ml.py

robustness:
	$(PYTHON) scripts/09_robustness_checks.py

diagnostics:
	$(PYTHON) scripts/09b_error_diagnostics.py

significance:
	$(PYTHON) scripts/09c_significance_analysis.py

figures:
	$(PYTHON) scripts/10_make_figures_tables.py

analysis: predict compare-llms-monthly causality causal-ml robustness diagnostics significance figures

all: ingest enrich score validate index panel analysis
