.PHONY: setup audit ingest enrich score validate index panel predict causality robustness figures all

setup:
	pip install -r requirements.txt

audit:
	python scripts/00_audit_legacy_project.py

ingest:
	python scripts/01_ingest_sources.py

enrich:
	python scripts/02_enrich_full_text.py

score:
	python scripts/03_run_llm_scoring.py

validate:
	python scripts/04_human_validation_metrics.py

index:
	python scripts/05_build_disruption_index.py

panel:
	python scripts/06_build_model_panels.py

predict:
	python scripts/07_prediction_benchmarks.py

causality:
	python scripts/08_causality_analysis.py

robustness:
	python scripts/09_robustness_checks.py

figures:
	python scripts/10_make_figures_tables.py

all: ingest enrich score validate index panel predict causality robustness figures
