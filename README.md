# SC-Disrupt

Lightweight toolkit and data pipelines for constructing a supply‑chain disruption index, running experiments, and validating models.

Contents
- `data/`: raw, interim, processed, and validation datasets (not tracked; see note below)
- `scripts/`: runnable scripts for ingestion, enrichment, scoring, and analysis
- `src/`: project source code and modules
- `notebooks/`: exploratory analysis and reproducible workflows
- `outputs/`: generated figures, tables, logs, and reports

Quickstart
1. Create environment (Conda) or use `requirements.txt`:

	conda env create -f environment.yml
	conda activate sc-disrupt

	or

	python -m venv .venv
	.venv\Scripts\activate
	pip install -r requirements.txt

2. Populate `data/` (this repository does not include large raw data).
3. Run ingestion/enrichment/scoring examples:

	python scripts/01_ingest_sources.py
	python scripts/02_enrich_full_text.py
	python scripts/03_run_llm_scoring.py

Testing

	pytest -q

Notes
- The `data/` directory is intentionally ignored from Git (large, sensitive files). Add only metadata or small examples if needed.
- Configs and paths live under `configs/` (see `configs/paths.yaml`).

Contributing
- Open an issue or submit a PR with tests where appropriate.

---
Short README: for a longer project README, expand sections above and add examples.

