```text
sc-disrupt/
├── configs/
│   ├── paths.yaml
│   ├── research_config.yaml
│   ├── taxonomy.yaml
│   ├── scoring_rubric.yaml
│   └── model_registry.yaml
├── data/
│   ├── 00_legacy/
│   ├── 01_raw/
│   │   ├── news/
│   │   ├── freight/
│   │   ├── controls/
│   │   └── exogenous/
│   ├── 02_interim/
│   ├── 03_processed/
│   ├── 04_validation/
│   ├── 05_indices/
│   └── 06_model_ready/
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
│   ├── robustness/
│   └── visualization/
├── annotation/
├── experiments/
├── notebooks/
├── outputs/
├── paper/
├── scripts/
├── tests/
├── README.md
├── TREE.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
└── Makefile

```