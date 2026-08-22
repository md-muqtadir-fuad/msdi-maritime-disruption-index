# Human-validation registry

This directory separates the three completed validation exercises. Generated
results are kept apart from blank templates and immutable completed ratings.

```text
validation/
├── pilot/                  # 50-article pilot
│   ├── templates/          # blinded sheets and master sample
│   ├── completed/          # completed independent ratings
│   ├── intermediate/       # retained cleaning artifact
│   └── results/            # reliability and sample-size calculations
├── full/                   # 400-article final LLM-score validation
│   ├── templates/          # 350 additional-article sheets and master sample
│   ├── completed/          # completed independent ratings
│   └── results/            # gold labels, reliability, and LLM benchmarks
└── event_pairs/            # 300-pair semantic-event validation
    ├── inputs/             # blinded packages and the audited sampling frame
    ├── completed/          # completed ratings, separated by threshold stratum
    └── results/            # adjudication, IAA, and threshold evaluation
```

Do not edit files under `completed/` after analysis. Regenerated computational
outputs belong under `results/`. The human-selected event threshold is 0.82;
its evidence is `event_pairs/results/event_threshold_evaluation_matrix.csv`.

Scripts:

- `scripts/33_sample_pilot_validation.py`: create pilot templates.
- `scripts/34_compute_pilot_agreement.py`: pilot agreement.
- `scripts/35_human_validation_sample_size_and_allocation.py`: final-sample design.
- `scripts/12_human_validation_and_llm_benchmark.py`: final validation and LLM benchmark.
- `scripts/37_sample_event_pair_validation.py`: create event-pair templates.
- `scripts/38_event_pair_reliability_and_threshold_evaluation.py`: event-pair analysis.
- `scripts/42_apply_human_validated_event_threshold.py`: apply the selected
  threshold to saved similarities without rerunning embeddings.
