# Semantic-event pair validation

- `inputs/`: the audited 300-pair sample and blinded annotation templates.
- `completed/`: immutable completed sheets for the 0.82, 0.85, 0.88, and 0.90 strata.
- `results/`: adjudicated labels, inter-annotator agreement, and threshold metrics.
- `event_pair_annotation_rubric_guide.md`: annotation decision rules.

The final threshold is 0.82, selected by maximum sample F1 on the adjudicated
300-pair reference. SCFI and downstream forecast results were not used in this
selection.
