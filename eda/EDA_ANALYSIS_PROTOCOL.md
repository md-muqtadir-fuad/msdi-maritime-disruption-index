# EDA analysis protocol

## Purpose

This document is the controlling protocol for exploratory analysis of the SCFI,
MSDI, LLM scores, news/event data, and external controls. Analysis must proceed
point by point in the order below. No available variable is accepted, rejected,
or assigned a final model role from an isolated correlation or an undocumented
assumption.

The monthly sample is primary. Weekly analysis is supplementary and is used
where the underlying data genuinely support weekly variation.

## Study scope

- Primary period: February 2016 through the last common SCFI month.
- Outcome: SCFI level and future 1-, 3-, and 6-month SCFI log changes.
- News signal: peak MSDI, breadth MSDI, seven disruption dimensions, model-
  specific scores, consensus scores, event weights, and coverage variables.
- Existing controls: Brent, CPB world trade, GSCPI, UNCTAD LSBCI, and IMF
  PortWatch.
- Newly collected controls: RWI/ISL container throughput, China merchandise
  exports, CNY/USD, geopolitical risk, trade-policy uncertainty, US-China
  tension, oil-price uncertainty, and infectious-disease EMV.
- LLMs: Gemma, Qwen model variants, and GLM present in the model registry and
  merged score file.

## Non-selection rule

Every successfully parsed variable remains in the EDA. A variable may later be
excluded from a particular specification only for a documented reason such as
unrecoverable data errors, inadequate temporal support, target leakage,
construct overlap, lack of availability at forecast time, or a causal role that
makes conditioning inappropriate. Statistical significance, SCFI correlation,
or forecast improvement alone is not an exclusion or inclusion rule.

## Point 1 — Raw-data integrity

### Questions

- What is the source, unit, native frequency, release schedule, and update date?
- What are the first and last usable observations?
- Are dates missing, duplicated, malformed, or unexpectedly irregular?
- Are values non-numeric, duplicated, infinite, or outside defensible bounds?
- Does the file cover February 2016 through the last common SCFI month?
- Is the series revised after first publication?

### Outputs

- Raw-file inventory with size, format, worksheets, and provenance.
- Series-level coverage and integrity report.
- Missing-date and duplicate-date report.
- Value-validity and extreme-value report.
- Machine-readable audit manifest.

### Decision rule

No variable is discarded here. Problems are classified as repairable,
restriction-requiring, or unresolved.

## Point 2 — Variable construction

### Questions

- What does each variable measure and in what unit?
- Is a higher value expected to increase or decrease freight pressure?
- Is the measure a level, rate, count, price, index, score, or uncertainty proxy?
- How were SCFI returns, MSDI paths, dimension scores, consensus scores, model
  disagreement, coverage counts, and event weights constructed?

### Outputs

- Variable dictionary.
- Construction formulas and aggregation rules.
- Expected-direction table with economic justification.
- Mapping from raw fields to cleaned fields.

### Decision rule

Expected signs are documented before outcome modelling and are not altered to
match observed coefficients.

## Point 3 — Temporal properties

### Checks

- Levels, log levels where valid, monthly changes, and year-over-year changes.
- Trend, seasonality, ACF, PACF, and persistence.
- ADF and KPSS stationarity tests with explicit transformation labels.
- Distribution, skewness, kurtosis, zero mass, and heavy tails.
- Robust extreme-value flags without automatically deleting genuine events.
- Structural-break diagnostics around COVID-19, Ever Given, the Russia-Ukraine
  war, and Red Sea disruption periods.

### Outputs

- Temporal diagnostics tables.
- Stationarity table.
- Outlier/event concordance table.
- Structural-break and regime summaries.
- Panel figures with sentence-case panel titles and no main figure title.

## Point 4 — Availability and leakage

### Checks

- Observation date versus first public availability date.
- Conservative publication delay and revision status.
- Whether monthly aggregation uses information available at forecast origin.
- Whether interpolation, backfilling, or forward filling leaks future values.
- Effective sample after backward-only availability joins.
- Construct overlap between GSCPI, related freight measures, SCFI, and MSDI.

### Outputs

- Source-contract and release-lag table.
- Leakage-safe monthly panel audit.
- Availability mismatch report.
- Complete-case counts by forecast origin and predictor family.

## Point 5 — Relationship with SCFI

### Checks for every variable

- Contemporaneous levels, labelled nonstationary and descriptive.
- Monthly changes and year-over-year changes.
- Pearson and Spearman associations.
- Predictor-leading lags of one through six months.
- Reverse lags to determine whether SCFI leads the predictor.
- FDR-adjusted inference across declared comparison families.
- Cross-correlation after prewhitening.
- Nonlinear scatter diagnostics.
- Rolling and regime-specific relationships.

### Guardrail

One significant lag is insufficient evidence of a stable predictive or causal
relationship.

## Point 6 — Relationships among controls

### Checks

- Pearson and Spearman correlation matrices.
- VIF and numerical conditioning.
- Near-duplicate series and hierarchical clustering.
- CPB, China exports, and RWI as potentially overlapping demand measures.
- GPR, TPU, US-China tension, oil-price uncertainty, infectious-disease EMV,
  and MSDI as potentially overlapping news-attention measures.

### Outputs

- Redundancy matrix and clusters.
- VIF table under complete-case and family-specific samples.
- Pre-specified alternative predictor families rather than outcome-selected
  subsets.

## Point 7 — MSDI-specific validation

### Checks

- Articles and sources per month and source concentration.
- Event counts, event sizes, date spans, and duplicate clusters.
- Results under semantic thresholds 0.85, 0.88, 0.90, and 0.92 when available.
- Novelty and event-weight distributions.
- Seven dimension distributions, zero mass, and extreme scores.
- Agreement among Gemma, Qwen variants, and GLM.
- Model-specific temporal patterns and disagreement.
- Peak-versus-breadth divergence.
- Relationship between news volume and MSDI.
- Dominance of COVID-related content across periods.

### Outputs

- Corpus/event integrity tables.
- LLM agreement and score-distribution tables.
- Threshold comparison tables pending human validation.
- Dimension and coverage diagnostics.

## Point 8 — Forecasting readiness

### Low-cost checks

- Distribution and availability of 1-, 3-, and 6-month targets.
- Effective observations for every predictor family.
- SCFI persistence and naive-baseline strength.
- Incremental information beyond lagged SCFI using small leakage-safe rolling
  regressions without hyperparameter tuning.
- Dependence created by overlapping multi-month targets.
- Evidence of nonlinearity that could justify later nonlinear models.

### Guardrail

EDA uses only lightweight fixed specifications. RF, XGBoost, LSTM, Prophet,
GRU, N-BEATSx, BiTCN, Optuna, and full tuning remain outside this stage.

## Point 9 — Causal-readiness audit

### Classification

- Exposure.
- Outcome.
- Confounder.
- Mediator.
- Competing proxy.
- Benchmark.
- External shock or event variable.

### Checks

- Temporal ordering and data availability.
- Support and overlap across exposure levels.
- Whether a proposed control could block part of the MSDI-to-SCFI pathway.
- Whether common causes are measured or materially omitted.
- Whether any defensible identification design exists beyond observational
  dynamic association.

### Output

- Variable-role table, proposed DAG inputs, identification limitations, and a
  claim gate that prevents causal language unless identification assumptions
  are defensible.

## Execution order and status

| Point | Analysis | Status |
|---:|---|---|
| 1 | Raw-data integrity | Test scope complete; three large-table deep scans pending |
| 2 | Variable construction | Test scope complete |
| 3 | Temporal properties | Test scope complete |
| 4 | Availability and leakage | Test scope complete; release-contract gaps documented |
| 5 | Relationship with SCFI | Test scope complete; descriptive only |
| 6 | Relationships among controls | Test scope complete |
| 7 | MSDI-specific validation | Computational test scope complete; human validation pending |
| 8 | Forecasting readiness | Test scope complete; no tuning performed |
| 9 | Causal-readiness audit | Test scope complete; causal claim gate closed |

## Reproducibility and figure rules

- Raw inputs are read-only.
- Cleaned files are written only under `data/02_interim/`.
- Generated EDA products are written under `eda/outputs/`.
- Every output records input paths, sample dates, transformations, and run mode.
- Figures use readable sentence-case labels, no raw variable names, and no main
  title. Multi-panel figures may use a short title on each panel.
- Test mode performs no model tuning or computationally expensive forecasting.
- `99_run_eda.py` is the only start-to-finish entry point. It runs Points 1–9
  in order, stops on failure, records logs and a manifest, and refuses a normal
  run when any point script is missing.
