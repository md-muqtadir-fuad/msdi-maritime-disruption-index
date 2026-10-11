# Maritime Supply Chain Disruption Index (MSDI): dataset

Version 1.0.0. License: CC BY 4.0 (see `LICENSE.txt`).

The MSDI is a news-based measure of maritime supply chain disruption covering
January 2015 to January 2026. Four open-weight LLMs scored 6,542 news
articles from 94 publishers on a 0–4 severity scale across seven
disruption dimensions. Article scores were combined by ensemble median,
down-weighted for syndicated duplicates and repeated coverage of the same
event, and aggregated into monthly (primary) and weekly (robustness) series.
LLM scores were checked against two independent human annotators.

The analysis code is archived separately; see the related software record.

## Contents

```text
articles/
  articles.csv              6,542 articles: ID, date, headline, title, publisher, URL,
                            duplicate group and event-cluster assignments
  article_scores.csv        per-model and ensemble scores for every dimension,
                            between-model differences, weights and index contributions
index/
  msdi_monthly.csv          monthly MSDI, 2015-01 to 2026-01 (primary)
  msdi_weekly.csv           weekly MSDI, 2015-01-02 to 2026-01-02 (robustness)
  *_model_specific.csv      the same series built from each LLM separately
  *_model_spread.csv        between-model spread, for uncertainty bands
validation/
  pilot/                    50-article pilot, two annotators
  full/                     400-article validation, inter-rater reliability, LLM-vs-human benchmark
  event_pairs/              300 article pairs rated as same or different event; threshold evaluation
data_dictionary.csv         every column in articles/ and index/, with description
SHA256SUMS.txt              checksums for every file
```

`articles.csv` and `article_scores.csv` join on `article_id`. The validation
files join to articles through `validation/full/templates/final_validation_master.csv`,
which maps every `validation_id` (pilot `P…` and full `V…`) to `article_id`.
Event-pair files carry `article_id_a` and `article_id_b` directly.

## Scoring

| Score | Meaning |
|---|---|
| 0 | no disruption, or irrelevant |
| 1 | low or localized disruption |
| 2 | moderate operational disruption |
| 3 | high regional or multi-actor disruption |
| 4 | severe systemic or key-route disruption |

Dimensions: port, global container movement, freight cost, supply, production,
COVID-19-related, and demand. `overall_disruption_max` is a model's maximum
across them.

Models: Gemma-2-9B-it, Qwen3-8B, Qwen2.5-7B-Instruct, GLM-4-9B-0414. Columns
ending in `__<model>` hold that model's score. Columns ending in
`__ensemble_median` and similar hold the cross-model statistics.

The primary article score is `peak_severity_score`, the ensemble median of
`overall_disruption_max`. The secondary score is `breadth_severity_score`, the
mean of the seven dimension medians.

## Index construction

Each article starts with a source weight of 1/sqrt(duplicate group size).
Articles reporting the same event in the same period then share a total
weight of 1 (`novelty_weight`), so heavy coverage of one event counts once. Events are semantic clusters of article embeddings at cosine
similarity 0.82, a threshold chosen with the human event-pair validation.
A period's index sums the weighted article scores and corrects for media
volume. Columns ending in `_z_expanding_realtime` are standardized using only
data available up to each period, so they can be used in forecasting without
look-ahead.

## What is not included

- **Article text.** The articles are copyrighted by their publishers. Use the
  URL to retrieve an article. In the validation files the text columns have
  been removed, but titles, IDs and every rating are unchanged. Annotators
  rated the full text.
- **Freight rates and control variables.** The Shanghai Containerized Freight
  Index (SCFI), Brent crude and the macroeconomic controls used in the study
  belong to third parties and must be obtained from their providers.

## Known limitations

- Some URLs will stop resolving over time. Archived copies may be available
  through the Internet Archive.
- Three articles have no parseable publication date and are excluded from the
  time series.
- 19 articles did not receive enough successful model scores to enter the
  index (`ensemble_eligible` is False).

## Citation

Cite this dataset with its Zenodo DOI.

Authors:

- Md. Muqtadir Fuad, Institute of Appropriate Technology, Bangladesh
  University of Engineering and Technology (ORCID 0009-0009-4362-0780)
- Md. Ashraful Huda Asif, Department of Industrial and Production Engineering,
  Bangladesh University of Engineering and Technology (ORCID 0009-0000-3874-234X)
- Abdullah Al Mazid, Department of Industrial and Production Engineering,
  Bangladesh University of Engineering and Technology (ORCID 0009-0003-9834-4652)

Contributors: Jannatul Maoya Hossain (Department of Industrial and Production
Engineering, Bangladesh Army University of Science and Technology;
ORCID 0009-0009-4222-5949) and Syed Mithun Ali (Department of Industrial and
Production Engineering, Bangladesh University of Engineering and Technology).
