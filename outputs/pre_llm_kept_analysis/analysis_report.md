# Pre-LLM Kept Dataset Analysis

## Dataset overview

- Input rows: **6,542**
- Unique publisher names: **110**
- Unique source domains: **94**
- Parsed publication-date coverage: **99.95%**
- Publication range: **2015-01-01** to **2026-01-01**
- Mean full-text length: **3,854.09 characters**
- Median full-text length: **3,050.50 characters**
- Mean word count: **614.73 words**
- Median word count: **485.00 words**

## Top publishers

| Source | Domain | Articles | Share |
|---|---|---:|---:|
| Supply Chain Dive | supplychaindive.com | 641 | 9.80% |
| Supply Chain Brain | supplychainbrain.com | 521 | 7.96% |
| Business Insider | businessinsider.com | 425 | 6.50% |
| gCaptain | gcaptain.com | 414 | 6.33% |
| CNBC | cnbc.com | 389 | 5.95% |
| South China Morning Post | scmp.com | 362 | 5.53% |
| Supply Chain Digital | supplychaindigital.com | 298 | 4.56% |
| Container News | container-news.com | 284 | 4.34% |
| ET Infra | infra.economictimes.indiatimes.com | 263 | 4.02% |
| bbc.co.uk | bbc.co.uk | 235 | 3.59% |

## Annual distribution

| Year | Articles | Share |
|---:|---:|---:|
| 2015 | 380 | 5.81% |
| 2016 | 372 | 5.69% |
| 2017 | 393 | 6.01% |
| 2018 | 444 | 6.79% |
| 2019 | 470 | 7.18% |
| 2020 | 634 | 9.69% |
| 2021 | 728 | 11.13% |
| 2022 | 826 | 12.63% |
| 2023 | 729 | 11.14% |
| 2024 | 842 | 12.87% |
| 2025 | 720 | 11.01% |
| 2026 | 1 | 0.02% |

## Data-quality checks

These checks are warnings for review; the script does not delete records.

| Check | Affected rows | Share |
|---|---:|---:|
| total_rows | 6,542 | 100.00% |
| missing_or_empty_full_text | 0 | 0.00% |
| missing_or_empty_title | 0 | 0.00% |
| missing_source_name | 320 | 4.89% |
| missing_source_domain | 0 | 0.00% |
| missing_url | 0 | 0.00% |
| unparseable_or_missing_publication_date | 3 | 0.05% |
| very_short_text_below_250_characters | 0 | 0.00% |
| short_text_below_500_characters | 3 | 0.05% |
| long_text_at_least_25000_characters | 15 | 0.23% |
| extreme_text_at_least_50000_characters | 0 | 0.00% |
| duplicate_article_id_rows | 0 | 0.00% |
| duplicate_nonempty_url_rows | 0 | 0.00% |
| duplicate_nonempty_normalized_text_rows | 0 | 0.00% |
| suspected_topic_tag_search_or_index_pages | 96 | 1.47% |

## Generated files

See `tables/` for CSV statistics and `figures/` for PNG charts.