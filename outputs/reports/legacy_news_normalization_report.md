# Legacy News Normalization Report

## Summary
- Total normalized rows: **10,784**
- Unique headline-date rows: **5,779**
- Rows with usable URLs: **4,443**
- Unique rows with usable URLs for full-text retrieval: **4,407**
- Duplicate rows captured: **6,656**
- Rows matched with legacy LLM score: **6,676**
- Date range: **2015-01-01** to **2031-12-19**

## Missing/quality checks
| issue                   |   count |      share |
|:------------------------|--------:|-----------:|
| missing_headline        |       0 | 0          |
| missing_date            |     400 | 0.037092   |
| missing_url             |    6341 | 0.588001   |
| duplicate_headline      |    6656 | 0.617211   |
| duplicate_headline_date |    6384 | 0.591988   |
| missing_legacy_score    |    4108 | 0.380935   |
| suspicious_future_date  |      13 | 0.00120549 |

## Source summary
| source_type              | source_file                                                                                                                        |   rows |   unique_headline_date |   rows_with_url |   missing_date |   suspicious_future_date |   loaded_rows_before_cleaning |
|:-------------------------|:-----------------------------------------------------------------------------------------------------------------------------------|-------:|-----------------------:|----------------:|---------------:|-------------------------:|------------------------------:|
| main_headline_with_links | C:\Users\DELL\Desktop\venv-python\sc-disrupt\sc-disrupt\data\01_raw\news\headlines\final_dataset_with_links_legacy.xlsx            |   5586 |                   5548 |            4443 |              0 |                        0 |                          5586 |
| manual_collected         | C:\Users\DELL\Desktop\venv-python\sc-disrupt\sc-disrupt\data\01_raw\news\headlines\manually_collected\datasetasif2_raw.xlsx        |   2242 |                   1339 |               0 |            200 |                        1 |                          2242 |
| manual_collected         | C:\Users\DELL\Desktop\venv-python\sc-disrupt\sc-disrupt\data\01_raw\news\headlines\manually_collected\datasetasif3_cleaned_v5.xlsx |   1143 |                   1141 |               0 |              0 |                        0 |                          1143 |
| manual_collected         | C:\Users\DELL\Desktop\venv-python\sc-disrupt\sc-disrupt\data\01_raw\news\headlines\manually_collected\datasetasif3_raw.xlsx        |   1143 |                   1141 |               0 |              1 |                       12 |                          1143 |
| manual_collected         | C:\Users\DELL\Desktop\venv-python\sc-disrupt\sc-disrupt\data\01_raw\news\headlines\manually_collected\datasetasif_raw.xlsx         |    670 |                    668 |               0 |            199 |                        0 |                           670 |

## Output files
- `data/02_interim/news_cleaned/legacy_news_normalized.csv`
- `data/02_interim/news_cleaned/legacy_news_unique_for_retrieval.csv`
- `data/02_interim/news_cleaned/legacy_news_duplicates.csv`
- `data/02_interim/news_cleaned/legacy_news_missing_report.csv`
- `data/02_interim/news_cleaned/legacy_news_source_summary.csv`

## Next step
Use `legacy_news_unique_for_retrieval.csv` as the input for full-text retrieval.