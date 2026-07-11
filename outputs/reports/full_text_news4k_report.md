# Full-text Retrieval Report: Newspaper4k

Generated at UTC: `2026-07-02T23:59:50.386531+00:00`

## Inputs

- Input file: `C:\Users\DELL\Desktop\venv-python\sc-disrupt\sc-disrupt\data\02_interim\news_cleaned\legacy_news_unique_for_retrieval.csv`
- Input rows loaded: **5,779**
- Max rows argument: `None`
- Resume mode: `True`
- Sleep seconds: `1.0`
- Language: `en`
- NLP enabled: `False`

## Outputs

- Results: `C:\Users\DELL\Desktop\venv-python\sc-disrupt\sc-disrupt\data\02_interim\article_enrichment\full_text_news4k\full_text_news4k_results.csv`
- Failures: `C:\Users\DELL\Desktop\venv-python\sc-disrupt\sc-disrupt\data\02_interim\article_enrichment\full_text_news4k\full_text_news4k_failed.csv`
- Checkpoint: `C:\Users\DELL\Desktop\venv-python\sc-disrupt\sc-disrupt\data\02_interim\article_enrichment\full_text_news4k\full_text_news4k_checkpoint.csv`

## Summary

- Total attempted/written rows: **5,779**
- Successful full-text rows: **2,176**
- Empty-text rows: **17**
- Failed/invalid/skipped rows: **3,586**
- Success rate among written rows: **37.65%**

## Status counts

```json
{
  "success": 2176,
  "failed": 2163,
  "invalid_url": 1372,
  "too_short": 51,
  "empty_text": 17
}
```

## Top source domains

```json
{
  "bbc.co.uk": 1754,
  "huffingtonpost.com": 1488,
  "": 1372,
  "huffpost.com": 825,
  "bbc.com": 316,
  "huffingtonpost.comhttp:": 19,
  "huffingtonpost.comhttps:": 5
}
```

## Notes

- `success` means Newspaper4k returned non-empty article text.
- `empty_text` usually means paywall, blocked page, JavaScript-heavy page, non-article URL, or parser failure.
- Use `full_text_news4k_failed.csv` for later fallback extraction using requests/trafilatura/Playwright/GDELT.