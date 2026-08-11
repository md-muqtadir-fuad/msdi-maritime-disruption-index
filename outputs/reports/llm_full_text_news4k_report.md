# LLM News Full-text Retrieval Report: Newspaper4k

Generated at UTC: `2026-08-11T06:28:56.302405+00:00`

## Input

- File: `C:\Users\DELL\Desktop\venv-python\sc-disrupt\sc-disrupt\data\02_interim\news_cleaned\llm_search_results_cleaned.csv`
- Rows loaded: **11,121**
- Unique normalized article links: **11,115**
- Duplicate links removed before retrieval: **6**

## Run settings

- Language: `en`
- Minimum words for success: `80`
- Request timeout: `25` seconds
- Base sleep: `0.5` seconds
- Random jitter: `0.5` seconds
- NLP enabled: `False`
- Resume mode: `True`

## Results

- Total attempted rows saved: **11,115**
- Succeeded rows: **7,046**
- Unsuccessful rows: **4,069**
- Success rate: **63.39%**
- Succeeded file: `C:\Users\DELL\Desktop\venv-python\sc-disrupt\sc-disrupt\data\02_interim\article_enrichment\full_text_llm_news4k\llm_full_text_news4k_succeeded.csv`
- Unsuccessful file: `C:\Users\DELL\Desktop\venv-python\sc-disrupt\sc-disrupt\data\02_interim\article_enrichment\full_text_llm_news4k\llm_full_text_news4k_unsucceeded.csv`

### Retrieval status counts

```json
{
  "success": 7046,
  "failed": 2527,
  "empty_text": 920,
  "too_short": 622
}
```

## Status definitions

- `success`: extracted text met the minimum word threshold.
- `too_short`: extracted text was below the threshold.
- `empty_text`: no article body was extracted.
- `failed`: downloading or parsing raised an exception.
- `invalid_url`: the input URL was missing or malformed.