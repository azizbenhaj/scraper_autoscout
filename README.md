# scraper_autoscout

`autoscout_scraper.py` reads AutoScout24 `.de` / `.ch` listing JSON from embedded `__NEXT_DATA__` (search cards in catalog mode; optional listing pages for doors/seats). Obey ToS, robots, and polite rate limits—fast concurrency risks 403/429.

**Full DE scrape (English), catalog + checkpoints + detail merge to CSV:**

```bash
python3 autoscout_scraper.py --country de --language en --catalog-all --concurrency 4 --delay-min 1 --delay-max 3 --checkpoint-every 20000 --out de_full_en.csv
```

Checkpoints: `de_full_en_catalog_snippet.csv` beside `--out`. Restarts merge existing `--out` unless `--fresh-out`. By default catalog raises `pricefrom` from max price in loaded rows; add `--no-catalog-floor-from-input-max-price` to crawl from EUR 0. Catalog-only: `--no-details`; detail-only on a saved CSV: `--csv-only --resume-from-csv FILE` (no `--no-details`).
