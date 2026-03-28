# Amazon India Luggage Competitive Intelligence Dashboard

Interactive dashboard that scrapes Amazon India listings + reviews for 6 luggage brands and synthesizes pricing, discounting, sentiment, and review themes into decision-ready competitive intelligence.

## Brand set
- Safari
- Skybags
- American Tourister
- VIP
- Aristocrat
- Nasher Miles

## Setup (Windows / PowerShell)
1. Create a virtualenv and install deps:
   - `python -m venv .venv`
   - `.\.venv\Scripts\Activate.ps1`
   - `pip install -r requirements.txt`
2. Install Playwright browsers:
   - `python -m playwright install chromium`
3. Copy env template:
   - `copy .env.example .env`
    - Fill LLM settings if using LLM themes/aspects.
       - OpenRouter: set `OPENROUTER_API_KEY` and (optionally) `OPENROUTER_SITE_URL` + `OPENROUTER_APP_NAME`.
          - Default model in `.env.example` is `nvidia/nemotron-3-super-120b-a12b:free`.
       - Or use any OpenAI-compatible endpoint by setting `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`.
   - Optional: set `MAX_LLM_REVIEWS` to control LLM cost/time (cached results are reused).

## Run the pipeline
1. Scrape products:
   - `python scripts\scrape_products.py --brands Safari Skybags "American Tourister" VIP Aristocrat "Nasher Miles" --max-products-per-brand 12`
2. Scrape reviews:
   - `python scripts\scrape_reviews.py --max-reviews-per-asin 40 --max-reviews-per-brand 120`
3. Clean + analyze:
   - `python scripts\analyze.py`
4. Launch dashboard:
   - `streamlit run app\app.py`

## How to reproduce results
Run these commands in order (PowerShell from the repo root):
1. `python scripts\scrape_products.py --brands Safari Skybags "American Tourister" VIP Aristocrat "Nasher Miles" --max-products-per-brand 12`
2. `python scripts\scrape_reviews.py --max-reviews-per-asin 40 --max-reviews-per-brand 120`
3. `python scripts\analyze.py --max-llm-reviews 300`
4. `streamlit run app\app.py`

## Methodology (high level)
- Pricing: extract price + list price where available; compute discount %.
- Sentiment: hybrid score blending star rating sentiment with text sentiment (VADER).
- Themes: aspect-level extraction (wheels/handle/zipper/material/durability/size/weight/service) via an OpenAI-compatible LLM with strict JSON outputs and caching for reproducibility.
- Competitive intelligence: value-for-money index, discount reliance, anomaly flags, and light trust signals.

## Architecture
This project runs in two modes:
- Mode A (local pipeline): scrape + analyze to produce/update `data/processed/*`.
- Mode B (deployed app): Streamlit reads `data/processed/*` (no live Amazon scraping), so demos are stable.

![Architecture diagram](docs/Architecture_Diag.png)

## Output datasets
- Primary names are `*.parquet`, but on environments without a Parquet engine (e.g. Python 3.13 without `pyarrow` wheels), the pipeline will automatically write `*.csv` with the same base name.
- `data/processed/products.parquet` (or `products.csv`)
- `data/processed/reviews.parquet` (or `reviews.csv`)
- `data/processed/product_metrics.parquet` (or `product_metrics.csv`)
- `data/processed/brand_metrics.parquet` (or `brand_metrics.csv`)
- `data/processed/agent_insights.parquet` (or `agent_insights.csv`)

## Limitations
- Amazon pages can trigger bot protections/captchas; scraping is throttled and will fail gracefully with partial outputs.
- Some listings do not expose list price/MRP.
- Category/size inference is heuristic (from titles).
