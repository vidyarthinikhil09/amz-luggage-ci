from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from datetime import datetime, timezone

import pandas as pd

from src.scrape.amazon import AmazonSearchQuery, extract_asin_cards, html_to_soup, jitter_sleep
from src.scrape.playwright_client import BrowserConfig, PlaywrightBrowser
from src.utils.env import settings
from src.utils.io import read_parquet, write_parquet
from src.utils.paths import processed_path, raw_path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def scrape_brand_products(page, brand: str, max_products: int, throttle_ms: int) -> list[dict]:
    url = AmazonSearchQuery(brand=brand).url()
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_timeout(1200)
    jitter_sleep(throttle_ms)

    html = page.content()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    html_path = raw_path("products", brand.replace(" ", "_"), f"search_{stamp}.html")
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    soup = html_to_soup(html)
    rows = []
    for item in extract_asin_cards(soup):
        item["brand"] = brand
        item["scraped_at"] = _utc_now_iso()
        rows.append(item)
        if len(rows) >= max_products:
            break

    return rows


def upsert_products(existing: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    incoming = pd.DataFrame(new_rows)
    if incoming.empty:
        return existing

    if existing is None or existing.empty:
        merged = incoming
    else:
        # Prefer most recent scrape per ASIN
        merged = pd.concat([existing, incoming], ignore_index=True)
        merged = merged.sort_values("scraped_at").drop_duplicates(subset=["asin"], keep="last")

    merged = merged.reset_index(drop=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brands", nargs="+", required=True)
    parser.add_argument("--max-products-per-brand", type=int, default=int(os.getenv("MAX_PRODUCTS_PER_BRAND", "12")))
    parser.add_argument("--headful", action="store_true")
    parser.add_argument("--throttle-ms", type=int, default=settings.throttle_ms)
    args = parser.parse_args()

    existing = read_parquet(processed_path("products.parquet"))

    cfg = BrowserConfig(headless=(not args.headful and settings.headless), user_agent=settings.user_agent)
    all_new: list[dict] = []

    with PlaywrightBrowser(cfg) as br:
        page = br.new_page()
        for brand in args.brands:
            rows = scrape_brand_products(page, brand=brand, max_products=args.max_products_per_brand, throttle_ms=args.throttle_ms)
            all_new.extend(rows)
            jitter_sleep(args.throttle_ms)

    merged = upsert_products(existing, all_new)

    # Compute a lightweight discount now if list_price exists.
    if "discount_pct" not in merged.columns:
        merged["discount_pct"] = None
    mask = merged["price"].notna() & merged["list_price"].notna() & (merged["list_price"] > 0)
    merged.loc[mask, "discount_pct"] = (merged.loc[mask, "list_price"] - merged.loc[mask, "price"]) / merged.loc[mask, "list_price"]

    write_parquet(merged, processed_path("products.parquet"))
    print(f"Saved {merged.shape[0]} products -> data/processed/products.parquet")


if __name__ == "__main__":
    main()
