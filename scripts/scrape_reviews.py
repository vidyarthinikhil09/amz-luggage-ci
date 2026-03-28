from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.scrape.amazon import AMAZON_IN, html_to_soup, jitter_sleep, parse_int
from src.scrape.playwright_client import BrowserConfig, PlaywrightBrowser
from src.utils.env import settings
from src.utils.io import read_parquet, write_parquet
from src.utils.paths import processed_path, raw_path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _review_page_url(asin: str, page_number: int) -> str:
    return f"{AMAZON_IN}/product-reviews/{asin}/?pageNumber={page_number}&reviewerType=all_reviews"


def _product_page_url(asin: str) -> str:
    return f"{AMAZON_IN}/dp/{asin}"


_rating_re = re.compile(r"([0-5](?:\.\d)?)\s*out of\s*5")


def _parse_review_rating(text: str | None) -> float | None:
    if not text:
        return None
    m = _rating_re.search(text)
    return float(m.group(1)) if m else None


def _extract_reviews(soup: BeautifulSoup, asin: str, brand: str) -> list[dict]:
    rows: list[dict] = []

    # Review blocks appear as div[data-hook='review'] on /product-reviews pages
    # and as li[data-hook='review'] on product detail pages.
    for block in soup.select("div[data-hook='review'], li[data-hook='review']"):
        review_id = block.get("id")

        title_el = block.select_one("a[data-hook='review-title'] span")
        review_title = title_el.get_text(" ", strip=True) if title_el else None

        body_el = block.select_one("span[data-hook='review-body']")
        review_text = body_el.get_text(" ", strip=True) if body_el else None

        rating_el = block.select_one("i[data-hook='review-star-rating'] span.a-icon-alt")
        if not rating_el:
            rating_el = block.select_one("i[data-hook='cmps-review-star-rating'] span.a-icon-alt")
        rating = _parse_review_rating(rating_el.get_text(" ", strip=True) if rating_el else None)

        date_el = block.select_one("span[data-hook='review-date']")
        review_date = date_el.get_text(" ", strip=True) if date_el else None

        verified_el = block.select_one("span[data-hook='avp-badge']")
        verified_purchase = bool(verified_el)

        helpful_el = block.select_one("span[data-hook='helpful-vote-statement']")
        helpful_votes = parse_int(helpful_el.get_text(" ", strip=True) if helpful_el else None)

        rows.append(
            {
                "review_id": review_id,
                "asin": asin,
                "brand": brand,
                "rating": rating,
                "review_title": review_title,
                "review_text": review_text,
                "review_date": review_date,
                "verified_purchase": verified_purchase,
                "helpful_votes": helpful_votes,
                "scraped_at": _utc_now_iso(),
            }
        )

    return rows


def _looks_like_blocked(html: str) -> bool:
    h = html.lower()
    # crude but effective signals
    return any(
        s in h
        for s in [
            "enter the characters you see below",
            "type the characters you see",
            "robot check",
            "captcha",
            "sorry, we just need to make sure you're not a robot",
            # Unified auth / forced sign-in pages (common when hitting /product-reviews)
            "ap_login_form",
            "name=\"signIn\"",
            "action=\"/ax/claim",
            # Common bot-block copy
            "to discuss automated access to amazon data",
        ]
    )


def _fallback_review_key(row: dict) -> str:
    # Used when review_id is missing
    t = (row.get("review_title") or "").strip().lower()
    d = (row.get("review_date") or "").strip().lower()
    x = (row.get("review_text") or "").strip().lower()
    x = re.sub(r"\s+", " ", x)
    return f"{row.get('asin')}|{d}|{t}|{x[:200]}"


def _dedupe_reviews(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["_fallback_key"] = df.apply(lambda r: _fallback_review_key(r.to_dict()), axis=1)

    if "review_id" in df.columns:
        has_id = df["review_id"].notna() & (df["review_id"].astype(str).str.len() > 0)
        df_id = df[has_id].drop_duplicates(subset=["review_id"], keep="last")
        df_no = df[~has_id].drop_duplicates(subset=["_fallback_key"], keep="last")
        out = pd.concat([df_id, df_no], ignore_index=True)
    else:
        out = df.drop_duplicates(subset=["_fallback_key"], keep="last")

    out = out.drop(columns=["_fallback_key"]).reset_index(drop=True)
    return out


def _ps_escape_single_quotes(s: str) -> str:
    return s.replace("'", "''")


def _fetch_html_product_page(url: str, user_agent: str) -> str:
    """Fetch HTML for product pages.

    Notes:
    - In some environments, python-requests gets served captcha/partial content.
    - On Windows, PowerShell's Invoke-WebRequest often returns the full HTML.
    """
    headers = {
        "User-Agent": user_agent,
        "Accept-Language": "en-IN,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        html = r.text
        # If this looks usable, keep it.
        if len(html) > 200_000 and "customerreviews" in html.lower():
            return html
        # Captcha/blocks often return small HTML.
        if not _looks_like_blocked(html) and len(html) > 200_000:
            return html
    except Exception:
        html = ""

    # Windows fallback: use Invoke-WebRequest which tends to get the full page.
    if sys.platform.startswith("win"):
        ua = _ps_escape_single_quotes(user_agent)
        u = _ps_escape_single_quotes(url)
        ps = (
            "$ProgressPreference='SilentlyContinue'; "
            "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
            f"$headers=@{{'User-Agent'='{ua}';'Accept-Language'='en-IN,en;q=0.9'}}; "
            f"$resp=Invoke-WebRequest -Uri '{u}' -Headers $headers -UseBasicParsing -MaximumRedirection 5 -ErrorAction Stop; "
            "$resp.Content"
        )
        try:
            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    ps,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=True,
            )
            return out.stdout
        except Exception:
            pass

    return html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-reviews-per-asin", type=int, default=int(os.getenv("MAX_REVIEWS_PER_ASIN", "40")))
    parser.add_argument("--max-reviews-per-brand", type=int, default=int(os.getenv("MAX_REVIEWS_PER_BRAND", "120")))
    parser.add_argument(
        "--source",
        choices=["product_page", "reviews_page"],
        default=os.getenv("REVIEW_SOURCE", "product_page"),
        help="Where to scrape reviews from. 'product_page' scrapes embedded reviews on /dp/{asin} (no pagination). 'reviews_page' uses /product-reviews/{asin} (may force sign-in).",
    )
    parser.add_argument("--headful", action="store_true")
    parser.add_argument("--throttle-ms", type=int, default=settings.throttle_ms)
    args = parser.parse_args()

    products = read_parquet(processed_path("products.parquet"))
    if products.empty:
        raise SystemExit("No products found. Run scripts/scrape_products.py first.")

    existing = read_parquet(processed_path("reviews.parquet"))

    cfg = BrowserConfig(headless=(not args.headful and settings.headless), user_agent=settings.user_agent)

    all_rows: list[dict] = []

    if args.source == "product_page":
        for brand in sorted(products["brand"].unique().tolist()):
            brand_asins = products[products["brand"] == brand]["asin"].dropna().unique().tolist()

            brand_count = 0
            for asin in brand_asins:
                if brand_count >= args.max_reviews_per_brand:
                    break

                jitter_sleep(args.throttle_ms)
                url = _product_page_url(asin)
                html = _fetch_html_product_page(url, user_agent=cfg.user_agent)
                if not html:
                    print(f"Failed to fetch {asin} product page: empty response")
                    continue

                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                if _looks_like_blocked(html):
                    blocked_path = raw_path("reviews", brand.replace(" ", "_"), asin, f"BLOCKED_product_{stamp}.html")
                    os.makedirs(os.path.dirname(blocked_path), exist_ok=True)
                    with open(blocked_path, "w", encoding="utf-8") as f:
                        f.write(html)
                    print(f"Blocked/captcha encountered for {asin} (product page). Saved {blocked_path}.")
                    continue

                html_path = raw_path("reviews", brand.replace(" ", "_"), asin, f"product_{stamp}.html")
                os.makedirs(os.path.dirname(html_path), exist_ok=True)
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html)

                soup = html_to_soup(html)
                per_asin_rows = _extract_reviews(soup, asin=asin, brand=brand)
                per_asin_rows = per_asin_rows[: args.max_reviews_per_asin]

                if per_asin_rows:
                    remaining = max(0, args.max_reviews_per_brand - brand_count)
                    per_asin_rows = per_asin_rows[:remaining]

                all_rows.extend(per_asin_rows)
                brand_count += len(per_asin_rows)
    else:
        with PlaywrightBrowser(cfg) as br:
            page = br.new_page()

            for brand in sorted(products["brand"].unique().tolist()):
                brand_asins = products[products["brand"] == brand]["asin"].dropna().unique().tolist()

                brand_count = 0
                for asin in brand_asins:
                    if brand_count >= args.max_reviews_per_brand:
                        break

                    per_asin_rows: list[dict] = []

                    # iterate pages until we hit the limit or no results
                    page_num = 1
                    while len(per_asin_rows) < args.max_reviews_per_asin:
                        url = _review_page_url(asin, page_num)
                        page.goto(url, wait_until="domcontentloaded")
                        page.wait_for_timeout(1200)
                        jitter_sleep(args.throttle_ms)

                        html = page.content()
                        if _looks_like_blocked(html):
                            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                            blocked_path = raw_path("reviews", brand.replace(" ", "_"), asin, f"BLOCKED_{stamp}.html")
                            os.makedirs(os.path.dirname(blocked_path), exist_ok=True)
                            with open(blocked_path, "w", encoding="utf-8") as f:
                                f.write(html)
                            print(f"Blocked/captcha encountered for {asin}. Saved {blocked_path}.")
                            break

                        # Save raw html for first 2 pages per ASIN
                        if page_num <= 2:
                            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                            html_path = raw_path("reviews", brand.replace(" ", "_"), asin, f"page_{page_num}_{stamp}.html")
                            os.makedirs(os.path.dirname(html_path), exist_ok=True)
                            with open(html_path, "w", encoding="utf-8") as f:
                                f.write(html)

                        soup = html_to_soup(html)
                        rows = _extract_reviews(soup, asin=asin, brand=brand)
                        if not rows:
                            break

                        per_asin_rows.extend(rows)

                        page_num += 1
                        jitter_sleep(args.throttle_ms)

                    if per_asin_rows:
                        # Respect brand cap
                        remaining = max(0, args.max_reviews_per_brand - brand_count)
                        per_asin_rows = per_asin_rows[:remaining]

                    all_rows.extend(per_asin_rows)
                    brand_count += len(per_asin_rows)

    incoming = pd.DataFrame(all_rows)
    merged = pd.concat([existing, incoming], ignore_index=True) if not existing.empty else incoming
    merged = _dedupe_reviews(merged)

    write_parquet(merged, processed_path("reviews.parquet"))
    print(f"Saved {merged.shape[0]} reviews -> data/processed/reviews.parquet")


if __name__ == "__main__":
    main()
