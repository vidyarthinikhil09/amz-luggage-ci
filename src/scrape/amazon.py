from __future__ import annotations

import random
import re
import time
import urllib.parse
from dataclasses import dataclass
from typing import Iterable

from bs4 import BeautifulSoup


AMAZON_IN = "https://www.amazon.in"


@dataclass(frozen=True)
class AmazonSearchQuery:
    brand: str
    keywords: str = "trolley bag luggage"

    def url(self) -> str:
        q = f"{self.brand} {self.keywords}".strip()
        return f"{AMAZON_IN}/s?k={urllib.parse.quote_plus(q)}"


_money_re = re.compile(r"(\d[\d,]*)")


def parse_inr(text: str | None) -> float | None:
    if not text:
        return None
    m = _money_re.search(text.replace("₹", ""))
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


_rating_re = re.compile(r"([0-5](?:\.\d)?)\s*out of\s*5")


def parse_rating(text: str | None) -> float | None:
    if not text:
        return None
    m = _rating_re.search(text)
    if not m:
        return None
    return float(m.group(1))


def parse_int(text: str | None) -> int | None:
    if not text:
        return None
    digits = re.sub(r"[^0-9]", "", text)
    return int(digits) if digits else None


def jitter_sleep(throttle_ms: int) -> None:
    base = max(0, throttle_ms) / 1000.0
    time.sleep(base + random.uniform(0.1, 0.6))


def html_to_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _canonicalize_product_url(href: str) -> str | None:
    if not href:
        return None

    # Sponsored results often use /sspa/click with an encoded `url` query param.
    # Prefer the decoded destination URL so we can reuse it elsewhere.
    if href.startswith("/sspa/click") or "amazon.in/sspa/click" in href:
        try:
            parsed = urllib.parse.urlparse(href)
            qs = urllib.parse.parse_qs(parsed.query)
            dest = (qs.get("url") or [None])[0]
            if dest:
                dest = urllib.parse.unquote(dest)
                if dest.startswith("http"):
                    return dest
                if dest.startswith("/"):
                    return f"{AMAZON_IN}{dest}"
        except Exception:
            pass

    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return f"{AMAZON_IN}{href}"
    return None


def _href_mentions_asin(href: str, asin: str) -> bool:
    if not href:
        return False
    return f"/dp/{asin}" in href or f"%2Fdp%2F{asin}" in href or f"asin={asin}" in href


def _best_title_from_anchors(card: BeautifulSoup, asin: str) -> str | None:
    candidates: list[str] = []

    # Classic desktop markup
    title_el = card.select_one("h2 a span")
    if title_el:
        t = title_el.get_text(" ", strip=True)
        if t:
            candidates.append(t)

    # Newer 'puis' cards often have the title as anchor text with line-clamp classes.
    for a in card.select("a[href]"):
        href = a.get("href")
        if not href or not _href_mentions_asin(href, asin):
            continue
        if a.get("aria-hidden") == "true":
            continue
        t = a.get_text(" ", strip=True)
        if t and len(t) >= 8:
            candidates.append(t)

    if not candidates:
        return None
    # Prefer the longest non-empty text.
    return max(candidates, key=len)


def _extract_review_count(card: BeautifulSoup) -> int | None:
    # Common: <a aria-label="27,982 ratings" ...>
    a = card.select_one("a[aria-label*='rating'], a[aria-label*='ratings']")
    if a:
        return parse_int(a.get("aria-label"))
    # Older fallback
    review_count_el = card.select_one("span.a-size-base.s-underline-text")
    return parse_int(review_count_el.get_text(" ", strip=True) if review_count_el else None)


def extract_asin_cards(soup: BeautifulSoup) -> Iterable[dict]:
    # Amazon search result cards have data-asin; empty values are placeholders.
    # Restrict to real search-result containers to avoid carousels/widgets.
    selector = "div[data-component-type='s-search-result'][data-asin]"
    for card in soup.select(selector):
        asin = (card.get("data-asin") or "").strip()
        if not asin or len(asin) < 8:
            continue

        # URL + title: handle both classic and newer 'puis' cards (incl. sponsored).
        href: str | None = None
        for a in card.select("a[href]"):
            h = a.get("href")
            if h and _href_mentions_asin(h, asin):
                href = h
                break
        product_url = _canonicalize_product_url(href) if href else None
        title = _best_title_from_anchors(card, asin)

        rating_el = card.select_one("span.a-icon-alt")
        rating_avg = parse_rating(rating_el.get_text(" ", strip=True) if rating_el else None)

        review_count = _extract_review_count(card)

        # Price shown on listing is best-effort
        price_el = card.select_one("span.a-price span.a-offscreen")
        price = parse_inr(price_el.get_text(" ", strip=True) if price_el else None)

        list_price = None
        mrp_el = card.select_one("span.a-price.a-text-price span.a-offscreen")
        if mrp_el:
            list_price = parse_inr(mrp_el.get_text(" ", strip=True))

        yield {
            "asin": asin,
            "title": title,
            "product_url": product_url,
            "price": price,
            "list_price": list_price,
            "rating_avg": rating_avg,
            "review_count": review_count,
        }
