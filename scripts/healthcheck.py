from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from src.utils.env import settings
from src.utils.io import read_parquet
from src.utils.paths import processed_path


REQUIRED_PRODUCTS_COLS = [
    "asin",
    "brand",
    "title",
    "product_url",
    "price",
    "list_price",
    "discount_pct",
    "rating_avg",
    "review_count",
]

REQUIRED_REVIEWS_COLS = [
    "asin",
    "brand",
    "review_text",
    "rating",
    "scraped_at",
    "sentiment_score",
    "sentiment_label",
]


def _exists_parquet_or_csv(base_parquet_path: str) -> bool:
    if os.path.exists(base_parquet_path):
        return True
    csv_path = os.path.splitext(base_parquet_path)[0] + ".csv"
    return os.path.exists(csv_path)


def _check_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [c for c in required if c not in df.columns]


def _print_kv(k: str, v: str) -> None:
    print(f"- {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end project healthcheck")
    parser.add_argument("--min-products-per-brand", type=int, default=10)
    parser.add_argument("--min-reviews-per-brand", type=int, default=50)
    args = parser.parse_args()

    print("Healthcheck: Amazon Luggage CI Dashboard")

    print("\n**Config**")
    _print_kv("LLM base URL", settings.openai_base_url)
    _print_kv("LLM model", settings.openai_model)
    _print_kv("LLM key present", str(bool(settings.openai_api_key)))

    required_outputs = [
        "products.parquet",
        "reviews.parquet",
        "brand_metrics.parquet",
        "product_metrics.parquet",
        "review_aspects.parquet",
        "agent_insights.parquet",
    ]

    print("\n**Files**")
    missing_files: list[str] = []
    for name in required_outputs:
        p = processed_path(name)
        ok = _exists_parquet_or_csv(p)
        _print_kv(name, "OK" if ok else "MISSING")
        if not ok:
            missing_files.append(name)

    if missing_files:
        print("\nFAIL: missing processed outputs. Run:")
        print(r"  .\\.venv\\Scripts\\python scripts\\analyze.py --skip-llm")
        raise SystemExit(2)

    products = read_parquet(processed_path("products.parquet"))
    reviews = read_parquet(processed_path("reviews.parquet"))
    brand_metrics = read_parquet(processed_path("brand_metrics.parquet"))
    product_metrics = read_parquet(processed_path("product_metrics.parquet"))
    review_aspects = read_parquet(processed_path("review_aspects.parquet"))
    agent_insights = read_parquet(processed_path("agent_insights.parquet"))

    print("\n**Dataset sizes**")
    _print_kv("products rows", str(len(products)))
    _print_kv("reviews rows", str(len(reviews)))
    _print_kv("brand_metrics rows", str(len(brand_metrics)))
    _print_kv("product_metrics rows", str(len(product_metrics)))
    _print_kv("review_aspects rows", str(len(review_aspects)))
    _print_kv("agent_insights rows", str(len(agent_insights)))

    print("\n**Schema checks**")
    prod_missing = _check_columns(products, REQUIRED_PRODUCTS_COLS)
    rev_missing = _check_columns(reviews, REQUIRED_REVIEWS_COLS)
    _print_kv("products required columns", "OK" if not prod_missing else f"MISSING: {prod_missing}")
    _print_kv("reviews required columns", "OK" if not rev_missing else f"MISSING: {rev_missing}")

    if prod_missing or rev_missing:
        print("\nFAIL: schema mismatch. Re-run:")
        print(r"  .\\.venv\\Scripts\\python scripts\\analyze.py --skip-llm")
        raise SystemExit(3)

    print("\n**Coverage checks**")
    if products.empty or reviews.empty:
        print("FAIL: products/reviews are empty. Re-run scraping:")
        print(r"  .\\.venv\\Scripts\\python scripts\\pipeline.py")
        raise SystemExit(4)

    products_per_brand = products.groupby("brand")["asin"].nunique().sort_values(ascending=False)
    reviews_per_brand = reviews.groupby("brand")["review_text"].count().sort_values(ascending=False)

    _print_kv("brands (products)", str(int(products["brand"].nunique())))
    _print_kv("brands (reviews)", str(int(reviews["brand"].nunique())))
    _print_kv("min products/brand", str(int(products_per_brand.min())))
    _print_kv("min reviews/brand", str(int(reviews_per_brand.min())))

    low_products = products_per_brand[products_per_brand < args.min_products_per_brand]
    low_reviews = reviews_per_brand[reviews_per_brand < args.min_reviews_per_brand]

    if not low_products.empty:
        print("\nFAIL: some brands below min products/brand")
        print(low_products.to_string())
        raise SystemExit(5)

    if not low_reviews.empty:
        print("\nFAIL: some brands below min reviews/brand")
        print(low_reviews.to_string())
        raise SystemExit(6)

    print("\n**LLM outputs**")
    if settings.openai_api_key:
        _print_kv("aspects populated", "YES" if not review_aspects.empty else "NO (LLM returned none)")
        _print_kv("agent insights populated", "YES" if not agent_insights.empty else "NO (LLM returned none)")
    else:
        _print_kv("aspects populated", "SKIPPED (no API key)")
        _print_kv("agent insights populated", "SKIPPED (no API key)")

    print("\nPASS: pipeline outputs look healthy.")
    print("Next: run the dashboard:")
    print(r"  .\\.venv\\Scripts\\streamlit run app\\app.py")


if __name__ == "__main__":
    main()
