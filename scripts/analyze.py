from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.cleaning import (
    attach_product_fields_to_reviews,
    ensure_product_schema,
    ensure_review_schema,
    standardize_brands,
)
from src.metrics.compute import MetricConfig, compute_brand_metrics, compute_product_metrics, compute_trust_signals, top_themes
from src.nlp.agent_insights import generate_agent_insights
from src.nlp.aspects import build_review_aspects
from src.nlp.sentiment import add_sentiment_columns
from src.utils.env import settings
from src.utils.io import read_parquet, write_parquet
from src.utils.paths import processed_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM-powered aspect extraction and agent insights (fast smoke-test).",
    )
    parser.add_argument(
        "--max-llm-reviews",
        type=int,
        default=None,
        help="Override MAX_LLM_REVIEWS for this run (caps number of uncached reviews sent to the LLM).",
    )
    args = parser.parse_args()

    if args.max_llm_reviews is not None:
        os.environ["MAX_LLM_REVIEWS"] = str(int(args.max_llm_reviews))

    products = read_parquet(processed_path("products.parquet"))
    reviews = read_parquet(processed_path("reviews.parquet"))

    products = ensure_product_schema(products)
    reviews = ensure_review_schema(reviews)

    products = standardize_brands(products)
    reviews = standardize_brands(reviews)

    # Attach product fields (price, size bucket) to reviews for filtering.
    reviews = attach_product_fields_to_reviews(products, reviews)

    # Sentiment
    reviews = add_sentiment_columns(reviews)

    # Aspect extraction (LLM cached). Slowest step when using hosted LLMs.
    prompt_path = os.path.join("docs", "prompts", "aspects_v1.md")
    if args.skip_llm or not settings.openai_api_key:
        review_aspects = build_review_aspects(reviews.iloc[0:0], prompt_path=prompt_path)
    else:
        review_aspects = build_review_aspects(reviews, prompt_path=prompt_path)

    # Metrics
    brand_metrics = compute_brand_metrics(products, reviews, cfg=MetricConfig())
    product_metrics = compute_product_metrics(products, reviews, review_aspects=review_aspects)

    # Themes
    brand_themes = top_themes(review_aspects, group_cols=["brand"], n=5)
    product_themes = top_themes(review_aspects, group_cols=["brand", "asin"], n=5)

    if not brand_themes.empty:
        brand_metrics = brand_metrics.merge(brand_themes, on="brand", how="left")

    if not product_themes.empty:
        product_metrics = product_metrics.merge(product_themes, on=["brand", "asin"], how="left")

    # Trust signals (brand-level)
    trust = compute_trust_signals(reviews)
    if not trust.empty:
        brand_metrics = brand_metrics.merge(trust, on="brand", how="left")

    # Agent insights (LLM; optional). Heuristic fallback always produces 5.
    insights_prompt = os.path.join("docs", "prompts", "agent_insights_v1.md")
    agent_insights = generate_agent_insights(
        brand_metrics,
        prompt_path=insights_prompt,
        use_llm=(not args.skip_llm and bool(settings.openai_api_key)),
    )

    # Persist
    write_parquet(products, processed_path("products.parquet"))
    write_parquet(reviews, processed_path("reviews.parquet"))
    write_parquet(review_aspects, processed_path("review_aspects.parquet"))
    write_parquet(product_metrics, processed_path("product_metrics.parquet"))
    write_parquet(brand_metrics, processed_path("brand_metrics.parquet"))
    write_parquet(agent_insights, processed_path("agent_insights.parquet"))

    print("Wrote:")
    for f in [
        "products.parquet",
        "reviews.parquet",
        "review_aspects.parquet",
        "product_metrics.parquet",
        "brand_metrics.parquet",
        "agent_insights.parquet",
    ]:
        print(" -", os.path.join("data", "processed", f))


if __name__ == "__main__":
    main()
