from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.utils.text import infer_size_bucket


REQUIRED_PRODUCT_COLS = [
    "asin",
    "brand",
    "title",
    "product_url",
    "price",
    "list_price",
    "discount_pct",
    "rating_avg",
    "review_count",
    "scraped_at",
]

REQUIRED_REVIEW_COLS = [
    "review_id",
    "asin",
    "brand",
    "rating",
    "review_title",
    "review_text",
    "review_date",
    "helpful_votes",
    "verified_purchase",
    "scraped_at",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_product_schema(products: pd.DataFrame) -> pd.DataFrame:
    df = products.copy() if products is not None else pd.DataFrame()
    for col in REQUIRED_PRODUCT_COLS:
        if col not in df.columns:
            df[col] = None

    # type normalization
    for col in ["price", "list_price", "rating_avg", "discount_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").astype("Int64")

    # derived fields
    if "size_bucket" not in df.columns:
        df["size_bucket"] = df["title"].astype(str).map(infer_size_bucket)

    # recompute discount if possible
    mask = df["price"].notna() & df["list_price"].notna() & (df["list_price"] > 0)
    df.loc[mask, "discount_pct"] = (df.loc[mask, "list_price"] - df.loc[mask, "price"]) / df.loc[mask, "list_price"]

    df["scraped_at"] = df["scraped_at"].fillna(_utc_now_iso())
    return df


def ensure_review_schema(reviews: pd.DataFrame) -> pd.DataFrame:
    df = reviews.copy() if reviews is not None else pd.DataFrame()
    for col in REQUIRED_REVIEW_COLS:
        if col not in df.columns:
            df[col] = None

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce").astype("Int64")
    df["verified_purchase"] = df["verified_purchase"].fillna(False).astype(bool)
    df["scraped_at"] = df["scraped_at"].fillna(_utc_now_iso())

    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["review_title"] = df["review_title"].fillna("").astype(str)

    return df


def standardize_brands(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "brand" not in df.columns:
        return df

    mapping = {
        "american tourister": "American Tourister",
        "americantourister": "American Tourister",
        "nasher miles": "Nasher Miles",
        "nashermiles": "Nasher Miles",
        "vip": "VIP",
        "aristocrat": "Aristocrat",
        "safari": "Safari",
        "skybags": "Skybags",
    }

    out = df.copy()
    out["brand"] = out["brand"].astype(str).map(lambda b: mapping.get(b.strip().lower(), b.strip()))
    return out


def attach_product_fields_to_reviews(products: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    if products.empty or reviews.empty:
        return reviews

    cols = ["asin", "title", "price", "list_price", "discount_pct", "rating_avg", "review_count", "size_bucket"]
    p = products[cols].drop_duplicates(subset=["asin"], keep="last")

    out = reviews.merge(p, on="asin", how="left", suffixes=("", "_product"))
    return out


def winsorize(series: pd.Series, lower_q: float = 0.02, upper_q: float = 0.98) -> pd.Series:
    if series.dropna().empty:
        return series
    lo = series.quantile(lower_q)
    hi = series.quantile(upper_q)
    return series.clip(lower=lo, upper=hi)


def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    if s.dropna().empty:
        return s
    mu = float(s.mean())
    sigma = float(s.std(ddof=0))
    if sigma == 0 or math.isnan(sigma):
        return s * 0
    return (s - mu) / sigma
