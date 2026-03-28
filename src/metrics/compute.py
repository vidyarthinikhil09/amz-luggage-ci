from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.cleaning import zscore
from src.utils.text import normalize_review_text


@dataclass(frozen=True)
class MetricConfig:
    vfm_alpha: float = 0.35


def compute_trust_signals(reviews: pd.DataFrame) -> pd.DataFrame:
    if reviews.empty:
        return pd.DataFrame(columns=["brand", "dup_exact_rate", "short_review_rate"]) 

    df = reviews.copy()
    df["_norm_text"] = df["review_text"].astype(str).map(normalize_review_text)

    by_brand = []
    for brand, g in df.groupby("brand"):
        n = len(g)
        if n == 0:
            continue

        exact_dups = int(g.duplicated(subset=["_norm_text"]).sum())
        dup_exact_rate = exact_dups / max(1, n)

        short_rate = float((g["_norm_text"].str.len() < 25).mean())

        by_brand.append({"brand": brand, "dup_exact_rate": dup_exact_rate, "short_review_rate": short_rate})

    return pd.DataFrame(by_brand)


def compute_brand_metrics(products: pd.DataFrame, reviews: pd.DataFrame, cfg: MetricConfig = MetricConfig()) -> pd.DataFrame:
    if products.empty:
        return pd.DataFrame()

    # product-level aggregates by brand
    p = products.copy()

    agg_products = (
        p.groupby("brand")
        .agg(
            avg_price=("price", "mean"),
            avg_list_price=("list_price", "mean"),
            avg_discount_pct=("discount_pct", "mean"),
            avg_rating=("rating_avg", "mean"),
            products=("asin", "nunique"),
        )
        .reset_index()
    )

    if reviews.empty or "sentiment_score" not in reviews.columns:
        agg_reviews = pd.DataFrame({"brand": agg_products["brand"], "reviews": 0, "sentiment_mean": np.nan, "sentiment_median": np.nan})
    else:
        agg_reviews = (
            reviews.groupby("brand")
            .agg(
                reviews=("asin", "count"),
                sentiment_mean=("sentiment_score", "mean"),
                sentiment_median=("sentiment_score", "median"),
                avg_review_rating=("rating", "mean"),
            )
            .reset_index()
        )

    out = agg_products.merge(agg_reviews, on="brand", how="left")

    # Indices
    out["price_z"] = zscore(out["avg_price"].fillna(out["avg_price"].median()))
    out["discount_z"] = zscore(out["avg_discount_pct"].fillna(0))

    out["vfm_index"] = out["sentiment_mean"].fillna(0) - cfg.vfm_alpha * out["price_z"].fillna(0)
    out["discount_reliance"] = 0.6 * out["discount_z"].fillna(0) + 0.4 * out["price_z"].fillna(0)

    # Flags
    out["premium_positioned"] = out["price_z"] > 0.75
    out["high_discounting"] = out["avg_discount_pct"].fillna(0) > 0.25

    return out


def compute_product_metrics(products: pd.DataFrame, reviews: pd.DataFrame, review_aspects: pd.DataFrame | None = None) -> pd.DataFrame:
    if products.empty:
        return pd.DataFrame()

    p = products.copy()

    # review aggregates at product level
    if reviews.empty:
        r_agg = pd.DataFrame(columns=["asin", "reviews", "sentiment_mean", "sentiment_median", "sentiment_neg_rate", "avg_review_rating"]) 
    else:
        r = reviews.copy()
        if "sentiment_score" not in r.columns:
            r["sentiment_score"] = np.nan
        if "sentiment_label" not in r.columns:
            r["sentiment_label"] = None

        r_agg = (
            r.groupby(["asin", "brand"])
            .agg(
                reviews=("review_text", "count"),
                sentiment_mean=("sentiment_score", "mean"),
                sentiment_median=("sentiment_score", "median"),
                sentiment_neg_rate=("sentiment_label", lambda s: float((s == "negative").mean())),
                avg_review_rating=("rating", "mean"),
            )
            .reset_index()
        )

    out = p.merge(r_agg, on=["asin", "brand"], how="left")

    # aspect theme counts (simple and explainable)
    if review_aspects is not None and not review_aspects.empty:
        a = review_aspects.copy()
        a = a[a["aspect"].notna()]

        pivot = (
            a.pivot_table(index=["asin", "brand"], columns=["aspect", "polarity"], values="review_id", aggfunc="count", fill_value=0)
            .reset_index()
        )

        # flatten columns
        pivot.columns = [
            ("_".join([c for c in col if c]) if isinstance(col, tuple) else col)
            for col in pivot.columns
        ]

        out = out.merge(pivot, on=["asin", "brand"], how="left")

        # durable complaint rate proxy
        dur_col = "durability_negative"
        if dur_col in out.columns:
            out["durability_complaints"] = out[dur_col].fillna(0)
        else:
            out["durability_complaints"] = 0

    return out


def top_themes(review_aspects: pd.DataFrame, *, group_cols: list[str], n: int = 5) -> pd.DataFrame:
    if review_aspects is None or review_aspects.empty:
        return pd.DataFrame(columns=group_cols + ["top_pros", "top_cons"]) 

    a = review_aspects.copy()
    a = a[a["aspect"].notna() & a["polarity"].notna()]

    # Count per aspect + polarity
    counts = (
        a.groupby(group_cols + ["aspect", "polarity"]).size().reset_index(name="count")
    )

    out_rows: list[dict] = []
    for keys, g in counts.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = dict(zip(group_cols, keys))

        pros = g[g["polarity"] == "positive"].sort_values("count", ascending=False).head(n)
        cons = g[g["polarity"] == "negative"].sort_values("count", ascending=False).head(n)

        base["top_pros"] = ", ".join([f"{r.aspect} ({int(r['count'])})" for _, r in pros.iterrows()])
        base["top_cons"] = ", ".join([f"{r.aspect} ({int(r['count'])})" for _, r in cons.iterrows()])

        out_rows.append(base)

    return pd.DataFrame(out_rows)
