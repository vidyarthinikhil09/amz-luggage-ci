from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Amazon Luggage CI Dashboard", layout="wide")

st.title("Amazon India Luggage Competitive Intelligence")

st.caption(
    "Pipeline: `python scripts\\scrape_products.py` → `python scripts\\scrape_reviews.py` → `python scripts\\analyze.py`"
)

st.caption(
    "Decision-ready view of price positioning, discount reliance, sentiment, and review themes across brands."
)


DATA_DIR = os.path.join("data", "processed")


@st.cache_data(show_spinner=False)
def _read_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        # fall back to csv
        csv_path = os.path.splitext(path)[0] + ".csv"
        return pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        csv_path = os.path.splitext(path)[0] + ".csv"
        return pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()


def _load_data() -> dict[str, pd.DataFrame]:
    return {
        "products": _read_parquet(os.path.join(DATA_DIR, "products.parquet")),
        "reviews": _read_parquet(os.path.join(DATA_DIR, "reviews.parquet")),
        "review_aspects": _read_parquet(os.path.join(DATA_DIR, "review_aspects.parquet")),
        "brand_metrics": _read_parquet(os.path.join(DATA_DIR, "brand_metrics.parquet")),
        "product_metrics": _read_parquet(os.path.join(DATA_DIR, "product_metrics.parquet")),
        "agent_insights": _read_parquet(os.path.join(DATA_DIR, "agent_insights.parquet")),
    }


data = _load_data()
products = data["products"]
reviews = data["reviews"]
review_aspects = data["review_aspects"]
agent_insights = data["agent_insights"]

if products.empty or reviews.empty:
    st.info(
        "No data found yet. Run the pipeline to generate `data/processed/*.parquet` and refresh this page."
    )


def _sidebar_filters(products_df: pd.DataFrame, reviews_df: pd.DataFrame) -> dict:
    st.sidebar.header("Filters")

    brands = sorted(products_df["brand"].dropna().unique().tolist()) if not products_df.empty else []
    selected_brands = st.sidebar.multiselect("Brands", options=brands, default=brands)

    prices = products_df["price"].dropna() if (not products_df.empty and "price" in products_df.columns) else pd.Series([], dtype=float)
    if not prices.empty:
        min_price = float(prices.min())
        max_price = float(prices.max())
        price_range = st.sidebar.slider("Price range (INR)", min_value=min_price, max_value=max_price, value=(min_price, max_price))
    else:
        price_range = (0.0, 1.0)

    min_rating = st.sidebar.slider("Minimum rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1)

    size_opts = ["all", "cabin", "medium", "large", "unknown"]
    size_bucket = st.sidebar.selectbox("Luggage size", options=size_opts, index=0)

    sent_opts = ["all", "positive", "neutral", "negative"]
    sentiment = st.sidebar.selectbox("Sentiment", options=sent_opts, index=0)

    return {
        "brands": selected_brands,
        "price_range": price_range,
        "min_rating": min_rating,
        "size_bucket": size_bucket,
        "sentiment": sentiment,
    }


filters = _sidebar_filters(products, reviews)


def _apply_filters(products_df: pd.DataFrame, reviews_df: pd.DataFrame, review_aspects_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = products_df.copy() if products_df is not None else pd.DataFrame()
    r = reviews_df.copy() if reviews_df is not None else pd.DataFrame()
    a = review_aspects_df.copy() if review_aspects_df is not None else pd.DataFrame()

    if not p.empty and filters["brands"]:
        p = p[p["brand"].isin(filters["brands"])]

    if not p.empty and "price" in p.columns:
        lo, hi = filters["price_range"]
        p = p[p["price"].fillna(-1).between(lo, hi)]

    if not p.empty and "rating_avg" in p.columns:
        p = p[(p["rating_avg"].fillna(0) >= filters["min_rating"]) | p["rating_avg"].isna()]

    if not p.empty and filters["size_bucket"] != "all" and "size_bucket" in p.columns:
        if filters["size_bucket"] == "unknown":
            p = p[p["size_bucket"].isna()]
        else:
            p = p[p["size_bucket"] == filters["size_bucket"]]

    if not r.empty and not p.empty:
        r = r[r["asin"].isin(p["asin"].dropna().unique())]

    if not r.empty and filters["sentiment"] != "all" and "sentiment_label" in r.columns:
        r = r[r["sentiment_label"] == filters["sentiment"]]

    if not a.empty and not r.empty and "review_id" in a.columns and "review_id" in r.columns:
        a = a[a["review_id"].isin(r["review_id"].dropna().unique())]

    return p, r, a


products_f, reviews_f, review_aspects_f = _apply_filters(products, reviews, review_aspects)


def _kpis(products_df: pd.DataFrame, reviews_df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Brands", int(products_df["brand"].nunique()) if not products_df.empty else 0)
    c2.metric("Products", int(products_df["asin"].nunique()) if not products_df.empty else 0)
    c3.metric("Reviews", int(reviews_df.shape[0]) if not reviews_df.empty else 0)
    if not reviews_df.empty and "sentiment_score" in reviews_df.columns:
        c4.metric("Avg sentiment", float(reviews_df["sentiment_score"].mean()))
    else:
        c4.metric("Avg sentiment", 0.0)


def _leaders_laggards(df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """Return leader/laggard brand names for key metrics.

    Keeps the UI decision-ready without introducing new filters or pages.
    """
    if df is None or df.empty:
        return {}, {}

    metrics: list[tuple[str, bool]] = [
        ("vfm_index", False),
        ("sentiment_mean", False),
        ("avg_rating", False),
        ("reviews", False),
        ("avg_discount_pct", False),
        ("avg_price", False),
    ]

    leaders: dict[str, str] = {}
    laggards: dict[str, str] = {}
    for col, ascending in metrics:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.dropna().empty:
            continue
        leaders[col] = str(df.loc[s.idxmax(), "brand"])
        laggards[col] = str(df.loc[s.idxmin(), "brand"])

    return leaders, laggards


def _add_winner_loser_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add table-friendly winner/loser callouts per metric.

    This makes leaders/laggards visible *inside* the comparison table
    without relying on color styling (keeps it robust across themes).
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    leaders, laggards = _leaders_laggards(out)

    label_map = {
        "vfm_index": "VFM",
        "sentiment_mean": "Sentiment",
        "avg_rating": "Rating",
        "reviews": "Review volume",
        "avg_discount_pct": "Discount",
        "avg_price": "Price (premium)",
    }

    def build_list(brand: str, mapping: dict[str, str]) -> str:
        hits = [label_map[k] for k, v in mapping.items() if v == brand and k in label_map]
        return ", ".join(hits)

    out["winner_metrics"] = out["brand"].astype(str).map(lambda b: build_list(b, leaders))
    out["loser_metrics"] = out["brand"].astype(str).map(lambda b: build_list(b, laggards))
    out["winner_metrics"] = out["winner_metrics"].fillna("")
    out["loser_metrics"] = out["loser_metrics"].fillna("")
    return out


def _display_brand_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format the brand comparison table for readability.

    Keeps raw metrics intact but adds human-friendly display columns.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    # Stable display columns
    if "avg_discount_pct" in out.columns:
        out["avg_discount_%"] = pd.to_numeric(out["avg_discount_pct"], errors="coerce") * 100.0
    if "avg_price" in out.columns:
        out["avg_price_₹"] = pd.to_numeric(out["avg_price"], errors="coerce")
    if "avg_rating" in out.columns:
        out["avg_rating"] = pd.to_numeric(out["avg_rating"], errors="coerce")
    if "sentiment_mean" in out.columns:
        out["sentiment_mean"] = pd.to_numeric(out["sentiment_mean"], errors="coerce")
    if "vfm_index" in out.columns:
        out["vfm_index"] = pd.to_numeric(out["vfm_index"], errors="coerce")
    if "reviews" in out.columns:
        out["reviews"] = pd.to_numeric(out["reviews"], errors="coerce").fillna(0).astype(int)
    if "products" in out.columns:
        out["products"] = pd.to_numeric(out["products"], errors="coerce").fillna(0).astype(int)

    # Preferred ordering (only keep columns that exist)
    preferred = [
        "brand",
        "avg_price_₹",
        "avg_discount_%",
        "avg_rating",
        "products",
        "reviews",
        "sentiment_mean",
        "vfm_index",
        "winner_metrics",
        "loser_metrics",
        "top_pros",
        "top_cons",
        "premium_positioned",
        "high_discounting",
        "discount_reliance",
    ]
    keep = [c for c in preferred if c in out.columns]
    if keep:
        out = out[keep]

    return out


def _brand_agg(products_df: pd.DataFrame, reviews_df: pd.DataFrame, aspects_df: pd.DataFrame) -> pd.DataFrame:
    if products_df.empty:
        return pd.DataFrame()

    p = products_df.copy()
    agg_p = (
        p.groupby("brand")
        .agg(
            avg_price=("price", "mean"),
            avg_discount_pct=("discount_pct", "mean"),
            avg_rating=("rating_avg", "mean"),
            products=("asin", "nunique"),
        )
        .reset_index()
    )

    if reviews_df.empty or "sentiment_score" not in reviews_df.columns:
        agg_r = pd.DataFrame({"brand": agg_p["brand"], "reviews": 0, "sentiment_mean": 0.0})
    else:
        agg_r = (
            reviews_df.groupby("brand")
            .agg(reviews=("review_text", "count"), sentiment_mean=("sentiment_score", "mean"))
            .reset_index()
        )

    out = agg_p.merge(agg_r, on="brand", how="left")

    # Plotly bubble sizes cannot contain NaN.
    if "reviews" in out.columns:
        out["reviews"] = pd.to_numeric(out["reviews"], errors="coerce").fillna(0)

    # Compute premium/value and discount-reliance signals for decision-ready callouts.
    price_std = float(out["avg_price"].std(ddof=0)) if out["avg_price"].notna().any() else 0.0
    if price_std == 0 or pd.isna(price_std):
        out["price_z"] = 0.0
    else:
        out["price_z"] = (out["avg_price"] - float(out["avg_price"].mean())) / price_std

    disc = pd.to_numeric(out["avg_discount_pct"], errors="coerce").fillna(0)
    disc_std = float(disc.std(ddof=0)) if disc.notna().any() else 0.0
    if disc_std == 0 or pd.isna(disc_std):
        discount_z = 0.0
    else:
        discount_z = (disc - float(disc.mean())) / disc_std
    out["discount_reliance"] = 0.6 * discount_z + 0.4 * out["price_z"].fillna(0)
    out["premium_positioned"] = out["price_z"] > 0.75
    out["high_discounting"] = disc > 0.25

    # VFM index (same definition as analysis stage, but recomputed on filtered data)
    price_z = (out["avg_price"] - out["avg_price"].mean()) / (out["avg_price"].std(ddof=0) or 1.0)
    out["vfm_index"] = out["sentiment_mean"].fillna(0) - 0.35 * price_z.fillna(0)

    if aspects_df is not None and not aspects_df.empty:
        counts = (
            aspects_df.groupby(["brand", "aspect", "polarity"]).size().reset_index(name="count")
        )
        pros = counts[counts["polarity"] == "positive"].sort_values(["brand", "count"], ascending=[True, False])
        cons = counts[counts["polarity"] == "negative"].sort_values(["brand", "count"], ascending=[True, False])

        def _top_str(df: pd.DataFrame) -> str:
            return ", ".join([f"{r.aspect} ({int(r['count'])})" for _, r in df.head(5).iterrows()])

        def _group_apply_top(df: pd.DataFrame, *, out_col: str) -> pd.DataFrame:
            """Return a 2-col dataframe: brand + out_col.

            Pandas can return either Series or DataFrame from groupby().apply(),
            and only Series.reset_index supports the `name=` kwarg.
            """
            gb = df.groupby("brand")
            try:
                applied = gb.apply(_top_str, include_groups=False)
            except TypeError:
                applied = gb.apply(_top_str)

            out_df = applied.reset_index()
            if out_col not in out_df.columns:
                value_cols = [c for c in out_df.columns if c != "brand"]
                if len(value_cols) == 1:
                    out_df = out_df.rename(columns={value_cols[0]: out_col})
                else:
                    # Last-resort fallback: pick the last non-brand column.
                    if value_cols:
                        out_df = out_df.rename(columns={value_cols[-1]: out_col})

            if out_col not in out_df.columns:
                out_df[out_col] = ""

            return out_df[["brand", out_col]]

        out = out.merge(
            _group_apply_top(pros, out_col="top_pros"),
            on="brand",
            how="left",
        )
        out = out.merge(
            _group_apply_top(cons, out_col="top_cons"),
            on="brand",
            how="left",
        )

    # Keep a stable schema for UI even when LLM/aspects are unavailable.
    for col in ["top_pros", "top_cons"]:
        if col not in out.columns:
            out[col] = ""
        else:
            out[col] = out[col].fillna("")

    return out.sort_values("vfm_index", ascending=False)


def _fmt_supporting_metrics(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if isinstance(x, list):
        parts = []
        for item in x:
            if isinstance(item, dict):
                n = str(item.get("name", "")).strip()
                v = str(item.get("value", "")).strip()
                if n and v:
                    parts.append(f"{n}: {v}")
        return " · ".join(parts)
    return str(x)


def _brand_selector(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None

    # Try click-to-select via dataframe selection if supported; otherwise fallback to selectbox.
    try:
        event = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        if event is not None and getattr(event, "selection", None):
            rows = event.selection.get("rows") or []
            if rows:
                return str(df.iloc[rows[0]]["brand"])
    except TypeError:
        pass

    return st.selectbox("Select brand", options=df["brand"].tolist())


def _theme_evidence(aspects_df: pd.DataFrame, *, brand: str, asin: str | None, polarity: str, n: int = 6) -> pd.DataFrame:
    if aspects_df.empty:
        return pd.DataFrame()
    df = aspects_df[aspects_df["brand"] == brand].copy()
    if asin is not None:
        df = df[df["asin"] == asin]
    df = df[df["polarity"] == polarity]
    if df.empty:
        return df
    counts = df.groupby("aspect").size().sort_values(ascending=False)
    top_aspects = counts.head(5).index.tolist()
    out = df[df["aspect"].isin(top_aspects)][["aspect", "evidence", "reason"]].dropna().head(n)
    return out


tab_overview, tab_brands, tab_product = st.tabs(["Overview", "Brand Comparison", "Product Drilldown"])

with tab_overview:
    _kpis(products_f, reviews_f)

    if products_f.empty:
        st.warning("No products match the filters.")
    else:
        brand_view = _brand_agg(products_f, reviews_f, review_aspects_f)

        st.divider()
        st.subheader("Pricing and sentiment overview")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Average price by brand")
            fig = px.bar(brand_view.sort_values("avg_price", ascending=False), x="brand", y="avg_price", color="brand")
            fig.update_layout(showlegend=False, xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

            # Product-level price spread (makes positioning obvious beyond averages).
            if "price" in products_f.columns and not products_f["price"].dropna().empty:
                st.subheader("Price spread (product-level)")
                fig = px.box(
                    products_f.dropna(subset=["price"]).copy(),
                    x="brand",
                    y="price",
                    points="all",
                )
                fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Price (INR)")
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Sentiment vs price")
            fig = px.scatter(
                brand_view,
                x="avg_price",
                y="sentiment_mean",
                size="reviews",
                color="brand",
                hover_data=["avg_discount_pct", "avg_rating"],
            )
            fig.update_layout(xaxis_title="Avg price (INR)", yaxis_title="Sentiment mean")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Pricing + discount snapshot")
        c3, c4 = st.columns(2)
        with c3:
            fig = px.bar(brand_view.sort_values("avg_discount_pct", ascending=False), x="brand", y="avg_discount_pct", color="brand")
            fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Avg discount %")
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.bar(brand_view.sort_values("vfm_index", ascending=False), x="brand", y="vfm_index", color="brand")
            fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Value-for-money index")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Brand table")
        st.caption("Winner/loser columns highlight which brands lead or lag key metrics.")
        brand_table = _display_brand_table(_add_winner_loser_cols(brand_view))
        st.dataframe(
            brand_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "avg_price_₹": st.column_config.NumberColumn("Avg price (₹)", format="%.0f"),
                "avg_discount_%": st.column_config.NumberColumn("Avg discount (%)", format="%.0f"),
                "avg_rating": st.column_config.NumberColumn("Avg rating", format="%.2f"),
                "sentiment_mean": st.column_config.NumberColumn("Sentiment", format="%.2f"),
                "vfm_index": st.column_config.NumberColumn("VFM index", format="%.2f"),
                "winner_metrics": st.column_config.TextColumn("Leads"),
                "loser_metrics": st.column_config.TextColumn("Lags"),
            },
        )

with tab_brands:
    st.subheader("Brand comparison")
    brand_view = _brand_agg(products_f, reviews_f, review_aspects_f)
    if brand_view.empty:
        st.warning("No data for brand comparison under current filters.")
    else:
        st.subheader("Decision summary")

        premium_row = brand_view.sort_values("avg_price", ascending=False).iloc[0]
        value_row = brand_view.sort_values("vfm_index", ascending=False).iloc[0]
        discount_row = brand_view.sort_values("avg_discount_pct", ascending=False).iloc[0]
        sentiment_row = brand_view.sort_values("sentiment_mean", ascending=False).iloc[0]

        st.markdown(
            "\n".join(
                [
                    f"- **Premium-positioned:** {premium_row['brand']} (avg price ≈ ₹{premium_row['avg_price']:.0f})",
                    f"- **Best value-for-money:** {value_row['brand']} (VFM index {value_row['vfm_index']:.2f})",
                    f"- **Most discount-reliant:** {discount_row['brand']} (avg discount ≈ {discount_row['avg_discount_pct']*100:.0f}%)",
                    f"- **Sentiment leader:** {sentiment_row['brand']} (sentiment {sentiment_row['sentiment_mean']:.2f})",
                ]
            )
        )

        # Clear winner/loser callouts per metric (fast to scan).
        leaders, laggards = _leaders_laggards(brand_view)
        if leaders and laggards:
            st.caption(
                " | ".join(
                    [
                        f"VFM: {leaders.get('vfm_index','—')} vs {laggards.get('vfm_index','—')}",
                        f"Sentiment: {leaders.get('sentiment_mean','—')} vs {laggards.get('sentiment_mean','—')}",
                        f"Rating: {leaders.get('avg_rating','—')} vs {laggards.get('avg_rating','—')}",
                        f"Discount: {leaders.get('avg_discount_pct','—')} vs {laggards.get('avg_discount_pct','—')}",
                    ]
                )
            )

        # Show exactly 5 insights prominently (LLM-backed or heuristic fallback).
        if agent_insights is not None and not agent_insights.empty:
            st.subheader("Agent insights (top 5)")
            ins = agent_insights[agent_insights["brand"].isin(brand_view["brand"].tolist())].copy()
            ins = ins.head(5)
            for idx, row in ins.iterrows():
                metrics_str = _fmt_supporting_metrics(row.get("supporting_metrics"))
                st.markdown(f"**{len(ins) and (ins.index.get_loc(idx)+1)}. {row.get('brand','')}** — {row.get('claim','')}")
                if metrics_str:
                    st.caption(metrics_str)

        brand_table = _display_brand_table(_add_winner_loser_cols(brand_view))

        selected_brand = _brand_selector(brand_table)

        if selected_brand:
            st.markdown(f"### Selected: {selected_brand}")

            left, right = st.columns(2)
            with left:
                st.write("Top pros")
                pros = _theme_evidence(review_aspects_f, brand=selected_brand, asin=None, polarity="positive")
                st.dataframe(pros, use_container_width=True, hide_index=True)
            with right:
                st.write("Top cons")
                cons = _theme_evidence(review_aspects_f, brand=selected_brand, asin=None, polarity="negative")
                st.dataframe(cons, use_container_width=True, hide_index=True)

            st.subheader("Brand signals")
            row = brand_view[brand_view["brand"] == selected_brand].iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg price", f"₹{row['avg_price']:.0f}" if pd.notna(row["avg_price"]) else "—")
            c2.metric("Avg discount", f"{row['avg_discount_pct']*100:.0f}%" if pd.notna(row["avg_discount_pct"]) else "—")
            c3.metric("Sentiment", f"{row['sentiment_mean']:.2f}" if pd.notna(row["sentiment_mean"]) else "—")
            c4.metric("VFM index", f"{row['vfm_index']:.2f}" if pd.notna(row["vfm_index"]) else "—")

        if agent_insights is not None and not agent_insights.empty and selected_brand:
            per_brand = agent_insights[agent_insights["brand"] == selected_brand].copy()
            if not per_brand.empty:
                st.subheader("Insights for selected brand")
                for i, (_, row) in enumerate(per_brand.head(5).iterrows(), start=1):
                    metrics_str = _fmt_supporting_metrics(row.get("supporting_metrics"))
                    st.markdown(f"**{i}.** {row.get('claim','')}")
                    if metrics_str:
                        st.caption(metrics_str)

with tab_product:
    st.subheader("Product drilldown")
    if products_f.empty:
        st.warning("No products match the filters.")
    else:
        brand = st.selectbox("Brand", sorted(products_f["brand"].dropna().unique().tolist()))
        psub = products_f[products_f["brand"] == brand].copy()

        # Sort by review_count then rating
        if "review_count" in psub.columns:
            psub = psub.sort_values(["review_count", "rating_avg"], ascending=[False, False])

        asin = st.selectbox(
            "Product",
            options=psub["asin"].tolist(),
            format_func=lambda a: f"{a} — {psub[psub['asin']==a].iloc[0]['title'][:80]}",
        )

        prow = psub[psub["asin"] == asin].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"₹{prow['price']:.0f}" if pd.notna(prow["price"]) else "—")
        c2.metric("List price", f"₹{prow['list_price']:.0f}" if pd.notna(prow["list_price"]) else "—")
        c3.metric("Discount", f"{prow['discount_pct']*100:.0f}%" if pd.notna(prow["discount_pct"]) else "—")
        c4.metric("Rating", f"{prow['rating_avg']:.1f}" if pd.notna(prow["rating_avg"]) else "—")

        st.write(prow["title"])

        # Context: where this product sits in its brand's price band.
        if "price" in psub.columns and psub["price"].dropna().shape[0] >= 3 and pd.notna(prow.get("price")):
            st.caption("Price context within the selected brand")
            band = psub.dropna(subset=["price"]).copy()
            fig = px.box(band, y="price", points="all")
            fig.add_scatter(y=[float(prow["price"])], mode="markers", name="Selected", marker=dict(size=12))
            fig.update_layout(showlegend=False, yaxis_title="Price (INR)", xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

        rsub = reviews_f[(reviews_f["asin"] == asin) & (reviews_f["brand"] == brand)].copy()
        st.caption(f"Reviews in current filter context: {len(rsub)}")

        left, right = st.columns(2)
        with left:
            st.write("Top complaint themes (with evidence)")
            cons = _theme_evidence(review_aspects_f, brand=brand, asin=asin, polarity="negative")
            st.dataframe(cons, use_container_width=True, hide_index=True)
        with right:
            st.write("Top appreciation themes (with evidence)")
            pros = _theme_evidence(review_aspects_f, brand=brand, asin=asin, polarity="positive")
            st.dataframe(pros, use_container_width=True, hide_index=True)

        st.write("Review synthesis")
        if rsub.empty:
            st.info("No reviews match the current filters for this product.")
        else:
            # Show representative positive/negative snippets
            if "sentiment_label" in rsub.columns:
                pos = rsub[rsub["sentiment_label"] == "positive"].head(3)
                neg = rsub[rsub["sentiment_label"] == "negative"].head(3)
            else:
                pos = rsub.head(3)
                neg = rsub.tail(3)

            cpos, cneg = st.columns(2)
            with cpos:
                st.write("Representative praise")
                for _, rr in pos.iterrows():
                    st.write(f"- {rr.get('review_title','').strip()}: {rr.get('review_text','')[:220]}…")
            with cneg:
                st.write("Representative complaints")
                for _, rr in neg.iterrows():
                    st.write(f"- {rr.get('review_title','').strip()}: {rr.get('review_text','')[:220]}…")
