from __future__ import annotations

import os
from typing import Any

import pandas as pd

from src.nlp.openai_compat import OpenAICompatClient, OpenAICompatConfig
from src.utils.env import settings


def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


_OUT_COLS = ["scope", "brand", "asin", "claim", "supporting_metrics", "supporting_themes"]


def _mk_metric(name: str, value: Any) -> dict[str, str]:
    return {"name": str(name), "value": "—" if value is None else str(value)}


def _heuristic_agent_insights(brand_metrics: pd.DataFrame, n: int = 5) -> list[dict[str, Any]]:
    """Deterministic fallback insights.

    Uses only computed metrics; never invents numbers.
    """
    if brand_metrics is None or brand_metrics.empty:
        return []

    df = brand_metrics.copy()

    # Ensure expected numeric cols exist
    for c in [
        "avg_price",
        "avg_discount_pct",
        "avg_rating",
        "reviews",
        "sentiment_mean",
        "vfm_index",
        "discount_reliance",
        "dup_exact_rate",
        "short_review_rate",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def pick(col: str, ascending: bool) -> pd.Series:
        if col not in df.columns or df[col].dropna().empty:
            return df.iloc[0]
        return df.sort_values(col, ascending=ascending).iloc[0]

    ins: list[dict[str, Any]] = []

    best_vfm = pick("vfm_index", ascending=False)
    ins.append(
        {
            "scope": "brand",
            "brand": best_vfm.get("brand"),
            "asin": None,
            "claim": "Best value-for-money (sentiment adjusted for price) among tracked brands.",
            "supporting_metrics": [
                _mk_metric("vfm_index", best_vfm.get("vfm_index")),
                _mk_metric("avg_price", best_vfm.get("avg_price")),
                _mk_metric("sentiment_mean", best_vfm.get("sentiment_mean")),
            ],
            "supporting_themes": [],
        }
    )

    premium = pick("avg_price", ascending=False)
    ins.append(
        {
            "scope": "brand",
            "brand": premium.get("brand"),
            "asin": None,
            "claim": "Most premium-positioned brand by average selling price; validate that sentiment keeps up with the premium.",
            "supporting_metrics": [
                _mk_metric("avg_price", premium.get("avg_price")),
                _mk_metric("sentiment_mean", premium.get("sentiment_mean")),
                _mk_metric("avg_discount_pct", premium.get("avg_discount_pct")),
            ],
            "supporting_themes": [],
        }
    )

    discount_heavy = pick("avg_discount_pct", ascending=False)
    ins.append(
        {
            "scope": "brand",
            "brand": discount_heavy.get("brand"),
            "asin": None,
            "claim": "Most discount-reliant brand (highest average discount); may be buying demand via promotions.",
            "supporting_metrics": [
                _mk_metric("avg_discount_pct", discount_heavy.get("avg_discount_pct")),
                _mk_metric("discount_reliance", discount_heavy.get("discount_reliance")),
                _mk_metric("reviews", discount_heavy.get("reviews")),
            ],
            "supporting_themes": [],
        }
    )

    sentiment_leader = pick("sentiment_mean", ascending=False)
    ins.append(
        {
            "scope": "brand",
            "brand": sentiment_leader.get("brand"),
            "asin": None,
            "claim": "Strongest sentiment leader; use this brand as the benchmark for what customers praise most.",
            "supporting_metrics": [
                _mk_metric("sentiment_mean", sentiment_leader.get("sentiment_mean")),
                _mk_metric("avg_rating", sentiment_leader.get("avg_rating")),
                _mk_metric("reviews", sentiment_leader.get("reviews")),
            ],
            "supporting_themes": [],
        }
    )

    # Trust/risk: if signals exist, flag highest risk; else flag lowest sentiment at similar price
    if "dup_exact_rate" in df.columns and df["dup_exact_rate"].dropna().any():
        trust_risk = pick("dup_exact_rate", ascending=False)
        ins.append(
            {
                "scope": "brand",
                "brand": trust_risk.get("brand"),
                "asin": None,
                "claim": "Highest duplicate-review rate signal; treat review sentiment with a bit more caution.",
                "supporting_metrics": [
                    _mk_metric("dup_exact_rate", trust_risk.get("dup_exact_rate")),
                    _mk_metric("short_review_rate", trust_risk.get("short_review_rate")),
                    _mk_metric("reviews", trust_risk.get("reviews")),
                ],
                "supporting_themes": [],
            }
        )
    else:
        worst_sent = pick("sentiment_mean", ascending=True)
        ins.append(
            {
                "scope": "brand",
                "brand": worst_sent.get("brand"),
                "asin": None,
                "claim": "Weakest sentiment performer; investigate recurring complaints and whether discounting is masking issues.",
                "supporting_metrics": [
                    _mk_metric("sentiment_mean", worst_sent.get("sentiment_mean")),
                    _mk_metric("avg_discount_pct", worst_sent.get("avg_discount_pct")),
                    _mk_metric("avg_price", worst_sent.get("avg_price")),
                ],
                "supporting_themes": [],
            }
        )

    # De-duplicate by brand+claim and cap to n
    out: list[dict[str, Any]] = []
    seen = set()
    for x in ins:
        key = (str(x.get("brand")), str(x.get("claim")))
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
        if len(out) >= n:
            break
    return out


def generate_agent_insights(
    brand_metrics: pd.DataFrame,
    prompt_path: str,
    *,
    min_insights: int = 5,
    use_llm: bool = True,
) -> pd.DataFrame:
    if brand_metrics is None or brand_metrics.empty:
        return pd.DataFrame(columns=_OUT_COLS)

    rows: list[dict[str, Any]] = []

    if use_llm and settings.openai_api_key:
        system = _load_prompt(prompt_path)

        # Provide only what we need to avoid hallucinated numbers.
        cols = [
            "brand",
            "avg_price",
            "avg_discount_pct",
            "avg_rating",
            "reviews",
            "sentiment_mean",
            "vfm_index",
            "discount_reliance",
            "top_pros",
            "top_cons",
            "dup_exact_rate",
            "short_review_rate",
        ]
        payload_df = brand_metrics[[c for c in cols if c in brand_metrics.columns]].copy()

        user = "BRAND_METRICS_JSON:\n" + payload_df.to_json(orient="records")

        client = OpenAICompatClient(
            OpenAICompatConfig(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                model=settings.openai_model,
                extra_headers=settings.llm_extra_headers(),
            )
        )

        try:
            out = client.chat_json(system=system, user=user, temperature=0.2, max_tokens=900)
            for ins in out.get("insights", []) or []:
                rows.append(
                    {
                        "scope": ins.get("scope") or "brand",
                        "brand": ins.get("brand"),
                        "asin": ins.get("asin"),
                        "claim": ins.get("claim"),
                        "supporting_metrics": ins.get("supporting_metrics"),
                        "supporting_themes": ins.get("supporting_themes"),
                    }
                )
        except Exception:
            rows = []

    # Ensure at least min_insights exist.
    if len(rows) < min_insights:
        need = min_insights - len(rows)
        rows.extend(_heuristic_agent_insights(brand_metrics, n=need))

    df = pd.DataFrame(rows, columns=_OUT_COLS)
    df = df.dropna(subset=["brand", "claim"], how="any")
    # If we somehow got more than requested, keep top min_insights deterministically.
    if len(df) > min_insights:
        df = df.head(min_insights).reset_index(drop=True)
    return df
