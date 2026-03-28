from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_analyzer = SentimentIntensityAnalyzer()


def rating_to_sentiment(rating: float | None) -> float | None:
    if rating is None or pd.isna(rating):
        return None
    # 1 -> -1, 3 -> 0, 5 -> +1
    return max(-1.0, min(1.0, (float(rating) - 3.0) / 2.0))


def text_to_sentiment(text: str) -> float:
    if not text:
        return 0.0
    vs = _analyzer.polarity_scores(text)
    return float(vs.get("compound", 0.0))


@dataclass(frozen=True)
class SentimentConfig:
    rating_weight: float = 0.70
    text_weight: float = 0.30
    min_text_len: int = 20


def blended_sentiment(rating: float | None, text: str, cfg: SentimentConfig = SentimentConfig()) -> tuple[float | None, str | None, float | None]:
    rs = rating_to_sentiment(rating)
    if rs is None:
        return None, None, None

    text = text or ""
    if len(text.strip()) < cfg.min_text_len:
        score = rs
        conf = 0.55 + 0.35 * abs(score)
    else:
        ts = text_to_sentiment(text)
        score = cfg.rating_weight * rs + cfg.text_weight * ts
        score = max(-1.0, min(1.0, score))
        conf = 0.60 + 0.35 * abs(score)

    if score >= 0.15:
        label = "positive"
    elif score <= -0.15:
        label = "negative"
    else:
        label = "neutral"

    return float(score), label, float(min(0.95, conf))


def add_sentiment_columns(reviews: pd.DataFrame) -> pd.DataFrame:
    if reviews.empty:
        return reviews

    out = reviews.copy()
    scores = out.apply(lambda r: blended_sentiment(r.get("rating"), r.get("review_text", "")), axis=1)
    out["sentiment_score"] = [s[0] for s in scores]
    out["sentiment_label"] = [s[1] for s in scores]
    out["sentiment_confidence"] = [s[2] for s in scores]
    return out
