from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.nlp.openai_compat import OpenAICompatClient, OpenAICompatConfig
from src.utils.env import settings
from src.utils.io import read_json, write_json
from src.utils.paths import data_dir


ASPECTS = ["wheels", "handle", "zipper", "material", "durability", "size", "weight", "service", "other"]


def _hash_text(text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h


def _cache_path(review_text: str) -> str:
    key = _hash_text(review_text)
    return os.path.join(data_dir("processed", "llm_cache", "aspects"), f"{key}.json")


def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@dataclass(frozen=True)
class AspectExtractionConfig:
    prompt_path: str


def extract_aspects_for_text(review_text: str, cfg: AspectExtractionConfig) -> dict[str, Any] | None:
    review_text = (review_text or "").strip()
    if not review_text:
        return {"aspects": []}

    cache_path = _cache_path(review_text)
    if os.path.exists(cache_path):
        return read_json(cache_path)

    if not settings.openai_api_key:
        return None

    system = _load_prompt(cfg.prompt_path)
    user = f"REVIEW TEXT:\n{review_text}\n"

    client = OpenAICompatClient(
        OpenAICompatConfig(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            extra_headers=settings.llm_extra_headers(),
        )
    )

    try:
        out = client.chat_json(system=system, user=user, temperature=0.0, max_tokens=700)
    except Exception as e:
        # Don't crash the pipeline; just skip this review.
        out = {"aspects": [], "_error": str(e)}

    write_json(out, cache_path)
    return out


def build_review_aspects(reviews: pd.DataFrame, prompt_path: str) -> pd.DataFrame:
    cols = ["review_id", "asin", "brand", "aspect", "polarity", "evidence", "reason"]
    if reviews.empty:
        return pd.DataFrame(columns=cols)

    cfg = AspectExtractionConfig(prompt_path=prompt_path)

    # Cost/time control: only run LLM for up to N reviews per analysis run.
    # Cached items do not count against this limit.
    max_llm_reviews = int(os.getenv("MAX_LLM_REVIEWS", "300"))

    rows: list[dict] = []

    processed = 0
    for _, r in tqdm(reviews.iterrows(), total=len(reviews), desc="Aspect extraction", disable=len(reviews) < 50):
        text = r.get("review_text", "")
        cache_path = _cache_path((text or "").strip())
        is_cached = os.path.exists(cache_path)
        if not is_cached and processed >= max_llm_reviews:
            continue

        payload = extract_aspects_for_text(text, cfg)
        if payload is None:
            continue

        if not is_cached:
            processed += 1
        for a in payload.get("aspects", []) or []:
            rows.append(
                {
                    "review_id": r.get("review_id"),
                    "asin": r.get("asin"),
                    "brand": r.get("brand"),
                    "aspect": a.get("aspect"),
                    "polarity": a.get("polarity"),
                    "evidence": a.get("evidence"),
                    "reason": a.get("reason"),
                }
            )

    return pd.DataFrame(rows, columns=cols)
