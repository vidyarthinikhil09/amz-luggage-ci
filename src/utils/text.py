from __future__ import annotations

import re


_ws_re = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return _ws_re.sub(" ", (text or "").strip())


def normalize_review_text(text: str) -> str:
    text = normalize_whitespace(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = _ws_re.sub(" ", text).strip()
    return text


def infer_size_bucket(title: str) -> str | None:
    t = (title or "").lower()

    # Cabin / medium / large hints
    if any(k in t for k in ["cabin", "carry on", "carry-on"]):
        return "cabin"
    if "medium" in t:
        return "medium"
    if "large" in t:
        return "large"

    # cm or inch hints
    cm = re.search(r"(\d{2})\s*cm", t)
    if cm:
        v = int(cm.group(1))
        if v <= 55:
            return "cabin"
        if v <= 65:
            return "medium"
        return "large"

    inch = re.search(r"(\d{2})\s*(?:inch|in)", t)
    if inch:
        v = int(inch.group(1))
        if v <= 22:
            return "cabin"
        if v <= 26:
            return "medium"
        return "large"

    return None
