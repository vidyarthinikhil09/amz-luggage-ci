from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class OpenAICompatConfig:
    api_key: str
    base_url: str
    model: str
    extra_headers: dict[str, str] | None = None
    timeout_s: float = 60.0


class OpenAICompatClient:
    def __init__(self, cfg: OpenAICompatConfig):
        self._cfg = cfg

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> dict[str, Any]:
        url = self._cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self._cfg.api_key}"}
        if self._cfg.extra_headers:
            headers.update(self._cfg.extra_headers)

        payload_base: dict[str, Any] = {
            "model": self._cfg.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        # Some OpenAI-compatible providers/models don't support `response_format`.
        payload = dict(payload_base)
        payload["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=self._cfg.timeout_s) as client:
            try:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                # Retry without response_format if provider rejects it.
                body = (e.response.text or "").lower() if e.response is not None else ""
                if e.response is not None and e.response.status_code in (400, 422) and "response_format" in body:
                    resp = client.post(url, headers=headers, json=payload_base)
                    resp.raise_for_status()
                else:
                    raise

            data = resp.json()

        content = data["choices"][0]["message"]["content"]

        import json

        try:
            return json.loads(content)
        except Exception:
            # Fallback: extract the first JSON object from the content.
            # Keeps the pipeline resilient if the provider ignores response_format.
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(content[start : end + 1])
            raise
