import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


load_dotenv()


def _env_str(name: str) -> str:
    return (os.getenv(name) or "").strip()


@dataclass(frozen=True)
class Settings:
    headless: bool = os.getenv("HEADLESS", "1") not in {"0", "false", "False"}
    throttle_ms: int = int(os.getenv("THROTTLE_MS", "1200"))
    user_agent: str = os.getenv(
        "USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    )

    # OpenAI-compatible LLM configuration.
    # Supports OpenRouter by mapping OPENROUTER_* into OPENAI_*.
    openrouter_api_key: str | None = _env_str("OPENROUTER_API_KEY") or None
    openrouter_model: str = _env_str("OPENROUTER_MODEL")
    openrouter_site_url: str = _env_str("OPENROUTER_SITE_URL")
    openrouter_app_name: str = _env_str("OPENROUTER_APP_NAME")

    openai_api_key: str | None = field(
        default_factory=lambda: _env_str("OPENAI_API_KEY") or _env_str("OPENROUTER_API_KEY") or None
    )
    openai_base_url: str = field(
        default_factory=lambda: _env_str("OPENAI_BASE_URL")
        or ("https://openrouter.ai/api/v1" if _env_str("OPENROUTER_API_KEY") else "https://api.openai.com/v1")
    )
    openai_model: str = field(
        default_factory=lambda: _env_str("OPENAI_MODEL") or _env_str("OPENROUTER_MODEL") or "gpt-4o-mini"
    )

    def llm_extra_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        # OpenRouter recommends these for attribution; they're optional.
        if self.openrouter_site_url:
            headers["HTTP-Referer"] = self.openrouter_site_url
        if self.openrouter_app_name:
            headers["X-Title"] = self.openrouter_app_name
        return headers


settings = Settings()
