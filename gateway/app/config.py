"""Gateway settings, read once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # Where Ollama listens (inside docker compose it is the "ollama" service).
    upstream_url: str = "http://ollama:11434"
    # SHA-256 hashes of the API keys allowed to call the gateway.
    # Plain keys are never stored: generate them with tools/new_key.py.
    api_key_hashes: list[str] = field(default_factory=list)
    # Models that clients may request. Empty list = any model pulled in Ollama.
    allowed_models: list[str] = field(default_factory=list)
    # Requests allowed per key and per minute.
    rate_limit_per_minute: int = 60
    # Prompts and answers are NOT logged unless explicitly enabled.
    log_prompts: bool = False
    # Upstream timeout in seconds (local models can be slow on first load).
    upstream_timeout: float = 300.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            upstream_url=os.getenv("UPSTREAM_URL", cls.upstream_url).rstrip("/"),
            api_key_hashes=[h.lower() for h in _csv(os.getenv("API_KEY_HASHES"))],
            allowed_models=_csv(os.getenv("ALLOWED_MODELS")),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", cls.rate_limit_per_minute)),
            log_prompts=os.getenv("LOG_PROMPTS", "false").lower() in {"1", "true", "yes"},
            upstream_timeout=float(os.getenv("UPSTREAM_TIMEOUT", cls.upstream_timeout)),
        )
