from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    openai_api_key: str
    openai_model: str
    openai_base_url: str | None
    context_dir: str
    history_limit: int
    context_char_limit: int


def load_config() -> Config:
    load_dotenv()

    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    context_dir = os.getenv("CONTEXT_DIR", "context").strip()
    history_limit = int(os.getenv("HISTORY_LIMIT", "10").strip())
    context_char_limit = int(os.getenv("CONTEXT_CHAR_LIMIT", "35000").strip())

    if not telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    return Config(
        telegram_bot_token=telegram_bot_token,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        context_dir=context_dir,
        history_limit=history_limit,
        context_char_limit=context_char_limit,
    )

