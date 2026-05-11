from __future__ import annotations

import os

import httpx
from openai import AsyncOpenAI

from bot.memory import ChatMessage


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def build_messages(system_prompt: str, context_blob: str, history: list[ChatMessage], user_text: str):
    system = system_prompt
    if context_blob:
        system = (
            system
            + "\n\n========================\nКОНТЕКСТ (единственный источник истины)\n========================\n"
            + context_blob
        )

    messages = [{"role": "system", "content": system}]
    for m in history:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": user_text})
    return messages


async def ask_llm(
    *,
    api_key: str,
    base_url: str | None,
    model: str,
    system_prompt: str,
    context_blob: str,
    history: list[ChatMessage],
    user_text: str,
) -> str:
    # Force a valid base URL even if a system-level env var is malformed.
    resolved_base_url = base_url or DEFAULT_OPENAI_BASE_URL
    insecure_skip_verify = os.getenv("OPENAI_INSECURE_SKIP_VERIFY", "0").strip() in {
        "1",
        "true",
        "True",
        "yes",
        "YES",
    }

    http_client = None
    if insecure_skip_verify:
        http_client = httpx.AsyncClient(verify=False)

    client = AsyncOpenAI(api_key=api_key, base_url=resolved_base_url, http_client=http_client)

    messages = build_messages(system_prompt, context_blob, history, user_text)

    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    out = (resp.choices[0].message.content or "").strip()
    return out or "В контексте нет данных: пустой ответ модели."

