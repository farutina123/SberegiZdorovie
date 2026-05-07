from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


class ChatMemory:
    def __init__(self, limit: int) -> None:
        self._limit = max(1, limit)
        self._by_chat: dict[int, list[ChatMessage]] = {}

    def reset(self, chat_id: int) -> None:
        self._by_chat.pop(chat_id, None)

    def add(self, chat_id: int, role: Role, content: str) -> None:
        msgs = self._by_chat.setdefault(chat_id, [])
        msgs.append(ChatMessage(role=role, content=content))
        if len(msgs) > self._limit:
            self._by_chat[chat_id] = msgs[-self._limit :]

    def get(self, chat_id: int) -> list[ChatMessage]:
        return list(self._by_chat.get(chat_id, []))

