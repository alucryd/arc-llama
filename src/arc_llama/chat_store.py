"""Persistent chat-history store for arc-llama.

Chats are stored as individual JSON files under ``<state_dir>/chats``. Each
file contains a lightweight chat record with a message list so that the agent
can reference prior conversations.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ChatMessage:
    """A single message inside a chat."""

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        return cls(
            role=data.get("role", ""),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class Chat:
    """A persisted conversation."""

    id: str
    title: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: list[ChatMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chat:
        return cls(
            id=data.get("id", ""),
            title=data.get("title", "Untitled chat"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            messages=[ChatMessage.from_dict(m) for m in data.get("messages", [])],
        )

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary without full message contents."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
        }


class ChatStore:
    """On-disk JSON store for chat histories."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _chat_path(self, chat_id: str) -> Path:
        """Return the path to a chat file, normalising the id."""
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", chat_id)
        return self.directory / f"{safe_id}.json"

    def create(self, chat_id: str, title: str, messages: list[ChatMessage] | None = None) -> Chat:
        """Create a new chat. Raises ``FileExistsError`` if it already exists."""
        path = self._chat_path(chat_id)
        if path.exists():
            raise FileExistsError(f"Chat already exists: {chat_id}")
        now = time.time()
        chat = Chat(
            id=chat_id,
            title=title,
            created_at=now,
            updated_at=now,
            messages=list(messages or []),
        )
        self._save(chat)
        return chat

    def get(self, chat_id: str) -> Chat | None:
        """Load a chat by id. Returns None if not found."""
        path = self._chat_path(chat_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return Chat.from_dict(data)

    def save(self, chat: Chat) -> None:
        """Persist a chat, updating ``updated_at``."""
        chat.updated_at = time.time()
        self._save(chat)

    def _save(self, chat: Chat) -> None:
        path = self._chat_path(chat.id)
        path.write_text(
            json.dumps(chat.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def delete(self, chat_id: str) -> bool:
        """Delete a chat. Returns True if it existed."""
        path = self._chat_path(chat_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_chats(self) -> list[Chat]:
        """Return all stored chats sorted by updated_at descending."""
        chats: list[Chat] = []
        for path in self.directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                chats.append(Chat.from_dict(data))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(chats, key=lambda c: c.updated_at, reverse=True)

    def search(self, query: str, limit: int = 20) -> list[tuple[Chat, list[int]]]:
        """Search chat titles and message contents for *query*.

        Returns a list of ``(chat, matching_message_indices)`` tuples, ordered by
        most recently updated chat first. The query is matched case-insensitively
        and supports no special syntax.
        """
        query_lower = query.lower()
        results: list[tuple[Chat, list[int]]] = []
        for chat in self.list_chats():
            matches: list[int] = []
            if query_lower in chat.title.lower():
                matches.append(-1)
            for idx, msg in enumerate(chat.messages):
                if query_lower in msg.content.lower():
                    matches.append(idx)
            if matches:
                results.append((chat, matches[:limit]))
        return results[:limit]

    def wipe(self) -> None:
        """Delete the entire store directory. Useful for tests."""
        if self.directory.exists():
            shutil.rmtree(self.directory)
            self.directory.mkdir(parents=True, exist_ok=True)
