"""Tests for the chat-history store and server endpoints."""
from __future__ import annotations

from pathlib import Path

import pytest

from arc_llama.chat_store import ChatMessage, ChatStore


@pytest.fixture
def store(tmp_path: Path) -> ChatStore:
    return ChatStore(tmp_path / "chats")


def test_create_and_get(store: ChatStore) -> None:
    chat = store.create("chat-1", "First chat")
    assert chat.id == "chat-1"
    assert chat.title == "First chat"
    assert chat.messages == []

    loaded = store.get("chat-1")
    assert loaded is not None
    assert loaded.title == "First chat"


def test_create_duplicate_raises(store: ChatStore) -> None:
    store.create("chat-1", "First chat")
    with pytest.raises(FileExistsError):
        store.create("chat-1", "Duplicate")


def test_save_appends_messages(store: ChatStore) -> None:
    chat = store.create("chat-1", "First chat")
    chat.messages.append(ChatMessage(role="user", content="hello"))
    chat.messages.append(ChatMessage(role="assistant", content="hi"))
    store.save(chat)

    loaded = store.get("chat-1")
    assert loaded is not None
    assert len(loaded.messages) == 2
    assert loaded.messages[0].role == "user"
    assert loaded.messages[0].content == "hello"
    assert loaded.messages[1].role == "assistant"
    assert loaded.updated_at >= chat.created_at


def test_list_chats_sorted_by_updated(store: ChatStore) -> None:
    chat_a = store.create("a", "A")
    store.create("b", "B")
    chat_a.messages.append(ChatMessage(role="user", content="update"))
    store.save(chat_a)

    chats = store.list_chats()
    assert [c.id for c in chats] == ["a", "b"]


def test_delete(store: ChatStore) -> None:
    store.create("chat-1", "First chat")
    assert store.delete("chat-1") is True
    assert store.get("chat-1") is None
    assert store.delete("chat-1") is False


def test_search_matches_title(store: ChatStore) -> None:
    store.create("project-x", "Project X planning")
    store.create("notes", "Random notes")
    results = store.search("project x")
    assert len(results) == 1
    assert results[0][0].id == "project-x"
    assert results[0][1] == [-1]


def test_search_matches_message_content(store: ChatStore) -> None:
    chat = store.create("chat-1", "Untitled chat")
    chat.messages.append(ChatMessage(role="user", content="I love rust"))
    chat.messages.append(ChatMessage(role="assistant", content="Rust is great"))
    store.save(chat)

    results = store.search("rust")
    assert len(results) == 1
    assert results[0][1] == [0, 1]


def test_summary(store: ChatStore) -> None:
    chat = store.create("chat-1", "Summary test")
    chat.messages.append(ChatMessage(role="user", content="hi"))
    summary = chat.summary()
    assert summary["id"] == "chat-1"
    assert summary["title"] == "Summary test"
    assert summary["message_count"] == 1
    assert "created_at" in summary
    assert "updated_at" in summary


