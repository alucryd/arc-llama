"""Tests for the /v1/chats server endpoints."""
from __future__ import annotations

from fastapi.testclient import TestClient

import arc_llama.server as server_mod
from arc_llama.config import Config
from arc_llama.server import create_app


class FakeRouter:
    def __init__(self, cfg, log_dir=None):
        self.cfg = cfg
        self._servers = {}

    def all_models(self):
        return []

    async def shutdown(self):
        return None


class FakeUpstreamManager:
    def __init__(self, upstreams=None):
        pass

    async def models(self):
        return []

    def find_model(self, model_id):
        return None

    async def proxy(self, upstream, path, body, headers, streaming_ok=True):
        raise RuntimeError("should not be called")

    def upstreams_status(self):
        return []


def _app(tmp_path):
    cfg = Config()
    cfg.paths.state_dir = str(tmp_path / "state")
    app = create_app(cfg)
    return app


def test_create_list_get_delete_chat(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    app = _app(tmp_path)

    with TestClient(app) as client:
        create_resp = client.post("/v1/chats", json={"id": "chat-1", "title": "Planning"})
        assert create_resp.status_code == 200
        assert create_resp.json()["id"] == "chat-1"
        assert create_resp.json()["title"] == "Planning"

        list_resp = client.get("/v1/chats")
        assert list_resp.status_code == 200
        data = list_resp.json()["data"]
        assert len(data) == 1
        assert data[0]["id"] == "chat-1"
        assert data[0]["message_count"] == 0

        get_resp = client.get("/v1/chats/chat-1")
        assert get_resp.status_code == 200
        assert get_resp.json()["title"] == "Planning"

        delete_resp = client.delete("/v1/chats/chat-1")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["deleted"] is True

        assert client.get("/v1/chats/chat-1").status_code == 404


def test_patch_appends_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    app = _app(tmp_path)

    with TestClient(app) as client:
        client.post("/v1/chats", json={"id": "chat-1", "title": "T"})
        patch_resp = client.patch(
            "/v1/chats/chat-1",
            json={
                "title": "Renamed",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
            },
        )
        assert patch_resp.status_code == 200
        body = patch_resp.json()
        assert body["title"] == "Renamed"
        assert len(body["messages"]) == 2
        assert body["messages"][1]["content"] == "world"


def test_search_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    app = _app(tmp_path)

    with TestClient(app) as client:
        client.post("/v1/chats", json={"id": "chat-1", "title": "Rust ideas"})
        client.patch(
            "/v1/chats/chat-1",
            json={"messages": [{"role": "user", "content": "I want to learn rust"}]},
        )

        resp = client.post("/v1/chats/search", json={"query": "rust"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 1
        assert data[0]["chat"]["id"] == "chat-1"
        assert data[0]["matching_message_indices"] == [-1, 0]
