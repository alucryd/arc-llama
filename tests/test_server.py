"""Tests for arc_llama.server — HTTP surface with TestClient."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from arc_llama.config import Config, GPUConfig, ModelConfig, PathsConfig, ServerConfig, default_config_path
from arc_llama.server import create_app


@pytest.fixture
def sample_cfg() -> Config:
    return Config(
        server=ServerConfig(host="127.0.0.1", port=11437, single_resident=True),
        paths=PathsConfig(llama_server="/bin/llama-server", state_dir="/tmp/arc-llama-test"),
        gpus=[
            GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24 * 1024),
        ],
        models=[
            ModelConfig(
                name="qwen3-7b",
                path="/models/qwen3-7b.gguf",
                port=18080,
                gpu_pci_slot="0000:03:00.0",
                display_name="Qwen 3 7B",
                recipe={"ctx": 32768, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
            ),
        ],
    )


class FakeRouter:
    """Mock router that doesn't spawn subprocesses."""
    def __init__(self, cfg: Config, log_dir: Path | None = None):
        self.cfg = cfg
        self._servers: dict[str, Any] = {}
        for m in cfg.models:
            self._servers[m.name] = type(
                "FakeSrv",
                (),
                {
                    "is_running": False,
                    "plan": type("Plan", (), {"backend_url": f"http://127.0.0.1:{m.port}"})(),
                },
            )()

    def resolve(self, query: str):
        m = next((x for x in self.cfg.models if x.name == query), None)
        if m is None:
            return None
        g = self.cfg.find_gpu(m.gpu_pci_slot)
        return m, g, self._servers[m.name]

    def all_models(self):
        return list(self.cfg.models)

    def backend_url_for(self, name: str):
        return f"http://127.0.0.1:{next(m.port for m in self.cfg.models if m.name == name)}"

    async def ensure_active(self, query: str):
        resolved = self.resolve(query)
        if resolved is None:
            raise KeyError(f"Unknown model: {query!r}")
        m, g, srv = resolved
        srv.is_running = True
        return m, srv

    async def stop_one(self, name: str):
        srv = self._servers.get(name)
        if srv is None or not srv.is_running:
            return False
        srv.is_running = False
        return True

    async def stop_all(self):
        stopped = 0
        for srv in self._servers.values():
            if srv.is_running:
                srv.is_running = False
                stopped += 1
        return stopped

    async def rebuild_model(self, name: str):
        return True, False

    async def shutdown(self):
        pass


@pytest.fixture
def client(sample_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Build a TestClient with a mocked router injected directly into app.state."""
    app = create_app(sample_cfg)
    # Bypass lifespan by setting state directly
    app.state.cfg = sample_cfg
    app.state.router = FakeRouter(sample_cfg)
    app.state.config_path = default_config_path()
    app.state.upstream_cache = {"models": []}
    app.state.upstream_cache_ts = float("inf")
    return TestClient(app)


class TestHealth:
    def test_health_ok(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestListModels:
    def test_lists_registered_models(self, client: TestClient):
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()["data"]
        assert any(m["id"] == "qwen3-7b" for m in data)


class TestAdminStatus:
    def test_snapshot(self, client: TestClient):
        r = client.get("/admin/status")
        assert r.status_code == 200
        s = r.json()
        assert s["server"]["port"] == 11437
        assert len(s["gpus"]) == 1
        assert s["gpus"][0]["arch"] == "battlemage"
        assert len(s["models"]) == 1
        assert s["models"][0]["name"] == "qwen3-7b"
        assert s["models"][0]["ctx"] == 32768


class TestAdminLoadStop:
    def test_load_unknown_404(self, client: TestClient):
        r = client.post("/admin/load/no-such-model")
        assert r.status_code == 404

    def test_load_known(self, client: TestClient):
        r = client.post("/admin/load/qwen3-7b")
        assert r.status_code == 200
        assert r.json()["loaded"] is True

    def test_stop_unknown_404(self, client: TestClient):
        r = client.post("/admin/stop/no-such-model")
        assert r.status_code == 404

    def test_stop_known(self, client: TestClient):
        client.post("/admin/load/qwen3-7b")
        r = client.post("/admin/stop/qwen3-7b")
        assert r.status_code == 200
        assert r.json()["was_running"] is True

    def test_stop_all(self, client: TestClient):
        client.post("/admin/load/qwen3-7b")
        r = client.post("/admin/stop-all")
        assert r.status_code == 200
        assert r.json()["stopped"] == 1


class TestAdminEditModel:
    def test_edit_ctx(self, client: TestClient):
        r = client.post("/admin/models/qwen3-7b/edit", json={"ctx": 16384})
        assert r.status_code == 200
        j = r.json()
        assert "ctx" in j["changed"]
        assert j["recipe"]["ctx"] == 16384

    def test_edit_invalid_ctx_type(self, client: TestClient):
        r = client.post("/admin/models/qwen3-7b/edit", json={"ctx": "not_a_number"})
        assert r.status_code == 400

    def test_edit_ctx_out_of_range(self, client: TestClient):
        r = client.post("/admin/models/qwen3-7b/edit", json={"ctx": 100})
        assert r.status_code == 400

    def test_edit_invalid_kv(self, client: TestClient):
        r = client.post("/admin/models/qwen3-7b/edit", json={"cache_type_k": "q3_0"})
        assert r.status_code == 400

    def test_edit_no_fields(self, client: TestClient):
        r = client.post("/admin/models/qwen3-7b/edit", json={"foo": "bar"})
        assert r.status_code == 400

    def test_edit_unknown_model(self, client: TestClient):
        r = client.post("/admin/models/ghost/edit", json={"ctx": 4096})
        assert r.status_code == 404


class TestAdminScan:
    def test_scan_empty(self, client: TestClient, monkeypatch: pytest.MonkeyPatch):
        import arc_llama.models as models_mod
        monkeypatch.setattr(models_mod, "discover_ggufs", lambda cfg, extra_paths=None: [])
        monkeypatch.setattr(models_mod, "register_discovered", lambda cfg, found, **kw: [])
        r = client.post("/admin/scan")
        assert r.status_code == 200
        assert r.json()["found"] == 0
        assert r.json()["added"] == []


class TestChatProxy:
    def test_unknown_model_404(self, client: TestClient):
        r = client.post("/v1/chat/completions", json={"model": "ghost", "messages": []})
        assert r.status_code == 404

    def test_invalid_json_400(self, client: TestClient):
        r = client.post("/v1/chat/completions", content="not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 400

    def test_proxies_to_backend(self, client: TestClient, monkeypatch: pytest.MonkeyPatch):
        import httpx
        called = {}

        async def _fake_post(self, url, **kwargs):
            called["url"] = url
            return type("R", (), {
                "content": b'{"choices":[]}',
                "status_code": 200,
                "headers": {"content-type": "application/json"},
            })()

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
        r = client.post("/v1/chat/completions", json={"model": "qwen3-7b", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        assert called["url"] == "http://127.0.0.1:18080/v1/chat/completions"
