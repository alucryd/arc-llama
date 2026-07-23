from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from arc_llama.config import (
    Config,
    GPUConfig,
    MCPServerConfig,
    ModelConfig,
    ProfileConfig,
    ServerConfig,
)
from arc_llama.server import create_app


class FakeServerPlan:
    backend_url = "http://fake-upstream"


class FakeBackend:
    plan = FakeServerPlan()
    is_running = True


class FakeRouter:
    def __init__(self, cfg, log_dir=None):
        self.cfg = cfg
        self.model = ModelConfig(
            name="qwen",
            path="/models/qwen.gguf",
            port=18080,
            gpu_pci_slot="0000:03:00.0",
            display_name="Qwen",
            aliases=["qwen.gguf"],
        )
        self._servers = {"qwen": FakeBackend()}
        self.metrics = {
            "loads": 5,
            "stops": 2,
            "load_errors": 1,
            "last_load_at": 1234.0,
            "last_error": None,
        }

    def all_models(self):
        return [self.model]

    async def ensure_active(self, query):
        if query not in {"qwen", "qwen.gguf"}:
            raise KeyError(query)
        return self.model, FakeBackend()

    async def shutdown(self):
        return None


class FakeResponse:
    status_code = 200
    headers = {
        "content-type": "application/json",
        "content-length": "999",
        "transfer-encoding": "chunked",
        "x-upstream": "ok",
    }
    content = b'{"ok": true}'


class FakeUpstreamStream:
    status_code = 200
    headers = {
        "content-type": "text/event-stream",
        "content-length": "999",
        "transfer-encoding": "chunked",
        "x-upstream": "ok",
    }
    closed = False

    async def aiter_raw(self):
        yield b"data: one\n\n"
        yield b"data: two\n\n"

    async def aclose(self):
        self.closed = True


class FakeAsyncClient:
    last_stream = None

    def __init__(self, timeout=None):
        self.timeout = timeout
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    def build_request(self, method, url, content=None, headers=None):
        return {
            "method": method,
            "url": url,
            "content": content,
            "headers": headers,
        }

    async def send(self, request, stream=False):
        assert request["url"] == "http://fake-upstream/v1/chat/completions"
        assert stream is True
        FakeAsyncClient.last_stream = FakeUpstreamStream()
        return FakeAsyncClient.last_stream

    async def post(self, url, content=None, headers=None):
        assert url == "http://fake-upstream/v1/chat/completions"
        return FakeResponse()

    async def aclose(self):
        self.closed = True


class FakeUpstreamManager:
    def __init__(self, upstreams=None):
        self._upstreams = upstreams or []
        self._models = []

    async def models(self):
        return self._models

    def find_model(self, model_id):
        return None

    async def proxy(self, upstream, path, body, headers, streaming_ok=True):
        raise RuntimeError("should not be called")

    def upstreams_status(self):
        return []


class FakeUpstreamModel:
    def __init__(self, model_id, upstream_name, upstream_url):
        self.id = model_id
        self.upstream_name = upstream_name
        self.upstream_url = upstream_url
        self.metadata = {}


class FakeUpstreamResponse:
    status_code = 200
    headers = {"content-type": "application/json", "x-upstream": "upstream-ok"}
    _content = b'{"upstream": true}'
    closed = False

    async def aread(self):
        return self._content

    async def aclose(self):
        self.closed = True

    async def aiter_raw(self):
        yield self._content


class FakeUpstreamManagerWithModels:
    def __init__(self, upstreams=None):
        self._upstreams = upstreams or []
        self._models = [FakeUpstreamModel("llama3.1", "ollama", "http://127.0.0.1:11434")]

    async def models(self):
        return self._models

    def find_model(self, model_id):
        for m in self._models:
            if m.id == model_id:
                return m
        return None

    async def proxy(self, upstream, path, body, headers, streaming_ok=True):
        resp = FakeUpstreamResponse()
        return resp

    def upstreams_status(self):
        return [{"name": "ollama", "url": "http://127.0.0.1:11434", "model_count": 1, "last_fetch": 123.0}]


class FakeAsyncClientUpstream:
    """httpx.AsyncClient that simulates upstream proxy responses."""
    def __init__(self, timeout=None):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def build_request(self, method, url, content=None, headers=None):
        return {"method": method, "url": url, "content": content, "headers": headers}

    async def send(self, request, stream=False):
        resp = FakeUpstreamResponse()
        return resp

    async def aclose(self):
        pass


class FakeUpstreamStreamResponse:
    status_code = 200
    headers = {
        "content-type": "text/event-stream",
        "content-length": "999",
        "transfer-encoding": "chunked",
        "x-upstream": "stream-ok",
    }
    closed = False

    async def aiter_raw(self):
        yield b"data: upstream chunk 1\n\n"
        yield b"data: upstream chunk 2\n\n"

    async def aclose(self):
        self.closed = True


class FakeUpstreamManagerStreaming:
    def __init__(self, upstreams=None):
        self._upstreams = upstreams or []
        self._models = [FakeUpstreamModel("llama3.1", "ollama", "http://127.0.0.1:11434")]
        self.last_stream = None
        self.last_streaming_ok = None

    async def models(self):
        return self._models

    def find_model(self, model_id):
        for m in self._models:
            if m.id == model_id:
                return m
        return None

    async def proxy(self, upstream, path, body, headers, streaming_ok=True):
        self.last_streaming_ok = streaming_ok
        self.last_stream = FakeUpstreamStreamResponse()
        return self.last_stream

    def upstreams_status(self):
        return []


def test_non_streaming_proxy_strips_hop_by_hop_headers(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response.headers["x-upstream"] == "ok"
    assert "content-length" in response.headers
    assert "transfer-encoding" not in response.headers


def test_streaming_proxy_forwards_raw_sse_chunks_and_closes_upstream(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app()

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        ) as response:
            body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert body == b"data: one\n\ndata: two\n\n"
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-upstream"] == "ok"
    assert "transfer-encoding" not in response.headers
    assert FakeAsyncClient.last_stream.closed is True


def test_upstream_model_proxy(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManagerWithModels)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClientUpstream)
    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.1", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert response.json() == {"upstream": True}
    assert response.headers["x-upstream"] == "upstream-ok"


def test_list_models_includes_upstream(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManagerWithModels)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()["data"]
    ids = {m["id"] for m in data}
    assert "qwen" in ids
    assert "llama3.1" in ids
    upstream = next(m for m in data if m["id"] == "llama3.1")
    assert upstream["owned_by"] == "upstream:ollama"


def test_admin_status_includes_upstreams(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManagerWithModels)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app(Config())

    with TestClient(app) as client:
        response = client.get("/admin/status")

    assert response.status_code == 200
    status = response.json()
    assert "upstreams" in status
    assert len(status["upstreams"]) == 1
    assert status["upstreams"][0]["name"] == "ollama"


def test_admin_load_rejects_upstream(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManagerWithModels)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app(Config())

    with TestClient(app) as client:
        response = client.post("/admin/load/llama3.1")

    assert response.status_code == 400
    assert "Upstream model" in response.json()["detail"]


def test_upstream_streaming_proxy_forwards_sse_and_closes_upstream(monkeypatch):
    import arc_llama.server as server_mod

    mgr = FakeUpstreamManagerStreaming()
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", lambda upstreams=None: mgr)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app()

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "llama3.1", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        ) as response:
            body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert body == b"data: upstream chunk 1\n\ndata: upstream chunk 2\n\n"
    assert response.headers["content-type"].startswith("text/event-stream")


def test_health_includes_loaded_models_and_uptime(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app()

    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["loaded_model_count"] == 1
    assert "qwen" in data["loaded_models"]
    assert data["uptime_seconds"] >= 0


def test_session_token_served_to_loopback_peer(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    cfg = Config(server=ServerConfig(admin_token="secret"))
    app = create_app(cfg)

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        r = client.get("/admin/session-token")
    assert r.status_code == 200
    assert r.json()["admin_token"] == "secret"


def test_session_token_refused_for_non_loopback_peer(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    cfg = Config(server=ServerConfig(admin_token="secret"))
    app = create_app(cfg)

    with TestClient(app, client=("192.168.1.50", 12345)) as client:
        r = client.get("/admin/session-token")
    assert r.status_code == 403


def test_admin_metrics_returns_counters_and_gpus(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    cfg = Config(gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage")])
    app = create_app(cfg)

    with TestClient(app) as client:
        r = client.get("/admin/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["loads"] == 5
    assert data["stops"] == 2
    assert data["load_errors"] == 1
    assert data["active_models"] == ["qwen"]
    assert any(g["pci_slot"] == "0000:03:00.0" for g in data["gpus"])


class CapturingMCPClientManager:
    started_servers = []

    def __init__(self, servers):
        self.servers = servers

    async def start(self):
        CapturingMCPClientManager.started_servers = list(self.servers)

    async def stop(self):
        pass


def test_server_lifespan_uses_active_profile_mcp_servers(monkeypatch):
    import arc_llama.server as server_mod
    from arc_llama.config import Config

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod, "MCPClientManager", CapturingMCPClientManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)

    cfg = Config()
    cfg.mcp_servers = [
        MCPServerConfig(name="fs", command="npx"),
        MCPServerConfig(name="gh", command="npx"),
    ]
    cfg.profiles = [ProfileConfig(name="work", mcp_servers=["fs"])]
    cfg.agent.profile = "work"

    app = create_app(cfg)
    with TestClient(app):
        pass

    assert [s.name for s in CapturingMCPClientManager.started_servers] == ["fs"]


class AgentFakeAsyncClient:
    """httpx.AsyncClient stand-in that lets /v1/agent complete without a real LLM."""

    def __init__(self, timeout=None, base_url=None):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "done"}}]
        }
        return resp

    async def aclose(self):
        pass


def _app_with_admin_token(monkeypatch, token: str):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", AgentFakeAsyncClient)
    cfg = Config(server=ServerConfig(admin_token=token))
    return create_app(cfg)


def test_admin_status_requires_token_when_configured(monkeypatch):
    app = _app_with_admin_token(monkeypatch, "secret")

    with TestClient(app) as client:
        assert client.get("/admin/status").status_code == 401
        assert client.get(
            "/admin/status", headers={"Authorization": "Bearer wrong"}
        ).status_code == 403
        assert client.get(
            "/admin/status", headers={"Authorization": "Bearer secret"}
        ).status_code == 200


def test_admin_load_requires_token_when_configured(monkeypatch):
    app = _app_with_admin_token(monkeypatch, "secret")

    with TestClient(app) as client:
        assert client.post("/admin/load/qwen").status_code == 401
        assert client.post(
            "/admin/load/qwen", headers={"Authorization": "Bearer secret"}
        ).status_code == 200


def test_agent_auto_confirm_requires_admin_token(monkeypatch):
    app = _app_with_admin_token(monkeypatch, "secret")

    with TestClient(app) as client:
        r = client.post("/v1/agent", json={
            "model": "qwen",
            "task": "hello",
            "auto_confirm": True,
        })
        assert r.status_code == 401

        r = client.post("/v1/agent", json={
            "model": "qwen",
            "task": "hello",
            "auto_confirm": True,
        }, headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200


def test_agent_without_auto_confirm_allows_unauthenticated_request(monkeypatch):
    app = _app_with_admin_token(monkeypatch, "secret")

    with TestClient(app) as client:
        r = client.post("/v1/agent", json={
            "model": "qwen",
            "task": "hello",
            "auto_confirm": False,
        })
        assert r.status_code == 200


def test_agent_confirm_endpoint_requires_admin_token(monkeypatch):
    app = _app_with_admin_token(monkeypatch, "secret")

    with TestClient(app) as client:
        assert client.post("/v1/agent/run-1/confirm", json={"approved": True}).status_code == 401
        assert client.post(
            "/v1/agent/run-1/confirm",
            json={"approved": True},
            headers={"Authorization": "Bearer secret"},
        ).status_code == 404  # run not found, but auth passed


def test_agent_plan_endpoint_requires_admin_token(monkeypatch):
    app = _app_with_admin_token(monkeypatch, "secret")

    with TestClient(app) as client:
        assert client.post("/v1/agent/run-1/plan", json={"approved": True}).status_code == 401
        assert client.post(
            "/v1/agent/run-1/plan",
            json={"approved": True},
            headers={"Authorization": "Bearer secret"},
        ).status_code == 404  # run not found, but auth passed


# ---------------------------------------------------------------------------
# /admin/models/{name}/edit — perf recipe fields
# ---------------------------------------------------------------------------

class FakeRouterWithRebuild(FakeRouter):
    async def rebuild_model(self, name):
        return True, False


def _edit_app(monkeypatch, tmp_path):
    import arc_llama.server as server_mod

    # Keep cfg.save() away from the real ~/.config.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(server_mod, "Router", FakeRouterWithRebuild)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    cfg = Config(
        server=ServerConfig(admin_token=None),
        gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24576)],
        models=[ModelConfig(
            name="qwen", path="/models/qwen.gguf", port=18080,
            gpu_pci_slot="0000:03:00.0", recipe={"ctx": 8192},
        )],
    )
    return create_app(cfg), cfg


def test_admin_edit_accepts_flash_attn_and_batch_size(monkeypatch, tmp_path):
    app, cfg = _edit_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/admin/models/qwen/edit",
            json={"flash_attn": "on", "batch_size": 2048, "ubatch_size": 1024},
        )
    assert response.status_code == 200
    body = response.json()
    assert set(body["changed"]) == {"flash_attn", "batch_size", "ubatch_size"}
    model = next(m for m in cfg.models if m.name == "qwen")
    assert model.recipe["flash_attn"] == "on"
    assert model.recipe["batch_size"] == 2048
    assert model.recipe["ubatch_size"] == 1024


def test_admin_edit_flash_attn_null_clears(monkeypatch, tmp_path):
    app, cfg = _edit_app(monkeypatch, tmp_path)
    model = next(m for m in cfg.models if m.name == "qwen")
    model.recipe["flash_attn"] = "on"
    with TestClient(app) as client:
        response = client.post("/admin/models/qwen/edit", json={"flash_attn": None})
    assert response.status_code == 200
    assert "flash_attn" not in model.recipe


def test_admin_edit_rejects_bad_flash_attn(monkeypatch, tmp_path):
    app, _cfg = _edit_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.post("/admin/models/qwen/edit", json={"flash_attn": "yes"})
    assert response.status_code == 400


def test_admin_edit_rejects_bad_batch_size(monkeypatch, tmp_path):
    app, _cfg = _edit_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.post("/admin/models/qwen/edit", json={"batch_size": 0})
    assert response.status_code == 400
