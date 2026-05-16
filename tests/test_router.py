"""Tests for arc_llama.router — model swap policy and lifecycle."""
from __future__ import annotations

import pytest

from arc_llama.config import Config, GPUConfig, ModelConfig
from arc_llama.router import Router


@pytest.fixture
def sample_cfg() -> Config:
    return Config(
        server=type("S", (), {"host": "127.0.0.1", "port": 11437, "single_resident": True})(),
        paths=type("P", (), {"llama_server": "/bin/llama-server"})(),
        gpus=[
            GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24 * 1024),
            GPUConfig(pci_slot="0000:04:00.0", sycl_index=1, arch="alchemist", vram_mb=16 * 1024),
        ],
        models=[
            ModelConfig(name="model-a", path="/a.gguf", port=18080, gpu_pci_slot="0000:03:00.0"),
            ModelConfig(name="model-b", path="/b.gguf", port=18081, gpu_pci_slot="0000:04:00.0"),
            ModelConfig(name="model-c", path="/c.gguf", port=18082, gpu_pci_slot="0000:03:00.0"),
        ],
    )


class FakeServer:
    """Drop-in replacement for LlamaServer in router tests."""
    def __init__(self, plan, name="llama-server"):
        self.plan = plan
        self.name = name
        self._running = False
        self.stopped = False

    @property
    def is_running(self):
        return self._running

    def start(self, log_dir=None):
        self._running = True
        self.stopped = False

    def stop(self):
        if self._running:
            self.stopped = True
        self._running = False

    async def wait_ready(self, timeout=1):
        return True


class UnhealthyFakeServer(FakeServer):
    async def wait_ready(self, timeout=1):
        return False


@pytest.fixture
def patched_router(sample_cfg: Config, monkeypatch: pytest.MonkeyPatch):
    """Return a Router with all LlamaServer instances mocked."""
    monkeypatch.setattr("arc_llama.router.LlamaServer", FakeServer)
    return Router(sample_cfg)


class TestRouterResolution:
    def test_resolve_known_model(self, sample_cfg: Config, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("arc_llama.router.LlamaServer", FakeServer)
        rt = Router(sample_cfg)
        result = rt.resolve("model-a")
        assert result is not None
        model, gpu, srv = result
        assert model.name == "model-a"
        assert gpu.pci_slot == "0000:03:00.0"
        assert srv.name == "model-a"

    def test_resolve_unknown_returns_none(self, sample_cfg: Config, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("arc_llama.router.LlamaServer", FakeServer)
        rt = Router(sample_cfg)
        assert rt.resolve("not-real") is None

    def test_all_models(self, sample_cfg: Config, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("arc_llama.router.LlamaServer", FakeServer)
        rt = Router(sample_cfg)
        assert len(rt.all_models()) == 3

    def test_backend_url(self, sample_cfg: Config, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("arc_llama.router.LlamaServer", FakeServer)
        rt = Router(sample_cfg)
        assert rt.backend_url_for("model-a") == "http://127.0.0.1:18080"


class TestRouterSwapPolicy:
    @pytest.mark.asyncio
    async def test_single_resident_evicts_everything(self, sample_cfg: Config, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("arc_llama.router.LlamaServer", FakeServer)
        rt = Router(sample_cfg)

        await rt.ensure_active("model-a")
        assert rt._servers["model-a"].is_running is True

        # Switch to model-b (different GPU); single_resident should evict model-a
        await rt.ensure_active("model-b")
        assert rt._servers["model-b"].is_running is True
        assert rt._servers["model-a"].is_running is False

    @pytest.mark.asyncio
    async def test_multi_resident_same_gpu_eviction(self, sample_cfg: Config, monkeypatch: pytest.MonkeyPatch):
        sample_cfg.server.single_resident = False
        monkeypatch.setattr("arc_llama.router.LlamaServer", FakeServer)
        rt = Router(sample_cfg)

        await rt.ensure_active("model-a")
        # model-c is on the SAME GPU (03:00.0) — should evict model-a
        await rt.ensure_active("model-c")
        assert rt._servers["model-a"].is_running is False
        assert rt._servers["model-c"].is_running is True

    @pytest.mark.asyncio
    async def test_multi_resident_different_gpu_no_eviction(self, sample_cfg: Config, monkeypatch: pytest.MonkeyPatch):
        sample_cfg.server.single_resident = False
        monkeypatch.setattr("arc_llama.router.LlamaServer", FakeServer)
        rt = Router(sample_cfg)

        await rt.ensure_active("model-a")  # GPU 03:00.0
        await rt.ensure_active("model-b")  # GPU 04:00.0
        # model-a should NOT be stopped
        assert rt._servers["model-a"].is_running is True
        assert rt._servers["model-b"].is_running is True

    @pytest.mark.asyncio
    async def test_stop_one(self, sample_cfg: Config, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("arc_llama.router.LlamaServer", FakeServer)
        rt = Router(sample_cfg)
        rt._servers["model-a"]._running = True
        stopped = await rt.stop_one("model-a")
        assert stopped is True
        stopped_again = await rt.stop_one("model-a")
        assert stopped_again is False

    @pytest.mark.asyncio
    async def test_stop_all(self, sample_cfg: Config, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("arc_llama.router.LlamaServer", FakeServer)
        rt = Router(sample_cfg)
        for s in rt._servers.values():
            s._running = True
        count = await rt.stop_all()
        assert count == 3

    @pytest.mark.asyncio
    async def test_rebuild_model(self, sample_cfg: Config, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("arc_llama.router.LlamaServer", FakeServer)
        rt = Router(sample_cfg)
        rt._servers["model-a"]._running = True
        rebuilt, was_running = await rt.rebuild_model("model-a")
        assert rebuilt is True
        assert was_running is True

    @pytest.mark.asyncio
    async def test_unknown_model_raises_keyerror(self, sample_cfg: Config, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("arc_llama.router.LlamaServer", FakeServer)
        rt = Router(sample_cfg)
        with pytest.raises(KeyError, match="Unknown model"):
            await rt.ensure_active("no-such-model")

    @pytest.mark.asyncio
    async def test_unhealthy_server_raises_runtimeerror(self, sample_cfg: Config, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("arc_llama.router.LlamaServer", UnhealthyFakeServer)
        rt = Router(sample_cfg)
        with pytest.raises(RuntimeError, match="did not become healthy"):
            await rt.ensure_active("model-a")
        assert rt._servers["model-a"].is_running is False
