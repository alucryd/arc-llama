"""Model swap policy.

The router owns the lifecycle of every llama-server subprocess and decides which
one is currently allowed to hold its GPU's VRAM. Two policies are supported:

  * **single_resident** (default): only one model is loaded across *all* GPUs
    at any time — switching models stops the previous one before starting the
    next. This matches conservative thermal/power use.

  * **multi_resident**: models on *different* GPUs can coexist; only models on
    the *same* GPU contend. Models still get loaded on demand and stay up for
    follow-up requests.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from arc_llama.config import Config, GPUConfig, ModelConfig
from arc_llama.launcher import LlamaServer, build_plan

log = logging.getLogger("arc_llama.router")


class Router:
    """Owns one LlamaServer per registered model and serialises swaps."""

    def __init__(self, cfg: Config, log_dir: Path | None = None):
        self.cfg = cfg
        self.log_dir = log_dir
        self._servers: dict[str, LlamaServer] = {}  # keyed by model.name
        self._lock = asyncio.Lock()
        self._build_servers()

    def _build_servers(self) -> None:
        """(Re)build the per-model LlamaServer registry from cfg.

        Idempotent — existing servers (running or not) are preserved by name,
        only new model entries get fresh LlamaServer instances. Use after a
        runtime config mutation (e.g. an admin scan).
        """
        for m in self.cfg.models:
            if m.name in self._servers:
                continue
            gpu = self.cfg.find_gpu(m.gpu_pci_slot)
            if gpu is None:
                log.warning(
                    "model %s references unknown GPU %s; skipping",
                    m.name, m.gpu_pci_slot,
                )
                continue
            plan = build_plan(self.cfg, m, gpu, host=self.cfg.server.host)
            self._servers[m.name] = LlamaServer(plan, name=m.name)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def resolve(self, query: str) -> tuple[ModelConfig, GPUConfig, LlamaServer] | None:
        m = self.cfg.find_model(query)
        if m is None:
            return None
        gpu = self.cfg.find_gpu(m.gpu_pci_slot)
        if gpu is None:
            return None
        srv = self._servers.get(m.name)
        if srv is None:
            return None
        return m, gpu, srv

    def all_models(self) -> list[ModelConfig]:
        return list(self.cfg.models)

    def backend_url_for(self, model_name: str) -> str | None:
        srv = self._servers.get(model_name)
        return srv.plan.backend_url if srv else None

    # ------------------------------------------------------------------
    # Swap
    # ------------------------------------------------------------------

    async def ensure_active(self, query: str) -> tuple[ModelConfig, LlamaServer]:
        """Make sure the requested model is the resident one (per policy) and
        return its (config, LlamaServer). Caller forwards the request to
        `srv.plan.backend_url`.
        """
        async with self._lock:
            resolved = self.resolve(query)
            if resolved is None:
                raise KeyError(f"Unknown model: {query!r}")
            target_model, target_gpu, target_srv = resolved
            await self._evict_for(target_model, target_gpu)
            if not target_srv.is_running:
                target_srv.start(log_dir=self.log_dir)
                ready = await target_srv.wait_ready()
                if not ready:
                    log.error(
                        "model %s failed health-check; stopping it",
                        target_model.name,
                    )
                    target_srv.stop()
                    raise RuntimeError(
                        f"llama-server for {target_model.name} did not become healthy"
                    )
            return target_model, target_srv

    async def _evict_for(self, target: ModelConfig, target_gpu: GPUConfig) -> None:
        """Stop the right neighbours so the target can have its GPU."""
        single = self.cfg.server.single_resident
        for name, srv in self._servers.items():
            if name == target.name:
                continue
            if not srv.is_running:
                continue
            other_model = next((m for m in self.cfg.models if m.name == name), None)
            if other_model is None:
                srv.stop()
                continue
            if single or other_model.gpu_pci_slot == target_gpu.pci_slot:
                log.info("evicting %s before starting %s", name, target.name)
                srv.stop()

    async def stop_one(self, name: str) -> bool:
        """Stop a single model's llama-server. Returns True if it was running."""
        async with self._lock:
            srv = self._servers.get(name)
            if srv is None or not srv.is_running:
                return False
            srv.stop()
            return True

    async def stop_all(self) -> int:
        """Stop every running llama-server. Returns the count stopped."""
        async with self._lock:
            stopped = 0
            for srv in self._servers.values():
                if srv.is_running:
                    srv.stop()
                    stopped += 1
            return stopped

    async def rebuild_model(self, name: str) -> tuple[bool, bool]:
        """Drop and rebuild the LlamaServer for one model after a config edit.

        If the model is currently loaded, it's stopped first — the recipe is
        consumed at process start, so an in-flight server can't pick up new
        flags. Returns (rebuilt, was_running).
        """
        async with self._lock:
            old = self._servers.pop(name, None)
            was_running = bool(old and old.is_running)
            if old is not None and old.is_running:
                old.stop()
            cfg_model = next((m for m in self.cfg.models if m.name == name), None)
            if cfg_model is None:
                return False, was_running
            gpu = self.cfg.find_gpu(cfg_model.gpu_pci_slot)
            if gpu is None:
                return False, was_running
            plan = build_plan(self.cfg, cfg_model, gpu, host=self.cfg.server.host)
            self._servers[name] = LlamaServer(plan, name=name)
            return True, was_running

    async def shutdown(self) -> None:
        async with self._lock:
            for srv in self._servers.values():
                srv.stop()
