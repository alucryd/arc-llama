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
import time
from pathlib import Path
from typing import Any

from arc_llama.config import Config, GPUConfig, ModelConfig
from arc_llama.gguf_meta import estimate_weight_vram_bytes
from arc_llama.launcher import LlamaServer, build_plan
from arc_llama.recipes import KVCacheType, estimate_kv_bytes

log = logging.getLogger("arc_llama.router")

# Rough overhead budgets for VRAM estimation (MiB).
_VRAM_COMPUTE_BUFFER_MB = 768
_VRAM_SAFETY_MARGIN_MB = 256


def _estimate_model_vram_mb(model: ModelConfig) -> int:
    """Rough VRAM footprint for one model instance.

    Uses GGUF tensor metadata to estimate the decompressed weight footprint,
    which is much closer to reality for heavily quantized files than the raw
    file size. Falls back to file size if the GGUF cannot be read.
    """
    path = Path(model.path)
    weight_bytes = estimate_weight_vram_bytes(path)
    if weight_bytes is None:
        try:
            weight_bytes = path.stat().st_size
        except OSError:
            weight_bytes = 0
        log.debug(
            "VRAM estimate for %s falling back to file size: %.0f MiB",
            model.name, weight_bytes / (1_048_576),
        )
    weight_mb = weight_bytes // (1_048_576)
    recipe = model.recipe or {}
    ctx = int(recipe.get("ctx", 8192))
    kv_type = KVCacheType(recipe.get("cache_type_k", "f16"))
    kv_mb = estimate_kv_bytes(ctx, kv_type, model.kv_class) // (1_048_576)
    return weight_mb + kv_mb + _VRAM_COMPUTE_BUFFER_MB + _VRAM_SAFETY_MARGIN_MB


class Router:
    """Owns one LlamaServer per registered model and serialises swaps."""

    def __init__(self, cfg: Config, log_dir: Path | None = None):
        self.cfg = cfg
        self.log_dir = log_dir
        self._servers: dict[str, LlamaServer] = {}  # keyed by model.name
        self._lock = asyncio.Lock()
        self._loading_futures: dict[str, asyncio.Future[tuple[ModelConfig, LlamaServer]]] = {}
        self.metrics: dict[str, Any] = {
            "loads": 0,
            "stops": 0,
            "load_errors": 0,
            "last_load_at": None,
            "last_error": None,
        }
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

        Fast-path: if the model is already running and no eviction is needed,
        return immediately without acquiring the swap lock.

        Concurrent requests for the same model that is currently loading will
        wait on a shared future instead of each trying to start a new process.
        """
        # Fast-path: already running → no lock, no eviction, no start.
        fast = self.resolve(query)
        if fast is not None:
            target_model, target_gpu, target_srv = fast
            if target_srv.is_running:
                # Verify policy: if single-resident, we are the one; if multi,
                # same-GPU contention would have been resolved when we started.
                return target_model, target_srv

        # Slow path: may need to swap / start. Serialize with the lock.
        async with self._lock:
            # Another task may have finished loading while we waited.
            resolved = self.resolve(query)
            if resolved is None:
                raise KeyError(f"Unknown model: {query!r}")
            target_model, target_gpu, target_srv = resolved

            # If someone else is already loading this model, wait on them.
            existing_future = self._loading_futures.get(target_model.name)
            if existing_future is not None:
                return await existing_future

            await self._evict_for(target_model, target_gpu)

            if target_srv.is_running:
                return target_model, target_srv

            self._check_vram_fit(target_model, target_gpu)

            # We are the one responsible for starting.
            log.info("loading model %s on GPU %s ...", target_model.name, target_gpu.pci_slot)
            loop = asyncio.get_running_loop()
            future: asyncio.Future[tuple[ModelConfig, LlamaServer]] = loop.create_future()
            self._loading_futures[target_model.name] = future
            try:
                target_srv.start(log_dir=self.log_dir)
                ready = await target_srv.wait_ready()
                if not ready:
                    tail = target_srv.tail_log(lines=40)
                    log.error(
                        "model %s failed health-check; stopping it",
                        target_model.name,
                    )
                    target_srv.stop()
                    self.metrics["load_errors"] += 1
                    self.metrics["last_error"] = f"{target_model.name} did not become healthy"
                    detail = f"llama-server for {target_model.name} did not become healthy"
                    if tail:
                        detail += "\n\n--- last log lines ---\n" + tail
                    raise RuntimeError(detail)
                self.metrics["loads"] += 1
                self.metrics["last_load_at"] = time.time()
                self.metrics["last_error"] = None
                result = (target_model, target_srv)
                future.set_result(result)
                return result
            except Exception as exc:
                if not future.done():
                    self.metrics["load_errors"] += 1
                    self.metrics["last_error"] = str(exc)
                    future.set_exception(RuntimeError(
                        f"llama-server for {target_model.name} did not become healthy"
                    ))
                raise
            finally:
                self._loading_futures.pop(target_model.name, None)

    def _check_vram_fit(self, target: ModelConfig, target_gpu: GPUConfig) -> None:
        """Refuse to load *target* if its estimated VRAM won't fit on target_gpu.

        In multi-resident mode this also accounts for other loaded models that
        share the same GPU.
        """
        if not target_gpu.vram_mb:
            return
        used_mb = _estimate_model_vram_mb(target)
        for name, srv in self._servers.items():
            if name == target.name or not srv.is_running:
                continue
            other = next((m for m in self.cfg.models if m.name == name), None)
            if other is None or other.gpu_pci_slot != target_gpu.pci_slot:
                continue
            used_mb += _estimate_model_vram_mb(other)
        if used_mb > target_gpu.vram_mb:
            target_mb = _estimate_model_vram_mb(target)
            raise RuntimeError(
                f"model {target.name!r} needs ~{target_mb} MiB on GPU "
                f"{target_gpu.pci_slot} but only {target_gpu.vram_mb} MiB is available "
                f"(estimated total with co-residents: {used_mb} MiB)"
            )

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
            self.metrics["stops"] += 1
            return True

    async def stop_all(self) -> int:
        """Stop every running llama-server. Returns the count stopped."""
        async with self._lock:
            stopped = 0
            for srv in self._servers.values():
                if srv.is_running:
                    srv.stop()
                    stopped += 1
            self.metrics["stops"] += stopped
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
