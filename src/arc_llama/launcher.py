"""Manage llama-server subprocesses.

A `LlamaServer` owns one llama-server process bound to one model on one GPU.
It builds the command line from an arch profile + recipe + model config so the
SYCL gotchas are applied uniformly.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from arc_llama.arch import Arch, ArchProfile, profile_for
from arc_llama.config import Config, GPUConfig, ModelConfig

log = logging.getLogger("arc_llama.launcher")

DEFAULT_HEALTH_TIMEOUT = 120  # seconds — generous for cold-start SYCL JIT
HEALTH_POLL_INTERVAL = 1.5


@dataclass
class LaunchPlan:
    """Everything needed to invoke llama-server for one model."""
    argv: list[str]
    env: dict[str, str]
    cwd: str | None = None
    health_url: str = ""
    backend_url: str = ""


def build_env(profile: ArchProfile, sycl_index: int) -> dict[str, str]:
    """Compose the environment, layering arch defaults over the user's shell env."""
    env = os.environ.copy()
    # Strip env vars known to break this arch (even if the user inherited them).
    for k in profile.sycl_env_remove:
        env.pop(k, None)
    # Apply arch-recommended values, but override the device selector with the
    # specific GPU index this model is bound to.
    env.update(profile.sycl_env)
    env["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{sycl_index}"
    return env


def build_plan(
    cfg: Config, model: ModelConfig, gpu: GPUConfig, host: str = "127.0.0.1"
) -> LaunchPlan:
    arch = Arch(gpu.arch) if gpu.arch else Arch.UNKNOWN
    profile = profile_for(arch)
    env = build_env(profile, gpu.sycl_index)
    recipe = model.launch_recipe()
    argv: list[str] = [
        cfg.paths.llama_server,
        "-m", model.path,
        "--host", host,
        "--port", str(model.port),
    ]
    argv.extend(recipe.to_argv())
    backend_url = f"http://{host}:{model.port}"
    return LaunchPlan(
        argv=argv,
        env=env,
        backend_url=backend_url,
        health_url=f"{backend_url}/health",
    )


class LlamaServer:
    """One llama-server subprocess. Lifecycle: start → wait_ready → stop."""

    def __init__(self, plan: LaunchPlan, name: str = "llama-server"):
        self.plan = plan
        self.name = name
        self.process: subprocess.Popen[bytes] | None = None
        self.started_at: float | None = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, log_dir: Path | None = None) -> None:
        if self.is_running:
            log.debug("[%s] already running, pid=%s", self.name, self.process.pid)  # type: ignore[union-attr]
            return
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{self.name}.log"
            stdout = open(log_path, "ab")
            stderr = subprocess.STDOUT
        log.info("[%s] starting: %s", self.name, " ".join(self.plan.argv))
        self.process = subprocess.Popen(
            self.plan.argv,
            env=self.plan.env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        self.started_at = time.time()

    async def wait_ready(self, timeout: float = DEFAULT_HEALTH_TIMEOUT) -> bool:
        deadline = time.time() + timeout
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.time() < deadline:
                if not self.is_running:
                    log.warning("[%s] process exited before becoming healthy", self.name)
                    return False
                try:
                    r = await client.get(self.plan.health_url)
                    if r.status_code == 200 and r.json().get("status") == "ok":
                        return True
                except Exception:
                    pass
                await asyncio.sleep(HEALTH_POLL_INTERVAL)
        return False

    def stop(self, drain_seconds: float = 3.0) -> None:
        if not self.is_running:
            return
        proc = self.process
        assert proc is not None
        log.info("[%s] stopping pid=%s", self.name, proc.pid)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=drain_seconds)
        except subprocess.TimeoutExpired:
            log.warning("[%s] SIGTERM timed out, sending SIGKILL", self.name)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=drain_seconds)
            except subprocess.TimeoutExpired:
                pass
        self.process = None
        self.started_at = None
