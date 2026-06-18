"""Manage llama-server subprocesses.

A `LlamaServer` owns one llama-server process bound to one model on one GPU.
It builds the command line from an arch profile + recipe + model config so the
SYCL gotchas are applied uniformly.

Subprocesses are launched with `setsid` (so they own a fresh process group we
can `killpg`) AND `PR_SET_PDEATHSIG=SIGTERM` so the kernel reaps the child if
the arc-llama parent dies hard — no orphan llama-servers holding VRAM.
"""
from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from arc_llama.arch import Arch, ArchProfile, profile_for
from arc_llama.config import Config, GPUConfig, ModelConfig
from arc_llama.gguf_meta import has_mtp_heads, is_hybrid_ssm

log = logging.getLogger("arc_llama.launcher")

DEFAULT_HEALTH_TIMEOUT = 120  # seconds — generous for cold-start SYCL JIT
HEALTH_POLL_INTERVAL = 1.5

# Log rotation for llama-server subprocess logs.
_MAX_LOG_BYTES = 50 * 1024 * 1024
_LOG_BACKUPS = 3

_IS_WINDOWS = sys.platform == "win32"
# Not defined on POSIX. Fallback lets the Windows code path stay import-safe
# when exercised under tests that monkeypatch _IS_WINDOWS on a Linux runner.
_CTRL_BREAK_EVENT = getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM)

# Linux prctl(2) constant. We don't import a real binding — one syscall.
_PR_SET_PDEATHSIG = 1


_libc: ctypes.CDLL | None = None


def _load_libc() -> ctypes.CDLL | None:
    global _libc
    if _libc is not None:
        return _libc
    try:
        lib = ctypes.CDLL("libc.so.6", use_errno=True)
        # int prctl(int option, unsigned long arg2, ...arg5)
        lib.prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong,
        ]
        lib.prctl.restype = ctypes.c_int
        _libc = lib
    except OSError:
        _libc = None
    return _libc


def _preexec_isolate_and_pdeathsig() -> None:
    """preexec_fn: detach into a new session and tie our lifetime to the parent's.

    Runs in the child between fork and exec. setsid() makes the child a new
    process-group/session leader (so we can `killpg` it cleanly), prctl with
    PR_SET_PDEATHSIG ensures the kernel sends us SIGTERM the moment the parent
    arc-llama process exits — covers crashes, SIGKILL, oom-killer, etc.

    NOTE: PDEATHSIG tracks the *thread* that did the fork, not the whole parent
    process. If Python's main thread dies but a worker thread is what spawned us,
    we wouldn't get the signal. arc-llama spawns from the asyncio loop running
    in the main thread, so this is a non-issue today, but a future move to a
    thread-pool launcher would need rework.
    """
    os.setsid()
    libc = _load_libc()
    if libc is None:
        return
    rc = libc.prctl(_PR_SET_PDEATHSIG, ctypes.c_ulong(int(signal.SIGTERM)), 0, 0, 0)
    if rc != 0:
        # Can't really log here — preexec_fn runs in a fragile post-fork state.
        # The child will simply not receive PDEATHSIG; not fatal.
        pass


def _rotate_log(log_path: Path) -> None:
    """Rotate an existing log file so it doesn't grow unbounded.

    Keeps up to ``_LOG_BACKUPS`` historic files (``.log.1``, ``.log.2``, ...).
    """
    if not log_path.exists():
        return
    try:
        if log_path.stat().st_size < _MAX_LOG_BYTES:
            return
    except OSError:
        return
    for i in range(_LOG_BACKUPS, 0, -1):
        src = log_path.parent / f"{log_path.name}.{i}"
        dst = log_path.parent / f"{log_path.name}.{i + 1}"
        if src.exists():
            try:
                src.replace(dst)
            except OSError:
                pass
    try:
        log_path.replace(log_path.parent / f"{log_path.name}.1")
    except OSError:
        pass


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

    # --- MTP head detection & safety wiring ---
    mtp_present = has_mtp_heads(model.path)
    hybrid_ssm = is_hybrid_ssm(model.path)

    # 1. Auto-inject -ub 8 for MTP models (prevents SSM compute-buffer OOM).
    if mtp_present and recipe.ubatch_size is None:
        recipe.ubatch_size = 8
        log.info(
            "[%s] MTP heads detected; auto-setting ubatch_size=8",
            model.name,
        )

    # 2. Warn if the user explicitly asked for draft-mtp on a model that
    #    does not actually contain MTP heads.
    if recipe.spec_type == "draft-mtp" and not mtp_present:
        log.warning(
            "[%s] recipe.spec_type='draft-mtp' but GGUF has no MTP heads "
            "(nextn_predict_layers == 0). Speculative decoding will likely "
            "degenerate or crash.",
            model.name,
        )

    # 3. Backend recommendation for hybrid SSM + MTP on Xe2 (Battlemage,
    #    Lunar Lake). GDN sequential state passes make SYCL MTP net-negative.
    if mtp_present and hybrid_ssm and arch in (Arch.BATTLEMAGE, Arch.LUNAR_LAKE):
        log.info(
            "[%s] Hybrid SSM+attention model with MTP heads on Xe2 (%s): "
            "SYCL MTP speculative decoding is net-negative here because GDN "
            "layers force serial state passes. Consider a Vulkan backend "
            "build for ~+9%% throughput with --spec-type draft-mtp.",
            model.name,
            arch.value,
        )

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
        self._log_file: Any = None  # file handle opened in start(), closed in stop()
        self._log_path: Path | None = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, log_dir: Path | None = None) -> None:
        if self.is_running:
            log.debug("[%s] already running, pid=%s", self.name, self.process.pid)  # type: ignore[union-attr]
            return
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
        self._log_path = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{self.name}.log"
            _rotate_log(log_path)
            self._log_path = log_path
            self._log_file = open(log_path, "ab")
            stdout = self._log_file
            stderr = subprocess.STDOUT
        log.info("[%s] starting: %s", self.name, " ".join(self.plan.argv))
        popen_kwargs: dict[str, Any] = {}
        if _IS_WINDOWS:
            # A new process group lets us terminate the whole subtree cleanly
            # without Unix-specific killpg. The constant is only defined on
            # Windows; the getattr guard keeps tests on Linux valid.
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            popen_kwargs["preexec_fn"] = _preexec_isolate_and_pdeathsig
        self.process = subprocess.Popen(
            self.plan.argv,
            env=self.plan.env,
            stdout=stdout,
            stderr=stderr,
            **popen_kwargs,
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

    def tail_log(self, lines: int = 50) -> str:
        """Return the last *lines* of the llama-server log, if any."""
        if self._log_path is None:
            return ""
        try:
            text = self._log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        all_lines = text.splitlines()
        return "\n".join(all_lines[-lines:])

    def stop(self, drain_seconds: float = 3.0) -> None:
        if not self.is_running:
            return
        proc = self.process
        assert proc is not None
        log.info("[%s] stopping pid=%s", self.name, proc.pid)
        if _IS_WINDOWS:
            # proc.terminate()/kill() both just call TerminateProcess on Windows —
            # there's no graceful/forceful distinction, and neither touches child
            # processes. CTRL_BREAK_EVENT goes to the whole CREATE_NEW_PROCESS_GROUP
            # (the closest equivalent to SIGTERM here); taskkill /T kills the whole
            # subtree, mirroring killpg on the Linux side below.
            try:
                proc.send_signal(_CTRL_BREAK_EVENT)
            except (ProcessLookupError, OSError):
                pass
            try:
                proc.wait(timeout=drain_seconds)
            except subprocess.TimeoutExpired:
                log.warning(
                    "[%s] CTRL_BREAK timed out, force-killing process tree", self.name
                )
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    timeout=drain_seconds,
                )
                try:
                    proc.wait(timeout=drain_seconds)
                except subprocess.TimeoutExpired:
                    pass
        else:
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
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
