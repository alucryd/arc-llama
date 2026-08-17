"""End-to-end inference smoke test.

This test is skipped by default. To run it locally on a machine with a model
and GPU:

    ARC_LLAMA_SMOKE_MODEL=qwen2.5-0.5b-instruct-q4_k_m pytest tests/test_smoke.py

It starts a real `arc-llama serve`, loads the requested model, sends a chat
completion, and verifies a non-empty response. It will not run in CI because
GitHub Actions has no Intel Arc GPU.
"""
from __future__ import annotations

import json
import os
import pwd
import signal
import subprocess
import sys
import time

import pytest

_SKIP_REASON = "Set ARC_LLAMA_SMOKE_MODEL to run inference smoke test"


def _real_home() -> str:
    """Return the real user's home, ignoring any test-isolated HOME env var."""
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except KeyError:
        return os.path.expanduser("~")


def _wait_for_server(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = subprocess.run(
                ["curl", "-fsS", f"{url}/v1/models"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        time.sleep(1)
    raise RuntimeError(f"arc-llama serve did not become ready at {url}")


@pytest.mark.skipif(not os.environ.get("ARC_LLAMA_SMOKE_MODEL"), reason=_SKIP_REASON)
def test_inference_smoke() -> None:
    model = os.environ["ARC_LLAMA_SMOKE_MODEL"]
    config_path = os.environ.get(
        "ARC_LLAMA_SMOKE_CONFIG",
        os.path.join(_real_home(), ".config", "arc-llama", "config.toml"),
    )
    server_url = os.environ.get("ARC_LLAMA_SMOKE_URL", "http://127.0.0.1:11436")

    real_home = _real_home()
    env = os.environ.copy()
    env["HOME"] = real_home
    env["USERPROFILE"] = real_home
    env["XDG_CONFIG_HOME"] = os.path.join(real_home, ".config")
    env["APPDATA"] = env["XDG_CONFIG_HOME"]
    env["LOCALAPPDATA"] = os.path.join(real_home, "AppData", "Local")
    # Make sure the subprocess uses the same Python/venv as the test runner.
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "arc_llama.cli",
            "-c",
            config_path,
            "serve",
            "--no-scan",
            "--no-auto-tune",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    try:
        _wait_for_server(server_url)

        result = subprocess.run(
            [
                "curl",
                "-fsS",
                f"{server_url}/v1/chat/completions",
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": "Say hello"}],
                        "max_tokens": 16,
                    }
                ),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        content = data["choices"][0]["message"]["content"]
        assert content and isinstance(content, str)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
