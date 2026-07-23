from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, PathsConfig


def _cfg(models_dir: Path) -> Config:
    models_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        paths=PathsConfig(models_dir=str(models_dir)),
        gpus=[
            GPUConfig(
                pci_slot="0000:03:00.0",
                sycl_index=0,
                arch="battlemage",
                vram_mb=24576,
                enabled=True,
                name="Arc Pro B60",
            )
        ],
        models=[],
    )


def _fake_app() -> MagicMock:
    app = MagicMock()
    app.state.router = None  # serve's atexit shutdown no-ops on None
    return app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_serve_auto_registers_new_gguf(runner, tmp_path, monkeypatch):
    """serve discovers a GGUF dropped into models_dir since the last run."""
    models_dir = tmp_path / "models"
    cfg = _cfg(models_dir)
    (models_dir / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"fake")
    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    with (
        patch("arc_llama.models.has_mtp_heads", return_value=False),
        patch("arc_llama.models.is_moe", return_value=False),
        patch("arc_llama.server.create_app", return_value=_fake_app()),
        patch("uvicorn.run"),
        patch("signal.signal"),
        patch("atexit.register"),
    ):
        result = runner.invoke(
            cli, ["--config", str(tmp_path / "config.toml"), "serve"]
        )

    assert result.exit_code == 0, result.output
    assert "Auto-registered" in result.output
    assert len(cfg.models) == 1


def test_serve_no_scan_skips_discovery(runner, tmp_path, monkeypatch):
    """--no-scan leaves the registry untouched even with a GGUF present."""
    models_dir = tmp_path / "models"
    cfg = _cfg(models_dir)
    (models_dir / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"fake")
    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    with (
        patch("arc_llama.models.has_mtp_heads", return_value=False),
        patch("arc_llama.models.is_moe", return_value=False),
        patch("arc_llama.server.create_app", return_value=_fake_app()),
        patch("uvicorn.run"),
        patch("signal.signal"),
        patch("atexit.register"),
    ):
        result = runner.invoke(
            cli, ["--config", str(tmp_path / "config.toml"), "serve", "--no-scan"]
        )

    assert result.exit_code == 0, result.output
    assert "Auto-registered" not in result.output
    assert cfg.models == []


def test_serve_scan_is_idempotent(runner, tmp_path, monkeypatch):
    """A GGUF already registered is not added again on the next serve."""
    models_dir = tmp_path / "models"
    cfg = _cfg(models_dir)
    (models_dir / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"fake")
    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    patches = (
        patch("arc_llama.models.has_mtp_heads", return_value=False),
        patch("arc_llama.models.is_moe", return_value=False),
        patch("arc_llama.server.create_app", return_value=_fake_app()),
        patch("uvicorn.run"),
        patch("signal.signal"),
        patch("atexit.register"),
    )
    for _ in range(2):
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = runner.invoke(
                cli, ["--config", str(tmp_path / "config.toml"), "serve"]
            )
            assert result.exit_code == 0, result.output

    assert len(cfg.models) == 1  # not duplicated on the second run
