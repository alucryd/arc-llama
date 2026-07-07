from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, PathsConfig


def _make_test_config(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(models_dir=str(tmp_path / "models")),
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
    )


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_init_fails_when_llama_server_binary_missing(monkeypatch, tmp_path, cli_runner):
    """`arc-llama init` must exit with a clear error if the binary is missing."""
    config_file = tmp_path / "config.toml"
    cfg = _make_test_config(tmp_path)

    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)
    monkeypatch.setattr("arc_llama.cli.default_config_path", lambda: config_file)
    monkeypatch.setattr("arc_llama.cli.detect_gpus", lambda: cfg.gpus)
    monkeypatch.setattr(
        "arc_llama.cli.init_config_from_detection",
        lambda gpus, llama_server_path: cfg,
    )

    result = cli_runner.invoke(
        cli,
        [
            "--config", str(config_file), "init", "--no-scan",
            "--llama-server", "/nonexistent/llama-server",
        ],
    )
    assert result.exit_code == 3
    assert "llama-server binary not found" in result.output
    assert "/nonexistent/llama-server" in result.output
