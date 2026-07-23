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
        models=[],
    )


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_benchmark_rejects_unknown_model(monkeypatch, tmp_path, cli_runner):
    """`arc-llama benchmark` must exit cleanly when the model isn't registered."""
    config_file = tmp_path / "config.toml"
    cfg = _make_test_config(tmp_path)
    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    result = cli_runner.invoke(
        cli,
        ["--config", str(config_file), "benchmark", "not-a-model"],
    )
    assert result.exit_code == 1
    assert "not registered" in result.output
