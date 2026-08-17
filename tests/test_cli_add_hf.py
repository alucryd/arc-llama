from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, PathsConfig


def _make_test_config(tmp_path: Path) -> Config:
    """Build a minimal config with one GPU for CLI tests."""
    cfg = Config(
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
    return cfg


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_cli_add_from_hf_success(monkeypatch, tmp_path, cli_runner):
    """Successful add from HF: download mocked, config updated."""
    cfg = _make_test_config(tmp_path)
    config_file = tmp_path / "config.toml"
    config_file.write_text("")

    # Mock load_config
    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    # Mock download_from_hf to return a fake file
    fake_model = tmp_path / "models" / "repo" / "model-Q4_K_M.gguf"
    fake_model.parent.mkdir(parents=True)
    fake_model.write_bytes(b"fake gguf")

    def mock_download(spec, *, target_dir, token=None, progress=True):
        return fake_model

    monkeypatch.setattr("arc_llama.cli.download_from_hf", mock_download)

    # Mock add_local_model
    mock_mc = MagicMock()
    mock_mc.name = "repo-q4_k_m"
    mock_mc.port = 18080
    monkeypatch.setattr("arc_llama.cli.add_local_model", lambda *a, **kw: mock_mc)

    # Mock _save_or_die so we don't write to real config
    saved = []
    monkeypatch.setattr("arc_llama.cli._save_or_die", lambda c, p: saved.append((c, p)))

    result = cli_runner.invoke(
        cli,
        [
            "-c",
            str(config_file),
            "add",
            "--from-hf",
            "org/repo:Q4_K_M",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Registered repo-q4_k_m" in result.output
    assert len(saved) == 1


def test_cli_add_from_hf_download_failure(monkeypatch, tmp_path, cli_runner):
    """When download_from_hf raises, CLI should print error and exit 1."""
    cfg = _make_test_config(tmp_path)
    config_file = tmp_path / "config.toml"
    config_file.write_text("")

    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    def mock_download(*a, **kw):
        raise RuntimeError("network timeout")

    monkeypatch.setattr("arc_llama.cli.download_from_hf", mock_download)

    result = cli_runner.invoke(
        cli,
        [
            "-c",
            str(config_file),
            "add",
            "--from-hf",
            "org/repo:Q4_K_M",
        ],
    )

    assert result.exit_code == 1
    assert "network timeout" in result.output


def test_cli_add_from_hf_with_name_override(monkeypatch, tmp_path, cli_runner):
    """--name should be passed through to add_local_model."""
    cfg = _make_test_config(tmp_path)
    config_file = tmp_path / "config.toml"
    config_file.write_text("")

    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    fake_model = tmp_path / "models" / "repo" / "model.gguf"
    fake_model.parent.mkdir(parents=True)
    fake_model.write_bytes(b"fake")

    monkeypatch.setattr(
        "arc_llama.cli.download_from_hf",
        lambda spec, *, target_dir, token=None, progress=True: fake_model,
    )

    captured = {}

    def capture_add_local_model(*args, **kwargs):
        captured.update(kwargs)
        mock_mc = MagicMock()
        mock_mc.name = kwargs.get("name", "unknown")
        mock_mc.port = 18080
        mock_recipe = MagicMock()
        mock_recipe.ctx = 8192
        mock_recipe.ubatch_size = None
        mock_recipe.batch_size = None
        mock_mc.launch_recipe.return_value = mock_recipe
        return mock_mc

    monkeypatch.setattr("arc_llama.cli.add_local_model", capture_add_local_model)
    monkeypatch.setattr("arc_llama.cli._save_or_die", lambda c, p: None)

    result = cli_runner.invoke(
        cli,
        [
            "-c",
            str(config_file),
            "add",
            "--from-hf",
            "org/repo",
            "--name",
            "my-custom-name",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured.get("name") == "my-custom-name"
    assert "Registered my-custom-name" in result.output


def test_cli_add_from_hf_with_batch_size_override(monkeypatch, tmp_path, cli_runner):
    """--batch-size should be passed as a recipe override."""
    cfg = _make_test_config(tmp_path)
    config_file = tmp_path / "config.toml"
    config_file.write_text("")

    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    fake_model = tmp_path / "models" / "repo" / "model.gguf"
    fake_model.parent.mkdir(parents=True)
    fake_model.write_bytes(b"fake")

    monkeypatch.setattr(
        "arc_llama.cli.download_from_hf",
        lambda spec, *, target_dir, token=None, progress=True: fake_model,
    )

    captured = {}

    def capture_add_local_model(*args, **kwargs):
        captured.update(kwargs)
        mock_mc = MagicMock()
        mock_mc.name = "repo"
        mock_mc.port = 18080
        mock_recipe = MagicMock()
        mock_recipe.ctx = 8192
        mock_recipe.ubatch_size = None
        mock_recipe.batch_size = kwargs.get("recipe_overrides", {}).get("batch_size")
        mock_mc.launch_recipe.return_value = mock_recipe
        return mock_mc

    monkeypatch.setattr("arc_llama.cli.add_local_model", capture_add_local_model)
    monkeypatch.setattr("arc_llama.cli._save_or_die", lambda c, p: None)

    result = cli_runner.invoke(
        cli,
        [
            "-c",
            str(config_file),
            "add",
            "--from-hf",
            "org/repo",
            "--batch-size",
            "4096",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured.get("recipe_overrides", {}).get("batch_size") == 4096
    assert "b=4096" in result.output


def test_cli_add_from_hf_file_not_found_after_download(monkeypatch, tmp_path, cli_runner):
    """If download succeeds but file is missing, add_local_model raises FileNotFoundError."""
    cfg = _make_test_config(tmp_path)
    config_file = tmp_path / "config.toml"
    config_file.write_text("")

    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    # Return a path that does NOT exist
    missing_path = tmp_path / "models" / "repo" / "missing.gguf"

    monkeypatch.setattr(
        "arc_llama.cli.download_from_hf",
        lambda spec, *, target_dir, token=None, progress=True: missing_path,
    )

    result = cli_runner.invoke(
        cli,
        [
            "-c",
            str(config_file),
            "add",
            "--from-hf",
            "org/repo:Q4_K_M",
        ],
    )

    # add_local_model will raise FileNotFoundError which is caught and exits 1
    assert result.exit_code == 1
    assert "File not found" in result.output or "Model file not found" in result.output


def test_cli_add_from_hf_parses_spec_correctly(monkeypatch, tmp_path, cli_runner):
    """Verify that the CLI parses the HF spec and passes it to download_from_hf."""
    cfg = _make_test_config(tmp_path)
    config_file = tmp_path / "config.toml"
    config_file.write_text("")

    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    fake_model = tmp_path / "models" / "repo" / "model-Q4_K_M.gguf"
    fake_model.parent.mkdir(parents=True)
    fake_model.write_bytes(b"fake")

    captured_spec = {}

    def mock_download(spec, *, target_dir, token=None, progress=True):
        captured_spec["repo"] = spec.repo
        captured_spec["quant"] = spec.quant
        return fake_model

    monkeypatch.setattr("arc_llama.cli.download_from_hf", mock_download)
    monkeypatch.setattr(
        "arc_llama.cli.add_local_model", lambda *a, **kw: MagicMock(name="test", port=18080)
    )
    monkeypatch.setattr("arc_llama.cli._save_or_die", lambda c, p: None)

    result = cli_runner.invoke(
        cli,
        [
            "-c",
            str(config_file),
            "add",
            "--from-hf",
            "org/repo:Q4_K_M",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured_spec["repo"] == "org/repo"
    assert captured_spec["quant"] == "Q4_K_M"


def test_cli_add_from_hf_invalid_spec(monkeypatch, tmp_path, cli_runner):
    """Invalid HF spec should show error and exit 1."""
    cfg = _make_test_config(tmp_path)
    config_file = tmp_path / "config.toml"
    config_file.write_text("")

    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    result = cli_runner.invoke(
        cli,
        [
            "-c",
            str(config_file),
            "add",
            "--from-hf",
            "not-a-valid-spec",
        ],
    )

    assert result.exit_code == 1
    assert "Invalid HF spec" in result.output


def test_cli_add_from_hf_with_token(monkeypatch, tmp_path, cli_runner):
    """--hf-token should be passed to download_from_hf."""
    cfg = _make_test_config(tmp_path)
    config_file = tmp_path / "config.toml"
    config_file.write_text("")

    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    fake_model = tmp_path / "models" / "repo" / "model.gguf"
    fake_model.parent.mkdir(parents=True)
    fake_model.write_bytes(b"fake")

    captured = {}

    def mock_download(spec, *, target_dir, token=None, progress=True):
        captured["token"] = token
        return fake_model

    monkeypatch.setattr("arc_llama.cli.download_from_hf", mock_download)
    monkeypatch.setattr(
        "arc_llama.cli.add_local_model", lambda *a, **kw: MagicMock(name="test", port=18080)
    )
    monkeypatch.setattr("arc_llama.cli._save_or_die", lambda c, p: None)

    result = cli_runner.invoke(
        cli,
        [
            "-c",
            str(config_file),
            "add",
            "--from-hf",
            "org/repo",
            "--hf-token",
            "hf_secret_123",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured.get("token") == "hf_secret_123"
