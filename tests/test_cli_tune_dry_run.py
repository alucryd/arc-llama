"""`arc-llama tune --dry-run` must not record the model as tuned.

Both branches of tune_cmd called set_tuned_state() and saved the config even
when apply_ was False. The recipe itself was restored, so the run looked
side-effect free, but the model was now marked tuned with a matching
fingerprint and background auto-tune skipped it forever. A look-don't-touch
run became a permanent opt-out.

No GPU needed: tune_model/tune_all are stubbed with canned reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from click.testing import CliRunner

from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, ModelConfig, PathsConfig


@dataclass
class _Report:
    model: str = "qwen"
    error: str | None = None
    aborted: bool = False
    stages: list = field(default_factory=list)
    baseline_toks: float = 10.0
    best_toks: float = 12.0


def _cfg(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(models_dir=str(tmp_path / "models")),
        gpus=[
            GPUConfig(
                pci_slot="0000:03:00.0",
                sycl_index=0,
                arch="battlemage",
                vram_mb=24576,
            )
        ],
        models=[
            ModelConfig(
                name="qwen",
                path=str(tmp_path / "qwen.gguf"),
                port=18080,
                gpu_pci_slot="0000:03:00.0",
            )
        ],
    )


@pytest.fixture
def runner():
    return CliRunner()


def _invoke_tune(monkeypatch, tmp_path, runner, extra_args):
    cfg = _cfg(tmp_path)
    saved: list[Path] = []
    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)
    monkeypatch.setattr(Config, "save", lambda self, p=None: saved.append(p) or p)

    async def fake_tune_model(url, model, **kw):
        return _Report(model=model)

    async def fake_tune_all(url, names, **kw):
        return [_Report(model=n) for n in names]

    monkeypatch.setattr("arc_llama.tune.tune_model", fake_tune_model)
    monkeypatch.setattr("arc_llama.tune.tune_all", fake_tune_all)
    # print_report/print_multi_summary render rich tables from real reports;
    # keep them out of the way.
    monkeypatch.setattr("arc_llama.tune.print_report", lambda r: None, raising=False)
    monkeypatch.setattr("arc_llama.tune.print_multi_summary", lambda r: None, raising=False)

    result = runner.invoke(cli, ["--config", str(tmp_path / "config.toml"), "tune", *extra_args])
    return cfg, saved, result


def test_dry_run_leaves_tune_state_untouched(monkeypatch, tmp_path, runner):
    cfg, saved, result = _invoke_tune(monkeypatch, tmp_path, runner, ["qwen", "--dry-run"])
    assert result.exit_code == 0, result.output
    model = cfg.models[0]
    assert model.tune_state == "untuned", f"dry-run recorded tune state: {model.tune_state}"
    assert not saved, "dry-run persisted the config"


def test_dry_run_all_leaves_tune_state_untouched(monkeypatch, tmp_path, runner):
    cfg, saved, result = _invoke_tune(monkeypatch, tmp_path, runner, ["--all", "--dry-run"])
    assert result.exit_code == 0, result.output
    model = cfg.models[0]
    assert model.tune_state == "untuned", f"dry-run --all recorded tune state: {model.tune_state}"
    assert not saved, "dry-run --all persisted the config"


def test_apply_still_records_tune_state(monkeypatch, tmp_path, runner):
    """The guard must not break the normal path."""
    cfg, saved, result = _invoke_tune(monkeypatch, tmp_path, runner, ["qwen"])
    assert result.exit_code == 0, result.output
    model = cfg.models[0]
    assert model.tune_state == "tuned", "apply run failed to record tune state"
    assert saved, "apply run failed to persist the config"
