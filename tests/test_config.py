from __future__ import annotations

from pathlib import Path

import pytest

from arc_llama.config import Config, GPUConfig, ModelConfig, load_config


def test_config_round_trips_models_gpus_and_paths(tmp_path):
    path = tmp_path / "config.toml"
    cfg = Config()
    cfg.paths.llama_server = "/opt/llama.cpp/llama-server"
    cfg.paths.scan_paths = [str(tmp_path / "extra-models")]
    cfg.gpus = [
        GPUConfig(
            pci_slot="0000:03:00.0",
            sycl_index=0,
            arch="battlemage",
            vram_mb=24576,
            name="Arc Pro B60",
        )
    ]
    cfg.models = [
        ModelConfig(
            name="qwen",
            path=str(tmp_path / "Qwen3-7B-Q4_K_M.gguf"),
            port=18080,
            gpu_pci_slot="0000:03:00.0",
            display_name="Qwen 3 7B",
            aliases=["Qwen3-7B-Q4_K_M.gguf"],
            recipe={"ctx": 32768, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
        )
    ]

    cfg.save(path)
    loaded = load_config(path)

    assert loaded.paths.llama_server == "/opt/llama.cpp/llama-server"
    assert loaded.paths.scan_paths == [str(tmp_path / "extra-models")]
    assert loaded.gpus[0].name == "Arc Pro B60"
    assert loaded.models[0].recipe["ctx"] == 32768


def test_find_model_matches_name_alias_display_name_and_filename(tmp_path):
    cfg = Config(
        models=[
            ModelConfig(
                name="qwen",
                path=str(tmp_path / "Qwen3-7B-Q4_K_M.gguf"),
                port=18080,
                gpu_pci_slot="0000:03:00.0",
                display_name="Qwen 3 7B",
                aliases=["chat-default"],
            )
        ]
    )

    assert cfg.find_model("qwen").name == "qwen"
    assert cfg.find_model("chat-default").name == "qwen"
    assert cfg.find_model("qwen 3").name == "qwen"
    assert cfg.find_model("Q4_K_M").name == "qwen"
    assert cfg.find_model("missing") is None


def test_windows_default_paths_use_appdata(monkeypatch):
    import os

    from arc_llama import config as config_mod

    monkeypatch.setattr(config_mod.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", r"C:\Users\test\AppData\Roaming")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\test\AppData\Local")
    appdata = Path(os.environ["APPDATA"])
    localappdata = Path(os.environ["LOCALAPPDATA"])
    assert config_mod.default_config_path() == appdata / "arc-llama" / "config.toml"
    assert config_mod.default_models_dir() == localappdata / "arc-llama" / "models"
    assert config_mod.default_state_dir() == localappdata / "arc-llama"


def test_migrate_config_adds_missing_sections():
    from arc_llama.config import CONFIG_VERSION, migrate_config

    raw = migrate_config({})
    assert raw["version"] == CONFIG_VERSION
    assert raw["server"] == {}
    assert raw["paths"] == {}
    assert raw["gpus"] == []
    assert raw["models"] == []
    assert raw["upstreams"] == []


def test_validate_config_rejects_bad_structure():
    from arc_llama.config import validate_config

    with pytest.raises(ValueError, match="version"):
        validate_config({"version": "not-an-int"})
    with pytest.raises(ValueError, match="gpus"):
        validate_config({"version": 1, "gpus": {}})
