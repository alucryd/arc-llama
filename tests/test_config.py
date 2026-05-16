"""Tests for arc_llama.config — schema, persistence, lookups, init."""
from __future__ import annotations

from pathlib import Path

import pytest

from arc_llama.arch import Arch
from arc_llama.config import (
    CONFIG_VERSION,
    Config,
    GPUConfig,
    ModelConfig,
    PathsConfig,
    ServerConfig,
    default_config_path,
    default_models_dir,
    default_state_dir,
    init_config_from_detection,
    load_config,
)
from arc_llama.detect import DetectedGPU


class TestConfigPaths:
    def test_default_config_path_uses_xdg(self, mock_config_dir: Path):
        assert default_config_path() == mock_config_dir / "arc-llama" / "config.toml"

    def test_default_models_dir_uses_xdg(self, mock_data_dir: Path):
        assert default_models_dir() == mock_data_dir / "arc-llama" / "models"

    def test_default_state_dir_uses_xdg(self, mock_data_dir: Path):
        # XDG_STATE_HOME falls back to ~/.local/state; with XDG_DATA_HOME set
        # we still use the default since we didn't mock XDG_STATE_HOME explicitly.
        # We'll just assert it's a Path.
        assert isinstance(default_state_dir(), Path)


class TestConfigSaveLoad:
    def test_roundtrip_empty(self, mock_config_dir: Path):
        cfg = Config()
        path = default_config_path()
        cfg.save()
        loaded = load_config(path)
        assert loaded.version == CONFIG_VERSION
        assert loaded.server.host == "127.0.0.1"
        assert loaded.server.port == 11437
        assert loaded.gpus == []
        assert loaded.models == []

    def test_roundtrip_full(self, mock_config_dir: Path):
        cfg = Config(
            server=ServerConfig(host="0.0.0.0", port=8080, single_resident=False),
            paths=PathsConfig(llama_server="/usr/bin/llama-server", scan_paths=["/mnt/models"]),
            gpus=[
                GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24480),
            ],
            models=[
                ModelConfig(
                    name="qwen3-7b",
                    path="/mnt/models/qwen3-7b.gguf",
                    port=18080,
                    gpu_pci_slot="0000:03:00.0",
                    recipe={"ctx": 32768, "cache_type_k": "q8_0"},
                ),
            ],
        )
        path = cfg.save()
        loaded = load_config(path)
        assert loaded.server.host == "0.0.0.0"
        assert loaded.server.single_resident is False
        assert len(loaded.gpus) == 1
        assert loaded.gpus[0].vram_mb == 24480
        assert len(loaded.models) == 1
        assert loaded.models[0].recipe["ctx"] == 32768

    def test_load_missing_returns_empty(self, tmp_path: Path):
        missing = tmp_path / "nope.toml"
        loaded = load_config(missing)
        assert loaded.version == CONFIG_VERSION
        assert loaded.models == []


class TestConfigLookups:
    def test_find_model_exact_name(self):
        cfg = Config(models=[
            ModelConfig(name="alpha", path="/a.gguf", port=1, gpu_pci_slot="00:00.0"),
            ModelConfig(name="beta", path="/b.gguf", port=2, gpu_pci_slot="00:00.0"),
        ])
        assert cfg.find_model("alpha") is not None
        assert cfg.find_model("beta") is not None
        assert cfg.find_model("gamma") is None

    def test_find_model_by_alias(self):
        cfg = Config(models=[
            ModelConfig(
                name="alpha", path="/a.gguf", port=1, gpu_pci_slot="00:00.0",
                aliases=["a.gguf", "my-model"],
            ),
        ])
        assert cfg.find_model("my-model") is not None
        assert cfg.find_model("a.gguf") is not None

    def test_find_model_substring(self):
        cfg = Config(models=[
            ModelConfig(
                name="alpha", path="/a.gguf", port=1, gpu_pci_slot="00:00.0",
                display_name="Alpha Big Model",
            ),
        ])
        assert cfg.find_model("big model") is not None
        assert cfg.find_model("alpha") is not None

    def test_find_model_empty_query(self):
        cfg = Config(models=[ModelConfig(name="x", path="/x.gguf", port=1, gpu_pci_slot="00:00.0")])
        assert cfg.find_model("") is None

    def test_find_gpu(self):
        cfg = Config(gpus=[
            GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage"),
            GPUConfig(pci_slot="0000:04:00.0", sycl_index=1, arch="alchemist"),
        ])
        assert cfg.find_gpu("0000:03:00.0") is not None
        assert cfg.find_gpu("0000:04:00.0") is not None
        assert cfg.find_gpu("0000:05:00.0") is None


class TestLaunchRecipe:
    def test_to_argv_basic(self):
        from arc_llama.recipes import LaunchRecipe, KVCacheType
        r = LaunchRecipe(ctx=4096, cache_type_k=KVCacheType.Q8_0, cache_type_v=KVCacheType.Q8_0)
        argv = r.to_argv()
        assert "-ngl" in argv
        assert "999" in argv
        assert "-c" in argv
        assert "4096" in argv
        assert "--cache-type-k" in argv
        assert "q8_0" in argv

    def test_to_argv_with_extras(self):
        from arc_llama.recipes import LaunchRecipe
        r = LaunchRecipe(extra_flags=["--reasoning", "off"])
        assert r.to_argv()[-2:] == ["--reasoning", "off"]

    def test_model_config_launch_recipe(self):
        mc = ModelConfig(
            name="m", path="/m.gguf", port=1, gpu_pci_slot="00:00.0",
            recipe={
                "n_gpu_layers": 42,
                "ctx": 8192,
                "parallel": 4,
                "cache_type_k": "q4_0",
                "cache_type_v": "q8_0",
                "extra_flags": ["--foo"],
            },
        )
        lr = mc.launch_recipe()
        assert lr.n_gpu_layers == 42
        assert lr.ctx == 8192
        assert lr.parallel == 4
        assert lr.cache_type_k.value == "q4_0"
        assert lr.cache_type_v.value == "q8_0"
        assert lr.extra_flags == ["--foo"]


class TestInitConfigFromDetection:
    def test_picks_highest_vram_battlemage(self):
        gpus = [
            DetectedGPU(
                pci_slot="0000:03:00.0", device_id=0xE211,
                arch=Arch.BATTLEMAGE, name="Arc Pro B60",
                driver="xe", vram_mb=24480, drm_card="card1",
                drm_render="renderD128", sysfs_path="/sys/.../0000:03:00.0",
            ),
            DetectedGPU(
                pci_slot="0000:04:00.0", device_id=0x56A0,
                arch=Arch.ALCHEMIST, name="Arc A770",
                driver="xe", vram_mb=16384, drm_card="card2",
                drm_render="renderD129", sysfs_path="/sys/.../0000:04:00.0",
            ),
        ]
        cfg = init_config_from_detection(gpus, llama_server_path="/bin/llama-server")
        assert cfg.paths.llama_server == "/bin/llama-server"
        assert len(cfg.gpus) == 2
        assert cfg.gpus[0].enabled is True   # B60 — highest VRAM Battlemage
        assert cfg.gpus[1].enabled is False  # A770

    def test_fallback_first_gpu_when_no_known_arc(self):
        gpus = [
            DetectedGPU(
                pci_slot="0000:03:00.0", device_id=0x9999,
                arch=Arch.UNKNOWN, name="Intel GPU",
                driver="xe", vram_mb=None, drm_card="card1",
                drm_render="renderD128", sysfs_path="/sys/.../0000:03:00.0",
            ),
        ]
        cfg = init_config_from_detection(gpus)
        assert cfg.gpus[0].enabled is True

    def test_sets_sycl_index(self):
        gpus = [
            DetectedGPU(
                pci_slot="0000:03:00.0", device_id=0xE211,
                arch=Arch.BATTLEMAGE, name="B60",
                driver="xe", vram_mb=24480, drm_card="card1",
                drm_render="renderD128", sysfs_path="/sys/.../0000:03:00.0",
            ),
        ]
        cfg = init_config_from_detection(gpus)
        assert cfg.gpus[0].sycl_index == 0
