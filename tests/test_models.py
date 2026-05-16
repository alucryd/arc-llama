"""Tests for arc_llama.models — discovery, registration, HF parsing."""
from __future__ import annotations

from pathlib import Path

import pytest

from arc_llama.config import Config, GPUConfig, ModelConfig
from arc_llama.models import (
    HFModelSpec,
    _next_free_port,
    _short_name_from,
    add_local_model,
    discover_ggufs,
    infer_display_name,
    infer_kv_class,
    parse_hf_spec,
    register_discovered,
    short_name_from_path,
)


class TestParseHfSpec:
    def test_repo_only(self):
        s = parse_hf_spec("unsloth/gemma-4-31B-it-GGUF")
        assert s.repo == "unsloth/gemma-4-31B-it-GGUF"
        assert s.file is None
        assert s.quant is None

    def test_repo_with_quant_hint(self):
        s = parse_hf_spec("unsloth/gemma-4-31B-it-GGUF:Q4_K_M")
        assert s.repo == "unsloth/gemma-4-31B-it-GGUF"
        assert s.file is None
        assert s.quant == "Q4_K_M"

    def test_repo_with_filename(self):
        s = parse_hf_spec("unsloth/gemma-4-31B-it-GGUF:model-Q4_K_M.gguf")
        assert s.repo == "unsloth/gemma-4-31B-it-GGUF"
        assert s.file == "model-Q4_K_M.gguf"
        assert s.quant is None

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            parse_hf_spec("not-a-valid-spec")

    def test_iq_quant_hint(self):
        s = parse_hf_spec("repo/name:IQ4_XS")
        assert s.quant == "IQ4_XS"
        assert s.file is None


class TestNextFreePort:
    def test_no_used(self):
        assert _next_free_port(set(), start=18080) == 18080

    def test_skips_used(self):
        assert _next_free_port({18080, 18081}, start=18080) == 18082


class TestShortNameFrom:
    def test_strips_gguf_suffix(self):
        assert _short_name_from("unsloth/Qwen-7B-gguf", None) == "qwen-7b"

    def test_appends_quant(self):
        assert _short_name_from("repo/name", "model-Q4_K_M.gguf") == "name-q4_k_m"


class TestInferKvClass:
    def test_gemma(self):
        assert infer_kv_class("gemma-4-27B-Q4_K_M.gguf") == "gemma_swa"
        assert infer_kv_class("gemma3-7b.gguf") == "gemma_swa"

    def test_qwen3_27b_dense(self):
        assert infer_kv_class("Qwen3-27B-Q4_K_M.gguf") == "qwen3_27b_dense"
        assert infer_kv_class("qwen3.6-27b-q4_k_m.gguf") == "qwen3_27b_dense"

    def test_moe_a3b(self):
        assert infer_kv_class("Qwen3-30B-A3B-Q4_K_M.gguf") == "moe_a3b"
        assert infer_kv_class("huihui-qwen3-30b-a3b.gguf") == "moe_a3b"

    def test_default_fallback(self):
        assert infer_kv_class("llama-3-8b.gguf") == "default"


class TestShortNameFromPath:
    def test_from_stem(self):
        p = Path("/models/Qwen3-7B-Q4_K_M.gguf")
        assert short_name_from_path(p, set()) == "qwen3-7b-q4_k_m"

    def test_deduplicates(self):
        p = Path("/models/Qwen3-7B-Q4_K_M.gguf")
        used = {"qwen3-7b-q4_k_m"}
        assert short_name_from_path(p, used) == "qwen3-7b-q4_k_m-2"

    def test_fallback_to_parent(self):
        p = Path("/models/---.gguf")
        assert short_name_from_path(p, set()) != ""


class TestInferDisplayName:
    def test_replaces_underscores(self):
        p = Path("/models/Qwen3_7B_Q4_K_M.gguf")
        assert infer_display_name(p) == "Qwen3 7B Q4 K M"


class TestDiscoverGgufs:
    def test_finds_ggufs(self, tmp_path: Path):
        cfg = Config(paths=type("P", (), {"models_dir": str(tmp_path), "scan_paths": []})())
        (tmp_path / "a.gguf").write_text("")
        (tmp_path / "b.txt").write_text("")
        found = discover_ggufs(cfg)
        assert len(found) == 1
        assert found[0].name == "a.gguf"

    def test_respects_max_depth(self, tmp_path: Path):
        deep = tmp_path / "d1" / "d2" / "d3" / "d4" / "d5"
        deep.mkdir(parents=True)
        (deep / "deep.gguf").write_text("")
        cfg = Config(paths=type("P", (), {"models_dir": str(tmp_path), "scan_paths": []})())
        found = discover_ggufs(cfg, max_depth=2)
        assert len(found) == 0

    def test_skips_hidden(self, tmp_path: Path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.gguf").write_text("")
        cfg = Config(paths=type("P", (), {"models_dir": str(tmp_path), "scan_paths": []})())
        found = discover_ggufs(cfg)
        assert len(found) == 0

    def test_dedupes_across_scan_paths(self, tmp_path: Path):
        cfg = Config(paths=type("P", (), {
            "models_dir": str(tmp_path),
            "scan_paths": [str(tmp_path)],
        })())
        (tmp_path / "a.gguf").write_text("")
        found = discover_ggufs(cfg)
        assert len(found) == 1


class TestRegisterDiscovered:
    def test_adds_new_models(self, tmp_path: Path):
        cfg = Config(
            gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24 * 1024)],
            models=[],
        )
        p = tmp_path / "Qwen3-7B-Q4_K_M.gguf"
        p.write_bytes(b"\x00" * (4 * 1024 * 1024))
        added = register_discovered(cfg, [p])
        assert len(added) == 1
        assert added[0].name == "qwen3-7b-q4_k_m"
        assert added[0].gpu_pci_slot == "0000:03:00.0"

    def test_skips_already_registered(self, tmp_path: Path):
        p = tmp_path / "Qwen3-7B-Q4_K_M.gguf"
        p.write_bytes(b"\x00" * (4 * 1024 * 1024))
        cfg = Config(
            gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24 * 1024)],
            models=[ModelConfig(name="existing", path=str(p.resolve()), port=18080, gpu_pci_slot="0000:03:00.0")],
        )
        added = register_discovered(cfg, [p])
        assert len(added) == 0

    def test_raises_without_gpus(self, tmp_path: Path):
        cfg = Config(gpus=[])
        with pytest.raises(ValueError, match="No GPUs"):
            register_discovered(cfg, [tmp_path / "x.gguf"])


class TestAddLocalModel:
    def test_basic_registration(self, tmp_path: Path):
        p = tmp_path / "model.gguf"
        p.write_text("")
        cfg = Config(
            gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24 * 1024)],
        )
        mc = add_local_model(cfg, name="my-model", path=str(p), gpu_pci_slot="0000:03:00.0")
        assert mc.name == "my-model"
        assert mc.port == 18080
        assert "my-model" in {m.name for m in cfg.models}

    def test_duplicate_name_raises(self, tmp_path: Path):
        p = tmp_path / "model.gguf"
        p.write_text("")
        cfg = Config(
            gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24 * 1024)],
            models=[ModelConfig(name="my-model", path="/other.gguf", port=18080, gpu_pci_slot="0000:03:00.0")],
        )
        with pytest.raises(ValueError, match="already registered"):
            add_local_model(cfg, name="my-model", path=str(p), gpu_pci_slot="0000:03:00.0")

    def test_missing_file_raises(self, tmp_path: Path):
        cfg = Config(
            gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24 * 1024)],
        )
        with pytest.raises(FileNotFoundError):
            add_local_model(cfg, name="x", path="/does/not/exist.gguf", gpu_pci_slot="0000:03:00.0")

    def test_invalid_name_raises(self, tmp_path: Path):
        p = tmp_path / "model.gguf"
        p.write_text("")
        cfg = Config(
            gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24 * 1024)],
        )
        with pytest.raises(ValueError, match="must match"):
            add_local_model(cfg, name="bad name!", path=str(p), gpu_pci_slot="0000:03:00.0")

    def test_auto_port_assignment(self, tmp_path: Path):
        p = tmp_path / "model.gguf"
        p.write_text("")
        cfg = Config(
            gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24 * 1024)],
            models=[ModelConfig(name="first", path="/a.gguf", port=18080, gpu_pci_slot="0000:03:00.0")],
        )
        mc = add_local_model(cfg, name="second", path=str(p), gpu_pci_slot="0000:03:00.0")
        assert mc.port == 18081
