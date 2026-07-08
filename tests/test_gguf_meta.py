"""Tests for arc_llama.gguf_meta — MTP head detection from GGUF metadata."""
from __future__ import annotations

from pathlib import Path

import pytest

from arc_llama.gguf_meta import (
    expert_count,
    has_mtp_heads,
    is_hybrid_ssm,
    is_moe,
    mtp_info,
    read_gguf_meta,
)

# Real GGUFs on the host's storage, discovered during exploration.
_BASE_QWEN = Path("/mnt/storage/models/qwen3.6-27b/Qwen_Qwen3.6-27B-Q4_K_M.gguf")
_MTP_QWEN = Path("/mnt/storage/models/qwen3.6-27b/Qwen3.6-27B-MTP-UD-Q4_K_XL.gguf")
_GEMMA_MOE = Path("/mnt/storage/models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf")
_QWEN_CODER_MOE = Path(
    "/mnt/storage/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"
)


def _have_fixtures() -> bool:
    return _BASE_QWEN.exists() and _MTP_QWEN.exists()


def _have_moe_fixtures() -> bool:
    return _GEMMA_MOE.exists() and _QWEN_CODER_MOE.exists()


class TestReadGgufMeta:
    @pytest.mark.skipif(not _have_fixtures(), reason="fixture GGUFs not on disk")
    def test_reads_architecture(self):
        meta = read_gguf_meta(_BASE_QWEN)
        assert meta["architecture"] == "qwen35"

    @pytest.mark.skipif(not _have_fixtures(), reason="fixture GGUFs not on disk")
    def test_reads_nextn_and_layers(self):
        meta = read_gguf_meta(_MTP_QWEN)
        assert meta["nextn_predict_layers"] == 1
        assert meta["block_count"] == 65

    def test_missing_file_returns_empty(self):
        meta = read_gguf_meta("/nonexistent/file.gguf")
        assert meta == {}


class TestMtpDetection:
    @pytest.mark.skipif(not _have_fixtures(), reason="fixture GGUFs not on disk")
    def test_base_qwen_has_no_mtp(self):
        assert has_mtp_heads(_BASE_QWEN) is False

    @pytest.mark.skipif(not _have_fixtures(), reason="fixture GGUFs not on disk")
    def test_mtp_qwen_has_mtp(self):
        assert has_mtp_heads(_MTP_QWEN) is True

    def test_missing_file_is_false(self):
        assert has_mtp_heads("/nonexistent.gguf") is False


class TestHybridSsmDetection:
    @pytest.mark.skipif(not _have_fixtures(), reason="fixture GGUFs not on disk")
    def test_qwen_is_hybrid_ssm(self):
        assert is_hybrid_ssm(_BASE_QWEN) is True
        assert is_hybrid_ssm(_MTP_QWEN) is True

    def test_missing_file_is_false(self):
        assert is_hybrid_ssm("/nonexistent.gguf") is False


class TestMtpInfo:
    @pytest.mark.skipif(not _have_fixtures(), reason="fixture GGUFs not on disk")
    def test_summary_keys(self):
        info = mtp_info(_MTP_QWEN)
        assert info["has_mtp_heads"] is True
        assert info["is_hybrid_ssm"] is True
        assert info["nextn_predict_layers"] == 1


class TestMoeDetection:
    def test_missing_file_is_not_moe(self):
        assert is_moe("/nonexistent.gguf") is False
        assert expert_count("/nonexistent.gguf") is None

    def test_detects_moe_from_architecture_prefix(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "arc_llama.gguf_meta.read_gguf_meta",
            lambda _path: {"architecture": "qwen2moe"},
        )
        assert is_moe("/fake.gguf") is True

    def test_expert_count_reads_arch_specific_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "arc_llama.gguf_meta.read_gguf_meta",
            lambda _path: {"architecture": "qwen2moe", "qwen2moe.expert_count": 64},
        )
        assert expert_count("/fake.gguf") == 64

    def test_expert_count_falls_back_to_generic_keys(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "arc_llama.gguf_meta.read_gguf_meta",
            lambda _path: {"architecture": "llama4", "expert_count": 16},
        )
        assert expert_count("/fake.gguf") == 16

    def test_expert_count_returns_none_when_unknown(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "arc_llama.gguf_meta.read_gguf_meta",
            lambda _path: {"architecture": "llama"},
        )
        assert expert_count("/fake.gguf") is None

    @pytest.mark.skipif(not _have_fixtures(), reason="fixture GGUFs not on disk")
    def test_dense_qwen_is_not_moe(self):
        # Regression guard: qwen35 (dense) must not be misdetected as MoE.
        assert is_moe(_BASE_QWEN) is False
        assert expert_count(_BASE_QWEN) is None

    @pytest.mark.skipif(not _have_moe_fixtures(), reason="MoE fixture GGUFs not on disk")
    def test_gemma_moe_detected_via_expert_count_not_arch_prefix(self):
        # gemma4 is deliberately excluded from _MOE_ARCH_PREFIXES since dense
        # and MoE Gemma-4 GGUFs share the same architecture string -- this
        # must be detected via the expert_count metadata field instead.
        meta = read_gguf_meta(_GEMMA_MOE)
        assert meta["architecture"] == "gemma4"
        assert is_moe(_GEMMA_MOE) is True
        assert expert_count(_GEMMA_MOE) == 128

    @pytest.mark.skipif(not _have_moe_fixtures(), reason="MoE fixture GGUFs not on disk")
    def test_qwen3_coder_moe_detected_via_arch_prefix(self):
        meta = read_gguf_meta(_QWEN_CODER_MOE)
        assert meta["architecture"] == "qwen3moe"
        assert is_moe(_QWEN_CODER_MOE) is True
        assert expert_count(_QWEN_CODER_MOE) == 128
