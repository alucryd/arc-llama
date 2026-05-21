"""Tests for arc_llama.gguf_meta — MTP head detection from GGUF metadata."""
from __future__ import annotations

from pathlib import Path

import pytest

from arc_llama.gguf_meta import has_mtp_heads, is_hybrid_ssm, mtp_info, read_gguf_meta

# Real GGUFs on the host's storage, discovered during exploration.
_BASE_QWEN = Path("/mnt/storage/models/qwen3.6-27b/Qwen_Qwen3.6-27B-Q4_K_M.gguf")
_MTP_QWEN = Path("/mnt/storage/models/qwen3.6-27b/Qwen3.6-27B-MTP-UD-Q4_K_XL.gguf")


def _have_fixtures() -> bool:
    return _BASE_QWEN.exists() and _MTP_QWEN.exists()


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
