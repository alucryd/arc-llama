from __future__ import annotations

from pathlib import Path

import pytest

from arc_llama.arch import Backend
from arc_llama.binary import detect_backends, detect_llama_server_backend


def _make_fake_binary(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "llama-server"
    path.write_bytes(content)
    return path


class TestDetectBackends:
    def test_detects_sycl(self, tmp_path: Path):
        binary = _make_fake_binary(tmp_path, b"some_prefix ggml_backend_sycl blah")
        assert detect_backends(binary) == {Backend.SYCL}
        assert detect_llama_server_backend(binary) == Backend.SYCL

    def test_detects_vulkan(self, tmp_path: Path):
        binary = _make_fake_binary(tmp_path, b"some_prefix ggml_backend_vulkan blah")
        assert detect_backends(binary) == {Backend.VULKAN}
        assert detect_llama_server_backend(binary) == Backend.VULKAN

    def test_detects_both_prefers_sycl(self, tmp_path: Path):
        binary = _make_fake_binary(
            tmp_path,
            b"ggml_backend_sycl ggml_backend_vulkan",
        )
        assert detect_backends(binary) == {Backend.SYCL, Backend.VULKAN}
        assert detect_llama_server_backend(binary) == Backend.SYCL

    def test_no_match_returns_none(self, tmp_path: Path):
        binary = _make_fake_binary(tmp_path, b"cpu only binary with no gpu backends")
        assert detect_backends(binary) == set()
        assert detect_llama_server_backend(binary) is None

    def test_missing_file_returns_empty(self, tmp_path: Path):
        missing = tmp_path / "does-not-exist"
        assert detect_backends(missing) == set()
        assert detect_llama_server_backend(missing) is None

    @pytest.mark.parametrize(
        "marker,expected",
        [
            (b"libsycl.so.7", Backend.SYCL),
            (b"libze_intel_gpu.so.1", Backend.SYCL),
            (b"libvulkan.so.1", Backend.VULKAN),
            (b"vkGetInstanceProcAddr", Backend.VULKAN),
        ],
    )
    def test_alternative_markers(self, tmp_path: Path, marker: bytes, expected: Backend):
        binary = _make_fake_binary(tmp_path, b"header " + marker + b" footer")
        assert detect_backends(binary) == {expected}
