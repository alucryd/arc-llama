"""Tests for host/driver competitive-inference checks."""
from __future__ import annotations

from pathlib import Path

from arc_llama.platform_checks import (
    format_bytes,
    max_memory_bar_bytes,
    parse_kernel_version,
    rebar_likely_enabled,
)


def _write_resource(path: Path, lines: list[str]) -> Path:
    sysfs = path / "0000:03:00.0"
    sysfs.mkdir(parents=True)
    (sysfs / "resource").write_text("\n".join(lines) + "\n")
    return sysfs


class TestMaxMemoryBar:
    def test_reads_largest_mem_bar(self, tmp_path: Path):
        # 256 MiB BAR (typical without ReBAR) + tiny IO region
        sysfs = _write_resource(
            tmp_path,
            [
                "0x00000000f0000000 0x00000000f0ffffff 0x0000000000040200",  # 16 MiB
                "0x00000000e0000000 0x00000000efffffff 0x0000000000040200",  # 256 MiB
                "0x000000000000e000 0x000000000000e0ff 0x0000000000000101",  # IO
            ],
        )
        size = max_memory_bar_bytes(sysfs)
        assert size == 256 * 1024 * 1024

    def test_large_rebar_aperture(self, tmp_path: Path):
        # 16 GiB aperture
        start = 0x0000380000000000
        end = start + 16 * 1024**3 - 1
        sysfs = _write_resource(
            tmp_path,
            [f"0x{start:016x} 0x{end:016x} 0x000000000014020e"],
        )
        size = max_memory_bar_bytes(sysfs)
        assert size == 16 * 1024**3


class TestRebarHeuristic:
    def test_small_bar_is_off(self, tmp_path: Path):
        sysfs = _write_resource(
            tmp_path,
            ["0x00000000e0000000 0x00000000efffffff 0x0000000000040200"],  # 256 MiB
        )
        assert rebar_likely_enabled(sysfs, vram_mb=24 * 1024) is False

    def test_large_bar_is_on(self, tmp_path: Path):
        start = 0x0000380000000000
        end = start + 24 * 1024**3 - 1
        sysfs = _write_resource(
            tmp_path,
            [f"0x{start:016x} 0x{end:016x} 0x000000000014020e"],
        )
        assert rebar_likely_enabled(sysfs, vram_mb=24 * 1024) is True

    def test_missing_resource_unknown(self, tmp_path: Path):
        empty = tmp_path / "nodev"
        empty.mkdir()
        assert rebar_likely_enabled(empty) is None


class TestHelpers:
    def test_format_bytes(self):
        assert format_bytes(256 * 1024 * 1024) == "256 MiB"
        assert "GiB" in format_bytes(16 * 1024**3)

    def test_parse_kernel_version(self):
        assert parse_kernel_version("6.14.0-generic") == (6, 14)
        assert parse_kernel_version("5.15.0") == (5, 15)
        assert parse_kernel_version("bogus") is None
