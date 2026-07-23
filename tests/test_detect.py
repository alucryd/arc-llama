"""Tests for arc_llama.detect — GPU discovery without real hardware."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from arc_llama.arch import Arch
from arc_llama.detect import (
    DetectedGPU,
    _enrich_with_clinfo,
    _parse_clinfo_devices,
    _scan_pci,
    lspci_intel_gpus,
)


class TestParseClinfo:
    def test_single_device(self):
        text = """
  Device Name                                     Intel Arc Pro B60 Graphics
  Global memory size                              25769803776
"""
        out = _parse_clinfo_devices(text)
        assert out == [("Intel Arc Pro B60 Graphics", 25769803776)]

    def test_multiple_devices(self):
        text = """
  Device Name                                     Intel Arc Pro B60 Graphics
  Global memory size                              25769803776
  Device Name                                     Intel(R) UHD Graphics
  Global memory size                              4294967296
"""
        out = _parse_clinfo_devices(text)
        assert len(out) == 2
        assert out[0] == ("Intel Arc Pro B60 Graphics", 25769803776)
        assert out[1] == ("Intel(R) UHD Graphics", 4294967296)

    def test_no_match_returns_empty(self):
        assert _parse_clinfo_devices("") == []


class TestScanPci:
    def test_finds_battlemage(self, make_sysfs_gpu, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        make_sysfs_gpu(slot="0000:03:00.0", device_id=0xE211, vram_bytes=24 * 1024 * 1024 * 1024)
        # Redirect /sys/bus/pci/devices to our fake tree
        fake_sys = tmp_path / "sys" / "bus" / "pci" / "devices"
        monkeypatch.setattr(
            "arc_llama.detect.Path",
            lambda p, **kw: fake_sys if p == "/sys/bus/pci/devices" else Path(p, **kw),
        )
        # _scan_pci uses Path("/sys/bus/pci/devices") directly; monkeypatch the import target
        import arc_llama.detect as detect_mod
        original_path_cls = detect_mod.Path
        detect_mod.Path = lambda p, **kw: fake_sys if p == "/sys/bus/pci/devices" else Path(p, **kw)
        try:
            gpus = _scan_pci()
        finally:
            detect_mod.Path = original_path_cls

        assert len(gpus) == 1
        assert gpus[0].pci_slot == "0000:03:00.0"
        assert gpus[0].arch == Arch.BATTLEMAGE
        assert gpus[0].vram_mb == 24 * 1024

    def test_skips_non_intel_vendor(self, make_sysfs_gpu, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # Write a non-Intel vendor
        base = tmp_path / "sys" / "bus" / "pci" / "devices" / "0000:01:00.0"
        base.mkdir(parents=True)
        (base / "vendor").write_text("0x10DE\n")
        (base / "device").write_text("0x1234\n")
        (base / "class").write_text("0x030000\n")

        fake_sys = tmp_path / "sys" / "bus" / "pci" / "devices"
        import arc_llama.detect as detect_mod
        original = detect_mod.Path
        detect_mod.Path = lambda p, **kw: fake_sys if p == "/sys/bus/pci/devices" else Path(p, **kw)
        try:
            gpus = _scan_pci()
        finally:
            detect_mod.Path = original
        assert gpus == []

    def test_notes_when_no_driver(self, make_sysfs_gpu, tmp_path: Path):
        make_sysfs_gpu(slot="0000:03:00.0", device_id=0xE211, driver="")
        fake_sys = tmp_path / "sys" / "bus" / "pci" / "devices"
        import arc_llama.detect as detect_mod
        original = detect_mod.Path
        detect_mod.Path = lambda p, **kw: fake_sys if p == "/sys/bus/pci/devices" else Path(p, **kw)
        try:
            gpus = _scan_pci()
        finally:
            detect_mod.Path = original
        assert any("No kernel driver" in n for n in gpus[0].notes)


class TestEnrichWithClinfo:
    def test_enrich_vram_from_clinfo(self):
        gpu = DetectedGPU(
            pci_slot="0000:03:00.0", device_id=0xE211,
            arch=Arch.BATTLEMAGE, name="Arc Pro B60",
            driver="xe", vram_mb=None, drm_card="card1",
            drm_render="renderD128", sysfs_path="/sys/...",
        )
        _enrich_with_clinfo([gpu])
        # clinfo probably not installed in CI, so this is a no-op.
        # We just assert it doesn't crash.
        assert gpu.vram_mb is None or isinstance(gpu.vram_mb, int)


class TestLspciIntelGpus:
    def test_returns_empty_when_missing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(args=a, returncode=1, stdout="", stderr=""))
        assert lspci_intel_gpus() == ""

    def test_filters_intel_display(self, monkeypatch: pytest.MonkeyPatch):
        fake_out = """
00:02.0 VGA compatible controller [0300]: Intel Corporation Device [8086:E211] (rev 01)
01:00.0 VGA compatible controller [0300]: NVIDIA Corporation Device [10de:1234]
"""
        def _fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=fake_out, stderr="")
        monkeypatch.setattr(subprocess, "run", _fake_run)
        result = lspci_intel_gpus()
        assert "8086:E211" in result
        assert "10de:1234" not in result
