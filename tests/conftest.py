"""Shared fixtures and helpers for arc-llama tests."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure src/ is on the path when running tests directly.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def fake_gguf(tmp_path: Path) -> Path:
    """Create a dummy .gguf file and return its path."""
    p = tmp_path / "Qwen3-7B-Q4_K_M.gguf"
    p.write_bytes(b"GGUF\x00" + b"\x00" * 128)
    return p


@pytest.fixture
def fake_gguf_large(tmp_path: Path) -> Path:
    """Create a larger dummy .gguf (simulates a big model)."""
    p = tmp_path / "gemma-4-27B-Q4_K_M.gguf"
    # 18 GiB file
    p.write_bytes(b"GGUF\x00" + b"\x00" * (18 * 1024 * 1024))
    return p


@pytest.fixture
def make_sysfs_gpu(tmp_path: Path):
    """Factory to build a fake /sys/bus/pci/devices tree for one Intel GPU."""
    def _make(
        slot: str = "0000:03:00.0",
        device_id: int = 0xE211,
        klass: int = 0x030000,
        vram_bytes: int | None = 24 * 1024 * 1024 * 1024,
        driver: str = "xe",
        drm_card: str = "card1",
        drm_render: str = "renderD128",
    ) -> Path:
        base = tmp_path / "sys" / "bus" / "pci" / "devices" / slot
        base.mkdir(parents=True)
        (base / "vendor").write_text("0x8086\n")
        (base / "device").write_text(f"0x{device_id:04X}\n")
        (base / "class").write_text(f"0x{klass:06X}\n")

        if vram_bytes is not None:
            (base / "mem_info_vram_total").write_text(str(vram_bytes) + "\n")

        if driver:
            drv = base / "driver"
            drv.symlink_to(f"/sys/bus/pci/drivers/{driver}")

        drm = base / "drm"
        drm.mkdir()
        (drm / drm_card).mkdir()
        (drm / drm_render).mkdir()
        return base
    return _make


@pytest.fixture
def mock_config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect XDG_CONFIG_HOME to a temp dir."""
    cfg_dir = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    return cfg_dir


@pytest.fixture
def mock_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect XDG_DATA_HOME to a temp dir."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_dir))
    return data_dir
