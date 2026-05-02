"""Discover Intel GPUs on the local machine.

Detection is layered so we work even on minimal systems:
  1. Read `/sys/bus/pci/devices/` directly — fast, root-free, always available.
  2. Cross-reference with `/sys/class/drm/cardN` to identify the kernel driver
     (`i915`, `xe`) and pull VRAM size from `device/mem_info_vram_total`.
  3. Optionally call `clinfo` if installed, to enrich with OpenCL platform info.

We never assume `clinfo`, `sycl-ls`, or oneAPI are available — `arc-llama doctor`
checks for those separately and tells the user how to install them.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from arc_llama.arch import Arch, arch_for_device_id

INTEL_VENDOR_ID = 0x8086
PCI_CLASS_VGA = 0x030000      # 0x03_00_00 — VGA-compatible controller
PCI_CLASS_DISPLAY = 0x038000  # 0x03_80_00 — other display controller


@dataclass
class DetectedGPU:
    """One Intel GPU discovered on this host."""
    pci_slot: str          # e.g. "0000:03:00.0"
    device_id: int         # PCI device ID (e.g. 0xE212 for Arc Pro B60)
    arch: Arch
    name: str              # marketing name from arch table or fallback
    driver: str | None     # kernel driver bound: "xe", "i915", or None
    vram_mb: int | None    # total VRAM in MiB if discoverable, else None
    drm_card: str | None   # e.g. "card1"
    drm_render: str | None # e.g. "renderD128"
    sysfs_path: str        # /sys/bus/pci/devices/<slot>
    notes: list[str] = field(default_factory=list)

    @property
    def vram_gb(self) -> float | None:
        return None if self.vram_mb is None else round(self.vram_mb / 1024, 1)

    @property
    def sycl_index_hint(self) -> int:
        """Best-guess index to feed into ONEAPI_DEVICE_SELECTOR=level_zero:N.

        We can't know the actual SYCL enumeration order without calling sycl-ls,
        so this is just the order we discovered the device. Override at runtime
        if you have multiple Intel GPUs and the wrong one gets picked.
        """
        return getattr(self, "_index", 0)


def _read_hex(path: Path) -> int | None:
    try:
        return int(path.read_text().strip(), 16)
    except (OSError, ValueError):
        return None


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _driver_name(sysfs: Path) -> str | None:
    """Resolve the bound kernel driver from /sys/bus/pci/devices/<slot>/driver."""
    driver_link = sysfs / "driver"
    if not driver_link.exists():
        return None
    try:
        return os.path.basename(os.readlink(driver_link))
    except OSError:
        return None


def _drm_nodes(sysfs: Path) -> tuple[str | None, str | None]:
    """Find (cardN, renderDN) under /sys/bus/pci/devices/<slot>/drm/."""
    drm_dir = sysfs / "drm"
    if not drm_dir.exists():
        return None, None
    card = render = None
    for entry in drm_dir.iterdir():
        n = entry.name
        if n.startswith("card") and card is None:
            card = n
        elif n.startswith("renderD") and render is None:
            render = n
    return card, render


def _vram_mib(sysfs: Path) -> int | None:
    """Pull VRAM size from xe driver sysfs. i915 doesn't expose this consistently."""
    # xe driver:  device/mem_info_vram_total (bytes, integer)
    candidates = [
        sysfs / "mem_info_vram_total",
        sysfs / "device" / "mem_info_vram_total",
        sysfs / "tile0" / "physical_vram_size_bytes",
    ]
    for c in candidates:
        v = _read_int(c)
        if v:
            return v // (1024 * 1024)
    return None


def _scan_pci() -> list[DetectedGPU]:
    """Walk /sys/bus/pci/devices and collect Intel GPUs."""
    pci_root = Path("/sys/bus/pci/devices")
    if not pci_root.exists():
        return []
    found: list[DetectedGPU] = []
    for slot_dir in sorted(pci_root.iterdir()):
        vendor = _read_hex(slot_dir / "vendor")
        if vendor != INTEL_VENDOR_ID:
            continue
        klass = _read_hex(slot_dir / "class")
        if klass not in (PCI_CLASS_VGA, PCI_CLASS_DISPLAY):
            continue
        device_id = _read_hex(slot_dir / "device")
        if device_id is None:
            continue
        arch, name = arch_for_device_id(device_id)
        driver = _driver_name(slot_dir)
        card, render = _drm_nodes(slot_dir)
        vram = _vram_mib(slot_dir)
        gpu = DetectedGPU(
            pci_slot=slot_dir.name,
            device_id=device_id,
            arch=arch,
            name=name,
            driver=driver,
            vram_mb=vram,
            drm_card=card,
            drm_render=render,
            sysfs_path=str(slot_dir),
        )
        if driver is None:
            gpu.notes.append("No kernel driver bound — install `xe` or `i915` modules.")
        found.append(gpu)
    for i, g in enumerate(found):
        g._index = i  # type: ignore[attr-defined]
    return found


def _enrich_with_clinfo(gpus: list[DetectedGPU]) -> None:
    """Best-effort: pull OpenCL device names from `clinfo -l` and attach to notes.

    This *does* spawn a subprocess, but `clinfo -l` is read-only and lightweight.
    Skips silently if clinfo isn't installed.
    """
    try:
        out = subprocess.run(
            ["clinfo", "-l"], capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return
    if out.returncode != 0:
        return
    intel_devices = [
        line.strip() for line in out.stdout.splitlines()
        if "Intel" in line and "Device" in line
    ]
    if not intel_devices:
        return
    for g in gpus:
        for d in intel_devices:
            if g.name.split()[0] in d or f"0x{g.device_id:04X}" in d:
                g.notes.append(f"OpenCL: {d}")


def detect_gpus(enrich: bool = True) -> list[DetectedGPU]:
    """Return every Intel GPU on this host. Sorted by PCI slot.

    Args:
        enrich: if True, also call `clinfo -l` to enrich notes. Set False in tests
            or when GPUs are in active use and you want zero subprocess noise.
    """
    gpus = _scan_pci()
    if enrich and gpus:
        _enrich_with_clinfo(gpus)
    return gpus


def lspci_intel_gpus() -> str:
    """Return raw `lspci -nn` output filtered to Intel display devices.

    Useful for `arc-llama doctor` and for issue reports when arc-llama doesn't
    recognise a card.
    """
    try:
        out = subprocess.run(
            ["lspci", "-nn"], capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if out.returncode != 0:
        return ""
    return "\n".join(
        line for line in out.stdout.splitlines()
        if re.search(r"\[8086:[0-9A-Fa-f]+\]", line) and (
            "VGA" in line or "Display" in line or "3D" in line
        )
    )
