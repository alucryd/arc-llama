"""Host / driver checks that matter for Arc inference performance.

Used by ``arc-llama doctor``. Detection is best-effort and never requires
root. Unknown results return None so the UI can say "could not determine".
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# PCI resource flag bits (linux/ioport.h)
_IORESOURCE_MEM = 0x00000200

# Without ReBAR, Arc typically exposes a 256 MiB aperture. Treat < 512 MiB as
# "ReBAR off / not usable"; >= 1 GiB as "ReBAR likely on".
_REBAR_OFF_MAX = 512 * 1024 * 1024
_REBAR_ON_MIN = 1024 * 1024 * 1024


@dataclass
class CheckResult:
    """One doctor line item."""
    name: str
    ok: bool | None  # True pass, False fail, None unknown/skip
    detail: str
    severity: str = "info"  # info | warn | fail
    hint: str = ""


@dataclass
class DoctorReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.ok is False and c.severity == "fail"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.ok is False and c.severity == "warn"]

    def add(
        self,
        name: str,
        ok: bool | None,
        detail: str,
        *,
        severity: str = "info",
        hint: str = "",
    ) -> None:
        if ok is False and severity == "info":
            severity = "warn"
        self.checks.append(
            CheckResult(name=name, ok=ok, detail=detail, severity=severity, hint=hint)
        )


def max_memory_bar_bytes(sysfs_path: str | Path) -> int | None:
    """Largest MMIO BAR size (bytes) for a PCI device via sysfs ``resource``."""
    path = Path(sysfs_path) / "resource"
    if not path.is_file():
        return None
    try:
        text = path.read_text()
    except OSError:
        return None
    largest = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start = int(parts[0], 16)
            end = int(parts[1], 16)
            flags = int(parts[2], 16)
        except ValueError:
            continue
        if start == 0 and end == 0:
            continue
        if not (flags & _IORESOURCE_MEM):
            continue
        size = end - start + 1
        if size > largest:
            largest = size
    return largest or None


def rebar_likely_enabled(
    sysfs_path: str | Path,
    vram_mb: int | None = None,
) -> bool | None:
    """Heuristic: is Resizable BAR exposing a large VRAM aperture?

    Returns True / False / None (unknown).
    """
    bar = max_memory_bar_bytes(sysfs_path)
    if bar is None:
        return None
    if bar < _REBAR_OFF_MAX:
        return False
    if vram_mb and vram_mb > 0:
        # Half of device VRAM mapped is a strong ReBAR signal.
        if bar >= (vram_mb * 1024 * 1024) // 2:
            return True
    if bar >= _REBAR_ON_MIN:
        return True
    # Between 512 MiB and 1 GiB — ambiguous (some partial BAR configs).
    return None


def format_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.0f} MiB"
    return f"{n} B"


def user_in_groups(*needed: str) -> dict[str, bool]:
    """Return membership for each group name (Linux). Empty on Windows."""
    if sys.platform == "win32":
        return {g: False for g in needed}
    try:
        out = subprocess.run(
            ["id", "-nG"], capture_output=True, text=True, timeout=2, check=False,
        )
        groups = set(out.stdout.split())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        groups = set()
    return {g: g in groups for g in needed}


def level_zero_loader_present() -> tuple[bool, str]:
    """Look for Level Zero loader library on common paths / ldconfig."""
    names = (
        "libze_loader.so.1",
        "libze_loader.so",
        "libze_intel_gpu.so.1",
        "libze_intel_gpu.so",
    )
    search_dirs = [
        Path("/usr/lib"),
        Path("/usr/lib64"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/local/lib"),
        Path("/opt/intel/oneapi/lib"),
        Path("/opt/intel/oneapi/lib/intel64"),
    ]
    # Also walk ONEAPI_ROOT if set.
    oneapi_root = os.environ.get("ONEAPI_ROOT") or os.environ.get("CMPLR_ROOT")
    if oneapi_root:
        search_dirs.append(Path(oneapi_root))
        search_dirs.append(Path(oneapi_root) / "lib")
        search_dirs.append(Path(oneapi_root) / "lib" / "intel64")

    for d in search_dirs:
        if not d.is_dir():
            continue
        for name in names:
            # Direct child
            candidate = d / name
            if candidate.exists():
                return True, str(candidate)
            # One level of subdirs (e.g. lib/intel64)
            try:
                for child in d.iterdir():
                    if child.is_dir():
                        c2 = child / name
                        if c2.exists():
                            return True, str(c2)
            except OSError:
                pass

    # ldconfig -p is authoritative when present.
    ldconfig = shutil.which("ldconfig")
    if ldconfig:
        try:
            out = subprocess.run(
                [ldconfig, "-p"], capture_output=True, text=True, timeout=5, check=False,
            )
            for name in names:
                if name in out.stdout:
                    m = re.search(rf"{re.escape(name)}[^\n]*=>\s*(\S+)", out.stdout)
                    return True, m.group(1) if m else name
        except (OSError, subprocess.TimeoutExpired):
            pass
    return False, ""


def oneapi_setvars_path() -> Path | None:
    if sys.platform == "win32":
        base = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        p = base / "Intel" / "oneAPI" / "setvars.bat"
        return p if p.exists() else None
    p = Path("/opt/intel/oneapi/setvars.sh")
    return p if p.exists() else None


def kernel_module_loaded(name: str) -> bool:
    return Path(f"/sys/module/{name}").exists()


def parse_kernel_version(release: str | None = None) -> tuple[int, int] | None:
    """Return (major, minor) from ``uname -r`` style string."""
    if release is None:
        release = os.uname().release if hasattr(os, "uname") else ""
    m = re.match(r"(\d+)\.(\d+)", release or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))
