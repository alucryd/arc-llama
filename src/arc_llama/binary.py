"""Introspect a local llama-server binary to discover its compute backend(s)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from arc_llama.arch import Backend

# Byte signatures that reliably appear in llama.cpp binaries built with the
# corresponding backend enabled.  We search for these instead of relying on
# `--version`, which does not currently print backend tags.
# NOTE: bare backend NAMES ("sycl", "vulkan", "oneapi") are deliberately excluded
# from the markers below. llama.cpp's meta-loader libggml.so enumerates every
# backend name as a plain string in ALL builds, regardless of which backends are
# actually compiled, so those bare names are not evidence a backend is present.
# Only registration symbols (ggml_backend_*) and real runtime library / API
# symbols (libsycl, libze_intel_gpu, libze_loader, libvulkan,
# vkGetInstanceProcAddr) are used as evidence.
_BACKEND_MARKERS: dict[Backend, list[bytes]] = {
    Backend.SYCL: [
        b"ggml_backend_sycl",
        b"ggml_backend_sycl_reg",
        b"libsycl",
        b"libze_intel_gpu",
        b"libze_loader",
    ],
    Backend.VULKAN: [
        b"ggml_backend_vulkan",
        b"ggml_backend_vulkan_reg",
        b"libvulkan",
        b"vkGetInstanceProcAddr",
    ],
}


def _scan_with_strings(path: Path) -> set[Backend]:
    """Use the system ``strings`` utility when available; much faster than a
    pure-Python scan on multi-hundred-megabyte binaries."""
    strings_bin = shutil.which("strings")
    if strings_bin is None:
        raise FileNotFoundError("strings")

    proc = subprocess.run(
        [strings_bin, "-n", "4", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "strings failed")

    text = proc.stdout
    found: set[Backend] = set()
    for backend, markers in _BACKEND_MARKERS.items():
        lowered = text.lower()
        if any(marker.decode().lower() in lowered for marker in markers):
            found.add(backend)
    return found


def _scan_with_python(path: Path, chunk_size: int = 1_048_576) -> set[Backend]:
    """Pure-Python fallback that scans the binary for backend markers."""
    found: set[Backend] = set()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            for backend, markers in _BACKEND_MARKERS.items():
                if backend in found:
                    continue
                if any(marker in chunk for marker in markers):
                    found.add(backend)
    return found


def _scan_file(path: Path) -> set[Backend]:
    """Scan a single file for backend markers, falling back to pure Python."""
    try:
        return _scan_with_strings(path)
    except (FileNotFoundError, RuntimeError, subprocess.SubprocessError):
        return _scan_with_python(path)


def detect_backends(binary_path: str | Path) -> set[Backend]:
    """Return the set of compute backends embedded in a ``llama-server`` binary.

    Modern official llama.cpp release builds are modular: the compute backend
    lives in sibling shared libraries next to the executable (e.g.
    ``libggml-vulkan.so`` holds the Vulkan markers, ``libggml-sycl.so`` holds
    the SYCL markers). This function scans the target file AND its sibling
    ``libggml*.so*`` files, unioning the results. It never executes the binary.
    """
    path = Path(binary_path)
    if not path.exists() or not path.is_file():
        return set()
    found = _scan_file(path)
    for sibling in sorted(path.parent.glob("libggml*.so*")):
        if not sibling.is_file() or sibling == path:
            continue
        try:
            found |= _scan_file(sibling)
        except OSError:
            continue
        if {Backend.SYCL, Backend.VULKAN} <= found:
            break
    return found


def detect_llama_server_backend(binary_path: str | Path) -> Backend | None:
    """Return the most capable backend detected in the binary, if any.

    Prefers SYCL over Vulkan because that is the current Arc default; returns
    ``None`` when the binary cannot be read or no supported backend is found.
    """
    backends = detect_backends(binary_path)
    if Backend.SYCL in backends:
        return Backend.SYCL
    if Backend.VULKAN in backends:
        return Backend.VULKAN
    return None
