"""Introspect a local llama-server binary to discover its compute backend(s)."""

from __future__ import annotations

import re
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


_VULKAN_DEVICE_RE = re.compile(r"^\s*Vulkan(\d+):\s*(.+?)\s*$")
# Trailing memory report, e.g. " (24480 MiB, 21994 MiB free)". Stripped so the
# name keeps its own parentheses: device names legitimately contain them
# ("Intel(R) Arc(tm) Pro B60 Graphics (BMG G21)"), so we cannot simply cut at
# the first '(' — that yields the useless name "Intel".
_VULKAN_MEM_SUFFIX_RE = re.compile(r"\s*\([^()]*MiB[^()]*\)\s*$")


def _name_tokens(name: str) -> set[str]:
    """Alphanumeric tokens of a device name, lowercased.

    Vendor decorations differ between how a GPU is detected and how Vulkan
    reports it ("Arc Pro B60 Graphics" vs "Intel(R) Arc(tm) Pro B60 Graphics
    (BMG G21)"), so neither string contains the other. Comparing token sets
    sidesteps the decorations entirely.
    """
    return {t for t in re.split(r"[^0-9a-z]+", name.lower()) if t}


def list_vulkan_devices(
    binary_path: str | Path, *, timeout: float = 30.0
) -> list[tuple[int, str]]:
    """Return [(vulkan_index, device_name)] as reported by ``--list-devices``.

    Returns an empty list if the binary cannot be run or prints nothing we
    recognise; callers must treat that as "unknown", never as "no GPUs".
    """
    try:
        proc = subprocess.run(  # noqa: S603
            [str(binary_path), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    devices: list[tuple[int, str]] = []
    for line in (proc.stdout + proc.stderr).splitlines():
        m = _VULKAN_DEVICE_RE.match(line)
        if m:
            name = _VULKAN_MEM_SUFFIX_RE.sub("", m.group(2)).strip()
            devices.append((int(m.group(1)), name))
    return devices


def resolve_vulkan_index(
    devices: list[tuple[int, str]], *, gpu_name: str = ""
) -> int | None:
    """Pick the Vulkan index for an Intel GPU out of ``devices``.

    Vulkan enumerates every vendor, so the Arc card is not necessarily index 0.
    Matching is by name because ``--list-devices`` reports no PCI address.

    Returns None when the choice is ambiguous (several Intel devices and no
    usable *gpu_name* to disambiguate) or when nothing matches. A caller that
    gets None must not guess: guessing is what sent models to the wrong GPU.
    """
    if not devices:
        return None

    intel = [(i, n) for i, n in devices if "intel" in n.lower()]
    if not intel:
        return None
    if len(intel) == 1:
        return intel[0][0]

    # Several Intel devices: disambiguate on the detected name, e.g. two Arc
    # cards where only one is a B60. Substring matching does not work here
    # because of vendor decorations, so require every token of the detected
    # name to appear in the Vulkan name.
    if gpu_name:
        wanted = _name_tokens(gpu_name)
        if wanted:
            exact = [i for i, n in intel if wanted <= _name_tokens(n)]
            if len(exact) == 1:
                return exact[0]
    return None
