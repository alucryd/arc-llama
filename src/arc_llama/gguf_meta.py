"""Lightweight GGUF metadata peeking for arc-llama.

Uses llama.cpp's `gguf-py` to read key metadata (architecture,
nextn_predict_layers, block_count) without loading tensor data.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gguf  # type: ignore[import-untyped]

log = logging.getLogger("arc_llama.gguf_meta")


# ---------------------------------------------------------------------------
# Metadata reading
# ---------------------------------------------------------------------------

def read_gguf_meta(path: Path | str) -> dict[str, Any]:
    """Read a GGUF file and return a small dict of metadata we care about.

    Returns empty dict if the file can't be read.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        reader = gguf.GGUFReader(p)
    except Exception as exc:
        log.debug("gguf read failed for %s: %s", p, exc)
        return {}

    meta: dict[str, Any] = {}
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is not None:
        meta["architecture"] = str(arch_field.contents())

    arch = meta.get("architecture", "")
    if arch:
        # nextn_predict_layers is the definitive MTP signal in the pr-22673 branch
        nextn_field = reader.get_field(f"{arch}.nextn_predict_layers")
        if nextn_field is not None:
            try:
                meta["nextn_predict_layers"] = int(nextn_field.contents())
            except (TypeError, ValueError):
                pass
        # layer count, useful for diagnostics
        n_layer_field = reader.get_field(f"{arch}.block_count")
        if n_layer_field is not None:
            try:
                meta["block_count"] = int(n_layer_field.contents())
            except (TypeError, ValueError):
                pass

    return meta


# ---------------------------------------------------------------------------
# MTP detection
# ---------------------------------------------------------------------------

def has_mtp_heads(path: Path | str) -> bool:
    """Return True if the GGUF at *path* contains real MTP heads.

    Checks metadata — not the filename. The canonical signal is
    ``nextn_predict_layers > 0`` in the GGUF kv store. Stand-alone MTP-only
    GGUFs (architecture ``qwen35_mtp`` / ``qwen35moe_mtp``) also count.
    """
    meta = read_gguf_meta(path)
    if not meta:
        return False
    arch = meta.get("architecture", "")
    if arch in ("qwen35_mtp", "qwen35moe_mtp"):
        return True
    nextn = meta.get("nextn_predict_layers", 0)
    if isinstance(nextn, int) and nextn > 0:
        return True
    return False


def is_hybrid_ssm(path: Path | str) -> bool:
    """Return True if the GGUF is a hybrid SSM+attention architecture.

    Today this means the Qwen3.5/3.6 family (dense or MoE) which use GDN
    (gated delta net) layers — a recurrent state-space-like attention
    hybrid. These architectures are known to perform poorly with SYCL MTP
    speculative decoding on Xe2 (Battlemage, Lunar Lake).
    """
    meta = read_gguf_meta(path)
    arch = meta.get("architecture", "")
    # qwen35 and qwen35moe (with or without the _mtp suffix) are the
    # known hybrid SSM+attention families today.
    return arch.startswith("qwen35")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def mtp_info(path: Path | str) -> dict[str, Any]:
    """Return a human-readable summary of MTP-relevant metadata."""
    meta = read_gguf_meta(path)
    return {
        "path": str(path),
        "architecture": meta.get("architecture", "unknown"),
        "block_count": meta.get("block_count", "unknown"),
        "nextn_predict_layers": meta.get("nextn_predict_layers", 0),
        "has_mtp_heads": has_mtp_heads(path),
        "is_hybrid_ssm": is_hybrid_ssm(path),
    }
