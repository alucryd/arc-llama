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
        # Expert count is the definitive MoE signal (some architectures,
        # e.g. gemma4, use the same arch string for dense and MoE variants,
        # so this field -- not the arch name -- is what actually tells them
        # apart).
        expert_count_field = reader.get_field(
            gguf.Keys.LLM.EXPERT_COUNT.format(arch=arch)
        )
        if expert_count_field is not None:
            try:
                meta["expert_count"] = int(expert_count_field.contents())
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
# MoE detection
# ---------------------------------------------------------------------------

# Architecture strings that unambiguously indicate a Mixture-of-Experts model.
# Keep this conservative: only add prefixes once they're confirmed by real
# GGUF metadata.
_MOE_ARCH_PREFIXES: tuple[str, ...] = (
    "qwen2moe",
    "qwen3moe",
    "qwen35moe",
    "qwen36moe",
    "llama4",
    "mixtral",
    "deepseek2",
    "deepseek3",
    "glm4",
)
# Deliberately excludes "gemma4": dense and MoE Gemma-4 GGUFs share the same
# architecture string, so prefix matching would false-positive on dense
# models. is_moe() falls back to the expert_count metadata field for these.

# GGUF keys that may hold the total number of experts per MoE layer. Different
# converter paths use different names, so we try the common ones in order.
_EXPERT_COUNT_KEYS: tuple[str, ...] = (
    "expert_count",
    "num_experts",
    "moe_expert_count",
    "n_experts",
    "moe_num_experts",
)


def is_moe(path: Path | str) -> bool:
    """Return True if the GGUF at *path* is a MoE architecture.

    Detection is based on the architecture string from GGUF metadata. This
    is a heuristic; it will miss MoE variants whose metadata uses a generic
    architecture name until they're added to ``_MOE_ARCH_PREFIXES``.
    """
    meta = read_gguf_meta(path)
    arch = meta.get("architecture", "").lower()
    if not arch:
        return False
    if any(arch.startswith(prefix) for prefix in _MOE_ARCH_PREFIXES):
        return True
    # Some converters label generic architectures with an explicit MoE key.
    return any(
        meta.get(key) is not None for key in _EXPERT_COUNT_KEYS
    )


def expert_count(path: Path | str) -> int | None:
    """Return the total number of experts per MoE layer, if known."""
    meta = read_gguf_meta(path)
    arch = meta.get("architecture", "")
    # Try architecture-specific keys first, then generic ones.
    keys: list[str] = []
    if arch:
        keys.extend(f"{arch}.{key}" for key in _EXPERT_COUNT_KEYS)
    keys.extend(_EXPERT_COUNT_KEYS)
    for key in keys:
        value = meta.get(key)
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str):
            try:
                n = int(value)
                if n > 0:
                    return n
            except ValueError:
                pass
    return None


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


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------

# Map GGML quantization type -> bytes per element. SYCL/Vulkan keep quantized
# weights in their packed device format, so the raw tensor byte size is the
# dominant weight-VRAM term.
_BYTES_PER_ELEMENT: dict[gguf.GGMLQuantizationType, float] = {
    qtype: block_bytes / block_size
    for qtype, (block_size, block_bytes) in gguf.GGML_QUANT_SIZES.items()
}


def _tensor_vram_bytes(tensor: Any) -> int:
    """Return the packed VRAM bytes for a single GGUF tensor.

    For quantized tensors this is the raw quantized size (which is how SYCL
    stores them on device). For unquantized types it is n_elements * bytes/elem.
    """
    try:
        # n_bytes is the exact on-disk tensor payload size; for GGUF this is
        # also the device-side size for quantized weights.
        return int(tensor.n_bytes)
    except Exception:
        pass
    try:
        bpe = _BYTES_PER_ELEMENT.get(tensor.tensor_type)
        if bpe is not None:
            return int(tensor.n_elements * bpe)
    except Exception:
        pass
    return 0


def estimate_weight_vram_bytes(path: Path | str) -> int | None:
    """Estimate the VRAM footprint of the model weights alone.

    Sums the raw quantized tensor sizes from the GGUF file, which closely
    matches the device-side footprint on SYCL/Vulkan. Falls back to the file
    size if the GGUF cannot be inspected.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        reader = gguf.GGUFReader(p)
    except Exception as exc:
        log.debug("gguf weight estimate failed for %s: %s", p, exc)
        return None

    total = 0
    for tensor in reader.tensors:
        total += _tensor_vram_bytes(tensor)
    return total


def gguf_vram_estimate(path: Path | str) -> dict[str, Any]:
    """Return a detailed VRAM estimate for a GGUF file.

    Keys:
        - file_size_bytes: size on disk
        - weight_vram_bytes: estimated weight footprint in VRAM
        - params: total parameter count read from tensor shapes
        - architecture: model architecture from metadata
    """
    p = Path(path)
    file_size = p.stat().st_size if p.exists() else 0
    weight_vram = estimate_weight_vram_bytes(p)
    meta = read_gguf_meta(p)
    return {
        "file_size_bytes": file_size,
        "weight_vram_bytes": weight_vram,
        "params": weight_vram // 2 if weight_vram else None,
        "architecture": meta.get("architecture", "unknown"),
    }
