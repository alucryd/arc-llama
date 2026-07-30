"""Lightweight GGUF metadata peeking for arc-llama.

Uses llama.cpp's `gguf-py` to read key metadata (architecture,
nextn_predict_layers, block_count) without loading tensor data.
"""
from __future__ import annotations

import logging
import re
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
        # apart). Different converters spell the key differently, so try the
        # architecture-prefixed form first, then the bare keys.
        for key in _expert_count_key_candidates(arch):
            expert_count_field = reader.get_field(key)
            if expert_count_field is not None:
                try:
                    value = int(expert_count_field.contents())
                    if value > 0:
                        meta["expert_count"] = value
                        break
                except (TypeError, ValueError):
                    continue
        # Trained context length: llama.cpp silently clamps the served ctx
        # to this value, so any configured ctx above it is a lie.
        ctx_field = reader.get_field(f"{arch}.context_length")
        if ctx_field is not None:
            try:
                meta["context_length"] = int(ctx_field.contents())
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


def _expert_count_key_candidates(arch: str) -> list[str]:
    """Return GGUF keys to inspect for the total expert count.

    Architecture-prefixed variants are tried first (the convention used by
    most upstream converters), then the bare key names for tools that omit
    the prefix.
    """
    keys: list[str] = []
    if arch:
        keys.extend(f"{arch}.{key}" for key in _EXPERT_COUNT_KEYS)
    keys.extend(_EXPERT_COUNT_KEYS)
    return keys


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
    keys = _expert_count_key_candidates(arch)
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


def trained_context_length(path: Path | str) -> int | None:
    """Return the model's trained context length, if known.

    Never raises: missing or malformed files return None.
    """
    meta = read_gguf_meta(path)
    arch = meta.get("architecture", "")
    # Architecture-specific key is canonical; keep a bare fallback so mocked
    # metadata and alternate converter spellings still work.
    keys: list[str] = []
    if arch:
        keys.append(f"{arch}.context_length")
    keys.append("context_length")
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


# ---------------------------------------------------------------------------
# Override-tensor helpers (--override-tensor <pattern>=CPU)
# ---------------------------------------------------------------------------

_OVERRIDABLE_BUFFER_TYPE = "CPU"


def _expert_projection_class(name: str) -> str | None:
    """Projection class of a routed-expert tensor, e.g. 'gate_up', 'down'.

    Some upstream MoE checkpoints spell the routed-expert marker ``chexps``
    (e.g. ``blk.0.ffn_down_chexps.weight``); the optional ``ch`` prefix is
    kept out of the projection class so pattern generation groups by the
    projection type, not by the spelling variant.
    """
    m = re.match(r"^blk\.\d+\.ffn_(.+?)_(?:ch)?exps\.", name)
    return m.group(1) if m else None


def override_tensor_saved_bytes(table: dict[str, int], patterns: list[str]) -> int:
    """Bytes that would move off the GPU for the given ``--override-tensor`` patterns.

    Verified against llama.cpp ``src/llama-model-loader.cpp:1161-1162``: each
    pattern is applied with ``std::regex_search`` over the full tensor name, so
    the Python side uses ``re.search`` for a faithful byte count.
    """
    compiled: list[re.Pattern[str]] = []
    for pat in patterns:
        try:
            compiled.append(re.compile(pat))
        except re.error as exc:
            raise ValueError(f"invalid override-tensor regex {pat!r}: {exc}") from exc
    return sum(
        nbytes
        for name, nbytes in table.items()
        if any(c.search(name) for c in compiled)
    )


def validate_override_patterns(table: dict[str, int] | None, patterns: list[str]) -> tuple[bool, str]:
    """Return (ok, error_message) for a proposed list of regex patterns.

    A pattern that matches zero tensors is rejected: unlike ``--n-cpu-moe``,
    ``-ot`` silently does nothing when its regex is wrong, and the only way
    to catch that without real hardware is to validate against the model's
    actual tensor list.
    """
    if table is None:
        return True, ""  # cannot validate without a readable GGUF
    for pat in patterns:
        try:
            compiled = re.compile(pat)
        except re.error as exc:
            return False, f"override_tensor regex {pat!r} is invalid: {exc}"
        if not any(compiled.search(name) for name in table):
            return False, f"override_tensor pattern {pat!r} matches zero tensors"
    return True, ""


def propose_override_tensor_patterns(table: dict[str, int]) -> list[str]:
    """Generate candidate ``--override-tensor`` regexes from real tensor names.

    Candidates are ordered from the cheapest expected throughput cost to the
    most expensive. The projection class that offloads the fewest bytes is
    tried first because moving less weight off the GPU means fewer PCIe
    round-trips per token; heavier options follow. The last candidate always
    offloads every routed-expert tensor, mirroring full ``--n-cpu-moe``.
    """
    if not table:
        return []
    by_class: dict[str, int] = {}
    for name, nbytes in table.items():
        if _expert_tensor_layer(name) is None:
            continue
        cls = _expert_projection_class(name)
        if cls is None:
            continue
        by_class[cls] = by_class.get(cls, 0) + nbytes
    if not by_class:
        return []
    # Cheapest-first: offload the smallest projection class first.
    ordered = sorted(by_class, key=lambda c: by_class[c])
    # Match both plain ``_exps`` and the upstream ``_chexps`` spelling, but
    # not shared experts (``_shexp``) or router gates (``_inp``).
    catch_all = r"blk\.\d+\.ffn_.*_(?:ch)?exps\."
    candidates: list[str] = []
    for cls in ordered:
        pat = rf"blk\.\d+\.ffn_{re.escape(cls)}_(?:ch)?exps\."
        # Skip if the catch-all would be identical (only one projection class).
        if pat != catch_all:
            candidates.append(pat)
    # Final fallback: all routed expert tensors, regardless of projection.
    candidates.append(catch_all)
    return candidates


# Routed-expert tensors are the ones llama.cpp's --n-cpu-moe N moves to the
# host: the per-layer `_exps` tensors of layers 0..N-1. Shared experts
# (ffn_*_shexp) and router gates (ffn_gate_inp) stay on the GPU, as do
# attention and embedding tensors.
#
# Match on the `_exps` marker rather than an enumerated list of projection
# names. Models fuse projections differently -- Gemma 4 26B-A4B ships
# `blk.N.ffn_gate_up_exps.weight`, a single fused gate+up tensor -- and an
# enumeration of {gate,up,down}_exps silently counted only the down
# projection there, under-reporting expert bytes by ~2.7x (143 MB/layer
# against a measured 393 MB/layer). Under-counting pushes
# min_moe_offload_layers to demand far more offloaded layers than needed,
# which is the expensive direction: on a B60 at ctx 16384, full offload cost
# 79% of prompt-eval throughput. Any future fusion spelling is matched by
# construction here.
#
# `.scale` sub-tensors (e.g. blk.N.ffn_down_exps.scale) are counted: they are
# quantisation metadata for those expert weights and travel with them to the
# host, so excluding them would under-report the saving again, just by less.
_EXPERT_TENSOR_RE = re.compile(r"^blk\.(\d+)\.ffn_\w*exps\b")
_SHARED_EXPERT_MARKER = "shexp"

# A sparse MoE keeps the large majority of its weights in routed experts. If
# the matched share falls below this, the tensor naming almost certainly moved
# again and we are silently under-counting, so warn loudly and name the
# tensors we failed to match. This is the check that would have caught the
# fused-projection bug without hardware.
_MIN_PLAUSIBLE_EXPERT_SHARE = 0.20


def _expert_tensor_layer(name: str) -> int | None:
    """Layer index of a routed-expert tensor name, or None.

    Shared experts are excluded: they are resident on every token, so
    llama.cpp keeps them on the device and --n-cpu-moe does not move them.
    """
    if _SHARED_EXPERT_MARKER in name:
        return None
    m = _EXPERT_TENSOR_RE.match(name)
    return int(m.group(1)) if m else None


def scan_weight_tensors(path: Path | str) -> tuple[int, dict[int, int]] | None:
    """One pass over the GGUF tensor table.

    Returns ``(total_weight_bytes, {layer_idx: routed_expert_bytes})``, or
    None when the file cannot be read. This is the single parser behind both
    ``estimate_weight_vram_bytes`` and the MoE offload accounting — the VRAM
    estimator and the offload search must never disagree about what a model
    weighs, so they share this scan.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        reader = gguf.GGUFReader(p)
    except Exception as exc:
        log.debug("gguf weight scan failed for %s: %s", p, exc)
        return None

    total = 0
    expert_by_layer: dict[int, int] = {}
    unmatched_expert_names: list[str] = []
    for tensor in reader.tensors:
        nbytes = _tensor_vram_bytes(tensor)
        total += nbytes
        name = getattr(tensor, "name", "") or ""
        layer = _expert_tensor_layer(name)
        if layer is not None:
            expert_by_layer[layer] = expert_by_layer.get(layer, 0) + nbytes
        elif "exps" in name and _SHARED_EXPERT_MARKER not in name:
            # Looks like a routed-expert tensor but did not parse as one.
            unmatched_expert_names.append(name)

    expert_bytes = sum(expert_by_layer.values())
    looks_moe = bool(expert_by_layer) or bool(unmatched_expert_names)
    if looks_moe and total > 0:
        share = expert_bytes / total
        if share < _MIN_PLAUSIBLE_EXPERT_SHARE:
            log.warning(
                "%s: routed-expert tensors are only %.1f%% of total weight bytes, "
                "which is implausibly low for a MoE model -- expert offload "
                "accounting is probably under-counting. Unmatched expert-like "
                "tensors: %s",
                p.name, share * 100,
                ", ".join(sorted(set(unmatched_expert_names))[:8]) or "(none)",
            )
    return total, expert_by_layer


def estimate_weight_vram_bytes(path: Path | str, *, n_cpu_moe: int = 0) -> int | None:
    """Estimate the VRAM footprint of the model weights alone.

    Sums the raw quantized tensor sizes from the GGUF file, which closely
    matches the device-side footprint on SYCL/Vulkan. Falls back to the file
    size if the GGUF cannot be inspected.

    When ``n_cpu_moe`` > 0, subtracts the routed-expert tensor bytes of the
    first N layers — exactly the bytes ``--n-cpu-moe N`` keeps on the host
    (N is a *layer* count, not an expert count). Returns None when the GGUF
    cannot be read, and also when offload accounting was requested but the
    model is MoE yet has no recognisable expert tensors: in that case the
    offloaded bytes are genuinely unknown and callers must NOT fall back to
    counting full weights — that fallback is what made offload-configured
    models unloadable.
    """
    scan = scan_weight_tensors(path)
    if scan is None:
        return None
    total, expert_by_layer = scan
    if n_cpu_moe <= 0:
        return total
    if not expert_by_layer:
        # Parsed fine but nothing to offload: either a dense model (the flag
        # is inert and the full count is correct) or an MoE variant whose
        # tensor names we don't recognise (the full count would be wrong).
        if is_moe(path):
            log.debug(
                "gguf %s is MoE but no routed-expert tensors were found; "
                "offload bytes undetermined",
                path,
            )
            return None
        return total
    saved = sum(b for layer, b in expert_by_layer.items() if layer < n_cpu_moe)
    return total - saved


def expert_tensor_bytes_by_layer(path: Path | str) -> dict[int, int] | None:
    """Routed-expert tensor bytes per layer index, or None if unreadable."""
    scan = scan_weight_tensors(path)
    return scan[1] if scan is not None else None


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------


def weight_tensor_table(path: Path | str) -> dict[str, int] | None:
    """Return ``{tensor_name: packed_bytes}`` for every weight tensor, or None.

    This is the single parser behind both the offload byte counters and the
    override-tensor pattern generator, so they can never disagree about what
    a model weighs or what a regex actually matches.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        reader = gguf.GGUFReader(p)
    except Exception as exc:
        log.debug("gguf weight table read failed for %s: %s", p, exc)
        return None
    return {
        getattr(tensor, "name", "") or "": _tensor_vram_bytes(tensor)
        for tensor in reader.tensors
    }


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
