"""Default llama.cpp launch recipes.

A *recipe* is the set of `llama-server` flags we'll feed for a given
(GPU arch, model size, model file size, target context length). Defaults are
chosen to be safe rather than maximal — we'd rather start small and let the user
crank context up than have a first-run experience that OOMs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from arc_llama.arch import Arch, ArchProfile, Backend, profile_for


class KVCacheType(str, Enum):
    F16 = "f16"
    F32 = "f32"
    Q8_0 = "q8_0"
    Q5_1 = "q5_1"
    Q5_0 = "q5_0"
    Q4_1 = "q4_1"
    Q4_0 = "q4_0"


# Approx KV bytes per token at f16 for a few well-known architectures.
# Numbers are tuned against `memory_breakdown_print` measurements on a real
# Battlemage B60 stack. They're upper bounds for sizing; actual usage with
# sliding-window attention (Gemma) is several × smaller again.
KV_PER_TOKEN_F16_BYTES: dict[str, int] = {
    "default": 70 * 1024,        # 70 KiB/token f16 — covers most ≤30B dense models
    "moe_a3b": 20 * 1024,        # ~20 KiB — Qwen3 30B/35B-A3B-class MoE
    "qwen3_dense": 67 * 1024,    # Qwen3 0.6B–32B dense (incl. Coder, Instruct)
    "qwen3_27b_dense": 67 * 1024,# kept for backwards compatibility
    "qwen2_5": 70 * 1024,        # Qwen2.5 / Qwen2.5-Coder dense
    "gemma_swa": 16 * 1024,      # Gemma 2/3/4 interleaved sliding-window attn
    "phi4": 72 * 1024,           # Phi-4 / Phi-4-reasoning 14.7B dense
    "llama3": 75 * 1024,         # Llama 3.x / 4 dense & small MoE distills
    "deepseek_r1_distill": 70 * 1024, # R1 distill on Llama/Qwen
}


FLASH_ATTN_VALUES = ("on", "off", "auto")


@dataclass
class LaunchRecipe:
    """A complete llama-server invocation, minus the model path and port."""
    n_gpu_layers: int = 999
    ctx: int = 8192
    parallel: int = 1
    cache_type_k: KVCacheType = KVCacheType.F16
    cache_type_v: KVCacheType = KVCacheType.F16
    threads: int | None = None
    temp: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    spec_type: str | None = None
    """Speculative decoding type, e.g. 'draft-mtp'."""
    spec_draft_n_max: int | None = None
    """Tokens to draft for speculative decoding (--spec-draft-n-max).

    Measured on Arc Pro B60 / Qwen3.6-27B-MTP (2026-07): n_max 1–4 give
    similar gen (~19–20 tok/s); n_max 5–6 regress gen to ~13–15. Prefer ≤4.
    """
    spec_draft_model: str | None = None
    """Path to a sidecar speculative-draft GGUF (--spec-draft-model).

    Some models ship their MTP/EAGLE heads as a separate small GGUF next to
    the main weights (e.g. `mtp-gemma-*.gguf`) rather than embedded. When set,
    llama-server loads it as the draft — this is what makes draft-mtp work for
    models whose main GGUF has no embedded MTP heads."""
    spec_draft_ngl: int | None = None
    """GPU layers for the draft model (--spec-draft-ngl); 999 = fully offloaded."""
    ubatch_size: int | None = None
    """Ubatch size (-ub). Leave unset to let llama.cpp pick the default."""
    batch_size: int | None = None
    """Logical batch size (-b). Must be >= ubatch_size when both are set."""
    flash_attn: str | None = None
    """Flash Attention: 'on' | 'off' | 'auto' | None (binary default).

    Old llama-server builds (pre ~b6300) expose -fa as a boolean that defaults
    to off; new builds take -fa {on,off,auto} and default to auto. `to_argv`
    translates per the probed binary style — see server_caps.probe_server_caps.
    """
    no_mmap: bool = False
    """Disable mmap (--no-mmap): slower reload, but the whole model is read
    up-front — avoids page-cache thrash when VRAM spill keeps tensors host-side."""
    mlock: bool = False
    """--mlock: pin host-side weights in RAM so they can't be swapped out."""
    n_cpu_moe: int | None = None
    """Number of MoE layers whose routed-expert tensors stay on CPU (--n-cpu-moe).

    N is a *layer* count: the expert weights of layers 0..N-1 are host-resident.
    Not an expert count — llama.cpp matches blk.N.ffn_{gate,up,down}_exps.*."""
    extra_flags: list[str] = field(default_factory=list)
    """Anything else the user wants appended to the command line verbatim."""

    override_tensor: list[str] | None = None
    """Tensor-buffer overrides as repeated ``--override-tensor <pattern>=CPU``.

    Each string is a regex over tensor names; together they move a subset of
    expert tensors to host memory with finer granularity than ``--n-cpu-moe``.
    If both ``override_tensor`` and ``n_cpu_moe`` are present, ``override_tensor``
    wins and ``n_cpu_moe`` is cleared: the two flags are alternative means to
    the same end and applying both would double-count the offload.
    """

    def to_argv(self, fa_takes_value: bool = True) -> list[str]:
        argv = [
            "-ngl", str(self.n_gpu_layers),
            "-c", str(self.ctx),
            "--parallel", str(self.parallel),
            "--cache-type-k", self.cache_type_k.value,
            "--cache-type-v", self.cache_type_v.value,
        ]
        if self.threads is not None:
            argv += ["-t", str(self.threads)]
        if self.temp is not None:
            argv += ["--temp", str(self.temp)]
        if self.top_p is not None:
            argv += ["--top-p", str(self.top_p)]
        if self.top_k is not None:
            argv += ["--top-k", str(self.top_k)]
        if self.spec_type:
            argv += ["--spec-type", self.spec_type]
        if self.spec_draft_model:
            argv += ["--spec-draft-model", self.spec_draft_model]
        if self.spec_draft_ngl is not None:
            argv += ["--spec-draft-ngl", str(self.spec_draft_ngl)]
        if self.spec_draft_n_max is not None:
            argv += ["--spec-draft-n-max", str(self.spec_draft_n_max)]
        if self.ubatch_size is not None:
            argv += ["-ub", str(self.ubatch_size)]
        if self.batch_size is not None:
            argv += ["-b", str(self.batch_size)]
        if self.flash_attn in FLASH_ATTN_VALUES:
            if fa_takes_value:
                argv += ["-fa", self.flash_attn]
            elif self.flash_attn == "on":
                # Old boolean-style flag; 'off' is that style's default and
                # 'auto' is inexpressible, so both fall through to no flag.
                argv += ["-fa"]
        if self.no_mmap:
            argv += ["--no-mmap"]
        if self.mlock:
            argv += ["--mlock"]
        if self.override_tensor:
            # llama.cpp common/arg.cpp:247 parses --override-tensor as
            # <pattern>=<buffer_type>, splitting each value on its first '='.
            # CPU is a valid backend buffer name, so render as ``pat=CPU``.
            for pat in self.override_tensor:
                argv += ["--override-tensor", f"{pat}=CPU"]
        elif self.n_cpu_moe is not None:
            argv += ["--n-cpu-moe", str(self.n_cpu_moe)]
        argv += list(self.extra_flags)
        return argv


def estimate_kv_bytes(ctx: int, kv_type: KVCacheType, kv_class: str = "default") -> int:
    """Rough estimate of KV-cache bytes at runtime."""
    f16_per_token = KV_PER_TOKEN_F16_BYTES.get(kv_class, KV_PER_TOKEN_F16_BYTES["default"])
    scale = {
        KVCacheType.F32: 2.0,
        KVCacheType.F16: 1.0,
        KVCacheType.Q8_0: 0.5,
        KVCacheType.Q5_1: 0.375,
        KVCacheType.Q5_0: 0.375,
        KVCacheType.Q4_1: 0.3125,
        KVCacheType.Q4_0: 0.3125,
    }.get(kv_type, 1.0)
    return int(ctx * f16_per_token * scale)


DEFAULT_CTX_CAP = 131072
"""Hard ceiling on auto-suggested context length. VRAM math will sometimes
say a 500k+ ctx fits, but real models top out around 128k–256k and our
KV-per-token estimates are inherently approximate. Cap the auto-suggestion
at 131072 — users who want more can override the recipe per-model."""


def suggest_ctx(
    vram_mb: int,
    model_file_mb: int,
    kv_type: KVCacheType,
    kv_class: str = "default",
    compute_buffer_mb: int = 768,
    safety_margin_mb: int = 256,
    ctx_cap: int = DEFAULT_CTX_CAP,
    trained_ctx: int | None = None,
    parallel: int = 1,
) -> int:
    """Pick the largest power-of-2-ish context that fits comfortably in VRAM.

    Rounds *down* to the nearest multiple of 4096 and clamps to `ctx_cap`.
    If `trained_ctx` is known, the result is also clamped to it (the model
    silently uses its trained length as a ceiling at runtime).

    The KV cache scales linearly with the number of parallel sequences, so
    `parallel` multiplies the per-token estimate.
    """
    free_for_kv = vram_mb - model_file_mb - compute_buffer_mb - safety_margin_mb
    if free_for_kv <= 0:
        return 4096  # last-resort minimum; user should pick a smaller quant
    f16_per_token = KV_PER_TOKEN_F16_BYTES.get(kv_class, KV_PER_TOKEN_F16_BYTES["default"])
    scale = {
        KVCacheType.F32: 2.0,
        KVCacheType.F16: 1.0,
        KVCacheType.Q8_0: 0.5,
        KVCacheType.Q5_1: 0.375,
        KVCacheType.Q5_0: 0.375,
        KVCacheType.Q4_1: 0.3125,
        KVCacheType.Q4_0: 0.3125,
    }.get(kv_type, 1.0)
    if parallel < 1:
        parallel = 1
    bytes_per_token = int(f16_per_token * scale) * parallel
    if bytes_per_token <= 0:
        return 4096
    max_tokens = (free_for_kv * 1024 * 1024) // bytes_per_token
    rounded = (max_tokens // 4096) * 4096
    # Clamp to the smallest applicable ceiling: trained context, then ctx_cap.
    ceiling = ctx_cap
    if trained_ctx is not None and trained_ctx < ceiling:
        ceiling = trained_ctx
    return max(4096, min(rounded, ceiling))


PERF_UBATCH_MIN_VRAM_MB = 16384
"""Only default to a large ubatch on cards with real VRAM headroom — the
compute buffer grows roughly linearly with ubatch, and on 8–12 GB cards a
previously-fitting model could stop fitting."""

PERF_UBATCH = 1024
"""Prompt processing on Arc is very sensitive to ubatch. Measured on Arc Pro
B60 / Qwen3.6-27B-MTP (2026-07): raising -ub 512→1024 lifts prompt-eval from
~340 to ~420 tok/s (~23%, all runs non-overlapping) with no gen regression.
See bench_results/SUMMARY.md."""

PERF_BATCH = 2048
"""Logical batch size (-b) paired with PERF_UBATCH. llama.cpp requires
batch_size >= ubatch_size; 2048 is the upstream stock default."""

PERF_COMPUTE_BUFFER_MB = 1536
"""Compute-buffer estimate used for ctx sizing when the perf ubatch applies
(vs. the conservative 768 MiB default at llama.cpp's stock ubatch of 512)."""


def default_recipe(
    arch: Arch,
    vram_mb: int,
    model_file_mb: int,
    kv_class: str = "default",
    prefer_q8_kv: bool = True,
    backend: Backend = Backend.SYCL,
    trained_ctx: int | None = None,
    parallel: int = 1,
) -> LaunchRecipe:
    """A safe starting recipe for a freshly added model on a given arch/backend."""
    profile: ArchProfile = profile_for(arch)
    extra_flags: list[str] = []
    # SYCL: express flash-attn via the recipe field so server_caps can emit
    # the right dialect. Vulkan quantized V-cache still injects via extra_flags
    # (policy also enforces this for older configs without flash_attn set).
    flash_attn: str | None = "auto"
    if backend == Backend.VULKAN:
        # Vulkan quantized V-cache needs --flash-attn (llama.cpp requirement).
        # SYCL production configs run fine with q8 V and no FA flag — do not
        # inject it there (verified on B60 production stack).
        use_q8 = prefer_q8_kv and profile.safe_kv_q8_vulkan
        if use_q8:
            extra_flags.extend(["--flash-attn", "on"])
            flash_attn = None  # avoid emitting both -fa auto and --flash-attn on
        else:
            flash_attn = "auto"
    else:
        use_q8 = prefer_q8_kv and profile.safe_kv_q8
    kv_type = KVCacheType.Q8_0 if use_q8 else KVCacheType.F16
    # Bump ubatch above llama.cpp's stock 512 when the card can absorb the
    # bigger compute buffer; budget the larger buffer into the ctx suggestion.
    perf_batching = vram_mb >= PERF_UBATCH_MIN_VRAM_MB
    ctx = suggest_ctx(
        vram_mb=vram_mb,
        model_file_mb=model_file_mb,
        kv_type=kv_type,
        kv_class=kv_class,
        compute_buffer_mb=PERF_COMPUTE_BUFFER_MB if perf_batching else 768,
        trained_ctx=trained_ctx,
        parallel=parallel,
    )
    return LaunchRecipe(
        n_gpu_layers=999,
        ctx=ctx,
        parallel=parallel,
        cache_type_k=kv_type,
        cache_type_v=kv_type,
        flash_attn=flash_attn,
        ubatch_size=PERF_UBATCH if perf_batching else None,
        batch_size=PERF_BATCH if perf_batching else None,
        extra_flags=extra_flags,
    )


def recipe_to_dict(recipe: LaunchRecipe) -> dict:
    """Serialise a recipe to the TOML-friendly dict stored in ModelConfig.recipe.

    Only always-meaningful fields are emitted unconditionally; optional fields
    are included only when set, so configs stay small and None never reaches
    the TOML writer.
    """
    d: dict = {
        "n_gpu_layers": recipe.n_gpu_layers,
        "ctx": recipe.ctx,
        "parallel": recipe.parallel,
        "cache_type_k": recipe.cache_type_k.value,
        "cache_type_v": recipe.cache_type_v.value,
    }
    if recipe.threads is not None:
        d["threads"] = recipe.threads
    if recipe.temp is not None:
        d["temp"] = recipe.temp
    if recipe.top_p is not None:
        d["top_p"] = recipe.top_p
    if recipe.top_k is not None:
        d["top_k"] = recipe.top_k
    if recipe.flash_attn is not None:
        d["flash_attn"] = recipe.flash_attn
    if recipe.ubatch_size is not None:
        d["ubatch_size"] = recipe.ubatch_size
    if recipe.batch_size is not None:
        d["batch_size"] = recipe.batch_size
    if recipe.spec_type is not None:
        d["spec_type"] = recipe.spec_type
    if recipe.spec_draft_model is not None:
        d["spec_draft_model"] = recipe.spec_draft_model
    if recipe.spec_draft_ngl is not None:
        d["spec_draft_ngl"] = recipe.spec_draft_ngl
    if recipe.spec_draft_n_max is not None:
        d["spec_draft_n_max"] = recipe.spec_draft_n_max
    if recipe.n_cpu_moe is not None:
        d["n_cpu_moe"] = recipe.n_cpu_moe
    if recipe.override_tensor:
        d["override_tensor"] = list(recipe.override_tensor)
    if recipe.no_mmap:
        d["no_mmap"] = True
    if recipe.mlock:
        d["mlock"] = True
    if recipe.extra_flags:
        d["extra_flags"] = list(recipe.extra_flags)
    return d
