"""Model registry — adding, removing, and (optionally) downloading GGUFs from HF.

The downloader is intentionally a thin shim around `huggingface_hub.hf_hub_download`
so users who already have models on disk never need network access.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc_llama.config import (
    Config,
    ModelConfig,
)
from arc_llama.recipes import KVCacheType, default_recipe

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
HF_SPEC_RE = re.compile(
    r"^(?P<repo>[^@:\s/]+/[^@:\s/]+)(?::(?P<file>[^@\s]+))?$"
)


@dataclass
class HFModelSpec:
    """Parsed user input like `unsloth/gemma-4-31B-it-GGUF:Q4_K_M`."""
    repo: str
    file: str | None  # exact filename; if None, we glob the repo for a match
    quant: str | None # short hint like "Q4_K_M", used when file is None


def parse_hf_spec(spec: str) -> HFModelSpec:
    m = HF_SPEC_RE.match(spec)
    if not m:
        raise ValueError(
            f"Invalid HF spec '{spec}'. Expected 'org/repo' or 'org/repo:filename' "
            f"or 'org/repo:Q4_K_M'."
        )
    repo = m.group("repo")
    file = m.group("file")
    quant = None
    if file and not file.endswith(".gguf"):
        # Treat short tokens like Q4_K_M as a quant hint, not a filename.
        if re.fullmatch(r"(IQ|Q|UD-)?[A-Z0-9_]+", file):
            quant = file
            file = None
    return HFModelSpec(repo=repo, file=file, quant=quant)


def _next_free_port(used: set[int], start: int = 18080) -> int:
    p = start
    while p in used:
        p += 1
    return p


def _short_name_from(repo: str, file: str | None) -> str:
    """Generate a short, slug-friendly name from an HF repo/filename."""
    base = repo.split("/")[-1].lower()
    # Strip common GGUF suffixes
    for suffix in ("-gguf", "_gguf"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    base = re.sub(r"[^a-z0-9._-]+", "-", base).strip("-")
    if file:
        # Append the quant tier if obvious from the filename
        m = re.search(r"(IQ\d[A-Z_]*|Q\d[A-Z_]*|UD-[A-Z0-9_]+)", file, re.IGNORECASE)
        if m:
            base = f"{base}-{m.group(1).lower()}"
    return base or "model"


def add_local_model(
    cfg: Config,
    *,
    name: str,
    path: str,
    gpu_pci_slot: str,
    port: int | None = None,
    display_name: str = "",
    kv_class: str = "default",
    aliases: list[str] | None = None,
    recipe_overrides: dict[str, Any] | None = None,
) -> ModelConfig:
    """Register an already-downloaded GGUF in the config.

    Picks a recipe based on the bound GPU's arch and VRAM, then applies any overrides.
    """
    if not NAME_RE.match(name):
        raise ValueError(
            f"Model name '{name}' must match [a-z0-9][a-z0-9._-]*"
        )
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    gpu = cfg.find_gpu(gpu_pci_slot)
    if gpu is None:
        raise ValueError(f"GPU {gpu_pci_slot} not in config — run `arc-llama init` first.")
    used_ports = {m.port for m in cfg.models}
    port = port or _next_free_port(used_ports)
    if port in used_ports:
        raise ValueError(f"Port {port} already in use by another model.")
    if any(m.name == name for m in cfg.models):
        raise ValueError(f"Model name '{name}' already registered.")
    # Build a recipe that fits this GPU.
    from arc_llama.arch import Arch
    arch = Arch(gpu.arch) if gpu.arch else Arch.UNKNOWN
    recipe = default_recipe(
        arch=arch,
        vram_mb=gpu.vram_mb or 8192,
        model_file_mb=p.stat().st_size // (1024 * 1024),
        kv_class=kv_class,
    )
    recipe_dict: dict[str, Any] = {
        "n_gpu_layers": recipe.n_gpu_layers,
        "ctx": recipe.ctx,
        "parallel": recipe.parallel,
        "cache_type_k": recipe.cache_type_k.value,
        "cache_type_v": recipe.cache_type_v.value,
    }
    if recipe_overrides:
        recipe_dict.update(recipe_overrides)
    mc = ModelConfig(
        name=name,
        path=str(p),
        port=port,
        gpu_pci_slot=gpu_pci_slot,
        display_name=display_name or name,
        kv_class=kv_class,
        recipe=recipe_dict,
        aliases=aliases or [p.name],
    )
    cfg.models.append(mc)
    return mc


def download_from_hf(
    spec: HFModelSpec,
    *,
    target_dir: Path,
    token: str | None = None,
    progress: bool = True,
) -> Path:
    """Resolve a HFModelSpec to a concrete file path under `target_dir`.

    Imported lazily so users without huggingface-hub can still use arc-llama
    with already-downloaded files.
    """
    target_dir = Path(target_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface-hub is required for downloads. "
            "Install with `pip install huggingface-hub` or pre-download the GGUF "
            "and use `arc-llama add --path /path/to/model.gguf`."
        ) from e

    file = spec.file
    if file is None:
        api = HfApi(token=token)
        files = [
            f for f in api.list_repo_files(spec.repo)
            if f.endswith(".gguf")
        ]
        if spec.quant:
            ql = spec.quant.lower()
            matches = [f for f in files if ql in f.lower()]
            if not matches:
                raise FileNotFoundError(
                    f"No GGUF in {spec.repo} matched quant hint '{spec.quant}'. "
                    f"Available: {', '.join(sorted(files))}"
                )
            # Prefer uniform quants (no "_xl" / "ud-") if multiple matched.
            uniform = [f for f in matches if "_xl" not in f.lower() and "ud-" not in f.lower()]
            file = sorted(uniform or matches)[0]
        elif len(files) == 1:
            file = files[0]
        else:
            raise ValueError(
                f"Repo {spec.repo} has {len(files)} GGUF files; specify one with "
                f"`{spec.repo}:<filename>` or `{spec.repo}:Q4_K_M`."
            )
    return Path(hf_hub_download(
        repo_id=spec.repo,
        filename=file,
        local_dir=str(target_dir),
        token=token,
    ))
