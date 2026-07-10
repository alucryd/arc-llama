"""On-disk config schema (TOML) at $XDG_CONFIG_HOME/arc-llama/config.toml.

Schema (v1):

```toml
[server]
host = "127.0.0.1"
port = 11436
single_resident = true   # only one llama-server alive at a time across all GPUs

[paths]
llama_server = "/usr/local/bin/llama-server"
models_dir   = "~/.local/share/arc-llama/models"
state_dir    = "~/.local/state/arc-llama"

[[gpus]]
pci_slot   = "0000:03:00.0"
sycl_index = 0          # passed as ONEAPI_DEVICE_SELECTOR=level_zero:N
arch       = "battlemage"
vram_mb    = 24480
enabled    = true

[[models]]
name             = "qwen3.6-27b"
display_name     = "Qwen 3.6 27B (dense)"
path             = "/mnt/storage/models/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf"
gpu_pci_slot     = "0000:03:00.0"
port             = 8083
kv_class         = "default"
[models.recipe]
ctx              = 131072
cache_type_k     = "q8_0"
cache_type_v     = "q8_0"
n_gpu_layers     = 999
parallel         = 1
extra_flags      = ["--reasoning", "off"]
```
"""
from __future__ import annotations

import logging
import os
import secrets
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

if sys.version_info >= (3, 11):
    import tomllib  # noqa: F401
    _toml_load = tomllib.load  # type: ignore[attr-defined]
else:
    import tomli  # type: ignore[import-not-found]
    _toml_load = tomli.load

from arc_llama.arch import Arch, Backend
from arc_llama.recipes import KVCacheType, LaunchRecipe

CONFIG_VERSION = 1


def _xdg_config_home() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def _xdg_data_home() -> Path:
    if sys.platform == "win32":
        return Path(
            os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        )
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def _xdg_state_home() -> Path:
    if sys.platform == "win32":
        return Path(
            os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        )
    return Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")


def default_config_path() -> Path:
    return _xdg_config_home() / "arc-llama" / "config.toml"


def default_models_dir() -> Path:
    return _xdg_data_home() / "arc-llama" / "models"


def default_state_dir() -> Path:
    return _xdg_state_home() / "arc-llama"


def default_skills_dir() -> Path:
    return _xdg_config_home() / "arc-llama" / "skills"


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 11437
    single_resident: bool = True
    admin_token: str | None = None
    """Bearer token required for destructive admin endpoints (load, stop, edit, scan).
    Set via `arc-llama serve --admin-token` or env var ARC_LLAMA_ADMIN_TOKEN."""
    """Why 11437? Ollama owns 11434 by default, and IPEX-LLM-Ollama installs
    sometimes use 11435/11436. 11437 is the first port in that neighbourhood
    that nobody else seems to claim."""
    """If True, only one llama-server runs at a time. If False, models share VRAM
    on a best-effort basis — set this only if you have generous VRAM headroom."""


@dataclass
class UpstreamConfig:
    """An OpenAI-compatible API endpoint whose models are merged into
    arc-llama's model list and forwarded transparently.

    Models from upstreams are shown in the web UI with an "upstream" source
    tag. When a request targets an upstream model, arc-llama proxies it to
    the upstream instead of starting a local llama-server.
    """
    url: str
    """Base URL of the upstream, e.g. 'http://127.0.0.1:11435' or 'http://192.168.1.50:8080'."""
    name: str = ""
    """Short label for the UI (e.g. 'ollama', 'proxy'). Default: hostname from url."""


@dataclass
class PathsConfig:
    llama_server: str = "llama-server"
    """Path to the llama-server binary. Plain `llama-server` resolves via PATH."""
    models_dir: str = field(default_factory=lambda: str(default_models_dir()))
    state_dir: str = field(default_factory=lambda: str(default_state_dir()))
    skills_dir: str = field(default_factory=lambda: str(default_skills_dir()))
    """Directory containing user skill Python files."""
    scan_paths: list[str] = field(default_factory=list)
    """Extra directories `arc-llama scan` walks looking for GGUFs. The
    `models_dir` is always scanned in addition to these."""


@dataclass
class AgentConfig:
    root: str = "."
    """Default filesystem root for the agent file/shell tools.

    Request-level `root` overrides this. Use an absolute path if you access
    arc-llama from another machine and `.` (the server's working directory)
    is not what you want.
    """
    profile: str | None = None
    """Default profile name for selecting which MCP servers are active.

    A profile references MCP servers by name. When a profile is active,
    only the MCP servers listed in that profile are loaded. If unset, all
    configured MCP servers are loaded.
    """


@dataclass
class ProfileConfig:
    """Named whitelist of MCP servers."""

    name: str = ""
    mcp_servers: list[str] = field(default_factory=list)


@dataclass
class MCPServerConfig:
    """Configuration for one stdio MCP server."""

    name: str = ""
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class GPUConfig:
    pci_slot: str
    sycl_index: int
    arch: str  # Arch.value
    vram_mb: int | None = None
    enabled: bool = True
    name: str = ""
    backend: str = Backend.SYCL.value  # Backend.value


@dataclass
class ModelConfig:
    name: str                  # short id, also URL-safe (e.g. "qwen3.6-27b")
    path: str                  # absolute path to the GGUF
    port: int                  # backend port for this model's llama-server
    gpu_pci_slot: str          # which detected GPU to bind to
    display_name: str = ""
    kv_class: str = "default"  # used for VRAM estimation
    recipe: dict[str, Any] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    """Extra strings that should match this model in the OpenAI `model` field."""

    def launch_recipe(self) -> LaunchRecipe:
        r = self.recipe or {}
        return LaunchRecipe(
            n_gpu_layers=int(r.get("n_gpu_layers", 999)),
            ctx=int(r.get("ctx", 8192)),
            parallel=int(r.get("parallel", 1)),
            cache_type_k=KVCacheType(r.get("cache_type_k", "f16")),
            cache_type_v=KVCacheType(r.get("cache_type_v", "f16")),
            threads=r.get("threads"),
            temp=r.get("temp"),
            top_p=r.get("top_p"),
            top_k=r.get("top_k"),
            spec_type=r.get("spec_type"),
            spec_draft_n_max=(
                int(r["spec_draft_n_max"]) if r.get("spec_draft_n_max") is not None else None
            ),
            ubatch_size=r.get("ubatch_size"),
            batch_size=r.get("batch_size"),
            flash_attn=r.get("flash_attn"),
            no_mmap=bool(r.get("no_mmap", False)),
            mlock=bool(r.get("mlock", False)),
            n_cpu_moe=r.get("n_cpu_moe"),
            extra_flags=list(r.get("extra_flags", [])),
        )


@dataclass
class Config:
    version: int = CONFIG_VERSION
    server: ServerConfig = field(default_factory=ServerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    gpus: list[GPUConfig] = field(default_factory=list)
    models: list[ModelConfig] = field(default_factory=list)
    upstreams: list[UpstreamConfig] = field(default_factory=list)
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)
    profiles: list[ProfileConfig] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def find_profile(self, name: str | None) -> ProfileConfig | None:
        if not name:
            return None
        for p in self.profiles:
            if p.name == name:
                return p
        return None

    def active_profile_name(self, profile_name: str | None = None) -> str | None:
        """Return the effective profile name.

        Explicit ``profile_name`` wins, then ``agent.profile`` in config,
        then None (meaning all MCP servers are active).
        """
        if profile_name:
            return profile_name
        if self.agent.profile:
            return self.agent.profile
        return None

    def active_mcp_servers(
        self, profile_name: str | None = None
    ) -> list[MCPServerConfig]:
        """Return MCP servers that belong to the active profile.

        If no profile is active, all configured servers are returned. Unknown
        server names listed in a profile are ignored with a warning.
        """
        name = self.active_profile_name(profile_name)
        if not name:
            return list(self.mcp_servers)
        profile = self.find_profile(name)
        if profile is None:
            logging.getLogger("arc_llama.config").warning(
                "Profile %r not found; loading all MCP servers", name
            )
            return list(self.mcp_servers)
        allowed = set(profile.mcp_servers)
        found: dict[str, MCPServerConfig] = {}
        for server in self.mcp_servers:
            if server.name in allowed:
                found[server.name] = server
        for missing in allowed - set(found):
            logging.getLogger("arc_llama.config").warning(
                "Profile %r references unknown MCP server %r", name, missing
            )
        return [found[name] for name in profile.mcp_servers if name in found]

    def find_model(self, query: str) -> ModelConfig | None:
        """Match a user-supplied model id against name/display_name/aliases.

        Match is exact-name first, then substring-on-aliases, then case-insensitive
        substring on display_name and basename(path) — so the OpenAI request body's
        `model` field can be the GGUF filename, the short name, or a friendly alias.
        """
        if not query:
            return None
        for m in self.models:
            if m.name == query:
                return m
        for m in self.models:
            if query in m.aliases:
                return m
        ql = query.lower()
        for m in self.models:
            haystacks = [
                m.name.lower(),
                m.display_name.lower(),
                Path(m.path).name.lower(),
                *(a.lower() for a in m.aliases),
            ]
            if any(ql in h for h in haystacks):
                return m
        return None

    def find_gpu(self, pci_slot: str) -> GPUConfig | None:
        for g in self.gpus:
            if g.pci_slot == pci_slot:
                return g
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_toml_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": self.version,
            "server": asdict(self.server),
            "paths": asdict(self.paths),
            "agent": asdict(self.agent),
            "gpus": [asdict(g) for g in self.gpus],
            "models": [asdict(m) for m in self.models],
            "upstreams": [asdict(u) for u in self.upstreams],
            "mcp_servers": [asdict(s) for s in self.mcp_servers],
            "profiles": [asdict(p) for p in self.profiles],
        }
        return _strip_none(d)

    def save(self, path: Path | None = None) -> Path:
        path = path or default_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            tomli_w.dump(self.to_toml_dict(), f)
        return path


def _strip_none(obj: Any) -> Any:
    """Recursively remove dict keys whose value is None.

    TOML has no null type, so None values crash tomli_w. This is applied
    to the whole config tree before persistence.
    """
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


def migrate_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Bump an on-disk config dict to the current schema version.

    Applies field-level migrations so older configs (0.1 → 0.2 → 0.3) pick up
    new defaults without losing user edits.
    """
    version = int(raw.get("version", 1))
    if version > CONFIG_VERSION:
        raise ValueError(
            f"config version {version} is newer than the supported version "
            f"{CONFIG_VERSION}; upgrade arc-llama"
        )

    # Ensure all top-level sections exist so downstream code can assume them.
    raw.setdefault("server", {})
    raw.setdefault("paths", {})
    raw.setdefault("agent", {})
    raw.setdefault("gpus", [])
    raw.setdefault("models", [])
    raw.setdefault("upstreams", [])
    raw.setdefault("mcp_servers", [])
    raw.setdefault("profiles", [])

    # 0.2 → 0.3: GPU backend field (SYCL vs Vulkan). Default to SYCL to match
    # pre-0.3 behaviour, but log so the user knows they can set it explicitly.
    for gpu in raw.get("gpus", []):
        if not isinstance(gpu, dict):
            continue
        if "backend" not in gpu:
            gpu["backend"] = Backend.SYCL.value
            logging.getLogger("arc_llama.config").warning(
                "GPU %s is missing the 'backend' field; defaulting to '%s'. "
                "Set it to '%s' if you are using a Vulkan llama-server build.",
                gpu.get("pci_slot", "?"),
                Backend.SYCL.value,
                Backend.VULKAN.value,
            )

    # 0.3: agent settings default to a server-side project root of ".".
    agent = raw.get("agent", {})
    if "root" not in agent:
        agent["root"] = "."
    if "profile" not in agent:
        agent["profile"] = None

    # Ensure newer server fields exist with safe defaults.
    server = raw.get("server", {})
    if "admin_token" not in server:
        server["admin_token"] = None

    # Ensure model defaults that were introduced across releases.
    for model in raw.get("models", []):
        if not isinstance(model, dict):
            continue
        if "kv_class" not in model:
            model["kv_class"] = "default"
        if "display_name" not in model:
            model["display_name"] = ""
        if "aliases" not in model:
            model["aliases"] = []

    raw["version"] = CONFIG_VERSION
    return raw


def validate_config(raw: dict[str, Any]) -> None:
    """Basic structural validation for a loaded config dict."""
    if not isinstance(raw.get("version"), int):
        raise ValueError("config 'version' must be an integer")
    if not isinstance(raw.get("server", {}), dict):
        raise ValueError("config 'server' must be a table")
    if not isinstance(raw.get("paths", {}), dict):
        raise ValueError("config 'paths' must be a table")
    if not isinstance(raw.get("agent", {}), dict):
        raise ValueError("config 'agent' must be a table")
    if not isinstance(raw.get("gpus", []), list):
        raise ValueError("config 'gpus' must be an array")
    if not isinstance(raw.get("models", []), list):
        raise ValueError("config 'models' must be an array")
    if not isinstance(raw.get("upstreams", []), list):
        raise ValueError("config 'upstreams' must be an array")
    if not isinstance(raw.get("mcp_servers", []), list):
        raise ValueError("config 'mcp_servers' must be an array")
    if not isinstance(raw.get("profiles", []), list):
        raise ValueError("config 'profiles' must be an array")


def _resolve_admin_token(cfg: Config, path: Path, *, persist: bool) -> None:
    """Fill in cfg.server.admin_token so admin/auto_confirm auth is never a no-op.

    ARC_LLAMA_ADMIN_TOKEN always wins and is never written to disk (so it can
    be overridden per-invocation, e.g. in containers). Otherwise, if no token
    is configured yet, generate one and persist it so it survives restarts --
    admin endpoints and `auto_confirm` agent runs would otherwise be
    unauthenticated by default.
    """
    env_token = os.environ.get("ARC_LLAMA_ADMIN_TOKEN")
    if env_token:
        cfg.server.admin_token = env_token
        return
    if cfg.server.admin_token:
        return
    cfg.server.admin_token = secrets.token_urlsafe(32)
    if persist:
        cfg.save(path)
    logging.getLogger("arc_llama.config").warning(
        "No admin_token was configured -- generated one and saved it to %s. "
        "Admin endpoints and auto_confirm agent runs now require "
        "'Authorization: Bearer %s'.",
        path,
        cfg.server.admin_token,
    )


def load_config(path: Path | None = None) -> Config:
    path = path or default_config_path()
    if not path.exists():
        cfg = Config()
        _resolve_admin_token(cfg, path, persist=False)
        return cfg
    with open(path, "rb") as f:
        raw = _toml_load(f)
    raw = migrate_config(raw)
    validate_config(raw)
    cfg = Config(
        version=int(raw.get("version", CONFIG_VERSION)),
        server=ServerConfig(**raw.get("server", {})),
        paths=PathsConfig(**raw.get("paths", {})),
        agent=AgentConfig(**raw.get("agent", {})),
        gpus=[GPUConfig(**g) for g in raw.get("gpus", [])],
        models=[ModelConfig(**m) for m in raw.get("models", [])],
        upstreams=[UpstreamConfig(**u) for u in raw.get("upstreams", [])],
        mcp_servers=[MCPServerConfig(**s) for s in raw.get("mcp_servers", [])],
        profiles=[ProfileConfig(**p) for p in raw.get("profiles", [])],
    )
    _resolve_admin_token(cfg, path, persist=True)
    return cfg


def init_config_from_detection(detected_gpus, llama_server_path: str | None = None) -> Config:
    """Build a fresh Config from a detect.detect_gpus() result."""
    cfg = Config()
    if llama_server_path:
        cfg.paths.llama_server = llama_server_path
    enabled_set = False
    for g in detected_gpus:
        gc = GPUConfig(
            pci_slot=g.pci_slot,
            sycl_index=g.sycl_index_hint,
            arch=g.arch.value if hasattr(g.arch, "value") else str(g.arch),
            vram_mb=g.vram_mb,
            enabled=False,
            name=g.name,
            backend=Backend.SYCL.value,
        )
        # Prefer the highest-VRAM Battlemage / Alchemist as the default GPU.
        if not enabled_set and g.arch in (Arch.BATTLEMAGE, Arch.ALCHEMIST) and g.vram_mb:
            gc.enabled = True
            enabled_set = True
        cfg.gpus.append(gc)
    if cfg.gpus and not enabled_set:
        cfg.gpus[0].enabled = True
    return cfg
