# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-07-23

Highlights: `arc-llama install-runtime` downloads a prebuilt portable Vulkan llama-server, removing the need to install oneAPI or build from source.

### Added
- `arc-llama install-runtime` command to download a prebuilt llama-server from official ggml-org/llama.cpp releases. Vulkan is the default backend, portable, and requires no oneAPI installation. SYCL remains optional. Verified end to end on Battlemage B60.
- `arc-llama tune` staged autotuner for KV cache type, ubatch, and flash attention. It measures performance on your card and writes the winning recipe. Added `arc-llama tune --all` to sweep every registered model in one run.
- `arc-llama benchmark` harness for prompt-eval and generation tokens per second.
- Vulkan backend support via `backend = "vulkan"` in the config, alongside SYCL.
- Windows support for the launcher, CLI, and config paths. CI is green on windows-latest and ubuntu-latest for Python 3.10 to 3.12.
- Auto-scan for new GGUFs on `serve` startup, and auto-detection of sidecar speculative-draft (MTP) models.
- Experimental agent loop, terminal agent UI, MCP client, checkpoints, chat persistence, and chat export/import. Gated behind `ARC_LLAMA_EXPERIMENTAL_AGENT`.

### Changed
- `arc-llama init` now writes a config even when no llama-server is present, and points users to `install-runtime`.
- `init` and `install-runtime` set each GPU's `backend` to match the actual binary. Previously, it always defaulted to SYCL, which mismatched Vulkan builds.
- `doctor` now surfaces AOT build guidance, device-ID VRAM fallback, metadata-based KV class, and host checks. Added VRAM guard before load, crash-log surfacing, config migration, and Prometheus-style metrics.

### Fixed
- Backend detection now scans sibling `libggml-*.so` shared libraries for modern modular builds, and no longer false-matches bare backend NAME strings in `libggml.so`. Previously, a downloaded Vulkan build reported its backend as "unknown", then as "sycl".
- Benchmark generation and prompt-eval measurement accuracy.
- MTP `ubatch_size` regression.
- MoE detection.

## [0.4.0]
Initial public releases.
