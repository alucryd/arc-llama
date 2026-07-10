#!/usr/bin/env python3
"""Empirical llama-server throughput on real Arc hardware.

Methodology (required):
  * POST /completion (not /v1/chat/completions)
  * every request: cache_prompt=false
  * read timings.prompt_per_second / timings.predicted_per_second from the
    response body (llama-server native timings — no wall-clock math)

Does NOT encode policy. Prints a table and writes JSON for humans to decide
what to wire into default_recipe / build_plan / policy.

Example:
  python scripts/empirical_throughput.py --quick
  python scripts/empirical_throughput.py --suite full
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults for this host (B60 + local SYCL build + Qwen3.6-27B MTP)
# ---------------------------------------------------------------------------

DEFAULT_LLAMA = "/mnt/storage/llama.cpp/build/bin/llama-server"
DEFAULT_MODEL = (
    "/mnt/storage/models/qwen3.6-27b/Qwen3.6-27B-MTP-UD-Q4_K_XL.gguf"
)
DEFAULT_PORT = 18199
DEFAULT_HOST = "127.0.0.1"

# Battlemage SYCL env — match arc_llama.arch BATTLEMAGE_PROFILE
BASE_ENV = {
    "ONEAPI_DEVICE_SELECTOR": "level_zero:0",
    "ZES_ENABLE_SYSMAN": "1",
    "SYCL_CACHE_PERSISTENT": "0",
}
STRIP_ENV = (
    "GGML_SYCL_DISABLE_OPT",
    "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
)

PROMPT_TARGET_TOKENS = 1024
N_PREDICT = 64
WARMUP_PREDICT = 8
# Gen tok/s under MTP is noisy (~20% spread on identical config) and
# prompt-sensitive. Prefer ≥5 before ranking configs.
REPEATS = 5

HEALTH_TIMEOUT_S = 300  # generous for SYCL JIT + 17GB load


@dataclass
class ConfigRun:
    name: str
    argv_extra: list[str]
    notes: str = ""


@dataclass
class TimingRow:
    name: str
    argv: list[str]
    prompt_tok_s: float | None = None
    gen_tok_s: float | None = None
    prompt_n: int | None = None
    gen_n: int | None = None
    ctx_reported: int | None = None
    error: str | None = None
    raw_best: dict = field(default_factory=dict)


def build_prompt(approx_tokens: int) -> str:
    """Build a long, open-ended prompt that does not immediately EOS.

    Temperature-0 completions of nonsense loops often emit a single token and
    stop, which makes generation tok/s look like 1e6. Use a narrative prompt
    the model continues for n_predict tokens.
    """
    unit = (
        "In a detailed technical report on GPU inference, describe the memory "
        "hierarchy, kernel fusion, quantization trade-offs, and how batch size "
        "interacts with prompt evaluation throughput on discrete Arc GPUs. "
        "Expand with numbered sections, examples, and measured intuition. "
    )
    # ~4 chars/token English; pad well past target so tokeniser lands ~approx_tokens.
    need = max(approx_tokens * 5, 512)
    body = (unit * (need // len(unit) + 1))[:need]
    return (
        "You are a performance engineer. Continue the following report with "
        "many full sentences; do not stop early.\n\nReport draft:\n" + body
        + "\n\nContinuation:\n"
    )


def _oneapi_ld_library_path() -> str:
    """Capture LD_LIBRARY_PATH after oneAPI setvars (required for libsvml/libsycl)."""
    setvars = Path("/opt/intel/oneapi/setvars.sh")
    if setvars.is_file():
        try:
            out = subprocess.run(
                [
                    "bash",
                    "-lc",
                    f'source "{setvars}" --force >/dev/null 2>&1 && printf %s "$LD_LIBRARY_PATH"',
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    # Fallback for this host's oneAPI 2026.0 layout
    return ":".join(
        p
        for p in (
            "/opt/intel/oneapi/compiler/2026.0/lib",
            "/opt/intel/oneapi/compiler/2026.0/opt/compiler/lib",
            "/opt/intel/oneapi/mkl/2026.0/lib",
            "/opt/intel/oneapi/tbb/2023.0/lib/intel64/gcc4.8",
            "/opt/intel/oneapi/tcm/1.5/lib",
            "/opt/intel/oneapi/umf/1.1/lib",
        )
        if Path(p).is_dir()
    )


def make_env(llama_bin: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for k in STRIP_ENV:
        env.pop(k, None)
    env.update(BASE_ENV)
    # Prefer the binary's own lib dir for libggml-sycl.so, then oneAPI.
    libdir = str(Path(llama_bin or DEFAULT_LLAMA).resolve().parent)
    parts = [libdir, _oneapi_ld_library_path()]
    if env.get("LD_LIBRARY_PATH"):
        parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(p for p in parts if p)
    return env


def http_json(method: str, url: str, body: dict | None = None, timeout: float = 600.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_health(base: str, timeout: float = HEALTH_TIMEOUT_S) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = http_json("GET", f"{base}/health", timeout=5.0)
            if r.get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1.5)
    return False


def completion(
    base: str,
    prompt: str,
    n_predict: int,
) -> dict:
    body = {
        "prompt": prompt,
        "n_predict": n_predict,
        # Non-zero temp + ignore_eos so we actually generate n_predict tokens
        # instead of immediate stop (which corrupts predicted_per_second).
        "temperature": 0.7,
        "top_p": 0.95,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": False,
    }
    return http_json("POST", f"{base}/completion", body, timeout=900.0)


def extract_timings(resp: dict) -> tuple[float | None, float | None, int | None, int | None]:
    t = resp.get("timings") or {}
    # Field names used by llama-server across versions
    pps = t.get("prompt_per_second")
    gpts = t.get("predicted_per_second")
    pn = t.get("prompt_n")
    gn = t.get("predicted_n")
    return (
        float(pps) if pps is not None else None,
        float(gpts) if gpts is not None else None,
        int(pn) if pn is not None else None,
        int(gn) if gn is not None else None,
    )


def start_server(llama: str, model: str, host: str, port: int, extra: list[str]) -> subprocess.Popen:
    argv = [
        llama,
        "-m", model,
        "--host", host,
        "--port", str(port),
        "-ngl", "999",
        *extra,
    ]
    log_path = Path(f"/tmp/arc-llama-emp-{port}.log")
    logf = open(log_path, "wb")
    print(f"  starting: {' '.join(argv)}", flush=True)
    print(f"  log: {log_path}", flush=True)
    proc = subprocess.Popen(
        argv,
        env=make_env(llama),
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._logf = logf  # type: ignore[attr-defined]
    proc._log_path = log_path  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=5)
    logf = getattr(proc, "_logf", None)
    if logf is not None:
        try:
            logf.close()
        except Exception:
            pass


def tail_log(proc: subprocess.Popen | None, n: int = 40) -> str:
    if proc is None:
        return ""
    path = getattr(proc, "_log_path", None)
    if not path:
        return ""
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return ""


def measure_config(
    name: str,
    extra: list[str],
    *,
    llama: str,
    model: str,
    host: str,
    port: int,
    prompt_tokens: int,
    n_predict: int,
    repeats: int,
) -> TimingRow:
    base = f"http://{host}:{port}"
    row = TimingRow(name=name, argv=extra)
    proc = None
    try:
        proc = start_server(llama, model, host, port, extra)
        if not wait_health(base):
            row.error = "health timeout\n" + tail_log(proc)
            return row

        prompt = build_prompt(prompt_tokens)
        # Warmup (also JIT) — still cache_prompt=false
        try:
            completion(base, build_prompt(64), WARMUP_PREDICT)
        except Exception as e:
            row.error = f"warmup failed: {e}\n" + tail_log(proc)
            return row

        best_p = -1.0
        best_g = -1.0
        best_raw: dict = {}
        last_pn = last_gn = None
        for i in range(repeats):
            try:
                resp = completion(base, prompt, n_predict)
            except Exception as e:
                row.error = f"completion failed run {i}: {e}\n" + tail_log(proc)
                return row
            pps, gpts, pn, gn = extract_timings(resp)
            last_pn, last_gn = pn, gn
            if pps is None or gpts is None:
                row.error = f"missing timings: {resp.get('timings')}"
                return row
            if gn is not None and gn < max(8, n_predict // 4):
                # Treat truncated gens as invalid samples (EOS / think-stop).
                print(
                    f"    run {i+1}/{repeats}: SKIP truncated gen n_pred={gn} "
                    f"(wanted {n_predict})",
                    flush=True,
                )
                continue
            # Track best gen; keep paired prompt from same run when gen improves.
            if gpts > best_g:
                best_g = gpts
                best_p = pps
                best_raw = resp.get("timings") or {}
            print(
                f"    run {i+1}/{repeats}: prompt={pps:.2f} tok/s  gen={gpts:.2f} tok/s  "
                f"(n_prompt={pn} n_pred={gn})",
                flush=True,
            )

        if best_g < 0:
            row.error = f"no valid runs (last n_pred={last_gn})"
            return row
        row.prompt_tok_s = best_p
        row.gen_tok_s = best_g
        row.prompt_n = last_pn
        row.gen_n = last_gn
        row.raw_best = best_raw
        # Try to parse n_ctx from props if available
        try:
            props = http_json("GET", f"{base}/props", timeout=10.0)
            row.ctx_reported = (
                props.get("default_generation_settings", {}).get("n_ctx")
                or props.get("n_ctx")
            )
        except Exception:
            pass
        return row
    finally:
        stop_server(proc)
        time.sleep(2)  # let GPU memory settle


def suite_configs(suite: str) -> list[ConfigRun]:
    """Return ordered configs. 'quick' is a minimal high-value subset.

    """
    # Production-like baseline: matches real SYCL deploy (q8 KV, no FA flag).
    # ctx 32768 is a working context for 27B Q4 + 24GB; 131k is a separate probe.
    prod = [
        "-c", "32768",
        "--fit", "off",
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "-b", "2048",
        "-ub", "512",
        # No -fa: SYCL production serves q8 V without flash-attn.
        "--parallel", "1",
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", "3",
    ]

    runs: list[ConfigRun] = [
        ConfigRun(
            "A_prod_manual",
            list(prod),
            "Production-like: fixed ctx/batch, fit off, no FA flag, q8 KV, draft-mtp n_max=3",
        ),
        ConfigRun(
            "B_fit_on_minimal",
            [
                "--fit", "on",
                "-ctk", "q8_0",
                "-ctv", "q8_0",
                "--parallel", "1",
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", "3",
            ],
            "Let -fit size ctx; no manual -c/-b/-ub; no FA flag",
        ),
        ConfigRun(
            "C_fit_on_keep_batch",
            [
                "--fit", "on",
                "-ctk", "q8_0",
                "-ctv", "q8_0",
                "-b", "2048",
                "-ub", "512",
                "--parallel", "1",
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", "3",
            ],
            "fit on but keep explicit -b/-ub",
        ),
        ConfigRun(
            "D_fa_on_sycl",
            [
                "-c", "32768",
                "--fit", "off",
                "-ctk", "q8_0",
                "-ctv", "q8_0",
                "-b", "2048",
                "-ub", "512",
                "-fa", "on",
                "--parallel", "1",
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", "3",
            ],
            "Same as prod but explicit FA on (optional SYCL probe)",
        ),
        ConfigRun(
            "E_fa_off_q8v",
            [
                "-c", "32768",
                "--fit", "off",
                "-ctk", "q8_0",
                "-ctv", "q8_0",
                "-b", "2048",
                "-ub", "512",
                "-fa", "off",
                "--parallel", "1",
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", "3",
            ],
            "Explicit FA off + q8 V (production path variant; SYCL allows this)",
        ),
    ]

    if suite == "quick":
        return runs

    # Full suite: batch sweep + MTP draft length + large-ctx probe.
    # Default path omits -fa (matches production SYCL).
    for b in (512, 1024, 2048, 4096):
        runs.append(
            ConfigRun(
                f"F_batch_b{b}",
                [
                    "-c", "32768",
                    "--fit", "off",
                    "-ctk", "q8_0",
                    "-ctv", "q8_0",
                    "-b", str(b),
                    "-ub", "512",
                    "--parallel", "1",
                    "--spec-type", "draft-mtp",
                    "--spec-draft-n-max", "3",
                ],
                f"batch-size {b}, ubatch fixed 512, no FA flag",
            )
        )
    for nmax in (1, 2, 3, 4, 5, 6):
        runs.append(
            ConfigRun(
                f"G_mtp_nmax{nmax}",
                [
                    "-c", "32768",
                    "--fit", "off",
                    "-ctk", "q8_0",
                    "-ctv", "q8_0",
                    "-b", "2048",
                    "-ub", "512",
                    "--parallel", "1",
                    "--spec-type", "draft-mtp",
                    "--spec-draft-n-max", str(nmax),
                ],
                f"draft-mtp with --spec-draft-n-max {nmax}",
            )
        )
    runs.append(
        ConfigRun(
            "H_no_mtp",
            [
                "-c", "32768",
                "--fit", "off",
                "-ctk", "q8_0",
                "-ctv", "q8_0",
                "-b", "2048",
                "-ub", "512",
                "--parallel", "1",
                "--spec-type", "none",
            ],
            "Control: no speculative decoding",
        )
    )
    runs.append(
        ConfigRun(
            "I_ctx131k_prod",
            [
                "-c", "131072",
                "--fit", "off",
                "-ctk", "q8_0",
                "-ctv", "q8_0",
                "-b", "2048",
                "-ub", "512",
                "--parallel", "1",
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", "3",
            ],
            "131072 ctx + q8 KV, no FA flag (production-like large ctx)",
        )
    )
    return runs


def print_table(rows: list[TimingRow]) -> None:
    print()
    print(
        f"{'config':<22} {'prompt tok/s':>12} {'gen tok/s':>10} {'ctx':>7}  notes/error"
    )
    print("-" * 100)
    for r in rows:
        if r.error:
            err = r.error.splitlines()[0][:60]
            print(f"{r.name:<22} {'ERR':>12} {'ERR':>10} {'?':>7}  {err}")
        else:
            print(
                f"{r.name:<22} {r.prompt_tok_s or 0:12.2f} {r.gen_tok_s or 0:10.2f} "
                f"{str(r.ctx_reported or '?'):>7}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llama", default=DEFAULT_LLAMA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--suite", choices=("quick", "full"), default="quick")
    ap.add_argument("--prompt-tokens", type=int, default=PROMPT_TARGET_TOKENS)
    ap.add_argument("--n-predict", type=int, default=N_PREDICT)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parents[1] / "bench_results" / "empirical.json"
        ),
    )
    ap.add_argument(
        "--only",
        default="",
        help="Comma-separated config name prefixes to run (e.g. A_,D_,G_)",
    )
    args = ap.parse_args()

    if not Path(args.llama).is_file():
        print(f"ERROR: llama-server not found: {args.llama}", file=sys.stderr)
        return 2
    if not Path(args.model).is_file():
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        return 2

    configs = suite_configs(args.suite)
    if args.only:
        prefixes = [p.strip() for p in args.only.split(",") if p.strip()]
        configs = [c for c in configs if any(c.name.startswith(p) for p in prefixes)]

    print(f"model:  {args.model}")
    print(f"llama:  {args.llama}")
    print(f"suite:  {args.suite} ({len(configs)} configs)")
    print(f"prompt≈{args.prompt_tokens} tok, n_predict={args.n_predict}, repeats={args.repeats}")
    print(f"endpoint: POST http://{args.host}:{args.port}/completion  cache_prompt=false")
    print()

    rows: list[TimingRow] = []
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg.name}: {cfg.notes}", flush=True)
        row = measure_config(
            cfg.name,
            cfg.argv_extra,
            llama=args.llama,
            model=args.model,
            host=args.host,
            port=args.port,
            prompt_tokens=args.prompt_tokens,
            n_predict=args.n_predict,
            repeats=args.repeats,
        )
        rows.append(row)
        if row.error:
            print(f"  ERROR: {row.error.splitlines()[0][:120]}", flush=True)
        else:
            print(
                f"  BEST prompt={row.prompt_tok_s:.2f} tok/s  gen={row.gen_tok_s:.2f} tok/s  "
                f"ctx={row.ctx_reported}",
                flush=True,
            )

    print_table(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "llama": args.llama,
        "suite": args.suite,
        "prompt_tokens_target": args.prompt_tokens,
        "n_predict": args.n_predict,
        "methodology": {
            "endpoint": "/completion",
            "cache_prompt": False,
            "metrics": ["timings.prompt_per_second", "timings.predicted_per_second"],
        },
        "rows": [asdict(r) for r in rows],
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
