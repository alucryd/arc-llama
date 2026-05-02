"""OpenAI-compatible HTTP server.

Mounts on `cfg.server.host:cfg.server.port` and forwards requests to whichever
llama-server backend the router decides is the right one for the model id in
the request body.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from arc_llama.config import Config, load_config
from arc_llama.router import Router

log = logging.getLogger("arc_llama.server")


def _strip_response_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        k: v for k, v in headers.items()
        if k.lower() not in (
            "transfer-encoding", "content-encoding", "content-length", "connection",
        )
    }


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    state_dir = None
    if cfg.paths.state_dir:
        from pathlib import Path
        state_dir = Path(cfg.paths.state_dir).expanduser()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router = Router(cfg, log_dir=state_dir)
        app.state.cfg = cfg
        try:
            yield
        finally:
            await app.state.router.shutdown()

    app = FastAPI(title="arc-llama", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict:
        rt: Router = request.app.state.router
        data = []
        for m in rt.all_models():
            srv = rt._servers.get(m.name)
            data.append({
                "id": m.name,
                "object": "model",
                "owned_by": "arc-llama",
                "created": 0,
                "metadata": {
                    "display_name": m.display_name,
                    "path": m.path,
                    "gpu_pci_slot": m.gpu_pci_slot,
                    "loaded": bool(srv and srv.is_running),
                    "aliases": list(m.aliases),
                },
            })
            for alias in m.aliases:
                if alias != m.name:
                    data.append({
                        "id": alias,
                        "object": "model",
                        "owned_by": "arc-llama-alias",
                        "created": 0,
                        "metadata": {"canonical": m.name},
                    })
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    @app.post("/v1/completions")
    async def chat_or_completions(request: Request):
        return await _proxy_post(request, request.url.path)

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        return await _proxy_post(request, "/v1/embeddings", streaming_ok=False)

    return app


async def _proxy_post(request: Request, target_path: str, streaming_ok: bool = True):
    rt: Router = request.app.state.router
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
    model_query = body.get("model", "")
    try:
        model, srv = await rt.ensure_active(model_query)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_query!r}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    target_url = f"{srv.plan.backend_url}{target_path}"
    want_stream = streaming_ok and bool(body.get("stream"))
    fwd_headers = {"Content-Type": "application/json"}
    if want_stream:
        async def gen():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST", target_url, content=body_bytes, headers=fwd_headers,
                ) as r:
                    async for chunk in r.aiter_raw():
                        yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")
    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(target_url, content=body_bytes, headers=fwd_headers)
        return Response(
            content=r.content,
            status_code=r.status_code,
            headers=_strip_response_headers(dict(r.headers)),
            media_type=r.headers.get("content-type", "application/json"),
        )
