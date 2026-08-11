"""Upstream proxy responses must close BOTH the response and its client.

``UpstreamManager.proxy`` builds a throwaway ``httpx.AsyncClient`` per call
and hands the streaming response to ``_proxy_post``. The response alone was
returned once, and only a ``BackgroundTask`` closed it — so:

  * the client (and its connection pool) was leaked on EVERY proxied call,
    streaming or not, because nothing after the return held a reference;
  * when the upstream died mid-stream, Starlette skipped the BackgroundTask
    and the response itself leaked too.

The local-model path had the same bug class fixed in
test_streaming_inflight.py; these tests pin the upstream twin of it.

None of this needs a GPU or a real llama.cpp backend.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from test_server import FakeRouter


class _Stream:
    """Upstream SSE response. Optionally dies partway through."""

    status_code = 200
    headers = {"content-type": "text/event-stream"}

    def __init__(self, *, fail_after: int | None = None, chunks: int = 3):
        self.closed = False
        self._fail_after = fail_after
        self._chunks = chunks

    async def aiter_raw(self):
        for i in range(self._chunks):
            if self._fail_after is not None and i >= self._fail_after:
                raise httpx.ReadError("upstream died mid-stream")
            yield f"data: chunk{i}\n\n".encode()

    async def aread(self):
        return b'{"upstream": true}'

    async def aclose(self):
        self.closed = True


class _Client:
    """Stand-in for the per-call httpx.AsyncClient; records closure."""

    instances: list[_Client] = []

    def __init__(self, timeout=None):
        self.timeout = timeout
        self.closed = False
        _Client.instances.append(self)

    async def aclose(self):
        self.closed = True


class _UpstreamModel:
    id = "llama3.1"
    upstream_name = "ollama"
    upstream_url = "http://127.0.0.1:11434"
    metadata = {}


class _Manager:
    """UpstreamManager fake whose proxied stream is controllable per test."""

    stream_factory = staticmethod(lambda: _Stream())

    def __init__(self, upstreams=None):
        self.last_client = None
        self.last_stream = None

    async def models(self):
        return [_UpstreamModel()]

    def find_model(self, model_id):
        return _UpstreamModel() if model_id == "llama3.1" else None

    async def proxy(self, upstream, path, body, headers, streaming_ok=True):
        self.last_client = _Client()
        self.last_stream = _Manager.stream_factory()
        return self.last_client, self.last_stream

    def upstreams_status(self):
        return []


@pytest.fixture
def app(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", _Manager)
    _Client.instances = []
    _Manager.stream_factory = staticmethod(lambda: _Stream())
    return server_mod.create_app()


def _post(client, **kw):
    return client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "llama3.1", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        **kw,
    )


def _mgr(app):
    return app.state.upstream_mgr


def test_upstream_stream_closes_response_and_client_on_success(app):
    with TestClient(app) as c:
        with _post(c) as r:
            assert b"chunk0" in b"".join(r.iter_bytes())
    assert _mgr(app).last_stream.closed, "upstream response leaked after a clean stream"
    assert _mgr(app).last_client.closed, "httpx client leaked after a clean stream"


def test_upstream_stream_closes_both_when_upstream_dies(app):
    """The real leak: Starlette skips the BackgroundTask when the body raises,
    so closure has to happen in the generator's finally."""
    _Manager.stream_factory = staticmethod(lambda: _Stream(fail_after=1))
    with TestClient(app) as c:
        with pytest.raises(httpx.ReadError):
            with _post(c) as r:
                b"".join(r.iter_bytes())
    assert _mgr(app).last_stream.closed, "upstream response leaked on mid-stream failure"
    assert _mgr(app).last_client.closed, "httpx client leaked on mid-stream failure"


def test_upstream_stream_closes_both_on_client_disconnect(app):
    _Manager.stream_factory = staticmethod(lambda: _Stream(chunks=50))
    with TestClient(app) as c:
        with _post(c) as r:
            next(r.iter_bytes())  # read one chunk, then abandon the response
    assert _mgr(app).last_stream.closed, "upstream response leaked on disconnect"
    assert _mgr(app).last_client.closed, "httpx client leaked on disconnect"


def test_upstream_non_streaming_closes_both(app):
    with TestClient(app) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "llama3.1", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
    assert _mgr(app).last_stream.closed, "upstream response leaked on a non-stream call"
    assert _mgr(app).last_client.closed, "httpx client leaked on a non-stream call"


def test_upstream_non_streaming_closes_both_when_read_fails(app):
    class _BadRead(_Stream):
        async def aread(self):
            raise httpx.ReadError("body read failed")

    _Manager.stream_factory = staticmethod(lambda: _BadRead())
    with TestClient(app) as c:
        with pytest.raises(httpx.ReadError):
            c.post(
                "/v1/chat/completions",
                json={"model": "llama3.1", "messages": [{"role": "user", "content": "hi"}]},
            )
    assert _mgr(app).last_client.closed, "httpx client leaked when aread() failed"
