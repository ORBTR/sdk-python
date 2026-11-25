"""Fixture loading and transports answering from fixtures while recording what the SDK sent. The fixtures in
fixtures/ are HTTP exchanges recorded from api.orbtr.ai's handlers; each names the handler it came from."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import httpx

ROOT = Path(__file__).resolve().parent.parent


def fixture(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "fixtures" / f"{name}.json").read_text())


def fixture_names() -> list[str]:
    return sorted(p.stem for p in (ROOT / "fixtures").glob("*.json"))


def _chunks(body: bytes, size: int) -> Iterator[bytes]:
    for i in range(0, len(body), size):
        yield body[i : i + size]


class FixtureTransport(httpx.BaseTransport):
    """Answers each request with the next of the given fixtures, in 7-byte chunks so stream parsing meets
    events cut across reads; every request is kept in sent."""

    def __init__(self, *names: str) -> None:
        self.queue = [fixture(n) for n in names]
        self.sent: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.sent.append(request)
        fx = self.queue.pop(0)
        r = fx["response"]
        return httpx.Response(r["status"], headers=r["headers"], content=_chunks(r["body"].encode(), 7))


class AsyncFixtureTransport(httpx.AsyncBaseTransport):
    """FixtureTransport for the async client."""

    def __init__(self, *names: str) -> None:
        self.queue = [fixture(n) for n in names]
        self.sent: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.sent.append(request)
        fx = self.queue.pop(0)
        r = fx["response"]
        return httpx.Response(r["status"], headers=r["headers"], content=r["body"].encode())


def assert_sent(request: httpx.Request, name: str) -> None:
    fx = fixture(name)["request"]
    assert request.method == fx["method"], name
    assert request.url.path == fx["path"], name
    assert request.url.query.decode() == fx.get("query", ""), name
    body = json.loads(request.content) if request.content else None
    assert body == fx.get("body"), name
    assert request.headers["authorization"] == "Bearer test-key"
    assert "x-api-key" not in request.headers
    assert request.headers["user-agent"].startswith("orbtr-sdk-python/")
    if fx.get("body") is not None:
        assert request.headers["content-type"] == "application/json"
