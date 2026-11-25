"""The transport every generated resource calls.

It holds the base URL and the API key, sent as a bearer token, and retries a 429, a 5xx or a connection
failure with backoff, honouring Retry-After. Consumed by _client.py (Orbtr and AsyncOrbtr extend the clients
here) and by the generated resource classes.
"""

from __future__ import annotations

import asyncio
import email.utils
import json
import os
import random
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, TypeVar

import httpx

from ._errors import APIConnectionError, APITimeoutError, OrbtrError, make_api_error
from ._version import __version__

if TYPE_CHECKING:
    from ._streaming import AsyncMessageStream, AsyncStream, MessageStream, Stream

T = TypeVar("T")

DEFAULT_BASE_URL = "https://api.orbtr.ai"
"""The default API origin."""
DEFAULT_MAX_RETRIES = 2
"""How many times a failed call is retried unless a client or call says otherwise."""
DEFAULT_TIMEOUT = 600.0
"""Seconds a call may wait on the network unless a client or call says otherwise."""
MAX_RETRY_AFTER = 60.0
"""The longest Retry-After, in seconds, a retry waits for; a longer one is raised to the caller instead."""
_INITIAL_BACKOFF = 0.5
_MAX_BACKOFF = 8.0


@dataclass
class RequestOptions:
    """Per-call settings."""

    extra_headers: Mapping[str, str] | None = None
    timeout: float | None = None
    max_retries: int | None = None
    idempotency_key: str | None = None


def compact(values: dict[str, Any]) -> dict[str, Any]:
    """The values that were given: a parameter left as None is not sent."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in values.items() if v is not None}


def is_retryable_status(status: int) -> bool:
    """Whether a response status is retried: 429 and every 5xx."""
    return status == 429 or status >= 500


def backoff(attempt: int, rand: Callable[[], float] = random.random) -> float:
    """Seconds before retry number attempt (0 for the first retry): exponential from 0.5, capped at 8, with up
    to a quarter taken off at random so clients refused together do not retry together."""
    return min(_INITIAL_BACKOFF * 2**attempt, _MAX_BACKOFF) * (1 - 0.25 * rand())


def parse_retry_after(value: str | None, now: float | None = None) -> float | None:
    """Reads Retry-After as seconds or an HTTP date. Returns None when absent or unreadable."""
    if value is None or not value.strip():
        return None
    v = value.strip()
    try:
        return max(0.0, float(v))
    except ValueError:
        pass
    try:
        at = email.utils.parsedate_to_datetime(v).timestamp()
    except (TypeError, ValueError, IndexError):
        return None
    return max(0.0, at - (time.time() if now is None else now))


def _retry_wait(response: httpx.Response, attempt: int) -> float | None:
    """Seconds to wait before retrying a refused response, or None when it is not retried."""
    if not is_retryable_status(response.status_code):
        return None
    after = parse_retry_after(response.headers.get("retry-after"))
    if after is None:
        return backoff(attempt)
    if after > MAX_RETRY_AFTER:
        return None
    return after


def _error_from(response: httpx.Response) -> OrbtrError:
    text = response.text
    try:
        body: Any = json.loads(text) if text else None
    except ValueError:
        body = None
    return make_api_error(response.status_code, body, text, response.headers)


class _BaseClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        max_retries: int,
        timeout: float,
        default_headers: Mapping[str, str] | None,
    ) -> None:
        key = api_key or os.environ.get("ORBTR_API_KEY")
        if not key:
            raise OrbtrError("No API key: pass api_key or set the ORBTR_API_KEY environment variable.")
        self._api_key = key
        self.base_url = (base_url or os.environ.get("ORBTR_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        self._default_headers = dict(default_headers or {})

    def _build(
        self,
        method: str,
        path: str,
        *,
        body: Any,
        query: Mapping[str, Any] | None,
        options: RequestOptions,
        accept: str,
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        headers = {
            "Accept": accept,
            "User-Agent": f"orbtr-sdk-python/{__version__}",
            **self._default_headers,
            **dict(options.extra_headers or {}),
        }
        for k in [k for k in headers if k.lower() == "authorization"]:
            del headers[k]
        headers["Authorization"] = f"Bearer {self._api_key}"
        if body is not None:
            headers["Content-Type"] = "application/json"
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        params = {k: (str(v).lower() if isinstance(v, bool) else v) for k, v in (query or {}).items()}
        return {
            "method": method,
            "url": self.base_url + path,
            "headers": headers,
            "params": params or None,
            "content": None if body is None else json.dumps(body).encode(),
            "timeout": options.timeout if options.timeout is not None else self.timeout,
        }


class SyncAPIClient(_BaseClient):
    """The synchronous client every resource calls through."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = DEFAULT_TIMEOUT,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=timeout, default_headers=default_headers)
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client()

    def close(self) -> None:
        """Closes the client's connections, when the client made its own HTTP client."""
        if self._owns_http:
            self._http.close()

    def __enter__(self: T) -> T:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _send(
        self,
        method: str,
        path: str,
        *,
        body: Any,
        query: Mapping[str, Any] | None,
        options: RequestOptions,
        accept: str,
        idempotent: bool,
        stream: bool,
    ) -> httpx.Response:
        max_retries = options.max_retries if options.max_retries is not None else self.max_retries
        key = (options.idempotency_key or f"orbtr-sdk-{uuid.uuid4()}") if idempotent else None
        spec = self._build(method, path, body=body, query=query, options=options, accept=accept, idempotency_key=key)
        attempt = 0
        while True:
            request = self._http.build_request(
                spec["method"], spec["url"], headers=spec["headers"], params=spec["params"], content=spec["content"], timeout=spec["timeout"]
            )
            try:
                response = self._http.send(request, stream=stream)
            except httpx.TimeoutException as err:
                if attempt < max_retries:
                    time.sleep(backoff(attempt))
                    attempt += 1
                    continue
                raise APITimeoutError() from err
            except httpx.TransportError as err:
                if attempt < max_retries:
                    time.sleep(backoff(attempt))
                    attempt += 1
                    continue
                raise APIConnectionError() from err
            if response.is_success:
                return response
            wait = _retry_wait(response, attempt) if attempt < max_retries else None
            if wait is not None:
                response.close()
                time.sleep(wait)
                attempt += 1
                continue
            if stream:
                response.read()
            response.close()
            raise _error_from(response)

    def _request_json(self, method: str, path: str, *, parse: Callable[[Any], T], options: RequestOptions, body: Any = None, query: Mapping[str, Any] | None = None, idempotent: bool = False) -> T:
        response = self._send(method, path, body=body, query=query, options=options, accept="application/json", idempotent=idempotent, stream=False)
        return parse(response.json())

    def _request_ndjson(self, method: str, path: str, *, parse: Callable[[Any], T], options: RequestOptions, body: Any = None, query: Mapping[str, Any] | None = None, idempotent: bool = False) -> list[T]:
        response = self._send(method, path, body=body, query=query, options=options, accept="application/x-ndjson", idempotent=idempotent, stream=False)
        return [parse(json.loads(line)) for line in response.text.splitlines() if line.strip()]

    def _message_stream(self, method: str, path: str, *, options: RequestOptions, body: Any = None, query: Mapping[str, Any] | None = None, idempotent: bool = False) -> MessageStream:
        from ._streaming import MessageStream

        response = self._send(method, path, body=body, query=query, options=options, accept="text/event-stream", idempotent=idempotent, stream=True)
        return MessageStream(response)

    def _chunk_stream(self, method: str, path: str, *, parse: Callable[[Any], T], options: RequestOptions, body: Any = None, query: Mapping[str, Any] | None = None, idempotent: bool = False) -> Stream[T]:
        from ._streaming import Stream, chunk_parser

        response = self._send(method, path, body=body, query=query, options=options, accept="text/event-stream", idempotent=idempotent, stream=True)
        return Stream(response, chunk_parser(parse))


class AsyncAPIClient(_BaseClient):
    """The asynchronous client every resource calls through."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = DEFAULT_TIMEOUT,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=timeout, default_headers=default_headers)
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient()

    async def close(self) -> None:
        """Closes the client's connections, when the client made its own HTTP client."""
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self: T) -> T:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _send(
        self,
        method: str,
        path: str,
        *,
        body: Any,
        query: Mapping[str, Any] | None,
        options: RequestOptions,
        accept: str,
        idempotent: bool,
        stream: bool,
    ) -> httpx.Response:
        max_retries = options.max_retries if options.max_retries is not None else self.max_retries
        key = (options.idempotency_key or f"orbtr-sdk-{uuid.uuid4()}") if idempotent else None
        spec = self._build(method, path, body=body, query=query, options=options, accept=accept, idempotency_key=key)
        attempt = 0
        while True:
            request = self._http.build_request(
                spec["method"], spec["url"], headers=spec["headers"], params=spec["params"], content=spec["content"], timeout=spec["timeout"]
            )
            try:
                response = await self._http.send(request, stream=stream)
            except httpx.TimeoutException as err:
                if attempt < max_retries:
                    await asyncio.sleep(backoff(attempt))
                    attempt += 1
                    continue
                raise APITimeoutError() from err
            except httpx.TransportError as err:
                if attempt < max_retries:
                    await asyncio.sleep(backoff(attempt))
                    attempt += 1
                    continue
                raise APIConnectionError() from err
            if response.is_success:
                return response
            wait = _retry_wait(response, attempt) if attempt < max_retries else None
            if wait is not None:
                await response.aclose()
                await asyncio.sleep(wait)
                attempt += 1
                continue
            if stream:
                await response.aread()
            await response.aclose()
            raise _error_from(response)

    async def _request_json(self, method: str, path: str, *, parse: Callable[[Any], T], options: RequestOptions, body: Any = None, query: Mapping[str, Any] | None = None, idempotent: bool = False) -> T:
        response = await self._send(method, path, body=body, query=query, options=options, accept="application/json", idempotent=idempotent, stream=False)
        return parse(response.json())

    async def _request_ndjson(self, method: str, path: str, *, parse: Callable[[Any], T], options: RequestOptions, body: Any = None, query: Mapping[str, Any] | None = None, idempotent: bool = False) -> list[T]:
        response = await self._send(method, path, body=body, query=query, options=options, accept="application/x-ndjson", idempotent=idempotent, stream=False)
        return [parse(json.loads(line)) for line in response.text.splitlines() if line.strip()]

    async def _message_stream(self, method: str, path: str, *, options: RequestOptions, body: Any = None, query: Mapping[str, Any] | None = None, idempotent: bool = False) -> AsyncMessageStream:
        from ._streaming import AsyncMessageStream

        response = await self._send(method, path, body=body, query=query, options=options, accept="text/event-stream", idempotent=idempotent, stream=True)
        return AsyncMessageStream(response)

    async def _chunk_stream(self, method: str, path: str, *, parse: Callable[[Any], T], options: RequestOptions, body: Any = None, query: Mapping[str, Any] | None = None, idempotent: bool = False) -> AsyncStream[T]:
        from ._streaming import AsyncStream, chunk_parser

        response = await self._send(method, path, body=body, query=query, options=options, accept="text/event-stream", idempotent=idempotent, stream=True)
        return AsyncStream(response, chunk_parser(parse))


class SyncAPIResource:
    """The base of every generated synchronous resource class."""

    def __init__(self, client: SyncAPIClient) -> None:
        self._client = client


class AsyncAPIResource:
    """The base of every generated asynchronous resource class."""

    def __init__(self, client: AsyncAPIClient) -> None:
        self._client = client
