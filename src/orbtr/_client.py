"""The clients applications construct: the transports of _core.py with the generated resources attached."""

from __future__ import annotations

from typing import Mapping

import httpx

from ._core import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, AsyncAPIClient, SyncAPIClient
from ._generated.resources import AsyncResources, Resources

__all__ = ["Orbtr", "AsyncOrbtr"]


class Orbtr(SyncAPIClient, Resources):
    """A client for api.orbtr.ai.

    Args:
        api_key: An ORBTR API key. Defaults to the ORBTR_API_KEY environment variable.
        base_url: The API origin. Defaults to the ORBTR_BASE_URL environment variable, then https://api.orbtr.ai.
        max_retries: Retries after a 429, a 5xx or a connection failure. Defaults to 2.
        timeout: Seconds a call may wait on the network. Defaults to 600.
        default_headers: Headers sent with every call.
        http_client: The httpx.Client to send through; the client makes its own when absent.
    """

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
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
            default_headers=default_headers,
            http_client=http_client,
        )
        self._init_resources()


class AsyncOrbtr(AsyncAPIClient, AsyncResources):
    """An asynchronous client for api.orbtr.ai; its arguments are Orbtr's, with an httpx.AsyncClient."""

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
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
            default_headers=default_headers,
            http_client=http_client,
        )
        self._init_resources()
