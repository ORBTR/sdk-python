"""The errors every call can raise.

_core.py builds an APIError subclass from a refused response's status and body; _streaming.py raises
StreamError when a stream reports a failure after it began.
"""

from __future__ import annotations

from typing import Any

import httpx

__all__ = [
    "OrbtrError",
    "APIError",
    "BadRequestError",
    "AuthenticationError",
    "PaymentRequiredError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "UnprocessableEntityError",
    "RateLimitError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
    "StreamError",
]


class OrbtrError(Exception):
    """The base of every error this SDK raises."""


class APIError(OrbtrError):
    """A response api.orbtr.ai refused, with its status, its error code and its parsed body."""

    status: int
    """The HTTP status."""
    code: str | None
    """The error code the body names, such as invalid_api_key or insufficient_ot."""
    body: Any
    """The parsed JSON body, or the raw text when it is not JSON."""
    headers: httpx.Headers
    """The response headers."""
    reason: str | None
    """The reason a key_expired_or_revoked refusal names (revoked, rotated or expired)."""

    def __init__(self, status: int, code: str | None, message: str, body: Any, headers: httpx.Headers, reason: str | None = None) -> None:
        text = f"{status} {code}: {message}" if code and not message.startswith(code) else f"{status} {message}"
        super().__init__(text)
        self.status = status
        self.code = code
        self.message = message
        self.body = body
        self.headers = headers
        self.reason = reason


class BadRequestError(APIError):
    """400: the request is malformed."""


class AuthenticationError(APIError):
    """401: the key is missing, unknown, revoked, rotated or expired."""


class PaymentRequiredError(APIError):
    """402: the OT balance or a spending bound is short."""


class PermissionDeniedError(APIError):
    """403: the key lacks the scope, the plan lacks the feature, or the request is above a limit."""


class NotFoundError(APIError):
    """404: the model, flow or batch is unknown or not readable by the key."""


class ConflictError(APIError):
    """409: the request conflicts with the conversation or an idempotency key."""


class UnprocessableEntityError(APIError):
    """422: the request is well-formed but cannot be processed."""


class RateLimitError(APIError):
    """429: a rate, quota or budget limit refused the call."""


class InternalServerError(APIError):
    """5xx: the service failed or is overloaded."""


class APIConnectionError(OrbtrError):
    """The request never received a response."""

    def __init__(self, message: str = "Connection error") -> None:
        super().__init__(message)


class APITimeoutError(APIConnectionError):
    """The request took longer than its timeout."""

    def __init__(self, message: str = "Request timed out") -> None:
        super().__init__(message)


class StreamError(OrbtrError):
    """A stream reported a failure after its response began."""

    def __init__(self, code: str | None, message: str, body: Any) -> None:
        super().__init__(f"{code}: {message}" if code else message)
        self.code = code
        self.message = message
        self.body = body


def describe_error_body(body: Any) -> tuple[str | None, str | None, str | None]:
    """Reads (code, message, reason) from the three error bodies api.orbtr.ai answers: the Anthropic shape
    {"type":"error","error":{"type","message"}}, the OpenAI shape {"error":{"message","type","code"}} and the
    key gateway's {"error":"<code>","reason"?}."""
    if not isinstance(body, dict):
        return None, None, None
    reason = body.get("reason") if isinstance(body.get("reason"), str) else None
    err = body.get("error")
    if isinstance(err, str):
        msg = body.get("message")
        return err, msg if isinstance(msg, str) else None, reason
    if isinstance(err, dict):
        code = err.get("code") if isinstance(err.get("code"), str) and err.get("code") else err.get("type")
        msg = err.get("message")
        return (code if isinstance(code, str) else None), (msg if isinstance(msg, str) else None), reason
    msg = body.get("message")
    return None, msg if isinstance(msg, str) else None, reason


_BY_STATUS: dict[int, type[APIError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    402: PaymentRequiredError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    422: UnprocessableEntityError,
    429: RateLimitError,
}


def make_api_error(status: int, body: Any, text: str, headers: httpx.Headers) -> APIError:
    """Builds the APIError subclass for a refused response."""
    code, message, reason = describe_error_body(body)
    msg = message or code or text or f"HTTP {status}"
    cls = _BY_STATUS.get(status) or (InternalServerError if status >= 500 else APIError)
    return cls(status, code, msg, body if body is not None else text, headers, reason)
