"""The ORBTR SDK for Python: messages, streaming, models, flows as models, token counting, embeddings,
OpenAI-compatible chat and completions, and message batches on api.orbtr.ai."""

from . import types
from ._client import AsyncOrbtr, Orbtr
from ._core import DEFAULT_BASE_URL, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, MAX_RETRY_AFTER, RequestOptions
from ._errors import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    OrbtrError,
    PaymentRequiredError,
    PermissionDeniedError,
    RateLimitError,
    StreamError,
    UnprocessableEntityError,
)
from ._flows import flow_model, is_flow_model, parse_flow_model
from ._streaming import AsyncMessageStream, AsyncStream, ExtensionStreamEvent, MessageStream, SSEDecoder, ServerSentEvent, Stream
from ._version import __version__

__all__ = [
    "Orbtr",
    "AsyncOrbtr",
    "types",
    "RequestOptions",
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT",
    "MAX_RETRY_AFTER",
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
    "flow_model",
    "is_flow_model",
    "parse_flow_model",
    "Stream",
    "AsyncStream",
    "MessageStream",
    "AsyncMessageStream",
    "ExtensionStreamEvent",
    "SSEDecoder",
    "ServerSentEvent",
    "__version__",
]
