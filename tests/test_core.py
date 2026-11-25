"""The hand-written layer: configuration, retries and Retry-After, timeouts, error bodies, the event-stream
decoder and the flow helpers."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import pytest

from orbtr import (
    MAX_RETRY_AFTER,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    Orbtr,
    OrbtrError,
    PaymentRequiredError,
    RateLimitError,
    SSEDecoder,
    StreamError,
    flow_model,
    is_flow_model,
    parse_flow_model,
)
from orbtr._core import backoff, is_retryable_status, parse_retry_after
from orbtr._errors import describe_error_body
from orbtr._streaming import ExtensionStreamEvent

OK = (200, {}, '{"object":"list","data":[]}')


class Scripted(httpx.BaseTransport):
    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.sent: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.sent.append(request)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        status, headers, body = r
        return httpx.Response(status, headers=headers, content=body.encode())


def client(*replies: Any, **kw: Any) -> tuple[Orbtr, Scripted]:
    t = Scripted(*replies)
    kw.setdefault("api_key", "k")
    return Orbtr(http_client=httpx.Client(transport=t), **kw), t


def test_key_from_environment_as_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBTR_API_KEY", "env-key")
    monkeypatch.delenv("ORBTR_BASE_URL", raising=False)
    t = Scripted(OK)
    Orbtr(http_client=httpx.Client(transport=t)).models.list()
    assert t.sent[0].headers["authorization"] == "Bearer env-key"
    assert str(t.sent[0].url) == "https://api.orbtr.ai/v1/models"


def test_no_key_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORBTR_API_KEY", raising=False)
    with pytest.raises(OrbtrError):
        Orbtr()


def test_base_url_from_argument_then_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    c, t = client(OK, base_url="http://127.0.0.1:9/")
    c.models.list()
    assert str(t.sent[0].url) == "http://127.0.0.1:9/v1/models"
    monkeypatch.setenv("ORBTR_BASE_URL", "http://127.0.0.2:9")
    c, t = client(OK)
    c.models.list()
    assert str(t.sent[0].url) == "http://127.0.0.2:9/v1/models"


def test_a_callers_header_cannot_replace_authorization() -> None:
    c, t = client(OK, default_headers={"Authorization": "Bearer other"})
    c.models.list(extra_headers={"authorization": "Bearer x", "X-Trace": "1"})
    assert t.sent[0].headers.get_list("authorization") == ["Bearer k"]
    assert t.sent[0].headers["x-trace"] == "1"


def test_retry_after_is_honoured() -> None:
    c, t = client((429, {"Retry-After": "1"}, '{"error":"api_key_rate_limit_exceeded"}'), OK)
    started = time.monotonic()
    c.models.list()
    assert len(t.sent) == 2
    assert time.monotonic() - started >= 0.95


def test_a_long_retry_after_is_raised() -> None:
    body = '{"type":"error","error":{"type":"no_servable_placement","message":"nothing can serve the model now"}}'
    c, t = client((503, {"Retry-After": "3600"}, body))
    with pytest.raises(InternalServerError) as info:
        c.models.list()
    assert info.value.code == "no_servable_placement"
    assert info.value.headers["retry-after"] == "3600"
    assert len(t.sent) == 1


def test_5xx_retried_up_to_max_retries() -> None:
    fail = (529, {"Retry-After": "0"}, '{"type":"error","error":{"type":"fleet_overload","message":"busy"}}')
    c, t = client(fail, fail, fail, max_retries=2)
    with pytest.raises(InternalServerError):
        c.models.list()
    assert len(t.sent) == 3


def test_a_calls_max_retries_overrides_the_clients() -> None:
    c, t = client((500, {"Retry-After": "0"}, "{}"), OK, max_retries=5)
    with pytest.raises(InternalServerError):
        c.models.list(max_retries=0)
    assert len(t.sent) == 1


@pytest.mark.parametrize("status,cls", [(400, BadRequestError), (402, PaymentRequiredError)])
def test_4xx_not_retried(status: int, cls: type) -> None:
    c, t = client((status, {}, '{"type":"error","error":{"type":"x","message":"no"}}'))
    with pytest.raises(cls):
        c.models.list()
    assert len(t.sent) == 1


def test_connection_failure_retried_then_raised() -> None:
    req = httpx.Request("GET", "http://x")
    c, t = client(httpx.ConnectError("refused", request=req), httpx.ConnectError("refused", request=req), max_retries=1)
    with pytest.raises(APIConnectionError) as info:
        c.models.list()
    assert not isinstance(info.value, APITimeoutError)
    assert len(t.sent) == 2


def test_timeout_raises_api_timeout_error() -> None:
    req = httpx.Request("GET", "http://x")
    c, _ = client(httpx.ReadTimeout("slow", request=req), max_retries=0)
    with pytest.raises(APITimeoutError):
        c.models.list()


def test_every_retry_of_a_batch_create_sends_the_same_key() -> None:
    c, t = client((503, {"Retry-After": "0"}, "{}"), (200, {}, '{"id":"b","type":"message_batch","processing_status":"in_progress","request_counts":{"processing":0,"succeeded":0,"errored":0,"canceled":0,"expired":0},"created_at":"x","expires_at":"y"}'))
    c.messages.batches.create(requests=[])
    keys = [r.headers.get("idempotency-key") for r in t.sent]
    assert len(keys) == 2 and keys[0] and keys[0] == keys[1]


def test_no_other_call_sends_an_idempotency_key() -> None:
    c, t = client((200, {}, '{"input_tokens":1}'))
    c.messages.count_tokens(model="m", messages=[])
    assert "idempotency-key" not in t.sent[0].headers


def test_parameters_left_as_none_are_not_sent() -> None:
    c, t = client((200, {}, '{"input_tokens":1}'))
    c.messages.count_tokens(model="m", messages=({"role": "user", "content": "x"},), system=None)
    assert json.loads(t.sent[0].content) == {"model": "m", "messages": [{"role": "user", "content": "x"}]}


def test_parse_retry_after() -> None:
    assert parse_retry_after(None) is None
    assert parse_retry_after("2") == 2.0
    assert parse_retry_after("0.5") == 0.5
    now = 1_795_600_000.0
    assert parse_retry_after(time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(now + 30)), now) == pytest.approx(30, abs=1)
    assert parse_retry_after(time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(now - 30)), now) == 0
    assert parse_retry_after("soon") is None
    assert MAX_RETRY_AFTER == 60.0


def test_backoff() -> None:
    assert backoff(0, lambda: 0) == 0.5
    assert backoff(1, lambda: 0) == 1.0
    assert backoff(10, lambda: 0) == 8.0
    assert backoff(0, lambda: 1) == 0.375
    assert is_retryable_status(429) and is_retryable_status(500) and is_retryable_status(529)
    assert not (is_retryable_status(400) or is_retryable_status(404) or is_retryable_status(409))


def test_describe_error_bodies() -> None:
    assert describe_error_body({"type": "error", "error": {"type": "insufficient_ot", "message": "short"}}) == ("insufficient_ot", "short", None)
    assert describe_error_body({"error": {"message": "m", "type": "authentication_error", "code": None, "param": None}}) == ("authentication_error", "m", None)
    assert describe_error_body({"error": "key_expired_or_revoked", "reason": "rotated"}) == ("key_expired_or_revoked", None, "rotated")
    assert describe_error_body({"type": "error", "error": "assistant_mismatch", "stored": "a", "requested": "b", "message": "pinned"}) == ("assistant_mismatch", "pinned", None)
    assert describe_error_body("text") == (None, None, None)


def test_revoked_keys_reason() -> None:
    c, _ = client((401, {}, '{"error":"key_expired_or_revoked","reason":"revoked"}'))
    with pytest.raises(AuthenticationError) as info:
        c.models.list()
    assert info.value.reason == "revoked"


def test_gateway_rate_limit() -> None:
    fail = (429, {"Retry-After": "0"}, '{"error":"api_key_rate_limit_exceeded"}')
    c, _ = client(fail, fail, fail)
    with pytest.raises(RateLimitError) as info:
        c.models.list()
    assert info.value.code == "api_key_rate_limit_exceeded"


def decode_all(text: str, size: int) -> list[tuple[str | None, str]]:
    d = SSEDecoder()
    data = text.encode()
    out = []
    for i in range(0, len(data), size):
        out += d.push(data[i : i + size])
    out += d.end()
    return [(e.event, e.data) for e in out]


@pytest.mark.parametrize("size", [1, 2, 3, 5, 100])
def test_sse_decoder(size: int) -> None:
    text = ": comment\r\nevent: a\r\ndata: 1\r\ndata: 2\r\n\r\nevent: b\rdata:x\r\rdata: tail"
    assert decode_all(text, size) == [("a", "1\n2"), ("b", "x"), (None, "tail")]


def test_sse_decoder_keeps_split_multibyte_characters() -> None:
    assert decode_all("data: café ☕\n\n", 1) == [(None, "café ☕")]


def sse_client(body: str) -> Orbtr:
    t = Scripted((200, {"Content-Type": "text/event-stream"}, body))
    return Orbtr(api_key="k", http_client=httpx.Client(transport=t))


START = 'event: message_start\ndata: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"x","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'


def test_error_event_raises_stream_error() -> None:
    body = START + 'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"inference stream failed"}}\n\n'
    stream = sse_client(body).messages.stream(model="x", messages=[])
    seen = []
    with pytest.raises(StreamError) as info:
        for ev in stream:
            seen.append(ev.type)
    assert info.value.code == "api_error"
    assert seen == ["message_start"]


def test_progress_events_pass_through() -> None:
    body = 'event: orbtr.ai.turn.started\ndata: {"turn":"t1"}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n'
    events = list(sse_client(body).messages.stream(model="x", messages=[]))
    assert events[0] == ExtensionStreamEvent(type="orbtr.ai.turn.started", data={"turn": "t1"})
    assert events[1].type == "message_stop"


def test_tool_input_gathered_from_json_pieces() -> None:
    body = "\n\n".join([
        START.strip(),
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"lookup","input":{}}}',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\": \\"Sy"}}',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"dney\\"}"}}',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5}}',
        'event: message_stop\ndata: {"type":"message_stop"}',
    ]) + "\n\n"
    m = sse_client(body).messages.stream(model="x", messages=[]).final_message()
    assert m.content[0].input == {"city": "Sydney"}
    assert m.stop_reason == "tool_use"
    assert (m.usage.input_tokens, m.usage.output_tokens) == (1, 5)


def test_flow_models() -> None:
    assert flow_model("partition:p1:flow:council") == "partition:p1:flow:council"
    assert flow_model("partition:p1:flow:council", 3) == "partition:p1:flow:council@3"
    for bad in [("orbtr-model-1", None), ("partition:p1:flow:council@2", 3), ("partition:p1:flow:council", 0), ("partition:p1:flow:council", True)]:
        with pytest.raises(ValueError):
            flow_model(*bad)  # type: ignore[arg-type]
    assert is_flow_model("partition:p1:flow:council@2") and not is_flow_model("orbtr-model-1")
    assert parse_flow_model("partition:p1:flow:council@2") == ("partition:p1:flow:council", 2)
    assert parse_flow_model("partition:p1:flow:council") == ("partition:p1:flow:council", None)
    assert parse_flow_model("orbtr-model-1") is None


def test_unknown_response_fields_are_kept() -> None:
    from orbtr.types import Model

    m = Model.from_dict({"id": "a", "object": "model", "created": 0, "owned_by": "orbtr", "orbtr": {"kind": "flow"}})
    assert m.extra == {"orbtr": {"kind": "flow"}}
    assert m.to_dict()["orbtr"] == {"kind": "flow"}
