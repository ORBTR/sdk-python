"""Each call against the exchange recorded from api.orbtr.ai's handler: the SDK sends the recorded request
(method, path, query, body) with the API key as a bearer token, and reads the recorded response."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from orbtr import AsyncOrbtr, AuthenticationError, NotFoundError, Orbtr, RateLimitError, flow_model
from orbtr.types import ContentBlockDeltaEvent, Message, MessageStartEvent, PingEvent

from helpers import AsyncFixtureTransport, FixtureTransport, assert_sent, fixture_names

SEA = [{"role": "user", "content": "Write one sentence about the sea."}]


def client(*names: str, max_retries: int = 0) -> tuple[Orbtr, FixtureTransport]:
    t = FixtureTransport(*names)
    c = Orbtr(api_key="test-key", base_url="https://api.orbtr.test", max_retries=max_retries, http_client=httpx.Client(transport=t))
    return c, t


def test_every_recorded_fixture_is_used() -> None:
    used = {
        "messages-create", "messages-stream", "messages-stream-rendered", "messages-count-tokens",
        "messages-create-flow", "messages-stream-flow", "chat-completions-create", "chat-completions-stream",
        "chat-completions-create-flow", "completions-create", "embeddings-create", "models-list",
        "batches-create", "batches-list", "batches-retrieve", "batches-cancel", "batches-results",
        "error-flow-not-found", "error-messages-no-key", "error-chat-no-key", "error-missing-bearer-token",
        "error-rate-limited",
    }
    assert fixture_names() == sorted(used)


def test_messages_create() -> None:
    c, t = client("messages-create")
    msg = c.messages.create(model="orbtr-model-1", max_tokens=256, messages=SEA)
    assert_sent(t.sent[0], "messages-create")
    assert "idempotency-key" not in t.sent[0].headers
    assert isinstance(msg, Message)
    assert msg.id == "msg_sdk_01"
    assert msg.content[0].type == "text"
    assert msg.content[0].text == "The sea is wide and grey."
    assert msg.stop_reason == "end_turn"
    assert (msg.usage.input_tokens, msg.usage.output_tokens, msg.usage.ot_consumed) == (14, 9, 12)
    assert msg.usage.billing_profile == "bp_1"
    assert msg.to_dict()["usage"]["reservation_id"] == "res_sdk_1"


def test_messages_stream_native() -> None:
    c, t = client("messages-stream")
    with c.messages.stream(model="orbtr-model-1", max_tokens=256, messages=SEA) as stream:
        events = list(stream)
        final = stream.final_message()
    assert_sent(t.sent[0], "messages-stream")
    assert t.sent[0].headers["accept"] == "text/event-stream"
    assert [e.type for e in events] == [
        "message_start", "content_block_start", "ping", "content_block_delta", "content_block_delta",
        "content_block_stop", "message_delta", "message_stop",
    ]
    assert isinstance(events[0], MessageStartEvent)
    assert isinstance(events[2], PingEvent)
    assert "".join(e.delta.text or "" for e in events if isinstance(e, ContentBlockDeltaEvent)) == "The sea is wide and grey."
    assert final.content[0].text == "The sea is wide and grey."
    assert final.stop_reason == "end_turn"
    assert final.usage.output_tokens == 9
    assert final.usage.input_tokens == 14


def test_messages_stream_rendered() -> None:
    c, t = client("messages-stream-rendered")
    stream = c.messages.stream(model="orbtr-model-1", max_tokens=256, messages=SEA)
    assert stream.final_text() == "The sea is wide and grey."
    assert_sent(t.sent[0], "messages-stream-rendered")


def test_text_stream() -> None:
    c, _ = client("messages-stream")
    stream = c.messages.stream(model="orbtr-model-1", max_tokens=256, messages=SEA)
    assert "".join(stream.text_stream()) == "The sea is wide and grey."


def test_count_tokens() -> None:
    c, t = client("messages-count-tokens")
    n = c.messages.count_tokens(model="orbtr-model-1", messages=[{"role": "user", "content": "How long is this request?"}])
    assert_sent(t.sent[0], "messages-count-tokens")
    assert n.input_tokens == 14


def test_flow_called_as_a_model_pinned_to_a_version() -> None:
    c, t = client("messages-create-flow")
    msg = c.messages.create(
        model=flow_model("partition:p1:flow:council", 2), max_tokens=64, messages=[{"role": "user", "content": "review this"}]
    )
    assert_sent(t.sent[0], "messages-create-flow")
    assert msg.content[0].text == "the council agrees"
    run = msg.usage.orbtr
    assert run is not None
    assert (run.kind, run.flow_id, run.version, run.run_id, run.total_ot) == ("flow", "partition:p1:flow:council", 2, "run1", 7)
    assert sum(s.ot for s in run.steps) == run.total_ot
    assert run.steps[0].model_id == "orbtr-small"
    assert run.to_dict()["flowId"] == "partition:p1:flow:council"


def test_streamed_flow_carries_the_runs_usage() -> None:
    c, t = client("messages-stream-flow")
    stream = c.messages.stream(model=flow_model("partition:p1:flow:council"), max_tokens=64, messages=[{"role": "user", "content": "review this"}])
    final = stream.final_message()
    assert_sent(t.sent[0], "messages-stream-flow")
    assert final.content[0].text == "the council agrees"
    assert final.usage.orbtr is not None and final.usage.orbtr.run_id == "run1"


def test_chat_completions() -> None:
    c, t = client("chat-completions-create", "chat-completions-stream", "chat-completions-create-flow")
    reply = c.chat.completions.create(model="orbtr-model-1", messages=SEA)
    assert_sent(t.sent[0], "chat-completions-create")
    assert reply.choices[0].message.content == "The sea is wide and grey."
    assert reply.usage.total_tokens == 23
    chunks = list(c.chat.completions.stream(model="orbtr-model-1", messages=SEA))
    assert_sent(t.sent[1], "chat-completions-stream")
    assert "".join(ch.choices[0].delta.content or "" for ch in chunks) == "The sea is wide and grey."
    assert chunks[-1].choices[0].finish_reason == "stop"
    flow = c.chat.completions.create(model="partition:p1:flow:council", messages=[{"role": "user", "content": "review this"}])
    assert_sent(t.sent[2], "chat-completions-create-flow")
    assert flow.usage.orbtr is not None and flow.usage.orbtr.version == 2


def test_completions_and_embeddings() -> None:
    c, t = client("completions-create", "embeddings-create")
    reply = c.completions.create(model="orbtr-model-1", prompt="The sea is", max_tokens=32)
    assert_sent(t.sent[0], "completions-create")
    assert reply.choices[0].text == "The sea is wide and grey."
    out = c.embeddings.create(model="orbtr-embed-1", input=["the sea", "the sky"])
    assert_sent(t.sent[1], "embeddings-create")
    assert out.data[0].embedding == [0.25, -0.5, 0.125]
    assert out.usage.prompt_tokens == 6


def test_models_list() -> None:
    c, t = client("models-list")
    models = c.models.list()
    assert_sent(t.sent[0], "models-list")
    assert [m.id for m in models.data] == ["orbtr-embed-1", "orbtr-model-1", "orbtr-model-2"]
    assert models.data[1].context_length == 131072
    assert models.data[0].context_length is None


BATCH = [
    {"custom_id": "first", "params": {"model": "orbtr-model-1", "max_tokens": 64, "messages": [{"role": "user", "content": "Say hello."}]}},
    {"custom_id": "second", "params": {"model": "orbtr-model-2", "max_tokens": 64, "messages": [{"role": "user", "content": "Say hello."}]}},
]


def test_batches() -> None:
    c, t = client("batches-create", "batches-create", "batches-list", "batches-retrieve", "batches-cancel", "batches-results")
    batch = c.messages.batches.create(requests=BATCH)
    assert_sent(t.sent[0], "batches-create")
    assert t.sent[0].headers["idempotency-key"].startswith("orbtr-sdk-")
    assert batch.processing_status == "in_progress"
    assert batch.request_counts.processing == 2
    c.messages.batches.create(requests=BATCH, idempotency_key="nightly-2026-11-25")
    assert t.sent[1].headers["idempotency-key"] == "nightly-2026-11-25"
    page = c.messages.batches.list(limit=20)
    assert_sent(t.sent[2], "batches-list")
    assert page.has_more is False and page.last_id == "msgbatch_sdk_01"
    assert c.messages.batches.retrieve("msgbatch_sdk_01").processing_status == "ended"
    assert_sent(t.sent[3], "batches-retrieve")
    assert c.messages.batches.cancel("msgbatch_sdk_01").processing_status == "canceling"
    assert_sent(t.sent[4], "batches-cancel")
    results = c.messages.batches.results("msgbatch_sdk_01")
    assert_sent(t.sent[5], "batches-results")
    assert t.sent[5].headers["accept"] == "application/x-ndjson"
    assert [(r.custom_id, r.result.type) for r in results] == [("first", "succeeded"), ("second", "errored")]
    assert results[0].result.message is not None and results[0].result.message.content[0].text == "Hello."
    assert results[1].result.error is not None and results[1].result.error.type == "model_not_entitled"


def test_unknown_flow_raises_not_found() -> None:
    c, _ = client("error-flow-not-found")
    with pytest.raises(NotFoundError) as info:
        c.messages.create(model="partition:p1:flow:unknown", max_tokens=64, messages=[{"role": "user", "content": "review this"}])
    assert info.value.status == 404 and info.value.code == "not_found_error"


@pytest.mark.parametrize(
    "name,code",
    [("error-messages-no-key", "authentication_error"), ("error-chat-no-key", "authentication_error"), ("error-missing-bearer-token", "missing_bearer_token")],
)
def test_refusal_shapes_raise_authentication_error(name: str, code: str) -> None:
    c, _ = client(name)
    with pytest.raises(AuthenticationError) as info:
        c.models.list()
    assert info.value.status == 401 and info.value.code == code


def test_a_429_is_retried_then_raised() -> None:
    c, t = client("error-rate-limited", "error-rate-limited", max_retries=1)
    with pytest.raises(RateLimitError) as info:
        c.messages.create(model="orbtr-model-1", max_tokens=256, messages=SEA)
    assert info.value.code == "partition_inflight_exceeded"
    assert info.value.headers["x-admission-reject-reason"] == "partition_inflight_exceeded"
    assert len(t.sent) == 2


def test_async_client_against_fixtures() -> None:
    async def run() -> None:
        t = AsyncFixtureTransport("messages-create", "messages-stream", "batches-results")
        c = AsyncOrbtr(api_key="test-key", base_url="https://api.orbtr.test", max_retries=0, http_client=httpx.AsyncClient(transport=t))
        msg = await c.messages.create(model="orbtr-model-1", max_tokens=256, messages=SEA)
        assert_sent(t.sent[0], "messages-create")
        assert msg.content[0].text == "The sea is wide and grey."
        async with await c.messages.stream(model="orbtr-model-1", max_tokens=256, messages=SEA) as stream:
            text = "".join([piece async for piece in stream.text_stream()])
        assert text == "The sea is wide and grey."
        assert_sent(t.sent[1], "messages-stream")
        results = await c.messages.batches.results("msgbatch_sdk_01")
        assert_sent(t.sent[2], "batches-results")
        assert len(results) == 2

    asyncio.run(run())
