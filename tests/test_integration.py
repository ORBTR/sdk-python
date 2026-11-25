"""The package over real HTTP against a local fake of the /v1 surface (fake_server.py): every call, both
clients, streaming, flows as models, retries after 429 and 503, typed errors, and each example in
examples/."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from typing import Iterator

import pytest

from orbtr import AsyncOrbtr, AuthenticationError, NotFoundError, Orbtr, StreamError, flow_model

from fake_server import FAKE_KEY, Fake
from helpers import ROOT


@pytest.fixture(scope="module")
def fake() -> Iterator[Fake]:
    f = Fake()
    yield f
    f.close()


def user(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def test_a_whole_session(fake: Fake) -> None:
    with Orbtr(api_key=FAKE_KEY, base_url=fake.url) as c:
        assert any(m.id == "orbtr-model-1" for m in c.models.list().data)
        assert c.messages.create(model="orbtr-model-1", max_tokens=256, messages=user("sea")).content[0].text == "The sea is wide and grey."
        with c.messages.stream(model="orbtr-model-1", max_tokens=256, messages=user("sea")) as s:
            assert s.final_text() == "The sea is wide and grey."
        assert c.messages.count_tokens(model="orbtr-model-1", messages=user("how long?")).input_tokens == 14
        flow = c.messages.create(model=flow_model("partition:p1:flow:council", 2), max_tokens=64, messages=user("review this"))
        assert flow.usage.orbtr is not None and flow.usage.orbtr.version == 2
        with pytest.raises(NotFoundError):
            c.messages.create(model=flow_model("partition:p1:flow:missing"), max_tokens=64, messages=user("x"))
        assert len(c.embeddings.create(model="orbtr-embed-1", input=["the sea", "the sky"]).data) == 2
        assert c.chat.completions.create(model="orbtr-model-1", messages=user("hi")).choices[0].message.content == "The sea is wide and grey."
        assert "".join(ch.choices[0].delta.content or "" for ch in c.chat.completions.stream(model="orbtr-model-1", messages=user("hi"))) == "The sea is wide and grey."
        assert c.completions.create(model="orbtr-model-1", prompt="The sea is").choices[0].text == "The sea is wide and grey."
        batch = c.messages.batches.create(requests=[{"custom_id": "first", "params": {"model": "orbtr-model-1", "max_tokens": 64, "messages": user("Say hello.")}}])
        assert batch.id == "msgbatch_sdk_01"
        assert len(c.messages.batches.list(limit=20).data) == 1
        assert c.messages.batches.retrieve(batch.id).processing_status == "ended"
        assert c.messages.batches.cancel(batch.id).processing_status == "canceling"
        assert [r.result.type for r in c.messages.batches.results(batch.id)] == ["succeeded", "errored"]
        with pytest.raises(NotFoundError):
            c.messages.batches.retrieve("msgbatch_missing")
    for r in fake.requests:
        assert r["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
        assert "X-Api-Key" not in r["headers"] and "x-api-key" not in r["headers"]


def test_the_async_client(fake: Fake) -> None:
    async def run() -> None:
        async with AsyncOrbtr(api_key=FAKE_KEY, base_url=fake.url) as c:
            msg = await c.messages.create(model="orbtr-model-1", max_tokens=256, messages=user("sea"))
            assert msg.content[0].text == "The sea is wide and grey."
            stream = await c.messages.stream(model="orbtr-model-1", max_tokens=256, messages=user("sea"))
            assert await stream.final_text() == "The sea is wide and grey."
            chunks = [ch async for ch in await c.chat.completions.stream(model="orbtr-model-1", messages=user("hi"))]
            assert chunks[-1].choices[0].finish_reason == "stop"
            results = await c.messages.batches.results("msgbatch_sdk_01")
            assert len(results) == 2

    asyncio.run(run())


def test_retries_after_429_and_503_and_one_idempotency_key(fake: Fake) -> None:
    c = Orbtr(api_key=FAKE_KEY, base_url=fake.url)
    n = len(fake.requests)
    assert c.messages.create(model="orbtr-model-1", max_tokens=16, messages=user("hi"), extra_headers={"X-Fake-Scenario": "retry-429:a"}).id == "msg_sdk_01"
    assert len(fake.requests) - n == 2
    n = len(fake.requests)
    c.messages.batches.create(requests=[], extra_headers={"X-Fake-Scenario": "retry-503:b"})
    sent = fake.requests[n:]
    assert len(sent) == 2
    assert sent[0]["headers"]["Idempotency-Key"] and sent[0]["headers"]["Idempotency-Key"] == sent[1]["headers"]["Idempotency-Key"]


def test_a_wrong_key_is_not_retried(fake: Fake) -> None:
    n = len(fake.requests)
    with pytest.raises(AuthenticationError) as info:
        Orbtr(api_key="wrong", base_url=fake.url).models.list()
    assert info.value.code == "invalid_api_key"
    assert len(fake.requests) - n == 1


def test_an_error_event_ends_a_stream(fake: Fake) -> None:
    stream = Orbtr(api_key=FAKE_KEY, base_url=fake.url).messages.stream(
        model="orbtr-model-1", max_tokens=16, messages=user("hi"), extra_headers={"X-Fake-Scenario": "stream-error"}
    )
    with pytest.raises(StreamError) as info:
        stream.final_message()
    assert info.value.code == "api_error"


EXPECTED = {
    "send_message.py": r"orbtr-model-1\n[\s\S]*The sea is wide and grey\.",
    "stream_reply.py": r"The sea is wide and grey\.",
    "count_tokens.py": r"input tokens: 14\ncontext length: 131072",
    "chat_completions.py": r"The sea is wide and grey\.",
    "flow_model.py": r"the council agrees\nflow partition:p1:flow:council version 2: 7 OT over 2 steps",
}


def test_every_example_runs_against_the_fake(fake: Fake) -> None:
    files = sorted(p.name for p in (ROOT / "examples").glob("*.py"))
    assert files == sorted(EXPECTED)
    env = {
        **os.environ,
        "ORBTR_API_KEY": FAKE_KEY,
        "ORBTR_BASE_URL": fake.url,
        "ORBTR_MODEL": "orbtr-model-1",
        "ORBTR_FLOW_ID": "partition:p1:flow:council",
        "ORBTR_FLOW_VERSION": "2",
    }
    for f in files:
        out = subprocess.run([sys.executable, str(ROOT / "examples" / f)], env=env, capture_output=True, text=True, timeout=120, check=False)
        assert out.returncode == 0, f"{f}: {out.stderr}"
        assert re.search(EXPECTED[f], out.stdout), f"{f}: {out.stdout}"
