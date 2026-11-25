"""Server-sent event streams: the line decoder, the streams every streamed call returns, and MessageStream,
which also gathers a Messages stream's events into the final Message.

A Messages stream yields the generated event type for each Anthropic event type, and ExtensionStreamEvent for
any other event name (ORBTR's orbtr.ai.* progress events); an error event raises StreamError. An
OpenAI-compatible stream yields each data line's JSON, parsed, until data: [DONE].
"""

from __future__ import annotations

import codecs
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Generic, Iterator, TypeVar, Union

import httpx

from ._errors import StreamError
from ._generated.types import (
    ContentBlock,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    Message,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStreamEvent,
    parse_message_stream_event,
)

T = TypeVar("T")

__all__ = ["ServerSentEvent", "SSEDecoder", "ExtensionStreamEvent", "Stream", "AsyncStream", "MessageStream", "AsyncMessageStream"]


@dataclass
class ServerSentEvent:
    """One decoded server-sent event."""

    event: str | None
    data: str
    id: str | None


@dataclass
class ExtensionStreamEvent:
    """An event a Messages stream carries beyond the Anthropic event types, such as orbtr.ai.turn.started."""

    type: str
    data: Any


class SSEDecoder:
    """Splits a byte stream into server-sent events, following the event-stream format: lines end in LF, CR
    or CRLF; a blank line ends an event; a line starting with a colon is a comment; data lines join with LF."""

    def __init__(self) -> None:
        self._buffer = ""
        self._event: str | None = None
        self._data: list[str] = []
        self._id: str | None = None
        self._text = codecs.getincrementaldecoder("utf-8")()

    def push(self, chunk: bytes) -> list[ServerSentEvent]:
        """Decodes one chunk and returns the events it completes."""
        self._buffer += self._text.decode(chunk)
        return self._drain(final=False)

    def end(self) -> list[ServerSentEvent]:
        """Returns the events still buffered when the body ends."""
        self._buffer += self._text.decode(b"", final=True)
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[ServerSentEvent]:
        out: list[ServerSentEvent] = []
        while True:
            i = next((j for j, c in enumerate(self._buffer) if c in "\r\n"), -1)
            if i < 0:
                break
            if self._buffer[i] == "\r":
                if i == len(self._buffer) - 1 and not final:
                    # A CR at the buffer's end may be the first half of a CRLF split across chunks.
                    break
                end = i + 2 if self._buffer[i + 1 : i + 2] == "\n" else i + 1
            else:
                end = i + 1
            line = self._buffer[:i]
            self._buffer = self._buffer[end:]
            ev = self._line(line)
            if ev:
                out.append(ev)
        if final:
            if self._buffer:
                ev = self._line(self._buffer)
                self._buffer = ""
                if ev:
                    out.append(ev)
            ev = self._line("")
            if ev:
                out.append(ev)
        return out

    def _line(self, line: str) -> ServerSentEvent | None:
        if line == "":
            if not self._data and self._event is None:
                return None
            ev = ServerSentEvent(self._event, "\n".join(self._data), self._id)
            self._event = None
            self._data = []
            return ev
        if line.startswith(":"):
            return None
        field, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._event = value
        elif field == "data":
            self._data.append(value)
        elif field == "id":
            self._id = value
        return None


_DONE = object()
_SKIP = object()

_MESSAGE_EVENT_TYPES = {
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
    "ping",
}


def _json(data: str) -> Any:
    try:
        return json.loads(data)
    except ValueError:
        return data


def _stream_error(data: Any) -> StreamError:
    err = data.get("error") if isinstance(data, dict) else None
    if not isinstance(err, dict):
        err = {}
    code = err.get("code") or err.get("type")
    return StreamError(code if isinstance(code, str) else None, err.get("message") or "The stream failed", data)


def message_event_parser(ev: ServerSentEvent) -> Any:
    """Parses a Messages stream's event."""
    data = _json(ev.data)
    name = ev.event or (data.get("type") if isinstance(data, dict) else None)
    if name == "error":
        raise _stream_error(data)
    if name is None:
        return _SKIP
    if name in _MESSAGE_EVENT_TYPES and isinstance(data, dict):
        return parse_message_stream_event({**data, "type": name})
    return ExtensionStreamEvent(type=name, data=data)


def chunk_parser(parse: Callable[[Any], T]) -> Callable[[ServerSentEvent], Any]:
    """Parses an OpenAI-compatible stream's data lines until [DONE]."""

    def run(ev: ServerSentEvent) -> Any:
        if ev.data == "[DONE]":
            return _DONE
        if ev.data == "":
            return _SKIP
        data = _json(ev.data)
        if isinstance(data, dict) and "error" in data and "choices" not in data:
            raise _stream_error(data)
        return parse(data)

    return run


class Stream(Generic[T]):
    """A streamed reply: iterate it with for. A stream can be iterated once; close() ends it early. It is
    also a context manager that closes it."""

    def __init__(self, response: httpx.Response, parse: Callable[[ServerSentEvent], Any]) -> None:
        self.response = response
        self._parse = parse
        self._consumed = False

    def __iter__(self) -> Iterator[T]:
        if self._consumed:
            raise RuntimeError("A stream can be iterated only once.")
        self._consumed = True
        decoder = SSEDecoder()
        try:
            for chunk in self.response.iter_bytes():
                for ev in decoder.push(chunk):
                    value = self._parse(ev)
                    if value is _DONE:
                        return
                    if value is not _SKIP:
                        yield self._observe(value)
            for ev in decoder.end():
                value = self._parse(ev)
                if value is _DONE:
                    return
                if value is not _SKIP:
                    yield self._observe(value)
        finally:
            self.response.close()

    def _observe(self, value: Any) -> Any:
        return value

    def close(self) -> None:
        """Closes the stream and its connection."""
        self.response.close()

    def __enter__(self) -> Stream[T]:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncStream(Generic[T]):
    """A streamed reply: iterate it with async for. A stream can be iterated once; aclose() ends it early.
    It is also an async context manager that closes it."""

    def __init__(self, response: httpx.Response, parse: Callable[[ServerSentEvent], Any]) -> None:
        self.response = response
        self._parse = parse
        self._consumed = False

    async def __aiter__(self) -> AsyncIterator[T]:
        if self._consumed:
            raise RuntimeError("A stream can be iterated only once.")
        self._consumed = True
        decoder = SSEDecoder()
        try:
            async for chunk in self.response.aiter_bytes():
                for ev in decoder.push(chunk):
                    value = self._parse(ev)
                    if value is _DONE:
                        return
                    if value is not _SKIP:
                        yield self._observe(value)
            for ev in decoder.end():
                value = self._parse(ev)
                if value is _DONE:
                    return
                if value is not _SKIP:
                    yield self._observe(value)
        finally:
            await self.response.aclose()

    def _observe(self, value: Any) -> Any:
        return value

    async def aclose(self) -> None:
        """Closes the stream and its connection."""
        await self.response.aclose()

    async def __aenter__(self) -> AsyncStream[T]:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


class _Gatherer:
    """Gathers a Messages stream's events into the Message they describe."""

    def __init__(self) -> None:
        self.message: Message | None = None
        self._partial: dict[int, str] = {}

    def observe(self, ev: Any) -> Any:
        if isinstance(ev, MessageStartEvent):
            self.message = Message.from_dict(ev.message.to_dict())
        elif isinstance(ev, ContentBlockStartEvent) and self.message is not None:
            block = ContentBlock.from_dict(ev.content_block.to_dict())
            while len(self.message.content) <= ev.index:
                self.message.content.append(ContentBlock(type="text"))
            self.message.content[ev.index] = block
        elif isinstance(ev, ContentBlockDeltaEvent) and self.message is not None and ev.index < len(self.message.content):
            block = self.message.content[ev.index]
            if ev.delta.type == "text_delta":
                block.text = (block.text or "") + (ev.delta.text or "")
            elif ev.delta.type == "input_json_delta":
                self._partial[ev.index] = self._partial.get(ev.index, "") + (ev.delta.partial_json or "")
        elif isinstance(ev, ContentBlockStopEvent) and self.message is not None:
            partial = self._partial.pop(ev.index, "")
            if partial and ev.index < len(self.message.content):
                self.message.content[ev.index].input = json.loads(partial)
        elif isinstance(ev, MessageDeltaEvent) and self.message is not None:
            if ev.delta is not None:
                self.message.stop_reason = ev.delta.stop_reason
                self.message.stop_sequence = ev.delta.stop_sequence
            if ev.usage is not None:
                merged = {**self.message.usage.to_dict(), **ev.usage.to_dict()}
                self.message.usage = type(self.message.usage).from_dict(merged)
        return ev

    def final(self) -> Message:
        if self.message is None:
            raise StreamError(None, "The stream ended before message_start", None)
        return self.message

    @staticmethod
    def text(message: Message) -> str:
        return "".join(b.text or "" for b in message.content if b.type == "text")


MessageStreamItem = Union[MessageStreamEvent, ExtensionStreamEvent]


class MessageStream(Stream[MessageStreamItem]):
    """A streamed Messages reply. Besides its events, it gathers the reply: final_message() returns the
    Message the events describe once the stream ends."""

    def __init__(self, response: httpx.Response) -> None:
        super().__init__(response, message_event_parser)
        self._gatherer = _Gatherer()

    def _observe(self, value: Any) -> Any:
        return self._gatherer.observe(value)

    @property
    def current_message(self) -> Message | None:
        """The reply as gathered so far, or None before message_start."""
        return self._gatherer.message

    def final_message(self) -> Message:
        """Reads the rest of the stream and returns the whole reply."""
        if not self._consumed:
            for _ in self:
                pass
        return self._gatherer.final()

    def final_text(self) -> str:
        """Reads the rest of the stream and returns the reply's text."""
        return _Gatherer.text(self.final_message())

    def text_stream(self) -> Iterator[str]:
        """Iterates the reply's text as it arrives."""
        for ev in self:
            if isinstance(ev, ContentBlockDeltaEvent) and ev.delta.type == "text_delta" and ev.delta.text:
                yield ev.delta.text


class AsyncMessageStream(AsyncStream[MessageStreamItem]):
    """A streamed Messages reply on the async client; see MessageStream."""

    def __init__(self, response: httpx.Response) -> None:
        super().__init__(response, message_event_parser)
        self._gatherer = _Gatherer()

    def _observe(self, value: Any) -> Any:
        return self._gatherer.observe(value)

    @property
    def current_message(self) -> Message | None:
        """The reply as gathered so far, or None before message_start."""
        return self._gatherer.message

    async def final_message(self) -> Message:
        """Reads the rest of the stream and returns the whole reply."""
        if not self._consumed:
            async for _ in self:
                pass
        return self._gatherer.final()

    async def final_text(self) -> str:
        """Reads the rest of the stream and returns the reply's text."""
        return _Gatherer.text(await self.final_message())

    async def text_stream(self) -> AsyncIterator[str]:
        """Iterates the reply's text as it arrives."""
        async for ev in self:
            if isinstance(ev, ContentBlockDeltaEvent) and ev.delta.type == "text_delta" and ev.delta.text:
                yield ev.delta.text
