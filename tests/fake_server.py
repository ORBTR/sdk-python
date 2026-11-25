"""A local fake of api.orbtr.ai's /v1 surface for the integration test.

Each route answers with the body recorded from its handler (fixtures/), after the checks api.orbtr.ai makes
that a client can get wrong: the bearer key, the JSON body, the Idempotency-Key a batch create needs and
refuses elsewhere, and a flow id the key cannot read. The X-Fake-Scenario header scripts failures:
"retry-429" and "retry-503" fail the first call of a scenario id ("<kind>:<id>") with Retry-After: 0, and
"stream-error" ends a stream with an error event.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from helpers import fixture

FAKE_KEY = "test-key"
FLOW = "partition:p1:flow:council"


class Fake:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.failed: set[str] = set()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                fake.handle(self)

            def do_POST(self) -> None:
                fake.handle(self)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    @staticmethod
    def send(h: BaseHTTPRequestHandler, status: int, headers: dict[str, str], body: str) -> None:
        data = body.encode()
        h.send_response(status)
        for k, v in headers.items():
            h.send_header(k, v)
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)

    def fixture(self, h: BaseHTTPRequestHandler, name: str) -> None:
        r = fixture(name)["response"]
        self.send(h, r["status"], r["headers"], r["body"])

    def json(self, h: BaseHTTPRequestHandler, status: int, body: Any, headers: dict[str, str] | None = None) -> None:
        self.send(h, status, {"Content-Type": "application/json", **(headers or {})}, json.dumps(body))

    def handle(self, h: BaseHTTPRequestHandler) -> None:
        n = int(h.headers.get("Content-Length") or 0)
        raw = h.rfile.read(n).decode() if n else ""
        path, _, _query = h.path.partition("?")
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            return self.json(h, 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "request body is not valid JSON"}})
        self.requests.append({"method": h.command, "path": path, "headers": dict(h.headers.items()), "body": body})

        auth = h.headers.get("Authorization") or ""
        if not auth.startswith("Bearer "):
            return self.fixture(h, "error-missing-bearer-token")
        if auth != f"Bearer {FAKE_KEY}":
            return self.json(h, 401, {"error": "invalid_api_key"})

        scenario = h.headers.get("X-Fake-Scenario") or ""
        kind = scenario.split(":")[0]
        if kind in ("retry-429", "retry-503") and scenario not in self.failed:
            self.failed.add(scenario)
            if kind == "retry-429":
                return self.json(h, 429, {"type": "error", "error": {"type": "partition_inflight_exceeded", "message": "too many concurrent inference requests for this partition"}}, {"Retry-After": "0"})
            return self.json(h, 503, {"type": "error", "error": {"type": "no_servable_placement", "message": "nothing can serve the model now"}}, {"Retry-After": "0"})

        b = body if isinstance(body, dict) else {}
        route = f"{h.command} {path}"
        idem = h.headers.get("Idempotency-Key")
        if idem is not None and route not in ("POST /v1/messages/batches", "POST /v1/messages/count_tokens") and not route.endswith("/cancel"):
            return self.json(h, 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "this route takes no Idempotency-Key"}})
        model = b.get("model")
        is_flow = isinstance(model, str) and model.startswith("partition:") and ":flow:" in model
        if is_flow and model.split("@")[0] != FLOW:
            return self.fixture(h, "error-flow-not-found")

        if route == "GET /v1/models":
            return self.fixture(h, "models-list")
        if route == "POST /v1/messages":
            if b.get("stream") and kind == "stream-error":
                start = fixture("messages-stream")["response"]["body"].split("\n\n")[0]
                err = 'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"inference stream failed"}}'
                return self.send(h, 200, {"Content-Type": "text/event-stream"}, f"{start}\n\n{err}\n\n")
            if is_flow:
                return self.fixture(h, "messages-stream-flow" if b.get("stream") else "messages-create-flow")
            return self.fixture(h, "messages-stream" if b.get("stream") else "messages-create")
        if route == "POST /v1/messages/count_tokens":
            return self.fixture(h, "messages-count-tokens")
        if route == "POST /v1/embeddings":
            return self.fixture(h, "embeddings-create")
        if route == "POST /v1/chat/completions":
            if is_flow:
                return self.fixture(h, "chat-completions-create-flow")
            return self.fixture(h, "chat-completions-stream" if b.get("stream") else "chat-completions-create")
        if route == "POST /v1/completions":
            return self.fixture(h, "completions-create")
        if route == "POST /v1/messages/batches":
            if not idem:
                return self.json(h, 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "Idempotency-Key is required"}})
            return self.fixture(h, "batches-create")
        if route == "GET /v1/messages/batches":
            return self.fixture(h, "batches-list")
        m = re.fullmatch(r"/v1/messages/batches/([^/]+)(/cancel|/results)?", path)
        if m:
            if m.group(1) != "msgbatch_sdk_01":
                return self.json(h, 404, {"type": "error", "error": {"type": "not_found_error", "message": "message batch not found"}})
            if h.command == "GET" and not m.group(2):
                return self.fixture(h, "batches-retrieve")
            if h.command == "POST" and m.group(2) == "/cancel":
                return self.fixture(h, "batches-cancel")
            if h.command == "GET" and m.group(2) == "/results":
                return self.fixture(h, "batches-results")
        return self.json(h, 404, {"type": "error", "error": {"type": "not_found_error", "message": f"no route {route}"}})
