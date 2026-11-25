# ORBTR SDK for Python

`orbtr` calls api.orbtr.ai: messages, streaming, models, flows as models, token counting, embeddings,
OpenAI-compatible chat and completions, and message batches, with a synchronous and an asynchronous client.

Its types and calls are generated from the API contract api.orbtr.ai serves (`contract/orbtr-v1.json`); a
small hand-written layer adds the API key, the base URL, streaming, retries and typed errors.

## Install

```bash
pip install orbtr
```

Python 3.11 or later. The one dependency is `httpx`.

## Make a client

```python
from orbtr import Orbtr

client = Orbtr()  # reads ORBTR_API_KEY
```

Make an API key at [manage.orbtr.ai/api_keys](https://manage.orbtr.ai/api_keys). The client sends it as
`Authorization: Bearer <key>`. Each call needs the key to carry the scope the call names
(`orbtr.inference.invoke` to send messages, `orbtr.inference.read` to list models and count tokens, and
`orbtr.messages.batch.*` for batches).

| Argument | Default | What it does |
|---|---|---|
| `api_key` | `ORBTR_API_KEY` | The API key. |
| `base_url` | `ORBTR_BASE_URL`, then `https://api.orbtr.ai` | The API origin. |
| `max_retries` | `2` | Retries after a 429, a 5xx or a connection failure. |
| `timeout` | `600` | Seconds a call may wait on the network. |
| `default_headers` | none | Headers sent with every call. |
| `http_client` | a new `httpx.Client` | The HTTP client to send through. |

`AsyncOrbtr` takes the same arguments (with an `httpx.AsyncClient`) and its methods are awaited.

## Send a message

```python
models = client.models.list()
message = client.messages.create(
    model=models.data[0].id,
    max_tokens=256,
    messages=[{"role": "user", "content": "Write one sentence about the sea."}],
)
print(message.content[0].text)
```

Requests are plain dictionaries (typed as `TypedDict`s in `orbtr.types`); replies are dataclasses whose
attributes follow the API's fields, with `to_dict()` for the JSON and `extra` for fields this version does not
know.

## Stream a reply

```python
with client.messages.stream(
    model="MODEL_ID",
    max_tokens=512,
    messages=[{"role": "user", "content": "Explain what a Space is in three sentences."}],
) as stream:
    for text in stream.text_stream():
        print(text, end="", flush=True)
```

Iterating the stream itself gives each event, in the Anthropic Messages streaming format.
`stream.final_message()` gathers them into the whole reply. `client.chat.completions.stream(...)` and
`client.completions.stream(...)` stream the OpenAI-compatible routes, ending at `[DONE]`. On `AsyncOrbtr`, use
`async with` and `async for`.

## Call a flow as a model

A published flow answers as a model. Name its id as the `model`; pin a published version with `flow_model`:

```python
from orbtr import flow_model

reply = client.messages.create(
    model=flow_model("partition:PARTITION:flow:NAME", 2),  # "…@2"; leave the version out for the newest
    max_tokens=1024,
    messages=[{"role": "user", "content": "Review this paragraph for tone."}],
)
print(reply.usage.orbtr)  # the version that answered, the run's OT and one line per step
```

## Count tokens

```python
count = client.messages.count_tokens(
    model="MODEL_ID",
    messages=[{"role": "user", "content": "How long is this request?"}],
)
print(count.input_tokens)
```

## Embeddings, OpenAI-compatible calls and batches

```python
client.embeddings.create(model="EMBEDDING_MODEL_ID", input=["the sea", "the sky"])
client.chat.completions.create(model="MODEL_ID", messages=[{"role": "user", "content": "Hello"}])

batch = client.messages.batches.create(
    requests=[{"custom_id": "first", "params": {"model": "MODEL_ID", "max_tokens": 64, "messages": [{"role": "user", "content": "Hello"}]}}],
)
results = client.messages.batches.results(batch.id)  # one result per request
```

Creating a batch sends an `Idempotency-Key`, the same on every retry, so a retried create never makes a second
batch. Pass `idempotency_key=` to choose it.

## Errors and retries

A refused call raises an `APIError` subclass carrying `status`, `code` (such as `invalid_api_key` or
`insufficient_ot`), `message`, `body` and `headers`:

| Status | Error |
|---|---|
| 400 | `BadRequestError` |
| 401 | `AuthenticationError` (`reason` says whether a key was revoked, rotated or expired) |
| 402 | `PaymentRequiredError` |
| 403 | `PermissionDeniedError` |
| 404 | `NotFoundError` |
| 409 | `ConflictError` |
| 422 | `UnprocessableEntityError` |
| 429 | `RateLimitError` |
| 5xx | `InternalServerError` |

`APIConnectionError` and `APITimeoutError` mean no response arrived; `StreamError` means a stream failed after it
began. A 429, a 5xx and a connection failure are retried up to `max_retries` times, waiting as long as a
`Retry-After` header asks (up to 60 seconds; a longer wait is raised to you instead) or, without one, backing
off from half a second to eight. Every call takes `max_retries=`, `timeout=` and `extra_headers=` to change this.

## Examples

`examples/` holds one program per recipe on [docs.orbtr.ai](https://docs.orbtr.ai/recipes/index):
`send_message.py`, `stream_reply.py`, `count_tokens.py`, `chat_completions.py` and `flow_model.py`.

## Developing

```bash
uv venv && uv pip install -e . pytest==9.1.1
.venv/bin/pytest                      # unit tests against recorded fixtures, then the integration test against a local fake
python3 scripts/generate.py           # regenerate src/orbtr/_generated from contract/orbtr-v1.json
```

`fixtures/` holds HTTP exchanges recorded from api.orbtr.ai's handlers, each naming the handler it came from.
Commits follow Conventional Commits; release-please turns them into versions and `CHANGELOG.md`.

## Licence

Proprietary. See [LICENSE](LICENSE): you may use this package with an ORBTR account under ORBTR's Terms of
Service.
