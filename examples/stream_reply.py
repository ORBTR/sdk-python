"""Stream a reply: the reply arrives as server-sent events while it is written (docs.orbtr.ai, Recipes,
"Stream a reply"). Needs ORBTR_API_KEY and ORBTR_MODEL."""

import os

from orbtr import Orbtr
from orbtr.types import ContentBlockDeltaEvent

client = Orbtr()
with client.messages.stream(
    model=os.environ.get("ORBTR_MODEL", "MODEL_ID"),
    max_tokens=512,
    messages=[{"role": "user", "content": "Explain what a Space is in three sentences."}],
) as stream:
    # The text arrives in content_block_delta events, and message_stop ends the stream.
    for event in stream:
        if isinstance(event, ContentBlockDeltaEvent) and event.delta.type == "text_delta":
            print(event.delta.text, end="", flush=True)
print()
