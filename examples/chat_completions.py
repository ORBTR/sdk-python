"""The OpenAI-compatible endpoint: the same message through /v1/chat/completions (docs.orbtr.ai, Recipes,
"Send a message with curl" and "Use the OpenAI Python SDK"). Needs ORBTR_API_KEY and ORBTR_MODEL."""

import os

from orbtr import Orbtr

client = Orbtr()
reply = client.chat.completions.create(
    model=os.environ.get("ORBTR_MODEL", "MODEL_ID"),
    messages=[{"role": "user", "content": "Write one sentence about the sea."}],
)
print(reply.choices[0].message.content)
