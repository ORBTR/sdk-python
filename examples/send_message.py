"""Send a message: list the model ids, choose one, and send a message (docs.orbtr.ai, Recipes, "Send a message
with curl"). Needs ORBTR_API_KEY with the orbtr.inference.invoke and orbtr.inference.read scopes; ORBTR_MODEL
chooses the model, otherwise the first listed model answers."""

import os

from orbtr import Orbtr

client = Orbtr()

# 1. List the model ids.
models = client.models.list()
for m in models.data:
    print(m.id)

# 2. Choose one.
model = os.environ.get("ORBTR_MODEL") or models.data[0].id

# 3. Send a message.
message = client.messages.create(
    model=model,
    max_tokens=256,
    messages=[{"role": "user", "content": "Write one sentence about the sea."}],
)
print(message.content[0].text)
