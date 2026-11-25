"""Count tokens before you send: count_tokens takes the same body as a message and answers with the number of
input tokens (docs.orbtr.ai, Recipes, "Count tokens before you send"). Needs ORBTR_API_KEY with the
orbtr.inference.read scope, and ORBTR_MODEL."""

import os

from orbtr import Orbtr

client = Orbtr()
model = os.environ.get("ORBTR_MODEL", "MODEL_ID")
count = client.messages.count_tokens(
    model=model,
    messages=[{"role": "user", "content": "How long is this request?"}],
)
print(f"input tokens: {count.input_tokens}")

# Compare it with the model's context_length from the model list. A request larger than the model's context
# limit is refused with an error that names the limit; it is never cut short.
listed = next((m for m in client.models.list().data if m.id == model), None)
if listed is not None and listed.context_length:
    print(f"context length: {listed.context_length}")
