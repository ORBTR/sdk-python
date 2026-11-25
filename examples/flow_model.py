"""Call a published flow as a model (docs.orbtr.ai, Concepts, "Flows as models"): name the flow's id as the
model, and add @ and a version number to pin a published version. Needs ORBTR_API_KEY, ORBTR_FLOW_ID and, to
pin a version, ORBTR_FLOW_VERSION."""

import os

from orbtr import Orbtr, flow_model

client = Orbtr()
flow_id = os.environ.get("ORBTR_FLOW_ID", "partition:PARTITION:flow:NAME")
version = int(os.environ["ORBTR_FLOW_VERSION"]) if os.environ.get("ORBTR_FLOW_VERSION") else None

message = client.messages.create(
    model=flow_model(flow_id, version),
    max_tokens=1024,
    messages=[{"role": "user", "content": "Review this paragraph for tone."}],
)
print(message.content[0].text)

# The usage carries the run: the version that answered, its total OT and one line per step.
run = message.usage.orbtr
if run is not None:
    print(f"flow {run.flow_id} version {run.version}: {run.total_ot} OT over {len(run.steps)} steps")
