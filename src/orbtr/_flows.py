"""Flows as models: a published flow is called by naming its id as a call's model, and <flow id>@<version>
pins a published version. api.orbtr.ai reads a model that starts with "partition:" and holds ":flow:" as a
flow; any other model is a model id."""

from __future__ import annotations

__all__ = ["flow_model", "is_flow_model", "parse_flow_model"]


def is_flow_model(model: str) -> bool:
    """Whether a model names a flow."""
    return model.startswith("partition:") and ":flow:" in model


def flow_model(flow_id: str, version: int | None = None) -> str:
    """The model that calls a published flow: its newest published version, or the version given."""
    if not is_flow_model(flow_id) or "@" in flow_id:
        raise ValueError(f"Not a flow id: {flow_id!r} (a flow id looks like partition:<partition>:flow:<name>)")
    if version is None:
        return flow_id
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f"A flow version is a whole number from 1, not {version!r}")
    return f"{flow_id}@{version}"


def parse_flow_model(model: str) -> tuple[str, int | None] | None:
    """Splits a flow model into its flow id and pinned version; None for a model that names no flow."""
    if not is_flow_model(model):
        return None
    flow_id, at, pin = model.rpartition("@")
    if not at:
        return model, None
    if pin.isdigit() and int(pin) >= 1:
        return flow_id, int(pin)
    return flow_id, None
