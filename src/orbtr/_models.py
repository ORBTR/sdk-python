"""The base of the generated response types and the helpers their from_dict methods call."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class ResponseModel:
    """The base of every response type."""

    def to_dict(self) -> dict[str, Any]:
        """The object as JSON-shaped data, under the API's field names, with the fields it did not know."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            if f.name == "extra":
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            out[f.metadata.get("json", f.name)] = _plain(value)
        out.update(getattr(self, "extra", {}) or {})
        return out


def _plain(value: Any) -> Any:
    if isinstance(value, ResponseModel):
        return value.to_dict()
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def parse_obj(value: Any, parse: Callable[[dict[str, Any]], T]) -> T | None:
    """Reads a nested object, or None when it is absent."""
    if value is None:
        return None
    return parse(value)


def parse_list(value: Any, parse: Callable[[dict[str, Any]], T] | None) -> list[Any] | None:
    """Reads a list, each item through parse when one is given, or None when it is absent."""
    if value is None:
        return None
    if parse is None:
        return list(value)
    return [parse(v) for v in value]


def extra_keys(data: dict[str, Any], known: tuple[str, ...]) -> dict[str, Any]:
    """The fields of data outside known."""
    return {k: v for k, v in data.items() if k not in known}
