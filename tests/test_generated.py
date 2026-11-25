"""src/orbtr/_generated is what contract/orbtr-v1.json generates: a change to the contract lands with its
regenerated files, and a hand edit of a generated file fails here."""

from __future__ import annotations

import importlib.util
import json

from orbtr import AsyncOrbtr, Orbtr

from helpers import ROOT


def load_generator():
    spec = importlib.util.spec_from_file_location("generate", ROOT / "scripts" / "generate.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generated_files_match_the_contract() -> None:
    gen = load_generator()
    contract = json.loads((ROOT / "contract" / "orbtr-v1.json").read_text())
    for rel, text in gen.generate(contract).items():
        assert (ROOT / rel).read_text() == text, rel


def test_every_contract_operation_is_a_method_of_both_clients() -> None:
    contract = json.loads((ROOT / "contract" / "orbtr-v1.json").read_text())
    names = {"countTokens": "count_tokens"}
    for c in (Orbtr(api_key="k"), AsyncOrbtr(api_key="k")):
        for item in contract["paths"].values():
            for op in item.values():
                target = c
                for seg in op["x-orbtr-resource"].split("."):
                    target = getattr(target, seg)
                methods = [names.get(op["x-orbtr-method"], op["x-orbtr-method"])] + (["stream"] if op.get("x-orbtr-stream") else [])
                for m in methods:
                    assert callable(getattr(target, m)), f"{op['x-orbtr-resource']}.{m}"
