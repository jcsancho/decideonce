"""Load JSON Schemas from decideonce/schemas/ and validate with jsonschema."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import jsonschema

SCHEMA_DIR = Path(__file__).parent / "schemas"


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict:
    with open(SCHEMA_DIR / f"{name}.json") as f:
        return json.load(f)


def validate(name: str, value: dict) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    validator = jsonschema.Draft202012Validator(load_schema(name))
    return [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}"
            for e in validator.iter_errors(value)]
