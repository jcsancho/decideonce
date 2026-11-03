"""Error helpers: every tool returns {ok: true, ...} or {ok: false, error, message}."""
from __future__ import annotations


class ToolError(Exception):
    def __init__(self, code: str, message: str, extra: dict | None = None):
        super().__init__(message)
        self.code = code
        self.extra = extra or {}


def err(code: str, message: str, **extra) -> dict:
    out = {"ok": False, "error": code, "message": message}
    out.update(extra)
    return out


def ok(**fields) -> dict:
    return {"ok": True, **fields}
