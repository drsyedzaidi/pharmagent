"""Observe the real tool registry without changing it: every execute() is recorded, then passed through."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Call:
    tool: str
    args: dict[str, Any]
    ok: bool
    error: str = ""
    error_type: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    owner_agent: str = ""
    expensive: bool = False
    allow_expensive: bool = False


class RecordingRegistry:
    """Wraps a ToolRegistry. Same interface for the code under test; adds ``calls`` for the harness."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls: list[Call] = []

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def execute(self, name: str, *, args: dict[str, Any], allow_expensive: bool = False, **kw):
        try:
            tool = self._inner.get(name)
            meta = dict(owner_agent=tool.agent, expensive=tool.expensive)
        except KeyError:
            meta = dict(owner_agent="", expensive=False)
        try:
            state, res = self._inner.execute(name, args=args, allow_expensive=allow_expensive, **kw)
        except Exception as exc:
            self.calls.append(Call(name, dict(args or {}), False, str(exc), type(exc).__name__,
                                   allow_expensive=allow_expensive, **meta))
            raise
        self.calls.append(Call(name, dict(args or {}), True, result=res.result,
                               allow_expensive=allow_expensive, **meta))
        return state, res

    def mark(self) -> int:
        return len(self.calls)

    def since(self, mark: int) -> list[Call]:
        return self.calls[mark:]
