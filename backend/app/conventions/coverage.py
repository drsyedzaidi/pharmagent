"""Convention-to-test coverage: every `tests/<file>.py::<test_function>` a convention cites must exist."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from .schema import Catalog

_REF = re.compile(r"^(tests/[\w./-]+\.py)::([A-Za-z_]\w*)$")


@dataclass(frozen=True)
class CoverageProblem:
    convention_id: str
    ref: str
    reason: str


def _test_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def check_test_coverage(catalog: Catalog, backend_root: Path) -> list[CoverageProblem]:
    problems: list[CoverageProblem] = []
    cache: dict[Path, set[str]] = {}
    for conv in catalog.conventions:
        for ref in conv.tests:
            m = _REF.match(ref)
            if not m:
                problems.append(CoverageProblem(conv.id, ref, "not of the form tests/<file>.py::<test_function>"))
                continue
            path = backend_root / m.group(1)
            if not path.is_file():
                problems.append(CoverageProblem(conv.id, ref, "test file does not exist"))
                continue
            if path not in cache:
                cache[path] = _test_names(path)
            if m.group(2) not in cache[path]:
                problems.append(CoverageProblem(conv.id, ref, "test function not found in file"))
    return problems
