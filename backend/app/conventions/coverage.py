"""Convention-to-test citation check.

Each reference must be a top-level, pytest-collectable test function, `tests/<path>/test_*.py::test_*`, inside the
tests root. This verifies that the cited test exists and would be collected. It does NOT show that the test exercises
the rule: that judgement belongs to owner review.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .schema import Catalog

_REF = re.compile(r"^(?P<file>[^:]+)::(?P<func>[A-Za-z_]\w*)$")


@dataclass(frozen=True)
class CoverageProblem:
    convention_id: str
    ref: str
    reason: str


def _collectable_file(name: str) -> bool:
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def _resolve(ref_file: str, tests_root: Path, backend_root: Path) -> Path | str:
    """The file a reference points at, or why it is not allowed."""
    if "\\" in ref_file:
        return "backslashes are not allowed in a test path"
    rel = PurePosixPath(ref_file)
    if rel.is_absolute():
        return "an absolute test path is outside the tests root"
    if ".." in rel.parts:
        return "'..' in a test path is outside the tests root"
    if not rel.parts or rel.parts[0] != "tests":
        return "test path must start with tests/ (outside the tests root)"
    if not _collectable_file(rel.name):
        return "file is not one pytest collects (test_*.py or *_test.py)"
    path = backend_root.joinpath(*rel.parts)
    try:
        real = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return "test file does not exist"
    if not real.is_relative_to(tests_root):
        return "test file resolves outside the tests root"
    return real


def _top_level_tests(path: Path) -> set[str] | str:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return f"cannot read test file: {type(exc).__name__}"
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError) as exc:
        return f"cannot parse test file: {type(exc).__name__}"
    return {n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")}


def check_test_coverage(catalog: Catalog, backend_root: Path) -> list[CoverageProblem]:
    backend_root = Path(backend_root)
    try:
        tests_root = (backend_root / "tests").resolve(strict=True)
    except (OSError, RuntimeError):
        return [CoverageProblem(c.id, t, "tests root does not exist") for c in catalog.conventions for t in c.tests]
    problems: list[CoverageProblem] = []
    cache: dict[Path, set[str] | str] = {}
    for conv in catalog.conventions:
        for ref in conv.tests:
            m = _REF.match(ref)
            if not m:
                problems.append(CoverageProblem(conv.id, ref, "not of the form tests/<file>.py::<test_function>"))
                continue
            target = _resolve(m["file"], tests_root, backend_root)
            if isinstance(target, str):
                problems.append(CoverageProblem(conv.id, ref, target))
                continue
            if target not in cache:
                cache[target] = _top_level_tests(target)
            names = cache[target]
            if isinstance(names, str):
                problems.append(CoverageProblem(conv.id, ref, names))
            elif not m["func"].startswith("test"):
                problems.append(CoverageProblem(conv.id, ref, "not a test function name (must start with 'test')"))
            elif m["func"] not in names:
                problems.append(CoverageProblem(conv.id, ref, "no top-level test function with that name"))
    return problems
