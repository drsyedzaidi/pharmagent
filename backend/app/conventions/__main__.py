"""CLI: `python -m app.conventions [report] [--format md|json] [--output FILE] [--check] [--catalog FILE]`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from .catalog_io import load_catalog
from .coverage import check_test_coverage
from .render import render_report_json, render_report_markdown

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.conventions", description=__doc__)
    ap.add_argument("command", nargs="?", choices=["report"], default="report")
    ap.add_argument("--format", choices=["md", "json"], default="md")
    ap.add_argument("--output", help="write the report here instead of stdout")
    ap.add_argument("--check", action="store_true", help="validate and check test coverage; exit 1 on any problem")
    ap.add_argument("--catalog", help="catalog file (default: the shipped catalog.json)")
    args = ap.parse_args(argv)
    try:
        catalog = load_catalog(args.catalog)
    except (ValidationError, ValueError, OSError) as exc:
        print(f"catalog invalid: {exc}", file=sys.stderr)
        return 1
    problems = check_test_coverage(catalog, BACKEND_ROOT)
    if args.check:
        for p in problems:
            print(f"{p.convention_id}: {p.ref}: {p.reason}", file=sys.stderr)
        print(f"{len(catalog.conventions)} conventions, {len(problems)} coverage problems")
        return 1 if problems else 0
    text = (render_report_json if args.format == "json" else render_report_markdown)(catalog, problems)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
