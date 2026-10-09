"""The AST scanners of the hygiene guards (r7 round 3, extended in round 4): a hand-typed character set or terminal, and
a literal terminal inside a regex fragment.

``hits(path)``: (line, kind, literal) of every ``frozenset``/``set`` over characters, ``in "xy"``, a Compare against a
one-character string, a one-character tuple or set display, a ``startswith`` / ``endswith`` of one character, and a
``strip`` / ``rstrip`` / ``lstrip`` whose argument is built from string constants (a character set). r7 round 4: the
strip check reads the characters the call strips: ``gaps.strip(text)`` (the GAP set by construction) and a constant
inside a subscript or a call of the argument (a dict key: ``gaps.strip(rec["value"])``) are no character set.

``regex_terminals(path)``: (line, the literal rest) of every string constant that builds a regex (an argument of a
``re`` call, or a module-level string expression that holds a ``char_class`` / ``re.escape`` / ``_alt`` call, a
``re`` call or a name already known to build one) once its regex syntax is taken out: groups, alternation,
quantifiers, anchors, the shorthand classes, the digit ranges ``[0-9]`` and ``[1-9]``, and the digits of a numeral.
What remains (an escaped bracket or caret, a slash, '=', a word such as 'per') is a terminal typed into the regex.
"""
from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

_STRIPS = ("strip", "rstrip", "lstrip")
_SYNTAX = re.compile(r"\(\?(?::|=|!|<=|<!|P<[a-z_]+>)|\[0-9\]|\[1-9\]|\{[0-9]+(?:,[0-9]*)?\}|\\[ZbBsSdDwWtrn]|[()|?*+.]|"
                     r"^\^|\$$|[0-9]")
_BUILDERS = ("char_class", "escape", "_alt")


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text("utf-8"))


def _pieces(node: ast.AST) -> Iterator[ast.Constant]:
    """The string constants of a string-building expression (not inside a subscript, a call or an attribute)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node
    elif isinstance(node, ast.BinOp):
        yield from _pieces(node.left)
        yield from _pieces(node.right)
    elif isinstance(node, ast.JoinedStr):
        for v in node.values:
            yield from _pieces(v)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for e in node.elts:
            yield from _pieces(e)


def _call_name(fn: ast.AST) -> str:
    return fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")


def _gaps_strip(fn: ast.AST) -> bool:
    return isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "gaps"


def _call_hits(n: ast.Call) -> list[tuple[int, str, str]]:
    name = _call_name(n.func)
    out = []
    if name in ("frozenset", "set"):
        for a in n.args:
            for c in ([a] if isinstance(a, ast.Constant) else getattr(a, "elts", [])):
                if isinstance(c, ast.Constant) and isinstance(c.value, str) and (c is a or len(c.value) == 1):
                    out.append((n.lineno, name, c.value))
    if name in ("startswith", "endswith"):
        out += [(n.lineno, name, c.value) for a in n.args for c in _pieces(a) if len(c.value) == 1]
    if name in _STRIPS and not _gaps_strip(n.func):
        out += [(n.lineno, name, c.value) for a in n.args for c in _pieces(a) if c.value]
    return out


def _compare_hits(n: ast.Compare) -> list[tuple[int, str, str]]:
    out = []
    for side in (n.left, *n.comparators):
        if isinstance(side, ast.Constant) and isinstance(side.value, str) and (
                len(side.value) == 1 or (len(side.value) > 1 and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops)
                                         and side is not n.left)):
            out.append((n.lineno, "char-literal", side.value))
        if isinstance(side, (ast.Tuple, ast.Set, ast.List)):
            chars = sorted(e.value for e in side.elts
                           if isinstance(e, ast.Constant) and isinstance(e.value, str) and len(e.value) == 1)
            if chars:                                    # one hit per display: its one-character elements
                out.append((n.lineno, "char-tuple", "".join(chars)))
    return out


def hits(path: Path) -> list[tuple[int, str, str]]:
    """(line, kind, literal) of every hand-typed character set or one-character terminal (see the module docstring)."""
    out: list[tuple[int, str, str]] = []
    for n in ast.walk(_tree(path)):
        if isinstance(n, ast.Call):
            out += _call_hits(n)
        if isinstance(n, ast.Set):
            out += [(n.lineno, "char-set-display", e.value) for e in n.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str) and len(e.value) == 1]
        if isinstance(n, ast.Compare):
            out += _compare_hits(n)
    return out


def _builds_regex(node: ast.AST, marked: set[str]) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and (_call_name(n.func) in _BUILDERS or (
                isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "re")):
            return True
        if isinstance(n, ast.Name) and n.id in marked:
            return True
    return False


def regex_terminals(path: Path) -> list[tuple[int, str]]:
    """(line, literal rest) of every regex fragment that types a terminal (see the module docstring)."""
    tree = _tree(path)
    assigns = [a for a in tree.body if isinstance(a, ast.Assign) and len(a.targets) == 1
               and isinstance(a.targets[0], ast.Name) and isinstance(a.value, (ast.BinOp, ast.Constant, ast.Call))]
    marked: set[str] = set()
    changed = True
    while changed:
        changed = False
        for a in assigns:
            if a.targets[0].id not in marked and _builds_regex(a.value, marked):
                marked.add(a.targets[0].id)
                changed = True
    consts = [c for a in assigns if a.targets[0].id in marked for c in _pieces(a.value)]
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                and n.func.value.id == "re":
            consts += [c for a in n.args for c in _pieces(a)]
    found = {(c.lineno, _SYNTAX.sub("", c.value)) for c in consts}
    return sorted((line, rest) for line, rest in found if rest)


# ------------------------------------------------------------------------------------------- named exemptions ---
# the modules that ARE the lists: the four permitted lists and the one equivalents table, and the word lists
LIST_MODULES = {"numgrammar/alphabet.py": "the four permitted lists and every role set and terminal of the reader",
                "numgrammar/equivalents.py": "the one table of approved equivalents (and the claim-mode group space)",
                "word_lists.py": "whole WORDS of the flag checks, the header keys and the dose rule ('k', 'n', 'p' are "
                                 "one-letter words, never a character set)"}
# (module, literal) -> why a hand-typed literal there never classifies a character of a number
NAMED_LITERALS = {
    ("extract_rules.py", "_"): "the template's word joiner in a header key ('Cmax_ss'): template syntax",
    ("image_routes.py", "12"): "the zoom query values of the page-image route ('1' or '2'), never page text",
    ("parse_guard.py", "h"): "pdfplumber's edge orientation ('h' horizontal) of a ruling, never page text",
    ("qc_flags.py", "n"): "a template field named 'n' (the sample size), never page text",
    ("qc_flags.py", " _"): "the joiners of a template's bound / interval field pair ('Cmax_lower'), never page text",
    ("screen_rules.py", "s"): "the plural stemming of a screening keyword ('rats' -> 'rat'), never a value",
}
# (module, regex rest) -> why a regex there types its terminals: prose, file names and identifiers, never a number
NAMED_REGEX = {
    ("config_tools.py", "[A-Za-z_][A-Za-z-_]"): "the identifier shape of a template field name",
    ("export_tools.py", "[;,][=\\-@]"): "the CSV formula-injection guard of an exported cell",
    ("extract_rules.py", "doseesedingageadminist"): "the dose-context prose pattern ('dose', 'administered')",
    ("ingest.py", "[^A-Za-z-_-]"): "file-name sanitising of an upload",
    ("ingest.py", "[A-Za-z-]"): "file-name sanitising of an upload",
    ("parse_tables.py", "Table"): "the caption word of a table ('Table 2')",
    ("parse_tables.py", "TableFigureFig\\"): "the caption words that end a layout table ('Figure 3', 'Fig. 3')",
    ("prompts.py", "</document[^<>]>"): "the document delimiter of the model prompt",
    ("proposal_check.py", "ignoredisregardoverrideforget[^]instructionsrulespromptsguidelinesapprove "
                          "everythingalleverysystem promptyou are an as an aiautomated readers"):
        "the instruction-text detector (prompt injection), prose only",
    ("qc_flags.py", "[a-z-_]"): "the identifier shape of a model flag code",
    ("qc_flags.py", "^[^a-z]sesdsemcvrse$[^a-z]"): "the dispersion field-name pattern ('Cmax_sd')",
    ("qc_flags.py", "[_]"): "the bound / interval field-name joiner ('Cmax_lower')",
    ("qc_flags.py", "^[_]rangeminmaxminimummaximum$[_]"): "the interval field-name pattern ('AUC_range')",
    ("screen_rules.py", ";"): "the separator of screening alternatives",
    ("screen_rules.py", "or"): "the word that separates screening alternatives",
    ("numgrammar/alphabet.py", "\\cid:\\"): "the pdfminer marker '(cid:N)', one OUTSIDE character",
    ("numgrammar/equivalents.py", " "): "the claim-mode digit group: an ASCII space in thousands position",
}


__all__ = ["LIST_MODULES", "NAMED_LITERALS", "NAMED_REGEX", "hits", "regex_terminals"]
