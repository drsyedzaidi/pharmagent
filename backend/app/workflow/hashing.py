"""Canonical JSON and SHA-256 for workflow records.

The canonical form (sorted keys, compact separators, ASCII-escaped, no NaN/Infinity) is the one the audit seal already
uses, but it is implemented here independently so this layer imports nothing from the guarded audit modules. It is a
content fingerprint for records, not a tamper-evident log: persistence, chaining and sealing belong to the existing
audit system and are not part of this package.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .errors import ContractError


def canonical_json(value: Any) -> bytes:
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        return text.encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        failure = type(exc).__name__
    raise ContractError(f"value is not canonical JSON ({failure})")           # outside the handler: nothing is chained


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
