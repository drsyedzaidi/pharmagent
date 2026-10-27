"""Errors raised by the workflow-contract layer."""
from __future__ import annotations


class ContractError(ValueError):
    """A workflow record is malformed, inconsistent, or cannot be read."""
