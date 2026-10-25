"""Reduce review-finding ids to their kind, so no subject id reaches an LLM prompt.

The adversarial reviewer (``app.compute.adversarial``) names a per-subject finding
``<prefix><subject id>``, for example ``recompute-cmax-1771``. The subject id is whatever
the dataset calls the subject (digits, letters, hyphens), so it must never be sent to a
hosted model. ``finding_kind`` keeps the part that says WHAT was found and drops WHO it
was found for; the finding counts already say how many.

The reduction is a prefix allow-list that mirrors what the engine emits, not a pattern on
the subject id, so it does not depend on what a subject id looks like. A drift test fails
if the engine gains a per-subject family this list does not know. An id in one of the
subject-capable namespaces that the list does not know is reduced to the namespace
(``nca-*``): failing closed costs a little detail, never a subject id.
"""
from __future__ import annotations

#: Prefixes the review engine puts in front of a SUBJECT id (``f"<prefix>{sid}"`` in adversarial.py).
SUBJECT_FINDING_PREFIXES: tuple[str, ...] = (
    "nca-aucorder-", "nca-extrap-", "nca-clf-sign-", "nca-clf-mag-", "nca-vzf-", "nca-thalf-",
    "nca-clf-consist-", "recompute-cmax-", "recompute-aucband-",
)
#: Families in the same namespaces whose tail is not a subject (``nca-monotonic-<dose>-<dose>``).
_SUBJECT_FREE_PREFIXES: tuple[str, ...] = ("nca-monotonic-",)
#: Namespaces that can carry a subject id; an unrecognised id in one of them fails closed.
_SUBJECT_CAPABLE_NAMESPACES: tuple[str, ...] = ("nca-", "recompute-")
_KINDS = frozenset(p[:-1] for p in SUBJECT_FINDING_PREFIXES)


def finding_kind(finding_id: str) -> str:
    """``recompute-cmax-1771`` -> ``recompute-cmax``. Ids that carry no subject are returned unchanged;
    an unknown id in a subject-capable namespace becomes ``<namespace>*``. Idempotent."""
    if finding_id in _KINDS:
        return finding_id
    for prefix in SUBJECT_FINDING_PREFIXES:
        if finding_id.startswith(prefix):
            return prefix[:-1]
    if finding_id.startswith(_SUBJECT_FREE_PREFIXES):
        return finding_id
    for namespace in _SUBJECT_CAPABLE_NAMESPACES:
        if finding_id.startswith(namespace):
            return f"{namespace}*"
    return finding_id
