"""Hash-chain audit trail.

Every tool invocation appends an entry whose hash incorporates the previous
entry's hash. New entries use the v2 canonical form — every semantic field,
including the version itself, encoded as canonical JSON (sorted keys, compact
separators, ASCII-escaped) and hashed:

    entry_hash = SHA256(canonical_json({hash_version, index, prev_hash, timestamp,
                        agent, tool, action, inputs_hash, outputs_hash, actor,
                        reason}))

JSON is used rather than a delimiter join because `actor` and `reason` are
user-supplied: any raw separator can be injected to shift a field boundary and
collide two different entries onto one digest. JSON escapes control characters,
so the encoding is injective.

Any modification to an intermediate entry invalidates every subsequent hash,
giving verifiable computational traceability. ``hash_version=1`` reproduces the
original undelimited concatenation (no index/action) so chains persisted before
versioning still verify — but only in the DEGRADED sense reported by
verify_status(), since a v1 entry's approved/rejected action is not bound.
A version outside SUPPORTED_HASH_VERSIONS is unverifiable and fails closed.

TWO LIMITS, both deliberate and worth stating plainly:

1. Traceability is not correctness. The chain proves *what was run and that the
   log was not altered*, not that the analysis was scientifically right.
2. The digest is unkeyed (plain SHA-256, no HMAC) and the head is not anchored
   anywhere outside the row that stores the chain. It is therefore tamper-
   EVIDENT against edits made through the application, and against a partial or
   careless edit of the stored JSON — but an actor who can write the audit
   column directly can recompute every hash forward with this same public
   algorithm, or truncate the tail, and the result still verifies. Resisting
   that requires a keyed MAC plus an append-only or externally anchored head,
   which is a deployment/key-management decision, not a code-local one.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

GENESIS = "0" * 64

#: Hash formats this build knows how to recompute. An entry claiming anything else
#: cannot be verified — it is reported as unverifiable rather than assumed valid.
SUPPORTED_HASH_VERSIONS = (1, 2)

#: v1 does not bind `index` or `action`, so a legacy entry's approved/rejected
#: e-signature can be flipped without breaking its hash. Chains containing v1
#: entries verify only in the DEGRADED sense — see AuditChain.verify_status.
DEGRADED_HASH_VERSIONS = (1,)


class UnknownHashVersion(ValueError):
    """An entry claims a hash format this build cannot recompute."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_payload(obj) -> str:
    """Stable SHA-256 of an arbitrary JSON-serializable payload."""
    return _sha256(json.dumps(obj, sort_keys=True, default=str))


@dataclass
class AuditEntry:
    index: int
    timestamp: str
    agent: str
    tool: str
    action: str
    inputs_hash: str
    outputs_hash: str
    prev_hash: str
    entry_hash: str = ""
    actor: str = ""          # authenticated identity that triggered the entry
    reason: str = ""         # reason-for-change / approval note (Part 11)
    hash_version: int = 1    # 1 = legacy format; 2 additionally binds index + action

    def compute_hash(self) -> str:
        if self.hash_version == 2:
            # v2 canonical form: EVERY semantic field, encoded as canonical JSON.
            # Encoding matters as much as coverage. A plain concatenation lets any
            # two adjacent variable-length fields be re-split for a byte-identical
            # digest; a raw separator only narrows that to fields containing the
            # separator, and `actor`/`reason` are user-supplied through the API, so
            # actor="a"/reason="b\x1fc" and actor="a\x1fb"/reason="c" would still
            # collide. JSON escapes every control character, so the encoding is
            # injective. `hash_version` is inside the payload, so the version itself
            # cannot be re-labelled without breaking the hash; `index` and `action`
            # make reordering and flipping the human-review approved/rejected
            # e-signature tamper-evident.
            return _sha256(json.dumps({
                "hash_version": 2,
                "index": self.index,
                "prev_hash": self.prev_hash,
                "timestamp": self.timestamp,
                "agent": self.agent,
                "tool": self.tool,
                "action": self.action,
                "inputs_hash": self.inputs_hash,
                "outputs_hash": self.outputs_hash,
                "actor": self.actor,
                "reason": self.reason,
            }, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
        if self.hash_version != 1:
            raise UnknownHashVersion(
                f"unsupported audit hash_version: {self.hash_version!r}")
        # v1 (legacy): the original undelimited concatenation, kept byte-exact so
        # chains persisted before versioning still verify. Never emitted for new
        # entries — AuditChain.append() stamps hash_version=2.
        return _sha256(
            self.prev_hash
            + self.timestamp
            + self.agent
            + self.tool
            + self.inputs_hash
            + self.outputs_hash
            + self.actor
            + self.reason
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditChain:
    entries: list[AuditEntry] = field(default_factory=list)

    @property
    def head(self) -> str:
        return self.entries[-1].entry_hash if self.entries else GENESIS

    def append(
        self,
        *,
        agent: str,
        tool: str,
        action: str,
        inputs,
        outputs,
        timestamp: str,
        actor: str = "",
        reason: str = "",
    ) -> AuditEntry:
        entry = AuditEntry(
            index=len(self.entries),
            timestamp=timestamp,
            agent=agent,
            tool=tool,
            action=action,
            inputs_hash=hash_payload(inputs),
            outputs_hash=hash_payload(outputs),
            prev_hash=self.head,
            actor=actor or "",
            reason=reason or "",
            hash_version=2,      # new entries bind index + action (tamper-evident)
        )
        entry.entry_hash = entry.compute_hash()
        self.entries.append(entry)
        return entry

    def verify(self) -> bool:
        """Recompute the chain; True iff every entry is intact AND recomputable.

        This is the strict answer. It does NOT distinguish a chain of fully-bound
        v2 entries from one containing legacy v1 entries whose action can still be
        flipped undetectably — call verify_status() for that.
        """
        return self.verify_status()["ok"]

    def verify_status(self) -> dict:
        """Verification with its caveats attached.

        ``ok``          — every entry recomputes and the prev-links hold.
        ``degraded``    — ok, but some entries use a format that does not bind all
                          semantic fields (v1: no index/action), so "verified" is
                          weaker than it looks for those rows.
        ``unverifiable``— entries claiming a hash format this build cannot
                          recompute. These make ``ok`` False: an unrecognised
                          version must never be read as a pass.
        """
        prev, ok, degraded, unverifiable = GENESIS, True, 0, 0
        for e in self.entries:
            if e.hash_version not in SUPPORTED_HASH_VERSIONS:
                ok, unverifiable = False, unverifiable + 1
                break
            try:
                digest = e.compute_hash()
            except (UnknownHashVersion, TypeError, ValueError):
                # a malformed/injected field (e.g. a null action) is not a pass
                ok, unverifiable = False, unverifiable + 1
                break
            if e.prev_hash != prev or digest != e.entry_hash:
                ok = False
                break
            if e.hash_version in DEGRADED_HASH_VERSIONS:
                degraded += 1
            prev = e.entry_hash
        return {"ok": ok, "degraded": ok and degraded > 0,
                "legacy_entries": degraded, "unverifiable_entries": unverifiable,
                "n_entries": len(self.entries)}

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]

    @classmethod
    def from_list(cls, entries: list[dict]) -> AuditChain:
        """Rebuild a chain from persisted entry dicts (e.g. loaded from the DB).

        Persisted JSON is untrusted input: it is read at startup for EVERY session,
        so one malformed row must not take the process down. Unknown keys are
        dropped and hash_version is coerced, rather than reaching
        ``AuditEntry(**e)`` and raising out of Orchestrator.__init__. A row that
        cannot be rebuilt at all is not silently skipped — dropping entries would
        forge a shorter valid chain — so it is kept as a placeholder whose hash
        cannot verify, making the damage visible instead.
        """
        str_fields = {"timestamp", "agent", "tool", "action", "inputs_hash",
                      "outputs_hash", "prev_hash", "entry_hash", "actor", "reason"}
        fields = {f.name for f in dataclasses.fields(AuditEntry)}
        chain = cls()

        def _corrupt() -> AuditEntry:
            return AuditEntry(
                index=len(chain.entries), timestamp="", agent="", tool="",
                action="unreadable_entry", inputs_hash="", outputs_hash="",
                prev_hash="", entry_hash="corrupt")

        for e in entries or []:
            if not isinstance(e, Mapping):    # e.g. a bare string or null in the JSON
                chain.entries.append(_corrupt())
                continue
            kw = {k: v for k, v in e.items() if k in fields}
            # Coerce field types rather than trusting them: a null/int `action` would
            # otherwise blow up inside compute_hash, turning one bad row into a failed
            # startup or a 500 on every audit read.
            for k in str_fields & kw.keys():
                if kw[k] is None:
                    kw[k] = ""
                elif not isinstance(kw[k], str):
                    kw[k] = str(kw[k])
            for k in ("index", "hash_version"):
                if k in kw:
                    try:
                        kw[k] = int(kw[k])
                    except (TypeError, ValueError):
                        kw[k] = 0 if k == "index" else 1
            try:
                chain.entries.append(AuditEntry(**kw))
            except TypeError:  # required field missing — keep it visible
                chain.entries.append(_corrupt())
        return chain
