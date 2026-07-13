"""Hash-chain audit trail.

Every tool invocation appends an entry whose hash incorporates the previous
entry's hash:

    entry_hash = SHA256(prev_hash + timestamp + agent + tool + inputs_hash + outputs_hash)

Any modification to an intermediate entry invalidates every subsequent hash,
giving verifiable computational traceability.

NOTE: traceability is not correctness. The chain proves *what was run and that
the log was not altered*, not that the analysis was scientifically right.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

GENESIS = "0" * 64


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

    def compute_hash(self) -> str:
        # actor/reason are appended last; both default to "" so pre-identity
        # chains hash bit-identically (empty-string concat is a no-op) and still
        # verify, while any recorded identity/reason is tamper-evident.
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
        )
        entry.entry_hash = entry.compute_hash()
        self.entries.append(entry)
        return entry

    def verify(self) -> bool:
        """Recompute the chain; return True iff intact."""
        prev = GENESIS
        for e in self.entries:
            if e.prev_hash != prev or e.compute_hash() != e.entry_hash:
                return False
            prev = e.entry_hash
        return True

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]

    @classmethod
    def from_list(cls, entries: list[dict]) -> AuditChain:
        """Rebuild a chain from persisted entry dicts (e.g. loaded from the DB)."""
        chain = cls()
        chain.entries = [AuditEntry(**e) for e in entries]
        return chain
