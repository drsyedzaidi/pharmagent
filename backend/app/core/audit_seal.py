"""Keyed audit snapshot seals and an out-of-database monotonic anchor.

The public SHA-256 chain in :mod:`app.core.audit` detects accidental/partial
edits, but a writer of the primary SQLite row can recompute it or replay a valid
prefix.  This module closes that threat for enforced deployments:

* an HMAC-SHA-256 seal commits to the complete semantic audit snapshot and
  session identity; and
* a monotonic anchor outside the primary database rejects older valid seals.

The local file anchor is intended for a single-host deployment and must be
placed outside the database writer's trust boundary.  The ``AnchorStore``
protocol is deliberately small so a remote/WORM implementation can replace it.
"""
from __future__ import annotations

import base64
import binascii
import fcntl
import hashlib
import hmac
import json
import os
import re
import stat
import threading
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import IO, Protocol

from app.core.audit import GENESIS, AuditChain

NO_SEAL = "0" * 64
SEAL_VERSION = 1
MAC_ALGORITHM = "HMAC-SHA-256"
_SEAL_DOMAIN = b"pharmagent.audit-seal.v1\x00"
_SNAPSHOT_DOMAIN = b"pharmagent.audit-snapshot.v1\x00"
_KEY_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_UNSET = object()


class AuditSecurityError(RuntimeError):
    """Base class for enforced audit-security failures."""


class AuditConfigurationError(AuditSecurityError):
    """Audit security is enabled but its trust material is invalid."""


class AuditIntegrityError(AuditSecurityError):
    """Persisted audit evidence does not match its keyed seal or anchor."""


class AuditEnrollmentRequired(AuditSecurityError):
    """A pre-seal legacy session needs explicit operator enrollment."""


class AuditAnchorConflict(AuditIntegrityError):
    """The external anchor did not have the expected monotonic predecessor."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AuditIntegrityError("audit data is not canonical JSON") from exc


def audit_snapshot_digest(entries: list[dict]) -> str:
    """Commit to every persisted entry field, including v1-unbound semantics."""
    return hashlib.sha256(_SNAPSHOT_DOMAIN + _canonical_json(entries)).hexdigest()


@dataclass(frozen=True)
class AuditKeyring:
    """Active signing key plus retained verification keys.

    Key material is copied into an immutable mapping and omitted from repr.
    Every key must contain at least 256 bits of entropy.  Key IDs, not keys, are
    persisted in seals so rotations can retain old verification keys.
    """

    active_key_id: str
    keys: Mapping[str, bytes] = field(repr=False)

    def __post_init__(self) -> None:
        copied: dict[str, bytes] = {}
        for key_id, key in self.keys.items():
            if not isinstance(key_id, str) or not _KEY_ID.fullmatch(key_id):
                raise AuditConfigurationError("invalid audit key id")
            if not isinstance(key, bytes) or len(key) < 32:
                raise AuditConfigurationError(
                    f"audit key {key_id!r} must be at least 32 bytes"
                )
            copied[key_id] = bytes(key)
        if self.active_key_id not in copied:
            raise AuditConfigurationError("active audit key id is absent from keyring")
        object.__setattr__(self, "keys", MappingProxyType(copied))

    @classmethod
    def from_base64_json(cls, active_key_id: str, raw: str) -> AuditKeyring:
        """Parse ``{"key-id": "base64..."}`` without ever echoing key values."""
        try:
            encoded = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise AuditConfigurationError("audit keyring must be a JSON object") from exc
        if not isinstance(encoded, dict) or not encoded:
            raise AuditConfigurationError("audit keyring must be a non-empty JSON object")
        decoded: dict[str, bytes] = {}
        for key_id, value in encoded.items():
            if not isinstance(value, str):
                raise AuditConfigurationError("audit key values must be base64 strings")
            try:
                decoded[key_id] = base64.b64decode(value, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise AuditConfigurationError(
                    f"audit key {key_id!r} is not valid base64"
                ) from exc
        return cls(active_key_id=active_key_id, keys=decoded)

    def sign(self, payload: bytes) -> tuple[str, str]:
        key_id = self.active_key_id
        return key_id, hmac.new(self.keys[key_id], payload, hashlib.sha256).hexdigest()

    def verify(self, key_id: str, payload: bytes, candidate: str) -> bool:
        key = self.keys.get(key_id)
        if key is None or not isinstance(candidate, str):
            return False
        expected = hmac.new(key, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, candidate)


@dataclass(frozen=True)
class AuditSeal:
    seal_version: int
    mac_algorithm: str
    installation_id: str
    session_id: str
    owner: str
    created_at: str
    generation: int
    entry_count: int
    entry_head: str
    semantic_digest: str
    previous_seal_mac: str
    key_id: str
    baseline: str
    legacy_entry_count: int
    seal_mac: str = ""

    def payload(self) -> dict:
        value = asdict(self)
        value.pop("seal_mac")
        return value

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping | None) -> AuditSeal:
        if not isinstance(value, Mapping):
            raise AuditIntegrityError("audit seal is missing or malformed")
        expected = {item.name for item in fields(cls)}
        if set(value) != expected:
            raise AuditIntegrityError("audit seal has an unsupported schema")
        try:
            seal = cls(**dict(value))
        except TypeError as exc:
            raise AuditIntegrityError("audit seal has invalid fields") from exc
        string_values = (
            seal.mac_algorithm,
            seal.installation_id,
            seal.session_id,
            seal.owner,
            seal.created_at,
            seal.entry_head,
            seal.semantic_digest,
            seal.previous_seal_mac,
            seal.key_id,
            seal.baseline,
            seal.seal_mac,
        )
        if (
            seal.seal_version != SEAL_VERSION
            or seal.mac_algorithm != MAC_ALGORITHM
            or not all(isinstance(item, str) for item in string_values)
            or type(seal.generation) is not int
            or seal.generation < 1
            or type(seal.entry_count) is not int
            or seal.entry_count < 0
            or type(seal.legacy_entry_count) is not int
            or not 0 <= seal.legacy_entry_count <= seal.entry_count
            or not _KEY_ID.fullmatch(seal.key_id)
            or not _HEX64.fullmatch(seal.entry_head)
            or not _HEX64.fullmatch(seal.semantic_digest)
            or not _HEX64.fullmatch(seal.previous_seal_mac)
            or not _HEX64.fullmatch(seal.seal_mac)
            or seal.baseline not in {"fresh", "operator_attested_legacy"}
        ):
            raise AuditIntegrityError("audit seal contains unsupported values")
        return seal


class AnchorStore(Protocol):
    installation_id: str

    def head(self, session_id: str) -> AuditSeal | None: ...

    def list_heads(self) -> dict[str, AuditSeal]: ...

    def history(self, session_id: str) -> tuple[AuditSeal, ...]: ...

    def advance(self, expected_mac: str, seal: AuditSeal) -> None: ...


class MemoryAnchorStore:
    """Thread-safe monotonic anchor used by tests and embedded deployments."""

    def __init__(self, installation_id: str | None = None) -> None:
        self.installation_id = installation_id or f"inst_{uuid.uuid4().hex}"
        self._heads: dict[str, AuditSeal] = {}
        self._history: dict[str, tuple[AuditSeal, ...]] = {}
        self._lock = threading.Lock()

    def head(self, session_id: str) -> AuditSeal | None:
        with self._lock:
            return self._heads.get(session_id)

    def list_heads(self) -> dict[str, AuditSeal]:
        with self._lock:
            return dict(self._heads)

    def history(self, session_id: str) -> tuple[AuditSeal, ...]:
        with self._lock:
            return self._history.get(session_id, ())

    def advance(self, expected_mac: str, seal: AuditSeal) -> None:
        with self._lock:
            current = self._heads.get(seal.session_id)
            if current is not None and hmac.compare_digest(current.seal_mac, seal.seal_mac):
                return  # idempotent retry after an uncertain response
            current_mac = current.seal_mac if current is not None else NO_SEAL
            current_generation = current.generation if current is not None else 0
            if not hmac.compare_digest(current_mac, expected_mac):
                raise AuditAnchorConflict("audit anchor predecessor changed")
            if (
                seal.previous_seal_mac != current_mac
                or seal.generation != current_generation + 1
                or seal.installation_id != self.installation_id
            ):
                raise AuditAnchorConflict("audit anchor successor is not monotonic")
            self._heads = {**self._heads, seal.session_id: seal}
            self._history = {
                **self._history,
                seal.session_id: (*self._history.get(seal.session_id, ()), seal),
            }


class FileAnchorStore:
    """Append-only, fsync'd single-host anchor journal.

    The journal and its lock file are required to be owner-only.  Put the path
    on a separately permissioned volume; co-locating it with a DB-writable
    directory does not create an independent trust boundary.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock_path = self.path.with_name(self.path.name + ".lock")
        self._ensure_file()
        with self._locked() as handle:
            installation_id, _, _ = self._read_locked(handle)
        self.installation_id = installation_id

    def _ensure_file(self) -> None:
        parent_mode = stat.S_IMODE(self.path.parent.stat().st_mode)
        if parent_mode & 0o022:
            raise AuditConfigurationError(
                "audit anchor directory must not be group- or world-writable"
            )
        if not self.path.exists():
            header = {
                "type": "pharmagent_audit_anchor",
                "version": 1,
                "installation_id": f"inst_{uuid.uuid4().hex}",
            }
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            try:
                descriptor = os.open(self.path, flags, 0o600)
            except FileExistsError:
                pass
            else:
                try:
                    os.write(descriptor, _canonical_json(header) + b"\n")
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        mode = stat.S_IMODE(self.path.stat().st_mode)
        if mode & 0o077:
            raise AuditConfigurationError(
                "audit anchor journal must not be accessible by group or others"
            )
        lock_descriptor = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        os.close(lock_descriptor)
        os.chmod(self._lock_path, 0o600)

    @contextmanager
    def _locked(self) -> IO[bytes]:
        with self._lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                with self.path.open("r+b") as journal:
                    yield journal
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _read_locked(
        handle: IO[bytes],
    ) -> tuple[str, dict[str, AuditSeal], dict[str, tuple[AuditSeal, ...]]]:
        handle.seek(0)
        lines = handle.read().splitlines()
        if not lines:
            raise AuditIntegrityError("audit anchor journal is empty")
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise AuditIntegrityError("audit anchor header is corrupt") from exc
        if (
            not isinstance(header, dict)
            or header.get("type") != "pharmagent_audit_anchor"
            or header.get("version") != 1
            or not isinstance(header.get("installation_id"), str)
        ):
            raise AuditIntegrityError("audit anchor header is invalid")
        installation_id = header["installation_id"]
        heads: dict[str, AuditSeal] = {}
        histories: dict[str, tuple[AuditSeal, ...]] = {}
        for raw in lines[1:]:
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise AuditIntegrityError("audit anchor journal is corrupt") from exc
            seal = AuditSeal.from_dict(item)
            previous = heads.get(seal.session_id)
            previous_mac = previous.seal_mac if previous is not None else NO_SEAL
            previous_generation = previous.generation if previous is not None else 0
            if (
                seal.installation_id != installation_id
                or seal.previous_seal_mac != previous_mac
                or seal.generation != previous_generation + 1
            ):
                raise AuditIntegrityError("audit anchor journal is not monotonic")
            heads = {**heads, seal.session_id: seal}
            histories = {
                **histories,
                seal.session_id: (*histories.get(seal.session_id, ()), seal),
            }
        return installation_id, heads, histories

    def head(self, session_id: str) -> AuditSeal | None:
        with self._locked() as handle:
            _, heads, _ = self._read_locked(handle)
        return heads.get(session_id)

    def list_heads(self) -> dict[str, AuditSeal]:
        with self._locked() as handle:
            _, heads, _ = self._read_locked(handle)
        return heads

    def history(self, session_id: str) -> tuple[AuditSeal, ...]:
        with self._locked() as handle:
            _, _, histories = self._read_locked(handle)
        return histories.get(session_id, ())

    def advance(self, expected_mac: str, seal: AuditSeal) -> None:
        with self._locked() as handle:
            installation_id, heads, _ = self._read_locked(handle)
            current = heads.get(seal.session_id)
            if current is not None and hmac.compare_digest(current.seal_mac, seal.seal_mac):
                return
            current_mac = current.seal_mac if current is not None else NO_SEAL
            current_generation = current.generation if current is not None else 0
            if not hmac.compare_digest(current_mac, expected_mac):
                raise AuditAnchorConflict("audit anchor predecessor changed")
            if (
                seal.installation_id != installation_id
                or seal.previous_seal_mac != current_mac
                or seal.generation != current_generation + 1
            ):
                raise AuditAnchorConflict("audit anchor successor is not monotonic")
            handle.seek(0, os.SEEK_END)
            handle.write(_canonical_json(seal.to_dict()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())


@dataclass(frozen=True)
class AuditSecurity:
    keyring: AuditKeyring
    anchor: AnchorStore

    def anchor_head(self, session_id: str) -> AuditSeal | None:
        try:
            return self.anchor.head(session_id)
        except OSError as exc:
            raise AuditIntegrityError("external audit anchor is unavailable") from exc

    def anchor_heads(self) -> dict[str, AuditSeal]:
        try:
            return self.anchor.list_heads()
        except OSError as exc:
            raise AuditIntegrityError("external audit anchor is unavailable") from exc

    def advance(self, expected_mac: str, seal: AuditSeal) -> None:
        try:
            self.anchor.advance(expected_mac, seal)
        except OSError as exc:
            raise AuditIntegrityError("external audit anchor is unavailable") from exc

    def _seal_payload(self, seal: AuditSeal) -> bytes:
        return _SEAL_DOMAIN + _canonical_json(seal.payload())

    def create_seal(
        self,
        *,
        session_id: str,
        owner: str | None,
        created_at: str,
        entries: list[dict],
        previous: AuditSeal | None = None,
        baseline: str = "fresh",
        legacy_entry_count: int = 0,
    ) -> AuditSeal:
        if previous is not None:
            if previous.session_id != session_id:
                raise AuditIntegrityError("audit seal session changed")
            baseline = previous.baseline
            legacy_entry_count = previous.legacy_entry_count
        elif baseline not in {"fresh", "operator_attested_legacy"}:
            raise AuditIntegrityError("unsupported audit seal baseline")
        entry_head = entries[-1].get("entry_hash", GENESIS) if entries else GENESIS
        if not isinstance(entry_head, str):
            raise AuditIntegrityError("audit entry head is malformed")
        unsigned = AuditSeal(
            seal_version=SEAL_VERSION,
            mac_algorithm=MAC_ALGORITHM,
            installation_id=self.anchor.installation_id,
            session_id=session_id,
            owner=owner or "",
            created_at=created_at,
            generation=(previous.generation + 1) if previous is not None else 1,
            entry_count=len(entries),
            entry_head=entry_head,
            semantic_digest=audit_snapshot_digest(entries),
            previous_seal_mac=previous.seal_mac if previous is not None else NO_SEAL,
            key_id=self.keyring.active_key_id,
            baseline=baseline,
            legacy_entry_count=legacy_entry_count,
        )
        key_id, seal_mac = self.keyring.sign(self._seal_payload(unsigned))
        return AuditSeal(**{**unsigned.to_dict(), "key_id": key_id, "seal_mac": seal_mac})

    def _verify_mac(self, seal: AuditSeal) -> bool:
        if seal.installation_id != self.anchor.installation_id:
            return False
        return self.keyring.verify(
            seal.key_id, self._seal_payload(seal), seal.seal_mac
        )

    def _anchor_history_valid(
        self, session_id: str, expected_head: AuditSeal
    ) -> bool:
        previous_mac = NO_SEAL
        previous_generation = 0
        try:
            history = self.anchor.history(session_id)
        except OSError as exc:
            raise AuditIntegrityError("external audit anchor is unavailable") from exc
        for seal in history:
            if (
                seal.session_id != session_id
                or seal.generation != previous_generation + 1
                or not hmac.compare_digest(seal.previous_seal_mac, previous_mac)
                or not self._verify_mac(seal)
            ):
                return False
            previous_mac = seal.seal_mac
            previous_generation = seal.generation
        return bool(
            history
            and hmac.compare_digest(history[-1].seal_mac, expected_head.seal_mac)
        )

    def _verify_snapshot(
        self,
        session_id: str,
        entries: list[dict],
        seal: AuditSeal,
        *,
        owner: str | None | object = _UNSET,
        created_at: str | object = _UNSET,
    ) -> bool:
        head = entries[-1].get("entry_hash", GENESIS) if entries else GENESIS
        return (
            seal.session_id == session_id
            and (owner is _UNSET or seal.owner == (owner or ""))
            and (created_at is _UNSET or seal.created_at == created_at)
            and seal.entry_count == len(entries)
            and seal.entry_head == head
            and hmac.compare_digest(seal.semantic_digest, audit_snapshot_digest(entries))
            and self._verify_mac(seal)
        )

    def assert_current(
        self,
        session_id: str,
        entries: list[dict],
        seal_value: Mapping | None,
        *,
        owner: str | None | object = _UNSET,
        created_at: str | object = _UNSET,
        recover: bool = False,
    ) -> AuditSeal:
        chain_status = AuditChain.from_list(entries).verify_status()
        if not chain_status["ok"]:
            raise AuditIntegrityError("audit hash chain is invalid")
        if seal_value is None:
            if self.anchor_head(session_id) is not None:
                raise AuditIntegrityError("persisted audit seal was removed")
            raise AuditEnrollmentRequired(
                f"session {session_id} requires explicit audit enrollment"
            )
        seal = AuditSeal.from_dict(seal_value)
        if not self._verify_snapshot(
            session_id, entries, seal, owner=owner, created_at=created_at
        ):
            raise AuditIntegrityError("audit snapshot does not match its keyed seal")
        anchored = self.anchor_head(session_id)
        if anchored is None:
            if (
                recover
                and seal.generation == 1
                and hmac.compare_digest(seal.previous_seal_mac, NO_SEAL)
            ):
                self.advance(NO_SEAL, seal)
                return seal
            raise AuditIntegrityError("external audit anchor is missing")
        if not self._anchor_history_valid(session_id, anchored):
            raise AuditIntegrityError("external audit anchor MAC is invalid")
        if hmac.compare_digest(anchored.seal_mac, seal.seal_mac):
            return seal
        if (
            recover
            and seal.generation == anchored.generation + 1
            and hmac.compare_digest(seal.previous_seal_mac, anchored.seal_mac)
        ):
            self.advance(anchored.seal_mac, seal)
            return seal
        raise AuditIntegrityError("database audit seal diverges from external anchor")

    def inspect(
        self,
        session_id: str,
        entries: list[dict],
        seal_value: Mapping | None,
        *,
        owner: str | None | object = _UNSET,
        created_at: str | object = _UNSET,
    ) -> dict:
        chain_status = AuditChain.from_list(entries).verify_status()
        status = {
            **chain_status,
            "chain_ok": chain_status["ok"],
            "mode": "enforced",
            "mac_ok": False,
            "anchor_ok": False,
            "verified": False,
            "baseline": None,
            "generation": None,
            "key_id": None,
            "trusted_since_index": None,
        }
        try:
            seal = AuditSeal.from_dict(seal_value)
        except AuditIntegrityError:
            return status
        mac_ok = self._verify_snapshot(
            session_id, entries, seal, owner=owner, created_at=created_at
        )
        anchored = self.anchor_head(session_id)
        anchor_ok = (
            anchored is not None
            and self._anchor_history_valid(session_id, anchored)
            and hmac.compare_digest(anchored.seal_mac, seal.seal_mac)
        )
        return {
            **status,
            "mac_ok": mac_ok,
            "anchor_ok": anchor_ok,
            "verified": bool(chain_status["ok"] and mac_ok and anchor_ok),
            "baseline": seal.baseline,
            "generation": seal.generation,
            "key_id": seal.key_id,
            "trusted_since_index": seal.legacy_entry_count,
        }


def audit_security_from_settings(config) -> AuditSecurity | None:
    """Build enforced audit security or return explicit hash-only dev mode."""
    mode = getattr(config, "audit_mode", "off")
    configured = (
        getattr(config, "audit_keys_json", None),
        getattr(config, "audit_active_key_id", None),
        getattr(config, "audit_anchor_path", None),
    )
    if mode == "off":
        if getattr(config, "api_token", None):
            raise AuditConfigurationError(
                "authenticated deployments require PHARMAGENT_AUDIT_MODE=enforce"
            )
        if any(value is not None for value in configured):
            raise AuditConfigurationError(
                "audit key/anchor settings require PHARMAGENT_AUDIT_MODE=enforce"
            )
        return None
    if mode != "enforce" or any(value is None for value in configured):
        raise AuditConfigurationError(
            "enforced audit requires keys, active key id, and anchor path"
        )
    secret, active_key_id, anchor_path = configured
    raw = secret.get_secret_value() if hasattr(secret, "get_secret_value") else str(secret)
    keyring = AuditKeyring.from_base64_json(str(active_key_id), raw)
    try:
        anchor = FileAnchorStore(anchor_path)
    except OSError as exc:
        raise AuditConfigurationError(
            "audit anchor path is unavailable"
        ) from exc
    return AuditSecurity(keyring=keyring, anchor=anchor)


def enroll_legacy_session(
    store,
    security: AuditSecurity,
    session_id: str,
    *,
    operator: str,
    timestamp: str,
) -> dict:
    """Explicitly attest and seal one structurally valid pre-#57 session.

    This is deliberately not called during startup.  The operator is asserting
    that the current legacy snapshot is the accepted migration baseline; the
    returned status reports where authenticated history begins and does not
    claim the earlier entries were originally created under HMAC protection.
    """
    if not operator.strip():
        raise AuditConfigurationError("legacy enrollment requires an operator identity")
    row = store.get(session_id)
    if row is None:
        raise KeyError(f"unknown session: {session_id}")
    if row["audit_seal"] is not None or security.anchor_head(session_id) is not None:
        raise AuditIntegrityError("session is already sealed or externally anchored")
    chain = AuditChain.from_list(row["audit"])
    chain_status = chain.verify_status()
    if not chain_status["ok"]:
        raise AuditIntegrityError("cannot enroll a corrupt legacy audit chain")
    legacy_count = len(chain.entries)
    legacy_digest = audit_snapshot_digest(row["audit"])
    chain.append(
        agent="system",
        tool="audit_security",
        action="legacy_snapshot_attested",
        inputs={
            "legacy_entry_count": legacy_count,
            "legacy_semantic_digest": legacy_digest,
        },
        outputs={"audit_mode": "enforce"},
        timestamp=timestamp,
        actor=operator,
        reason="explicit operator enrollment into keyed audit sealing",
    )
    entries = chain.to_list()
    seal = security.create_seal(
        session_id=session_id,
        owner=row["owner"],
        created_at=row["created_at"],
        entries=entries,
        baseline="operator_attested_legacy",
        legacy_entry_count=legacy_count,
    )
    store.save_secured(
        id=session_id,
        owner=row["owner"],
        created_at=row["created_at"],
        updated_at=timestamp,
        state=row["state"],
        audit=entries,
        audit_seal=seal.to_dict(),
        history=row["history"],
        pending=row["pending"],
        params=row["params"],
        dataset_id=row["dataset_id"],
        dataset_path=row["dataset_path"],
        commands=row["commands"],
        expected_audit_json=row["_audit_json_raw"],
        expected_audit_seal_json=None,
    )
    security.advance(NO_SEAL, seal)
    return security.inspect(session_id, entries, seal.to_dict())
