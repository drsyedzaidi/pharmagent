# Audit authenticity and external anchoring

PharmAgent has two distinct audit integrity layers:

1. Every entry participates in the versioned public SHA-256 chain in
   `app/core/audit.py`. This catches malformed and partial edits and keeps old
   audit exports readable.
2. Enforced deployments HMAC the complete semantic audit snapshot and advance a
   monotonic anchor outside the primary SQLite database. This prevents a
   database-only writer from recomputing the public chain or replaying a genuine
   older prefix.

`GET /api/sessions/{sid}/audit` reports `verified: true` only when the structural
chain, keyed semantic seal, and current external anchor all agree. The default
keyless development mode reports `mode: hash_only` and `verified: false`.

## Enforced configuration

Set all of:

```text
PHARMAGENT_AUDIT_MODE=enforce
PHARMAGENT_AUDIT_KEYS_JSON={"2026-07":"<base64-encoded-32+-byte-key>"}
PHARMAGENT_AUDIT_ACTIVE_KEY_ID=2026-07
PHARMAGENT_AUDIT_ANCHOR_PATH=/protected/pharmagent/audit-anchor.jsonl
```

The HMAC key must be independent of the bearer API token and contain at least 32
random bytes. The local anchor journal is created owner-only and uses file
locking plus `fsync`, but its path must also be outside the primary database
writer's permissions. A multi-node or regulated deployment should implement the
same `AnchorStore` interface with a remote append-only/WORM service.

When bearer authentication is configured, startup rejects `audit_mode=off`.
Partial audit configuration, weak keys, missing historical keys, an insecure
anchor file mode, or an inconsistent database/anchor state also fail closed.

## Existing databases

Enforced startup never silently authenticates an unsealed legacy row. First
inventory it:

```bash
cd /path/to/pharmagent
PYTHONPATH=backend backend/.venv/bin/python \
  backend/scripts/enroll_audit_security.py
```

After taking a backup and reviewing the inventory, explicitly attest the current
snapshot:

```bash
PYTHONPATH=backend backend/.venv/bin/python \
  backend/scripts/enroll_audit_security.py \
  --apply --operator security-admin@example.org --yes
```

The command structurally verifies each legacy chain, records an
`legacy_snapshot_attested` audit event, HMAC-seals the full snapshot, and anchors
it. The status retains `baseline: operator_attested_legacy` and
`trusted_since_index`; it does not claim that pre-enrollment history was
originally authenticated.

## What the seal binds (and does not)

A seal (`seal_version` 2) is an HMAC over: session id, owner, `created_at`,
generation, the complete semantic audit snapshot (entry count, head, digest),
the previous seal's MAC, key id, baseline, and **`state_digest`** — SHA-256 of
the canonical `sessions.state_json` content. `state_digest` excludes only the
transient `run_provenance_status` / `run_provenance_assurance` fields, which are
recomputed from the audit chain on every read and are never trusted from storage.

Every save re-seals when the state or audit changes; the SQLite write is a
compare-and-swap on the exact audit, seal **and state** rows that were
verified. Loading a session, `assert_current` and `GET .../audit` verify the
state against the seal. Editing `state_json` (e.g. `nca_parameters`,
`be_results`) directly in the database therefore raises `AuditIntegrityError` at
load and reports `verified: false`, `state_binding: "mismatch"`.

Not bound by the seal: `history_json`, `pending_json`, `params_json`,
`commands_json`, `dataset_id`/`dataset_path`, `updated_at`, and the dataset file
contents (the dataset file is covered separately by its recorded SHA-256, checked
at load). Rewriting those rows is not detected by the keyed seal.

`state_binding` values in the audit status: `verified` (v2 seal, state matches),
`absent` (legacy v1 seal, state is **not** covered), `mismatch` (v2 seal,
state differs → `verified: false`).

### LLM attribution

Genesis provenance records the provider/model the session was created under
(`llm_provider`, `llm_model`; sessions created earlier have no such fields). The
model can be switched at runtime for all sessions; each switch appends an
`llm_config_changed` entry (provider, model, base-URL host, actor — never a key)
to every loaded writable session, and every chat turn appends a
`chat_turn(llm=provider:model)` entry. Genesis provenance is never rewritten.

### Seals created before `seal_version` 2

Existing sealed sessions carry v1 seals with no `state_digest`. They keep
loading and verifying exactly as before, but report `state_binding: "absent"`:
their state row is not tamper-evident. The next legitimate save (any tool run,
chat turn, or review decision) writes a v2 seal (generation + 1, linked to the
v1 seal's MAC), after which the state is bound. A v1 seal found in a session
that has already advanced to v2 in the anchor is rejected as a rollback by the
anchor check as before. Until a session is re-sealed this way, treat its state as
unprotected.

## Rotation and recovery

To rotate, add the new key to `AUDIT_KEYS_JSON`, retain every old key, and change
`AUDIT_ACTIVE_KEY_ID`. The next persisted operation creates a successor seal
under the new key. Removing a referenced old key makes verification fail.

SQLite is committed before the external anchor. A crash in that narrow window
may leave exactly one signed successor in the database. Startup can advance only
that HMAC-valid, previous-seal-linked generation. Larger gaps, forks, a database
behind the anchor, whole-row deletion, unknown keys, and seal/snapshot mismatch
fail closed.

## Threat boundary and limitations

This control protects against an actor who can read/write/delete the primary
SQLite database but cannot access the HMAC keys or modify/roll back the external
anchor. It does not protect against compromise of the application process, MAC
keys, and anchor together; host-administrator rollback of a local anchor;
denial-of-service; untrusted wall-clock time; or deliberately false content
signed by compromised application code.

HMAC authenticates the application's record. It is not a per-user digital
signature or cryptographic non-repudiation control.
