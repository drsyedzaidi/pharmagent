# Workflow orchestration: stage contracts

Status: **Phase A (domain contracts) only.** Design document, not a decision record. Nothing described under "Later
phases" exists yet, and none of it is approved.

## Purpose

Pharmagent has substantial functionality, but each stage hands the next one a different slice of a flat `PharmState`.
Phase A adds the vocabulary that lets one dataset version, its assumptions, executions, findings and human reviews be
described the same way at every stage. It adds records and a read-only projection. It does not execute, persist,
route, sign or approve anything.

## What exists after Phase A

`backend/app/workflow/`:

| Module | Contents |
|---|---|
| `base.py` | `WorkflowContract` (the common record header), `FrozenMap`, `Unknown`, `MigrationInfo`, validated scalar types |
| `contracts.py` | `ArtifactRef`, `DatasetVersion`, `DatasetVersionRef`, `TransformationRecord`, `AssumptionSet`, `ExecutionRecord`, `Finding`, `ReviewRecord` |
| `lineage.py` | `check_lineage`: graph-level checks over dataset versions and transformations |
| `stages.py` | `WorkflowStage` (7 stages), `ReviewGate` (7 gates), allowed transitions, `StageTransition` |
| `events.py` | `WorkflowEvent` (a domain-event **schema** only) and `AuditEntryRef` |
| `compat.py` | Read-only projection of a legacy `PharmState` or serialized session row |
| `hashing.py` | Canonical JSON and SHA-256 |
| `errors.py` | `ContractError`, raised by `validated_copy`, `check_lineage`, the compat projection and `replace`/`model_copy` for unusable input. Constructing, `create` and `load` raise pydantic's `ValidationError` instead |

Tests are in `backend/tests/workflow/` (fixtures are sanitized serialized legacy sessions).

## Record rules

Every record is a frozen Pydantic model with `extra="forbid"` and carries `schema_version`, a stable `id`, `created_at`
(always UTC; naive times are refused), `producer`, `parents` (lineage), `hash_algorithm`, `content_hash` and optional
`migration` metadata.

* **Deep immutability.** Records are frozen, and every collection inside them is immutable: lists are tuples and
  arbitrary mappings are `FrozenMap`, a read-only, serializable mapping that deep-freezes its values on construction. It
  holds JSON-like values only (sets and bytes are refused) and limits nesting to 64 levels however it was assembled.
  Input containers are copied when a record is built, so mutating a caller-held dict or list afterwards cannot change
  the record, and `model_dump()` hands out fresh plain copies. `object.__setattr__`, `model_construct`, pickle,
  `copy.copy` and `copy.deepcopy` (whose pydantic implementations skip validation) can still bypass Python-level
  guards, as can reaching into `FrozenMap` internals; `verify_integrity()` and `validated_copy()` are the backstop for
  that, not the primary mechanism. "Immutable" means immutable through the normal API, not against a deliberate hack.
* `content_hash` is SHA-256 over the canonical JSON (sorted keys, compact separators, ASCII, no NaN/Infinity) of a
  domain-separated payload: `{domain, record_type, schema_version, fields}`, where `fields` is every other field. The
  record type is an explicit, unique `RECORD_TYPE` string declared on each class (a subclass without its own, or with a
  duplicate, cannot even be defined), so renaming a Python class never changes stored hashes. Two record types, or two
  schema versions, therefore never share a hash even if their fields coincide. The canonical form is **Python's**, not
  RFC 8785 (JCS): float text is Python `repr`, object keys sort by code point rather than UTF-16 unit, and integers
  above `sys.get_int_max_str_digits()` digits cannot be hashed, so a non-Python verifier would need to reproduce it.
  Pinned hash vectors cover every record family plus the risky encodings (floats, `-0.0`, a 2**70 integer, non-ASCII
  and astral text, microseconds, a `+05:30` offset, populated maps); ten of them were computed by hand from literal
  JSON outside the model. They fail loudly if the recipe, a field or a serialization rule ever changes; the response is
  a `schema_version` bump and a migration, never a silent re-pin.
* **Sealing has a name; validating never seals.** `Model.create(**fields)` builds a NEW record and seals it, and
  `record.replace(**changes)` makes a new sealed record from an existing one. Every other path (the constructor,
  `model_validate`, `model_validate_json`, `load` and `model_copy`) requires an existing, correct `content_hash` and
  refuses a missing or blank one, so a stored record whose content was altered, or whose hash was stripped, cannot be
  laundered by reading it back. `model_copy(update=...)` can no longer change content (use `replace`). Stored bytes are
  not themselves canonical: a validated record re-serializes to its canonical form, so a store that hashes raw bytes
  would disagree with `content_hash`. The hash is an **unkeyed corruption check, not proof of authenticity**: anyone who can write a record can compute a
  matching hash. Authenticity comes only from the existing keyed audit seal.
* `schema_version` must be the exact integer of a supported version (not `1.0`, `"1"` or `True`), refused in both
  directions. Counts, sizes, seeds and `from_schema_version` are strict integers and `reversible` a strict boolean (no
  `"3"`, `3.0`, `True` or `"no"`); float fields accept numbers only. Timestamps must be a `datetime` or an ISO-8601 string with an explicit offset and at most microsecond
  precision; epoch numbers, naive strings and over-precise fractions are refused rather than coerced or truncated.
* Nested model instances are re-validated when a record is built, so a forged `model_construct` child inside a record
  is refused at construction.
* A value the source never recorded is an explicit `Unknown(status, reason)`. It is never filled in.
* Text fields (labels, messages, and the `reason` and `note` of `Unknown` and `MigrationInfo`) refuse control, format
  (bidi, zero-width), line-separator, private-use and unassigned characters, the Unicode Default_Ignorable ranges
  (variation selectors, tag characters, fillers), a fixed list of other blank-rendering characters (braille blank, Khmer
  inherent vowels), and edge whitespace, and must contain a letter, number, punctuation mark or symbol (marks alone do
  not count). The ignorable ranges are written out by hand, so the set can drift from a future Unicode version;
  confusable (look-alike) letters are not detected, and labels are display text that carries no authority. Validation
  errors hide the offending input in their printed form, but `ValidationError.errors()` still carries it (as does
  FastAPI's default 422 body) unless the caller passes `include_input=False`. Strings inside JSON content must be valid UTF-8 (no lone surrogates), keys included; numbers refuse booleans; JSON objects are capped at 64 KiB and collections at fixed counts. Defaults for
  mapping fields are immutable too (pydantic does not validate defaults, so they are `FrozenMap` already). Equality is
  Python value equality (`1 == 1.0`); `content_hash` is stricter and treats `1` and `1.0`, and `0.0` and `-0.0`, as
  different, so two records that compare equal but carry different numeric spellings have different hashes.
* `ArtifactRef` is content-addressed: its `id` must equal `sha256:<digest>`. Generated code, control streams, logs and
  outputs are all artifacts, so each has a hash that a review can bind to.

## Dataset lineage is a DAG

A `DatasetVersion` has an opaque, stable `id` and zero or more `parents` (other version ids). Its `dataset_id` is the
logical dataset and may differ from its parents' (an analysis dataset built from dosing and observations has a new
identity). `display_version` is an optional number for display; it is never identity and imposes no ordering.

A `TransformationRecord` has one or more `inputs` and one or more `outputs`, each a `DatasetVersionRef` that pins a
version id and that record's content hash. This covers branches from one source, merges, joins and one-to-many outputs.
`check_lineage(versions, transformations)` verifies the supplied set, reading freshly validated copies: no
duplicate version or transformation ids, no missing parents, no cycles, every referenced version present with the
exact recorded hash, every output descended from exactly its transformation's inputs, and no output claimed by two
transformations. Ids are chosen by the caller, so this shows the set is internally consistent, not that it is the
genuine history. A version with parents but no transformation record is allowed, and timestamps are not ordered.

## Stages and transitions

`data_curation → eda → nca → model_development → model_evaluation → simulation → reporting`, with these forward moves
only: EDA may go to NCA or straight to model development; NCA may go to model development or reporting; evaluation may
go to simulation or reporting. Going back (to repair or refit) is a `rollback`, repeating a stage is a `rerun`; both
require a reason. `StageTransition` refuses anything else.

## Review gates

Seven gates: cleaned dataset, NCA exclusions and BLQ handling, base structural model, covariate model, final model,
simulation scenario, final report. A `ReviewRecord` binds a decision to the exact digests of the artifacts reviewed
(an `ArtifactBinding`'s id must be the content address of its digest). `binding_status(current)` reports `current`,
`stale` (only an inconsistent caller map can produce it: binding ids are the content address of the digest, so a changed
artifact has a new id and reports `missing`), `missing`, or `invalid` (the record fails its own integrity check or no longer
re-validates); only an `approved` record with a `current` binding is a current approval. Re-validation catches records
that are structurally invalid even if their hash was recomputed. It does **not** catch a structurally valid forgery
whose hash was recomputed: that needs the keyed audit seal. The decision is made on a freshly validated copy, never the live object, because a `str`, `int` or `datetime`
subclass with a lying comparison could otherwise be smuggled in through `model_construct` plus a rehash. The copy is
rebuilt from the record's JSON form, so every field comes back as an exact built-in type, and a record with a blank
stored hash is refused. Re-validation checks
structure and the content hash. The advisory name lint judges only NEW records (`create` and `replace`); `load`,
`model_validate`, `model_copy` and `validated_copy` never apply it, so tightening the lint later cannot void or make
unreadable approvals already stored (the lint is triggered through a private validation-context key; that is not a
security boundary). Any other validator that is
tightened is a schema change and needs a `schema_version` bump; so does a change to the Unicode categories that
`Label` and `Message` reject. The `current` mapping must be plain strings, is size-capped and read once; anything else raises
`ContractError`. None of this defends against someone who can forge records and compute hashes: that is what the keyed
audit seal is for.

**No authority lives in the record.** `actor_label` is display text. `claimed_actor_kind` is optional and is only what
the submitter said about themselves. Neither affects `binding_status` or `is_current_approval`, and tests prove that.
`authenticated_principal_ref` exists but can only be `None` in Phase A. Legacy reviewer values go in `legacy_metadata`.
A name lint on `actor_label` (after Unicode normalization) refuses labels that identify an automated agent; it is a
lint, not authentication, is easy to evade and will reject a human who shares such a name. Phase D must derive
authority from an authenticated principal and policy evaluation.

## Relationship to the existing audit system

`WorkflowEvent` is a schema. It has no sequence, previous-hash, signature, seal or anchor fields on purpose. It can
carry `source_audit_refs` (`AuditEntryRef`: session id, entry index, entry hash) naming the audit entries it was
derived from, so the audit chain stays the compliance evidence and the event only points at it. `claimed_actor_kind` is a
claim, not evidence, exactly like the same-named field on a review, and an event's `subject_id` and `subject_hash`
carry no record type, so a hash from the wrong record family is not detectable at this layer. `AuditEntryRef` is an
unverified pointer: nothing checks that the entry exists or that the hash is what the audit chain computes. The audit
chain (`core/audit.py`), keyed seals and external anchors (`core/audit_seal.py`) and run provenance
(`core/provenance.py`) remain the only tamper-evident record. A later phase connects workflow events to that system; no
second log is introduced.

## Legacy compatibility

`compat.project_pharmstate(state, as_of=...)` and `project_session_row(row, as_of=...)` are pure: no clock, disk,
database or audit access. The caller supplies `as_of`. They report what a legacy state holds (dataset identity and
digest, columns and roles, subject count, environment id derived from the recorded run provenance, which result fields
are populated per stage). Units, dose-record counts, executions, reviews and assumptions were not recorded in
`PharmState`, so they are `Unknown(unavailable, ...)`. "Stage evidence" says a result field is populated; it says
nothing about completion, validity or approval. The projected dataset has `display_version=None` and an id of the form
`dataset:<dataset_id>:legacy:<first 32 hex of its data digest>` (or `...:legacy:nodigest`), so different data under the
same dataset id never collides: legacy state tracked no versions, so none is invented. Projection ids come from three
disjoint namespaces: `legacy:none`, `legacy:id:<session id>` and `legacy:h:<32 hex of a hash>`. Its `source` is the data file's base name, which
may contain PHI or PII, so treat projected records as sensitive until the Phase D privacy review.

## Required before records are persisted or acted on (Phase D)

Raised by the Phase A security review and deliberately not built here:

* Anchor each `ReviewRecord` hash in the existing audit chain, and read approvals back through it, so authenticity and
  later rejection or invalidation are decided by the audit system and not by the record alone.
* Resolve the reviewer through authentication into `authenticated_principal_ref`, and evaluate policy on that.
* Bind a review to more than artifact hashes: the dataset version, the execution ids, the gate's required artifact
  kinds and minimum set, a link to the review it supersedes, and a server-assigned timestamp. Without these an approval
  can be replayed against a later state with the same digests.
* Enforce that a rollback targets a stage that was actually visited; Phase A checks only that it goes backwards.
* Privacy review of dataset file names carried by the projection, and CSV/formula escaping on every export.

## Later phases (not implemented, not approved)

* **B, coordinator, in shadow mode.** The existing orchestrator stays authoritative. The coordinator never parses
  audit storage and never depends only on a current-state projection. It consumes normalized `WorkflowEvent` objects
  through a `WorkflowEventSource` protocol:

  ```
  WorkflowEventSource  (protocol)
        ↑ implemented by
  LegacyAuditEventAdapter   reads existing audit entries, read-only
        │                   emits WorkflowEvent records that keep AuditEntryRef (session, index, hash)
        ↓
  Shadow Coordinator        records and compares proposed transitions; performs no scientific
                            execution, writes nothing back, is not authoritative
  ```

  Activation needs a separately reviewed guarded integration, because it touches the orchestrator, the review gate and
  provenance.
* **C, stage adapters; D, artifact and approval binding; E, routing; F, backend API; G, UI.** Each is its own
  reviewable change. Routing (E) coordinates with the routing fixes already open as separate PRs.

## Explicit non-goals of Phase A

No change to scientific results, engine defaults, routing, API payloads, persisted sessions, the conventions catalog,
review gates, audit formats or the frontend. No existing production module is modified. The overlapping BE statistical
implementations in other open PRs are untouched and need their own scientific comparison.
