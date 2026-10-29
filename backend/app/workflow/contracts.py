"""The canonical analysis records: artifacts, dataset versions, transformations, assumptions, executions, findings and
reviews. All are immutable, versioned and content-hashed (see ``base``)."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    StrictBool,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .base import (
    ColumnName,
    FiniteFloat,
    Frozen,
    FrozenMap,
    Identifier,
    IdTuple,
    JsonObject,
    Label,
    Message,
    NonNegInt,
    Sha256Hex,
    Timestamp,
    Unknown,
    WorkflowContract,
    creation_policy,
    frozen_map_of,
    numbers_only,
    require_unique,
    to_utc,
    unrecorded,
)
from .errors import ContractError
from .hashing import sha256_hex
from .stages import ReviewGate, WorkflowStage

_U = "A-Za-z0-9\u00b5\u03bc%\u00b0/*^().\u00b7\\-"  # characters a unit may contain; interior spaces allowed, edges not
_UNIT = rf"^(?:[{_U}]|[{_U}][{_U} ]{{0,30}}[{_U}])$"
Unit = Annotated[str, StringConstraints(pattern=_UNIT)]
_TOKEN = r"[a-z0-9][a-z0-9!#$&^_.+-]{0,126}"
MediaType = Annotated[str, StringConstraints(pattern=rf"^{_TOKEN}/{_TOKEN}$")]


# --- artifacts ------------------------------------------------------------------------------------
ArtifactKind = Literal["dataset", "code", "control_stream", "log", "output", "plot", "report", "config", "other"]


class ArtifactRef(WorkflowContract):
    """A reference to bytes by their SHA-256 digest. The id *is* the content address, so the same bytes always have the
    same id and different bytes never do; ``parents`` records the artifacts this one was derived from."""
    RECORD_TYPE = "ArtifactRef"
    kind: ArtifactKind
    digest: Sha256Hex
    digest_algorithm: Literal["sha256"] = "sha256"
    media_type: MediaType
    size_bytes: NonNegInt

    @model_validator(mode="after")
    def _content_addressed(self):
        if self.id != f"{self.digest_algorithm}:{self.digest}":
            raise ValueError(f"id must be the content address '{self.digest_algorithm}:<digest>'")
        return self

    @classmethod
    def from_bytes(cls, data: bytes, *, kind: ArtifactKind, media_type: str, producer: str, created_at: datetime,
                   parents: IdTuple = ()) -> ArtifactRef:
        digest = sha256_hex(data)
        return cls.create(id=f"sha256:{digest}", digest=digest, kind=kind, media_type=media_type, size_bytes=len(data),
                   producer=producer, created_at=created_at, parents=parents)


class ArtifactBinding(Frozen):
    """A reviewed artifact, by content address: the id must be ``sha256:<digest>``, so a binding cannot name one
    artifact and carry another's hash."""
    artifact_id: Identifier
    digest: Sha256Hex

    @model_validator(mode="after")
    def _content_addressed(self):
        if self.artifact_id != f"sha256:{self.digest}":
            raise ValueError("artifact_id must be the content address 'sha256:<digest>'")
        return self


# --- datasets -------------------------------------------------------------------------------------
class ColumnSchema(Frozen):
    name: ColumnName
    role: Label | None = None
    dtype: Label | None = None
    unit: Unit | Unknown = Field(default_factory=unrecorded)


class BlqRepresentation(Frozen):
    basis: Literal["cens_flag", "dv_nonpositive", "none"]
    censoring_column: ColumnName | None = None
    lloq: Annotated[float, BeforeValidator(numbers_only), Field(ge=0, allow_inf_nan=False)] | Unknown = Field(
        default_factory=unrecorded)

    @model_validator(mode="after")
    def _column_matches_basis(self):
        if self.basis == "cens_flag" and self.censoring_column is None:
            raise ValueError("a cens_flag basis must name its censoring_column")
        if self.basis != "cens_flag" and self.censoring_column is not None:
            raise ValueError("censoring_column is only valid with the cens_flag basis")
        return self


class DatasetVersion(WorkflowContract):
    RECORD_TYPE = "DatasetVersion"
    dataset_id: Identifier
    display_version: Annotated[int, Field(ge=1, strict=True)] | None = None       # for display only; never identity
    source: Label | Unknown = Field(default_factory=unrecorded)
    data_digest: Sha256Hex | Unknown = Field(default_factory=unrecorded)
    subject_id_column: ColumnName | Unknown = Field(default_factory=unrecorded)
    columns: Annotated[tuple[ColumnSchema, ...], Field(max_length=2000)] = ()
    n_subjects: NonNegInt | Unknown = Field(default_factory=unrecorded)
    n_dose_records: NonNegInt | Unknown = Field(default_factory=unrecorded)
    n_observation_records: NonNegInt | Unknown = Field(default_factory=unrecorded)
    blq: BlqRepresentation | Unknown = Field(default_factory=unrecorded)

    @model_validator(mode="after")
    def _columns(self):
        names = tuple(c.name for c in self.columns)
        require_unique(names, "column names")
        if names and isinstance(self.subject_id_column, str) and self.subject_id_column not in names:
            raise ValueError("subject_id_column is not among the listed columns")
        return self


class DatasetVersionRef(Frozen):
    """Pins one exact dataset-version record: its opaque id and that record's content hash."""
    version_id: Identifier
    record_hash: Sha256Hex

    @classmethod
    def to(cls, version: DatasetVersion) -> DatasetVersionRef:
        return cls(version_id=version.id, record_hash=version.content_hash)


RefTuple = Annotated[tuple[DatasetVersionRef, ...], Field(min_length=1, max_length=100)]


class TransformationRecord(WorkflowContract):
    """One or more input dataset versions become one or more output versions. ``parents`` must be exactly the inputs'
    version ids; ``check_lineage`` verifies, against the versions, that each output descends from them."""
    RECORD_TYPE = "TransformationRecord"
    inputs: RefTuple
    outputs: RefTuple
    operation: Label
    parameters: JsonObject = Field(default_factory=FrozenMap)
    actor: Label | Unknown = Field(default_factory=unrecorded)
    reason: Message | Unknown = Field(default_factory=unrecorded)
    affected_rows: NonNegInt | Unknown = Field(default_factory=unrecorded)
    reversible: StrictBool | Unknown = Field(default_factory=unrecorded)
    code_artifact_id: Identifier | None = None

    @model_validator(mode="after")
    def _links(self):
        ins, outs = [r.version_id for r in self.inputs], [r.version_id for r in self.outputs]
        require_unique(tuple(ins), "input version ids")
        require_unique(tuple(outs), "output version ids")
        if set(ins) & set(outs):
            raise ValueError("a version cannot be both an input and an output")
        if set(self.parents) != set(ins):
            raise ValueError("parents must be exactly the input version ids")
        return self


# --- scientific assumptions -----------------------------------------------------------------------
class Bounds(Frozen):
    lower: FiniteFloat | None = None
    upper: FiniteFloat | None = None

    @model_validator(mode="after")
    def _ordered(self):
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("lower bound exceeds upper bound")
        return self


Estimates = frozen_map_of(Label, FiniteFloat)
BoundsMap = frozen_map_of(Label, Bounds)
RuleVersions = frozen_map_of(Label, Label)
Route = Literal["iv_bolus", "iv_infusion", "oral", "subcutaneous", "other"]


class AssumptionSet(WorkflowContract):
    RECORD_TYPE = "AssumptionSet"
    route: Route | Unknown = Field(default_factory=unrecorded)
    dose_regimen: Message | Unknown = Field(default_factory=unrecorded)
    analyte: Label | Unknown = Field(default_factory=unrecorded)
    matrix: Label | Unknown = Field(default_factory=unrecorded)
    blq_policy: Message | Unknown = Field(default_factory=unrecorded)
    exclusion_policy: Message | Unknown = Field(default_factory=unrecorded)
    missing_data_policy: Message | Unknown = Field(default_factory=unrecorded)
    model_assumptions: Annotated[tuple[Message, ...], Field(max_length=100)] | Unknown = Field(
        default_factory=unrecorded)
    parameterization: Message | Unknown = Field(default_factory=unrecorded)
    initial_estimates: Estimates | Unknown = Field(default_factory=unrecorded)
    bounds: BoundsMap | Unknown = Field(default_factory=unrecorded)
    covariate_assumptions: Annotated[tuple[Message, ...], Field(max_length=100)] | Unknown = Field(
        default_factory=unrecorded)

    @model_validator(mode="after")
    def _estimates_inside_bounds(self):
        if isinstance(self.initial_estimates, Mapping) and isinstance(self.bounds, Mapping):
            for name, value in self.initial_estimates.items():
                b = self.bounds.get(name)
                if b and ((b.lower is not None and value < b.lower) or (b.upper is not None and value > b.upper)):
                    raise ValueError("an initial estimate is outside its bounds")
        return self


MAX_CURRENT_ARTIFACTS = 10_000


def _snapshot_current(current: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(current, Mapping):
        raise ContractError("current artifacts must be a mapping of artifact id to digest")
    snapshot: dict[str, str] = {}
    try:
        for count, key in enumerate(current):
            if count >= MAX_CURRENT_ARTIFACTS:
                raise ContractError("too many current artifacts")
            value = current[key]
            if type(key) is not str or type(value) is not str:
                raise ContractError("current artifacts must map plain strings to plain strings")
            snapshot[key] = value
    except ContractError:
        raise
    except Exception:
        snapshot = None
    if snapshot is None:                               # raised outside the handler: the caller's exception is not kept
        raise ContractError("current artifacts could not be read")
    return snapshot


# --- execution, findings, review ------------------------------------------------------------------
TERMINAL = frozenset({"succeeded", "failed", "skipped"})


class ExecutionRecord(WorkflowContract):
    RECORD_TYPE = "ExecutionRecord"
    stage: WorkflowStage
    tool: Label
    engine: Label | Unknown = Field(default_factory=unrecorded)
    engine_version: Label | Unknown = Field(default_factory=unrecorded)
    environment_id: Label | Unknown = Field(default_factory=unrecorded)
    code_artifact_ids: IdTuple = ()
    input_artifact_ids: IdTuple = ()
    output_artifact_ids: IdTuple = ()
    log_artifact_ids: IdTuple = ()
    config: JsonObject = Field(default_factory=FrozenMap)
    seed: Annotated[int, Field(strict=True)] | Unknown = Field(default_factory=unrecorded)
    warnings: Annotated[tuple[Message, ...], Field(max_length=100)] = ()
    status: Literal["pending", "running", "succeeded", "failed", "skipped"]
    started_at: Timestamp
    finished_at: Timestamp | None = None

    _norm_times = field_validator("started_at", "finished_at")(lambda v: None if v is None else to_utc(v))

    @field_validator("code_artifact_ids", "input_artifact_ids", "output_artifact_ids", "log_artifact_ids")
    @classmethod
    def _unique_ids(cls, v: IdTuple) -> IdTuple:
        require_unique(v, "artifact ids")
        return v

    @model_validator(mode="after")
    def _status_and_times(self):
        if self.status in TERMINAL and self.finished_at is None:
            raise ValueError(f"finished_at is required once status is {self.status}")
        if self.status not in TERMINAL and self.finished_at is not None:
            raise ValueError(f"finished_at must be empty while status is {self.status}")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at is before started_at")
        return self


class Finding(WorkflowContract):
    RECORD_TYPE = "Finding"
    stage: WorkflowStage
    severity: Literal["info", "warning", "error", "blocking"]
    code: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{1,100}$")]
    message: Message
    source: Literal["automated", "scientist"]
    artifact_ids: IdTuple = ()
    rule_version: Label | None = None


#: Names that identify an automated author. A LINT on the display label, not authentication: it is easy to evade and it
#: will reject a human who shares such a name. Authority comes only from an authenticated principal (Phase D).
AUTOMATED_NAME_WORDS = frozenset({"assistant", "llm", "bot", "agent"})              # matched as whole words
AUTOMATED_NAME_BRANDS = ("codex", "claude", "gemini", "chatgpt", "copilot", "openai", "anthropic",
                         "mockllm")                                                 # matched anywhere in the name
_GPT_TOKEN = re.compile(r"^gpt(\d\w*)?$")                                           # gpt, gpt4, gpt5o; not 'egypt'


def _label_lint(value: str, info: ValidationInfo) -> str:
    """Advisory lint on a reviewer label. It runs only under the creation policy, so it judges a NEW label, or one that
    ``replace()`` changes; a stored label is never re-linted by loading, validating, or an unrelated ``replace()``."""
    if not creation_policy(info):
        return value
    text = "".join(c for c in unicodedata.normalize("NFKC", value) if unicodedata.category(c) != "Cf").casefold()
    words = set(re.split(r"[^a-z0-9]+", text))
    if (AUTOMATED_NAME_WORDS & words or any(b in text for b in AUTOMATED_NAME_BRANDS)
            or any(_GPT_TOKEN.fullmatch(w) for w in words)):
        raise ValueError("label must name a person, not an automated agent")
    return value


class ReviewRecord(WorkflowContract):
    """A review decision bound to exact artifact hashes.

    ``actor_label`` is display text and ``claimed_actor_kind`` is what the submitter said about themselves; neither is
    evidence and neither affects ``binding_status`` or ``is_current_approval``. The name lint on ``actor_label`` is a
    lint, not authentication. ``authenticated_principal_ref`` is always ``None`` until Phase D resolves the reviewer
    through authentication and policy evaluation. Legacy reviewer values belong in ``legacy_metadata``."""
    RECORD_TYPE = "ReviewRecord"
    gate: ReviewGate
    decision: Literal["approved", "rejected"]
    actor_label: Annotated[Label, AfterValidator(_label_lint)]
    claimed_actor_kind: Literal["human", "agent", "system", "unknown"] | None = None
    authenticated_principal_ref: None = None
    reason: Message
    comments: Annotated[tuple[Message, ...], Field(max_length=100)] = ()
    bound_artifacts: Annotated[tuple[ArtifactBinding, ...], Field(max_length=256)] = ()
    rule_versions: RuleVersions = Field(default_factory=FrozenMap)
    legacy_metadata: JsonObject = Field(default_factory=FrozenMap)

    @model_validator(mode="after")
    def _binding(self):
        if self.decision == "approved" and not self.bound_artifacts:
            raise ValueError("an approval must bind at least one artifact hash")
        require_unique(tuple(b.artifact_id for b in self.bound_artifacts), "bound artifact ids")
        return self

    def _validated(self) -> ReviewRecord | None:
        try:
            return self.validated_copy()
        except ContractError:
            return None

    def _status_of(self, current: Mapping[str, str]) -> Literal["current", "stale", "missing"]:
        snapshot = _snapshot_current(current)
        if any(b.artifact_id in snapshot and snapshot[b.artifact_id] != b.digest for b in self.bound_artifacts):
            return "stale"
        if any(b.artifact_id not in snapshot for b in self.bound_artifacts):
            return "missing"
        return "current"

    def binding_status(self, current: Mapping[str, str]) -> Literal["current", "stale", "missing", "invalid"]:
        """Compare the bound hashes with the artifacts as they are now. The comparison runs on a freshly validated
        copy, never the live object: ``invalid`` if that fails (content hash or any structural rule), else ``stale``
        if a bound artifact changed, else ``missing`` if one is absent, else ``current``. ``current`` must be a plain
        mapping of str to str (bounded, read once); anything else raises ``ContractError``. This does not defend
        against someone who can forge records and their hashes; authenticity comes from the audit seal."""
        checked = self._validated()
        return "invalid" if checked is None else checked._status_of(current)

    def is_current_approval(self, current: Mapping[str, str]) -> bool:
        """STRUCTURAL check only, not an authorization: True when this record re-validates, says ``approved``, and every
        bound artifact is currently present with the bound digest. It says nothing about who approved (the label is free
        text and the hash is unkeyed), whether a later decision superseded it, or whether the gate required these
        artifacts. Authority is established elsewhere (authenticated principal, audit seal, policy) in a later phase."""
        checked = self._validated()
        return checked is not None and checked.decision == "approved" and checked._status_of(current) == "current"
