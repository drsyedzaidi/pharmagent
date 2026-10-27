"""Graph-level checks over dataset versions and transformations: lineage is a DAG of opaque version ids."""
from __future__ import annotations

from collections.abc import Iterable

from .base import WorkflowContract
from .contracts import DatasetVersion, TransformationRecord
from .errors import ContractError


def check_lineage(versions: Iterable[DatasetVersion], transformations: Iterable[TransformationRecord] = ()) -> None:
    """Raise ``ContractError`` unless the supplied set is internally consistent: every record still matches its own
    hash, the versions form a DAG whose every parent is present, and every transformation references present versions
    by their exact record hash and produces outputs descended from exactly its inputs. Ids are chosen by the caller, so
    this proves consistency of the supplied set, not that the set is the genuine history."""
    by_id: dict[str, DatasetVersion] = {}
    transformations = list(transformations)
    if any(type(t) is not TransformationRecord for t in transformations):
        raise ContractError("check_lineage needs exact TransformationRecord records")
    transformations = [WorkflowContract.validated_copy(t) for t in transformations]
    for original in versions:
        if type(original) is not DatasetVersion:
            raise ContractError("check_lineage needs exact DatasetVersion records")
        v = WorkflowContract.validated_copy(original)      # decisions below read the validated copy, never the live one
        if v.id in by_id:
            raise ContractError(f"duplicate dataset version id '{v.id}'")
        by_id[v.id] = v
    for v in by_id.values():
        for p in v.parents:
            if p not in by_id:
                raise ContractError(f"dataset version '{v.id}' has a missing parent '{p}'")
    _refuse_cycles(by_id)
    seen_ids: set[str] = set()
    produced_by: dict[str, str] = {}
    for t in transformations:
        if t.id in seen_ids:
            raise ContractError(f"duplicate transformation id '{t.id}'")
        seen_ids.add(t.id)
        for ref in t.outputs:
            if produced_by.setdefault(ref.version_id, t.id) != t.id:
                raise ContractError(f"version '{ref.version_id}' is produced by more than one transformation")
        for ref in (*t.inputs, *t.outputs):
            v = by_id.get(ref.version_id)
            if v is None:
                raise ContractError(f"transformation '{t.id}' references a missing dataset version '{ref.version_id}'")
            if v.content_hash != ref.record_hash:
                raise ContractError(f"transformation '{t.id}': record_hash does not match version '{ref.version_id}'")
        inputs = {r.version_id for r in t.inputs}
        for ref in t.outputs:
            if set(by_id[ref.version_id].parents) != inputs:
                raise ContractError(f"transformation '{t.id}': parents of output '{ref.version_id}' must be exactly "
                                    f"the transformation's inputs")


def _refuse_cycles(by_id: dict[str, DatasetVersion]) -> None:
    visiting, done = set(), set()
    for start in by_id:
        if start in done:
            continue
        stack = [(start, iter(by_id[start].parents))]
        visiting.add(start)
        while stack:
            node, parents = stack[-1]
            for p in parents:
                if p in visiting:
                    raise ContractError(f"dataset lineage contains a cycle through '{p}'")
                if p not in done:
                    visiting.add(p)
                    stack.append((p, iter(by_id[p].parents)))
                    break
            else:
                stack.pop()
                visiting.discard(node)
                done.add(node)
