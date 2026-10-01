"""Registry choke point: opt-in read-only audit snapshot for tools that must cite entries.

Only a Tool declared ``uses_audit_trail=True`` receives it, as an immutable tuple of
entry dicts for the entries that exist BEFORE its own result is appended. The audit
append itself is untouched, and the caller's ToolContext is never mutated.
"""
from app.core.audit import AuditChain
from app.core.pharmstate import PharmState
from app.tools.base import Tool, ToolContext, ToolRegistry, ToolResult

SEEN: dict = {}


def _spy(state, ctx, args):
    SEEN["trail"] = ctx.audit_trail
    SEEN["data_dir"] = ctx.data_dir
    return ToolResult(summary="spy", result={"n": len(ctx.audit_trail)})


def _registry(flag: bool) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool("spy", "d", "report", {"type": "object"}, _spy, uses_audit_trail=flag))
    return reg


def _chain(n: int) -> AuditChain:
    chain = AuditChain()
    for i in range(n):
        chain.append(agent="nca", tool="compute_nca", action=f"a{i}", inputs={}, outputs={"i": i},
                     timestamp="t")
    return chain


def test_opted_in_tool_sees_prior_entries_only_and_as_immutable_dicts():
    audit, ctx = _chain(3), ToolContext(data_dir="x")
    _registry(True).execute("spy", state=PharmState(), ctx=ctx, args={}, audit=audit,
                            timestamp="t", actor="u")
    trail = SEEN["trail"]
    assert isinstance(trail, tuple) and [e["index"] for e in trail] == [0, 1, 2]
    assert all(isinstance(e, dict) and "outputs_hash" in e for e in trail)
    assert len(audit.entries) == 4 and audit.entries[3].tool == "spy"   # own entry came after
    assert audit.verify()
    assert SEEN["data_dir"] == "x"                                      # rest of ctx preserved


def test_snapshot_cannot_alter_the_chain():
    audit = _chain(2)

    def mutate(state, ctx, args):
        ctx.audit_trail[0]["tool"] = "forged"        # a copy: must not reach the chain
        return ToolResult(summary="m")

    reg = ToolRegistry()
    reg.register(Tool("mut", "d", "report", {"type": "object"}, mutate, uses_audit_trail=True))
    reg.execute("mut", state=PharmState(), ctx=ToolContext(), args={}, audit=audit, timestamp="t")
    assert audit.entries[0].tool == "compute_nca" and audit.verify()


def test_default_tools_get_no_trail_and_callers_ctx_is_not_mutated():
    audit, ctx = _chain(3), ToolContext()
    _registry(False).execute("spy", state=PharmState(), ctx=ctx, args={}, audit=audit, timestamp="t")
    assert SEEN["trail"] == ()
    _registry(True).execute("spy", state=PharmState(), ctx=ctx, args={}, audit=audit, timestamp="t")
    assert ctx.audit_trail == ()                     # the shared session ctx stays clean
