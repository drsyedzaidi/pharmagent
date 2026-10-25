"""Divergence analysis and markdown summary for app.evals.planner_divergence.

Pure functions over replicate records: no I/O except reading the finished
records, no LLM, no registry. Nothing here edits a rule; it only names what
varied (see docs/planner_divergence.md).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.core import policy

REPORT_VERSION = 2
MIN_REPLICATES = 2          # a single run has nothing to diverge from
RATIONALE_CHARS = 240       # markdown only; the JSON keeps the full text
VALUE_CHARS = 40            # markdown only: longer argument values get a hash suffix

CONVERGED = "CONVERGED"
DIVERGED_ARGS = "DIVERGED-ARGS"
DIVERGED_TOOL = "DIVERGED-TOOL"
UNREPLICATED = "UNREPLICATED"   # one replicate reached this point: nothing to compare
ALL_CONVERGED = "ALL-CONVERGED"
DIVERGED = "DIVERGED"
INCOMPLETE = "INCOMPLETE"       # an invalid replicate or no decision: no verdict on the rules

# Pseudo-tools for a leg that ended WITHOUT a planner proposal at that point, so
# "stopped here" vs "proposed X here" is itself a comparable decision.
END_STOP = "(planner stop)"         # a reasoned stop ("goal met")
END_SILENT = "(silent stop)"        # no tool and no text
END_PLANNER_ERROR = "(planner error)"
END_HARNESS_ERROR = "(harness error)"
END_TOOLS = frozenset({END_STOP, END_SILENT, END_PLANNER_ERROR, END_HARNESS_ERROR})

# Stops that say nothing about the rules: an infrastructure failure, an unknown
# stop, or the registry refusing an expensive tool the policy admitted (a policy
# regression). Such replicates are reported but kept out of the divergence tree.
INVALID_STOPS: dict[str, str] = {
    "dataset_error": "the dataset could not be loaded (an input problem, not a planner decision)",
    "planner_error": "the planner call failed (network, timeout, malformed answer)",
    "harness_error": "the leg raised inside the harness",
    "expensive_refused": "the policy admitted an expensive tool and the registry refused it: "
                         "a policy regression",
    "stopped": "the leg stopped for an unrecognised reason",
}

OWNER_DECISION = ("Nothing here is applied automatically. Converting a divergence into a policy "
                  "(app/core/policy.py) or mikebrain rule is an owner decision.")
GATED_KEY_NOTE = ("a policy scientific-choice key, so every proposal carrying it is gated: replicates "
                  "disagree on the value, a judgment branch for the human at the gate or a default the "
                  "owner may choose to write down")
SCIENTIFIC_REJECTED_KEY_NOTE = ("a policy scientific-choice key, but the policy rejected at least one "
                                "variant carrying it, so that value neither ran nor reached the human "
                                "gate and no human is known to have seen it (see the policy rejections "
                                "for why): replicates disagree on the value, and a human saw it only "
                                "where a variant carrying it was gated")
REVIEWED_KEY_NOTE = ("not a policy scientific-choice key, but every variant carrying it was gated for "
                     "another reason (the tool itself or another argument), so a human saw the value; a "
                     "candidate for SCIENTIFIC_CHOICE_KEYS only if it should be gated on its own")
REJECTED_KEY_NOTE = ("not a policy scientific-choice key, and the policy rejected at least one variant "
                     "carrying it, so that value neither ran nor reached the human gate and no human is "
                     "known to have seen it (see the policy rejections for why); a candidate for "
                     "SCIENTIFIC_CHOICE_KEYS only if it should be gated on its own")
UNGATED_KEY_NOTE = ("not a policy scientific-choice key and admitted WITHOUT a human gate in at least one "
                    "replicate: if the value changes the science, a candidate for SCIENTIFIC_CHOICE_KEYS "
                    "or a planner rule")
END_CHOICE_NOTE = ("some replicates ended here while others proposed a call: the planner may lack "
                   "information it needs (check the state summary it sees) or a rule saying the step "
                   "is possible")
STOP_KIND_NOTE = ("replicates ended differently without proposing anything (a reasoned stop vs a "
                  "silent one): the planner's stop behaviour is unstable at this point")
TOOL_CHOICE_NOTE = ("no rule orders these tools at this point: a candidate ordering rule (policy "
                    "PREREQUISITES or planner guidance), or an order that genuinely does not matter")
METHOD_SOURCE = ("Pritchard-Bell et al., CPT Pharmacometrics Syst Pharmacol 2026, "
                 "doi:10.1002/psp4.70317, PMID 42557662 (method re-expressed)")


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _unique(items: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(i for i in items if i not in ("", None)))


# -- divergence tree ---------------------------------------------------------
def _varied_arg_keys(variants: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Per tool proposed in 2+ variants: the argument keys whose value (or presence) differs."""
    by_tool: dict[str, list[Mapping[str, Any]]] = {}
    for v in variants:
        by_tool.setdefault(v["tool"], []).append(v["args"])
    varied: dict[str, list[str]] = {}
    for tool, arglists in by_tool.items():
        keys = sorted(set().union(*arglists)) if len(arglists) > 1 else []
        diff = [k for k in keys if len({(k in a, _canon(a.get(k))) for a in arglists}) > 1]
        if diff:
            varied[tool] = diff
    return varied


def _point(step: int, prefix: Sequence[Mapping[str, Any]], groups: Mapping[tuple[str, str], list[int]],
           seqs: Mapping[int, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    ids = sorted(i for group in groups.values() for i in group)
    variants = []
    for (tool, _), members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        first = seqs[members[0]][step]
        outcomes: dict[str, int] = {}
        for i in members:
            outcome = str(seqs[i][step].get("outcome"))
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        variants.append({"tool": tool, "args": first["args"], "count": len(members),
                         "replicates": members, "policy_mode": first.get("mode"),
                         "outcomes": outcomes, "policy_reason": first.get("reason", ""),
                         "rationales": _unique(seqs[i][step].get("rationale", "") for i in members)})
    tools = {v["tool"] for v in variants}
    label = (UNREPLICATED if len(ids) < MIN_REPLICATES else CONVERGED if len(variants) == 1
             else DIVERGED_ARGS if len(tools) == 1 else DIVERGED_TOOL)
    return {"step": step, "prefix": [dict(c) for c in prefix], "replicates": ids,
            "n_replicates": len(ids), "label": label, "variants": variants,
            "varied_arg_keys": _varied_arg_keys(variants)}


def decision_points(replicates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compare replicates decision by decision, but only among replicates whose
    earlier decisions were identical (a prefix tree): a decision is comparable when
    the same calls on the same data led up to it. Calls are keyed by
    ``policy.canonical_call``, the identity the policy's repeat guard uses."""
    seqs = {r["replicate"]: r["decisions"] for r in replicates}
    points: list[dict[str, Any]] = []
    frontier: list[tuple[list[dict[str, Any]], list[int]]] = [([], sorted(seqs))]
    step = 0
    while frontier:
        nxt = []
        for prefix, ids in frontier:
            groups: dict[tuple[str, str], list[int]] = {}
            for i in ids:
                if step < len(seqs[i]):
                    d = seqs[i][step]
                    groups.setdefault(policy.canonical_call(d["tool"], d["args"]), []).append(i)
            if groups:
                points.append(_point(step, prefix, groups, seqs))
                for members in groups.values():
                    d = seqs[members[0]][step]
                    nxt.append(([*prefix, {"tool": d["tool"], "args": d["args"]}], members))
        frontier, step = nxt, step + 1
    return points


# -- what varied --------------------------------------------------------------
def _key_note(scientific: bool, unreviewed: bool, rejected: bool) -> str:
    if scientific:
        return SCIENTIFIC_REJECTED_KEY_NOTE if rejected else GATED_KEY_NOTE
    if unreviewed:
        return UNGATED_KEY_NOTE
    return REJECTED_KEY_NOTE if rejected else REVIEWED_KEY_NOTE


def _choice_note(competing: Sequence[str]) -> str:
    ended = END_TOOLS & set(competing)
    if ended and ended == set(competing):
        return STOP_KIND_NOTE
    return END_CHOICE_NOTE if ended else TOOL_CHOICE_NOTE


def candidate_rule_gaps(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Name what varied; never edit a rule. An argument key is classified by whether
    the policy treats it as a scientific choice and whether any variant CARRYING it
    was admitted autonomously (it ran with no human seeing the value) or rejected
    (it never reached the human gate)."""
    arg_keys: dict[tuple[str, str], dict[str, Any]] = {}
    tool_choices = []
    for p in points:
        if p["label"] == DIVERGED_TOOL:
            competing = _unique(v["tool"] for v in p["variants"])
            tool_choices.append({"step": p["step"], "replicates": p["replicates"],
                                 "after": p["prefix"][-1] if p["prefix"] else None,
                                 "competing_tools": competing,
                                 "note": _choice_note(competing)})
        for tool, keys in p["varied_arg_keys"].items():
            for key in keys:
                carriers = [v for v in p["variants"] if v["tool"] == tool and key in v["args"]]
                rec = arg_keys.setdefault((tool, key), {
                    "tool": tool, "key": key, "steps": [],
                    "policy_scientific_choice": key in policy.SCIENTIFIC_CHOICE_KEYS.get(tool, frozenset()),
                    "admitted_without_gate": False, "rejected_by_policy": False})
                if p["step"] not in rec["steps"]:
                    rec["steps"].append(p["step"])
                rec["admitted_without_gate"] |= any(
                    v["policy_mode"] == policy.MODE_AUTONOMOUS for v in carriers)
                rec["rejected_by_policy"] |= any(v["policy_mode"] == policy.MODE_REJECTED for v in carriers)
    for rec in arg_keys.values():
        rec["note"] = _key_note(rec["policy_scientific_choice"], rec["admitted_without_gate"],
                                rec["rejected_by_policy"])
    return {"owner_decision": OWNER_DECISION,
            "arg_keys": [arg_keys[k] for k in sorted(arg_keys)],
            "tool_choices": tool_choices}


def policy_rejections(replicates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Proposals the policy refused, grouped by (tool, reason) across replicates."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for r in replicates:
        for rej in r["rejected"]:
            g = groups.setdefault((rej["tool"], rej["reason"]), {
                "tool": rej["tool"], "reason": rej["reason"], "count": 0, "replicates": []})
            g["count"] += 1
            if r["replicate"] not in g["replicates"]:
                g["replicates"].append(r["replicate"])
    return sorted(groups.values(), key=lambda g: (-g["count"], g["tool"], g["reason"]))


def _guarantee(replicates: Sequence[Mapping[str, Any]], stops: Mapping[str, int]) -> dict[str, Any]:
    """Observed, not asserted: counts taken at the registry choke point and from the audit chains."""
    calls = [c for r in replicates for c in r["execute_calls"]]
    flagged = sum(1 for c in calls if c["allow_expensive"])
    expensive = sum(len(r["expensive_executed"]) for r in replicates)
    scanned = sum(r["audit_entries_scanned"] for r in replicates)
    return {"execute_calls": len(calls), "allow_expensive_true_calls": flagged,
            "audit_entries_scanned": scanned, "expensive_tools_executed": expensive,
            "statement": (f"{len(calls)} tool executions observed at ToolRegistry.execute, "
                          f"{flagged} with allow_expensive=True; {scanned} audit entries scanned, "
                          f"{expensive} expensive tool executions; stops: "
                          + ", ".join(f"{k} ×{v}" for k, v in stops.items()) + ".")}


def _exercised(valid: Sequence[Mapping[str, Any]]) -> bool:
    """True when some replicate made a planner decision beyond the injected data load
    that the policy acted on: otherwise the run says nothing about the rules."""
    return any(d["tool"] not in END_TOOLS and d["tool"] != "load_dataset"
               and d.get("outcome") in ("ran", "gated", "rejected")
               for r in valid for d in r["decisions"])


def _problems(replicates: Sequence[Mapping[str, Any]], valid: Sequence[Mapping[str, Any]],
              points: Sequence[Mapping[str, Any]], dataset: Mapping[str, Any]) -> list[str]:
    problems = [f"replicate {r['replicate']}: {INVALID_STOPS[r['stop_reason']]} "
                f"({plain(r['stop_detail'], 200)})"
                for r in replicates if r["stop_reason"] in INVALID_STOPS]
    if len(valid) < MIN_REPLICATES:
        problems.append(f"only {len(valid)} valid replicate(s); at least {MIN_REPLICATES} are needed")
    if not points:
        problems.append("no planner decision was recorded")
    elif valid and not _exercised(valid):
        problems.append("no replicate made a planner decision beyond loading the data (a model "
                        "without tool calling, empty replies, or a first-step failure): nothing "
                        "about the rules was exercised")
    problems += [f"step {p['step']}: the same call {v['tool']} had different outcomes "
                 f"{v['outcomes']} across replicates (a non-deterministic tool failure)"
                 for p in points for v in p["variants"] if len(v["outcomes"]) > 1]
    if dataset.get("sha256_before") != dataset.get("sha256_after"):
        problems.append("the dataset file changed during the run")
    return problems


def build_report(replicates: Sequence[Mapping[str, Any]], *, llm: Mapping[str, Any],
                 dataset: Mapping[str, Any], workflow: str) -> dict[str, Any]:
    valid = [r for r in replicates if r["stop_reason"] not in INVALID_STOPS]
    points = decision_points(valid)
    diverged = [p for p in points if p["label"] in (DIVERGED_ARGS, DIVERGED_TOOL)]
    stops: dict[str, int] = {}
    for r in replicates:
        stops[r["stop_reason"]] = stops.get(r["stop_reason"], 0) + 1
    stops = dict(sorted(stops.items()))
    problems = _problems(replicates, valid, points, dataset)
    comparable = len(valid) >= MIN_REPLICATES and bool(points)
    analysis = DIVERGED if diverged else ALL_CONVERGED
    return {
        "harness": "planner_divergence", "version": REPORT_VERSION, "workflow": workflow,
        "dataset": dict(dataset), "llm": dict(llm), "n_replicates": len(replicates),
        "n_valid": len(valid),
        "verdict": INCOMPLETE if problems else analysis,
        "analysis_verdict": analysis if comparable else None, "incomplete_reasons": problems,
        "first_divergence_step": min((p["step"] for p in diverged), default=None),
        "stop_reasons": stops,
        "guarantee": _guarantee(replicates, stops),
        "decision_points": points,
        "candidate_rule_gaps": candidate_rule_gaps(points),
        "policy_rejections": policy_rejections(valid),
        "replicates": list(replicates),
        "method_source": METHOD_SOURCE,
    }


# -- markdown summary --------------------------------------------------------
_MD_SPECIAL = set("\\`*_[]<>|#!")


def plain(text: Any, limit: int = RATIONALE_CHARS) -> str:
    """One line from untrusted (LLM- or data-derived) text: whitespace flattened,
    control characters (terminal escapes included) dropped, length capped."""
    flat = "".join(ch for ch in " ".join(str(text).split()) if ch.isprintable())
    return flat if len(flat) <= limit else flat[:limit - 1] + "…"


def md(text: Any, limit: int = RATIONALE_CHARS) -> str:
    """``plain`` text with markdown syntax escaped: it cannot add headings, emphasis,
    tables, HTML or hidden link text (a bare URL may still be auto-linked, visibly)."""
    flat = plain(text, limit)
    return "".join("\\" + ch if ch in _MD_SPECIAL else ch for ch in flat)


def _short(value: Any) -> str:
    text = _canon(value)
    if len(text) <= VALUE_CHARS:
        return text
    return f"{text[:VALUE_CHARS - 16]}…#{hashlib.sha256(text.encode()).hexdigest()[:8]}"


def call_text(tool: str, args: Mapping[str, Any]) -> str:
    """Plain (unescaped) compact call; a long value keeps its head plus a short hash,
    so two values that differ anywhere never render the same."""
    inner = ", ".join(f"{k}={_short(args[k])}" for k in sorted(args))
    return f"{tool}({inner})" if inner else tool


def _after(prefix: Sequence[Mapping[str, Any]]) -> str:
    return call_text(prefix[-1]["tool"], prefix[-1]["args"]) if prefix else "(start)"


def _sampling_text(llm: Mapping[str, Any], n: int) -> str:
    if llm["provider"] != "local":
        return "provider default sampling"
    if not llm.get("temperature"):
        return ("temperature 0 (greedy): agreement shows the preferred path is stable, not that "
                "the rules are adequate; sample with --temperature to surface near-ties")
    seeds = ("no seed" if llm.get("seed_base") is None
             else f"seeds {llm['seed_base']}..{llm['seed_base'] + n - 1}")
    return f"temperature {llm['temperature']}, {seeds}"


def _header(report: Mapping[str, Any]) -> list[str]:
    llm, n = report["llm"], report["n_replicates"]
    first = report["first_divergence_step"]
    lines = [f"# Planner divergence: {md(report['workflow'])}", "",
             f"- Dataset: {md(report['dataset']['name'])} "
             f"(sha256 {report['dataset']['sha256_before'][:12]})",
             f"- Planner: {md(llm['provider'])} / {md(llm['model'])} ({md(_sampling_text(llm, n), 400)})",
             f"- Replicates: {n} ({report['n_valid']} valid)",
             f"- No-fit guarantee (observed): {md(report['guarantee']['statement'], 600)}",
             f"- **Verdict: {report['verdict']}**"
             + (f" (first divergence at step {first})" if first is not None else "")]
    if report["incomplete_reasons"]:
        if report["analysis_verdict"] is not None:
            lines.append(f"- Analysis of the valid replicates alone: {report['analysis_verdict']}")
        lines += ["", "## Why the run is incomplete", ""]
        lines += [f"- {md(reason, 400)}" for reason in report["incomplete_reasons"]]
    return lines


def _points_section(report: Mapping[str, Any]) -> list[str]:
    lines = ["", "## Decision points", "",
             "| step | after | replicates | label | variants |", "|---|---|---|---|---|"]
    for p in report["decision_points"]:
        variants = "; ".join(f"{v['count']}× {call_text(v['tool'], v['args'])}" for v in p["variants"])
        lines.append(f"| {p['step']} | {md(_after(p['prefix']), 120)} | "
                     f"{', '.join(map(str, p['replicates']))} | {p['label']} | {md(variants)} |")
    for p in report["decision_points"]:
        if p["label"] not in (DIVERGED_ARGS, DIVERGED_TOOL):
            continue
        lines += ["", f"### Step {p['step']} after {md(_after(p['prefix']), 120)}: {p['label']}"]
        for v in p["variants"]:
            lines.append(f"- {md(call_text(v['tool'], v['args']), 400)} ×{v['count']} "
                         f"(replicates {', '.join(map(str, v['replicates']))}; "
                         f"policy: {md(v['policy_mode'] or 'n/a')})")
            lines += [f"  - \"{md(t)}\"" for t in v["rationales"]]
    return lines


def _replicates_section(report: Mapping[str, Any]) -> list[str]:
    lines = ["", "## Replicates", "",
             "| # | stop | autonomous tools | gated proposal | rejected | elapsed s |",
             "|---|---|---|---|---|---|"]
    for r in report["replicates"]:
        gp = r["gated_proposal"]
        gated = md(call_text(gp["tool"], gp["args"]), 160) if gp else "-"
        lines.append(f"| {r['replicate']} | {md(r['stop_reason'])} | "
                     f"{md(', '.join(r['autonomous_tools']), 200)} | {gated} | {len(r['rejected'])} | "
                     f"{r['elapsed_s']} |")
    return lines


def _gaps_section(report: Mapping[str, Any]) -> list[str]:
    gaps = report["candidate_rule_gaps"]
    lines = ["", "## Candidate rule gaps", "", gaps["owner_decision"], ""]
    if not gaps["arg_keys"] and not gaps["tool_choices"]:
        lines.append("None: every compared decision point converged.")
    for g in gaps["arg_keys"]:
        lines.append(f"- {md(g['tool'], 80)}.{md(g['key'], 80)} varied at step(s) "
                     f"{', '.join(map(str, g['steps']))}: {g['note']}.")
    for t in gaps["tool_choices"]:
        competing = " vs ".join(md(c, 80) for c in t["competing_tools"])
        lines.append(f"- step {t['step']} after {md(_after([t['after']] if t['after'] else []), 120)} "
                     f"(replicates {', '.join(map(str, t['replicates']))}): {competing} competed: "
                     f"{t['note']}.")
    if report["policy_rejections"]:
        lines += ["", "## Policy rejections", ""]
        lines += [f"- {md(g['tool'], 80)} ×{g['count']} "
                  f"(replicates {', '.join(map(str, g['replicates']))}): {md(g['reason'])}"
                  for g in report["policy_rejections"]]
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [*_header(report), *_points_section(report), *_replicates_section(report),
             *_gaps_section(report), "", f"Method: {report['method_source']}.", ""]
    return "\n".join(lines)
