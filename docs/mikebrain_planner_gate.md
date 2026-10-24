# mikebrain at the planner's human gate (M3)

**Status:** built, **off by default**, advisory only. It gives the human who is
about to approve a planner proposal the owner's literature as a second opinion.
It decides nothing: the deterministic policy, the gate and the human's options
are exactly what they were without it.

Read `docs/mikebrain.md` first for the corpus, the critic tool and the
quote-grounding rule. This page covers only what M3 adds.

## What it does

When the goal planner (`Orchestrator._advance_goal`) proposes a step that the
policy has already classified as **gated** (a structural model choice, an
expensive fit, the report), the proposal is parked in `state.pending_tool` for a
human, as before. M3 adds one step just before it is parked:

```
planner proposes  ->  policy.decide  ->  "gated"           (unchanged, already final)
                                            |
                                            v
                        gate_critic.critique_proposal      (new, advisory)
                          registry.execute("consult_mikebrain")   audit entry, agent "reviewer"
                                            |
                                            v
             pending_tool = { ...as before..., "critic": { stance, citations, ... } }
             audit "propose_tool" outputs carry the same critic block
                                            |
                                            v
                         human approves / rejects         (unchanged)
```

The question put to the literature names the proposed tool, its arguments
(configuration such as model key, method or alpha), the policy's reason for
gating, the planner's stated rationale and the goal. The rationale is presented
as **a claim to test against the excerpts**, not as background, so the proposer
does not get to frame its own critic. Arguments and planner text are size-capped
and have subject references removed (best effort: "subject 12", "ID=1771",
"patients 1, 2", and per-subject finding ids such as `nca-clf-sign-1771` reduced
to their kind; an identifier with no digit in it is not recognised).
The consultation also sees the same aggregate-only state summary as the M1
critic, never rows. That summary's blocking-finding ids are reduced to their
kind (`recompute-cmax-1771` -> `recompute-cmax`), which also tightens the M1
critic, where the same ids used to reach the prompt, the retrieval query and
`knowledge_results.state_summary`.

## Enabling it

```bash
PHARMAGENT_MIKEBRAIN_GATE_CRITIC=true
```

Restart the server afterwards (never with `--reload`). It only has an effect
when the owner's corpus exists (`python -m app.knowledge add ...`). With the
switch on and no corpus, no LLM call is made and the block says
`not_consulted`.

**Why it is off by default.** The consultation is a synchronous LLM call inside
the request that parks the proposal, under the session lock, like the planner
call itself. Measured on a local `gemma4:31b` it takes 75 to 120 seconds, and
the first call after idle longer; the hard limit is the critic call's 300 s
timeout plus embedding time. A goal run parks at most `max_gated` (6)
proposals. A smaller local model answers in seconds. While it runs, other
requests for the same session wait (reading a session takes the same lock), and
a client that times out and calls `start_workflow` again restarts the planner
from step 0, a window the planner call already had and this widens. Whether to
turn it on by default is an owner decision.

## The `critic` block

Present on `pending_tool` (and therefore on the `proposal` in the workflow
response and in `state`) only when the switch is on:

| Field | Meaning |
|---|---|
| `stance` | `objection` (a retrieved passage was cited as contradicting the choice), `corroboration` (cited as supporting it), `no_evidence` (consulted, nothing cited), `not_consulted` (no corpus), `unavailable` (could not answer) |
| `meaning` | one plain sentence for the reader at the gate |
| `note` | "Advisory only: ..." verbatim, every time, including that a quote shows what a passage says, not that it applies |
| `status`, `verdict`, `grounding` | the underlying `consult_mikebrain` record |
| `rationale`, `citations`, `citations_total` | the model's reason and at most 3 **verbatim** quotes of at most 300 characters (a verbatim prefix when cut, flagged `truncated`), each validated against the retrieved chunk by the M1 rule |
| `corpus_version`, `chunk_hashes`, `provider`, `model`, `elapsed_s` | what the answer was based on and who gave it |

`objection` and `corroboration` require at least one validated verbatim
citation: a verdict the validator downgraded to `no_evidence` is never shown as
an objection. What grounding proves is that the quote is in a retrieved passage,
not that the passage entails the verdict or applies to this analysis; the stance
wording says so and a person has to read the quote.

Against the design's Phase B exit criterion ("every gated proposal carries at
least one audited objection or an explicit 'no objection'"): a consulted
proposal carries an explicit, audited stance. A proposal that was `not_consulted`
or `unavailable` carries a stance too, but it is neither an objection nor a "no
objection", and has no consultation entry. The criterion is met only for
consulted proposals.

## Audit

With the switch on, a gated proposal produces these entries, in this order:

1. `planner` / `policy_gated` (unchanged)
2. `reviewer` / `consult_mikebrain` (new; its outputs hash binds the full
   consultation record: question, chunk hashes, citations, model, elapsed time)
3. `planner` / `propose_tool` (unchanged fields, plus `critic` in its outputs,
   so the advisory the human saw is bound to the proposal)

Human decisions keep today's entries: the signed `human_review`, then
`approve_tool` or `reject_tool`, with identical inputs and outputs hashes with
and without the critic (tested). With the switch off the chain is identical to a
build without M3: no extra entry and no extra clock reading (tested entry for
entry).

**What the signed decision does not do.** It does not reference the advisory the
human saw. `pending_review.proposal` and the `human_review` entry are unchanged,
so the link runs one way: the `propose_tool` entry carries the block, and the
audit chain stores hashes only, so the block itself can be checked only while
the proposal is pending or the record is still among the last 20
`knowledge_results`. Making the signed decision cite the advisory would mean
editing the gate and e-signature path, which is the owner's call, not this PR's.

## What an objection cannot do

An `objection` is information for a person. It cannot block, delay, reroute or
alter anything, and the tests pin each of these with a run that has the critic
and a run that does not:

- The proposal's tool, agent, arguments, rationale, `gate_reason`, workflow and
  the `review` payload are identical. (`proposed_at` is later by the clock ticks
  the consultation used.)
- The policy's `plan` and `policy_gated` audit entries are identical.
- Rejecting writes the same signed entries; approving an expensive tool still
  returns `awaiting_job` and consumes nothing until it is re-issued as a job.
- The `critic` field never reaches the approved call: `take_pending_tool` reads
  only `tool`, `agent` and `args`, and a test asserts the returned call has no
  trace of it.
- Chat-proposed tools (`run_bootstrap` and the like) are not critiqued; the only
  caller of the critic is `_propose_goal_step`.

Whether a high-severity objection may ever block on its own is still the
owner's open question 3 in the planner design. M3 answers "no, advisory", and
changing that would be a deliberate, reviewed change to the gate, not a flag.

## Failure behaviour

The critic never stops a proposal reaching the human. In each case the
proposal is parked exactly as without M3, the audit chain verifies, and the
block states why there is no opinion:

| Situation | `stance` / `status` |
|---|---|
| no corpus | `not_consulted` / `no_corpus`, no LLM call |
| corrupt manifest | `unavailable` / `corpus_error` |
| LLM unreachable, timed out, model missing | `unavailable` / `llm_error` |
| any unexpected exception | `unavailable` / `error` (details go to the server log only; the block carries no error text or path) |

## Where the data goes

Retrieval stays on the owner's machine. The excerpts and the question go to the
**configured** LLM, which is local Ollama by default and a hosted API only if
the owner switched to one (same statement as `docs/mikebrain.md`, "Where corpus
text travels"). The critic uses the app's configured LLM, not a separate model.

Quotes from the owner's private corpus now land in `pending_tool.critic`, the
workflow response's `proposal` and `state`, in addition to
`state.knowledge_results` (M1, up to 12 quotes of 600 characters per record, the
last 20 records). The gate block is capped at 3 quotes of 300 characters per
proposal and at most 6 proposals per run. The corpus is deployment-wide, not
per user: with the switch on, **anyone who can start a workflow on this server
receives quotes from it.** Enable it only where that is acceptable. On a failed
LLM call the underlying record in `knowledge_results` also keeps the exception
text (M1 behaviour, first 300 characters, which can name the model server URL);
the gate block carries no error text.

## Not done in M3

- **No UI.** The data is on `pending_tool.critic`, in the workflow response's
  `proposal` and in state. The frontend has no planner-gate screen yet, so there
  is nothing to bind it to.
- **No planner reply.** The design's Phase B lets the planner revise or answer
  the objection; M3 only informs the human.
- **No generic adversarial critic.** The design's critic persona is an LLM that
  attacks the proposal from the diagnostics alone. M3 is the literature-grounded
  critic only.
- **No per-role LLM routing**, so the critic cannot yet be a different model
  from the planner.
- **Rules (M2) are not consulted here.** Owner-approved rules already act in
  `adversarial_review`; feeding them to the gate is a separate decision.
