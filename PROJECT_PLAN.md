# PharmAgent — Project Plan

An agentic AI platform for end-to-end pharmacometric analysis. Hierarchical
multi-agent architecture (Supervisor → domain agents → modeling specialists),
schema-only privacy, human-in-the-loop review gates, and a SHA-256 hash-chain
audit trail. Built for PmatricsAI.

> **Design principle:** *Agents decide, tools execute.* The LLM reasons about
> analytical strategy; deterministic Python tools (numpy/scipy/pandas) perform
> all computation. Reproducibility comes from the tools, not the model.

---

## Architecture (target)

```
Level 0  Supervisor            two-stage routing (keyword score -> LLM fallback)
Level 1  Domain agents         Data Manager, NCA, QC, Report, PBPK, Statistical,
                               Simulator, Regulatory Intel, Modeler Manager
Level 2  Modeling specialists  PopPK Expert, PKPD Expert, E-R Expert
```

All agents communicate **only** through `PharmState` — a typed shared state with
per-agent write-access rules. Every tool call is wrapped by the audit chain.

```
User → Supervisor → Agent → Tool(s) → PharmState → (review gate) → next step
                                  └── AuditChain (SHA-256, append-only)
SchemaExtractor sits between every dataset and the LLM: metadata only.
```

---

## Phased delivery

### Phase 1 — Vertical slice (FOUNDATION)   ← current
Prove the entire spine on the easiest analytical method (NCA).
- [x] Repo + venv + deps
- [x] `PharmState` typed model + write-access enforcement
- [x] Audit hash-chain (SHA-256, persisted)
- [x] `SchemaExtractor` (privacy: metadata-only summaries)
- [x] Tool base + registry (audit-wrapped)
- [x] Agent base + Supervisor (two-stage routing)
- [x] Data Manager agent + data tools
- [x] NCA agent + NCA compute (linear-up/log-down)
- [x] QC agent + diagnostic checklist
- [x] Report agent + DOCX export
- [x] FastAPI app (chat endpoint, workflow runner)
- [x] End-to-end test: load → NCA → QC → report, no API key needed
- [ ] Minimal React chat UI (Phase 1b)

### Phase 2 — Product hardening
- Auth + multi-user projects, dataset storage, persistence layer
- SSE streaming of agent steps to the UI
- Review-gate UX (approve/reject at decision points)
- Full QC 15-point checklist; bioequivalence + dose-proportionality tools
- Plotly visualizations (spaghetti, GOF, forest, E-R) returned as widgets
- Workflow templates engine (ordered steps + parameters + gates)

### Phase 3 — Modeling specialists
- Modeler Manager + PopPK Expert (1-/2-cmt, IIV, covariate SHAP screen)
- Shell-out to R (nlmixr2 / mrgsolve) for estimation; NONMEM control-stream gen
- PKPD Expert (Emax, indirect response); E-R Expert (logistic, survival)
- **Hardest phase — validated estimation in an agent loop. Do not rush.**

### Phase 4 — Regulatory + simulation
- Regulatory Intel agent: RAG over FDA/EMA/ICH guidances (cited responses)
- Simulator agent: Monte Carlo, virtual populations, target attainment
- PBPK agent
- Methods-section auto-generation from the audit trail

### Phase 5 — Compliance + deploy
- 21 CFR Part 11 posture: audit export, e-sign metadata, read-only raw data
- Validation documentation per tool
- Containerized deploy, secrets management, rate limiting

---

## Stack

| Layer | Choice |
|-------|--------|
| Backend | FastAPI (Python 3.13) |
| Agents | Anthropic Claude (tool-use); custom orchestrator, no heavy framework |
| Compute | numpy / scipy / pandas; R (nlmixr2/mrgsolve) shelled out in Phase 3 |
| State | `PharmState` (pydantic) + SQLAlchemy persistence |
| Audit | SHA-256 hash chain (append-only) |
| Reports | python-docx |
| Frontend | React + Vite + Plotly.js |

---

## Non-negotiables (carried from the design)

1. **Privacy:** raw patient data never reaches the LLM. SchemaExtractor enforces it.
2. **Determinism:** all numbers come from tools, not the model.
3. **Human-in-the-loop:** review gates at scientific decision points.
4. **Traceability:** every tool call in the audit chain. (Traceability ≠ correctness — a human still verifies.)
