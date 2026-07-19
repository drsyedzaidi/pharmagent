<!-- Generated 2026-07-18 via a 4-analyst + synthesis workflow, every claim file:line-traced against source.
Independently re-verified before commit: scipy surface (solve_ivp/expm/least_squares/minimize/stats.* all in Pyodide);
python-docx needs lxml via loadPackage; nlme SCM self-disables to serial in WASM (nlme.py:1402-1403, cpu_count->1);
jobs.py ThreadPool is the P1 drop; main.py:656 already serves frontend/dist (the hybrid basis). -->

# SPEC: PharmAgent Browser-Native (Pyodide/WASM) Migration
*Single source of truth for the build/no-build decision. Synthesizes four subsystem audits (compute/engines, LLM/agent-loop, persistence/audit/files/exports, frontend/build). All file:line refs traced by the analysts against source.*

---

## 1. VERDICT

**Feasible. High confidence (~85%).** The codebase already isolated every seam that matters: compute is pure numpy/scipy/pandas (26 of 28 modules port unchanged), `api.ts` is the sole network module (App.tsx never touches `fetch`), FastAPI endpoints are thin pass-throughs whose return shapes are already worker-ready, and the LLM is a pure decision function (`classify`/`select_tool`, `llm.py:20-23`) that never runs a tool — so **all pharmacometrics execution stays 100% client-side under every configuration**, LLM or not. The single hardest constraint is **not** technical portability; it is the **audit-chain tamper-evidence gap**: a SHA-256 hash chain moved into the analyst's own IndexedDB proves internal consistency but no longer proves authorship — the audited party can forge a chain that passes `verify()` byte-for-byte. That cannot be fixed by moving code; it needs an external hash-only witness or an honest downgrade of the Part-11 framing. The remaining known risk (WASM numerical parity of FOCE/SAEM vs the CPython reference) is real but measurable — it must be proven, not assumed, before committing.

---

## 2. TARGET ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────┐
│  BROWSER TAB — static host (Netlify / Pages / Cloudflare). No app server. │
│                                                                           │
│  ┌───────────────────────── UI THREAD ─────────────────────────────┐     │
│  │  React 19 SPA (App.tsx, flexplot/, types.ts, index.css)          │     │
│  │  — UNCHANGED. Imports the `api` object (App.tsx:8), ~35 sites.    │     │
│  │  api.ts  ← the ONLY file whose transport changes                 │     │
│  │     req()/download()  →  rpc(method,args) via Comlink proxy      │     │
│  └───────────────────────────────┬──────────────────────────────────┘     │
│                    Comlink RPC    │ structured-clone. UI never blocks.     │
│  ┌──────────────────────── WEB WORKER ▼───────────────────────────┐       │
│  │  pyodide.worker.ts  →  loadPyodide + loadPackage(numpy,pandas,  │       │
│  │                        scipy[lazy]) + micropip(python-docx)     │       │
│  │  worker_api.py  (NEW ~60-line dispatch shim, replaces main.py)  │       │
│  │     dispatch(method,args) → Orchestrator singleton             │       │
│  │  ┌──────────────────────────────────────────────────────────┐  │       │
│  │  │  app/  (pure Python, unchanged):                         │  │       │
│  │  │   orchestrator · agents · supervisor · tools             │  │       │
│  │  │   compute/*  (nca, nlme FOCE/SAEM, pk_fit, vpc, cdisc…)  │  │       │
│  │  │   core: pharmstate · audit (hashlib sha256) · provenance │  │       │
│  │  │   engines: native FOCE + native SAEM · mock · ensemble   │  │       │
│  │  │            (nlmixr2 EXCLUDED — no R in WASM)             │  │       │
│  │  │   MockLLM (default, keyless)  ─or─  BrowserLLM (await JS) │  │       │
│  │  └──────────────────────────────────────────────────────────┘  │       │
│  │  MEMFS /data (uploaded CSV, .docx/.zip out) │ IndexedDB (persist)│      │
│  └────────────────────────────────────────────────┬───────────────┘       │
│  ┌── SERVICE WORKER: Cache Storage of pyodide.asm.wasm, numpy/scipy/       │
│  │   pandas wheels, app-*.whl, SPA → instant + offline repeat visits ┘     │
└─────────────────────────────────────────────────────┼─────────────────────┘
                                                       ▼ (ONLY if RealLLM tier)
                                    ┌────────────────────────────────────┐
                                    │ OPTIONAL stateless LLM proxy       │
                                    │ (holds key + CORS). classify /     │
                                    │ select_tool decisions ONLY.        │
                                    │ ~50-line Worker/edge fn. No data.  │
                                    └────────────────────────────────────┘
```

**How much server is left: essentially none.** Static hosting serves the SPA + Pyodide + wheels. All compute, audit hashing, state, persistence (IndexedDB), and DOCX/CSV/ZIP generation run in-browser. The *only* optional server piece is a tiny stateless LLM proxy — needed **iff** you want real-Claude chat without asking users for their own key. Ship with MockLLM and the app is 100% serverless, offline, keyless. The invariant across all tiers: **the model decision (≤7 short calls/turn) is the sole network egress, and only on real-LLM tiers; no dataset, dataframe, or patient value ever leaves the machine** (`ToolContext` raw dataframes are never serialized to the model, `base.py:23`; `state_summary` is explicitly privacy-safe, `base.py:32-46`).

---

## 3. PORTABILITY LEDGER

| Subsystem | Verdict | Blocker (file:line) | Fix |
|---|---|---|---|
| `compute/` — nca, nca_ss, pk_models, pk_simulate, compartmental, pk_fit, vpc, diagnostics, poppk, bioequivalence, dose_proportionality, dose_sweep, forecast, dosing, nmexport, adversarial, flexplot (17 modules) | **PORTS-AS-IS** | none — numpy/scipy/pandas/stdlib only. `solve_ivp` LSODA + `expm` + `least_squares` + `minimize` + `scipy.stats` all present in Pyodide's scipy wheel | — |
| `compute/nlme.py` (FOCE-I / SAEM / SCM) | **WITH-CHANGES** | `ProcessPoolExecutor` (nlme.py:52, pool at :1402) — SCM fan-out only | Self-disables: `os.cpu_count()`→1 in WASM, `>1` guard (:1403) picks `pool=None`→serial `_fit_batch` (:1329). **Zero code change to run.** Must run in a Web Worker (long runtime). Serial SCM is slower but correct. |
| `engines/` — base, select, ensemble, scoring, dataset_io, native, mock, runner, demo (9 modules) | **PORTS-AS-IS** | none — math/dataclasses/pandas; native inherits nlme worker concern via `population_fit` | — |
| `engines/nlmixr2.py` (R shell-out) | **CANNOT-PORT** | `subprocess`+`Rscript` (:19, :190) — needs R + nlmixr2 package | No fix — R can't run in Pyodide. **Fails safe:** `available()`→False→runner records `absent` row (runner.py:16-18), no crash. `demo.py:46-49` already swaps a `MockEngineAdapter` in. |
| LLM decision layer (`core/llm.py`) | **WITH-CHANGES** | `RealLLM` `import anthropic` + holds key (:64-69). Lazy import → default build has zero network | Delete `RealLLM` from worker; add `BrowserLLM` that `await`s a JS async fn across the Pyodide boundary (JS does the `fetch`). MockLLM (:29-61) pure, stays in-worker default. |
| Tool execution / registry / agents | **PORTS-AS-IS** | none | Decision/exec seam already split (`base.py:52-70`): `registry.execute → tool.run` (:90) is pure, deterministic, audited, no network — runs in WASM unchanged. |
| `core/store.py` + `skills.py` (SQLite) | **WITH-CHANGES** | `sqlite3` file lives in MEMFS → dies on reload. `threading.Lock` is a **no-op** in single-thread WASM (false blocker) | Keep SQL logic verbatim. Snapshot `conn.serialize()`→IndexedDB blob on save; `deserialize()` on boot (fallback: `FS.readFile`). |
| `core/audit.py` (SHA-256 chain) | **PORTS-AS-IS (integrity caveat)** | none — `hashlib` compiled-in; `verify()` runs unchanged | Code ports. **But tamper-evidence weakens** — see §4b. |
| `core/provenance.py` | **WITH-CHANGES** | `subprocess` git SHA (:11, :27-37) | Inject build-time constant (`__GIT_SHA__` via Vite `define`). `importlib.metadata.version()` keeps working; `platform.*`→`Emscripten-…-wasm32` is *better* provenance. Add `pyodide` to `_PACKAGES`. |
| File upload (`data_tools.py`) | **WITH-CHANGES** | `_safe_path` assumes disk roots | `File.arrayBuffer()`→`FS.writeFile('/data/ds.csv')`→`pd.read_csv` unchanged. Repoint `allowed_data_dirs` to `['/data','/samples']` — confinement keeps teeth. `pd.read_sas` (.xpt) pure-python, works. |
| Exports — CSV (`exporters.py`), ZIP+define.xml (`cdisc.py`) | **PORTS-AS-IS** | none — stdlib `csv`/`io`/`zipfile`/`xml`; already return str/bytes | Marshal buffer → Blob + `<a download>`. No generator change. |
| Exports — DOCX (`report_tools.py`) | **WITH-CHANGES** | python-docx transitively needs **`lxml`** (C-ext) | `doc.save(BytesIO())`→Blob. python-docx wheel via micropip **but `lxml` must come from `pyodide.loadPackage('lxml')`**, not micropip. |
| `core/jobs.py` (ThreadPoolExecutor) | **DROP** | `ThreadPoolExecutor` starts threads (:43) → fails in WASM | Delete. The Worker *is* the background context; submit+poll collapses to one awaited RPC. |
| `main.py` (FastAPI/ASGI) | **DROP** | uvicorn/starlette/multipart | Replaced by `worker_api.py` dispatch shim. Reuse Pydantic request models (main.py:121-192) for arg validation — pydantic runs in Pyodide. |

**Genuine losses:**
- **R/nlmixr2 cross-engine estimator — lost, moderate impact.** This is the one truly-independent third-party FOCEI/SAEM. Client-side it reports `absent`. The cross-engine machinery **still functions**: `pharmagent_focei` vs `pharmagent_saem` are two distinct engines ranked engine-agnostically on prediction metrics (`pred_rmse`/`vpc_coverage90`, select.py:28-35), optionally joined by `ensemble`/`mock`. What's lost is *independent* cross-validation against a non-PharmAgent engine. **If independent-engine cross-validation is a stated differentiator, only the WebR route (not Pyodide) retains R** — that is a different, heavier architecture.
- **ProcessPool SCM parallelism — degraded, low impact.** Collapses to serial (self-disabling). A real covariate search runs minutes-to-tens-of-minutes sequentially instead of parallel. Mitigation exists (nested Web Workers, K× ~30-50 MB Pyodide instances) but is a heavy re-architecture; near-term answer is serial SCM + candidate cap + progress UI.

---

## 4. THE THREE HARD PROBLEMS

### (a) LLM key / CORS

The loop needs only the final message (no streaming — `llm.py:80,97`) and a tiny serializable payload (system + user msg + tool schemas + privacy-safe state; back = `{name,input}` or `None`). Four options, pick a primary + layer the rest:

| Option | What it is | Verdict |
|---|---|---|
| **1. MockLLM** (exists today) | Pure Python, keyless, auto-selected with no key. Ships unchanged. | **Always-on floor.** Enough for free/demo tier + CI *because the pharmacometrics work is fully deterministic client-side*; only NL convenience thins. Direct-tool UI buttons bypass NL entirely (`api.ts:104-231`). |
| **2. BYO-key** (browser→api.anthropic.com) | Needs header `anthropic-dangerous-direct-browser-access: true` + `x-api-key`. User's own key. | **Best "real Claude, literally zero server."** Cost: user must paste a key. Hold key in memory/`sessionStorage` (XSS), warn, recommend low-limit key. ⚠️ **Verify the header name against current Anthropic docs before shipping.** |
| **3. Thin stateless proxy** | One `POST /llm` forwards to Anthropic with server-side key. Stateless, no DB, no compute. ~50-line edge fn. | **Best mainstream UX** — user configures nothing. Cost: you own one tiny hosted dependency + abuse surface (rate limits, `max_tokens` cap, model allowlist, spend cap, Turnstile). |
| **4a. OpenAI-compat adapter** | BYO base_url+key → OpenRouter, vLLM, LM Studio, **local Ollama** (offline/private). | Cheap high-value **add-on** on top of whichever primary. |

**Recommendation.** If *"no user setup / no account"* is the north-star (matches DrLevy): **primary = Option 3 (proxy)**, with 1/2/4a layered underneath. If *"zero server, period"* is a hard constraint: **promote Option 2 (BYOK) to primary**, Option 1 as no-key fallback, 4a for local/offline — then the app is a pure static bundle. **Fallback within either config:** MockLLM always guarantees a working app.

**Rewire (only real change to the loop):** replace `RealLLM` with `BrowserLLM.select_tool` that `await`s a JS promise. The loop body (`for _ in range(MAX_TOOL_STEPS)` + `registry.execute`, `base.py:52-70`) is structurally identical; execution stays in WASM. `agent.run_turn` / `orchestrator.chat` / `supervisor.route` become `async` (mechanical; MockLLM path awaits an already-resolved value).

### (b) Persistence across reloads + client-side audit integrity

**Persistence — SQLite→IndexedDB blob (recommended).** On `save()`, after commit: `conn.serialize()`→`Uint8Array`→`IDBObjectStore.put(blob,"pharmagent.db")`. On boot: read→`conn.deserialize()` (fallback `FS.readFile` if `serialize` absent in the Pyodide build). Keeps `store.py`/`skills.py` logic identical (matters — CLAUDE.md gates this code as human-review). Single-writer, atomic; the `.sqlite` blob doubles as export/backup. Debounce snapshots.
- **Rejected: IDBFS `syncfs`** — multi-tab-unsafe (whole-dir last-writer-wins → corruption), unacceptable for a regulated app.
- **Rejected: rewrite store to raw IndexedDB KV** — most native but churns human-review-gated code.
- **Multi-tab:** guard snapshot with Web Locks API or a BroadcastChannel-elected writer. Single-user local tool → last-writer-wins + warning banner acceptable, but *state it*.
- **The real gap:** `store.py:3-6` re-reads the dataset from `dataset_path` on load — no re-readable path after reload in the browser. **Must persist uploaded CSV bytes**, content-addressed by `dataset_sha256`; `dataset_path`→`blob://<sha256>`, rehydrated into MEMFS on restore. (Optional: a git-lite versioned VFS — `blobs`/`versions`/`refs` stores, copy-on-write trees — gives time-travel + integrity + competitor parity, and pairs naturally with the SHA-256 audit chain.)
- **Risk:** IndexedDB eviction under pressure → call `navigator.storage.persist()`.

**Audit integrity — the honest problem.** Moving the chain into the analyst's own IndexedDB makes it **weaker as tamper-evidence, stronger as reproducibility + privacy.** On a server the audited party didn't hold the infra, so history-rewrite was hard and the server witnessed the head. In the browser the audited party can edit an input and re-hash forward — `verify()` returns True on a fully self-consistent forgery. A hash chain proves consistency, never authorship. **No client code recovers this.** Two moves:
1. **Export a signed chain** — canonicalize (already deterministic, `sort_keys=True` audit.py:29), sign with a WebCrypto ECDSA-P256 non-extractable key bound to the Part-11 identity in the chain. *Necessary, not sufficient* — only binds "same browser."
2. **The real fix — hash-only anchor.** Periodically POST *only the current head* (32 bytes, zero patient data) to an external witness: RFC-3161 TSA, OpenTimestamps (zero-server), a transparency log, or a thin co-sign endpoint. Restores "existed-before-T, witnessed by X" **without breaching the no-data-leaves-browser promise.** Add `verify_against_anchor(head, receipt)`.

**Until the anchor ships, the honest claim is session-integrity, not non-repudiation — and the "Part 11 / ALCOA+ / regulated" framing must be softened accordingly.** The compensating strength: deterministic numpy/scipy + fixed seed (`report_tools.py:231`) means a reviewer can *re-execute the same inputs and reproduce identical output hashes* — reproducibility-as-integrity, a stronger scientific story than an opaque server "it happened."

### (c) Cold-start / bundle + long single-threaded fits

**Bundle (verify against the pinned Pyodide release's lock):**

| Asset | ~uncompressed | ~wire (Brotli) | When |
|---|---|---|---|
| Pyodide core + numpy + pandas + SPA | ~30 MB | ~12-16 MB | **eager** (upload, schema, NCA usable) |
| **scipy** | **~30-40 MB** | **~12-18 MB** | **LAZY** (first fit) — the single dominant chunk |
| python-docx + lxml | ~5 MB | ~2 MB | lazy (first report) |

**Mitigations:** (1) self-host + Brotli-precompress all wasm/wheels, content-hashed `Cache-Control: immutable` — avoid CDN for compute assets (CSP + offline + version drift); (2) Service-Worker precache → repeat visits instant + fully offline (this *is* the DrLevy "load once" feel); (3) lazy scipy loaded on first `scipy.optimize`/`scipy.stats` action; (4) determinate progress bar with stage labels; (5) `requestIdleCallback` warm-on-idle prefetch of scipy; (6) pin+hash via `pyodide-lock.json`. UX: splash → interactive at eager-load done → scipy warms in background. Never a blank white wait.

**Long fits — the dominant *runtime* risk (not a blocker).** FOCE-I ~40s on CPython becomes **~2-5 min single-threaded WASM**; SAEM (default 300 iters, pure-Python MCMC E-step + Gauss-Newton M-step orchestrating many small `simulate()` calls, nlme.py:1151-1201) likely worse. WASM's penalty lands hardest on the *pure-Python glue* (conditional-mode loops, numeric Hessians nlme.py:440-471, outer `minimize`) — slows ~2-10×, and FOCE is glue-heavy; compiled BLAS/LAPACK only slows ~1.5-3×.
- **The fit MUST run in a Web Worker** — on the main thread the tab freezes for minutes with no progress and no cancel.
- **Latent mitigation:** closed-form compartmental models (iv/oral 1-3cmt) use the matrix-exponential fast-path (pk_simulate.py:69-81) and **avoid `solve_ivp` entirely** — stay tolerable. The painful cases are ODE-only models (Michaelis-Menten, transit, PK/PD) that hit LSODA every residual eval.
- **Submit+poll → one awaited RPC.** Pyodide compute is synchronous; a 40s fit is one blocking Worker call. Collapse submit+poll but keep `api.nlme`/`api.pollJob` *shapes* so App.tsx:1925-1961 is untouched. `onTick` runs off a UI-side timer; real iteration progress via a Comlink-proxied callback wired to `scipy.minimize(callback=…)`.
- **No server-side compute means no escape hatch for huge datasets** — MEMFS load is bounded by browser RAM. Chunk/stream very large CSVs, or keep a server tier for them (see §8).

---

## 5. WHAT GETS BETTER

- **Zero-install, no-account** — matches DrLevy's feel; a static bundle + Pyodide, load once, works offline forever after (Service Worker).
- **Data never leaves the device** — the strongest win. A real regulatory/privacy selling point (PHI never transits a server), and it's *architecturally enforced*: only the model *decision* ever egresses, only on real-LLM tiers, and never any dataframe.
- **No backend to run, scale, secure, or patch** — the entire FastAPI monolith, auth, ownership checks, and JobManager collapse. Attack surface shrinks to a static host (+ optional 50-line proxy).
- **Reproducibility-as-integrity** — deterministic compute + fixed seed lets any reviewer re-derive identical output hashes; a stronger scientific claim than a server "witness."
- **Trivial hosting** — Netlify / GitHub Pages / Cloudflare Pages. No ops, near-zero cost, infinite horizontal scale.
- **On-device audit** — sha256 chain runs locally in-Worker; nothing to trust on the wire.

---

## 6. WHAT GETS WORSE / RISKS

- **nlmixr2 independent estimator lost** (§3). Cross-engine still works native-vs-native, but the one non-PharmAgent estimator is gone unless you go WebR.
- **WASM numerical-parity — the must-prove risk.** FOCE/SAEM *functionally* port, but **you must re-run the Theophylline reference-validation suite INSIDE Pyodide** and diff against the published/CPython numbers before trusting it. Two independent sources of drift: (i) WASM float/BLAS differences in iterative optimizers; (ii) **Pyodide ships older pins** — its numpy ~2.2 / scipy ~1.14-1.16 / pandas ~2.2 vs the backend's numpy 2.4 / scipy 1.17 / pandas 3.0. pandas 3.0→2.2 (Copy-on-Write, PyArrow-string defaults) is the likeliest behavioral divergence, concentrated in `adversarial.py`/`flexplot.py`/`dataset_io.py`. **Validate against Pyodide's actual pinned versions, not the backend's.**
- **Performance** — interactive FOCE ~2-5 min, SAEM/SCM worse, serial SCM minutes-to-tens-of-minutes. Acceptable for a solo/desktop tool with good progress UX; not for high-throughput.
- **Bundle** — ~12-16 MB eager wire, +12-18 MB on first fit. Mitigated by lazy scipy + SW cache, but first-ever load is heavy.
- **No server-side compute** — huge datasets hit the browser RAM ceiling with no fallback.
- **Audit non-repudiation lost** without the external anchor (§4b) — a framing/claims risk, not just a technical one.
- **Multi-tab persistence corruption** if the write lock is skipped.

---

## 7. PHASED MIGRATION PLAN

| Phase | Scope | Exit criterion | Effort |
|---|---|---|---|
| **P0 — Numerical spike (DE-RISK FIRST)** | Run `compute/nca.py`, `compute/flexplot.py`, and **`compute/nlme.py` FOCE-I + SAEM** in a bare Pyodide (node or browser). Feed the **Theophylline reference dataset**. Diff every output hash + key estimates vs CPython. Note the exact Pyodide-pinned numpy/scipy/pandas versions. | NCA matches to ~2%; FOCE/SAEM parameter estimates + VPC coverage match the reference within the suite's existing tolerance. **If they don't match, stop and investigate before any UI work.** | **M** |
| **P1 — Tool execution in Worker, MockLLM keyless** | `pyodide.worker.ts` + `worker_api.py` dispatch shim; Comlink; port `api.ts` transport (Strategy A); MockLLM default; upload→MEMFS→`load_dataset`; drop `jobs.py`; exclude nlmixr2; serial SCM guard. In-memory state only. | Full deterministic happy-path (upload → NCA → PK fit → VPC → NLME) runs end-to-end in-browser via the UI, App.tsx unchanged, no server. | **L** |
| **P2 — Persistence + exports** | SQLite→IndexedDB blob snapshot/restore; persist CSV bytes content-addressed; build-time git-SHA + provenance constants; DOCX (`loadPackage('lxml')`) / CSV / CDISC-ZIP → Blob download. | Reload restores session + dataset; all three export formats download correctly; provenance reports real build SHA + Pyodide version. | **M** |
| **P3 — LLM path** | `BrowserLLM` await-JS bridge; async-ify call chain; wire chosen primary (proxy or BYOK) + MockLLM fallback + optional OpenAI-compat adapter. | Real-Claude chat routes/selects tools with execution staying in WASM; keyless MockLLM tier still works; graceful with no key. | **M** |
| **P4 — Packaging + hosting + audit anchor** | `app/` as pinned pure-python wheel; Service-Worker precache; Brotli + content-hash; lazy scipy + warm-on-idle; determinate load UX. Ship signed-chain export; wire hash-only timestamp anchor (or formally downgrade Part-11 claims). | Repeat visit loads offline & instant; cold start shows staged progress; audit either anchored or claims softened to "session-integrity." | **M** |

---

## 8. RECOMMENDATION

**Do not go all-in client-side yet. Adopt the hybrid path: "keep the server, add a browser-native free tier."**

Rationale: (1) the audit non-repudiation property is load-bearing for the Part-11/regulated positioning, and it *degrades* in the browser until an external anchor ships — a server-backed tier can retain true witnessed audit for paying/regulated users; (2) nlmixr2 independent cross-validation and unbounded dataset size need a server; (3) the browser-native build is the *logical endpoint* of the single-origin desktop mode that already exists (`main.py:655-701` serves `frontend/dist`) — it's additive, not a rewrite. So: ship the **same `app/` compute core** two ways — the existing FastAPI server for the regulated/heavy tier, and a Pyodide static bundle for a **zero-install, private, offline, keyless free/demo tier** that directly answers DrLevy. The compute code is shared; only the transport/host layer forks (`worker_api.py` vs `main.py`). Reassess going full-client only after P0 proves parity and the audit anchor is in place.

**The ONE thing to prototype first (P0): run `compute/nlme.py` FOCE-I + SAEM on the Theophylline reference dataset inside Pyodide and diff the numbers against CPython.** This de-risks the entire decision. Everything else — transport swaps, Blob downloads, IndexedDB, lazy loading — is mechanical and known-good. The two things that are *not* yet proven are (i) whether WASM + Pyodide's older scipy/pandas pins reproduce your validated FOCE/SAEM estimates, and (ii) whether the interactive fit latency is tolerable. Both are answered by a one-week spike with no UI. If the numbers match and a fit finishes in single-digit minutes, the port is a green light. If they don't, you've spent a week instead of a quarter finding out.
