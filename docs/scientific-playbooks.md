# Scientific Playbooks — Design Note, Architecture, and Security Specification

## 1. Executive Summary & Naming Rationale

PharmAgent already uses the term **skills** (`backend/app/core/skills.py`) for **captured, replayable tool-call histories** recorded from completed user sessions (`Skill`, `SkillStep`, `extract_skill_from_session`, `prepare_skill_replay`, persisted under `data/skills/*.json`).

To prevent domain collision, API ambiguity, or confusion in regulatory audit trails, this capability introduces **Scientific Playbooks**:

| Dimension | Existing Session Skills (`app.core.skills`) | Scientific Playbooks (`app.playbooks`) |
|---|---|---|
| **User-facing term** | Session Skill / Replayable Skill | **Scientific Playbook** |
| **Python domain model** | `Skill`, `SkillStep` (`backend/app/core/skills.py`) | `ScientificPlaybook`, `PlaybookSummary`, `PlaybookResource` (`backend/app/playbooks/`) |
| **Artifact location** | `data/skills/*.json` (runtime session storage) | `skills/<playbook-name>/SKILL.md` + `references/*` (version-controlled Agent Skills format) |
| **What it represents** | A captured, concrete sequence of past tool calls with specific dataset arguments for replay | Authored, versioned, content-addressed procedural guidance teaching an agent how to sequence existing reviewed PharmAgent tools |
| **Computes numbers?** | Re-runs recorded tool calls when replayed through the orchestrator | **Never** computes numbers or executes scripts; reviewed PharmAgent tools remain the sole numerical authority |
| **Relationship** | **Separate systems** — not renamed, replaced, migrated, or merged | **Separate systems** — read-only procedural guidance layer |

Per `AGENTS.md`, `skills/` is the canonical repository directory for `SKILL.md` definitions (`AGENTS.md` forbids creating `.claude/skills/`). Nothing is added under `.claude/skills/`.

---

## 2. External Evidence Assessment: Nova In Silico ACoP 2026 Poster M-019

Before designing or authoring any playbook, we audited the primary external source cited in the task specification:

- **Primary source inspected**: ACoP 2026 Poster `M-019` (*"Empowering AI Agents for QSP and PBPK Modeling with Open-Source Agent Skills"*, J. Bourlier, S.aiser, P.ert, A.ert, Nova In Silico; official poster popup `PosterID=834010`).
- **Findings**:
  1. The official poster abstract describes a modular collection of domain-specific skills built on the open `agentskills.io` format (`SKILL.md` with YAML frontmatter + Markdown instructions + progressive disclosure), organized across literature extraction, model building, trial simulation, and analysis/review.
  2. However, the poster's sole URL link points to `https://agentskills.io/specification` (the generic Agent Skills specification), **not** to an open-source QSP/PBPK modeling skill repository.
  3. A verification search of Nova In Silico's public GitHub organization (`https://github.com/novainsilico`) identified `https://github.com/novainsilico/jinko-skills` (MIT License, releases `0.1.0` through `1.12.3`), which packages operational skills (`jinko`, `pbpk`, `project-setup`, `literate-modeling`) that wrap Nova's proprietary Jinkō SaaS API (`jinko-sdk` / `jinko.nova-insilico.ai`) and require Jinkō authentication and server-side engines. No standalone, platform-independent Nova M-019 PBPK/QSP Agent Skills library for external pharmacometric engines exists.
- **Compliance & Attribution Policy**:
  - **No Nova skill files were downloaded, copied, or adapted.** All PharmAgent playbooks are authored from scratch against PharmAgent's verified tool catalog.
  - **Nova M-019 is cited strictly as high-level design inspiration** for applying progressive-disclosure `SKILL.md` playbooks to pharmacometric workflows.
  - **Self-reported productivity figures**: The `30–50%` time-savings and `~50%` productivity figures mentioned in the Nova M-019 poster are **self-reported internal team estimates from Nova In Silico**, **not controlled benchmark results**, and must never be cited as empirical or benchmarked performance claims for PharmAgent.

---

## 3. Phase 0 Discovery: Verified PharmAgent Tool Inventory (`origin/main`)

Inspection of `backend/app/tools/builtins.py` (`default_registry()`) verified **49 registered tools** across 7 specialist agents on `origin/main` (`5d099fd576e06dd6d18553202f1a0d87cd5729ec`):

| Category | Verified Registered Tools (`default_registry()`) | Runtime Engine / Backing Module |
|---|---|---|
| **Data Loading & Profiling** | `load_dataset`, `profile_pk_dataset`, `validate_cdisc`, `normalize_units`, `Flag_BLQ`, `export_272_dataset` | `app.compute.cdisc`, `app.compute.units` |
| **Non-Compartmental Analysis (NCA)** | `compute_nca`, `run_bioequivalence`, `generate_nca_table`, `plot_conc_time`, `run_superposition` | `app.compute.nca` (deterministic trapezoidal NCA) |
| **PK / PopPK Modeling & Covariates** | `list_pk_models`, `fit_compartmental`, `run_poppk`, `fit_pk_model`, `run_nlme`, `run_scm`, `compare_models`, `run_engine_comparison`, `compute_ebe`, `generate_pk_summary` | `app.compute.pk`, `app.compute.nlme` (`scipy_nlme`, `saem_npde`, `nlmixr2_r`, `map_ebe`, `naive_pooled`), `app.compute.scm` |
| **Model Diagnostics & Plots** | `run_diagnostics`, `run_vpc`, `run_bootstrap`, `run_npde`, `run_sir`, `run_llp`, `run_covariate_forest`, `plot_goodness_of_fit`, `plot_vpc`, `plot_individual_fits` | `app.compute.diagnostics`, `app.compute.forest` |
| **Simulation, Dose Sweeps & E-R** | `simulate_pk_profile`, `run_dose_sweep`, `run_clinsim`, `run_special_population`, `run_pediatric_simulation`, `fit_exposure_response`, `select_optimal_dose`, `recommend_dosing_regimen` | `app.compute.simulation`, `app.compute.clinsim`, `app.compute.pediatric`, `app.compute.er` |
| **Quality Control & Review** | `run_qc`, `verify_citations`, `check_regulatory_compliance`, `adversarial_review`, `explain_Concept` | `app.compute.qc`, `app.core.citations` |
| **Regulatory & Report Exports** | `generate_272`, `generate_CSR_section`, `generate_model_report`, `export_report_bundle`, `draft_executive_summary` | `app.compute.nmexport` (`build_nonmem`, `build_mrgsolve` for 6 closed-form models: `iv_1cmt`, `oral_1cmt`, `oral_1cmt_lag`, `iv_2cmt`, `oral_2cmt`, `iv_3cmt`), `app.compute.reports` |

### Capability Boundaries Confirmed During Discovery

1. **No PBPK or QSP Engine**: PharmAgent has **zero** PBPK or QSP solvers. No PBPK/QSP execution is claimed or implemented.
2. **NONMEM & mrgsolve are Export-Only**: `app.compute.nmexport` deterministically generates NONMEM `.ctl` (`ADVAN1/2/3/4/11`) and mrgsolve `.cpp` (`$PARAM`, `$CMT`, `$MAIN`, `$ODE`/`$PK`, `$TABLE`) code for the 6 built-in closed-form models, but PharmAgent **cannot execute `nmfe` or `mrgsolve` at runtime**.
3. **Unmerged Branches Excluded**:
   - PR #21 (PopPK physiological plausibility checker) is unmerged and is not referenced as an active runtime tool.
   - PR #17 (plot digitizer) is unmerged and excluded.
   - PR #20 (conventions run provenance) is unmerged; `app.playbooks` provides standalone provenance helpers (`playbook_provenance()`, `catalog_provenance_snapshot()`) without modifying guarded `provenance.py` or `pharmstate.py`.
4. **No Literature/PDF Extraction Backend**: Only `verify_citations` (which checks internal `PharmState` keys / artifact hashes / `NCT` / `PMID` strings) exists. There is no PDF/HTML/XML parser, OCR engine, table extractor, or unverified-evidence staging table on `origin/main`.

---

## 4. Supported Scientific Playbooks (`skills/`)

All shipped playbooks live under the canonical `skills/` directory and keep `approval_status: pending_owner_review` until the repository owner explicitly approves their exact text:

| Playbook ID | Directory (`skills/<name>/`) | Version | Status | Approval Status | Runtime Available | Purpose |
|---|---|---|---|---|---|---|
| `pharmagent.protocol_simulation` | `skills/pharmagent-protocol-simulation/` | `1.0.0` | `reviewed` | `pending_owner_review` | `true` | 10-step clinical protocol simulation & dose-selection review using `load_dataset`, `profile_pk_dataset`, `validate_cdisc`, `compute_nca`, `simulate_pk_profile`, `run_dose_sweep`, `run_clinsim`, `run_special_population`, `run_pediatric_simulation`, `fit_exposure_response`, `select_optimal_dose`, `run_qc`, and `adversarial_review`. Includes confined reference `references/scenario-audit-template.md`. |
| `pharmagent.model_review` | `skills/pharmagent-model-review/` | `1.0.0` | `reviewed` | `pending_owner_review` | `true` | 10-step compartmental PK / PopPK model review and NONMEM/mrgsolve translation verification checklist using PharmAgent's 17 verified modeling, diagnostic, simulation, and QC tools. Explicitly forbids cross-engine raw `OFV`/`AIC`/`BIC` comparison and documents NONMEM/mrgsolve as export-only. Includes confined reference `references/translation-verification-checklist.md`. |
| `pharmagent.literature_extraction` | `skills/pharmagent-literature-extraction/` | `0.1.0` | `draft` | `pending_owner_review` | **`false`** (disabled) | Readiness assessment and security/provenance specification for future literature evidence extraction. Fails closed (`PlaybookDisabledError`) on runtime `load_playbook()` and `read_playbook_resource()` unless `allow_disabled=True` is passed for explicit specification review. Includes confined reference `references/evidence-record-schema-spec.md`. |

---

## 5. Architecture & Progressive Disclosure

To prevent prompt bloat and context pollution, `backend/app/playbooks/` implements a three-tier **progressive-disclosure** read-only API:

```
Tier 1: list_playbooks()
  └─> Returns tuple[PlaybookSummary, ...]
      (Discovery metadata, governance status, canonical_digest, allowed_tools,
       required_inputs, limitations, resource_paths — NO instructions or resource bodies)

Tier 2: load_playbook(playbook_id, version=None)
  └─> Returns ScientificPlaybook
      (Validated instructions body + full identity/governance metadata + untrusted_data=True;
       blocks disabled playbooks unless allow_disabled=True; never executes scripts)

Tier 3: read_playbook_resource(playbook_id, resource)
  └─> Returns PlaybookResource
      (Reads a single allowlisted static file confined inside the playbook directory,
       verifies SHA-256 against the playbook's resource_digests, and returns identity metadata)
```

Additional deterministic validation and governance entrypoints:
- `validate_playbook(playbook_dir_or_file, *, skills_root=None, registry=None) -> PlaybookValidationResult`
- `validate_all_playbooks(skills_root=None, *, registry=None) -> CatalogValidationReport`
- `check_tool_allowed(playbook, tool_name, *, registry=None) -> bool`
- `playbook_provenance(playbook_id, ...) -> PlaybookProvenance`
- `catalog_provenance_snapshot(skills_root=None, ...) -> MappingProxyType[str, Any]`
- CLI check command:
  ```bash
  python -m app.playbooks --check
  python -m app.playbooks --check --format json
  ```

### Why Reviewed Tools Remain Authoritative for Numbers

1. **Agents decide; reviewed tools execute.** A Scientific Playbook is procedural text that guides tool selection, scenario matrix design, and diagnostic verification.
2. **Zero numerical fabrication.** Playbooks cannot perform arithmetic, integrate ODEs, convert units silently, or impute missing data. Every numerical value in a PharmAgent run must originate from a deterministic PharmAgent tool execution recorded in `AuditChain` or be explicitly labelled `UNRESOLVED`.

---

## 6. Security, Confinement & Trust Model

All playbook files (`SKILL.md` and `references/*`) are treated as **untrusted data**, never trusted code:

1. **Deep Immutability**:
   - All models (`PlaybookSummary`, `ScientificPlaybook`, `PlaybookResource`, `PlaybookCompatibility`, `PlaybookValidationIssue`, `PlaybookValidationResult`, `CatalogValidationReport`, `PlaybookProvenance`) use `ConfigDict(extra="forbid", frozen=True)` with `tuple` collections.
   - `to_metadata_mapping()` and `freeze_value()` recursively wrap dictionaries in `types.MappingProxyType` and sequences in `tuple`. Callers cannot mutate cached or returned state.
2. **Strict YAML Frontmatter Parsing (`_StrictSafeLoader`)**:
   - Subclasses `yaml.SafeLoader`.
   - Rejects duplicate YAML keys (`duplicate_yaml_key`), custom YAML tags/constructors (`!!python/...`, `!!binary`), YAML anchors (`&`) and aliases (`*`) (`unsafe_yaml`), and non-string mapping keys.
   - Disables implicit YAML timestamp conversion so calendar dates are validated deterministically as ISO `YYYY-MM-DD` strings.
3. **Filesystem Confinement & Size Caps**:
   - `SKILL.md` capped at `128 KiB` (`MAX_SKILL_MD_BYTES`); resource files capped at `64 KiB` (`MAX_RESOURCE_BYTES`); adapter payloads capped at `64 KiB` (`MAX_ADAPTER_OUTPUT_BYTES`).
   - Rejects path traversal (`..`, `.`, backslashes `\`), absolute paths (`/etc/...`, `C:\...`), remote or local URIs (`http://`, `https://`, `file://`, `ftp://`, `data:`), symlinks anywhere in `skills_root` or playbook subdirectories (`symlink_escape`), hidden dotfiles/dotdirs (`hidden_file`), null bytes, and invalid UTF-8 (`invalid_encoding`).
4. **No Bundled Scripts or Executables in the MVP**:
   - Rejects `scripts/`, `bin/`, and `__pycache__/` directories.
   - Rejects `.py`, `.sh`, `.bash`, `.zsh`, `.js`, `.ts`, `.r`, `.rb`, `.pl`, `.php`, `.exe`, `.bin`, `.bat`, `.ps1`, `.wasm`, `.so`, `.dylib`, `.dll`, any file with executable mode bits (`st_mode & 0o111`), and any resource starting with a `#!` shebang (`executable_or_script_file`).
   - Static resources are restricted to allowlisted extensions: `.md`, `.txt`, `.json`, `.csv`, `.yaml`, `.yml`.
   - AST inspection tests (`test_mutation_guard_4_allow_script_execution_is_caught`) verify that `app.playbooks` never imports `subprocess`, `pty`, `shlex`, or `ctypes` and never calls `eval`, `exec`, `compile`, or `os.system`.
5. **Tool Registry Verification & Per-Playbook Allowlisting**:
   - Every tool listed in `allowed_tools` must exist in the active `ToolRegistry` (`non_existent_tool` rejected).
   - Both `SKILL.md` and all confined resource files are scanned for references to registered tool names outside `allowed_tools` (`undeclared_tool` rejected).
   - `check_tool_allowed(playbook, tool_name)` enforces both registry membership and per-playbook allowlist membership at runtime.
6. **Prompt-Injection Containment**:
   - Loaded playbooks and resources carry `untrusted_data: True` and `trust_notice`.
   - Adversarial text inside `SKILL.md` or `references/*` cannot alter `approval_status`, `requires_human_review`, `allowed_tools`, `PharmState` (`writes={}` in the adapter), `ToolRegistry` agent ownership (`CrossAgentToolError`), or expensive-tool admission control (`ExpensiveToolError`).

---

## 7. Approval Lifecycle & Canonical Digest Reproduction

### Approval Lifecycle

- Allowed `status` values: `draft`, `reviewed`, `approved`, `deprecated`.
- Allowed `approval_status` values: `pending_owner_review`, `approved`, `rejected`.
- **Governance invariants**:
  - `requires_human_review` must always be `true`.
  - All new or modified playbooks default to `approval_status: pending_owner_review` and must not set `approved_by` or `approved_date`.
  - A playbook with `status: approved` must also have `approval_status: approved`, a non-empty `approved_by`, and a valid calendar `approved_date` (`YYYY-MM-DD`).

### Reproducing the Canonical Digest (`canonical_digest`)

Every playbook's `canonical_digest` is a deterministic SHA-256 hex digest over normalized frontmatter metadata, normalized Markdown instructions, and the sorted SHA-256 digests of all confined resource files:

1. Normalize all text newlines (`\r\n` and `\r` $\rightarrow$ `\n`).
2. For each confined resource file under the playbook directory (excluding `SKILL.md`), compute `sha256(normalized_utf8_bytes)` and sort pairs ascending by POSIX relative path (`[{"path": rel_posix, "sha256": hex_digest}, ...]`).
3. Construct the canonical payload dictionary:
   ```json
   {
     "digest_version": 1,
     "instructions": "<normalized_trimmed_markdown_body>",
     "metadata": {
       "allowed_tools": ["..."],
       "approval_status": "pending_owner_review",
       "approved_by": null,
       "approved_date": null,
       "compatibility": {
         "engines": ["..."],
         "min_version": "0.1.0",
         "notes": "...",
         "product": "PharmAgent"
       },
       "description": "...",
       "domain": "...",
       "license": "...",
       "limitations": ["..."],
       "name": "...",
       "playbook_id": "...",
       "produced_artifacts": ["..."],
       "required_inputs": ["..."],
       "requires_human_review": true,
       "runtime_available": true,
       "source": "...",
       "status": "...",
       "version": "..."
     },
     "resources": [
       {"path": "references/...", "sha256": "..."}
     ]
   }
   ```
4. Serialize with `json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)` and compute `hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()`.

You can reproduce and verify all digests via:
```bash
cd backend
python -m app.playbooks --check --format json
```

Current shipped digests:
- `pharmagent.protocol_simulation` (`1.0.0`): `4a8ab71ef4941f964e2e362bb412c02b926787424d587ac4ac0cb112645a6fb7`
- `pharmagent.model_review` (`1.0.0`): `5e99d8a20d6dc6281c2b89b2d3eb2977149a0148cfef1c01216c357262d61556`
- `pharmagent.literature_extraction` (`0.1.0`, disabled spec): `0c4992eaded92678c0925cb90f5623338f605d5a7c0b7a98bc7874b68c0834eb`

---

## 8. Deferred Integrations & Exact Guarded Diff Specification

### 8.1 Deferred Literature Ingestion Backend

`skills/pharmagent-literature-extraction/SKILL.md` (`runtime_available: false`, `status: draft`) specifies the full schema, provenance, unit-preservation, OCR-confidence, prompt-injection sanitation, and human-verification requirements for literature extraction. Runtime activation is deferred until PharmAgent adds:
1. A sandboxed, offline-capable document parser (`pdfplumber`/`pypdf` + structured XML/HTML reader) with strict byte/page/timeout bounds;
2. An isolated `unverified_literature_extractions` staging table that cannot write directly to `PharmState` modeling inputs; and
3. A human reviewer sign-off endpoint that promotes verified records into analysis datasets.

### 8.2 Deferred Runtime Registration (Guarded Files Not Modified in MVP)

In compliance with the MVP boundary, **zero guarded files were modified**. The tool adapter (`backend/app/playbooks/adapter.py`, `build_playbook_tools()`, `run_playbook_adapter()`) is tested in complete isolation.

When the repository owner approves wiring `inspect_scientific_playbook` and playbook provenance into production routing, the exact minimal guarded diffs required are:

#### 1. `backend/app/tools/builtins.py` (Register read-only tool in `default_registry()`)
```diff
--- a/backend/app/tools/builtins.py
+++ b/backend/app/tools/builtins.py
@@ -16,6 +16,7 @@ from app.tools.base import Tool, ToolContext, ToolRegistry, ToolResult
+from app.playbooks.adapter import build_playbook_tools
 
 def default_registry() -> ToolRegistry:
     reg = ToolRegistry()
@@ -1520,5 +1521,7 @@ def default_registry() -> ToolRegistry:
     for t in _ build_all_tools():
         reg.register(t)
+    for pb_tool in build_playbook_tools(registry=reg):
+        reg.register(pb_tool)
     return reg
```

#### 2. `backend/app/core/pharmstate.py` & `backend/app/core/provenance.py` (Record active playbook provenance)
```diff
--- a/backend/app/core/pharmstate.py
+++ b/backend/app/core/pharmstate.py
@@ -88,6 +88,7 @@ class PharmState(BaseModel):
     active_conventions: list[dict[str, Any]] = Field(default_factory=list)
+    active_playbooks: list[dict[str, Any]] = Field(default_factory=list)
```
```diff
--- a/backend/app/core/provenance.py
+++ b/backend/app/core/provenance.py
@@ -45,6 +45,7 @@ class RunProvenance(BaseModel):
     dataset_sha256: str | None = None
+    playbooks_used: list[dict[str, Any]] = Field(default_factory=list)
```

#### 3. `backend/app/main.py` (Optional read-only discovery endpoint)
```diff
--- a/backend/app/main.py
+++ b/backend/app/main.py
@@ -30,6 +30,7 @@ from app.playbooks import list_playbooks
+@app.get("/api/playbooks")
+def get_scientific_playbooks(include_disabled: bool = False):
+    return [s.model_dump() for s in list_playbooks(include_disabled=include_disabled)]
```

None of the diffs above have been applied in this branch; `git diff origin/main -- <guarded_files>` is completely empty.
