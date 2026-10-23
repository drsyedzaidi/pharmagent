# Scientific Playbooks — Design Note, Architecture, and Security Specification

## 1. Executive Summary & Naming Rationale

PharmAgent already uses the term **skills** (`backend/app/core/skills.py`) for **captured, replayable tool-call histories** recorded from completed user sessions (`Skill`, `SkillStore`, `distill_steps`, persisted in the SQLite `skills` table inside the session database).

To prevent domain collision, API ambiguity, or confusion in regulatory audit trails, this capability introduces **Scientific Playbooks**:

| Dimension | Existing Session Skills (`app.core.skills`) | Scientific Playbooks (`app.playbooks`) |
|---|---|---|
| **User-facing term** | Session Skill / Replayable Skill | **Scientific Playbook** |
| **Python domain model** | `Skill`, `SkillStore`, `distill_steps` (`backend/app/core/skills.py`) | `ScientificPlaybook`, `PlaybookSummary`, `PlaybookResource` (`backend/app/playbooks/`) |
| **Artifact location** | SQLite `skills` table (`SkillStore`, runtime session database) | `skills/<playbook-name>/SKILL.md` + `references/*` (version-controlled Agent Skills format) |
| **What it represents** | A captured, concrete sequence of past tool calls with specific dataset arguments for replay | Authored, versioned, content-addressed procedural guidance teaching an agent how to sequence existing reviewed PharmAgent tools |
| **Computes numbers?** | Re-runs recorded tool calls when replayed through the orchestrator | **Never** computes numbers or executes scripts; reviewed PharmAgent tools remain the sole numerical authority |
| **Relationship** | **Separate systems** — not renamed, replaced, migrated, or merged | **Separate systems** — read-only procedural guidance layer |

Per `AGENTS.md`, `skills/` is the canonical repository directory for `SKILL.md` definitions (`AGENTS.md` forbids creating `.claude/skills/`). Nothing is added under `.claude/skills/`.

---

## 2. External Evidence Assessment: Nova In Silico ACoP 2026 Poster M-019

Before designing or authoring any playbook, we audited the primary external source cited in the task specification:

- **Primary source inspected**: ACoP 2026 Poster `M-019` (*"Empowering AI Agents for QSP and PBPK Modeling with Open-Source Agent Skills"*, J. Bourlier et al., Nova In Silico; official poster popup `PosterID=834010`).
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

Inspection of `backend/app/tools/builtins.py` (`default_registry()`) verified **49 registered tools** across 22 `*_tools.py` modules on `origin/main` (`524d4b0c47a78bff0dfa6a67ff411e46ba76caa1`):

| Category | Verified Registered Tools (`default_registry()`, 49 total) | Backing Tool Module(s) |
|---|---|---|
| **Data Loading, Profiling & CDISC** | `load_dataset`, `profile_pk_dataset`, `generate_spaghetti_plot`, `validate_cdisc` | `data_tools.py`, `cdisc_tools.py` |
| **Non-Compartmental Analysis & Dose Proportionality** | `compute_nca`, `run_dose_proportionality`, `run_be_analysis`, `run_ddi_analysis`, `generate_be_plot` | `nca_tools.py`, `be_tools.py` |
| **Compartmental PK & PopPK Estimation** | `list_pk_models`, `fit_compartmental`, `run_poppk`, `fit_pk_model`, `run_nlme`, `run_scm`, `run_prior_check`, `run_engine_comparison` | `pk_tools.py`, `pk_model_tools.py`, `nlme_tools.py`, `scm_tools.py`, `prior_tools.py`, `engine_tools.py` |
| **Model Diagnostics, Uncertainty & Covariate Forest** | `run_diagnostics`, `run_vpc`, `run_bootstrap`, `run_sir`, `run_profile`, `run_covariate_forest`, `run_simest` | `diagnostics_tools.py`, `vpc_tools.py`, `bootstrap_tools.py`, `sir_tools.py`, `profile_tools.py`, `forest_tools.py`, `simest_tools.py` |
| **Simulation, Dose Sweeps, Special Pops & E-R** | `simulate_pk_profile`, `run_dose_sweep`, `run_clinsim`, `run_exposure_forest`, `run_special_population`, `run_individual_exposures`, `run_pediatric_simulation`, `fit_exposure_response`, `bootstrap_exposure_response`, `select_optimal_dose` | `pk_model_tools.py`, `clinsim_tools.py`, `pediatric_tools.py`, `er_tools.py`, `dose_select_tools.py` |
| **Deterministic PK Calculators** | `calc_half_life`, `calc_dose_regimen`, `calc_allometric`, `calc_accumulation`, `calc_renal_function`, `convert_concentration`, `quick_one_compartment` | `calculator_tools.py` |
| **Quality Control, Adversarial Review & Reporting** | `run_qc`, `adversarial_review`, `build_briefing_memo`, `generate_report`, `get_regulatory_guidance`, `get_Submission_checklist`, `generate_regulatory_table`, `export_regulatory_package`, `assess_program_risk`, `recommend_next_studies`, `evaluate_go_nogo` | `qc_tools.py`, `review_tools.py`, `memo_tools.py`, `report_tools.py`, `regulatory_tools.py`, `strategy_tools.py` |

### Capability Boundaries Confirmed During Discovery

1. **No PBPK or QSP Engine**: PharmAgent has **zero** PBPK or QSP solvers. No PBPK/QSP execution is claimed or implemented.
2. **NONMEM & mrgsolve are Export-Only**: `app.compute.nmexport` deterministically generates NONMEM `.ctl` (`ADVAN1/2/3/4/11`) and mrgsolve `.cpp` (`$PARAM`, `$CMT`, `$MAIN`, `$ODE`/`$PK`, `$TABLE`) code for the 6 built-in closed-form models (`iv_1cmt`, `oral_1cmt`, `oral_1cmt_lag`, `iv_2cmt`, `oral_2cmt`, `iv_3cmt`), but PharmAgent **cannot execute `nmfe` or `mrgsolve` at runtime**.
3. **Unmerged Branches & Guarded Provenance Boundary**:
   - PR #21 (PopPK physiological plausibility checker) is unmerged and is not referenced as an active runtime tool.
   - PR #17 (plot digitizer) is unmerged and excluded.
   - PR #20 (conventions run provenance) is merged on `origin/main` (`524d4b0c47a78bff0dfa6a67ff411e46ba76caa1`); to preserve guarded-file isolation, `app.playbooks` provides standalone provenance helpers (`playbook_provenance()`, `catalog_provenance_snapshot()`) without modifying guarded `provenance.py` or `pharmstate.py`.
4. **No Literature/PDF Extraction Backend**: Only `build_briefing_memo` and internal citation/audit helpers (which check internal `PharmState` keys and artifact hashes) exist. There is no PDF/HTML/XML parser, OCR engine, table extractor, or unverified-evidence staging table on `origin/main`.

---

## 4. Supported Scientific Playbooks (`skills/`)

All shipped playbooks live under the canonical `skills/` directory and keep `approval_status: pending_owner_review` until the repository owner explicitly approves their exact text:

| Playbook ID | Directory (`skills/<name>/`) | Version | Status | Approval Status | Runtime Available | Purpose |
|---|---|---|---|---|---|---|
| `pharmagent.protocol_simulation` | `skills/pharmagent-protocol-simulation/` | `1.0.0` | `reviewed` | `pending_owner_review` | `true` | 10-step clinical protocol simulation & dose-selection review across 31 allowed tools (`load_dataset`, `profile_pk_dataset`, `validate_cdisc`, `compute_nca`, `simulate_pk_profile`, `run_dose_sweep`, `run_clinsim`, `run_special_population`, `run_pediatric_simulation`, `fit_exposure_response`, `select_optimal_dose`, `run_qc`, `adversarial_review`, etc.). Includes confined reference `references/scenario-audit-template.md`. |
| `pharmagent.model_review` | `skills/pharmagent-model-review/` | `1.0.0` | `reviewed` | `pending_owner_review` | `true` | 6-step compartmental PK / PopPK model review and NONMEM/mrgsolve translation verification checklist across 22 verified modeling, diagnostic, simulation, and QC tools. Explicitly forbids cross-engine raw `OFV`/`AIC`/`BIC` comparison and documents NONMEM/mrgsolve as export-only. Includes confined reference `references/translation-verification-checklist.md`. |
| `pharmagent.literature_extraction` | `skills/pharmagent-literature-extraction/` | `0.1.0` | `draft` | `pending_owner_review` | **`false`** (disabled) | Readiness assessment and security/provenance specification for future literature evidence extraction. Fails closed (`PlaybookDisabledError`) on runtime `load_playbook()` and `read_playbook_resource()`; inspectable only via explicit review APIs (`load_playbook_for_review(..., allow_disabled=True)` and `read_playbook_resource_for_review(..., allow_disabled=True)`). Includes confined reference `references/evidence-record-schema-spec.md`. |

---

## 5. Architecture & Progressive Disclosure

To prevent prompt bloat and context pollution, `backend/app/playbooks/` implements a three-tier **progressive-disclosure** read-only API:

```
Tier 1: list_playbooks()
  └─> Returns tuple[PlaybookSummary, ...]
      (Discovery metadata, governance status, canonical_digest, allowed_tools,
       required_inputs, limitations, resource_paths — NO instructions or resource bodies.
       Lists both approved and pending_owner_review playbooks.)

Tier 2 (Runtime): load_approved_playbook(playbook_id, version=None) / load_playbook(playbook_id, version=None)
  └─> Returns ScientificPlaybook ONLY when status=="approved", approval_status=="approved",
      runtime_available is True, approved_by/approved_date are valid, version >= 1.0.0,
      and canonical_digest is verified. Rejects pending_owner_review playbooks with PlaybookApprovalError.
      Accepts NO bypass parameter (`allow_disabled` is forbidden on all runtime APIs).

Tier 2 (Human Review): load_playbook_for_review(playbook_id, version=None, allow_disabled=False)
  └─> Returns ScientificPlaybook for explicit human inspection of pending_owner_review
      (or disabled draft when allow_disabled=True) playbooks.

Tier 3 (Runtime): read_playbook_resource(playbook_id, resource)
  └─> Reads a single allowlisted static file confined inside an owner-approved playbook directory,
      verifies SHA-256 against resource_digests, and returns identity metadata.
      Accepts NO bypass parameter (`allow_disabled` is forbidden on all runtime APIs).
      (Human review counterpart: read_playbook_resource_for_review(..., allow_disabled=False))
```

Additional deterministic validation and governance entrypoints:
- `validate_playbook(playbook_dir_or_file, *, skills_root=None, registry=None) -> PlaybookValidationResult`
- `validate_all_playbooks(skills_root=None, *, registry=None) -> CatalogValidationReport`
- `authorize_playbook_tool(playbook_or_id, tool_name, *, skills_root=None, registry=None, expected_digest=None) -> bool`
- `check_tool_allowed(playbook, tool_name, *, skills_root=None, registry=None, expected_digest=None) -> bool`
  (Both `authorize_playbook_tool` and `check_tool_allowed` re-verify owner approval and canonical digest against the on-disk catalog on every call, preventing bypass via in-memory object construction/mutation or stale cached state.)
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
2. **Strict YAML Frontmatter Parsing (`yaml_loader.py` / `_StrictSafeLoader`)**:
   - Explicitly pins `PyYAML==6.0.3` (MIT license, zero sub-dependencies, zero known vulnerabilities under `pip-audit`) as a direct runtime dependency in `backend/requirements.txt` rather than relying on transitive `uvicorn[standard]` installation.
   - Subclasses `yaml.SafeLoader`.
   - Rejects duplicate YAML keys (`duplicate_yaml_key`), custom YAML tags/constructors (`!!python/...`, `!!binary`), YAML anchors (`&`), aliases (`*`), merge keys (`<<`), exceeding nesting depth (`MAX_YAML_DEPTH = 8`) (`unsafe_yaml`), and non-string mapping keys.
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
  - A playbook with `status: approved` must also have `approval_status: approved`, `runtime_available: true`, stable semantic `version >= 1.0.0`, a non-empty `approved_by`, and a valid calendar `approved_date` (`YYYY-MM-DD`).
  - Runtime APIs (`load_playbook`, `load_approved_playbook`, `read_playbook_resource`, `authorize_playbook_tool`, `check_tool_allowed`, `run_playbook_adapter`) expose **no bypass flags** (`allow_disabled` exists only on the explicit human-review functions `load_playbook_for_review` and `read_playbook_resource_for_review`).

### Approval Trust Anchor Requirement (Prerequisite Before Marking Any Playbook `approved` or Enabling Runtime Registration)

In the current unregistered MVP, all three shipped playbooks remain `approval_status: pending_owner_review` and cannot execute at runtime.

Because `approval_status`, `approved_by`, and `approved_date` are stored in `SKILL.md` frontmatter and hashed into `canonical_digest`, **frontmatter fields alone are not a standalone cryptographic trust anchor against an actor who can edit `skills/*/SKILL.md` on disk** (who could edit `SKILL.md` to set `approval_status: approved` and let the loader compute a new digest).

Before any playbook is promoted to `approved` or wired into production runtime registration (`builtins.py` / `orchestrator.py`), PharmAgent must anchor approval outside the untrusted `skills/` directory via at least one of the following controls:
1. **Trusted Python Pin Registry (Guarded Code / Manifest)**: A read-only mapping in guarded backend code (or a signed manifest outside `skills/`) that pins `(playbook_id, version) -> expected_canonical_digest` (comparable to `backend/app/core/conventions.py` digest pins) and passes `expected_digest` into `load_approved_playbook(...)` and `authorize_playbook_tool(...)`. Any edit to `SKILL.md` or `references/*`—even one claiming `approval_status: approved`—will then fail with `PlaybookSecurityError("digest_mismatch")` unless the guarded pin table is updated in an owner-approved PR.
2. **Repository CODEOWNERS & Signed-Commit Governance**: Protect both `skills/**` and the guarded digest-pin registry via `.github/CODEOWNERS` and branch-protection rules requiring signed repository-owner review before merge.

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
- `pharmagent.protocol_simulation` (`1.0.0`): `2c87b79a9907156f337734187fd569a320e59e2b781b11f50117739d90ac6ca8`
- `pharmagent.model_review` (`1.0.0`): `7f7b9d11075811f956d79d05f6ce6834c66bda413a1cd5e2a2c153d57d4db4db`
- `pharmagent.literature_extraction` (`0.1.0`, disabled spec): `236bcab23abe510e270da754d25deb49b91d65f8d5269fa76b202c30f45f639b`

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
@@ -25,6 +25,7 @@ from app.tools import (
     stats_tools,
     viz_tools,
 )
+from app.playbooks.adapter import build_playbook_tools
 from app.tools.base import ToolRegistry
@@ -37,4 +38,6 @@ def default_registry() -> ToolRegistry:
                 stats_tools, clinpharm_tools, er_tools, indirect_tools, memo_tools):
         for tool in mod.TOOLS:
             reg.register(tool)
+    for pb_tool in build_playbook_tools(registry=reg):
+        reg.register(pb_tool)
     return reg
```

#### 2. `backend/app/core/pharmstate.py` & `backend/app/core/provenance.py` (Record active playbook provenance)
```diff
--- a/backend/app/core/pharmstate.py
+++ b/backend/app/core/pharmstate.py
@@ -45,6 +45,7 @@ class PharmState(BaseModel):
     pending_tool: dict[str, Any] | None = None
+    active_playbooks: list[dict[str, Any]] = Field(default_factory=list)
```
```diff
--- a/backend/app/core/provenance.py
+++ b/backend/app/core/provenance.py
@@ -49,5 +49,6 @@ def collect_provenance(app_version: str = "0.1.0") -> dict[str, str]:
     for pkg in _PACKAGES:
         prov[pkg] = _pkg_version(pkg)
+    # Optional: attach catalog_provenance_snapshot() in session audit metadata
     return dict(prov)
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

---

## 9. Independent Review & Correction Cycle Disposition

### 9.1 Correction Cycle Blockers Addressed

| Blocker | Requirement | Resolution |
|---|---|---|
| **Blocker 1** | Split oversized files (`catalog.py`, `test_playbooks_security_and_mutation.py`) below the 800-line `AGENTS.md` limit | Refactored `backend/app/playbooks/` into 9 focused modules (`schema.py`, `yaml_loader.py`, `filesystem.py`, `digest.py`, `validation.py`, `catalog.py`, `adapter.py`, `__main__.py`, `__init__.py`; max 715 lines) and split tests into 7 focused files (`test_playbooks_catalog.py`, `test_playbooks_yaml_security.py`, `test_playbooks_filesystem_security.py`, `test_playbooks_tool_allowlist.py`, `test_playbooks_digest_and_immutability.py`, `test_playbooks_adapter_and_cli.py`, `test_playbooks_mutation_guards.py`; max 605 lines). Removed `test_playbooks_security_and_mutation.py`. |
| **Blocker 2** | Declare PyYAML as a direct runtime dependency and audit advisories | Pinned `PyYAML==6.0.3` in `backend/requirements.txt` (zero sub-dependencies, zero PyYAML advisories under `pip-audit -r backend/requirements.txt`), verified in an isolated throwaway venv without `uvicorn[standard]`, and added `test_pyyaml_declared_directly_in_requirements_txt`. |
| **Blocker 3** | Enforce that `pending_owner_review` playbooks cannot execute at runtime or authorize tools | Added `PlaybookApprovalError(PlaybookDisabledError)`, explicit review APIs (`load_playbook_for_review`, `read_playbook_resource_for_review`), and runtime APIs (`load_approved_playbook`, `authorize_playbook_tool`, plus default runtime enforcement on `load_playbook`, `read_playbook_resource`, `check_tool_allowed`, and `run_playbook_adapter`). Enforces `status=="approved"`, `approval_status=="approved"`, `runtime_available is True`, `version >= 1.0.0`, valid `approved_by`/`approved_date`, and fresh on-disk digest re-verification. |
| **Blocker 4** | Rebase-aware final verification on `origin/main` (`524d4b0c`) | Verified `git merge-base HEAD origin/main == 524d4b0c47a78bff0dfa6a67ff411e46ba76caa1`, 0 guarded files changed, unchanged canonical digests for all 3 shipped playbooks, >94% statement coverage across `app/playbooks/`, and full backend test suite green alongside merged PR #20 provenance/audit/quarantine tests. |

### 9.2 Independent Follow-Up Review Findings & Dispositions

| Finding ID | Reviewer | Severity | Summary | Disposition |
|---|---|---|---|---|
| `ARCH-C-HIGH-1` | Architecture | `high` | Unstaged deletion of old 1,500-line `backend/tests/test_playbooks_security_and_mutation.py` in Git index | **Fixed**: Staged `git rm backend/tests/test_playbooks_security_and_mutation.py` so the file is completely removed from the branch diff. |
| `TQ-C-MED-1` | Test-Quality | `medium` | `test_prompt_injection_in_instructions_and_resources_cannot_escalate_permissions` short-circuited on `PlaybookApprovalError` before reaching `undeclared_tool` and adapter load paths | **Fixed**: Tested both the `pending_owner_review` rejection (`PlaybookApprovalError`) and an `approved` injected playbook (`PlaybookSecurityError("undeclared_tool")` on all 4 forbidden tools + adapter `status=="ok"` with `writes=={}`). |
| `TQ-C-MED-2` | Test-Quality | `medium` | `forged_pb` and on-disk modification tests in `test_playbooks_tool_allowlist.py` did not separately assert `expected_digest` mismatch vs `PlaybookApprovalError` | **Fixed**: Added explicit assertions for both stale-digest forgery (`PlaybookSecurityError("digest_mismatch")`), recomputed-digest forgery (`PlaybookApprovalError`), in-memory `PlaybookSummary` tampering (`PlaybookSecurityError("digest_mismatch")`), and on-disk modification to a different approved digest (`PlaybookSecurityError("digest_mismatch")`). |
| `TQ-C-MED-3` | Test-Quality | `medium` | Module-level mutable `_VALIDATED_PLAYBOOK_ROOTS` cache in `validation.py` introduced global state | **Fixed**: Removed `_VALIDATED_PLAYBOOK_ROOTS` so `authorize_playbook_tool` and `check_tool_allowed` are 100% stateless and default strictly to `DEFAULT_SKILLS_ROOT` unless `skills_root` is explicitly provided. |
| `SEC-C-LOW-1` / `ARCH-C-LOW-1` | Security / Architecture | `low` | `MAX_YAML_DEPTH` was `16` in `yaml_loader.py` vs `8` in documentation | **Fixed**: Set `MAX_YAML_DEPTH = 8` in `backend/app/playbooks/yaml_loader.py`. |
| `SEC-C-LOW-2` | Security | `low` | `load_playbook_for_review` and `read_playbook_resource_for_review` did not forward/check `expected_digest` | **Fixed**: Added `expected_digest` verification to `load_playbook_for_review` and `read_playbook_resource_for_review`, and removed `allow_disabled` from runtime `load_playbook` and `read_playbook_resource`. |
| `SEC-C-LOW-3` | Security | `low` | `authorize_playbook_tool` checked `allowed_tools` equality rather than full `PlaybookSummary` equality against `on_disk.to_summary()` | **Fixed**: Enforced `on_disk.to_summary() == expected_summary` in `authorize_playbook_tool` and added a test for tampered `PlaybookSummary` metadata. |
| `TQ-C-LOW-1` / `TQ-C-LOW-2` | Test-Quality | `low` | Direct test coverage for `_scan_tools_and_links` zero-width stripping, `approved` + `runtime_available=False`, and `_read_utf8_bounded` defensive branches | **Fixed**: Added direct assertions in `test_playbooks_filesystem_security.py`, `test_playbooks_catalog.py`, and `test_playbooks_tool_allowlist.py`. |
| `ARCH-C-LOW-2` | Architecture | `low` | `docs/scientific-playbooks.md` referenced pre-rebase `origin/main` SHA and unmerged PR #20 | **Fixed**: Updated documentation to cite `origin/main` `524d4b0c47a78bff0dfa6a67ff411e46ba76caa1` (PR #20 merged). |
