# Page-image viewer

Status: built (backend + frontend component). NOT wired into the reviewer panel:
the attachment point is an open owner decision.

## Goal
Let a reviewer see the rendered page of a source PDF beside a value, with a
highlighted bbox, to compare. Standalone; no dependency on the evidence package
(numgrammar, normalize, parse_guard, regions, evidence store/models).

## Owner decisions (applied)
1. PDF upload + a session-scoped document store: added as NEW routes/modules.
2. Rendering dependency: `pypdfium2==5.13.0` only (bundles PDFium; no system
   packages). No Pillow, no poppler: the PNG is encoded in the worker with
   `zlib` + `struct`; crops use pypdfium2's own `crop` render option.
3. PDF page views are NOT audited (read-only endpoint, no audit writes).
4. The component and hook ship; nothing in `ReviewPanel` or any gate changed.

## What was built

### Backend (`backend/app/pageimage/`, all new files)
| File | Role |
|---|---|
| `types.py` | pydantic `PageLocator{document_id, page}`, `NormBBox{x0,y0,x1,y1}` (each in [0,1], x0<x1, y0<y1, no NaN/inf), `RenderParams{scale, bbox}` |
| `limits.py` | every cap in one place |
| `documents.py` | session-scoped store; routes never see a client path |
| `router.py` | `build_documents_router(owned_session)` factory |
| `runner.py` | parent side of the sandbox (spawn, wall-clock kill, output cap, concurrency) |
| `worker.py` | the ONLY code that parses an untrusted PDF; stdlib + pypdfium2 only |

Registration in `app/main.py`: one import and one `app.include_router(...)` line
(placed before the SPA catch-all). `owned_session` is passed in unchanged.

### Routes (all under `owned_session`)
- `POST /api/sessions/{sid}/documents` (multipart `file`) -> 201 `{id, filename, size_bytes, pages, created_at}`
- `GET  /api/sessions/{sid}/documents` -> `{documents: [...]}`
- `GET  /api/sessions/{sid}/documents/{doc_id}` -> metadata
- `GET  /api/sessions/{sid}/documents/{doc_id}/pages/{n}.png?scale=&x0=&y0=&x1=&y1=` -> PNG
  - `scale` 0.5..3.0 (default 1.5); `x0,y0,x1,y1` normalized, top-left origin, all-or-none.
  - Headers: `Content-Type: image/png`, `X-Content-Type-Options: nosniff`,
    `Cache-Control: private, no-store`.
- The existing `POST /api/sessions/{sid}/upload` is unchanged and still CSV only
  (regression-tested, including that it still rejects `.pdf`).

Status codes: 400 invalid bbox / region too large / bad upload (including a malformed multipart body: a media type
other than exactly `multipart/form-data`, no boundary, bad framing or part headers, an unknown charset, more than one
file part, more than 4 other fields or a field over 64 KiB, or no `file` part, all answered "upload must be a
well-formed multipart/form-data body with one 'file' part" in the standard envelope, with no ERROR log; a client that
disconnects mid-body gets 400 "upload was interrupted", also without an ERROR log); 404 unknown
document or page (bad `{n}.png` shape is also 404); 409 per-session document
quota; 413 upload or output too large; 422 `scale` out of range (FastAPI
validation) or page could not be rendered; 429 renderer busy (no `Retry-After`: the app's shared HTTP error handler drops
exception headers);
503 renderer unavailable; 504 render timeout. Error bodies use the standard
envelope with fixed messages: no server path, library text, or traceback.

### Storage layout
`<data_dir>/documents/<session id>/<doc_id>.pdf` + `<doc_id>.json` metadata
sidecar. `doc_id` = server `uuid4().hex`. The client filename is never part of a
path; it is kept only as sanitized display text (basename, control chars
stripped, 120 chars). Uploads stream to `<doc_id>.part`, are validated, then
atomically renamed; a failed upload leaves nothing behind. The session id is
re-validated (`^[A-Za-z0-9_-]{1,64}$`) and the resolved directory must stay
inside `<data_dir>/documents`.

### Upload validation
`%PDF-` magic at byte 0 (the client's suffix and Content-Type are ignored in both
directions: a `.pdf` of HTML is refused, real PDF bytes named `.txt` are
accepted); 50 MiB cap (413); page count measured IN THE SANDBOX, 1..2000;
password-protected, malformed, truncated and zero-page PDFs refused (400);
20 documents per session.

**Request-body gate (step-5 item 1).** The whole request body is capped at 50 MiB + 256 KiB before FastAPI parses it
and before any dependency (auth, ownership) runs: a declared `Content-Length` over the cap is refused with 413, a
non-numeric or negative one with 400, and a body that has no length or understates it is cut off once the bytes
received pass the cap. Without this, FastAPI spools the multipart body to disk ahead of its dependencies, so a caller
that would fail auth could still make the server buffer an arbitrarily large upload. Side effect, intended: an
oversize request is answered 413 even when it would have been 401/403. The body is cut off here only while it is being
read, which now happens after authentication and reservation (see the next section). The gate lives in `GatedRoute` in
`pageimage/router.py` (new code); auth and ownership code are untouched. Tests: `test_pageimage_bodycap.py`.

### Render sandbox
`python -I worker.py '<json>'` via `subprocess.Popen`: argv carries one JSON
object of typed, pre-validated values (no shell); cleared environment (no
tokens leak in); cwd = fresh empty temp dir; stdin closed; stderr carries ONE
attestation line (below), nothing else is kept; own process group. Limits:
- `RLIMIT_CPU` 10 s (hard limit 12 s, so `SIGXCPU` arrives first), `RLIMIT_AS` 1 GiB,
  `RLIMIT_FSIZE` 0, `RLIMIT_CORE` 0, `RLIMIT_NOFILE` 64, applied inside the worker before
  pypdfium2 is imported. An inherited tighter SOFT limit is kept, never raised; the HARD
  limit is lowered to the request so worker code cannot restore it.
- **Attestation (protocol v1).** After lowering each limit the worker confirms it with
  `getrlimit` and writes exactly one JSON line to stderr: `{"pageimage_limits": {"v": 1,
  "ok": true, "limits": [{name, requested, soft, hard, state}, ...]}}`. The parent accepts an
  outcome only if there is exactly one report, version 1, `ok` true, exactly the five
  expected limits once each with the requested value we asked for, and every `enforced`
  entry confirmed (soft <= request, soft <= hard <= request, CPU hard <= request + 2). This
  is required for EVERY outcome after limits were applied and parsing began (success,
  malformed, too large, memory, crash signal, `SIGXCPU`, wall-clock kill, output overflow).
  Only a pre-parse bad request (exit 2) and a setup failure (exit 9) are mapped without it.
  Anything else is `limit_unavailable` (503). The report is an honesty check against our own
  mistakes, not a defence against a compromised worker.
- Kinds: `memory_limit` (413 on render, 400 on upload), `cpu_limit` (504 / 400), `timeout`
  (504), `limit_unavailable` (503), `failed` (422).
- Wall clock 15 s then `SIGKILL` of the whole process group.
- stdout read is capped at 8 MiB; exceeding it kills the worker.
- Output pixels: each side <= 4096 and area <= 16 Mpx, checked from the page size
  BEFORE rendering (so a 14400 pt page at scale 3 is refused, not allocated) and
  again on the bitmap. A small bbox crop of a huge page is allowed.
- **Render slots (`pageimage/slots.py`).** At most 2 renders run at once across ALL server workers sharing the data
  directory, and one session may hold at most 1 of them. They are `flock` files, so the OS frees a slot when its
  process dies. The 2 global slots are `<data_dir>/.pageimage-locks/global-<i>`; a session's slot is
  `<data_dir>/documents/<session id>/.render-<sha256 of the id>-<j>`, beside that session's document storage (no
  separate namespace; quota and cleanup ignore it). The name carries the full digest of the exact id, so two different
  sessions never share a lock (an earlier design hashed ids into 256 groups, which let a chosen id keep a victim at 429).
  A render takes its session slot first and a global slot second, so a session waiting behind its own earlier render
  holds no global slot and cannot starve anyone. Waiting is bounded: at most 8 threads per process wait (2 per
  session); beyond that a request is refused with 429 immediately, and a request that waits 5 s without a slot is
  refused with 429. If `fcntl`, the lock directory, a plain session id or a real (non-link) session directory is
  unavailable, the render is refused (503). This isolates sessions from each other. It does not rank callers (the wait
  is a poll, not a FIFO queue), and in open mode a client that can create unlimited sessions can still occupy every
  global slot. Upload validation (the page-count probe) uses the same slots, so an upload whose probe finds the slots
  busy is refused with 429 after its body has been received.
The worker re-validates its whole request and exits with a coarse code the
parent maps to a fixed message.

### Frontend (`frontend/src/pageimage/`)
- `PageImage` (hook + view), `PageImageView` (pure view), `usePageImage(locator, {scale, crop})`.
- States: `empty` (null locator), `loading`, `ready`, `error` (role=alert, text
  escaped by React; server message taken from the JSON envelope).
- Bbox overlay: percentages of the image box, so it scales with the image. When
  the image was fetched with a `crop`, pass the same `crop` and the box is
  re-expressed in the crop frame (hidden if it falls outside).
- The loader refuses to build a URL for anything the server would reject
  (bad ids, page, scale, bbox), only accepts `image/png`, aborts on unmount or
  param change, and revokes the object URL.
- Nothing imports it yet; `ReviewPanel` and the gate are untouched.

## Tests
- Backend: `tests/test_pageimage_render.py` (types, worker/runner, caps, kill
  paths), `tests/test_pageimage_api.py` (store, routes, headers, CSV regression,
  ownership), `tests/test_pageimage_security.py` (hostile uploads and filenames,
  traversal on every path segment, zero-page/encrypted/malformed/truncated PDFs,
  oversize dimensions, timeout kill with a liveness check, crashing worker,
  non-PNG worker output, cross-user access, no path leakage). PDFs are
  hand-built (`tests/pdf_fixtures.py`), including an RC4-encrypted one.
- Frontend: `npm test` (Node's built-in runner, zero new dependencies).
  `tests/pageimage/pageImageLoader.test.ts` covers URL building, bbox geometry
  and the abort/revoke lifecycle; `PageImageView.test.mjs` renders the component
  through Vite's SSR loader + `react-dom/server`.
  Limit: there is no DOM in these tests, so effects do not run. The hook is
  covered by its first paint plus the loader tests (the hook is a thin wrapper:
  `useEffect` -> `startPageImageLoad` -> `cancel`). A jsdom/Testing-Library suite
  would need new devDependencies and was deliberately not added.

## Threat model
| Threat | Mitigation |
|---|---|
| Parser memory/CPU bomb (huge page, malformed xref) | subprocess, rlimits, wall kill, pixel cap checked before allocation |
| Renderer exploit (native PDFium bug) | see "Residual risk" |
| Path traversal | client never supplies a path; `doc_id` is a server uuid hex validated by regex; session id regex + resolved-path containment; page segment must match `^\d{1,4}\.png$` |
| Cross-user read | `owned_session` (403) then a lookup confined to that session's own directory (404 for another session's doc id) |
| Output flooding / DoS | param ranges, stdout byte cap, per-session and global render slots, per-session quota |
| Argv / env injection | one JSON argv element of typed values; empty environment |
| Content sniffing / XSS | fixed `image/png`, `nosniff`; worker output must start with the PNG signature; frontend uses `<img>` on an object URL, never HTML |
| PDF JS / external actions / forms | never executed: rasterization only, forms not drawn |
| Cache leakage | `private, no-store` |
| Secrets via worker env | cleared environment |

### Storage quotas and stale-file cleanup (step-5 item 2)
**Order of an upload.** (1) the route gate caps the whole body (above); (2) `owned_session` authenticates and checks
ownership; (3) quota is reserved; (4) only then is the multipart form read, by the endpoint's own `_read_form` (Starlette's
`MultiPartParser`, driven directly so parse errors map to 400 and every spooled temp file is closed). The endpoint has no
`UploadFile` parameter, because FastAPI would parse and spool such a body before steps 2 and 3. A request refused at
step 2 or 3 reads none of its body and creates no temporary file, and concurrent bodies are bounded by quota because
each one holds a reservation while it is parsed.

All admission, heartbeat, commit and cleanup decisions run under one `flock` on `<data_dir>/documents/.quota.lock`
(`pageimage/quota.py`), so several workers sharing a data directory cannot jointly exceed a quota (a thread lock would
only cover one process). Where `fcntl` does not exist, or the lock cannot be taken within 5 s, uploads are refused.
- **Budgets:** 20 documents and 200 MiB per session, 2 GiB across all sessions (open mode allows unlimited sessions).
  Constants in `limits.py`.
- **Reservation:** an upload first writes `<id>.resv` (JSON: id, session, bytes, pid, created) holding a document slot and
  50 MiB (always the worst case, never the client-declared length) of both byte budgets; the file may not exceed that. A document's cost is
  `max(reserved, .part + .pdf)`, a complete document costs its PDF, so nothing is counted twice. If creating the
  reservation fails part-way, the partial file is removed before the upload is refused.
- **Heartbeat:** each 1 MiB chunk copied to the `.part` refreshes the reservation under the same lock cleanup uses, so
  cleanup cannot classify an upload stale and delete it after a heartbeat. A heartbeat for a reservation that was
  already reclaimed fails (409) instead of recreating its timestamp. KNOWN GAP: the copy starts only after the whole
  multipart body has been received, so nothing refreshes the reservation while the body is still arriving. A body
  slower than the stale age can lose its reservation if any other upload starts (and so runs cleanup) after that age;
  the slow upload is then refused with 409 at the first chunk of its copy. With no other upload in between it completes.
- **States:** active (fresh reservation), stale (reservation idle > 1 h), committed (`.pdf` AND `.json`), abandoned
  (pieces with no reservation).
- **Commit order:** metadata first (`.json.tmp` → `.json`), then `.part` → `.pdf` as the commit point, then the
  reservation is released (a failure there is ignored: the document is complete). A document is not visible until its
  PDF exists, so every crash point leaves either a complete document or pieces cleanup reclaims, never an inaccessible
  PDF holding quota. On a failed commit the endpoint discards the upload (reservation, `.part`, `.json.tmp`).
- **Cleanup** runs inside every reservation. It removes the pieces of an upload that never became a complete document: a
  stale reservation together with its `.part`/`.json.tmp`/`.pdf`/`.json`, and such pieces with no reservation once older than
  1 h. It never removes a complete document (`.pdf` + `.json`) and has no retention policy for those. It never follows
  a symlink and ignores any name that is not `<32 hex>.<ext>` inside a plain-named, non-link session directory.
- **Links:** a session directory that is a symlink is rejected everywhere (also when it points elsewhere inside the
  documents root); it is neither read, written, counted nor cleaned.
- **Deletions are locked too:** discarding a failed upload (its `.part`, `.json.tmp` and reservation) takes the storage
  lock, and commit, which already holds it, releases its reservation through a lock-held helper, so a scan normally
  never sees a file disappear half-way. A discard runs on the failure path, so it never raises and never waits long:
  if the lock is not free within 0.5 s, or the request is being cancelled, it deletes the pieces without the lock
  rather than leave a reservation holding a document slot and 50 MiB for an hour. That is safe because of the next
  rule. If a file disappears mid-scan (that fallback, or anything outside the app), that listing is thrown away and the
  directory is listed again; a directory that keeps changing after 3 listings fails closed (503 for every upload). A
  vanished entry never makes a session look empty. An unreadable or non-UTF-8 reservation is counted at the worst
  case (50 MiB) and is reclaimed by stale cleanup like any other. **Operators** who delete files under
  `<data_dir>/documents` by hand should stop the server or hold `.quota.lock` (flock) while they do it.
- **Fail closed:** if usage cannot be read (I/O error, lock busy, a directory that will not hold still; an unreadable
  reservation counts as 50 MiB) the upload is refused with 503; nothing is deleted to make room. An upload idle for over an hour loses its reservation and its
  commit is refused (409 "upload expired").
- **Status codes:** 409 document count or session byte budget; 503 global byte budget or accounting unavailable.
- **Not done:** retention of complete documents. (Per-session render slots and the narrowed override are described in
  "Render sandbox" and "Platform caveats".)

## Residual risk (read this)
The subprocess + rlimits limit blast radius; they are NOT a sandbox. A native
PDFium vulnerability (memory corruption) in the worker would run with the API
server's OS user and filesystem/network access: the worker has no seccomp,
namespace, chroot, or network block. PDFium is a mature, heavily fuzzed engine
and the worker only rasterizes, but this remains the main risk. Mitigations to
do for any exposed deployment: keep pypdfium2 current (CI runs `pip-audit`
but only advisorily, `|| true`), keep running as the unprivileged `pharma`
user the Docker image already uses (not the case for the desktop app), and
consider a seccomp profile / `sandbox-exec` / a dedicated no-network container
for the worker. Uploaded PDFs from untrusted parties should not be enabled until then.

### Platform caveats
- macOS (including the desktop app on a Mac): `RLIMIT_AS` is not enforced by the kernel, so
  the renderer REFUSES to run (`limit_unavailable`) unless an operator sets
  `PHARMAGENT_PAGEIMAGE_UNSAFE_ALLOW_UNENFORCED_LIMITS=true`. That is an UNSAFE operator
  override, not a development mode: nothing verifies the server is local. It is default off,
  ignored in a frozen build, and logged as a warning on every render. With it, the CPU limit,
  the wall-clock kill, the pixel caps and the output cap still apply, the memory cap does NOT.
  **The override excuses exactly one thing: a missing `RLIMIT_AS`, on a platform that cannot enforce it.** The worker
  and the parent both refuse (`limit_unavailable`) if the CPU-time, file-size, core-size or open-file limit is not
  set and confirmed, whatever the override says, and on Linux, where `RLIMIT_AS` exists, a missing `RLIMIT_AS` is
  refused even with the override. (The skipped test in the dedicated Linux CI step is the macOS-only "refuses by
  default" test; it applies only to platforms where `RLIMIT_AS` is unavailable, so it is expected to skip on Linux.)
- Frozen/PyInstaller desktop builds: the page-image routes are ABSENT (see "Feature switch"): not matched, not in
  OpenAPI, bodies never parsed. As a second guard the renderer itself also refuses there ("unavailable"), because
  `sys.executable` is the app binary, not a Python interpreter. "Frozen" means the PyInstaller bundle only: the
  checked-in `PharmAgent.app` shell launcher runs the development venv's uvicorn unfrozen, so there the protection is
  the default-off switch (plus, on macOS, the RLIMIT_AS refusal unless the unsafe override is also set).
- Windows: the feature does not work at all. There is no `fcntl`, so the storage lock cannot be taken and every upload
  is refused with 503; with no upload there is no document, so a page request is 404. The render path would refuse as
  well (no `fcntl` for the render slots, and no `resource` module for the worker's rlimits). Not tested.

## Feature switch
`PHARMAGENT_PAGEIMAGE_ENABLED` (default false). When off, or in a frozen build, the routes are
ABSENT: the router's `matches` returns no match before FastAPI reads the body, so a disabled
server never parses multipart input or spools it to disk, and `/openapi.json` omits them.

## Guarded-file sign-offs required
What has and has not been approved by a human, as of `6367117` plus this correction. Approvals given in chat are quoted
with their date; anything not listed as approved is still open.
- `backend/app/config.py` (GUARDED). PR #2 adds two settings, `pageimage_enabled` and
  `pageimage_unsafe_allow_unenforced_limits`, both default `False` (`07014c6`, `8951c65`), and their comment (corrected
  in `f0ce584`). **Approved:** the comment change only: "I approve the guarded `backend/app/config.py` comment change in
  `f0ce584`" (owner, 2026-10-19). **Not yet approved:** the two settings themselves.
- `backend/app/main.py` (guarded-adjacent). The base feature (`4d45660`) adds 4 lines: the router import and one
  `app.include_router(build_documents_router(owned_session))` before the SPA fallback; no existing line changed, and
  PR #2 does not touch the file. **No sign-off recorded.**
- `.github/workflows/ci.yml`. PR #2 makes two edits: (1) a dedicated step running `tests/test_pageimage_limits.py`
  with `-vv -rs -s` so executed tests, skips and diagnostics stay visible on Ubuntu (`9261d93`), **no sign-off
  recorded**; (2) `fetch-depth: 0` on the backend checkout so the 384 routing-baseline tests run instead of skipping
  (`6367117`), **authorized by the owner** in chat on 2026-10-19: "add the fetch-depth: 0 commit and push to #2".
- `backend/requirements.txt`: `pypdfium2==5.13.0` (base feature, `87d1f4a`); the dependency choice is owner decision 2
  above. `frontend/package.json` (`test` script) and `frontend/tsconfig*.json` (tests project added to the build
  graph): base feature, **no sign-off recorded**. Note that CI does not run the frontend `test` script.

## Notes for reviewers
- Pre-existing, not changed: dataset loading may read any file under `data_dir`;
  stored PDFs live under `data_dir/documents/`, so they are inside that allowed
  root (they are not CSVs, so parsing fails, but the path policy is broader than
  per-session). Worth tightening separately.
- Audit: page views are not audited by owner decision. If a regulator needs
  "who viewed which source page", that is a new audit event type (guarded code).
- Per-session documents are not deleted when a session is; no retention policy
  yet.
- Next step (owner): decide where the viewer attaches (reviewer panel vs
  evidence list) and whether upload needs a UI button.
