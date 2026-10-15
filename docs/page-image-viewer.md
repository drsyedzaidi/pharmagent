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

Status codes: 400 invalid bbox / region too large / bad upload; 404 unknown
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

### Render sandbox
`python -I worker.py '<json>'` via `subprocess.Popen`: argv carries one JSON
object of typed, pre-validated values (no shell); cleared environment (no
tokens leak in); cwd = fresh empty temp dir; stdin closed; stderr discarded;
own process group. Limits:
- `RLIMIT_CPU` 10 s, `RLIMIT_AS` 1 GiB, `RLIMIT_FSIZE` 0, `RLIMIT_CORE` 0,
  `RLIMIT_NOFILE` 64 (applied inside the worker before pypdfium2 is imported).
- Wall clock 15 s then `SIGKILL` of the whole process group.
- stdout read is capped at 8 MiB; exceeding it kills the worker.
- Output pixels: each side <= 4096 and area <= 16 Mpx, checked from the page size
  BEFORE rendering (so a 14400 pt page at scale 3 is refused, not allocated) and
  again on the bitmap. A small bbox crop of a huge page is allowed.
- At most 2 concurrent renders; a third waits 5 s then gets 429.
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
| Output flooding / DoS | param ranges, stdout byte cap, concurrency semaphore, per-session quota |
| Argv / env injection | one JSON argv element of typed values; empty environment |
| Content sniffing / XSS | fixed `image/png`, `nosniff`; worker output must start with the PNG signature; frontend uses `<img>` on an object URL, never HTML |
| PDF JS / external actions / forms | never executed: rasterization only, forms not drawn |
| Cache leakage | `private, no-store` |
| Secrets via worker env | cleared environment |

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
- macOS (including the desktop app on a Mac): `RLIMIT_AS` is not enforced by
  the kernel (setrlimit either errors or is a no-op), so the memory cap does
  NOT apply there; the CPU limit, the wall-clock kill, the pixel caps and the
  output cap still do. The address-space test is therefore skipped off Linux
  (and was not executed in the development environment, which is macOS).
- Frozen/PyInstaller desktop builds: `sys.executable` is the app binary, not a
  Python interpreter, so the worker cannot be spawned. The renderer reports
  "unavailable" (503) rather than trying. Bundling needs a separate decision.
- Windows: `resource` does not exist, so no rlimits; wall-clock and output caps
  only. Not tested.

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
