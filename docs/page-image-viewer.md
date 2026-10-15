# Page-image viewer: design note (STOPPED at design, awaiting sign-off)

Status: design only. No code written. Two blockers need an owner decision (see "Blockers").

## Goal
Let a reviewer see the rendered page of a source PDF beside a value, with a
highlighted bbox, to compare. Standalone; no dependency on the evidence package
(numgrammar, normalize, parse_guard, regions, evidence store/models).

## What main actually has (checked on d68842c)
- Uploads: `POST /api/sessions/{sid}/upload` accepts `.csv` ONLY
  (`ALLOWED_UPLOAD_SUFFIXES = {".csv"}`), saved to `settings.data_dir`,
  parsed as a dataset. Only `dataset_path` is stored per session
  (`core/store.py`). There is no PDF/source-document store, model or route.
- Ownership: `owned_session` (resolves the session, 403/404) over
  `current_owner` (bearer token to hashed principal). Guarded; reuse unchanged.
- No PDF renderer in `requirements*.txt` (fastapi, pandas, numpy, scipy,
  python-docx, anthropic only). Dockerfile is python:3.13-slim, no poppler.
  `pdftoppm` exists on the dev Mac only (not a dependency, not in the image).
- Frontend has `src/shell`, `src/flexplot`, `App.tsx`; no reviewer panel
  component that shows source documents. (Review gate lives backend-side.)

## Blockers (why stopped)
1. No stored source PDF exists on main. "Reuse the existing store" is
   impossible. Options: (a) extend upload to accept `.pdf` into a session-scoped
   document store (touches the upload route and session persistence; the
   session store is near the guarded owner/audit code, so human review needed),
   or (b) a separate `documents` table. Both are a new store, which the task
   forbids doing silently. Slice is larger than described.
2. No PDF-rendering dependency. Smallest options, each needs sign-off:
   - `pypdfium2` (Apache-2/BSD wheels, bundles PDFium, no system deps, works on
     slim image + macOS arm64). RECOMMENDED. Pillow also needed for crop/PNG
     encode (or pypdfium2's own bitmap-to-PIL; Pillow is already a transitive
     dep of python-docx? No: python-docx needs lxml only, so Pillow is a second
     new dep unless cropping is done by PDFium crop box + PNG via `zlib`).
   - `pymupdf` (AGPL; license problem for a product; not recommended).
   - poppler `pdftoppm` subprocess (system package; must add to Dockerfile and
     desktop bundle; no Python dep but an OS dep).

## Proposed slice (once unblocked)
Types (pydantic, new module `app/pageimage/types.py`):
`PageLocator{page:int>=0}`, `NormBBox{x0,y0,x1,y1: float in [0,1], x0<x1, y0<y1}`.

Endpoint (read-only):
`GET /api/sessions/{sid}/documents/{doc_id}/pages/{page}.png`
query: `scale` (float, 0.5..3.0, default 1.5), optional `x0,y0,x1,y1` crop.
Auth: `Depends(owned_session)` exactly as other routes; doc looked up only
through the session's own document list (never by path from the client).
Response: `image/png`, `Cache-Control: private, no-store`,
`X-Content-Type-Options: nosniff`.
Errors: 400 invalid params, 404 unknown doc/page, 403 not owner (via
dependency), 413 output too large, 422 render failed/timeout.

Renderer: separate subprocess (`python -I -m app.pageimage.worker`), input =
file path chosen by server + validated params on argv, output PNG bytes on
stdout. Limits: `resource.setrlimit` RLIMIT_CPU (10 s), RLIMIT_AS (1 GiB),
RLIMIT_FSIZE, no network (not needed by pdfium), wall-clock timeout 15 s with
kill, cleared env, cwd = empty temp dir. Caps: page index <= 2000 and < page
count, pixel dims <= 4096 x 4096 and <= 16 Mpx, PNG <= 8 MiB, PDF input
<= 50 MiB (existing upload cap). Concurrency semaphore (2).

Frontend: `PageImage` component + `usePageImage` hook (fetch with auth header,
object-URL lifecycle with revoke, loading/error/empty states, absolutely
positioned bbox overlay in normalized coords). Ship with component tests.
Integration into the reviewer panel: list as a human sign-off item; do not
touch gate logic.

## Threat model (untrusted PDF, planned mitigations)
| Threat | Mitigation |
|---|---|
| Parser memory/CPU bomb (huge page, decompression bomb, malformed xref) | subprocess, rlimits CPU/AS/FSIZE, wall timeout + kill, pixel cap |
| Renderer exploit (native PDFium bug) | no network, cleared env, empty cwd, non-root container user; recommend seccomp/sandbox-exec later |
| Path traversal | client never supplies a path; doc_id is a server-issued uuid hex validated by regex, resolved from the session's doc list; resolved path must be `is_relative_to(data_dir)` |
| Cross-user read | `owned_session` dependency; doc must belong to that session |
| Output flooding / DoS | param ranges, byte cap on stdout read, concurrency semaphore, rate via existing admission pattern |
| Param injection into argv | floats/ints parsed and range-checked in parent; argv built from typed values only; no shell |
| Content sniffing / XSS via image | fixed `image/png`, nosniff; frontend uses object URL on `<img>`, no HTML injection |
| PDF JS / external actions / embedded files | never executed; rasterization only |
| Cache leakage | `private, no-store` |

## Deliberately out
Text extraction, OCR, region detection, value-to-bbox matching, any evidence
package integration, multi-page streaming, annotations, editing documents,
the review gate, audit/provenance writes (the endpoint is read-only and does
not write audit entries; decide whether view events must be audited).

## Security review of the render path
Paper review only (no code yet). Findings: the main residual risk is a native
PDFium vulnerability; subprocess + rlimits limit blast radius but are not a
sandbox. RLIMIT_AS is unreliable on macOS (desktop app); rely on wall timeout
there. Upload of PDFs (blocker 1) widens the attack surface: needs magic-byte
check (`%PDF-`), size cap, and session-scoped storage.

## Decisions needed
1. Approve adding PDF to uploads + a session-scoped document store (guarded
   adjacent code: human review).
2. Approve dependency: pypdfium2 (+ Pillow if crop via PIL) vs poppler.
3. Whether PDF views must be audited.
