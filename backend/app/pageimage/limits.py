"""Hard caps for the page-image viewer (single source of truth for the parent
process; the worker receives them in its request and re-checks them)."""
from __future__ import annotations

MAX_PDF_BYTES = 50 * 1024 * 1024      # same cap as the CSV upload
MULTIPART_OVERHEAD_BYTES = 256 * 1024  # boundary lines, part headers and filename around the file bytes
MAX_PAGES = 2000                      # page-count cap and page-index ceiling
MAX_DOCS_PER_SESSION = 20
MAX_SESSION_BYTES = 200 * 1024 * 1024        # committed + in-flight PDF bytes per session
MAX_GLOBAL_BYTES = 2 * 1024 * 1024 * 1024    # the same across ALL sessions (open mode allows unlimited sessions)
STALE_SECONDS = 3600                         # an in-flight upload idle this long (no chunk, no heartbeat) is abandoned
LOCK_WAIT_SECONDS = 5.0                      # wait for the storage lock before failing closed
MAX_BODY_BYTES = MAX_PDF_BYTES + MULTIPART_OVERHEAD_BYTES  # whole request body, checked before auth and parsing

MIN_SCALE = 0.5
MAX_SCALE = 3.0
DEFAULT_SCALE = 1.5
MAX_DIM_PX = 4096                     # per output side
MAX_PIXELS = 16_000_000               # output area
MAX_PNG_BYTES = 8 * 1024 * 1024       # encoded output (also the stdout read cap)

CPU_SECONDS = 10                      # RLIMIT_CPU inside the worker
ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024  # RLIMIT_AS (not enforced on macOS)
NOFILE_LIMIT = 64                     # RLIMIT_NOFILE inside the worker
WALL_SECONDS = 15.0                   # parent kills the worker after this
QUEUE_WAIT_SECONDS = 5.0              # wait for a render slot before 429
MAX_CONCURRENT_RENDERS = 2           # across ALL processes sharing the data directory (flock slots)
MAX_RENDERS_PER_SESSION = 1          # one session never holds more than this many of those slots
MAX_WAITING_RENDERS = 8              # per process: threads allowed to block waiting for a slot; more get 429 at once
MAX_WAITING_PER_SESSION = 2          # per process: of those, how many one session may occupy
SLOT_POLL_SECONDS = 0.01
