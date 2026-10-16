"""Hard caps for the page-image viewer (single source of truth for the parent
process; the worker receives them in its request and re-checks them)."""
from __future__ import annotations

MAX_PDF_BYTES = 50 * 1024 * 1024      # same cap as the CSV upload
MAX_PAGES = 2000                      # page-count cap and page-index ceiling
MAX_DOCS_PER_SESSION = 20

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
MAX_CONCURRENT_RENDERS = 2
