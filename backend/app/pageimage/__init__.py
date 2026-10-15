"""Page-image viewer: session-scoped PDF documents rendered to PNG pages.

Read-only and standalone: no dependency on the evidence package. PDFs are
untrusted input; every parse/render happens in a resource-limited subprocess
(``worker.py``), never in the API process.
"""
