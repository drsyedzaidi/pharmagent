"""mikebrain — a literature-grounded, advisory-only knowledge index.

The owner supplies pharmacometrics books, papers and guidelines; nothing is
shipped in the repository. Documents are recorded in a content-addressed
manifest (``manifest.py``), chunked and retrieved by hybrid BM25 + vector
search (``index.py``), and every answer the critic tool gives is validated
against the retrieved text by a deterministic quote-grounding rule
(``grounding.py``). The corpus lives under ``<data_dir>/knowledge/`` and is
never committed.

mikebrain refutes or corroborates; it never decides. See docs/mikebrain.md.
"""
