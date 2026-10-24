"""Hybrid retrieval over the corpus: BM25 + vector similarity fused by RRF.

Ported minimally from the owner's rag-pharma ``hybrid_search``: each retriever
ranks its own top-N candidate pool and the pools are fused by Reciprocal Rank
Fusion over their UNION, so an exact-term hit (parameter name, drug, method)
that embeddings rank low still surfaces, and a chunk absent from one retriever's
pool simply contributes nothing from that retriever instead of a punishing
worst-case rank.

The vector side uses Ollama's ``nomic-embed-text`` over plain HTTP when a local
Ollama is reachable. When it is not (CI, keyless tests, a laptop without Ollama)
retrieval still works BM25-only and the result says so: ``embedding:
"unavailable"``. Embeddings are cached per document under ``<root>/embeddings``
keyed by chunk hash, so re-indexing never re-embeds unchanged text.

Everything here is deterministic given the corpus and the embedder's output;
ties are broken by chunk id.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import requests

from app.knowledge.chunking import Chunk, chunk_document
from app.knowledge.manifest import Manifest, load_manifest, load_text

log = logging.getLogger("pharmagent")

EMBED_MODEL = "nomic-embed-text"
EMBED_DIR = "embeddings"
EMBED_MAX_CHARS = 1800        # nomic's effective context (~2048 tokens); longer input is truncated
EMBED_BATCH = 32
PROBE_TIMEOUT_S = 2.0
EMBED_TIMEOUT_S = 120.0
EMBEDDING_UNAVAILABLE = "unavailable"

RRF_K = 60                    # standard RRF constant; insensitive at this scale
POOL_FACTOR = 10              # per-retriever candidate pool = max(k * POOL_FACTOR, POOL_MIN)
POOL_MIN = 50
DEFAULT_K = 6
MAX_K = 12
BM25_K1 = 1.5
BM25_B = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ── BM25 (Okapi) ────────────────────────────────────────────────────────────────
class BM25:
    """Okapi BM25 over pre-tokenised documents. Pure Python; corpora here are a few
    thousand chunks at most, so a full-corpus score is milliseconds."""

    def __init__(self, docs: list[list[str]], k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avgdl = (sum(self.lens) / self.n) if self.n else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.idf = {t: math.log(1.0 + (self.n - n_t + 0.5) / (n_t + 0.5)) for t, n_t in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = [0.0] * self.n
        if not self.n or not query:
            return out
        for i, tf in enumerate(self.tfs):
            denom_norm = self.k1 * (1.0 - self.b + self.b * self.lens[i] / self.avgdl) if self.avgdl else self.k1
            s = 0.0
            for t in query:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1.0) / (f + denom_norm)
            out[i] = s
        return out


# ── embedders ───────────────────────────────────────────────────────────────────
class Embedder(Protocol):
    name: str
    def available(self) -> bool: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class NoEmbedder:
    """Vector side switched off: retrieval is BM25-only."""
    name = EMBEDDING_UNAVAILABLE

    def available(self) -> bool:
        return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embeddings unavailable")


class OllamaEmbedder:
    """``nomic-embed-text`` through a local Ollama server's native API."""

    def __init__(self, base_url: str, model: str = EMBED_MODEL) -> None:
        url = base_url.rstrip("/")
        self.root = url[:-3] if url.endswith("/v1") else url
        self.model = model
        self.name = f"ollama:{model}"

    def available(self) -> bool:
        try:
            r = requests.get(f"{self.root}/api/tags", timeout=PROBE_TIMEOUT_S)
            if r.status_code != 200:
                return False
            names = [m.get("name", "") for m in (r.json().get("models") or [])]
        except (requests.RequestException, ValueError):
            return False
        return any(n == self.model or n.startswith(self.model + ":") for n in names)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = [t[:EMBED_MAX_CHARS] for t in texts[i:i + EMBED_BATCH]]
            r = requests.post(f"{self.root}/api/embed", json={"model": self.model, "input": batch},
                              timeout=EMBED_TIMEOUT_S)
            if r.status_code >= 400:
                raise RuntimeError(f"embedding failed: HTTP {r.status_code} from {self.root}/api/embed")
            vecs = r.json().get("embeddings")
            if not isinstance(vecs, list) or len(vecs) != len(batch):
                raise RuntimeError("embedding server returned a malformed response")
            out.extend([float(x) for x in v] for v in vecs)
        return out


def default_embedder() -> Embedder:
    """Ollama on this machine. A hosted LLM base URL is irrelevant here: embeddings
    are only ever fetched from a loopback server."""
    from app.config import settings
    from app.core.llm import is_loopback
    url = settings.llm_base_url if settings.llm_base_url and is_loopback(settings.llm_base_url) \
        else settings.DEFAULT_OLLAMA_URL
    return OllamaEmbedder(url)


# ── results ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_sha: str
    chunk_hash: str
    title: str
    section: str
    offsets: tuple[int, int]
    text: str
    score: float
    bm25_rank: int | None
    vector_rank: int | None

    def to_dict(self, *, with_text: bool = True) -> dict[str, Any]:
        d = asdict(self)
        d["offsets"] = list(self.offsets)
        if not with_text:
            d.pop("text")
        return d


@dataclass(frozen=True)
class SearchResult:
    query: str
    hits: tuple[Hit, ...]
    embedding: str                 # embedder name, or "unavailable"
    corpus_version: str
    n_chunks: int
    embedding_error: str | None = None

    def to_dict(self, *, with_text: bool = True) -> dict[str, Any]:
        return {"query": self.query, "hits": [h.to_dict(with_text=with_text) for h in self.hits],
                "embedding": self.embedding, "corpus_version": self.corpus_version,
                "n_chunks": self.n_chunks, "embedding_error": self.embedding_error}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ── the index ───────────────────────────────────────────────────────────────────
class CorpusIndex:
    """All chunks of the corpus under ``root`` with BM25 and (optional) cached vectors."""

    def __init__(self, root: Path, manifest: Manifest, chunks: tuple[Chunk, ...]) -> None:
        self.root = Path(root)
        self.manifest = manifest
        self.chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        self._titles = {d.sha256: d.title for d in manifest.documents}
        self._bm25: BM25 | None = None

    @classmethod
    def load(cls, root: Path) -> CorpusIndex:
        manifest = load_manifest(root)
        chunks: list[Chunk] = []
        for doc in manifest.documents:
            chunks.extend(chunk_document(doc.sha256, load_text(root, doc.sha256)))
        return cls(root, manifest, tuple(chunks))

    @property
    def corpus_version(self) -> str:
        return self.manifest.corpus_version

    def chunk(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def _bm25_index(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([tokenize(c.text) for c in self.chunks])
        return self._bm25

    # -- embeddings cache ----------------------------------------------------
    def _cache_path(self, doc_sha: str, embedder: Embedder) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", embedder.name)
        return self.root / EMBED_DIR / f"{doc_sha}.{safe}.json"

    def ensure_embeddings(self, embedder: Embedder) -> dict[str, list[float]]:
        """chunk_hash → vector for every chunk, embedding what the cache lacks."""
        import json
        vectors: dict[str, list[float]] = {}
        by_doc: dict[str, list[Chunk]] = {}
        for c in self.chunks:
            by_doc.setdefault(c.doc_sha, []).append(c)
        for doc_sha, chunks in by_doc.items():
            path = self._cache_path(doc_sha, embedder)
            cached: dict[str, list[float]] = {}
            if path.is_file():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    cached = {k: v for k, v in (raw.get("vectors") or {}).items()
                              if isinstance(v, list)} if raw.get("model") == embedder.name else {}
                except (OSError, ValueError):
                    cached = {}
            missing = [c for c in chunks if c.chunk_hash not in cached]
            if missing:
                new = embedder.embed([c.text for c in missing])
                cached = {**cached, **{c.chunk_hash: v for c, v in zip(missing, new)}}
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"model": embedder.name, "vectors": cached}), encoding="utf-8")
            vectors.update({c.chunk_hash: cached[c.chunk_hash] for c in chunks})
        return vectors

    # -- search ---------------------------------------------------------------
    def search(self, query: str, k: int = DEFAULT_K, embedder: Embedder | None = None) -> SearchResult:
        k = max(1, min(int(k), MAX_K))
        n = len(self.chunks)
        if n == 0:
            return SearchResult(query, (), EMBEDDING_UNAVAILABLE, self.corpus_version, 0)
        pool = min(n, max(k * POOL_FACTOR, POOL_MIN))

        bm25_scores = self._bm25_index().scores(tokenize(query))
        nonzero = sorted((i for i in range(n) if bm25_scores[i] > 0.0),
                         key=lambda i: (-bm25_scores[i], self.chunks[i].chunk_id))[:pool]
        bm25_rank = {self.chunks[i].chunk_id: r for r, i in enumerate(nonzero)}

        vector_rank: dict[str, int] = {}
        embedding, embedding_error = EMBEDDING_UNAVAILABLE, None
        if embedder is not None and embedder.available():
            try:
                vectors = self.ensure_embeddings(embedder)
                qv = embedder.embed([query])[0]
                sims = [(_cosine(qv, vectors[c.chunk_hash]), c.chunk_id) for c in self.chunks]
                ordered = sorted(sims, key=lambda t: (-t[0], t[1]))[:pool]
                vector_rank = {cid: r for r, (_, cid) in enumerate(ordered)}
                embedding = embedder.name
            except (requests.RequestException, RuntimeError, KeyError, ValueError) as exc:
                embedding_error = str(exc)[:200]
                log.warning("knowledge_embedding_failed", extra={"error": embedding_error})

        candidates = set(bm25_rank) | set(vector_rank)
        fused = {cid: (1.0 / (RRF_K + bm25_rank[cid]) if cid in bm25_rank else 0.0)
                 + (1.0 / (RRF_K + vector_rank[cid]) if cid in vector_rank else 0.0)
                 for cid in candidates}
        order = sorted(candidates, key=lambda cid: (-fused[cid], cid))[:k]
        hits = tuple(Hit(
            chunk_id=cid, doc_sha=(c := self._by_id[cid]).doc_sha, chunk_hash=c.chunk_hash,
            title=self._titles.get(c.doc_sha, ""), section=c.section, offsets=(c.start, c.end),
            text=c.text, score=round(fused[cid], 6),
            bm25_rank=bm25_rank.get(cid), vector_rank=vector_rank.get(cid),
        ) for cid in order)
        return SearchResult(query, hits, embedding, self.corpus_version, n, embedding_error)
