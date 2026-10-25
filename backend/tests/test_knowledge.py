"""mikebrain M0: content-addressed manifest, section-aware chunking, BM25-only and
hybrid retrieval, and the deterministic quote-grounding validator.

Keyless and offline: no Ollama is contacted (``NoEmbedder`` / a fake embedder).
The one real-embedding test skips unless a local Ollama with nomic-embed-text answers.
Fixtures are three tiny SYNTHETIC notes under tests/fixtures/knowledge (no copyrighted text).
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.knowledge import cli
from app.knowledge import manifest as mf
from app.knowledge.chunking import CHUNK_OVERLAP_WORDS, CHUNK_WORDS, chunk_document, is_heading, split_sections
from app.knowledge.extract import ExtractionError, extract_text
from app.knowledge.grounding import (
    GROUNDING_ABSTAINED,
    GROUNDING_FAILED,
    GROUNDING_OK,
    GROUNDING_PARTIAL,
    GROUNDING_UNPARSEABLE,
    MIN_QUOTE_CHARS,
    NO_EVIDENCE,
    ground,
    parse_critic_json,
    quote_in_chunk,
    validate_citations,
)
from app.knowledge.index import BM25, EMBEDDING_UNAVAILABLE, MAX_K, CorpusIndex, NoEmbedder, OllamaEmbedder, tokenize

FIXTURES = Path(__file__).parent / "fixtures" / "knowledge"
NOTES = sorted(FIXTURES.glob("*.txt"))
NOW = datetime.now(UTC).isoformat(timespec="seconds")


def _record(path: Path, **over) -> mf.DocumentRecord:
    base = dict(path=str(path), sha256=mf.file_sha256(path), title=path.stem, year=2026,
                source_type="paper", added_by="tester", added_at=NOW)
    return mf.DocumentRecord(**{**base, **over})


def _build(root: Path, paths=NOTES) -> mf.Manifest:
    m = mf.Manifest()
    for p in paths:
        m = mf.add_document(root, _record(p), extract_text(p))
    return m


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "knowledge"
    _build(root)
    return root


# ── manifest ────────────────────────────────────────────────────────────────────

def test_fixtures_are_three_tiny_synthetic_notes():
    assert len(NOTES) == 3
    for p in NOTES:
        assert "synthetic fixture" in p.read_text().lower()
        assert p.stat().st_size < 4000


def test_manifest_hashes_and_version_change_on_add_and_remove(tmp_path: Path):
    root = tmp_path / "k"
    m2 = _build(root, NOTES[:2])
    v2 = m2.corpus_version
    assert len(v2) == 64 and mf.load_manifest(root).corpus_version == v2
    for d in m2.documents:
        assert d.sha256 == mf.file_sha256(Path(d.path))
        assert mf.text_path(root, d.sha256).is_file()
    m3 = mf.add_document(root, _record(NOTES[2]), extract_text(NOTES[2]))
    assert m3.corpus_version != v2
    back = mf.remove_document(root, m3.documents[-1].sha256)
    assert back.corpus_version == v2                      # same document set → same version
    assert not mf.text_path(root, m3.documents[-1].sha256).exists()


def test_corpus_version_is_order_independent_and_well_defined_when_empty():
    a, b = "a" * 64, "b" * 64
    assert mf.corpus_version_of([a, b]) == mf.corpus_version_of([b, a])
    assert mf.Manifest().corpus_version == mf.corpus_version_of([])
    assert mf.Manifest().is_empty


def test_manifest_rejects_duplicates_bad_fields_and_malformed_file(tmp_path: Path):
    root = tmp_path / "k"
    _build(root, NOTES[:1])
    with pytest.raises(mf.ManifestError):
        mf.add_document(root, _record(NOTES[0]), "x" * 300)
    with pytest.raises(mf.ManifestError):
        mf.remove_document(root, "f" * 64)
    with pytest.raises(mf.ManifestError):
        _record(NOTES[0], source_type="blog").validate()
    with pytest.raises(mf.ManifestError):
        _record(NOTES[0], sha256="short").validate()
    mf.manifest_path(root).write_text("{not json", encoding="utf-8")
    with pytest.raises(mf.ManifestError):       # fails closed, never reads as an empty corpus
        mf.load_manifest(root)


def test_missing_manifest_is_an_empty_corpus(tmp_path: Path):
    assert mf.load_manifest(tmp_path / "nowhere").is_empty


def test_sha256_must_be_lowercase_hex_and_never_a_path(tmp_path: Path):
    traversal = ("../" * 21 + "x")[:64]
    assert len(traversal) == 64 and not mf.is_sha256(traversal)
    assert not mf.is_sha256("A" * 64) and mf.is_sha256("a" * 64)
    with pytest.raises(mf.ManifestError):
        _record(NOTES[0], sha256=traversal).validate()
    with pytest.raises(mf.ManifestError):           # a tampered digest cannot become a path component
        mf.text_path(tmp_path, traversal)
    with pytest.raises(mf.ManifestError):
        mf.load_text(tmp_path, "A" * 64)


def test_manifest_fails_closed_on_bad_year_and_non_object_entries(tmp_path: Path):
    root = tmp_path / "k"
    _build(root, NOTES[:1])
    good = json.loads(mf.manifest_path(root).read_text())
    bad_year = {**good, "documents": [{**good["documents"][0], "year": "abc"}]}
    mf.manifest_path(root).write_text(json.dumps(bad_year))
    with pytest.raises(mf.ManifestError):
        mf.load_manifest(root)
    mixed = {**good, "documents": [good["documents"][0], "not-an-object"]}
    mf.manifest_path(root).write_text(json.dumps(mixed))
    with pytest.raises(mf.ManifestError):           # never silently dropped (that would change corpus_version)
        mf.load_manifest(root)


# ── extraction ──────────────────────────────────────────────────────────────────

def test_extract_text_reads_txt_and_rejects_unsupported_or_empty(tmp_path: Path):
    assert "shrinkage" in extract_text(NOTES[-1]).lower()
    with pytest.raises(ExtractionError):
        extract_text(tmp_path / "x.docx")
    short = tmp_path / "short.txt"
    short.write_text("too short to index")
    with pytest.raises(ExtractionError):
        extract_text(short)


def test_extract_pdf_branch_refuses_blank_and_corrupt_pdfs(tmp_path: Path):
    pypdf = pytest.importorskip("pypdf")
    blank = tmp_path / "blank.pdf"
    w = pypdf.PdfWriter()
    w.add_blank_page(width=200, height=200)
    with blank.open("wb") as fh:
        w.write(fh)
    with pytest.raises(ExtractionError, match="yielded only"):   # image-only / empty → refused, not indexed
        extract_text(blank)
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4 garbage" + bytes(range(256)) * 4)
    with pytest.raises(ExtractionError):
        extract_text(corrupt)


# ── chunking ────────────────────────────────────────────────────────────────────

def test_headings_detected_and_sections_split():
    assert is_heading("## 2 Why it matters")
    assert is_heading("3.2 Shrinkage of eta estimates")
    assert is_heading("METHODS")
    assert not is_heading("The model was fitted with FOCE-I.")
    text = extract_text(NOTES[-1])
    heads = [h for h, _, _ in split_sections(text)]
    assert any("shrinkage" in h.lower() for h in heads) and len(heads) >= 3


def test_chunks_carry_exact_offsets_and_are_deterministic():
    text = extract_text(NOTES[0])
    sha = mf.file_sha256(NOTES[0])
    chunks = chunk_document(sha, text)
    assert chunks and chunks == chunk_document(sha, text)
    for c in chunks:
        assert text[c.start:c.end] == c.text
        assert c.chunk_id == f"{sha[:12]}:{c.ordinal}" and c.doc_sha == sha
        assert c.chunk_hash == mf.sha256_text(c.text)


def test_long_section_windows_overlap_and_cover_everything():
    words = [f"w{i}" for i in range(CHUNK_WORDS * 3 + 17)]
    text = "LONG SECTION\n" + " ".join(words)
    chunks = chunk_document("d" * 64, text)
    assert len(chunks) >= 3
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    for a, b in zip(chunks, chunks[1:]):
        assert a.end > b.start                      # overlapping windows
        shared = set(a.text.split()) & set(b.text.split())
        assert len(shared) >= CHUNK_OVERLAP_WORDS - 1
    assert words[-1] in chunks[-1].text


# ── retrieval ───────────────────────────────────────────────────────────────────

def test_bm25_scores_exact_terms():
    bm = BM25([tokenize("eta shrinkage shrinkage EBE"), tokenize("allometric exponent"), tokenize("aic delta")])
    s = bm.scores(tokenize("shrinkage"))
    assert s[0] > 0 and s[1] == 0 and s[2] == 0


@pytest.mark.parametrize("query, expected_title", [
    ("eta shrinkage EBE covariate screening", "shrinkage_note"),
    ("AIC difference of less than two points runner-up", "model_selection_note"),
    ("allometric exponent 0.75 clearance reference weight", "allometry_note"),
])
def test_bm25_only_retrieval_over_fixtures(corpus: Path, query: str, expected_title: str):
    index = CorpusIndex.load(corpus)
    res = index.search(query, k=3, embedder=NoEmbedder())
    assert res.embedding == EMBEDDING_UNAVAILABLE and res.embedding_error is None
    assert res.corpus_version == mf.load_manifest(corpus).corpus_version
    assert res.hits and res.hits[0].title == expected_title
    top = res.hits[0]
    assert index.chunk(top.chunk_id) is not None
    assert top.text == mf.load_text(corpus, top.doc_sha)[top.offsets[0]:top.offsets[1]]
    assert top.score > 0 and top.bm25_rank == 0 and top.vector_rank is None
    d = top.to_dict()
    assert {"chunk_id", "doc_sha", "offsets", "text", "score", "chunk_hash"} <= set(d)
    assert "text" not in top.to_dict(with_text=False)


def test_search_without_embedder_is_bm25_only_and_k_is_bounded(corpus: Path):
    index = CorpusIndex.load(corpus)
    res = index.search("shrinkage", k=999)
    assert res.embedding == EMBEDDING_UNAVAILABLE and len(res.hits) <= MAX_K
    assert index.search("zzzz qqqq", k=3, embedder=NoEmbedder()).hits == ()


def test_empty_corpus_search_returns_nothing(tmp_path: Path):
    res = CorpusIndex.load(tmp_path / "none").search("anything")
    assert res.hits == () and res.n_chunks == 0 and res.embedding == EMBEDDING_UNAVAILABLE


class _FakeEmbedder:
    """Deterministic 'vectors': bag of a few marker words, so similarity is checkable."""
    name = "fake:test"
    markers = ("shrinkage", "allometric", "aic", "convergence")

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail

    def available(self) -> bool:
        return True

    def embed(self, texts):
        if self.fail:
            raise RuntimeError("embedding server down")
        self.calls.append(list(texts))
        return [[float(t.lower().count(m)) for m in self.markers] + [1e-3] for t in texts]


def test_hybrid_search_fuses_vectors_and_caches_embeddings(corpus: Path):
    index = CorpusIndex.load(corpus)
    emb = _FakeEmbedder()
    res = index.search("allometric", k=3, embedder=emb)
    assert res.embedding == "fake:test"
    assert res.hits[0].title == "allometry_note"
    assert res.hits[0].vector_rank is not None and res.hits[0].bm25_rank is not None
    cache = list((corpus / "embeddings").glob("*.json"))
    assert len(cache) == 3 and all(json.loads(p.read_text())["model"] == "fake:test" for p in cache)
    n_embed_calls = len(emb.calls)
    index.search("allometric", k=3, embedder=emb)
    assert len(emb.calls) == n_embed_calls + 1          # only the query was embedded again


def test_embedding_failure_falls_back_to_bm25_and_says_so(corpus: Path):
    res = CorpusIndex.load(corpus).search("shrinkage", k=2, embedder=_FakeEmbedder(fail=True))
    assert res.embedding == EMBEDDING_UNAVAILABLE and "down" in (res.embedding_error or "")
    assert res.hits and res.hits[0].title == "shrinkage_note"


def test_stale_or_corrupt_vector_cache_is_detected(corpus: Path):
    index = CorpusIndex.load(corpus)
    emb = _FakeEmbedder()
    index.search("shrinkage", k=2, embedder=emb)                       # populates the cache (5-dim vectors)
    cache_files = sorted((corpus / "embeddings").glob("*.json"))
    # a same-name model update that changed dimensions: cached vectors no longer match the query vector
    for p in cache_files:
        raw = json.loads(p.read_text())
        p.write_text(json.dumps({"model": raw["model"], "vectors": {h: v[:3] for h, v in raw["vectors"].items()}}))
    fresh = CorpusIndex.load(corpus)
    res = fresh.search("shrinkage", k=2, embedder=emb)
    assert res.embedding == EMBEDDING_UNAVAILABLE and "dimension mismatch" in (res.embedding_error or "")
    assert res.hits and res.hits[0].title == "shrinkage_note"          # BM25 still answers
    # a corrupt cache file is treated as missing and re-embedded, not crashed on
    cache_files[0].write_text("[1, 2, 3]")
    cache_files[1].write_text(json.dumps({"model": emb.name, "vectors": {"h": ["x", "y"]}}))
    before = len(emb.calls)
    again = CorpusIndex.load(corpus)
    import shutil
    shutil.rmtree(corpus / "embeddings")                                # start clean so dims agree again
    res = again.search("allometric", k=2, embedder=emb)
    assert res.embedding == emb.name and len(emb.calls) > before


def test_load_index_is_cached_until_the_corpus_changes(tmp_path: Path):
    from app.knowledge.index import load_index
    root = tmp_path / "k"
    _build(root, NOTES[:2])
    first = load_index(root)
    assert load_index(root) is first
    n_before = len(first.chunks)
    mf.add_document(root, _record(NOTES[2]), extract_text(NOTES[2]))
    second = load_index(root)
    assert second is not first and len(second.chunks) > n_before
    assert second.corpus_version == mf.load_manifest(root).corpus_version


def test_ollama_embedder_unavailable_when_server_unreachable(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(requests, "get", boom)
    assert OllamaEmbedder("http://127.0.0.1:1/v1").available() is False


# ── grounding validator ─────────────────────────────────────────────────────────

CHUNKS = {
    "aaaaaaaaaaaa:2": ("When eta-shrinkage exceeds 30 percent, individual parameter estimates are drawn\n"
                       "toward the population mean and EBE-based diagnostics lose their power to reveal\n"
                       "model misspecification."),
    "bbbbbbbbbbbb:0": "A difference in AIC of less than two points does not distinguish the two models.",
}
VERBATIM = "EBE-based diagnostics lose their power to reveal model misspecification"


def _reply(verdict="supported", citations=None, rationale="because") -> str:
    return json.dumps({"verdict": verdict, "rationale": rationale, "citations": citations or []})


def test_quote_in_chunk_tolerates_only_whitespace_layout():
    assert quote_in_chunk(VERBATIM, CHUNKS["aaaaaaaaaaaa:2"])
    assert quote_in_chunk("drawn toward the population mean", CHUNKS["aaaaaaaaaaaa:2"])
    assert not quote_in_chunk("EBE-based diagnostics lose their power to show misspecification",
                              CHUNKS["aaaaaaaaaaaa:2"])
    assert not quote_in_chunk("", CHUNKS["aaaaaaaaaaaa:2"])


def test_validator_accepts_verbatim_rejects_paraphrase_foreign_id_short_long_duplicate():
    offered = [
        {"chunk_id": "aaaaaaaaaaaa:2", "quote": VERBATIM},
        {"chunk_id": "aaaaaaaaaaaa:2", "quote": "EBE-based diagnostics lose power to reveal misspecification"},
        {"chunk_id": "zzzzzzzzzzzz:9", "quote": VERBATIM},
        {"chunk_id": "bbbbbbbbbbbb:0", "quote": "AIC of less"},
        {"chunk_id": "bbbbbbbbbbbb:0", "quote": "x" * 700},
        {"chunk_id": "aaaaaaaaaaaa:2", "quote": VERBATIM},
        "not an object",
    ]
    valid, dropped = validate_citations(offered, CHUNKS)
    assert valid == [{"chunk_id": "aaaaaaaaaaaa:2", "quote": VERBATIM}]
    reasons = [d["reason"] for d in dropped]
    assert any("verbatim" in r for r in reasons)
    assert any("not among retrieved" in r for r in reasons)
    assert any(f"shorter than {MIN_QUOTE_CHARS}" in r for r in reasons)
    assert any("longer than" in r for r in reasons)
    assert any("duplicate" in r for r in reasons)
    assert any("not an object" in r for r in reasons)
    assert len(dropped) == 6


def test_quote_must_be_verbatim_in_the_cited_chunk_not_merely_somewhere():
    aic = "A difference in AIC of less than two points"
    valid, dropped = validate_citations([{"chunk_id": "aaaaaaaaaaaa:2", "quote": aic},
                                         {"chunk_id": "bbbbbbbbbbbb:0", "quote": aic}], CHUNKS)
    assert valid == [{"chunk_id": "bbbbbbbbbbbb:0", "quote": aic}]
    assert dropped[0]["chunk_id"] == "aaaaaaaaaaaa:2" and "verbatim" in dropped[0]["reason"]


def test_ground_supported_with_verbatim_citation_is_ok():
    g = ground(_reply(citations=[{"chunk_id": "aaaaaaaaaaaa:2", "quote": VERBATIM}]), CHUNKS)
    assert g.verdict == "supported" and g.grounding == GROUNDING_OK and len(g.citations) == 1
    assert g.to_dict()["citations"][0]["quote"] == VERBATIM


def test_ground_partial_when_some_citations_dropped():
    g = ground(_reply("contradicted", [{"chunk_id": "aaaaaaaaaaaa:2", "quote": VERBATIM},
                                        {"chunk_id": "nope:1", "quote": VERBATIM}]), CHUNKS)
    assert g.verdict == "contradicted" and g.grounding == GROUNDING_PARTIAL
    assert len(g.citations) == 1 and len(g.dropped) == 1


def test_ground_downgrades_to_no_evidence_when_no_citation_survives():
    paraphrase = [{"chunk_id": "aaaaaaaaaaaa:2", "quote": "shrinkage above 30% ruins EBE diagnostics entirely"}]
    g = ground(_reply("supported", paraphrase), CHUNKS)
    assert g.verdict == NO_EVIDENCE and g.grounding == GROUNDING_FAILED and g.raw_verdict == "supported"
    assert g.citations == () and len(g.dropped) == 1
    g2 = ground(_reply("contradicted"), CHUNKS)          # evidence claimed, nothing cited
    assert g2.verdict == NO_EVIDENCE and g2.grounding == GROUNDING_FAILED


def test_ground_abstained_unparseable_and_unknown_verdict():
    g = ground(_reply("no_evidence", rationale="nothing relevant"), CHUNKS)
    assert g.verdict == NO_EVIDENCE and g.grounding == GROUNDING_ABSTAINED and g.rationale == "nothing relevant"
    g = ground("I think the model is fine.", CHUNKS)
    assert g.verdict == NO_EVIDENCE and g.grounding == GROUNDING_UNPARSEABLE and g.parse_error
    g = ground(_reply("maybe", [{"chunk_id": "aaaaaaaaaaaa:2", "quote": VERBATIM}]), CHUNKS)
    assert g.verdict == NO_EVIDENCE and g.grounding == GROUNDING_FAILED and g.citations == ()


def test_parse_critic_json_accepts_fenced_and_embedded_objects():
    inner = {"verdict": "no_evidence", "rationale": "", "citations": []}
    assert parse_critic_json("```json\n" + json.dumps(inner) + "\n```") == inner
    assert parse_critic_json("Sure! " + json.dumps(inner) + " Hope this helps.") == inner
    with pytest.raises(ValueError):
        parse_critic_json("[1, 2]")
    with pytest.raises(ValueError):
        parse_critic_json("")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def test_cli_add_list_search_remove(tmp_path: Path, capsys, monkeypatch):
    emb = _FakeEmbedder()
    monkeypatch.setattr(cli, "default_embedder", lambda: emb)           # offline: never probe Ollama
    root = tmp_path / "k"
    assert cli.main(["--root", str(root), "add", str(NOTES[-1]), "--title", "Shrinkage", "--year", "2026",
                     "--type", "guideline", "--by", "owner"]) == 0
    added = json.loads(capsys.readouterr().out)
    assert added["documents"] == 1 and added["corpus_chunks"] >= 3
    assert added["embeddings"] == emb.name and emb.calls                # cache warmed at add time
    assert list((root / "embeddings").glob("*.json"))
    assert cli.main(["--root", str(root), "add", str(NOTES[0]), "--no-embed"]) == 0
    assert json.loads(capsys.readouterr().out)["embeddings"] == EMBEDDING_UNAVAILABLE
    assert cli.main(["--root", str(root), "remove", mf.file_sha256(NOTES[0])[:10]]) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(root), "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["documents"][0]["title"] == "Shrinkage" and listed["documents"][0]["source_type"] == "guideline"
    assert listed["corpus_version"] == added["corpus_version"]
    assert cli.main(["--root", str(root), "search", "eta shrinkage", "-k", "2", "--no-embed"]) == 0
    found = json.loads(capsys.readouterr().out)
    assert found["embedding"] == EMBEDDING_UNAVAILABLE and found["hits"]
    assert cli.main(["--root", str(root), "remove", added["added"][:8]]) == 0
    assert json.loads(capsys.readouterr().out)["documents"] == 0
    assert cli.main(["--root", str(root), "add", str(tmp_path / "missing.pdf")]) == 2
    mf.manifest_path(root).write_text("{broken", encoding="utf-8")
    assert cli.main(["--root", str(root), "list"]) == 2                 # error message, not a traceback
    assert "error:" in capsys.readouterr().err


# ── opt-in: real embeddings ─────────────────────────────────────────────────────

def test_real_ollama_embeddings_when_reachable(corpus: Path):
    from app.knowledge.index import default_embedder
    emb = default_embedder()
    if not emb.available():
        pytest.skip("no local Ollama with nomic-embed-text reachable")
    res = CorpusIndex.load(corpus).search("How does high eta shrinkage affect EBE diagnostics?", k=3, embedder=emb)
    assert res.embedding == emb.name and res.embedding_error is None
    assert res.hits and res.hits[0].title == "shrinkage_note"
    assert all(h.vector_rank is not None for h in res.hits)
