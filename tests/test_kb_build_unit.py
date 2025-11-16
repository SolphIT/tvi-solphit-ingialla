from pathlib import Path
import json
import numpy as np
import types

from tvi.solphit.ingialla import kb as kb_mod
from tvi.solphit.ingialla.kb import BuildConfig, build_kb

def test_build_kb_happy_path(tmp_path, monkeypatch):
    # Create one fake article file with minimal XML
    art_dir = tmp_path / "articles"; art_dir.mkdir()
    page = art_dir / "a.xml"
    page.write_text('<page><title>T</title><revision><text>Alpha Beta Gamma Delta</text></revision></page>', encoding="utf-8")

    # Stub ES facade
    class FakeES:
        def __init__(self): self.bulk_rows = []
        def get(self, *a, **k): return {}
        def index(self, *a, **k): return {}
        def update(self, *a, **k): return {}
        def search(self, *a, **k): return {"hits": {"hits": []}}

    fake_es = FakeES()

    def fake_ensure_articles_index():
        return fake_es

    def fake_ensure_chunks_index(dims: int):
        assert dims == 4  # from our fake embedder below

    def fake_get_unprocessed_for_kb(es, max_pages):
        return [("T", str(page))]

    captured_rows = []
    def fake_bulk_index_chunks(es, rows):
        captured_rows.extend(rows)

    # Stub embedder to return 4D unit vectors
    class FakeEmbedder:
        def __init__(self, cfg):
            self.dim = 4
        def embed(self, chunks):
            # return one 4D vec per chunk
            return np.array([[1,0,0,0] for _ in chunks], dtype="float32")

    monkeypatch.setattr(kb_mod, "ensure_articles_index", fake_ensure_articles_index)
    monkeypatch.setattr(kb_mod, "ensure_chunks_index", fake_ensure_chunks_index)
    monkeypatch.setattr(kb_mod, "get_unprocessed_for_kb", fake_get_unprocessed_for_kb)
    monkeypatch.setattr(kb_mod, "bulk_index_chunks", fake_bulk_index_chunks)
    monkeypatch.setattr(kb_mod, "Embedder", FakeEmbedder)

    cfg = BuildConfig(
        input_dir=str(art_dir),
        output_dir=str(tmp_path / "kb"),
        db_path="(ignored)",
        include_redirects=False,
        chunk_size=10,
        overlap=2,
        embed_backend="ollama",
        embed_model="nomic-embed-text",
        embed_batch=8,
        commit_every=500,
        max_pages=None,
    )

    meta = build_kb(cfg)
    # sanity checks
    assert meta["stats"]["n_docs"] == 1
    assert meta["stats"]["n_chunks"] >= 1
    # docmap was written
    docmap = (tmp_path / "kb" / "docmap.jsonl")
    assert docmap.exists()
    line = json.loads(docmap.read_text(encoding="utf-8").splitlines()[0])
    assert line["title"] == "T"
    assert captured_rows  # we did index something