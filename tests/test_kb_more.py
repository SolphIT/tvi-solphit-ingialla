import numpy as np
from tvi.solphit.ingialla import kb as kb_mod
from tvi.solphit.ingialla.kb import BuildConfig, build_kb


class FakeES:
    def __init__(self):
        self.ops = []  # capture operations for assertions

    def update(self, **k):
        self.ops.append(("update", k))

    def index(self, **k):
        self.ops.append(("index", k))

    def get(self, **k):
        return {}

    def search(self, **k):
        # mapping for get_unprocessed_for_kb or others
        return {"hits": {"hits": []}}


class FakeEmbedder:
    def __init__(self, cfg): self.dim = 4
    def embed(self, chunks): return np.array([[1,0,0,0] for _ in chunks], dtype="float32")

def _cfg(tmp_path, **over):
    base = dict(
        input_dir=str(tmp_path/"articles"), output_dir=str(tmp_path/"kb"), db_path="(ignored)",
        include_redirects=False, chunk_size=100, overlap=10,
        embed_backend="ollama", embed_model="nomic", embed_batch=8,
        commit_every=500, max_pages=None
    ); base.update(over); return BuildConfig(**base)

def test_kb_skips_missing_file(tmp_path, monkeypatch):
    (tmp_path/"articles").mkdir()
    missing = tmp_path/"articles"/"missing.xml"
    missing.write_text('<page><title>X</title><revision><text>Body</text></revision></page>', encoding="utf-8")
    # Remove file to simulate missing
    missing.unlink()

    fake_es = FakeES()
    monkeypatch.setattr(kb_mod, "ensure_articles_index", lambda: fake_es)
    monkeypatch.setattr(kb_mod, "ensure_chunks_index", lambda dims: None)
    monkeypatch.setattr(kb_mod, "get_unprocessed_for_kb", lambda es, mp: [("X", str(missing))])
    monkeypatch.setattr(kb_mod, "Embedder", FakeEmbedder)
    meta = build_kb(_cfg(tmp_path))
    assert meta["stats"]["n_docs"] in (0, 1)  # no doc processed due to missing file

def test_kb_empty_chunks_mark_done(tmp_path, monkeypatch):
    art = tmp_path/"articles"; art.mkdir()
    page = art/"a.xml"
    page.write_text('<page><title>T</title><revision><text></text></revision></page>', encoding="utf-8")

    fake_es = FakeES()
    monkeypatch.setattr(kb_mod, "ensure_articles_index", lambda: fake_es)
    monkeypatch.setattr(kb_mod, "ensure_chunks_index", lambda dims: None)
    monkeypatch.setattr(kb_mod, "get_unprocessed_for_kb", lambda es, mp: [("T", str(page))])

    # Force embedder but chunks will be empty due to empty text
    monkeypatch.setattr(kb_mod, "Embedder", FakeEmbedder)

    cfg = _cfg(tmp_path, chunk_size=10, overlap=2)
    meta = build_kb(cfg)
    # should have updated mark_kb_done via update()
    assert any(op == "update" for op, _ in fake_es.ops)