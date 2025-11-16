from tvi.solphit.ingialla import es as es_mod

class FakeIndices:
    def __init__(self): self.calls = []
    def create(self, **kw): self.calls.append(("create", kw))

class FakeES:
    def __init__(self): self.indices = FakeIndices()
    def get(self, **kw): return {"_source": {"split_done": True}}
    def update(self, **kw): return {}
    def index(self, **kw): return {}
    def search(self, **kw): return {"hits": {"hits": []}}

def test_ensure_articles_index():
    # monkeypatch es_client to return our fake
    fake = FakeES()
    def fake_client(): return fake
    old = es_mod.es_client
    es_mod.es_client = fake_client
    try:
        es = es_mod.ensure_articles_index()
        assert es is fake
        assert any(c[0] == "create" and c[1]["index"] == es_mod.ARTICLES for c in fake.indices.calls)
    finally:
        es_mod.es_client = old

def test_ensure_chunks_index_sets_dims(monkeypatch):
    fake = FakeES()
    monkeypatch.setattr(es_mod, "es_client", lambda: fake)
    es_mod.ensure_chunks_index(dims=128)
    # assert it attempted 'create' on CHUNKS
    assert any(c[0] == "create" and c[1]["index"] == es_mod.CHUNKS for c in fake.indices.calls)

def test_already_split_true_and_exception(monkeypatch):
    fake = FakeES()
    monkeypatch.setattr(es_mod, "es_client", lambda: fake)
    # True case
    assert es_mod.already_split(fake, "/x.xml") is True
    # Exception path -> False
    class ErrES(FakeES):
        def get(self, **kw): raise RuntimeError("es down")
    assert es_mod.already_split(ErrES(), "/x.xml") is False

def test_bulk_index_chunks_calls_helpers(monkeypatch):
    called = {}
    def fake_bulk(es, actions, chunk_size=0, request_timeout=0):
        called["count"] = sum(1 for _ in actions)
        called["chunk_size"] = chunk_size

    monkeypatch.setattr(es_mod.helpers, "bulk", fake_bulk)
    es_mod.bulk_index_chunks(object(), [
        {"chunk_id": "c1", "doc_id": "d", "text": "a", "vector": [0.1], "created_at": 0},
        {"chunk_id": "c2", "doc_id": "d", "text": "b", "vector": [0.2], "created_at": 0},
    ])
    assert called["count"] == 2