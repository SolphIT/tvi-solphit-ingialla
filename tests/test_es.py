import numpy as np
import types
from tvi.solphit.ingialla import es as es_mod


# --- Utilities / fakes used by multiple tests --------------------------------

class FakeIndices:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        # Record the call for assertions
        self.calls.append(("create", kw))


class FakeES:
    """
    A minimal fake ES client with just the attributes/methods we need.
    Individual tests can subclass/extend to override behavior.
    """
    def __init__(self):
        self.indices = FakeIndices()
        self.ops = []  # capture operations for assertions

    # Stubs overridden where needed
    def get(self, **kw):
        return {"_source": {"split_done": True}}

    def update(self, **kw):
        self.ops.append(("update", kw))

    def index(self, **kw):
        self.ops.append(("index", kw))

    def search(self, **kw):
        return {"hits": {"hits": []}}


# --- es_client ---------------------------------------------------------------

def test_es_client_constructs_with_headers(monkeypatch):
    # Ensure we point to a harmless URL and the client still constructs.
    monkeypatch.setenv("ELASTIC_URL", "http://example:9200")
    es = es_mod.es_client()
    assert es is not None  # basic construction (headers/timeouts set in es.py)


# --- ensure_*_index ----------------------------------------------------------

def test_ensure_articles_index_creates(monkeypatch):
    fake = FakeES()

    # Replace the real client with our fake
    monkeypatch.setattr(es_mod, "es_client", lambda: fake)

    es = es_mod.ensure_articles_index()
    assert es is fake

    # One of the recorded index create calls must target the ARTICLES index
    assert any(
        c[0] == "create" and c[1]["index"] == es_mod.ARTICLES
        for c in fake.indices.calls
    )


def test_ensure_chunks_index_sets_dims(monkeypatch):
    fake = FakeES()
    monkeypatch.setattr(es_mod, "es_client", lambda: fake)

    es_mod.ensure_chunks_index(dims=128)

    # Confirm the CHUNKS index was created (with dims embedded in the mapping)
    assert any(
        c[0] == "create" and c[1]["index"] == es_mod.CHUNKS
        for c in fake.indices.calls
    )


# --- already_split -----------------------------------------------------------

def test_already_split_true_and_exception(monkeypatch):
    # True path: FakeES.get returns {"_source": {"split_done": True}}
    fake = FakeES()
    assert es_mod.already_split(fake, "/x.xml") is True

    # Exception path -> False
    class ErrES(FakeES):
        def get(self, **kw):
            raise RuntimeError("es down")

    assert es_mod.already_split(ErrES(), "/x.xml") is False


# --- mark_split_done ---------------------------------------------------------

def test_mark_split_done_indexes_document():
    fake = FakeES()
    xml_path = "/x/y/z.xml"
    title = "T"
    es_mod.mark_split_done(fake, xml_path, title)

    # We should have produced an index operation with correct doc
    op = [op for op in fake.ops if op[0] == "index"]
    assert op, "Expected an index operation"
    _, kwargs = op[0]
    assert kwargs["index"] == es_mod.ARTICLES
    assert kwargs["id"] == es_mod._art_id(xml_path)
    assert kwargs["document"]["title"] == title
    assert kwargs["document"]["xml_path"] == xml_path
    assert kwargs["document"]["split_done"] is True
    assert kwargs["document"]["kb_done"] is False
    assert isinstance(kwargs["document"]["created_at"], int)
    assert isinstance(kwargs["document"]["updated_at"], int)
    assert kwargs["request_timeout"] == 30


# --- mark_kb_done (covers lines 67–69) --------------------------------------

def test_mark_kb_done_calls_update_with_upsert():
    captured = {}

    class E:
        def update(self, **kw):
            captured.update(kw)

    es = E()
    xml_path = "/some/path/article.xml"

    es_mod.mark_kb_done(es, xml_path)

    # Assert: correct target and id
    assert captured["index"] == es_mod.ARTICLES
    assert captured["id"] == es_mod._art_id(xml_path)

    # Assert: upsert and timeout exactly as specified (lines 67–69)
    assert captured["doc_as_upsert"] is True
    assert captured["request_timeout"] == 30

    # Assert: payload
    assert captured["doc"]["kb_done"] is True
    assert isinstance(captured["doc"]["updated_at"], int)


# --- get_unprocessed_for_kb --------------------------------------------------

def test_get_unprocessed_default_size_and_mapping(monkeypatch):
    class E:
        def search(self, **kw):
            # Default size when max_pages is None must be 10000
            assert kw.get("size") == 10000
            # Return two sample hits
            return {
                "hits": {
                    "hits": [
                        {"_source": {"title": "T1", "xml_path": "/a.xml"}},
                        {"_source": {"title": "T2", "xml_path": "/b.xml"}},
                    ]
                }
            }

    rows = es_mod.get_unprocessed_for_kb(E(), None)
    assert rows == [("T1", "/a.xml"), ("T2", "/b.xml")]


# --- bulk_index_chunks -------------------------------------------------------

def test_bulk_index_chunks_calls_helpers(monkeypatch):
    called = {"count": 0, "chunk_size": None, "timeout": None}

    def fake_bulk(es, actions, chunk_size=0, request_timeout=0):
        # Consume the generator to ensure we count the documents
        called["count"] = sum(1 for _ in actions)
        called["chunk_size"] = chunk_size
        called["timeout"] = request_timeout

    # Patch helpers.bulk to our fake
    monkeypatch.setattr(es_mod.helpers, "bulk", fake_bulk)

    rows = [
        {"chunk_id": "c1", "doc_id": "d", "text": "a", "vector": [0.1], "created_at": 0},
        {"chunk_id": "c2", "doc_id": "d", "text": "b", "vector": [0.2], "created_at": 0},
    ]
    es_mod.bulk_index_chunks(object(), rows)

    assert called["count"] == 2
    assert called["chunk_size"] == 2000
    assert called["timeout"] == 300