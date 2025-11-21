from tvi.solphit.ingialla.ask import knn_search, Generator
import pytest


class FakeES:
    def __init__(self):
        self.kw = None

    def search(self, **kw):
        self.kw = kw
        return {"hits": {
            "hits": [{"_source": {"title": "T", "text": "C", "source_path": "/p", "chunk_index": 0}, "_score": 1.0}]}}


def test_knn_search_payload():
    es = FakeES()
    resp = knn_search(es, "kb_chunks", "vector", [0.1, 0.2], k=3, num_candidates=123)
    assert "hits" in resp
    assert es.kw["knn"]["k"] == 3 and es.kw["knn"]["num_candidates"] == 123
    assert "vector" in str(es.kw["knn"])


def test_generator_none_returns_contexts():
    g = Generator(provider="none", model="n/a")
    out = g.generate("Q?", ["A", "B"])
    assert "[Context only]" in out and "Q?" in out and "A" in out and "B" in out


def test_generator_ollama_success(monkeypatch):
    class R:
        def raise_for_status(self): pass

        def json(self): return {"response": "Answer"}

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setattr("tvi.solphit.ingialla.ask.requests.post", lambda url, json, timeout: R())
    g = Generator(provider="ollama", model="mistral:7b-instruct")
    out = g.generate("Q?", ["C1", "C2"])
    assert "Answer" in out


def test_generator_ollama_error_fallback(monkeypatch):
    class R:
        def raise_for_status(self): raise RuntimeError("bad gateway")

    monkeypatch.setattr("tvi.solphit.ingialla.ask.requests.post", lambda url, json, timeout: R())
    g = Generator(provider="ollama", model="m")
    out = g.generate("Q?", ["C"])
    assert "[Context only]" in out


def test_generator_unknown_provider_raises():
    with pytest.raises(ValueError):
        Generator(provider="xyz", model="irrelevant")


def test_knn_search_cache(monkeypatch):
    import time
    from tvi.solphit.ingialla.ask import knn_search, _KNN_CACHE, _KNN_TTL_SEC
    _KNN_CACHE.clear()  # <-- Clear cache before testing!

    class FakeES:
        def __init__(self):
            self.calls = []

        def search(self, **kw):
            self.calls.append(kw)
            return {"hits": {"hits": [{"_source": {"text": "A"}}]}}

    es = FakeES()
    qvec = [0.1, 0.2]
    # First call populates cache
    out1 = knn_search(es, "kb_chunks", "vector", qvec, k=3, num_candidates=123)
    # Second call should hit cache (calls should not increment)
    out2 = knn_search(es, "kb_chunks", "vector", qvec, k=3, num_candidates=123)
    assert es.calls == [es.calls[0]]  # Only one call
    assert out1 == out2
    # Simulate TTL expiry
    key = next(iter(_KNN_CACHE))
    old_time = _KNN_CACHE[key][0]
    monkeypatch.setitem(_KNN_CACHE, key, (old_time - _KNN_TTL_SEC - 1, _KNN_CACHE[key][1]))
    out3 = knn_search(es, "kb_chunks", "vector", qvec, k=3, num_candidates=123)
    assert len(es.calls) == 2  # Called again after TTL