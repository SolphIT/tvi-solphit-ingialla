from tvi.solphit.ingialla.ask import knn_search, _KNN_CACHE

class FakeES:
    def __init__(self):
        self.last = None
    def search(self, **kw):
        self.last = kw
        return {"hits": {"hits": []}}

def test_knn_search_builds_payload():
    _KNN_CACHE.clear()  # <-- Clear cache before testing!
    es = FakeES()
    resp = knn_search(es, index="kb_chunks", field="vector", qvec=[0.1,0.2], k=3, num_candidates=123)
    assert resp["hits"]["hits"] == []
    kw = es.last
    assert kw["index"] == "kb_chunks"
    assert kw["knn"]["field"] == "vector"
    assert kw["knn"]["k"] == 3 and kw["knn"]["num_candidates"] == 123
    assert kw["_source"]  # default fields present

def test_knn_search_cache_key_rounding():
    from tvi.solphit.ingialla.ask import _qvec_key
    # Should round floats to 5 decimals
    key1 = _qvec_key([0.123456789, 0.987654321])
    key2 = _qvec_key([0.1234567, 0.9876543])
    assert key1 == key2
    assert all(isinstance(x, float) for x in key1)