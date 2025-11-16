from tvi.solphit.ingialla.ask import knn_search

class FakeES:
    def __init__(self): self.last = None
    def search(self, **kw):
        self.last = kw
        return {"hits": {"hits": []}}

def test_knn_search_builds_payload():
    es = FakeES()
    resp = knn_search(es, index="kb_chunks", field="vector", qvec=[0.1,0.2], k=3, num_candidates=123)
    assert resp["hits"]["hits"] == []
    kw = es.last
    assert kw["index"] == "kb_chunks"
    assert kw["knn"]["field"] == "vector"
    assert kw["knn"]["k"] == 3 and kw["knn"]["num_candidates"] == 123
    assert kw["_source"]  # default fields present