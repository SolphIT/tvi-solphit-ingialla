import numpy as np
from tvi.solphit.ingialla.embed import EmbedConfig, Embedder

def test_embedder_st_backend(monkeypatch):
    class FakeST:
        def __init__(self, model, device=None): pass
        def get_sentence_embedding_dimension(self): return 3
        def encode(self, texts, batch_size=0, show_progress_bar=False, normalize_embeddings=True):
            # normalized vectors in 3D
            vecs = []
            for t in texts:
                v = np.array([1.0, 0.0, 0.0], dtype="float32")
                vecs.append(v)
            return np.vstack(vecs)

    monkeypatch.setattr("tvi.solphit.ingialla.embed.SentenceTransformer", FakeST)
    e = Embedder(EmbedConfig(backend="st", model="any", batch_size=2))
    arr = e.embed(["a", "b"])
    assert arr.shape == (2, 3)
    assert np.allclose(arr[0], [1.0, 0.0, 0.0])

def test_embedder_ollama_backend(monkeypatch):
    class R:
        def __init__(self, payload): self._payload = payload
        def raise_for_status(self): pass
        def json(self): return self._payload

    calls = []

    def fake_post(url, json=None, timeout=0):
        calls.append((url, json))
        if json.get("input") == "probe":
            # init probe returns 4D embedding
            return R({"embeddings": [[0,0,0,1]]})
        embs = [[1,0,0,0],[0,1,0,0]]
        return R({"embeddings": embs})

    monkeypatch.setattr("tvi.solphit.ingialla.embed.requests.post", fake_post)
    e = Embedder(EmbedConfig(backend="ollama", model="nomic-embed-text", batch_size=128))
    arr = e.embed(["x", "y"])
    assert arr.shape == (2, 4)
    # normalized
    assert np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-6)
    # saw both probe and batch calls
    assert any("/api/embed" in u for u, _ in calls)