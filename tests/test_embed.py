import builtins
import importlib
import sys
import types
from typing import List

import numpy as np
import pytest


# -------------------------------
# Helpers to (re)load the module
# -------------------------------

PKG_MOD = "tvi.solphit.ingialla.embed"


def _unload_embed_module():
    """Remove the embed module (and only it) so we can reload with different import scenarios."""
    sys.modules.pop(PKG_MOD, None)


def _reload_embed_with_import_hook(monkeypatch, *, st_import_raises: bool):
    """
    Reload tvi.solphit.ingialla.embed with a temporary import hook that either:
      - raises ImportError for 'sentence_transformers' (to cover the except path), or
      - behaves normally.
    Returns the freshly imported module object.
    """
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if st_import_raises and name == "sentence_transformers":
            raise ImportError("sentence-transformers missing")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    _unload_embed_module()
    return importlib.import_module(PKG_MOD)


def _fake_st_class(dim=3):
    """
    Minimal Fake SentenceTransformer:
      - keeps 'device'
      - returns deterministic, normalized vectors
      - returns (0, dim) for empty input
    """
    class FakeST:
        def __init__(self, model, device=None):
            self.model = model
            self.device = device

        def get_sentence_embedding_dimension(self):
            return dim

        def encode(self, texts: List[str], batch_size=0, show_progress_bar=False, normalize_embeddings=True):
            if not texts:
                return np.zeros((0, dim), dtype="float32")
            v = np.zeros(dim, dtype="float32")
            if dim > 0:
                v[0] = 1.0
            return np.vstack([v.copy() for _ in texts])

    return FakeST


# ------------------------------------
# 1) Cover lines 8–9: import fallback
# ------------------------------------

def test_import_fallback_missing_sentence_transformers(monkeypatch):
    """
    Forces ImportError when importing 'sentence_transformers' so that the module-level
    except path sets SentenceTransformer = None, and then verifies ST backend raises.
    """
    embed_mod = _reload_embed_with_import_hook(monkeypatch, st_import_raises=True)
    from tvi.solphit.ingialla.embed import EmbedConfig, Embedder  # noqa: E402

    assert embed_mod.SentenceTransformer is None  # top-level fallback executed

    with pytest.raises(RuntimeError):
        _ = Embedder(EmbedConfig(backend="st", model="any", batch_size=8))


# ------------------------------------
# 2) ST backend: CUDA + CPU + torch fail + dim=0 edge
# ------------------------------------

def test_st_cuda_available_branch(monkeypatch):
    """
    Covers ST branch with CUDA available = True and dim>0.
    """
    embed_mod = _reload_embed_with_import_hook(monkeypatch, st_import_raises=False)

    FakeST = _fake_st_class(dim=5)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeST),
    )
    monkeypatch.setattr(embed_mod, "SentenceTransformer", FakeST, raising=True)

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    from tvi.solphit.ingialla.embed import EmbedConfig, Embedder  # noqa: E402

    e = Embedder(EmbedConfig(backend="st", model="unused", batch_size=2))
    assert e.dim == 5
    out = e.embed(["a", "b"])
    assert out.shape == (2, 5)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_st_cpu_branch_and_import_torch_failure(monkeypatch):
    """
    Covers CPU fallback when torch import fails inside _init_backend (dim>0).
    """
    embed_mod = _reload_embed_with_import_hook(monkeypatch, st_import_raises=False)

    FakeST = _fake_st_class(dim=7)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeST),
    )
    monkeypatch.setattr(embed_mod, "SentenceTransformer", FakeST, raising=True)

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "torch":
            raise ImportError("torch not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from tvi.solphit.ingialla.embed import EmbedConfig, Embedder  # noqa: E402

    e = Embedder(EmbedConfig(backend="st", model="unused", batch_size=3))
    assert e.dim == 7
    arr = e.embed([])
    assert arr.shape == (0, 7)


def test_st_dim_zero_edge(monkeypatch):
    """
    Covers ST path where SentenceTransformer reports dim=0, then embed([]) returns shape (0, 0).
    This exercises a distinct outcome through the ST block (lines ~31–32) vs the usual dim>0 path.
    """
    embed_mod = _reload_embed_with_import_hook(monkeypatch, st_import_raises=False)

    FakeST = _fake_st_class(dim=0)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeST),
    )
    monkeypatch.setattr(embed_mod, "SentenceTransformer", FakeST, raising=True)

    # Make CUDA unavailable to take the CPU branch (also fine)
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    from tvi.solphit.ingialla.embed import EmbedConfig, Embedder  # noqa: E402

    e = Embedder(EmbedConfig(backend="st", model="unused", batch_size=2))
    assert e.dim == 0
    out = e.embed([])
    assert out.shape == (0, 0)


# ------------------------------------
# 3) Ollama backend: probe + empty + norm + empty-probe edge + default URL
# ------------------------------------

def test_ollama_probe_and_empty_and_normalization(monkeypatch):
    """
    Covers Ollama path with env URL:
      - probe call on __init__ (to get dim=4)
      - empty input returns zeros with shape (0, 4)
      - batch call returns normalized vectors
    """
    embed_mod = _reload_embed_with_import_hook(monkeypatch, st_import_raises=False)

    calls = []

    class R:
        def __init__(self, payload):
            self._p = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._p

    def fake_post(url, json=None, timeout=0):
        calls.append((url, json))
        if json and json.get("input") == "probe":
            return R({"embeddings": [[0, 0, 0, 1]]})  # 4-D
        return R({"embeddings": [[1, 0, 0, 0], [0, 3, 0, 0]]})

    monkeypatch.setattr(embed_mod, "requests", types.SimpleNamespace(post=fake_post), raising=True)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    from tvi.solphit.ingialla.embed import EmbedConfig, Embedder  # noqa: E402

    e = Embedder(EmbedConfig(backend="ollama", model="nomic-embed-text", batch_size=128))
    assert e.dim == 4
    empty = e.embed([])
    assert empty.shape == (0, 4)

    arr = e.embed(["x", "y"])
    assert arr.shape == (2, 4)
    np.testing.assert_allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-6)
    assert any("/api/embed" in u for u, _ in calls)


def test_ollama_probe_empty_vector_sets_dim_zero_and_default_url(monkeypatch):
    """
    Covers alternate Ollama init where probe returns an empty vector (len 0), dim=0,
    and uses the default base URL (no env var). embed([]) returns (0, 0).
    """
    embed_mod = _reload_embed_with_import_hook(monkeypatch, st_import_raises=False)

    calls = []

    class R:
        def __init__(self, payload):
            self._p = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._p

    def fake_post(url, json=None, timeout=0):
        calls.append((url, json))
        if json and json.get("input") == "probe":
            return R({"embeddings": [[]]})  # dim=0
        return R({"embeddings": []})

    monkeypatch.setattr(embed_mod, "requests", types.SimpleNamespace(post=fake_post), raising=True)
    # Intentionally NOT setting OLLAMA_BASE_URL to hit default URL path

    from tvi.solphit.ingialla.embed import EmbedConfig, Embedder  # noqa: E402

    e = Embedder(EmbedConfig(backend="ollama", model="nomic-embed-text", batch_size=64))
    assert e.dim == 0
    out = e.embed([])
    assert out.shape == (0, 0)
    assert any("/api/embed" in u for u, _ in calls)


# ------------------------------------
# 4) Invalid backend in __init__ and post-init in embed()
# ------------------------------------

def test_invalid_backend_in_init_raises(monkeypatch):
    _reload_embed_with_import_hook(monkeypatch, st_import_raises=False)
    from tvi.solphit.ingialla.embed import EmbedConfig, Embedder  # noqa: E402
    with pytest.raises(ValueError):
        _ = Embedder(EmbedConfig(backend="bogus", model="m", batch_size=8))


def test_embed_invalid_backend_branch_after_init(monkeypatch):
    """
    Construct with valid backend, then flip cfg.backend to hit the defensive branch in embed().
    """
    embed_mod = _reload_embed_with_import_hook(monkeypatch, st_import_raises=False)

    FakeST = _fake_st_class(dim=3)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeST),
    )
    monkeypatch.setattr(embed_mod, "SentenceTransformer", FakeST, raising=True)

    from tvi.solphit.ingialla.embed import EmbedConfig, Embedder  # noqa: E402

    e = Embedder(EmbedConfig(backend="st", model="unused", batch_size=2))
    e.cfg.backend = "after-init-bogus"
    with pytest.raises(ValueError):
        _ = e.embed(["x"])


# ------------------------------------
# 5) QueryEmbedder subclass smoke
# ------------------------------------

def test_query_embedder_subclass_for_st(monkeypatch):
    embed_mod = _reload_embed_with_import_hook(monkeypatch, st_import_raises=False)

    FakeST = _fake_st_class(dim=6)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeST),
    )
    monkeypatch.setattr(embed_mod, "SentenceTransformer", FakeST, raising=True)

    from tvi.solphit.ingialla.embed import EmbedConfig, QueryEmbedder  # noqa: E402

    qe = QueryEmbedder(EmbedConfig(backend="st", model="unused", batch_size=2))
    vecs = qe.embed(["a", "b", "c"])
    assert vecs.shape == (3, 6)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-6)