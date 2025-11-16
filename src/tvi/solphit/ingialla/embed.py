from __future__ import annotations
import os, requests, numpy as np
from dataclasses import dataclass
from typing import List

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

@dataclass
class EmbedConfig:
    backend: str
    model: str
    batch_size: int

class Embedder:
    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        self.dim = None
        self._init_backend()

    def _init_backend(self):
        be = self.cfg.backend.lower()
        if be == "st":
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not installed.")
            device = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
            except Exception:
                pass
            self.model = SentenceTransformer(self.cfg.model, device=device)
            self.dim = self.model.get_sentence_embedding_dimension()
        elif be == "ollama":
            self.ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            self.model_name = self.cfg.model
            r = requests.post(f"{self.ollama_url}/api/embed", json={"model": self.model_name, "input": "probe"}, timeout=60)
            r.raise_for_status()
            vec = (r.json().get("embeddings") or [[]])[0]
            self.dim = len(vec)
        else:
            raise ValueError(f"Unknown embeddings backend: {self.cfg.backend}")

    def embed(self, texts: List[str]) -> np.ndarray:
        be = self.cfg.backend.lower()
        if be == "st":
            embs = self.model.encode(texts, batch_size=self.cfg.batch_size, show_progress_bar=False, normalize_embeddings=True)
            arr = np.asarray(embs, dtype="float32")
            self.dim = arr.shape[1]
            return arr
        elif be == "ollama":
            arrs = []
            for i in range(0, len(texts), self.cfg.batch_size):
                batch = texts[i:i+self.cfg.batch_size]
                r = requests.post(f"{self.ollama_url}/api/embed", json={"model": self.model_name, "input": batch}, timeout=180)
                r.raise_for_status()
                embs = r.json().get("embeddings") or []
                import numpy as _np
                arrs.append(_np.vstack([_np.array(v, dtype="float32") for v in embs]))
            allv = np.vstack(arrs) if arrs else np.zeros((0, self.dim), dtype="float32")
            norms = np.linalg.norm(allv, axis=1, keepdims=True) + 1e-12
            return (allv / norms).astype("float32")
        else:
            raise ValueError("Invalid backend")

class QueryEmbedder(Embedder):
    pass
