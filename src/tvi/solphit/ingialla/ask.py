from __future__ import annotations

import os
from typing import List, Sequence, Optional

import requests
from elasticsearch import Elasticsearch

from tvi.solphit.base.logging import SolphitLogger
from tvi.solphit.ingialla.es import CHUNKS

log = SolphitLogger.get_logger("tvi.solphit.ingialla.ask")


def knn_search(
    es: Elasticsearch,
    index: str,
    field: str,
    qvec: Sequence[float],
    k: int,
    num_candidates: int = 1000,
    include_fields: Optional[Sequence[str]] = None,
):
    """
    Run an ES dense_vector k-NN search.

    :param es: Elasticsearch client
    :param index: index name (e.g., CHUNKS)
    :param field: vector field name (e.g., "vector")
    :param qvec: query vector (1D)
    :param k: top-k
    :param num_candidates: HNSW candidate pool per shard
    :param include_fields: optional list of fields to include in _source
    """
    return es.search(
        index=index,
        knn={
            "field": field,
            "query_vector": list(qvec),
            "k": int(k),
            "num_candidates": int(num_candidates),
        },
        _source=list(include_fields) if include_fields else ["title", "source_path", "chunk_index", "text"],
    )


class Generator:
    """
    Very small LLM wrapper (Ollama | none).
    - provider="none": returns the retrieved contexts (no generation).
    - provider="ollama": calls Ollama /api/generate with a grounded prompt.
    """

    def __init__(self, provider: str, model: str, verbose: bool = False) -> None:
        self.provider = (provider or "").lower()
        self.model = model
        self.verbose = verbose

        if self.provider == "ollama":
            self.ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        elif self.provider == "none":
            self.ollama_url = None
        else:
            raise ValueError(f"Unknown LLM provider: {provider!r}")

    def generate(self, question: str, contexts: List[str], *, temperature: float = 0.2, timeout: int = 600) -> str:
        """
        :param question: user question
        :param contexts: list of context snippets (already labeled/ordered by caller)
        :param temperature: sampling temperature for Ollama
        :param timeout: request timeout (seconds)
        """
        if self.provider == "none":
            # Just echo the retrieved contexts (no model call)
            return "[Context only]\nQ: " + question + "\n\n" + "\n\n".join(contexts)

        # Ollama generation
        sys_prompt = (
            "You are a helpful assistant. Answer the question using ONLY the provided context. "
            "Cite sources in brackets with their labels (e.g., [1], [2]). "
            "If you are unsure or the context is insufficient, say so."
        )
        labeled = [f"[{i}] {c}" for i, c in enumerate(contexts, start=1)]
        user_prompt = f"Question: {question}\n\nContext:\n" + "\n\n".join(labeled)

        payload = {
            "model": self.model,
            "prompt": f"{sys_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {"temperature": float(temperature)},
        }

        try:
            r = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or "").strip()
        except Exception as ex:
            log.error(f"Ollama request failed: {ex}")
            # Graceful fallback: return the contexts so callers still see something
            return "[Context only]\nQ: " + question + "\n\n" + "\n\n".join(contexts)