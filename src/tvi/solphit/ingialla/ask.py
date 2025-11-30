
# tvi/solphit/ingialla/ask_v23.py
from __future__ import annotations
import os
from typing import List, Sequence, Optional
import requests
from elasticsearch import Elasticsearch
from tvi.solphit.base.logging import SolphitLogger
from tvi.solphit.ingialla.es import CHUNKS

log = SolphitLogger.get_logger("tvi.solphit.ingialla.ask_v23")


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
    Very small LLM wrapper (Ollama | none), now chat-aware and with minimal vision support.

    - provider="none": echoes retrieved contexts (no model call).
    - provider="ollama": /api/chat with messages (supports stream/non-stream).
    - generate_vision(...): /api/chat with a base64 image on the user message.
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

    # ---------------------- system guards ----------------------
    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a helpful assistant. Use ONLY the provided context snippets when answering. "
            "Cite snippets with bracketed numbers like [1], [2]. "
            "If the context is insufficient or unrelated, say so and avoid guessing."
        )

    @staticmethod
    def _context_message(contexts: List[str]) -> str:
        if not contexts:
            return "No retrieval context available."
        labeled = [f"[{i}] {c}" for i, c in enumerate(contexts, start=1)]
        return "Context:\n" + "\n\n".join(labeled)

    @staticmethod
    def _build_messages(
        question: str,
        contexts: List[str],
        history: Optional[List[dict]] = None,
        system_preprompt: Optional[str] = None,
    ) -> List[dict]:
        messages: List[dict] = []
        # System guardrails + retrieval context as system
        sys_text = (system_preprompt or Generator._system_prompt()) + "\n\n" + Generator._context_message(contexts)
        messages.append({"role": "system", "content": sys_text})
        # Prior turns (already role-tagged "user"/"assistant")
        for m in (history or []):
            r = m.get("role")
            c = m.get("content", "")
            if r in {"user", "assistant"} and c:
                messages.append({"role": r, "content": c})
        # Current user question last
        messages.append({"role": "user", "content": question})
        return messages

    # ---------------------- text generation ----------------------
    def generate(
        self,
        question: str,
        contexts: List[str],
        *,
        history: Optional[List[dict]] = None,
        temperature: float = 0.2,
        timeout: int = 600,
        system_preprompt: Optional[str] = None,
    ) -> str:
        if self.provider == "none":
            labeled = [f"[{i}] {c}" for i, c in enumerate(contexts, start=1)]
            return "[Context only]\nQ: " + question + "\n\n" + "\n\n".join(labeled)

        # Ollama chat (non-streaming)
        messages = self._build_messages(question, contexts, history, system_preprompt)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": float(temperature)},
        }
        try:
            r = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json() or {}
            # Prefer /api/chat's canonical shape
            if "message" in data and isinstance(data["message"], dict):
                return (data["message"].get("content") or "").strip()
            if "response" in data:  # some builds mirror /api/generate fielding
                return (data.get("response") or "").strip()
            return (str(data) or "").strip()
        except Exception as ex:
            log.error(f"Ollama chat request failed: {ex}")
            labeled = [f"[{i}] {c}" for i, c in enumerate(contexts, start=1)]
            return "[Context only]\nQ: " + question + "\n\n" + "\n\n".join(labeled)

    def generate_stream(
        self,
        question: str,
        contexts: List[str],
        *,
        history: Optional[List[dict]] = None,
        temperature: float = 0.2,
        timeout: int = 600,
        system_preprompt: Optional[str] = None,
    ):
        """
        Stream token-like chunks from Ollama /api/chat with stream=True.
        Yields strings (may be partial words).
        """
        if self.provider != "ollama":
            # Fall back to non-streaming but yield once
            yield self.generate(
                question,
                contexts,
                history=history,
                temperature=temperature,
                timeout=timeout,
                system_preprompt=system_preprompt,
            )
            return

        messages = self._build_messages(question, contexts, history, system_preprompt)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": float(temperature)},
        }
        try:
            with requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=timeout, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    # Typical line JSON: {"message":{"role":"assistant","content":"..."}, "done":false}
                    # Final line: {"done":true}
                    try:
                        import json as _json
                        obj = _json.loads(line)
                        if obj.get("done"):
                            break
                        msg = obj.get("message") or {}
                        chunk = msg.get("content") or ""
                        if chunk:
                            yield chunk
                    except Exception:
                        # non-JSON keep-alives
                        pass
        except Exception as ex:
            log.error(f"Ollama chat stream failed: {ex}")
            # Degrade to one-shot
            yield self.generate(
                question,
                contexts,
                history=history,
                temperature=temperature,
                timeout=timeout,
                system_preprompt=system_preprompt,
            )

    # ---------------------- vision (the missing piece) ----------------------
    def generate_vision(
        self,
        image_b64: str,
        question: str,
        *,
        timeout: int = 600,
        guidelines: str = "",
    ) -> dict:
        """
        Minimal vision support for Ollama /api/chat:
          - user message includes 'images': [<base64>] and textual question.
          - returns a compact analysis dict: caption + raw json, with model hint fields.
        """
        if self.provider != "ollama":
            raise ValueError(f"Vision is only supported with provider='ollama'; got {self.provider!r}")

        sys_text = (guidelines or self._system_prompt())
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": question or "Describe the image objectively.", "images": [image_b64]},
        ]
        payload = {
            "model": self.model,  # e.g., "llava:latest"
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            r = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json() or {}
            text = ""
            if "message" in data and isinstance(data["message"], dict):
                text = (data["message"].get("content") or "").strip()
            elif "response" in data:
                text = (data.get("response") or "").strip()
            return {
                "caption": text or "",
                "objects": [],        # keep simple; could be extracted via follow-up prompt
                "ocr_text": "",       # optional; add OCR via separate flow if desired
                "model_version": data.get("model", ""),
                "cost_hint": None,
                "raw_json": data,
            }
        except Exception as ex:
            raise RuntimeError(f"Ollama vision request failed: {ex}")
