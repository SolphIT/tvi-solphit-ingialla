from __future__ import annotations
from typing import List

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text: return []
    chunks, step = [], max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        chunk = text[i:i+chunk_size]
        if chunk: chunks.append(chunk)
        if i + chunk_size >= len(text): break
    return chunks
