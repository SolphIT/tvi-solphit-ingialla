from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import time, json, hashlib
from typing import Optional

from tvi.solphit.ingialla.es import (
    ensure_articles_index,
    ensure_chunks_index,
    get_unprocessed_for_kb,
    mark_kb_done,
    bulk_index_chunks,
)
from tvi.solphit.ingialla.parsing import extract_title_and_text
from tvi.solphit.ingialla.clean import simple_wikitext_clean
from tvi.solphit.ingialla.chunk import chunk_text
from tvi.solphit.ingialla.embed import EmbedConfig, Embedder
from tvi.solphit.base.logging import SolphitLogger

log = SolphitLogger.get_logger("tvi.solphit.ingialla.kb")


@dataclass
class BuildConfig:
    input_dir: str
    output_dir: str
    db_path: str                # kept for legacy parity; not used
    include_redirects: bool
    chunk_size: int
    overlap: int
    embed_backend: str          # "st" | "ollama"
    embed_model: str            # e.g. "nomic-embed-text"
    embed_batch: int
    commit_every: int           # kept for parity; not used
    max_pages: Optional[int]


def build_kb(cfg: BuildConfig) -> dict:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docmap_path = out_dir / "docmap.jsonl"
    meta_path = out_dir / "meta.json"

    # Ensure indices and initialize embedder (also gives us the embedding dimension)
    es = ensure_articles_index()
    embedder = Embedder(EmbedConfig(cfg.embed_backend, cfg.embed_model, cfg.embed_batch))
    ensure_chunks_index(dims=embedder.dim)

    # Pull pages marked split_done && not kb_done
    articles = get_unprocessed_for_kb(es, cfg.max_pages)

    # If nothing to do, still emit a meta file with zeros so downstream tools are stable
    if not articles:
        log.info("No new articles to process.")
        meta = {
            "build_started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": asdict(cfg),
            "stats": {
                "n_docs": 0,
                "n_redirects_skipped": 0,
                "n_chunks": 0,
                "embedding_backend": cfg.embed_backend,
                "embedding_model": cfg.embed_model,
                "embedding_dim": embedder.dim,
                "seconds": 0.0,
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    t0 = time.time()
    n_docs = 0
    n_redirects = 0
    n_chunks = 0

    with docmap_path.open("a", encoding="utf-8") as docmap_f:
        for title, xml_path in articles:
            p = Path(xml_path)
            if not p.exists():
                log.warning(f"[SKIP] File not found: {xml_path}")
                continue

            title2, wikitext, is_redirect = extract_title_and_text(p)

            if is_redirect and not cfg.include_redirects:
                n_redirects += 1
                # Mark as done so it won't be re-queued on the next run
                mark_kb_done(es, xml_path)
                continue

            cleaned = simple_wikitext_clean(wikitext)
            chunks = chunk_text(cleaned, cfg.chunk_size, cfg.overlap)

            # Mark done even if empty so we don't retry forever
            if not chunks:
                mark_kb_done(es, xml_path)
                n_docs += 1
                continue

            vectors = embedder.embed(chunks)

            doc_id = hashlib.sha1(str(p).encode("utf-8")).hexdigest()
            now_ms = int(time.time() * 1000)

            # Prepare bulk rows (one row per chunk)
            rows = []
            for local_idx, ch in enumerate(chunks):
                chunk_id = hashlib.sha1((str(p) + "#" + str(local_idx)).encode("utf-8")).hexdigest()
                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": f"doc-{doc_id}",
                        "title": title2,
                        "source_path": str(p),
                        "chunk_index": local_idx,
                        "text": ch,
                        "vector": vectors[local_idx].tolist(),
                        "created_at": now_ms,
                    }
                )

            bulk_index_chunks(es, rows)

            # Append to docmap
            docmap_f.write(
                json.dumps(
                    {
                        "doc_id": f"doc-{doc_id}",
                        "title": title2,
                        "source_path": str(p),
                        "n_chunks": len(chunks),
                        "is_redirect": is_redirect,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            n_chunks += len(chunks)
            n_docs += 1
            mark_kb_done(es, xml_path)

    meta = {
        "build_started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(cfg),
        "stats": {
            "n_docs": n_docs,
            "n_redirects_skipped": n_redirects,
            "n_chunks": n_chunks,
            "embedding_backend": cfg.embed_backend,
            "embedding_model": cfg.embed_model,
            "embedding_dim": embedder.dim,
            "seconds": round(time.time() - t0, 2),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Incremental ES build complete.")
    return meta