
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
    db_path: str  # kept for legacy parity; not used
    include_redirects: bool
    chunk_size: int
    overlap: int
    embed_backend: str  # "st" | "ollama"
    embed_model: str    # e.g. "nomic-embed-text"
    embed_batch: int
    commit_every: int   # kept for parity; not used
    max_pages: Optional[int]

def build_kb(cfg: BuildConfig) -> dict:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docmap_path = out_dir / "docmap.jsonl"
    meta_path = out_dir / "meta.json"

    # Ensure indices and initialize embedder (also gives us the embedding dimension)
    log.info("[phase:init] ensuring indices & initializing embedder backend=%s model=%s batch=%s",
             cfg.embed_backend, cfg.embed_model, cfg.embed_batch)
    es = ensure_articles_index()
    embedder = Embedder(EmbedConfig(cfg.embed_backend, cfg.embed_model, cfg.embed_batch))
    ensure_chunks_index(dims=embedder.dim)
    log.info("[phase:init] embedding dimension=%s", embedder.dim)

    # Pull pages marked split_done && not kb_done
    log.info("[phase:discover] querying ES for unprocessed pages (split_done && !kb_done) max_pages=%s",
             cfg.max_pages or "all")
    articles = get_unprocessed_for_kb(es, cfg.max_pages)
    log.info("[phase:discover] found %s articles to process", len(articles))

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
    last_report = t0

    with docmap_path.open("a", encoding="utf-8") as docmap_f:
        for idx, (title, xml_path) in enumerate(articles, start=1):
            p = Path(xml_path)
            if not p.exists():
                log.warning("[SKIP] File not found: %s", xml_path)
                continue

            log.debug("[article %s/%s] loading xml=%s", idx, len(articles), p)
            title2, wikitext, is_redirect = extract_title_and_text(p)
            if is_redirect and not cfg.include_redirects:
                n_redirects += 1
                # Mark as done so it won't be re-queued on the next run
                mark_kb_done(es, xml_path)
                log.debug("[article %s/%s] redirect skipped title=%r", idx, len(articles), title2)
                continue

            cleaned = simple_wikitext_clean(wikitext)
            chunks = chunk_text(cleaned, cfg.chunk_size, cfg.overlap)
            log.info("[article %s/%s] title=%r chunks=%s (size=%s overlap=%s)",
                     idx, len(articles), title2, len(chunks), cfg.chunk_size, cfg.overlap)

            # Mark done even if empty so we don't retry forever
            if not chunks:
                mark_kb_done(es, xml_path)
                n_docs += 1
                # also write to docmap so you can tail during long runs
                docmap_f.write(
                    json.dumps({"doc_id": f"doc-{hashlib.sha1(str(p).encode('utf-8')).hexdigest()}",
                                "title": title2, "source_path": str(p),
                                "n_chunks": 0, "is_redirect": is_redirect}, ensure_ascii=False) + "\n"
                )
                meta_path.write_text(json.dumps({
                    "build_started": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "config": asdict(cfg),
                    "stats": {
                        "n_docs": n_docs, "n_redirects_skipped": n_redirects, "n_chunks": n_chunks,
                        "embedding_backend": cfg.embed_backend,
                        "embedding_model": cfg.embed_model,
                        "embedding_dim": embedder.dim,
                        "seconds": round(time.time() - t0, 2),
                    }}, indent=2), encoding="utf-8")
                continue

            t1 = time.time()
            vectors = embedder.embed(chunks)
            log.info("[article %s/%s] embedded %s chunks in %.2fs",
                     idx, len(articles), len(chunks), time.time() - t1)

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
            log.debug("[article %s/%s] indexing %s chunk rows into ES", idx, len(articles), len(rows))
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

            # periodic progress report to meta.json
            if (time.time() - last_report) > 5:
                last_report = time.time()
                meta_path.write_text(json.dumps({
                    "build_started": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "config": asdict(cfg),
                    "stats": {
                        "n_docs": n_docs, "n_redirects_skipped": n_redirects, "n_chunks": n_chunks,
                        "embedding_backend": cfg.embed_backend,
                        "embedding_model": cfg.embed_model,
                        "embedding_dim": embedder.dim,
                        "seconds": round(time.time() - t0, 2),
                    }}, indent=2), encoding="utf-8")

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
    log.info("[phase:done] Incremental ES build complete. docs=%s chunks=%s seconds=%.2f",
             n_docs, n_chunks, time.time() - t0)
    return meta
