
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import time, json, hashlib
from typing import Optional, Callable, Set
from tqdm import tqdm

from tvi.solphit.ingialla.es import (
    ensure_articles_index,
    ensure_chunks_index,
    iter_unprocessed_for_kb,
    get_unprocessed_for_kb,  # still used for initial "found N" snapshot
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

@dataclass
class BuildUiOptions:
    """UI/verbosity knobs for nicer progress output."""
    log_every: int = 50            # summary every N docs
    status_interval: float = 5.0   # write kb-status.json every X seconds
    show_titles: bool = False      # INFO log per title; otherwise DEBUG only
    inline_summary: bool = True    # update a single line via tqdm postfix (no new lines)

@dataclass
class BuildStreamOptions:
    """Streaming knobs beyond 10k and watch for new work while running."""
    scan_size: int = 1000          # scan/scroll page size
    watch_seconds: float = 0.0     # if > 0, after finishing one scan round, sleep and rescan
    watch_idle_rounds: int = 0     # stop after N consecutive rounds with zero new IDs (0 = infinite)

def _write_status(status_path: Path, *, n_docs: int, n_redirects: int, n_chunks: int,
                  embed_backend: str, embed_model: str, dim: int,
                  t0: float, last_title: str | None):
    now = time.time()
    elapsed = max(0.001, now - t0)
    status = {
        "n_docs": n_docs,
        "n_redirects_skipped": n_redirects,
        "n_chunks": n_chunks,
        "embedding_backend": embed_backend,
        "embedding_model": embed_model,
        "embedding_dim": dim,
        "elapsed_seconds": round(elapsed, 2),
        "rate_docs_per_sec": round(n_docs / elapsed, 3),
        "rate_chunks_per_sec": round(n_chunks / elapsed, 3),
        "last_title": (last_title or "")[:120],
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    except Exception as ex:
        log.debug("Failed to write kb status file %s: %s", status_path, ex)

def build_kb(cfg: BuildConfig, ui: BuildUiOptions | None = None,
             stream: BuildStreamOptions | None = None) -> dict:
    """Build the KB (clean → chunk → embed → index) with bar-based UI and streaming beyond 10k."""
    ui = ui or BuildUiOptions()
    stream = stream or BuildStreamOptions()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docmap_path = out_dir / "docmap.jsonl"
    meta_path = out_dir / "meta.json"
    status_path = out_dir / "kb-status.json"

    # Ensure indices and initialize embedder (also gives us the embedding dimension)
    log.info("[phase:init] ensuring indices & initializing embedder backend=%s model=%s batch=%s",
             cfg.embed_backend, cfg.embed_model, cfg.embed_batch)
    es = ensure_articles_index()
    embedder = Embedder(EmbedConfig(cfg.embed_backend, cfg.embed_model, cfg.embed_batch))
    ensure_chunks_index(dims=embedder.dim)
    log.info("[phase:init] embedding dimension=%s", embedder.dim)

    # Snapshot: how many are visible right now (bounded by 10k search window, informational only)
    visible_now = len(get_unprocessed_for_kb(es, cfg.max_pages))
    log.info("[phase:discover] visible now=%s (informational, streaming will pull all)", visible_now)

    # --- Bars -----------------------------------------------------------------
    # Position 0: overall docs processed (unknown total → None)
    docs_bar = tqdm(total=None, unit="doc", desc="Docs", leave=True, position=0)
    # Position 1: total chunks indexed
    chunks_bar = tqdm(total=None, unit="chunk", desc="Chunks", leave=True, position=1)

    t0 = time.time()
    last_status_write = 0.0
    n_docs = 0
    n_redirects = 0
    n_chunks = 0
    last_title: str | None = None

    processed_ids: Set[str] = set()
    rounds_without_new = 0

    # Helper to write meta + status periodically
    def _periodic_meta_status():
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
        _write_status(status_path, n_docs=n_docs, n_redirects=n_redirects, n_chunks=n_chunks,
                      embed_backend=cfg.embed_backend, embed_model=cfg.embed_model, dim=embedder.dim,
                      t0=t0, last_title=last_title)

    # --- Streaming + optional watching ---------------------------------------
    with docmap_path.open("a", encoding="utf-8") as docmap_f:
        while True:
            new_ids_this_round = 0
            # Stream all current unprocessed docs
            for title, xml_path, _id in iter_unprocessed_for_kb(es, page_size=stream.scan_size):
                if cfg.max_pages is not None and n_docs >= cfg.max_pages:
                    break
                # Skip duplicates within the same invocation
                if _id in processed_ids:
                    continue
                processed_ids.add(_id)
                new_ids_this_round += 1

                p = Path(xml_path)
                if not p.exists():
                    log.warning("[SKIP] File not found: %s", xml_path)
                    # represent slot consumed even when file missing
                    docs_bar.update(1)
                    n_docs += 1
                    continue

                # Extract text
                title2, wikitext, is_redirect = extract_title_and_text(p)
                last_title = title2

                # Redirect handling
                if is_redirect and not cfg.include_redirects:
                    n_redirects += 1
                    mark_kb_done(es, xml_path)
                    docs_bar.update(1)
                    # Inline summary or log line for skips
                    summary = f"skip redirect title='{(title2 or 'untitled')[:40]}'"
                    if ui.inline_summary:
                        docs_bar.set_postfix_str(summary)
                    else:
                        log.info("[article %s] %s", n_docs + 1, summary)
                    n_docs += 1
                    continue

                # Clean + chunk
                cleaned = simple_wikitext_clean(wikitext)
                chunks = chunk_text(cleaned, cfg.chunk_size, cfg.overlap)
                chunk_count = len(chunks)

                # Mark done even if empty so we don't retry forever
                if not chunks:
                    mark_kb_done(es, xml_path)
                    n_docs += 1
                    docs_bar.update(1)
                    # docmap entry (0 chunks)
                    docmap_f.write(
                        json.dumps({
                            "doc_id": f"doc-{hashlib.sha1(str(p).encode('utf-8')).hexdigest()}",
                            "title": title2, "source_path": str(p),
                            "n_chunks": 0, "is_redirect": is_redirect
                        }, ensure_ascii=False) + "\n"
                    )
                    # periodic status/meta by time
                    now = time.time()
                    if (now - last_status_write) >= max(0.5, float(ui.status_interval)):
                        last_status_write = now
                        _periodic_meta_status()
                    continue

                # --- Embedding (timed, postfix) -----------------------------------
                t1 = time.time()
                vectors = embedder.embed(chunks)
                embed_secs = time.time() - t1

                # --- Indexing (streaming, updates the chunks bar) ------------------
                t2 = time.time()
                def _cb(_n: int):
                    chunks_bar.update(_n)
                bulk_index_chunks(es, (
                    {
                        "chunk_id": hashlib.sha1((str(p) + "#" + str(local_idx)).encode("utf-8")).hexdigest(),
                        "doc_id": f"doc-{hashlib.sha1(str(p).encode('utf-8')).hexdigest()}",
                        "title": title2,
                        "source_path": str(p),
                        "chunk_index": local_idx,
                        "text": ch,
                        "vector": vectors[local_idx].tolist(),
                        "created_at": int(time.time() * 1000),
                    }
                    for local_idx, ch in enumerate(chunks)
                ), progress_cb=_cb)
                index_secs = time.time() - t2

                # Append to docmap
                docmap_f.write(
                    json.dumps(
                        {
                            "doc_id": f"doc-{hashlib.sha1(str(p).encode('utf-8')).hexdigest()}",
                            "title": title2,
                            "source_path": str(p),
                            "n_chunks": len(chunks),
                            "is_redirect": is_redirect,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                # Counters + bars
                n_chunks += chunk_count
                n_docs += 1
                mark_kb_done(es, xml_path)
                docs_bar.update(1)

                # Inline summary or log line for this doc
                summary = (
                    f"title='{(title2 or 'untitled')[:40]}' "
                    f"chunks={chunk_count} (size={cfg.chunk_size} overlap={cfg.overlap}) "
                    f"embed={embed_secs:.2f}s index={index_secs:.2f}s"
                )
                if ui.inline_summary:
                    docs_bar.set_postfix_str(summary)
                else:
                    log.info("[article %s] %s", n_docs, summary)

                # periodic progress report to meta.json & status file
                now = time.time()
                if (now - last_status_write) >= max(0.5, float(ui.status_interval)) or (n_docs % max(1, ui.log_every) == 0):
                    last_status_write = now
                    _periodic_meta_status()

            # Round complete — decide whether to watch for new work
            if cfg.max_pages is not None and n_docs >= cfg.max_pages:
                break
            if stream.watch_seconds > 0:
                if new_ids_this_round == 0:
                    rounds_without_new += 1
                    if stream.watch_idle_rounds > 0 and rounds_without_new >= stream.watch_idle_rounds:
                        log.info("[watch] no new work for %s rounds; exiting.", rounds_without_new)
                        break
                else:
                    rounds_without_new = 0
                # short sleep before rescanning
                time.sleep(stream.watch_seconds)
                continue
            # no watching → we’re done
            break

    # Final meta
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
    # Close bars
    docs_bar.close()
    chunks_bar.close()
    log.info("[phase:done] KB build complete. docs=%s chunks=%s seconds=%.2f",
             n_docs, n_chunks, time.time() - t0)
    return meta
