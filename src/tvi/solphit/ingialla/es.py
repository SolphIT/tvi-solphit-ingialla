
from __future__ import annotations
import os, hashlib, time
from typing import List, Tuple, Optional, Iterable, Callable, Iterator
from elasticsearch import Elasticsearch, helpers
from elastic_transport import ConnectionError as ElasticConnectionError  # type: ignore
from tvi.solphit.base.logging import SolphitLogger

log = SolphitLogger.get_logger("tvi.solphit.ingialla.es")

ARTICLES = "kb_articles"
CHUNKS = "kb_chunks"
PERSONALITY = "personality"

def es_client() -> Elasticsearch:
    url = os.environ.get("ELASTIC_URL", "http://localhost:9200")
    headers = {
        "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8",
    }
    es = Elasticsearch(url, headers=headers, request_timeout=60)
    # Lightweight connectivity check; times out quickly if not reachable
    try:
        if not es.ping():
            raise RuntimeError("ES ping returned False")
    except Exception as ex:
        raise RuntimeError(
            f"Elasticsearch not reachable at {url}. "
            "Start ES (e.g., docker run -p 9200:9200 ...) or set $ELASTIC_URL."
        ) from ex
    return es

def ensure_articles_index():
    es = es_client()
    es.indices.create(
        index=ARTICLES,
        settings={"index": {"number_of_shards": 1, "number_of_replicas": 0, "codec": "best_compression"}},
        mappings={"properties": {
            "title": {"type": "text"},
            "xml_path": {"type": "keyword"},
            "split_done": {"type": "boolean"},
            "kb_done": {"type": "boolean"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        }},
        ignore=[400],
        request_timeout=60,
    )
    return es

def ensure_chunks_index(dims: int):
    es = es_client()
    es.indices.create(
        index=CHUNKS,
        settings={"index": {"number_of_shards": 3, "number_of_replicas": 0, "codec": "best_compression"}},
        mappings={"properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "title": {"type": "text", "fields": {"kw": {"type": "keyword"}}},
            "source_path": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "text": {"type": "match_only_text"},
            "vector": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"},
            "created_at": {"type": "date"},
        }},
        ignore=[400],
        request_timeout=120,
    )
    return es

def _art_id(xml_path: str) -> str:
    return hashlib.sha1(xml_path.encode("utf-8")).hexdigest()

def already_split(es: Elasticsearch, xml_path: str) -> bool:
    doc_id = _art_id(xml_path)
    try:
        resp = es.get(index=ARTICLES, id=doc_id, _source_includes=["split_done"], request_timeout=30)
        return bool(resp.get("_source", {}).get("split_done"))
    except Exception:
        return False

def mark_split_done(es: Elasticsearch, xml_path: str, title: str):
    now = int(time.time() * 1000)
    doc_id = _art_id(xml_path)
    es.index(index=ARTICLES, id=doc_id, document={
        "title": title, "xml_path": xml_path, "split_done": True, "kb_done": False,
        "created_at": now, "updated_at": now
    }, request_timeout=30)

def mark_kb_done(es: Elasticsearch, xml_path: str):
    now = int(time.time() * 1000)
    es.update(index=ARTICLES, id=_art_id(xml_path),
              doc={"kb_done": True, "updated_at": now},
              doc_as_upsert=True, request_timeout=30)

def get_unprocessed_for_kb(es: Elasticsearch, max_pages: Optional[int]) -> List[tuple[str, str]]:
    """
    Kept for backward compatibility (single-shot search). Limited by index.max_result_window.
    Prefer iter_unprocessed_for_kb() for streaming beyond 10k.
    """
    size = max_pages or 10000
    resp = es.search(index=ARTICLES, size=size,
                     query={"bool": {"filter": [{"term": {"split_done": True}}, {"term": {"kb_done": False}}]}},
                     request_timeout=60)
    return [(h["_source"].get("title",""), h["_source"]["xml_path"]) for h in resp["hits"]["hits"]]

def iter_unprocessed_for_kb(es: Elasticsearch, page_size: int = 1000) -> Iterator[tuple[str, str, str]]:
    """
    Stream all unprocessed articles using helpers.scan() (scroll under the hood),
    which bypasses the 10k 'from+size' limit. Yields (title, xml_path, _id).
    """
    query = {"bool": {"filter": [{"term": {"split_done": True}}, {"term": {"kb_done": False}}]}}
    for h in helpers.scan(
        client=es,
        index=ARTICLES,
        query={"query": query},
        size=max(1, int(page_size)),
        preserve_order=True,
        _source_includes=["title", "xml_path"],
        request_timeout=300,
    ):
        src = h.get("_source", {})
        title = src.get("title", "")
        xml_path = src.get("xml_path")
        if not xml_path:
            continue
        yield title, xml_path, h.get("_id")

def bulk_index_chunks(es: Elasticsearch, rows: Iterable[dict], progress_cb: Callable[[int], None] | None = None):
    """
    Streamed bulk indexing. If progress_cb is provided, it's called once per action
    (use it to update a tqdm bar without spamming the log).
    """
    actions = ({"_op_type": "index", "_index": CHUNKS, "_id": r["chunk_id"], **r} for r in rows)
    for ok, info in helpers.streaming_bulk(es, actions, chunk_size=2000, request_timeout=300):
        if progress_cb:
            try:
                progress_cb(1)
            except Exception:
                pass
        if not ok:
            # Keep failures in DEBUG to avoid noisy INFO logs
            log.debug("[es:bulk] failed action: %s", info)
