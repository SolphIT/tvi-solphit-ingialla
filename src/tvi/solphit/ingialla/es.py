from __future__ import annotations
import os, hashlib, time
from typing import List, Tuple, Optional, Iterable
from elasticsearch import Elasticsearch, helpers

ARTICLES = "kb_articles"
CHUNKS = "kb_chunks"

def es_client() -> Elasticsearch:
    url = os.environ.get("ELASTIC_URL", "http://localhost:9200")
    headers = {
        "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8",
    }
    return Elasticsearch(url, headers=headers, request_timeout=60)

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
    size = max_pages or 10000
    resp = es.search(index=ARTICLES, size=size,
                     query={"bool": {"filter": [{"term": {"split_done": True}}, {"term": {"kb_done": False}}]}},
                     request_timeout=60)
    return [(h["_source"].get("title",""), h["_source"]["xml_path"]) for h in resp["hits"]["hits"]]

def bulk_index_chunks(es: Elasticsearch, rows: Iterable[dict]):
    actions = ({"_op_type": "index", "_index": CHUNKS, "_id": r["chunk_id"], **r} for r in rows)
    helpers.bulk(es, actions, chunk_size=2000, request_timeout=300)
