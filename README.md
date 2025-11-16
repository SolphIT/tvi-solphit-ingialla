# tvi-solphit-ingialla

**Ingialla** is the domain logic layer for SolphIT’s knowledge base tooling: it splits Wikipedia-like dumps into per-article files, cleans & chunks text, builds embeddings, stores/searches vectors in Elasticsearch, and provides thin Q&A helpers.

> Modules included: `wikidump`, `parsing`, `clean`, `chunk`, `embed`, `es`, and light `ask/KB` helpers. [1](https://tobivisionco-my.sharepoint.com/personal/tobias_opcenter_io/Documents/Microsoft%20Copilot%20Chat%20Files/README.md)

---

## Table of contents

- Features
- Quick install
- Configuration
- Quickstart
  - Split a dump to articles
  - Clean and chunk text
  - Create embeddings
  - Elasticsearch helpers
  - Lightweight “ask” helpers
- CLI / scripts
- Development
- Testing & coverage
- Versioning & changelog
- License

---

## Features

- **Wikidump splitter**: streams a dump, writes each `<page>` to a hashed directory tree with safe file names; handles “already split”, page limits, and write errors. [2](https://tobivisionco-my.sharepoint.com/personal/tobias_opcenter_io/Documents/Microsoft%20Copilot%20Chat%20Files/test_wikidump_filename_and_path.py)  
- **Filename safety**: `safe_filename(title, max_len)` trims/normalizes and protects Windows-reserved names (e.g., `CON`, `COM1`) with a short hash suffix. [2](https://tobivisionco-my.sharepoint.com/personal/tobias_opcenter_io/Documents/Microsoft%20Copilot%20Chat%20Files/test_wikidump_filename_and_path.py)  
- **Parsing**: extract title/text and detect redirects (either `<redirect/>` element or `#REDIRECT` text) with an iterparse fast path and a robust fallback. [3](https://tobivisionco-my.sharepoint.com/personal/tobias_opcenter_io/Documents/Microsoft%20Copilot%20Chat%20Files/wikidump.py)  
- **Chunking**: `chunk_text(text, chunk_size, overlap)` with overlap and guard rails for edge cases (empty text, extreme overlap). [2](https://tobivisionco-my.sharepoint.com/personal/tobias_opcenter_io/Documents/Microsoft%20Copilot%20Chat%20Files/test_wikidump_filename_and_path.py)  
- **Embeddings**: unified `Embedder` for SentenceTransformers (`backend="st"`) and Ollama (`backend="ollama"`), with batch support and L2-normalized vectors. [1](https://tobivisionco-my.sharepoint.com/personal/tobias_opcenter_io/Documents/Microsoft%20Copilot%20Chat%20Files/README.md)  
- **Elasticsearch utilities**: client factory, index creation (`ARTICLES`, `CHUNKS`), bulk indexing, “already split” checks, and “KB done” markers. [1](https://tobivisionco-my.sharepoint.com/personal/tobias_opcenter_io/Documents/Microsoft%20Copilot%20Chat%20Files/README.md)

---

## Quick install

```bash
# Base
pip install tvi-solphit-ingialla

# Optional: SentenceTransformers backend for embeddings
pip install "tvi-solphit-ingialla[st]"

# (Ollama backend has no PyPI dependency here; you only need a running Ollama.)
```

## Configuration

Environment variables the modules honor:

ELASTIC_URL — Elasticsearch URL (default: http://localhost:9200).
OLLAMA_BASE_URL — Ollama base URL (default: http://localhost:11434).

## Quickstart
### Split a dump to articles

```python
from tvi.solphit.ingialla.wikidump import extract_articles

saved = extract_articles(
    xml_path="/path/to/dump.xml",
    output_dir="/path/to/articles",
    max_pages=None,  # or an int limit
)
print(f"Saved {saved} pages")
```

- Creates a hashed directory tree under output_dir.
- Uses get_article_path(...) + safe_filename(...) to keep filenames cross‑platform safe.
- Skips pages already split (already_split), logs write errors, and marks split completion.

### Clean and chunk text

```python 
from tvi.solphit.ingialla.clean import simple_wikitext_clean
from tvi.solphit.ingialla.chunk import chunk_text

text = """
== Heading ==
[[File:x.png]]
Some content...
"""

clean = simple_wikitext_clean(text)
chunks = chunk_text(clean, chunk_size=500, overlap=50)
```

- chunk_text creates overlapping windows; protects against empty input and odd overlaps.

### Create embeddings

```python
import numpy as np
from tvi.solphit.ingialla.embed import EmbedConfig, Embedder

cfg = EmbedConfig(backend="st", model="all-MiniLM-L6-v2", batch_size=32)
embedder = Embedder(cfg)
vectors: np.ndarray = embedder.embed(["A sentence", "Another sentence"])
print(vectors.shape)
```

- backend="st" uses SentenceTransformers (CUDA if available); backend="ollama" probes dimension at init and normalizes outputs.

### Elasticsearch helpers

```python
from tvi.solphit.ingialla.es import (
    es_client, ensure_articles_index, ensure_chunks_index,
    bulk_index_chunks, get_unprocessed_for_kb, mark_kb_done
)

es = ensure_articles_index()
ensure_chunks_index(dims=384)

rows = get_unprocessed_for_kb(es, max_pages=None)  # -> List[(title, xml_path)]
# ... build chunks + vectors ...
# bulk_index_chunks(es, list_of_chunk_dicts)
# mark_kb_done(es, xml_path)
```

- Includes dense vector mapping (cosine) for the CHUNKS index and robust defaults for timeouts/headers.

### Lightweight “ask” helpers

- KNN wrapper & generator utilities (Ollama or “none” pass‑through). See tests for example usage.

## CLI / scripts

No CLI is shipped; use the module APIs directly. (You can add a small wrapper script that calls extract_articles(...), then chunk_text(...) + Embedder to prep your KB.)

## Development

```bash
# clone your repo
python -m venv .venv && source .venv/bin/activate
pip install -e ".[st]"  # include ST extra if you use that backend
pip install -r requirements-dev.txt  # if you keep one; otherwise:
pip install pytest pytest-cov

# run tests
pytest -q --maxfail=1 --disable-warnings
```

- Source layout uses src/ with namespace package tvi.solphit.ingialla.

## Testing & coverage

We use single-file, per‑module tests aimed at 100% line coverage and high branch coverage.

```bash
pytest -q --maxfail=1 --disable-warnings \
  --cov=tvi.solphit.ingialla --cov-report=term-missing
```
Coverage configuration is in pyproject.toml (branch = true, source = ["tvi.solphit.ingialla", "ingialla"]).

## Versioning & changelog

- Semantic Versioning (MAJOR.MINOR.PATCH). 
- See CHANGELOG.md for details.

## License

This project is licensed under the terms found in LICENSE.