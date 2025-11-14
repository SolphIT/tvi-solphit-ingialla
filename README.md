# tvi-solphit-ingialla

Ingialla provides higher-level utilities under the shared `tvi.solphit` namespace, building on `tvi-solphit-base` for consistent logging and conventions.

- **Distribution name (PyPI):** `tvi-solphit-ingialla`
- **Import path:** `tvi.solphit.ingialla`
- **Namespace packaging:** PEP 420 (no `__init__.py` in `tvi/` or `tvi/solphit/`)

> **Author:** Tobias Hagenbeek (SolphIT LLC)

## Installation

```bash
pip install tvi-solphit-ingialla
```

> Requires Python >=3.9.

## Quickstart

```python
from tvi.solphit.ingialla import hello

print(hello("world"))
```

### Logging via the base package
```bash
export TVI_SOLPHIT_LOG_LEVEL=DEBUG
python - <<'PY'
from tvi.solphit.ingialla import hello
hello("SolphIT")
PY
```

### Development
```python
python -m venv .venv
source .venv/bin/activate
pip install -U pip build twine pytest
pip install -e .     # editable install
pytest -q
python -m build
```

## Versioning

This project uses Semantic Versioning. See the CHANGELOG.