# Contributing

Thanks for your interest in improving **tvi-solphit-ingialla**!

## Development setup

```bash
git clone <your-repo-url>
cd tvi-solphit-ingialla
python -m venv .venv && source .venv/bin/activate
pip install -e ".[st]"    # optionally include extras you need
pip install pytest pytest-cov
```

The package uses a src/ layout with the namespace tvi.solphit.ingialla.