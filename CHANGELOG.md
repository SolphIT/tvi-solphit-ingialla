# Changelog

All notable changes to **tvi-solphit-ingialla** will be documented in this file.

The format follows Semantic Versioning and the keep-a-changelog structure.

## [0.2.0] - 2025-11-16
### Added
- High-coverage, single-file tests for each module (embed, es, chunk, parsing, wikidump).
- Robust `get_article_path` filename shortening with hashed directories; clarified tests to assert **filename** shortening rather than full absolute path constraints.
- Documentation: comprehensive `README.md`, `CONTRIBUTING.md`.
- Project configuration for coverage (`branch=true`) as defined in `pyproject.toml`.

### Improved
- Test fixtures for redirect parsing fallback and Ollama/ST embedding branches.
- ES helpers tests consolidated; ensured `mark_kb_done` upsert semantics are validated.

### Fixed
- Edge-case assertions in tests for reserved Windows filenames and long paths.

## [0.1.0] - 2025-11-13
### Added
> Did not document