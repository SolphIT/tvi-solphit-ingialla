# Changelog
All notable changes to **tvi-solphit-ingialla** will be documented in this file.
The format follows Semantic Versioning and the keep-a-changelog structure.

## [0.3.3] - 2025-11-29
### Added
- logging/information for file splitting 

## [0.3.2] - 2025-11-29
### Added
- logging/information for build kb
 
## [0.3.1] - 2025-11-29
???

## [0.3.0] - 2025-11-17
### Added
- Conversational context support: Generator now accepts a history of prior turns (user/assistant) for true multi-turn Q&A.
- Streaming chat generation: Ollama backend supports `/api/chat` with streaming tokens, enabling live responses.
- API surface for history: `ask.py` and downstream consumers can pass `history` and tune `history_limit`.
- Backward-compatible: Single-turn and context-free flows remain supported.

### Changed
- Generator uses `/api/chat` endpoint for Ollama instead of `/api/generate` for chat-aware answers.
- Service layer (`discera_service/service.py`) now reconstructs conversation history from Elasticsearch and passes it to the generator.
- Improved error handling for streaming and non-streaming generation.

### Fixed
- Edge cases in history reconstruction (avoids duplicate user turns).
- Defensive fallback for partial or malformed streaming responses.

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