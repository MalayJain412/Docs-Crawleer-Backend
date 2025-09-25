<!-- AI assistant guide for quickly understanding and contributing to this repo -->
# Copilot instructions — AI contributor quick reference

Purpose
- Short, actionable guidance so an AI coding agent can be productive immediately in this FastAPI + FAISS project.

High-level architecture (the "why")
- FastAPI app (src/main.py + src/api/endpoints.py) exposes async endpoints to: crawl sites, save documents, create embeddings, and answer queries via a RAG pipeline.
- Crawl → Parse → Store (JSON + YAML) → Embed → FAISS vector store → RAG query
- Async-first throughout: crawler, embeddings, and many I/O helpers use asyncio/aiohttp/async SDKs.

Key directories & responsibilities
- `src/` — application source
  - `main.py` — app entrypoint (creates FastAPI app; used by gunicorn/uvicorn)
  - `api/` — FastAPI routes (`endpoints.py`), Pydantic request/response models (`models.py`)
  - `crawler/` — `web_crawler.py`, `content_parser.py` (aiohttp + trafilatura)
  - `storage/` — `storage_manager.py` (domain folders, JSON/YAML), `azure_blob.py` (async blob helpers)
  - `embeddings/` — `embedding_service.py` (Gemini primary, sentence-transformers fallback), `vector_store.py` (FAISS operations)
  - `qa/` — `rag_pipeline.py` (RAG orchestration and domain index management)

Project-specific conventions
- Domain-based storage: data/{domain}/json, data/{domain}/yaml, data/{domain}/faiss. Use StorageManager.get_domain_folder().
- Dual-format persistence: every dataset saved in both JSON (machine) and YAML (human) for the same domain.
- Async-first I/O: prefer async helpers (see `storage/azure_blob.py` which exposes async and sync wrappers).
- Import fallback pattern: modules try absolute imports then append project root to sys.path for direct execution; preserve this when adding files.

Developer workflows (commands you can run)
- Install deps: `pip install -r requirements.txt`
- Run locally (dev):
  - from project root: `cd src` then `python main.py` (main uses uvicorn)
  - or use gunicorn as in Dockerfile / Procfile: `gunicorn --config gunicorn.conf.py src.main:app`
- Docker build & test (from repo root):
  - `docker build -t faiss-backend .`
  - `docker run -e API_PORT=8000 -p 8000:8000 faiss-backend`

Important env & config
- `src/config/settings.py` holds defaults. Common env vars used:
  - `API_HOST`, `API_PORT`, `API_RELOAD`, `LOG_LEVEL`
  - `DATA_DIR` — base path for domain folders
  - `GEMINI_API_KEY` — primary embedding provider
  - `AZURE_STORAGE_CONNECTION_STRING` or `AZURE_STORAGE_ACCOUNT_URL` — storage access
  - `AZURE_BLOB_CONTAINER` (default `faiss-indexes`)

Storage & cloud integration notes
- Blob access: `src/storage/azure_blob.py` supports both connection-string and Managed Identity (DefaultAzureCredential). When using Container Apps prefer Managed Identity and set `AZURE_STORAGE_ACCOUNT_URL=https://<account>.blob.core.windows.net`.
- FAISS indexes are stored in `data/{domain}/faiss/` alongside `metadata.json` and `index_info.json`.

Embedding & RAG specifics
- `embedding_service.py` tries Gemini (google-generativeai) first, falls back to sentence-transformers. Keep batched embedding shape and chunking (CHUNK_SIZE, CHUNK_OVERLAP) consistent with settings.
- VectorStore/FAISS operations live in `embeddings/vector_store.py` and are domain-scoped.

Patterns for new code
- Prefer async functions for I/O; if a sync consumer exists, provide small sync wrappers that call asyncio.run(...) (see `azure_blob.py`).
<!-- AI assistant guide for quickly understanding and contributing to this repo -->
# Copilot instructions — AI contributor quick reference

Purpose
- Quick, actionable notes to make an AI coding agent productive in this FastAPI + FAISS RAG crawler.

Core architecture (short)
- FastAPI app: `src/main.py` + `src/api/endpoints.py` expose async endpoints for crawling, saving, embedding and RAG querying.
- Pipeline: Crawl -> Parse -> Store (JSON + YAML) -> Embed -> FAISS (domain-scoped) -> RAG pipeline (`src/qa/rag_pipeline.py`).

Key folders & files (what to open first)
- `src/main.py` — app entry (used by uvicorn/gunicorn).
- `src/api/endpoints.py` + `src/api/models.py` — public API surface and Pydantic shapes.
- `src/crawler/web_crawler.py`, `src/crawler/content_parser.py` — fetching and extraction (async + trafilatura).
- `src/storage/storage_manager.py`, `src/storage/azure_blob.py` — domain folder layout + cloud helpers.
- `src/embeddings/embedding_service.py`, `src/embeddings/vector_store.py` — embedding providers + FAISS ops.
- `src/qa/rag_pipeline.py` — retrieval + LLM answer generation and fallback.

Important conventions (project-specific)
- Domain-based storage: each domain under `data/{domain}/json/`, `yaml/`, `faiss/`. Use `StorageManager.get_domain_folder()` to construct paths.
- Dual persistence: every dataset saved as JSON (machine) and YAML (human) — keep both updated when modifying storage logic.
- Async-first: prefer async functions for I/O. If a sync API is required, add a thin wrapper that uses `asyncio.run(...)` (see `azure_blob.py`).
- Import-fallback: many modules attempt absolute imports then append repo `src/` to `sys.path` for direct execution; preserve that pattern to avoid import errors.

Env, run & quick checks
- Defaults: `src/config/settings.py`. Common env vars: `API_PORT`, `DATA_DIR`, `GEMINI_API_KEY`, `AZURE_STORAGE_CONNECTION_STRING` / `AZURE_STORAGE_ACCOUNT_URL`, `AZURE_BLOB_CONTAINER`.
- Install: `pip install -r requirements.txt`.
- Run (dev): cd into `src/` then `python main.py` (uses uvicorn). Or from project root: `python run_server.py`.
- Docker: `docker build -t faiss-backend .` then `docker run -e API_PORT=8000 -p 8000:8000 faiss-backend`.
- Smoke: GET `/` or `/health` to verify server; `test_crawl.py` and `test_imports.py` are quick local checks.

Embedding & RAG notes
- `embedding_service.py` prefers Gemini (`google-generativeai`) and falls back to sentence-transformers. Respect CHUNK_SIZE/CHUNK_OVERLAP constants when chunking text.
- FAISS files live in `data/{domain}/faiss/` with `index.faiss`, `index_info.json`, and `metadata.json`.
- RAG pipeline includes LLM-first answers with a structured fallback (`_create_fallback_answer()`) — change prompts in `RAGPipeline._generate_llm_answer()`.

Where to be careful (gotchas)
- Always run from `src/` if encountering import errors; the repo uses import fallback but tests and main expect `src` as working dir.
- StorageManager is authoritative for folder names; changing layout will need updates across crawler, embeddings, and QA code.
- When adding endpoints, register routes in `DocumentCrawlerAPI._setup_routes()` (see `src/api/endpoints.py`) and add corresponding Pydantic models in `src/api/models.py`.

Examples (copy-paste patterns)
- Add a new endpoint:
  - File: `src/api/endpoints.py` under DocumentCrawlerAPI._setup_routes(); return Pydantic model from `src/api/models.py`.
- Use storage manager:
  - from `src.storage.storage_manager import StorageManager`
  - path = StorageManager.get_domain_folder(domain, subfolder='faiss')

Next steps
- If any area needs expansion (CI/CD, tests, or cloud deployment steps), tell me which section to expand and I will iterate.

---
Please review this condensed guide and tell me any missing examples or workflows to add.