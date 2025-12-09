# AI-Powered Documentation Crawler & Q/A System - Copilot Instructions

## Architecture Overview
FastAPI-based RAG system for crawling docs and Q/A. Layered: Crawler → Parser → Storage → Embeddings → Vector Store → RAG Pipeline. Domain-based data isolation with separate FAISS indexes per domain.

## Core Components
- **Backend**: `src/main.py`, `src/api/endpoints.py` (async REST API with background tasks)
- **Storage**: `src/storage/` (domain folders: json/yaml/faiss subdirs; dual JSON/YAML persistence)
- **Frontend**: `dfrontend/` (static web app, independent deployment)
- **RAG**: `src/qa/rag_pipeline.py` (Gemini LLM + FAISS search with configurable context chunks)
- **Embeddings**: `src/embeddings/` (sentence-transformers primary, Gemini fallback; chunking with overlap)
- **Config**: `src/config/settings.py` (dotenv-based env vars, no hardcoded values)

## Key Patterns
- **Domain Organization**: Data in `src/data/{domain}/` with json/yaml/faiss subdirs; per-domain vector stores
- **Multi-Domain Queries**: Frontend supports checkbox selection; backend uses `MultiDomainVectorStore` for cross-domain search
- **Async-First**: Use `aiohttp`, not `requests`; all ops async with `await`; background tasks for long-running ops
- **Import Strategy**: Absolute from `src/`; `run_server.py` handles PYTHONPATH; fallback imports in endpoints.py
- **Config**: `src/config/settings.py` via env vars; require `GEMINI_API_KEY`; server defaults to port 5002
- **Fallbacks**: Embedding: sentence-transformers → Gemini; HTTP retries with backoff; dual JSON/YAML storage
- **Routes**: Added in `DocumentCrawlerAPI._setup_routes()`; use `BackgroundTasks` for crawl/embed; task tracking with IDs
- **Models**: Pydantic in `src/api/models.py`; schemas in `src/storage/schemas.py`

## Development Workflows
- **Run Dev**: `python run_server.py` (handles PYTHONPATH, venv activation, runs on port 5002)
- **Quick Deploy**: `.\deploy.bat` (starts backend + frontend servers with proper venv setup)
- **Frontend**: `dfrontend/` serves static files; `dfrontend/js/config.js` for API endpoints
- **API Dev**: Add routes in `DocumentCrawlerAPI._setup_routes()`; models in `src/api/models.py`; use `BackgroundTasks`
- **Storage**: Use `StorageManager` for domain folders; saves json+yaml auto; `get_domain_folder()` for paths
- **Vectors**: Per-domain `VectorStore`; FAISS with metadata; `MultiDomainVectorStore` for cross-domain queries
- **Testing**: Integration in `test_crawl.py`; imports in `test_imports.py`; API docs at `/docs`

## Integration Points
- Crawler → Storage: `DocumentContent` objects persisted per domain via `StorageManager.save_documents()`
- Storage → Embeddings: `ContentChunk` with embeddings from `EmbeddingService.embed_documents()`
- Embeddings → Vector: FAISS indexes from chunks via `VectorStore.create_index()`
- Vector → RAG: Domain stores loaded for retrieval in `RAGPipeline.query()`
- API → All: Endpoints trigger background tasks; status via `/tasks/{task_id}`

## Common Gotchas
- Import errors: Use `python run_server.py`, not direct `main.py`; run_server.py sets PYTHONPATH automatically
- Missing GEMINI_API_KEY: Fails LLM ops; set in .env or venv activation
- Domain filtering: Embed dropdown shows domains WITHOUT embeddings; query shows domains WITH embeddings
- URL field mapping: Multi-domain results check multiple field names (`url`, `source_url`, `link`, `page_url`)
- Async: Don't mix sync/async; use await; background tasks for crawl/embed
- Memory: FAISS loads per domain; large domains use RAM; chunk size configurable
- Config: All via env vars; server runs on port 5002, frontend on 3000
- Routes: Decorators in _setup_routes(); not class methods
- Storage: Dual json/yaml; individual docs in subfolders with domain isolation

## Testing
- Integration: `test_crawl.py`
- Imports: `test_imports.py`
- API: `/docs` for OpenAPI
- Frontend: `dfrontend/test-copy.html`