# AI-Powered Documentation Crawler & Q/A System - Copilot Instructions

## Architecture Overview
FastAPI-based RAG system for crawling docs and Q/A. Layered: Crawler → Parser → Storage → Embeddings → Vector Store → RAG Pipeline. Domain-based data isolation.

## Core Components
- **Backend**: `src/main.py`, `src/api/endpoints.py` (async REST API)
- **Storage**: `src/storage/` (domain folders: json/yaml/faiss)
- **Frontend**: `dfrontend/` (static web app, independent deployment)
- **RAG**: `src/qa/rag_pipeline.py` (Gemini LLM + FAISS search)
- **Embeddings**: `src/embeddings/` (sentence-transformers primary, Gemini fallback)

## Key Patterns
- **Domain Organization**: Data in `data/{domain}/` with json/yaml/faiss subdirs
- **Async-First**: Use `aiohttp`, not `requests`; all ops async with `await`
- **Import Strategy**: Absolute from `src/`; run `cd src && python main.py`; set `PYTHONPATH=/path/to/src`
- **Config**: `src/config/settings.py` via env vars; require `GEMINI_API_KEY`
- **Fallbacks**: Embedding: ST → Gemini; HTTP retries with backoff; dual JSON/YAML storage

## Development Workflows
- **Run Dev**: `cd src && python main.py`
- **Prod**: `gunicorn -c gunicorn.conf.py src.main:app`
- **API Dev**: Add routes in `DocumentCrawlerAPI._setup_routes()`; models in `src/api/models.py`; use `BackgroundTasks`
- **Storage**: Use `StorageManager` for domain folders; saves json+yaml auto
- **Vectors**: Per-domain `VectorStore`; FAISS with metadata; `MultiDomainVectorStore` for cross-domain

## Integration Points
- Crawler → Storage: `DocumentContent` objects persisted per domain
- Storage → Embeddings: `ContentChunk` with embeddings
- Embeddings → Vector: FAISS indexes from chunks
- Vector → RAG: Domain stores loaded for retrieval

## Common Gotchas
- Import errors: Run from `src/` or set PYTHONPATH
- Missing GEMINI_API_KEY: Fails LLM ops
- Domain isolation: Separate embed/index per domain
- Async: Don't mix sync/async; use await
- Memory: FAISS loads per domain; large domains use RAM

## Testing
- Integration: `test_crawl.py`
- Imports: `test_imports.py`
- API: `/docs` for OpenAPI
- Frontend: `dfrontend/test-copy.html`