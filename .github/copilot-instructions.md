# AI-Powered Documentation Crawler & Q/A System - Copilot Instructions

## Architecture Overview

This is a FastAPI-based RAG (Retrieval-Augmented Generation) system that crawls documentation websites and provides intelligent Q/A capabilities. The system follows a layered architecture with domain-based data organization.

### Core Components

- **FastAPI Backend** (`src/main.py`, `src/api/endpoints.py`): REST API with async background task processing
- **Domain-Based Storage** (`src/storage/`): Each crawled domain gets isolated JSON/YAML/FAISS storage 
- **Dual Frontend**: Static web UI in `dfrontend/` + FastAPI serves API endpoints
- **RAG Pipeline** (`src/qa/rag_pipeline.py`): Gemini LLM + FAISS vector search for Q/A
- **Embedding Strategy** (`src/embeddings/`): sentence-transformers primary, Gemini fallback

### Data Flow Pattern

```
URL → WebCrawler → ContentParser → StorageManager → EmbeddingService → VectorStore → RAGPipeline
```

## Key Patterns & Conventions

### 1. Domain-Based Organization
All data is organized by domain in `data/{domain-name}/`:
```
data/docs-livekit-io/
├── json/          # Machine-readable storage
├── yaml/          # Human-readable storage  
└── faiss/         # Vector indexes + metadata
```

### 2. Async-First Architecture
- All major operations are async: crawling, embedding, querying
- Background task tracking via `DocumentCrawlerAPI.background_tasks`
- Use `aiohttp` for web requests, not `requests`

### 3. Import Path Strategy
The codebase uses absolute imports from `src/`. Always ensure:
- Run from `src/` directory or set `PYTHONPATH=/path/to/src`
- Main entry: `python src/main.py` or `cd src && python main.py`
- Import pattern: `from config.settings import settings` (not relative imports)

### 4. Configuration Management
Settings centralized in `src/config/settings.py` using environment variables:
- **Required**: `GEMINI_API_KEY` for LLM operations
- **Key Settings**: `MAX_CONCURRENT_REQUESTS`, `CHUNK_SIZE`, `LOG_LEVEL`
- Use `settings.PROPERTY_NAME` throughout codebase

### 5. Error Handling & Fallbacks
- Embedding: sentence-transformers → Gemini fallback 
- HTTP: Retry logic with exponential backoff in `WebCrawler`
- Storage: Both JSON and YAML formats for redundancy
- Always check `try/except ImportError` blocks for module loading

## Development Workflows

### Running the System
```bash
# Development server
cd src && python main.py

# Production (uses gunicorn.conf.py)
gunicorn -c gunicorn.conf.py src.main:app
```

### API Development Pattern
New endpoints in `src/api/endpoints.py` follow this pattern:
1. Add route to `DocumentCrawlerAPI._setup_routes()`
2. Create request/response models in `src/api/models.py`
3. Use `BackgroundTasks` for long-running operations
4. Return task IDs for status tracking

### Storage Operations
Always use `StorageManager` for file operations:
- `get_domain_folder()` for path resolution
- Saves both JSON and YAML automatically
- Individual documents saved in `individual/` subdirectories

### Vector Operations
- Each domain gets its own `VectorStore` instance
- FAISS indexes stored with metadata in domain folders
- Use `MultiDomainVectorStore` for cross-domain queries

## Critical Integration Points

### 1. Crawler → Storage
`WebCrawler.crawl_domain()` returns `DocumentContent` objects that `StorageManager` persists in domain-specific folders.

### 2. Storage → Embeddings  
`EmbeddingService.embed_documents()` takes `DocumentContent` list and returns `ContentChunk` objects with embeddings.

### 3. Embeddings → Vector Store
`VectorStore.create_index()` builds FAISS indexes from `ContentChunk` objects, storing metadata separately.

### 4. Vector Store → RAG
`RAGPipeline` loads domain-specific vector stores and uses them for retrieval during Q/A.

## Frontend Integration

The `dfrontend/` contains a standalone static web app:
- **API Communication**: Uses `js/api.js` to call FastAPI endpoints
- **Configuration**: Backend URL set in `js/config.js`
- **Deployment**: Completely independent from backend (static hosting)

## Deployment Considerations

### Environment Setup
- **Development**: Run from `src/` with `.env` file
- **Production**: Uses `gunicorn` with Uvicorn workers
- **Azure**: Container Apps deployment with Blob Storage integration

### Data Persistence
- **Local**: `data/` directory with domain folders
- **Azure**: Blob Storage integration via `src/storage/azure_blob.py`
- **Vector Indexes**: FAISS files persisted alongside metadata

### Scaling Points
- Concurrent crawling: `MAX_CONCURRENT_REQUESTS` setting
- Memory usage: FAISS indexes loaded per domain
- Background tasks: Tracked in-memory (consider Redis for production)

## Common Gotchas

1. **Import Errors**: Always run from `src/` or set `PYTHONPATH`
2. **Missing API Keys**: Gemini operations fail without `GEMINI_API_KEY`
3. **Domain Isolation**: Each domain needs separate embedding/indexing workflow
4. **Async Context**: Most operations require `await` - don't mix sync/async
5. **FAISS Memory**: Large domains can consume significant RAM for vector indexes

## Testing Patterns

- **Integration Tests**: Use `test_crawl.py` for end-to-end crawler testing
- **Import Validation**: `test_imports.py` validates all module dependencies
- **API Testing**: FastAPI auto-generates OpenAPI docs at `/docs`
- **Local Development**: Use `dfrontend/test-copy.html` for frontend testing