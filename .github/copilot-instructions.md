# AI-Powered Documentation Crawler Copilot Instructions

## Architecture Overview

This is a **RAG-powered documentation crawler** with a 3-stage pipeline: **Crawl → Embed → Query**. The system is domain-oriented, storing each crawled site separately with JSON/YAML dual formats and FAISS vector indexes.

```
FastAPI ← → [Crawler → Parser → Storage] → [Embeddings → FAISS] → [RAG Pipeline]
```

## Key Components & Data Flow

### 1. **Domain-Based Storage Pattern**
- Each crawled site creates a domain folder: `data/{domain}/json/`, `data/{domain}/yaml/`, `data/{domain}/faiss/`
- Always use `StorageManager.get_domain_folder()` to ensure proper structure
- JSON for machine processing, YAML for human inspection, FAISS for vector search

### 2. **Async-First Architecture**  
- **All** I/O operations use async/await (crawler, embeddings, API)
- WebCrawler uses `aiohttp.ClientSession` with configurable concurrency limits
- Background tasks tracked in `DocumentCrawlerAPI.background_tasks` dict

### 3. **Import Path Management**
Critical: This project uses complex path resolution. Always follow the import pattern:
```python
# In any src/ file:
try:
    from config.settings import settings
    from utils.logger import default_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    # then repeat imports
```

## Development Workflows

### Running the System
```bash
# Always run from src/ directory
cd src
python main.py  # Starts FastAPI server on port 8000

# Alternative: from project root
python run_server.py  # Wrapper script
```

### Testing Workflow
1. **Crawl**: `POST /crawl {"url": "https://docs.example.com", "domain_name": "example-docs"}`
2. **Embed**: `POST /embed {"domain": "example-docs"}`  
3. **Query**: `POST /query {"query": "...", "domain": "example-docs"}`

Use `test_crawl.py` for development testing - it contains working examples.

## Critical Configuration Patterns

### Environment Setup
- Copy `.env.example` → `.env` and set `GEMINI_API_KEY`
- Settings in `src/config/settings.py` use dotenv with intelligent defaults
- Storage paths always relative to `DATA_DIR` setting
- **LLM Configuration**: New settings for enhanced responses: `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `RAG_CONTEXT_CHUNKS`

### RAG Pipeline Architecture
- **Dual Response System**: LLM-powered comprehensive responses with intelligent fallback
- **Primary**: Gemini-generated structured responses using complex prompt templates
- **Fallback**: Rich structured responses from `_create_fallback_answer()` when LLM unavailable
- **Prompt Engineering**: Senior engineer-level prompt in `_generate_llm_answer()` produces actionable, copy-pasteable guidance

### Embedding Strategy
- **Primary**: Gemini embeddings via `google-generativeai`
- **Fallback**: sentence-transformers when Gemini unavailable  
- Chunking: 1000 chars with 100 overlap, max 50 chunks per doc

### Error Handling Conventions
- Use `default_logger` from `utils.logger` consistently
- Background tasks store status in `background_tasks` dict with task_id keys
- API returns proper HTTP status codes with detailed error messages
- **RAG Failures**: Graceful degradation from LLM → structured fallback → basic response

## Integration Points

### Adding New Endpoints
Extend `DocumentCrawlerAPI._setup_routes()` in `src/api/endpoints.py`. Follow async patterns:
```python
@self.app.post("/new-endpoint", response_model=ResponseModel)
async def new_endpoint(request: RequestModel):
    # Always use try/except with proper logging
    # Return structured responses using Pydantic models
```

### Enhancing RAG Responses
- **Prompt Templates**: Modify comprehensive prompt in `RAGPipeline._generate_llm_answer()`
- **Fallback Logic**: Enhance `_create_fallback_answer()` for structured responses without LLM
- **Content Extraction**: Use helper methods like `_extract_implementation_steps()`, `_extract_code_examples()`
- **Response Structure**: Follow senior engineer pattern: TL;DR → Overview → Prerequisites → Implementation → Examples

### Extending Crawling Logic  
- Modify `WebCrawler` in `src/crawler/web_crawler.py`
- Content parsing via `ContentParser` uses trafilatura for clean text extraction
- URL normalization handled by `URLUtils.normalize_url()`

### Custom Embedding Providers
Implement in `EmbeddingService.generate_embedding()` with fallback pattern:
```python
try:
    # Try primary provider
    return await self._gemini_embedding(text)
except Exception as e:
    # Fall back to sentence-transformers
    return self._sentence_transformer_embedding(text)
```

## Project-Specific Conventions

- **Domain names** extracted via `URLUtils.extract_domain_name()` or custom via API
- **File naming**: Individual docs saved as `{url_slug}.json` in `individual/` subdirs
- **Vector metadata**: Stored alongside FAISS indexes with document chunk mappings
- **Dual storage**: Every document saved in both JSON and YAML simultaneously
- **Logging**: Module-specific loggers via `setup_logger(module_name, level)`

## Debugging Tips

1. **Import issues**: Always run from `src/` directory, check Python path setup
2. **Storage problems**: Verify domain folder structure with `StorageManager.get_domain_folder()`
3. **Embedding failures**: Check both Gemini API key and sentence-transformers fallback
4. **Crawler issues**: Monitor `visited_urls` set and `failed_urls` for debugging loops
5. **API issues**: Check background task status via `/tasks/{task_id}` endpoint
6. **RAG Response Quality**: 
   - If getting basic responses: verify `GEMINI_API_KEY` is uncommented in `.env`
   - Check `self._llm_client` initialization in `RAGPipeline.__init__()`
   - Monitor fallback vs LLM usage in logs
   - Adjust `LLM_TEMPERATURE`, `LLM_MAX_TOKENS` for response style

## Common Patterns

- **Task tracking**: Generate UUID, store in `background_tasks`, return task_id to client
- **Domain validation**: Always validate domain exists before operations via `storage_manager.get_domain_info()`
- **Async resource management**: Use context managers for aiohttp sessions and file operations
- **Configuration access**: Import `settings` globally, access via `settings.PROPERTY_NAME`
- **Response Enhancement**: Use `_create_fallback_answer()` pattern for structured responses even without LLM
- **Content Analysis**: Implement helper methods for extracting steps, code examples, and topics from documentation