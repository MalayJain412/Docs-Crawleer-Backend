# 📄 AI-Powered Documentation Crawler & Q/A System

A comprehensive solution for crawling documentation websites and providing intelligent Q/A capabilities using Retrieval-Augmented Generation (RAG).

## 🌟 Features

- **Intelligent Web Crawling**: Recursively crawl documentation websites with domain restriction
- **Content Extraction**: Clean text extraction using trafilatura
- **Dual Storage Format**: Save crawled content in both JSON and YAML formats
- **Domain-Based Organization**: Organize data by domain with separate folders for JSON, YAML, and FAISS indexes
- **Advanced Embeddings**: Support for Gemini (primary) and sentence-transformers (fallback)
- **Vector Search**: FAISS-based similarity search for fast retrieval
- **RAG Pipeline**: Intelligent Q/A using retrieved context and LLM generation
- **FastAPI REST API**: Complete API endpoints for all operations
- **Background Processing**: Async crawling and embedding generation

## 🏗️ Architecture

```mermaid
graph TD
    A[Start URL] --> B[Crawler Layer<br/>(aiohttp)]
    B --> C[Parser Layer<br/>(trafilatura)]
    C --> D[Storage Layer<br/>(JSON + YAML)]
    D --> E[All Docs Collected]
    E --> F[Embedding Layer<br/>(Gemini + ST fallback)]
    F --> G[Vector DB<br/>(FAISS)]
    G --> H[Q/A Agent<br/>(RAG Pipeline via FastAPI)]
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd docs-crawler

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and configure:

```env
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Crawler Settings
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=30
RETRY_ATTEMPTS=3
DELAY_BETWEEN_REQUESTS=1

# Storage Settings
DATA_DIR=./data
LOG_LEVEL=INFO

# Embedding Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
MAX_CHUNKS_PER_DOC=50

# FastAPI Settings
API_HOST=0.0.0.0
API_PORT=8000
```

### 3. Run the Server

```bash
cd src
python main.py
```

The API will be available at `http://localhost:8000`

## 📡 API Endpoints

### Health Check
```bash
GET /
```

### Start Crawling
```bash
POST /crawl
{
    "url": "https://docs.livekit.io",
    "domain_name": "livekit-docs",  # optional
    "max_depth": 10  # optional
}
```

### Generate Embeddings
```bash
POST /embed
{
    "domain": "livekit-docs"
}
```

### Query Documentation
```bash
POST /query
{
    "query": "How do I set up authentication?",
    "domain": "livekit-docs",
    "top_k": 5,
    "include_context": true
}
```

### List Domains
```bash
GET /domains
```

### Get Documents
```bash
GET /domains/{domain_name}/documents?format_type=json
```

### Check Task Status
```bash
GET /tasks/{task_id}
```

### System Status
```bash
GET /status
```

## 📁 Project Structure

```
src/
├── main.py                    # FastAPI application entry point
├── config/
│   ├── __init__.py
│   └── settings.py            # Configuration settings
├── crawler/
│   ├── __init__.py
│   ├── web_crawler.py         # Main crawler logic
│   └── content_parser.py      # Content extraction with trafilatura
├── storage/
│   ├── __init__.py
│   ├── storage_manager.py     # JSON/YAML storage with domain folders
│   └── schemas.py             # Data schemas
├── embeddings/
│   ├── __init__.py
│   ├── embedding_service.py   # Gemini + sentence-transformers
│   └── vector_store.py        # FAISS operations
├── qa/
│   ├── __init__.py
│   └── rag_pipeline.py        # RAG Q/A system
├── api/
│   ├── __init__.py
│   ├── endpoints.py           # FastAPI routes
│   └── models.py              # Pydantic models
└── utils/
    ├── __init__.py
    ├── url_utils.py           # URL normalization and validation
    └── logger.py              # Logging configuration

data/                          # Domain-based storage structure
├── {domain_name}/            # e.g., livekit-docs/
│   ├── json/                 # JSON files
│   ├── yaml/                 # YAML files
│   └── faiss/                # FAISS index files
└── logs/                     # Application logs
```

## 💾 Data Storage Structure

### Domain Folder Organization
```
data/
└── livekit-docs/                    # Domain folder
    ├── json/
    │   ├── crawl_session.json       # Crawl metadata
    │   ├── documents.json           # All documents
    │   └── individual/              # Individual document files
    │       ├── getting-started.json
    │       └── api-reference.json
    ├── yaml/
    │   ├── crawl_session.yaml       # Human-readable metadata
    │   ├── documents.yaml           # All documents
    │   └── individual/              # Individual document files
    │       ├── getting-started.yaml
    │       └── api-reference.yaml
    └── faiss/
        ├── index.faiss              # FAISS vector index
        ├── metadata.json            # Chunk metadata
        └── index_info.json          # Index information
```

## 🔧 Usage Examples

### Basic Crawling and Q/A

```python
import asyncio
import aiohttp
import json

async def crawl_and_query():
    base_url = "http://localhost:8000"
    
    # 1. Start crawling
    async with aiohttp.ClientSession() as session:
        # Crawl documentation
        crawl_data = {
            "url": "https://docs.livekit.io",
            "domain_name": "livekit-docs"
        }
        async with session.post(f"{base_url}/crawl", json=crawl_data) as resp:
            result = await resp.json()
            print(f"Crawling started: {result['message']}")
        
        # Wait for crawling to complete (check task status)
        await asyncio.sleep(30)  # Adjust based on site size
        
        # Generate embeddings
        embed_data = {"domain": "livekit-docs"}
        async with session.post(f"{base_url}/embed", json=embed_data) as resp:
            result = await resp.json()
            print(f"Embedding started: {result['message']}")
        
        # Wait for embedding generation
        await asyncio.sleep(10)
        
        # Query the documentation
        query_data = {
            "query": "How do I authenticate users?",
            "domain": "livekit-docs",
            "top_k": 3
        }
        async with session.post(f"{base_url}/query", json=query_data) as resp:
            result = await resp.json()
            print(f"Answer: {result['answer']}")
            print(f"Sources: {len(result['sources'])} found")

# Run the example
asyncio.run(crawl_and_query())
```

### Direct API Usage

```python
from src.crawler.web_crawler import WebCrawler
from src.storage.storage_manager import StorageManager
from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.vector_store import VectorStore
from src.qa.rag_pipeline import RAGPipeline

async def direct_usage():
    # 1. Crawl documentation
    async with WebCrawler() as crawler:
        session = await crawler.crawl_domain("https://docs.livekit.io")
        documents = crawler.crawled_documents
    
    # 2. Save to storage
    storage = StorageManager()
    domain_folder = storage.get_domain_folder(session.start_url, "livekit-docs")
    storage.save_documents(documents, domain_folder)
    
    # 3. Generate embeddings
    embedding_service = EmbeddingService()
    await embedding_service.initialize()
    chunks = await embedding_service.embed_documents(documents)
    
    # 4. Create vector store
    vector_store = VectorStore("livekit-docs", domain_folder)
    vector_store.create_index(chunks, "sentence-transformer")
    
    # 5. Setup RAG pipeline
    rag = RAGPipeline()
    await rag.initialize()
    rag.add_vector_store("livekit-docs", vector_store)
    
    # 6. Query
    from src.storage.schemas import QueryRequest
    query = QueryRequest(
        query="How do I set up authentication?",
        domain="livekit-docs"
    )
    response = await rag.query(query)
    print(f"Answer: {response.answer}")
```

## 🎯 Key Features Explained

### 1. Domain-Based Organization
- Each crawled domain gets its own folder
- Separate storage for JSON (machine-readable) and YAML (human-readable)
- FAISS indexes stored per domain for efficient retrieval

### 2. Intelligent Crawling
- Respects robots.txt and rate limits
- Domain restriction prevents crawling external sites
- Async processing for high performance
- Retry logic with exponential backoff

### 3. Advanced Content Processing
- Clean text extraction using trafilatura
- Automatic chunking for optimal embedding
- Metadata preservation (title, URL, links)
- Link classification (internal/external)

### 4. Flexible Embedding Strategy
- Primary: Gemini embeddings for high quality
- Fallback: sentence-transformers for reliability
- Configurable chunk size and overlap
- Efficient batch processing

### 5. Powerful Q/A System
- Retrieval-Augmented Generation (RAG)
- Context-aware answer generation
- Source attribution and confidence scoring
- Configurable retrieval parameters

## 🔍 Monitoring and Logging

### Check System Status
```bash
curl http://localhost:8000/status
```

### View Logs
```bash
tail -f data/logs/crawler_$(date +%Y%m%d).log
```

### Monitor Background Tasks
```bash
curl http://localhost:8000/tasks/{task_id}
```

## 🛠️ Customization

### Adding New Embedding Models
1. Extend `EmbeddingService` class
2. Add model initialization in `_init_*` methods
3. Implement embedding generation logic

### Custom Content Parsers
1. Extend `ContentParser` class
2. Override parsing methods as needed
3. Add custom metadata extraction

### Additional API Endpoints
1. Add routes in `endpoints.py`
2. Create request/response models in `models.py`
3. Implement business logic

## 🔒 Security Considerations

- API key management via environment variables
- Input validation on all endpoints
- Rate limiting (implement as needed)
- CORS configuration
- File path sanitization

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Issues**: Ensure Gemini API key is set correctly
   ```bash
   export GEMINI_API_KEY=your_key_here
   ```

3. **Memory Issues**: Reduce `MAX_CONCURRENT_REQUESTS` for large sites
   ```env
   MAX_CONCURRENT_REQUESTS=5
   ```

4. **Slow Crawling**: Adjust delay settings
   ```env
   DELAY_BETWEEN_REQUESTS=0.5
   ```

### Debug Mode
Enable debug logging:
```env
LOG_LEVEL=DEBUG
```

## 📝 License

[Add your license information here]

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the logs in `data/logs/`
- Review the API documentation at `http://localhost:8000/docs`