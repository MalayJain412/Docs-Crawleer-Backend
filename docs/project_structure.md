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

.env                          # Environment variables
README.md                     # Project documentation
requirements.txt              # Dependencies