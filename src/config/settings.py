"""Configuration settings for the documentation crawler."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings."""
    
    # API Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    # Gemini model selection (configurable via environment)
    # Choose embedding and LLM models from available Gemini model names
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    GEMINI_LLM_MODEL: str = os.getenv("GEMINI_LLM_MODEL", "models/gemini-2.5-flash")
    
    # Model selection
    MODEL_USE: str = os.getenv("MODEL_USE", "gemini")
    
    # Jina Embeddings
    JINA_API_KEY: Optional[str] = os.getenv("JINA_API_KEY")
    
    # Azure OpenAI settings
    AZURE_OPENAI_ENDPOINT: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_DEPLOYMENT: Optional[str] = os.getenv("AZURE_DEPLOYMENT")
    AZURE_API_VERSION: Optional[str] = os.getenv("AZURE_API_VERSION")
    AZURE_EMBEDDING_DEPLOYMENT: str = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    AZURE_LLM_DEPLOYMENT: str = os.getenv("AZURE_DEPLOYMENT", "gpt-35-turbo")
    
    # Crawler Settings
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    RETRY_ATTEMPTS: int = int(os.getenv("RETRY_ATTEMPTS", "3"))
    DELAY_BETWEEN_REQUESTS: float = float(os.getenv("DELAY_BETWEEN_REQUESTS", "1"))
    
    # Storage Settings
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Embedding Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    MAX_CHUNKS_PER_DOC: int = int(os.getenv("MAX_CHUNKS_PER_DOC", "50"))
    
    # FAISS Settings
    FAISS_INDEX_TYPE: str = os.getenv("FAISS_INDEX_TYPE", "IndexFlatIP")
    VECTOR_DIMENSION: int = int(os.getenv("VECTOR_DIMENSION", "384"))
    
    # LLM/RAG Settings
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))  # Increased for comprehensive responses
    RAG_CONTEXT_CHUNKS: int = int(os.getenv("RAG_CONTEXT_CHUNKS", "5"))
    RAG_CONTENT_LENGTH: int = int(os.getenv("RAG_CONTENT_LENGTH", "1500"))
    
    # FastAPI Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8006"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"
    
    # Retry and Backoff Settings
    EMBED_BATCH_SIZE_JINA: int = int(os.getenv("EMBED_BATCH_SIZE_JINA", "10"))
    EMBED_BATCH_SIZE_GEMINI: int = int(os.getenv("EMBED_BATCH_SIZE_GEMINI", "5"))
    EMBED_BATCH_SIZE_AZURE: int = int(os.getenv("EMBED_BATCH_SIZE_AZURE", "10"))
    RETRY_MAX_ATTEMPTS: int = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    RETRY_BACKOFF_FACTOR: float = float(os.getenv("RETRY_BACKOFF_FACTOR", "1.0"))
    
    # FAISS Advanced Settings
    FAISS_INDEX_FACTORY: str = os.getenv("FAISS_INDEX_FACTORY", "IndexFlatIP")
    
    # Optional Cosmos DB settings
    USE_COSMOS_INDEX_REGISTRY: bool = os.getenv("USE_COSMOS_INDEX_REGISTRY", "false").lower() == "true"
    COSMOS_ENDPOINT: Optional[str] = os.getenv("COSMOS_ENDPOINT")
    COSMOS_KEY: Optional[str] = os.getenv("COSMOS_KEY")


# Global settings instance
settings = Settings()