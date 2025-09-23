"""Data schemas for the documentation crawler."""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, HttpUrl, Field
from datetime import datetime


class LinkInfo(BaseModel):
    """Information about a link found in a document."""
    url: str
    text: Optional[str] = None
    is_internal: bool = Field(default=False, description="Whether link is internal to docs domain")
    is_visited: bool = Field(default=False, description="Whether link has been crawled")


class DocumentContent(BaseModel):
    """Structured content of a crawled document."""
    url: str
    title: Optional[str] = None
    content: str = Field(description="Main text content extracted from the page")
    raw_html: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    links: List[LinkInfo] = Field(default_factory=list)
    crawled_at: datetime = Field(default_factory=datetime.now)
    content_length: int = Field(default=0, description="Length of extracted content")
    language: Optional[str] = None


class CrawlSession(BaseModel):
    """Information about a crawling session."""
    domain: str
    start_url: str
    domain_folder: str = Field(description="Folder name for storing domain data")
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    total_pages: int = Field(default=0)
    successful_pages: int = Field(default=0)
    failed_pages: int = Field(default=0)
    status: str = Field(default="in_progress")  # in_progress, completed, failed
    
    
class ContentChunk(BaseModel):
    """A chunk of content for embedding."""
    chunk_id: str = Field(description="Unique identifier for the chunk")
    source_url: str = Field(description="URL of the source document")
    content: str = Field(description="Text content of the chunk")
    chunk_index: int = Field(description="Position of chunk in the document")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    
class EmbeddingIndex(BaseModel):
    """Information about an embedding index."""
    domain: str
    total_chunks: int = Field(default=0)
    vector_dimension: int = Field(default=384)
    model_name: str = Field(description="Name of the embedding model used")
    created_at: datetime = Field(default_factory=datetime.now)
    index_file_path: str = Field(description="Path to FAISS index file")
    metadata_file_path: str = Field(description="Path to chunk metadata file")


class QueryRequest(BaseModel):
    """Request model for Q/A queries."""
    query: str = Field(description="Natural language query")
    domain: str = Field(description="Domain to search in")
    top_k: int = Field(default=5, description="Number of relevant chunks to retrieve")
    include_context: bool = Field(default=True, description="Whether to include source context")


class QueryResponse(BaseModel):
    """Response model for Q/A queries."""
    query: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: Optional[float] = None
    processing_time: Optional[float] = None