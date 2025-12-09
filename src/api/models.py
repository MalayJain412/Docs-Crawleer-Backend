"""Pydantic models for FastAPI endpoints."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, HttpUrl, validator


class CrawlRequest(BaseModel):
    """Request model for crawling documentation."""
    url: str = Field(description="Starting URL for crawling")
    domain_name: Optional[str] = Field(None, description="Custom name for domain folder (optional)")
    base_domain: Optional[str] = Field(None, description="Base domain to restrict crawling (auto-detected if not provided)")
    max_depth: Optional[int] = Field(None, description="Maximum crawling depth (unlimited if not provided)")
    

class CrawlResponse(BaseModel):
    """Response model for crawl operations."""
    success: bool
    message: str
    domain: str
    domain_folder: str
    session_info: Dict[str, Any]
    statistics: Dict[str, Any]
    

class EmbedRequest(BaseModel):
    """Request model for generating embeddings."""
    domain: str = Field(description="Domain to generate embeddings for")
    

class EmbedResponse(BaseModel):
    """Response model for embedding operations."""
    success: bool
    message: str
    domain: str
    embedding_info: Dict[str, Any]
    total_chunks: int
    processing_time: float
    

class QueryRequest(BaseModel):
    """Request model for single-domain Q/A queries."""
    query: str = Field(description="Natural language query")
    domain: str = Field(description="Domain to search in")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of relevant chunks to retrieve")
    include_context: bool = Field(default=True, description="Whether to use LLM for answer generation")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "How to configure authentication?",
                "domain": "docs-site-1",
                "top_k": 5
            }
        }


class MultiDomainQueryRequest(BaseModel):
    """Request model for multi-domain Q/A queries."""
    query: str = Field(description="Natural language query")
    domains: List[str] = Field(description="Multiple domains to search", min_items=1, max_items=10)
    top_k: int = Field(default=5, ge=1, le=50, description="Number of relevant chunks to retrieve")
    include_context: bool = Field(default=True, description="Whether to use LLM for answer generation")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "How to configure authentication?",
                "domains": ["docs-site-1", "docs-site-2"],
                "top_k": 10
            }
        }
    

class QueryResponse(BaseModel):
    """Response model for Q/A queries."""
    query: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    

class DomainInfo(BaseModel):
    """Information about a crawled domain."""
    domain_name: str
    total_documents: int
    last_crawl: Optional[str] = None
    has_embeddings: bool = False
    embedding_info: Optional[Dict[str, Any]] = None
    statistics: Dict[str, Any] = Field(default_factory=dict)
    

class StatusResponse(BaseModel):
    """General status response."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    available_domains: List[str]
    system_info: Dict[str, Any]
    

class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    details: Optional[str] = None
    error_code: Optional[str] = None


class DomainsValidationRequest(BaseModel):
    """Request model for domain validation."""
    domains: List[str] = Field(description="List of domains to validate")


class DomainsValidationResponse(BaseModel):
    """Response model for domain validation."""
    domain_status: Dict[str, bool] = Field(description="Status of each domain")
    valid_domains: List[str] = Field(description="List of valid domains")
    invalid_domains: List[str] = Field(description="List of invalid domains")


class AvailableDomainsResponse(BaseModel):
    """Response model for available domains."""
    domains: List[str] = Field(description="List of available domains")
    total_count: int = Field(description="Total number of domains")


class MultiDomainQueryResponse(BaseModel):
    """Response model for multi-domain queries."""
    query: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    domains_searched: List[str] = Field(default_factory=list)
    total_results: int = 0
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    domain_status: Optional[Dict[str, bool]] = None