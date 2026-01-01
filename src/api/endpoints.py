"""FastAPI endpoints for the documentation crawler."""

import asyncio
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from config.settings import settings
    from utils.logger import default_logger
    from utils.url_utils import URLUtils
    from storage.storage_manager import StorageManager
    from storage.schemas import CrawlSession, DocumentContent, EmbeddingIndex
    from crawler.web_crawler import WebCrawler
    from embeddings.embedding_service import EmbeddingService
    from embeddings.vector_store import VectorStore
    from qa.rag_pipeline import RAGPipeline
    from api.models import (
        CrawlRequest, CrawlResponse, EmbedRequest, EmbedResponse,
        QueryRequest, MultiDomainQueryRequest, QueryResponse, DomainInfo, StatusResponse,
        HealthResponse, ErrorResponse, DomainsValidationRequest,
        DomainsValidationResponse, AvailableDomainsResponse, MultiDomainQueryResponse
    )
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from config.settings import settings
    from utils.logger import default_logger
    from utils.url_utils import URLUtils
    from storage.storage_manager import StorageManager
    from storage.schemas import CrawlSession, DocumentContent, EmbeddingIndex
    from crawler.web_crawler import WebCrawler
    from embeddings.embedding_service import EmbeddingService
    from embeddings.vector_store import VectorStore
    from qa.rag_pipeline import RAGPipeline
    from api.models import (
        CrawlRequest, CrawlResponse, EmbedRequest, EmbedResponse,
        QueryRequest, MultiDomainQueryRequest, QueryResponse, DomainInfo, StatusResponse,
        HealthResponse, ErrorResponse, DomainsValidationRequest,
        DomainsValidationResponse, AvailableDomainsResponse, MultiDomainQueryResponse
    )


class DocumentCrawlerAPI:
    """FastAPI application for the documentation crawler."""
    
    def __init__(self):
        """Initialize the API."""
        self.app = FastAPI(
            title="AI-Powered Documentation Crawler & Q/A System",
            description="Crawl documentation websites and provide intelligent Q/A capabilities",
            version="1.0.0"
        )
        
        self.logger = default_logger
        self.storage_manager = StorageManager()
        self.rag_pipeline = RAGPipeline()
        
        # Background task tracking
        self.background_tasks: Dict[str, Dict[str, Any]] = {}
        
        self._setup_middleware()
        self._setup_routes()
    
    async def initialize_services(self):
        """Initialize services on startup."""
        try:
            await self.rag_pipeline.initialize()
            
            # Load existing vector stores
            await self._load_existing_domains()
            
            self.logger.info("API services initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize API services: {e}")
            raise
    
    async def cleanup_services(self):
        """Clean up resources on shutdown."""
        try:
            # Close any open resources
            self.logger.info("API services cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during API cleanup: {e}")
    
    async def check_readiness(self) -> Dict[str, Any]:
        """Check if services are ready."""
        try:
            # Check if RAG pipeline is initialized
            rag_ready = self.rag_pipeline._llm_client is not None
            
            # Check if embedding service is initialized
            embedding_ready = hasattr(self.rag_pipeline, '_embedding_service') and \
                            self.rag_pipeline._embedding_service._current_model is not None
            
            # Check available domains
            domains = await self._get_available_domains()
            
            return {
                "rag_pipeline": rag_ready,
                "embedding_service": embedding_ready,
                "available_domains": len(domains),
                "domains": domains
            }
        except Exception as e:
            self.logger.error(f"Readiness check failed: {e}")
            return {"error": str(e)}
        
    def _setup_middleware(self):
        """Setup CORS middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure as needed
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                available_domains = self.rag_pipeline.get_available_domains()
                
                return HealthResponse(
                    status="healthy",
                    version="1.0.0",
                    available_domains=available_domains,
                    system_info={
                        "total_domains": len(available_domains),
                        "embedding_service": "initialized",
                        "rag_pipeline": "initialized"
                    }
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint for monitoring and frontend connection testing."""
            try:
                available_domains = self.rag_pipeline.get_available_domains()
                
                return HealthResponse(
                    status="healthy",
                    version="1.0.0",
                    available_domains=available_domains,
                    system_info={
                        "total_domains": len(available_domains),
                        "embedding_service": "initialized",
                        "rag_pipeline": "initialized",
                        "timestamp": time.time()
                    }
                )
                
            except Exception as e:
                default_logger.error(f"Health check failed: {e}")
                return HealthResponse(
                    status="unhealthy",
                    version="1.0.0",
                    available_domains=[],
                    system_info={
                        "total_domains": 0,
                        "embedding_service": "error",
                        "rag_pipeline": "error",
                        "timestamp": time.time(),
                        "error": str(e)
                    }
                )
        
        @self.app.post("/crawl", response_model=CrawlResponse)
        async def crawl_documentation(request: CrawlRequest, background_tasks: BackgroundTasks):
            """Start crawling a documentation website."""
            try:
                # Validate URL
                if not URLUtils.is_valid_url(request.url):
                    raise HTTPException(status_code=400, detail="Invalid URL provided")
                
                # Create domain folder
                domain_folder = self.storage_manager.get_domain_folder(
                    request.url, 
                    request.domain_name
                )
                
                # Generate domain name and unique task ID with timestamp
                domain = request.base_domain or URLUtils.extract_domain_name(request.url)
                domain_name_safe = domain.replace('.', '-').replace('_', '-').lower()  # Safe for task ID
                task_id = f"crawl_{domain_name_safe}_{int(time.time() * 1000)}"
                
                # Initialize task with complete structure
                self.background_tasks[task_id] = {
                    "task_id": task_id,
                    "status": "started",
                    "message": f"Initializing crawl for {request.url}",
                    "progress": 0,
                    "domain": domain,
                    "start_time": time.time(),
                    "url": request.url,
                    "end_time": None,
                    "error": None
                }
                
                # Start crawling in background
                background_tasks.add_task(
                    self._crawl_task,
                    task_id,
                    request.url,
                    request.base_domain,
                    request.max_depth,
                    domain_folder
                )
                
                return CrawlResponse(
                    success=True,
                    message=f"Crawling started for {request.url}",
                    domain=domain,
                    domain_folder=domain_folder,
                    session_info={"task_id": task_id, "status": "started"},
                    statistics={}
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Crawl request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/embed", response_model=EmbedResponse)
        async def generate_embeddings(request: EmbedRequest, background_tasks: BackgroundTasks):
            """Generate embeddings for a crawled domain."""
            try:
                # Find domain folder
                domains = self.storage_manager.list_domains()
                domain_info = next((d for d in domains if d['domain_name'] == request.domain), None)
                
                if not domain_info:
                    raise HTTPException(status_code=404, detail=f"Domain '{request.domain}' not found")
                
                domain_folder = domain_info['domain_folder']
                
                # Generate unique task ID with domain name
                domain_name_safe = request.domain.replace('.', '-').replace('_', '-').lower()  # Safe for task ID
                task_id = f"embed_{domain_name_safe}_{int(time.time() * 1000)}"
                
                # Initialize task with complete structure
                self.background_tasks[task_id] = {
                    "task_id": task_id,
                    "status": "started",
                    "message": f"Initializing embedding generation for {request.domain}",
                    "progress": 0,
                    "domain": request.domain,
                    "start_time": time.time(),
                    "end_time": None,
                    "error": None
                }
                
                # Start embedding in background
                background_tasks.add_task(
                    self._embed_task,
                    task_id,
                    request.domain,
                    domain_folder
                )
                
                return EmbedResponse(
                    success=True,
                    message=f"Embedding generation started for domain '{request.domain}'",
                    domain=request.domain,
                    embedding_info={"task_id": task_id, "status": "started"},
                    total_chunks=0,
                    processing_time=0.0
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Embed request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/query", response_model=QueryResponse)
        async def query_documentation(request: QueryRequest):
            """Query the documentation using enhanced RAG pipeline."""
            start_time = time.time()
            
            try:
                # Validate domain
                if request.domain not in self.rag_pipeline.get_available_domains():
                    available_domains = self.rag_pipeline.get_available_domains()
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Domain '{request.domain}' not available or no embeddings found. Available domains: {available_domains}"
                    )
                
                # Log the query for monitoring
                self.logger.info(f"Processing query for domain '{request.domain}': {request.query[:100]}{'...' if len(request.query) > 100 else ''}")
                
                # Process query
                from storage.schemas import QueryRequest as StorageQueryRequest
                storage_request = StorageQueryRequest(
                    query=request.query,
                    domain=request.domain,
                    top_k=request.top_k,
                    include_context=request.include_context
                )
                
                response = await self.rag_pipeline.query(storage_request)
                
                # Log response metrics
                processing_time = time.time() - start_time
                self.logger.info(f"Query completed for domain '{request.domain}' in {processing_time:.3f}s, confidence: {response.confidence:.3f}")
                
                return QueryResponse(
                    query=response.query,
                    answer=response.answer,
                    sources=response.sources,
                    confidence=response.confidence,
                    processing_time=response.processing_time
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Query failed for domain '{request.domain}': {str(e)}")
                raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
        
        @self.app.post("/query-multi-domain", response_model=MultiDomainQueryResponse)
        async def query_multi_domain(request: MultiDomainQueryRequest):
            """Query across multiple domains simultaneously."""
            start_time = time.time()
            
            try:
                # Get domains list directly (no normalization needed)
                domains = request.domains
                
                # Validate domains list
                if len(domains) > 10:  # Reasonable limit
                    raise HTTPException(
                        status_code=400, 
                        detail="Too many domains requested (max 10)"
                    )
                
                # Log the query for monitoring
                self.logger.info(f"Processing multi-domain query for domains {domains}: {request.query[:100]}{'...' if len(request.query) > 100 else ''}")
                
                # Call RAG pipeline
                result = await self.rag_pipeline.answer_query_multi_domain(
                    query=request.query,
                    domains=domains,
                    top_k=request.top_k
                )
                
                # Log response metrics
                processing_time = time.time() - start_time
                self.logger.info(f"Multi-domain query completed for domains {domains} in {processing_time:.3f}s")
                
                return MultiDomainQueryResponse(
                    query=request.query,
                    answer=result.get("answer", ""),
                    sources=result.get("sources", []),
                    domains_searched=result.get("domains_searched", domains),
                    total_results=result.get("total_results", 0),
                    processing_time=processing_time,
                    domain_status=result.get("domain_status")
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Multi-domain query failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Multi-domain query processing failed: {str(e)}")
        
        @self.app.get("/domains/available", response_model=AvailableDomainsResponse)
        async def get_available_domains():
            """Get list of all available domains with FAISS indexes."""
            try:
                available_domains = []
                # Use the same data directory as StorageManager
                data_dir = Path(settings.DATA_DIR)
                
                if data_dir.exists():
                    for domain_path in data_dir.iterdir():
                        if domain_path.is_dir() and domain_path.name != 'logs':
                            faiss_dir = domain_path / "faiss"
                            index_file = faiss_dir / "index.faiss"
                            
                            if index_file.exists():
                                available_domains.append(domain_path.name)
                
                # Sort domains alphabetically
                available_domains.sort()
                
                return AvailableDomainsResponse(
                    domains=available_domains,
                    total_count=len(available_domains)
                )
                
            except Exception as e:
                self.logger.error(f"Failed to get available domains: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/domains/validate", response_model=DomainsValidationResponse)
        async def validate_domains(request: DomainsValidationRequest):
            """Validate that requested domains have available FAISS indexes."""
            try:
                if not request.domains:
                    return DomainsValidationResponse(
                        domain_status={},
                        valid_domains=[],
                        invalid_domains=[]
                    )
                
                domain_status = {}
                valid_domains = []
                invalid_domains = []
                
                for domain in request.domains:
                    try:
                        # Construct path directly to match the data structure
                        data_dir = Path(settings.DATA_DIR)
                        domain_folder = data_dir / domain
                        faiss_dir = domain_folder / "faiss"
                        index_file = faiss_dir / "index.faiss"
                        
                        is_valid = index_file.exists()
                        domain_status[domain] = is_valid
                        
                        if is_valid:
                            valid_domains.append(domain)
                        else:
                            invalid_domains.append(domain)
                            
                    except Exception as e:
                        self.logger.warning(f"Error validating domain {domain}: {e}")
                        domain_status[domain] = False
                        invalid_domains.append(domain)
                
                return DomainsValidationResponse(
                    domain_status=domain_status,
                    valid_domains=valid_domains,
                    invalid_domains=invalid_domains
                )
                
            except Exception as e:
                self.logger.error(f"Domain validation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/domains", response_model=List[DomainInfo])
        async def list_domains():
            """List all available domains with their information."""
            try:
                domains = self.storage_manager.list_domains()
                domain_list = []
                
                for domain_data in domains:
                    domain_name = domain_data['domain_name']
                    
                    # Check if embeddings exist
                    has_embeddings = domain_name in self.rag_pipeline.get_available_domains()
                    embedding_info = None
                    
                    if has_embeddings:
                        embedding_info = self.rag_pipeline.get_domain_stats(domain_name)
                    
                    domain_info = DomainInfo(
                        domain_name=domain_name,
                        total_documents=domain_data.get('total_documents', 0),
                        last_crawl=domain_data.get('last_crawl'),
                        has_embeddings=has_embeddings,
                        embedding_info=embedding_info,
                        statistics=domain_data
                    )
                    
                    domain_list.append(domain_info)
                
                return domain_list
                
            except Exception as e:
                self.logger.error(f"List domains failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/domains/{domain_name}/documents")
        async def get_domain_documents(domain_name: str, format_type: str = "json"):
            """Get all documents for a specific domain."""
            try:
                domains = self.storage_manager.list_domains()
                domain_info = next((d for d in domains if d['domain_name'] == domain_name), None)
                
                if not domain_info:
                    raise HTTPException(status_code=404, detail=f"Domain '{domain_name}' not found")
                
                if format_type not in ["json", "yaml"]:
                    raise HTTPException(status_code=400, detail="Format must be 'json' or 'yaml'")
                
                domain_folder = domain_info['domain_folder']
                documents = self.storage_manager.load_documents(domain_folder, format_type)
                
                return {
                    "domain": domain_name,
                    "format": format_type,
                    "total_documents": len(documents),
                    "documents": [doc.dict() for doc in documents]  # Return all documents
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Get documents failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/domains/{domain_name}/embeddings")
        async def delete_domain_embeddings(domain_name: str):
            """Delete embeddings (FAISS index) for a specific domain."""
            try:
                # Check if domain exists
                domains = self.storage_manager.list_domains()
                domain_info = next((d for d in domains if d['domain_name'] == domain_name), None)
                if not domain_info:
                    raise HTTPException(status_code=404, detail=f"Domain '{domain_name}' not found")
                
                # Check if embeddings exist
                if domain_name not in self.rag_pipeline.get_available_domains():
                    raise HTTPException(status_code=404, detail=f"No embeddings found for domain '{domain_name}'")
                
                # Delete the vector store
                self.rag_pipeline.remove_vector_store(domain_name)
                
                # Delete FAISS files
                domain_folder = domain_info['domain_folder']
                vector_store = VectorStore(domain_name, domain_folder)
                vector_store.delete_index()
                
                return {
                    "success": True,
                    "message": f"Embeddings deleted for domain '{domain_name}'",
                    "domain": domain_name
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to delete embeddings for {domain_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/domains/{domain_name}/reembed")
        async def reembed_domain(domain_name: str, background_tasks: BackgroundTasks):
            """Re-embed a domain (delete existing embeddings and regenerate)."""
            try:
                # Check if domain exists
                domains = self.storage_manager.list_domains()
                domain_info = next((d for d in domains if d['domain_name'] == domain_name), None)
                if not domain_info:
                    raise HTTPException(status_code=404, detail=f"Domain '{domain_name}' not found")
                
                domain_folder = domain_info['domain_folder']
                
                # Generate unique task ID
                task_id = f"reembed_{domain_name.replace('.', '-').replace('_', '-').lower()}_{int(time.time() * 1000)}"
                
                # Initialize task
                self.background_tasks[task_id] = {
                    "task_id": task_id,
                    "status": "started",
                    "message": f"Starting re-embedding for {domain_name}",
                    "progress": 0,
                    "domain": domain_name,
                    "start_time": time.time(),
                    "end_time": None,
                    "error": None
                }
                
                # Start re-embedding in background
                background_tasks.add_task(
                    self._reembed_task,
                    task_id,
                    domain_name,
                    domain_folder
                )
                
                return {
                    "success": True,
                    "message": f"Re-embedding started for domain '{domain_name}'",
                    "task_id": task_id
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to start re-embedding for {domain_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/domains/{domain_name}")
        async def delete_domain(domain_name: str, background_tasks: BackgroundTasks, delete_blob: bool = False):
            """Delete an entire domain (documents, embeddings, etc.)."""
            try:
                # Check if domain exists
                domains = self.storage_manager.list_domains()
                domain_info = next((d for d in domains if d['domain_name'] == domain_name), None)
                if not domain_info:
                    raise HTTPException(status_code=404, detail=f"Domain '{domain_name}' not found")
                
                domain_folder = domain_info['domain_folder']
                
                # Generate unique task ID
                task_id = f"delete_domain_{domain_name.replace('.', '-').replace('_', '-').lower()}_{int(time.time() * 1000)}"
                
                # Initialize task
                self.background_tasks[task_id] = {
                    "task_id": task_id,
                    "status": "started",
                    "message": f"Starting domain deletion for {domain_name}",
                    "progress": 0,
                    "domain": domain_name,
                    "start_time": time.time(),
                    "end_time": None,
                    "error": None
                }
                
                # Start deletion in background
                background_tasks.add_task(
                    self._delete_domain_task,
                    task_id,
                    domain_name,
                    domain_folder,
                    delete_blob
                )
                
                return {
                    "success": True,
                    "message": f"Domain deletion started for '{domain_name}'",
                    "task_id": task_id
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to start domain deletion for {domain_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tasks/{task_id}")
        async def get_task_status(task_id: str):
            """Get status of a background task."""
            if task_id not in self.background_tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            return self.background_tasks[task_id]
        
        @self.app.get("/tasks")
        async def get_all_tasks():
            """Get all background tasks."""
            try:
                # Convert the tasks dict to a list and add task_id to each task
                tasks_list = []
                for task_id, task_data in self.background_tasks.items():
                    task_with_id = task_data.copy()
                    task_with_id["task_id"] = task_id
                    tasks_list.append(task_with_id)
                
                # Sort by start_time (most recent first)
                tasks_list.sort(key=lambda x: x.get("start_time", 0), reverse=True)
                
                return tasks_list
                
            except Exception as e:
                self.logger.error(f"Failed to get all tasks: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/status")
        async def get_system_status():
            """Get system status and statistics."""
            try:
                pipeline_info = self.rag_pipeline.get_pipeline_info()
                domains = self.storage_manager.list_domains()
                
                return StatusResponse(
                    success=True,
                    message="System status retrieved",
                    data={
                        "pipeline_info": pipeline_info,
                        "total_domains": len(domains),
                        "active_tasks": len(self.background_tasks),
                        "storage_domains": domains
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Status check failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            """Global exception handler."""
            self.logger.error(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="Internal server error",
                    details=str(exc)
                ).dict()
            )
    
    async def _crawl_task(self, task_id: str, url: str, base_domain: Optional[str], 
                         max_depth: Optional[int], domain_folder: str):
        """Background crawling task."""
        try:
            self.logger.info(f"Starting crawl task {task_id} for {url}")
            
            # Update task status (domain stays static like start_time)
            self.background_tasks[task_id].update({
                "task_id": task_id,  # Ensure task_id is always present
                "status": "crawling",
                "message": f"Starting to crawl documentation from {url}",  # Always provide a message
                "progress": 10
            })
            
            # Create progress callback
            async def progress_callback(progress_data):
                successful = progress_data['successful_pages']
                total = progress_data['total_processed']
                remaining = progress_data['queue_remaining']
                
                # Calculate progress percentage (estimate based on discovered pages so far)
                # We'll use a heuristic: 90% for crawling, 10% for initial setup
                estimated_total = max(total + remaining, 1)  # Avoid division by zero
                crawl_progress = min(90, int((total / estimated_total) * 90))
                final_progress = 10 + crawl_progress  # 10% for initial setup + crawl progress
                
                # Update task with real-time progress (domain stays static)
                self.background_tasks[task_id].update({
                    "task_id": task_id,  # Ensure task_id is always present
                    "message": f"Processing pages... ({successful} successful, {remaining} in queue)",
                    "progress": final_progress
                })
            
            # Perform crawling with progress updates
            async with WebCrawler() as crawler:
                session = await crawler.crawl_domain(url, base_domain, max_depth, progress_callback)
                
                # Update with crawling results (domain stays static)
                self.background_tasks[task_id].update({
                    "task_id": task_id,  # Ensure task_id is always present
                    "message": f"Crawling complete. Found {session.successful_pages} pages",
                    "progress": 95
                })
                
                # Save results
                self.storage_manager.save_crawl_session(session, domain_folder)
                documents_paths = self.storage_manager.save_documents(
                    crawler.crawled_documents, 
                    domain_folder
                )
                
                # Final completion status (domain stays static)
                self.background_tasks[task_id].update({
                    "task_id": task_id,  # Ensure task_id is always present
                    "status": "completed",
                    "message": f"Successfully crawled {session.successful_pages} pages",
                    "progress": 100,
                    "end_time": time.time(),
                    "results": {
                        "total_pages": session.total_pages,
                        "successful_pages": session.successful_pages,
                        "failed_pages": session.failed_pages,
                        "documents_saved": len(crawler.crawled_documents),
                        "files_created": documents_paths
                    }
                })
                
                self.logger.info(f"Crawl task {task_id} completed successfully")
                
        except Exception as e:
            self.logger.error(f"Crawl task {task_id} failed: {e}")
            self.background_tasks[task_id].update({
                "task_id": task_id,
                "status": "failed",
                "message": f"Crawling failed: {str(e)}",
                "progress": 0,
                "end_time": time.time(),
                "error": str(e)
            })
    
    async def _embed_task(self, task_id: str, domain: str, domain_folder: str):
        """Background embedding task."""
        try:
            self.logger.info(f"Starting embed task {task_id} for domain {domain}")
            
            # Update task status (domain stays static like start_time)
            self.background_tasks[task_id].update({
                "task_id": task_id,
                "status": "processing",
                "message": f"Loading documents for {domain}",
                "progress": 5
            })
            
            # Load documents
            documents = self.storage_manager.load_documents(domain_folder, "json")
            if not documents:
                raise Exception("No documents found for embedding")
            
            # Update progress (domain stays static)
            self.background_tasks[task_id].update({
                "task_id": task_id,
                "message": f"Loaded {len(documents)} documents. Initializing embedding service...",
                "progress": 10
            })
            
            # Initialize embedding service
            embedding_service = EmbeddingService()
            await embedding_service.initialize()
            
            # Create progress callback for embedding
            async def embedding_progress_callback(progress_data):
                stage = progress_data['stage']
                
                if stage == 'chunking':
                    docs_processed = progress_data['documents_processed']
                    total_docs = progress_data['total_documents']
                    chunks_created = progress_data['chunks_created']
                    
                    # Chunking gets 10-30% of progress
                    chunk_progress = int(10 + (docs_processed / total_docs) * 20)
                    
                    self.background_tasks[task_id].update({
                        "task_id": task_id,
                        "message": f"Chunking documents... ({docs_processed}/{total_docs} docs, {chunks_created} chunks)",
                        "progress": chunk_progress
                    })
                    
                elif stage == 'embedding':
                    chunks_processed = progress_data['chunks_processed']
                    total_chunks = progress_data['total_chunks']
                    
                    # Embedding gets 30-80% of progress
                    embed_progress = int(30 + (chunks_processed / total_chunks) * 50)
                    
                    self.background_tasks[task_id].update({
                        "task_id": task_id,
                        "message": f"Generating embeddings... ({chunks_processed}/{total_chunks} chunks)",
                        "progress": embed_progress
                    })
                    
                elif stage == 'completed':
                    self.background_tasks[task_id].update({
                        "task_id": task_id,
                        "message": f"Embeddings generated. Creating vector index...",
                        "progress": 85
                    })
            
            # Generate embeddings with progress tracking
            chunks = await embedding_service.embed_documents(documents, embedding_progress_callback)
            
            # Create vector store
            vector_store = VectorStore(domain, domain_folder)
            model_info = embedding_service.get_model_info()
            model_name = model_info.get('model', model_info.get('type', 'unknown'))
            embedding_info = vector_store.create_index(
                chunks, 
                model_name
            )
            
            # Add to RAG pipeline
            self.rag_pipeline.add_vector_store(domain, vector_store)
            
            # Final completion status (domain stays static)
            self.background_tasks[task_id].update({
                "task_id": task_id,
                "status": "completed",
                "message": f"Successfully generated embeddings for {len(chunks)} chunks",
                "progress": 100,
                "end_time": time.time(),
                "results": {
                    "total_chunks": len(chunks),
                    "embedding_dimension": embedding_info.vector_dimension,
                    "model_name": embedding_info.model_name,
                    "index_file": embedding_info.index_file_path
                }
            })
            
            self.logger.info(f"Embed task {task_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Embed task {task_id} failed: {e}")
            self.background_tasks[task_id].update({
                "task_id": task_id,
                "status": "failed",
                "message": f"Embedding generation failed: {str(e)}",
                "progress": 0,
                "end_time": time.time(),
                "error": str(e)
            })
    
    async def _reembed_task(self, task_id: str, domain: str, domain_folder: str):
        """Background re-embedding task (delete existing and regenerate)."""
        try:
            self.logger.info(f"Starting re-embed task {task_id} for domain {domain}")
            
            # Update task status
            self.background_tasks[task_id].update({
                "task_id": task_id,
                "status": "processing",
                "message": f"Deleting existing embeddings for {domain}",
                "progress": 10
            })
            
            # Delete existing embeddings
            if domain in self.rag_pipeline.get_available_domains():
                self.rag_pipeline.remove_vector_store(domain)
                vector_store = VectorStore(domain, domain_folder)
                vector_store.delete_index()
            
            # Update progress
            self.background_tasks[task_id].update({
                "task_id": task_id,
                "message": f"Regenerating embeddings for {domain}",
                "progress": 20
            })
            
            # Load documents
            documents = self.storage_manager.load_documents(domain_folder, "json")
            if not documents:
                raise Exception("No documents found for re-embedding")
            
            # Initialize embedding service
            embedding_service = EmbeddingService()
            await embedding_service.initialize()
            
            # Generate embeddings
            async def embedding_progress_callback(progress_data):
                stage = progress_data.get('stage', 'processing')
                if stage == 'chunking':
                    self.background_tasks[task_id].update({
                        "task_id": task_id,
                        "message": f"Chunking documents...",
                        "progress": 40
                    })
                elif stage == 'embedding':
                    chunks_processed = progress_data.get('chunks_processed', 0)
                    total_chunks = progress_data.get('total_chunks', 1)
                    embed_progress = int(50 + (chunks_processed / total_chunks) * 40)
                    self.background_tasks[task_id].update({
                        "task_id": task_id,
                        "message": f"Generating embeddings... ({chunks_processed}/{total_chunks} chunks)",
                        "progress": embed_progress
                    })
            
            chunks = await embedding_service.embed_documents(documents, embedding_progress_callback)
            
            # Create vector store
            vector_store = VectorStore(domain, domain_folder)
            model_info = embedding_service.get_model_info()
            model_name = model_info.get('model', model_info.get('type', 'unknown'))
            embedding_info = vector_store.create_index(
                chunks, 
                model_name
            )
            
            # Add to RAG pipeline
            self.rag_pipeline.add_vector_store(domain, vector_store)
            
            # Completion
            self.background_tasks[task_id].update({
                "task_id": task_id,
                "status": "completed",
                "message": f"Successfully re-embedded {len(chunks)} chunks for {domain}",
                "progress": 100,
                "end_time": time.time(),
                "results": {
                    "total_chunks": len(chunks),
                    "embedding_dimension": embedding_info.vector_dimension,
                    "model_name": embedding_info.model_name
                }
            })
            
            self.logger.info(f"Re-embed task {task_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Re-embed task {task_id} failed: {e}")
            self.background_tasks[task_id].update({
                "task_id": task_id,
                "status": "failed",
                "message": f"Re-embedding failed: {str(e)}",
                "progress": 0,
                "end_time": time.time(),
                "error": str(e)
            })

    async def _delete_domain_task(self, task_id: str, domain: str, domain_folder: str, delete_blob: bool):
        """Background domain deletion task."""
        try:
            self.logger.info(f"Starting domain deletion task {task_id} for domain {domain}")
            
            # Update task status
            self.background_tasks[task_id].update({
                "task_id": task_id,
                "status": "processing",
                "message": f"Deleting domain data for {domain}",
                "progress": 50
            })
            
            # Remove from RAG pipeline if exists
            if domain in self.rag_pipeline.get_available_domains():
                self.rag_pipeline.remove_vector_store(domain)
            
            # Delete domain folder
            import shutil
            if Path(domain_folder).exists():
                shutil.rmtree(domain_folder)
            
            # TODO: If delete_blob is True, also delete from Azure Blob Storage (if configured)
            if delete_blob:
                # Implement blob deletion logic here if needed
                pass
            
            # Completion
            self.background_tasks[task_id].update({
                "task_id": task_id,
                "status": "completed",
                "message": f"Successfully deleted domain '{domain}'",
                "progress": 100,
                "end_time": time.time()
            })
            
            self.logger.info(f"Domain deletion task {task_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Domain deletion task {task_id} failed: {e}")
            self.background_tasks[task_id].update({
                "task_id": task_id,
                "status": "failed",
                "message": f"Domain deletion failed: {str(e)}",
                "progress": 0,
                "end_time": time.time(),
                "error": str(e)
            })
    
    async def _load_existing_domains(self):
        """Load existing domains and their vector stores."""
        try:
            domains = self.storage_manager.list_domains()
            
            for domain_data in domains:
                domain_name = domain_data['domain_name']
                domain_folder = domain_data['domain_folder']
                
                # Try to load vector store
                if self.rag_pipeline.load_vector_store(domain_name, domain_folder):
                    self.logger.info(f"Loaded existing domain: {domain_name}")
                
        except Exception as e:
            self.logger.warning(f"Failed to load existing domains: {e}")
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app


# Create the API instance
api = DocumentCrawlerAPI()
app = api.get_app()