"""Multi-domain vector store for cross-domain similarity search."""

import asyncio
import hashlib
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

try:
    from config.settings import settings
    from utils.logger import default_logger
    from storage.storage_manager import StorageManager
    from .vector_store import VectorStore
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from config.settings import settings
    from utils.logger import default_logger
    from storage.storage_manager import StorageManager
    from embeddings.vector_store import VectorStore


@dataclass
class DomainSearchResult:
    """Result from a single domain search."""
    domain: str
    chunk_id: str
    source_url: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any]
    raw_score: float
    normalized_score: float
    faiss_index: int


class MultiDomainVectorStore:
    """Orchestrates searches across multiple domain vector stores."""
    
    def __init__(self, max_concurrent_domains: int = 5):
        """
        Initialize multi-domain vector store.
        
        Args:
            max_concurrent_domains: Maximum number of domains to search concurrently
        """
        self.logger = default_logger
        self.max_concurrent_domains = max_concurrent_domains
        self._domain_stores: Dict[str, VectorStore] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_domains)
    
    async def search_domains(
        self, 
        domains: List[str], 
        query_embedding: List[float], 
        top_k: int = 5,
        per_domain_k: Optional[int] = None
    ) -> List[DomainSearchResult]:
        """
        Search across multiple domains and return merged results.
        
        Args:
            domains: List of domain names to search
            query_embedding: Query embedding vector
            top_k: Total number of results to return
            per_domain_k: Number of candidates to retrieve per domain (defaults to top_k)
            
        Returns:
            List of DomainSearchResult objects, sorted by normalized score (desc)
        """
        if not domains:
            return []
        
        if per_domain_k is None:
            # Get more candidates per domain to improve final ranking
            per_domain_k = min(top_k * 2, 20)
        
        self.logger.info(f"Searching {len(domains)} domains with per_domain_k={per_domain_k}, final_top_k={top_k}")
        
        # Search all domains concurrently
        search_tasks = [
            self._search_single_domain(domain, query_embedding, per_domain_k)
            for domain in domains
        ]
        
        domain_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Collect and merge results
        all_results = []
        successful_domains = []
        
        for domain, result in zip(domains, domain_results):
            if isinstance(result, Exception):
                self.logger.warning(f"Search failed for domain {domain}: {result}")
                continue
            
            if result:  # Non-empty results
                all_results.extend(result)
                successful_domains.append(domain)
        
        if not all_results:
            self.logger.warning("No results found across any domains")
            return []
        
        # Sort by normalized score (descending) and deduplicate
        merged_results = self._merge_and_deduplicate(all_results, top_k)
        
        self.logger.info(
            f"Found {len(merged_results)} results from {len(successful_domains)} domains: {successful_domains}"
        )
        
        return merged_results
    
    async def _search_single_domain(
        self, 
        domain: str, 
        query_embedding: List[float], 
        top_k: int
    ) -> List[DomainSearchResult]:
        """Search a single domain with concurrency control."""
        async with self._semaphore:
            try:
                # Load or get cached domain store
                store = await self._get_domain_store(domain)
                if not store:
                    return []
                
                # Perform search
                raw_results = store.search(query_embedding, top_k)
                
                # Convert to DomainSearchResult with normalized scores
                domain_results = []
                for result in raw_results:
                    normalized_score = self._normalize_score(
                        result['similarity_score'], 
                        store.index
                    )
                    
                    domain_result = DomainSearchResult(
                        domain=domain,
                        chunk_id=result['chunk_id'],
                        source_url=result['source_url'],
                        content=result['content'],
                        chunk_index=result['chunk_index'],
                        metadata=result['metadata'],
                        raw_score=result['similarity_score'],
                        normalized_score=normalized_score,
                        faiss_index=result['faiss_index']
                    )
                    domain_results.append(domain_result)
                
                return domain_results
                
            except Exception as e:
                self.logger.error(f"Error searching domain {domain}: {e}")
                return []
    
    async def _get_domain_store(self, domain: str) -> Optional[VectorStore]:
        """Get or load domain vector store."""
        if domain in self._domain_stores:
            return self._domain_stores[domain]
        
        try:
            # Construct domain folder path directly
            data_dir = Path(settings.DATA_DIR)
            domain_folder = data_dir / domain
            
            # Create and load vector store
            store = VectorStore(domain, str(domain_folder))
            
            # Load index in thread pool to avoid blocking
            loaded = await asyncio.get_event_loop().run_in_executor(
                None, store.load_index
            )
            
            if not loaded:
                self.logger.warning(f"No FAISS index found for domain: {domain}")
                return None
            
            # Cache the loaded store
            self._domain_stores[domain] = store
            self.logger.debug(f"Loaded and cached vector store for domain: {domain}")
            
            return store
            
        except Exception as e:
            self.logger.error(f"Failed to load vector store for domain {domain}: {e}")
            return None
    
    def _normalize_score(self, raw_score: float, faiss_index) -> float:
        """
        Normalize FAISS scores to consistent similarity score (higher = better).
        
        Args:
            raw_score: Raw score from FAISS search
            faiss_index: FAISS index object to determine score type
            
        Returns:
            Normalized similarity score (0.0 to 1.0, higher = better)
        """
        index_type = type(faiss_index).__name__
        
        if "IP" in index_type:  # Inner Product
            # Inner product scores are already similarity (higher = better)
            # Clamp to [0, 1] range
            return max(0.0, min(1.0, raw_score))
        
        elif "L2" in index_type:  # L2 Distance
            # L2 distances: lower = better, convert to similarity
            # Use exponential decay: similarity = exp(-distance)
            similarity = math.exp(-abs(raw_score))
            return max(0.0, min(1.0, similarity))
        
        else:
            # Unknown index type, assume similarity score
            self.logger.warning(f"Unknown FAISS index type: {index_type}, treating as similarity")
            return max(0.0, min(1.0, abs(raw_score)))
    
    def _merge_and_deduplicate(
        self, 
        results: List[DomainSearchResult], 
        top_k: int
    ) -> List[DomainSearchResult]:
        """
        Merge results and remove duplicates based on content similarity.
        
        Args:
            results: List of results from all domains
            top_k: Number of final results to return
            
        Returns:
            Deduplicated and ranked results
        """
        if not results:
            return []
        
        # Sort by normalized score (descending)
        sorted_results = sorted(results, key=lambda x: x.normalized_score, reverse=True)
        
        # Deduplicate based on content hash
        seen_hashes: Set[str] = set()
        deduplicated = []
        
        for result in sorted_results:
            # Create content hash for deduplication
            content_hash = hashlib.md5(
                result.content.strip().lower().encode('utf-8')
            ).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                deduplicated.append(result)
                
                if len(deduplicated) >= top_k:
                    break
        
        return deduplicated
    
    def get_loaded_domains(self) -> List[str]:
        """Get list of currently loaded domains."""
        return list(self._domain_stores.keys())
    
    def clear_cache(self):
        """Clear all cached domain stores."""
        self._domain_stores.clear()
        self.logger.info("Cleared domain store cache")
    
    async def validate_domains(self, domains: List[str]) -> Dict[str, bool]:
        """
        Validate that domains have available FAISS indexes.
        
        Args:
            domains: List of domain names to validate
            
        Returns:
            Dict mapping domain -> availability status
        """
        validation_tasks = [
            self._check_domain_availability(domain) 
            for domain in domains
        ]
        
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        domain_status = {}
        for domain, result in zip(domains, results):
            if isinstance(result, Exception):
                domain_status[domain] = False
            else:
                domain_status[domain] = result
        
        return domain_status
    
    async def _check_domain_availability(self, domain: str) -> bool:
        """Check if a domain has an available FAISS index."""
        try:
            data_dir = Path(settings.DATA_DIR)
            domain_folder = data_dir / domain
            faiss_dir = domain_folder / "faiss"
            index_file = faiss_dir / "index.faiss"
            
            return index_file.exists()
            
        except Exception:
            return False