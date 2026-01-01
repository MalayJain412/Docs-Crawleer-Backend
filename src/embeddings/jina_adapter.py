"""Jina embeddings adapter."""

import asyncio
from typing import List, Any, Dict, Optional
from .adapter_base import EmbeddingAdapter

try:
    from langchain_community.embeddings import JinaEmbeddings
except ImportError:
    JinaEmbeddings = None

try:
    from config.settings import settings
    from utils.logger import default_logger
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config.settings import settings
    from utils.logger import default_logger


class JinaEmbeddingAdapter(EmbeddingAdapter):
    """Jina embeddings adapter implementation."""
    
    def __init__(self):
        self.logger = default_logger
        self._embeddings: Optional[JinaEmbeddings] = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize Jina embeddings."""
        try:
            if not JinaEmbeddings:
                self.logger.warning("langchain-community not installed, skipping Jina")
                return False
            
            if not settings.JINA_API_KEY:
                self.logger.warning("JINA_API_KEY not set, skipping Jina")
                return False
            
            self._embeddings = JinaEmbeddings(
                jina_api_key=settings.JINA_API_KEY,
                model_name="jina-embeddings-v2-base-en"
            )
            
            # Test the embeddings
            test_result = await self._embeddings.aembed_query("Test sentence")
            if test_result and len(test_result) > 0:
                self._initialized = True
                self.logger.info("Jina embeddings initialized successfully")
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Jina embeddings: {e}")
            return False
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        if not self._initialized or not self._embeddings:
            raise RuntimeError("Jina embeddings not initialized")
        
        try:
            # Process in batches
            batch_size = settings.EMBED_BATCH_SIZE_JINA
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = await self._embeddings.aembed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Small delay between batches
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"Jina embedding failed: {e}")
            raise
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        if not self._initialized or not self._embeddings:
            raise RuntimeError("Jina embeddings not initialized")
        
        try:
            return await self._embeddings.aembed_query(text)
        except Exception as e:
            self.logger.error(f"Jina query embedding failed: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return 768  # Jina v2 base dimension
    
    def get_info(self) -> Dict[str, Any]:
        """Get adapter info."""
        return {
            "type": "jina",
            "model": "jina-embeddings-v2-base-en",
            "dimension": self.get_dimension(),
            "initialized": self._initialized,
            "batch_size": settings.EMBED_BATCH_SIZE_JINA
        }