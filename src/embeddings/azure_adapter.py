"""Azure OpenAI embedding adapter."""

import asyncio
from typing import List, Dict, Any
from .adapter_base import EmbeddingAdapter

try:
    from langchain_openai import AzureOpenAIEmbeddings
except ImportError:
    AzureOpenAIEmbeddings = None


class AzureEmbeddingAdapter(EmbeddingAdapter):
    """Azure OpenAI embedding adapter."""
    
    def __init__(self, embeddings):
        self._embeddings = embeddings
        self._initialized = True
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        from config.settings import settings
        
        batch_size = settings.EMBED_BATCH_SIZE_AZURE
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self._embeddings.aembed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        return all_embeddings
    
    async def embed_query(self, text: str) -> List[float]:
        return await self._embeddings.aembed_query(text)
    
    def get_dimension(self) -> int:
        return 1536
    
    def get_info(self) -> Dict[str, Any]:
        return {"type": "azure", "dimension": 1536, "initialized": True}
    
    async def initialize(self) -> bool:
        return True