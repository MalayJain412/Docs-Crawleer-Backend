"""Sentence transformer embedding adapter."""

import asyncio
from typing import List, Dict, Any
from .adapter_base import EmbeddingAdapter


class SentenceTransformerAdapter(EmbeddingAdapter):
    """Sentence transformer embedding adapter."""
    
    def __init__(self, model):
        self._model = model
        self._initialized = True
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self._model.encode, texts)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
    
    async def embed_query(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._model.encode, text)
        return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
    
    def get_dimension(self) -> int:
        return 384
    
    def get_info(self) -> Dict[str, Any]:
        return {"type": "sentence-transformer", "dimension": 384, "initialized": True}
    
    async def initialize(self) -> bool:
        return True