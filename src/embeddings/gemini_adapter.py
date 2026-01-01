"""Gemini embedding adapter."""

import asyncio
from typing import List, Dict, Any
from .adapter_base import EmbeddingAdapter


class GeminiEmbeddingAdapter(EmbeddingAdapter):
    """Gemini embedding adapter."""
    
    def __init__(self):
        self._initialized = True
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        from config.settings import settings
        
        import google.generativeai as genai
        
        embeddings = []
        batch_size = settings.EMBED_BATCH_SIZE_GEMINI
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            for text in batch:
                result = genai.embed_content(
                    model=settings.GEMINI_EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                if 'embedding' in result:
                    batch_embeddings.append(result['embedding'])
            embeddings.extend(batch_embeddings)
            if i + batch_size < len(texts):
                await asyncio.sleep(1)
        return embeddings
    
    async def embed_query(self, text: str) -> List[float]:
        from config.settings import settings
        
        import google.generativeai as genai
        
        result = genai.embed_content(
            model=settings.GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        return result.get('embedding', [])
    
    def get_dimension(self) -> int:
        return 768
    
    def get_info(self) -> Dict[str, Any]:
        return {"type": "gemini", "dimension": 768, "initialized": True}
    
    async def initialize(self) -> bool:
        return True