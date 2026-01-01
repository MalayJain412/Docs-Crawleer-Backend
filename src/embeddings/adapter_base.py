"""Base adapter interface for embedding and LLM services."""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict


class EmbeddingAdapter(ABC):
    """Abstract base class for embedding adapters."""
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        pass
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get adapter information."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the adapter."""
        pass


class LLMAdapter(ABC):
    """Abstract base class for LLM adapters."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get adapter information."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the adapter."""
        pass