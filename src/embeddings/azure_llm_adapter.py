"""Azure OpenAI LLM adapter."""

from typing import Dict, Any
from .adapter_base import LLMAdapter

try:
    from langchain_openai import AzureChatOpenAI
except ImportError:
    AzureChatOpenAI = None


class AzureLLMAdapter(LLMAdapter):
    """Azure OpenAI LLM adapter."""
    
    def __init__(self, llm):
        self._llm = llm
        self._initialized = True
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Azure OpenAI."""
        try:
            # Use ainvoke for async generation
            response = await self._llm.ainvoke(prompt, **kwargs)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            raise Exception(f"Azure OpenAI generation failed: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get adapter information."""
        return {
            "type": "azure",
            "initialized": self._initialized
        }
    
    async def initialize(self) -> bool:
        """Initialize Azure LLM (already done in constructor)."""
        return True