"""Gemini LLM adapter."""

from typing import Dict, Any
from .adapter_base import LLMAdapter


class GeminiLLMAdapter(LLMAdapter):
    """Gemini LLM adapter."""
    
    def __init__(self):
        self._initialized = False
        self._model = None
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Gemini."""
        if not self._model:
            raise Exception("Gemini LLM not initialized")
        
        try:
            response = self._model.generate_content(prompt, **kwargs)
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            raise Exception(f"Gemini generation failed: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get adapter information."""
        from config.settings import settings
        return {
            "type": "gemini",
            "model": settings.GEMINI_LLM_MODEL,
            "initialized": self._initialized
        }
    
    async def initialize(self) -> bool:
        """Initialize Gemini LLM."""
        try:
            from config.settings import settings
            import google.generativeai as genai
            
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._model = genai.GenerativeModel(settings.GEMINI_LLM_MODEL)
            
            # Test
            test_response = self._model.generate_content("Hello")
            if test_response:
                self._initialized = True
                return True
            
            return False
            
        except Exception as e:
            print(f"Failed to initialize Gemini LLM: {e}")
            return False