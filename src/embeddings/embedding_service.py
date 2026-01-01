"""Embedding service with Jina primary and fallback options."""

import asyncio
from typing import List, Optional, Dict, Any, Tuple
import time
import re

from .adapter_base import EmbeddingAdapter
from .jina_adapter import JinaEmbeddingAdapter
from .azure_adapter import AzureEmbeddingAdapter
from .gemini_adapter import GeminiEmbeddingAdapter
from .sentence_transformer_adapter import SentenceTransformerAdapter

try:
    from langchain_openai import AzureOpenAIEmbeddings
except ImportError:
    AzureOpenAIEmbeddings = None

try:
    from config.settings import settings
    from utils.logger import default_logger
    from storage.schemas import DocumentContent, ContentChunk
    from utils.url_utils import URLUtils  # Import URLUtils
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from config.settings import settings
    from utils.logger import default_logger
    from storage.schemas import DocumentContent, ContentChunk
    from utils.url_utils import URLUtils  # Import URLUtils


class EmbeddingService:
    """Service for generating embeddings with multiple fallback options."""
    
    def __init__(self):
        """Initialize the embedding service."""
        self.logger = default_logger
        self._current_adapter: Optional[EmbeddingAdapter] = None
        self._adapters: Dict[str, EmbeddingAdapter] = {}
        
    async def initialize(self):
        """Initialize embedding adapters."""
        try:
            # Try Jina first (primary)
            jina_adapter = JinaEmbeddingAdapter()
            if await jina_adapter.initialize():
                self._adapters["jina"] = jina_adapter
                self._current_adapter = jina_adapter
                self.logger.info("Initialized Jina embeddings (primary)")
                return
            
            # Fallback based on MODEL_USE
            if settings.MODEL_USE == "azure":
                # Try Azure OpenAI
                azure_adapter = await self._init_azure_adapter()
                if azure_adapter:
                    self._adapters["azure"] = azure_adapter
                    self._current_adapter = azure_adapter
                    self.logger.info("Initialized Azure OpenAI embeddings (fallback)")
                    return
                
                # Fallback to Gemini
                gemini_adapter = await self._init_gemini_adapter()
                if gemini_adapter:
                    self._adapters["gemini"] = gemini_adapter
                    self._current_adapter = gemini_adapter
                    self.logger.info("Initialized Gemini embeddings (fallback)")
                    return
            
            elif settings.MODEL_USE == "gemini":
                # Try Gemini first
                gemini_adapter = await self._init_gemini_adapter()
                if gemini_adapter:
                    self._adapters["gemini"] = gemini_adapter
                    self._current_adapter = gemini_adapter
                    self.logger.info("Initialized Gemini embeddings (fallback)")
                    return
            
            # Final fallback: sentence-transformers
            st_adapter = await self._init_sentence_transformer_adapter()
            if st_adapter:
                self._adapters["sentence-transformer"] = st_adapter
                self._current_adapter = st_adapter
                self.logger.info("Initialized sentence-transformers embeddings (final fallback)")
                return
            
            raise Exception("No embedding adapter could be initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding adapters: {e}")
            raise
    
    async def _init_gemini(self) -> Optional[Any]:
        """Initialize Gemini embedding model."""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=settings.GEMINI_API_KEY)
            # Try to test the embedding model first using configured model name
            try:
                test_result = genai.embed_content(
                    model=settings.GEMINI_EMBEDDING_MODEL,
                    content="Test",
                    task_type="retrieval_document"
                )

                if isinstance(test_result, dict) and 'embedding' in test_result:
                    return genai
            except Exception:
                # If embedding test fails, try a lightweight generation test using the LLM model
                try:
                    model = genai.GenerativeModel(settings.GEMINI_LLM_MODEL)
                    _ = model.generate_content("Test")
                    return genai
                except Exception as e:
                    self.logger.warning(f"Gemini generation test failed: {e}")
                    return None
            
        except ImportError:
            self.logger.warning("google-generativeai not installed, skipping Gemini")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to initialize Gemini: {e}")
            return None
    
    async def _init_azure_adapter(self) -> Optional[EmbeddingAdapter]:
        """Initialize Azure OpenAI embedding adapter."""
        try:
            if not AzureOpenAIEmbeddings:
                self.logger.warning("langchain-openai not installed, skipping Azure OpenAI")
                return None
            
            if not all([
                settings.AZURE_OPENAI_ENDPOINT,
                settings.AZURE_OPENAI_API_KEY,
                settings.AZURE_API_VERSION
            ]):
                self.logger.warning("Azure OpenAI settings incomplete, skipping Azure OpenAI")
                return None
            
            embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_API_VERSION,
                azure_deployment=settings.AZURE_EMBEDDING_DEPLOYMENT,
                model="text-embedding-ada-002"
            )
            
            # Test
            test_result = await embeddings.aembed_query("Test sentence")
            if test_result and len(test_result) > 0:
                return AzureEmbeddingAdapter(embeddings)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Azure OpenAI embeddings: {e}")
            return None
    
    async def _init_gemini_adapter(self) -> Optional[EmbeddingAdapter]:
        """Initialize Gemini embedding adapter."""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # Test
            test_result = genai.embed_content(
                model=settings.GEMINI_EMBEDDING_MODEL,
                content="Test",
                task_type="retrieval_document"
            )
            if isinstance(test_result, dict) and 'embedding' in test_result:
                return GeminiEmbeddingAdapter()
            
            return None
            
        except ImportError:
            self.logger.warning("google-generativeai not installed, skipping Gemini")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to initialize Gemini: {e}")
            return None
    
    async def _init_sentence_transformer_adapter(self) -> Optional[EmbeddingAdapter]:
        """Initialize sentence-transformer adapter."""
        try:
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            test_embedding = model.encode("Test sentence")
            if test_embedding is not None and len(test_embedding) > 0:
                return SentenceTransformerAdapter(model)
            
            return None
            
        except ImportError:
            self.logger.warning("sentence-transformers not installed")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to initialize sentence-transformers: {e}")
            return None
        """Initialize sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use a lightweight, efficient model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Test the model
            test_embedding = model.encode("Test sentence")
            if test_embedding is not None and len(test_embedding) > 0:
                return model
            
            return None
            
        except ImportError:
            self.logger.warning("sentence-transformers not installed")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to initialize sentence-transformers: {e}")
            return None
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using current adapter.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        if not self._current_adapter:
            raise Exception("No embedding adapter initialized")
        
        try:
            return await self._current_adapter.embed_texts(texts)
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings with current adapter: {e}")
            # Try fallback adapters
            for adapter_name, adapter in self._adapters.items():
                if adapter != self._current_adapter:
                    try:
                        self.logger.info(f"Trying fallback adapter: {adapter_name}")
                        self._current_adapter = adapter
                        return await adapter.embed_texts(texts)
                    except Exception as fallback_e:
                        self.logger.warning(f"Fallback adapter {adapter_name} failed: {fallback_e}")
                        continue
            
            raise Exception("All embedding adapters failed")
    
    async def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        if not self._current_adapter:
            raise Exception("No embedding adapter initialized")
        
        try:
            return await self._current_adapter.embed_query(text)
        except Exception as e:
            self.logger.error(f"Failed to embed query with current adapter: {e}")
            # Try fallback adapters
            for adapter_name, adapter in self._adapters.items():
                if adapter != self._current_adapter:
                    try:
                        self.logger.info(f"Trying fallback adapter for query: {adapter_name}")
                        self._current_adapter = adapter
                        return await adapter.embed_query(text)
                    except Exception as fallback_e:
                        self.logger.warning(f"Fallback adapter {adapter_name} failed for query: {fallback_e}")
                        continue
            
            raise Exception("All embedding adapters failed for query")
    
    def chunk_document(self, document: DocumentContent) -> List[ContentChunk]:
        """
        Split document into chunks for embedding.
        
        Args:
            document: DocumentContent to chunk
            
        Returns:
            List of ContentChunk objects
        """
        if not document.content:
            return []
        
        chunks = []
        content = document.content
        
        # Simple chunking strategy: split by paragraphs and combine to target size
        paragraphs = self._split_into_paragraphs(content)
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, create a chunk
            if (len(current_chunk) + len(paragraph) > settings.CHUNK_SIZE and 
                current_chunk.strip()):
                
                chunk = ContentChunk(
                    chunk_id=f"{URLUtils.extract_domain_name(document.url)}_{chunk_index}",
                    source_url=document.url,
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    metadata={
                        'title': document.title,
                        'source_domain': URLUtils.extract_domain_name(document.url),
                        'chunk_length': len(current_chunk.strip()),
                        'original_doc_length': document.content_length
                    }
                )
                
                chunks.append(chunk)
                chunk_index += 1
                current_chunk = paragraph
                
                # Prevent too many chunks per document
                if len(chunks) >= settings.MAX_CHUNKS_PER_DOC:
                    break
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add remaining content as final chunk
        if current_chunk.strip() and len(chunks) < settings.MAX_CHUNKS_PER_DOC:
            chunk = ContentChunk(
                chunk_id=f"{URLUtils.extract_domain_name(document.url)}_{chunk_index}",
                source_url=document.url,
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                metadata={
                    'title': document.title,
                    'source_domain': URLUtils.extract_domain_name(document.url),
                    'chunk_length': len(current_chunk.strip()),
                    'original_doc_length': document.content_length
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs, handling various separators."""
        # Split on multiple newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Further split long paragraphs on sentence boundaries
        result = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph is too long, split on sentences
            if len(para) > settings.CHUNK_SIZE:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_group = ""
                
                for sentence in sentences:
                    if len(current_group + sentence) <= settings.CHUNK_SIZE:
                        current_group += " " + sentence if current_group else sentence
                    else:
                        if current_group:
                            result.append(current_group.strip())
                        current_group = sentence
                
                if current_group:
                    result.append(current_group.strip())
            else:
                result.append(para)
        
        return result
    
    async def embed_documents(self, documents: List[DocumentContent], progress_callback=None) -> List[ContentChunk]:
        """
        Create embeddings for a list of documents.
        
        Args:
            documents: List of DocumentContent objects
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of ContentChunk objects with embeddings
        """
        if not documents:
            return []
        
        self.logger.info(f"Processing {len(documents)} documents for embedding")
        
        # Report initial progress
        if progress_callback:
            await progress_callback({
                'stage': 'chunking',
                'documents_processed': 0,
                'total_documents': len(documents),
                'chunks_created': 0
            })
        
        # Chunk all documents with progress tracking
        all_chunks = []
        for i, doc in enumerate(documents):
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            
            # Report chunking progress
            if progress_callback:
                await progress_callback({
                    'stage': 'chunking',
                    'documents_processed': i + 1,
                    'total_documents': len(documents),
                    'chunks_created': len(all_chunks)
                })
        
        self.logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        if not all_chunks:
            return []
        
        # Extract text content for embedding
        chunk_texts = [chunk.content for chunk in all_chunks]
        
        # Report embedding start
        if progress_callback:
            await progress_callback({
                'stage': 'embedding',
                'chunks_processed': 0,
                'total_chunks': len(all_chunks)
            })
        
        # Generate embeddings with batch processing for progress tracking
        try:
            # Process in batches to provide better progress updates
            batch_size = 50  # Process 50 chunks at a time
            embeddings = []
            
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i:i + batch_size]
                batch_embeddings = await self.generate_embeddings(batch_texts)
                embeddings.extend(batch_embeddings)
                
                # Report embedding progress
                if progress_callback:
                    await progress_callback({
                        'stage': 'embedding',
                        'chunks_processed': len(embeddings),
                        'total_chunks': len(all_chunks)
                    })
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk.embedding = embedding
            
            # Report completion
            if progress_callback:
                await progress_callback({
                    'stage': 'completed',
                    'chunks_processed': len(all_chunks),
                    'total_chunks': len(all_chunks)
                })
            
            self.logger.info(f"Successfully generated embeddings for {len(all_chunks)} chunks")
            return all_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from current adapter."""
        if self._current_adapter:
            return self._current_adapter.get_dimension()
        return settings.VECTOR_DIMENSION
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current embedding adapter."""
        if self._current_adapter:
            info = self._current_adapter.get_info()
            info['available_adapters'] = list(self._adapters.keys())
            return info
        return {
            'type': 'none',
            'dimension': settings.VECTOR_DIMENSION,
            'available_adapters': list(self._adapters.keys()),
            'initialized': False
        }