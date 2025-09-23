"""Embedding service with Gemini primary and sentence-transformers fallback."""

import asyncio
from typing import List, Optional, Dict, Any, Tuple
import time
import re

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
        self._gemini_model = None
        self._sentence_transformer = None
        self._current_model = None
        
    async def initialize(self):
        """Initialize embedding models."""
        try:
            # Try to initialize sentence-transformers first (primary)
            self._sentence_transformer = await self._init_sentence_transformer()
            if self._sentence_transformer:
                self._current_model = "sentence-transformer"
                self.logger.info("Initialized sentence-transformers embeddings (primary)")
                return
            
            # Fallback to Gemini if sentence-transformers fails
            if settings.GEMINI_API_KEY:
                self._gemini_model = await self._init_gemini()
                if self._gemini_model:
                    self._current_model = "gemini"
                    self.logger.info("Initialized Gemini embeddings (fallback)")
                    return
            
            raise Exception("No embedding model could be initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding models: {e}")
            raise
    
    async def _init_gemini(self) -> Optional[Any]:
        """Initialize Gemini embedding model."""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # Test the model with Gemini Flash (latest version available for embeddings and generation)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            test_response = model.generate_content("Test")
            
            return genai
            
        except ImportError:
            self.logger.warning("google-generativeai not installed, skipping Gemini")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to initialize Gemini: {e}")
            return None
    
    async def _init_sentence_transformer(self) -> Optional[Any]:
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
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        if self._current_model == "gemini":
            return await self._generate_gemini_embeddings(texts)
        elif self._current_model == "sentence-transformer":
            return await self._generate_st_embeddings(texts)
        else:
            raise Exception("No embedding model available")
    
    async def _generate_gemini_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Gemini."""
        try:
            import google.generativeai as genai
            
            embeddings = []
            
            # Gemini has rate limits, so process in smaller batches
            batch_size = 5
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = []
                
                for text in batch:
                    try:
                        # Use text embedding model
                        result = genai.embed_content(
                            model="models/embedding-001",
                            content=text,
                            task_type="retrieval_document"
                        )
                        
                        if 'embedding' in result:
                            batch_embeddings.append(result['embedding'])
                        else:
                            # Fallback: create zero vector
                            batch_embeddings.append([0.0] * settings.VECTOR_DIMENSION)
                        
                        # Rate limiting
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to embed text with Gemini: {e}")
                        batch_embeddings.append([0.0] * settings.VECTOR_DIMENSION)
                
                embeddings.extend(batch_embeddings)
                
                # Delay between batches
                if i + batch_size < len(texts):
                    await asyncio.sleep(1)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Gemini embedding failed: {e}")
            # Try fallback
            return await self._generate_st_embeddings(texts)
    
    async def _generate_st_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence-transformers."""
        try:
            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, 
                self._sentence_transformer.encode, 
                texts
            )
            
            # Convert numpy arrays to lists
            return [embedding.tolist() for embedding in embeddings]
            
        except Exception as e:
            self.logger.error(f"Sentence-transformer embedding failed: {e}")
            raise
    
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
    
    async def embed_documents(self, documents: List[DocumentContent]) -> List[ContentChunk]:
        """
        Create embeddings for a list of documents.
        
        Args:
            documents: List of DocumentContent objects
            
        Returns:
            List of ContentChunk objects with embeddings
        """
        if not documents:
            return []
        
        self.logger.info(f"Processing {len(documents)} documents for embedding")
        
        # Chunk all documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        self.logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        if not all_chunks:
            return []
        
        # Extract text content for embedding
        chunk_texts = [chunk.content for chunk in all_chunks]
        
        # Generate embeddings
        try:
            embeddings = await self.generate_embeddings(chunk_texts)
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk.embedding = embedding
            
            self.logger.info(f"Successfully generated embeddings for {len(all_chunks)} chunks")
            return all_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from current model."""
        if self._current_model == "gemini":
            return 768  # Gemini embedding dimension
        elif self._current_model == "sentence-transformer":
            return 384  # all-MiniLM-L6-v2 dimension
        else:
            return settings.VECTOR_DIMENSION
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current embedding model."""
        return {
            'model_type': self._current_model,
            'dimension': self.get_embedding_dimension(),
            'available_models': {
                'gemini': self._gemini_model is not None,
                'sentence_transformer': self._sentence_transformer is not None
            }
        }