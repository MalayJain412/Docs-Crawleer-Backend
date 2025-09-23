"""FAISS vector store for similarity search."""

import faiss
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

try:
    from config.settings import settings
    from utils.logger import default_logger
    from storage.schemas import ContentChunk, EmbeddingIndex
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from config.settings import settings
    from utils.logger import default_logger
    from storage.schemas import ContentChunk, EmbeddingIndex


class VectorStore:
    """FAISS-based vector store for document embeddings."""
    
    def __init__(self, domain: str, domain_folder: str):
        """
        Initialize vector store for a domain.
        
        Args:
            domain: Domain name
            domain_folder: Path to domain folder
        """
        self.domain = domain
        self.domain_folder = Path(domain_folder)
        self.faiss_dir = self.domain_folder / "faiss"
        self.faiss_dir.mkdir(exist_ok=True)
        
        self.logger = default_logger
        
        # FAISS index
        self.index: Optional[faiss.Index] = None
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.embedding_dimension: int = settings.VECTOR_DIMENSION
        
        # File paths
        self.index_file = self.faiss_dir / "index.faiss"
        self.metadata_file = self.faiss_dir / "metadata.json"
        self.info_file = self.faiss_dir / "index_info.json"
    
    def create_index(self, chunks: List[ContentChunk], model_name: str = "unknown") -> EmbeddingIndex:
        """
        Create a new FAISS index from content chunks.
        
        Args:
            chunks: List of ContentChunk objects with embeddings
            model_name: Name of the embedding model used
            
        Returns:
            EmbeddingIndex object with index information
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        # Validate that all chunks have embeddings
        chunks_with_embeddings = [chunk for chunk in chunks if chunk.embedding]
        if not chunks_with_embeddings:
            raise ValueError("No chunks with embeddings found")
        
        self.logger.info(f"Creating FAISS index for {len(chunks_with_embeddings)} chunks")
        
        # Extract embeddings and metadata
        embeddings = np.array([chunk.embedding for chunk in chunks_with_embeddings], dtype=np.float32)
        self.embedding_dimension = embeddings.shape[1]
        
        # Create FAISS index
        if settings.FAISS_INDEX_TYPE == "IndexFlatIP":
            # Inner product (cosine similarity for normalized vectors)
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
        else:
            # L2 distance (Euclidean)
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
        
        # Normalize embeddings for cosine similarity
        if settings.FAISS_INDEX_TYPE == "IndexFlatIP":
            faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Prepare metadata
        self.chunk_metadata = []
        for chunk in chunks_with_embeddings:
            metadata = {
                'chunk_id': chunk.chunk_id,
                'source_url': chunk.source_url,
                'content': chunk.content,
                'chunk_index': chunk.chunk_index,
                'metadata': chunk.metadata
            }
            self.chunk_metadata.append(metadata)
        
        # Save to disk
        self._save_index()
        
        # Create index info
        index_info = EmbeddingIndex(
            domain=self.domain,
            total_chunks=len(chunks_with_embeddings),
            vector_dimension=self.embedding_dimension,
            model_name=model_name,
            index_file_path=str(self.index_file),
            metadata_file_path=str(self.metadata_file)
        )
        
        # Save index info
        self._save_index_info(index_info)
        
        self.logger.info(f"Successfully created FAISS index with {len(chunks_with_embeddings)} vectors")
        return index_info
    
    def load_index(self) -> bool:
        """
        Load existing FAISS index from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not self.index_file.exists() or not self.metadata_file.exists():
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_file))
            
            # Load metadata
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.chunk_metadata = json.load(f)
            
            self.embedding_dimension = self.index.d
            
            self.logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors from {self.index_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load FAISS index: {e}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        if not self.index:
            raise ValueError("No index loaded. Call load_index() or create_index() first.")
        
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(f"Query embedding dimension {len(query_embedding)} doesn't match index dimension {self.embedding_dimension}")
        
        # Prepare query vector
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Normalize for cosine similarity
        if settings.FAISS_INDEX_TYPE == "IndexFlatIP":
            faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_metadata):
                continue
                
            metadata = self.chunk_metadata[idx].copy()
            metadata['similarity_score'] = float(score)
            metadata['faiss_index'] = int(idx)
            
            results.append(metadata)
        
        return results
    
    def add_chunks(self, chunks: List[ContentChunk]) -> int:
        """
        Add new chunks to existing index.
        
        Args:
            chunks: List of ContentChunk objects with embeddings
            
        Returns:
            Number of chunks added
        """
        if not self.index:
            raise ValueError("No index loaded. Call load_index() or create_index() first.")
        
        chunks_with_embeddings = [chunk for chunk in chunks if chunk.embedding]
        if not chunks_with_embeddings:
            return 0
        
        # Extract embeddings
        embeddings = np.array([chunk.embedding for chunk in chunks_with_embeddings], dtype=np.float32)
        
        # Normalize for cosine similarity
        if settings.FAISS_INDEX_TYPE == "IndexFlatIP":
            faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Add metadata
        for chunk in chunks_with_embeddings:
            metadata = {
                'chunk_id': chunk.chunk_id,
                'source_url': chunk.source_url,
                'content': chunk.content,
                'chunk_index': chunk.chunk_index,
                'metadata': chunk.metadata
            }
            self.chunk_metadata.append(metadata)
        
        # Save updated index
        self._save_index()
        
        self.logger.info(f"Added {len(chunks_with_embeddings)} chunks to FAISS index")
        return len(chunks_with_embeddings)
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.chunk_metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved FAISS index to {self.index_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save FAISS index: {e}")
            raise
    
    def _save_index_info(self, index_info: EmbeddingIndex):
        """Save index information to disk."""
        try:
            with open(self.info_file, 'w', encoding='utf-8') as f:
                json.dump(index_info.dict(), f, indent=2, default=str, ensure_ascii=False)
                
        except Exception as e:
            self.logger.warning(f"Failed to save index info: {e}")
    
    def get_index_info(self) -> Optional[EmbeddingIndex]:
        """Get index information."""
        try:
            if self.info_file.exists():
                with open(self.info_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return EmbeddingIndex(**data)
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to load index info: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        stats = {
            'domain': self.domain,
            'total_vectors': 0,
            'embedding_dimension': self.embedding_dimension,
            'index_file_exists': self.index_file.exists(),
            'metadata_file_exists': self.metadata_file.exists(),
            'index_loaded': self.index is not None
        }
        
        if self.index:
            stats['total_vectors'] = self.index.ntotal
            stats['total_metadata'] = len(self.chunk_metadata)
        
        # File sizes
        if self.index_file.exists():
            stats['index_file_size'] = self.index_file.stat().st_size
        
        if self.metadata_file.exists():
            stats['metadata_file_size'] = self.metadata_file.stat().st_size
        
        return stats
    
    def delete_index(self):
        """Delete the FAISS index and associated files."""
        try:
            files_to_delete = [self.index_file, self.metadata_file, self.info_file]
            
            for file_path in files_to_delete:
                if file_path.exists():
                    file_path.unlink()
                    
            self.index = None
            self.chunk_metadata = []
            
            self.logger.info(f"Deleted FAISS index for domain {self.domain}")
            
        except Exception as e:
            self.logger.error(f"Failed to delete FAISS index: {e}")
            raise