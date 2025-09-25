"""Storage manager for handling JSON/YAML files with domain-based organization."""

import json
import yaml
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from config.settings import settings
    from utils.logger import default_logger
    from utils.url_utils import URLUtils
    from storage.schemas import DocumentContent, CrawlSession
    from storage.azure_blob import upload_file_sync, download_file_sync
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from config.settings import settings
    from utils.logger import default_logger
    from utils.url_utils import URLUtils
    from storage.schemas import DocumentContent, CrawlSession
    from storage.azure_blob import upload_file_sync, download_file_sync


class StorageManager:
    """Manages storage operations for crawled documentation."""
    
    def __init__(self, base_data_dir: str = None):
        """
        Initialize storage manager.
        
        Args:
            base_data_dir: Base directory for data storage
        """
        self.base_dir = Path(base_data_dir or settings.DATA_DIR)
        self.logger = default_logger
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def get_domain_folder(self, url: str, custom_name: Optional[str] = None) -> str:
        """
        Get or create domain-specific folder for storage.
        
        Args:
            url: URL to extract domain from
            custom_name: Custom name for the domain folder
            
        Returns:
            Path to domain folder
        """
        if custom_name:
            folder_name = self._sanitize_folder_name(custom_name)
        else:
            folder_name = URLUtils.extract_domain_name(url)
        
        domain_path = self.base_dir / folder_name
        
        # Create subdirectories
        for subdir in ['json', 'yaml', 'faiss']:
            (domain_path / subdir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created/verified domain folder: {domain_path}")
        return str(domain_path)
    
    def _sanitize_folder_name(self, name: str) -> str:
        """Sanitize folder name for filesystem compatibility."""
        import re
        # Replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        return sanitized.strip('_')
    
    def save_crawl_session(self, session: CrawlSession, domain_folder: str) -> str:
        """
        Save crawl session metadata.
        
        Args:
            session: CrawlSession object
            domain_folder: Domain folder path
            
        Returns:
            Path to saved session file
        """
        session_data = session.dict()
        
        # Save as JSON
        json_path = Path(domain_folder) / 'json' / 'crawl_session.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, default=str, ensure_ascii=False)
        
        # Save as YAML
        yaml_path = Path(domain_folder) / 'yaml' / 'crawl_session.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(session_data, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"Saved crawl session: {json_path}")
        return str(json_path)
    
    def save_documents(self, documents: List[DocumentContent], domain_folder: str) -> Dict[str, str]:
        """
        Save crawled documents in both JSON and YAML formats.
        
        Args:
            documents: List of DocumentContent objects
            domain_folder: Domain folder path
            
        Returns:
            Dictionary with paths to saved files
        """
        # Prepare data for serialization
        documents_data = {
            'metadata': {
                'total_documents': len(documents),
                'saved_at': datetime.now().isoformat(),
                'format_version': '1.0'
            },
            'documents': [doc.dict() for doc in documents]
        }
        
        # Save as JSON
        json_path = Path(domain_folder) / 'json' / 'documents.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(documents_data, f, indent=2, default=str, ensure_ascii=False)
        
        # Save as YAML
        yaml_path = Path(domain_folder) / 'yaml' / 'documents.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(documents_data, f, default_flow_style=False, allow_unicode=True)
        
        # Also save individual document files for easier access
        json_docs_dir = Path(domain_folder) / 'json' / 'individual'
        yaml_docs_dir = Path(domain_folder) / 'yaml' / 'individual'
        json_docs_dir.mkdir(exist_ok=True)
        yaml_docs_dir.mkdir(exist_ok=True)
        
        for i, doc in enumerate(documents):
            # Create safe filename
            filename = self._create_safe_filename(doc.url, i)
            
            # Save individual JSON
            json_doc_path = json_docs_dir / f"{filename}.json"
            with open(json_doc_path, 'w', encoding='utf-8') as f:
                json.dump(doc.dict(), f, indent=2, default=str, ensure_ascii=False)
            
            # Save individual YAML
            yaml_doc_path = yaml_docs_dir / f"{filename}.yaml"
            with open(yaml_doc_path, 'w', encoding='utf-8') as f:
                yaml.dump(doc.dict(), f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"Saved {len(documents)} documents to {domain_folder}")
        return {
            'json': str(json_path),
            'yaml': str(yaml_path),
            'json_individual': str(json_docs_dir),
            'yaml_individual': str(yaml_docs_dir)
        }
    
    def _create_safe_filename(self, url: str, index: int) -> str:
        """Create a safe filename from URL."""
        from urllib.parse import urlparse
        import re
        
        try:
            parsed = urlparse(url)
            path = parsed.path.strip('/')
            
            if path:
                # Use path as filename
                filename = path.replace('/', '_')
            else:
                # Use domain + index
                filename = f"{parsed.netloc}_{index}"
            
            # Clean filename
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            filename = re.sub(r'_+', '_', filename)
            filename = filename.strip('_')
            
            # Limit length
            if len(filename) > 200:
                filename = filename[:200]
            
            return filename or f"document_{index}"
        except Exception:
            return f"document_{index}"
    
    def load_documents(self, domain_folder: str, format_type: str = 'json') -> List[DocumentContent]:
        """
        Load documents from storage.
        
        Args:
            domain_folder: Domain folder path
            format_type: 'json' or 'yaml'
            
        Returns:
            List of DocumentContent objects
        """
        file_path = Path(domain_folder) / format_type / 'documents.json' if format_type == 'json' else Path(domain_folder) / format_type / 'documents.yaml'
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if format_type == 'json':
                    data = json.load(f)
                else:
                    data = yaml.safe_load(f)
            
            documents = [DocumentContent(**doc_data) for doc_data in data['documents']]
            self.logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except FileNotFoundError:
            self.logger.warning(f"Documents file not found: {file_path}")
            return []
        except Exception as e:
            self.logger.error(f"Error loading documents from {file_path}: {e}")
            return []
    
    def get_domain_stats(self, domain_folder: str) -> Dict[str, Any]:
        """
        Get statistics for a domain folder.
        
        Args:
            domain_folder: Domain folder path
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'domain_folder': domain_folder,
            'json_files': 0,
            'yaml_files': 0,
            'faiss_files': 0,
            'total_documents': 0,
            'last_crawl': None
        }
        
        domain_path = Path(domain_folder)
        
        if domain_path.exists():
            # Count files
            json_dir = domain_path / 'json'
            yaml_dir = domain_path / 'yaml'
            faiss_dir = domain_path / 'faiss'
            
            if json_dir.exists():
                stats['json_files'] = len(list(json_dir.glob('*.json')))
            
            if yaml_dir.exists():
                stats['yaml_files'] = len(list(yaml_dir.glob('*.yaml')))
                
            if faiss_dir.exists():
                stats['faiss_files'] = len(list(faiss_dir.glob('*')))
            
            # Try to get document count and last crawl time
            try:
                session_file = json_dir / 'crawl_session.json'
                if session_file.exists():
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                        stats['total_documents'] = session_data.get('total_pages', 0)
                        stats['last_crawl'] = session_data.get('started_at')
            except Exception:
                pass
        
        return stats
    
    def list_domains(self) -> List[Dict[str, Any]]:
        """
        List all available domains with their statistics.
        
        Returns:
            List of domain information
        """
        domains = []
        
        if self.base_dir.exists():
            for domain_dir in self.base_dir.iterdir():
                if domain_dir.is_dir() and domain_dir.name != 'logs':
                    stats = self.get_domain_stats(str(domain_dir))
                    stats['domain_name'] = domain_dir.name
                    domains.append(stats)
        
        return domains
    
    def save_faiss_index_atomic(self, index_data: bytes, domain_folder: str, filename: str = "index.faiss") -> str:
        """
        Atomically save FAISS index file and optionally upload to blob storage.
        
        Args:
            index_data: FAISS index data as bytes
            domain_folder: Domain folder path
            filename: FAISS index filename
            
        Returns:
            Path to saved index file
        """
        faiss_dir = Path(domain_folder) / 'faiss'
        faiss_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = faiss_dir / filename
        temp_path = faiss_dir / f"{filename}.tmp"
        
        try:
            # Write to temporary file first
            with open(temp_path, 'wb') as f:
                f.write(index_data)
            
            # Atomic move to final location
            os.replace(temp_path, index_path)
            
            self.logger.info(f"Saved FAISS index: {index_path}")
            
            # Upload to blob storage if configured
            self._upload_to_blob_if_configured(str(index_path), domain_folder, filename)
            
            return str(index_path)
            
        except Exception as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def _upload_to_blob_if_configured(self, local_path: str, domain_folder: str, filename: str):
        """
        Upload file to blob storage if Azure storage is configured.
        
        Args:
            local_path: Local file path
            domain_folder: Domain folder path (used to create blob name)
            filename: Filename for blob
        """
        try:
            # Check if Azure storage is configured
            storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            storage_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
            
            if not (storage_conn or storage_url):
                self.logger.debug("Azure storage not configured, skipping blob upload")
                return
            
            # Create blob name with domain prefix
            domain_name = Path(domain_folder).name
            blob_name = f"{domain_name}/{filename}"
            
            # Upload using sync wrapper (since this method is sync)
            upload_file_sync(local_path, blob_name=blob_name)
            
            self.logger.info(f"Uploaded {filename} to blob: {blob_name}")
            
        except Exception as e:
            # Log error but don't fail the main operation
            self.logger.warning(f"Failed to upload {filename} to blob storage: {e}")
    
    def download_faiss_index_from_blob(self, domain_folder: str, filename: str = "index.faiss") -> Optional[str]:
        """
        Download FAISS index from blob storage if available.
        
        Args:
            domain_folder: Domain folder path
            filename: FAISS index filename
            
        Returns:
            Path to downloaded file or None if not available
        """
        try:
            # Check if Azure storage is configured
            storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            storage_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
            
            if not (storage_conn or storage_url):
                self.logger.debug("Azure storage not configured, skipping blob download")
                return None
            
            faiss_dir = Path(domain_folder) / 'faiss'
            faiss_dir.mkdir(parents=True, exist_ok=True)
            
            local_path = faiss_dir / filename
            domain_name = Path(domain_folder).name
            blob_name = f"{domain_name}/{filename}"
            
            # Download using sync wrapper
            download_file_sync(str(local_path), blob_name=blob_name)
            
            self.logger.info(f"Downloaded {filename} from blob: {blob_name}")
            return str(local_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to download {filename} from blob storage: {e}")
            return None