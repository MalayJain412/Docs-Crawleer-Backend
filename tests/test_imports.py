"""Simple test script to check if all imports work correctly."""

import sys
from pathlib import Path

# Add src to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

try:
    print("Testing imports...")
    
    # Test basic imports
    from config.settings import settings
    print("✓ Settings imported successfully")
    
    from utils.logger import setup_logger
    print("✓ Logger imported successfully")
    
    from utils.url_utils import URLUtils
    print("✓ URL utils imported successfully")
    
    from storage.schemas import DocumentContent, CrawlSession
    print("✓ Storage schemas imported successfully")
    
    from storage.storage_manager import StorageManager
    print("✓ Storage manager imported successfully")
    
    # Test API imports
    from api.models import CrawlRequest, QueryRequest
    print("✓ API models imported successfully")
    
    from api.endpoints import DocumentCrawlerAPI
    print("✓ API endpoints imported successfully")
    
    # Try to create the app
    api = DocumentCrawlerAPI()
    app = api.get_app()
    print("✓ FastAPI app created successfully")
    
    print("\n✅ All imports successful!")
    print(f"Server would run on: {settings.API_HOST}:{settings.API_PORT}")
    print(f"Reload mode: {settings.API_RELOAD}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()