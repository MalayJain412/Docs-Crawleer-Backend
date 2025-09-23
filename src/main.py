"""Main entry point for the AI-Powered Documentation Crawler & Q/A System."""

import asyncio
from sympy import false, true
import uvicorn
from pathlib import Path
import sys
import os

# Add src to Python path for absolute imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Also add the project root to Python path
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Set environment variable to help with imports
os.environ['PYTHONPATH'] = str(current_dir)

try:
    from config.settings import settings
    from utils.logger import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the src directory")
    sys.exit(1)


def create_app():
    """Create and return the FastAPI app."""
    try:
        from api.endpoints import DocumentCrawlerAPI
        api_instance = DocumentCrawlerAPI()
        return api_instance.get_app()
    except ImportError as e:
        print(f"Failed to import FastAPI app: {e}")
        sys.exit(1)


# Create the app instance for import string usage
app = create_app()


def main():
    """Main function to run the FastAPI server."""
    # Setup logging
    logger = setup_logger("main", settings.LOG_LEVEL)
    logger.info("Starting AI-Powered Documentation Crawler & Q/A System")
    
    # Run the FastAPI server
    if settings.API_RELOAD:
        # Use import string for reload mode
        uvicorn.run(
            "main:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=false,
            log_level=settings.LOG_LEVEL.lower()
        )
    else:
        # Use app object for non-reload mode
        app = create_app()
        uvicorn.run(
            app,
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=False,
            log_level=settings.LOG_LEVEL.lower()
        )


if __name__ == "__main__":
    main()