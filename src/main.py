"""Main entry point for the AI-Powered Documentation Crawler & Q/A System."""

import asyncio
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
        app = api_instance.get_app()
        
        # Add startup and shutdown events
        @app.on_event("startup")
        async def startup_event():
            """Initialize services on startup."""
            logger = setup_logger("startup", settings.LOG_LEVEL)
            logger.info("Application startup: initializing services")
            # Initialize embedding service and other async resources
            try:
                await api_instance.initialize_services()
                logger.info("Services initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize services: {e}")
                raise
        
        @app.on_event("shutdown")
        async def shutdown_event():
            """Clean up resources on shutdown."""
            logger = setup_logger("shutdown", settings.LOG_LEVEL)
            logger.info("Application shutdown: cleaning up resources")
            try:
                await api_instance.cleanup_services()
                logger.info("Resources cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        
        # Add health endpoints
        @app.get("/health")
        async def health_check():
            """Basic health check endpoint."""
            return {"status": "healthy", "service": "documentation-crawler"}
        
        @app.get("/ready")
        async def readiness_check():
            """Readiness check endpoint."""
            try:
                # Check if services are ready
                ready_status = await api_instance.check_readiness()
                return {
                    "status": "ready" if ready_status else "not ready",
                    "service": "documentation-crawler",
                    "details": ready_status
                }
            except Exception as e:
                return {
                    "status": "error",
                    "service": "documentation-crawler",
                    "error": str(e)
                }
        
        return app
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
            reload=True,
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