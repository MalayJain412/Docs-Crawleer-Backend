"""Production entry point for Documentation Crawler."""
from pathlib import Path
import sys
import os

# Set Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from api.endpoints import DocumentCrawlerAPI
from utils.logger import setup_logger
from config.settings import settings


def create_app() -> FastAPI:
    """Create production FastAPI app."""
    api_instance = DocumentCrawlerAPI()
    app = api_instance.get_app()

    @app.on_event("startup")
    async def startup_services():
        """Initialize services when the FastAPI app starts."""
        logger = setup_logger("startup", settings.LOG_LEVEL)
        logger.info(f"Documentation crawler started on port {settings.API_PORT}")
        await api_instance.initialize_services()

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        log_level="info",
    )