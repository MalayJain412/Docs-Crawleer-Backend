import sys
import os
from pathlib import Path
from dotenv import load_dotenv  # ensure python-dotenv is installed

# ---- Load environment variables ----
project_root = Path(__file__).parent.resolve()
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Default port = 8006 if not set in .env
os.environ.setdefault("API_PORT", "8006")
os.environ.setdefault("PORT", os.getenv("API_PORT"))

# ---- Setup paths ----
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))
os.chdir(src_dir)

# ---- Import the FastAPI app ----
from main import app

# ---- Expose as WSGI entry point ----
application = app
