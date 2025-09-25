#!/usr/bin/env python3
"""
Run script for the AI-Powered Documentation Crawler & Q/A System.
This script handles the proper setup and execution of the FastAPI server.
"""

import sys
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.resolve()
src_dir = project_root / "src"

# Add src to Python path
sys.path.insert(0, str(src_dir))

# Change to src directory
os.chdir(src_dir)

# Import and run main
try:
    from main import main

    if __name__ == "__main__":
        # Respect container apps / App Service ports if provided
        port = os.getenv("API_PORT") or os.getenv("PORT") or os.getenv("WEBSITES_PORT") or "8000"
        os.environ["API_PORT"] = port
        os.environ["PORT"] = port
        print(f"Starting server from: {os.getcwd()} (port={port})")
        print(f"Python path includes: {src_dir}")

        # Print detected Azure storage settings (non-sensitive)
        storage_conn = bool(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
        storage_url = bool(os.getenv("AZURE_STORAGE_ACCOUNT_URL"))
        print(f"AZURE_STORAGE_CONNECTION_STRING set: {storage_conn}")
        print(f"AZURE_STORAGE_ACCOUNT_URL set: {storage_url} (use Managed Identity if true)")

        # Start main (main() should read settings / env as needed)
        main()

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting server: {e}")
    sys.exit(1)