#!/usr/bin/env python3
"""Server runner with PYTHONPATH setup and diagnostics."""

import os
import sys
import argparse
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Set working directory to project root
os.chdir(project_root)

def print_diagnostics():
    """Print environment diagnostics."""
    print(f"🚀 Starting server from: {project_root}")
    print(f"📁 Python path includes: {src_path}")
    
    # Check key environment variables
    env_vars = [
        'GEMINI_API_KEY', 'JINA_API_KEY', 'AZURE_OPENAI_API_KEY', 
        'MODEL_USE', 'DATA_DIR', 'API_PORT'
    ]
    print("\n🔧 Environment:")
    for var in env_vars:
        value = os.getenv(var, 'Not set')
        # Mask API keys
        if 'API_KEY' in var and value != 'Not set':
            value = f"{value[:8]}..." if len(value) > 8 else "***"
        print(f"   {var}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the documentation crawler server")
    parser.add_argument('--dev', action='store_true', help='Run in development mode with reload')
    parser.add_argument('--port', type=int, default=5002, help='Port to run on')
    args = parser.parse_args()
    
    print_diagnostics()
    
    import uvicorn
    
    # Import after path setup
    from main import app
    
    # Run with or without auto-reload
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.dev,
        reload_dirs=[str(src_path)] if args.dev else None,
        log_level="info"
    )