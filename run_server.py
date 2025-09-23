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
        print(f"Starting server from: {os.getcwd()}")
        print(f"Python path includes: {src_dir}")
        main()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting server: {e}")
    sys.exit(1)