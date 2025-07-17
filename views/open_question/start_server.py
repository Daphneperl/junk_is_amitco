#!/usr/bin/env python3
"""
Startup script for the Open Question Search Server
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True

def start_server():
    """Start the Flask server"""
    print("Starting server...")
    try:
        subprocess.run([sys.executable, "server.py"])
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    print("=== Open Question Search Server ===")
    
    # Check if we're in the right directory
    if not os.path.exists("server.py"):
        print("Error: server.py not found. Please run this script from the open_question directory.")
        sys.exit(1)
    
    # Install requirements
    if install_requirements():
        print("\nStarting server on http://localhost:8000")
        print("Open openQuestion_with_search.html in your browser")
        print("Press Ctrl+C to stop the server\n")
        start_server()
    else:
        print("Failed to install requirements. Please check your Python environment.")
        sys.exit(1) 