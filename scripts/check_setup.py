#!/usr/bin/env python3
"""
Setup validation script for Invoice OCR Extractor.

Checks all system requirements and provides guidance for missing components.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print('=' * 60)


def print_check(name: str, status: bool, message: str = ""):
    """Print a check result."""
    icon = "✅" if status else "❌"
    print(f"{icon} {name}: {message}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"⚠️  {message}")


def print_info(message: str):
    """Print an info message."""
    print(f"ℹ️  {message}")


def check_python_version():
    """Check Python version."""
    print_header("Python Version")
    
    version = sys.version_info
    required = (3, 9)
    
    is_ok = version >= required
    print_check(
        "Python",
        is_ok,
        f"{version.major}.{version.minor}.{version.micro} "
        f"({'OK' if is_ok else f'requires {required[0]}.{required[1]}+'})"
    )
    
    return is_ok


def check_tesseract():
    """Check Tesseract installation."""
    print_header("Tesseract OCR")
    
    # Check if tesseract is available
    tesseract_cmd = shutil.which("tesseract")
    
    if tesseract_cmd:
        try:
            result = subprocess.run(
                [tesseract_cmd, "--version"],
                capture_output=True,
                text=True,
            )
            version_line = result.stdout.split('\n')[0] if result.stdout else "unknown"
            print_check("Tesseract", True, f"Found at {tesseract_cmd}")
            print_info(f"Version: {version_line}")
            return True
        except Exception as e:
            print_check("Tesseract", False, f"Error: {e}")
            return False
    else:
        print_check("Tesseract", False, "Not found in PATH")
        print_info("Install with:")
        print_info("  macOS: brew install tesseract")
        print_info("  Ubuntu: sudo apt install tesseract-ocr")
        print_info("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False


def check_poppler():
    """Check Poppler installation (for PDF support)."""
    print_header("Poppler (PDF Support)")
    
    # Check for pdfinfo or pdftoppm
    poppler_cmd = shutil.which("pdfinfo") or shutil.which("pdftoppm")
    
    if poppler_cmd:
        print_check("Poppler", True, f"Found: {poppler_cmd}")
        return True
    else:
        print_check("Poppler", False, "Not found (PDF support may not work)")
        print_info("Install with:")
        print_info("  macOS: brew install poppler")
        print_info("  Ubuntu: sudo apt install poppler-utils")
        return False


def check_ollama():
    """Check Ollama installation and status."""
    print_header("Ollama (Local LLM)")
    
    ollama_cmd = shutil.which("ollama")
    
    if not ollama_cmd:
        print_check("Ollama", False, "Not installed")
        print_info("Install from: https://ollama.com/download")
        return False
    
    print_check("Ollama CLI", True, f"Found at {ollama_cmd}")
    
    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            print_check("Ollama Server", True, "Running")
            
            if models:
                print_info(f"Available models: {', '.join(model_names[:5])}")
                if len(models) > 5:
                    print_info(f"  ... and {len(models) - 5} more")
            else:
                print_warning("No models installed. Run: ollama pull llama3.2")
            
            # Check for vision models
            vision_models = [m for m in model_names if "llava" in m.lower() or "vision" in m.lower()]
            if vision_models:
                print_info(f"Vision models available: {', '.join(vision_models)}")
            else:
                print_warning("No vision models. For vision support: ollama pull llava")
            
            return True
        else:
            print_check("Ollama Server", False, "Not responding correctly")
            return False
    except requests.exceptions.ConnectionError:
        print_check("Ollama Server", False, "Not running")
        print_info("Start with: ollama serve")
        return False
    except ImportError:
        print_warning("requests library not installed - cannot check Ollama server")
        return False


def check_lm_studio():
    """Check LM Studio server."""
    print_header("LM Studio (Local LLM)")
    
    try:
        import requests
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            print_check("LM Studio Server", True, "Running at localhost:1234")
            return True
        else:
            print_check("LM Studio Server", False, "Not responding correctly")
            return False
    except requests.exceptions.ConnectionError:
        print_check("LM Studio Server", False, "Not running")
        print_info("Download from: https://lmstudio.ai")
        print_info("Start the local server after loading a model")
        return False
    except ImportError:
        print_warning("requests library not installed - cannot check LM Studio")
        return False


def check_python_packages():
    """Check required Python packages."""
    print_header("Python Packages")
    
    required_packages = [
        "streamlit",
        "pytesseract",
        "opencv-python",
        "PIL",  # Pillow
        "pdf2image",
        "pydantic",
        "openpyxl",
        "pandas",
        "requests",
        "tenacity",
    ]
    
    # Map import names to package names
    import_map = {
        "PIL": "Pillow",
        "cv2": "opencv-python",
    }
    
    all_ok = True
    
    for package in required_packages:
        import_name = package
        display_name = import_map.get(package, package)
        
        # Handle special cases
        if package == "opencv-python":
            import_name = "cv2"
        
        try:
            __import__(import_name)
            print_check(display_name, True, "Installed")
        except ImportError:
            print_check(display_name, False, "Not installed")
            all_ok = False
    
    if not all_ok:
        print_info("\nInstall missing packages with:")
        print_info("  pip install -r requirements.txt")
    
    return all_ok


def check_env_file():
    """Check for .env file."""
    print_header("Environment Configuration")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print_check(".env file", True, "Found")
        
        # Check for Deepseek API key
        from dotenv import load_dotenv
        load_dotenv()
        
        if os.getenv("DEEPSEEK_API_KEY"):
            print_check("Deepseek API Key", True, "Configured")
        else:
            print_info("Deepseek API key not set (cloud fallback won't work)")
        
        return True
    else:
        print_check(".env file", False, "Not found")
        if env_example.exists():
            print_info("Copy .env.example to .env and configure:")
            print_info("  cp .env.example .env")
        return False


def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("  Invoice OCR Extractor - Setup Validation")
    print("=" * 60)
    
    results = {}
    
    # Run checks
    results["python"] = check_python_version()
    results["tesseract"] = check_tesseract()
    results["poppler"] = check_poppler()
    results["ollama"] = check_ollama()
    results["lm_studio"] = check_lm_studio()
    results["packages"] = check_python_packages()
    results["env"] = check_env_file()
    
    # Summary
    print_header("Summary")
    
    critical_ok = results["python"] and results["tesseract"]
    llm_ok = results["ollama"] or results["lm_studio"]
    
    if critical_ok and llm_ok:
        print("✅ System is ready to run the application!")
        print("\nStart with:")
        print("  streamlit run app/main.py")
    else:
        print("❌ Some requirements are missing:")
        
        if not results["python"]:
            print("  - Python 3.9+ required")
        if not results["tesseract"]:
            print("  - Tesseract OCR required for text extraction")
        if not llm_ok:
            print("  - At least one LLM provider (Ollama or LM Studio) required")
        if not results["packages"]:
            print("  - Some Python packages missing (run: pip install -r requirements.txt)")
    
    print()
    return 0 if (critical_ok and llm_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
