"""
Configuration module for Invoice OCR App.

Handles settings for LLM providers, API keys, Tesseract configuration,
and application-wide settings with validation.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMProvider(Enum):
    """Supported LLM providers for invoice extraction."""
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    DEEPSEEK = "deepseek"
    VISION_OLLAMA = "vision_ollama"  # For vision models like llava


class ExtractionMode(Enum):
    """Mode for extracting data from invoices."""
    OCR_LLM = "ocr_llm"  # Traditional OCR + LLM text parsing
    VISION = "vision"    # Direct image-to-text via vision model


@dataclass
class TesseractConfig:
    """Configuration for Tesseract OCR engine."""
    tesseract_cmd: Optional[str] = None
    language: str = "eng"
    oem: int = 3  # OCR Engine Mode: 3 = Default, based on what is available
    psm: int = 6  # Page Segmentation Mode: 6 = Assume uniform block of text
    
    def get_config_string(self) -> str:
        """Generate Tesseract configuration string."""
        return f"--oem {self.oem} --psm {self.psm}"
    
    @staticmethod
    def find_tesseract() -> Optional[str]:
        """Attempt to find Tesseract installation."""
        # Check common installation paths
        common_paths = [
            "/usr/local/bin/tesseract",  # macOS Homebrew
            "/opt/homebrew/bin/tesseract",  # macOS M1/M2 Homebrew
            "/usr/bin/tesseract",  # Linux
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",  # Windows
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",  # Windows x86
        ]
        
        # First check if tesseract is in PATH
        tesseract_path = shutil.which("tesseract")
        if tesseract_path:
            return tesseract_path
        
        # Check common installation paths
        for path in common_paths:
            if Path(path).exists():
                return path
        
        return None
    
    @staticmethod
    def validate_installation() -> tuple[bool, str]:
        """
        Validate Tesseract installation.
        
        Returns:
            Tuple of (is_valid, message)
        """
        tesseract_path = TesseractConfig.find_tesseract()
        
        if not tesseract_path:
            return False, (
                "Tesseract OCR is not installed or not found in PATH.\n\n"
                "Installation instructions:\n"
                "• macOS: brew install tesseract\n"
                "• Ubuntu/Debian: sudo apt install tesseract-ocr\n"
                "• Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
            )
        
        try:
            # Verify it runs
            result = subprocess.run(
                [tesseract_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version_info = result.stdout.split('\n')[0]
                return True, f"Tesseract found: {version_info}"
            else:
                return False, f"Tesseract found but returned error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "Tesseract command timed out"
        except Exception as e:
            return False, f"Error validating Tesseract: {str(e)}"


@dataclass
class OllamaConfig:
    """Configuration for Ollama local LLM."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:8b"  # Recommended for structured extraction
    vision_model: str = "llava:13b"  # For vision-based extraction
    temperature: float = 0.1  # Low temperature for deterministic extraction
    timeout: int = 120  # Seconds
    context_length: int = 4096  # Context window size
    
    @staticmethod
    def validate_connection(base_url: str = "http://localhost:11434") -> tuple[bool, str]:
        """Validate Ollama is running and accessible."""
        import requests
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "unknown") for m in models]
                return True, f"Ollama connected. Available models: {', '.join(model_names[:5])}"
            return False, f"Ollama returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, (
                "Cannot connect to Ollama. Ensure Ollama is running:\n"
                "• Start Ollama: ollama serve\n"
                "• Pull a model: ollama pull llama3.2:8b"
            )
        except Exception as e:
            return False, f"Error connecting to Ollama: {str(e)}"


@dataclass
class LMStudioConfig:
    """Configuration for LM Studio (OpenAI-compatible API)."""
    base_url: str = "http://localhost:1234/v1"
    model: str = "qwen3-vl-4b-instruct"  # Currently loaded model in LM Studio
    vision_model: str = "qwen3-vl-4b-instruct"  # Vision-capable model for direct image extraction
    api_key: str = "not-needed"  # LM Studio doesn't require API key
    temperature: float = 0.1
    timeout: int = 180  # Vision models need more time
    max_tokens: int = 4096  # Max tokens for response
    
    @staticmethod
    def validate_connection(base_url: str = "http://localhost:1234/v1") -> tuple[bool, str, list]:
        """Validate LM Studio is running and accessible.
        
        Returns:
            Tuple of (is_valid, message, loaded_models)
        """
        import requests
        try:
            response = requests.get(f"{base_url}/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m.get("id", "unknown") for m in data.get("data", [])]
                model_str = ", ".join(models) if models else "no models listed"
                return True, f"LM Studio connected. Loaded models: {model_str}", models
            return False, f"LM Studio returned status {response.status_code}", []
        except requests.exceptions.ConnectionError:
            return False, (
                "Cannot connect to LM Studio. Ensure LM Studio is running:\n"
                "• Open LM Studio application\n"
                "• Load a model\n"
                "• Start the local server (Developer tab)"
            ), []
        except Exception as e:
            return False, f"Error connecting to LM Studio: {str(e)}", []


@dataclass
class DeepseekConfig:
    """Configuration for Deepseek cloud API."""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    temperature: float = 0.1
    timeout: int = 60
    
    def validate_api_key(self) -> tuple[bool, str]:
        """Validate Deepseek API key is set."""
        if not self.api_key:
            return False, (
                "Deepseek API key not set.\n"
                "Set it via environment variable: DEEPSEEK_API_KEY=your_key\n"
                "Or enter it in the settings panel."
            )
        if len(self.api_key) < 20:
            return False, "Deepseek API key appears to be invalid (too short)"
        return True, "Deepseek API key is configured"


@dataclass
class AppConfig:
    """Main application configuration."""
    # LLM Settings
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    extraction_mode: ExtractionMode = ExtractionMode.OCR_LLM
    
    # Provider-specific configs
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    lm_studio: LMStudioConfig = field(default_factory=LMStudioConfig)
    deepseek: DeepseekConfig = field(default_factory=DeepseekConfig)
    tesseract: TesseractConfig = field(default_factory=TesseractConfig)
    
    # Processing settings
    max_file_size_mb: int = 5
    ocr_timeout_seconds: int = 30
    llm_timeout_seconds: int = 120
    
    # Excel export settings
    excel_date_format: str = "YYYY-MM-DD"
    excel_currency_format: str = 'R #,##0.00'
    
    # Validation settings
    validate_numeric_fields: bool = True
    vat_rate: float = 0.15  # South African VAT rate (15%)
    
    def get_active_llm_config(self) -> dict:
        """Get configuration for the currently selected LLM provider."""
        if self.llm_provider == LLMProvider.OLLAMA:
            return {
                "provider": "ollama",
                "base_url": self.ollama.base_url,
                "model": self.ollama.model,
                "temperature": self.ollama.temperature,
                "timeout": self.ollama.timeout,
            }
        elif self.llm_provider == LLMProvider.VISION_OLLAMA:
            return {
                "provider": "ollama_vision",
                "base_url": self.ollama.base_url,
                "model": self.ollama.vision_model,
                "temperature": self.ollama.temperature,
                "timeout": self.ollama.timeout,
            }
        elif self.llm_provider == LLMProvider.LM_STUDIO:
            return {
                "provider": "lm_studio",
                "base_url": self.lm_studio.base_url,
                "model": self.lm_studio.model,
                "api_key": self.lm_studio.api_key,
                "temperature": self.lm_studio.temperature,
                "timeout": self.lm_studio.timeout,
            }
        elif self.llm_provider == LLMProvider.DEEPSEEK:
            return {
                "provider": "deepseek",
                "base_url": self.deepseek.base_url,
                "model": self.deepseek.model,
                "api_key": self.deepseek.api_key,
                "temperature": self.deepseek.temperature,
                "timeout": self.deepseek.timeout,
            }
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")


def validate_system_requirements() -> dict:
    """
    Validate all system requirements on startup.
    
    Returns:
        Dictionary with validation results for each requirement.
    """
    import requests
    
    results = {}
    
    # Check Tesseract
    tesseract_valid, tesseract_msg = TesseractConfig.validate_installation()
    tesseract_path = TesseractConfig.find_tesseract()
    version = None
    if tesseract_path:
        try:
            result = subprocess.run(
                [tesseract_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
        except Exception:
            pass
    
    results["tesseract"] = {
        "installed": tesseract_valid,
        "message": tesseract_msg,
        "path": tesseract_path,
        "version": version,
    }
    
    # Check Ollama
    ollama_available = False
    ollama_models = []
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            ollama_available = True
            models_data = response.json().get("models", [])
            ollama_models = [m.get("name", "") for m in models_data]
    except Exception:
        pass
    
    results["ollama"] = {
        "available": ollama_available,
        "models": ollama_models,
    }
    
    # Check LM Studio
    lm_studio_available = False
    lm_studio_models = []
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            lm_studio_available = True
            models_data = response.json().get("data", [])
            lm_studio_models = [m.get("id", "") for m in models_data]
    except Exception:
        pass
    
    results["lm_studio"] = {
        "available": lm_studio_available,
        "models": lm_studio_models,
    }
    
    # Check Deepseek configuration
    config = get_config()
    deepseek_configured = bool(config.deepseek.api_key)
    
    results["deepseek"] = {
        "configured": deepseek_configured,
    }
    
    # Check Python dependencies
    try:
        import pytesseract
        import cv2
        import PIL
        import openpyxl
        results["python_deps"] = {
            "installed": True,
            "message": "All Python dependencies installed"
        }
    except ImportError as e:
        results["python_deps"] = {
            "installed": False,
            "message": f"Missing Python dependency: {e.name}"
        }
    
    return results


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
        # Auto-detect Tesseract path
        tesseract_path = TesseractConfig.find_tesseract()
        if tesseract_path:
            _config.tesseract.tesseract_cmd = tesseract_path
    return _config


def update_config(**kwargs) -> AppConfig:
    """Update configuration with new values."""
    global _config
    config = get_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
