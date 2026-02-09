"""
Multi-provider LLM client for invoice extraction.

Supports:
- Ollama (local)
- LM Studio (local)
- Deepseek (cloud)

With automatic fallback and retry logic.
"""

import base64
import json
import logging
import time
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union

import requests
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import (
    AppConfig,
    DeepseekConfig,
    LLMProvider,
    LMStudioConfig,
    OllamaConfig,
    get_config,
)
from app.llm.prompts import get_extraction_prompt, get_vision_prompt

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMConnectionError(LLMClientError):
    """Error connecting to LLM service."""
    pass


class LLMResponseError(LLMClientError):
    """Error in LLM response."""
    pass


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def extract_from_text(self, ocr_text: str) -> str:
        """Extract invoice data from OCR text."""
        pass
    
    @abstractmethod
    def extract_from_image(self, image: Union[str, Path, bytes, Image.Image]) -> str:
        """Extract invoice data directly from image using vision model."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name."""
        pass
    
    def _prepare_image(self, image: Union[str, Path, bytes, Image.Image], max_size: int = 1536) -> tuple[str, str]:
        """
        Prepare image for LLM vision API: validate, resize, convert, encode.
        
        Returns:
            Tuple of (base64_string, mime_type) e.g. ("abc...", "image/jpeg")
        """
        # Step 1: Load into PIL Image
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image)
        elif isinstance(image, bytes):
            pil_img = Image.open(BytesIO(image))
        elif isinstance(image, Image.Image):
            pil_img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Step 2: Handle EXIF orientation
        try:
            from PIL import ImageOps
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception:
            pass  # Not all images have EXIF data
        
        # Step 3: Convert to RGB (handles CMYK, RGBA, palette modes)
        if pil_img.mode not in ("RGB", "L"):
            pil_img = pil_img.convert("RGB")
        elif pil_img.mode == "L":
            pil_img = pil_img.convert("RGB")
        
        # Step 4: Resize if too large (preserve aspect ratio)
        w, h = pil_img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h}")
        
        # Step 5: Save as PNG (universally supported, lossless)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG", optimize=True)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        logger.info(f"Prepared image: {pil_img.size[0]}x{pil_img.size[1]}, PNG, {len(buffer.getvalue()):,} bytes")
        return b64, "image/png"
    
    def _encode_image_base64(self, image: Union[str, Path, bytes, Image.Image]) -> str:
        """Legacy method - returns just the base64 string."""
        b64, _ = self._prepare_image(image)
        return b64


class OllamaClient(BaseLLMClient):
    """Client for Ollama local LLM."""
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        """Initialize Ollama client."""
        self.config = config or get_config().ollama
        self.base_url = self.config.base_url.rstrip("/")
    
    def get_provider_name(self) -> str:
        return "Ollama"
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Check if our configured model is available
                return any(
                    self.config.model in name or name in self.config.model
                    for name in model_names
                )
            return False
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(LLMConnectionError),
    )
    def extract_from_text(self, ocr_text: str) -> str:
        """Extract invoice data using Ollama."""
        prompt = get_extraction_prompt(ocr_text)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_ctx": self.config.context_length,
                    },
                },
                timeout=self.config.timeout,
            )
            
            if response.status_code != 200:
                raise LLMResponseError(f"Ollama returned status {response.status_code}: {response.text}")
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(f"Failed to connect to Ollama: {e}")
        except requests.exceptions.Timeout:
            raise LLMConnectionError("Ollama request timed out")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(LLMConnectionError),
    )
    def extract_from_image(self, image: Union[str, Path, bytes, Image.Image]) -> str:
        """Extract invoice data using Ollama vision model."""
        if not self.config.vision_model:
            raise LLMClientError("No vision model configured for Ollama")
        
        image_base64 = self._encode_image_base64(image)
        prompt = get_vision_prompt()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config.vision_model,
                    "prompt": prompt,
                    "images": [image_base64],
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_ctx": self.config.context_length,
                    },
                },
                timeout=self.config.timeout * 2,  # Vision takes longer
            )
            
            if response.status_code != 200:
                raise LLMResponseError(f"Ollama returned status {response.status_code}: {response.text}")
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(f"Failed to connect to Ollama: {e}")
        except requests.exceptions.Timeout:
            raise LLMConnectionError("Ollama vision request timed out")


class LMStudioClient(BaseLLMClient):
    """Client for LM Studio local server."""
    
    def __init__(self, config: Optional[LMStudioConfig] = None):
        """Initialize LM Studio client."""
        self.config = config or get_config().lm_studio
        self.base_url = self.config.base_url.rstrip("/")
    
    def get_provider_name(self) -> str:
        return "LM Studio"
    
    def is_available(self) -> bool:
        """Check if LM Studio server is running."""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                timeout=5,
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"LM Studio not available: {e}")
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(LLMConnectionError),
    )
    def extract_from_text(self, ocr_text: str) -> str:
        """Extract invoice data using LM Studio."""
        prompt = get_extraction_prompt(ocr_text)
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.config.model or "local-model",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
                timeout=self.config.timeout,
            )
            
            if response.status_code != 200:
                raise LLMResponseError(f"LM Studio returned status {response.status_code}: {response.text}")
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(f"Failed to connect to LM Studio: {e}")
        except requests.exceptions.Timeout:
            raise LLMConnectionError("LM Studio request timed out")
    
    def extract_from_image(self, image: Union[str, Path, bytes, Image.Image]) -> str:
        """Extract invoice data using LM Studio vision model."""
        # Use vision_model if set, otherwise fall back to main model
        vision_model = getattr(self.config, 'vision_model', None) or self.config.model
        if not vision_model:
            raise LLMClientError("No vision model configured for LM Studio")
        
        image_base64, mime_type = self._prepare_image(image)
        prompt = get_vision_prompt()
        
        logger.info(f"Sending image to LM Studio vision model: {vision_model} ({mime_type})")
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
                timeout=self.config.timeout * 2,
            )
            
            if response.status_code != 200:
                raise LLMResponseError(f"LM Studio returned status {response.status_code}: {response.text}")
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(f"Failed to connect to LM Studio: {e}")
        except requests.exceptions.Timeout:
            raise LLMConnectionError("LM Studio vision request timed out")


class DeepseekClient(BaseLLMClient):
    """Client for Deepseek cloud API."""
    
    def __init__(self, config: Optional[DeepseekConfig] = None):
        """Initialize Deepseek client."""
        self.config = config or get_config().deepseek
        self.base_url = self.config.base_url.rstrip("/")
    
    def get_provider_name(self) -> str:
        return "Deepseek"
    
    def is_available(self) -> bool:
        """Check if Deepseek API key is configured."""
        return bool(self.config.api_key)
    
    def _get_headers(self) -> dict:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(LLMConnectionError),
    )
    def extract_from_text(self, ocr_text: str) -> str:
        """Extract invoice data using Deepseek."""
        prompt = get_extraction_prompt(ocr_text)
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json={
                    "model": self.config.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
                timeout=self.config.timeout,
            )
            
            if response.status_code == 401:
                raise LLMClientError("Invalid Deepseek API key")
            elif response.status_code == 429:
                raise LLMConnectionError("Deepseek rate limit exceeded")
            elif response.status_code != 200:
                raise LLMResponseError(f"Deepseek returned status {response.status_code}: {response.text}")
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(f"Failed to connect to Deepseek: {e}")
        except requests.exceptions.Timeout:
            raise LLMConnectionError("Deepseek request timed out")
    
    def extract_from_image(self, image: Union[str, Path, bytes, Image.Image]) -> str:
        """
        Extract invoice data from image using Deepseek.
        
        Note: Deepseek vision support depends on the API version.
        Falls back to text extraction if vision not available.
        """
        if not self.config.vision_model:
            raise LLMClientError("Vision extraction not available for Deepseek")
        
        image_base64, mime_type = self._prepare_image(image)
        prompt = get_vision_prompt()
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json={
                    "model": self.config.vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
                timeout=self.config.timeout * 2,
            )
            
            if response.status_code != 200:
                raise LLMResponseError(f"Deepseek returned status {response.status_code}: {response.text}")
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(f"Failed to connect to Deepseek: {e}")
        except requests.exceptions.Timeout:
            raise LLMConnectionError("Deepseek vision request timed out")


class LLMClient:
    """
    Unified LLM client with automatic provider selection and fallback.
    
    Tries providers in order of preference until one succeeds.
    Default order: Ollama -> LM Studio -> Deepseek
    """
    
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        preferred_provider: Optional[LLMProvider] = None,
    ):
        """
        Initialize the unified LLM client.
        
        Args:
            config: Application configuration
            preferred_provider: Preferred provider to try first
        """
        self.config = config or get_config()
        self.preferred_provider = preferred_provider or self.config.default_llm_provider
        
        # Initialize clients
        self.clients: dict[LLMProvider, BaseLLMClient] = {
            LLMProvider.OLLAMA: OllamaClient(self.config.ollama),
            LLMProvider.LM_STUDIO: LMStudioClient(self.config.lm_studio),
            LLMProvider.DEEPSEEK: DeepseekClient(self.config.deepseek),
        }
        
        self._last_used_provider: Optional[LLMProvider] = None
    
    def _get_provider_order(self) -> list[LLMProvider]:
        """Get providers in order of preference."""
        order = [self.preferred_provider]
        for provider in LLMProvider:
            if provider not in order and provider in self.clients:
                order.append(provider)
        # Filter to only providers that have client implementations
        return [p for p in order if p in self.clients]
    
    def get_available_providers(self) -> list[LLMProvider]:
        """Get list of currently available providers."""
        available = []
        for provider, client in self.clients.items():
            if client.is_available():
                available.append(provider)
        return available
    
    @property
    def last_used_provider(self) -> Optional[LLMProvider]:
        """Get the last provider that was successfully used."""
        return self._last_used_provider
    
    def extract_from_text(
        self,
        ocr_text: str,
        provider: Optional[LLMProvider] = None,
    ) -> tuple[str, LLMProvider]:
        """
        Extract invoice data from OCR text.
        
        Args:
            ocr_text: OCR-extracted text from invoice
            provider: Specific provider to use (optional)
            
        Returns:
            Tuple of (extracted JSON string, provider used)
            
        Raises:
            LLMClientError: If all providers fail
        """
        if provider:
            providers = [provider]
        else:
            providers = self._get_provider_order()
        
        errors = []
        for prov in providers:
            client = self.clients[prov]
            
            if not client.is_available():
                logger.info(f"{prov.value} not available, skipping")
                continue
            
            try:
                logger.info(f"Attempting extraction with {prov.value}")
                result = client.extract_from_text(ocr_text)
                self._last_used_provider = prov
                return result, prov
            except Exception as e:
                logger.warning(f"{prov.value} failed: {e}")
                errors.append(f"{prov.value}: {str(e)}")
        
        raise LLMClientError(f"All providers failed: {'; '.join(errors)}")
    
    def extract_from_image(
        self,
        image: Union[str, Path, bytes, Image.Image],
        provider: Optional[LLMProvider] = None,
    ) -> tuple[str, LLMProvider]:
        """
        Extract invoice data directly from image using vision model.
        
        Args:
            image: Invoice image
            provider: Specific provider to use (optional)
            
        Returns:
            Tuple of (extracted JSON string, provider used)
            
        Raises:
            LLMClientError: If all providers fail
        """
        if provider:
            providers = [provider]
        else:
            providers = self._get_provider_order()
        
        errors = []
        for prov in providers:
            client = self.clients[prov]
            
            if not client.is_available():
                logger.info(f"{prov.value} not available for vision, skipping")
                continue
            
            try:
                logger.info(f"Attempting vision extraction with {prov.value}")
                result = client.extract_from_image(image)
                self._last_used_provider = prov
                return result, prov
            except LLMClientError as e:
                if "vision" in str(e).lower():
                    logger.info(f"{prov.value} does not support vision")
                    continue
                raise
            except Exception as e:
                logger.warning(f"{prov.value} vision failed: {e}")
                errors.append(f"{prov.value}: {str(e)}")
        
        raise LLMClientError(f"Vision extraction failed with all providers: {'; '.join(errors)}")


def create_client(
    provider: Optional[LLMProvider] = None,
    config: Optional[AppConfig] = None,
) -> LLMClient:
    """
    Create an LLM client.
    
    Args:
        provider: Preferred provider
        config: Application configuration
        
    Returns:
        Configured LLMClient instance
    """
    return LLMClient(config=config, preferred_provider=provider)
