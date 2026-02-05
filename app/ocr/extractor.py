"""
OCR extraction module using Tesseract.

Handles text extraction from images and PDFs with:
- Tesseract OCR integration
- PDF to image conversion
- Multi-page document support
- Configurable OCR settings
"""

import logging
import tempfile
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import pytesseract
from PIL import Image

from app.config import TesseractConfig, get_config
from app.ocr.preprocessor import ImagePreprocessor, PreprocessingLevel, PreprocessingResult

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result of OCR text extraction."""
    text: str
    page_count: int
    processing_time_seconds: float
    confidence: float  # Average confidence if available
    preprocessing_applied: List[str]
    errors: List[str]
    warnings: List[str]


class OCRExtractor:
    """
    Extracts text from images and PDFs using Tesseract OCR.
    
    Supports:
    - Single images (PNG, JPG, JPEG, TIFF, BMP, GIF, WEBP)
    - Multi-page PDFs (converted to images)
    - Configurable preprocessing
    - Confidence scoring
    """
    
    SUPPORTED_IMAGE_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".jpe", ".jfif",  # Common formats
        ".tiff", ".tif", ".bmp", ".gif", ".webp",   # Other formats
        ".ppm", ".pgm", ".pbm",                      # Netpbm formats
    }
    SUPPORTED_PDF_EXTENSIONS = {".pdf"}
    
    def __init__(
        self,
        tesseract_config: Optional[TesseractConfig] = None,
        preprocessing_level: PreprocessingLevel = PreprocessingLevel.STANDARD,
    ):
        """
        Initialize the OCR extractor.
        
        Args:
            tesseract_config: Tesseract configuration (uses default if not provided)
            preprocessing_level: Level of image preprocessing to apply
        """
        self.config = tesseract_config or get_config().tesseract
        self.preprocessor = ImagePreprocessor(level=preprocessing_level)
        self.preprocessing_level = preprocessing_level
        
        # Set Tesseract command path if configured
        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd
    
    def extract(
        self,
        file_input: Union[str, Path, bytes],
        file_extension: Optional[str] = None,
    ) -> OCRResult:
        """
        Extract text from an image or PDF file.
        
        Args:
            file_input: File path, bytes content, or file-like object
            file_extension: File extension (required if input is bytes)
            
        Returns:
            OCRResult with extracted text and metadata
        """
        start_time = time.time()
        errors = []
        warnings = []
        preprocessing_messages = []
        
        # Determine file type
        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            extension = path.suffix.lower()
            is_file_path = True
        else:
            if not file_extension:
                raise ValueError("file_extension is required when input is bytes")
            extension = file_extension.lower()
            if not extension.startswith("."):
                extension = f".{extension}"
            is_file_path = False
        
        # Process based on file type
        if extension in self.SUPPORTED_PDF_EXTENSIONS:
            text, page_count, preprocessing_messages = self._extract_from_pdf(
                file_input, is_file_path, errors, warnings
            )
        elif extension in self.SUPPORTED_IMAGE_EXTENSIONS:
            text, preprocessing_messages = self._extract_from_image(
                file_input, is_file_path, errors, warnings
            )
            page_count = 1
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        
        # Calculate confidence (using Tesseract's data output)
        confidence = self._calculate_confidence(text)
        
        processing_time = time.time() - start_time
        
        return OCRResult(
            text=text,
            page_count=page_count,
            processing_time_seconds=processing_time,
            confidence=confidence,
            preprocessing_applied=preprocessing_messages,
            errors=errors,
            warnings=warnings,
        )
    
    def _extract_from_image(
        self,
        image_input: Union[str, Path, bytes],
        is_file_path: bool,
        errors: List[str],
        warnings: List[str],
    ) -> tuple[str, List[str]]:
        """Extract text from a single image."""
        preprocessing_messages = []
        
        try:
            # Load image
            if is_file_path:
                image = Image.open(str(image_input))
            else:
                image = Image.open(BytesIO(image_input))
            
            # Force load the image data (some formats are lazy-loaded)
            image.load()
            
            # Ensure image is in a compatible mode for processing
            if image.mode not in ('RGB', 'L', '1'):
                if image.mode == 'RGBA':
                    # Create white background for transparent images
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])
                    image = background
                elif image.mode == 'P':
                    # Palette mode - convert to RGB
                    image = image.convert('RGB')
                else:
                    # Other modes - convert to RGB
                    image = image.convert('RGB')
            
            # Preprocess image
            result: PreprocessingResult = self.preprocessor.preprocess(image)
            preprocessing_messages = result.messages
            processed_image = result.image
            
            # Ensure the processed image is in a pytesseract-compatible format
            # pytesseract works best with PIL Images in RGB, L, or 1 mode
            if hasattr(processed_image, 'mode') and processed_image.mode not in ('RGB', 'L', '1'):
                processed_image = processed_image.convert('RGB')
            
            # Convert PIL Image to numpy array for pytesseract
            # This ensures maximum compatibility with pytesseract
            if isinstance(processed_image, Image.Image):
                # Force load the image data
                processed_image.load()
                # Convert to numpy array
                img_array = np.array(processed_image)
                logger.debug(f"Image array shape: {img_array.shape}, dtype: {img_array.dtype}")
            else:
                img_array = processed_image
            
            # Perform OCR
            custom_config = self.config.get_config_string()
            text = pytesseract.image_to_string(
                img_array,
                lang=self.config.language,
                config=custom_config,
            )
            
            # Clean up text
            text = self._clean_text(text)
            
            if not text.strip():
                warnings.append("OCR produced empty result - image may be blank or unreadable")
            
            return text, preprocessing_messages
            
        except pytesseract.TesseractNotFoundError as e:
            errors.append(f"Tesseract not found: {str(e)}")
            return "", preprocessing_messages
        except Exception as e:
            errors.append(f"OCR extraction error: {str(e)}")
            logger.exception("OCR extraction failed")
            return "", preprocessing_messages
    
    def _extract_from_pdf(
        self,
        pdf_input: Union[str, Path, bytes],
        is_file_path: bool,
        errors: List[str],
        warnings: List[str],
    ) -> tuple[str, int, List[str]]:
        """Extract text from a PDF by converting pages to images."""
        preprocessing_messages = []
        
        try:
            # Import pdf2image (optional dependency)
            try:
                from pdf2image import convert_from_path, convert_from_bytes
            except ImportError:
                errors.append(
                    "pdf2image not installed. Install with: pip install pdf2image\n"
                    "Also requires poppler: brew install poppler (macOS) or apt install poppler-utils (Linux)"
                )
                return "", 0, preprocessing_messages
            
            # Convert PDF to images
            if is_file_path:
                images = convert_from_path(
                    str(pdf_input),
                    dpi=300,
                    fmt="png",
                )
            else:
                images = convert_from_bytes(
                    pdf_input,
                    dpi=300,
                    fmt="png",
                )
            
            page_count = len(images)
            if page_count == 0:
                warnings.append("PDF contains no pages")
                return "", 0, preprocessing_messages
            
            # Extract text from each page
            all_text = []
            for i, page_image in enumerate(images):
                page_text, page_messages = self._extract_from_image(
                    page_image, is_file_path=False, errors=errors, warnings=warnings
                )
                
                if i == 0:
                    preprocessing_messages = page_messages
                
                if page_text.strip():
                    all_text.append(f"--- Page {i + 1} ---\n{page_text}")
            
            combined_text = "\n\n".join(all_text)
            
            return combined_text, page_count, preprocessing_messages
            
        except Exception as e:
            errors.append(f"PDF extraction error: {str(e)}")
            logger.exception("PDF extraction failed")
            return "", 0, preprocessing_messages
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace while preserving structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove leading/trailing whitespace from each line
            cleaned_line = line.strip()
            # Collapse multiple spaces into one
            cleaned_line = ' '.join(cleaned_line.split())
            cleaned_lines.append(cleaned_line)
        
        # Remove excessive blank lines (keep max 2 consecutive)
        result_lines = []
        blank_count = 0
        for line in cleaned_lines:
            if not line:
                blank_count += 1
                if blank_count <= 2:
                    result_lines.append(line)
            else:
                blank_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _calculate_confidence(self, text: str) -> float:
        """
        Estimate extraction confidence based on text characteristics.
        
        This is a heuristic approach since pytesseract.image_to_data
        confidence is per-word and computationally expensive.
        """
        if not text.strip():
            return 0.0
        
        # Count recognizable patterns
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        
        # Count alphanumeric characters (good indicator of successful OCR)
        alphanumeric = sum(1 for c in text if c.isalnum())
        
        # Count typical invoice keywords as confidence boosters
        invoice_keywords = [
            "invoice", "tax", "vat", "date", "total", "amount",
            "qty", "quantity", "price", "subtotal", "account",
        ]
        keyword_matches = sum(1 for kw in invoice_keywords if kw in text.lower())
        
        # Base confidence on alphanumeric ratio
        base_confidence = min(alphanumeric / total_chars, 1.0) * 0.7
        
        # Bonus for invoice keywords (up to 0.3)
        keyword_bonus = min(keyword_matches * 0.03, 0.3)
        
        return min(base_confidence + keyword_bonus, 1.0)
    
    def extract_with_layout(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
    ) -> dict:
        """
        Extract text with bounding box information for layout analysis.
        
        Useful for table structure detection.
        
        Returns:
            Dictionary with text blocks and their positions
        """
        # Load image
        if isinstance(image_input, Image.Image):
            image = image_input
        elif isinstance(image_input, bytes):
            image = Image.open(BytesIO(image_input))
        else:
            image = Image.open(str(image_input))
        
        # Preprocess
        result = self.preprocessor.preprocess(image)
        processed_image = result.image
        
        # Get detailed OCR data
        custom_config = self.config.get_config_string()
        data = pytesseract.image_to_data(
            processed_image,
            lang=self.config.language,
            config=custom_config,
            output_type=pytesseract.Output.DICT,
        )
        
        # Organize into blocks
        blocks = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if text:
                blocks.append({
                    'text': text,
                    'confidence': data['conf'][i],
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'block_num': data['block_num'][i],
                    'line_num': data['line_num'][i],
                    'word_num': data['word_num'][i],
                })
        
        return {
            'blocks': blocks,
            'image_size': result.processed_size,
            'preprocessing': result.messages,
        }


def extract_text(
    file_path: Union[str, Path],
    preprocessing_level: PreprocessingLevel = PreprocessingLevel.STANDARD,
) -> OCRResult:
    """
    Convenience function to extract text from a file.
    
    Args:
        file_path: Path to image or PDF file
        preprocessing_level: Level of preprocessing to apply
        
    Returns:
        OCRResult with extracted text
    """
    extractor = OCRExtractor(preprocessing_level=preprocessing_level)
    return extractor.extract(file_path)
