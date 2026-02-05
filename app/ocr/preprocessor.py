"""
Image preprocessing module for OCR optimization.

Provides image enhancement techniques to improve OCR accuracy:
- Grayscale conversion
- Noise reduction
- Contrast enhancement
- Deskewing (rotation correction)
- Adaptive thresholding
"""

import logging
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class PreprocessingLevel(Enum):
    """Level of preprocessing to apply."""
    NONE = "none"          # No preprocessing, use original image
    LIGHT = "light"        # Basic grayscale and minor enhancement
    STANDARD = "standard"  # Recommended: grayscale, denoise, threshold
    AGGRESSIVE = "aggressive"  # Full preprocessing including deskew


@dataclass
class PreprocessingResult:
    """Result of image preprocessing."""
    image: Image.Image
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    preprocessing_level: PreprocessingLevel
    was_deskewed: bool = False
    rotation_angle: float = 0.0
    messages: list[str] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []


class ImagePreprocessor:
    """
    Preprocesses images for optimal OCR results.
    
    Applies various image enhancement techniques based on the
    configured preprocessing level. Higher levels may improve
    OCR accuracy but take longer to process.
    """
    
    def __init__(
        self,
        level: PreprocessingLevel = PreprocessingLevel.STANDARD,
        target_dpi: int = 300,
        max_dimension: int = 4000,
    ):
        """
        Initialize the image preprocessor.
        
        Args:
            level: Preprocessing intensity level
            target_dpi: Target DPI for image scaling (300 recommended for OCR)
            max_dimension: Maximum image dimension to prevent memory issues
        """
        self.level = level
        self.target_dpi = target_dpi
        self.max_dimension = max_dimension
    
    def preprocess(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
    ) -> PreprocessingResult:
        """
        Preprocess an image for OCR.
        
        Args:
            image_input: Image as file path, bytes, PIL Image, or numpy array
            
        Returns:
            PreprocessingResult with processed image and metadata
        """
        # Load image
        cv_image, original_pil = self._load_image(image_input)
        original_size = (cv_image.shape[1], cv_image.shape[0])
        messages = []
        
        if self.level == PreprocessingLevel.NONE:
            # Even with no preprocessing, ensure image is in a compatible mode
            # Convert to RGB if necessary for pytesseract compatibility
            if original_pil.mode not in ('RGB', 'L', '1'):
                if original_pil.mode == 'RGBA':
                    background = Image.new('RGB', original_pil.size, (255, 255, 255))
                    background.paste(original_pil, mask=original_pil.split()[3])
                    original_pil = background
                else:
                    original_pil = original_pil.convert('RGB')
                messages.append("Converted image mode for compatibility")
            else:
                messages.append("No preprocessing applied")
            
            return PreprocessingResult(
                image=original_pil,
                original_size=original_size,
                processed_size=original_size,
                preprocessing_level=self.level,
                messages=messages
            )
        
        # Resize if too large
        cv_image = self._resize_if_needed(cv_image, messages)
        
        # Convert to grayscale
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            messages.append("Converted to grayscale")
        else:
            gray = cv_image
        
        # Apply preprocessing based on level
        was_deskewed = False
        rotation_angle = 0.0
        
        if self.level in [PreprocessingLevel.LIGHT, PreprocessingLevel.STANDARD, PreprocessingLevel.AGGRESSIVE]:
            # Light enhancement
            gray = self._enhance_contrast(gray)
            messages.append("Applied contrast enhancement")
        
        if self.level in [PreprocessingLevel.STANDARD, PreprocessingLevel.AGGRESSIVE]:
            # Denoise
            gray = self._remove_noise(gray)
            messages.append("Applied noise reduction")
            
            # Adaptive thresholding for document images
            gray = self._apply_threshold(gray)
            messages.append("Applied adaptive thresholding")
        
        if self.level == PreprocessingLevel.AGGRESSIVE:
            # Deskew
            gray, rotation_angle, was_deskewed = self._deskew(gray)
            if was_deskewed:
                messages.append(f"Deskewed by {rotation_angle:.2f}°")
        
        # Convert back to PIL Image
        processed_pil = Image.fromarray(gray)
        processed_size = (gray.shape[1], gray.shape[0])
        
        return PreprocessingResult(
            image=processed_pil,
            original_size=original_size,
            processed_size=processed_size,
            preprocessing_level=self.level,
            was_deskewed=was_deskewed,
            rotation_angle=rotation_angle,
            messages=messages,
        )
    
    def _load_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
    ) -> tuple[np.ndarray, Image.Image]:
        """Load image from various input types."""
        if isinstance(image_input, np.ndarray):
            cv_image = image_input
            if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(cv_image)
        elif isinstance(image_input, Image.Image):
            pil_image = image_input
            # Convert to RGB if necessary (handles RGBA, P, L, etc.)
            if pil_image.mode not in ('RGB', 'L'):
                if pil_image.mode == 'RGBA':
                    # Create white background and paste image
                    background = Image.new('RGB', pil_image.size, (255, 255, 255))
                    background.paste(pil_image, mask=pil_image.split()[3])
                    pil_image = background
                else:
                    pil_image = pil_image.convert('RGB')
            
            if pil_image.mode == 'L':
                # Grayscale image
                cv_image = np.array(pil_image)
            else:
                cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, bytes):
            pil_image = Image.open(BytesIO(image_input))
            # Convert to RGB if necessary
            if pil_image.mode not in ('RGB', 'L'):
                if pil_image.mode == 'RGBA':
                    background = Image.new('RGB', pil_image.size, (255, 255, 255))
                    background.paste(pil_image, mask=pil_image.split()[3])
                    pil_image = background
                else:
                    pil_image = pil_image.convert('RGB')
            
            if pil_image.mode == 'L':
                cv_image = np.array(pil_image)
            else:
                cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, (str, Path)):
            path = str(image_input)
            cv_image = cv2.imread(path)
            if cv_image is None:
                raise ValueError(f"Failed to load image from: {path}")
            pil_image = Image.open(path)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")
        
        return cv_image, pil_image
    
    def _resize_if_needed(self, image: np.ndarray, messages: list) -> np.ndarray:
        """Resize image if it exceeds maximum dimensions."""
        height, width = image.shape[:2]
        
        if max(height, width) > self.max_dimension:
            scale = self.max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            messages.append(f"Resized from {width}x{height} to {new_width}x{new_height}")
        
        return image
    
    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    def _remove_noise(self, gray: np.ndarray) -> np.ndarray:
        """Remove noise using bilateral filter (preserves edges)."""
        return cv2.bilateralFilter(gray, 9, 75, 75)
    
    def _apply_threshold(self, gray: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for document images."""
        # Use Otsu's thresholding for automatic threshold selection
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary
    
    def _deskew(self, image: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """
        Detect and correct image skew.
        
        Returns:
            Tuple of (deskewed_image, rotation_angle, was_corrected)
        """
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=100, maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            return image, 0.0, False
        
        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # Avoid division by zero
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider near-horizontal lines
                if abs(angle) < 45:
                    angles.append(angle)
        
        if not angles:
            return image, 0.0, False
        
        # Use median angle to be robust against outliers
        median_angle = np.median(angles)
        
        # Only correct if skew is significant but not too extreme
        if abs(median_angle) < 0.5 or abs(median_angle) > 15:
            return image, median_angle, False
        
        # Rotate image to correct skew
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Calculate new image bounds
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust rotation matrix for new bounds
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation with white background
        deskewed = cv2.warpAffine(
            image, rotation_matrix, (new_width, new_height),
            borderMode=cv2.BORDER_CONSTANT, borderValue=255
        )
        
        return deskewed, median_angle, True
    
    def preprocess_for_table_extraction(self, image: np.ndarray) -> np.ndarray:
        """
        Special preprocessing for images with tables.
        
        Enhances line detection for table structure extraction.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply morphological operations to enhance table lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Detect horizontal lines
        horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_h)
        
        # Detect vertical lines
        vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_v)
        
        # Combine
        table_mask = cv2.add(horizontal, vertical)
        
        return table_mask


def preprocess_image(
    image_input: Union[str, Path, bytes, Image.Image],
    level: PreprocessingLevel = PreprocessingLevel.STANDARD,
) -> PreprocessingResult:
    """
    Convenience function to preprocess an image.
    
    Args:
        image_input: Image as file path, bytes, or PIL Image
        level: Preprocessing intensity level
        
    Returns:
        PreprocessingResult with processed image
    """
    preprocessor = ImagePreprocessor(level=level)
    return preprocessor.preprocess(image_input)
