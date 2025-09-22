"""
Advanced image quality detection algorithms for exposure and saturation analysis.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """Result structure for individual quality detection."""
    is_detected: bool
    score: float
    confidence: float
    threshold_used: float
    pixel_percentage: float


class ImageQualityDetector:
    """
    Comprehensive image quality detector for exposure and saturation issues.
    """
    
    def __init__(
        self,
        overexposure_threshold: float = 240.0,
        overexposure_percentage: float = 5.0,
        underexposure_threshold: float = 15.0,
        underexposure_percentage: float = 10.0,
        oversaturation_threshold: float = 0.9,
        oversaturation_percentage: float = 15.0,
        undersaturation_threshold: float = 0.2,
        undersaturation_percentage: float = 60.0
    ):
        """
        Initialize quality detector with configurable thresholds.
        
        Args:
            overexposure_threshold: Pixel intensity threshold for overexposure (0-255)
            overexposure_percentage: Percentage of pixels above threshold to flag as overexposed
            underexposure_threshold: Pixel intensity threshold for underexposure (0-255)
            underexposure_percentage: Percentage of pixels below threshold to flag as underexposed
            oversaturation_threshold: Saturation threshold for oversaturation (0-1)
            oversaturation_percentage: Percentage of pixels above threshold to flag as oversaturated
            undersaturation_threshold: Saturation threshold for undersaturation (0-1)
            undersaturation_percentage: Percentage of pixels below threshold to flag as undersaturated
        """
        self.overexposure_threshold = overexposure_threshold
        self.overexposure_percentage = overexposure_percentage
        self.underexposure_threshold = underexposure_threshold
        self.underexposure_percentage = underexposure_percentage
        self.oversaturation_threshold = oversaturation_threshold
        self.oversaturation_percentage = oversaturation_percentage
        self.undersaturation_threshold = undersaturation_threshold
        self.undersaturation_percentage = undersaturation_percentage
    
    def detect_overexposure(self, image: np.ndarray) -> QualityResult:
        """
        Detect overexposure using optimized luminance analysis.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            QualityResult: Overexposure detection results
        """
        try:
            # Optimized luminance conversion using vectorized operations
            if len(image.shape) == 3:
                # Use optimized RGB to luminance conversion (faster than cv2.cvtColor)
                # Y = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601)
                luminance = np.dot(image[...,:3], [0.114, 0.587, 0.299]).astype(np.uint8)
            else:
                luminance = image

            # Direct pixel counting (faster than histogram for threshold operations)
            total_pixels = luminance.size
            overexposed_pixels = np.count_nonzero(luminance >= self.overexposure_threshold)
            overexposed_percentage = (overexposed_pixels / total_pixels) * 100

            # Optimized confidence calculation
            confidence = min(1.0, abs(overexposed_percentage - self.overexposure_percentage) / max(self.overexposure_percentage, 1e-6) + 0.5)

            # Normalize score (0 = no overexposure, 1 = severe overexposure)
            score = min(1.0, overexposed_percentage / max(self.overexposure_percentage * 2, 1e-6))

            is_overexposed = overexposed_percentage > self.overexposure_percentage

            return QualityResult(
                is_detected=is_overexposed,
                score=score,
                confidence=confidence,
                threshold_used=self.overexposure_threshold,
                pixel_percentage=overexposed_percentage
            )

        except Exception as e:
            logger.error(f"Error in overexposure detection: {e}")
            return QualityResult(False, 0.0, 0.0, self.overexposure_threshold, 0.0)
    
    def detect_underexposure(self, image: np.ndarray) -> QualityResult:
        """
        Detect underexposure using optimized luminance analysis.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            QualityResult: Underexposure detection results
        """
        try:
            # Optimized luminance conversion using vectorized operations
            if len(image.shape) == 3:
                # Use optimized RGB to luminance conversion
                luminance = np.dot(image[...,:3], [0.114, 0.587, 0.299]).astype(np.uint8)
            else:
                luminance = image

            # Direct pixel counting (faster than histogram)
            total_pixels = luminance.size
            underexposed_pixels = np.count_nonzero(luminance <= self.underexposure_threshold)
            underexposed_percentage = (underexposed_pixels / total_pixels) * 100

            # Optimized confidence calculation
            confidence = min(1.0, abs(underexposed_percentage - self.underexposure_percentage) / max(self.underexposure_percentage, 1e-6) + 0.5)

            # Normalize score (0 = no underexposure, 1 = severe underexposure)
            score = min(1.0, underexposed_percentage / max(self.underexposure_percentage * 2, 1e-6))

            is_underexposed = underexposed_percentage > self.underexposure_percentage

            return QualityResult(
                is_detected=is_underexposed,
                score=score,
                confidence=confidence,
                threshold_used=self.underexposure_threshold,
                pixel_percentage=underexposed_percentage
            )

        except Exception as e:
            logger.error(f"Error in underexposure detection: {e}")
            return QualityResult(False, 0.0, 0.0, self.underexposure_threshold, 0.0)
    
    def detect_oversaturation(self, image: np.ndarray) -> QualityResult:
        """
        Detect oversaturation using optimized HSV analysis.

        Args:
            image: Input image (BGR format)

        Returns:
            QualityResult: Oversaturation detection results
        """
        try:
            # Skip saturation analysis for grayscale images
            if len(image.shape) != 3:
                return QualityResult(False, 0.0, 1.0, self.oversaturation_threshold, 0.0)

            # Optimized HSV conversion - only extract saturation channel
            # Use faster manual conversion for saturation-only analysis
            b, g, r = image[:,:,0].astype(np.float32), image[:,:,1].astype(np.float32), image[:,:,2].astype(np.float32)

            # Vectorized min/max operations for saturation calculation
            max_val = np.maximum(np.maximum(r, g), b)
            min_val = np.minimum(np.minimum(r, g), b)

            # Avoid division by zero
            saturation = np.divide(max_val - min_val, max_val, out=np.zeros_like(max_val), where=max_val!=0)

            # Direct threshold comparison (no normalization needed as already 0-1)
            total_pixels = saturation.size
            oversaturated_pixels = np.count_nonzero(saturation > self.oversaturation_threshold)
            oversaturated_percentage = (oversaturated_pixels / total_pixels) * 100

            # Optimized confidence calculation
            confidence = min(1.0, abs(oversaturated_percentage - self.oversaturation_percentage) / max(self.oversaturation_percentage, 1e-6) + 0.5)

            # Normalize score
            score = min(1.0, oversaturated_percentage / max(self.oversaturation_percentage * 2, 1e-6))

            is_oversaturated = oversaturated_percentage > self.oversaturation_percentage

            return QualityResult(
                is_detected=is_oversaturated,
                score=score,
                confidence=confidence,
                threshold_used=self.oversaturation_threshold,
                pixel_percentage=oversaturated_percentage
            )

        except Exception as e:
            logger.error(f"Error in oversaturation detection: {e}")
            return QualityResult(False, 0.0, 0.0, self.oversaturation_threshold, 0.0)
    
    def detect_undersaturation(self, image: np.ndarray) -> QualityResult:
        """
        Detect undersaturation using optimized HSV analysis.

        Args:
            image: Input image (BGR format)

        Returns:
            QualityResult: Undersaturation detection results
        """
        try:
            # Skip saturation analysis for grayscale images
            if len(image.shape) != 3:
                return QualityResult(False, 0.0, 1.0, self.undersaturation_threshold, 0.0)

            # Reuse saturation calculation from oversaturation (avoid duplicate computation)
            b, g, r = image[:,:,0].astype(np.float32), image[:,:,1].astype(np.float32), image[:,:,2].astype(np.float32)

            # Vectorized min/max operations for saturation calculation
            max_val = np.maximum(np.maximum(r, g), b)
            min_val = np.minimum(np.minimum(r, g), b)

            # Avoid division by zero
            saturation = np.divide(max_val - min_val, max_val, out=np.zeros_like(max_val), where=max_val!=0)

            # Direct threshold comparison
            total_pixels = saturation.size
            undersaturated_pixels = np.count_nonzero(saturation < self.undersaturation_threshold)
            undersaturated_percentage = (undersaturated_pixels / total_pixels) * 100

            # Optimized confidence calculation
            confidence = min(1.0, abs(undersaturated_percentage - self.undersaturation_percentage) / max(self.undersaturation_percentage, 1e-6) + 0.5)

            # Normalize score
            score = min(1.0, undersaturated_percentage / max(self.undersaturation_percentage * 2, 1e-6))

            is_undersaturated = undersaturated_percentage > self.undersaturation_percentage

            return QualityResult(
                is_detected=is_undersaturated,
                score=score,
                confidence=confidence,
                threshold_used=self.undersaturation_threshold,
                pixel_percentage=undersaturated_percentage
            )

        except Exception as e:
            logger.error(f"Error in undersaturation detection: {e}")
            return QualityResult(False, 0.0, 0.0, self.undersaturation_threshold, 0.0)
    
    def _calculate_saturation_optimized(self, image: np.ndarray) -> np.ndarray:
        """
        Optimized saturation calculation to avoid duplicate computation.

        Args:
            image: Input image (BGR format)

        Returns:
            Saturation array (0-1 range)
        """
        if len(image.shape) != 3:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        b, g, r = image[:,:,0].astype(np.float32), image[:,:,1].astype(np.float32), image[:,:,2].astype(np.float32)

        # Vectorized min/max operations
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)

        # Avoid division by zero
        saturation = np.divide(max_val - min_val, max_val, out=np.zeros_like(max_val), where=max_val!=0)
        return saturation

    def analyze_comprehensive_quality(self, image: np.ndarray) -> Dict[str, QualityResult]:
        """
        Perform optimized comprehensive quality analysis.

        Args:
            image: Input image (BGR format)

        Returns:
            Dict containing all quality analysis results
        """
        # Pre-calculate luminance and saturation to avoid duplicate computation
        if len(image.shape) == 3:
            luminance = np.dot(image[...,:3], [0.114, 0.587, 0.299]).astype(np.uint8)
            saturation = self._calculate_saturation_optimized(image)
        else:
            luminance = image
            saturation = np.zeros_like(image, dtype=np.float32)

        total_pixels = luminance.size

        # Optimized exposure detection using pre-calculated luminance
        overexposed_pixels = np.count_nonzero(luminance >= self.overexposure_threshold)
        overexposed_percentage = (overexposed_pixels / total_pixels) * 100

        underexposed_pixels = np.count_nonzero(luminance <= self.underexposure_threshold)
        underexposed_percentage = (underexposed_pixels / total_pixels) * 100

        # Optimized saturation detection using pre-calculated saturation
        oversaturated_pixels = np.count_nonzero(saturation > self.oversaturation_threshold)
        oversaturated_percentage = (oversaturated_pixels / total_pixels) * 100

        undersaturated_pixels = np.count_nonzero(saturation < self.undersaturation_threshold)
        undersaturated_percentage = (undersaturated_pixels / total_pixels) * 100

        # Build results efficiently
        results = {
            'overexposure': QualityResult(
                is_detected=overexposed_percentage > self.overexposure_percentage,
                score=min(1.0, overexposed_percentage / max(self.overexposure_percentage * 2, 1e-6)),
                confidence=min(1.0, abs(overexposed_percentage - self.overexposure_percentage) / max(self.overexposure_percentage, 1e-6) + 0.5),
                threshold_used=self.overexposure_threshold,
                pixel_percentage=overexposed_percentage
            ),
            'underexposure': QualityResult(
                is_detected=underexposed_percentage > self.underexposure_percentage,
                score=min(1.0, underexposed_percentage / max(self.underexposure_percentage * 2, 1e-6)),
                confidence=min(1.0, abs(underexposed_percentage - self.underexposure_percentage) / max(self.underexposure_percentage, 1e-6) + 0.5),
                threshold_used=self.underexposure_threshold,
                pixel_percentage=underexposed_percentage
            ),
            'oversaturation': QualityResult(
                is_detected=oversaturated_percentage > self.oversaturation_percentage,
                score=min(1.0, oversaturated_percentage / max(self.oversaturation_percentage * 2, 1e-6)),
                confidence=min(1.0, abs(oversaturated_percentage - self.oversaturation_percentage) / max(self.oversaturation_percentage, 1e-6) + 0.5),
                threshold_used=self.oversaturation_threshold,
                pixel_percentage=oversaturated_percentage
            ),
            'undersaturation': QualityResult(
                is_detected=undersaturated_percentage > self.undersaturation_percentage,
                score=min(1.0, undersaturated_percentage / max(self.undersaturation_percentage * 2, 1e-6)),
                confidence=min(1.0, abs(undersaturated_percentage - self.undersaturation_percentage) / max(self.undersaturation_percentage, 1e-6) + 0.5),
                threshold_used=self.undersaturation_threshold,
                pixel_percentage=undersaturated_percentage
            )
        }

        return results
