"""
Blur Type Classification Module

This module provides functionality to classify different types of blur in images:
- Motion blur (caused by camera or subject movement)
- Gaussian blur (out-of-focus blur)
- Defocus blur (depth-of-field related)
- Mixed blur (combination of multiple blur types)

The classifier works in conjunction with the existing BlurDetector to provide
detailed blur analysis.
"""

import cv2
import numpy as np
from enum import Enum
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BlurType(Enum):
    """Enumeration of different blur types."""
    SHARP = "sharp"
    MOTION_BLUR = "motion_blur"
    GAUSSIAN_BLUR = "gaussian_blur"
    DEFOCUS_BLUR = "defocus_blur"
    MIXED_BLUR = "mixed_blur"


class BlurTypeClassifier:
    """
    Classifier for identifying different types of blur in images.
    
    This classifier analyzes various image characteristics to determine
    the type of blur present in an image.
    """
    
    def __init__(self):
        """Initialize the blur type classifier."""
        self.motion_threshold = 0.15
        self.gaussian_threshold = 0.12
        self.defocus_threshold = 0.10
        self.mixed_threshold = 0.08
        
    def classify_blur_type(self, image: np.ndarray, blur_result: dict, 
                          blur_score: float) -> Tuple[BlurType, float, Dict[str, Any]]:
        """
        Classify the type of blur in an image.
        
        Args:
            image: Input grayscale image (2D numpy array)
            blur_result: Blur detection result from BlurDetector
            blur_score: Overall blur score from blur detection
            
        Returns:
            Tuple containing:
            - BlurType: The detected blur type
            - float: Confidence score for the classification (0-1)
            - Dict: Additional analysis details
        """
        if blur_score < 0.3:  # Image is relatively sharp
            return BlurType.SHARP, 0.9, {"reason": "Low blur score indicates sharp image"}
        
        # Analyze different blur characteristics
        motion_score = self._analyze_motion_blur(image)
        gaussian_score = self._analyze_gaussian_blur(image)
        defocus_score = self._analyze_defocus_blur(image, blur_result)
        
        # Create analysis details
        analysis_details = {
            "motion_score": float(motion_score),
            "gaussian_score": float(gaussian_score),
            "defocus_score": float(defocus_score),
            "blur_score": float(blur_score)
        }
        
        # Determine blur type based on scores
        scores = {
            BlurType.MOTION_BLUR: motion_score,
            BlurType.GAUSSIAN_BLUR: gaussian_score,
            BlurType.DEFOCUS_BLUR: defocus_score
        }
        
        # Find the dominant blur type
        max_blur_type = max(scores, key=scores.get)
        max_score = scores[max_blur_type]
        
        # Check for mixed blur (multiple high scores)
        high_scores = [score for score in scores.values() if score > self.mixed_threshold]
        if len(high_scores) >= 2:
            confidence = min(0.8, max_score)
            analysis_details["reason"] = "Multiple blur types detected"
            return BlurType.MIXED_BLUR, confidence, analysis_details
        
        # Return the dominant blur type
        confidence = min(0.95, max_score * 1.2)  # Scale confidence
        analysis_details["reason"] = f"Dominant {max_blur_type.value} characteristics"
        
        return max_blur_type, confidence, analysis_details
    
    def _analyze_motion_blur(self, image: np.ndarray) -> float:
        """
        Analyze image for motion blur characteristics.
        
        Motion blur typically shows:
        - Strong directional patterns in frequency domain
        - Linear streaking artifacts
        - Asymmetric blur patterns
        """
        try:
            # Apply FFT to analyze frequency domain
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Analyze directional patterns using Sobel filters
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude and direction
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            gradient_direction = np.arctan2(sobel_y, sobel_x)
            
            # Analyze directional consistency (motion blur has consistent direction)
            direction_std = np.std(gradient_direction[gradient_magnitude > np.percentile(gradient_magnitude, 75)])
            direction_consistency = 1.0 / (1.0 + direction_std)
            
            # Analyze frequency domain for linear patterns
            center_y, center_x = np.array(magnitude_spectrum.shape) // 2
            y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
            
            # Create radial mask to analyze frequency distribution
            radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            radial_profile = np.bincount(radius.astype(int).ravel(), magnitude_spectrum.ravel())
            radial_profile = radial_profile / np.bincount(radius.astype(int).ravel())
            
            # Motion blur shows specific frequency characteristics
            if len(radial_profile) > 10:
                high_freq_ratio = np.mean(radial_profile[-5:]) / (np.mean(radial_profile[2:7]) + 1e-6)
            else:
                high_freq_ratio = 0.1
            
            # Combine metrics
            motion_score = (direction_consistency * 0.6 + 
                          min(high_freq_ratio / 2.0, 1.0) * 0.4)
            
            return float(motion_score)
            
        except Exception as e:
            logger.warning(f"Error in motion blur analysis: {e}")
            return 0.0
    
    def _analyze_gaussian_blur(self, image: np.ndarray) -> float:
        """
        Analyze image for Gaussian blur characteristics.
        
        Gaussian blur typically shows:
        - Uniform blur in all directions
        - Smooth frequency rolloff
        - Symmetric blur patterns
        """
        try:
            # Calculate Laplacian variance (measure of blur)
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            laplacian_var = laplacian.var()
            
            # Analyze frequency domain characteristics
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Calculate frequency distribution
            center_y, center_x = np.array(magnitude_spectrum.shape) // 2
            y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
            radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Gaussian blur shows smooth frequency rolloff
            max_radius = min(center_x, center_y) // 2
            if max_radius > 5:
                radial_mask = radius <= max_radius
                freq_values = magnitude_spectrum[radial_mask]
                freq_smoothness = 1.0 / (1.0 + np.std(freq_values) / (np.mean(freq_values) + 1e-6))
            else:
                freq_smoothness = 0.5
            
            # Analyze edge characteristics
            edges = cv2.Canny(image.astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Gaussian blur reduces edge density uniformly
            edge_score = 1.0 - min(edge_density * 10, 1.0)
            
            # Combine metrics
            gaussian_score = (freq_smoothness * 0.4 + 
                            edge_score * 0.4 + 
                            (1.0 / (1.0 + laplacian_var / 100)) * 0.2)
            
            return float(gaussian_score)
            
        except Exception as e:
            logger.warning(f"Error in Gaussian blur analysis: {e}")
            return 0.0
    
    def _analyze_defocus_blur(self, image: np.ndarray, blur_result: dict) -> float:
        """
        Analyze defocus blur characteristics.
        
        Args:
            image: Input grayscale image
            blur_result: Blur detection result from BlurDetector
            
        Returns:
            float: Defocus blur score (0-1)
        """
        try:
            # For defocus blur, we'll use the blur score from the result
            # and analyze image characteristics
            blur_score = blur_result.get('blur_score', 0.0)
            
            # Analyze spatial frequency distribution
            h, w = image.shape
            center_x, center_y = w // 2, h // 2
            
            # Sample regions from center to edges
            region_size = min(h, w) // 8
            regions = []
            
            for i in range(0, h - region_size, region_size):
                for j in range(0, w - region_size, region_size):
                    region = image[i:i+region_size, j:j+region_size]
                    regions.append(region)
            
            if not regions:
                return 0.0
            
            # Calculate variance for each region
            variances = [np.var(region) for region in regions]
            
            # Defocus blur typically shows gradual decrease from center
            # Calculate distance-based variance pattern
            variance_std = np.std(variances) if len(variances) > 1 else 0.0
            
            # Combine with blur score
            defocus_score = min(1.0, blur_score * 0.7 + variance_std * 0.3)
            
            return float(defocus_score)
            
        except Exception as e:
            logger.warning(f"Error in defocus blur analysis: {e}")
            return 0.0
