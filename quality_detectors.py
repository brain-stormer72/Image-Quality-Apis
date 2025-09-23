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


def detect_overexposure(image: np.ndarray, threshold: float = 0.95) -> QualityResult:
    """
    Detect overexposure in an image with optimized calculations.
    Enhanced with vectorized operations and early termination.
    
    Args:
        image: Input image as numpy array
        threshold: Overexposure threshold (0-1)
        
    Returns:
        QualityResult: Detection results with optimization metrics
    """
    start_time = time.time()
    
    try:
        # Convert to grayscale if needed using optimized weights
        if len(image.shape) == 3:
            # Use optimized luminance calculation (ITU-R BT.709)
            gray = np.dot(image[...,:3], [0.2126, 0.7152, 0.0722])
        else:
            gray = image.copy()
        
        # Normalize to 0-1 range efficiently
        if gray.dtype != np.float32:
            gray = gray.astype(np.float32) / 255.0
        
        # Early termination: check if image is obviously not overexposed
        max_brightness = np.max(gray)
        if max_brightness < 0.8:
            return QualityResult(
                score=0.0,
                is_good_quality=True,
                confidence=0.9,
                details={
                    'max_brightness': float(max_brightness),
                    'overexposed_pixels': 0,
                    'overexposed_percentage': 0.0,
                    'processing_time': time.time() - start_time,
                    'early_termination': 'low_max_brightness'
                }
            )
        
        # Vectorized overexposure detection
        overexposed_mask = gray >= threshold
        overexposed_pixels = np.sum(overexposed_mask)
        total_pixels = gray.size
        overexposed_percentage = (overexposed_pixels / total_pixels) * 100
        
        # Calculate overexposure score with improved scaling
        if overexposed_percentage > 20:  # Severe overexposure
            score = min(1.0, overexposed_percentage / 30.0)
            confidence = 0.9
        elif overexposed_percentage > 5:  # Moderate overexposure
            score = overexposed_percentage / 20.0
            confidence = 0.8
        else:  # Minimal overexposure
            score = overexposed_percentage / 10.0
            confidence = 0.7
        
        # Additional quality metrics with vectorized operations
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Check for blown highlights in color channels if available
        blown_highlights = 0
        if len(image.shape) == 3:
            # Vectorized blown highlight detection
            for channel in range(3):
                channel_data = image[:, :, channel].astype(np.float32) / 255.0
                blown_highlights += np.sum(channel_data >= 0.98)
        
        is_good_quality = score < 0.3 and overexposed_percentage < 10
        
        processing_time = time.time() - start_time
        
        return QualityResult(
            score=float(score),
            is_good_quality=is_good_quality,
            confidence=float(confidence),
            details={
                'overexposed_pixels': int(overexposed_pixels),
                'total_pixels': int(total_pixels),
                'overexposed_percentage': float(overexposed_percentage),
                'mean_brightness': float(mean_brightness),
                'brightness_std': float(brightness_std),
                'max_brightness': float(max_brightness),
                'blown_highlights': int(blown_highlights),
                'processing_time': processing_time,
                'threshold_used': threshold
            }
        )
        
    except Exception as e:
        return QualityResult(
            score=0.5,
            is_good_quality=False,
            confidence=0.1,
            details={'error': str(e), 'processing_time': time.time() - start_time}
        )


def detect_underexposure(image: np.ndarray, threshold: float = 0.05) -> QualityResult:
    """
    Detect underexposure in an image with optimized vectorized operations.
    
    Args:
        image: Input image as numpy array
        threshold: Underexposure threshold (0-1)
        
    Returns:
        QualityResult: Detection results
    """
    start_time = time.time()
    
    try:
        # Convert to grayscale efficiently
        if len(image.shape) == 3:
            gray = np.dot(image[...,:3], [0.2126, 0.7152, 0.0722])
        else:
            gray = image.copy()
        
        # Normalize efficiently
        if gray.dtype != np.float32:
            gray = gray.astype(np.float32) / 255.0
        
        # Early termination for obviously well-exposed images
        min_brightness = np.min(gray)
        if min_brightness > 0.2:
            return QualityResult(
                score=0.0,
                is_good_quality=True,
                confidence=0.9,
                details={
                    'min_brightness': float(min_brightness),
                    'underexposed_pixels': 0,
                    'underexposed_percentage': 0.0,
                    'processing_time': time.time() - start_time,
                    'early_termination': 'high_min_brightness'
                }
            )
        
        # Vectorized underexposure detection
        underexposed_mask = gray <= threshold
        underexposed_pixels = np.sum(underexposed_mask)
        total_pixels = gray.size
        underexposed_percentage = (underexposed_pixels / total_pixels) * 100
        
        # Calculate score with improved scaling
        if underexposed_percentage > 25:
            score = min(1.0, underexposed_percentage / 40.0)
            confidence = 0.9
        elif underexposed_percentage > 10:
            score = underexposed_percentage / 30.0
            confidence = 0.8
        else:
            score = underexposed_percentage / 20.0
            confidence = 0.7
        
        # Additional metrics
        mean_brightness = np.mean(gray)
        
        is_good_quality = score < 0.3 and underexposed_percentage < 15
        
        return QualityResult(
            score=float(score),
            is_good_quality=is_good_quality,
            confidence=float(confidence),
            details={
                'underexposed_pixels': int(underexposed_pixels),
                'total_pixels': int(total_pixels),
                'underexposed_percentage': float(underexposed_percentage),
                'mean_brightness': float(mean_brightness),
                'min_brightness': float(min_brightness),
                'processing_time': time.time() - start_time,
                'threshold_used': threshold
            }
        )
        
    except Exception as e:
        return QualityResult(
            score=0.5,
            is_good_quality=False,
            confidence=0.1,
            details={'error': str(e), 'processing_time': time.time() - start_time}
        )


def detect_noise(image: np.ndarray) -> QualityResult:
    """
    Detect noise in an image using optimized statistical methods.
    Enhanced with vectorized operations and multiple noise metrics.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        QualityResult: Noise detection results
    """
    start_time = time.time()
    
    try:
        # Convert to grayscale efficiently
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Ensure float32 for calculations
        if gray.dtype != np.float32:
            gray = gray.astype(np.float32)
        
        # Multiple noise detection methods for better accuracy
        
        # 1. Laplacian variance method (fast)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # 2. Local standard deviation method (vectorized)
        # Use a smaller kernel for speed
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(gray, -1, kernel)
        local_variance = cv2.filter2D(gray**2, -1, kernel) - local_mean**2
        noise_estimate = np.mean(np.sqrt(np.maximum(local_variance, 0)))
        
        # 3. High-frequency content analysis (optimized)
        # Apply high-pass filter
        high_pass_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        high_freq = cv2.filter2D(gray, -1, high_pass_kernel)
        high_freq_energy = np.mean(np.abs(high_freq))
        
        # Combine metrics for final score
        # Normalize each metric
        laplacian_score = min(1.0, laplacian_var / 1000.0)  # Adjust scaling
        noise_score = min(1.0, noise_estimate / 50.0)       # Adjust scaling
        high_freq_score = min(1.0, high_freq_energy / 30.0) # Adjust scaling
        
        # Weighted combination
        final_score = (laplacian_score * 0.4 + noise_score * 0.4 + high_freq_score * 0.2)
        
        # Determine quality and confidence
        if final_score > 0.7:
            is_good_quality = False
            confidence = 0.9
        elif final_score > 0.4:
            is_good_quality = False
            confidence = 0.7
        else:
            is_good_quality = True
            confidence = 0.8
        
        processing_time = time.time() - start_time
        
        return QualityResult(
            score=float(final_score),
            is_good_quality=is_good_quality,
            confidence=float(confidence),
            details={
                'laplacian_variance': float(laplacian_var),
                'noise_estimate': float(noise_estimate),
                'high_freq_energy': float(high_freq_energy),
                'laplacian_score': float(laplacian_score),
                'noise_score': float(noise_score),
                'high_freq_score': float(high_freq_score),
                'processing_time': processing_time
            }
        )
        
    except Exception as e:
        return QualityResult(
            score=0.5,
            is_good_quality=False,
            confidence=0.1,
            details={'error': str(e), 'processing_time': time.time() - start_time}
        )


def analyze_comprehensive_quality(image: np.ndarray, 
                                enable_blur_detection: bool = True,
                                enable_exposure_detection: bool = True,
                                enable_noise_detection: bool = True,
                                blur_detector=None) -> dict:
    """
    Perform comprehensive quality analysis with optimized processing.
    Enhanced with parallel processing and selective analysis.
    
    Args:
        image: Input image as numpy array
        enable_blur_detection: Enable blur detection
        enable_exposure_detection: Enable exposure detection  
        enable_noise_detection: Enable noise detection
        blur_detector: Optional blur detector instance
        
    Returns:
        dict: Comprehensive quality analysis results
    """
    start_time = time.time()
    results = {}
    
    try:
        # Parallel processing of independent analyses
        import concurrent.futures
        
        def run_blur_analysis():
            if enable_blur_detection and blur_detector:
                blur_result = blur_detector.detectBlur(image)
                return ('blur', blur_result)
            return ('blur', None)
        
        def run_exposure_analysis():
            if enable_exposure_detection:
                overexp = detect_overexposure(image)
                underexp = detect_underexposure(image)
                return ('exposure', {'overexposure': overexp, 'underexposure': underexp})
            return ('exposure', None)
        
        def run_noise_analysis():
            if enable_noise_detection:
                noise_result = detect_noise(image)
                return ('noise', noise_result)
            return ('noise', None)
        
        # Execute analyses in parallel for better performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            if enable_blur_detection:
                futures.append(executor.submit(run_blur_analysis))
            if enable_exposure_detection:
                futures.append(executor.submit(run_exposure_analysis))
            if enable_noise_detection:
                futures.append(executor.submit(run_noise_analysis))
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                analysis_type, result = future.result()
                if result is not None:
                    results[analysis_type] = result
        
        # Calculate overall quality score
        quality_scores = []
        confidence_scores = []
        
        if 'blur' in results and results['blur']:
            blur_score = 1.0 - results['blur'].get('blur_score', 0.5)
            quality_scores.append(blur_score * 0.4)  # 40% weight
            confidence_scores.append(results['blur'].get('confidence', 0.5))
        
        if 'exposure' in results and results['exposure']:
            exp_results = results['exposure']
            overexp_score = 1.0 - exp_results['overexposure'].score
            underexp_score = 1.0 - exp_results['underexposure'].score
            exposure_score = min(overexp_score, underexp_score)
            quality_scores.append(exposure_score * 0.35)  # 35% weight
            confidence_scores.append((exp_results['overexposure'].confidence + 
                                    exp_results['underexposure'].confidence) / 2)
        
        if 'noise' in results and results['noise']:
            noise_score = 1.0 - results['noise'].score
            quality_scores.append(noise_score * 0.25)  # 25% weight
            confidence_scores.append(results['noise'].confidence)
        
        # Calculate final metrics
        overall_score = sum(quality_scores) if quality_scores else 0.5
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        is_good_quality = overall_score > 0.6
        
        processing_time = time.time() - start_time
        
        # Add summary to results
        results['summary'] = {
            'overall_score': float(overall_score),
            'is_good_quality': is_good_quality,
            'overall_confidence': float(overall_confidence),
            'processing_time': processing_time,
            'analyses_performed': list(results.keys()),
            'parallel_processing': True
        }
        
        return results
        
    except Exception as e:
        return {
            'error': str(e),
            'processing_time': time.time() - start_time,
            'summary': {
                'overall_score': 0.5,
                'is_good_quality': False,
                'overall_confidence': 0.1,
                'error': True
            }
        }
