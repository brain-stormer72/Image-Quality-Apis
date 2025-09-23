"""
Simplified intelligent image resizing module for optimizing image processing performance.
"""

import cv2
import numpy as np
import logging
import time
from typing import Tuple, Optional
from dataclasses import dataclass

from resizing_config import get_resizing_config, ResizingConfig

logger = logging.getLogger(__name__)


@dataclass
class ResizingMetrics:
    """Simplified metrics for tracking resizing performance."""
    original_width: int
    original_height: int
    new_width: int
    new_height: int
    original_size_bytes: int
    new_size_bytes: int
    resize_time_ms: float
    memory_reduction_percent: float
    dimension_reduction_percent: float


class IntelligentImageResizer:
    """
    Simplified intelligent image resizer that optimizes images for processing while preserving quality.
    """
    
    def __init__(self, config: Optional[ResizingConfig] = None):
        """
        Initialize the image resizer.
        
        Args:
            config: Optional resizing configuration. If None, loads from environment.
        """
        self.config = config or get_resizing_config()
        
        logger.info(f"IntelligentImageResizer initialized: "
                   f"enabled={self.config.enabled}, max_dimension={self.config.max_dimension}")
    
    def resize_image(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[ResizingMetrics]]:
        """
        Simplified resize image using intelligent resizing logic.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple[np.ndarray, Optional[ResizingMetrics]]: Resized image and metrics
        """
        if not self.config.enabled:
            return image, None
        
        start_time = time.time()
        original_height, original_width = image.shape[:2]
        
        # Check if resizing is needed
        if not self.config.should_resize(original_width, original_height):
            return image, None
        
        # Calculate new dimensions
        new_width, new_height = self.config.calculate_new_dimensions(original_width, original_height)
        
        # Perform resizing
        try:
            resized_image = cv2.resize(
                image, 
                (new_width, new_height), 
                interpolation=self.config.get_interpolation_method()
            )
            
            resize_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Calculate simplified metrics
            metrics = self._calculate_metrics(
                image, resized_image, 
                original_width, original_height,
                new_width, new_height,
                resize_time
            )
            
            logger.info(f"Resized image: {original_width}x{original_height} → {new_width}x{new_height} "
                       f"({metrics.memory_reduction_percent:.1f}% memory reduction)")
            
            return resized_image, metrics
            
        except Exception as e:
            logger.error(f"Failed to resize image from {original_width}x{original_height} "
                        f"to {new_width}x{new_height}: {e}")
            return image, None
    
    def _calculate_metrics(self, original_image: np.ndarray, resized_image: np.ndarray,
                          original_width: int, original_height: int,
                          new_width: int, new_height: int,
                          resize_time: float) -> ResizingMetrics:
        """Calculate simplified resizing performance metrics."""
        
        # Calculate memory usage
        original_size = original_image.nbytes
        new_size = resized_image.nbytes
        
        # Calculate reduction percentages
        memory_reduction = ((original_size - new_size) / original_size) * 100 if original_size > 0 else 0
        
        original_pixels = original_width * original_height
        new_pixels = new_width * new_height
        dimension_reduction = ((original_pixels - new_pixels) / original_pixels) * 100 if original_pixels > 0 else 0
        
        return ResizingMetrics(
            original_width=original_width,
            original_height=original_height,
            new_width=new_width,
            new_height=new_height,
            original_size_bytes=original_size,
            new_size_bytes=new_size,
            resize_time_ms=resize_time,
            memory_reduction_percent=memory_reduction,
            dimension_reduction_percent=dimension_reduction
        )


# Global resizer instance
_global_resizer: Optional[IntelligentImageResizer] = None


def get_image_resizer() -> IntelligentImageResizer:
    """
    Get the global image resizer instance.
    
    Returns:
        IntelligentImageResizer: Global resizer instance
    """
    global _global_resizer
    if _global_resizer is None:
        _global_resizer = IntelligentImageResizer()
    return _global_resizer


def resize_image_intelligently(image: np.ndarray) -> Tuple[np.ndarray, Optional[ResizingMetrics]]:
    """
    Simplified convenience function to resize an image using the global resizer.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Tuple[np.ndarray, Optional[ResizingMetrics]]: Resized image and metrics
    """
    resizer = get_image_resizer()
    return resizer.resize_image(image)
