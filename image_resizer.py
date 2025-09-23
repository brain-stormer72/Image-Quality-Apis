"""
Intelligent image resizing module for optimizing image processing performance.
"""

import cv2
import numpy as np
import logging
import time
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

from resizing_config import get_resizing_config, ResizingConfig

logger = logging.getLogger(__name__)


@dataclass
class ResizingMetrics:
    """Metrics for tracking resizing performance."""
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
    Intelligent image resizer that optimizes images for processing while preserving quality.
    """
    
    def __init__(self, config: Optional[ResizingConfig] = None):
        """
        Initialize the image resizer.
        
        Args:
            config: Optional resizing configuration. If None, loads from environment.
        """
        self.config = config or get_resizing_config()
        self.metrics_history: list[ResizingMetrics] = []
        
        logger.info(f"IntelligentImageResizer initialized with config: "
                   f"enabled={self.config.enabled}, max_dimension={self.config.max_dimension}")
    
    def resize_image(self, image: np.ndarray, preserve_original_metadata: bool = True) -> Tuple[np.ndarray, Optional[ResizingMetrics]]:
        """
        Resize image using intelligent resizing logic.
        
        Args:
            image: Input image as numpy array
            preserve_original_metadata: Whether to preserve original image metadata
            
        Returns:
            Tuple[np.ndarray, Optional[ResizingMetrics]]: Resized image and metrics
        """
        if not self.config.enabled:
            return image, None
        
        start_time = time.time()
        original_height, original_width = image.shape[:2]
        
        # Check if resizing is needed
        if not self.config.should_resize(original_width, original_height):
            if self.config.enable_performance_logging:
                logger.debug(f"Image {original_width}x{original_height} does not need resizing")
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
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                image, resized_image, 
                original_width, original_height,
                new_width, new_height,
                resize_time
            )
            
            # Log performance if enabled
            if self.config.enable_performance_logging:
                self._log_resize_performance(metrics)
            
            # Store metrics for analysis
            self.metrics_history.append(metrics)
            
            # Limit metrics history to prevent memory growth
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-500:]
            
            return resized_image, metrics
            
        except Exception as e:
            logger.error(f"Failed to resize image from {original_width}x{original_height} "
                        f"to {new_width}x{new_height}: {e}")
            return image, None
    
    def _calculate_metrics(self, original_image: np.ndarray, resized_image: np.ndarray,
                          original_width: int, original_height: int,
                          new_width: int, new_height: int,
                          resize_time: float) -> ResizingMetrics:
        """Calculate resizing performance metrics."""
        
        # Calculate memory usage (approximate)
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
    
    def _log_resize_performance(self, metrics: ResizingMetrics):
        """Log resizing performance metrics."""
        logger.info(
            f"Image resized: {metrics.original_width}x{metrics.original_height} → "
            f"{metrics.new_width}x{metrics.new_height} "
            f"({metrics.dimension_reduction_percent:.1f}% pixel reduction, "
            f"{metrics.memory_reduction_percent:.1f}% memory reduction, "
            f"{metrics.resize_time_ms:.2f}ms)"
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary of all resizing operations.
        
        Returns:
            Dict[str, Any]: Performance summary statistics
        """
        if not self.metrics_history:
            return {"message": "No resizing operations performed yet"}
        
        total_operations = len(self.metrics_history)
        avg_memory_reduction = sum(m.memory_reduction_percent for m in self.metrics_history) / total_operations
        avg_dimension_reduction = sum(m.dimension_reduction_percent for m in self.metrics_history) / total_operations
        avg_resize_time = sum(m.resize_time_ms for m in self.metrics_history) / total_operations
        
        total_memory_saved = sum(m.original_size_bytes - m.new_size_bytes for m in self.metrics_history)
        
        return {
            "total_operations": total_operations,
            "average_memory_reduction_percent": round(avg_memory_reduction, 2),
            "average_dimension_reduction_percent": round(avg_dimension_reduction, 2),
            "average_resize_time_ms": round(avg_resize_time, 2),
            "total_memory_saved_bytes": total_memory_saved,
            "total_memory_saved_mb": round(total_memory_saved / (1024 * 1024), 2)
        }
    
    def reset_metrics(self):
        """Reset performance metrics history."""
        self.metrics_history.clear()
        logger.info("Resizing metrics history reset")


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
    Convenience function to resize an image using the global resizer.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Tuple[np.ndarray, Optional[ResizingMetrics]]: Resized image and metrics
    """
    resizer = get_image_resizer()
    return resizer.resize_image(image)


def get_resizing_performance_summary() -> Dict[str, Any]:
    """
    Get performance summary from the global resizer.
    
    Returns:
        Dict[str, Any]: Performance summary statistics
    """
    resizer = get_image_resizer()
    return resizer.get_performance_summary()
