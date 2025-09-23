"""
Configuration management for intelligent image resizing functionality.
"""

import cv2
import logging
from typing import Tuple, Optional
from dataclasses import dataclass
from constants import (
    ENABLE_INTELLIGENT_RESIZING,
    RESIZE_MAX_DIMENSION,
    RESIZE_TARGET_DIMENSION,
    RESIZE_INTERPOLATION,
    RESIZE_PRESERVE_ASPECT_RATIO
)

logger = logging.getLogger(__name__)


@dataclass
class ResizingConfig:
    """Simplified configuration class for intelligent image resizing."""
    
    # Core resizing settings - simplified and optimized
    enabled: bool = ENABLE_INTELLIGENT_RESIZING
    max_dimension: int = RESIZE_MAX_DIMENSION
    target_dimension: int = RESIZE_TARGET_DIMENSION
    
    # Quality settings - simplified
    interpolation_method: int = cv2.INTER_AREA  # Best for downsampling
    preserve_aspect_ratio: bool = RESIZE_PRESERVE_ASPECT_RATIO
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.max_dimension <= 0:
            raise ValueError("max_dimension must be positive")
        
        if self.target_dimension <= 0:
            raise ValueError("target_dimension must be positive")
        
        if self.resize_factor is not None:
            if not 0.1 <= self.resize_factor <= 1.0:
                raise ValueError("resize_factor must be between 0.1 and 1.0")
        
        if self.target_max_width <= 0 or self.target_max_height <= 0:
            raise ValueError("target_max dimensions must be positive")
    
    def should_resize(self, width: int, height: int) -> bool:
        """
        Simplified logic to determine if an image should be resized.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            bool: True if image should be resized
        """
        if not self.enabled:
            return False
        
        # Simple criteria: resize if either dimension exceeds target or max dimension
        exceeds_target = max(width, height) > self.target_dimension
        exceeds_max = max(width, height) > self.max_dimension
        
        return exceeds_target or exceeds_max
    
    def calculate_new_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """
        Calculate new dimensions for resizing with simplified logic.
        
        Args:
            width: Original image width
            height: Original image height
            
        Returns:
            Tuple[int, int]: New (width, height)
        """
        if not self.should_resize(width, height):
            return width, height
        
        # Calculate scale factor to fit within target dimension
        max_current = max(width, height)
        scale_factor = self.target_dimension / max_current
        
        if self.preserve_aspect_ratio:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        else:
            # If not preserving aspect ratio, fit to square
            new_width = new_height = self.target_dimension
        
        # Ensure minimum dimensions
        new_width = max(new_width, 64)
        new_height = max(new_height, 64)
        
        return new_width, new_height
    
    def get_interpolation_method(self) -> int:
        """Get the OpenCV interpolation method for resizing."""
        return self.interpolation_method


def load_resizing_config() -> ResizingConfig:
    """
    Load simplified resizing configuration.
    
    Returns:
        ResizingConfig: Configured resizing settings
    """
    try:
        config = ResizingConfig()
        
        logger.info(f"Loaded simplified resizing configuration: enabled={config.enabled}, "
                   f"max_dimension={config.max_dimension}, target_dimension={config.target_dimension}")
        
        return config
        
    except Exception as e:
        logger.warning(f"Failed to load resizing configuration: {e}")
        logger.info("Using default resizing configuration")
        return ResizingConfig()


# Global configuration instance
_resizing_config: Optional[ResizingConfig] = None


def get_resizing_config() -> ResizingConfig:
    """
    Get the global resizing configuration instance.
    
    Returns:
        ResizingConfig: Global configuration instance
    """
    global _resizing_config
    if _resizing_config is None:
        _resizing_config = load_resizing_config()
    return _resizing_config


def reload_resizing_config() -> ResizingConfig:
    """
    Reload the resizing configuration.
    
    Returns:
        ResizingConfig: Reloaded configuration instance
    """
    global _resizing_config
    _resizing_config = load_resizing_config()
    return _resizing_config
    return _resizing_config
