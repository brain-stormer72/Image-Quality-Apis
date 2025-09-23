"""
Configuration management for intelligent image resizing functionality.
"""

import os
import cv2
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ResizingConfig:
    """Configuration class for intelligent image resizing."""
    
    # Core resizing settings - optimized for performance
    enabled: bool = True
    max_dimension: int = 1536  # Reduced from 2048 for better performance
    min_threshold_width: int = 1280  # Reduced from 1920 for more aggressive resizing
    min_threshold_height: int = 720   # Reduced from 1080 for more aggressive resizing
    
    # Resize strategy settings - optimized
    resize_factor: Optional[float] = 0.65  # More aggressive resizing (was 0.75)
    target_max_width: int = 1280   # Reduced from 1920
    target_max_height: int = 720   # Reduced from 1080
    
    # Quality settings - optimized for speed
    interpolation_method: int = cv2.INTER_AREA  # Best for downsampling
    preserve_aspect_ratio: bool = True
    
    # Performance settings - enhanced
    enable_performance_logging: bool = True
    memory_optimization: bool = True
    enable_early_termination: bool = True  # New: early termination for obvious cases
    blur_threshold_for_early_exit: float = 0.1  # New: exit early if very sharp
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.max_dimension <= 0:
            raise ValueError("max_dimension must be positive")
        
        if self.min_threshold_width <= 0 or self.min_threshold_height <= 0:
            raise ValueError("min_threshold dimensions must be positive")
        
        if self.resize_factor is not None:
            if not 0.1 <= self.resize_factor <= 1.0:
                raise ValueError("resize_factor must be between 0.1 and 1.0")
        
        if self.target_max_width <= 0 or self.target_max_height <= 0:
            raise ValueError("target_max dimensions must be positive")
    
    def should_resize(self, width: int, height: int) -> bool:
        """
        Determine if an image should be resized based on its dimensions.
        Enhanced with more aggressive resizing criteria.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            bool: True if image should be resized
        """
        if not self.enabled:
            return False
        
        # More aggressive resizing criteria for better performance
        exceeds_threshold = (
            width > self.min_threshold_width or 
            height > self.min_threshold_height
        )
        
        # Check if image exceeds maximum dimension
        exceeds_max = max(width, height) > self.max_dimension
        
        # Also resize if image is significantly larger than target dimensions
        significantly_larger = (
            width > self.target_max_width * 1.5 or
            height > self.target_max_height * 1.5
        )
        
        return exceeds_threshold or exceeds_max or significantly_larger
    
    def calculate_new_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """
        Calculate new dimensions for resizing with enhanced optimization.
        
        Args:
            width: Original image width
            height: Original image height
            
        Returns:
            Tuple[int, int]: New (width, height)
        """
        if not self.should_resize(width, height):
            return width, height
        
        if self.resize_factor is not None:
            # Use percentage-based resizing with optimization
            new_width = int(width * self.resize_factor)
            new_height = int(height * self.resize_factor)
        else:
            # Use target dimensions with aspect ratio preservation
            if self.preserve_aspect_ratio:
                # Calculate scale to fit within target dimensions
                scale_w = self.target_max_width / width
                scale_h = self.target_max_height / height
                scale = min(scale_w, scale_h, 1.0)  # Don't upscale
                
                new_width = int(width * scale)
                new_height = int(height * scale)
            else:
                new_width = min(width, self.target_max_width)
                new_height = min(height, self.target_max_height)
        
        # Ensure minimum dimensions but allow smaller sizes for performance
        new_width = max(new_width, 32)  # Reduced from 64
        new_height = max(new_height, 32)  # Reduced from 64
        
        return new_width, new_height
    
    def get_interpolation_method(self) -> int:
        """Get the OpenCV interpolation method for resizing."""
        return self.interpolation_method
    
    def should_use_early_termination(self) -> bool:
        """Check if early termination optimization is enabled."""
        return self.enable_early_termination


def load_resizing_config() -> ResizingConfig:
    """
    Load resizing configuration from environment variables with enhanced defaults.
    
    Returns:
        ResizingConfig: Configured resizing settings
    """
    try:
        # Parse interpolation method
        interpolation_str = os.getenv("RESIZE_INTERPOLATION", "INTER_AREA").upper()
        interpolation_map = {
            "INTER_AREA": cv2.INTER_AREA,
            "INTER_LINEAR": cv2.INTER_LINEAR,
            "INTER_CUBIC": cv2.INTER_CUBIC,
            "INTER_LANCZOS4": cv2.INTER_LANCZOS4
        }
        interpolation_method = interpolation_map.get(interpolation_str, cv2.INTER_AREA)
        
        # Parse resize factor (optional) - more aggressive default
        resize_factor_str = os.getenv("RESIZE_FACTOR")
        resize_factor = float(resize_factor_str) if resize_factor_str else 0.65
        
        config = ResizingConfig(
            enabled=os.getenv("ENABLE_INTELLIGENT_RESIZING", "true").lower() == "true",
            max_dimension=int(os.getenv("RESIZE_MAX_DIMENSION", "1536")),  # Reduced default
            min_threshold_width=int(os.getenv("RESIZE_MIN_THRESHOLD_WIDTH", "1280")),  # Reduced default
            min_threshold_height=int(os.getenv("RESIZE_MIN_THRESHOLD_HEIGHT", "720")),  # Reduced default
            resize_factor=resize_factor,
            target_max_width=int(os.getenv("RESIZE_TARGET_MAX_WIDTH", "1280")),  # Reduced default
            target_max_height=int(os.getenv("RESIZE_TARGET_MAX_HEIGHT", "720")),  # Reduced default
            interpolation_method=interpolation_method,
            preserve_aspect_ratio=os.getenv("RESIZE_PRESERVE_ASPECT_RATIO", "true").lower() == "true",
            enable_performance_logging=os.getenv("RESIZE_ENABLE_PERFORMANCE_LOGGING", "true").lower() == "true",
            memory_optimization=os.getenv("RESIZE_MEMORY_OPTIMIZATION", "true").lower() == "true",
            enable_early_termination=os.getenv("RESIZE_ENABLE_EARLY_TERMINATION", "true").lower() == "true",
            blur_threshold_for_early_exit=float(os.getenv("RESIZE_BLUR_THRESHOLD_EARLY_EXIT", "0.1"))
        )
        
        logger.info(f"Loaded optimized resizing configuration: enabled={config.enabled}, "
                   f"max_dimension={config.max_dimension}, resize_factor={config.resize_factor}, "
                   f"early_termination={config.enable_early_termination}")
        
        return config
        
    except Exception as e:
        logger.warning(f"Failed to load resizing configuration from environment: {e}")
        logger.info("Using optimized default resizing configuration")
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
    Reload the resizing configuration from environment variables.
    
    Returns:
        ResizingConfig: Reloaded configuration instance
    """
    global _resizing_config
    _resizing_config = load_resizing_config()
    return _resizing_config
