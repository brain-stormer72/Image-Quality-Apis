"""
Configuration constants for the Image Quality API.
This file contains all hardcoded configuration values that were previously
loaded from environment variables.
"""

# Server Configuration
HOST = "0.0.0.0"
PORT = 8000
LOG_LEVEL = "info"
LOG_FILE = "logs/api.log"

# Rate Limiting Configuration
RATE_LIMIT_PER_MINUTE = 60
RATE_LIMIT_PER_HOUR = 1000

# Image Processing Configuration
MAX_IMAGE_DIMENSIONS = (10000, 10000)  # max width, height
MAX_FILE_SIZE_MB = 1  # Maximum file size in MB - optimized for performance

# Resizing Configuration (Simplified)
RESIZE_INTERPOLATION = "INTER_AREA"
ENABLE_INTELLIGENT_RESIZING = True
RESIZE_MAX_DIMENSION = 1024  # Reduced for better performance
RESIZE_TARGET_DIMENSION = 1024  # Simplified to single target dimension
RESIZE_PRESERVE_ASPECT_RATIO = True

# Blur Detection Thresholds
BLUR_THRESHOLD_LOW = 100.0
BLUR_THRESHOLD_HIGH = 300.0
LAPLACIAN_VARIANCE_THRESHOLD = 100.0

# Quality Detection Thresholds
OVEREXPOSURE_THRESHOLD = 0.02
UNDEREXPOSURE_THRESHOLD = 0.02
OVERSATURATION_THRESHOLD = 0.02
UNDERSATURATION_THRESHOLD = 0.02

# Performance Optimization
ENABLE_CACHING = True
CACHE_TTL_SECONDS = 3600
ENABLE_PARALLEL_PROCESSING = True
MAX_WORKER_THREADS = 4

# Image Download Configuration
DOWNLOAD_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
CHUNK_SIZE = 8192