"""
Logging configuration for the Image Quality Check API.
"""

import logging
import logging.handlers
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str = None, performance_mode: bool = False):
    """
    Set up logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        performance_mode: If True, reduces logging verbosity for better performance
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    # Suppress noisy third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


class APILogger:
    """Centralized logger for API operations."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def log_request(self, endpoint: str, params: dict = None):
        """Log incoming API request."""
        self.logger.info(f"API Request - Endpoint: {endpoint}, Params: {params}")
    
    def log_response(self, endpoint: str, status_code: int, processing_time: float):
        """Log API response."""
        self.logger.info(f"API Response - Endpoint: {endpoint}, Status: {status_code}, Time: {processing_time:.3f}s")
    
    def log_error(self, endpoint: str, error: Exception, context: dict = None):
        """Log API error."""
        self.logger.error(f"API Error - Endpoint: {endpoint}, Error: {str(error)}, Context: {context}")
    
    def log_image_processing(self, url: str, success: bool, processing_time: float, error: str = None):
        """Log image processing results."""
        if success:
            self.logger.info(f"Image Processing Success - URL: {url}, Time: {processing_time:.3f}s")
        else:
            self.logger.warning(f"Image Processing Failed - URL: {url}, Error: {error}, Time: {processing_time:.3f}s")
    
    def log_blur_detection(self, url: str, is_blurred: bool, blur_score: float, confidence: float):
        """Log blur detection results."""
        self.logger.info(f"Blur Detection - URL: {url}, Blurred: {is_blurred}, Score: {blur_score:.3f}, Confidence: {confidence:.3f}")


# Create API logger instance
api_logger = APILogger("image_quality_api")
