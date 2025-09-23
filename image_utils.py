"""
Image utilities for downloading, validating, and processing images from URLs.
"""

import cv2
import numpy as np
import requests
import aiohttp
import asyncio
import logging
from typing import Optional, Tuple
from io import BytesIO
from PIL import Image
import tempfile
import os
import time
from urllib.parse import urlparse

# Import intelligent resizing functionality
from image_resizer import resize_image_intelligently, ResizingMetrics
from performance_monitor import track_performance

# Supported image formats (only JPG, JPEG, PNG)
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png'}

# Configure logging
logger = logging.getLogger(__name__)
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB max file size
DOWNLOAD_TIMEOUT = 30  # seconds
MAX_IMAGE_DIMENSIONS = (10000, 10000)  # max width, height


class ImageDownloadError(Exception):
    """Custom exception for image download errors."""
    pass


class ImageValidationError(Exception):
    """Custom exception for image validation errors."""
    pass


def is_valid_image_url(url: str) -> bool:
    """
    Check if URL appears to be a valid image URL based on extension or known patterns.

    Args:
        url: The URL to validate

    Returns:
        bool: True if URL appears to be an image URL
    """
    try:
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()

        # Check if path ends with supported image extension
        if any(path.endswith(ext) for ext in SUPPORTED_FORMATS):
            return True

        # Allow URLs that might serve images without extensions (like httpbin.org/image/jpeg)
        # Check for common image service patterns
        image_patterns = ['/image/', '/img/', '/photo/', '/picture/']
        if any(pattern in path for pattern in image_patterns):
            return True

        # If no extension and no pattern, we'll let the download attempt handle validation
        # This is more permissive but allows for dynamic image URLs
        return True

    except Exception:
        return False


async def download_image_from_url_async(url: str, timeout: int = 30, max_size: int = 50 * 1024 * 1024) -> np.ndarray:
    """
    Asynchronously download and process an image from a URL.

    Args:
        url: Image URL to download
        timeout: Request timeout in seconds
        max_size: Maximum file size in bytes

    Returns:
        np.ndarray: Processed image as numpy array

    Raises:
        ImageDownloadError: If download fails
        ImageValidationError: If image validation fails
    """
    with track_performance("download_image_async"):
        try:
            # Create aiohttp session with optimized settings
            connector = aiohttp.TCPConnector(
                limit=100,  # Connection pool limit
                limit_per_host=30,  # Per-host connection limit
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )

            timeout_config = aiohttp.ClientTimeout(total=timeout, connect=10)

            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout_config,
                headers={'User-Agent': 'ImageQualityAPI/1.1.0'}
            ) as session:

                async with session.get(url) as response:
                    # Check response status
                    if response.status != 200:
                        raise ImageDownloadError(f"HTTP {response.status}: Failed to download image from {url}")

                    # Check content length
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > max_size:
                        raise ImageValidationError(f"Image too large: {content_length} bytes (max: {max_size})")

                    # Validate content type
                    content_type = response.headers.get('content-type', '').lower()
                    parsed_url = urlparse(url)

                    if not content_type.startswith('image/') and content_type != 'application/octet-stream':
                        if not any(ext in parsed_url.path.lower() for ext in SUPPORTED_FORMATS):
                            raise ImageValidationError(f"URL does not point to a supported image format (JPG, JPEG, PNG). Content-Type: {content_type}")

                    # Read image data with size limit
                    image_data = BytesIO()
                    size = 0

                    async for chunk in response.content.iter_chunked(8192):  # 8KB chunks
                        size += len(chunk)
                        if size > max_size:
                            raise ImageValidationError(f"Image too large: {size} bytes (max: {max_size})")
                        image_data.write(chunk)

                    image_data.seek(0)

                    # Process image in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, _process_image_data_sync, image_data, parsed_url.path.lower())

        except aiohttp.ClientError as e:
            raise ImageDownloadError(f"Network error downloading image: {e}")
        except asyncio.TimeoutError:
            raise ImageDownloadError(f"Timeout downloading image from {url}")
        except Exception as e:
            if isinstance(e, (ImageDownloadError, ImageValidationError)):
                raise
            raise ImageDownloadError(f"Unexpected error downloading image: {e}")


def _optimize_image_for_processing(image: np.ndarray, max_dimension: int = 2048) -> Tuple[np.ndarray, Optional[ResizingMetrics]]:
    """
    Optimize image for processing using intelligent resizing.

    Args:
        image: Input image
        max_dimension: Maximum dimension for processing (legacy parameter, now handled by resizing config)

    Returns:
        Tuple[np.ndarray, Optional[ResizingMetrics]]: Optimized image and resizing metrics
    """
    # Use intelligent resizing instead of basic resizing
    optimized_image, metrics = resize_image_intelligently(image)

    # Fallback to legacy resizing if intelligent resizing is disabled
    if metrics is None and max_dimension:
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            # Calculate scaling factor
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Use INTER_AREA for downsampling (better quality)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Legacy resize: {width}x{height} to {new_width}x{new_height}")
            return resized, None

    return optimized_image, metrics


def _process_image_data_sync(image_data: BytesIO, file_path: str) -> np.ndarray:
    """
    Synchronous image processing helper for async operations.
    """
    try:
        # Check if file extension is supported
        file_ext = None
        for ext in SUPPORTED_FORMATS:
            if ext in file_path.lower():
                file_ext = ext
                break

        if file_ext is None:
            # Try to process anyway for URLs without clear extensions
            pass

        # Handle standard image formats (JPG, JPEG, PNG only)
        # Try OpenCV first (faster for most formats)
        image_data.seek(0)
        image_bytes = image_data.read()

        # Decode with OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            # Fallback to PIL for formats OpenCV doesn't support well
            image_data.seek(0)
            with Image.open(image_data) as pil_image:
                # Convert to RGB if needed
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')

                # Convert PIL to OpenCV format
                image_array = np.array(pil_image)
                image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        if image is None:
            raise ImageValidationError("Failed to decode image data")

        # Validate image dimensions
        height, width = image.shape[:2]
        max_width, max_height = MAX_IMAGE_DIMENSIONS
        if height > max_height or width > max_width:
            raise ImageValidationError(f"Image dimensions too large: {width}x{height} (max: {max_width}x{max_height})")

        # Optimize image for processing with intelligent resizing
        optimized_image, resize_metrics = _optimize_image_for_processing(image)

        # Log resize performance if metrics are available
        if resize_metrics:
            logger.info(f"Image optimized: {width}x{height} → {resize_metrics.new_width}x{resize_metrics.new_height} "
                       f"({resize_metrics.memory_reduction_percent:.1f}% memory reduction)")
        else:
            logger.info(f"Successfully processed image: {width}x{height}")

        return optimized_image

    except Exception as e:
        if isinstance(e, ImageValidationError):
            raise
        raise ImageValidationError(f"Failed to process image: {e}")


def download_image_from_url(url: str) -> np.ndarray:
    """
    Download an image from URL and return as OpenCV numpy array.

    Args:
        url: The image URL to download

    Returns:
        np.ndarray: Image as OpenCV numpy array (grayscale)

    Raises:
        ImageDownloadError: If download fails
        ImageValidationError: If image validation fails
    """
    with track_performance("download_image_sync"):
        try:
            logger.info(f"Downloading image from URL: {url}")
            
            # Set headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Download the image
            response = requests.get(url, headers=headers, timeout=DOWNLOAD_TIMEOUT, stream=True)
            response.raise_for_status()
            
            # Check content type - be more flexible for cloud storage services
            content_type = response.headers.get('content-type', '').lower()
            parsed_url = urlparse(url)
            if not content_type.startswith('image/') and content_type != 'application/octet-stream':
                # Only reject if it's clearly not an image and not a generic binary type
                if not any(ext in parsed_url.path.lower() for ext in SUPPORTED_FORMATS):
                    raise ImageValidationError(f"URL does not point to a supported image format (JPG, JPEG, PNG). Content-Type: {content_type}")

            # Log content type for debugging
            logger.debug(f"Content-Type: {content_type} for URL: {url}")
            
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > MAX_IMAGE_SIZE:
                raise ImageValidationError(f"Image too large: {content_length} bytes (max: {MAX_IMAGE_SIZE})")
            
            # Read image data
            image_data = BytesIO()
            downloaded_size = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    downloaded_size += len(chunk)
                    if downloaded_size > MAX_IMAGE_SIZE:
                        raise ImageValidationError(f"Image too large: {downloaded_size} bytes (max: {MAX_IMAGE_SIZE})")
                    image_data.write(chunk)
            
            image_data.seek(0)
            
            # Validate and convert image
            return validate_and_convert_image(image_data.getvalue(), url)

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image from {url}: {e}")
            raise ImageDownloadError(f"Failed to download image: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error downloading image from {url}: {e}")
            raise ImageDownloadError(f"Unexpected error: {str(e)}")


def validate_and_convert_image(image_data: bytes, url: str) -> np.ndarray:
    """
    Validate image data and convert to OpenCV format.
    
    Args:
        image_data: Raw image bytes
        url: Original URL for logging
        
    Returns:
        np.ndarray: Image as OpenCV numpy array (grayscale)
        
    Raises:
        ImageValidationError: If image validation fails
    """
    try:
        # Try to open with PIL first for validation
        pil_image = Image.open(BytesIO(image_data))
        
        # Check image dimensions
        width, height = pil_image.size
        if width > MAX_IMAGE_DIMENSIONS[0] or height > MAX_IMAGE_DIMENSIONS[1]:
            raise ImageValidationError(f"Image dimensions too large: {width}x{height} (max: {MAX_IMAGE_DIMENSIONS[0]}x{MAX_IMAGE_DIMENSIONS[1]})")
        
        # Convert to RGB if necessary (for consistent processing)
        if pil_image.mode in ('RGBA', 'LA', 'P'):
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode not in ('RGB', 'L'):
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL image to OpenCV format
        if pil_image.mode == 'L':
            # Already grayscale
            cv_image = np.array(pil_image)
        else:
            # Convert RGB to BGR for OpenCV, then to grayscale
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        logger.info(f"Successfully processed image from {url}: {cv_image.shape}")
        return cv_image
        
    except Exception as e:
        logger.error(f"Failed to validate/convert image from {url}: {e}")
        raise ImageValidationError(f"Invalid image format or corrupted data: {str(e)}")


def save_temp_image(image: np.ndarray) -> str:
    """
    Save OpenCV image to temporary file.
    
    Args:
        image: OpenCV image array
        
    Returns:
        str: Path to temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    cv2.imwrite(temp_file.name, image)
    temp_file.close()
    return temp_file.name


def cleanup_temp_file(file_path: str) -> None:
    """
    Clean up temporary file.
    
    Args:
        file_path: Path to temporary file to delete
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary file {file_path}: {e}")


def get_image_info(image: np.ndarray) -> dict:
    """
    Get basic information about an image.
    
    Args:
        image: OpenCV image array
        
    Returns:
        dict: Image information
    """
    height, width = image.shape[:2]
    return {
        'width': int(width),
        'height': int(height),
        'channels': len(image.shape),
        'dtype': str(image.dtype)
    }
