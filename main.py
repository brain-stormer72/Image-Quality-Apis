"""
FastAPI application for Image Quality Check API.
Provides endpoints for blur detection in images from URLs.
"""

import time
import logging
import numpy as np
import cv2
import os
import asyncio
from contextlib import asynccontextmanager
from typing import List, Tuple

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import (
    ImageRequest, BatchImageRequest, ImageAnalysisResponse,
    BatchAnalysisResponse, BlurResult, BlurTypeDetails, HealthResponse, ErrorResponse,
    ComprehensiveAnalysisResponse, ComprehensiveBatchAnalysisResponse,
    ComprehensiveQualityResult, QualityDetectionResult
)
from image_utils import (
    download_image_from_url, download_image_from_url_async, ImageDownloadError, ImageValidationError,
    is_valid_image_url, cleanup_temp_file
)
from BlurDetector import BlurDetector
from quality_detectors import ImageQualityDetector
from blur_type_classifier import BlurTypeClassifier, BlurType
from cache_manager import get_cache_manager, init_cache_manager
from logging_config import setup_logging, api_logger
from middleware import RateLimitMiddleware, RequestLoggingMiddleware, SecurityHeadersMiddleware


class ThreadSafeBlurDetector:
    """Thread-safe wrapper for BlurDetector that handles reinitialization."""

    def __init__(self, **kwargs):
        self.init_params = kwargs
        self._detector = None

    def detectBlur(self, image):
        """Detect blur in image, reinitializing detector if needed."""
        # Create a new detector instance for each detection to avoid state issues
        detector = BlurDetector(**self.init_params)
        return detector.detectBlur(image)

# Setup logging
log_level = os.getenv("LOG_LEVEL", "INFO")  # Back to INFO for production
log_file = os.getenv("LOG_FILE", "logs/api.log")
setup_logging(log_level, log_file)

logger = logging.getLogger(__name__)

# Global variables
blur_detector = None
quality_detector = None
blur_type_classifier = None
cache_manager = None
app_start_time = time.time()
API_VERSION = "1.2.0"  # Updated version for performance optimizations


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global blur_detector, quality_detector, blur_type_classifier, cache_manager

    # Startup
    logger.info("Starting Image Quality Check API...")
    try:
        # Initialize thread-safe blur detector with optimized parameters
        blur_detector = ThreadSafeBlurDetector(
            downsampling_factor=4,
            num_scales=3,
            scale_start=2,
            entropy_filt_kernel_sze=7,
            sigma_s_RF_filter=15,
            sigma_r_RF_filter=0.25,
            num_iterations_RF_filter=3,
            show_progress=False  # Disable progress display for API
        )
        logger.info("ThreadSafeBlurDetector initialized successfully")

        # Initialize quality detector with configurable thresholds
        quality_detector = ImageQualityDetector(
            overexposure_threshold=240.0,
            overexposure_percentage=5.0,
            underexposure_threshold=15.0,
            underexposure_percentage=10.0,
            oversaturation_threshold=0.9,
            oversaturation_percentage=15.0,
            undersaturation_threshold=0.2,
            undersaturation_percentage=60.0
        )
        logger.info("ImageQualityDetector initialized successfully")

        # Initialize blur type classifier
        blur_type_classifier = BlurTypeClassifier()
        logger.info("BlurTypeClassifier initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize detectors: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Image Quality Check API...")
    if cache_manager:
        cache_manager.close()


# Create FastAPI app
app = FastAPI(
    title="Image Quality Check API",
    description="""
    Comprehensive API for image quality analysis including:
    - **Blur detection** using advanced computer vision algorithms with blur type classification:
      * Motion blur (camera/subject movement)
      * Gaussian blur (out-of-focus)
      * Defocus blur (depth-of-field)
      * Mixed blur (combination of types)
    - **Overexposure detection** via histogram analysis
    - **Underexposure detection** via histogram analysis
    - **Oversaturation detection** using HSV color space analysis
    - **Undersaturation detection** using HSV color space analysis

    Supports both individual analysis and batch processing of images from URLs.

    **Supported Image Formats:** JPG, JPEG, PNG only

    **New Feature:** Blur type classification provides detailed analysis of blur characteristics
    with confidence scores and explanations for each detected blur type.
    """,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    calls_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
    calls_per_hour=int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def calculate_blur_metrics(blur_map: np.ndarray) -> Tuple[bool, float, float]:
    """
    Calculate blur metrics from the blur detection map.

    Args:
        blur_map: Blur detection map from BlurDetector

    Returns:
        tuple: (is_blurred, blur_score, confidence)
    """
    # Calculate blur score as the median of the blur map
    blur_score = float(np.median(blur_map))

    # Calculate confidence based on the consistency of the blur map
    # Higher standard deviation means less consistent, lower confidence
    std_dev = float(np.std(blur_map))
    confidence = max(0.0, min(1.0, 1.0 - (std_dev * 2)))  # Normalize std_dev to confidence

    # Determine if image is blurred based on threshold
    # Lower blur_score means more blurred (counter-intuitive but matches the algorithm)
    blur_threshold = 0.3
    is_blurred = blur_score < blur_threshold

    # Normalize blur_score to 0-1 range where 1 is most blurred
    normalized_blur_score = max(0.0, min(1.0, 1.0 - blur_score))

    return is_blurred, normalized_blur_score, confidence


def calculate_blur_metrics_with_type(blur_map: np.ndarray, image_gray: np.ndarray) -> Tuple[bool, float, float, BlurTypeDetails]:
    """
    Calculate blur metrics including blur type classification.

    Args:
        blur_map: Blur detection map from BlurDetector
        image_gray: Grayscale image for blur type analysis

    Returns:
        tuple: (is_blurred, blur_score, confidence, blur_type_details)
    """
    # Get basic blur metrics
    is_blurred, blur_score, confidence = calculate_blur_metrics(blur_map)

    # Classify blur type
    blur_type, type_confidence, analysis_details = blur_type_classifier.classify_blur_type(
        image_gray, blur_map, blur_score
    )

    # Create blur type details
    blur_type_details = BlurTypeDetails(
        blur_type=blur_type.value,
        type_confidence=type_confidence,
        motion_score=analysis_details.get('motion_score', 0.0),
        gaussian_score=analysis_details.get('gaussian_score', 0.0),
        defocus_score=analysis_details.get('defocus_score', 0.0),
        analysis_reason=analysis_details.get('reason', 'Blur type analysis completed')
    )

    return is_blurred, blur_score, confidence, blur_type_details


def calculate_overall_quality_score(quality_results: dict, blur_result: dict) -> float:
    """
    Calculate overall quality score based on all detection results.

    Args:
        quality_results: Dictionary of quality detection results
        blur_result: Blur detection result

    Returns:
        float: Overall quality score (0 = poor, 1 = excellent)
    """
    # Weight factors for different quality aspects
    weights = {
        'blur': 0.3,
        'overexposure': 0.2,
        'underexposure': 0.2,
        'oversaturation': 0.15,
        'undersaturation': 0.15
    }

    # Calculate weighted score (invert scores so higher = better quality)
    total_score = 0.0
    total_weight = 0.0

    # Blur score (invert since blur_score represents amount of blur)
    blur_quality = 1.0 - blur_result['blur_score']
    total_score += blur_quality * weights['blur']
    total_weight += weights['blur']

    # Other quality scores (invert since scores represent amount of issue)
    for quality_type, result in quality_results.items():
        if quality_type in weights:
            quality_score = 1.0 - result.score
            total_score += quality_score * weights[quality_type]
            total_weight += weights[quality_type]

    # Normalize to 0-1 range
    overall_score = total_score / total_weight if total_weight > 0 else 0.0
    return max(0.0, min(1.0, overall_score))


async def analyze_single_image(image_url: str) -> ImageAnalysisResponse:
    """
    Analyze a single image for blur detection.
    
    Args:
        image_url: URL of the image to analyze
        
    Returns:
        ImageAnalysisResponse: Analysis results
    """
    start_time = time.time()
    
    try:
        # Validate URL format
        if not is_valid_image_url(image_url):
            return ImageAnalysisResponse(
                success=False,
                image_url=image_url,
                result=None,
                error="URL does not appear to be a valid image URL"
            )
        
        # Download and process image
        api_logger.log_request("analyze_image", {"url": image_url})
        image = download_image_from_url(image_url)

        # Convert to grayscale if needed for blur detection
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        # Detect blur
        blur_map = blur_detector.detectBlur(image_gray)

        # Calculate metrics with blur type classification
        is_blurred, blur_score, confidence, blur_type_details = calculate_blur_metrics_with_type(blur_map, image_gray)

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        result = BlurResult(
            is_blurred=is_blurred,
            blur_score=blur_score,
            confidence=confidence,
            processing_time_ms=processing_time,
            blur_type_details=blur_type_details
        )

        # Log results
        api_logger.log_blur_detection(image_url, is_blurred, blur_score, confidence)
        api_logger.log_image_processing(image_url, True, processing_time / 1000)

        return ImageAnalysisResponse(
            success=True,
            image_url=image_url,
            result=result,
            error=None
        )
        
    except (ImageDownloadError, ImageValidationError) as e:
        processing_time = (time.time() - start_time) * 1000
        api_logger.log_image_processing(image_url, False, processing_time / 1000, str(e))
        return ImageAnalysisResponse(
            success=False,
            image_url=image_url,
            result=None,
            error=str(e)
        )
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        api_logger.log_error("analyze_image", e, {"url": image_url})
        api_logger.log_image_processing(image_url, False, processing_time / 1000, str(e))
        return ImageAnalysisResponse(
            success=False,
            image_url=image_url,
            result=None,
            error=f"Internal processing error: {str(e)}"
        )


async def analyze_comprehensive_quality_optimized(image_url: str) -> ComprehensiveAnalysisResponse:
    """
    Optimized comprehensive image quality analysis with caching and async I/O.

    Args:
        image_url: URL of the image to analyze

    Returns:
        ComprehensiveAnalysisResponse: Comprehensive analysis results
    """
    start_time = time.time()

    try:
        # Check cache first
        cached_result = cache_manager.get_cached_result(image_url, "comprehensive")
        if cached_result:
            logger.info(f"Returning cached result for {image_url}")
            return ComprehensiveAnalysisResponse(
                success=True,
                image_url=image_url,
                result=cached_result["result"],
                error=None
            )

        # Validate URL format
        if not is_valid_image_url(image_url):
            return ComprehensiveAnalysisResponse(
                success=False,
                image_url=image_url,
                result=None,
                error="URL does not appear to be a valid image URL"
            )

        # Download and process image asynchronously
        api_logger.log_request("analyze_comprehensive", {"url": image_url})
        image = await download_image_from_url_async(image_url)

        # Convert to grayscale for blur detection and BGR for quality analysis
        if len(image.shape) == 3:
            # Convert color image to grayscale for blur detection
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_bgr = image
        else:
            # Already grayscale
            image_gray = image
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Run blur detection and quality analysis in parallel using thread pool
        loop = asyncio.get_event_loop()

        # Create tasks for parallel execution
        blur_task = loop.run_in_executor(None, blur_detector.detectBlur, image_gray)
        quality_task = loop.run_in_executor(None, quality_detector.analyze_comprehensive_quality, image_bgr)

        # Wait for both tasks to complete
        results = await asyncio.gather(blur_task, quality_task)
        blur_map = results[0]
        quality_results = results[1]

        # Process results with blur type classification
        is_blurred, blur_score, blur_confidence, blur_type_details = calculate_blur_metrics_with_type(blur_map, image_gray)

        blur_processing_time = (time.time() - start_time) * 1000

        blur_result = BlurResult(
            is_blurred=is_blurred,
            blur_score=blur_score,
            confidence=blur_confidence,
            processing_time_ms=blur_processing_time,
            blur_type_details=blur_type_details
        )

        # Convert quality results to response format
        quality_response = {}
        detected_issues = []

        if is_blurred:
            detected_issues.append("blur")

        for quality_type, result in quality_results.items():
            quality_response[quality_type] = QualityDetectionResult(
                is_detected=result.is_detected,
                score=result.score,
                confidence=result.confidence,
                threshold_used=result.threshold_used,
                pixel_percentage=result.pixel_percentage
            )

            if result.is_detected:
                detected_issues.append(quality_type)

        # Calculate overall quality score
        overall_score = calculate_overall_quality_score(quality_results, {
            'blur_score': blur_score
        })

        total_processing_time = (time.time() - start_time) * 1000

        comprehensive_result = ComprehensiveQualityResult(
            blur=blur_result,
            overexposure=quality_response['overexposure'],
            underexposure=quality_response['underexposure'],
            oversaturation=quality_response['oversaturation'],
            undersaturation=quality_response['undersaturation'],
            overall_quality_score=overall_score,
            detected_issues=detected_issues,
            processing_time_ms=total_processing_time
        )

        # Cache the result
        cache_manager.cache_result(image_url, comprehensive_result.model_dump(), "comprehensive")

        # Log results
        api_logger.log_blur_detection(image_url, is_blurred, blur_score, blur_confidence)
        logger.info(f"Comprehensive analysis completed for {image_url}: {len(detected_issues)} issues detected")
        api_logger.log_image_processing(image_url, True, total_processing_time / 1000)

        return ComprehensiveAnalysisResponse(
            success=True,
            image_url=image_url,
            result=comprehensive_result,
            error=None
        )

    except (ImageDownloadError, ImageValidationError) as e:
        processing_time = (time.time() - start_time) * 1000
        api_logger.log_image_processing(image_url, False, processing_time / 1000, str(e))
        return ComprehensiveAnalysisResponse(
            success=False,
            image_url=image_url,
            result=None,
            error=str(e)
        )
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        api_logger.log_error("analyze_comprehensive", e, {"url": image_url})
        api_logger.log_image_processing(image_url, False, processing_time / 1000, str(e))
        return ComprehensiveAnalysisResponse(
            success=False,
            image_url=image_url,
            result=None,
            error=f"Internal processing error: {str(e)}"
        )


async def analyze_comprehensive_quality(image_url: str) -> ComprehensiveAnalysisResponse:
    """
    Perform comprehensive image quality analysis including blur and all quality detections.

    Args:
        image_url: URL of the image to analyze

    Returns:
        ComprehensiveAnalysisResponse: Comprehensive analysis results
    """
    start_time = time.time()

    try:
        # Validate URL format
        if not is_valid_image_url(image_url):
            return ComprehensiveAnalysisResponse(
                success=False,
                image_url=image_url,
                result=None,
                error="URL does not appear to be a valid image URL"
            )

        # Download and process image
        api_logger.log_request("analyze_comprehensive", {"url": image_url})
        image = download_image_from_url(image_url)

        # Convert grayscale to BGR for quality analysis if needed
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image

        # Perform blur detection
        blur_map = blur_detector.detectBlur(image)
        is_blurred, blur_score, blur_confidence = calculate_blur_metrics(blur_map)

        blur_processing_time = (time.time() - start_time) * 1000

        blur_result = BlurResult(
            is_blurred=is_blurred,
            blur_score=blur_score,
            confidence=blur_confidence,
            processing_time_ms=blur_processing_time
        )

        # Perform comprehensive quality analysis
        quality_results = quality_detector.analyze_comprehensive_quality(image_bgr)

        # Convert quality results to response format
        quality_response = {}
        detected_issues = []

        if is_blurred:
            detected_issues.append("blur")

        for quality_type, result in quality_results.items():
            quality_response[quality_type] = QualityDetectionResult(
                is_detected=result.is_detected,
                score=result.score,
                confidence=result.confidence,
                threshold_used=result.threshold_used,
                pixel_percentage=result.pixel_percentage
            )

            if result.is_detected:
                detected_issues.append(quality_type)

        # Calculate overall quality score
        overall_score = calculate_overall_quality_score(quality_results, {
            'blur_score': blur_score
        })

        total_processing_time = (time.time() - start_time) * 1000

        comprehensive_result = ComprehensiveQualityResult(
            blur=blur_result,
            overexposure=quality_response['overexposure'],
            underexposure=quality_response['underexposure'],
            oversaturation=quality_response['oversaturation'],
            undersaturation=quality_response['undersaturation'],
            overall_quality_score=overall_score,
            detected_issues=detected_issues,
            processing_time_ms=total_processing_time
        )

        # Log results
        api_logger.log_blur_detection(image_url, is_blurred, blur_score, blur_confidence)
        logger.info(f"Comprehensive analysis completed for {image_url}: {len(detected_issues)} issues detected")
        api_logger.log_image_processing(image_url, True, total_processing_time / 1000)

        return ComprehensiveAnalysisResponse(
            success=True,
            image_url=image_url,
            result=comprehensive_result,
            error=None
        )

    except (ImageDownloadError, ImageValidationError) as e:
        processing_time = (time.time() - start_time) * 1000
        api_logger.log_image_processing(image_url, False, processing_time / 1000, str(e))
        return ComprehensiveAnalysisResponse(
            success=False,
            image_url=image_url,
            result=None,
            error=str(e)
        )
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        api_logger.log_error("analyze_comprehensive", e, {"url": image_url})
        api_logger.log_image_processing(image_url, False, processing_time / 1000, str(e))
        return ComprehensiveAnalysisResponse(
            success=False,
            image_url=image_url,
            result=None,
            error=f"Internal processing error: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app_start_time
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        uptime_seconds=uptime
    )


@app.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics and performance metrics.

    Returns:
        Cache statistics and hit rates
    """
    if not cache_manager:
        return {"error": "Cache manager not initialized"}

    return cache_manager.get_cache_stats()


@app.post("/cache/clear")
async def clear_cache():
    """
    Clear all cached analysis results.

    Returns:
        Success status
    """
    if not cache_manager:
        return {"error": "Cache manager not initialized"}

    success = cache_manager.clear_all_cache()
    return {"success": success, "message": "Cache cleared" if success else "Failed to clear cache"}


@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(request: ImageRequest):
    """
    Analyze a single image for blur detection with blur type classification.

    This endpoint provides comprehensive blur analysis including:
    - Basic blur detection (is_blurred, blur_score, confidence)
    - Detailed blur type classification (motion, gaussian, defocus, mixed)
    - Individual blur characteristic scores
    - Analysis explanation and reasoning

    Args:
        request: Image analysis request containing the image URL

    Returns:
        ImageAnalysisResponse: Analysis results with blur type details
    """
    return await analyze_single_image(str(request.image_url))


@app.post("/analyze-quality", response_model=ComprehensiveAnalysisResponse)
async def analyze_quality(request: ImageRequest):
    """
    Perform comprehensive image quality analysis including blur, exposure, and saturation.

    Args:
        request: Image analysis request containing the image URL

    Returns:
        ComprehensiveAnalysisResponse: Comprehensive quality analysis results
    """
    return await analyze_comprehensive_quality_optimized(str(request.image_url))


@app.post("/analyze-batch", response_model=BatchAnalysisResponse)
async def analyze_batch(request: BatchImageRequest):
    """
    Analyze multiple images for blur detection (backward compatibility).

    Args:
        request: Batch analysis request containing multiple image URLs

    Returns:
        BatchAnalysisResponse: Batch analysis results
    """
    results = []
    successful = 0
    failed = 0

    for image_url in request.image_urls:
        result = await analyze_single_image(str(image_url))
        results.append(result)

        if result.success:
            successful += 1
        else:
            failed += 1

    return BatchAnalysisResponse(
        total_processed=len(request.image_urls),
        successful=successful,
        failed=failed,
        results=results
    )


@app.post("/analyze-quality-batch", response_model=ComprehensiveBatchAnalysisResponse)
async def analyze_quality_batch(request: BatchImageRequest):
    """
    Perform comprehensive quality analysis on multiple images with parallel processing.

    Args:
        request: Batch analysis request containing multiple image URLs

    Returns:
        ComprehensiveBatchAnalysisResponse: Comprehensive batch analysis results
    """
    # Process images in parallel with concurrency limit
    max_concurrent = min(5, len(request.image_urls))  # Limit concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_image(image_url: str) -> ComprehensiveAnalysisResponse:
        async with semaphore:
            return await analyze_comprehensive_quality_optimized(image_url)

    # Create tasks for parallel processing
    tasks = [process_single_image(str(url)) for url in request.image_urls]

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and handle exceptions
    processed_results = []
    successful = 0
    failed = 0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Handle exceptions from individual tasks
            error_result = ComprehensiveAnalysisResponse(
                success=False,
                image_url=str(request.image_urls[i]),
                result=None,
                error=f"Processing error: {str(result)}"
            )
            processed_results.append(error_result)
            failed += 1
        else:
            processed_results.append(result)
            if result.success:
                successful += 1
            else:
                failed += 1

    logger.info(f"Batch processing completed: {successful} successful, {failed} failed")

    return ComprehensiveBatchAnalysisResponse(
        total_processed=len(request.image_urls),
        successful=successful,
        failed=failed,
        results=processed_results
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
