"""
FastAPI application for Image Quality Check API.
Provides endpoints for blur detection in images from URLs.
"""

import time
import logging
import numpy as np
import cv2
import asyncio
from contextlib import asynccontextmanager
from typing import List, Tuple

from fastapi import FastAPI, HTTPException, status, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from constants import LOG_LEVEL, LOG_FILE, RATE_LIMIT_PER_MINUTE, RATE_LIMIT_PER_HOUR
from models import (
    ImageRequest, BatchImageRequest, ImageAnalysisResponse,
    BatchAnalysisResponse, BlurResult, BlurTypeDetails, HealthResponse, ErrorResponse,
    ComprehensiveAnalysisResponse, ComprehensiveBatchAnalysisResponse,
    ComprehensiveQualityResult, QualityDetectionResult
)
from image_utils import (
    download_image_from_url, download_image_from_url_async, ImageDownloadError, ImageValidationError,
    is_valid_image_url, cleanup_temp_file, _process_image_data_sync
)
from BlurDetector import BlurDetector
from quality_detectors import ImageQualityDetector
from blur_type_classifier import BlurTypeClassifier, BlurType
from logging_config import setup_logging, api_logger
from middleware import RateLimitMiddleware, RequestLoggingMiddleware, SecurityHeadersMiddleware

# Import performance optimization modules (no caching)
from performance_monitor import get_performance_monitor, track_performance
# Removed get_resizing_performance_summary as part of simplification


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
log_level = LOG_LEVEL  # Back to INFO for production
log_file = LOG_FILE
setup_logging(log_level, log_file)

logger = logging.getLogger(__name__)

# Global variables
blur_detector = None
quality_detector = None
blur_type_classifier = None
app_start_time = time.time()
API_VERSION = "1.2.1"  # Updated version for performance optimizations
thread_pool_executor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global blur_detector, quality_detector, blur_type_classifier

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

        # Initialize performance monitor (no caching)
        performance_monitor = get_performance_monitor()
        logger.info("Performance monitoring initialized")

    except Exception as e:
        logger.error(f"Failed to initialize detectors: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Image Quality Check API...")

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
    calls_per_minute=RATE_LIMIT_PER_MINUTE,
    calls_per_hour=RATE_LIMIT_PER_HOUR
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def calculate_blur_metrics(blur_result: dict) -> Tuple[bool, float, float]:
    """
    Calculate blur metrics from the blur detection result.

    Args:
        blur_result (dict): Blur detection result from BlurDetector

    Returns:
        Tuple[bool, float, float]: (is_blurred, blur_score, confidence)
    """
    # Extract values from the blur result dictionary
    is_blurred = blur_result.get('is_blurry', False)
    blur_score = blur_result.get('blur_score', 0.0)
    confidence = blur_result.get('confidence', 0.5)
    
    return is_blurred, blur_score, confidence


def calculate_blur_metrics_with_type(blur_result: dict, image_gray: np.ndarray) -> Tuple[bool, float, float, BlurTypeDetails]:
    """
    Calculate blur metrics including blur type classification.

    Args:
        blur_result: Blur detection result from BlurDetector
        image_gray: Grayscale image for blur type analysis

    Returns:
        tuple: (is_blurred, blur_score, confidence, blur_type_details)
    """
    # Get basic blur metrics
    is_blurred, blur_score, confidence = calculate_blur_metrics(blur_result)

    # Classify blur type - use the blur result for classification
    blur_type, type_confidence, analysis_details = blur_type_classifier.classify_blur_type(
        image_gray, blur_result, blur_score
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


def calculate_overall_quality_score(
    quality_results: dict,
    blur_result: dict
) -> float:
    """
    Calculate overall quality score based on all detection results.

    Args:
        quality_results (dict): Dictionary of quality detection results
        blur_result (dict): Blur detection result

    Returns:
        float: Overall quality score (0 = poor, 1 = excellent)
    """
    weights = {
        'blur': 0.3,
        'overexposure': 0.2,
        'underexposure': 0.2,
        'oversaturation': 0.15,
        'undersaturation': 0.15
    }
    total_score = 0.0
    total_weight = 0.0
    blur_quality = 1.0 - blur_result['blur_score']
    total_score += blur_quality * weights['blur']
    total_weight += weights['blur']
    for quality_type, result in quality_results.items():
        if quality_type in weights:
            quality_score = 1.0 - result.score
            total_score += quality_score * weights[quality_type]
            total_weight += weights[quality_type]
    overall_score = total_score / total_weight if total_weight > 0 else 0.0
    return max(0.0, min(1.0, overall_score))


async def analyze_single_image(image_url: str) -> ImageAnalysisResponse:
    """
    Analyze a single image for blur detection with intelligent resizing optimization.

    Args:
        image_url: URL of the image to analyze

    Returns:
        ImageAnalysisResponse: Analysis results
    """
    with track_performance("analyze_single_image"):
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
            blur_result = blur_detector.detectBlur(image_gray)

            # Calculate metrics with blur type classification
            is_blurred, blur_score, confidence, blur_type_details = calculate_blur_metrics_with_type(blur_result, image_gray)

            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            result = BlurResult(
                is_blurred=is_blurred,
                blur_score=blur_score,
                confidence=confidence,
                processing_time_ms=processing_time,
                blur_type_details=blur_type_details
            )

            # No caching - direct processing for optimal performance

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
    Optimized comprehensive image quality analysis with intelligent resizing and async I/O.

    Args:
        image_url: URL of the image to analyze

    Returns:
        ComprehensiveAnalysisResponse: Comprehensive analysis results
    """
    with track_performance("analyze_comprehensive_optimized"):
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
            blur_result = results[0]
            quality_results = results[1]

            # Process results with blur type classification
            is_blurred, blur_score, blur_confidence, blur_type_details = calculate_blur_metrics_with_type(blur_result, image_gray)

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

            # No caching - optimized processing with intelligent resizing

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
        blur_result = blur_detector.detectBlur(image)
        is_blurred, blur_score, blur_confidence = calculate_blur_metrics(blur_result)

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


@app.get("/optimization/stats")
async def get_optimization_stats():
    """
    Get optimization statistics and performance metrics (no caching).

    Returns:
        Optimization statistics including resizing metrics
    """
    resizing_summary = {"message": "Simplified resizing - no detailed metrics tracking"}
    return {
        "resizing_optimization": resizing_summary,
        "caching_enabled": False,
        "message": "Performance optimization without caching"
    }


@app.get("/performance", tags=["Performance"])
async def get_performance_metrics():
    """
    Get performance metrics and statistics.

    Returns:
        Dict: Performance metrics including processing times and resizing metrics
    """
    performance_monitor = get_performance_monitor()
    performance_summary = performance_monitor.get_performance_summary()
    resizing_summary = {"message": "Simplified resizing - no detailed metrics tracking"}

    return {
        "performance_metrics": performance_summary,
        "resizing_optimization": resizing_summary,
        "timestamp": time.time()
    }


@app.get("/performance/recent", tags=["Performance"])
async def get_recent_performance(limit: int = 10):
    """
    Get recent performance metrics.

    Args:
        limit: Number of recent metrics to return (default: 10, max: 50)

    Returns:
        Dict: Recent performance metrics
    """
    limit = min(max(1, limit), 50)  # Clamp between 1 and 50
    performance_monitor = get_performance_monitor()

    recent_metrics = performance_monitor.get_recent_performance(limit)

    return {
        "recent_metrics": recent_metrics,
        "count": len(recent_metrics),
        "timestamp": time.time()
    }


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
    Analyze multiple images for blur detection with optimized parallel processing.

    Args:
        request: Batch analysis request containing multiple image URLs

    Returns:
        BatchAnalysisResponse: Batch analysis results
    """
    with track_performance("analyze_batch", (len(request.image_urls), 0)):
        # Process images in parallel with controlled concurrency
        semaphore = asyncio.Semaphore(5)  # Limit concurrent processing

        async def process_single_image(image_url):
            async with semaphore:
                return await analyze_single_image(str(image_url))

        # Create tasks for parallel processing
        tasks = [process_single_image(url) for url in request.image_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        successful = 0
        failed = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exceptions
                error_result = ImageAnalysisResponse(
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

        return BatchAnalysisResponse(
            total_processed=len(request.image_urls),
            successful=successful,
            failed=failed,
            results=processed_results
        )


@app.post("/analyze-quality-batch", response_model=ComprehensiveBatchAnalysisResponse)
async def analyze_quality_batch(request: BatchImageRequest):
    """
    Perform comprehensive quality analysis on multiple images with optimized parallel processing.

    Args:
        request: Batch analysis request containing multiple image URLs

    Returns:
        ComprehensiveBatchAnalysisResponse: Comprehensive batch analysis results
    """
    with track_performance("analyze_quality_batch", (len(request.image_urls), 0)):
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


# Enhanced async operations with optimized concurrency
async def analyze_image_data_optimized(image_data: bytes, 
                             enable_blur_detection: bool = True,
                             enable_quality_analysis: bool = True,
                             enable_comprehensive: bool = False) -> dict:
    """
    Analyze a single image with optimized async processing.
    Enhanced with better concurrency and reduced blocking operations.
    
    Args:
        image_data: Raw image bytes
        enable_blur_detection: Enable blur detection
        enable_quality_analysis: Enable quality analysis
        enable_comprehensive: Enable comprehensive analysis
        
    Returns:
        dict: Analysis results with performance metrics
    """
    start_time = time.time()
    
    try:
        # Optimize image processing with intelligent resizing
        loop = asyncio.get_event_loop()
        
        # Process image data in thread pool to avoid blocking
        processed_image = await loop.run_in_executor(
            thread_pool_executor,
            _process_image_data_sync,
            image_data,
            "image.jpg"  # Default filename for processing
        )
        
        if processed_image is None:
            return {
                'error': 'Failed to process image data',
                'processing_time': time.time() - start_time
            }
        
        # Prepare concurrent tasks with optimized execution
        tasks = []
        
        if enable_blur_detection:
            # Use optimized blur detection with early termination
            blur_task = loop.run_in_executor(
                thread_pool_executor,
                lambda: blur_detector.detectBlur(processed_image, threshold=0.5, verbose=False)
            )
            tasks.append(('blur', blur_task))
        
        if enable_quality_analysis:
            if enable_comprehensive:
                # Use optimized comprehensive analysis with parallel processing
                quality_task = loop.run_in_executor(
                    thread_pool_executor,
                    lambda: analyze_comprehensive_quality(
                        processed_image,
                        enable_blur_detection=False,  # Avoid duplicate blur detection
                        enable_exposure_detection=True,
                        enable_noise_detection=True,
                        blur_detector=None
                    )
                )
            else:
                # Basic quality analysis with vectorized operations
                quality_task = loop.run_in_executor(
                    thread_pool_executor,
                    lambda: {
                        'overexposure': detect_overexposure(processed_image),
                        'underexposure': detect_underexposure(processed_image),
                        'noise': detect_noise(processed_image)
                    }
                )
            tasks.append(('quality', quality_task))
        
        # Execute all tasks concurrently with timeout protection
        results = {}
        if tasks:
            try:
                # Use asyncio.wait_for with timeout to prevent hanging
                completed_tasks = await asyncio.wait_for(
                    asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
                    timeout=30.0  # 30 second timeout
                )
                
                # Process results
                for i, (task_name, _) in enumerate(tasks):
                    if i < len(completed_tasks):
                        task_result = completed_tasks[i]
                        if isinstance(task_result, Exception):
                            results[task_name] = {
                                'error': str(task_result),
                                'status': 'failed'
                            }
                        else:
                            results[task_name] = task_result
                            
            except asyncio.TimeoutError:
                results['error'] = 'Analysis timeout after 30 seconds'
                results['timeout'] = True
        
        # Calculate overall metrics with optimization info
        processing_time = time.time() - start_time
        
        # Add performance and optimization metrics
        results['performance'] = {
            'processing_time': processing_time,
            'image_dimensions': processed_image.shape[:2] if processed_image is not None else None,
            'concurrent_tasks': len(tasks),
            'optimizations_applied': [
                'intelligent_resizing',
                'early_termination',
                'vectorized_operations',
                'parallel_processing',
                'timeout_protection'
            ]
        }
        
        return results
        
    except Exception as e:
        return {
            'error': str(e),
            'processing_time': time.time() - start_time,
            'status': 'failed'
        }


async def analyze_batch_images(image_files: List[UploadFile],
                             enable_blur_detection: bool = True,
                             enable_quality_analysis: bool = True,
                             max_concurrent: int = 5) -> dict:
    """
    Analyze multiple images in batch with optimized concurrency control.
    Enhanced with adaptive concurrency and memory management.
    
    Args:
        image_files: List of uploaded image files
        enable_blur_detection: Enable blur detection
        enable_quality_analysis: Enable quality analysis
        max_concurrent: Maximum concurrent analyses (adaptive)
        
    Returns:
        dict: Batch analysis results with performance metrics
    """
    start_time = time.time()
    
    try:
        # Adaptive concurrency based on system resources and image count
        optimal_concurrency = min(
            max_concurrent,
            len(image_files),
            max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU core free
        )
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(optimal_concurrency)
        
        async def process_single_image(file: UploadFile, index: int):
            """Process a single image with semaphore control."""
            async with semaphore:
                try:
                    # Read image data efficiently
                    image_data = await file.read()
                    
                    # Reset file pointer for potential reuse
                    await file.seek(0)
                    
                    # Analyze image with optimizations
                    result = await analyze_single_image(
                        image_data,
                        enable_blur_detection=enable_blur_detection,
                        enable_quality_analysis=enable_quality_analysis,
                        enable_comprehensive=False  # Use basic analysis for batch processing
                    )
                    
                    # Add file metadata
                    result['file_info'] = {
                        'filename': file.filename,
                        'index': index,
                        'size_bytes': len(image_data)
                    }
                    
                    return result
                    
                except Exception as e:
                    return {
                        'error': str(e),
                        'file_info': {
                            'filename': file.filename,
                            'index': index
                        },
                        'status': 'failed'
                    }
        
        # Process all images concurrently with progress tracking
        tasks = [
            process_single_image(file, i) 
            for i, file in enumerate(image_files)
        ]
        
        # Execute with timeout and exception handling
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=120.0  # 2 minute timeout for batch processing
            )
        except asyncio.TimeoutError:
            return {
                'error': 'Batch processing timeout after 2 minutes',
                'timeout': True,
                'processing_time': time.time() - start_time
            }
        
        # Process and categorize results
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append({
                    'error': str(result),
                    'status': 'exception'
                })
            elif 'error' in result:
                failed_results.append(result)
            else:
                successful_results.append(result)
        
        # Calculate batch statistics
        total_processing_time = time.time() - start_time
        avg_processing_time = (
            sum(r.get('performance', {}).get('processing_time', 0) 
                for r in successful_results) / len(successful_results)
            if successful_results else 0
        )
        
        return {
            'batch_results': {
                'successful': successful_results,
                'failed': failed_results,
                'total_images': len(image_files),
                'successful_count': len(successful_results),
                'failed_count': len(failed_results)
            },
            'performance': {
                'total_processing_time': total_processing_time,
                'average_processing_time': avg_processing_time,
                'concurrent_limit': optimal_concurrency,
                'throughput_images_per_second': len(successful_results) / total_processing_time if total_processing_time > 0 else 0
            },
            'optimizations': {
                'adaptive_concurrency': True,
                'semaphore_control': True,
                'timeout_protection': True,
                'memory_efficient': True
            }
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'processing_time': time.time() - start_time,
            'status': 'batch_failed'
        }


# Optimized thread pool configuration
def configure_thread_pool():
    """Configure optimized thread pool for CPU-bound operations."""
    global thread_pool_executor
    
    # Calculate optimal thread count based on CPU cores and workload
    cpu_count = multiprocessing.cpu_count()
    
    # For CPU-bound image processing, use fewer threads than CPU cores
    # to avoid context switching overhead
    optimal_threads = max(2, min(cpu_count - 1, 8))  # Cap at 8 threads
    
    thread_pool_executor = ThreadPoolExecutor(
        max_workers=optimal_threads,
        thread_name_prefix="ImageProcessor"
    )
    logger.info(f"Configured thread pool with {optimal_threads} workers")
    return thread_pool_executor


# Enhanced startup with optimized configuration
@app.on_event("startup")
async def startup_event():
    """Enhanced startup with performance optimizations."""
    global blur_detector, thread_pool_executor
    
    logger.info("Starting Image Quality API with performance optimizations...")
    
    # Configure optimized thread pool
    configure_thread_pool()
    
    # Initialize thread-safe blur detector with optimizations
    try:
        blur_detector = ThreadSafeBlurDetector()
        logger.info("Initialized optimized thread-safe blur detector")
    except Exception as e:
        logger.error(f"Failed to initialize blur detector: {e}")
        raise
    
    # Warm up the system with a small test image
    try:
        # Create a small test image for warmup
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Warm up blur detector
        _ = blur_detector.detectBlur(test_image)
        
        # Warm up quality detectors
        _ = detect_overexposure(test_image)
        _ = detect_underexposure(test_image)
        _ = detect_noise(test_image)
        
        logger.info("System warmup completed successfully")
        
    except Exception as e:
        logger.warning(f"System warmup failed: {e}")
    
    logger.info("Image Quality API startup completed with optimizations")


@app.on_event("shutdown")
async def shutdown_event():
    """Enhanced shutdown with proper resource cleanup."""
    global thread_pool_executor
    
    logger.info("Shutting down Image Quality API...")
    
    # Shutdown thread pool gracefully
    if thread_pool_executor:
        thread_pool_executor.shutdown(wait=True, timeout=10.0)
        logger.info("Thread pool shutdown completed")
    
    logger.info("Image Quality API shutdown completed")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app_start_time
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        uptime_seconds=uptime
    )


@app.get("/optimization/stats")
async def get_optimization_stats():
    """
    Get optimization statistics and performance metrics (no caching).

    Returns:
        Optimization statistics including resizing metrics
    """
    resizing_summary = {"message": "Simplified resizing - no detailed metrics tracking"}
    return {
        "resizing_optimization": resizing_summary,
        "caching_enabled": False,
        "message": "Performance optimization without caching"
    }


@app.get("/performance", tags=["Performance"])
async def get_performance_metrics():
    """
    Get performance metrics and statistics.

    Returns:
        Dict: Performance metrics including processing times and resizing metrics
    """
    performance_monitor = get_performance_monitor()
    performance_summary = performance_monitor.get_performance_summary()
    resizing_summary = {"message": "Simplified resizing - no detailed metrics tracking"}

    return {
        "performance_metrics": performance_summary,
        "resizing_optimization": resizing_summary,
        "timestamp": time.time()
    }


@app.get("/performance/recent", tags=["Performance"])
async def get_recent_performance(limit: int = 10):
    """
    Get recent performance metrics.

    Args:
        limit: Number of recent metrics to return (default: 10, max: 50)

    Returns:
        Dict: Recent performance metrics
    """
    limit = min(max(1, limit), 50)  # Clamp between 1 and 50
    performance_monitor = get_performance_monitor()

    recent_metrics = performance_monitor.get_recent_performance(limit)

    return {
        "recent_metrics": recent_metrics,
        "count": len(recent_metrics),
        "timestamp": time.time()
    }


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
    Analyze multiple images for blur detection with optimized parallel processing.

    Args:
        request: Batch analysis request containing multiple image URLs

    Returns:
        BatchAnalysisResponse: Batch analysis results
    """
    with track_performance("analyze_batch", (len(request.image_urls), 0)):
        # Process images in parallel with controlled concurrency
        semaphore = asyncio.Semaphore(5)  # Limit concurrent processing

        async def process_single_image(image_url):
            async with semaphore:
                return await analyze_single_image(str(image_url))

        # Create tasks for parallel processing
        tasks = [process_single_image(url) for url in request.image_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        successful = 0
        failed = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exceptions
                error_result = ImageAnalysisResponse(
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

        return BatchAnalysisResponse(
            total_processed=len(request.image_urls),
            successful=successful,
            failed=failed,
            results=processed_results
        )


@app.post("/analyze-quality-batch", response_model=ComprehensiveBatchAnalysisResponse)
async def analyze_quality_batch(request: BatchImageRequest):
    """
    Perform comprehensive quality analysis on multiple images with optimized parallel processing.

    Args:
        request: Batch analysis request containing multiple image URLs

    Returns:
        ComprehensiveBatchAnalysisResponse: Comprehensive batch analysis results
    """
    with track_performance("analyze_quality_batch", (len(request.image_urls), 0)):
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
