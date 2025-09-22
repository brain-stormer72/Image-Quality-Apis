"""
Pydantic models for the Image Quality Check API.
"""

from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List
from enum import Enum


class ImageRequest(BaseModel):
    """Request model for single image blur detection."""
    image_url: HttpUrl = Field(..., description="URL of the image to analyze for blur (supports JPG, JPEG, PNG only)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_url": "https://example.com/image.jpg"
            }
        }


class BatchImageRequest(BaseModel):
    """Request model for batch image blur detection."""
    image_urls: List[HttpUrl] = Field(..., min_items=1, max_items=10, description="List of image URLs to analyze (max 10, supports JPG, JPEG, PNG only)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_urls": [
                    "https://example.com/image1.jpg",
                    "https://example.com/image2.png"
                ]
            }
        }


class QualityDetectionResult(BaseModel):
    """Response model for individual quality detection results."""
    is_detected: bool = Field(..., description="Whether the quality issue is detected")
    score: float = Field(..., ge=0.0, le=1.0, description="Quality issue score between 0 (no issue) and 1 (severe issue)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level of the detection")
    threshold_used: float = Field(..., description="Threshold value used for detection")
    pixel_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of pixels contributing to the detection")


class BlurTypeDetails(BaseModel):
    """Details about blur type classification."""
    blur_type: str = Field(..., description="Type of blur detected (sharp, motion_blur, gaussian_blur, defocus_blur, mixed_blur)")
    type_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in blur type classification")
    motion_score: float = Field(..., ge=0.0, le=1.0, description="Motion blur characteristics score")
    gaussian_score: float = Field(..., ge=0.0, le=1.0, description="Gaussian blur characteristics score")
    defocus_score: float = Field(..., ge=0.0, le=1.0, description="Defocus blur characteristics score")
    analysis_reason: str = Field(..., description="Explanation of blur type classification")


class BlurResult(BaseModel):
    """Response model for blur detection results (backward compatibility)."""
    is_blurred: bool = Field(..., description="Whether the image is detected as blurred")
    blur_score: float = Field(..., ge=0.0, le=1.0, description="Blur score between 0 (sharp) and 1 (blurred)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level of the detection")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    blur_type_details: Optional[BlurTypeDetails] = Field(None, description="Detailed blur type classification information")

    class Config:
        json_schema_extra = {
            "example": {
                "is_blurred": True,
                "blur_score": 0.75,
                "confidence": 0.92,
                "processing_time_ms": 1250.5,
                "blur_type_details": {
                    "blur_type": "motion_blur",
                    "type_confidence": 0.82,
                    "motion_score": 0.78,
                    "gaussian_score": 0.23,
                    "defocus_score": 0.15,
                    "analysis_reason": "Dominant motion_blur characteristics"
                }
            }
        }


class ComprehensiveQualityResult(BaseModel):
    """Response model for comprehensive image quality analysis."""
    blur: BlurResult = Field(..., description="Blur detection results")
    overexposure: QualityDetectionResult = Field(..., description="Overexposure detection results")
    underexposure: QualityDetectionResult = Field(..., description="Underexposure detection results")
    oversaturation: QualityDetectionResult = Field(..., description="Oversaturation detection results")
    undersaturation: QualityDetectionResult = Field(..., description="Undersaturation detection results")
    overall_quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score (0 = poor, 1 = excellent)")
    detected_issues: List[str] = Field(..., description="List of detected quality issues")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "blur": {
                    "is_blurred": False,
                    "blur_score": 0.25,
                    "confidence": 0.88,
                    "processing_time_ms": 1250.5
                },
                "overexposure": {
                    "is_detected": True,
                    "score": 0.75,
                    "confidence": 0.92,
                    "threshold_used": 240.0,
                    "pixel_percentage": 8.5
                },
                "underexposure": {
                    "is_detected": False,
                    "score": 0.15,
                    "confidence": 0.85,
                    "threshold_used": 15.0,
                    "pixel_percentage": 3.2
                },
                "oversaturation": {
                    "is_detected": False,
                    "score": 0.30,
                    "confidence": 0.78,
                    "threshold_used": 0.9,
                    "pixel_percentage": 12.1
                },
                "undersaturation": {
                    "is_detected": False,
                    "score": 0.40,
                    "confidence": 0.82,
                    "threshold_used": 0.2,
                    "pixel_percentage": 45.3
                },
                "overall_quality_score": 0.65,
                "detected_issues": ["overexposure"],
                "processing_time_ms": 2150.8
            }
        }


class ImageAnalysisResponse(BaseModel):
    """Response model for single image analysis (backward compatibility)."""
    success: bool = Field(..., description="Whether the analysis was successful")
    image_url: str = Field(..., description="The analyzed image URL")
    result: Optional[BlurResult] = Field(None, description="Blur detection results if successful")
    error: Optional[str] = Field(None, description="Error message if analysis failed")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "image_url": "https://example.com/image.jpg",
                "result": {
                    "is_blurred": True,
                    "blur_score": 0.75,
                    "confidence": 0.92,
                    "processing_time_ms": 1250.5
                },
                "error": None
            }
        }


class ComprehensiveAnalysisResponse(BaseModel):
    """Response model for comprehensive image quality analysis."""
    success: bool = Field(..., description="Whether the analysis was successful")
    image_url: str = Field(..., description="The analyzed image URL")
    result: Optional[ComprehensiveQualityResult] = Field(None, description="Comprehensive quality analysis results if successful")
    error: Optional[str] = Field(None, description="Error message if analysis failed")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "image_url": "https://example.com/image.jpg",
                "result": {
                    "blur": {
                        "is_blurred": False,
                        "blur_score": 0.25,
                        "confidence": 0.88,
                        "processing_time_ms": 1250.5
                    },
                    "overexposure": {
                        "is_detected": True,
                        "score": 0.75,
                        "confidence": 0.92,
                        "threshold_used": 240.0,
                        "pixel_percentage": 8.5
                    },
                    "underexposure": {
                        "is_detected": False,
                        "score": 0.15,
                        "confidence": 0.85,
                        "threshold_used": 15.0,
                        "pixel_percentage": 3.2
                    },
                    "oversaturation": {
                        "is_detected": False,
                        "score": 0.30,
                        "confidence": 0.78,
                        "threshold_used": 0.9,
                        "pixel_percentage": 12.1
                    },
                    "undersaturation": {
                        "is_detected": False,
                        "score": 0.40,
                        "confidence": 0.82,
                        "threshold_used": 0.2,
                        "pixel_percentage": 45.3
                    },
                    "overall_quality_score": 0.65,
                    "detected_issues": ["overexposure"],
                    "processing_time_ms": 2150.8
                },
                "error": None
            }
        }


class BatchAnalysisResponse(BaseModel):
    """Response model for batch image analysis (backward compatibility)."""
    total_processed: int = Field(..., description="Total number of images processed")
    successful: int = Field(..., description="Number of successfully processed images")
    failed: int = Field(..., description="Number of failed image analyses")
    results: List[ImageAnalysisResponse] = Field(..., description="Individual analysis results")


class ComprehensiveBatchAnalysisResponse(BaseModel):
    """Response model for comprehensive batch image analysis."""
    total_processed: int = Field(..., description="Total number of images processed")
    successful: int = Field(..., description="Number of successfully processed images")
    failed: int = Field(..., description="Number of failed image analyses")
    results: List[ComprehensiveAnalysisResponse] = Field(..., description="Individual comprehensive analysis results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_processed": 2,
                "successful": 1,
                "failed": 1,
                "results": [
                    {
                        "success": True,
                        "image_url": "https://example.com/image1.jpg",
                        "result": {
                            "is_blurred": False,
                            "blur_score": 0.25,
                            "confidence": 0.88,
                            "processing_time_ms": 980.2
                        },
                        "error": None
                    },
                    {
                        "success": False,
                        "image_url": "https://example.com/invalid.jpg",
                        "result": None,
                        "error": "Failed to download image: HTTP 404"
                    }
                ]
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": 3600.5
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid image URL",
                "detail": "The provided URL is not accessible or does not contain a valid image"
            }
        }
