# Image Quality Check API

A comprehensive FastAPI-based web service for advanced image quality analysis. This API accepts image URLs and performs multiple quality assessments including blur detection, exposure analysis, and saturation evaluation.

## Features

### Core Quality Detection
- **Advanced Blur Detection**: Uses spatially-varying blur detection based on multiscale fused and sorted transform coefficients of gradient magnitudes
- **Overexposure Detection**: Histogram analysis to detect overly bright images with excessive high-intensity pixels
- **Underexposure Detection**: Histogram analysis to identify overly dark images with excessive low-intensity pixels
- **Oversaturation Detection**: HSV color space analysis to detect unnaturally high color saturation
- **Undersaturation Detection**: HSV color space analysis to identify washed-out or low-vibrancy images

### API Features
- **REST API**: Clean RESTful endpoints with comprehensive error handling
- **Dual Analysis Modes**: Both legacy blur-only and comprehensive quality analysis
- **Batch Processing**: Analyze multiple images in a single request
- **Rate Limiting**: Built-in protection against API abuse
- **Comprehensive Logging**: Detailed logging for monitoring and debugging
- **Health Monitoring**: Health check endpoint for service monitoring
- **OpenAPI Documentation**: Automatic Swagger/ReDoc documentation
- **Security Headers**: Built-in security middleware
- **HEIC Support**: Full support for HEIC/HEIF image formats

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd image-quality-check-apis
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

The API will be available at `http://localhost:8000`

## API Documentation

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Endpoints

#### 1. Health Check

**GET** `/health`

Returns the current status of the API service.

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600.5
}
```

#### 2. Analyze Single Image (Blur Only)

**POST** `/analyze-image`

Analyzes a single image for blur detection only (backward compatibility).

**Request Body**:
```json
{
  "image_url": "https://example.com/image.jpg"
}
```

**Response**:
```json
{
  "success": true,
  "image_url": "https://example.com/image.jpg",
  "result": {
    "is_blurred": true,
    "blur_score": 0.75,
    "confidence": 0.92,
    "processing_time_ms": 1250.5
  },
  "error": null
}
```

#### 3. Comprehensive Quality Analysis

**POST** `/analyze-quality`

Performs comprehensive image quality analysis including blur, exposure, and saturation detection.

**Request Body**:
```json
{
  "image_url": "https://example.com/image.jpg"
}
```

**Response**:
```json
{
  "success": true,
  "image_url": "https://example.com/image.jpg",
  "result": {
    "blur": {
      "is_blurred": false,
      "blur_score": 0.25,
      "confidence": 0.88,
      "processing_time_ms": 1250.5
    },
    "overexposure": {
      "is_detected": true,
      "score": 0.75,
      "confidence": 0.92,
      "threshold_used": 240.0,
      "pixel_percentage": 8.5
    },
    "underexposure": {
      "is_detected": false,
      "score": 0.15,
      "confidence": 0.85,
      "threshold_used": 15.0,
      "pixel_percentage": 3.2
    },
    "oversaturation": {
      "is_detected": false,
      "score": 0.30,
      "confidence": 0.78,
      "threshold_used": 0.9,
      "pixel_percentage": 12.1
    },
    "undersaturation": {
      "is_detected": false,
      "score": 0.40,
      "confidence": 0.82,
      "threshold_used": 0.2,
      "pixel_percentage": 45.3
    },
    "overall_quality_score": 0.65,
    "detected_issues": ["overexposure"],
    "processing_time_ms": 2150.8
  },
  "error": null
}
```

#### 4. Analyze Multiple Images (Batch - Blur Only)

**POST** `/analyze-batch`

Analyzes multiple images for blur detection only (backward compatibility, max 10 images).

**Request Body**:
```json
{
  "image_urls": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.png"
  ]
}
```

**Response**: (Same format as single image analysis but in batch)

#### 5. Comprehensive Quality Batch Analysis

**POST** `/analyze-quality-batch`

Performs comprehensive quality analysis on multiple images (max 10 images).

**Request Body**:
```json
{
  "image_urls": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.png"
  ]
}
```

**Response**:
```json
{
  "total_processed": 2,
  "successful": 1,
  "failed": 1,
  "results": [
    {
      "success": true,
      "image_url": "https://example.com/image1.jpg",
      "result": {
        "is_blurred": false,
        "blur_score": 0.25,
        "confidence": 0.88,
        "processing_time_ms": 980.2
      },
      "error": null
    },
    {
      "success": false,
      "image_url": "https://example.com/invalid.jpg",
      "result": null,
      "error": "Failed to download image: HTTP 404"
    }
  ]
}
```

## Usage Examples

### Python

```python
import requests

# Single image analysis
response = requests.post(
    "http://localhost:8000/analyze-image",
    json={"image_url": "https://example.com/test-image.jpg"}
)
result = response.json()
print(f"Is blurred: {result['result']['is_blurred']}")
print(f"Blur score: {result['result']['blur_score']}")

# Batch analysis
response = requests.post(
    "http://localhost:8000/analyze-batch",
    json={
        "image_urls": [
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg"
        ]
    }
)
batch_result = response.json()
print(f"Processed: {batch_result['total_processed']}")
print(f"Successful: {batch_result['successful']}")
```

### cURL

```bash
# Single image analysis
curl -X POST "http://localhost:8000/analyze-image" \
     -H "Content-Type: application/json" \
     -d '{"image_url": "https://example.com/test-image.jpg"}'

# Batch analysis
curl -X POST "http://localhost:8000/analyze-batch" \
     -H "Content-Type: application/json" \
     -d '{"image_urls": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]}'

# Health check
curl -X GET "http://localhost:8000/health"
```

## Configuration

The API can be configured using environment variables:

- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) - Default: INFO
- `LOG_FILE`: Log file path - Default: logs/api.log
- `RATE_LIMIT_PER_MINUTE`: Requests per minute per IP - Default: 60
- `RATE_LIMIT_PER_HOUR`: Requests per hour per IP - Default: 1000

Example:
```bash
export LOG_LEVEL=DEBUG
export RATE_LIMIT_PER_MINUTE=100
python main.py
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- HEIC/HEIF (.heic, .heif) - Apple's High Efficiency Image Container format

## Limitations

- Maximum image size: 50MB
- Maximum image dimensions: 10,000 x 10,000 pixels
- Batch processing: Maximum 10 images per request
- Download timeout: 30 seconds per image

## Error Handling

The API provides comprehensive error handling for various scenarios:

- Invalid or unreachable URLs
- Unsupported image formats
- Network timeouts
- Image processing errors
- Rate limit exceeded

All errors return appropriate HTTP status codes and detailed error messages.

## Rate Limiting

The API includes built-in rate limiting to prevent abuse:

- Default: 60 requests per minute per IP
- Default: 1000 requests per hour per IP
- Rate limit headers are included in responses
- Configurable via environment variables

## Logging

Comprehensive logging is implemented for:

- Request/response tracking
- Image processing results
- Error conditions
- Performance metrics
- Rate limiting events

Logs are written to both console and file (configurable).

## Algorithm Details

### Blur Detection
The blur detection algorithm is based on the research paper:

> Golestaneh, S Alireza and Karam, Lina J. "Spatially-Varying Blur Detection Based on Multiscale Fused and Sorted Transform Coefficients of Gradient Magnitudes." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

The algorithm analyzes gradient magnitudes at multiple scales and uses DCT coefficients to detect blur patterns in images.

### Quality Detection Algorithms

#### Exposure Analysis
- **Overexposure**: Histogram analysis of luminance channel, detecting pixels above brightness threshold (default: 240/255)
- **Underexposure**: Histogram analysis of luminance channel, detecting pixels below darkness threshold (default: 15/255)

#### Saturation Analysis
- **Oversaturation**: HSV color space analysis, detecting pixels with excessive saturation (default: >90%)
- **Undersaturation**: HSV color space analysis, detecting washed-out pixels with low saturation (default: <20%)

#### Overall Quality Scoring
Weighted combination of all quality factors:
- Blur: 30% weight
- Overexposure: 20% weight
- Underexposure: 20% weight
- Oversaturation: 15% weight
- Undersaturation: 15% weight

All thresholds are configurable and the algorithms provide confidence scores based on detection consistency.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions, please create an issue in the repository or contact the development team.

## API Endpoints

### 1. Health Check
- **GET `/health`**
  - Returns API status, version, and uptime.

### 2. Blur Detection (with Type Classification)
- **POST `/analyze-image`**
  - Request Body:
    ```json
    {
      "image_url": "https://example.com/image.jpg"
    }
    ```
  - Response:
    ```json
    {
      "success": true,
      "image_url": "https://example.com/image.jpg",
      "result": {
        "is_blurred": true,
        "blur_score": 0.85,
        "confidence": 0.92,
        "processing_time_ms": 120,
        "blur_type_details": {
          "blur_type": "motion_blur",
          "type_confidence": 0.88,
          "motion_score": 0.9,
          "gaussian_score": 0.1,
          "defocus_score": 0.0,
          "analysis_reason": "High motion blur detected"
        }
      },
      "error": null
    }
    ```

### 3. Comprehensive Quality Analysis
- **POST `/analyze-quality`**
  - Request Body:
    ```json
    {
      "image_url": "https://example.com/image.jpg"
    }
    ```
  - Response: Includes blur, exposure, saturation, overall score, and detected issues.

### 4. Batch Blur Detection
- **POST `/analyze-batch`**
  - Request Body:
    ```json
    {
      "image_urls": [
        "https://example.com/image1.jpg",
        "https://example.com/image2.png"
      ]
    }
    ```

### 5. Cache Management
- **GET `/cache/stats`**: Returns cache statistics.
- **POST `/cache/clear`**: Clears all cached results.

## Usage

- All endpoints accept only JPG, JPEG, PNG image URLs.
- See `/docs` or `/redoc` for full OpenAPI documentation and schema details.
