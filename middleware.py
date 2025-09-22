"""
Custom middleware for the Image Quality Check API.
"""

import time
import logging
from typing import Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware to prevent API abuse.
    """
    
    def __init__(self, app, calls_per_minute: int = 60, calls_per_hour: int = 1000):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour
        self.minute_requests = defaultdict(deque)
        self.hour_requests = defaultdict(deque)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _clean_old_requests(self, client_ip: str):
        """Remove old requests from tracking."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        # Clean minute requests
        while (self.minute_requests[client_ip] and 
               self.minute_requests[client_ip][0] < minute_ago):
            self.minute_requests[client_ip].popleft()
        
        # Clean hour requests
        while (self.hour_requests[client_ip] and 
               self.hour_requests[client_ip][0] < hour_ago):
            self.hour_requests[client_ip].popleft()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting for health check
        if request.url.path == "/health":
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        now = datetime.now()
        
        # Clean old requests
        self._clean_old_requests(client_ip)
        
        # Check rate limits
        minute_count = len(self.minute_requests[client_ip])
        hour_count = len(self.hour_requests[client_ip])
        
        if minute_count >= self.calls_per_minute:
            logger.warning(f"Rate limit exceeded (per minute) for IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.calls_per_minute} requests per minute allowed"
                }
            )
        
        if hour_count >= self.calls_per_hour:
            logger.warning(f"Rate limit exceeded (per hour) for IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.calls_per_hour} requests per hour allowed"
                }
            )
        
        # Record this request
        self.minute_requests[client_ip].append(now)
        self.hour_requests[client_ip].append(now)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Minute-Limit"] = str(self.calls_per_minute)
        response.headers["X-RateLimit-Minute-Remaining"] = str(self.calls_per_minute - minute_count - 1)
        response.headers["X-RateLimit-Hour-Limit"] = str(self.calls_per_hour)
        response.headers["X-RateLimit-Hour-Remaining"] = str(self.calls_per_hour - hour_count - 1)
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging requests and responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details."""
        start_time = time.time()
        
        # Log request
        client_ip = request.client.host if request.client else "unknown"
        logger.info(f"Request: {request.method} {request.url.path} from {client_ip}")
        
        # Process request
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time
            
            # Log response
            logger.info(f"Response: {response.status_code} for {request.url.path} in {processing_time:.3f}s")
            
            # Add processing time header
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Request failed: {request.url.path} in {processing_time:.3f}s - {str(e)}")
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response
