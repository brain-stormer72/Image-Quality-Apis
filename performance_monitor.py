"""
Performance monitoring and metrics collection for the Image Quality Check API.
"""

import time
import logging
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for API operations."""
    endpoint: str
    processing_time_ms: float
    memory_usage_mb: float
    image_dimensions: tuple
    resized_dimensions: Optional[tuple] = None
    cache_hit: bool = False
    error_occurred: bool = False
    timestamp: float = field(default_factory=time.time)


class PerformanceMonitor:
    """
    Monitors and tracks performance metrics for the Image Quality Check API.
    """
    
    def __init__(self, max_history_size: int = 1000):
        """
        Initialize the performance monitor.
        
        Args:
            max_history_size: Maximum number of metrics to keep in history
        """
        self.max_history_size = max_history_size
        self.metrics_history: deque = deque(maxlen=max_history_size)
        self.endpoint_stats: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
        
        # Performance tracking
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.total_memory_saved = 0
        self.cache_hits = 0
        self.errors = 0
        
        logger.info(f"PerformanceMonitor initialized with max_history_size={max_history_size}")
    
    @contextmanager
    def track_performance(self, endpoint: str, image_dimensions: tuple = None):
        """
        Context manager to track performance of an operation.
        
        Args:
            endpoint: Name of the endpoint being tracked
            image_dimensions: Original image dimensions (width, height)
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        error_occurred = False
        
        try:
            yield
        except Exception as e:
            error_occurred = True
            logger.error(f"Error in {endpoint}: {e}")
            raise
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
            memory_usage = max(end_memory - start_memory, 0)  # Memory delta
            
            metrics = PerformanceMetrics(
                endpoint=endpoint,
                processing_time_ms=processing_time,
                memory_usage_mb=memory_usage,
                image_dimensions=image_dimensions or (0, 0),
                error_occurred=error_occurred
            )
            
            self._record_metrics(metrics)
    
    def record_cache_hit(self, endpoint: str):
        """Record a cache hit for performance tracking."""
        with self.lock:
            self.cache_hits += 1
            
            # Create a minimal metrics entry for cache hits
            metrics = PerformanceMetrics(
                endpoint=endpoint,
                processing_time_ms=0.1,  # Minimal time for cache hit
                memory_usage_mb=0.0,
                image_dimensions=(0, 0),
                cache_hit=True
            )
            self._record_metrics(metrics)
    
    def record_resize_operation(self, original_dims: tuple, new_dims: tuple, 
                              processing_time_ms: float, memory_saved_mb: float):
        """
        Record a resize operation for performance tracking.
        
        Args:
            original_dims: Original image dimensions (width, height)
            new_dims: New image dimensions after resizing
            processing_time_ms: Time taken for resizing
            memory_saved_mb: Memory saved by resizing
        """
        with self.lock:
            self.total_memory_saved += memory_saved_mb
            
            # Update metrics with resize information
            if self.metrics_history:
                latest_metrics = self.metrics_history[-1]
                latest_metrics.resized_dimensions = new_dims
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        with self.lock:
            self.metrics_history.append(metrics)
            self.endpoint_stats[metrics.endpoint].append(metrics.processing_time_ms)
            
            # Keep endpoint stats manageable
            if len(self.endpoint_stats[metrics.endpoint]) > 100:
                self.endpoint_stats[metrics.endpoint] = self.endpoint_stats[metrics.endpoint][-50:]
            
            # Update totals
            self.total_requests += 1
            self.total_processing_time += metrics.processing_time_ms
            
            if metrics.error_occurred:
                self.errors += 1
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not PSUTIL_AVAILABLE:
            return 0.0
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dict containing performance statistics
        """
        with self.lock:
            if not self.metrics_history:
                return {"message": "No performance data available"}
            
            # Calculate overall statistics
            total_requests = len(self.metrics_history)
            avg_processing_time = sum(m.processing_time_ms for m in self.metrics_history) / total_requests
            
            # Calculate endpoint-specific statistics
            endpoint_summary = {}
            for endpoint, times in self.endpoint_stats.items():
                if times:
                    endpoint_summary[endpoint] = {
                        "requests": len(times),
                        "avg_time_ms": round(sum(times) / len(times), 2),
                        "min_time_ms": round(min(times), 2),
                        "max_time_ms": round(max(times), 2)
                    }
            
            # Calculate cache statistics
            cache_hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
            error_rate = (self.errors / total_requests * 100) if total_requests > 0 else 0
            
            # Calculate resize statistics
            resize_operations = sum(1 for m in self.metrics_history if m.resized_dimensions)
            resize_rate = (resize_operations / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "total_requests": total_requests,
                "average_processing_time_ms": round(avg_processing_time, 2),
                "cache_hit_rate_percent": round(cache_hit_rate, 2),
                "error_rate_percent": round(error_rate, 2),
                "resize_operations": resize_operations,
                "resize_rate_percent": round(resize_rate, 2),
                "total_memory_saved_mb": round(self.total_memory_saved, 2),
                "endpoint_statistics": endpoint_summary,
                "current_memory_usage_mb": round(self._get_memory_usage(), 2)
            }
    
    def get_recent_performance(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent performance metrics.
        
        Args:
            limit: Number of recent metrics to return
            
        Returns:
            List of recent performance metrics
        """
        with self.lock:
            recent_metrics = list(self.metrics_history)[-limit:]
            return [
                {
                    "endpoint": m.endpoint,
                    "processing_time_ms": round(m.processing_time_ms, 2),
                    "memory_usage_mb": round(m.memory_usage_mb, 2),
                    "image_dimensions": m.image_dimensions,
                    "resized_dimensions": m.resized_dimensions,
                    "cache_hit": m.cache_hit,
                    "error_occurred": m.error_occurred,
                    "timestamp": m.timestamp
                }
                for m in recent_metrics
            ]
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        with self.lock:
            self.metrics_history.clear()
            self.endpoint_stats.clear()
            self.total_requests = 0
            self.total_processing_time = 0.0
            self.total_memory_saved = 0
            self.cache_hits = 0
            self.errors = 0
            
        logger.info("Performance metrics reset")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """
    Get the global performance monitor instance.
    
    Returns:
        PerformanceMonitor: Global monitor instance
    """
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def track_performance(endpoint: str, image_dimensions: tuple = None):
    """
    Decorator/context manager for tracking performance.
    
    Args:
        endpoint: Name of the endpoint being tracked
        image_dimensions: Original image dimensions
    """
    monitor = get_performance_monitor()
    return monitor.track_performance(endpoint, image_dimensions)
