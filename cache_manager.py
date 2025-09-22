"""
Intelligent caching system for image analysis results.
"""

import hashlib
import json
import logging
import time
from typing import Optional, Dict, Any
import redis
from dataclasses import asdict

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Redis-based caching system for image analysis results.
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        default_ttl: int = 3600,  # 1 hour
        enabled: bool = True
    ):
        """
        Initialize cache manager.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            default_ttl: Default time-to-live in seconds
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.default_ttl = default_ttl
        self.redis_client = None
        
        if self.enabled:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                
                # Test connection
                self.redis_client.ping()
                logger.info(f"Cache manager initialized: Redis at {redis_host}:{redis_port}")
                
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Caching disabled.")
                self.enabled = False
                self.redis_client = None
    
    def _generate_cache_key(self, image_url: str, analysis_type: str = "comprehensive") -> str:
        """
        Generate cache key for image analysis result.
        
        Args:
            image_url: Image URL
            analysis_type: Type of analysis (blur, comprehensive)
            
        Returns:
            Cache key string
        """
        # Create hash of URL for consistent key generation
        url_hash = hashlib.sha256(image_url.encode()).hexdigest()[:16]
        return f"image_analysis:{analysis_type}:{url_hash}"
    
    def get_cached_result(self, image_url: str, analysis_type: str = "comprehensive") -> Optional[Dict[str, Any]]:
        """
        Retrieve cached analysis result.
        
        Args:
            image_url: Image URL
            analysis_type: Type of analysis
            
        Returns:
            Cached result or None if not found
        """
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_cache_key(image_url, analysis_type)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                result = json.loads(cached_data)
                logger.debug(f"Cache hit for {image_url} ({analysis_type})")
                return result
            
            logger.debug(f"Cache miss for {image_url} ({analysis_type})")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None
    
    def cache_result(
        self,
        image_url: str,
        result: Dict[str, Any],
        analysis_type: str = "comprehensive",
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache analysis result.
        
        Args:
            image_url: Image URL
            result: Analysis result to cache
            analysis_type: Type of analysis
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key(image_url, analysis_type)
            ttl = ttl or self.default_ttl
            
            # Add timestamp to cached data
            cache_data = {
                "result": result,
                "cached_at": time.time(),
                "image_url": image_url,
                "analysis_type": analysis_type
            }
            
            # Store in Redis with TTL
            success = self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(cache_data, default=str)
            )
            
            if success:
                logger.debug(f"Cached result for {image_url} ({analysis_type}) with TTL {ttl}s")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error caching result: {e}")
            return False
    
    def invalidate_cache(self, image_url: str, analysis_type: str = "*") -> int:
        """
        Invalidate cached results for an image URL.
        
        Args:
            image_url: Image URL
            analysis_type: Type of analysis or "*" for all types
            
        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.redis_client:
            return 0
        
        try:
            if analysis_type == "*":
                # Delete all analysis types for this URL
                url_hash = hashlib.sha256(image_url.encode()).hexdigest()[:16]
                pattern = f"image_analysis:*:{url_hash}"
            else:
                cache_key = self._generate_cache_key(image_url, analysis_type)
                pattern = cache_key
            
            # Find matching keys
            keys = self.redis_client.keys(pattern)
            
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries for {image_url}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled or not self.redis_client:
            return {"enabled": False}
        
        try:
            info = self.redis_client.info()
            
            # Count image analysis keys
            analysis_keys = self.redis_client.keys("image_analysis:*")
            
            return {
                "enabled": True,
                "total_keys": len(analysis_keys),
                "memory_usage": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"enabled": True, "error": str(e)}
    
    def _calculate_hit_rate(self, info: Dict[str, Any]) -> float:
        """Calculate cache hit rate."""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return round((hits / total) * 100, 2)
    
    def clear_all_cache(self) -> bool:
        """
        Clear all image analysis cache entries.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            keys = self.redis_client.keys("image_analysis:*")
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries")
                return True
            
            logger.info("No cache entries to clear")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def close(self):
        """Close Redis connection."""
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info("Cache manager connection closed")
            except Exception as e:
                logger.error(f"Error closing cache connection: {e}")


# Global cache manager instance
cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global cache_manager
    if cache_manager is None:
        cache_manager = CacheManager()
    return cache_manager


def init_cache_manager(**kwargs) -> CacheManager:
    """Initialize global cache manager with custom settings."""
    global cache_manager
    cache_manager = CacheManager(**kwargs)
    return cache_manager
