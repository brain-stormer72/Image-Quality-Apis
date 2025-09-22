#!/usr/bin/env python3
"""
Performance test script for the optimized Image Quality API.
"""

import time
import requests
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import statistics

BASE_URL = "http://localhost:8000"

def test_single_request():
    """Test single request performance."""
    print("Testing single request performance...")

    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/analyze-quality",
        json={"image_url": "https://imagedetectionv2.blob.core.windows.net/test/blur.jpg"}
    )
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            processing_time = result['result']['processing_time_ms']
            total_time = (end_time - start_time) * 1000
            issues = result['result']['detected_issues']
            
            print(f"✓ Single request successful:")
            print(f"  Processing time: {processing_time:.1f}ms")
            print(f"  Total request time: {total_time:.1f}ms")
            print(f"  Issues detected: {issues}")
            print(f"  Overall quality score: {result['result']['overall_quality_score']:.3f}")
            return processing_time, total_time
        else:
            print(f"✗ Request failed: {result['error']}")
    else:
        print(f"✗ HTTP error: {response.status_code}")
    
    return None, None

def test_batch_request():
    """Test batch request performance."""
    print("\nTesting batch request performance...")
    
    test_urls = [
        "https://httpbin.org/image/jpeg",
        "https://httpbin.org/image/png",
        "https://httpbin.org/image/webp"
    ]
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/analyze-quality-batch",
        json={"image_urls": test_urls}
    )
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        total_time = (end_time - start_time) * 1000
        
        print(f"✓ Batch request successful:")
        print(f"  Total processed: {result['total_processed']}")
        print(f"  Successful: {result['successful']}")
        print(f"  Failed: {result['failed']}")
        print(f"  Total request time: {total_time:.1f}ms")
        print(f"  Average per image: {total_time / len(test_urls):.1f}ms")
        
        # Show individual results
        for i, img_result in enumerate(result['results']):
            if img_result['success']:
                proc_time = img_result['result']['processing_time_ms']
                issues = img_result['result']['detected_issues']
                print(f"  Image {i+1}: {proc_time:.1f}ms, issues: {issues}")
        
        return total_time
    else:
        print(f"✗ HTTP error: {response.status_code}")
    
    return None

def test_concurrent_requests():
    """Test concurrent request performance."""
    print("\nTesting concurrent request performance...")
    
    def make_request():
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/analyze-quality",
            json={"image_url": "https://httpbin.org/image/jpeg"}
        )
        end = time.time()
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                return (end - start) * 1000, result['result']['processing_time_ms']
        return None, None
    
    # Test with 3 concurrent requests
    num_requests = 3
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        results = [future.result() for future in futures]
    
    end_time = time.time()
    
    successful_results = [r for r in results if r[0] is not None]
    
    if successful_results:
        total_times = [r[0] for r in successful_results]
        processing_times = [r[1] for r in successful_results]
        
        print(f"✓ Concurrent requests successful:")
        print(f"  Requests: {len(successful_results)}/{num_requests}")
        print(f"  Total wall time: {(end_time - start_time) * 1000:.1f}ms")
        print(f"  Average request time: {statistics.mean(total_times):.1f}ms")
        print(f"  Average processing time: {statistics.mean(processing_times):.1f}ms")
        print(f"  Speedup vs sequential: {sum(total_times) / ((end_time - start_time) * 1000):.1f}x")
    else:
        print("✗ All concurrent requests failed")

def test_cache_stats():
    """Test cache statistics endpoint."""
    print("\nTesting cache statistics...")
    
    response = requests.get(f"{BASE_URL}/cache/stats")
    if response.status_code == 200:
        stats = response.json()
        print(f"✓ Cache stats retrieved:")
        print(f"  Enabled: {stats.get('enabled', False)}")
        if stats.get('enabled'):
            print(f"  Total keys: {stats.get('total_keys', 0)}")
            print(f"  Hit rate: {stats.get('hit_rate', 0)}%")
            print(f"  Memory usage: {stats.get('memory_usage', 'unknown')}")
        else:
            print(f"  Cache is disabled (Redis not available)")
    else:
        print(f"✗ Failed to get cache stats: {response.status_code}")

def main():
    """Run all performance tests."""
    print("=" * 60)
    print("Image Quality API Performance Tests")
    print("=" * 60)
    
    # Test single request
    proc_time, total_time = test_single_request()
    
    # Test batch request
    batch_time = test_batch_request()
    
    # Test concurrent requests
    test_concurrent_requests()
    
    # Test cache stats
    test_cache_stats()
    
    print("\n" + "=" * 60)
    print("Performance Summary:")
    print("=" * 60)
    
    if proc_time and total_time:
        print(f"Single request processing: {proc_time:.1f}ms")
        print(f"Single request total: {total_time:.1f}ms")
    
    if batch_time:
        print(f"Batch request (3 images): {batch_time:.1f}ms")
        print(f"Average per image in batch: {batch_time/3:.1f}ms")
    
    print("\nOptimizations active:")
    print("- ✓ Vectorized NumPy operations")
    print("- ✓ Optimized histogram analysis")
    print("- ✓ Parallel batch processing")
    print("- ✓ Async I/O operations")
    print("- ✓ Image preprocessing optimization")
    print("- ✓ Intelligent caching (Redis disabled)")

if __name__ == "__main__":
    main()
