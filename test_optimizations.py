#!/usr/bin/env python3
"""
Test script to verify performance optimizations are working correctly.
Tests intelligent image resizing, performance monitoring, and API endpoints.
"""

import asyncio
import time
import requests
import json
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_IMAGE_URLS = [
    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=2400&h=1600",  # Large landscape
    "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=1920&h=1080",  # HD image
    "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=800&h=600",   # Smaller image
]

def test_api_health():
    """Test API health endpoint."""
    print("🔍 Testing API health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API health check error: {e}")
        return False

def test_single_image_analysis(image_url: str) -> Dict[str, Any]:
    """Test single image analysis with performance tracking."""
    print(f"🖼️  Testing single image analysis: {image_url[:50]}...")
    
    start_time = time.time()
    try:
        payload = {"image_url": image_url}
        response = requests.post(
            f"{API_BASE_URL}/analyze-image", 
            json=payload, 
            timeout=30
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                result = data.get("result", {})
                api_processing_time = result.get("processing_time_ms", 0)
                
                print(f"✅ Analysis successful:")
                print(f"   - Total time: {processing_time:.1f}ms")
                print(f"   - API processing: {api_processing_time:.1f}ms")
                print(f"   - Blur detected: {result.get('is_blurred', 'unknown')}")
                print(f"   - Blur score: {result.get('blur_score', 0):.3f}")
                
                return {
                    "success": True,
                    "total_time_ms": processing_time,
                    "api_processing_time_ms": api_processing_time,
                    "blur_detected": result.get("is_blurred", False),
                    "blur_score": result.get("blur_score", 0)
                }
            else:
                print(f"❌ Analysis failed: {data.get('error', 'unknown error')}")
        else:
            print(f"❌ Request failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Analysis error: {e}")
    
    return {"success": False, "total_time_ms": processing_time}

def test_comprehensive_analysis(image_url: str) -> Dict[str, Any]:
    """Test comprehensive image analysis."""
    print(f"🔬 Testing comprehensive analysis: {image_url[:50]}...")
    
    start_time = time.time()
    try:
        payload = {"image_url": image_url}
        response = requests.post(
            f"{API_BASE_URL}/analyze-comprehensive", 
            json=payload, 
            timeout=45
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                result = data.get("result", {})
                api_processing_time = result.get("processing_time_ms", 0)
                detected_issues = result.get("detected_issues", [])
                overall_score = result.get("overall_quality_score", 0)
                
                print(f"✅ Comprehensive analysis successful:")
                print(f"   - Total time: {processing_time:.1f}ms")
                print(f"   - API processing: {api_processing_time:.1f}ms")
                print(f"   - Overall quality: {overall_score:.3f}")
                print(f"   - Issues detected: {len(detected_issues)}")
                
                return {
                    "success": True,
                    "total_time_ms": processing_time,
                    "api_processing_time_ms": api_processing_time,
                    "overall_quality_score": overall_score,
                    "issues_count": len(detected_issues)
                }
            else:
                print(f"❌ Comprehensive analysis failed: {data.get('error', 'unknown error')}")
        else:
            print(f"❌ Request failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Comprehensive analysis error: {e}")
    
    return {"success": False, "total_time_ms": processing_time}

def test_performance_metrics():
    """Test performance metrics endpoint."""
    print("📊 Testing performance metrics...")
    try:
        response = requests.get(f"{API_BASE_URL}/performance", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Performance metrics retrieved:")
            
            perf_metrics = data.get("performance_metrics", {})
            resizing_metrics = data.get("resizing_optimization", {})
            
            print(f"   - Caching enabled: {data.get('caching_enabled', 'unknown')}")
            print(f"   - Total operations: {perf_metrics.get('total_operations', 0)}")
            print(f"   - Average processing time: {perf_metrics.get('average_processing_time_ms', 0):.1f}ms")
            print(f"   - Images resized: {resizing_metrics.get('total_resized', 0)}")
            print(f"   - Average resize time: {resizing_metrics.get('average_resize_time_ms', 0):.1f}ms")
            
            return True
        else:
            print(f"❌ Performance metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Performance metrics error: {e}")
    
    return False

def test_batch_processing():
    """Test batch processing with multiple images."""
    print("📦 Testing batch processing...")
    
    start_time = time.time()
    try:
        payload = {"image_urls": TEST_IMAGE_URLS[:2]}  # Test with 2 images
        response = requests.post(
            f"{API_BASE_URL}/analyze-batch", 
            json=payload, 
            timeout=60
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            total_processed = data.get("total_processed", 0)
            successful = data.get("successful", 0)
            failed = data.get("failed", 0)
            
            print(f"✅ Batch processing completed:")
            print(f"   - Total time: {processing_time:.1f}ms")
            print(f"   - Images processed: {total_processed}")
            print(f"   - Successful: {successful}")
            print(f"   - Failed: {failed}")
            print(f"   - Average per image: {processing_time/max(1, total_processed):.1f}ms")
            
            return True
        else:
            print(f"❌ Batch processing failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
    
    return False

def main():
    """Run all optimization tests."""
    print("🚀 Starting Image Quality API Optimization Tests")
    print("=" * 60)
    
    # Test API health first
    if not test_api_health():
        print("❌ API is not available. Please start the server first.")
        return
    
    print("\n" + "=" * 60)
    
    # Test single image analysis
    results = []
    for i, url in enumerate(TEST_IMAGE_URLS):
        print(f"\n--- Test Image {i+1} ---")
        result = test_single_image_analysis(url)
        results.append(result)
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 60)
    
    # Test comprehensive analysis on first image
    print(f"\n--- Comprehensive Analysis Test ---")
    comprehensive_result = test_comprehensive_analysis(TEST_IMAGE_URLS[0])
    
    print("\n" + "=" * 60)
    
    # Test batch processing
    print(f"\n--- Batch Processing Test ---")
    batch_result = test_batch_processing()
    
    print("\n" + "=" * 60)
    
    # Test performance metrics
    print(f"\n--- Performance Metrics Test ---")
    metrics_result = test_performance_metrics()
    
    print("\n" + "=" * 60)
    print("📈 Test Summary:")
    
    successful_tests = sum(1 for r in results if r.get("success", False))
    avg_processing_time = sum(r.get("api_processing_time_ms", 0) for r in results if r.get("success", False)) / max(1, successful_tests)
    
    print(f"   - Single image tests: {successful_tests}/{len(results)} successful")
    print(f"   - Average processing time: {avg_processing_time:.1f}ms")
    print(f"   - Comprehensive analysis: {'✅' if comprehensive_result.get('success') else '❌'}")
    print(f"   - Batch processing: {'✅' if batch_result else '❌'}")
    print(f"   - Performance metrics: {'✅' if metrics_result else '❌'}")
    
    print("\n🎯 Optimization Status:")
    if avg_processing_time > 0:
        if avg_processing_time < 2000:  # Less than 2 seconds
            print("   ✅ Excellent performance - optimizations working well!")
        elif avg_processing_time < 5000:  # Less than 5 seconds
            print("   ⚡ Good performance - optimizations are effective")
        else:
            print("   ⚠️  Performance could be improved - check resizing settings")
    
    print("\n🔧 Next Steps:")
    print("   - Check /performance endpoint for detailed metrics")
    print("   - Monitor /optimization/stats for resizing statistics")
    print("   - Adjust RESIZE_FACTOR in .env for optimal performance")

if __name__ == "__main__":
    main()
