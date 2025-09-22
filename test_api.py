"""
Test script for the Image Quality Check API.
"""

import requests
import json
import time
from typing import Dict, Any


class APITester:
    """Test class for the Image Quality Check API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        print("Testing health check endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Health check passed: {data['status']}")
                print(f"  Version: {data['version']}")
                print(f"  Uptime: {data['uptime_seconds']:.2f}s")
                return True
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Health check error: {e}")
            return False
    
    def test_single_image_analysis(self, image_url: str) -> bool:
        """Test single image analysis endpoint."""
        print(f"Testing single image analysis with: {image_url}")
        try:
            payload = {"image_url": image_url}
            response = self.session.post(
                f"{self.base_url}/analyze-image",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    result = data['result']
                    print(f"✓ Analysis successful:")
                    print(f"  Is blurred: {result['is_blurred']}")
                    print(f"  Blur score: {result['blur_score']:.3f}")
                    print(f"  Confidence: {result['confidence']:.3f}")
                    print(f"  Processing time: {result['processing_time_ms']:.1f}ms")
                    return True
                else:
                    print(f"✗ Analysis failed: {data['error']}")
                    return False
            else:
                print(f"✗ Request failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False

        except Exception as e:
            print(f"✗ Analysis error: {e}")
            return False

    def test_comprehensive_quality_analysis(self, image_url: str) -> bool:
        """Test comprehensive quality analysis endpoint."""
        print(f"Testing comprehensive quality analysis with: {image_url}")
        try:
            payload = {"image_url": image_url}
            response = self.session.post(
                f"{self.base_url}/analyze-quality",
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    result = data['result']
                    print(f"✓ Comprehensive analysis successful:")
                    print(f"  Blur: {result['blur']['is_blurred']} (score: {result['blur']['blur_score']:.3f})")
                    print(f"  Overexposure: {result['overexposure']['is_detected']} (score: {result['overexposure']['score']:.3f})")
                    print(f"  Underexposure: {result['underexposure']['is_detected']} (score: {result['underexposure']['score']:.3f})")
                    print(f"  Oversaturation: {result['oversaturation']['is_detected']} (score: {result['oversaturation']['score']:.3f})")
                    print(f"  Undersaturation: {result['undersaturation']['is_detected']} (score: {result['undersaturation']['score']:.3f})")
                    print(f"  Overall quality: {result['overall_quality_score']:.3f}")
                    print(f"  Detected issues: {result['detected_issues']}")
                    print(f"  Processing time: {result['processing_time_ms']:.1f}ms")
                    return True
                else:
                    print(f"✗ Comprehensive analysis failed: {data['error']}")
                    return False
            else:
                print(f"✗ Request failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False

        except Exception as e:
            print(f"✗ Comprehensive analysis error: {e}")
            return False
    
    def test_batch_analysis(self, image_urls: list) -> bool:
        """Test batch image analysis endpoint."""
        print(f"Testing batch analysis with {len(image_urls)} images...")
        try:
            payload = {"image_urls": image_urls}
            response = self.session.post(
                f"{self.base_url}/analyze-batch",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Batch analysis completed:")
                print(f"  Total processed: {data['total_processed']}")
                print(f"  Successful: {data['successful']}")
                print(f"  Failed: {data['failed']}")
                
                for i, result in enumerate(data['results']):
                    if result['success']:
                        blur_result = result['result']
                        print(f"  Image {i+1}: Blurred={blur_result['is_blurred']}, Score={blur_result['blur_score']:.3f}")
                    else:
                        print(f"  Image {i+1}: Failed - {result['error']}")
                
                return True
            else:
                print(f"✗ Batch request failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Batch analysis error: {e}")
            return False
    
    def test_error_scenarios(self) -> bool:
        """Test various error scenarios."""
        print("Testing error scenarios...")
        
        # Test invalid URL
        print("  Testing invalid URL...")
        response = self.session.post(
            f"{self.base_url}/analyze-image",
            json={"image_url": "not-a-valid-url"}
        )
        if response.status_code == 422:  # Validation error
            print("  ✓ Invalid URL properly rejected")
        else:
            print(f"  ✗ Invalid URL not properly handled: {response.status_code}")
            return False
        
        # Test non-existent image
        print("  Testing non-existent image...")
        response = self.session.post(
            f"{self.base_url}/analyze-image",
            json={"image_url": "https://httpbin.org/status/404"}
        )
        if response.status_code == 200:
            data = response.json()
            if not data['success'] and 'error' in data:
                print("  ✓ Non-existent image properly handled")
            else:
                print("  ✗ Non-existent image not properly handled")
                return False
        else:
            print(f"  ✗ Unexpected status code: {response.status_code}")
            return False
        
        return True
    
    def run_all_tests(self) -> bool:
        """Run all tests."""
        print("=" * 50)
        print("Image Quality Check API Test Suite")
        print("=" * 50)
        
        # Test URLs (using publicly available test images)
        test_image_urls = [
            "https://httpbin.org/image/jpeg",  # Simple test image
            "https://httpbin.org/image/png",   # PNG test image
        ]
        
        batch_urls = [
            "https://httpbin.org/image/jpeg",
            "https://httpbin.org/image/png"
        ]
        
        all_passed = True
        
        # Run tests
        tests = [
            ("Health Check", lambda: self.test_health_check()),
            ("Single Image Analysis", lambda: self.test_single_image_analysis(test_image_urls[0])),
            ("Comprehensive Quality Analysis", lambda: self.test_comprehensive_quality_analysis(test_image_urls[0])),
            ("Batch Analysis", lambda: self.test_batch_analysis(batch_urls)),
            ("Error Scenarios", lambda: self.test_error_scenarios())
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'-' * 30}")
            print(f"Running: {test_name}")
            print(f"{'-' * 30}")
            
            try:
                result = test_func()
                if not result:
                    all_passed = False
                    print(f"✗ {test_name} FAILED")
                else:
                    print(f"✓ {test_name} PASSED")
            except Exception as e:
                print(f"✗ {test_name} ERROR: {e}")
                all_passed = False
        
        print(f"\n{'=' * 50}")
        if all_passed:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED!")
        print(f"{'=' * 50}")
        
        return all_passed


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Image Quality Check API")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--image-url",
        help="Test with a specific image URL"
    )
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    if args.image_url:
        # Test specific image
        print(f"Testing specific image: {args.image_url}")
        success = tester.test_single_image_analysis(args.image_url)
        if success:
            print("✓ Test completed successfully")
        else:
            print("✗ Test failed")
    else:
        # Run full test suite
        tester.run_all_tests()


if __name__ == "__main__":
    main()
