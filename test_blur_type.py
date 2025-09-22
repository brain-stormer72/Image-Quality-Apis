#!/usr/bin/env python3
"""
Test script for blur type classification functionality.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_blur_type_detection():
    """Test blur type detection with different images."""
    
    test_images = [
        {
            "name": "Blurred Image",
            "url": "https://imagedetectionv2.blob.core.windows.net/test/blur.jpg",
            "expected_blur": True
        },
        {
            "name": "Sharp Image", 
            "url": "https://imagedetectionv2.blob.core.windows.net/test/1%20mb.png",
            "expected_blur": False
        }
    ]
    
    print("=" * 80)
    print("Blur Type Classification Test")
    print("=" * 80)
    
    for test_case in test_images:
        print(f"\nTesting: {test_case['name']}")
        print(f"URL: {test_case['url']}")
        print("-" * 60)
        
        try:
            # Test legacy blur detection endpoint
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/analyze-image",
                json={"image_url": test_case['url']},
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    blur_result = result['result']
                    print(f"✓ Analysis successful:")
                    print(f"  Is Blurred: {blur_result['is_blurred']}")
                    print(f"  Blur Score: {blur_result['blur_score']:.3f}")
                    print(f"  Confidence: {blur_result['confidence']:.3f}")
                    print(f"  Processing Time: {blur_result['processing_time_ms']:.1f}ms")
                    
                    # Check if blur type details are included
                    if 'blur_type_details' in blur_result and blur_result['blur_type_details']:
                        details = blur_result['blur_type_details']
                        print(f"  Blur Type: {details['blur_type']}")
                        print(f"  Type Confidence: {details['type_confidence']:.3f}")
                        print(f"  Motion Score: {details['motion_score']:.3f}")
                        print(f"  Gaussian Score: {details['gaussian_score']:.3f}")
                        print(f"  Defocus Score: {details['defocus_score']:.3f}")
                        print(f"  Analysis Reason: {details['analysis_reason']}")
                    else:
                        print("  ⚠ No blur type details found")
                        
                    print(f"  Total Request Time: {(end_time - start_time) * 1000:.1f}ms")
                else:
                    print(f"✗ Analysis failed: {result['error']}")
            else:
                print(f"✗ HTTP Error: {response.status_code}")
                print(f"  Response: {response.text}")
                
        except Exception as e:
            print(f"✗ Request failed: {e}")
    
    print("\n" + "=" * 80)
    print("Testing comprehensive quality analysis...")
    print("=" * 80)
    
    # Test comprehensive analysis endpoint
    test_url = "https://imagedetectionv2.blob.core.windows.net/test/blur.jpg"
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/analyze-quality",
            json={"image_url": test_url},
            timeout=60
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                comprehensive_result = result['result']
                print(f"✓ Comprehensive analysis successful:")
                print(f"  Overall Quality Score: {comprehensive_result['overall_quality_score']:.3f}")
                print(f"  Issues Detected: {comprehensive_result['detected_issues']}")
                
                # Check blur details in comprehensive result
                blur_info = comprehensive_result['blur']
                print(f"  Blur Analysis:")
                print(f"    Is Blurred: {blur_info['is_blurred']}")
                print(f"    Blur Score: {blur_info['blur_score']:.3f}")
                print(f"    Confidence: {blur_info['confidence']:.3f}")
                
                if 'blur_type_details' in blur_info and blur_info['blur_type_details']:
                    details = blur_info['blur_type_details']
                    print(f"    Blur Type: {details['blur_type']}")
                    print(f"    Type Confidence: {details['type_confidence']:.3f}")
                    print(f"    Motion Score: {details['motion_score']:.3f}")
                    print(f"    Gaussian Score: {details['gaussian_score']:.3f}")
                    print(f"    Defocus Score: {details['defocus_score']:.3f}")
                    print(f"    Analysis Reason: {details['analysis_reason']}")
                else:
                    print("    ⚠ No blur type details found in comprehensive analysis")
                    
                print(f"  Total Request Time: {(end_time - start_time) * 1000:.1f}ms")
            else:
                print(f"✗ Comprehensive analysis failed: {result['error']}")
        else:
            print(f"✗ HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"✗ Comprehensive analysis request failed: {e}")

if __name__ == "__main__":
    test_blur_type_detection()
