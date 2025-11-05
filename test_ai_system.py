#!/usr/bin/env python3
"""
Test script for NagrikHelp AI Image Validation System
Tests all three models and the ensemble pipeline
"""

import sys
import time
import json
import base64
from pathlib import Path

try:
    import requests
except ImportError:
    print("❌ requests library not found. Install with: pip install requests")
    sys.exit(1)

# Configuration
AI_SERVER_URL = "http://127.0.0.1:8001"
TIMEOUT = 30

def create_test_image_base64():
    """Create a simple test image as base64"""
    from PIL import Image
    import io
    
    # Create a simple 224x224 test image
    img = Image.new('RGB', (224, 224), color='gray')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

def test_health_check():
    """Test GET / endpoint"""
    print("\n🔍 Testing Health Check Endpoint...")
    try:
        response = requests.get(f"{AI_SERVER_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Service: {data.get('service')}")
            print(f"   Models loaded: {data.get('models')}")
            print(f"   Confidence threshold: {data.get('confidence_threshold')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to AI server at {AI_SERVER_URL}")
        print("   Make sure the server is running:")
        print("   python -m uvicorn local_vision_server:app --host 0.0.0.0 --port 8001")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_status():
    """Test GET /model-status endpoint"""
    print("\n🔍 Testing Model Status Endpoint...")
    try:
        response = requests.get(f"{AI_SERVER_URL}/model-status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Model status check passed")
            print(f"   Loaded: {data.get('loaded')}")
            print(f"   All loaded: {data.get('all_loaded')}")
            return data.get('all_loaded', False)
        else:
            print(f"❌ Model status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model status error: {e}")
        return False

def test_validate_endpoint():
    """Test POST /validate endpoint with test image"""
    print("\n🔍 Testing Validate Endpoint...")
    print("   ⏳ Creating test image...")
    
    try:
        image_base64 = create_test_image_base64()
        print("   ⏳ Sending validation request (this may take 5-15 seconds on first run)...")
        
        start_time = time.time()
        response = requests.post(
            f"{AI_SERVER_URL}/validate",
            json={
                "image": f"data:image/jpeg;base64,{image_base64}",
                "description": "test image"
            },
            timeout=TIMEOUT
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Validate endpoint passed ({elapsed:.2f}s)")
            print(f"   Is Issue: {data.get('isIssue')}")
            print(f"   Category: {data.get('category')}")
            print(f"   Confidence: {data.get('confidence', 0)*100:.1f}%")
            print(f"   Models Used: {data.get('modelUsed')}")
            print(f"   Message: {data.get('message')}")
            if data.get('bbox'):
                print(f"   Bounding Box: {data.get('bbox')}")
            
            # Show debug info
            if data.get('debug'):
                print("\n   📊 Debug Info:")
                debug = data['debug']
                if debug.get('yolo'):
                    print(f"      YOLO: detected={debug['yolo'].get('detected')}, conf={debug['yolo'].get('conf', 0):.3f}")
                if debug.get('resnet'):
                    print(f"      ResNet: category={debug['resnet'].get('category')}, conf={debug['resnet'].get('conf', 0):.3f}")
                if debug.get('clip'):
                    print(f"      CLIP: {debug['clip']}")
            
            return True
        else:
            print(f"❌ Validate endpoint failed: {response.status_code}")
            try:
                error = response.json()
                print(f"   Error: {error}")
            except:
                print(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ Validate endpoint timeout (>{TIMEOUT}s)")
        print("   Models may still be loading. Try again in a minute.")
        return False
    except Exception as e:
        print(f"❌ Validate endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_classify_legacy():
    """Test POST /classify legacy endpoint"""
    print("\n🔍 Testing Classify (Legacy) Endpoint...")
    
    try:
        from PIL import Image
        import io
        
        # Create test image
        img = Image.new('RGB', (224, 224), color='gray')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        
        print("   ⏳ Sending classification request...")
        start_time = time.time()
        response = requests.post(
            f"{AI_SERVER_URL}/classify",
            data=img_bytes,
            headers={'Content-Type': 'application/octet-stream'},
            timeout=TIMEOUT
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Classify endpoint passed ({elapsed:.2f}s)")
            if isinstance(data, list) and len(data) > 0:
                print(f"   Top labels:")
                for i, item in enumerate(data[:3]):
                    print(f"      {i+1}. {item.get('label')}: {item.get('score', 0)*100:.1f}%")
            return True
        else:
            print(f"❌ Classify endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Classify endpoint error: {e}")
        return False

def main():
    print("=" * 60)
    print("🤖 NagrikHelp AI Validation System - Test Suite")
    print("=" * 60)
    
    # Track results
    results = {
        'health_check': False,
        'model_status': False,
        'validate': False,
        'classify': False
    }
    
    # Run tests
    results['health_check'] = test_health_check()
    
    if not results['health_check']:
        print("\n❌ Cannot proceed - AI server not reachable")
        sys.exit(1)
    
    results['model_status'] = test_model_status()
    results['validate'] = test_validate_endpoint()
    results['classify'] = test_classify_legacy()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your AI system is ready to use!")
        print("\nNext steps:")
        print("1. Open http://localhost:3000/citizen/create")
        print("2. Upload a civic issue image (pothole, garbage, etc.)")
        print("3. Watch the AI analyze and suggest category")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
        sys.exit(130)
