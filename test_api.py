"""
Test script for Voice Deepfake Detection API
"""

import requests
import base64
import json
import sys
from pathlib import Path


def test_api(audio_file_path: str, language: str = "en", api_url: str = "http://localhost:8000", api_key: str = "your-secret-api-key-here"):
    """
    Test the deepfake detection API
    
    Args:
        audio_file_path: Path to MP3 audio file
        language: Language code (ta/en/hi/ml/te)
        api_url: API base URL
        api_key: API key for authentication
    """
    
    # Check if file exists
    if not Path(audio_file_path).exists():
        print(f"❌ Error: File not found: {audio_file_path}")
        return
    
    # Read and encode audio file
    print(f"📁 Reading audio file: {audio_file_path}")
    try:
        with open(audio_file_path, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    print(f"📊 File size: {len(audio_data)} bytes")
    print(f"🔤 Base64 length: {len(audio_base64)} characters")
    
    # Prepare request
    url = f"{api_url}/detect"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    
    data = {
        "audio_base64": audio_base64,
        "language": language
    }
    
    # Make request
    print(f"\n🚀 Sending request to {url}")
    print(f"🌍 Language: {language}")
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        # Print response
        print(f"\n📡 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Detection Result:")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"🎯 Result: {result['result']}")
            print(f"📊 Confidence: {result['confidence']:.2%}")
            print(f"🌐 Language: {result['language']}")
            print(f"💡 Explanation: {result['explanation']}")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        else:
            print(f"\n❌ Error Response:")
            print(json.dumps(response.json(), indent=2))
            
    except requests.exceptions.Timeout:
        print("\n❌ Request timeout (>30s)")
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection error - is the API running?")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def test_health(api_url: str = "http://localhost:8000"):
    """Test the health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"❌ Health check failed: {e}")


if __name__ == "__main__":
    print("🎙️ Voice Deepfake Detection API - Test Script")
    print("=" * 50)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python test_api.py <audio_file.mp3> [language] [api_url] [api_key]")
        print("\nExample:")
        print("  python test_api.py sample.mp3 en http://localhost:8000 your-api-key")
        print("\nLanguages: ta, en, hi, ml, te")
        sys.exit(1)
    
    # Parse arguments
    audio_file = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "en"
    api_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000"
    api_key = sys.argv[4] if len(sys.argv) > 4 else "your-secret-api-key-here"
    
    # Test health first
    print("\n1️⃣ Health Check:")
    test_health(api_url)
    
    # Test detection
    print("\n2️⃣ Deepfake Detection:")
    test_api(audio_file, language, api_url, api_key)