"""
Script to test the YOLO Object Detection API locally
"""
import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8080"


def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        print(f"✓ Health check passed: {response.json()}")
        return True
    else:
        print(f"✗ Health check failed: {response.status_code}")
        return False


def test_predict_image(image_path: str):
    """Test prediction endpoint with image"""
    print(f"\nTesting prediction with image: {image_path}")

    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        return False

    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{API_URL}/predict", files=files)

    if response.status_code == 200:
        output_path = "test_annotated_result.jpg"
        with open(output_path, "wb") as out:
            out.write(response.content)
        print(f"✓ Prediction successful! Annotated image saved to: {output_path}")
        return True
    else:
        print(f"✗ Prediction failed: {response.status_code} - {response.text}")
        return False


def test_predict_json(image_path: str):
    """Test JSON prediction endpoint"""
    print(f"\nTesting JSON prediction with image: {image_path}")

    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        return False

    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{API_URL}/predict/json", files=files)

    if response.status_code == 200:
        data = response.json()
        print(f"✓ JSON prediction successful!")
        print(f"  Found {data['count']} objects")
        print(f"  Image shape: {data['image_shape']}")
        for i, detection in enumerate(data['detections'], 1):
            print(f"  Detection {i}: {detection['class_name']} "
                  f"(confidence: {detection['confidence']:.2f})")
        return True
    else:
        print(f"✗ JSON prediction failed: {response.status_code} - {response.text}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("YOLO Object Detection API Test Suite")
    print("=" * 60)

    # Test health
    if not test_health():
        print("\n✗ API is not running or unhealthy. Start the service first:")
        print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8080")
        sys.exit(1)

    # Test with image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_predict_image(image_path)
        test_predict_json(image_path)
    else:
        print("\n⚠ No image provided. To test prediction endpoints, run:")
        print("  python test_api.py path/to/your/image.jpg")

    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
