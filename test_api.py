"""Test script for local API testing before Azure deployment."""

import requests
import json

# API URL - change this after deployment
API_URL = "http://localhost:8000"


def test_health() -> None:
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_single_prediction() -> None:
    """Test single prediction."""
    print("Testing /predict endpoint...")

    # Test case: Weekday evening rush hour
    payload = {
        "distance_miles": 5.5,
        "time_of_day_hours": 17.5,
        "is_weekend": 0,
    }

    response = requests.post(f"{API_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Request: {json.dumps(payload, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_batch_predictions() -> None:
    """Test batch prediction."""
    print("Testing /predict/batch endpoint...")

    # Multiple test cases
    payloads = [
        {
            "distance_miles": 5.5,
            "time_of_day_hours": 9.0,
            "is_weekend": 0,
        },  # Morning rush
        {
            "distance_miles": 12.0,
            "time_of_day_hours": 14.5,
            "is_weekend": 1,
        },  # Weekend afternoon
        {
            "distance_miles": 3.0,
            "time_of_day_hours": 19.5,
            "is_weekend": 0,
        },  # Evening after rush
    ]

    response = requests.post(f"{API_URL}/predict/batch", json=payloads)
    print(f"Status: {response.status_code}")
    print(f"Request: {json.dumps(payloads, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_validation_errors() -> None:
    """Test input validation."""
    print("Testing input validation...")

    # Invalid distance (too high)
    invalid_payload = {
        "distance_miles": 25.0,  # Max is 20
        "time_of_day_hours": 12.0,
        "is_weekend": 0,
    }

    response = requests.post(f"{API_URL}/predict", json=invalid_payload)
    print(f"Status: {response.status_code}")
    print(f"Error response: {json.dumps(response.json(), indent=2)}\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("API Testing Script")
    print("=" * 60 + "\n")

    try:
        test_health()
        test_single_prediction()
        test_batch_predictions()
        test_validation_errors()

        print("=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API")
        print("Make sure the API is running:")
        print("  uvicorn app:app --reload")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
