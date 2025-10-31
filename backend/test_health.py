"""Test suite for health check and basic connectivity"""

import requests
import json
import time

def test_api_health():
    """Test API health endpoint"""
    api_url = "http://localhost:8000"
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "status" in data, "Missing 'status' field"
        assert "service" in data, "Missing 'service' field"
        assert "timestamp" in data, "Missing 'timestamp' field"
        assert data["status"] == "healthy", f"Status should be 'healthy', got {data['status']}"
        print("✓ Health check passed")
        return True
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API. Make sure the backend is running on http://localhost:8000")
        return False
    except AssertionError as e:
        print(f"✗ Health check failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_log_event():
    """Test event logging"""
    api_url = "http://localhost:8000"
    try:
        event = {
            "gesture": "Hello",
            "expression": "Happy",
            "confidence": 0.92
        }
        response = requests.post(f"{api_url}/log_event", json=event, timeout=5)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "id" in data, "Missing 'id' field in response"
        assert data["gesture"] == event["gesture"], f"Gesture mismatch"
        assert data["expression"] == event["expression"], f"Expression mismatch"
        assert data["confidence"] == event["confidence"], f"Confidence mismatch"
        print(f"✓ Event logging passed (ID: {data['id']})")
        return True
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API")
        return False
    except AssertionError as e:
        print(f"✗ Event logging test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_get_events():
    """Test retrieving events"""
    api_url = "http://localhost:8000"
    try:
        response = requests.get(f"{api_url}/events?limit=10", timeout=5)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert isinstance(data, list), "Expected response to be a list"
        print(f"✓ Retrieved {len(data)} events")
        return True
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API")
        return False
    except AssertionError as e:
        print(f"✗ Get events test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*50)
    print("Running Gesture AI Backend Tests")
    print("="*50 + "\n")
    
    results = []
    
    print("Test 1: Health Check")
    results.append(("Health Check", test_api_health()))
    print()
    
    print("Test 2: Log Event")
    results.append(("Log Event", test_log_event()))
    print()
    
    print("Test 3: Get Events")
    results.append(("Get Events", test_get_events()))
    print()
    
    print("="*50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("="*50 + "\n")
    
    return all(result for _, result in results)

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
