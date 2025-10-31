"""Test suite for gesture and facial expression models"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gesture_recognition import GestureRecognizer
from facial_expression import FacialExpressionRecognizer

def test_gesture_recognizer_init():
    """Test GestureRecognizer initialization"""
    try:
        recognizer = GestureRecognizer()
        assert recognizer.hands is not None, "Hands object is None"
        assert recognizer.classifier is not None or recognizer.scaler is not None, "Model not loaded"
        recognizer.close()
        print("✓ GestureRecognizer initialization passed")
        return True
    except Exception as e:
        print(f"✗ GestureRecognizer initialization failed: {e}")
        return False

def test_facial_expression_recognizer_init():
    """Test FacialExpressionRecognizer initialization"""
    try:
        recognizer = FacialExpressionRecognizer()
        assert recognizer.face_mesh is not None, "Face mesh object is None"
        assert len(recognizer.EXPRESSIONS) > 0, "No expressions defined"
        recognizer.close()
        print("✓ FacialExpressionRecognizer initialization passed")
        return True
    except Exception as e:
        print(f"✗ FacialExpressionRecognizer initialization failed: {e}")
        return False

def test_gesture_recognizer_output():
    """Test GestureRecognizer output format"""
    try:
        import numpy as np
        recognizer = GestureRecognizer()
        
        # Create dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        gesture, confidence = recognizer.recognize(dummy_image)
        assert isinstance(gesture, str), f"Gesture should be str, got {type(gesture)}"
        assert isinstance(confidence, float), f"Confidence should be float, got {type(confidence)}"
        assert 0 <= confidence <= 1, f"Confidence out of range: {confidence}"
        
        recognizer.close()
        print(f"✓ GestureRecognizer output format passed (Gesture: {gesture}, Confidence: {confidence:.2f})")
        return True
    except Exception as e:
        print(f"✗ GestureRecognizer output format test failed: {e}")
        return False

def test_facial_expression_recognizer_output():
    """Test FacialExpressionRecognizer output format"""
    try:
        import numpy as np
        recognizer = FacialExpressionRecognizer()
        
        # Create dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        expression, confidence = recognizer.recognize(dummy_image)
        assert isinstance(expression, str), f"Expression should be str, got {type(expression)}"
        assert isinstance(confidence, float), f"Confidence should be float, got {type(confidence)}"
        assert 0 <= confidence <= 1, f"Confidence out of range: {confidence}"
        
        recognizer.close()
        print(f"✓ FacialExpressionRecognizer output format passed (Expression: {expression}, Confidence: {confidence:.2f})")
        return True
    except Exception as e:
        print(f"✗ FacialExpressionRecognizer output format test failed: {e}")
        return False

def run_all_tests():
    """Run all model tests"""
    print("\n" + "="*60)
    print("Running Gesture AI Model Tests")
    print("="*60 + "\n")
    
    results = []
    
    print("Test 1: GestureRecognizer Initialization")
    results.append(("GestureRecognizer Init", test_gesture_recognizer_init()))
    print()
    
    print("Test 2: FacialExpressionRecognizer Initialization")
    results.append(("FacialExpressionRecognizer Init", test_facial_expression_recognizer_init()))
    print()
    
    print("Test 3: GestureRecognizer Output Format")
    results.append(("GestureRecognizer Output", test_gesture_recognizer_output()))
    print()
    
    print("Test 4: FacialExpressionRecognizer Output Format")
    results.append(("FacialExpressionRecognizer Output", test_facial_expression_recognizer_output()))
    print()
    
    print("="*60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60 + "\n")
    
    return all(result for _, result in results)

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
