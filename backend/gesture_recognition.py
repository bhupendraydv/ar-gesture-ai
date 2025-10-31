"""Gesture Recognition Module using MediaPipe and scikit-learn"""

import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict
import os
import pickle

class GestureRecognizer:
    """Hand gesture recognition using MediaPipe and ML classifier"""
    
    # Gesture labels
    GESTURES = ["Hello", "Help", "Yes", "No", "Stop", "Neutral"]
    
    def __init__(self, model_path: str = "./backend/models/gesture_model.pkl"):
        """Initialize gesture recognizer
        
        Args:
            model_path: Path to trained model file
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.model_path = model_path
        self.classifier = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load trained model from disk"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.classifier = model_data['classifier']
                    self.scaler = model_data['scaler']
                print(f"Loaded gesture model from {self.model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                self._init_default_model()
        else:
            print(f"Model not found at {self.model_path}. Using default model.")
            self._init_default_model()
    
    def _init_default_model(self):
        """Initialize default RandomForest classifier"""
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        self.scaler = StandardScaler()
        print("Initialized default RandomForest classifier")
    
    def extract_landmarks(self, hand_landmarks) -> Optional[np.ndarray]:
        """Extract hand landmarks as feature vector
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            Feature vector (42D: 21 landmarks * 2 coordinates)
        """
        if hand_landmarks is None:
            return None
        
        features = []
        for landmark in hand_landmarks.landmark:
            features.append(landmark.x)
            features.append(landmark.y)
        
        return np.array(features).reshape(1, -1)
    
    def recognize(self, image) -> Tuple[Optional[str], float]:
        """Recognize gesture from image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Tuple of (gesture_label, confidence_score)
        """
        if self.classifier is None:
            return "Unknown", 0.0
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            landmarks = results.multi_hand_landmarks[0]
            features = self.extract_landmarks(landmarks)
            
            if features is not None:
                features_scaled = self.scaler.transform(features)
                probabilities = self.classifier.predict_proba(features_scaled)[0]
                confidence = np.max(probabilities)
                gesture_idx = np.argmax(probabilities)
                
                # Return gesture and confidence
                return self.GESTURES[gesture_idx], float(confidence)
        
        return "No Hand", 0.0
    
    def draw_landmarks(self, image, hand_landmarks) -> None:
        """Draw hand landmarks on image
        
        Args:
            image: Input image (in-place modification)
            hand_landmarks: MediaPipe hand landmarks
        """
        if hand_landmarks is None:
            return
        
        import cv2
        h, w, _ = image.shape
        
        # Draw connections
        for connection in self.mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = hand_landmarks.landmark[start_idx]
            end = hand_landmarks.landmark[end_idx]
            
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
    
    def close(self):
        """Close MediaPipe resources"""
        self.hands.close()

# For module imports
try:
    import cv2
except ImportError:
    cv2 = None
