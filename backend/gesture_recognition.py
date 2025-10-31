"""Gesture Recognition Module using MediaPipe and scikit-learn"""

import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict
import os
import pickle
import logging

try:
    import cv2
except ImportError as e:
    logging.warning(f"cv2 not available: {e}")
    cv2 = None

logger = logging.getLogger(__name__)


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
        """Load pre-trained model from disk"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.classifier = data.get('classifier')
                    self.scaler = data.get('scaler')
                logger.info(f"Model loaded from {self.model_path}")
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                # Create dummy classifier and scaler
                self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                self.scaler = StandardScaler()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def extract_landmarks(self, hand_landmarks) -> Optional[np.ndarray]:
        """Extract features from hand landmarks
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            np.ndarray: Feature vector (42D) or None if extraction fails
        """
        try:
            if hand_landmarks is None:
                return None
            
            features = []
            for landmark in hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
            
            if len(features) != 63:  # 21 landmarks * 3 coordinates
                logger.warning(f"Expected 63 features, got {len(features)}")
                return None
            
            return np.array(features).reshape(1, -1)
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None
    
    def recognize(self, image) -> Tuple[str, float]:
        """Recognize gesture in image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Tuple[str, float]: (gesture_name, confidence_score)
        """
        try:
            if self.classifier is None or self.scaler is None:
                logger.warning("Classifier or scaler not initialized")
                return "Unknown", 0.0
            
            # Flip image for selfie-view
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if cv2 else image
            h, w, c = image.shape
            
            # Detect hand landmarks
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
        except Exception as e:
            logger.error(f"Error during gesture recognition: {e}")
            return "Error", 0.0
    
    def draw_landmarks(self, image, hand_landmarks) -> None:
        """Draw hand landmarks on image
        
        Args:
            image: Input image (in-place modification)
            hand_landmarks: MediaPipe hand landmarks
        """
        try:
            if hand_landmarks is None or cv2 is None:
                return
            
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
        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")
    
    def close(self):
        """Close MediaPipe resources"""
        try:
            self.hands.close()
        except Exception as e:
            logger.error(f"Error closing resources: {e}")
