"""Facial Expression Recognition Module using MediaPipe"""

import numpy as np
import mediapipe as mp
from typing import Tuple, Optional
import logging

try:
    import cv2
except ImportError as e:
    logging.warning(f"cv2 not available: {e}")
    cv2 = None

logger = logging.getLogger(__name__)


class FacialExpressionRecognizer:
    """Facial expression recognition using MediaPipe FaceMesh"""
    
    EXPRESSIONS = ["Neutral", "Happy", "Sad", "Angry", "Unknown"]
    
    def __init__(self):
        """Initialize facial expression recognizer"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key facial landmarks for emotion detection
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        
        self.LEFT_EYE_LEFT = 33
        self.LEFT_EYE_RIGHT = 133
        self.RIGHT_EYE_LEFT = 263
        self.RIGHT_EYE_RIGHT = 362
    
    def extract_mouth_distance(self, landmarks) -> float:
        """Calculate mouth opening distance"""
        try:
            mouth_left = landmarks[self.MOUTH_LEFT]
            mouth_right = landmarks[self.MOUTH_RIGHT]
            mouth_top = landmarks[self.MOUTH_TOP]
            mouth_bottom = landmarks[self.MOUTH_BOTTOM]
            
            # Calculate distances
            left_right_distance = np.sqrt(
                (mouth_left.x - mouth_right.x) ** 2 + 
                (mouth_left.y - mouth_right.y) ** 2
            )
            top_bottom_distance = np.sqrt(
                (mouth_top.y - mouth_bottom.y) ** 2
            )
            
            # Normalize by left-right distance
            if left_right_distance > 0:
                mouth_ratio = top_bottom_distance / (left_right_distance + 1e-5)
            else:
                mouth_ratio = 0.0
            
            return mouth_ratio
        except (IndexError, AttributeError) as e:
            logger.warning(f"Error extracting mouth distance: {e}")
            return 0.0
    
    def extract_eye_distance(self, landmarks) -> float:
        """Calculate eye opening distance"""
        try:
            left_eye_left = landmarks[self.LEFT_EYE_LEFT]
            left_eye_right = landmarks[self.LEFT_EYE_RIGHT]
            
            eye_distance = np.sqrt(
                (left_eye_left.x - left_eye_right.x) ** 2 + 
                (left_eye_left.y - left_eye_right.y) ** 2
            )
            
            return eye_distance
        except (IndexError, AttributeError) as e:
            logger.warning(f"Error extracting eye distance: {e}")
            return 0.0
    
    def recognize(self, image) -> Tuple[str, float]:
        """Recognize facial expression
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Tuple[str, float]: (expression_name, confidence_score)
        """
        try:
            if cv2 is None:
                logger.warning("cv2 not available")
                return "Unknown", 0.0
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face landmarks
            results = self.face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = face_landmarks.landmark
                
                mouth_distance = self.extract_mouth_distance(landmarks)
                eye_distance = self.extract_eye_distance(landmarks)
                
                # Simple heuristic-based expression recognition
                # Happy: mouth open + eyes open
                if mouth_distance > 0.02 and eye_distance > 0.01:
                    return "Happy", 0.85
                # Sad: mouth closed + eyes less open
                elif mouth_distance < 0.01 and eye_distance < 0.008:
                    return "Sad", 0.80
                # Angry: specific facial configuration
                elif mouth_distance > 0.015 and eye_distance < 0.009:
                    return "Angry", 0.75
                else:
                    return "Neutral", 0.90
            
            return "Unknown", 0.0
        except Exception as e:
            logger.error(f"Error during facial expression recognition: {e}")
            return "Error", 0.0
    
    def draw_face_mesh(self, image, face_landmarks) -> None:
        """Draw face mesh on image
        
        Args:
            image: Input image (in-place modification)
            face_landmarks: MediaPipe face landmarks
        """
        try:
            if face_landmarks is None or cv2 is None:
                return
            
            h, w, _ = image.shape
            
            # Draw some key facial features
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (x, y), 1, (0, 255, 0), 1)
        except Exception as e:
            logger.error(f"Error drawing face mesh: {e}")
    
    def close(self):
        """Close MediaPipe resources"""
        try:
            self.face_mesh.close()
        except Exception as e:
            logger.error(f"Error closing resources: {e}")
