"""Facial Expression Recognition Module using MediaPipe"""

import numpy as np
import mediapipe as mp
from typing import Tuple, Optional

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
        mouth_left = landmarks[self.MOUTH_LEFT]
        mouth_right = landmarks[self.MOUTH_RIGHT]
        mouth_top = landmarks[self.MOUTH_TOP]
        mouth_bottom = landmarks[self.MOUTH_BOTTOM]
        
        width = abs(mouth_left.x - mouth_right.x)
        height = abs(mouth_top.y - mouth_bottom.y)
        
        return height / (width + 1e-5)
    
    def extract_eye_openness(self, landmarks) -> float:
        """Calculate eye openness ratio"""
        left_eye_left = landmarks[self.LEFT_EYE_LEFT]
        left_eye_right = landmarks[self.LEFT_EYE_RIGHT]
        right_eye_left = landmarks[self.RIGHT_EYE_LEFT]
        right_eye_right = landmarks[self.RIGHT_EYE_RIGHT]
        
        left_width = abs(left_eye_left.x - left_eye_right.x)
        right_width = abs(right_eye_left.x - right_eye_right.x)
        
        return (left_width + right_width) / 2
    
    def recognize(self, image) -> Tuple[str, float]:
        """Recognize facial expression from image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Tuple of (expression, confidence)
        """
        try:
            import cv2
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            # If cv2 not available, assume RGB
            image_rgb = image
        
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Extract features
            mouth_distance = self.extract_mouth_distance(landmarks)
            eye_openness = self.extract_eye_openness(landmarks)
            
            # Simple heuristic-based classification
            if mouth_distance > 0.15:  # Mouth wide open
                return "Happy", 0.85
            elif mouth_distance > 0.08 and eye_openness < 0.25:
                return "Angry", 0.80
            elif mouth_distance < 0.05 and eye_openness < 0.20:
                return "Sad", 0.75
            else:
                return "Neutral", 0.90
        
        return "Unknown", 0.0
    
    def draw_face_mesh(self, image, face_landmarks) -> None:
        """Draw face mesh on image
        
        Args:
            image: Input image (in-place modification)
            face_landmarks: MediaPipe face landmarks
        """
        if face_landmarks is None:
            return
        
        try:
            import cv2
            h, w, _ = image.shape
            
            # Draw some key facial features
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (x, y), 1, (0, 255, 0), 1)
        except:
            pass
    
    def close(self):
        """Close MediaPipe resources"""
        self.face_mesh.close()
