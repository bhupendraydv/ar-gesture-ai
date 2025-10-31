"""UI Main Application - Real-time Gesture & Facial Expression Recognition"""

import cv2
import numpy as np
import requests
from typing import Tuple, Optional
from hud_elements import HUDRenderer
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from backend.gesture_recognition import GestureRecognizer
    from backend.facial_expression import FacialExpressionRecognizer
    logger.info("Backend modules imported successfully")
except ImportError as e:
    logger.warning(f"Backend modules not available: {e}. Running in limited mode.")
    GestureRecognizer = None
    FacialExpressionRecognizer = None


class GestureAIUI:
    """Main UI application for Gesture & Facial Expression Recognition"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        """Initialize the UI application
        
        Args:
            api_url: FastAPI backend URL
        """
        self.api_url = api_url
        self.running = False
        self.fps = 0
        self.frame_count = 0
        self.cap = None
        self.hud = None
        
        # Initialize recognizers
        self.gesture_recognizer = None
        self.facial_recognizer = None
        
        if GestureRecognizer:
            try:
                self.gesture_recognizer = GestureRecognizer()
                logger.info("Gesture recognizer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize gesture recognizer: {e}")
        
        if FacialExpressionRecognizer:
            try:
                self.facial_recognizer = FacialExpressionRecognizer()
                logger.info("Facial expression recognizer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize facial expression recognizer: {e}")
        
        # Initialize HUD renderer
        try:
            self.hud = HUDRenderer()
            logger.info("HUD renderer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize HUD renderer: {e}")
        
        # Initialize camera
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
            else:
                logger.info("Camera opened successfully")
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
    
    def check_api_health(self) -> bool:
        """Check if API is available"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"API health check failed: {e}")
            return False
    
    def log_event(self, gesture: str, expression: str, confidence: float):
        """Log event to API
        
        Args:
            gesture: Detected gesture
            expression: Detected expression
            confidence: Confidence score
        """
        try:
            event_data = {
                "gesture": gesture,
                "expression": expression,
                "confidence": confidence
            }
            response = requests.post(
                f"{self.api_url}/log_event",
                json=event_data,
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"Event logged successfully: {gesture} + {expression}")
            else:
                logger.warning(f"Failed to log event. Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error logging event to API: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while logging event: {e}")
    
    def run(self):
        """Run the UI application"""
        logger.info("Starting Gesture AI Communication System UI")
        logger.info("Press 'q' to quit, 'r' to record event")
        
        if self.cap is None or not self.cap.isOpened():
            logger.error("Camera is not available. Exiting.")
            return
        
        self.running = True
        api_available = self.check_api_health()
        
        if not api_available:
            logger.warning("API is not available. Gesture logging will not work.")
        
        prev_frame_time = 0
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                self.frame_count += 1
                
                # Flip frame for selfie-view
                frame = cv2.flip(frame, 1)
                
                # Recognize gesture
                gesture = "Unknown"
                gesture_conf = 0.0
                if self.gesture_recognizer:
                    gesture, gesture_conf = self.gesture_recognizer.recognize(frame)
                
                # Recognize expression
                expression = "Unknown"
                expression_conf = 0.0
                if self.facial_recognizer:
                    expression, expression_conf = self.facial_recognizer.recognize(frame)
                
                # Calculate FPS
                current_frame_time = cv2.getTickCount() / cv2.getTickFrequency()
                fps = 1 / (current_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
                prev_frame_time = current_frame_time
                self.fps = fps * 0.7 + self.fps * 0.3  # Smooth FPS
                
                # Draw HUD overlay
                if self.hud:
                    frame = self.hud.draw_overlay(
                        frame,
                        gesture=gesture,
                        gesture_confidence=gesture_conf,
                        expression=expression,
                        expression_confidence=expression_conf,
                        fps=self.fps
                    )
                
                # Display frame
                cv2.imshow("Gesture AI Communication System", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("User quit application")
                    self.running = False
                elif key == ord('r') and gesture != "Unknown":
                    logger.info(f"Recording event: {gesture} + {expression}")
                    if api_available:
                        self.log_event(gesture, expression, gesture_conf)
            
            except KeyboardInterrupt:
                logger.info("Application interrupted by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                # Continue running despite errors
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.cap:
                self.cap.release()
            if self.gesture_recognizer:
                self.gesture_recognizer.close()
            if self.facial_recognizer:
                self.facial_recognizer.close()
            cv2.destroyAllWindows()
            logger.info("UI closed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main entry point"""
    try:
        app = GestureAIUI(api_url="http://localhost:8000")
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
