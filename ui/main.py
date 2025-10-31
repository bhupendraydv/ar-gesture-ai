"""UI Main Application - Real-time Gesture & Facial Expression Recognition"""

import cv2
import numpy as np
import requests
from typing import Tuple, Optional
from hud_elements import HUDRenderer
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from backend.gesture_recognition import GestureRecognizer
    from backend.facial_expression import FacialExpressionRecognizer
except ImportError:
    print("Warning: Backend modules not available. Running in limited mode.")
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
        
        # Initialize recognizers
        if GestureRecognizer:
            self.gesture_recognizer = GestureRecognizer()
        else:
            self.gesture_recognizer = None
        
        if FacialExpressionRecognizer:
            self.facial_recognizer = FacialExpressionRecognizer()
        else:
            self.facial_recognizer = None
        
        # Initialize HUD
        self.hud = HUDRenderer()
        
        # Video capture
        self.cap = None
    
    def check_api_health(self) -> bool:
        """Check if FastAPI backend is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=2)
            return response.status_code == 200
        except:
            print(f"⚠️  Backend not available at {self.api_url}")
            print("   Running in offline mode (events won't be logged)")
            return False
    
    def log_event(self, gesture: str, expression: str, confidence: float):
        """Log an event to the backend API"""
        try:
            payload = {
                "gesture": gesture,
                "expression": expression,
                "confidence": confidence
            }
            response = requests.post(f"{self.api_url}/log_event", json=payload, timeout=2)
            if response.status_code != 200:
                pass  # Silent fail in UI
        except:
            pass  # Silent fail if API not available
    
    def run(self, camera_id: int = 0):
        """Run the main UI loop
        
        Args:
            camera_id: Camera device index
        """
        print("\n🎥 Starting Gesture AI UI...")
        print("Press 'q' to quit, 'r' to record event\n")
        
        # Check API health
        api_available = self.check_api_health()
        
        # Open camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print("❌ Failed to open camera")
            return
        
        self.running = True
        prev_frame_time = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Recognize gesture
            gesture = "Unknown"
            gesture_conf = 0.0
            if self.gesture_recognizer:
                gesture, gesture_conf = self.gesture_recognizer.recognize(frame)
            
            # Recognize facial expression
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
                print("\n👋 Exiting...")
                self.running = False
            elif key == ord('r') and gesture != "Unknown":
                print(f"📝 Recording event: {gesture} + {expression}")
                if api_available:
                    self.log_event(gesture, expression, gesture_conf)
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if self.gesture_recognizer:
            self.gesture_recognizer.close()
        if self.facial_recognizer:
            self.facial_recognizer.close()
        cv2.destroyAllWindows()
        print("✅ UI closed successfully")

def main():
    """Main entry point"""
    app = GestureAIUI(api_url="http://localhost:8000")
    app.run()

if __name__ == "__main__":
    main()
