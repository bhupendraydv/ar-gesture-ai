"""HUD Elements for UI Overlay"""

import cv2
import numpy as np
from typing import Tuple

class HUDRenderer:
    """Renders HUD overlay with gesture, expression, and FPS information"""
    
    def __init__(self, 
                 font=cv2.FONT_HERSHEY_SIMPLEX,
                 font_scale=0.6,
                 font_thickness=2):
        """Initialize HUD renderer
        
        Args:
            font: OpenCV font type
            font_scale: Font size scale
            font_thickness: Text thickness
        """
        self.font = font
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        
        # Colors (BGR format)
        self.color_gesture = (0, 255, 0)  # Green
        self.color_expression = (0, 165, 255)  # Orange
        self.color_confidence = (255, 0, 0)  # Blue
        self.color_fps = (0, 255, 255)  # Cyan
        self.color_background = (20, 20, 20)  # Dark gray
        self.color_text = (255, 255, 255)  # White
    
    def draw_overlay(self,
                     frame: np.ndarray,
                     gesture: str,
                     gesture_confidence: float,
                     expression: str,
                     expression_confidence: float,
                     fps: float) -> np.ndarray:
        """Draw complete HUD overlay on frame
        
        Args:
            frame: Input image
            gesture: Recognized gesture
            gesture_confidence: Gesture confidence score
            expression: Recognized expression
            expression_confidence: Expression confidence score
            fps: Frames per second
            
        Returns:
            Frame with HUD overlay
        """
        h, w, _ = frame.shape
        
        # Draw semi-transparent background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), self.color_background, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (10, 10), (400, 200), self.color_text, 2)
        
        y_offset = 35
        line_height = 30
        
        # Draw title
        cv2.putText(frame, "GESTURE AI SYSTEM", (20, y_offset),
                   self.font, 0.7, self.color_gesture, self.font_thickness)
        
        y_offset += line_height
        
        # Draw gesture info
        gesture_text = f"Gesture: {gesture}"
        cv2.putText(frame, gesture_text, (20, y_offset),
                   self.font, self.font_scale, self.color_gesture, self.font_thickness)
        
        # Draw gesture confidence
        conf_text = f"Conf: {gesture_confidence:.2f}"
        cv2.putText(frame, conf_text, (250, y_offset),
                   self.font, 0.5, self.color_confidence, 1)
        
        y_offset += line_height
        
        # Draw expression info
        expr_text = f"Expression: {expression}"
        cv2.putText(frame, expr_text, (20, y_offset),
                   self.font, self.font_scale, self.color_expression, self.font_thickness)
        
        # Draw expression confidence
        expr_conf_text = f"Conf: {expression_confidence:.2f}"
        cv2.putText(frame, expr_conf_text, (250, y_offset),
                   self.font, 0.5, self.color_confidence, 1)
        
        # Draw FPS in top right corner
        fps_text = f"FPS: {fps:.1f}"
        text_size = cv2.getTextSize(fps_text, self.font, 0.7, 2)[0]
        fps_x = w - text_size[0] - 10
        cv2.putText(frame, fps_text, (fps_x, 40),
                   self.font, 0.7, self.color_fps, 2)
        
        # Draw status bar at bottom
        cv2.rectangle(frame, (0, h - 25), (w, h), (30, 30, 30), -1)
        status_text = "Press 'q' to quit | 'r' to record"
        cv2.putText(frame, status_text, (10, h - 8),
                   self.font, 0.5, self.color_text, 1)
        
        return frame
    
    def draw_confidence_bar(self,
                           frame: np.ndarray,
                           x: int,
                           y: int,
                           width: int,
                           height: int,
                           confidence: float,
                           label: str = "") -> np.ndarray:
        """Draw a confidence bar
        
        Args:
            frame: Input image
            x, y: Position
            width, height: Size
            confidence: Confidence value (0-1)
            label: Label text
            
        Returns:
            Frame with confidence bar
        """
        # Draw background bar
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                     (50, 50, 50), -1)
        
        # Draw filled bar
        filled_width = int(width * min(max(confidence, 0), 1))
        color = self.color_gesture if confidence > 0.7 else self.color_confidence
        cv2.rectangle(frame, (x, y), (x + filled_width, y + height),
                     color, -1)
        
        # Draw border
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                     self.color_text, 1)
        
        # Draw label and value
        if label:
            text = f"{label}: {confidence:.0%}"
            cv2.putText(frame, text, (x, y - 5),
                       self.font, 0.4, self.color_text, 1)
        
        return frame
