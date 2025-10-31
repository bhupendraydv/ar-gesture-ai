"""HUD Elements for UI Overlay - Real-time gesture and expression visualization"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


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
        self.color_gesture = (0, 255, 0)         # Green
        self.color_expression = (0, 165, 255)   # Orange
        self.color_confidence = (255, 0, 0)     # Blue
        self.color_fps = (0, 255, 255)          # Cyan
        self.color_background = (20, 20, 20)    # Dark gray
        self.color_text = (255, 255, 255)       # White
    
    def draw_overlay(self,
                     frame: np.ndarray,
                     gesture: str,
                     gesture_confidence: float,
                     expression: str,
                     expression_confidence: float,
                     fps: float) -> np.ndarray:
        """Draw overlay with all information
        
        Args:
            frame: Input image
            gesture: Detected gesture
            gesture_confidence: Gesture confidence (0-1)
            expression: Detected expression
            expression_confidence: Expression confidence (0-1)
            fps: Frames per second
            
        Returns:
            Frame with overlay drawn
        """
        try:
            if frame is None:
                logger.warning("Frame is None")
                return None
            
            h, w = frame.shape[:2]
            
            # Draw semi-transparent background for text
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 150), self.color_background, -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Draw gesture information
            gesture_text = f"Gesture: {gesture}"
            cv2.putText(frame, gesture_text, (20, 40),
                       self.font, self.font_scale, self.color_gesture, self.font_thickness)
            
            # Draw gesture confidence bar
            frame = self.draw_confidence_bar(frame, 20, 55, 150, 15,
                                            gesture_confidence, "Confidence")
            
            # Draw expression information
            expression_text = f"Expression: {expression}"
            cv2.putText(frame, expression_text, (220, 40),
                       self.font, self.font_scale, self.color_expression, self.font_thickness)
            
            # Draw expression confidence bar
            frame = self.draw_confidence_bar(frame, 220, 55, 150, 15,
                                            expression_confidence, "")
            
            # Draw FPS counter
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
        except Exception as e:
            logger.error(f"Error drawing overlay: {e}")
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
        try:
            if frame is None:
                logger.warning("Frame is None")
                return None
            
            # Validate confidence value
            if not isinstance(confidence, (int, float)):
                logger.warning(f"Invalid confidence type: {type(confidence)}")
                confidence = 0.0
            
            # Clamp confidence to 0-1 range
            confidence = max(0.0, min(1.0, confidence))
            
            # Draw background bar
            cv2.rectangle(frame, (x, y), (x + width, y + height),
                         (50, 50, 50), -1)
            
            # Draw filled bar
            filled_width = int(width * confidence)
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
        except Exception as e:
            logger.error(f"Error drawing confidence bar: {e}")
            return frame
    
    def draw_text_box(self,
                     frame: np.ndarray,
                     text: str,
                     position: Tuple[int, int],
                     bg_color: Tuple[int, int, int] = None,
                     text_color: Tuple[int, int, int] = None) -> np.ndarray:
        """Draw text with background box
        
        Args:
            frame: Input image
            text: Text to draw
            position: (x, y) position
            bg_color: Background color (BGR)
            text_color: Text color (BGR)
            
        Returns:
            Frame with text box
        """
        try:
            if frame is None or not text:
                return frame
            
            bg_color = bg_color or self.color_background
            text_color = text_color or self.color_text
            
            x, y = position
            text_size = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)[0]
            
            # Draw background rectangle
            cv2.rectangle(frame, (x - 5, y - text_size[1] - 5),
                         (x + text_size[0] + 5, y + 5),
                         bg_color, -1)
            
            # Draw text
            cv2.putText(frame, text, position,
                       self.font, self.font_scale, text_color, self.font_thickness)
            
            return frame
        except Exception as e:
            logger.error(f"Error drawing text box: {e}")
            return frame
