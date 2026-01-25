#!/usr/bin/env python3
"""
Camera module for face recognition system
Handles camera initialization and frame capture
"""

import cv2
import numpy as np

class Camera:
    """Camera handler for video capture"""
    
    def __init__(self, camera_id=0, width=640, height=480):
        """
        Initialize camera
        
        Args:
            camera_id: Camera device ID (default 0)
            width: Frame width
            height: Frame height
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        
    def start(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera {self.camera_id} started: {self.width}x{self.height}")
        
    def read_frame(self):
        """
        Read a frame from camera
        
        Returns:
            tuple: (success, frame) where success is bool and frame is numpy array
        """
        if self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        return ret, frame
        
    def stop(self):
        """Stop camera capture"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Camera stopped")
            
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

def test_camera():
    """Test camera functionality"""
    print("Testing camera...")
    
    try:
        with Camera() as camera:
            print("Camera initialized successfully")
            print("Press 'q' to quit, 's' to save frame")
            
            frame_count = 0
            while True:
                ret, frame = camera.read_frame()
                if not ret:
                    print("Failed to read frame")
                    break
                    
                frame_count += 1
                
                # Display frame info
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Camera Test', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"test_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame: {filename}")
                    
    except Exception as e:
        print(f"Camera test failed: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()