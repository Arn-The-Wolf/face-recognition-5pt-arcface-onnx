#!/usr/bin/env python3
"""
Face detection module using Haar cascades
Implements CPU-only face detection as specified in the document
"""

import cv2
import numpy as np
import os

class HaarFaceDetector:
    """Haar cascade face detector"""
    
    def __init__(self):
        """Initialize Haar cascade detector"""
        # Load the default OpenCV Haar cascade for frontal faces
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade not found: {cascade_path}")
            
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade")
            
        print("Haar face detector initialized")
        
    def detect_faces(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces in image using Haar cascades
        
        Args:
            image: Input image (BGR or grayscale)
            scale_factor: How much the image size is reduced at each scale
            min_neighbors: How many neighbors each candidate rectangle should have to retain it
            min_size: Minimum possible face size
            
        Returns:
            list: List of face bounding boxes as (x, y, w, h) tuples
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list of tuples
        face_list = []
        for (x, y, w, h) in faces:
            face_list.append((x, y, w, h))
            
        return face_list
        
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes around detected faces
        
        Args:
            image: Input image
            faces: List of face bounding boxes
            color: Rectangle color (BGR)
            thickness: Rectangle thickness
            
        Returns:
            numpy.ndarray: Image with drawn rectangles
        """
        result = image.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            
        return result

def test_detection():
    """Test face detection with camera"""
    print("Testing Haar face detection...")
    
    try:
        # Import camera module
        from camera import Camera
        
        detector = HaarFaceDetector()
        
        with Camera() as camera:
            print("Face detection test started")
            print("Press 'q' to quit")
            
            while True:
                ret, frame = camera.read_frame()
                if not ret:
                    print("Failed to read frame")
                    break
                    
                # Detect faces
                faces = detector.detect_faces(frame)
                
                # Draw faces
                result = detector.draw_faces(frame, faces)
                
                # Display info
                cv2.putText(result, f"Faces detected: {len(faces)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result, "Press 'q' to quit", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show face coordinates
                for i, (x, y, w, h) in enumerate(faces):
                    cv2.putText(result, f"Face {i+1}: ({x},{y}) {w}x{h}", 
                               (10, 110 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                cv2.imshow('Face Detection Test', result)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
    except Exception as e:
        print(f"Detection test failed: {e}")
    finally:
        cv2.destroyAllWindows()

def test_detection_image(image_path):
    """Test detection on a single image"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
        
    detector = HaarFaceDetector()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
        
    # Detect faces
    faces = detector.detect_faces(image)
    print(f"Detected {len(faces)} faces in {image_path}")
    
    # Draw and display
    result = detector.draw_faces(image, faces)
    
    cv2.imshow('Face Detection - Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test on image file
        test_detection_image(sys.argv[1])
    else:
        # Test with camera
        test_detection()