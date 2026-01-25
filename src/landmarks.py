#!/usr/bin/env python3
"""
5-point facial landmark detection using MediaPipe FaceMesh
Extracts exactly 5 landmarks in fixed order as specified in the document
"""

import cv2
import numpy as np
import mediapipe as mp

class FivePtLandmarkDetector:
    """5-point landmark detector using MediaPipe FaceMesh"""
    
    def __init__(self):
        """Initialize MediaPipe FaceMesh"""
        self.mp_face_mesh = mp.solutions.face_mesh if hasattr(mp, 'solutions') else None
        if self.mp_face_mesh is None:
            # Fallback for newer MediaPipe versions
            self.face_mesh = mp.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        
        
        # MediaPipe FaceMesh landmark indices for 5 key points
        # These indices correspond to the 468-point face mesh
        self.landmark_indices = {
            'left_eye': 33,      # Left eye center
            'right_eye': 263,    # Right eye center  
            'nose_tip': 1,       # Nose tip
            'left_mouth': 61,    # Left mouth corner
            'right_mouth': 291   # Right mouth corner
        }
        
        print("5-point landmark detector initialized with MediaPipe FaceMesh")
        
    def detect_landmarks(self, image, face_bbox=None):
        """
        Detect 5-point landmarks in image
        
        Args:
            image: Input image (BGR)
            face_bbox: Optional face bounding box (x, y, w, h) for ROI processing
            
        Returns:
            numpy.ndarray: 5x2 array of landmark coordinates [(x,y), ...] in fixed order:
                          [left_eye, right_eye, nose_tip, left_mouth, right_mouth]
                          Returns None if no face detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
            
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract 5 key points in fixed order
        h, w = image.shape[:2]
        landmarks_5pt = []
        
        # Fixed order as specified in document
        point_order = ['left_eye', 'right_eye', 'nose_tip', 'left_mouth', 'right_mouth']
        
        for point_name in point_order:
            idx = self.landmark_indices[point_name]
            landmark = face_landmarks.landmark[idx]
            
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_5pt.append([x, y])
            
        return np.array(landmarks_5pt, dtype=np.float32)
        
    def draw_landmarks(self, image, landmarks_5pt, color=(0, 255, 0), radius=3):
        """
        Draw 5-point landmarks on image
        
        Args:
            image: Input image
            landmarks_5pt: 5x2 array of landmark coordinates
            color: Point color (BGR)
            radius: Point radius
            
        Returns:
            numpy.ndarray: Image with drawn landmarks
        """
        if landmarks_5pt is None:
            return image.copy()
            
        result = image.copy()
        point_names = ['L_Eye', 'R_Eye', 'Nose', 'L_Mouth', 'R_Mouth']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, (x, y) in enumerate(landmarks_5pt):
            x, y = int(x), int(y)
            cv2.circle(result, (x, y), radius, colors[i], -1)
            cv2.putText(result, point_names[i], (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)
            
        return result

def test_landmarks():
    """Test 5-point landmark detection with camera"""
    print("Testing 5-point landmark detection...")
    
    try:
        from camera import Camera
        from detect import HaarFaceDetector
        
        detector = HaarFaceDetector()
        landmark_detector = FivePtLandmarkDetector()
        
        with Camera() as camera:
            print("Landmark detection test started")
            print("Press 'q' to quit, 's' to save landmarks")
            
            while True:
                ret, frame = camera.read_frame()
                if not ret:
                    print("Failed to read frame")
                    break
                    
                # Detect faces first
                faces = detector.detect_faces(frame)
                
                # Process first face if detected
                landmarks_5pt = None
                if faces:
                    landmarks_5pt = landmark_detector.detect_landmarks(frame, faces[0])
                
                # Draw results
                result = detector.draw_faces(frame, faces, color=(255, 0, 0))
                if landmarks_5pt is not None:
                    result = landmark_detector.draw_landmarks(result, landmarks_5pt)
                
                # Display info
                cv2.putText(result, f"Faces: {len(faces)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if landmarks_5pt is not None:
                    cv2.putText(result, "5-pt landmarks detected", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show landmark coordinates
                    point_names = ['L_Eye', 'R_Eye', 'Nose', 'L_Mouth', 'R_Mouth']
                    for i, (x, y) in enumerate(landmarks_5pt):
                        cv2.putText(result, f"{point_names[i]}: ({int(x)},{int(y)})", 
                                   (10, 110 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    cv2.putText(result, "No landmarks detected", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(result, "Press 'q' to quit, 's' to save", (10, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('5-Point Landmark Detection', result)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and landmarks_5pt is not None:
                    # Save landmarks to file
                    filename = "landmarks_5pt.txt"
                    np.savetxt(filename, landmarks_5pt, fmt='%.2f')
                    print(f"Saved landmarks to {filename}")
                    
    except Exception as e:
        print(f"Landmark test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

def test_landmarks_image(image_path):
    """Test landmark detection on a single image"""
    import os
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
        
    from detect import HaarFaceDetector
    
    detector = HaarFaceDetector()
    landmark_detector = FivePtLandmarkDetector()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
        
    # Detect faces
    faces = detector.detect_faces(image)
    print(f"Detected {len(faces)} faces")
    
    if faces:
        # Detect landmarks
        landmarks_5pt = landmark_detector.detect_landmarks(image, faces[0])
        
        if landmarks_5pt is not None:
            print("5-point landmarks detected:")
            point_names = ['Left Eye', 'Right Eye', 'Nose Tip', 'Left Mouth', 'Right Mouth']
            for i, (x, y) in enumerate(landmarks_5pt):
                print(f"  {point_names[i]}: ({x:.1f}, {y:.1f})")
            
            # Draw and display
            result = detector.draw_faces(image, faces, color=(255, 0, 0))
            result = landmark_detector.draw_landmarks(result, landmarks_5pt)
            
            cv2.imshow('5-Point Landmarks - Image', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No landmarks detected")
    else:
        print("No faces detected")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test on image file
        test_landmarks_image(sys.argv[1])
    else:
        # Test with camera
        test_landmarks()