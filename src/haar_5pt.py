#!/usr/bin/env python3
"""
Combined Haar detection + 5-point landmarks module
Integrates face detection and landmark detection as specified in the document
"""

import cv2
import numpy as np

class HaarFivePtDetector:
    """Combined Haar face detection + MediaPipe 5-point landmarks"""
    
    def __init__(self):
        """Initialize combined detector"""
        from detect import HaarFaceDetector
        from landmarks import FivePtLandmarkDetector
        
        self.face_detector = HaarFaceDetector()
        self.landmark_detector = FivePtLandmarkDetector()
        
        print("Combined Haar + 5-point detector initialized")
        
    def detect_faces_and_landmarks(self, image, return_largest=True):
        """
        Detect faces and extract 5-point landmarks
        
        Args:
            image: Input image (BGR)
            return_largest: If True, return only the largest face
            
        Returns:
            list: List of (face_bbox, landmarks_5pt) tuples
                  face_bbox: (x, y, w, h)
                  landmarks_5pt: 5x2 array or None if detection failed
        """
        # Detect faces
        faces = self.face_detector.detect_faces(image)
        
        if not faces:
            return []
            
        # Sort by area if return_largest is True
        if return_largest and len(faces) > 1:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            faces = [faces[0]]  # Keep only largest
            
        results = []
        
        # Extract landmarks for each face
        for face_bbox in faces:
            landmarks_5pt = self.landmark_detector.detect_landmarks(image, face_bbox)
            results.append((face_bbox, landmarks_5pt))
            
        return results
        
    def process_frame(self, image):
        """
        Process a single frame and return detection results
        
        Args:
            image: Input image (BGR)
            
        Returns:
            dict: Processing results with keys:
                  - faces: List of face bounding boxes
                  - landmarks: List of 5-point landmarks (same order as faces)
                  - largest_face: Largest face bbox or None
                  - largest_landmarks: Landmarks for largest face or None
        """
        # Get all detections
        detections = self.detect_faces_and_landmarks(image, return_largest=False)
        
        faces = []
        landmarks = []
        
        for face_bbox, landmarks_5pt in detections:
            faces.append(face_bbox)
            landmarks.append(landmarks_5pt)
            
        # Find largest face
        largest_face = None
        largest_landmarks = None
        
        if faces:
            # Find face with largest area
            areas = [w * h for (x, y, w, h) in faces]
            largest_idx = np.argmax(areas)
            largest_face = faces[largest_idx]
            largest_landmarks = landmarks[largest_idx]
            
        return {
            'faces': faces,
            'landmarks': landmarks,
            'largest_face': largest_face,
            'largest_landmarks': largest_landmarks,
            'num_faces': len(faces)
        }
        
    def draw_detections(self, image, detections, draw_all=True):
        """
        Draw detection results on image
        
        Args:
            image: Input image
            detections: Results from process_frame()
            draw_all: If True, draw all faces; if False, draw only largest
            
        Returns:
            numpy.ndarray: Image with drawn detections
        """
        result = image.copy()
        
        if draw_all:
            # Draw all faces and landmarks
            faces = detections['faces']
            landmarks_list = detections['landmarks']
            
            for i, (face_bbox, landmarks_5pt) in enumerate(zip(faces, landmarks_list)):
                # Draw face bounding box
                x, y, w, h = face_bbox
                color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for first, blue for others
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
                
                # Draw face number
                cv2.putText(result, f"Face {i+1}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw landmarks if available
                if landmarks_5pt is not None:
                    result = self.landmark_detector.draw_landmarks(result, landmarks_5pt)
        else:
            # Draw only largest face
            largest_face = detections['largest_face']
            largest_landmarks = detections['largest_landmarks']
            
            if largest_face is not None:
                x, y, w, h = largest_face
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result, "Largest Face", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if largest_landmarks is not None:
                    result = self.landmark_detector.draw_landmarks(result, largest_landmarks)
                    
        return result
        
    def get_face_quality_score(self, face_bbox, landmarks_5pt, image_shape):
        """
        Compute face quality score based on size and landmark geometry
        
        Args:
            face_bbox: Face bounding box (x, y, w, h)
            landmarks_5pt: 5-point landmarks
            image_shape: Image shape (h, w, c)
            
        Returns:
            float: Quality score [0, 1]
        """
        if landmarks_5pt is None:
            return 0.0
            
        x, y, w, h = face_bbox
        img_h, img_w = image_shape[:2]
        
        # Size score (larger faces are better)
        face_area = w * h
        img_area = img_h * img_w
        size_score = min(face_area / (img_area * 0.1), 1.0)  # Normalize to [0, 1]
        
        # Position score (centered faces are better)
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        img_center_x = img_w // 2
        img_center_y = img_h // 2
        
        center_dist = np.sqrt((face_center_x - img_center_x)**2 + (face_center_y - img_center_y)**2)
        max_dist = np.sqrt(img_center_x**2 + img_center_y**2)
        position_score = 1.0 - (center_dist / max_dist)
        
        # Landmark geometry score
        from align import FaceAligner
        aligner = FaceAligner()
        geometry = aligner.get_alignment_quality(landmarks_5pt)
        
        if geometry:
            geometry_score = geometry['quality_score']
        else:
            geometry_score = 0.0
            
        # Combined score
        quality_score = (size_score * 0.4 + position_score * 0.3 + geometry_score * 0.3)
        
        return quality_score

def test_haar_5pt():
    """Test combined Haar + 5-point detection"""
    print("Testing combined Haar + 5-point detection...")
    
    try:
        from camera import Camera
        
        detector = HaarFivePtDetector()
        
        with Camera() as camera:
            print("Combined detection test started")
            print("Press 'q' to quit, 's' to save results, 'a' to toggle show all faces")
            
            show_all = True
            
            while True:
                ret, frame = camera.read_frame()
                if not ret:
                    print("Failed to read frame")
                    break
                    
                # Process frame
                detections = detector.process_frame(frame)
                
                # Draw results
                result = detector.draw_detections(frame, detections, draw_all=show_all)
                
                # Display info
                cv2.putText(result, f"Faces detected: {detections['num_faces']}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if detections['largest_face'] is not None:
                    # Show quality score for largest face
                    quality = detector.get_face_quality_score(
                        detections['largest_face'], 
                        detections['largest_landmarks'], 
                        frame.shape
                    )
                    cv2.putText(result, f"Quality: {quality:.2f}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                mode_text = "All faces" if show_all else "Largest only"
                cv2.putText(result, f"Mode: {mode_text}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.putText(result, "Press 'q' quit, 's' save, 'a' toggle mode", (10, 400),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Haar + 5-Point Detection', result)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save results
                    cv2.imwrite('haar_5pt_result.jpg', result)
                    print("Saved detection result")
                    
                    # Save detection data
                    if detections['largest_landmarks'] is not None:
                        np.savetxt('haar_5pt_landmarks.txt', detections['largest_landmarks'], fmt='%.2f')
                        print("Saved landmark coordinates")
                elif key == ord('a'):
                    show_all = not show_all
                    print(f"Show all faces: {show_all}")
                    
    except Exception as e:
        print(f"Combined detection test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

def test_haar_5pt_image(image_path):
    """Test combined detection on a single image"""
    import os
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
        
    detector = HaarFivePtDetector()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
        
    # Process
    detections = detector.process_frame(image)
    
    print(f"Detection results for {image_path}:")
    print(f"  Faces detected: {detections['num_faces']}")
    
    if detections['largest_face'] is not None:
        x, y, w, h = detections['largest_face']
        print(f"  Largest face: ({x}, {y}) {w}x{h}")
        
        if detections['largest_landmarks'] is not None:
            print(f"  5-point landmarks detected")
            quality = detector.get_face_quality_score(
                detections['largest_face'], 
                detections['largest_landmarks'], 
                image.shape
            )
            print(f"  Quality score: {quality:.3f}")
        else:
            print(f"  No landmarks detected")
    else:
        print(f"  No faces detected")
        
    # Draw and display results
    result = detector.draw_detections(image, detections, draw_all=True)
    
    cv2.imshow('Haar + 5-Point Detection - Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    cv2.imwrite('haar_5pt_image_result.jpg', result)
    print("Saved result as 'haar_5pt_image_result.jpg'")

def benchmark_detection_speed():
    """Benchmark detection speed"""
    print("Benchmarking detection speed...")
    
    try:
        from camera import Camera
        import time
        
        detector = HaarFivePtDetector()
        
        with Camera() as camera:
            print("Collecting frames for benchmark...")
            
            # Collect test frames
            test_frames = []
            for i in range(30):
                ret, frame = camera.read_frame()
                if ret:
                    test_frames.append(frame)
                    
            if not test_frames:
                print("No frames captured")
                return
                
            print(f"Benchmarking with {len(test_frames)} frames...")
            
            # Benchmark
            start_time = time.time()
            total_faces = 0
            
            for frame in test_frames:
                detections = detector.process_frame(frame)
                total_faces += detections['num_faces']
                
            end_time = time.time()
            
            # Results
            total_time = end_time - start_time
            fps = len(test_frames) / total_time
            avg_faces = total_faces / len(test_frames)
            
            print(f"\nBenchmark Results:")
            print(f"  Frames processed: {len(test_frames)}")
            print(f"  Total time: {total_time:.2f} seconds")
            print(f"  FPS: {fps:.1f}")
            print(f"  Average faces per frame: {avg_faces:.1f}")
            print(f"  Time per frame: {total_time/len(test_frames)*1000:.1f} ms")
            
    except Exception as e:
        print(f"Benchmark failed: {e}")

def main():
    """Main interface for combined detector"""
    print("=== Haar + 5-Point Detection System ===")
    print("1. Live detection (camera)")
    print("2. Test on image file")
    print("3. Benchmark speed")
    print("4. Quit")
    
    choice = input("Choose option: ").strip()
    
    if choice == '1':
        test_haar_5pt()
    elif choice == '2':
        image_path = input("Enter image path: ").strip()
        if image_path:
            test_haar_5pt_image(image_path)
    elif choice == '3':
        benchmark_detection_speed()
    elif choice == '4':
        pass
    else:
        print("Invalid choice")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test on image file
        test_haar_5pt_image(sys.argv[1])
    else:
        # Interactive mode
        main()