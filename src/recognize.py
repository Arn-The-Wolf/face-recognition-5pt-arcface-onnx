#!/usr/bin/env python3
"""
Live face recognition module
Implements real-time face recognition as specified in the document
"""

import cv2
import numpy as np
import os
import json
from collections import deque

class FaceRecognizer:
    """Live face recognition system"""
    
    def __init__(self, data_dir="data", threshold=0.4):
        """
        Initialize face recognizer
        
        Args:
            data_dir: Base data directory
            threshold: Recognition threshold (cosine similarity)
        """
        self.data_dir = data_dir
        self.db_dir = os.path.join(data_dir, "db")
        self.threshold = threshold
        
        # Recognition parameters
        self.temporal_window = 10  # Frames for temporal smoothing
        self.acceptance_hold = 30  # Frames to hold recognition result
        self.process_every_n = 3   # Process every N frames for performance
        
        # State variables
        self.frame_count = 0
        self.recent_predictions = deque(maxlen=self.temporal_window)
        self.hold_counter = 0
        self.held_result = None
        
        # Load database
        self.database_embeddings = {}
        self.database_metadata = {}
        self._load_database()
        
        print(f"Face recognizer initialized")
        print(f"Threshold: {self.threshold}")
        print(f"Database: {len(self.database_embeddings)} identities")
        
    def _load_database(self):
        """Load face database"""
        db_embeddings_path = os.path.join(self.db_dir, "face_db.npz")
        db_metadata_path = os.path.join(self.db_dir, "face_db.json")
        
        if os.path.exists(db_embeddings_path):
            db_data = np.load(db_embeddings_path)
            self.database_embeddings = {key: db_data[key] for key in db_data.files}
            print(f"Loaded {len(self.database_embeddings)} embeddings from database")
        else:
            print(f"Database not found: {db_embeddings_path}")
            
        if os.path.exists(db_metadata_path):
            with open(db_metadata_path, 'r') as f:
                self.database_metadata = json.load(f)
                
    def recognize_face(self, embedding):
        """
        Recognize face from embedding
        
        Args:
            embedding: Face embedding vector (512,)
            
        Returns:
            tuple: (name, confidence) where name is string or None, confidence is float
        """
        if embedding is None or len(self.database_embeddings) == 0:
            return None, 0.0
            
        best_name = None
        best_similarity = -1.0
        
        # Compare with all database embeddings
        for name, db_embedding in self.database_embeddings.items():
            similarity = np.dot(embedding, db_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name
                
        # Apply threshold
        if best_similarity >= self.threshold:
            return best_name, best_similarity
        else:
            return None, best_similarity
            
    def apply_temporal_smoothing(self, current_prediction):
        """
        Apply temporal smoothing to recognition results
        
        Args:
            current_prediction: Current frame prediction (name, confidence)
            
        Returns:
            tuple: Smoothed prediction (name, confidence)
        """
        self.recent_predictions.append(current_prediction)
        
        # Count votes for each name
        name_votes = {}
        confidence_sum = {}
        
        for name, conf in self.recent_predictions:
            if name is not None:
                name_votes[name] = name_votes.get(name, 0) + 1
                confidence_sum[name] = confidence_sum.get(name, 0) + conf
                
        if not name_votes:
            return None, 0.0
            
        # Find most voted name
        best_name = max(name_votes.keys(), key=lambda x: name_votes[x])
        avg_confidence = confidence_sum[best_name] / name_votes[best_name]
        
        # Require majority vote
        if name_votes[best_name] >= len(self.recent_predictions) // 2:
            return best_name, avg_confidence
        else:
            return None, 0.0
            
    def update_threshold(self, delta):
        """Update recognition threshold"""
        self.threshold = max(0.0, min(1.0, self.threshold + delta))
        print(f"Threshold updated: {self.threshold:.3f}")
        
    def draw_recognition_result(self, image, face_bbox, name, confidence):
        """
        Draw recognition result on image
        
        Args:
            image: Input image
            face_bbox: Face bounding box (x, y, w, h)
            name: Recognized name or None
            confidence: Recognition confidence
            
        Returns:
            numpy.ndarray: Image with drawn results
        """
        result = image.copy()
        x, y, w, h = face_bbox
        
        # Draw bounding box
        if name is not None:
            color = (0, 255, 0)  # Green for recognized
            label = f"{name} ({confidence:.2f})"
        else:
            color = (0, 0, 255)  # Red for unknown
            label = f"Unknown ({confidence:.2f})"
            
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result, (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), color, -1)
        
        # Draw label text
        cv2.putText(result, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result
        
    def draw_confidence_bar(self, image, confidence, threshold):
        """Draw confidence bar"""
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = image.shape[0] - 40
        
        # Background
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Confidence bar
        conf_width = int(bar_width * confidence)
        color = (0, 255, 0) if confidence >= threshold else (0, 0, 255)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), 
                     color, -1)
        
        # Threshold line
        thresh_x = bar_x + int(bar_width * threshold)
        cv2.line(image, (thresh_x, bar_y), (thresh_x, bar_y + bar_height), 
                (255, 255, 255), 2)
        
        # Labels
        cv2.putText(image, f"Confidence: {confidence:.3f}", (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Threshold: {threshold:.3f}", (bar_x + 120, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def live_recognition():
    """Live face recognition with camera"""
    print("=== Live Face Recognition ===")
    
    try:
        from camera import Camera
        from detect import HaarFaceDetector
        from landmarks import FivePtLandmarkDetector
        from align import FaceAligner
        from embed import ArcFaceEmbedder
        
        # Check if model exists
        model_path = "models/embedder_arcface.onnx"
        if not os.path.exists(model_path):
            print(f"ERROR: ArcFace model not found at {model_path}")
            print("Please download w600k_r50.onnx and place it in the models/ directory")
            return
            
        # Initialize components
        detector = HaarFaceDetector()
        landmark_detector = FivePtLandmarkDetector()
        aligner = FaceAligner()
        embedder = ArcFaceEmbedder(model_path)
        recognizer = FaceRecognizer()
        
        if len(recognizer.database_embeddings) == 0:
            print("ERROR: No enrolled faces found in database")
            print("Please run enrollment first: python src/enroll.py")
            return
            
        print("Live recognition started")
        print("Controls:")
        print("  +: Increase threshold")
        print("  -: Decrease threshold")
        print("  Q: Quit")
        
        with Camera() as camera:
            while True:
                ret, frame = camera.read_frame()
                if not ret:
                    print("Failed to read frame")
                    break
                    
                recognizer.frame_count += 1
                result = frame.copy()
                
                # Process every N frames for performance
                if recognizer.frame_count % recognizer.process_every_n == 0:
                    
                    # Detect faces
                    faces = detector.detect_faces(frame)
                    
                    if faces:
                        # Process first face
                        face_bbox = faces[0]
                        
                        # ROI-based detection (crop face region)
                        x, y, w, h = face_bbox
                        face_roi = frame[y:y+h, x:x+w]
                        
                        # Detect landmarks
                        landmarks_5pt = landmark_detector.detect_landmarks(frame, face_bbox)
                        
                        if landmarks_5pt is not None:
                            # Align face
                            aligned_face = aligner.align_face(frame, landmarks_5pt)
                            
                            if aligned_face is not None:
                                # Extract embedding
                                embedding = embedder.extract_embedding(aligned_face)
                                
                                if embedding is not None:
                                    # Recognize face
                                    name, confidence = recognizer.recognize_face(embedding)
                                    
                                    # Apply temporal smoothing
                                    smoothed_name, smoothed_conf = recognizer.apply_temporal_smoothing(
                                        (name, confidence)
                                    )
                                    
                                    # Apply acceptance hold
                                    if smoothed_name is not None:
                                        recognizer.held_result = (smoothed_name, smoothed_conf)
                                        recognizer.hold_counter = recognizer.acceptance_hold
                                    elif recognizer.hold_counter > 0:
                                        recognizer.hold_counter -= 1
                                        if recognizer.hold_counter == 0:
                                            recognizer.held_result = None
                                    
                                    # Use held result if available
                                    display_name, display_conf = recognizer.held_result or (None, confidence)
                                    
                                    # Draw results
                                    result = recognizer.draw_recognition_result(
                                        result, face_bbox, display_name, display_conf
                                    )
                                    
                                    # Draw confidence bar
                                    recognizer.draw_confidence_bar(result, confidence, recognizer.threshold)
                                    
                        # Draw face detection even if no landmarks
                        if not landmarks_5pt:
                            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.putText(result, "No landmarks", (x, y - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Display info
                cv2.putText(result, f"Database: {len(recognizer.database_embeddings)} people", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result, f"Threshold: {recognizer.threshold:.3f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result, "Controls: +/- threshold, Q quit", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Live Face Recognition', result)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('+') or key == ord('='):
                    recognizer.update_threshold(0.05)
                elif key == ord('-'):
                    recognizer.update_threshold(-0.05)
                    
    except Exception as e:
        print(f"Recognition failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

def test_recognition_image(image_path):
    """Test recognition on a single image"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
        
    from detect import HaarFaceDetector
    from landmarks import FivePtLandmarkDetector
    from align import FaceAligner
    from embed import ArcFaceEmbedder
    
    model_path = "models/embedder_arcface.onnx"
    if not os.path.exists(model_path):
        print(f"ERROR: ArcFace model not found at {model_path}")
        return
        
    # Initialize components
    detector = HaarFaceDetector()
    landmark_detector = FivePtLandmarkDetector()
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder(model_path)
    recognizer = FaceRecognizer()
    
    if len(recognizer.database_embeddings) == 0:
        print("ERROR: No enrolled faces found in database")
        return
        
    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
        
    # Process
    faces = detector.detect_faces(image)
    print(f"Detected {len(faces)} faces")
    
    if faces:
        face_bbox = faces[0]
        landmarks_5pt = landmark_detector.detect_landmarks(image, face_bbox)
        
        if landmarks_5pt is not None:
            aligned_face = aligner.align_face(image, landmarks_5pt)
            
            if aligned_face is not None:
                embedding = embedder.extract_embedding(aligned_face)
                
                if embedding is not None:
                    name, confidence = recognizer.recognize_face(embedding)
                    
                    print(f"Recognition result:")
                    print(f"  Name: {name or 'Unknown'}")
                    print(f"  Confidence: {confidence:.4f}")
                    print(f"  Threshold: {recognizer.threshold:.4f}")
                    
                    # Draw and display result
                    result = recognizer.draw_recognition_result(image, face_bbox, name, confidence)
                    recognizer.draw_confidence_bar(result, confidence, recognizer.threshold)
                    
                    cv2.imshow('Face Recognition - Image', result)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("Embedding extraction failed")
            else:
                print("Face alignment failed")
        else:
            print("Landmark detection failed")
    else:
        print("No faces detected")

def main():
    """Main recognition interface"""
    print("=== Face Recognition System ===")
    print("1. Live recognition (camera)")
    print("2. Test on image file")
    print("3. Quit")
    
    choice = input("Choose option: ").strip()
    
    if choice == '1':
        live_recognition()
    elif choice == '2':
        image_path = input("Enter image path: ").strip()
        if image_path:
            test_recognition_image(image_path)
    elif choice == '3':
        pass
    else:
        print("Invalid choice")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test on image file
        test_recognition_image(sys.argv[1])
    else:
        # Interactive mode
        main()