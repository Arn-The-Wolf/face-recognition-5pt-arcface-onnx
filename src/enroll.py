#!/usr/bin/env python3
"""
Face enrollment module for creating reference database
Implements enrollment pipeline as specified in the document
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime

class FaceEnroller:
    """Face enrollment system for building reference database"""
    
    def __init__(self, data_dir="data"):
        """
        Initialize face enroller
        
        Args:
            data_dir: Base data directory
        """
        self.data_dir = data_dir
        self.enroll_dir = os.path.join(data_dir, "enroll")
        self.db_dir = os.path.join(data_dir, "db")
        
        # Ensure directories exist
        os.makedirs(self.enroll_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Database files
        self.db_embeddings_path = os.path.join(self.db_dir, "face_db.npz")
        self.db_metadata_path = os.path.join(self.db_dir, "face_db.json")
        
        # Enrollment parameters
        self.min_samples = 3
        self.max_samples = 10
        self.quality_threshold = 0.5
        
        print(f"Face enroller initialized")
        print(f"Enroll directory: {self.enroll_dir}")
        print(f"Database directory: {self.db_dir}")
        
    def enroll_person(self, name, embeddings, aligned_faces=None, save_crops=True):
        """
        Enroll a person with multiple face samples
        
        Args:
            name: Person's name/identifier
            embeddings: List of embedding vectors
            aligned_faces: List of aligned face images (optional)
            save_crops: Whether to save aligned face crops
            
        Returns:
            bool: Success status
        """
        if len(embeddings) < self.min_samples:
            print(f"ERROR: Need at least {self.min_samples} samples, got {len(embeddings)}")
            return False
            
        try:
            # Create person directory
            person_dir = os.path.join(self.enroll_dir, name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Validate embeddings
            valid_embeddings = []
            valid_faces = []
            
            for i, embedding in enumerate(embeddings):
                if embedding is not None and embedding.shape == (512,):
                    valid_embeddings.append(embedding)
                    if aligned_faces and i < len(aligned_faces):
                        valid_faces.append(aligned_faces[i])
                        
            if len(valid_embeddings) < self.min_samples:
                print(f"ERROR: Only {len(valid_embeddings)} valid embeddings")
                return False
                
            # Compute mean embedding
            embeddings_array = np.array(valid_embeddings)
            mean_embedding = np.mean(embeddings_array, axis=0)
            
            # L2 normalize mean embedding
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
            
            # Save aligned face crops if requested
            if save_crops and valid_faces:
                for i, face in enumerate(valid_faces):
                    if face is not None:
                        crop_path = os.path.join(person_dir, f"{name}_{i:03d}.jpg")
                        cv2.imwrite(crop_path, face)
                        
            # Update database
            self._update_database(name, mean_embedding, valid_embeddings)
            
            print(f"Successfully enrolled {name} with {len(valid_embeddings)} samples")
            return True
            
        except Exception as e:
            print(f"Enrollment failed for {name}: {e}")
            return False
            
    def _update_database(self, name, mean_embedding, all_embeddings):
        """Update face database with new person"""
        
        # Load existing database
        db_embeddings = {}
        db_metadata = {}
        
        if os.path.exists(self.db_embeddings_path):
            db_data = np.load(self.db_embeddings_path)
            db_embeddings = {key: db_data[key] for key in db_data.files}
            
        if os.path.exists(self.db_metadata_path):
            with open(self.db_metadata_path, 'r') as f:
                db_metadata = json.load(f)
                
        # Add/update person
        db_embeddings[name] = mean_embedding
        db_metadata[name] = {
            'num_samples': len(all_embeddings),
            'enrollment_date': datetime.now().isoformat(),
            'embedding_dim': mean_embedding.shape[0],
            'embedding_norm': float(np.linalg.norm(mean_embedding))
        }
        
        # Save updated database
        np.savez(self.db_embeddings_path, **db_embeddings)
        
        with open(self.db_metadata_path, 'w') as f:
            json.dump(db_metadata, f, indent=2)
            
        print(f"Database updated: {len(db_embeddings)} identities")
        
    def load_database(self):
        """
        Load face database
        
        Returns:
            tuple: (embeddings_dict, metadata_dict)
        """
        embeddings = {}
        metadata = {}
        
        if os.path.exists(self.db_embeddings_path):
            db_data = np.load(self.db_embeddings_path)
            embeddings = {key: db_data[key] for key in db_data.files}
            
        if os.path.exists(self.db_metadata_path):
            with open(self.db_metadata_path, 'r') as f:
                metadata = json.load(f)
                
        return embeddings, metadata
        
    def list_enrolled_people(self):
        """List all enrolled people"""
        embeddings, metadata = self.load_database()
        
        print(f"\nEnrolled people ({len(embeddings)}):")
        for name in embeddings.keys():
            info = metadata.get(name, {})
            print(f"  {name}: {info.get('num_samples', 'N/A')} samples, "
                  f"enrolled {info.get('enrollment_date', 'unknown')}")
                  
        return list(embeddings.keys())
        
    def delete_person(self, name):
        """Delete a person from database"""
        embeddings, metadata = self.load_database()
        
        if name not in embeddings:
            print(f"Person {name} not found in database")
            return False
            
        # Remove from database
        del embeddings[name]
        if name in metadata:
            del metadata[name]
            
        # Save updated database
        np.savez(self.db_embeddings_path, **embeddings)
        with open(self.db_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Remove enrollment directory
        person_dir = os.path.join(self.enroll_dir, name)
        if os.path.exists(person_dir):
            import shutil
            shutil.rmtree(person_dir)
            
        print(f"Deleted {name} from database")
        return True

def interactive_enrollment():
    """Interactive enrollment using camera"""
    print("=== Face Enrollment System ===")
    
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
        enroller = FaceEnroller()
        
        # Get person name
        name = input("Enter person's name: ").strip()
        if not name:
            print("Invalid name")
            return
            
        print(f"\nEnrolling: {name}")
        print("Controls:")
        print("  SPACE: Capture sample")
        print("  A: Auto-capture mode")
        print("  S: Save enrollment")
        print("  R: Reset samples")
        print("  Q: Quit")
        
        # Enrollment state
        samples = []
        aligned_faces = []
        auto_capture = False
        auto_counter = 0
        
        with Camera() as camera:
            while True:
                ret, frame = camera.read_frame()
                if not ret:
                    print("Failed to read frame")
                    break
                    
                # Process frame
                faces = detector.detect_faces(frame)
                landmarks_5pt = None
                aligned_face = None
                embedding = None
                
                if faces:
                    landmarks_5pt = landmark_detector.detect_landmarks(frame, faces[0])
                    if landmarks_5pt is not None:
                        aligned_face = aligner.align_face(frame, landmarks_5pt)
                        if aligned_face is not None:
                            embedding = embedder.extract_embedding(aligned_face)
                
                # Auto-capture logic
                if auto_capture and embedding is not None and len(samples) < enroller.max_samples:
                    auto_counter += 1
                    if auto_counter >= 30:  # Capture every 30 frames (~1 second)
                        samples.append(embedding)
                        aligned_faces.append(aligned_face.copy())
                        auto_counter = 0
                        print(f"Auto-captured sample {len(samples)}")
                
                # Draw results
                result = detector.draw_faces(frame, faces, color=(255, 0, 0))
                if landmarks_5pt is not None:
                    result = landmark_detector.draw_landmarks(result, landmarks_5pt)
                
                # Display info
                cv2.putText(result, f"Enrolling: {name}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result, f"Samples: {len(samples)}/{enroller.max_samples}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if auto_capture:
                    cv2.putText(result, "AUTO-CAPTURE ON", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if embedding is not None:
                    cv2.putText(result, "Ready to capture", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(result, "No face detected", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show controls
                cv2.putText(result, "SPACE:Capture A:Auto S:Save R:Reset Q:Quit", (10, 400),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Face Enrollment', result)
                
                # Show aligned face if available
                if aligned_face is not None:
                    aligned_display = cv2.resize(aligned_face, (224, 224), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow('Aligned Face', aligned_display)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' ') and embedding is not None:
                    # Manual capture
                    if len(samples) < enroller.max_samples:
                        samples.append(embedding)
                        aligned_faces.append(aligned_face.copy())
                        print(f"Captured sample {len(samples)}")
                    else:
                        print("Maximum samples reached")
                elif key == ord('a'):
                    # Toggle auto-capture
                    auto_capture = not auto_capture
                    auto_counter = 0
                    print(f"Auto-capture: {'ON' if auto_capture else 'OFF'}")
                elif key == ord('s'):
                    # Save enrollment
                    if len(samples) >= enroller.min_samples:
                        success = enroller.enroll_person(name, samples, aligned_faces)
                        if success:
                            print(f"Successfully enrolled {name}!")
                            break
                        else:
                            print("Enrollment failed")
                    else:
                        print(f"Need at least {enroller.min_samples} samples")
                elif key == ord('r'):
                    # Reset samples
                    samples = []
                    aligned_faces = []
                    auto_capture = False
                    print("Samples reset")
                    
    except Exception as e:
        print(f"Enrollment failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

def main():
    """Main enrollment interface"""
    enroller = FaceEnroller()
    
    while True:
        print("\n=== Face Enrollment System ===")
        print("1. Interactive enrollment (camera)")
        print("2. List enrolled people")
        print("3. Delete person")
        print("4. Database info")
        print("5. Quit")
        
        choice = input("Choose option: ").strip()
        
        if choice == '1':
            interactive_enrollment()
        elif choice == '2':
            enroller.list_enrolled_people()
        elif choice == '3':
            name = input("Enter name to delete: ").strip()
            if name:
                enroller.delete_person(name)
        elif choice == '4':
            embeddings, metadata = enroller.load_database()
            print(f"\nDatabase info:")
            print(f"  Total identities: {len(embeddings)}")
            print(f"  Database files:")
            print(f"    {enroller.db_embeddings_path}")
            print(f"    {enroller.db_metadata_path}")
        elif choice == '5':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()