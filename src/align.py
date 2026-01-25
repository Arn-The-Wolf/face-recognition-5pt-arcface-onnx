#!/usr/bin/env python3
"""
Face alignment module using 5-point landmarks
Implements similarity transform to align faces to 112x112 as specified in the document
"""

import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform

class FaceAligner:
    """5-point face alignment using similarity transform"""
    
    def __init__(self, output_size=(112, 112)):
        """
        Initialize face aligner
        
        Args:
            output_size: Target aligned face size (width, height)
        """
        self.output_size = output_size
        
        # Standard 5-point template for 112x112 face alignment
        # These are the canonical positions for aligned faces
        # Order: [left_eye, right_eye, nose_tip, left_mouth, right_mouth]
        self.template_5pt = np.array([
            [38.2946, 51.6963],    # Left eye
            [73.5318, 51.5014],    # Right eye  
            [56.0252, 71.7366],    # Nose tip
            [41.5493, 92.3655],    # Left mouth corner
            [70.7299, 92.2041]     # Right mouth corner
        ], dtype=np.float32)
        
        print(f"Face aligner initialized for {output_size[0]}x{output_size[1]} output")
        
    def align_face(self, image, landmarks_5pt):
        """
        Align face using 5-point landmarks with similarity transform
        
        Args:
            image: Input image (BGR)
            landmarks_5pt: 5x2 array of landmark coordinates in fixed order
            
        Returns:
            numpy.ndarray: Aligned face image (112x112) or None if alignment fails
        """
        if landmarks_5pt is None or len(landmarks_5pt) != 5:
            return None
            
        try:
            # Estimate similarity transform from detected landmarks to template
            transform_matrix = self._estimate_similarity_transform(
                landmarks_5pt, self.template_5pt
            )
            
            # Apply transformation
            aligned_face = cv2.warpAffine(
                image, 
                transform_matrix, 
                self.output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            return aligned_face
            
        except Exception as e:
            print(f"Face alignment failed: {e}")
            return None
            
    def _estimate_similarity_transform(self, src_pts, dst_pts):
        """
        Estimate similarity transform (rotation, scale, translation) between point sets
        
        Args:
            src_pts: Source points (Nx2)
            dst_pts: Destination points (Nx2)
            
        Returns:
            numpy.ndarray: 2x3 transformation matrix
        """
        # Ensure float32 type
        src_pts = src_pts.astype(np.float32)
        dst_pts = dst_pts.astype(np.float32)
        
        # Center the points
        src_mean = np.mean(src_pts, axis=0)
        dst_mean = np.mean(dst_pts, axis=0)
        
        src_centered = src_pts - src_mean
        dst_centered = dst_pts - dst_mean
        
        # Compute scale
        src_scale = np.sqrt(np.sum(src_centered ** 2))
        dst_scale = np.sqrt(np.sum(dst_centered ** 2))
        
        if src_scale < 1e-6:
            # Degenerate case - return identity-like transform
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            
        scale = dst_scale / src_scale
        
        # Normalize points
        src_norm = src_centered / src_scale
        dst_norm = dst_centered / dst_scale
        
        # Compute rotation using Procrustes analysis
        H = np.dot(src_norm.T, dst_norm)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
            
        # Compute final transformation
        transform_matrix = np.zeros((2, 3), dtype=np.float32)
        transform_matrix[:2, :2] = scale * R
        transform_matrix[:, 2] = dst_mean - scale * np.dot(R, src_mean)
        
        return transform_matrix
        
    def get_alignment_quality(self, landmarks_5pt):
        """
        Assess alignment quality based on landmark geometry
        
        Args:
            landmarks_5pt: 5x2 array of landmark coordinates
            
        Returns:
            dict: Quality metrics (eye_distance, face_angle, etc.)
        """
        if landmarks_5pt is None or len(landmarks_5pt) != 5:
            return None
            
        left_eye, right_eye, nose, left_mouth, right_mouth = landmarks_5pt
        
        # Eye distance
        eye_distance = np.linalg.norm(right_eye - left_eye)
        
        # Face angle (deviation from horizontal eye line)
        eye_vector = right_eye - left_eye
        face_angle = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
        
        # Nose-eye alignment
        eye_center = (left_eye + right_eye) / 2
        nose_eye_distance = np.linalg.norm(nose - eye_center)
        
        # Mouth width
        mouth_width = np.linalg.norm(right_mouth - left_mouth)
        
        return {
            'eye_distance': eye_distance,
            'face_angle': face_angle,
            'nose_eye_distance': nose_eye_distance,
            'mouth_width': mouth_width,
            'quality_score': min(eye_distance / 50.0, 1.0)  # Normalized quality
        }

def test_alignment():
    """Test face alignment with camera"""
    print("Testing face alignment...")
    
    try:
        from camera import Camera
        from detect import HaarFaceDetector
        from landmarks import FivePtLandmarkDetector
        
        detector = HaarFaceDetector()
        landmark_detector = FivePtLandmarkDetector()
        aligner = FaceAligner()
        
        with Camera() as camera:
            print("Face alignment test started")
            print("Press 'q' to quit, 's' to save aligned face")
            
            while True:
                ret, frame = camera.read_frame()
                if not ret:
                    print("Failed to read frame")
                    break
                    
                # Detect faces and landmarks
                faces = detector.detect_faces(frame)
                landmarks_5pt = None
                aligned_face = None
                
                if faces:
                    landmarks_5pt = landmark_detector.detect_landmarks(frame, faces[0])
                    if landmarks_5pt is not None:
                        aligned_face = aligner.align_face(frame, landmarks_5pt)
                
                # Draw results on original frame
                result = detector.draw_faces(frame, faces, color=(255, 0, 0))
                if landmarks_5pt is not None:
                    result = landmark_detector.draw_landmarks(result, landmarks_5pt)
                
                # Display info
                cv2.putText(result, f"Faces: {len(faces)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if aligned_face is not None:
                    cv2.putText(result, "Face aligned", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show alignment quality
                    quality = aligner.get_alignment_quality(landmarks_5pt)
                    if quality:
                        cv2.putText(result, f"Quality: {quality['quality_score']:.2f}", (10, 110),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(result, f"Angle: {quality['face_angle']:.1f}°", (10, 140),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(result, "No alignment", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(result, "Press 'q' to quit, 's' to save", (10, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show original frame
                cv2.imshow('Face Alignment - Original', result)
                
                # Show aligned face if available
                if aligned_face is not None:
                    # Resize for better visibility
                    aligned_display = cv2.resize(aligned_face, (224, 224), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow('Face Alignment - Aligned (112x112)', aligned_display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and aligned_face is not None:
                    # Save aligned face
                    filename = "aligned_face.jpg"
                    cv2.imwrite(filename, aligned_face)
                    print(f"Saved aligned face: {filename}")
                    
    except Exception as e:
        print(f"Alignment test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

def test_alignment_image(image_path):
    """Test alignment on a single image"""
    import os
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
        
    from detect import HaarFaceDetector
    from landmarks import FivePtLandmarkDetector
    
    detector = HaarFaceDetector()
    landmark_detector = FivePtLandmarkDetector()
    aligner = FaceAligner()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
        
    # Process
    faces = detector.detect_faces(image)
    print(f"Detected {len(faces)} faces")
    
    if faces:
        landmarks_5pt = landmark_detector.detect_landmarks(image, faces[0])
        
        if landmarks_5pt is not None:
            aligned_face = aligner.align_face(image, landmarks_5pt)
            quality = aligner.get_alignment_quality(landmarks_5pt)
            
            print("Alignment successful!")
            if quality:
                print(f"Quality score: {quality['quality_score']:.2f}")
                print(f"Face angle: {quality['face_angle']:.1f}°")
                print(f"Eye distance: {quality['eye_distance']:.1f}px")
            
            if aligned_face is not None:
                # Show results
                result = detector.draw_faces(image, faces, color=(255, 0, 0))
                result = landmark_detector.draw_landmarks(result, landmarks_5pt)
                
                cv2.imshow('Original with Landmarks', result)
                cv2.imshow('Aligned Face (112x112)', aligned_face)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # Save aligned face
                cv2.imwrite('aligned_face_test.jpg', aligned_face)
                print("Saved aligned face as 'aligned_face_test.jpg'")
        else:
            print("No landmarks detected")
    else:
        print("No faces detected")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test on image file
        test_alignment_image(sys.argv[1])
    else:
        # Test with camera
        test_alignment()