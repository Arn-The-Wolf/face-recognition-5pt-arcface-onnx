#!/usr/bin/env python3
"""
Face embedding module using ArcFace ONNX model
Extracts L2-normalized 512-dimensional embeddings as specified in the document
"""

import cv2
import numpy as np
import onnxruntime as ort
import os

class ArcFaceEmbedder:
    """ArcFace ONNX embedder for face recognition"""
    
    def __init__(self, model_path="models/embedder_arcface.onnx"):
        """
        Initialize ArcFace embedder
        
        Args:
            model_path: Path to ArcFace ONNX model (w600k_r50.onnx)
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        
        self._load_model()
        
    def _load_model(self):
        """Load ONNX model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ArcFace model not found: {self.model_path}")
            
        try:
            # Create ONNX Runtime session (CPU only)
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Get input shape
            input_shape = self.session.get_inputs()[0].shape
            output_shape = self.session.get_outputs()[0].shape
            
            print(f"ArcFace model loaded successfully")
            print(f"Input shape: {input_shape}")
            print(f"Output shape: {output_shape}")
            print(f"Expected embedding dimension: 512")
            
            # Validate expected dimensions
            if output_shape[-1] != 512:
                print(f"WARNING: Expected 512-D embeddings, got {output_shape[-1]}-D")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load ArcFace model: {e}")
            
    def extract_embedding(self, aligned_face):
        """
        Extract face embedding from aligned face image
        
        Args:
            aligned_face: Aligned face image (112x112, BGR)
            
        Returns:
            numpy.ndarray: L2-normalized 512-dimensional embedding vector
                          Returns None if extraction fails
        """
        if aligned_face is None:
            return None
            
        try:
            # Preprocess image for ArcFace
            input_blob = self._preprocess_face(aligned_face)
            
            # Run inference
            embedding = self.session.run([self.output_name], {self.input_name: input_blob})[0]
            
            # Extract embedding vector (remove batch dimension)
            embedding = embedding[0]  # Shape: (512,)
            
            # L2 normalize embedding
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm > 0:
                embedding = embedding / embedding_norm
            else:
                print("WARNING: Zero embedding norm detected")
                return None
                
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"Embedding extraction failed: {e}")
            return None
            
    def _preprocess_face(self, aligned_face):
        """
        Preprocess aligned face for ArcFace model
        
        Args:
            aligned_face: Aligned face image (112x112, BGR)
            
        Returns:
            numpy.ndarray: Preprocessed input blob (1, 3, 112, 112)
        """
        # Ensure correct size
        if aligned_face.shape[:2] != (112, 112):
            aligned_face = cv2.resize(aligned_face, (112, 112))
            
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format (channels first)
        face_chw = np.transpose(face_normalized, (2, 0, 1))
        
        # Add batch dimension
        input_blob = np.expand_dims(face_chw, axis=0)  # Shape: (1, 3, 112, 112)
        
        return input_blob
        
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector (512,)
            embedding2: Second embedding vector (512,)
            
        Returns:
            float: Cosine similarity [-1, 1]
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        # Cosine similarity (embeddings are already L2-normalized)
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
        
    def validate_embedding(self, embedding):
        """
        Validate embedding properties
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            dict: Validation results
        """
        if embedding is None:
            return {'valid': False, 'reason': 'None embedding'}
            
        # Check dimension
        if embedding.shape != (512,):
            return {'valid': False, 'reason': f'Wrong dimension: {embedding.shape}'}
            
        # Check L2 norm (should be ~1.0 for normalized embeddings)
        norm = np.linalg.norm(embedding)
        if abs(norm - 1.0) > 0.01:
            return {'valid': False, 'reason': f'Not L2-normalized: norm={norm:.4f}'}
            
        # Check for NaN or inf values
        if not np.isfinite(embedding).all():
            return {'valid': False, 'reason': 'Contains NaN or inf values'}
            
        # Check embedding magnitude (before normalization should be ~15-30)
        # This is an approximation since we only have normalized embedding
        return {
            'valid': True,
            'dimension': embedding.shape[0],
            'norm': norm,
            'mean': np.mean(embedding),
            'std': np.std(embedding)
        }

def test_embedding():
    """Test embedding extraction with camera"""
    print("Testing ArcFace embedding extraction...")
    
    try:
        from camera import Camera
        from detect import HaarFaceDetector
        from landmarks import FivePtLandmarkDetector
        from align import FaceAligner
        
        # Check if model exists
        model_path = "models/embedder_arcface.onnx"
        if not os.path.exists(model_path):
            print(f"ERROR: ArcFace model not found at {model_path}")
            print("Please download w600k_r50.onnx and place it in the models/ directory")
            return
            
        detector = HaarFaceDetector()
        landmark_detector = FivePtLandmarkDetector()
        aligner = FaceAligner()
        embedder = ArcFaceEmbedder(model_path)
        
        with Camera() as camera:
            print("Embedding extraction test started")
            print("Press 'q' to quit, 's' to save embedding")
            
            while True:
                ret, frame = camera.read_frame()
                if not ret:
                    print("Failed to read frame")
                    break
                    
                # Full pipeline
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
                
                # Draw results
                result = detector.draw_faces(frame, faces, color=(255, 0, 0))
                if landmarks_5pt is not None:
                    result = landmark_detector.draw_landmarks(result, landmarks_5pt)
                
                # Display info
                cv2.putText(result, f"Faces: {len(faces)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if embedding is not None:
                    validation = embedder.validate_embedding(embedding)
                    if validation['valid']:
                        cv2.putText(result, "Embedding: OK", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(result, f"Dim: {validation['dimension']}", (10, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(result, f"Norm: {validation['norm']:.3f}", (10, 130),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(result, f"Embedding: {validation['reason']}", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(result, "No embedding", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(result, "Press 'q' to quit, 's' to save", (10, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show frames
                cv2.imshow('Embedding Extraction - Original', result)
                
                if aligned_face is not None:
                    aligned_display = cv2.resize(aligned_face, (224, 224), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow('Embedding Extraction - Aligned', aligned_display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and embedding is not None:
                    # Save embedding
                    np.save("test_embedding.npy", embedding)
                    print(f"Saved embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")
                    
    except Exception as e:
        print(f"Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

def test_embedding_image(image_path):
    """Test embedding extraction on a single image"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
        
    model_path = "models/embedder_arcface.onnx"
    if not os.path.exists(model_path):
        print(f"ERROR: ArcFace model not found at {model_path}")
        return
        
    from detect import HaarFaceDetector
    from landmarks import FivePtLandmarkDetector
    from align import FaceAligner
    
    detector = HaarFaceDetector()
    landmark_detector = FivePtLandmarkDetector()
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder(model_path)
    
    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
        
    faces = detector.detect_faces(image)
    print(f"Detected {len(faces)} faces")
    
    if faces:
        landmarks_5pt = landmark_detector.detect_landmarks(image, faces[0])
        if landmarks_5pt is not None:
            aligned_face = aligner.align_face(image, landmarks_5pt)
            if aligned_face is not None:
                embedding = embedder.extract_embedding(aligned_face)
                if embedding is not None:
                    validation = embedder.validate_embedding(embedding)
                    print(f"Embedding extracted successfully!")
                    print(f"Shape: {embedding.shape}")
                    print(f"Validation: {validation}")
                    
                    # Save results
                    cv2.imwrite('test_aligned.jpg', aligned_face)
                    np.save('test_embedding.npy', embedding)
                    print("Saved aligned face and embedding")
                else:
                    print("Embedding extraction failed")
            else:
                print("Face alignment failed")
        else:
            print("Landmark detection failed")
    else:
        print("No faces detected")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test on image file
        test_embedding_image(sys.argv[1])
    else:
        # Test with camera
        test_embedding()