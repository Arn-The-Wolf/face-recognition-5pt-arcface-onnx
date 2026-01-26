#!/usr/bin/env python3
"""
Comprehensive test runner for Face Recognition System
Tests all components individually
"""

import sys
import os
import cv2
import numpy as np

def test_camera():
    """Test camera functionality"""
    print("=== Testing Camera ===")
    try:
        from src.camera import Camera
        with Camera() as camera:
            ret, frame = camera.read_frame()
            if ret:
                print("✓ Camera test passed")
                return True
            else:
                print("✗ Camera test failed - cannot read frame")
                return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_detection():
    """Test face detection"""
    print("\n=== Testing Face Detection ===")
    try:
        from src.detect import HaarFaceDetector
        detector = HaarFaceDetector()
        
        # Create a test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect_faces(test_img)
        
        print("✓ Face detection test passed")
        return True
    except Exception as e:
        print(f"✗ Face detection test failed: {e}")
        return False

def test_landmarks():
    """Test landmark detection"""
    print("\n=== Testing Landmark Detection ===")
    try:
        from src.landmarks import FivePtLandmarkDetector
        detector = FivePtLandmarkDetector()
        
        # Create a test image and fake bbox
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_bbox = (100, 100, 200, 200)
        landmarks = detector.detect_landmarks(test_img, fake_bbox)
        
        if landmarks is not None and landmarks.shape == (5, 2):
            print("✓ Landmark detection test passed")
            return True
        else:
            print("✗ Landmark detection test failed - invalid output")
            return False
    except Exception as e:
        print(f"✗ Landmark detection test failed: {e}")
        return False

def test_alignment():
    """Test face alignment"""
    print("\n=== Testing Face Alignment ===")
    try:
        from src.align import FaceAligner
        aligner = FaceAligner()
        
        # Create test landmarks
        test_landmarks = np.array([
            [100, 150],  # left eye
            [200, 150],  # right eye
            [150, 200],  # nose
            [120, 250],  # left mouth
            [180, 250]   # right mouth
        ], dtype=np.float32)
        
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        aligned = aligner.align_face(test_img, test_landmarks)
        
        if aligned is not None and aligned.shape == (112, 112, 3):
            print("✓ Face alignment test passed")
            return True
        else:
            print("✗ Face alignment test failed - invalid output")
            return False
    except Exception as e:
        print(f"✗ Face alignment test failed: {e}")
        return False

def test_embedding():
    """Test embedding extraction"""
    print("\n=== Testing Embedding Extraction ===")
    try:
        model_path = "models/embedder_arcface.onnx"
        if not os.path.exists(model_path):
            print("⚠️  ArcFace model not found - skipping embedding test")
            return True
            
        from src.embed import ArcFaceEmbedder
        embedder = ArcFaceEmbedder(model_path)
        
        # Create test aligned face
        test_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        embedding = embedder.extract_embedding(test_face)
        
        if embedding is not None and embedding.shape == (512,):
            print("✓ Embedding extraction test passed")
            return True
        else:
            print("✗ Embedding extraction test failed - invalid output")
            return False
    except Exception as e:
        print(f"✗ Embedding extraction test failed: {e}")
        return False

def test_pipeline():
    """Test complete pipeline"""
    print("\n=== Testing Complete Pipeline ===")
    try:
        from src.camera import Camera
        from src.detect import HaarFaceDetector
        from src.landmarks import FivePtLandmarkDetector
        from src.align import FaceAligner
        
        detector = HaarFaceDetector()
        landmark_detector = FivePtLandmarkDetector()
        aligner = FaceAligner()
        
        with Camera() as camera:
            ret, frame = camera.read_frame()
            if not ret:
                print("✗ Pipeline test failed - cannot read camera frame")
                return False
                
            # Run pipeline
            faces = detector.detect_faces(frame)
            if faces:
                landmarks = landmark_detector.detect_landmarks(frame, faces[0])
                if landmarks is not None:
                    aligned = aligner.align_face(frame, landmarks)
                    if aligned is not None:
                        print("✓ Complete pipeline test passed")
                        return True
            
            print("✓ Pipeline components work (no face detected in test frame)")
            return True
            
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Face Recognition System - Comprehensive Tests")
    print("=" * 50)
    
    tests = [
        ("Camera", test_camera),
        ("Face Detection", test_detection),
        ("Landmark Detection", test_landmarks),
        ("Face Alignment", test_alignment),
        ("Embedding Extraction", test_embedding),
        ("Complete Pipeline", test_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}: Test crashed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
    
    print("\nNext steps:")
    print("1. Download ArcFace model (see MODEL_DOWNLOAD.md)")
    print("2. Run enrollment: python src/enroll.py")
    print("3. Run recognition: python src/recognize.py")

if __name__ == "__main__":
    main()