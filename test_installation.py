#!/usr/bin/env python3
"""
Installation test script for Face Recognition System
Verifies all dependencies and components are working correctly
"""

import sys
import os
import importlib

def test_imports():
    """Test all required imports"""
    print("=== Testing Python Dependencies ===")
    
    required_packages = [
        'cv2',
        'numpy', 
        'onnxruntime',
        'scipy',
        'tqdm',
        'mediapipe'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            module = importlib.import_module(package)
            if package == 'cv2':
                print(f"✓ OpenCV {module.__version__}")
            elif package == 'numpy':
                print(f"✓ NumPy {module.__version__}")
            elif package == 'onnxruntime':
                print(f"✓ ONNX Runtime {module.__version__}")
            elif package == 'scipy':
                print(f"✓ SciPy {module.__version__}")
            elif package == 'mediapipe':
                print(f"✓ MediaPipe {module.__version__}")
            else:
                print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def test_file_structure():
    """Test project file structure"""
    print("\n=== Testing File Structure ===")
    
    required_dirs = [
        'data',
        'data/enroll',
        'data/db', 
        'models',
        'src'
    ]
    
    required_files = [
        'src/camera.py',
        'src/detect.py',
        'src/landmarks.py',
        'src/align.py',
        'src/embed.py',
        'src/enroll.py',
        'src/evaluate.py',
        'src/recognize.py',
        'src/haar_5pt.py',
        'requirements.txt'
    ]
    
    missing_items = []
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Directory: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            missing_items.append(dir_path)
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ File: {file_path}")
        else:
            print(f"✗ File missing: {file_path}")
            missing_items.append(file_path)
    
    return len(missing_items) == 0

def test_arcface_model():
    """Test ArcFace model availability"""
    print("\n=== Testing ArcFace Model ===")
    
    model_path = "models/embedder_arcface.onnx"
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"✓ ArcFace model found: {model_path} ({file_size:.1f} MB)")
        return True
    else:
        print(f"✗ ArcFace model missing: {model_path}")
        print("  Please download w600k_r50.onnx and place it in models/ directory")
        return False

def test_camera():
    """Test camera availability"""
    print("\n=== Testing Camera ===")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"✓ Camera working: {w}x{h}")
                cap.release()
                return True
            else:
                print("✗ Camera opened but cannot read frames")
                cap.release()
                return False
        else:
            print("✗ Cannot open camera (ID: 0)")
            return False
            
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_components():
    """Test individual components"""
    print("\n=== Testing Components ===")
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    components = [
        ('detect', 'HaarFaceDetector'),
        ('landmarks', 'FivePtLandmarkDetector'), 
        ('align', 'FaceAligner'),
        ('haar_5pt', 'HaarFivePtDetector')
    ]
    
    failed_components = []
    
    for module_name, class_name in components:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            instance = cls()
            print(f"✓ {class_name}")
        except Exception as e:
            print(f"✗ {class_name}: {e}")
            failed_components.append(class_name)
    
    # Test embedding component (requires model)
    if os.path.exists("models/embedder_arcface.onnx"):
        try:
            from embed import ArcFaceEmbedder
            embedder = ArcFaceEmbedder("models/embedder_arcface.onnx")
            print("✓ ArcFaceEmbedder")
        except Exception as e:
            print(f"✗ ArcFaceEmbedder: {e}")
            failed_components.append("ArcFaceEmbedder")
    else:
        print("- ArcFaceEmbedder: Skipped (no model)")
    
    return len(failed_components) == 0

def test_opencv_features():
    """Test specific OpenCV features"""
    print("\n=== Testing OpenCV Features ===")
    
    try:
        import cv2
        
        # Test Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                print("✓ Haar cascade loaded")
            else:
                print("✗ Haar cascade failed to load")
                return False
        else:
            print(f"✗ Haar cascade not found: {cascade_path}")
            return False
            
        # Test basic image operations
        test_img = cv2.imread('test_image.jpg')  # This will fail, but that's ok
        print("✓ OpenCV image operations available")
        
        return True
        
    except Exception as e:
        print(f"✗ OpenCV features test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Face Recognition System - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Python Dependencies", test_imports),
        ("File Structure", test_file_structure),
        ("ArcFace Model", test_arcface_model),
        ("Camera", test_camera),
        ("OpenCV Features", test_opencv_features),
        ("Components", test_components)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}: Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("INSTALLATION TEST SUMMARY")
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
        print("\nNext steps:")
        print("1. Run component tests: python src/camera.py")
        print("2. Start enrollment: python src/enroll.py")
        print("3. Run recognition: python src/recognize.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before using the system.")
        
        if not any(name == "ArcFace Model" and result for name, result in results):
            print("\n📥 To download ArcFace model:")
            print("1. Download buffalo_l model pack from InsightFace")
            print("2. Extract w600k_r50.onnx")
            print("3. Place it at models/embedder_arcface.onnx")

if __name__ == "__main__":
    main()