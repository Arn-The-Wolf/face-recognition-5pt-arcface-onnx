#!/usr/bin/env python3
"""
Simple camera test to find working camera
"""

import cv2

def test_cameras():
    """Test different camera indices"""
    print("Testing available cameras...")
    
    for i in range(5):  # Test camera indices 0-4
        print(f"\nTrying camera {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"✓ Camera {i} works: {w}x{h}")
                
                # Show a quick preview
                cv2.imshow(f'Camera {i} Test', frame)
                print(f"Showing preview for camera {i} - press any key to continue")
                cv2.waitKey(2000)  # Show for 2 seconds
                cv2.destroyAllWindows()
                
                cap.release()
                return i  # Return first working camera
            else:
                print(f"✗ Camera {i} opened but can't read frames")
        else:
            print(f"✗ Camera {i} cannot be opened")
            
        cap.release()
    
    print("\nNo working cameras found!")
    return None

def test_face_detection_on_camera(camera_id):
    """Test face detection on working camera"""
    print(f"\nTesting face detection on camera {camera_id}...")
    
    from src.detect import HaarFaceDetector
    detector = HaarFaceDetector()
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    print("Face detection test - press 'q' to quit")
    print("Put your face in front of the camera! 😄")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
            
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Draw faces
        result = detector.draw_faces(frame, faces)
        
        # Add info
        cv2.putText(result, f"Faces detected: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result, "Press 'q' to quit", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if len(faces) > 0:
            cv2.putText(result, "FACE DETECTED! 😄", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Detection Test', result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🎥 Camera and Face Detection Test")
    print("=" * 40)
    
    # Find working camera
    camera_id = test_cameras()
    
    if camera_id is not None:
        print(f"\n✓ Using camera {camera_id} for face detection test")
        test_face_detection_on_camera(camera_id)
    else:
        print("\n❌ No working cameras found. Please check:")
        print("1. Camera is connected")
        print("2. Camera permissions are granted")
        print("3. No other applications are using the camera")