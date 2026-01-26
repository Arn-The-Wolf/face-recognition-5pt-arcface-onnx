#!/usr/bin/env python3
"""
Test face detection on saved image files
"""

import cv2
import os
import glob

def test_face_detection_on_image(image_path):
    """Test face detection on a single image"""
    print(f"Testing face detection on: {image_path}")
    
    from src.detect import HaarFaceDetector
    from src.landmarks import FivePtLandmarkDetector
    
    detector = HaarFaceDetector()
    landmark_detector = FivePtLandmarkDetector()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return False
        
    print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    # Detect faces
    faces = detector.detect_faces(image)
    print(f"🔍 Faces detected: {len(faces)}")
    
    if len(faces) == 0:
        print("😞 No faces detected in the image")
        # Still show the image
        cv2.imshow('No Faces Detected', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return False
    
    # Process each face
    result = image.copy()
    
    for i, face_bbox in enumerate(faces):
        x, y, w, h = face_bbox
        print(f"  Face {i+1}: ({x}, {y}) size {w}x{h}")
        
        # Draw face bounding box
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(result, f"Face {i+1}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Try to detect landmarks
        landmarks = landmark_detector.detect_landmarks(image, face_bbox)
        if landmarks is not None:
            print(f"  ✓ 5-point landmarks detected for face {i+1}")
            result = landmark_detector.draw_landmarks(result, landmarks)
        else:
            print(f"  ⚠️ No landmarks detected for face {i+1}")
    
    # Add summary text
    cv2.putText(result, f"Faces Found: {len(faces)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(result, "Press any key to close", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show result
    window_name = f'Face Detection Results - {os.path.basename(image_path)}'
    cv2.imshow(window_name, result)
    
    print(f"🎉 Showing results! Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    result_filename = f"detection_result_{os.path.basename(image_path)}"
    cv2.imwrite(result_filename, result)
    print(f"💾 Result saved as: {result_filename}")
    
    return True

def find_image_files():
    """Find image files in current directory"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob.glob(ext))
        image_files.extend(glob.glob(ext.upper()))
    
    return sorted(image_files)

def main():
    """Main function"""
    print("🖼️  Face Detection on Images")
    print("=" * 40)
    
    # Look for image files
    image_files = find_image_files()
    
    if not image_files:
        print("❌ No image files found in current directory")
        print("Please make sure you have image files (.jpg, .png, etc.) in this folder")
        return
    
    print(f"📁 Found {len(image_files)} image file(s):")
    for i, img_file in enumerate(image_files):
        print(f"  {i+1}. {img_file}")
    
    if len(image_files) == 1:
        # Test the single image
        print(f"\n🔍 Testing the image: {image_files[0]}")
        test_face_detection_on_image(image_files[0])
    else:
        # Let user choose
        print(f"\nWhich image would you like to test?")
        print("Enter the number (1-{}) or 'all' to test all images:".format(len(image_files)))
        
        choice = input("Your choice: ").strip().lower()
        
        if choice == 'all':
            for img_file in image_files:
                print(f"\n{'='*50}")
                test_face_detection_on_image(img_file)
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(image_files):
                    test_face_detection_on_image(image_files[idx])
                else:
                    print("❌ Invalid choice")
            except ValueError:
                print("❌ Invalid input")

if __name__ == "__main__":
    main()