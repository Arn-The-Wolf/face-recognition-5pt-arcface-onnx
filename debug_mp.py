import mediapipe as mp
import sys

with open("debug_mp.txt", "w") as f:
    f.write(f"Python version: {sys.version}\n")
    f.write(f"MediaPipe file: {mp.__file__}\n")
    f.write(f"MediaPipe dir: {dir(mp)}\n")
    try:
        import mediapipe.python.solutions
        f.write("import mediapipe.python.solutions SUCCESS\n")
    except Exception as e:
        f.write(f"import mediapipe.python.solutions FAILED: {e}\n")
    
    try:
        from mediapipe import solutions
        f.write("from mediapipe import solutions SUCCESS\n")
    except Exception as e:
        f.write(f"from mediapipe import solutions FAILED: {e}\n")

print("Debug script finished.")
