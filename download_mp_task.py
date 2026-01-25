import urllib.request
import os

url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
target = "models/face_landmarker.task"

print(f"Downloading {url} to {target}...")
try:
    urllib.request.urlretrieve(url, target)
    print("Download complete.")
except Exception as e:
    print(f"Download failed: {e}")
    exit(1)
