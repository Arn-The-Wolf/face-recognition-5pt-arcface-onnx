import onnxruntime as ort
import os

model_path = "models/embedder_arcface.onnx"
print(f"Testing model: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")
if os.path.exists(model_path):
    print(f"File size: {os.path.getsize(model_path)}")

try:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print("Model loaded successfully!")
    print(f"Inputs: {[x.name for x in sess.get_inputs()]}")
except Exception as e:
    print(f"Failed to load model: {e}")
