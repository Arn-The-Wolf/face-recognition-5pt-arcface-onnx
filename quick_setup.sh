#!/bin/bash

echo "========================================"
echo "Face Recognition System - Quick Setup"
echo "========================================"

echo
echo "Creating virtual environment..."
python3 -m venv .venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment"
    exit 1
fi

echo
echo "Activating virtual environment..."
source .venv/bin/activate

echo
echo "Installing dependencies..."
pip install --timeout 600 --retries 10 opencv-python numpy scipy tqdm mediapipe onnxruntime

echo
echo "Testing installation..."
python test_installation.py

echo
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "Next steps:"
echo "1. Download w600k_r50.onnx and place it in models/ directory"
echo "2. Run: python src/enroll.py (to enroll people)"
echo "3. Run: python src/recognize.py (for live recognition)"
echo