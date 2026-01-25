@echo off
echo ========================================
echo Face Recognition System - Quick Setup
echo ========================================

echo.
echo Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Installing dependencies...
pip install --timeout 600 --retries 10 opencv-python numpy scipy tqdm mediapipe onnxruntime

echo.
echo Testing installation...
python test_installation.py

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Download w600k_r50.onnx and place it in models/ directory
echo 2. Run: python src/enroll.py (to enroll people)
echo 3. Run: python src/recognize.py (for live recognition)
echo.
pause