#!/usr/bin/env python3
"""
Setup script for Face Recognition System
Automates the complete setup process
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {command}")
            return True
        else:
            print(f"✗ {command}")
            print(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ {command}")
        print(f"  Exception: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def setup_virtual_environment():
    """Create and activate virtual environment"""
    print("\n=== Setting up Virtual Environment ===")
    
    if os.path.exists(".venv"):
        print("✓ Virtual environment already exists")
        return True
    
    success = run_command("python -m venv .venv")
    return success

def install_dependencies():
    """Install required packages"""
    print("\n=== Installing Dependencies ===")
    
    packages = [
        "opencv-python",
        "numpy", 
        "scipy",
        "tqdm",
        "mediapipe",
        "onnxruntime"
    ]
    
    # Determine activation command based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = ".venv\\Scripts\\activate &&"
    else:  # Linux/Mac
        activate_cmd = "source .venv/bin/activate &&"
    
    for package in packages:
        cmd = f"{activate_cmd} pip install --timeout 600 --retries 10 {package}"
        success = run_command(cmd)
        if not success:
            print(f"Failed to install {package}")
            return False
    
    return True

def check_model():
    """Check if ArcFace model exists"""
    print("\n=== Model Setup ===")
    
    model_path = Path("models/embedder_arcface.onnx")
    if model_path.exists():
        print("✓ ArcFace model found")
        return True
    
    print("⚠️  ArcFace model not found")
    print("Please download w600k_r50.onnx and place it at models/embedder_arcface.onnx")
    print("See MODEL_DOWNLOAD.md for instructions")
    return False

def run_tests():
    """Run installation tests"""
    print("\n=== Running Tests ===")
    
    if os.name == 'nt':  # Windows
        activate_cmd = ".venv\\Scripts\\activate &&"
    else:  # Linux/Mac
        activate_cmd = "source .venv/bin/activate &&"
    
    cmd = f"{activate_cmd} python test_installation.py"
    return run_command(cmd)

def main():
    """Main setup function"""
    print("Face Recognition System - Automated Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\nPlease install Python 3.8 or higher")
        return False
    
    # Setup virtual environment
    if not setup_virtual_environment():
        print("\nFailed to create virtual environment")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\nFailed to install dependencies")
        return False
    
    # Check model
    model_exists = check_model()
    
    # Run tests
    if not run_tests():
        print("\nSome tests failed, but basic setup is complete")
    
    # Final instructions
    print("\n" + "=" * 50)
    print("SETUP COMPLETE!")
    print("=" * 50)
    
    if not model_exists:
        print("\n⚠️  IMPORTANT: Download the ArcFace model to complete setup")
        print("See MODEL_DOWNLOAD.md for detailed instructions")
    
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    if os.name == 'nt':
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    
    print("2. Test components:")
    print("   python src/camera.py")
    print("3. Enroll people:")
    print("   python src/enroll.py")
    print("4. Start recognition:")
    print("   python src/recognize.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)