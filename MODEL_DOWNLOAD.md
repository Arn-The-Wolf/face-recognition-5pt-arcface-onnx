# ArcFace Model Download Guide

The face recognition system requires the ArcFace ONNX model to function. Here's how to get it:

## Method 1: Direct Download (Recommended)

1. **Download the model file directly:**
   - Go to: https://github.com/deepinsight/insightface/releases/tag/v0.7
   - Download `buffalo_l.zip`
   - Extract the zip file
   - Find `w600k_r50.onnx` in the extracted folder
   - Copy it to `models/embedder_arcface.onnx` in this project

## Method 2: Using InsightFace Python Package

```bash
# Activate your virtual environment first
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install insightface
pip install insightface

# Download model using Python
python -c "
import insightface
app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print('Model downloaded successfully')
"
```

Then find the downloaded model in your system's cache and copy `w600k_r50.onnx` to `models/embedder_arcface.onnx`.

## Method 3: Manual Download from Hugging Face

1. Go to: https://huggingface.co/public-data/insightface
2. Navigate to the buffalo_l folder
3. Download `w600k_r50.onnx`
4. Place it at `models/embedder_arcface.onnx`

## Verification

After placing the model file, run:

```bash
python test_installation.py
```

You should see:
```
✓ ArcFace model found: models/embedder_arcface.onnx (XXX.X MB)
```

## Model Information

- **File name**: `w600k_r50.onnx`
- **Size**: ~250 MB
- **Architecture**: ResNet-50 backbone
- **Training data**: WebFace600K dataset
- **Output**: 512-dimensional embeddings
- **License**: Check InsightFace license terms

## Troubleshooting

**File not found error:**
- Ensure the file is named exactly `embedder_arcface.onnx`
- Check the file is in the `models/` directory
- Verify the file size is around 250 MB

**Permission errors:**
- Make sure you have write permissions to the models/ directory
- Try running as administrator if needed

**Download fails:**
- Try using a VPN if the download is blocked
- Use a download manager for large files
- Check your internet connection

## Important Notes

- The model file is required for the system to work
- Do not rename or modify the model file
- Keep a backup copy of the model file
- Respect the license terms of the InsightFace project