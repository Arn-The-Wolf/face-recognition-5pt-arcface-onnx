# Face Recognition with ArcFace ONNX and 5-Point Alignment

A complete CPU-only face recognition system implementing the exact architecture described in "Face Recognition with ArcFace ONNX and 5-Point Alignment" by Gabriel Baziramwabo.

## 🎯 System Overview

This system implements a modular face recognition pipeline with the following components:

1. **Detection**: Haar cascades for face detection
2. **Landmarks**: MediaPipe FaceMesh for 5-point landmarks  
3. **Alignment**: Similarity transform to 112×112
4. **Embedding**: ArcFace ONNX (w600k_r50.onnx) for 512-D embeddings
5. **Enrollment**: Reference database creation
6. **Evaluation**: Threshold optimization
7. **Recognition**: Live identification

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Arn-The-Wolf/face-recognition-5pt-arcface-onnx.git
cd face-recognition-5pt-arcface-onnx
```

### 2. Setup Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac  
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Download ArcFace Model
- Download the buffalo_l model pack from [InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package)
- Extract `w600k_r50.onnx` 
- Place it at `models/embedder_arcface.onnx`

### 4. Test Installation
```bash
python test_installation.py
```

### 5. Start Using
```bash
# Test components
python src/camera.py          # Test camera
python src/detect.py          # Test detection
python src/landmarks.py       # Test landmarks

# Enroll people
python src/enroll.py

# Live recognition
python src/recognize.py
```

## 📁 Project Structure

```
face-recognition-5pt-arcface-onnx/
├── data/
│   ├── enroll/           # Face crops storage
│   └── db/               # Database files
├── models/               # ArcFace ONNX model location
├── src/                  # Source modules
│   ├── camera.py         # Camera handling
│   ├── detect.py         # Haar face detection
│   ├── landmarks.py      # MediaPipe 5-point landmarks
│   ├── align.py          # Similarity transform alignment
│   ├── embed.py          # ArcFace ONNX embedding
│   ├── enroll.py         # Enrollment pipeline
│   ├── evaluate.py       # Threshold evaluation
│   ├── recognize.py      # Live recognition
│   └── haar_5pt.py       # Combined detection
├── test_installation.py  # Installation verification
├── USAGE_GUIDE.md        # Detailed usage guide
└── requirements.txt      # Dependencies
```

## 🎮 Usage

### Enrollment
```bash
python src/enroll.py
```
**Controls:**
- `SPACE`: Capture sample
- `A`: Auto-capture mode
- `S`: Save enrollment
- `R`: Reset samples
- `Q`: Quit

### Live Recognition
```bash
python src/recognize.py
```
**Controls:**
- `+`: Increase threshold (more restrictive)
- `-`: Decrease threshold (less restrictive)
- `Q`: Quit

### Evaluation
```bash
python src/evaluate.py
```
Analyzes recognition performance and suggests optimal threshold.

## 🔧 Technical Specifications

- **CPU-only execution** with ONNX Runtime
- **ArcFace w600k_r50.onnx** embedder (512-D vectors)
- **Haar cascades** for face detection
- **MediaPipe FaceMesh** for 5-point landmarks
- **Similarity transform** alignment to 112×112
- **L2-normalized embeddings**
- **Cosine similarity** matching with configurable threshold

## 📊 Performance Features

- **Temporal smoothing** over 10 frames
- **Acceptance hold** for 30 frames  
- **ROI-based processing** for speed
- **Frame skipping** (process every 3rd frame)
- **Quality scoring** for face selection

## 🛠️ Dependencies

- `opencv-python` - Computer vision operations
- `numpy` - Numerical computations
- `onnxruntime` - ONNX model inference
- `scipy` - Scientific computing
- `mediapipe` - Face landmark detection
- `tqdm` - Progress bars

## 📖 Documentation

- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Complete usage instructions
- [Component Testing](#component-testing) - Individual module tests
- [Troubleshooting](#troubleshooting) - Common issues and solutions

## 🧪 Component Testing

Test each component individually:

```bash
python src/camera.py          # Camera functionality
python src/detect.py          # Face detection
python src/landmarks.py       # 5-point landmarks
python src/align.py           # Face alignment
python src/embed.py           # Embedding extraction
python src/haar_5pt.py        # Combined pipeline
```

## 🔍 Troubleshooting

### Common Issues

**"ArcFace model not found"**
- Download `w600k_r50.onnx` and place in `models/` directory

**"No enrolled faces found"**  
- Run enrollment first: `python src/enroll.py`

**"Camera cannot be opened"**
- Check camera permissions and availability

**Poor recognition accuracy**
- Increase enrollment samples (5-10 per person)
- Ensure good lighting conditions
- Run evaluation to optimize threshold

## 🎯 Design Philosophy

This implementation strictly follows the document specifications:
- **CPU-first**: No GPU dependencies
- **Modular**: Each stage independently testable
- **Exact pipeline**: Detection → Landmarks → Alignment → Embedding
- **Specific models**: Haar + MediaPipe + ArcFace ONNX
- **Fixed alignment**: 5-point similarity transform
- **L2-normalized**: 512-dimensional embeddings

## 📄 License

This project is open source. Please ensure you comply with the licenses of all dependencies and models used.

## 🤝 Contributing

Contributions are welcome! Please ensure any changes maintain compatibility with the original document specifications.

## 📞 Support

For issues and questions, please use the GitHub Issues tab.
