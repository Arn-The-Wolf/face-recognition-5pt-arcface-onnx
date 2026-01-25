# Face Recognition System - Complete Usage Guide

## System Overview

This face recognition system implements the exact architecture described in "Face Recognition with ArcFace ONNX and 5-Point Alignment" by Gabriel Baziramwabo. It follows a modular pipeline design with CPU-only execution.

### Pipeline Architecture
1. **Detection**: Haar cascades for face detection
2. **Landmarks**: MediaPipe FaceMesh for 5-point landmarks
3. **Alignment**: Similarity transform to 112×112
4. **Embedding**: ArcFace ONNX (w600k_r50.onnx) for 512-D embeddings
5. **Enrollment**: Reference database creation
6. **Evaluation**: Threshold optimization
7. **Recognition**: Live identification

## Setup Instructions

### 1. Environment Setup
```bash
# Navigate to project directory
cd face-recognition-5pt

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download ArcFace Model
You need to download the ArcFace ONNX model:
1. Download the buffalo_l model pack from InsightFace
2. Extract `w600k_r50.onnx` 
3. Place it at `models/embedder_arcface.onnx`

**Important**: The system will not work without this model file.

## Component Testing

Test each component individually to ensure proper setup:

### Camera Test
```bash
python src/camera.py
```
- Tests camera initialization and frame capture
- Controls: 'q' to quit, 's' to save frame

### Face Detection Test
```bash
python src/detect.py
```
- Tests Haar cascade face detection
- Shows detected face bounding boxes
- Controls: 'q' to quit

### Landmark Detection Test
```bash
python src/landmarks.py
```
- Tests MediaPipe 5-point landmark detection
- Shows: left eye, right eye, nose tip, left mouth, right mouth
- Controls: 'q' to quit, 's' to save landmarks

### Face Alignment Test
```bash
python src/align.py
```
- Tests similarity transform alignment to 112×112
- Shows original and aligned face side by side
- Controls: 'q' to quit, 's' to save aligned face

### Embedding Extraction Test
```bash
python src/embed.py
```
- Tests ArcFace ONNX embedding extraction
- Validates 512-D L2-normalized embeddings
- Controls: 'q' to quit, 's' to save embedding

### Combined Detection Test
```bash
python src/haar_5pt.py
```
- Tests integrated Haar + 5-point pipeline
- Shows face quality scores
- Controls: 'q' to quit, 's' to save, 'a' to toggle show all faces

## Enrollment Process

### Interactive Enrollment
```bash
python src/enroll.py
```

**Enrollment Workflow:**
1. Enter person's name when prompted
2. Position face in camera view
3. Use controls to capture samples:
   - **SPACE**: Manual capture (recommended)
   - **A**: Toggle auto-capture mode
   - **S**: Save enrollment (requires 3+ samples)
   - **R**: Reset samples and start over
   - **Q**: Quit without saving

**Best Practices:**
- Capture 5-10 samples per person
- Vary head pose slightly between captures
- Ensure good lighting and clear face visibility
- Wait for "Ready to capture" message before capturing

**Database Storage:**
- Mean embeddings: `data/db/face_db.npz`
- Metadata: `data/db/face_db.json`
- Face crops: `data/enroll/<name>/`

## Evaluation System

### Threshold Evaluation
```bash
python src/evaluate.py
```

**What it does:**
- Loads aligned face crops from `data/enroll/`
- Computes genuine distances (same person pairs)
- Computes impostor distances (different person pairs)
- Generates FAR/FRR sweep table
- Suggests optimal threshold for FAR = 1%

**Output:**
- Distribution summaries
- FAR/FRR curves
- ROC and DET curves (if matplotlib available)
- Recommended threshold value

**Requirements:**
- At least 2 people enrolled
- Multiple samples per person (3+ recommended)

## Live Recognition

### Real-time Recognition
```bash
python src/recognize.py
```

**Recognition Features:**
- Real-time face detection and recognition
- Temporal smoothing over 10 frames
- Acceptance hold for 30 frames
- ROI-based processing for performance
- Process every 3rd frame for speed

**Controls:**
- **+**: Increase threshold (more restrictive)
- **-**: Decrease threshold (less restrictive)  
- **Q**: Quit

**Display Elements:**
- Green box: Recognized person
- Red box: Unknown person
- Confidence bar at bottom
- Threshold indicator line
- Person name and confidence score

### Image Recognition Test
```bash
python src/recognize.py path/to/image.jpg
```
- Test recognition on a single image file
- Shows recognition result and confidence

## System Parameters

### Detection Parameters
- **Scale factor**: 1.1 (Haar cascade)
- **Min neighbors**: 5 (Haar cascade)
- **Min face size**: 30×30 pixels

### Alignment Parameters
- **Output size**: 112×112 pixels
- **Transform**: Similarity (rotation, scale, translation)
- **Template**: Fixed 5-point canonical positions

### Embedding Parameters
- **Model**: ArcFace w600k_r50.onnx
- **Dimension**: 512
- **Normalization**: L2-normalized
- **Input**: 112×112×3 RGB, normalized to [0,1]

### Recognition Parameters
- **Default threshold**: 0.4 (cosine similarity)
- **Temporal window**: 10 frames
- **Acceptance hold**: 30 frames
- **Processing rate**: Every 3rd frame

## Troubleshooting

### Common Issues

**"ArcFace model not found"**
- Download w600k_r50.onnx and place in `models/` directory
- Ensure filename is exactly `embedder_arcface.onnx`

**"No enrolled faces found"**
- Run enrollment first: `python src/enroll.py`
- Ensure at least one person is successfully enrolled

**"Camera cannot be opened"**
- Check camera permissions
- Try different camera ID (0, 1, 2...)
- Ensure no other application is using camera

**Poor recognition accuracy**
- Increase number of enrollment samples (5-10 per person)
- Ensure good lighting during enrollment and recognition
- Adjust threshold using +/- keys during recognition
- Run evaluation to find optimal threshold

**Slow performance**
- System processes every 3rd frame by default
- Reduce camera resolution if needed
- Ensure CPU-only execution (no GPU dependencies)

### Performance Optimization

**For better speed:**
- Reduce camera resolution to 640×480
- Increase frame skip rate in recognition
- Use smaller face detection parameters

**For better accuracy:**
- Increase enrollment samples per person
- Use consistent lighting conditions
- Ensure faces are well-aligned during enrollment
- Run evaluation to optimize threshold

## File Structure

```
face-recognition-5pt/
├── data/
│   ├── enroll/           # Enrollment face crops
│   │   └── <name>/       # Per-person directories
│   └── db/               # Face database
│       ├── face_db.npz   # Embeddings
│       └── face_db.json  # Metadata
├── models/
│   └── embedder_arcface.onnx  # ArcFace model
├── src/
│   ├── camera.py         # Camera handling
│   ├── detect.py         # Haar face detection
│   ├── landmarks.py      # 5-point landmarks
│   ├── align.py          # Face alignment
│   ├── embed.py          # ArcFace embedding
│   ├── enroll.py         # Enrollment system
│   ├── evaluate.py       # Threshold evaluation
│   ├── recognize.py      # Live recognition
│   └── haar_5pt.py       # Combined detection
└── requirements.txt      # Dependencies
```

## Design Philosophy

This system strictly follows the document's specifications:
- **CPU-first**: No GPU assumptions
- **Modular**: Each stage is independently testable
- **Exact pipeline**: Detection → Landmarks → Alignment → Embedding
- **Specific models**: Haar cascades + MediaPipe + ArcFace ONNX
- **Fixed alignment**: 5-point similarity transform to 112×112
- **L2-normalized**: 512-dimensional embeddings
- **Threshold-based**: Cosine similarity with configurable threshold

The implementation prioritizes correctness and adherence to the document over performance optimizations or alternative approaches.