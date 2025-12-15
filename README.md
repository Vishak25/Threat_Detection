# Video Anomaly Detection using Weakly Supervised MIL

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.16](https://img.shields.io/badge/tensorflow-2.16-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning system for detecting anomalies (Fighting, Arrest) in surveillance videos using **Multiple Instance Learning (MIL)** with weak supervision.

## Quick Links

- **[Full Report (REPORT.md)](REPORT.md)** - Comprehensive documentation with methodology, experiments, and results
- **[Figures Directory](figures/)** - All generated visualizations

## Key Results

| Metric | Value |
|:-------|:------|
| Validation Accuracy | **74.31%** |
| Training Loss | **0.134** |
| Model Status | Stable (No mode collapse) |

## Architecture

```
Video → ResNet50V2 (Frozen) → MIL Scoring Head → Anomaly Score [0-1]
        (ImageNet)            (Trainable)
```

![Architecture](figures/fig2_architecture.png)

## Project Structure

```
camera_anomaly_detection/
├── config.py              # Hyperparameters
├── model.py               # MIL Scoring Head
├── dcsass_loader.py       # Data loading
├── extract_features.py    # Feature extraction
├── train.py               # Training script
├── realtime_inference.py  # Real-time detection
├── generate_figures.py    # Report figures
├── REPORT.md              # Full project report
├── figures/               # Generated visualizations
└── requirements.txt       # Dependencies
```

## Installation

```bash
git clone <repository>
cd camera_anomaly_detection
pip install -r requirements.txt
```

## Usage

### 1. Feature Extraction

```bash
python extract_features.py
```

### 2. Training

```bash
python train.py
```

### 3. Real-Time Inference

```bash
python realtime_inference.py \
    --video surveillance.mp4 \
    --weights checkpoints/model_epoch_20.weights.h5 \
    --threshold 0.4
```

## Training Curves

![Training](figures/fig4_training_curves.png)

## Requirements

- Python 3.8+
- TensorFlow 2.16+
- OpenCV
- NumPy < 2.0.0
- SciPy < 1.12

## Citation

```bibtex
@inproceedings{sultani2018real,
  title={Real-world anomaly detection in surveillance videos},
  author={Sultani, Waqas and Chen, Chen and Shah, Mubarak},
  booktitle={CVPR},
  year={2018}
}
```

**Author:** Vishak Nandakumar (G01494598) and Baalavignesh Arunachalam (G01486574)
