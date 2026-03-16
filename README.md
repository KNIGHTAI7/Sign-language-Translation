<div align="center">

# 🤟 NeuroGestures

### Real-Time American Sign Language Recognition powered by Transformer Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.7-green?style=for-the-badge)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

<br/>

> **Bridging the communication gap for 70 million deaf people worldwide**  
> NeuroGestures translates American Sign Language gestures into text and speech in real time using a custom Transformer deep learning model trained on the WLASL dataset.

<br/>

![NeuroGestures Demo](https://via.placeholder.com/900x400/0f0c29/667eea?text=NeuroGestures+%7C+ASL+Translator)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Supported Signs](#-supported-signs)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Acknowledgements](#-acknowledgements)

---

## 🧠 Overview

NeuroGestures is an end-to-end deep learning pipeline that recognizes American Sign Language (ASL) gestures from video input and converts them into readable text and audible speech.

The project addresses a real-world accessibility problem — only **1% of web content** is accessible to the deaf community, and communication barriers remain a daily challenge. NeuroGestures aims to lower that barrier using state-of-the-art computer vision and natural language processing techniques.

The system uses **MediaPipe Holistic** for skeletal pose extraction, a custom **signer-invariant feature engineering** pipeline to generalize across different people, and a **Transformer Encoder** architecture with a CLS token for sequence classification — the same attention-based design that powers modern NLP models like BERT.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📁 **Video Upload** | Upload any `.mp4`, `.avi`, `.mov` or `.mkv` file for translation |
| 📷 **Live Camera** | Real-time sign recognition using your webcam |
| 🦴 **Skeleton Overlay** | Visualize MediaPipe pose and hand keypoints on video frames |
| 🎯 **Top-5 Predictions** | See the top 5 predicted signs with confidence scores |
| 🔄 **Test-Time Augmentation** | 8-pass TTA for more stable and accurate predictions |
| 🔊 **Text-to-Speech** | Instantly convert predicted signs to spoken audio via gTTS |
| 📝 **Sentence Builder** | Chain multiple signs together to form complete sentences |
| ⚡ **GPU Accelerated** | CUDA support for faster inference on NVIDIA GPUs |
| 🎨 **Modern Dark UI** | Professional glass morphism interface built with Streamlit |

---

## 🎬 Demo

### Mode 1 — Upload Video
Upload a pre-recorded sign language video. NeuroGestures extracts keypoints frame by frame, runs the Transformer model, and returns the top predicted sign with confidence scores and a skeleton visualization.

### Mode 2 — Live Camera
Start your webcam and sign directly. The app captures 30 frames, extracts signer-invariant keypoints, and predicts the sign in real time. Use the Sentence Builder to chain multiple signs into a sentence and play it as speech.

---

## 🏗️ Architecture

NeuroGestures uses a multi-stage pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                    NeuroGestures Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Video / Webcam                                             │
│       │                                                     │
│       ▼                                                     │
│  MediaPipe Holistic                                         │
│  → Pose (33 landmarks) + Hands (21 × 2 landmarks)          │
│       │                                                     │
│       ▼                                                     │
│  Signer-Invariant Feature Engineering (243 features/frame) │
│  → Normalize by shoulder width & hip midpoint              │
│  → Hand keypoints relative to wrist                        │
│  → Finger joint angle unit vectors                         │
│       │                                                     │
│       ▼                                                     │
│  Velocity Features (frame-to-frame motion)                 │
│  → Concatenated → 486 features/frame                       │
│       │                                                     │
│       ▼                                                     │
│  Transformer Encoder (4 layers, 4 heads, d_model=256)      │
│  → Positional Encoding + CLS Token                         │
│  → Multi-Head Self-Attention × 4                           │
│  → Classification Head                                     │
│       │                                                     │
│       ▼                                                     │
│  Top-5 Predictions + TTA (8 passes)                        │
│       │                                                     │
│       ▼                                                     │
│  Text-to-Speech (gTTS)                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why Transformer over LSTM/CNN?

| Model | Val Accuracy | Notes |
|---|---|---|
| BiLSTM (3 layers) | 17.3% | Too many params for small dataset |
| 1D Temporal CNN | 44.0% | Better inductive bias for motion |
| **Transformer + Signer-Invariant** | **60.0%** | Best generalization across signers |

The key breakthrough was **signer-invariant feature engineering** — normalizing all keypoints relative to body proportions so the model learns *sign shape* rather than *person identity*.

---

## 📦 Dataset

NeuroGestures is trained on the **WLASL (Word-Level American Sign Language)** dataset.

| Property | Value |
|---|---|
| Source | [WLASL GitHub](https://github.com/dxli94/WLASL) / Kaggle |
| Total Videos | 21,083 |
| Total Classes | 2,000 ASL words |
| Videos Used | 499 (20-class subset) |
| Train Split | 350 videos (70%) |
| Val Split | 84 videos (17%) |
| Test Split | 65 videos (13%) |
| Resolution | 256 × 256 px |
| Avg Duration | ~0.84 seconds (~21 frames) |
| Split Strategy | Official signer-based splits (different signers in each set) |

The official signer-based splits ensure the model is evaluated on **unseen signers** — a much harder and more realistic evaluation than random splits.

---

## 📊 Model Performance

```
Training Configuration:
  Architecture  : Transformer Encoder (4 layers, 4 heads)
  Input Size    : 243 (signer-invariant keypoints)
  d_model       : 256
  Max Frames    : 30
  Batch Size    : 16
  Optimizer     : AdamW (lr=3e-4, weight_decay=1e-2)
  Scheduler     : Warmup (15 epochs) + Cosine Decay
  Augmentation  : Time scaling, noise, frame dropout,
                  temporal reverse, scale variation, Mixup
  Label Smoothing: 0.1
  GPU           : NVIDIA GeForce RTX 3050 6GB
```

| Metric | Score |
|---|---|
| **Val Accuracy (Top-1)** | **59.5%** |
| **Val Accuracy (Top-3)** | **81.0%** |
| **Test Accuracy (Top-1)** | **60.0%** |
| **Test Accuracy (Top-3)** | **73.8%** |
| Best Epoch | 86 |
| Total Parameters | ~2.1M |

> 📌 These results are on **official WLASL splits with unseen signers** — a significantly harder evaluation than random splits. Research papers report 60–80% on the full 2000-class dataset with much more data.

---

## 📁 Project Structure

```
NeuroGestures/
│
├── app.py                          # Streamlit application (entry point)
├── train.py                        # Model training script
├── extract_keypoints.py            # Keypoint extraction pipeline
├── explore_data.py                 # Dataset exploration & visualization
├── requirements.txt                # Python dependencies
├── packages.txt                    # System-level dependencies
├── README.md                       # Project documentation
│
├── .streamlit/
│   └── config.toml                 # Streamlit theme & server config
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── lstm_model.py           # Transformer model architecture
│   └── data/
│       ├── __init__.py
│       ├── dataset.py              # PyTorch Dataset & DataLoader
│       ├── build_index.py          # Video index builder from JSON
│       └── visualizer.py          # MediaPipe webcam visualizer
│
├── models/
│   ├── sign_language_lstm.pth      # Trained model weights
│   └── checkpoints/                # PyTorch Lightning checkpoints
│
├── data/
│   ├── raw/
│   │   └── wlasl/
│   │       ├── WLASL_v0.3.json     # Full dataset annotations
│   │       ├── nslt_100.json       # 100-class subset with video IDs
│   │       └── videos/             # 21,095 MP4 video files
│   ├── processed/                  # Extracted .npy keypoint files
│   │   └── {word}/
│   │       └── {video_id}.npy
│   └── splits/
│       ├── all_samples.csv         # Video paths + labels + splits
│       └── label_map.json          # Class index → word mapping
│
├── logs/
│   └── sign_transformer/           # CSVLogger training logs
│
└── notebooks/
    └── 01_data_exploration.ipynb
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU (recommended) with CUDA 11.8+
- Webcam (for live camera mode)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/neurogestures.git
cd neurogestures
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 4. Verify Setup
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); import mediapipe; print('MediaPipe OK')"
```

---

## 🚀 Usage

### Run the App
```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Train from Scratch

**Step 1 — Download WLASL Dataset**
```bash
kaggle datasets download -d sttaseen/wlasl2000-resized
```

**Step 2 — Build Video Index**
```bash
python src/data/build_index.py
```

**Step 3 — Extract Keypoints**
```bash
python extract_keypoints.py
```

**Step 4 — Train Model**
```bash
python train.py
```

---

## 🤟 Supported Signs

The current model recognizes **20 ASL signs**:

| | | | | |
|---|---|---|---|---|
| 📖 Book | 🥤 Drink | 💻 Computer | ⏮️ Before | 🪑 Chair |
| 🚶 Go | 👕 Clothes | ❓ Who | 🍬 Candy | 👨‍👩‍👧 Cousin |
| 🦻 Deaf | 👍 Fine | 🆘 Help | ❌ No | 🪶 Thin |
| 🚶 Walk | 📅 Year | ✅ Yes | 🌐 All | ⬛ Black |

---

## 🗺️ Roadmap

- [x] 20-class ASL recognition
- [x] Signer-invariant feature engineering
- [x] Transformer encoder architecture
- [x] Streamlit web application
- [x] Text-to-speech output
- [x] Sentence builder
- [ ] Scale to 100 ASL classes
- [ ] Scale to full 2000-class WLASL dataset
- [ ] Continuous sign recognition (full sentences)
- [ ] PHOENIX dataset (German Sign Language)
- [ ] Mobile app (React Native)
- [ ] Browser extension for real-time captioning
- [ ] Multi-language support (BSL, ISL, GSL)

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

```bash
# Fork the repo, then:
git clone https://github.com/YOUR_USERNAME/neurogestures.git
git checkout -b feature/your-feature-name
# make your changes
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
# Open a Pull Request
```

Please make sure your code follows the existing style and includes relevant comments.

---

## 🙏 Acknowledgements

- **WLASL Dataset** — Dongxu Li et al. — [Word-level Deep Sign Language Recognition from Video](https://arxiv.org/abs/1910.11006)
- **MediaPipe** — Google Research — Real-time pose and hand landmark detection
- **PyTorch Lightning** — Clean and scalable deep learning training
- **Streamlit** — Rapid ML application development
- **WLASL Kaggle Mirror** — [sttaseen/wlasl2000-resized](https://www.kaggle.com/datasets/sttaseen/wlasl2000-resized)

---

<div align="center">

**Made with ❤️ to improve accessibility for the deaf community**

⭐ Star this repo if you found it useful!

</div>
