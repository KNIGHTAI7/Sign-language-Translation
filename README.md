# 🤟 Sign Language Translation — Video to Text & Speech

> An AI-powered deep learning pipeline that translates sign language from video into text and speech in real-time, making digital communication accessible to 70M+ deaf individuals worldwide.

---

## 🧠 Overview

**Sign Language Translation** is a deep learning system that bridges the communication gap between the deaf/hard-of-hearing community and the hearing world. It accepts video input (live webcam or pre-recorded), detects hand gestures and body pose, and translates continuous sign sequences into readable text and audible speech.

This project supports **multiple sign languages** including **American Sign Language (ASL)** and **British Sign Language (BSL)**, and is designed for extensibility to other sign systems.

---

## ❗ Problem Statement

| Statistic | Impact |
|-----------|--------|
| **70 million+** | Deaf people worldwide face daily communication barriers |
| **99%** | Of web content remains inaccessible to the deaf community |
| **0** | Publicly available, real-time universal sign language translators |

Sign language is the primary language for millions of deaf people globally, yet it remains largely unsupported in digital systems. This project aims to solve that.

---

## ✨ Features

- 🎥 **Real-time webcam sign recognition** with live text overlay
- 🔄 **Continuous sign recognition** — full sentences, not just isolated words
- 🌍 **Multi-language support** — ASL, BSL, and extensible to ISL, JSL, Auslan, etc.
- 🦴 **Pose estimation** using MediaPipe and OpenPose (21 hand landmarks per hand)
- 🧠 **3D CNN + Transformer** for spatial-temporal feature extraction and translation
- 🔊 **Text-to-Speech (TTS)** output for complete Video → Text → Speech pipeline
- ⚡ **Low latency** — designed for <100ms real-time inference

---

## 🏗️ Architecture

The system is built on a multi-stage deep learning architecture:

```
Video Input
    │
    ▼
┌─────────────────────┐
│  Pose Estimation    │  ← MediaPipe Holistic / OpenPose
│  (Hand Landmarks)   │     21 keypoints per hand
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3D CNN             │  ← I3D / R(2+1)D
│  Feature Extraction │     Spatial + Temporal features
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Sequence Model     │  ← LSTM / GRU / Transformer
│  (Temporal Modeling)│     Sign sequence recognition
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Sign-to-Text       │  ← Seq2Seq Transformer
│  Translation        │     Beam search decoding
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Text-to-Speech     │  ← Google TTS / pyttsx3
│  Output             │
└─────────────────────┘
```

### Model Components

| Component | Model | Purpose |
|-----------|-------|---------|
| **3D CNN** | I3D, R(2+1)D | Capture spatial + temporal features from video |
| **Pose Estimation** | MediaPipe, OpenPose | Extract hand landmarks and body skeleton |
| **Sequence Model** | LSTM / GRU | Temporal modeling for continuous sign sequences |
| **Translation** | Transformer (Seq2Seq) | Attention-based sign-to-text conversion |
| **Speech** | Google TTS / pyttsx3 | Convert translated text to audio output |

---

## 📦 Datasets

| Dataset | Language | Videos | Classes | Description |
|---------|----------|--------|---------|-------------|
| [**WLASL**](https://dxli94.github.io/WLASL/) | ASL | 21,000 | 2,000 words | Word-level American Sign Language benchmark |
| [**MS-ASL**](https://www.microsoft.com/en-us/research/project/ms-asl/) | ASL | 25,000 | 1,000 classes | Microsoft's large-scale ASL dataset |
| [**PHOENIX-2014**](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) | German (DGS) | 7,096 | 1,081 signs | Weather broadcast continuous sign language |
| [**How2Sign**](https://how2sign.github.io/) | ASL | ~35 hours | Continuous | 35 hours of continuous ASL signing |

---

## 🛠️ Tech Stack

### Deep Learning
- [PyTorch](https://pytorch.org/) / [TensorFlow](https://www.tensorflow.org/)
- I3D / R(2+1)D (3D CNN architectures)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- LSTM / GRU networks

### Computer Vision
- [OpenCV](https://opencv.org/)
- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html)
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- FFmpeg (video processing)

### NLP & Speech
- Seq2Seq Transformer
- Beam Search Decoder
- Google Text-to-Speech / pyttsx3
- SentencePiece Tokenizer

### Infrastructure
- Python 3.10+
- CUDA (GPU acceleration)
- Flask / FastAPI (demo server)
- NumPy, Pandas

---

## 📁 Project Structure

```
sign-language-translation/
│
├── data/
│   ├── raw/                    # Raw dataset files
│   ├── processed/              # Preprocessed keypoints & features
│   └── splits/                 # Train / val / test splits
│
├── models/
│   ├── cnn3d/                  # I3D / R(2+1)D feature extractor
│   ├── pose/                   # MediaPipe / OpenPose wrappers
│   ├── sequence/               # LSTM / GRU / Transformer models
│   └── tts/                    # Text-to-Speech integration
│
├── scripts/
│   ├── preprocess.py           # Data preprocessing pipeline
│   ├── train.py                # Model training script
│   ├── evaluate.py             # Evaluation metrics (WER, BLEU)
│   └── inference.py            # Run inference on video/webcam
│
├── app/
│   ├── app.py                  # Flask/FastAPI demo server
│   ├── static/                 # Frontend assets
│   └── templates/              # HTML templates
│
├── notebooks/
│   ├── EDA.ipynb               # Exploratory data analysis
│   ├── model_training.ipynb    # Training experiments
│   └── demo.ipynb              # Interactive demo notebook
│
├── configs/
│   └── config.yaml             # Training & model hyperparameters
│
├── requirements.txt
├── README.md
└── LICENSE
```

*Made with ❤️ for accessibility and inclusion.*
