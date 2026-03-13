print("Testing environment...\n")

# PyTorch + CUDA
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("❌ CUDA not available - check driver")

# OpenCV
import cv2
print(f"\n✅ OpenCV: {cv2.__version__}")

# MediaPipe
import mediapipe as mp
print(f"✅ MediaPipe: {mp.__version__}")
from mediapipe.tasks import python as mp_tasks
print("✅ MediaPipe Tasks API: Loaded")

# Streamlit
import streamlit
print(f"✅ Streamlit: {streamlit.__version__}")

# Others
import numpy as np
import pandas as pd
import sklearn
import pytorch_lightning as pl
print(f"✅ NumPy: {np.__version__}")
print(f"✅ Pandas: {pd.__version__}")
print(f"✅ Scikit-learn: {sklearn.__version__}")
print(f"✅ PyTorch Lightning: {pl.__version__}")

print("\n🎉 All systems go! Ready to build.")