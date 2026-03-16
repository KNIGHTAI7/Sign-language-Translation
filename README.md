# 🤟 NeuroGestures

Real-time American Sign Language (ASL) recognition powered by Transformer Deep Learning.

## Features
- 📁 Upload video for sign translation
- 📷 Live camera real-time translation
- 🔊 Text-to-speech output
- 📝 Sentence builder
- 🧠 Transformer model with signer-invariant features

## Tech Stack
- **Model**: Transformer Encoder + CLS Token
- **Pose**: MediaPipe Holistic
- **App**: Streamlit
- **DL**: PyTorch + PyTorch Lightning

## Accuracy
- Top-1: 60% | Top-3: 73.8% on 20 ASL signs
- Trained on WLASL dataset (official splits)

## Supported Signs
book, drink, computer, before, chair, go, clothes, who,
candy, cousin, deaf, fine, help, no, thin, walk, year,
yes, all, black

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
