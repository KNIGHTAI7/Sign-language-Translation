import streamlit as st
import cv2
import numpy as np
import torch
import tempfile
import os
import sys
import time
from gtts import gTTS
import mediapipe as mp

# 1. Define these at the top level (not inside the class)
_Holistic = mp.solutions.holistic.Holistic
_POSE_CONNECTIONS = mp.solutions.holistic.POSE_CONNECTIONS
_HAND_CONNECTIONS = mp.solutions.holistic.HAND_CONNECTIONS
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# 2. Now your class uses those variables
class mp_holistic:
    Holistic         = _Holistic
    POSE_CONNECTIONS = _POSE_CONNECTIONS
    HAND_CONNECTIONS = _HAND_CONNECTIONS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from models.lstm_model import SignLanguageLSTM

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroGestures",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── MediaPipe ─────────────────────────────────────────────────


MODEL_PATH = "models/sign_language_lstm.pth"
MAX_FRAMES = 30
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────
# CSS — Professional Dark Theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.6);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    .stat-chip {
        background: rgba(102,126,234,0.2);
        border: 1px solid rgba(102,126,234,0.4);
        border-radius: 50px;
        padding: 0.4rem 1.2rem;
        color: #a78bfa;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Mode selector */
    .mode-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Cards */
    .glass-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }

    /* Top prediction display */
    .prediction-hero {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(102,126,234,0.4);
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { box-shadow: 0 8px 32px rgba(102,126,234,0.3); }
        to   { box-shadow: 0 8px 48px rgba(118,75,162,0.6); }
    }
    .prediction-word {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .prediction-conf {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.8);
        margin-top: 0.3rem;
    }

    /* Confidence bars */
    .conf-item {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.6rem;
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
    }
    .conf-rank {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: rgba(102,126,234,0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
        color: #a78bfa;
        flex-shrink: 0;
    }
    .conf-word {
        flex: 1;
        font-weight: 600;
        color: white;
        font-size: 0.95rem;
        text-transform: capitalize;
    }
    .conf-pct {
        font-size: 0.9rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .conf-bar-wrap {
        width: 80px;
        height: 6px;
        background: rgba(255,255,255,0.1);
        border-radius: 3px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #667eea, #a78bfa);
    }

    /* Tip box */
    .tip-box {
        background: rgba(250,204,21,0.1);
        border: 1px solid rgba(250,204,21,0.3);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        color: #fcd34d;
        font-size: 0.92rem;
        font-weight: 500;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Camera placeholder */
    .cam-placeholder {
        height: 380px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: rgba(255,255,255,0.02);
        border: 2px dashed rgba(102,126,234,0.4);
        border-radius: 20px;
        color: rgba(255,255,255,0.4);
        font-size: 1rem;
        gap: 0.8rem;
    }

    /* Status badges */
    .status-ok  { color: #34d399; font-weight: 700; }
    .status-no  { color: #f87171; font-weight: 700; }

    /* Sentence builder */
    .sentence-display {
        background: rgba(102,126,234,0.1);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        min-height: 50px;
        word-break: break-word;
        letter-spacing: 0.02em;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: rgba(255,255,255,0.9);
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Signs grid */
    .signs-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.5rem;
    }
    .sign-chip {
        background: rgba(102,126,234,0.15);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 20px;
        padding: 0.25rem 0.7rem;
        color: #a78bfa;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: capitalize;
    }

    /* Streamlit button overrides */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1.2rem !important;
        border: none !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102,126,234,0.6) !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(255,255,255,0.07) !important;
        color: rgba(255,255,255,0.8) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
    }

    /* Radio buttons */
    .stRadio > div {
        display: flex;
        gap: 1rem;
        flex-direction: row !important;
    }
    .stRadio label {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        cursor: pointer !important;
        color: rgba(255,255,255,0.7) !important;
        font-weight: 600 !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #a78bfa) !important;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.1) !important; }

    /* File uploader */
    .stFileUploader {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 16px !important;
        border: 2px dashed rgba(102,126,234,0.4) !important;
    }

    /* Success/info/error messages */
    .stSuccess, .stInfo, .stError, .stWarning {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Feature Extraction (must match training exactly)
# ─────────────────────────────────────────────────────────────
def extract_keypoints_raw(results):
    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                     for lm in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33 * 4)
    left_hand = np.array([[lm.x, lm.y, lm.z]
                           for lm in results.left_hand_landmarks.landmark]).flatten() \
                if results.left_hand_landmarks else np.zeros(21 * 3)
    right_hand = np.array([[lm.x, lm.y, lm.z]
                            for lm in results.right_hand_landmarks.landmark]).flatten() \
                 if results.right_hand_landmarks else np.zeros(21 * 3)
    return pose, left_hand, right_hand


def make_signer_invariant(pose, left_hand, right_hand):
    features   = []
    pose_pts   = pose.reshape(33, 4)[:, :3]
    l_shoulder = pose_pts[11]; r_shoulder = pose_pts[12]
    l_hip      = pose_pts[23]; r_hip      = pose_pts[24]
    shoulder_dist = np.linalg.norm(l_shoulder - r_shoulder) + 1e-6
    origin        = (l_hip + r_hip) / 2.0
    pose_norm     = (pose_pts - origin) / shoulder_dist
    key_joints    = [0, 11, 12, 13, 14, 15, 16, 23, 24]
    features.append(pose_norm[key_joints].flatten())
    for hand_raw in [left_hand, right_hand]:
        hand_pts     = hand_raw.reshape(21, 3)
        wrist        = hand_pts[0]
        hand_detected = np.any(hand_pts != 0)
        if hand_detected:
            hand_rel  = (hand_pts - wrist)
            hand_size = np.linalg.norm(hand_pts[12] - wrist) + 1e-6
            hand_rel  = hand_rel / hand_size
            angles    = []
            for finger in [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]:
                for j in range(len(finger)-1):
                    v = hand_pts[finger[j+1]] - hand_pts[finger[j]]
                    angles.append(v / (np.linalg.norm(v) + 1e-6))
            features.append(np.concatenate([hand_rel.flatten(), np.array(angles).flatten()]))
        else:
            features.append(np.zeros(108))
    return np.concatenate(features).astype(np.float32)


def pad_or_truncate(seq, max_frames=MAX_FRAMES):
    T = len(seq)
    if T >= max_frames:
        start = (T - max_frames) // 2
        return seq[start:start + max_frames]
    pad = np.zeros((max_frames - T, seq.shape[1]), dtype=np.float32)
    return np.concatenate([seq, pad], axis=0)


def normalize(seq):
    mean = seq.mean(axis=0, keepdims=True)
    std  = seq.std(axis=0,  keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    return (seq - mean) / std


def draw_landmarks(frame, results):
    mp_drawing.draw_landmarks(frame, results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS, mp_styles.get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style())
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style())
    return frame


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    ckpt      = torch.load(MODEL_PATH, map_location=DEVICE)
    label_map = ckpt['label_map']
    config    = ckpt['config']
    model = SignLanguageLSTM(
        input_size  = ckpt.get('input_size', 243),
        hidden_size = config['hidden_size'],
        num_layers  = config['num_layers'],
        num_classes = ckpt.get('num_classes', len(label_map)),
        dropout     = 0.0,
        nhead       = config.get('nhead', 4)
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model, label_map


def tta_predict(keypoints_seq, model, label_map, top_k=5, n_aug=8):
    """
    Test-Time Augmentation — run prediction N times with slight variations
    and average the probabilities. Much more accurate than single prediction.
    """
    if len(keypoints_seq) == 0:
        return []

    seq = np.array(keypoints_seq, dtype=np.float32)

    all_probs = []
    for i in range(n_aug):
        aug = seq.copy()

        if i > 0:  # first pass = no augmentation (clean)
            # slight noise
            aug = aug + np.random.normal(0, 0.01, aug.shape).astype(np.float32)
            # slight time shift
            if np.random.rand() < 0.5:
                shift = np.random.randint(-2, 3)
                aug   = np.roll(aug, shift, axis=0)

        aug = pad_or_truncate(aug)
        aug = normalize(aug)
        x   = torch.FloatTensor(aug).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        all_probs.append(probs)

    avg_probs  = np.mean(all_probs, axis=0)
    top_idx    = avg_probs.argsort()[::-1][:top_k]

    return [(label_map.get(str(i), "unknown"), round(float(avg_probs[i]), 4))
            for i in top_idx]


def extract_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_seq, frames_vis = [], []
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5,
                               model_complexity=1) as holistic:
        while len(keypoints_seq) < MAX_FRAMES * 2:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True
            vis = draw_landmarks(rgb.copy(), results)
            frames_vis.append(vis)
            if results.pose_landmarks:
                pose, lh, rh = extract_keypoints_raw(results)
                kp = make_signer_invariant(pose, lh, rh)
            else:
                kp = np.zeros(243)
            keypoints_seq.append(kp)
    cap.release()
    return keypoints_seq, frames_vis


def text_to_speech(text):
    try:
        tts  = gTTS(text=text, lang='en', slow=False)
        path = tempfile.mktemp(suffix='.mp3')
        tts.save(path)
        with open(path, 'rb') as f:
            audio = f.read()
        os.remove(path)
        return audio
    except:
        return None


def render_predictions(predictions):
    """Render styled prediction cards."""
    if not predictions:
        return
    top_word, top_conf = predictions[0]
    st.markdown(f"""
    <div class="prediction-hero">
        <div style="font-size:2.5rem; margin-bottom:0.3rem">🤟</div>
        <div class="prediction-word">{top_word}</div>
        <div class="prediction-conf">{top_conf*100:.1f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 All Predictions</div>', unsafe_allow_html=True)
    for rank, (word, conf) in enumerate(predictions, 1):
        bar_width = int(conf * 100)
        st.markdown(f"""
        <div class="conf-item">
            <div class="conf-rank">#{rank}</div>
            <div class="conf-word">{word}</div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:{bar_width}%"></div>
            </div>
            <div class="conf-pct">{conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# APP START
# ─────────────────────────────────────────────────────────────
model, label_map = load_model()

# Hero Section
device_icon = "⚡" if DEVICE == "cuda" else "💻"
st.markdown(f"""
<div class="hero">
    <div class="hero-title">🤟 NeuroGestures</div>
    <div class="hero-subtitle">
        Real-time American Sign Language recognition powered by Transformer Deep Learning
    </div>
    <div class="hero-stats">
        <span class="stat-chip">🧠 Transformer Model</span>
        <span class="stat-chip">👁️ MediaPipe Pose</span>
        <span class="stat-chip">{device_icon} {DEVICE.upper()}</span>
        {'<span class="stat-chip">✅ ' + str(len(label_map)) + ' Signs</span>' if label_map else '<span class="stat-chip">❌ Model Not Loaded</span>'}
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("❌ Model not found at `models/sign_language_lstm.pth`. Please train the model first.")
    st.stop()

# Mode selector
mode = st.radio("", ["📁  Upload Video", "📷  Live Camera"], horizontal=True)
st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MODE 1 — Upload Video
# ─────────────────────────────────────────────────────────────
if mode == "📁  Upload Video":
    col1, col2 = st.columns([1.3, 1], gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📂 Upload Your Video</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="tip-box">
            💡 <b>Tips for best results:</b> Ensure good lighting, keep hands visible,
            sign at a normal speed, and face the camera directly.
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop your video here or click to browse",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded:
            tmp = tempfile.mktemp(suffix=f".{uploaded.name.split('.')[-1]}")
            with open(tmp, 'wb') as f:
                f.write(uploaded.read())

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🎬 Preview</div>', unsafe_allow_html=True)
            st.video(tmp)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🔍  Translate Sign Language", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing video with AI..."):
                    keypoints_seq, frames_vis = extract_from_video(tmp)
                    if len(keypoints_seq) < 5:
                        st.error("❌ No pose detected. Make sure the person is visible in the video.")
                    else:
                        predictions = tta_predict(keypoints_seq, model, label_map)

                        with col2:
                            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                            st.markdown('<div class="section-header">🎯 Translation Result</div>', unsafe_allow_html=True)
                            render_predictions(predictions)

                            if predictions:
                                top_word = predictions[0][0]
                                audio    = text_to_speech(top_word)
                                if audio:
                                    st.markdown('<div class="section-header" style="margin-top:1rem">🔊 Text to Speech</div>', unsafe_allow_html=True)
                                    st.audio(audio, format='audio/mp3')

                                # Add to sentence
                                if 'upload_sentence' not in st.session_state:
                                    st.session_state.upload_sentence = []
                                st.session_state._last_upload_pred = top_word
                            st.markdown('</div>', unsafe_allow_html=True)

                        # Skeleton frames
                        if frames_vis:
                            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                            st.markdown('<div class="section-header">🦴 Skeleton Keypoints</div>', unsafe_allow_html=True)
                            step  = max(1, len(frames_vis) // 6)
                            cols  = st.columns(6)
                            for i, c in enumerate(cols):
                                idx = min(i * step, len(frames_vis) - 1)
                                c.image(frames_vis[idx], use_container_width=True,
                                        caption=f"Frame {idx+1}")
                            st.markdown('</div>', unsafe_allow_html=True)

            try: os.remove(tmp)
            except: pass

    with col2:
        # Sentence builder
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📝 Sentence Builder</div>', unsafe_allow_html=True)

        if 'upload_sentence' not in st.session_state:
            st.session_state.upload_sentence = []

        sentence_text = " ".join(st.session_state.upload_sentence)
        display_text  = sentence_text if sentence_text else "Your sentence will appear here..."
        color         = "white" if sentence_text else "rgba(255,255,255,0.3)"
        st.markdown(f'<div class="sentence-display" style="color:{color}">{display_text}</div>', unsafe_allow_html=True)

        b1, b2 = st.columns(2)
        with b1:
            if st.button("➕ Add Word", type="primary", use_container_width=True):
                last = st.session_state.get('_last_upload_pred')
                if last:
                    st.session_state.upload_sentence.append(last)
                    st.rerun()
        with b2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.upload_sentence = []
                st.rerun()

        if sentence_text:
            sent_audio = text_to_speech(sentence_text)
            if sent_audio:
                st.markdown('<div style="margin-top:0.8rem"></div>', unsafe_allow_html=True)
                st.audio(sent_audio, format='audio/mp3')

        st.markdown('</div>', unsafe_allow_html=True)

        # Supported signs
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📋 Supported Signs</div>', unsafe_allow_html=True)
        signs_html = "".join([f'<span class="sign-chip">{s}</span>' for s in label_map.values()])
        st.markdown(f'<div class="signs-grid">{signs_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MODE 2 — Live Camera
# ─────────────────────────────────────────────────────────────
elif mode == "📷  Live Camera":
    col1, col2 = st.columns([1.3, 1], gap="large")

    with col2:
        pred_ph   = st.empty()
        pred_ph.markdown("""
        <div class="glass-card" style="text-align:center; padding:2rem; color:rgba(255,255,255,0.3)">
            <div style="font-size:3rem">🤟</div>
            <div style="margin-top:0.5rem">Predictions will appear here</div>
        </div>
        """, unsafe_allow_html=True)

        audio_ph  = st.empty()
        st.divider()

        # Sentence builder
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📝 Sentence Builder</div>', unsafe_allow_html=True)

        if 'live_sentence' not in st.session_state:
            st.session_state.live_sentence = []
        if 'live_last_pred' not in st.session_state:
            st.session_state.live_last_pred = None

        sent_text = " ".join(st.session_state.live_sentence)
        disp_text = sent_text if sent_text else "Your sentence will appear here..."
        color     = "white" if sent_text else "rgba(255,255,255,0.3)"
        sent_ph   = st.empty()
        sent_ph.markdown(f'<div class="sentence-display" style="color:{color}">{disp_text}</div>', unsafe_allow_html=True)

        b1, b2 = st.columns(2)
        with b1:
            add_btn = st.button("➕ Add Word", type="primary", use_container_width=True)
        with b2:
            clr_btn = st.button("🗑️ Clear", use_container_width=True)

        if add_btn and st.session_state.live_last_pred:
            st.session_state.live_sentence.append(st.session_state.live_last_pred)
            st.rerun()
        if clr_btn:
            st.session_state.live_sentence = []
            st.rerun()

        if sent_text:
            sent_audio = text_to_speech(sent_text)
            if sent_audio:
                st.audio(sent_audio, format='audio/mp3')

        st.markdown('</div>', unsafe_allow_html=True)

        # Supported signs
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📋 Supported Signs</div>', unsafe_allow_html=True)
        signs_html = "".join([f'<span class="sign-chip">{s}</span>' for s in label_map.values()])
        st.markdown(f'<div class="signs-grid">{signs_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col1:
        st.markdown("""
        <div class="tip-box">
            💡 <b>Tips:</b> Stay in good lighting · Keep hands & upper body visible ·
            Hold each sign for 1–2 seconds · Click <b>Predict</b> after signing
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1: start_btn   = st.button("▶️  Start Camera",  type="primary",    use_container_width=True)
        with c2: predict_btn = st.button("🔍  Predict Sign",  use_container_width=True)
        with c3: stop_btn    = st.button("⏹️  Stop Camera",   use_container_width=True)

        frame_ph  = st.empty()
        status_ph = st.empty()

        if 'cam_running' not in st.session_state:
            st.session_state.cam_running  = False
        if 'cam_buffer' not in st.session_state:
            st.session_state.cam_buffer   = []

        if start_btn:
            st.session_state.cam_running = True
            st.session_state.cam_buffer  = []
        if stop_btn:
            st.session_state.cam_running = False

        if not st.session_state.cam_running:
            frame_ph.markdown("""
            <div class="cam-placeholder">
                <span style="font-size:3rem">📷</span>
                <span>Click <b>Start Camera</b> to begin</span>
                <span style="font-size:0.85rem; opacity:0.6">Camera access required</span>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.cam_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot access camera. Check permissions.")
                st.session_state.cam_running = False
            else:
                kp_buffer   = []
                frame_count = 0
                max_frames  = MAX_FRAMES * 4

                with mp_holistic.Holistic(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                ) as holistic:
                    while st.session_state.cam_running and frame_count < max_frames:
                        ret, frame = cap.read()
                        if not ret: break

                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb.flags.writeable = False
                        results = holistic.process(rgb)
                        rgb.flags.writeable = True

                        display = draw_landmarks(rgb.copy(), results)

                        pose_ok  = results.pose_landmarks is not None
                        lh_ok    = results.left_hand_landmarks is not None
                        rh_ok    = results.right_hand_landmarks is not None

                        # Status overlay
                        overlay_bg = display.copy()
                        cv2.rectangle(overlay_bg, (0, 0), (220, 110), (0,0,0), -1)
                        cv2.addWeighted(overlay_bg, 0.5, display, 0.5, 0, display)

                        def put(text, y, ok):
                            color = (52,211,153) if ok else (248,113,113)
                            cv2.putText(display, text, (10, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                        put(f"Pose:   {'OK' if pose_ok  else 'NOT DETECTED'}", 30,  pose_ok)
                        put(f"L.Hand: {'OK' if lh_ok    else 'NOT DETECTED'}", 60,  lh_ok)
                        put(f"R.Hand: {'OK' if rh_ok    else 'NOT DETECTED'}", 90,  rh_ok)

                        # Frame counter bar
                        filled = int((len(kp_buffer) / MAX_FRAMES) * display.shape[1])
                        cv2.rectangle(display, (0, display.shape[0]-8), (display.shape[1], display.shape[0]), (30,30,30), -1)
                        cv2.rectangle(display, (0, display.shape[0]-8), (filled, display.shape[0]), (102,126,234), -1)
                        cv2.putText(display, f"Buffer: {len(kp_buffer)}/{MAX_FRAMES}",
                                    (10, display.shape[0]-14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

                        frame_ph.image(display, channels="RGB", use_container_width=True)

                        if results.pose_landmarks:
                            pose, lh, rh = extract_keypoints_raw(results)
                            kp = make_signer_invariant(pose, lh, rh)
                        else:
                            kp = np.zeros(243)
                        kp_buffer.append(kp)

                        # Auto-predict when buffer full
                        if len(kp_buffer) >= MAX_FRAMES:
                            preds = tta_predict(kp_buffer[-MAX_FRAMES:], model, label_map)
                            if preds:
                                top_word, top_conf = preds[0]
                                st.session_state.live_last_pred = top_word

                                with pred_ph.container():
                                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                                    render_predictions(preds)
                                    st.markdown('</div>', unsafe_allow_html=True)

                                audio = text_to_speech(top_word)
                                if audio:
                                    audio_ph.audio(audio, format='audio/mp3')

                            kp_buffer = kp_buffer[MAX_FRAMES//2:]  # sliding window

                        status_ph.markdown(f"🔴 **Recording** · Frame {frame_count+1}/{max_frames} · Buffer: {len(kp_buffer)}/{MAX_FRAMES}")
                        frame_count += 1
                        time.sleep(0.033)

                cap.release()
                status_ph.success("✅ Camera stopped")

# Footer
st.markdown("""
<div style="text-align:center; color:rgba(255,255,255,0.3); font-size:0.8rem; padding:2rem 0 1rem">
    🤟 NeuroGestures · Built with PyTorch Transformer + MediaPipe + Streamlit<br>
    Empowering accessibility through deep learning · 60% accuracy on 20 ASL signs
</div>
""", unsafe_allow_html=True)
