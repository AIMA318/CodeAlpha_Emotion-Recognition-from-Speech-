import streamlit as st
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import pickle, os, glob
import tensorflow as tf

st.set_page_config(page_title="Emotion Recognition", page_icon="🎧", layout="wide")

# --- CSS ---
st.markdown("""
<style>
.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: white; }
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.emotion-box {
    background: linear-gradient(90deg, #6C63FF, #48C6EF);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: white;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:white;'>🎧 CodeAlpha - Speech Emotion Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#cccccc;'>Detect Human Emotions from Voice using Deep Learning (CNN + MFCC)</p>", unsafe_allow_html=True)
st.markdown("---")

def find_model():
    for p in ["ser_output/*.h5","*.h5","**/*.h5","**/*.keras"]:
        files = glob.glob(p, recursive=True)
        if files: return files[0]
    return None

def extract_features(y, sr):
    # Model ko (130, 120) chahiye - ye error se pata chala
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=120)
    mfcc = mfcc.T # shape = (time, 120)

    # 130 frames pe fix karo
    if mfcc.shape[0] < 130:
        # padding
        pad_width = 130 - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0,0)), mode='constant')
    else:
        # truncate
        mfcc = mfcc[:130, :]

    return np.expand_dims(mfcc, axis=0) # (1,130,120)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Audio File")
    uploaded_file = st.file_uploader("", type=["wav"])
    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        y, sr = librosa.load(uploaded_file, sr=22050)
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#6C63FF")
        ax.set_facecolor("none")
        fig.patch.set_alpha(0.0)
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Prediction Result")
    if uploaded_file:
        model_path = find_model()
        if model_path:
            st.success(f"Model loaded: {model_path}")
            model = tf.keras.models.load_model(model_path)

            features = extract_features(y, sr)
            st.write(f"Input shape to model: {features.shape}")

            pred = model.predict(features, verbose=0)[0]
            emotions = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']

            # Try to load encoder if exists
            try:
                enc_files = glob.glob("**/*.pkl", recursive=True)
                if enc_files:
                    with open(enc_files[0], 'rb') as f:
                        le = pickle.load(f)
                    emotions = list(le.classes_)
            except:
                pass

            emoji_map = {'happy':'😊','sad':'😢','angry':'😠','neutral':'😐','calm':'😌','fearful':'😨','disgust':'🤢','surprised':'😲'}
            final_emotion = emotions[np.argmax(pred)]

            st.markdown(f'<div class="emotion-box">{emoji_map.get(final_emotion,"🎙️")} {final_emotion.upper()}</div>', unsafe_allow_html=True)

            st.write("")
            for emo, score in zip(emotions, pred):
                st.progress(float(score), text=f"{emo} - {score*100:.1f}%")
        else:
            st.error("Model not found in ser_output!")
    else:
        st.info("👆 Please upload a.wav file to see prediction")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color:#888;'>Built with ❤️ by AIMA318</p>", unsafe_allow_html=True)