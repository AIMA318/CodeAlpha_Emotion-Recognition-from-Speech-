# 🎙️ Emotion Recognition from Speech - CodeAlpha Internship

> Task 1: Build a system that can recognize emotions from speech data.

**Repository:** https://github.com/AIMA318/CodeAlpha_Emotion-Recognition-from-Speech-
**Intern:** Aima | **GitHub:** @AIMA318

### 📌 Overview
This project detects human emotions (Happy, Sad, Angry, Fearful, Calm, Disgust, Surprised, Neutral) from speech audio using Machine Learning. It uses the RAVDESS dataset and includes an interactive Streamlit web app for real-time prediction.

### ✨ Features
- 🎧 Accepts .wav audio files
- 🧠 Trained on RAVDESS (1440 files, 24 actors)
- 📊 Feature Extraction: MFCC, Chroma, Mel Spectrogram
- 🤖 Model: MLP Classifier (80%+ accuracy)
- 💻 Interactive Streamlit GUI (app.py) with waveform & confidence chart

### 🛠️ Tech Stack
Python, Librosa, Scikit-learn, TensorFlow, Streamlit, NumPy, Matplotlib

### 📁 Files in this Repo
- `app.py` - Main Streamlit application
- `speech_emotion_recognition.py` - Training code
- `.gitignore` - To ignore heavy datasets
- `README.md` - Documentation

### 🚀 How to Run
1. Clone repo:
   git clone https://github.com/AIMA318/CodeAlpha_Emotion-Recognition-from-Speech-.git
2. Install libraries:
   pip install librosa scikit-learn streamlit tensorflow matplotlib
3. Run app:
   streamlit run app.py
4. Open http://localhost:8501 and upload a .wav file

### 📊 Results
- 8 Emotions classified
- Accuracy: ~85% on test set

### 🎯 Internship Info
This is submitted for CodeAlpha Machine Learning Internship - Task 1.

⭐ Star this repo if you like it!