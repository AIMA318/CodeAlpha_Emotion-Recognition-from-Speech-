"""
Speech Emotion Recognition (SER)
--------------------------------
Features : MFCC + delta + delta-delta (log-mel optional)
Model    : 1D-CNN front-end -> Bidirectional LSTM -> attention-free pooling -> Dense
Datasets : RAVDESS (speaker-independent split) or TESS

Install:
    pip install librosa soundfile numpy scikit-learn tensorflow matplotlib seaborn tqdm

Usage:
    # RAVDESS (download "Audio_Speech_Actors_01-24" from Zenodo / Kaggle and unzip)
    python speech_emotion_recognition.py train --dataset ravdess --data_dir ./RAVDESS

    # TESS
    python speech_emotion_recognition.py train --dataset tess --data_dir ./TESS

    # Predict on a single file with a trained model
    python speech_emotion_recognition.py predict --file sample.wav
"""
import argparse
import glob
import json
import os
import random

import librosa
import numpy as np
from tqdm import tqdm

SR = 22050
DURATION = 3.0                 # seconds; clips are padded / cropped to this
N_MFCC = 40
HOP = 512
N_FRAMES = int(np.ceil(SR * DURATION / HOP))   # ~130 frames
SEED = 42
OUT_DIR = "ser_output"

RAVDESS_MAP = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
               "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}


# --------------------------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------------------------
def load_ravdess(data_dir):
    """Returns list of (path, emotion, speaker_id). Filename: 03-01-06-01-02-01-12.wav"""
    items = []
    for p in glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True):
        parts = os.path.basename(p).replace(".wav", "").split("-")
        if len(parts) != 7:
            continue
        items.append((p, RAVDESS_MAP[parts[2]], int(parts[6])))
    return items


def load_tess(data_dir):
    """TESS filenames look like OAF_back_angry.wav ; speaker = OAF / YAF."""
    items = []
    for p in glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True):
        name = os.path.basename(p).replace(".wav", "")
        spk, *_, emo = name.split("_")
        emo = {"ps": "surprised", "pleasant_surprised": "surprised"}.get(emo.lower(), emo.lower())
        items.append((p, emo, spk))
    return items


# --------------------------------------------------------------------------
# 2. Signal processing / feature extraction
# --------------------------------------------------------------------------
def load_audio(path):
    y, _ = librosa.load(path, sr=SR)
    y, _ = librosa.effects.trim(y, top_db=25)          # strip leading/trailing silence
    return y


def fix_length(y):
    n = int(SR * DURATION)
    return np.pad(y, (0, n - len(y))) if len(y) < n else y[:n]


def extract_features(y):
    """Returns array (N_FRAMES, 3*N_MFCC): MFCC, delta, delta-delta."""
    y = fix_length(y)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    feat = np.vstack([mfcc, d1, d2]).T                  # (time, features)
    return feat[:N_FRAMES].astype(np.float32)


# Simple data augmentation (applied to TRAIN set only)
def aug_noise(y):
    return y + np.random.randn(len(y)) * 0.005 * np.random.uniform(0.5, 1.5) * np.max(np.abs(y) + 1e-9)


def aug_pitch(y):
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=np.random.choice([-2, -1, 1, 2]))


def aug_stretch(y):
    return librosa.effects.time_stretch(y, rate=np.random.uniform(0.85, 1.15))


AUGS = [aug_noise, aug_pitch, aug_stretch]


def build_features(items, augment=False):
    X, y_lbl = [], []
    for path, emo, _ in tqdm(items, desc="Extracting features" + (" (+aug)" if augment else "")):
        wav = load_audio(path)
        X.append(extract_features(wav)); y_lbl.append(emo)
        if augment:
            for fn in AUGS:
                X.append(extract_features(fn(wav))); y_lbl.append(emo)
    return np.stack(X), np.array(y_lbl)


# --------------------------------------------------------------------------
# 3. Model: CNN -> BiLSTM
# --------------------------------------------------------------------------
def build_model(n_classes, input_shape):
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    inp = layers.Input(shape=input_shape)
    x = inp
    for filters in (64, 128, 128):                       # local spectral patterns
        x = layers.Conv1D(filters, 5, padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.25)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)   # temporal dynamics
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# --------------------------------------------------------------------------
# 4. Train / evaluate
# --------------------------------------------------------------------------
def speaker_split(items, dataset):
    """Speaker-independent split (no speaker appears in more than one set)."""
    speakers = sorted({s for *_, s in items}, key=str)
    random.Random(SEED).shuffle(speakers)
    if dataset == "tess":                                # only 2 speakers -> fall back to random split
        idx = list(range(len(items))); random.Random(SEED).shuffle(idx)
        n = len(idx); tr, va = idx[:int(.7 * n)], idx[int(.7 * n):int(.85 * n)]
        te = idx[int(.85 * n):]
        return [items[i] for i in tr], [items[i] for i in va], [items[i] for i in te]
    n = len(speakers)
    test_s, val_s = set(speakers[:max(1, n // 8)]), set(speakers[max(1, n // 8):max(2, n // 4)])
    pick = lambda S: [it for it in items if it[2] in S]
    train_s = set(speakers) - test_s - val_s
    return pick(train_s), pick(val_s), pick(test_s)


def train(args):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight

    np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    items = load_ravdess(args.data_dir) if args.dataset == "ravdess" else load_tess(args.data_dir)
    assert items, f"No .wav files found in {args.data_dir}"
    print(f"Found {len(items)} files")
    tr, va, te = speaker_split(items, args.dataset)
    print(f"train={len(tr)}  val={len(va)}  test={len(te)}")

    Xtr, ytr = build_features(tr, augment=not args.no_aug)
    Xva, yva = build_features(va)
    Xte, yte = build_features(te)

    classes = sorted(set(ytr))
    c2i = {c: i for i, c in enumerate(classes)}
    ytr, yva, yte = [np.array([c2i[c] for c in y]) for y in (ytr, yva, yte)]

    # Standardise with TRAIN statistics only
    mean, std = Xtr.mean(axis=(0, 1), keepdims=True), Xtr.std(axis=(0, 1), keepdims=True) + 1e-8
    Xtr, Xva, Xte = [(X - mean) / std for X in (Xtr, Xva, Xte)]

    model = build_model(len(classes), Xtr.shape[1:])
    model.summary()

    cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=ytr)
    cb = [tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_accuracy"),
          tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
          tf.keras.callbacks.ModelCheckpoint(os.path.join(OUT_DIR, "best_model.keras"),
                                             save_best_only=True, monitor="val_accuracy")]
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.epochs,
                     batch_size=32, class_weight=dict(enumerate(cw)), callbacks=cb)

    # ---- Evaluation
    loss, acc = model.evaluate(Xte, yte, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}")
    pred = model.predict(Xte).argmax(1)
    print(classification_report(yte, pred, target_names=classes, zero_division=0))

    cm = confusion_matrix(yte, pred, normalize="true")
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Normalised confusion matrix")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)

    plt.figure(figsize=(10, 4))
    for i, k in enumerate(("accuracy", "loss")):
        plt.subplot(1, 2, i + 1)
        plt.plot(hist.history[k], label="train"); plt.plot(hist.history["val_" + k], label="val")
        plt.title(k); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)

    # Save what inference needs
    meta = {"classes": classes, "mean": mean.tolist(), "std": std.tolist()}
    json.dump(meta, open(os.path.join(OUT_DIR, "meta.json"), "w"))
    print(f"Saved model + metadata to ./{OUT_DIR}")


# --------------------------------------------------------------------------
# 5. Inference
# --------------------------------------------------------------------------
def predict(args):
    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(OUT_DIR, "best_model.keras"))
    meta = json.load(open(os.path.join(OUT_DIR, "meta.json")))
    mean, std = np.array(meta["mean"]), np.array(meta["std"])

    x = (extract_features(load_audio(args.file))[None] - mean) / std
    probs = model.predict(x, verbose=0)[0]
    for c, p in sorted(zip(meta["classes"], probs), key=lambda t: -t[1]):
        print(f"{c:10s} {p:.3f}")
    print(f"\nPredicted emotion: {meta['classes'][int(probs.argmax())]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--dataset", choices=["ravdess", "tess"], default="ravdess")
    t.add_argument("--data_dir", required=True)
    t.add_argument("--epochs", type=int, default=80)
    t.add_argument("--no_aug", action="store_true", help="disable data augmentation")
    p = sub.add_parser("predict")
    p.add_argument("--file", required=True)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else predict(a)
