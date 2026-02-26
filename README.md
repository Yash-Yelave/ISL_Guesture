# 🤟 ISL Gesture Detection & Translation System
### Complete Project Documentation — 24-Hour Hackathon Build

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Tech Stack](#tech-stack)
5. [Team Structure & Roles](#team-structure--roles)
6. [Project Scope](#project-scope)
7. [Environment Setup](#environment-setup)
8. [Folder Structure](#folder-structure)
9. [Implementation Guide](#implementation-guide)
   - [Data Collection](#phase-1-data-collection)
   - [Model Training](#phase-2-model-training)
   - [Backend Server](#phase-3-backend-server)
   - [Frontend App](#phase-4-frontend-app)
   - [Integration](#phase-5-integration)
10. [24-Hour Timeline](#24-hour-timeline)
11. [Deployment Guide](#deployment-guide)
12. [Testing Checklist](#testing-checklist)
13. [Presentation Strategy](#presentation-strategy)
14. [Troubleshooting](#troubleshooting)
15. [Future Roadmap](#future-roadmap)

---

## Project Overview

**Name:** ISL Real-Time Gesture Detection & Translation System  
**Type:** AI/ML + Full Stack Web Application  
**Goal:** Real-time Indian Sign Language gesture recognition that translates signs into text and speech, enabling communication accessibility for the deaf and hard-of-hearing community.

### Key Features
- Live camera-based gesture detection
- On-device landmark extraction (no raw frame upload)
- Real-time WebSocket communication
- BiLSTM/Transformer sequence model for temporal gesture understanding
- Text-to-speech output
- Progressive Web App (installable, no app store required)

---

## Problem Statement

Over **63 million people** in India are deaf or hard of hearing. Indian Sign Language (ISL) is their primary mode of communication, yet the vast majority of hearing people cannot understand it. This creates a massive communication barrier in hospitals, schools, public services, and everyday life.

Existing solutions are either:
- Expensive hardware gloves
- Desktop-only software
- Limited to small sign sets
- Not real-time

**Our solution** is a lightweight, browser-based, real-time ISL translator accessible on any device with a camera.

---

## Solution Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   USER'S DEVICE                          │
│                                                          │
│   📷 Camera Stream                                       │
│         ↓                                               │
│   🧠 MediaPipe Holistic (JS)                            │
│      - 21 hand landmarks × 2 hands                      │
│      - 33 pose landmarks                                 │
│      - Extracts ~258 float values per frame             │
│         ↓                                               │
│   📡 Socket.IO Client (WebSocket)                        │
│         ↓  [sends landmarks only, NOT raw frames]        │
└─────────────────────────────────────────────────────────┘
                         │
                    [WebSocket]
                         │
┌─────────────────────────────────────────────────────────┐
│                   FLASK SERVER                           │
│                                                          │
│   ⚙️  Flask-SocketIO                                     │
│         ↓                                               │
│   📦 30-Frame Sliding Window Buffer                      │
│         ↓  (fills at ~1 second @ 30fps)                 │
│   🤖 BiLSTM / Transformer Model                         │
│         ↓                                               │
│   🎯 Softmax Prediction (threshold: 0.85)               │
│         ↓                                               │
│   📤 Push result back via WebSocket                      │
└─────────────────────────────────────────────────────────┘
                         │
                    [WebSocket]
                         │
┌─────────────────────────────────────────────────────────┐
│                   FRONTEND OUTPUT                        │
│                                                          │
│   📝 Predicted Text Display                             │
│   🔊 Web Speech API (TTS)                               │
│   📜 Sentence Buffer                                     │
└─────────────────────────────────────────────────────────┘
```

### Why This Architecture?

| Design Decision | Reason |
|---|---|
| Send landmarks, not frames | 258 floats vs 921,600 pixels per frame — 3500x less data |
| WebSocket over HTTP | Persistent connection, no request overhead, bidirectional |
| Backend inference | LSTM/Transformer models too heavy for browser |
| MediaPipe Holistic | Captures hands + pose + face — ISL needs full body context |
| PWA over native app | No Play Store, no Android coding, same camera access |
| 30-frame sliding window | ~1 second of motion captures complete temporal gesture |

---

## Tech Stack

### Frontend
| Tool | Version | Purpose |
|---|---|---|
| React.js | 18+ | UI framework |
| @mediapipe/holistic | 0.5.x | On-device landmark extraction |
| socket.io-client | 4.x | WebSocket connection to backend |
| Web Speech API | Browser native | Text-to-speech output |
| PWA (manifest + SW) | — | Installable app experience |

### Backend
| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| Flask | 2.x | Web server |
| Flask-SocketIO | 5.x | WebSocket server |
| eventlet | latest | Async server for SocketIO |
| NumPy | latest | Array manipulation |
| scikit-learn | latest | Label encoding |

### AI / ML
| Tool | Version | Purpose |
|---|---|---|
| TensorFlow / Keras | 2.12+ | BiLSTM model training |
| MediaPipe | 0.9.x | Landmark extraction (data collection) |
| OpenCV | 4.x | Camera feed (data collection only) |
| NumPy | latest | Sequence arrays |
| Pickle | stdlib | Save label encoder |

### DevOps / Deployment
| Tool | Purpose |
|---|---|
| Cloudflare Tunnel | Expose local Flask server publicly (HTTPS) |
| GitHub | Version control (3 branches: frontend, backend, model) |
| VS Code | Development IDE |

---

## Team Structure & Roles

### Member 1 — AI/ML Engineer ⭐ (Most Critical)
**Owns:** Data collection script, model training, inference wrapper

**Tasks:**
- Write `collect.py` to record gesture sequences
- Perform and supervise data collection (15 signs × 30 sequences × 30 frames)
- Train BiLSTM model
- Export `isl_model.h5` and `label_encoder.pkl`
- Write `predict()` function for Flask integration
- Fine-tune model if accuracy < 90%

**Must coordinate with:** Member 3 on landmark array shape (must match browser output exactly)

---

### Member 2 — Frontend Developer
**Owns:** React app, MediaPipe JS integration, WebSocket client, UI/UX

**Tasks:**
- Set up React project and folder structure
- Integrate `@mediapipe/holistic` with webcam
- Flatten landmark output to match Member 1's format (258 values, same order)
- Set up Socket.IO client
- Build camera view + text output + sentence buffer UI
- Implement Web Speech API TTS
- Add PWA manifest and service worker

**Must coordinate with:** Member 1 on landmark format; Member 3 on WebSocket event names

---

### Member 3 — Backend / Integration Engineer
**Owns:** Flask server, WebSocket server, deployment, connecting all pieces

**Tasks:**
- Set up Flask + Flask-SocketIO project
- Implement `landmarks` WebSocket event handler
- Build 30-frame sliding window buffer
- Create **dummy prediction function first** (returns random label) so Member 2 can test immediately
- Replace dummy with real model once Member 1 delivers `isl_model.h5`
- Set up Cloudflare Tunnel for demo deployment
- Debug CORS and connection issues

**Must coordinate with:** Everyone — this is the integration role

---

### Support Member — QA, Documentation, Presentation
**Owns:** Testing, bug tracking, slides, demo video

- **Hours 0–8:** Help Member 1 perform gestures during data collection
- **Hours 8–16:** Write test scripts, test each sign repeatedly
- **Hours 16–20:** Track failing signs, document bugs
- **Hours 20–24:** Build presentation slides, record backup demo video, prepare 3-minute pitch

---

## Project Scope

> ⚠️ **Critical: Do NOT attempt full ISL vocabulary in 24 hours.**  
> Full ISL vocabulary is a research-level problem. A clean, reliable 15-sign demo beats a broken 100-sign system every time.

### Selected 15 Signs

| # | Sign | Priority | Reason |
|---|---|---|---|
| 1 | Hello | High | Universal greeting |
| 2 | Thank You | High | Common courtesy |
| 3 | Yes | High | Core response |
| 4 | No | High | Core response |
| 5 | Help | High | Emergency value |
| 6 | Water | High | Basic need |
| 7 | Food | High | Basic need |
| 8 | Doctor | High | Emergency value |
| 9 | Emergency | High | Assistive tech highlight |
| 10 | I / Me | Medium | Personal pronoun |
| 11 | You | Medium | Personal pronoun |
| 12 | Please | Medium | Common courtesy |
| 13 | Sorry | Medium | Common courtesy |
| 14 | Good | Medium | Positive feedback |
| 15 | Stop | Medium | Safety/control |

---

## Environment Setup

### Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install flask flask-socketio eventlet
pip install tensorflow numpy scikit-learn
pip install mediapipe opencv-python

# Verify
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import mediapipe; print('mediapipe ok')"
```

### Frontend Setup

```bash
npx create-react-app isl-frontend
cd isl-frontend

npm install socket.io-client
npm install @mediapipe/holistic @mediapipe/camera_utils @mediapipe/drawing_utils

# Start dev server
npm start
```

### requirements.txt

```
flask==2.3.2
flask-socketio==5.3.4
eventlet==0.33.3
tensorflow==2.12.0
numpy==1.24.3
scikit-learn==1.3.0
mediapipe==0.9.3
opencv-python==4.8.0.74
```

---

## Folder Structure

```
isl-gesture-system/
│
├── backend/
│   ├── app.py                    # Flask + SocketIO server
│   ├── requirements.txt
│   ├── model/
│   │   ├── isl_model.h5          # Trained BiLSTM model
│   │   └── label_encoder.pkl     # Scikit-learn label encoder
│   └── utils/
│       └── buffer.py             # Sliding window buffer helper
│
├── frontend/
│   ├── public/
│   │   ├── manifest.json         # PWA manifest
│   │   └── sw.js                 # Service worker
│   └── src/
│       ├── App.js
│       ├── socket.js             # Socket.IO connection singleton
│       └── components/
│           ├── Camera.js         # Webcam + MediaPipe component
│           └── Output.js         # Text display + TTS component
│
├── data_collection/
│   ├── collect.py                # Record gesture sequences
│   └── data/
│       ├── Hello/                # 30 .npy files, each shape (30, 258)
│       ├── Help/
│       └── ...
│
├── training/
│   ├── train.py                  # Model training script
│   └── evaluate.py               # Accuracy evaluation
│
└── README.md
```

---

## Implementation Guide

---

### Phase 1: Data Collection

> **Owner:** Member 1 + Support Member  
> **Duration:** Hours 1–6  
> **Output:** `data/` folder with 15 × 30 sequences

#### `data_collection/collect.py`

```python
import cv2
import numpy as np
import mediapipe as mp
import os

# ─── Config ───────────────────────────────────────────────
SIGNS = [
    'Hello', 'ThankYou', 'Yes', 'No', 'Help',
    'Water', 'Food', 'Doctor', 'Emergency', 'Me',
    'You', 'Please', 'Sorry', 'Good', 'Stop'
]
SEQUENCES = 30        # 30 sequences per sign
FRAMES = 30           # 30 frames per sequence
DATA_PATH = 'data'
# ──────────────────────────────────────────────────────────

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks(results):
    """
    Extract and flatten landmarks.
    Returns array of shape (258,)
    Order: [left_hand(63), right_hand(63), pose(132)]
    MUST match extractLandmarks() in Camera.js exactly.
    """
    # Left hand: 21 landmarks × 3 = 63
    lh = np.array([[r.x, r.y, r.z]
                   for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(63)

    # Right hand: 21 landmarks × 3 = 63
    rh = np.array([[r.x, r.y, r.z]
                   for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(63)

    # Pose: 33 landmarks × 4 (x, y, z, visibility) = 132
    pose = np.array([[r.x, r.y, r.z, r.visibility]
                     for r in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(132)

    return np.concatenate([lh, rh, pose])  # Total: 258


def create_folders():
    for sign in SIGNS:
        for seq in range(SEQUENCES):
            os.makedirs(os.path.join(DATA_PATH, sign, str(seq)), exist_ok=True)


def collect():
    create_folders()
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for sign in SIGNS:
            print(f"\n{'='*40}")
            print(f"  GET READY FOR: {sign}")
            print(f"{'='*40}")

            for seq in range(SEQUENCES):
                for frame_num in range(FRAMES):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img.flags.writeable = False
                    results = holistic.process(img)
                    img.flags.writeable = True
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    # Draw landmarks on screen
                    mp_drawing.draw_landmarks(
                        img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(
                        img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                    # Status text overlay
                    if frame_num == 0:
                        cv2.putText(img, 'STARTING COLLECTION', (80, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(
                            img,
                            f'Sign: {sign} | Seq: {seq+1}/{SEQUENCES}',
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                        )
                        cv2.imshow('Collection', img)
                        cv2.waitKey(1500)
                    else:
                        cv2.putText(
                            img,
                            f'Sign: {sign} | Seq: {seq+1}/{SEQUENCES} | Frame: {frame_num}',
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                        )

                    cv2.imshow('Collection', img)

                    # Save landmarks
                    landmarks = extract_landmarks(results)
                    npy_path = os.path.join(DATA_PATH, sign, str(seq), str(frame_num))
                    np.save(npy_path, landmarks)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Data collection complete!")
    print(f"   Total files: {15 * 30 * 30} .npy files across {len(SIGNS)} signs")


if __name__ == '__main__':
    collect()
```

> **After running:** `data/` should have 15 signs × 30 sequences × 30 frames = **13,500 `.npy` files**, each of shape `(258,)`.

---

### Phase 2: Model Training

> **Owner:** Member 1  
> **Duration:** Hours 6–10  
> **Output:** `isl_model.h5`, `label_encoder.pkl`

#### `training/train.py`

```python
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ─── Config ───────────────────────────────────────────────
DATA_PATH = '../data_collection/data'
SIGNS = [
    'Hello', 'ThankYou', 'Yes', 'No', 'Help',
    'Water', 'Food', 'Doctor', 'Emergency', 'Me',
    'You', 'Please', 'Sorry', 'Good', 'Stop'
]
SEQUENCES = 30
FRAMES = 30
FEATURES = 258
# ──────────────────────────────────────────────────────────


def load_data():
    X, y = [], []
    for sign in SIGNS:
        for seq in range(SEQUENCES):
            sequence = []
            for frame in range(FRAMES):
                path = os.path.join(DATA_PATH, sign, str(seq), f'{frame}.npy')
                frame_data = np.load(path)
                sequence.append(frame_data)
            X.append(sequence)
            y.append(sign)
    return np.array(X), np.array(y)


def train():
    print("Loading data...")
    X, y = load_data()
    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    # Save encoder immediately
    os.makedirs('../backend/model', exist_ok=True)
    with open('../backend/model/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print(f"✅ Label encoder saved. Classes: {list(le.classes_)}")

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42,
        stratify=y_encoded
    )

    # BiLSTM Model
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True),
                      input_shape=(FRAMES, FEATURES)),
        Dropout(0.2),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(len(SIGNS), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*40}")
    print(f"  Test Accuracy: {acc:.2%}")
    print(f"  Test Loss:     {loss:.4f}")
    print(f"{'='*40}")

    if acc < 0.90:
        print("⚠️  Accuracy below 90%. Consider:")
        print("   - Collecting more sequences (increase SEQUENCES to 50)")
        print("   - Checking for inconsistent gesture recording")
        print("   - Training longer (remove EarlyStopping)")

    # Confusion matrix
    y_pred = model.predict(X_test)
    y_pred_labels = le.inverse_transform(np.argmax(y_pred, axis=1))
    y_true_labels = le.inverse_transform(np.argmax(y_test, axis=1))
    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels))

    # Save model
    model.save('../backend/model/isl_model.h5')
    print("✅ Model saved to backend/model/isl_model.h5")

    return history


if __name__ == '__main__':
    train()
```

---

### Phase 3: Backend Server

> **Owner:** Member 3  
> **Duration:** Hours 1–10 (runs parallel with data collection)  
> **Output:** Working Flask-SocketIO server

#### `backend/app.py`

```python
from flask import Flask, request
from flask_socketio import SocketIO, emit
import numpy as np
import pickle
import os
from collections import deque

# ─── Init ─────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'isl_hackathon_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# ─── Constants ────────────────────────────────────────────
MODEL_PATH = 'model/isl_model.h5'
ENCODER_PATH = 'model/label_encoder.pkl'
CONFIDENCE_THRESHOLD = 0.85
BUFFER_SIZE = 30

# ─── State ────────────────────────────────────────────────
model = None
label_encoder = None
# Per-client buffers (keyed by session ID)
client_buffers = {}
client_last_prediction = {}


def load_model():
    global model, label_encoder
    try:
        from tensorflow.keras.models import load_model as keras_load
        model = keras_load(MODEL_PATH)
        with open(ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        print("✅ Model loaded successfully")
        print(f"   Classes: {list(label_encoder.classes_)}")
    except Exception as e:
        print(f"⚠️  Model not found, using dummy predictions: {e}")
        print("   Drop isl_model.h5 into backend/model/ and restart")


# ─── Socket Events ────────────────────────────────────────
@socketio.on('connect')
def handle_connect():
    sid = request.sid
    client_buffers[sid] = deque(maxlen=BUFFER_SIZE)
    client_last_prediction[sid] = None
    print(f"Client connected: {sid}")
    emit('status', {'message': 'Connected to ISL server', 'model_ready': model is not None})


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    client_buffers.pop(sid, None)
    client_last_prediction.pop(sid, None)
    print(f"Client disconnected: {sid}")


@socketio.on('landmarks')
def handle_landmarks(data):
    sid = request.sid
    landmarks = data.get('landmarks')

    if landmarks is None or len(landmarks) != 258:
        emit('error', {'message': f'Invalid landmark shape. Expected 258, got {len(landmarks) if landmarks else 0}'})
        return

    buffer = client_buffers.get(sid)
    if buffer is None:
        return

    buffer.append(landmarks)

    # Report buffer status
    emit('buffer_status', {
        'filled': len(buffer),
        'total': BUFFER_SIZE,
        'percent': int((len(buffer) / BUFFER_SIZE) * 100)
    })

    if len(buffer) < BUFFER_SIZE:
        return

    # ── Dummy prediction (model not yet loaded) ──
    if model is None:
        import random
        dummy_signs = ['Hello', 'Help', 'Water', 'Doctor', 'Emergency']
        emit('prediction', {
            'text': random.choice(dummy_signs),
            'confidence': 0.95,
            'status': 'dummy — model not loaded yet'
        })
        return

    # ── Real prediction ──
    try:
        sequence = np.array(list(buffer))           # (30, 258)
        sequence_input = np.expand_dims(sequence, 0)  # (1, 30, 258)
        predictions = model.predict(sequence_input, verbose=0)[0]
        confidence = float(np.max(predictions))
        class_idx = int(np.argmax(predictions))
        label = label_encoder.inverse_transform([class_idx])[0]

        last = client_last_prediction.get(sid)

        # Only emit if confident AND different from last prediction
        if confidence >= CONFIDENCE_THRESHOLD and label != last:
            client_last_prediction[sid] = label
            emit('prediction', {
                'text': label,
                'confidence': round(confidence, 3)
            })
            # Clear buffer after prediction to prepare for next sign
            buffer.clear()

    except Exception as e:
        print(f"Prediction error: {e}")
        emit('error', {'message': str(e)})


@socketio.on('reset')
def handle_reset():
    sid = request.sid
    if sid in client_buffers:
        client_buffers[sid].clear()
    client_last_prediction[sid] = None
    emit('status', {'message': 'Buffer reset'})


# ─── Run ──────────────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    print("🚀 ISL server starting on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
```

---

### Phase 4: Frontend App

> **Owner:** Member 2  
> **Duration:** Hours 1–10 (parallel with backend)  
> **Output:** React PWA with camera, MediaPipe, WebSocket, TTS

#### `frontend/src/socket.js`

```javascript
import { io } from 'socket.io-client';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5000';

const socket = io(BACKEND_URL, {
  transports: ['websocket'],
  reconnection: true,
  reconnectionDelay: 1000,
  reconnectionAttempts: 10,
});

export default socket;
```

#### `frontend/src/components/Camera.js`

```javascript
import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Holistic } from '@mediapipe/holistic';
import { Camera as MediaPipeCamera } from '@mediapipe/camera_utils';
import socket from '../socket';

export default function Camera({ onPrediction }) {
  const videoRef = useRef(null);
  const [bufferPercent, setBufferPercent] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const [modelReady, setModelReady] = useState(false);

  useEffect(() => {
    socket.on('connect', () => setIsConnected(true));
    socket.on('disconnect', () => setIsConnected(false));
    socket.on('status', (data) => {
      if (data.model_ready !== undefined) setModelReady(data.model_ready);
    });
    socket.on('buffer_status', ({ percent }) => setBufferPercent(percent));
    socket.on('prediction', (data) => {
      onPrediction(data);
      setBufferPercent(0);
    });
    socket.on('error', (err) => console.error('Server error:', err.message));

    return () => {
      socket.off('connect');
      socket.off('disconnect');
      socket.off('status');
      socket.off('buffer_status');
      socket.off('prediction');
      socket.off('error');
    };
  }, [onPrediction]);

  useEffect(() => {
    const holistic = new Holistic({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    holistic.onResults((results) => {
      const landmarks = extractLandmarks(results);
      // Validate shape before sending
      if (landmarks.length === 258) {
        socket.emit('landmarks', { landmarks });
      }
    });

    if (videoRef.current) {
      const camera = new MediaPipeCamera(videoRef.current, {
        onFrame: async () => {
          await holistic.send({ image: videoRef.current });
        },
        width: 640,
        height: 480,
      });
      camera.start();
    }

    return () => holistic.close();
  }, []);

  return (
    <div className="camera-container">
      <video ref={videoRef} className="camera-feed" autoPlay playsInline muted />

      <div className="status-bar">
        <span className={`dot ${isConnected ? 'green' : 'red'}`} />
        {isConnected ? (modelReady ? '🟢 Model Ready' : '🟡 Model Loading...') : '🔴 Disconnected'}
      </div>

      <div className="buffer-bar-container">
        <div
          className="buffer-bar-fill"
          style={{ width: `${bufferPercent}%` }}
        />
        <span className="buffer-label">
          {bufferPercent < 100 ? `Capturing... ${bufferPercent}%` : 'Predicting...'}
        </span>
      </div>
    </div>
  );
}

// ─── CRITICAL: Must match extract_landmarks() in collect.py exactly ───
// Order: [left_hand(63), right_hand(63), pose(132)] = 258 total
function extractLandmarks(results) {
  // Left hand: 21 landmarks × 3 values (x, y, z) = 63
  const lh = results.leftHandLandmarks
    ? results.leftHandLandmarks.flatMap(l => [l.x, l.y, l.z])
    : new Array(63).fill(0);

  // Right hand: 21 landmarks × 3 values (x, y, z) = 63
  const rh = results.rightHandLandmarks
    ? results.rightHandLandmarks.flatMap(l => [l.x, l.y, l.z])
    : new Array(63).fill(0);

  // Pose: 33 landmarks × 4 values (x, y, z, visibility) = 132
  const pose = results.poseLandmarks
    ? results.poseLandmarks.flatMap(l => [l.x, l.y, l.z, l.visibility])
    : new Array(132).fill(0);

  return [...lh, ...rh, ...pose];  // Total: 258
}
```

#### `frontend/src/components/Output.js`

```javascript
import React, { useState, useEffect, useCallback } from 'react';

export default function Output({ prediction }) {
  const [sentence, setSentence] = useState([]);
  const [lastSpoken, setLastSpoken] = useState('');

  const speak = useCallback((text) => {
    if (!('speechSynthesis' in window)) return;
    if (text === lastSpoken) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-IN';
    utterance.rate = 0.9;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
    setLastSpoken(text);
  }, [lastSpoken]);

  useEffect(() => {
    if (prediction?.text) {
      setSentence(prev => [...prev, prediction.text]);
      speak(prediction.text);
    }
  }, [prediction, speak]);

  const clearSentence = () => {
    setSentence([]);
    setLastSpoken('');
    window.speechSynthesis.cancel();
  };

  const speakFull = () => {
    if (sentence.length > 0) {
      const full = sentence.join(' ');
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(full);
      utterance.lang = 'en-IN';
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <div className="output-container">

      {/* Current prediction */}
      <div className="current-sign">
        {prediction?.text ? (
          <>
            <div className="sign-text">{prediction.text}</div>
            <div className="confidence">
              {(prediction.confidence * 100).toFixed(0)}% confidence
            </div>
          </>
        ) : (
          <div className="placeholder">Perform a sign to begin...</div>
        )}
      </div>

      {/* Sentence buffer */}
      <div className="sentence-buffer">
        {sentence.length > 0 ? (
          sentence.map((word, i) => (
            <span key={i} className="word-chip">{word}</span>
          ))
        ) : (
          <span className="placeholder-small">Words will appear here as you sign</span>
        )}
      </div>

      {/* Controls */}
      <div className="controls">
        <button
          className="btn-primary"
          onClick={speakFull}
          disabled={sentence.length === 0}
        >
          🔊 Speak Sentence
        </button>
        <button
          className="btn-secondary"
          onClick={clearSentence}
          disabled={sentence.length === 0}
        >
          🗑️ Clear
        </button>
      </div>

    </div>
  );
}
```

#### `frontend/src/App.js`

```javascript
import React, { useState } from 'react';
import Camera from './components/Camera';
import Output from './components/Output';
import './App.css';

export default function App() {
  const [prediction, setPrediction] = useState(null);

  return (
    <div className="app">
      <header className="app-header">
        <h1>🤟 ISL Translator</h1>
        <p>Indian Sign Language → Real-time Text & Speech</p>
      </header>

      <main className="app-main">
        <section className="camera-section">
          <h2>Camera</h2>
          <Camera onPrediction={setPrediction} />
        </section>

        <section className="output-section">
          <h2>Translation</h2>
          <Output prediction={prediction} />
        </section>
      </main>

      <footer>
        <small>Powered by MediaPipe Holistic + BiLSTM | Built for Hackathon 2024</small>
      </footer>
    </div>
  );
}
```

#### `frontend/public/manifest.json`

```json
{
  "name": "ISL Gesture Translator",
  "short_name": "ISL Translator",
  "description": "Real-time Indian Sign Language detection and translation",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#0f0f23",
  "theme_color": "#6c63ff",
  "orientation": "portrait",
  "icons": [
    {
      "src": "logo192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "logo512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}
```

---

### Phase 5: Integration

> **Owner:** All members  
> **Duration:** Hours 10–16  
> **Goal:** Full pipeline working end-to-end

#### Integration Checklist

- [ ] Member 3 starts Flask: `python backend/app.py`
- [ ] Member 2 starts React: `npm start`
- [ ] Green dot appears in UI (WebSocket connected)
- [ ] Landmark emits logged in Flask terminal
- [ ] Buffer bar fills progressively in UI
- [ ] Dummy prediction returns and shows in UI
- [ ] Member 1 drops real model into `backend/model/`
- [ ] Flask restarted, "Model Ready" shows in UI
- [ ] All 15 signs tested at least once
- [ ] TTS fires correctly on each prediction
- [ ] Sentence buffer accumulates multiple signs
- [ ] Confidence threshold filters false positives

#### Common Integration Bugs & Fixes

**Bug: 0 landmarks received by backend**  
Cause: Array shape mismatch  
Fix: Print `len(landmarks)` in Flask terminal and `console.log(landmarks.length)` in browser. Both must be 258.

**Bug: CORS error in browser console**  
Fix: Ensure `cors_allowed_origins="*"` in Flask-SocketIO init. Also check HTTPS/HTTP mismatch.

**Bug: Prediction fires repeatedly for same sign**  
Fix: The `last_prediction` debounce in `app.py` handles this. Verify `buffer.clear()` is called after each prediction.

**Bug: Very low confidence scores (< 0.5)**  
Cause: Landmark ordering mismatch between `collect.py` and `Camera.js`  
Fix: Verify BOTH use the order `[left_hand, right_hand, pose]` — this is the single most common integration bug.

**Bug: Model loads but accuracy in production is much lower than training**  
Cause: The person doing the live demo is different from the person who recorded data  
Fix: Record a few sequences with the demo person's gestures and add to training data.

---

## 24-Hour Timeline

```
Hour 00–01  │ ALL:    Setup — GitHub, npm/pip installs, folder structure
Hour 01–03  │ M1+S:  Data collection (first 5 signs: Hello, ThankYou, Yes, No, Help)
            │ M2:    React setup + webcam access via getUserMedia
            │ M3:    Flask + SocketIO boilerplate + dummy prediction endpoint
Hour 03–06  │ M1+S:  Data collection (remaining 10 signs)
            │ M2:    MediaPipe Holistic integration + landmark extraction
            │ M3:    30-frame buffer + WebSocket event handlers
Hour 06–08  │ M1:    Model training begins (BiLSTM, 100 epochs)
            │ M2:    Socket.IO client — connect + emit + receive
            │ M3:    Wire dummy prediction to frontend, verify full loop works
Hour 08–10  │ M1:    Evaluate model, retrain if accuracy < 90%
            │ M2:    Output.js UI + TTS integration
            │ M3:    Replace dummy with real model, test inference
Hour 10–13  │ ALL:   Integration — connect all pieces, first full test
Hour 13–16  │ ALL:   Debug pipeline — landmark format, CORS, confidence tuning
            │ S:     Log all sign test results (pass/fail per sign)
Hour 16–18  │ M1:    Retrain/fine-tune underperforming signs
            │ M2:    UI polish — responsive layout, App.css
            │ M3:    Confidence threshold tuning
Hour 18–20  │ ALL:   Full system test — all 15 signs, 5 times each
            │ S:     Complete testing checklist, record issues
Hour 20–22  │ M3:    Cloudflare Tunnel setup + deployment
            │ M2:    Update REACT_APP_BACKEND_URL + rebuild frontend
            │ S:     Build presentation slides
Hour 22–23  │ ALL:   Rehearse 3-minute demo script
            │ S:     Record backup demo video
Hour 23–24  │ Buffer: Fix critical bugs only
            │ 🚫 NO NEW FEATURES after Hour 20
```

---

## Deployment Guide

### Local Demo (Default for Development)

```bash
# Terminal 1 — Backend
cd backend
source venv/bin/activate
python app.py
# Server at: http://localhost:5000

# Terminal 2 — Frontend
cd frontend
npm start
# App at: http://localhost:3000
```

### Public Demo via Cloudflare Tunnel

```bash
# Download cloudflared (Linux)
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -o cloudflared && chmod +x cloudflared

# Start tunnel pointing to Flask
./cloudflared tunnel --url http://localhost:5000
# You'll get: https://some-random-name.trycloudflare.com
```

Update frontend environment:
```bash
# frontend/.env
REACT_APP_BACKEND_URL=https://some-random-name.trycloudflare.com
```

Rebuild and serve:
```bash
npm run build
npx serve -s build -l 3000
```

> ⚠️ **Critical Warning — Free Tier Services (Render/Railway):**  
> Free tiers sleep after 15 minutes of inactivity. Cold start takes 30–60 seconds.  
> This will **kill your live demo**. Use Cloudflare Tunnel on your own machine instead.

---

## Testing Checklist

### Pre-Integration (Hour 10)
- [ ] `collect.py` output shape is `(30, 258)` per sequence: `np.load('data/Hello/0/0.npy').shape`
- [ ] Model accuracy ≥ 90% on test set
- [ ] Flask starts without import errors
- [ ] Dummy prediction visible in browser
- [ ] MediaPipe running — landmarks logged in browser console

### Post-Integration (Hour 16)
- [ ] WebSocket connects on page load (green indicator)
- [ ] Buffer fills progressively (bar reaches 100%)
- [ ] Predictions appear after ~1 second of signing
- [ ] TTS speaks each prediction aloud
- [ ] Sentence buffer builds up multiple words
- [ ] Clear button resets everything
- [ ] Confidence score displayed
- [ ] Works with a different person than who collected data

### Full Sign Test (Hour 18–20)

| Sign | 1 | 2 | 3 | 4 | 5 | Pass Rate |
|---|---|---|---|---|---|---|
| Hello | | | | | | /5 |
| ThankYou | | | | | | /5 |
| Yes | | | | | | /5 |
| No | | | | | | /5 |
| Help | | | | | | /5 |
| Water | | | | | | /5 |
| Food | | | | | | /5 |
| Doctor | | | | | | /5 |
| Emergency | | | | | | /5 |
| Me | | | | | | /5 |
| You | | | | | | /5 |
| Please | | | | | | /5 |
| Sorry | | | | | | /5 |
| Good | | | | | | /5 |
| Stop | | | | | | /5 |

**Target:** 12/15 signs passing at 4+/5 before Hour 20.

---

## Presentation Strategy

### 3-Minute Demo Script

**[0:00–0:15] Hook**  
"63 million Indians are deaf. They communicate in Indian Sign Language — but almost no one around them understands it. We built a system that bridges that gap in real-time, using nothing but a browser."

**[0:15–1:45] Live Demo**  
Show 5 high-impact signs: Help → Doctor → Emergency → Water → Hello  
Let the TTS speak each one aloud. Show the sentence buffer building up.  
Say: "No special hardware. No app install. Just a camera."

**[1:45–2:15] Architecture (show diagram)**  
"Instead of sending raw video frames to the server — which would be slow and a privacy risk — we extract just 258 landmark coordinates on-device and stream those over WebSocket. The backend runs a BiLSTM sequence model on a 30-frame sliding window."

**[2:15–2:45] Impact + Future**  
"Today: 15 signs. Our roadmap includes 100+ signs, offline mode via TFLite, and emergency communication presets for hospitals and police stations."

**[2:45–3:00] Close**  
"This is what accessible AI looks like — real-time, private, and available to anyone with a phone."

### What Judges Notice
- Real-time demo with TTS output — proves the tech works
- Landmark-based architecture — shows depth of ML knowledge
- Assistive tech framing — strong social impact narrative
- Confidence score display — shows production-thinking
- Clean UI — signals you can ship, not just prototype

---

## Troubleshooting

### Model Won't Load
```bash
# Check TF version compatibility
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('backend/model/isl_model.h5')
print('Model loaded:', model.input_shape)
"
# If error: training and inference must use the same TF version
```

### WebSocket Not Connecting
```bash
# Check server is running
curl http://localhost:5000

# Check CORS in app.py
# Must have: socketio = SocketIO(app, cors_allowed_origins="*")

# Check browser console for:
# - Mixed content errors (HTTP vs HTTPS)
# - Connection refused (wrong port)
```

### Low Accuracy (< 85%)
```
1. Inconsistent recording: same lighting, same distance, multiple angles
2. Too few sequences: increase SEQUENCES = 30 → 50
3. Similar signs: check confusion matrix for worst pairs
4. Wrong landmark order: verify extract_landmarks() == extractLandmarks()
5. Train longer: increase EarlyStopping patience = 15 → 25
```

### High Latency (> 2 seconds to predict)
```
1. Reduce buffer size: BUFFER_SIZE = 30 → 20
2. Check model size: model.count_params() — should be < 500K params
3. Add prediction on partially-filled buffer at 80% confidence
4. Move to GPU if available: export CUDA_VISIBLE_DEVICES=0
```

### Camera Not Working on Mobile
```
Cause: Browsers require HTTPS for camera access on non-localhost domains
Fix: Serve frontend over HTTPS (Cloudflare Tunnel provides this automatically)
Fix: Test on localhost during development (HTTP allowed for localhost)
```

---

## Future Roadmap

### Phase 2 — Expanded Vocabulary (1–2 months post-hackathon)
- Grow from 15 to 100+ ISL signs
- Community data collection app for diverse recordings
- Per-sign accuracy dashboard with confusion matrix
- Transfer learning from ASL pre-trained models

### Phase 3 — Mobile Production (3–6 months)
- React Native with `react-native-vision-camera` for native frame processing
- TFLite model conversion for on-device inference (full offline)
- Sub-300ms end-to-end latency
- Google Play Store submission

### Phase 4 — Real-World Deployment (6–12 months)
- Emergency mode: pre-configured sign sets for hospital, police, fire
- Multi-language TTS (Hindi, Tamil, Telugu, regional languages)
- Sentence-level NLP post-processing using small language model
- Integration with hospital patient communication systems

---

## Quick Reference Card

```
START BACKEND:     cd backend && python app.py
START FRONTEND:    cd frontend && npm start
COLLECT DATA:      cd data_collection && python collect.py
TRAIN MODEL:       cd training && python train.py
TUNNEL DEPLOY:     ./cloudflared tunnel --url http://localhost:5000

LANDMARK SHAPE:    (258,) per frame
                   [left_hand(63) + right_hand(63) + pose(132)]
BUFFER SIZE:       30 frames ≈ 1 second at 30fps
CONFIDENCE GATE:   0.85 (85%) — only emit if above this
WEBSOCKET EVENTS:  emit → 'landmarks' | listen → 'prediction', 'buffer_status'
MODEL FILE:        backend/model/isl_model.h5
ENCODER FILE:      backend/model/label_encoder.pkl
```

---

*ISL Gesture Detection System | Hackathon 2024 | Team of 3 + 1 Support*
