import os
import random
import numpy as np
import eventlet
from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
from collections import deque
import tensorflow as tf
import pickle
from utils.buffer import LandmarkBuffer

# Set async mode to eventlet
eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Config
CONFIDENCE_THRESHOLD = 0.85
BUFFER_SIZE = 30
MODEL_PATH = os.path.join("model", "isl_model.h5")
LABEL_ENCODER_PATH = os.path.join("model", "label_encoder.pkl")

# Global State
frame_buffer = LandmarkBuffer(maxlen=BUFFER_SIZE)
MODEL_READY = False
model = None
label_encoder = None

# 15 Signs for Dummy Mode
SIGNS = [
    "Hello", "ThankYou", "Yes", "No", "Help", 
    "Water", "Food", "Doctor", "Emergency", "IMe", 
    "You", "Please", "Sorry", "Good", "Stop"
]

def load_ml_model():
    global MODEL_READY, model, label_encoder
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(LABEL_ENCODER_PATH, "rb") as f:
                label_encoder = pickle.load(f)
            MODEL_READY = True
            print("Model and Label Encoder loaded successfully.")
        else:
            print("Model files not found. Running in DUMMY MODE.")
    except Exception as e:
        print(f"Error loading model: {e}. Falling back to DUMMY MODE.")

load_ml_model()

def run_inference(sequence):
    """
    Takes a sequence of 30 frames and returns predicted label and confidence.
    """
    if not MODEL_READY:
        # Dummy Mode: Random sign with high confidence
        return random.choice(SIGNS), 0.99

    try:
        # sequence is (30, 258)
        arr = np.expand_dims(np.array(sequence), axis=0) # (1, 30, 258)
        probs = model.predict(arr, verbose=0)[0]
        max_idx = np.argmax(probs)
        confidence = float(probs[max_idx])
        
        if confidence < CONFIDENCE_THRESHOLD:
            return None, confidence
            
        label = label_encoder.inverse_transform([max_idx])[0]
        return label, confidence
    except Exception as e:
        print(f"Inference error: {e}")
        return None, 0.0

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_ready": MODEL_READY,
        "buffer_size": frame_buffer.size()
    })

@socketio.on("connect")
def handle_connect():
    emit("status", {"ready": MODEL_READY})
    print("Client connected")

@socketio.on("disconnect")
def handle_disconnect():
    frame_buffer.clear()
    print("Client disconnected, buffer cleared")

@socketio.on("landmarks")
def handle_landmarks(data):
    try:
        landmarks = data.get("landmarks", [])
        
        # Validate shape
        if len(landmarks) != 258:
            emit("error", {"message": f"Invalid landmark shape. Expected 258, got {len(landmarks)}"})
            return

        frame_buffer.add(np.array(landmarks, dtype=np.float32))
        
        # Emit buffer size for status bar updates
        emit("buffer_size", {"size": frame_buffer.size()})

        if frame_buffer.is_ready():
            sequence = frame_buffer.get_sequence()
            label, confidence = run_inference(sequence)
            
            if label:
                emit("prediction", {
                    "text": label,
                    "confidence": confidence
                })
            
    except Exception as e:
        print(f"Error handling landmarks: {e}")
        emit("error", {"message": "Internal server error during landmark processing"})

@socketio.on("reset_buffer")
def handle_reset_buffer():
    frame_buffer.clear()
    emit("buffer_size", {"size": 0})
    print("Buffer reset requested")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
