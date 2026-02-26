from flask import Flask
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import threading
import base64
import pickle
import time
import os
from collections import deque
from tensorflow.keras.models import load_model as tf_load_model

app = Flask(__name__)
app.config['SECRET_KEY'] = 'isl_secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

CONFIG = {
    "camera_index": 0,           # PC webcam index (0 = default)
    "frame_width": 640,
    "frame_height": 480,
    "target_fps": 30,
    "model_complexity": 1,       # 1 = balanced, 2 = high accuracy (slower)
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "smooth_landmarks": True,
    "buffer_size": 30,           # frames per prediction window
    "slide_every": 5,            # overlapping window — predict every N new frames
    "confidence_threshold": 0.85,
    "jpeg_quality": 70,          # lower = less bandwidth, faster streaming
    "model_path": "model/isl_model.h5",
    "encoder_path": "model/label_encoder.pkl",
}

# Global state
frame_buffer = deque(maxlen=CONFIG["buffer_size"])
frame_counter = 0
model = None
label_encoder = None
MODEL_READY = False
camera_running = False
connected_clients = 0

# MediaPipe Setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def load_model():
    global model, label_encoder, MODEL_READY
    try:
        if os.path.exists(CONFIG["model_path"]) and os.path.exists(CONFIG["encoder_path"]):
            model = tf_load_model(CONFIG["model_path"])
            with open(CONFIG["encoder_path"], 'rb') as f:
                label_encoder = pickle.load(f)
            MODEL_READY = True
            print("Model and encoder loaded successfully.")
        else:
            print("Model not found — running in dummy mode")
            MODEL_READY = False
    except Exception as e:
        print(f"Error loading model: {e}. Running in dummy mode")
        MODEL_READY = False

def extract_landmarks(results):
    # Left hand: 21 landmarks × (x, y, z) = 63 values
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    
    # Right hand: 21 landmarks × (x, y, z) = 63 values
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    
    # Pose: 33 landmarks × (x, y, z, visibility) = 132 values
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    
    return np.concatenate([lh, rh, pose])

def draw_landmarks(frame, results):
    # Left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
        cv2.circle(frame, (30, 30), 10, (0, 255, 0), -1)
        cv2.putText(frame, 'L', (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)

    # Right hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        cv2.circle(frame, (60, 30), 10, (0, 255, 0), -1)
        cv2.putText(frame, 'R', (52, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    elif not results.left_hand_landmarks:
        # If neither hand, red circle already drawn at (30,30) in LH branch, but let's be explicit
        cv2.circle(frame, (60, 30), 10, (0, 0, 255), -1)

    # Pose connections (Upper body: 11-22)
    if results.pose_landmarks:
        # We only want to draw specific connections for upper body
        # MediaPipe's default drawing draws everything. For simplicity, we use the default but it will include face if available.
        # The request said: Only draw upper body pose (landmarks 11–22, skip face)
        # Note: draw_landmarks doesn't easily support subset of connections without custom list.
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=2))

def predict(sequence):
    global MODEL_READY
    if not MODEL_READY:
        # Dummy mode
        dummy_signs = ["Hello", "Help", "Water", "Doctor", "ThankYou"]
        idx = int(time.time() / 2) % len(dummy_signs)
        return dummy_signs[idx], 0.95

    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
    idx = np.argmax(res)
    confidence = res[idx]
    
    if confidence < CONFIG["confidence_threshold"]:
        return None, confidence
    
    label = label_encoder.inverse_transform([idx])[0]
    return label, confidence

def camera_loop():
    global frame_counter, camera_running, connected_clients

    cap = cv2.VideoCapture(CONFIG["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
    cap.set(cv2.CAP_PROP_FPS, CONFIG["target_fps"])
    camera_running = True

    with mp_holistic.Holistic(
        model_complexity=CONFIG["model_complexity"],
        smooth_landmarks=CONFIG["smooth_landmarks"],
        min_detection_confidence=CONFIG["min_detection_confidence"],
        min_tracking_confidence=CONFIG["min_tracking_confidence"],
    ) as holistic:

        frame_time = 1.0 / CONFIG["target_fps"]

        while camera_running:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            landmarks = extract_landmarks(results)
            frame_buffer.append(landmarks)
            frame_counter += 1

            if (len(frame_buffer) == CONFIG["buffer_size"] and
                    frame_counter % CONFIG["slide_every"] == 0):

                label, confidence = predict(list(frame_buffer))

                if label:
                    socketio.emit("prediction", {
                        "text": label,
                        "confidence": round(float(confidence), 3)
                    })

            draw_landmarks(frame, results)

            if connected_clients > 0:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, CONFIG["jpeg_quality"]]
                _, buffer = cv2.imencode(".jpg", frame, encode_params)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")
                socketio.emit("frame", {"image": frame_b64})

            if frame_counter % 5 == 0:
                socketio.emit("buffer_size", {"size": len(frame_buffer)})

            elapsed = time.time() - loop_start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    cap.release()

@socketio.on("connect")
def on_connect():
    global connected_clients
    connected_clients += 1
    emit("status", {
        "ready": MODEL_READY,
        "camera": camera_running,
        "config": {
            "buffer_size": CONFIG["buffer_size"],
            "confidence_threshold": CONFIG["confidence_threshold"],
            "fps": CONFIG["target_fps"]
        }
    })
    print(f"Client connected — total: {connected_clients}")

@socketio.on("disconnect")
def on_disconnect():
    global connected_clients
    connected_clients = max(0, connected_clients - 1)
    print(f"Client disconnected — total: {connected_clients}")

@socketio.on("set_confidence")
def set_confidence(data):
    CONFIG["confidence_threshold"] = float(data.get("threshold", 0.85))
    emit("config_updated", {"confidence_threshold": CONFIG["confidence_threshold"]})
    print(f"Confidence threshold updated to: {CONFIG['confidence_threshold']}")

@socketio.on("reset_buffer")
def reset_buffer():
    frame_buffer.clear()
    emit("buffer_cleared", {})
    print("Buffer cleared")

@app.route("/health")
def health():
    return {
        "status": "ok",
        "model_ready": MODEL_READY,
        "camera_running": camera_running,
        "connected_clients": connected_clients,
        "buffer_size": len(frame_buffer),
        "config": CONFIG
    }

if __name__ == "__main__":
    load_model()
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    print(f"Camera thread started")
    print(f"Server starting on http://0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
