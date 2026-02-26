# 👥 ISL Project — Team Task Assignment
### 24-Hour Hackathon | Indian Sign Language Real-Time Translator

---

## ⚠️ Ground Rules Before Starting

Read these before touching any code.

1. **Never block each other.** If you're waiting for someone else's output, work on something else.
2. **Agree on landmark format in Hour 1.** Member 1 and Member 2 must both sign off on the 258-value array order. Everything breaks if this mismatches.
3. **Member 3 must have a dummy endpoint live by Hour 3.** Member 2 should never be waiting for the real model to test the frontend.
4. **Feature freeze at Hour 20.** Nothing new after that. Only fixing what's broken.
5. **Record backup demo video at Hour 22.** No exceptions.
6. **Commit to your branch every 2 hours.** If your laptop dies, your work should survive.

---

## 📋 Landmark Format Agreement (Sign Off Before Hour 1 Ends)

Both Member 1 (Python) and Member 2 (JavaScript) must extract landmarks in this exact order:

```
Index   0 –  62  →  Left Hand   21 landmarks × (x, y, z)           = 63 values
Index  63 – 125  →  Right Hand  21 landmarks × (x, y, z)           = 63 values
Index 126 – 257  →  Pose        33 landmarks × (x, y, z, visibility)= 132 values
─────────────────────────────────────────────────────────────────────
TOTAL = 258 values per frame
```

When a hand is **not detected** → fill with **zeros**, do NOT skip the frame.

**Member 1 sign:** _______________________ **Member 2 sign:** _______________________

---

---

# 🧠 Member 1 — AI/ML Engineer

> Your entire focus is data collection, model training, and inference. Do not touch frontend or backend server code.

---

### ✅ Hour 0–1 | Environment Setup

- [ ] Clone repo, checkout `model` branch
- [ ] Install dependencies:
  ```bash
  pip install mediapipe tensorflow opencv-python scikit-learn numpy
  ```
- [ ] Create folder structure for all 15 signs under `data/`:
  ```
  data/
  ├── Hello/
  ├── ThankYou/
  ├── Yes/
  ├── No/
  ├── Help/
  ├── Water/
  ├── Food/
  ├── Doctor/
  ├── Emergency/
  ├── IMe/
  ├── You/
  ├── Please/
  ├── Sorry/
  ├── Good/
  └── Stop/
  ```
- [ ] **Sit with Member 2 for 15 minutes** — agree on the 258-value landmark format, write it down, both sign off

---

### ✅ Hour 1–3 | Data Collection

- [ ] Run `collect.py` with Support Member performing gestures in front of camera
- [ ] Record **30 sequences × 30 frames** for all 15 signs
- [ ] After every 3 signs, verify shape:
  ```python
  import numpy as np
  arr = np.load("data/Hello/0/0.npy")
  print(arr.shape)  # Must print (258,)
  ```
- [ ] If shape is wrong, stop and fix `extract_landmarks()` before continuing
- [ ] Lighting check: hands must be clearly visible, no backlight
- [ ] Have Support Member and at least one other team member perform some sequences for natural variation

---

### ✅ Hour 3–6 | Model Training

- [ ] Run `train.py`
- [ ] Monitor training — watch for validation accuracy climbing above 85%
- [ ] If accuracy is stuck below 80% after 30 epochs, stop and re-collect data for the failing signs
- [ ] Target: **>90% test accuracy**
- [ ] Save outputs to `backend/model/`:
  ```
  backend/model/isl_model.h5
  backend/model/label_encoder.pkl
  ```
- [ ] While training is running, write the inference wrapper (next task)

---

### ✅ Hour 6–8 | Inference Wrapper

- [ ] Write `backend/utils/inference.py`:
  ```python
  import numpy as np
  import pickle
  from tensorflow.keras.models import load_model

  model = load_model("model/isl_model.h5")
  with open("model/label_encoder.pkl", "rb") as f:
      le = pickle.load(f)

  def predict(sequence, threshold=0.85):
      arr = np.expand_dims(np.array(sequence), axis=0)  # (1, 30, 258)
      probs = model.predict(arr, verbose=0)[0]
      confidence = float(np.max(probs))
      if confidence < threshold:
          return None, confidence
      label = le.inverse_transform([np.argmax(probs)])[0]
      return label, confidence
  ```
- [ ] Test inference standalone:
  ```python
  # Load a real 30-frame sequence from data/ and test prediction
  sequence = [np.load(f"data/Hello/0/{i}.npy") for i in range(30)]
  label, conf = predict(sequence)
  print(label, conf)  # Should print "Hello" with high confidence
  ```
- [ ] Hand off `isl_model.h5` and `label_encoder.pkl` to Member 3

---

### ✅ Hour 8–16 | Integration Support + Accuracy Tuning

- [ ] Notify Member 3 that model is ready, help them plug it into `app.py`
- [ ] During integration testing (Hour 10–14), monitor which signs fail most often
- [ ] For any sign with accuracy < 80%:
  - Collect 10–15 more sequences
  - Retrain only if more than 3 signs are failing (full retrain)
  - Otherwise, adjust confidence threshold per-class if needed
- [ ] Run a confusion matrix after integration to identify systematic misclassifications:
  ```python
  from sklearn.metrics import confusion_matrix, classification_report
  # See evaluate.py for full script
  ```

---

### ✅ Hour 16–20 | Final Model Polish

- [ ] Test all 15 signs personally at least 5 times each
- [ ] Ensure these high-priority signs are reliable (judges will see these):
  - `Hello`, `Help`, `ThankYou`, `Doctor`, `Emergency`
- [ ] Document final model accuracy in a comment at top of `train.py`
- [ ] Help Support Member with technical content in slides if needed

---

### ✅ Hour 20–24 | Buffer + Demo Prep

- [ ] Available for emergency model fixes if demo is broken
- [ ] Rehearse the architecture explanation section of the demo (30 seconds)
- [ ] Rest if pipeline is stable

---

### 📁 Your Deliverables

| File | Location | Due |
|---|---|---|
| `collect.py` | `data_collection/` | Hour 3 |
| `train.py` | `training/` | Hour 6 |
| `isl_model.h5` | `backend/model/` | Hour 8 |
| `label_encoder.pkl` | `backend/model/` | Hour 8 |
| `inference.py` | `backend/utils/` | Hour 8 |

---

---

# 💻 Member 2 — Frontend Developer

> Your entire focus is the React app, MediaPipe in the browser, WebSocket client, UI, and PWA setup. Do not touch the Python backend.

---

### ✅ Hour 0–1 | Environment Setup

- [ ] Clone repo, checkout `frontend` branch
- [ ] Create React app:
  ```bash
  npx create-react-app frontend
  cd frontend
  npm install @mediapipe/holistic @mediapipe/camera_utils socket.io-client
  ```
- [ ] **Sit with Member 1 for 15 minutes** — agree on the 258-value landmark format, write it down, both sign off
- [ ] Create component files:
  ```
  src/
  ├── App.js
  ├── socket.js
  ├── App.css
  └── components/
      ├── Camera.js
      └── Output.js
  ```

---

### ✅ Hour 1–4 | Camera + MediaPipe Integration

- [ ] Set up `socket.js`:
  ```javascript
  import { io } from "socket.io-client";
  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";
  export const socket = io(BACKEND_URL, { transports: ["websocket"] });
  ```
- [ ] Build `Camera.js` with MediaPipe Holistic:
  - Access webcam via `getUserMedia`
  - Load `@mediapipe/holistic` and process each frame
  - Extract landmarks in the agreed 258-value format
  - Add `console.log("Landmark count:", combined.length)` — must log 258
  - Emit via WebSocket: `socket.emit("landmarks", { landmarks })`
  - Only emit when at least one hand is detected
- [ ] Throttle emit rate to 30fps max (one emit per 33ms)
- [ ] Test that landmarks are emitting even before backend is connected (check Network tab in DevTools)

---

### ✅ Hour 4–7 | WebSocket Client + UI

- [ ] Listen for `prediction` event:
  ```javascript
  socket.on("prediction", (data) => {
    onPrediction(data.text, data.confidence);
  });
  ```
- [ ] Build `Output.js`:
  - Show current detected sign in large text
  - Show rolling sentence buffer (last 20 words)
  - Add "Clear" button to reset
  - Add "Speak All" button to read full sentence
- [ ] Build `App.js`:
  - Combine Camera + Output
  - Add debounce logic: same sign not added to sentence within 1500ms
  - Manage sentence array in state
- [ ] Layout: camera feed on left, output panel on right (stack vertically on mobile)
- [ ] Mirror the video feed horizontally (`transform: scaleX(-1)`) — feels natural

---

### ✅ Hour 7–10 | TTS + PWA

- [ ] Add Web Speech API TTS in `Output.js`:
  ```javascript
  const utterance = new SpeechSynthesisUtterance(currentSign);
  utterance.rate = 0.9;
  utterance.lang = "en-IN";
  window.speechSynthesis.speak(utterance);
  ```
- [ ] Trigger TTS only when sign changes, not on every render
- [ ] Add `public/manifest.json` for PWA:
  ```json
  {
    "name": "ISL Gesture Translator",
    "short_name": "ISL Translate",
    "display": "standalone",
    "start_url": "/",
    "theme_color": "#6c63ff",
    "background_color": "#0f0f0f"
  }
  ```
- [ ] Register a basic service worker in `index.js`
- [ ] Test on mobile Chrome: open app → "Add to Home Screen" should appear

---

### ✅ Hour 10–16 | Integration + Debugging

- [ ] Connect to Member 3's live Flask server
- [ ] Verify the full pipeline works: camera → landmarks → WebSocket → prediction → display
- [ ] Check browser console for errors — no red errors should remain after integration
- [ ] Common issues to check:
  - Landmark count is exactly 258 (not 256 or 260)
  - Socket connects successfully (check Network tab → WS)
  - Predictions appear within 1–2 seconds of performing a sign
- [ ] Test on mobile device (Android Chrome) — fix any layout issues

---

### ✅ Hour 16–20 | UI Polish

- [ ] Clean up styling — make it look presentable for judges:
  - Dark background (`#0f0f0f` or similar)
  - Camera feed with rounded corners
  - Current sign displayed in large (48px+), bold, colored text
  - Sentence buffer in readable size with clear separation
- [ ] Add a connection status indicator (green dot = connected, red = disconnected)
- [ ] Add a "Model Ready" indicator (received from backend on connect)
- [ ] Ensure mobile layout is clean — judges may test on phone
- [ ] Remove all `console.log` debug statements

---

### ✅ Hour 20–24 | Production Build + Demo Prep

- [ ] Run production build:
  ```bash
  REACT_APP_BACKEND_URL=https://your-tunnel.trycloudflare.com npm run build
  npx serve -s build -l 3000
  ```
- [ ] Test production build end-to-end (not dev server)
- [ ] Rehearse the live demo — perform all 5 demo signs smoothly
- [ ] Have a phone ready with the PWA installed as backup

---

### 📁 Your Deliverables

| File | Location | Due |
|---|---|---|
| `socket.js` | `frontend/src/` | Hour 2 |
| `Camera.js` | `frontend/src/components/` | Hour 4 |
| `Output.js` | `frontend/src/components/` | Hour 7 |
| `App.js` | `frontend/src/` | Hour 8 |
| `manifest.json` | `frontend/public/` | Hour 10 |
| Production build | `frontend/build/` | Hour 22 |

---

---

# ⚙️ Member 3 — Backend / Integration Engineer

> Your focus is the Flask server, WebSocket handling, connecting all pieces together, and deployment. You are the integration hub — everyone depends on you being unblocked and communicative.

---

### ✅ Hour 0–1 | Environment Setup

- [ ] Clone repo, checkout `backend` branch
- [ ] Install dependencies:
  ```bash
  pip install flask flask-socketio eventlet numpy tensorflow scikit-learn
  pip freeze > backend/requirements.txt
  ```
- [ ] Create folder structure:
  ```
  backend/
  ├── app.py
  ├── requirements.txt
  ├── model/
  │   ├── isl_model.h5        ← Member 1 will drop this here
  │   └── label_encoder.pkl   ← Member 1 will drop this here
  └── utils/
      └── buffer.py
  ```
- [ ] Set up Git properly — make sure all pushes go to `backend` branch

---

### ✅ Hour 1–3 | Flask + Dummy Endpoint

**This is your most critical early task. Member 2 must be able to test their frontend against your server before the real model exists.**

- [ ] Build `app.py` with dummy prediction mode:
  ```python
  from flask import Flask
  from flask_socketio import SocketIO, emit
  import numpy as np
  from collections import deque
  import random

  app = Flask(__name__)
  socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

  frame_buffer = deque(maxlen=30)
  MODEL_READY = False  # Set to True after real model loads

  DUMMY_SIGNS = ["Hello", "Help", "Water", "Food", "Doctor",
                 "ThankYou", "Yes", "No", "Emergency", "Please"]

  @socketio.on("connect")
  def on_connect():
      emit("status", {"ready": MODEL_READY})
      print("Client connected")

  @socketio.on("landmarks")
  def handle_landmarks(data):
      landmarks = np.array(data["landmarks"])
      frame_buffer.append(landmarks)
      if len(frame_buffer) == 30:
          # Dummy: return random sign every time buffer fills
          emit("prediction", {
              "text": random.choice(DUMMY_SIGNS),
              "confidence": 0.99
          })

  @app.route("/health")
  def health():
      return {"status": "ok", "model_ready": MODEL_READY}

  if __name__ == "__main__":
      socketio.run(app, host="0.0.0.0", port=5000, debug=False)
  ```
- [ ] Run server: `python backend/app.py`
- [ ] Test with Postman: connect WebSocket to `ws://localhost:5000`, emit `landmarks` with 258 zeros, verify prediction comes back
- [ ] Share `http://localhost:5000` with Member 2 so they can connect

---

### ✅ Hour 3–6 | Buffer Logic + Sliding Window

- [ ] Build `backend/utils/buffer.py`:
  ```python
  from collections import deque
  import numpy as np

  class LandmarkBuffer:
      def __init__(self, maxlen=30):
          self.buffer = deque(maxlen=maxlen)

      def add(self, landmarks):
          self.buffer.append(landmarks)

      def is_ready(self):
          return len(self.buffer) == self.buffer.maxlen

      def get_sequence(self):
          return list(self.buffer)

      def clear(self):
          self.buffer.clear()

      def size(self):
          return len(self.buffer)
  ```
- [ ] Add `reset_buffer` socket event in `app.py`:
  ```python
  @socketio.on("reset_buffer")
  def reset_buffer():
      frame_buffer.clear()
  ```
- [ ] Add buffer size debug emit (remove before demo):
  ```python
  emit("buffer_size", {"size": len(frame_buffer)})
  ```
- [ ] Test that buffer fills exactly at 30 frames and prediction fires once

---

### ✅ Hour 8–10 | Real Model Integration

When Member 1 drops `isl_model.h5` and `label_encoder.pkl` into `backend/model/`:

- [ ] Load model in `app.py`:
  ```python
  from tensorflow.keras.models import load_model
  import pickle

  model = load_model("model/isl_model.h5")
  with open("model/label_encoder.pkl", "rb") as f:
      label_encoder = pickle.load(f)
  MODEL_READY = True
  CONFIDENCE_THRESHOLD = 0.85
  ```
- [ ] Replace dummy prediction with real inference:
  ```python
  def predict(sequence):
      arr = np.expand_dims(np.array(sequence), axis=0)
      probs = model.predict(arr, verbose=0)[0]
      confidence = float(np.max(probs))
      if confidence < CONFIDENCE_THRESHOLD:
          return None, confidence
      label = label_encoder.inverse_transform([np.argmax(probs)])[0]
      return label, confidence
  ```
- [ ] Test real prediction with a known sequence
- [ ] Verify `GET /health` returns `{"model_ready": true}`
- [ ] Notify Member 2 that real model is live

---

### ✅ Hour 10–16 | Integration Debugging

You are the primary debugger during this phase. When things break, you coordinate the fix.

- [ ] CORS issues → set `cors_allowed_origins="*"` in SocketIO init
- [ ] Shape mismatch → add shape validation:
  ```python
  if landmarks.shape[0] != 258:
      emit("error", {"msg": f"Wrong shape: {landmarks.shape[0]}"})
      return
  ```
- [ ] Predictions firing too fast → coordinate with Member 2 to adjust debounce
- [ ] No predictions firing → check buffer size, emit debug info:
  ```python
  print(f"Buffer: {len(frame_buffer)}/30")
  ```
- [ ] Monitor server logs continuously during this phase
- [ ] Keep a running list of bugs and who is fixing them

---

### ✅ Hour 16–20 | Confidence Tuning + Stability

- [ ] Run server for 1 hour straight — verify no memory leaks or crashes
- [ ] Tune `CONFIDENCE_THRESHOLD`:
  - Too many wrong predictions → raise to 0.88
  - Too few predictions (long gaps) → lower to 0.80
  - Target: smooth, accurate predictions with minimal garbage
- [ ] Add error handling throughout `app.py` — no unhandled exceptions
- [ ] Remove all debug `print` statements except startup logs
- [ ] Test server restart: kill and restart, verify model reloads correctly

---

### ✅ Hour 20–22 | Deployment

- [ ] Install Cloudflare Tunnel:
  ```bash
  # Ubuntu
  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  sudo dpkg -i cloudflared-linux-amd64.deb

  # macOS
  brew install cloudflared
  ```
- [ ] Start Flask server on port 5000:
  ```bash
  python backend/app.py
  ```
- [ ] Start tunnel:
  ```bash
  cloudflared tunnel --url http://localhost:5000
  ```
- [ ] Copy the generated URL (e.g. `https://abc-def.trycloudflare.com`)
- [ ] Share tunnel URL with Member 2 for production build
- [ ] Test tunnel: visit `https://your-tunnel.trycloudflare.com/health` — must return `{"status": "ok"}`
- [ ] Keep tunnel terminal open and visible throughout the demo
- [ ] Add reconnection handler awareness — if tunnel drops, you restart it immediately

---

### ✅ Hour 22–24 | Demo Stability Watch

- [ ] Keep server running — do not restart unless absolutely necessary
- [ ] Monitor terminal for errors during demo rehearsal
- [ ] Have the tunnel restart command memorized
- [ ] Rehearse the architecture explanation section of the demo

---

### 📁 Your Deliverables

| File | Location | Due |
|---|---|---|
| `app.py` (dummy mode) | `backend/` | Hour 3 |
| `buffer.py` | `backend/utils/` | Hour 5 |
| `app.py` (real model) | `backend/` | Hour 10 |
| `requirements.txt` | `backend/` | Hour 1 |
| Cloudflare Tunnel live | — | Hour 21 |

---

---

# 🤝 Support Member

> Your role is to keep everyone unblocked, well-tested, and well-presented. You touch everything but own nothing exclusively — your job is to make the other three faster.

---

### ✅ Hour 0–1 | Setup

- [ ] Clone repo on your machine
- [ ] Install Python dependencies (same as Member 1)
- [ ] Set up the shared bug tracker (Google Doc or Notion page) with columns:
  - Bug description | Who found it | Who is fixing it | Status
- [ ] Set up a shared testing log for sign accuracy results

---

### ✅ Hour 1–3 | Data Collection Assistant

This is your most physically active period.

- [ ] Perform all 15 signs in front of the camera for Member 1's data collection
- [ ] Learn each sign correctly before recording — Member 1 will show you
- [ ] Perform at least **15 of the 30 sequences** per sign
- [ ] Get 1–2 other team members to perform some sequences for natural variation
- [ ] After collection, verify dataset integrity:
  ```python
  import os
  import numpy as np

  signs = ["Hello", "ThankYou", "Yes", "No", "Help", "Water", "Food",
           "Doctor", "Emergency", "IMe", "You", "Please", "Sorry", "Good", "Stop"]

  for sign in signs:
      count = 0
      for seq in range(30):
          for frame in range(30):
              path = f"data/{sign}/{seq}/{frame}.npy"
              if os.path.exists(path):
                  arr = np.load(path)
                  assert arr.shape == (258,), f"Wrong shape at {path}: {arr.shape}"
                  count += 1
      print(f"✅ {sign}: {count}/900 frames OK")
  ```

---

### ✅ Hour 3–10 | Documentation

While Members 1–3 are deep in code, you handle all documentation.

- [ ] Write `README.md` for the GitHub repo:
  - Project description
  - Setup instructions
  - How to run data collection, training, server, and frontend
  - Architecture diagram (text-based is fine)
  - Team member names and roles
- [ ] Start building the **presentation slides** (10 slides max):

  | Slide | Content |
  |---|---|
  | 1 | Title + team names |
  | 2 | Problem statement (63M hearing-impaired in India) |
  | 3 | Current solutions and why they fail |
  | 4 | Our solution — one-line pitch |
  | 5 | Architecture diagram |
  | 6 | Tech stack (clean visual) |
  | 7 | Live demo placeholder |
  | 8 | Model accuracy results |
  | 9 | Impact and use cases |
  | 10 | Future scope + thank you |

- [ ] Add architecture diagram to slide 5:
  ```
  Camera → MediaPipe (on-device) → WebSocket → Flask → BiLSTM → Text + TTS
  ```

---

### ✅ Hour 10–16 | Full System Testing

This is your most critical phase. Test everything that Members 1–3 are too busy to test.

- [ ] **Sign accuracy test** — test all 15 signs 5 times each, record results:

  | Sign | Attempt 1 | Attempt 2 | Attempt 3 | Attempt 4 | Attempt 5 | Pass Rate |
  |---|---|---|---|---|---|---|
  | Hello | | | | | | |
  | ThankYou | | | | | | |
  | Yes | | | | | | |
  | No | | | | | | |
  | Help | | | | | | |
  | Water | | | | | | |
  | Food | | | | | | |
  | Doctor | | | | | | |
  | Emergency | | | | | | |
  | IMe | | | | | | |
  | You | | | | | | |
  | Please | | | | | | |
  | Sorry | | | | | | |
  | Good | | | | | | |
  | Stop | | | | | | |

- [ ] Report results to Member 1 — signs below 60% pass rate need more data
- [ ] Test on mobile (Android Chrome):
  - [ ] Camera opens correctly
  - [ ] Landmarks extract (check browser console)
  - [ ] Predictions appear
  - [ ] TTS works
  - [ ] "Add to Home Screen" option appears
- [ ] Test the "Clear" and "Speak All" buttons
- [ ] Test what happens if you sign nothing for 10 seconds — should show nothing, not garbage
- [ ] Test what happens if internet drops — should reconnect gracefully
- [ ] Log every bug found into the shared bug tracker

---

### ✅ Hour 16–20 | Presentation Polish + Video

- [ ] Finalize all 10 slides — clean, minimal, professional
- [ ] Add final model accuracy numbers to slide 8 (get from Member 1)
- [ ] Practice the 3-minute demo script with the team:

  ```
  0:00 – 0:15  Problem statement
  0:15 – 1:45  Live demo (5 signs: Hello, IMe, Water, Help, ThankYou)
  1:45 – 2:15  Architecture explanation
  2:15 – 2:45  Impact + future scope
  ```

- [ ] Time the demo — it must fit in 3 minutes exactly
- [ ] Note which team member says what part

---

### ✅ Hour 20–22 | Backup Demo Video

**This is non-negotiable. A recorded demo can save you if the live demo fails.**

- [ ] Set up screen recording (OBS, Loom, or phone camera pointed at screen)
- [ ] Record a clean run of the demo:
  - Full screen showing camera feed + output
  - Perform: Hello → IMe → Water → Help → ThankYou
  - Each sign clearly detected, TTS audible
  - Sentence builds up on screen
- [ ] Watch the recording — if any sign was missed, record again
- [ ] Upload to Google Drive or keep on USB
- [ ] Confirm backup video plays correctly on the presentation machine

---

### ✅ Hour 22–24 | Pre-Demo Checks

- [ ] Full end-to-end test on presentation machine (not dev laptop)
- [ ] Verify Cloudflare Tunnel URL is in the frontend production build
- [ ] Check that production build is served and accessible
- [ ] Confirm microphone/speaker works for TTS during presentation
- [ ] Confirm backup video is accessible
- [ ] All team members know their speaking part
- [ ] Slides are loaded and on the right machine

---

### 📁 Your Deliverables

| File | Location | Due |
|---|---|---|
| `README.md` | repo root | Hour 8 |
| Presentation slides | Google Drive / USB | Hour 20 |
| Sign accuracy test results | Shared doc | Hour 16 |
| Backup demo video | Google Drive / USB | Hour 22 |
| Bug tracker log | Shared doc | Ongoing |

---

---

## 🗓️ Combined Timeline at a Glance

| Hour | Member 1 | Member 2 | Member 3 | Support |
|---|---|---|---|---|
| 0–1 | Setup + landmark agreement | Setup + landmark agreement | Setup + dummy server start | Setup + bug tracker |
| 1–3 | Data collection | Camera.js + MediaPipe | Flask dummy endpoint | Perform gestures for data |
| 3–6 | Training (running) + inference wrapper | UI + socket client | Buffer logic | Dataset verification + slides start |
| 6–8 | Training done, model handoff | TTS + PWA | Integrate real model | Slides + README |
| 8–10 | Accuracy checks | Production build prep | Integration testing | Full system first test |
| 10–14 | Integration debug support | Integration debug | Integration debug (lead) | All 15 sign testing |
| 14–16 | Retrain failing signs if needed | UI polish | Threshold tuning | Mobile testing |
| 16–18 | Final accuracy validation | Mobile layout fixes | Server stability | Demo rehearsal |
| 18–20 | 💤 Rest (rotating) | 💤 Rest (rotating) | 💤 Rest (rotating) | 💤 Rest (rotating) |
| 20–22 | Buffer / emergency fixes | Production build | Cloudflare Tunnel deploy | Backup video recording |
| 22–24 | Demo rehearsal | Demo rehearsal | Server watch | Final checks + slides |

---

## 🚨 Emergency Protocols

### If model accuracy is too low at Hour 14
- Member 1 re-collects 15 more sequences for failing signs
- Support Member performs gestures again immediately
- Retrain only the failing classes if possible, else full retrain
- Reduce sign vocabulary to top 10 most accurate signs if needed

### If WebSocket connection is unstable
- Member 3 adds reconnection logic to `app.py`
- Member 2 adds client-side reconnect: `socket.on("disconnect", () => setTimeout(() => socket.connect(), 2000))`

### If Cloudflare Tunnel fails during demo
- Switch to local network (hotspot from demo machine)
- Update `REACT_APP_BACKEND_URL` to local IP: `http://192.168.x.x:5000`
- Rebuild frontend: `npm run build`

### If live demo breaks completely during presentation
- Switch to backup demo video immediately — no apologies, no explanation
- Continue with slides and architecture explanation
- Judges care about the idea and thinking, not just the live demo

---

## ✅ Final Pre-Presentation Checklist

- [ ] All 15 signs detected reliably
- [ ] TTS works and is audible
- [ ] Mobile PWA tested and installable
- [ ] Cloudflare Tunnel live and health check passing
- [ ] Production frontend build running
- [ ] Backup demo video ready and accessible
- [ ] Presentation slides complete and on correct machine
- [ ] All team members know their speaking role
- [ ] Demo rehearsed at least twice end-to-end
- [ ] Every team member has slept at least 3 hours

---

*ISL Gesture Detection System — Team Task Assignment v1.0*
*Good luck. Build something that matters. 🤟*