# Indian Sign Language (ISL) Gesture Detection System

A production-ready real-time gesture detection system using a PC-centric architecture.

## Architecture

- **PC Backend**: Handles camera access, MediaPipe landmark extraction, and BiLSTM model inference.
- **Client (Frontend)**: A pure receiver that displays the MJPEG camera feed and prediction strings via WebSockets.

## Setup

### PC (runs everything)

1. **Install dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Add trained model files** to `backend/model/`:
   - `isl_model.h5`
   - `label_encoder.pkl`
   *(Dummy mode works without these files)*

3. **Find your PC's local IP address**:
   - **Windows**: `ipconfig` → IPv4 Address
   - **Mac/Linux**: `hostname -I`

4. **Start backend**:
   ```bash
   cd backend
   python app.py
   ```

5. **Build and serve frontend**:
   ```bash
   cd frontend
   npm install
   npm run build
   npx serve -s build -l 3000 --listen 0.0.0.0
   ```

### Phone or any other device (just a browser)

1. Connect to the same WiFi as the PC.
2. Open browser and navigate to: `http://YOUR_PC_LOCAL_IP:3000`
3. That's it — no permissions, no install, no setup.

### Multiple viewers

Any number of phones/tablets/laptops can connect simultaneously. All see the same camera feed and predictions in real time.
