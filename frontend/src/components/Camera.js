import React, { useEffect, useRef } from 'react';
import { Holistic } from '@mediapipe/holistic';
import { Camera as MPWebcam } from '@mediapipe/camera_utils';
import { socket } from '../socket';

const Camera = ({ onPrediction }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const lastEmitTime = useRef(0);

  useEffect(() => {
    const holistic = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    const onResults = (results) => {
      const now = Date.now();
      // Throttle to ~30fps (33ms)
      if (now - lastEmitTime.current < 33) return;

      const landmarks = extractLandmarks(results);
      
      // Check if at least one hand is detected (heuristic: not all zeros in hand regions)
      const hasLeftHand = landmarks.slice(0, 63).some(v => v !== 0);
      const hasRightHand = landmarks.slice(63, 126).some(v => v !== 0);

      if (hasLeftHand || hasRightHand) {
        socket.emit("landmarks", { landmarks });
        lastEmitTime.current = now;
      }
    };

    holistic.onResults(onResults);

    if (videoRef.current) {
      const camera = new MPWebcam(videoRef.current, {
        onFrame: async () => {
          await holistic.send({ image: videoRef.current });
        },
        width: 640,
        height: 480,
      });
      camera.start();
    }

    // Prediction listener
    const handlePrediction = (data) => {
      onPrediction(data.text, data.confidence);
    };

    socket.on("prediction", handlePrediction);

    return () => {
      holistic.close();
      socket.off("prediction", handlePrediction);
    };
  }, [onPrediction]);

  const extractLandmarks = (results) => {
    // Left Hand: 21 * 3 = 63
    let lh = new Array(63).fill(0);
    if (results.leftHandLandmarks) {
      lh = results.leftHandLandmarks.flatMap(l => [l.x, l.y, l.z]);
    }

    // Right Hand: 21 * 3 = 63
    let rh = new Array(63).fill(0);
    if (results.rightHandLandmarks) {
      rh = results.rightHandLandmarks.flatMap(l => [l.x, l.y, l.z]);
    }

    // Pose: 33 * 4 = 132
    let pose = new Array(132).fill(0);
    if (results.poseLandmarks) {
      pose = results.poseLandmarks.flatMap(l => [l.x, l.y, l.z, l.visibility || 0]);
    }

    // Total = 63 + 63 + 132 = 258
    return [...lh, ...rh, ...pose];
  };

  return (
    <div className="camera-container">
      <video
        ref={videoRef}
        className="camera-video"
        playsInline
        muted
      />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
};

export default Camera;
