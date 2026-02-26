import React, { useState } from 'react';
import { socket } from '../socket';

const Output = ({ prediction, confidence, sentence, onClear }) => {
  const [sensitivity, setSensitivity] = useState(0.85);

  const handleSensitivityChange = (e) => {
    const value = parseFloat(e.target.value);
    setSensitivity(value);
    socket.emit("set_confidence", { threshold: value });
  };

  return (
    <div className="output-panel">
      <div className="prediction-box">
        <h3>Current Sign</h3>
        <div className="prediction-text">
          {prediction || "---"}
          {prediction && <span className="confidence-pill">{(confidence * 100).toFixed(0)}%</span>}
        </div>
      </div>

      <div className="sentence-box">
        <h3>Sentence</h3>
        <div className="sentence-text">{sentence || "Start signing to see text here..."}</div>
        <button className="clear-btn" onClick={onClear}>Clear</button>
      </div>

      <div className="sensitivity-control">
        <div className="sensitivity-label">
          <span>Sensitivity</span>
          <span>{Math.round(sensitivity * 100)}%</span>
        </div>
        <input
          type="range"
          min="0.60"
          max="0.95"
          step="0.05"
          value={sensitivity}
          onChange={handleSensitivityChange}
          className="sensitivity-slider"
        />
      </div>
    </div>
  );
};

export default Output;
