import React, { useState, useEffect } from 'react';
import { socket } from '../socket';

const StatusBar = ({ connected, modelReady, bufferSize }) => {
  const progress = (bufferSize / 30) * 100;

  return (
    <div className="status-bar">
      <div className="status-item">
        <span className={`dot ${connected ? 'connected' : 'disconnected'}`}></span>
        <span className="status-label">{connected ? 'Connected' : 'Disconnected'}</span>
      </div>

      <div className="status-item">
        <span className={`badge ${modelReady ? 'model-ready' : 'dummy-mode'}`}>
          {modelReady ? 'Model Ready' : 'Dummy Mode'}
        </span>
      </div>

      <div className="status-item buffer-status">
        <span className="status-label">Buffer: {bufferSize}/30</span>
        <div className="progress-container">
          <div 
            className="progress-bar" 
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      </div>
    </div>
  );
};

export default StatusBar;
