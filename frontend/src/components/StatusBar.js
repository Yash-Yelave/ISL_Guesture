import React, { useEffect, useState } from 'react';
import { socket } from '../socket';

const StatusBar = ({ connected, bufferSize = 0 }) => {
  const [status, setStatus] = useState({
    modelReady: false,
    cameraReady: false
  });

  useEffect(() => {
    socket.on('status', (data) => {
      setStatus({
        modelReady: data.ready,
        cameraReady: data.camera
      });
    });

    return () => socket.off('status');
  }, []);

  const bufferPercentage = Math.min(100, (bufferSize / 30) * 100);

  return (
    <div className="status-bar">
      <div className="status-indicators">
        <div className={`indicator ${connected ? 'connected' : 'disconnected'}`}>
          <span className="dot"></span>
          {connected ? 'Connected' : 'Disconnected'}
        </div>
        
        <div className={`badge ${status.modelReady ? 'ready' : 'dummy'}`}>
          {status.modelReady ? 'Model Ready' : 'Dummy Mode'}
        </div>

        <div className={`badge ${status.cameraReady ? 'camera-on' : 'camera-off'}`}>
          {status.cameraReady ? 'Camera On' : 'Camera Off'}
        </div>
      </div>

      <div className="buffer-progress-container">
        <div 
          className="buffer-progress-bar" 
          style={{ width: `${bufferPercentage}%` }}
        ></div>
      </div>
    </div>
  );
};

export default StatusBar;
