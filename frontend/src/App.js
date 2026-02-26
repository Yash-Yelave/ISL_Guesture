import React, { useState, useEffect, useCallback, useRef } from 'react';
import Camera from './components/Camera';
import Output from './components/Output';
import StatusBar from './components/StatusBar';
import { socket } from './socket';
import './App.css';

function App() {
  const [currentSign, setCurrentSign] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [sentence, setSentence] = useState([]);
  const [connected, setConnected] = useState(socket.connected);
  const [modelReady, setModelReady] = useState(false);
  const [bufferSize, setBufferSize] = useState(0);

  const lastAddedRef = useRef({ sign: "", time: 0 });

  useEffect(() => {
    const onConnect = () => setConnected(true);
    const onDisconnect = () => setConnected(false);
    const onStatus = (data) => setModelReady(data.ready);
    const onBufferSize = (data) => setBufferSize(data.size);

    socket.on("connect", onConnect);
    socket.on("disconnect", onDisconnect);
    socket.on("status", onStatus);
    socket.on("buffer_size", onBufferSize);

    return () => {
      socket.off("connect", onConnect);
      socket.off("disconnect", onDisconnect);
      socket.off("status", onStatus);
      socket.off("buffer_size", onBufferSize);
    };
  }, []);

  const handlePrediction = useCallback((text, conf) => {
    setCurrentSign(text);
    setConfidence(conf);

    const now = Date.now();
    const { sign: lastSign, time: lastTime } = lastAddedRef.current;

    // Debounce: Same sign not added within 1500ms
    if (text && (text !== lastSign || now - lastTime > 1500)) {
      setSentence(prev => {
        const newSentence = [...prev, text];
        return newSentence.slice(-20); // Keep max 20 words
      });
      lastAddedRef.current = { sign: text, time: now };
    }
  }, []);

  const handleClear = () => {
    setCurrentSign("");
    setConfidence(0);
    setSentence([]);
    setBufferSize(0);
    socket.emit("reset_buffer");
  };

  return (
    <div className="app">
      <header className="header">
        <div className="logo-container">
          <span className="logo">🤟</span>
          <h1>ISL Translator</h1>
        </div>
      </header>

      <StatusBar 
        connected={connected} 
        modelReady={modelReady} 
        bufferSize={bufferSize} 
      />

      <main className="main-grid">
        <div className="camera-panel">
          <Camera onPrediction={handlePrediction} />
        </div>
        
        <div className="output-panel-container">
          <Output 
            currentSign={currentSign} 
            confidence={confidence} 
            sentence={sentence} 
            onClear={handleClear}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
