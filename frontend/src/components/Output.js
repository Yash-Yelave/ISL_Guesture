import React from 'react';

const Output = ({ currentSign, confidence, sentence, onClear }) => {
  const speakSentence = () => {
    if (sentence.length === 0) return;
    const utterance = new SpeechSynthesisUtterance(sentence.join(" "));
    utterance.lang = "en-IN";
    utterance.rate = 0.9;
    window.speechSynthesis.speak(utterance);
  };

  React.useEffect(() => {
    if (currentSign && currentSign !== "—") {
      const utterance = new SpeechSynthesisUtterance(currentSign);
      utterance.lang = "en-IN";
      utterance.rate = 0.9;
      window.speechSynthesis.speak(utterance);
    }
  }, [currentSign]);

  return (
    <div className="output-panel">
      <div className="current-sign-container">
        <h3 className="panel-title">Detected Sign</h3>
        <div className="sign-text">{currentSign || "—"}</div>
        {confidence > 0 && (
          <div className="confidence-text">
            Confidence: {(confidence * 100).toFixed(1)}%
          </div>
        )}
      </div>

      <div className="sentence-container">
        <h3 className="panel-title">Sentence</h3>
        <p className="sentence-buffer">
          {sentence.length > 0 ? sentence.join(" ") : "Start signing..."}
        </p>
      </div>

      <div className="button-group">
        <button className="btn-clear" onClick={onClear}>Clear</button>
        <button className="btn-speak" onClick={speakSentence}>Speak All</button>
      </div>
    </div>
  );
};

export default Output;
