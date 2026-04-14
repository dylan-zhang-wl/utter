import { useState } from "react";
import { TranscriptView } from "./components/TranscriptView";
import { ControlBar } from "./components/ControlBar";
import { Settings } from "./components/Settings";
import "./App.css";

function App() {
  const [showSettings, setShowSettings] = useState(false);

  return (
    <div className="app">
      <header className="app-header">
        <h1>LiveScribe</h1>
        <button className="btn-settings" onClick={() => setShowSettings(true)}>
          ⚙
        </button>
      </header>
      <TranscriptView />
      <ControlBar />
      {showSettings && <Settings onClose={() => setShowSettings(false)} />}
    </div>
  );
}

export default App;
