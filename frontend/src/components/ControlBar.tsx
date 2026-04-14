import { useAppStore } from "../store/appStore";
import { useWebSocket } from "../hooks/useWebSocket";
import "./ControlBar.css";

export function ControlBar() {
  const {
    recording, connected,
    audioSource, setAudioSource,
    translationEngine, setTranslationEngine,
    displayMode, setDisplayMode,
  } = useAppStore();
  const { send } = useWebSocket();

  const toggleRecording = () => {
    if (recording) {
      send("stop");
    } else {
      send("start");
    }
  };

  const switchSource = (source: "microphone" | "system") => {
    setAudioSource(source);
    send("switch_source", { source });
  };

  return (
    <div className="control-bar">
      <button
        className={`btn-record ${recording ? "recording" : ""}`}
        onClick={toggleRecording}
        disabled={!connected}
      >
        {recording ? "⏹ Stop" : "● Start"}
      </button>

      <div className="control-group">
        <label>Source:</label>
        <select
          value={audioSource}
          onChange={(e) => switchSource(e.target.value as "microphone" | "system")}
        >
          <option value="microphone">Microphone</option>
          <option value="system">System Audio</option>
        </select>
      </div>

      <div className="control-group">
        <label>Translate:</label>
        <select
          value={translationEngine}
          onChange={(e) => setTranslationEngine(e.target.value as "google" | "openai")}
        >
          <option value="google">Google</option>
          <option value="openai">OpenAI</option>
        </select>
      </div>

      <div className="control-group">
        <label>Display:</label>
        <select
          value={displayMode}
          onChange={(e) => setDisplayMode(e.target.value as "english" | "bilingual" | "chinese")}
        >
          <option value="bilingual">EN + 中文</option>
          <option value="english">English</option>
          <option value="chinese">中文</option>
        </select>
      </div>

      <div className={`status-dot ${connected ? "connected" : "disconnected"}`} />
    </div>
  );
}
