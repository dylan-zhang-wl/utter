import { useState, useEffect } from "react";
import "./Settings.css";

interface SettingsProps {
  onClose: () => void;
}

export function Settings({ onClose }: SettingsProps) {
  const [apiKey, setApiKey] = useState("");
  const [whisperModel, setWhisperModel] = useState("base");
  const [savePath, setSavePath] = useState("");

  useEffect(() => {
    fetch("http://127.0.0.1:8765/config")
      .then((r) => r.json())
      .then((data) => {
        setApiKey(data.openai_api_key || "");
        setWhisperModel(data.whisper_model);
        setSavePath(data.save_dir);
      })
      .catch(() => {});
  }, []);

  const save = () => {
    fetch("http://127.0.0.1:8765/config", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        openai_api_key: apiKey || null,
        whisper_model: whisperModel,
        save_dir: savePath,
      }),
    }).then(() => onClose());
  };

  return (
    <div className="settings-overlay">
      <div className="settings-panel">
        <h2>Settings</h2>

        <div className="setting-row">
          <label>Whisper Model</label>
          <select value={whisperModel} onChange={(e) => setWhisperModel(e.target.value)}>
            <option value="tiny">tiny (fastest)</option>
            <option value="base">base (recommended)</option>
            <option value="small">small (more accurate)</option>
            <option value="medium">medium (slower)</option>
          </select>
        </div>

        <div className="setting-row">
          <label>OpenAI API Key</label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-..."
          />
        </div>

        <div className="setting-row">
          <label>Save Directory</label>
          <input
            type="text"
            value={savePath}
            onChange={(e) => setSavePath(e.target.value)}
          />
        </div>

        <div className="setting-actions">
          <button className="btn-cancel" onClick={onClose}>Cancel</button>
          <button className="btn-save" onClick={save}>Save</button>
        </div>
      </div>
    </div>
  );
}
