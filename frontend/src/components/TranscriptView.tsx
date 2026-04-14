import { useEffect, useRef } from "react";
import { useAppStore } from "../store/appStore";
import "./TranscriptView.css";

export function TranscriptView() {
  const { entries, displayMode } = useAppStore();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries]);

  return (
    <div className="transcript-view">
      {entries.length === 0 && (
        <div className="transcript-empty">
          Press Start to begin transcription...
        </div>
      )}
      {entries.map((entry, i) => (
        <div key={i} className="transcript-entry">
          {displayMode !== "chinese" && (
            <p className="transcript-text">{entry.text}</p>
          )}
          {displayMode !== "english" && entry.translation && (
            <p className="transcript-translation">{entry.translation}</p>
          )}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
