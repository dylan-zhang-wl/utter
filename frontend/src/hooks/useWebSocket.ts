import { useEffect, useRef, useCallback } from "react";
import { useAppStore } from "../store/appStore";

const WS_URL = "ws://127.0.0.1:8765/ws";

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const { setConnected, setRecording, addEntry } = useAppStore();

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        setTimeout(connect, 3000); // auto-reconnect
      };
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "transcript") {
          addEntry({
            text: data.text,
            translation: data.translation,
            startTime: data.start_time,
            endTime: data.end_time,
            timestamp: Date.now(),
          });
        } else if (data.type === "status") {
          if (data.status === "recording") setRecording(true);
          if (data.status === "stopped") setRecording(false);
        }
      };

      wsRef.current = ws;
    };

    connect();
    return () => wsRef.current?.close();
  }, []);

  const send = useCallback((action: string, extra?: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action, ...extra }));
    }
  }, []);

  return { send };
}
