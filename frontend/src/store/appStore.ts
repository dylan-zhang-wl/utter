import { create } from "zustand";

export interface TranscriptEntry {
  text: string;
  translation: string;
  startTime: number;
  endTime: number;
  timestamp: number;
}

interface AppState {
  // Connection
  connected: boolean;
  setConnected: (v: boolean) => void;

  // Recording
  recording: boolean;
  setRecording: (v: boolean) => void;

  // Transcripts
  entries: TranscriptEntry[];
  addEntry: (entry: TranscriptEntry) => void;
  clearEntries: () => void;

  // Settings
  audioSource: "microphone" | "system";
  setAudioSource: (s: "microphone" | "system") => void;
  translationEngine: "google" | "openai";
  setTranslationEngine: (e: "google" | "openai") => void;
  displayMode: "english" | "bilingual" | "chinese";
  setDisplayMode: (m: "english" | "bilingual" | "chinese") => void;
}

export const useAppStore = create<AppState>((set) => ({
  connected: false,
  setConnected: (v) => set({ connected: v }),

  recording: false,
  setRecording: (v) => set({ recording: v }),

  entries: [],
  addEntry: (entry) => set((s) => ({ entries: [...s.entries, entry] })),
  clearEntries: () => set({ entries: [] }),

  audioSource: "microphone",
  setAudioSource: (audioSource) => set({ audioSource }),
  translationEngine: "google",
  setTranslationEngine: (translationEngine) => set({ translationEngine }),
  displayMode: "bilingual",
  setDisplayMode: (displayMode) => set({ displayMode }),
}));
