import { TranscriptView } from "./components/TranscriptView";
import { ControlBar } from "./components/ControlBar";
import "./App.css";

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>LiveScribe</h1>
      </header>
      <TranscriptView />
      <ControlBar />
    </div>
  );
}

export default App;
