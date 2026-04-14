import json
from datetime import datetime
from pathlib import Path


class Session:
    def __init__(self, audio_source: str, save_dir: str = ""):
        self.audio_source = audio_source
        self.save_dir = save_dir or str(Path.home() / "LiveScribe" / "sessions")
        self.entries: list[dict] = []
        self.started_at = datetime.now().isoformat()

    def add_entry(self, text: str, translation: str, start_time: float, end_time: float):
        self.entries.append({
            "text": text,
            "translation": translation,
            "start_time": start_time,
            "end_time": end_time,
            "timestamp": datetime.now().isoformat(),
        })

    def save(self) -> str:
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = str(Path(self.save_dir) / filename)
        data = {
            "started_at": self.started_at,
            "audio_source": self.audio_source,
            "entries": self.entries,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filepath

    def export(self, fmt: str) -> str:
        if fmt == "txt":
            return self._export_txt()
        elif fmt == "srt":
            return self._export_srt()
        elif fmt == "markdown":
            return self._export_markdown()
        raise ValueError(f"Unknown format: {fmt}")

    def _export_txt(self) -> str:
        lines = []
        for e in self.entries:
            lines.append(e["text"])
            if e.get("translation"):
                lines.append(e["translation"])
            lines.append("")
        return "\n".join(lines)

    def _export_srt(self) -> str:
        lines = []
        for i, e in enumerate(self.entries, 1):
            start = self._format_srt_time(e["start_time"])
            end = self._format_srt_time(e["end_time"])
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(e["text"])
            if e.get("translation"):
                lines.append(e["translation"])
            lines.append("")
        return "\n".join(lines)

    def _export_markdown(self) -> str:
        lines = [f"# LiveScribe Session — {self.started_at}", ""]
        lines.append(f"**Audio source:** {self.audio_source}\n")
        for e in self.entries:
            ts = self._format_time(e["start_time"])
            lines.append(f"**[{ts}]** {e['text']}")
            if e.get("translation"):
                lines.append(f"> {e['translation']}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _format_time(seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"
