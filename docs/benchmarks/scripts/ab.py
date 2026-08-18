"""A/B: does a growing buffer beat independent chunks?

The complaint is a blank screen for many seconds and then two sentences at
once. Both arms run the same VAD cuts and the same model; the only difference
is what audio each transcription sees.

  A (today)  each VAD segment transcribed alone. A ceiling cut lands
             mid-sentence, so the fragment has no full stop, so the assembler
             holds it — and you need TWO segments to see ONE sentence.

  B          the same segments, but a ceiling cut keeps its audio and the next
             transcription sees the whole thing. Whisper punctuates a complete
             sentence correctly. Same number of model calls; only the audio
             is longer, and 10s vs 18s was measured at 1.12s vs 1.27s.

Reported per sentence: the audio time it was spoken by, and the audio time it
would have appeared on screen (model time included).
"""
import sys, time
import numpy as np
import soundfile as sf

sys.path.insert(0, "/Users/d/Desktop/华工大/cc cowork/utter")
from backend.vad import VadSegmenter, SpeechEnd, SAMPLE_RATE, FRAME_SAMPLES
from backend.sentences import SentenceAssembler
from backend.providers.mlx import MlxWhisperProvider

AUDIO = "/private/tmp/claude-501/-Users-d-Desktop-----cc-cowork-utter/fcd94861-ba29-4991-bb2c-5ee806b006b0/scratchpad/speech.wav"
MAX_SEC, MIN_SEC, SILENCE_MS = 8, 1.5, 500

audio, sr = sf.read(AUDIO, dtype="float32")
assert sr == SAMPLE_RATE

stt = MlxWhisperProvider("balanced")
stt.transcribe(np.zeros(SAMPLE_RATE, dtype="float32"), language="en")  # warm


def cuts():
    seg = VadSegmenter(vad_silence_ms=SILENCE_MS, max_utterance_sec=MAX_SEC,
                       min_utterance_sec=MIN_SEC)
    out = []
    for i in range(0, len(audio) - FRAME_SAMPLES, FRAME_SAMPLES):
        for ev in seg.feed(audio[i:i + FRAME_SAMPLES]):
            if isinstance(ev, SpeechEnd):
                out.append(ev)
    for ev in seg.flush():
        if isinstance(ev, SpeechEnd):
            out.append(ev)
    return out


def run(mode, segments):
    asm, shown, calls, model_time = SentenceAssembler(), [], 0, 0.0
    carry = None          # B only: audio kept from a ceiling cut
    emitted = 0
    for ev in segments:
        clip = ev.audio
        if mode == "B":
            clip = ev.audio if carry is None else np.concatenate([carry, clip])
        t0 = time.perf_counter()
        text = stt.transcribe(clip, language="en").strip()
        model_time += time.perf_counter() - t0
        calls += 1
        # when this text could be on screen, in audio time
        at = ev.end_sample / SAMPLE_RATE + (time.perf_counter() - t0)

        if mode == "A":
            for s in asm.feed(text):
                shown.append((at, s))
        else:
            # re-transcribing the same audio re-derives earlier sentences too;
            # only what is past the high-water mark is new
            whole = SentenceAssembler()
            sentences = whole.feed(text)
            for s in sentences[emitted:]:
                shown.append((at, s))
            emitted = len(sentences)
            if ev.forced and whole.pending:
                carry = clip           # sentence unfinished: keep the audio
            else:
                if whole.pending:      # a real pause: nothing more is coming
                    shown.append((at, whole.pending))
                carry, emitted = None, 0
    if mode == "A":
        for s in asm.flush():
            shown.append((len(audio) / SAMPLE_RATE, s))
    return shown, calls, model_time


segments = cuts()
print(f"VAD 切出 {len(segments)} 段，"
      f"其中撞天花板 {sum(1 for s in segments if s.forced)} 段；"
      f"音频 {len(audio)/SAMPLE_RATE:.1f}s\n")

for mode in ("A", "B"):
    shown, calls, model_time = run(mode, segments)
    times = [t for t, _ in shown]
    gaps = [round(b - a, 1) for a, b in zip(times, times[1:])]
    label = "A 各段独立转录（现状）" if mode == "A" else "B 天花板切了就留住音频"
    print(f"--- {label} ---")
    print(f"  {len(shown)} 句，模型调用 {calls} 次，"
          f"共 {model_time:.1f}s（占空比 {model_time/(len(audio)/SAMPLE_RATE)*100:.0f}%）")
    print(f"  出字间隔最大 {max(gaps):.1f}s，中位 {sorted(gaps)[len(gaps)//2]:.1f}s")
    print(f"  间隔: {gaps}")
    for t, s in shown:
        print(f"    @{t:>5.1f}s  {s[:78]}")
    print()
