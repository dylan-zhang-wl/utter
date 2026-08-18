"""How much latency can we buy with the 87% of the machine we are not using?

The A/B showed the wait is not the model and not the punctuation hold — it is
that text can only appear when the VAD closes a segment, and with a speaker in
flow that is every 8 seconds by the ceiling. Model time was 8.5s for 63.4s of
audio: 13% duty.

So stop waiting for the VAD. Re-transcribe a growing buffer on a fixed tick and
emit sentences as they complete — ufal/whisper_streaming's shape, minus the
LocalAgreement re-checking, because we only ever emit text that Whisper has
already terminated with a full stop.

Reported: how long after a sentence was finished being spoken it appeared, and
what fraction of the machine that costs.
"""
import sys, time
import numpy as np
import soundfile as sf

sys.path.insert(0, "/Users/d/Desktop/华工大/cc cowork/utter")
from backend.sentences import SentenceAssembler
from backend.providers.mlx import MlxWhisperProvider
from backend.vad import SAMPLE_RATE

HERE = "/private/tmp/claude-501/-Users-d-Desktop-----cc-cowork-utter/fcd94861-ba29-4991-bb2c-5ee806b006b0/scratchpad"
audio, sr = sf.read(f"{HERE}/speech.wav", dtype="float32")
total = len(audio) / sr

stt = MlxWhisperProvider("balanced")
stt.transcribe(np.zeros(SAMPLE_RATE, dtype="float32"), language="en")

print(f"音频 {total:.1f}s\n")

MAX_BUFFER = 24.0     # Whisper's window is 30s; stay inside it


def run(tick):
    buffer_start = 0.0
    shown, model_time, calls = [], 0.0, 0
    emitted = 0
    at = tick
    while at <= total + tick:
        clip = audio[int(buffer_start * sr):int(min(at, total) * sr)]
        if len(clip) < sr * 0.5:
            at += tick
            continue
        t0 = time.perf_counter()
        text = stt.transcribe(clip, language="en").strip()
        cost = time.perf_counter() - t0
        model_time += cost
        calls += 1

        asm = SentenceAssembler()
        sentences = asm.feed(text)
        for s in sentences[emitted:]:
            shown.append((at + cost, s))       # audio time it hit the screen
        emitted = len(sentences)

        held = at - buffer_start
        if not asm.pending or held >= MAX_BUFFER:
            if asm.pending and held >= MAX_BUFFER:
                shown.append((at + cost, asm.pending))
            buffer_start, emitted = min(at, total), 0
        # the model cannot start the next round before this one finished
        at = max(at + tick, at + cost)
    return shown, model_time, calls


for tick in (8.0, 4.0, 2.5):
    shown, model_time, calls = run(tick)
    duty = model_time / total * 100
    print(f"--- 每 {tick:.1f}s 转一次整段缓冲 ---")
    print(f"  出 {len(shown)} 句，模型 {calls} 次共 {model_time:.1f}s，占空比 {duty:.0f}%")
    per_call = model_time / calls
    print(f"  说完到看见：约 {tick/2 + per_call:.1f}s（平均），最坏 {tick + per_call:.1f}s")
    times = [t for t, _ in shown]
    gaps = [round(b - a, 1) for a, b in zip(times, times[1:])]
    print(f"  屏幕空白最久 {max(gaps):.1f}s\n")
