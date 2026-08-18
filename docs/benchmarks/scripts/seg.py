"""Does the tick make sentence segmentation worse?

The worry is real and specific: a tick re-transcribes audio that has been cut
mid-sentence, and Whisper may put a full stop at the truncation. That would
split one spoken sentence into two on screen.

There is a ground truth here — the source text the TTS read — so boundaries can
be scored rather than eyeballed. Words are aligned with difflib (ASR text never
matches the source exactly), then each true sentence boundary is checked
against the boundaries the pipeline produced.
"""
import re, sys, time, difflib
import numpy as np
import soundfile as sf

sys.path.insert(0, "/Users/d/Desktop/华工大/cc cowork/utter")
from backend.sentences import SentenceAssembler, split_sentences
from backend.providers.mlx import MlxWhisperProvider
from backend.vad import VadSegmenter, SpeechEnd, SAMPLE_RATE, FRAME_SAMPLES

HERE = "/private/tmp/claude-501/-Users-d-Desktop-----cc-cowork-utter/fcd94861-ba29-4991-bb2c-5ee806b006b0/scratchpad"
audio, sr = sf.read(f"{HERE}/speech.wav", dtype="float32")
total = len(audio) / sr
source = open(f"{HERE}/speech.txt").read()

stt = MlxWhisperProvider("balanced")
stt.transcribe(np.zeros(SAMPLE_RATE, dtype="float32"), language="en")

norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).split()


def bounds(sentences):
    """Word stream plus the index each sentence ends at."""
    words, marks = [], []
    for s in sentences:
        words += norm(s)
        marks.append(len(words))
    return words, marks[:-1]        # the final end is not a decision


truth_sentences, tail = split_sentences(source)
if tail:
    truth_sentences.append(tail)
truth_words, truth_marks = bounds(truth_sentences)


def score(sentences):
    hyp_words, hyp_marks = bounds(sentences)
    # map truth word index -> hyp word index
    sm = difflib.SequenceMatcher(None, truth_words, hyp_words, autojunk=False)
    mapping = {}
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            mapping[a + k] = b + k
    def near(i):
        j = mapping.get(i)
        if j is None:                      # word dropped by ASR; take neighbours
            for d in (1, 2, 3):
                if i - d in mapping: j = mapping[i - d] + d; break
                if i + d in mapping: j = mapping[i + d] - d; break
        return j
    hit = 0
    for m in truth_marks:
        j = near(m)
        if j is not None and any(abs(j - h) <= 2 for h in hyp_marks):
            hit += 1
    extra = 0
    projected = {near(m) for m in truth_marks if near(m) is not None}
    for h in hyp_marks:
        if not any(abs(h - p) <= 2 for p in projected):
            extra += 1
    return hit, len(truth_marks), extra, len(hyp_marks)


def chunked():
    """Today: each VAD segment transcribed on its own."""
    seg = VadSegmenter(vad_silence_ms=500, max_utterance_sec=8, min_utterance_sec=1.5)
    events = []
    for i in range(0, len(audio) - FRAME_SAMPLES, FRAME_SAMPLES):
        events += [e for e in seg.feed(audio[i:i+FRAME_SAMPLES]) if isinstance(e, SpeechEnd)]
    events += [e for e in seg.flush() if isinstance(e, SpeechEnd)]
    asm, out, model = SentenceAssembler(), [], 0.0
    for ev in events:
        t0 = time.perf_counter()
        text = stt.transcribe(ev.audio, language="en").strip()
        model += time.perf_counter() - t0
        out += asm.feed(text)
    out += asm.flush()
    return out, model


def ticked(tick, max_buffer=24.0):
    start, out, model, emitted, at = 0.0, [], 0.0, 0, tick
    while at <= total + tick:
        clip = audio[int(start*sr):int(min(at, total)*sr)]
        if len(clip) < sr * 0.5:
            at += tick; continue
        t0 = time.perf_counter()
        text = stt.transcribe(clip, language="en").strip()
        cost = time.perf_counter() - t0
        model += cost
        asm = SentenceAssembler()
        sentences = asm.feed(text)
        out += sentences[emitted:]
        emitted = len(sentences)
        if not asm.pending or at - start >= max_buffer:
            if asm.pending:
                out.append(asm.pending)
            start, emitted = min(at, total), 0
        at = max(at + tick, at + cost)
    return out, model


print(f"真值 {len(truth_sentences)} 句，{len(truth_words)} 词，音频 {total:.1f}s\n")
print(f"{'方案':<26}{'句数':>5}{'边界对上':>10}{'多切的':>8}{'占空比':>8}")
runs = [("现状：VAD 切段各自转录", chunked()),
        ("节奏 4.0s", ticked(4.0)),
        ("节奏 2.5s", ticked(2.5))]
for label, (sentences, model) in runs:
    hit, need, extra, made = score(sentences)
    print(f"{label:<24}{len(sentences):>6}{hit:>6}/{need:<5}{extra:>6}{model/total*100:>9.0f}%")
print()
for label, (sentences, _) in runs:
    print(f"--- {label} ---")
    for s in sentences:
        print(f"   {s[:96]}")
    print()
