"""Sweep the ceiling only. No architecture change, no iron law touched.

A lower ceiling means text can appear sooner, but it also means more cuts land
mid-sentence, which the assembler answers by holding — so the win may be
given straight back. Worth knowing before recommending a number.
"""
import re, sys, time, difflib
import numpy as np, soundfile as sf
sys.path.insert(0, "/Users/d/Desktop/华工大/cc cowork/utter")
from backend.sentences import SentenceAssembler, split_sentences
from backend.providers.mlx import MlxWhisperProvider
from backend.vad import VadSegmenter, SpeechEnd, SAMPLE_RATE, FRAME_SAMPLES

HERE = "/private/tmp/claude-501/-Users-d-Desktop-----cc-cowork-utter/fcd94861-ba29-4991-bb2c-5ee806b006b0/scratchpad"
audio, sr = sf.read(f"{HERE}/speech.wav", dtype="float32"); total = len(audio)/sr
source = open(f"{HERE}/speech.txt").read()
stt = MlxWhisperProvider("balanced"); stt.transcribe(np.zeros(sr, dtype="float32"), language="en")

norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).split()
def bounds(ss):
    w, m = [], []
    for s in ss: w += norm(s); m.append(len(w))
    return w, m[:-1]
truth, tail = split_sentences(source)
if tail: truth.append(tail)
tw, tm = bounds(truth)
def score(ss):
    hw, hm = bounds(ss)
    sm = difflib.SequenceMatcher(None, tw, hw, autojunk=False); mp = {}
    for a,b,n in sm.get_matching_blocks():
        for k in range(n): mp[a+k] = b+k
    def near(i):
        if i in mp: return mp[i]
        for d in (1,2,3):
            if i-d in mp: return mp[i-d]+d
            if i+d in mp: return mp[i+d]-d
    hit = sum(1 for m in tm if near(m) is not None and any(abs(near(m)-h)<=2 for h in hm))
    proj = {near(m) for m in tm if near(m) is not None}
    return hit, len(tm), sum(1 for h in hm if not any(abs(h-p)<=2 for p in proj))

def run(ceiling):
    seg = VadSegmenter(vad_silence_ms=500, max_utterance_sec=ceiling, min_utterance_sec=1.5)
    evs = []
    for i in range(0, len(audio)-FRAME_SAMPLES, FRAME_SAMPLES):
        evs += [e for e in seg.feed(audio[i:i+FRAME_SAMPLES]) if isinstance(e, SpeechEnd)]
    evs += [e for e in seg.flush() if isinstance(e, SpeechEnd)]
    asm, out, model, shown = SentenceAssembler(), [], 0.0, []
    for ev in evs:
        t0 = time.perf_counter()
        text = stt.transcribe(ev.audio, language="en").strip()
        cost = time.perf_counter()-t0; model += cost
        got = asm.feed(text)
        out += got
        for _ in got: shown.append(ev.end_sample/SAMPLE_RATE + cost)
    rest = asm.flush(); out += rest
    for _ in rest: shown.append(total)
    gaps = [b-a for a, b in zip(shown, shown[1:])] or [0]
    return out, model, max(gaps), len(evs)

print(f"真值 {len(truth)} 句\n")
print(f"{'天花板':>6}{'段数':>6}{'句数':>6}{'边界对上':>10}{'多切的':>8}{'占空比':>8}{'空白最久':>10}")
for c in (8, 6, 5, 4):
    out, model, gap, n = run(c)
    hit, need, extra = score(out)
    print(f"{c:>5}s{n:>6}{len(out):>6}{hit:>6}/{need:<4}{extra:>6}{model/total*100:>8.0f}%{gap:>9.1f}s")
