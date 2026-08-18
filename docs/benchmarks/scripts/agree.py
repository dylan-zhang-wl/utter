"""Tick + LocalAgreement-2: emit only what two consecutive runs agree on.

The plain tick splits sentences at the truncation and invents words there
(「Pretty much everything.」, 「Chrissy was weak.」). Those artefacts exist in
exactly one run — the next run, with more audio, gets that stretch right. So
committing only the prefix two runs agree on should suppress them, which is
what ufal/whisper_streaming does and what the previous simulation skipped.

Cost: a word needs two runs to be confirmed, so latency gains one tick.
"""
import re, sys, time, difflib
import numpy as np
import soundfile as sf

sys.path.insert(0, "/Users/d/Desktop/华工大/cc cowork/utter")
from backend.sentences import SentenceAssembler, split_sentences
from backend.providers.mlx import MlxWhisperProvider

HERE = "/private/tmp/claude-501/-Users-d-Desktop-----cc-cowork-utter/fcd94861-ba29-4991-bb2c-5ee806b006b0/scratchpad"
audio, sr = sf.read(f"{HERE}/speech.wav", dtype="float32")
total = len(audio) / sr
source = open(f"{HERE}/speech.txt").read()
stt = MlxWhisperProvider("balanced")
stt.transcribe(np.zeros(sr, dtype="float32"), language="en")

norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).split()

def bounds(sentences):
    words, marks = [], []
    for s in sentences:
        words += norm(s); marks.append(len(words))
    return words, marks[:-1]

truth, tail = split_sentences(source)
if tail: truth.append(tail)
tw, tm = bounds(truth)

def score(sentences):
    hw, hm = bounds(sentences)
    sm = difflib.SequenceMatcher(None, tw, hw, autojunk=False)
    mapping = {}
    for a, b, n in sm.get_matching_blocks():
        for k in range(n): mapping[a+k] = b+k
    def near(i):
        j = mapping.get(i)
        if j is None:
            for d in (1,2,3):
                if i-d in mapping: return mapping[i-d]+d
                if i+d in mapping: return mapping[i+d]-d
        return j
    hit = sum(1 for m in tm if near(m) is not None and any(abs(near(m)-h)<=2 for h in hm))
    proj = {near(m) for m in tm if near(m) is not None}
    extra = sum(1 for h in hm if not any(abs(h-p)<=2 for p in proj))
    # word error against truth, on the emitted text only
    wrong = sum(1 for tag,_,_,_,_ in sm.get_opcodes() if tag != "equal")
    return hit, len(tm), extra, len(hw)

def common_prefix(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n].lower().strip(".,!?") == b[n].lower().strip(".,!?"):
        n += 1
    return n

def agreed(tick, max_buffer=24.0):
    start, out, model, emitted, at = 0.0, [], 0.0, 0, tick
    previous, confirmed_text = [], ""
    while at <= total + tick:
        clip = audio[int(start*sr):int(min(at, total)*sr)]
        if len(clip) < sr*0.5:
            at += tick; continue
        t0 = time.perf_counter()
        text = stt.transcribe(clip, language="en").strip()
        cost = time.perf_counter() - t0; model += cost
        words = text.split()
        done = at >= total                      # last round: nothing more is coming
        n = len(words) if done else common_prefix(previous, words)
        confirmed_text = " ".join(words[:n])
        previous = words

        asm = SentenceAssembler()
        sentences = asm.feed(confirmed_text)
        out += sentences[emitted:]
        emitted = len(sentences)
        if (not asm.pending and n and n == len(words)) or at - start >= max_buffer:
            if asm.pending: out.append(asm.pending)
            start, emitted, previous = min(at, total), 0, []
        at = max(at + tick, at + cost)
    return out, model

print(f"真值 {len(truth)} 句\n")
print(f"{'方案':<30}{'句数':>5}{'边界对上':>10}{'多切的':>8}{'占空比':>8}")
for tick in (4.0, 2.5):
    s, model = agreed(tick)
    hit, need, extra, _ = score(s)
    print(f"{'节奏 '+str(tick)+'s + 双次印证':<28}{len(s):>6}{hit:>6}/{need:<5}{extra:>6}{model/total*100:>9.0f}%")
    print("   " + "\n   ".join(x[:96] for x in s) + "\n")
