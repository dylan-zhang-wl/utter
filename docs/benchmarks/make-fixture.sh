#!/usr/bin/env bash
# Generate a reproducible speech fixture for Utter's latency checks.
#
# Why synthesised speech: the repo must not carry audio (铁律 3), and the
# original bench60.wav was never committed. `say` gives a byte-identical file
# on every run, so a latency number from one machine is comparable with another.
#
# VALID FOR LATENCY ONLY. With temperature=0.0 there is no fallback retry and
# Whisper pads every chunk to 30s regardless of content, so per-chunk cost is
# near content-independent — which is the thing being measured. Synthetic speech
# has no accent, no disfluency and no room noise, so it says NOTHING about
# accuracy. Do not compute WER from this file. Accuracy work needs the author's
# own recording and belongs to P2b (design §9 open item 3).
#
# Writes outside the repo: this repo is in iCloud and the wav is regenerable.
set -euo pipefail

OUT_DIR="${UTTER_FIXTURE_DIR:-${TMPDIR:-/tmp}/utter-fixtures}"
OUT="$OUT_DIR/bench-say-en.wav"
mkdir -p "$OUT_DIR"

for tool in say ffmpeg; do
  command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 1; }
done

# Prefer a plain en_GB/en_US voice; fall back to the system default. The novelty
# voices (Bells, Boing, ...) are useless here and sort first alphabetically.
VOICE=""
for candidate in Daniel Serena Alex Samantha; do
  if say -v '?' | awk '{print $1}' | grep -qx "$candidate"; then
    VOICE="$candidate"; break
  fi
done

# Sentences separated by explicit 700ms silences so the VAD segmenter has real
# boundaries to find. Content is deliberately in this project's own subject area
# so the transcript is easy to eyeball for gross errors.
read -r -d '' TEXT <<'EOF' || true
Translation is not a transparent window onto an original text. [[slnc 700]]
Every choice a translator makes leaves a trace in the target language. [[slnc 700]]
Lawrence Venuti called this the translator's invisibility. [[slnc 700]]
Domestication makes the foreign text read fluently in the receiving culture. [[slnc 700]]
Foreignisation, by contrast, keeps the reader aware that the text came from elsewhere. [[slnc 700]]
Neither strategy is neutral, and neither is simply correct. [[slnc 700]]
The question is what each one costs, and who pays that cost. [[slnc 700]]
This recording exists only to measure transcription latency. [[slnc 700]]
It is synthetic speech, so it tells you nothing about accuracy. [[slnc 700]]
Real accuracy testing needs a real voice in a real room.
EOF

AIFF="$(mktemp -t utter-fixture).aiff"
trap 'rm -f "$AIFF"' EXIT

if [ -n "$VOICE" ]; then
  say -v "$VOICE" -o "$AIFF" "$TEXT"
else
  say -o "$AIFF" "$TEXT"
fi

ffmpeg -loglevel error -y -i "$AIFF" -ar 16000 -ac 1 -c:a pcm_s16le "$OUT"

DUR="$(ffprobe -loglevel error -show_entries format=duration -of csv=p=0 "$OUT")"
printf 'fixture: %s\nvoice:   %s\nlength:  %.1fs @ 16kHz mono\n' \
  "$OUT" "${VOICE:-<system default>}" "$DUR"
