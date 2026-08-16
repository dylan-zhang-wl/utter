# Utter

A local-first speech tool for macOS, built for people who dictate arguments
rather than commands.

Two modes. **Speak** is finished and in daily use: hold a key, talk, and your
words appear at the cursor in whatever application you were already typing in.
**Listen** is not built yet: live transcription of an English lecture or
meeting, translated into Chinese, kept as a document.

[中文说明](README.zh-CN.md)

---

## Why another dictation tool

There are good ones already. This one exists because of a constraint the others
do not take seriously enough for academic work:

> **The text that lands in your document must be the words you said.**

Every dictation tool with an "AI enhancement" step sends your transcript to a
language model and adopts what comes back. Measured here, against models
explicitly instructed not to: GPT and Gemini both deleted 「因为」 — a causal
connective — from an academic argument, and both reordered clauses. The results
read beautifully. That is the danger: a fluent sentence that is no longer your
claim will not be caught on re-reading, and by then the audio is gone.

So Utter's polish step **cannot** change your words. Not "is told not to" —
cannot. The model's output is aligned against the transcript character by
character, and only punctuation is taken from it; every letter comes from what
you actually said. A model that deletes a word has that word restored; a model
that reorders a clause has the order restored; a model that invents a sentence
contributes nothing.

That property is the point of the project. Everything else is engineering.

---

## Speak — what it does

- **Push-to-talk**: hold a key, speak, release. The whole utterance is
  transcribed and pasted at the cursor.
- **Streaming** (off by default): double-tap a key and keep talking. Silero VAD cuts the audio at
  your own pauses and each clause is transcribed and injected while you are
  still speaking the next one — roughly one clause every four seconds, the last
  landing about a second after you stop.
- **Punctuation from your pauses.** Whisper does not punctuate a two-second
  fragment, so streamed clauses are closed by the silence you actually left:
  a breath is a comma, a longer gap is a full stop. Measured, not guessed.
- **A terminology list** that biases the recogniser, so 「foreignisation」 does
  not come back as 「Thorinization」. This is the *only* layer allowed to correct
  a misheard word, because it is the only one that still has the audio.
- **Nothing is ever lost silently.** The timing report after each utterance says
  how long you held the key, how much audio arrived, how much of it was speech,
  and where the text went. If they disagree, it says so.

### What it deliberately does not do

- **Never writes audio to disk.** Recordings live in memory for a few seconds
  and are discarded. This is also why the archive is tiny: 227 bytes per
  utterance, against the tens of gigabytes a year that tools storing audio
  accumulate.
- **Never revises text it has already injected.** No draft-then-correct. Your
  document is yours; a tool that edits behind you is worse than a slow one.
- **Never keeps the microphone open.** It opens on the hotkey and closes after.

---

## Listen — planned, not built

Live transcription of English lectures and meetings, translated into Chinese,
accumulating into a document you can edit and export. The pipeline underneath
it (VAD segmentation, provider abstraction, translation providers) exists and
is tested; the window and the session model do not. See
[docs/plans](docs/plans/) for the design.

Do not expect this to work yet.

---

## Models

Speech recognition runs locally. Models are downloaded on first use, not
bundled — the app is a few megabytes and the weights are hundreds.

| Engine | Chinese | English terminology | Mixed | Speed |
|---|---|---|---|---|
| **Whisper large-v3-turbo (MLX)** | good | **accurate** | **both preserved** | 90–146 ms per audio-second |
| SenseVoice int8 | good, denser punctuation | poor | **drops the English** | 14–20 ms |
| Whisper (CTranslate2) | good | accurate | both preserved | CPU — for Windows and Intel Macs |

Whisper turbo is the default. SenseVoice is four to twelve times faster and
better punctuated in pure Chinese, but in a Chinese sentence containing English
terms it does not mishear them — it *removes* them, silently, which for a
bilingual academic is disqualifying. It stays available and its menu entry says
so.

Polish is optional and runs through any OpenAI-compatible endpoint, Google
Vertex AI (via cloud credentials, no API key), or a local Ollama. Keys live in
the macOS Keychain and never touch a file.

---

## Install

macOS 13+, Apple Silicon or Intel. Python 3.11.

```bash
uv venv ~/.venvs/utter
uv pip install --python ~/.venvs/utter/bin/python \
    --overrides requirements-overrides.txt -r requirements.txt
uv pip install --python ~/.venvs/utter/bin/python --no-deps -e .
```

`--overrides` is not optional: `mlx-whisper` declares a dependency on torch it
does not use, and without the override you get 476 MB of it.

Then build the app:

```bash
# once — a persistent signing certificate, so macOS permissions survive rebuilds
~/.venvs/utter/bin/python packaging/build_app.py --make-cert

~/.venvs/utter/bin/python packaging/build_app.py --login-item
open /Applications/Utter.app
```

macOS will ask for **Accessibility** (the hotkey and paste) and **Microphone**.
Without Accessibility the hotkey receives nothing and raises nothing; without
the microphone macOS hands the app silence rather than an error. The app checks
both and says which is missing.

There is also a CLI, which is the better way to debug:

```bash
utter doctor            # what this machine is and what works on it
utter dictate --timing  # dictation with a per-stage latency report
utter mics              # open every input device and see which actually hears
utter polish '文字'      # run the polish prompt against your real model
```

---

## The rules this project holds itself to

Thirteen of them, in [CLAUDE.md](CLAUDE.md), each written after something went
wrong. The ones that shape the architecture:

- Audio never reaches disk.
- Injected text is never revised.
- Polish may add punctuation, remove filler and paragraph — never reword, never
  add or remove content. The raw transcript is archived alongside the polished
  one.
- API keys only in the Keychain.
- No torch, anywhere.
- Nothing is lost silently.

---

## Lessons

[docs/教训与经验.md](docs/教训与经验.md) is thirty-five findings from building
this, most of them counter-intuitive and several of them expensive. A sample:

- Whisper invents sentences during silence — two seconds of an empty room
  produced "you" and "Good job." Its Chinese hallucination is a subtitle credit.
- A `temperature` ladder must be bounded but not removed: the ladder is what
  escapes repetition loops, and a scalar leaves the model nowhere to go.
- PortAudio enumerates devices once at startup and never again, so a resident
  daemon quietly records silence after you plug in headphones.
- macOS denies a microphone by returning **exact zeros**, not an error. A real
  microphone in a silent room returns about 0.001. A perfect run of zeros is
  never a measurement; it is an `if` in somebody's software.
- A `.app` whose launcher `exec`s away is never seen by LaunchServices as
  having finished launching, and its menu bar icon is never laid out.
- Permissions follow the code signature, so a wrapper app must run an
  interpreter that lives inside the bundle.

If you are building anything with Whisper on macOS, that file will save you
days. It cost them.

---

## Status

Personal software, used daily by its author, published because the lessons are
worth more shared than kept. 775 tests. Speak mode is complete; Listen mode is
designed and not implemented; the bundle is not distributable to other machines
yet (Apple notarisation needs a paid developer account).

Issues and questions welcome. It is not seeking contributors.

## Licence

MIT. See [LICENSE](LICENSE).
