# Live STT (real-time, local, free)

Transcribe Zoom meetings / online lecture audio in real time and save it to `.txt`.
Korean/English auto-detect. **Speaker separation** (Me / Others) + local-LLM meeting minutes.
macOS-focused, runs entirely on your machine — no cloud, no cost.

## How speaker separation works

No diarization. Each speaker is a *separate audio source*:
- **Others** (meeting/lecture audio) = the whole **system output**, captured directly via a **ScreenCaptureKit sidecar** (`native/sysaudio`). No virtual audio device needed — just grant Screen Recording permission once. This is the default.
- **Me** = the physical **microphone**, captured directly.

The two sources are tagged separately -> 100% accurate speaker attribution without any ML.
Device/speaker mapping lives in `SOURCES` in [config.py](config.py).

> **BlackHole is a fallback, not the default.** The old routing (Others = BlackHole 2ch via a Multi-Output Device) still works — see [docs/system-audio-capture.md](docs/system-audio-capture.md).

## Layout

```
config.py         # all settings
audio_capture.py  # capture -> queue (producers: mic + system-audio sidecar)
vad_buffer.py     # split the block stream into utterance chunks on silence
transcriber.py    # Whisper wrapper (mlx / faster-whisper)
writer.py         # output (terminal/GUI) + real-time txt saving
minutes.py        # minutes generation (local LLM: Ollama)
main.py           # CLI, Ctrl+C handling
gui.py            # GUI — tabs (Live STT / Live Summary / Minutes) (customtkinter)
native/sysaudio.swift  # ScreenCaptureKit system-audio sidecar
```

## Install

```bash
brew install portaudio                     # sounddevice needs PortAudio
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install mlx-whisper                     # default backend (Apple GPU); skip if using faster-whisper

# Build the system-audio sidecar (default "Others" capture — no BlackHole needed)
bash native/build.sh                        # needs Xcode command-line tools (swiftc), macOS 13+, arm64
```

> The default `STT_BACKEND="mlx"` needs `mlx-whisper` (Apple silicon). On other hardware set `STT_BACKEND="faster-whisper"` in [config.py](config.py) — that backend is already in `requirements.txt`.
> Models auto-download on first run (hundreds of MB).

## Usage

### GUI (recommended)
```bash
python gui.py
```
On first launch, macOS asks for **Screen Recording** permission (needed to capture system audio) — allow it, then restart the app.

Three tabs:
- **🎙 Live STT**: ▶ Start / ■ Stop / 🆕 New session, model & language selection. Captions scroll live, colored per speaker. **Every utterance is auto-saved to txt** (nothing is lost even if the app crashes).
- **⚡ Live Summary**: live per-speaker accumulation, no LLM.
- **📝 Minutes**: press **✨ Generate** and a local LLM turns the transcript into clean minutes -> **💾 Save** (`minutes_*.md`).

`🆕 New session` = clear the screen + start a new file (for the next meeting).

#### Minutes feature = needs a local LLM (Ollama) (one-time install)
```bash
brew install ollama
ollama serve                 # keep running in a separate terminal
ollama pull qwen2.5:7b       # good multilingual model (~4.7GB). Use qwen2.5:3b for something lighter.
```
- Fully free, local, offline (meeting content is never sent anywhere)
- Change the model via `OLLAMA_MODEL` in [config.py](config.py)
- If you only need STT, Ollama is optional (only the Minutes tab is disabled without it)
- Minutes are written in the same language as the transcript (Korean audio -> Korean minutes, English -> English)

### CLI
```bash
python main.py                 # run, captions in the terminal
python main.py --list-devices  # list input devices
```
- Play Zoom / a lecture -> live captions
- `Ctrl+C` -> saves `transcripts/transcript_*.txt`

If a device name differs from the list, match the `device` value in `SOURCES` in [config.py](config.py).
To capture only the other party (no mic), delete the microphone line in `SOURCES`.

## Tuning (config.py)

| Symptom | What to change |
|------|--------|
| Transcription lags | `MODEL_SIZE="base"`, `BEAM_SIZE=1`, lower `MAX_CHUNK_SEC` |
| Short phrases get the wrong language | raise `VAD_SILENCE_SEC` |
| Phantom text on silence | raise `VAD_AGGRESSIVENESS` / raise `MIN_RMS` |
| Single-language lecture | `LANGUAGE="ko"` or `"en"` |
| Prefer accuracy | `MODEL_SIZE="medium"` or `"large-v3-turbo"` |
