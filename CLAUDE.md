# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Real-time, local, free STT for Zoom meetings / online lectures. Captures audio, transcribes with Whisper, writes `.txt` live, translates transcripts, and can summarize into meeting minutes via a local LLM (Ollama). Korean/English/Chinese auto-detect. macOS-focused. UI and code comments are in English.

## Commands

```bash
# Setup
brew install portaudio                     # sounddevice needs PortAudio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# System-audio sidecar (default "Others" capture — no BlackHole needed)
bash native/build.sh                       # compiles native/sysaudio (Swift/ScreenCaptureKit)
                                           # first run → grant Screen Recording permission

python gui.py                              # GUI (recommended) — tkinter/customtkinter
python main.py                             # CLI — captions to terminal, Ctrl+C saves txt
python main.py --list-devices              # list input devices (fix config.SOURCES names)

# Translation/minutes features (optional, local LLM)
brew install ollama && ollama serve        # keep running in a separate terminal
ollama pull qwen2.5:7b                      # model set in config.OLLAMA_MODEL
```

No test suite or linter. The only build step is `native/build.sh` (the Swift system-audio sidecar); verify code changes by running the app against live audio.

## Speaker separation — the core design idea

Diarization is avoided entirely. Instead, each speaker is a *separate audio source*:
- **Others** (meeting/lecture audio) = **the whole system output**, captured directly via a **ScreenCaptureKit sidecar** (`native/sysaudio`, `{"kind": "system"}` in `config.SOURCES`). No virtual audio device needed — just grant Screen Recording permission once. This is the default.
- **Me** = physical microphone (a `sounddevice` device source).

`MultiCapture` runs one producer per source — a `sounddevice` stream per device, or the sidecar subprocess for system audio — and tags every audio block with its speaker label → 100% accurate speaker attribution without ML. Mapping lives in `config.SOURCES`.

**BlackHole is now a fallback, not the default.** The old routing (Others = **BlackHole 2ch** via a **Multi-Output Device**) still works if you swap `config.SOURCES` to the device-based block (commented in `config.py`), but ScreenCaptureKit avoids the audio-routing setup entirely. See [docs/system-audio-capture.md](docs/system-audio-capture.md).

## Pipeline architecture (producer → consumer)

Audio flows through a threaded producer/consumer pipeline, shared by both `main.py` (CLI) and `gui.py` (GUI wraps the same stages in `STTController`):

1. **`audio_capture.py`** (`MultiCapture`) — producer threads. Two source kinds feed one shared `audio_queue` as `(speaker, mono_block)`: (a) `sounddevice` device callbacks (mic), (b) `SystemAudioCapture` — a subprocess running the `native/sysaudio` ScreenCaptureKit sidecar, whose 16kHz-mono-int16 stdout is read, converted to float32, and tagged. Stereo/multichannel is averaged to mono; 16kHz throughout.
2. **`vad_buffer.py`** (`VADBuffer`, one per speaker) — consumer side. `webrtcvad` splits the block stream into utterance chunks on silence gaps (`VAD_SILENCE_SEC`), drops sub-`MIN_CHUNK_SEC` noise, force-splits at `MAX_CHUNK_SEC`.
3. **`transcriber.py`** (`Transcriber`) — chunk → `(lang, text)`. Two backends selected by `config.STT_BACKEND`: `"mlx"` (Apple GPU via `mlx-whisper`) or `"faster-whisper"` (CPU). Three anti-hallucination gates: RMS volume floor (`MIN_RMS`) skips STT entirely, boilerplate-phrase match (`HALLUCINATION_PHRASES`), and repetition-loop detection.
4. **`writer.py`** (`Writer`) — appends each utterance to `transcripts/transcript_*.txt` with immediate `flush()` + `os.fsync()` so nothing is lost on crash/force-quit. Optional `on_line` callback feeds the GUI.
5. **`translator.py` / `minutes.py`** — separate, on-demand stages. They stream the full transcript to Ollama's HTTP API (`urllib`, no extra deps) and return translated transcript text or markdown minutes.

`main.py` `consume_loop` and `gui.py` `STTController._run` are two implementations of the same loop; keep pipeline changes in sync across both.

## Models in use

| Stage | Default model | Backend / how | Set in |
|-------|--------------|---------------|--------|
| STT | `large-v3-turbo` (Whisper) | mlx-whisper → repo `mlx-community/whisper-large-v3-turbo` (Apple GPU) | `config.MODEL_SIZE`, `config.STT_BACKEND`, `config.MLX_MODEL_MAP` |
| STT (CPU fallback) | same size | faster-whisper, `int8`, `beam_size=1` | `config.DEVICE`, `config.COMPUTE_TYPE`, `config.BEAM_SIZE` |
| Translation | `qwen2.5:7b` | Ollama local LLM, HTTP `localhost:11434`, streaming, `temperature=0.1`, `num_ctx=8192` | `config.OLLAMA_MODEL`, `config.OLLAMA_URL`, `config.OLLAMA_NUM_CTX` |
| Minutes | `qwen2.5:7b` | Ollama local LLM, HTTP `localhost:11434`, streaming, `temperature=0.2`, `num_ctx=8192` | `config.OLLAMA_MODEL`, `config.OLLAMA_URL`, `config.OLLAMA_NUM_CTX` |

STT model sizes: `base` / `small` / `medium` / `large-v3-turbo` / `large-v3`. Language `config.LANGUAGE=None` = auto-detect; pin to `ko`, `en`, or `zh` for Korean, English, or Chinese. Translation prompt is a hardcoded template in `translator.py` (`PROMPT_TEMPLATE`) and preserves timestamps plus speaker tags. Minutes prompt is a hardcoded template in `minutes.py` (`PROMPT_TEMPLATE`) that instructs the LLM to write minutes in the **same language as the transcript** — sections Summary / Discussion / Decisions / Action Items, using `[Me]`/`[Others]` speaker tags to attribute.

## Functions by module

**`audio_capture.py`**
- `find_device(name_substr)` → `(idx, dev)` by substring, input channels > 0.
- `list_input_devices()` — prints capturable devices (`--list-devices` / GUI "Devices").
- `AudioCapture` — single-device capture (legacy, unused by pipeline). `_callback`, `start`, `stop`.
- `SystemAudioCapture(audio_queue, speaker, binary=None)` — spawns `native/sysaudio` (raises if not built), reader thread slices its stdout int16 PCM into `BLOCK_SEC` blocks → float32 → `(speaker, block)`; `_stderr_relay` surfaces sidecar `[sysaudio]` logs (format/permission); `stop()` terminates the subprocess.
- `MultiCapture(sources)` — **the one in use.** Handles both source kinds: `{"kind":"system"}` → a `SystemAudioCapture`; `{"device":...}` → a `sounddevice` stream. `_make_callback(speaker)` tags device blocks; `start()` opens all sources (warns on missing devices, raises if none opened); `speakers()` lists active labels; `stop()` closes streams and sidecars.

**`vad_buffer.py`**
- `_float32_to_pcm16(audio)` — float32 → int16 PCM bytes (webrtcvad needs 16-bit).
- `VADBuffer` — one per speaker. `add(block)` → list of finished chunks; `_flush()` finalizes current utterance (None if < `MIN_CHUNK_SEC`); `_speech_len`, `_reset_speech`; `finalize()` returns trailing utterance on shutdown.

**`transcriber.py`**
- `_normalize(text)`, `is_hallucination(text)`, `is_repetitive(text)` — text-side hallucination gates (boilerplate match + low unique-word-ratio loop detection).
- `rms(audio)` — volume, for the `MIN_RMS` gate.
- `Transcriber` — `__init__` picks backend; `_init_mlx`/`_transcribe_mlx`, `_init_faster_whisper`/`_transcribe_faster_whisper`; public `transcribe(audio)` → `(lang, text)` runs RMS gate → backend → text gates. Both backends use `condition_on_previous_text=False`, `no_speech_threshold=0.6`.

**`writer.py`** — `Writer(on_line, echo, started_at)`: `_ensure_file` (append if same `started_at` = resume, else header), `emit(lang, text, speaker)` (print + fsync line), `save_txt()`, `close()`.

**`translator.py`** — `TARGET_LANGUAGES`, `is_available()` (Ollama `/api/tags` ping), `translate_transcript(transcript, target_language, on_token)` (streams `/api/generate`, raises `RuntimeError` on connection failure).

**`minutes.py`** — `is_available()` (Ollama `/api/tags` ping), `generate_minutes(transcript, when, on_token)` (streams `/api/generate`, raises `RuntimeError` on connection failure).

**`main.py`** — `consume_loop(capture, stt, out, stop_event)` (CLI pipeline loop), `main()` (wires threads, Ctrl+C → save).

**`gui.py`** — `STTController` runs the pipeline in a worker thread, passes lines to GUI via `line_queue`; `request_model()` triggers no-drop hot model swap. `App` builds customtkinter UI (4 tabs: Live STT / Live Summary / Translator / Minutes), `_poll_queue` drains queues onto the UI, `_update_live_summary` merges consecutive same-speaker turns (no LLM).

## Configuration

`config.py` is the single source of truth — nearly all behavior is tunable there (backend, model, VAD thresholds, sources, Ollama). The GUI **mutates `config` module globals at runtime** (`config.MODEL_SIZE`, `config.LANGUAGE`) when dropdowns change; the worker hot-swaps models without dropping audio (loads new model in the background, switches when ready). Treat `config` as live mutable state, not just static defaults.

## Audio routing (macOS operational setup)

**Default (ScreenCaptureKit): no routing at all.** With `{"kind":"system"}` in `config.SOURCES`, the `native/sysaudio` sidecar taps system output directly. One-time setup: `bash native/build.sh`, then grant **Screen Recording** permission on first run (System Settings → Privacy & Security → Screen Recording). No BlackHole, no Multi-Output Device. See [docs/system-audio-capture.md](docs/system-audio-capture.md).

**Fallback (BlackHole):** only if the sidecar can't be used. Requires routing system output to BlackHole; `docs/` covers two methods:
- **Multi-Output Device** (`docs/multi-output-setup.md`) — output goes to earphones + BlackHole simultaneously. Simple, but hardware volume keys stop working.
- **LadioCast** (`docs/ladiocast-setup.md`) — system output fixed to BlackHole, LadioCast forwards to hardware. Volume keys keep working; must stay running.
- `docs/run-guide.md` — end-user run checklist and a symptom/cause/fix table.

**Earphones vs. built-in speakers:** with the BlackHole fallback (or any mic that hears the room), speaker audio leaks into the mic and the other party's voice gets mislabeled as `Me` — use earphones. The ScreenCaptureKit path taps the audio stream before the speaker, so leakage is only a concern for the mic source.

## Design doc is historical

[docs/design-spec.md](docs/design-spec.md) is the original (2026-06-29) build spec and no longer matches the code. It targets **Intel Xeon / CPU / faster-whisper / `small` model** and lists a 4-stage CLI-only pipeline. The shipped code moved to **mlx / Apple GPU / `large-v3-turbo`** by default and added the GUI, speaker separation, minutes, and hallucination filters. Read it for original intent, not current behavior — trust the code and this file over it.

## Gotchas

- **`requirements.txt` does not install the default backend.** `config.STT_BACKEND` defaults to `"mlx"`, which needs `mlx-whisper` (not listed). Either `pip install mlx-whisper` or set `STT_BACKEND="faster-whisper"`. `MLX_MODEL_MAP` maps `MODEL_SIZE` → HuggingFace mlx-community repo; add an entry when introducing a new size for the mlx path.
- Device names in `config.SOURCES` use substring match; confirm exact names via `--list-devices` when audio doesn't capture.
- **The system-audio sidecar must be built before first use.** `SystemAudioCapture` raises if `native/sysaudio` is missing → run `bash native/build.sh` (needs Xcode command-line tools / `swiftc`, macOS 13+, arm64). First launch triggers the Screen Recording permission prompt; **no audio flows until granted and the app is restarted.** The sidecar logs its real input format to stderr (`[sysaudio] input format: …Hz`); if it isn't 16000 Hz the block math in `SystemAudioCapture._reader` is off (no resampling) — treat a non-16kHz warning as a real bug.
- `webrtcvad` requires `setuptools<81` (pinned) because it uses the removed `pkg_resources` API.
- Models auto-download on first run (hundreds of MB); the mlx backend warms up with a silent block at startup to front-load the download.
