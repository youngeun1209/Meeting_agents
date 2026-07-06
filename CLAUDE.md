# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Conversation language

Converse with the user in **Korean** (한국어). Keep code, comments, identifiers, and commit messages in English as the codebase already does.

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

Audio flows through a threaded producer/consumer pipeline. The pipeline lives in **one place** — `main.STTEngine` — and both entrypoints run it: `main.py` (CLI) and `gui.py` (GUI). There is no duplicated loop.

1. **`audio_capture.py`** (`MultiCapture`) — producer threads. Two source kinds feed one shared `audio_queue` as `(speaker, mono_block)`: (a) `sounddevice` device callbacks (mic), (b) `SystemAudioCapture` — a subprocess running the `native/sysaudio` ScreenCaptureKit sidecar, whose 16kHz-mono-int16 stdout is read, converted to float32, and tagged. Stereo/multichannel is averaged to mono; 16kHz throughout.
2. **`vad_buffer.py`** (`VADBuffer`, one per speaker) — consumer side. `webrtcvad` splits the block stream into utterance chunks on silence gaps (`VAD_SILENCE_SEC`), drops sub-`MIN_CHUNK_SEC` noise, force-splits at `MAX_CHUNK_SEC`.
3. **`transcriber.py`** (`Transcriber`) — chunk → `(lang, text)`. Two backends selected by `config.STT_BACKEND`: `"mlx"` (Apple GPU via `mlx-whisper`) or `"faster-whisper"` (CPU). Three anti-hallucination gates: RMS volume floor (`MIN_RMS`) skips STT entirely, boilerplate-phrase match (`HALLUCINATION_PHRASES`), and repetition-loop detection.
4. **`writer.py`** (`Writer`) — appends each utterance to `transcripts/transcript_*.txt` with immediate `flush()` + `os.fsync()` so nothing is lost on crash/force-quit. Optional `on_line` callback feeds the GUI.
5. **`translator.py` / `minutes.py`** — separate, on-demand stages. They stream the full transcript to Ollama's HTTP API (`urllib`, no extra deps) and return translated transcript text or markdown minutes.

Pipeline changes go in `main.STTEngine.run` only — both entrypoints inherit them. `main()` is a thin CLI (worker thread + Ctrl+C → save); `gui.STTController` is a thin adapter that constructs an `STTEngine` with `on_line`/`on_status` callbacks feeding a queue the UI drains.

## Models in use

| Stage | Default model | Backend / how | Set in |
|-------|--------------|---------------|--------|
| STT | `large-v3-turbo` (Whisper) | mlx-whisper → repo `mlx-community/whisper-large-v3-turbo` (Apple GPU) | `config.MODEL_SIZE`, `config.STT_BACKEND`, `config.MLX_MODEL_MAP` |
| STT (CPU fallback) | same size | faster-whisper, `int8`, `beam_size=1` | `config.DEVICE`, `config.COMPUTE_TYPE`, `config.BEAM_SIZE` |
| Translation | `qwen2.5:7b` | Ollama local LLM, HTTP `localhost:11434`, streaming, `temperature=0.1`, `num_ctx=8192` | `config.OLLAMA_MODEL`, `config.OLLAMA_URL`, `config.OLLAMA_NUM_CTX` |
| Minutes | `qwen2.5:7b` | Ollama local LLM, HTTP `localhost:11434`, streaming, `temperature=0.2`, `num_ctx=8192` | `config.OLLAMA_MODEL`, `config.OLLAMA_URL`, `config.OLLAMA_NUM_CTX` |

STT model sizes: `base` / `small` / `medium` / `large-v3-turbo` / `large-v3`. Language `config.LANGUAGE=None` = auto-detect; pin to `ko`, `en`, or `zh` for Korean, English, or Chinese. Translation prompt is a hardcoded template in `translator.py` (`PROMPT_TEMPLATE`) and preserves timestamps plus speaker tags. Minutes prompt is a hardcoded template in `minutes.py` (`PROMPT_TEMPLATE`) that instructs the LLM to write minutes in the **same language as the transcript** — sections Summary / Discussion / Decisions / Action Items, using `[Me]`/`[Others]` speaker tags to attribute.

## Functions by module

Every `.py` file, every function. Pipeline order: `audio_capture` → `vad_buffer` → `transcriber` → `writer`; `translator`/`minutes` are on-demand; `main` holds the engine, `gui` the UI.

**`audio_capture.py`** — producer side (audio → shared `audio_queue`).
- `find_device(name_substr)` → `(idx, dev)`. First input device (channels > 0) whose name contains the substring; `(None, None)` if none.
- `list_input_devices()` — prints every capturable input device with index/channels/samplerate (`--list-devices` / GUI "Devices").
- `AudioCapture` — single-device `sounddevice` capture. **Legacy, unused by the pipeline.**
  - `__init__()` — own `audio_queue`, no stream yet.
  - `_callback(indata, frames, time_info, status)` — averages stereo→mono, puts the block on the queue.
  - `start()` — resolves `config.INPUT_DEVICE`, opens the stream (raises if device missing).
  - `stop()` — stops + closes the stream.
- `SystemAudioCapture(audio_queue, speaker="Others", binary=None)` — whole system audio via the `native/sysaudio` ScreenCaptureKit sidecar.
  - `start()` — raises if the sidecar binary isn't built; spawns it, launches reader + stderr threads.
  - `_reader()` — slices the sidecar's 16kHz int16 stdout into `BLOCK_SEC` blocks → float32 → `(speaker, block)` onto the queue.
  - `_stderr_relay()` — surfaces sidecar `[sysaudio]` logs (format/permission diagnostics).
  - `stop()` — sets the stop flag, terminates (then kills) the subprocess.
- `MultiCapture(sources)` — **the one in use.** One producer per source, all feeding one queue.
  - `__init__(sources)` — holds the source list, shared `audio_queue`, stream/sidecar/active lists.
  - `_make_callback(speaker)` — returns a `sounddevice` callback that tags device blocks with the speaker.
  - `start()` — `{"kind":"system"}` → `SystemAudioCapture`; `{"device":...}` → a `sounddevice` stream. Warns on missing devices, raises if none opened.
  - `speakers()` → list of active speaker labels.
  - `stop()` — closes all streams and stops all sidecars.

**`vad_buffer.py`** — consumer side, one `VADBuffer` per speaker.
- `_float32_to_pcm16(audio)` — float32 [-1,1] → int16 PCM bytes (webrtcvad needs 16-bit).
- `VADBuffer`
  - `__init__()` — builds the `webrtcvad.Vad`, derives frame/silence/min/max sample counts from `config`.
  - `add(block)` → list of finished utterance chunks. Frames the block, marks speech/silence, ends an utterance on `VAD_SILENCE_SEC` of silence, force-splits at `MAX_CHUNK_SEC`.
  - `_speech_len()` — current accumulated utterance length in samples.
  - `_flush()` — finalizes the current utterance to a chunk; `None` if shorter than `MIN_CHUNK_SEC`.
  - `_reset_speech()` — clears the utterance accumulator + silence counter.
  - `finalize()` — on shutdown, returns the trailing utterance (`None` if none).

**`transcriber.py`** — chunk → `(lang, text)` with anti-hallucination gates.
- `_normalize(text)` — strip punctuation/whitespace, lowercase (for phrase matching).
- `is_hallucination(text)` — True if the whole chunk exactly matches a boilerplate phrase (`HALLUCINATION_PHRASES`) or contains all keywords of any `HALLUCINATION_KEYWORD_SETS` combo (YouTube-outro residue).
- `is_repetitive(text)` — True if unique-word ratio < 0.35 (hallucination loop).
- `rms(audio_float32)` — block volume, for the `MIN_RMS` gate.
- `Transcriber`
  - `__init__()` — picks backend from `config.STT_BACKEND`.
  - `_init_mlx()` — lazy-imports `mlx_whisper`, resolves the repo from `MLX_MODEL_MAP`, warms up with a silent block (front-loads download).
  - `_transcribe_mlx(audio)` → `(lang, text)`.
  - `_init_faster_whisper()` — lazy-imports + loads `WhisperModel` (CPU).
  - `_transcribe_faster_whisper(audio)` → `(lang, text)`.
  - `transcribe(audio)` → `(lang, text)`. RMS gate → backend → text gates (empty text if gated). Both backends use `condition_on_previous_text=False`, `no_speech_threshold=0.6`.

**`writer.py`** — output + crash-safe live saving.
- `Writer(on_line=None, echo=True, started_at=None)`
  - `_ensure_file()` — opens the file on the first utterance; appends if the same `started_at` file exists (resume), else writes a header.
  - `emit(lang, text, speaker=None)` — formats one line; echoes to terminal if `echo`; appends + `flush()` + `os.fsync()`; fires `on_line` callback (GUI).
  - `save_txt()` — already saved live; flushes and returns the path (`None` if nothing captured).
  - `close()` — flushes and closes the file handle (call from the emit thread).

**`translator.py`** — transcript / single-line translation via local LLM (Ollama).
- `TARGET_LANGUAGES` — supported targets (Korean/English/Chinese).
- `PROMPT_TEMPLATE` / `LINE_PROMPT_TEMPLATE` — full-transcript and single-line prompts.
- `_stream_generate(prompt, temperature, num_ctx, on_token=None)` — POSTs `/api/generate` (streaming), returns full text, raises `RuntimeError` on connection failure. Shared by both translate paths.
- `translate_line(text, target_language="Korean", on_token=None)` — one utterance → translated line, short prompt + small context (`num_ctx=2048`) for fast real-time translation. **Used by the GUI's live-translate.**
- `is_available()` — Ollama `/api/tags` ping.
- `translate_transcript(transcript, target_language="Korean", on_token=None)` — whole transcript → translated transcript (streams, preserves timestamps + speaker tags).

**`minutes.py`** — minutes generation via local LLM (Ollama).
- `PROMPT_TEMPLATE` — minutes prompt (same language as transcript; Summary / Discussion / Decisions / Action Items).
- `is_available()` — Ollama `/api/tags` ping.
- `generate_minutes(transcript, when="", on_token=None)` — transcript → markdown minutes (streams `/api/generate`, raises `RuntimeError` on failure).

**`main.py`** — the shared pipeline engine + CLI. **Single source of truth for the pipeline loop.**
- `STTEngine(on_line=None, on_status=None, echo=True)` — producer → VAD → STT → Writer, with live model hot-swap.
  - `request_model(model_size)` — requests a background model swap; the loop reloads and switches without dropping audio.
  - `run(model_size, language, stop_event, session_started=None)` — blocking pipeline loop (run on a worker thread). Loads the model, opens capture, per-speaker VAD → transcribe → `writer.emit`; handles background reload/swap; finalizes tails and closes capture/writer in `finally`. `session_started` reuse = append to the same file.
  - `save()` — returns the transcript path (`None` if nothing captured).
- `main()` — thin CLI. `--list-devices` short-circuits; else spawns `STTEngine(echo=True)` on a worker thread, prints captions via `Writer` echo, Ctrl+C → stop + save.

**`gui.py`** — customtkinter UI. **No pipeline logic.**
- `STTController` — thin adapter over `STTEngine`.
  - `__init__()` — creates the engine with `on_line`/`on_status` → `line_queue`, `echo=False`.
  - `running` (property) — proxies `engine.running`.
  - `request_model` / `stop` / `save` — delegate to the engine.
  - `_emit_line` / `_status` — push `("line"|"status", …)` onto `line_queue`.
  - `start(model_size, language, session_started=None)` — spawns the worker running `_run_engine`.
  - `_run_engine(...)` — runs `engine.run`, catches exceptions → `("error", …)` on the queue.
- `App` — the whole window.
  - Build: `_build_toolbar` (Start/Stop/New, model+language menus), `_build_statusbar` (status dot, Save/Devices), `_build_tabs` + `_build_stt_tab` / `_build_live_tab` / `_build_translator_tab` / `_build_minutes_tab`.
  - Session: `_on_start`, `_on_stop`, `_on_new`, `_on_save`, `_clear_captions`, `_idle_buttons`.
  - Dropdowns: `_on_lang_change` (mutates `config.LANGUAGE`), `_on_model_change` (live model swap via the controller).
  - Real-time translation (Translator tab): `_on_tab_change` (opening the tab turns live-translate on, leaving turns it off), `_enable_live_translate`, `_ensure_live_translate_worker`, `_pump_live_translate` (queues untranslated utterances), `_live_translate_worker` (translates one line at a time via `translator.translate_line`, pushes `("ltline", block)`).
  - Full translate: `_on_translate`, `_translation_worker` (streams `translate_transcript`), `_on_save_translation`.
  - Minutes: `_on_make_minutes`, `_minutes_worker` (streams `generate_minutes`), `_on_save_minutes`.
  - Render: `_speaker_tag`, `_append` (adds a caption; accumulates `session_lines` + `_utterances`; pumps live-translate), `_update_live_summary` / `_render_live_summary` (per-speaker merge, no LLM), `_append_minutes_token`, `_append_translation_token`, `_set_translation_text`, `_set_minutes_text`, `_set_status`, `_toast`.
  - Loop: `_poll_queue` (drains `line_queue` / `translation_queue` / `minutes_queue` onto the UI every 100ms), `_on_close`.
- `main()` — creates the `CTk` root, builds `App`, runs `mainloop`.

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
