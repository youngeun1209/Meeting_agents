"""Settings. Change behavior here and nowhere else."""
import os as _os

import settings as _settings

# --- STT backend ---
# "mlx"           = Apple silicon GPU (M1/M2/M3). turbo runs in real time. (recommended)
# "faster-whisper" = CPU based. For Macs/other platforms without a GPU.
STT_BACKEND = "mlx"

# --- STT model ---
MODEL_SIZE = "large-v3-turbo"  # base / small / medium / large-v3-turbo / large-v3
                            #  turbo = large-level accuracy + fast. Real time on mlx, slow on CPU.
DEVICE = "cpu"              # faster-whisper only. This Mac has no GPU -> cpu
COMPUTE_TYPE = "int8"       # optimal quantization for faster-whisper on CPU
BEAM_SIZE = 1               # 1~2 for real time (lower = faster)

# mlx keeps freed Metal buffers in a cache pool (unified memory). Trim it once the pool
# exceeds this, so a long session doesn't creep into swap and beach-ball the UI. 512MB.
MLX_CACHE_LIMIT_BYTES = 512 * 1024 * 1024

# mlx-whisper model repo mapping (MODEL_SIZE -> HuggingFace mlx-community repo)
MLX_MODEL_MAP = {
    "base":            "mlx-community/whisper-base-mlx",
    "small":           "mlx-community/whisper-small-mlx",
    "medium":          "mlx-community/whisper-medium-mlx",
    "large-v3-turbo":  "mlx-community/whisper-large-v3-turbo",
    "large-v3":        "mlx-community/whisper-large-v3-mlx",
}

# --- Language ---
# None = auto-detect. Can also pin to "ko", "en", or "zh".
LANGUAGE = None

# --- Audio input ---
# Capture is split per speaker into separate sources. Two source kinds:
#   {"kind": "system", ...}  = whole system audio (the other party / lecture sound).
#                              Uses the ScreenCaptureKit sidecar (native/sysaudio).
#                              -> No BlackHole / Multi-Output Device setup. Screen Recording permission once on first run.
#   {"device": "...", ...}   = sounddevice device (microphone = your own voice). Device name is a substring match.
# Confirm device names with `python main.py --list-devices` and match them here.
SOURCES = [
    {"kind": "system",                     "speaker": "Others"},
    {"device": "MacBook Pro Microphone",   "speaker": "Me"},
]
# To capture only the other party (no mic), delete the microphone line above.
#
# [Legacy fallback] To use BlackHole + a Multi-Output Device instead of the ScreenCaptureKit sidecar:
# SOURCES = [
#     {"device": "BlackHole 2ch",          "speaker": "Others"},
#     {"device": "MacBook Pro Microphone", "speaker": "Me"},
# ]

INPUT_DEVICE = "BlackHole 2ch"  # (legacy compatibility, unused)
SAMPLE_RATE = 16000             # Whisper expects 16kHz mono
BLOCK_SEC = 0.1                 # length read per callback (seconds)

# --- VAD (speech activity detection) ---
VAD_AGGRESSIVENESS = 2     # webrtcvad 0~3 (higher = stricter on silence). 3 cuts into real speech.
VAD_FRAME_MS = 30          # webrtcvad frame length: only 10/20/30 allowed
VAD_SILENCE_SEC = 0.7      # silence this long = "end of utterance"
MIN_CHUNK_SEC = 0.4        # utterances shorter than this are dropped (noise)
MAX_CHUNK_SEC = 12         # force-split long utterances. Longer = more context/accuracy, but long monologues lag more.

# If chunk RMS (volume) is below this, skip STT -> blocks silence/noise hallucinations at the source.
# This is the ONLY reliable gate: on near-silent mic noise Whisper hallucinates "Thank you"/"I love that"
# with no_speech_prob=0.0 and good logprob (it is *confident* in the garbage) -> confidence gates don't help.
# Measured silent-mic room noise ~0.006 RMS, so 0.005 let it through. 0.015 gates it; real speech is 0.02-0.1.
# Raise if hallucinations persist; lower (0.008) if quiet speech is being dropped.
MIN_RMS = 0.015

# --- Output / saving ---
# Everything the app produces -- transcripts, per-speaker WAVs, minutes, translations -- goes
# here. The default is a folder next to the app; saving somewhere else once (the folder button,
# or any Save dialog) makes that folder the default for every save afterwards, including after
# a restart. The user's choice lives in settings.json, so this stays the *default* only.
DEFAULT_OUTPUT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "transcripts")
OUTPUT_DIR = _settings.output_dir(DEFAULT_OUTPUT_DIR)
PRINT_LANG = True            # show the detected language in the terminal

# Also save the raw captured audio as WAV, one file per speaker, alongside the transcript.
# Disk cost: 16kHz mono int16 = ~1.9 MB/min per source, so ~115 MB/hour with two sources
# (Me + Others). Turn off if disk space is tight and only the transcript is needed.
SAVE_AUDIO = True

# --- Batch mode (audio file -> re-transcribe + speaker diarization) ---
# The realtime path tags speakers by AUDIO SOURCE (mic = "Me", system audio = "Others"), which is
# exact when it works -- but an in-person meeting has no system audio, so both people land on the
# one mic and every line comes out "[Me]". Batch mode solves that properly: it re-transcribes the
# saved WAV and separates speakers from the audio itself (voice characteristics), which also fixes
# a second problem -- realtime timestamps are stamped when transcription FINISHED (writer.py), 5-10s
# after the words were actually spoken, so they cannot be aligned against anything.
DIARIZATION_DIR = "models"

# pyannote segmentation-3.0, exported to ONNX. This is the SAME segmentation model pyannote 3.1
# uses; only the embedding + clustering stages differ.
#   https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
DIARIZATION_SEGMENTATION = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"

# Speaker embedding model. Measured on a 2-speaker fixture: WeSpeaker ResNet34 matched 3D-Speaker
# ERes2NetV2 at 99.0% frame accuracy while being 25MB vs 68MB and ~2x faster -> it is the default.
# Swap to "3dspeaker_eres2netv2.onnx" (or a larger WeSpeaker model) if real meeting audio -- which
# is much harder than the fixture: one mic across a room, reverb, overlapping speech -- proves it
# insufficient.
#   https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx
DIARIZATION_EMBEDDING = "wespeaker_resnet34.onnx"

# Cosine-distance threshold used ONLY when NUM_SPEAKERS is None. Lower = stricter = more, smaller
# clusters (risks splitting one person in two); higher = looser (risks merging two people into one).
# MEASURED on a 2-speaker fixture with clearly distinct voices: 0.2/0.3/0.4 all recover exactly 2
# speakers, while 0.5 and above collapse both into ONE. The cliff sits between 0.4 and 0.5, so 0.35
# is chosen to sit inside the working range with margin rather than on its edge.
# CAVEAT: that fixture is clean synthetic speech. Real meeting audio -- one mic across a room,
# reverb, overlapping speech, similar voices -- may need a different value, so re-check this against
# a real recording before trusting Auto. Naming the exact count in NUM_SPEAKERS bypasses this
# entirely and is the more reliable path whenever you know it.
DIARIZATION_THRESHOLD = 0.35

# None = let clustering decide how many speakers there are. Set an integer when you know (a 1:1
# interview is 2) -- telling it the count is markedly more reliable than making it guess.
NUM_SPEAKERS = None

# Ignore speech shorter than this / treat silence shorter than this as not a real gap (seconds).
DIARIZATION_MIN_ON = 0.3
DIARIZATION_MIN_OFF = 0.5

# --- Minutes / full-transcript translation (local LLM: Ollama) ---
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"   # good multilingual local model. Use qwen2.5:3b for something lighter.
OLLAMA_NUM_CTX = 8192         # context length to hold a long transcript
OLLAMA_TIMEOUT = 600          # max wait for generation (seconds). Generous since it runs on CPU.
# Keep the model resident between calls so a gap between utterances doesn't force a cold reload
# (default Ollama unloads after 5 min -> the next line eats a multi-second load stall). "-1" = never unload.
OLLAMA_KEEP_ALIVE = "30m"

# --- Real-time (per-line) translation ---
# Live translation fires one Ollama call per utterance, so latency per line matters far more than
# raw quality. Point this at a lighter model for snappier lines: `ollama pull qwen2.5:3b` then set
# LIVE_OLLAMA_MODEL = "qwen2.5:3b". Defaults to the main model so nothing breaks out of the box.
LIVE_OLLAMA_MODEL = OLLAMA_MODEL
LIVE_NUM_CTX = 1024           # a single line needs little context -> smaller ctx = faster prompt eval
LIVE_NUM_PREDICT = 128        # cap generated tokens per line (a translated line is short)
