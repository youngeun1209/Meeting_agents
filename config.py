"""Settings. Change behavior here and nowhere else."""

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

# mlx-whisper model repo mapping (MODEL_SIZE -> HuggingFace mlx-community repo)
MLX_MODEL_MAP = {
    "base":            "mlx-community/whisper-base-mlx",
    "small":           "mlx-community/whisper-small-mlx",
    "medium":          "mlx-community/whisper-medium-mlx",
    "large-v3-turbo":  "mlx-community/whisper-large-v3-turbo",
    "large-v3":        "mlx-community/whisper-large-v3-mlx",
}

# --- Language ---
# None = auto-detect (Korean/English). Can also pin to "ko" or "en".
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
# Lower catches quiet speech better (higher hallucination risk). Lower it (0.003) if speech is being missed.
MIN_RMS = 0.005

# --- Output / saving ---
OUTPUT_DIR = "transcripts"   # folder for .txt files
PRINT_LANG = True            # show the detected language in the terminal

# --- Minutes generation (local LLM: Ollama) ---
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"   # good multilingual local model. Use qwen2.5:3b for something lighter.
OLLAMA_NUM_CTX = 8192         # context length to hold a long transcript
OLLAMA_TIMEOUT = 600          # max wait for generation (seconds). Generous since it runs on CPU.
