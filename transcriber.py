"""STT wrapper. Picks backend (mlx / faster-whisper) + chunk -> text.

- mlx           : Apple silicon GPU. turbo runs in real time. (config.STT_BACKEND="mlx")
- faster-whisper: CPU. For environments without a GPU.
Both: post-filter for boilerplate hallucinations on silent segments.
"""
import re

import numpy as np

import config

# Boilerplate phrases Whisper often emits on silent/noisy segments (residue from YouTube training data).
# If the whole chunk matches one of these exactly (ignoring punctuation), it is dropped.
# NOTE: the Korean phrases are DATA, not UI text — they must stay Korean to match Korean hallucinations.
HALLUCINATION_PHRASES = {
    "감사합니다", "시청해주셔서 감사합니다", "시청해 주셔서 감사합니다",
    "구독과 좋아요 부탁드립니다", "구독 좋아요 부탁드려요",
    "다음 영상에서 만나요", "다음 시간에 만나요", "다음에 또 만나요",
    "여러분 안녕하세요", "자막 제공", "한국어 자막", "네", "예",
    "thank you", "thank you very much", "thanks for watching",
    "thank you for watching", "please subscribe", "bye", "you",
    "we'll be right back", "i'll be right back", "i'm glad",
    "okay", "ok", "so", "thanks", "bye bye", "yeah",
}
_HAL_NORMALIZED = None  # lazily computed cache


def _normalize(text):
    """Strip punctuation/whitespace, lowercase (for hallucination matching)."""
    t = re.sub(r"[^\w\s]", "", text).strip().lower()
    return re.sub(r"\s+", " ", t)


# Keyword combos that are ALWAYS YouTube-outro boilerplate in a meeting/lecture,
# regardless of the exact verb ending Whisper picks (부탁드려요 / 부탁드립니다 / 부탁해요 ...).
# Each tuple: chunk is dropped if it contains ALL of these substrings (normalized).
HALLUCINATION_KEYWORD_SETS = [
    ("구독", "좋아요"),        # "구독과 좋아요 부탁..." (subscribe + like)
    ("시청", "감사"),          # "시청해주셔서 감사합니다"
    ("한글자막",),             # "한글자막 by 한효정" — subtitle-credit residue (trailing name defeats exact match)
    ("자막", "by"),            # "자막 by ...", "번역/자막 by <name>"
    ("subscribe", "like"),
    ("thanks", "watching"),
    ("thank you", "watching"),
    ("subtitles", "by"),       # "subtitles by <name>" — English subtitle credit
]


def is_hallucination(text):
    """True if the whole chunk is a boilerplate hallucination."""
    global _HAL_NORMALIZED
    if _HAL_NORMALIZED is None:
        _HAL_NORMALIZED = {_normalize(p) for p in HALLUCINATION_PHRASES}
    norm = _normalize(text)
    if norm in _HAL_NORMALIZED:
        return True
    # Keyword-combo guard: catches every verb-ending variant of the same outro line.
    return any(all(kw in norm for kw in kws) for kws in HALLUCINATION_KEYWORD_SETS)


def is_repetitive(text):
    """True if the same word/short phrase repeats excessively (hallucination loop: 'I'm glad I'm glad ...')."""
    words = _normalize(text).split()
    if len(words) < 6:
        return False
    # Very low unique-word ratio = mostly the same words repeated
    return len(set(words)) / len(words) < 0.35


def rms(audio_float32):
    """Block volume (RMS). Used for the silence/noise gate."""
    if audio_float32.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_float32))))


class Transcriber:
    """Uses the mlx or faster-whisper backend depending on config.STT_BACKEND."""

    def __init__(self):
        self.backend = config.STT_BACKEND
        if self.backend == "mlx":
            self._init_mlx()
        else:
            self._init_faster_whisper()

    # ---------- mlx (Apple GPU) ----------
    def _init_mlx(self):
        import mlx_whisper  # lazy import
        self._mlx = mlx_whisper
        self._repo = config.MLX_MODEL_MAP.get(config.MODEL_SIZE)
        if self._repo is None:
            raise RuntimeError(
                f"No mlx model mapping for '{config.MODEL_SIZE}'. "
                f"Add it to config.MLX_MODEL_MAP or change MODEL_SIZE."
            )
        print(f"[stt] loading mlx model: {self._repo} (downloads on first run) ...")
        # Warm up with one silent block -> finishes the model download+load up front
        self._mlx.transcribe(
            np.zeros(config.SAMPLE_RATE, dtype=np.float32),
            path_or_hf_repo=self._repo, verbose=None,
        )
        print("[stt] model loaded (mlx / Apple GPU)")

    def _transcribe_mlx(self, audio_float32):
        res = self._mlx.transcribe(
            audio_float32,
            path_or_hf_repo=self._repo,
            language=config.LANGUAGE,           # None = auto-detect
            condition_on_previous_text=False,   # block hallucination loops
            no_speech_threshold=0.6,            # drop when silence probability is high
            compression_ratio_threshold=2.4,    # suppress repetitive (hallucinated) text
            verbose=None,                        # keep default temperature fallback (avoid dropping speech)
        )
        text = (res.get("text") or "").strip()
        lang = res.get("language") or (config.LANGUAGE or "")
        return lang, text

    # ---------- faster-whisper (CPU) ----------
    def _init_faster_whisper(self):
        from faster_whisper import WhisperModel  # lazy import
        print(f"[stt] loading model: {config.MODEL_SIZE} ({config.COMPUTE_TYPE}) ...")
        self.model = WhisperModel(
            config.MODEL_SIZE,
            device=config.DEVICE,
            compute_type=config.COMPUTE_TYPE,
        )
        print("[stt] model loaded (faster-whisper / CPU)")

    def _transcribe_faster_whisper(self, audio_float32):
        segments, info = self.model.transcribe(
            audio_float32,
            language=config.LANGUAGE,           # None = auto-detect
            beam_size=config.BEAM_SIZE,
            vad_filter=True,                    # built-in Silero VAD suppresses hallucinations
            condition_on_previous_text=False,   # block hallucination loops
            no_speech_threshold=0.6,            # drop when silence probability is high
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return info.language, text

    # ---------- common ----------
    def transcribe(self, audio_float32):
        """Audio (float32 1D, 16kHz) -> (language, text). Empty text if nothing -> (language, "")."""
        # First gate: if volume is too low, skip STT entirely (blocks silence/noise hallucinations at the source)
        if rms(audio_float32) < config.MIN_RMS:
            return (config.LANGUAGE or ""), ""
        if self.backend == "mlx":
            lang, text = self._transcribe_mlx(audio_float32)
        else:
            lang, text = self._transcribe_faster_whisper(audio_float32)
        # Second gate: drop boilerplate hallucinations or repetition loops
        if is_hallucination(text) or is_repetitive(text):
            return lang, ""
        return lang, text
