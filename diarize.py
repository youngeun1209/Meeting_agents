"""Speaker diarization (mode 2, post-session) — wraps sherpa-onnx's offline pipeline.

Segmentation model: pyannote segmentation-3-0 (voice-activity + speaker-change detection).
Embedding model: wespeaker_resnet34 (DEFAULT — smaller + 2x faster than the 3D-Speaker
alternative below, with equal measured accuracy on this project's benchmark audio).
An alternative embedding model (3dspeaker_eres2netv2, larger/slower) is also downloaded
to models/ but not used by default; swap EMBEDDING_MODEL below to try it.

Both models are downloaded ahead of time to <repo>/models/ (see MODEL_DOWNLOAD_INFO for
the exact source URLs) -- diarize.py does not fetch them itself, and raises with clear
instructions if they are missing rather than silently degrading.
"""
import json
import os
import subprocess
import sys
import threading
import wave

import numpy as np
import sherpa_onnx as so

import config

# Paths and tuning live in config.py ("Change behavior here and nowhere else"), so the embedding
# model can be swapped without editing this file.
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.DIARIZATION_DIR)

SEGMENTATION_MODEL = os.path.join(_MODELS_DIR, config.DIARIZATION_SEGMENTATION)
EMBEDDING_MODEL = os.path.join(_MODELS_DIR, config.DIARIZATION_EMBEDDING)

# Exact source + target paths for the error message when a model is missing.
MODEL_DOWNLOAD_INFO = """\
Missing diarization model file(s). Download and place them at these exact paths:

  segmentation (pyannote segmentation-3-0):
    https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
    -> extract so that this file exists: {seg}

  embedding (wespeaker resnet34, default -- smaller + 2x faster, equal accuracy):
    https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx
    -> save as: {emb}
""".format(seg=SEGMENTATION_MODEL, emb=EMBEDDING_MODEL)

# Clustering threshold for the unknown-speaker-count case lives in config.py
# (config.DIARIZATION_THRESHOLD) alongside the rest of the tuning.


def is_available():
    """True if both required model files are present on disk."""
    return os.path.isfile(SEGMENTATION_MODEL) and os.path.isfile(EMBEDDING_MODEL)


def diarize(audio_f32_16k, num_speakers=None, on_progress=None):
    """
    16kHz mono float32 audio -> list of (start, end, speaker_id) tuples, sorted by start time.

    num_speakers: exact speaker count if known (uses FastClusteringConfig(num_clusters=N)).
                  If None, uses threshold-based clustering (config.DIARIZATION_THRESHOLD).
    on_progress:  optional callback(processed_chunks, num_chunks) -> int. Return non-zero
                  to abort. Wired into sherpa-onnx's own process() progress callback.
    Raises RuntimeError if the model files are not present (see MODEL_DOWNLOAD_INFO).
    """
    if not is_available():
        raise RuntimeError(MODEL_DOWNLOAD_INFO)

    if num_speakers:
        clustering = so.FastClusteringConfig(num_clusters=num_speakers)
    else:
        clustering = so.FastClusteringConfig(threshold=config.DIARIZATION_THRESHOLD)

    cfg = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEGMENTATION_MODEL)),
        embedding=so.SpeakerEmbeddingExtractorConfig(model=EMBEDDING_MODEL),
        clustering=clustering,
        min_duration_on=config.DIARIZATION_MIN_ON,
        min_duration_off=config.DIARIZATION_MIN_OFF)
    if not cfg.validate():
        raise RuntimeError("[diarize] invalid sherpa-onnx diarization config")

    sd = so.OfflineSpeakerDiarization(cfg)
    samples = np.asarray(audio_f32_16k, dtype=np.float32)
    if on_progress is not None:
        result = sd.process(samples, callback=on_progress)
    else:
        result = sd.process(samples)
    result = result.sort_by_start_time()

    return [(float(s.start), float(s.end), s.speaker) for s in result]


# --- running diarization off the main process -------------------------------------------
# sherpa-onnx's process() is a long C++ call that does NOT release the GIL, so calling it from
# a worker thread starves the Tk main loop completely -- measured: a 3-minute recording froze
# the UI for 139 seconds straight. Wiring on_progress helps (the callback forces the GIL to be
# handed back) but still left 11-second stalls, which is a frozen app by any user's definition.
# So the work goes in a separate PROCESS, which shares no GIL with the UI at all.
#
# Audio is handed over as a WAV file rather than pickled through a pipe: an hour of 16kHz mono
# float32 is ~230MB, and pickling that per call is both slow and memory-hungry.

_PROGRESS_PREFIX = "PROGRESS "


def diarize_file(wav_path, num_speakers=None, on_progress=None):
    """Diarize a 16kHz mono WAV in a CHILD PROCESS. Same return shape as diarize().

    on_progress(done, total) is called as the child reports; it cannot abort the child.
    Raises RuntimeError if the child fails (its stderr is included).
    """
    if not is_available():
        raise RuntimeError(MODEL_DOWNLOAD_INFO)

    cmd = [sys.executable, os.path.abspath(__file__), wav_path]
    if num_speakers:
        cmd += ["--speakers", str(num_speakers)]
    # stdin is given explicitly, not inherited. A GUI process has no meaningful stdin, and what
    # it does hold can be actively hostile to a fresh interpreter: when the launcher's own stdin
    # is already closed, its open() for the log file is handed descriptor 0 (verified), so the
    # GUI ends up with a WRITE-ONLY file as fd 0 and passes it down. A child that then cannot
    # build sys.stdin dies before running a single line of our code -- "Fatal Python error:
    # init_sys_streams ... OSError: [Errno 9] Bad file descriptor", with no Python frame to show
    # for it. With all three streams supplied here, nothing about the parent's fds can reach it.
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__)))

    # Both pipes must be drained CONCURRENTLY. Draining stderr to EOF first and only then
    # reading stdout deadlocks the moment the result outgrows the OS pipe buffer (~64KB): the
    # child blocks writing its JSON to a full stdout pipe, so it never finishes and never closes
    # stderr, so the parent waits on stderr forever. That threshold is roughly 2,900 segments --
    # never reached by a short recording, always reached by a long meeting, which is why this
    # only ever failed on long files. Measured on the exact read pattern: 30KB completed,
    # 200KB hung indefinitely.
    errors = []

    def drain_stderr():
        for line in proc.stderr:
            if line.startswith(_PROGRESS_PREFIX):
                if on_progress is not None:
                    try:
                        done, total = line[len(_PROGRESS_PREFIX):].split()
                        on_progress(int(done), int(total))
                    except ValueError:
                        pass
            else:
                errors.append(line.rstrip())

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()
    out = proc.stdout.read()
    proc.wait()
    stderr_thread.join(timeout=5)   # the child is gone; its stderr is at EOF
    if proc.returncode != 0:
        # Include what the worker WAS, not just what it said: a startup failure prints no Python
        # frame, so the interpreter path is the only thing that distinguishes "the venv is gone"
        # from "the interpreter could not set up its streams".
        exe = sys.executable
        raise RuntimeError(
            f"[diarize] worker failed (exit {proc.returncode})\n"
            f"  interpreter: {exe} (exists: {os.path.exists(exe)})\n"
            f"  worker: {os.path.abspath(__file__)}\n"
            + "\n".join(errors[-15:]))
    return [(float(s), float(e), spk) for s, e, spk in json.loads(out)]


def _main():
    """Child-process entry point: WAV path in, JSON segments out, progress on stderr."""
    path = sys.argv[1]
    num_speakers = None
    if "--speakers" in sys.argv:
        num_speakers = int(sys.argv[sys.argv.index("--speakers") + 1])
    with wave.open(path, "rb") as r:
        audio = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    def progress(done, total):
        print(f"{_PROGRESS_PREFIX}{done} {total}", file=sys.stderr, flush=True)
        return 0

    segments = diarize(audio, num_speakers=num_speakers, on_progress=progress)
    json.dump(segments, sys.stdout)
    sys.stdout.flush()


if __name__ == "__main__":
    _main()
