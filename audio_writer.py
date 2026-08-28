"""Saves raw captured audio to WAV files, alongside the live transcript. (Consumer thread.)

Mirrors writer.Writer's real-time-save design: all disk I/O happens on one dedicated
thread so the sounddevice callback and the sidecar reader thread never block on it.
Unlike audio_capture.AUDIO_QUEUE_MAXLEN (which drops the oldest block under backlog),
this queue must never drop -- archival audio has to be complete -- so a full queue
blocks the caller instead, after logging once.
"""
import os
import queue
import struct
import sys
import threading
import time
from datetime import datetime

import numpy as np
import wave

import config

# Generous bound on the archival backlog. At BLOCK_SEC=0.1s and up to a few sources
# that's a handful of blocks/sec each; 6000 is minutes of buffering before this queue
# would ever back up. Disk writes are far faster than real-time audio, so filling this
# means something is actually wrong (disk stall) -- not routine backpressure to shrug off.
AUDIO_QUEUE_MAXLEN = 6000

_WAV_HEADER_SIZE = 44  # standard 16-bit-PCM WAV header: RIFF(12) + fmt(24) + data-header(8)

# How often the WAV size headers are rewritten + fsync'd. Blocks arrive every
# config.BLOCK_SEC (0.1s) per source, so flushing per block would mean ~20 fsync/sec
# with two sources -- needlessly expensive, and genuinely slow when OUTPUT_DIR lives in
# a cloud-synced folder (iCloud/OneDrive), which is exactly where this project runs.
# fsync is what makes a killed process still leave a playable file, so keep it -- just
# at a sane cadence. Worst case a hard kill loses the last FLUSH_INTERVAL_SEC of audio
# from the *header count*; the PCM bytes themselves are already written.
FLUSH_INTERVAL_SEC = 2.0


class _WavAppender:
    """One speaker's WAV file, opened for append-safe writing.

    Python's `wave` module has no append mode (Wave_write always starts a file fresh),
    so a fresh file is created via `wave` (correct header for 16kHz/mono/int16), then
    reopened in raw binary mode and PCM bytes are appended directly. On every flush the
    RIFF/data chunk sizes are rewritten in place -- the same sizes `wave` itself would
    compute -- so the file stays a valid, playable WAV even if the process is killed
    right after. Resuming (same started_at) re-reads the existing header via `wave` to
    validate format and current length, then continues appending after it.
    """

    def __init__(self, path):
        self.path = path
        self._data_size = 0
        if os.path.exists(path) and os.path.getsize(path) >= _WAV_HEADER_SIZE:
            with wave.open(path, "rb") as r:
                if (r.getframerate() != config.SAMPLE_RATE
                        or r.getnchannels() != 1 or r.getsampwidth() != 2):
                    raise RuntimeError(
                        f"[audio_writer] {path} exists with an unexpected format "
                        f"({r.getframerate()}Hz {r.getnchannels()}ch {r.getsampwidth()}B) "
                        f"-- refusing to append.")
                self._data_size = r.getnframes() * r.getsampwidth() * r.getnchannels()
            os.chmod(path, 0o600)
            self._file = open(path, "r+b")
            self._file.seek(0, os.SEEK_END)
        else:
            # Recorded meeting audio is at least as sensitive as the transcript -> 0600, same
            # reasoning as writer.Writer. `wave` gives no mode control, so create the file
            # restricted first and let wave write its header into it.
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600))
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)          # int16
                w.setframerate(config.SAMPLE_RATE)
                w.writeframes(b"")
            self._file = open(path, "r+b")
            self._file.seek(0, os.SEEK_END)

    def append(self, pcm_bytes):
        self._file.write(pcm_bytes)
        self._data_size += len(pcm_bytes)

    def flush(self):
        """Rewrite the RIFF/data chunk sizes so a crash right after still leaves a playable file."""
        pos = self._file.tell()
        self._file.seek(4)                                  # RIFF chunk size
        self._file.write(struct.pack("<I", 36 + self._data_size))
        self._file.seek(40)                                 # data chunk size
        self._file.write(struct.pack("<I", self._data_size))
        self._file.seek(pos)
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self):
        self.flush()
        self._file.close()


class AudioWriter:
    """Writes raw captured audio to 16kHz mono int16 WAV files, one per speaker.

    started_at: session start time, SAME convention as writer.Writer -- building an
                AudioWriter again with the same value appends to the same WAV files
                (Stop/Start resume), and keeps filenames lined up with the transcript.
    """

    def __init__(self, started_at=None):
        self.started_at = started_at or datetime.now()
        self._queue = queue.Queue(maxsize=AUDIO_QUEUE_MAXLEN)
        self._appenders = {}     # speaker -> _WavAppender (writer thread only)
        self._paths = {}         # speaker -> path (populated as files are opened)
        self._thread = None
        self._stop = threading.Event()
        self._overflow_warned = False
        self._last_flush = 0.0
        self._dirty = False

    def start(self):
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, speaker, block):
        """Queue a float32 mono block for the given speaker.
        Must never be called from a context that can't tolerate blocking -- unlike the
        STT audio_queue this does NOT drop on overflow (archival must be complete); if
        the queue is ever full it logs once and then blocks until there's room."""
        try:
            self._queue.put_nowait((speaker, block))
        except queue.Full:
            if not self._overflow_warned:
                self._overflow_warned = True
                print(f"[audio_writer] backlog full ({AUDIO_QUEUE_MAXLEN}) -- disk writer "
                      f"is falling behind; blocking until it catches up (audio will NOT be dropped)",
                      file=sys.stderr)
            self._queue.put((speaker, block))  # block rather than drop -- archival must be complete

    def _path_for(self, speaker):
        fname = self.started_at.strftime(f"audio_{speaker}_%Y-%m-%d_%H-%M.wav")
        return os.path.join(config.OUTPUT_DIR, fname)

    def _run(self):
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                # Caught up with the producers. This is the cheapest possible moment to
                # fsync, and it is also the ONLY thing that protects the tail: the
                # time-based flush below is driven by *incoming* blocks, so without this
                # the size headers would go stale the moment audio goes quiet, and a
                # crash would leave the file claiming fewer frames than it really holds.
                self._flush_all()
                if self._stop.is_set() and self._queue.empty():
                    break
                continue
            speaker, block = item
            appender = self._appenders.get(speaker)
            if appender is None:
                path = self._path_for(speaker)
                appender = _WavAppender(path)
                self._appenders[speaker] = appender
                self._paths[speaker] = path
            pcm16 = np.clip(block * 32768.0, -32768, 32767).astype(np.int16)
            appender.append(pcm16.tobytes())
            self._dirty = True
            # Sustained-streaming flush: under continuous audio the queue may never go
            # empty, so bound how much the headers can lag even then.
            if time.monotonic() - self._last_flush >= FLUSH_INTERVAL_SEC:
                self._flush_all()
        for appender in self._appenders.values():
            appender.close()

    def _flush_all(self):
        """Rewrite + fsync every open WAV's size headers. No-op when nothing was written
        since the last flush, so going idle doesn't fsync in a loop."""
        if not self._dirty:
            return
        self._dirty = False
        self._last_flush = time.monotonic()
        for a in self._appenders.values():
            a.flush()

    def close(self):
        """Signal the writer thread to drain the queue, close all files, and join."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def paths(self):
        """Return {speaker: wav_path} for files opened so far this session."""
        return dict(self._paths)
