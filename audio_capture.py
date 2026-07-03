"""Reads audio and pushes it onto the queue. (Producer threads.)

Supports two kinds of source:
  - sounddevice devices (microphone, BlackHole, etc.) -> AudioCapture / MultiCapture
  - ScreenCaptureKit sidecar (whole system audio) -> SystemAudioCapture
Both put (speaker, mono float32 block) onto the same audio_queue, so downstream is identical.
"""
import os
import queue
import subprocess
import sys
import threading

import numpy as np
import sounddevice as sd

import config


def find_device(name_substr):
    """Find an input device index by substring match on the device name."""
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if name_substr.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            return idx, dev
    return None, None


def list_input_devices():
    """Print the list of capturable input devices. (For debugging.)"""
    print("=== Capturable audio input devices ===")
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"  [{idx}] {dev['name']} (in={dev['max_input_channels']}, "
                  f"{int(dev['default_samplerate'])}Hz)")
    print("=" * 32)


class AudioCapture:
    """sounddevice callback-based capture. Puts float32 mono blocks onto audio_queue."""

    def __init__(self):
        self.audio_queue = queue.Queue()
        self.stream = None
        self.device_idx = None
        self.channels = 1

    def _callback(self, indata, frames, time_info, status):
        if status:
            # Overflow etc. -> don't handle heavily, just warn
            print(f"[audio] {status}", file=sys.stderr)
        # indata: (frames, channels) float32. Average to mono if stereo.
        if indata.shape[1] > 1:
            mono = indata.mean(axis=1)
        else:
            mono = indata[:, 0]
        self.audio_queue.put(mono.copy())

    def start(self):
        idx, dev = find_device(config.INPUT_DEVICE)
        if idx is None:
            list_input_devices()
            raise RuntimeError(
                f"Input device '{config.INPUT_DEVICE}' not found. "
                f"Check the device name in the list above and fix config.INPUT_DEVICE."
            )
        self.device_idx = idx
        # An aggregate device (mic + BlackHole) can have 3+ channels -> take them all and average to mono.
        # (Cutting to 2 would drop one source.) Safe upper bound of 8.
        self.channels = min(dev["max_input_channels"], 8)
        blocksize = int(config.SAMPLE_RATE * config.BLOCK_SEC)
        self.stream = sd.InputStream(
            device=idx,
            channels=self.channels,
            samplerate=config.SAMPLE_RATE,
            blocksize=blocksize,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()
        print(f"[audio] capture started: [{idx}] {dev['name']} "
              f"({self.channels}ch @ {config.SAMPLE_RATE}Hz)")

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("[audio] capture stopped")


class SystemAudioCapture:
    """Reads whole system audio from the ScreenCaptureKit sidecar (native/sysaudio).

    The sidecar streams 16kHz mono int16 PCM to stdout. This converts it to float32 blocks
    and puts them on the shared audio_queue with the given speaker label.
    -> Captures the other party (system) sound without BlackHole / a Multi-Output Device.
    """

    def __init__(self, audio_queue, speaker="Others", binary=None):
        self.audio_queue = audio_queue
        self.speaker = speaker
        self.binary = binary or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "native", "sysaudio")
        self.proc = None
        self._stop = threading.Event()
        self._reader_thread = None

    def start(self):
        if not os.path.exists(self.binary):
            raise RuntimeError(
                f"Sidecar binary not found: {self.binary}\n"
                f"  -> build it first with: bash native/build.sh"
            )
        self.proc = subprocess.Popen(
            [self.binary], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()
        threading.Thread(target=self._stderr_relay, daemon=True).start()
        print(f"[audio] system audio capture started -> speaker '{self.speaker}' "
              f"(ScreenCaptureKit)")

    def _reader(self):
        # Slice into config.BLOCK_SEC-length blocks before queueing (same granularity as other sources).
        bytes_per_block = int(config.SAMPLE_RATE * config.BLOCK_SEC) * 2  # int16=2B
        buf = bytearray()
        stream = self.proc.stdout
        while not self._stop.is_set():
            chunk = stream.read(4096)
            if not chunk:
                break
            buf.extend(chunk)
            while len(buf) >= bytes_per_block:
                block = bytes(buf[:bytes_per_block])
                del buf[:bytes_per_block]
                mono = np.frombuffer(block, dtype=np.int16).astype(np.float32) / 32768.0
                self.audio_queue.put((self.speaker, mono))

    def _stderr_relay(self):
        # Surface the sidecar's diagnostic/permission/format logs as-is.
        for line in iter(self.proc.stderr.readline, b""):
            if self._stop.is_set():
                break
            sys.stderr.write(f"[sysaudio] {line.decode(errors='replace')}")

    def stop(self):
        self._stop.set()
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        print("[audio] system audio capture stopped")


class MultiCapture:
    """
    Captures multiple sources at once, tags them with a speaker label,
    and puts (speaker, mono_block) onto a single shared queue.

    Source kinds:
      - {"device": "...", "speaker": "..."}  -> sounddevice device (mic, etc.)
      - {"kind": "system", "speaker": "..."} -> ScreenCaptureKit system audio
    -> Separating your voice (mic) and the other party (system) as sources = accurate speaker tagging.
    """

    def __init__(self, sources):
        self.sources = sources          # [{"device":...} | {"kind":"system"}, ...]
        self.audio_queue = queue.Queue()
        self.streams = []
        self.sys_caps = []              # SystemAudioCapture instances
        self.active = []                # actually opened (speaker, source_name)

    def _make_callback(self, speaker):
        def _cb(indata, frames, time_info, status):
            if status:
                print(f"[audio:{speaker}] {status}", file=sys.stderr)
            if indata.shape[1] > 1:
                mono = indata.mean(axis=1)
            else:
                mono = indata[:, 0]
            self.audio_queue.put((speaker, mono.copy()))
        return _cb

    def start(self):
        blocksize = int(config.SAMPLE_RATE * config.BLOCK_SEC)
        missing = []
        for src in self.sources:
            # System audio source (ScreenCaptureKit sidecar)
            if src.get("kind") == "system":
                speaker = src.get("speaker", "Others")
                cap = SystemAudioCapture(self.audio_queue, speaker)
                cap.start()
                self.sys_caps.append(cap)
                self.active.append((speaker, "System Audio"))
                continue

            # sounddevice device source (mic, etc.)
            idx, dev = find_device(src["device"])
            if idx is None:
                missing.append(src["device"])
                continue
            channels = min(dev["max_input_channels"], 8)
            stream = sd.InputStream(
                device=idx,
                channels=channels,
                samplerate=config.SAMPLE_RATE,
                blocksize=blocksize,
                dtype="float32",
                callback=self._make_callback(src["speaker"]),
            )
            stream.start()
            self.streams.append(stream)
            self.active.append((src["speaker"], dev["name"]))
            print(f"[audio] capture started: [{idx}] {dev['name']} "
                  f"-> speaker '{src['speaker']}' ({channels}ch)")

        if not self.streams and not self.sys_caps:
            list_input_devices()
            raise RuntimeError(
                f"Could not open any source: {missing}. "
                f"Check the list above and match the device names in config.SOURCES."
            )
        if missing:
            print(f"[audio] warning: devices not found {missing} — continuing with the rest")

    def speakers(self):
        return [s for s, _ in self.active]

    def stop(self):
        for stream in self.streams:
            stream.stop()
            stream.close()
        self.streams = []
        for cap in self.sys_caps:
            cap.stop()
        self.sys_caps = []
        print("[audio] capture stopped")
