"""The STT pipeline engine + CLI entrypoint.

`STTEngine` is the single source of truth for the producer -> VAD -> STT -> writer
pipeline (with live model hot-swap). Both the CLI (`main()` here) and the GUI
(`gui.py`) run this same engine — no duplicated loop. The GUI just wires the
engine's callbacks into its widgets; the CLI lets the Writer echo to the terminal.

Run `STTEngine.run()` on a worker thread and stop it by setting the `stop_event`.
"""
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import queue
import sys
import threading

import config
from audio_capture import MultiCapture, list_input_devices
from vad_buffer import VADBuffer
from writer import Writer

# Transcriber is heavy (model load) -> imported lazily inside run() so that
# importing this module (e.g. from gui.py) stays cheap.


class STTEngine:
    """The full STT pipeline, shared by the CLI and the GUI.

    Callbacks (all optional):
      on_line(line, lang, text, ts, speaker) — every saved utterance (GUI captions).
      on_status(msg)                         — pipeline status strings (GUI status bar).
      echo                                   — Writer prints to the terminal (CLI). False for the GUI.

    Usage: run `run()` on a worker thread; call `request_model()` to hot-swap the
    STT model without dropping audio; set the `stop_event` to finish.
    """

    def __init__(self, on_line=None, on_status=None, echo=True):
        self.on_line = on_line
        self.on_status = on_status or (lambda msg: None)
        self.echo = echo
        self.writer = None
        self.capture = None
        self.running = False
        self._pending_model = None   # model swap request while running (string)

    def request_model(self, model_size):
        """Request a model swap while running. The loop reloads in the background and swaps when ready."""
        self._pending_model = model_size

    def run(self, model_size, language, stop_event, session_started=None):
        """Blocking pipeline loop until stop_event is set. Call on a worker thread.

        session_started: reusing the same value appends to the same file (Stop/Start resume).
        """
        self.running = True
        try:
            config.MODEL_SIZE = model_size
            config.LANGUAGE = language
            self.on_status(f"Loading model: {model_size} …")
            from transcriber import Transcriber  # lazy import (heavy)
            stt = Transcriber()

            self.writer = Writer(on_line=self.on_line, echo=self.echo,
                                 started_at=session_started)
            self.capture = MultiCapture(config.SOURCES)
            self.capture.start()

            vads = {sp: VADBuffer() for sp in self.capture.speakers()}
            who = " + ".join(self.capture.speakers())
            current_model = model_size
            self._pending_model = None
            self.on_status(f"Transcribing · {who} · {current_model}")

            # Background model loader: keep transcribing with the old model until the new one is ready,
            # then swap the moment it finishes loading (no interruption).
            loader = {"thread": None, "model": None, "stt": None, "error": None}

            def _load(model_name, box):
                try:
                    config.MODEL_SIZE = model_name
                    from transcriber import Transcriber  # lazy import
                    box["stt"] = Transcriber()   # download+load (seconds to tens of seconds)
                    box["model"] = model_name
                except Exception as e:  # noqa: BLE001
                    box["error"] = str(e)

            while not stop_event.is_set():
                # 1) If there is a swap request and nothing is loading -> start a background load
                pending = self._pending_model
                if pending and pending != current_model and loader["thread"] is None:
                    self._pending_model = None
                    self.on_status(f"Loading {pending} … (still using {current_model})")
                    loader["thread"] = threading.Thread(
                        target=_load, args=(pending, loader), daemon=True)
                    loader["thread"].start()

                # 2) If the load finished -> swap right then
                if loader["thread"] is not None and not loader["thread"].is_alive():
                    if loader["error"]:
                        self.on_status(f"Model load failed: {loader['error']}")
                    elif loader["stt"] is not None:
                        stt = loader["stt"]
                        current_model = loader["model"]
                        self.on_status(f"Transcribing · {who} · {current_model}")
                    loader = {"thread": None, "model": None,
                              "stt": None, "error": None}

                try:
                    speaker, block = self.capture.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if speaker not in vads:
                    vads[speaker] = VADBuffer()
                for chunk in vads[speaker].add(block):
                    lang, text = stt.transcribe(chunk)
                    self.writer.emit(lang, text, speaker=speaker)

            # Handle the remaining utterance on shutdown
            for speaker, vad in vads.items():
                tail = vad.finalize()
                if tail is not None:
                    lang, text = stt.transcribe(tail)
                    self.writer.emit(lang, text, speaker=speaker)
        finally:
            if self.capture is not None:
                self.capture.stop()
                self.capture = None
            if self.writer is not None:
                self.writer.close()
            self.running = False
            self.on_status("Stopped")

    def save(self):
        """Return the transcript file path (already saved live). None if nothing captured."""
        return self.writer.save_txt() if self.writer is not None else None


def main():
    if "--list-devices" in sys.argv:
        list_input_devices()
        return

    engine = STTEngine(echo=True)          # CLI: Writer echoes captions to the terminal
    stop_event = threading.Event()
    worker = threading.Thread(
        target=engine.run,
        args=(config.MODEL_SIZE, config.LANGUAGE, stop_event),
        daemon=True,
    )
    worker.start()

    print("\n=== Transcription started. Press Ctrl+C to stop. ===\n")
    try:
        while worker.is_alive():
            worker.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\n[main] stop signal received — wrapping up...")
    finally:
        stop_event.set()
        worker.join(timeout=10)   # engine.run() closes capture + writer in its finally
        engine.save()


if __name__ == "__main__":
    main()
