"""Wires the threads together and runs. Ctrl+C saves the .txt on exit."""
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import queue
import sys
import threading

import config
from audio_capture import MultiCapture, list_input_devices
from transcriber import Transcriber
from vad_buffer import VADBuffer
from writer import Writer


def consume_loop(capture, stt, out, stop_event):
    """Consumer: queue -> per-speaker VAD buffering -> STT on finished utterance -> output."""
    vads = {sp: VADBuffer() for sp in capture.speakers()}
    while not stop_event.is_set():
        try:
            speaker, block = capture.audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if speaker not in vads:
            vads[speaker] = VADBuffer()
        for chunk in vads[speaker].add(block):
            lang, text = stt.transcribe(chunk)
            out.emit(lang, text, speaker=speaker)
    # Handle the remaining utterance on shutdown
    for speaker, vad in vads.items():
        tail = vad.finalize()
        if tail is not None:
            lang, text = stt.transcribe(tail)
            out.emit(lang, text, speaker=speaker)


def main():
    if "--list-devices" in sys.argv:
        list_input_devices()
        return

    out = Writer()
    stt = Transcriber()          # load the model first (a few seconds)
    capture = MultiCapture(config.SOURCES)
    stop_event = threading.Event()

    capture.start()
    worker = threading.Thread(
        target=consume_loop,
        args=(capture, stt, out, stop_event),
        daemon=True,
    )
    worker.start()

    print("\n=== Transcription started. Press Ctrl+C to stop. ===\n")
    try:
        while True:
            worker.join(timeout=0.5)
            if not worker.is_alive():
                break
    except KeyboardInterrupt:
        print("\n[main] stop signal received — wrapping up...")
    finally:
        stop_event.set()
        capture.stop()
        worker.join(timeout=10)
        out.save_txt()
        out.close()


if __name__ == "__main__":
    main()
