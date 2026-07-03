"""Output (terminal/GUI) + real-time file saving.

Every utterance line is appended to the file and flushed immediately.
-> Even if the app crashes or is force-quit, everything up to that point stays on disk.
"""
import os
from datetime import datetime

import config


class Writer:
    def __init__(self, on_line=None, echo=True, started_at=None):
        """
        on_line:    callback (line, lang, text, ts, speaker) — feeds an external sink such as the GUI.
        echo:       if True, also print to the terminal (for CLI). Recommended False for the GUI.
        started_at: session start time. Basis for the filename. Building a Writer again with the same
                    value appends to the same file -> continuous saving across Stop/Start resume.
        """
        self.lines = []
        self.started_at = started_at or datetime.now()
        self.on_line = on_line
        self.echo = echo
        self.path = None       # real-time save file path
        self._file = None      # open file handle (emit thread only)

    def _ensure_file(self):
        """Open the file on the first utterance. Append if it exists (resume), else write a header."""
        if self._file is not None:
            return
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        fname = self.started_at.strftime("transcript_%Y-%m-%d_%H-%M.txt")
        self.path = os.path.join(config.OUTPUT_DIR, fname)
        fresh = (not os.path.exists(self.path)
                 or os.path.getsize(self.path) == 0)
        self._file = open(self.path, "a", encoding="utf-8", buffering=1)
        if fresh:
            self._file.write(
                f"# STT transcript — started {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"# model: {config.MODEL_SIZE} / language: "
                f"{config.LANGUAGE or 'auto-detect'}\n\n"
            )
            self._file.flush()

    def emit(self, lang, text, speaker=None):
        """Output one utterance line and save it to the file immediately."""
        if not text:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        who = f"[{speaker}] " if speaker else ""
        if config.PRINT_LANG:
            line = f"[{ts}] {who}({lang}) {text}"
        else:
            line = f"[{ts}] {who}{text}"
        if self.echo:
            print(line, flush=True)
        self.lines.append(line)

        # Real-time save: write the line, then flush + fsync to push it all the way to disk
        self._ensure_file()
        self._file.write(line + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

        if self.on_line is not None:
            self.on_line(line, lang, text, ts, speaker)

    def save_txt(self):
        """
        Already saving in real time, so nothing extra to do.
        Returns the current file path (None if there is no content).
        """
        if self.path is None:
            print("[writer] nothing to save")
            return None
        if self._file is not None:
            self._file.flush()
        print(f"[writer] saved: {self.path} ({len(self.lines)} lines)")
        return self.path

    def close(self):
        """Close the file handle on session end. (Call from the same thread as emit.)"""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
