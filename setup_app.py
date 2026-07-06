"""First-run setup wizard (MATLAB-style initialization).

Double-clicking the app runs THIS first. It uses only the Python standard
library (tkinter) so it can run before any third-party package is installed.
It then builds everything the real app needs — a virtual environment, all pip
packages, the native system-audio helper, the Whisper model, and optionally
Ollama — streaming progress into a window. When every required step is done it
launches gui.py inside the virtual environment.

If everything is already in place it skips the window and launches the app
directly, so day-to-day startup stays instant.

Run:  python3 setup_app.py
"""

import os
import platform
import queue
import shutil
import subprocess
import sys
import threading

import tkinter as tk
from tkinter import scrolledtext

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, ".venv")
VENV_PY = os.path.join(VENV, "bin", "python")
MARKER = os.path.join(ROOT, ".setup_done")
IS_MAC = platform.system() == "Darwin"
IS_ARM = platform.machine() == "arm64"


# ---------------------------------------------------------------------------
# Environment probes — cheap checks used both for the "already ready?" fast
# path and for skipping steps whose work is already done.
# ---------------------------------------------------------------------------

def venv_ok():
    return os.path.exists(VENV_PY)


def packages_ok():
    """True if the venv can import the core runtime deps."""
    if not venv_ok():
        return False
    mods = ["customtkinter", "sounddevice", "numpy", "webrtcvad"]
    if IS_ARM:
        mods.append("mlx_whisper")
    else:
        mods.append("faster_whisper")
    code = "import importlib.util,sys;" + \
        "sys.exit(0 if all(importlib.util.find_spec(m) for m in %r) else 1)" % mods
    return _quiet_run([VENV_PY, "-c", code]) == 0


def sidecar_ok():
    # Only the macOS ScreenCaptureKit build matters; other platforms skip it.
    return (not IS_MAC) or os.path.exists(os.path.join(ROOT, "native", "sysaudio"))


def all_ready():
    return venv_ok() and packages_ok() and sidecar_ok() and os.path.exists(MARKER)


def _quiet_run(cmd):
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode
    except Exception:
        return 1


def have(binary):
    return shutil.which(binary) is not None


# ---------------------------------------------------------------------------
# Step implementations. Each takes a `log(text)` callback and returns True on
# success. Required steps gate the Launch button; optional ones only warn.
# ---------------------------------------------------------------------------

def _stream(cmd, log, cwd=ROOT, env=None):
    """Run a command, streaming combined stdout/stderr into the log."""
    log("$ " + " ".join(cmd) + "\n")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log(line)
    proc.wait()
    return proc.returncode == 0


def step_venv(log):
    if venv_ok():
        log("Virtual environment already exists.\n")
        return True
    log("Creating virtual environment (.venv) ...\n")
    return _stream([sys.executable, "-m", "venv", VENV], log)


def step_packages(log):
    if packages_ok():
        log("All Python packages already installed.\n")
        return True
    log("Upgrading pip ...\n")
    _stream([VENV_PY, "-m", "pip", "install", "--upgrade", "pip"], log)
    log("Installing packages from requirements.txt ...\n")
    ok = _stream([VENV_PY, "-m", "pip", "install", "-r",
                  os.path.join(ROOT, "requirements.txt")], log)
    if ok and IS_ARM:
        # Apple silicon default backend needs mlx-whisper (not in requirements).
        log("Installing mlx-whisper (Apple GPU backend) ...\n")
        ok = _stream([VENV_PY, "-m", "pip", "install", "mlx-whisper"], log)
    return ok


def step_portaudio(log):
    # sounddevice usually ships PortAudio in its wheel; this is best-effort.
    if not IS_MAC:
        log("Not macOS — skipping PortAudio (handled by the wheel).\n")
        return True
    if not have("brew"):
        log("Homebrew not found — skipping. Install manually if the mic fails:\n"
            "  brew install portaudio\n")
        return True
    if _quiet_run(["brew", "list", "portaudio"]) == 0:
        log("PortAudio already installed.\n")
        return True
    log("Installing PortAudio via Homebrew ...\n")
    return _stream(["brew", "install", "portaudio"], log)


def step_sidecar(log):
    if not IS_MAC:
        log("Not macOS — system-audio helper not needed.\n")
        return True
    if sidecar_ok():
        log("System-audio helper already built.\n")
        return True
    if not have("swiftc"):
        log("swiftc not found. Triggering Xcode Command Line Tools install ...\n"
            "A macOS dialog will open — click Install, wait for it to finish,\n"
            "then press 'Retry setup'.\n")
        _quiet_run(["xcode-select", "--install"])
        return False
    log("Building the ScreenCaptureKit system-audio helper ...\n")
    return _stream(["bash", os.path.join(ROOT, "native", "build.sh")], log)


def step_model(log):
    """Pre-download the Whisper model so the first real run is instant.

    Optional: if it fails, the model still auto-downloads at runtime.
    """
    log("Downloading the speech-to-text model (first time is large) ...\n")
    if IS_ARM:
        code = (
            "import numpy as np, mlx_whisper;"
            "mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32),"
            "path_or_hf_repo='mlx-community/whisper-large-v3-turbo')"
        )
    else:
        code = (
            "from faster_whisper import WhisperModel;"
            "WhisperModel('large-v3', device='cpu', compute_type='int8')"
        )
    ok = _stream([VENV_PY, "-c", code], log)
    if not ok:
        log("Model pre-download failed — it will download on first use instead.\n")
    return True  # never block launch on this


def step_ollama(log):
    """Optional: Ollama powers translation and minutes. App runs without it."""
    if have("ollama"):
        log("Ollama already installed. Pulling model qwen2.5:7b (if missing) ...\n")
        _stream(["ollama", "pull", "qwen2.5:7b"], log)
        return True
    if not have("brew"):
        log("Ollama not installed (optional). For translation/minutes install it:\n"
            "  https://ollama.com/download\n")
        return True
    log("Installing Ollama via Homebrew (optional) ...\n")
    if _stream(["brew", "install", "ollama"], log):
        log("Pulling model qwen2.5:7b ...\n")
        _stream(["ollama", "pull", "qwen2.5:7b"], log)
    return True


def step_permission(log):
    log("Screen Recording permission:\n"
        "  macOS requires you to grant this MANUALLY (it cannot be automated).\n"
        "  On the first capture a prompt appears — allow it, then restart the app.\n"
        "  System Settings > Privacy & Security > Screen Recording.\n")
    return True


# name, required, function
STEPS = [
    ("Virtual environment",        True,  step_venv),
    ("Python packages",            True,  step_packages),
    ("PortAudio (microphone)",     False, step_portaudio),
    ("System-audio helper",        True,  step_sidecar),
    ("Speech model download",      False, step_model),
    ("Ollama (translate/minutes)", False, step_ollama),
    ("Screen Recording note",      False, step_permission),
]


def launch_app():
    """Start the real GUI inside the venv and stop being the launcher."""
    subprocess.Popen([VENV_PY, os.path.join(ROOT, "gui.py")], cwd=ROOT)


# ---------------------------------------------------------------------------
# Setup window
# ---------------------------------------------------------------------------

class SetupWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Meeting STT — Setup")
        self.geometry("640x520")
        self.q = queue.Queue()
        self.rows = {}
        self.failed_required = False

        tk.Label(self, text="First-time setup", font=("Helvetica", 18, "bold")
                 ).pack(pady=(16, 2))
        tk.Label(self, text="Installing everything the app needs. This runs once.",
                 fg="#666").pack()

        checks = tk.Frame(self)
        checks.pack(fill="x", padx=20, pady=12)
        for name, required, _ in STEPS:
            row = tk.Frame(checks)
            row.pack(fill="x", anchor="w")
            dot = tk.Label(row, text="•", width=2, font=("Helvetica", 14))
            dot.pack(side="left")
            label = name + ("" if required else "  (optional)")
            tk.Label(row, text=label, anchor="w").pack(side="left")
            self.rows[name] = dot

        self.log_box = scrolledtext.ScrolledText(self, height=12, wrap="word",
                                                font=("Menlo", 10))
        self.log_box.pack(fill="both", expand=True, padx=20, pady=(0, 8))
        self.log_box.configure(state="disabled")

        bar = tk.Frame(self)
        bar.pack(fill="x", padx=20, pady=(0, 16))
        self.status = tk.Label(bar, text="Ready.", anchor="w", fg="#666")
        self.status.pack(side="left")
        self.action = tk.Button(bar, text="Start setup", command=self.start)
        self.action.pack(side="right")

        self.after(80, self._drain)

    # -- worker communication -------------------------------------------------
    def _post(self, kind, payload):
        self.q.put((kind, payload))

    def _drain(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log_box.configure(state="normal")
                    self.log_box.insert("end", payload)
                    self.log_box.see("end")
                    self.log_box.configure(state="disabled")
                elif kind == "dot":
                    name, mark, color = payload
                    self.rows[name].configure(text=mark, fg=color)
                elif kind == "status":
                    self.status.configure(text=payload)
                elif kind == "done":
                    self._on_done(ok=payload)
        except queue.Empty:
            pass
        self.after(80, self._drain)

    # -- control --------------------------------------------------------------
    def start(self):
        self.action.configure(state="disabled", text="Working ...")
        threading.Thread(target=self._run_steps, daemon=True).start()

    def _run_steps(self):
        self.failed_required = False
        for name, required, fn in STEPS:
            self._post("dot", (name, "▸", "#1e88e5"))
            self._post("status", "Running: " + name)
            try:
                ok = fn(lambda t: self._post("log", t))
            except Exception as e:  # never let one step kill the wizard
                self._post("log", "ERROR: %s\n" % e)
                ok = False
            if ok:
                self._post("dot", (name, "✓", "#2e7d32"))
            elif required:
                self._post("dot", (name, "✗", "#c62828"))
                self.failed_required = True
            else:
                self._post("dot", (name, "!", "#f9a825"))
        self._post("done", not self.failed_required)

    def _on_done(self, ok):
        if ok:
            with open(MARKER, "w") as f:
                f.write("ok\n")
            self.status.configure(text="Setup complete. Launching ...", fg="#2e7d32")
            self.action.configure(state="normal", text="Launch app",
                                  command=self._launch_and_close)
            self.after(600, self._launch_and_close)
        else:
            self.status.configure(text="A required step failed — see log.", fg="#c62828")
            self.action.configure(state="normal", text="Retry setup",
                                  command=self.start)

    def _launch_and_close(self):
        launch_app()
        self.destroy()


def _osa_quote(s):
    return "'" + s.replace("'", "'\\''") + "'"


def _dbg(msg):
    try:
        with open(os.path.join(ROOT, "setup_debug.log"), "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def main():
    _dbg("main start | all_ready=%s venv=%s pkgs=%s sidecar=%s marker=%s py=%s" % (
        all_ready(), venv_ok(), packages_ok(), sidecar_ok(),
        os.path.exists(MARKER), sys.executable))
    # Fast path: everything already in place -> straight into the app.
    if all_ready():
        _dbg("taking fast path -> launch_app")
        launch_app()
        return
    _dbg("opening SetupWindow")
    SetupWindow().mainloop()
    _dbg("mainloop returned")


if __name__ == "__main__":
    # When launched from Finder (.app) stdout/stderr are discarded, so any crash
    # would be invisible. Mirror it to a log next to the app and to a dialog.
    LOG = os.path.join(ROOT, "setup_error.log")
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        try:
            with open(LOG, "w") as f:
                f.write(tb)
        except Exception:
            pass
        if IS_MAC:
            msg = tb.strip().replace('"', "'").splitlines()[-1][:300]
            os.system('osascript -e %s' % _osa_quote(
                'display alert "Meeting STT failed to start" message "%s\\n\\nSee setup_error.log."' % msg))
        raise
