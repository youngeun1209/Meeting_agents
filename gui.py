"""STT transcription GUI — modern UI (customtkinter).

- Tab 1 🎙 Live STT: real-time scrolling captions colored per speaker
- Tab 2 🌐 Translator: translate the transcript with a local LLM (Ollama)
- Tab 3 📁 Audio File: re-transcribe a saved recording with speaker diarization
- Tab 4 📝 Minutes: turn the transcript into minutes with a local LLM (Ollama)
- Start/Stop/New session, model & language selection, automatic txt saving
"""
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import os
import queue
import resource
import sys
import threading
import time
from datetime import datetime

import customtkinter as ctk

import dock_icon
import settings
from tkinter import filedialog

import config
import batch
import diarize
import markdown_view
import minutes
import translator
from audio_capture import list_input_devices, put_drop_oldest
from main import STTEngine   # the shared pipeline engine — single source of truth

MODEL_CHOICES = ["base", "small", "medium", "large-v3-turbo", "large-v3"]
LANG_CHOICES = [("Auto", None), ("Korean", "ko"), ("English", "en"), ("Chinese", "zh")]
# Translation target name -> Whisper language code (to skip translating same-language lines)
TARGET_LANG_CODE = {"Korean": "ko", "English": "en", "Chinese": "zh"}

# --- Palette (modern dark) ---
ACCENT = "#6366f1"        # indigo
ACCENT_HOVER = "#4f46e5"
GREEN = "#22c55e"
GREEN_HOVER = "#16a34a"
NEUTRAL = "#2b2b35"
NEUTRAL_HOVER = "#3a3a46"
BORDER = "#3a3a46"
PANEL = "#17171f"
TEXT_MUTED = "#8b8b99"
# How often the streaming minutes are re-rendered as markdown (seconds). Low enough to look
# live, high enough that re-parsing the growing document stays off the critical path.
MINUTES_RENDER_INTERVAL_SEC = 0.5

SPEAKER_PALETTE = ["#7ec8ff", "#9cffb0", "#ffcf7e",
                   "#ff9ec8", "#c8a0ff", "#a0ffe8"]

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


def _open_private(path):
    """Create/truncate `path` with 0600 permissions before it is written.

    Saved translations, minutes and rebuilt transcripts all contain the full meeting content,
    so they get the same 0600 treatment as the raw transcript rather than the umask default.
    """
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600))


class STTController:
    """Thin GUI adapter over `STTEngine`.

    Owns no pipeline logic — it just runs the shared engine on a worker thread and
    funnels the engine's callbacks (line / status / error) into a queue the UI drains.
    """

    def __init__(self):
        self.line_queue = queue.Queue()    # ('line'|'status'|'error', payload)
        self.stop_event = threading.Event()
        self.worker = None
        self.engine = STTEngine(on_line=self._emit_line, on_status=self._status,
                                echo=False)   # GUI shows lines itself; no terminal echo

    @property
    def running(self):
        return self.engine.running

    def request_model(self, model_size):
        """Request a live model swap. Delegated straight to the engine."""
        self.engine.request_model(model_size)

    def _emit_line(self, line, lang, text, ts, speaker):
        self.line_queue.put(("line", (ts, speaker, lang, text)))

    def _status(self, msg):
        self.line_queue.put(("status", msg))

    def start(self, model_size, language, session_started=None):
        if self.engine.running:
            return
        self.stop_event.clear()
        self.worker = threading.Thread(
            target=self._run_engine,
            args=(model_size, language, session_started),
            daemon=True,
        )
        self.worker.start()

    def _run_engine(self, model_size, language, session_started):
        try:
            self.engine.run(model_size, language, self.stop_event, session_started)
        except Exception as e:  # noqa: BLE001
            self.line_queue.put(("error", f"Error: {e}"))

    def stop(self):
        self.stop_event.set()

    def save(self):
        return self.engine.save()


class App:
    def __init__(self, root):
        self.root = root
        self.ctrl = STTController()
        self.session_lines = []
        self.session_started = None
        self.translation_queue = queue.Queue()
        self.minutes_queue = queue.Queue()
        self.minutes_busy = False
        self.file_queue = queue.Queue()
        self.file_busy = False
        self.file_path = None
        # The textboxes show RENDERED markdown (syntax stripped), so the original text has to be
        # kept here -- reading it back out of the widget would save the rendering, not the markdown.
        self.file_md = ""
        self.minutes_md = ""
        self._minutes_last_render = 0.0   # throttle for the streaming re-render
        self._speaker_colors = {}

        # --- real-time translation ---
        self._utterances = []                 # [(ts, speaker, lang, text), ...] raw, for live translate
        self._live_translate_idx = 0          # how many utterances have been queued for translation
        self.live_translate = False           # on when the Translator tab is open + Ollama is up
        # Bounded: if Ollama translates slower than utterances arrive, keep only the newest
        # backlog (live pane is best-effort; the STT tab + saved txt are authoritative).
        self.live_translate_queue = queue.Queue(maxsize=200)   # feed to the live-translate worker
        self.live_translate_thread = None

        # --- diagnostics (always on, cheap) — pinpoint what freezes the UI ---
        # _poll_queue runs every 100ms on the MAIN thread. If the gap between runs
        # balloons, the main loop is being starved/blocked -> that's the beach ball.
        # We log the stall + resource stats to stderr (visible in the terminal, and
        # captured to gui_error.log when launched via the .app).
        self._diag_last_poll = time.monotonic()
        self._diag_last_stats = 0.0
        self._diag_max_gap = 0.0

        root.title("Live STT · Translator · Minutes")
        root.geometry("760x640")
        root.minsize(560, 460)
        root.configure(fg_color="#0f0f14")

        self.ui_font = ctk.CTkFont(size=13)
        self.ui_bold = ctk.CTkFont(size=13, weight="bold")
        self.title_font = ctk.CTkFont(size=15, weight="bold")
        self.mono_font = ctk.CTkFont(family="Menlo", size=13)

        self._build_toolbar()
        self._build_statusbar()
        self._build_tabs()

        self.root.after(100, self._poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI ----------
    def _build_toolbar(self):
        bar = ctk.CTkFrame(self.root, fg_color="transparent")
        bar.pack(fill="x", padx=16, pady=(14, 6))

        self.start_btn = ctk.CTkButton(
            bar, text="▶  Start", command=self._on_start, width=92, height=36,
            corner_radius=10, font=self.ui_bold,
            fg_color=GREEN, hover_color=GREEN_HOVER)
        self.start_btn.pack(side="left")

        self.stop_btn = ctk.CTkButton(
            bar, text="■  Stop", command=self._on_stop, width=88, height=36,
            corner_radius=10, font=self.ui_bold, state="disabled",
            fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER)
        self.stop_btn.pack(side="left", padx=(8, 0))

        self.new_btn = ctk.CTkButton(
            bar, text="🆕  New", command=self._on_new, width=96, height=36,
            corner_radius=10, font=self.ui_font,
            fg_color="transparent", hover_color=NEUTRAL,
            border_width=1, border_color=BORDER, text_color="#c9c9d4")
        self.new_btn.pack(side="left", padx=(8, 0))

        # Right side: language / model selection
        self.lang_var = ctk.StringVar(value=LANG_CHOICES[0][0])
        self.lang_menu = ctk.CTkOptionMenu(
            bar, values=[n for n, _ in LANG_CHOICES], variable=self.lang_var,
            command=self._on_lang_change,
            width=110, height=36, corner_radius=10, font=self.ui_font,
            fg_color=NEUTRAL, button_color=NEUTRAL, button_hover_color=NEUTRAL_HOVER)
        self.lang_menu.pack(side="right")
        ctk.CTkLabel(bar, text="Language", font=self.ui_font,
                     text_color=TEXT_MUTED).pack(side="right", padx=(0, 6))

        default_model = (config.MODEL_SIZE
                         if config.MODEL_SIZE in MODEL_CHOICES else "small")
        self.model_var = ctk.StringVar(value=default_model)
        self.model_menu = ctk.CTkOptionMenu(
            bar, values=MODEL_CHOICES, variable=self.model_var,
            command=self._on_model_change,
            width=132, height=36, corner_radius=10, font=self.ui_font,
            fg_color=NEUTRAL, button_color=NEUTRAL, button_hover_color=NEUTRAL_HOVER)
        self.model_menu.pack(side="right", padx=(0, 14))
        ctk.CTkLabel(bar, text="Model", font=self.ui_font,
                     text_color=TEXT_MUTED).pack(side="right", padx=(0, 6))

    def _build_statusbar(self):
        bar = ctk.CTkFrame(self.root, fg_color=PANEL, corner_radius=12, height=44)
        bar.pack(side="bottom", fill="x", padx=16, pady=(6, 14))
        bar.pack_propagate(False)

        self.status_dot = ctk.CTkLabel(bar, text="●", font=self.ui_font,
                                       text_color=TEXT_MUTED, width=16)
        self.status_dot.pack(side="left", padx=(14, 4))
        self.status_var = ctk.StringVar(value="Idle")
        ctk.CTkLabel(bar, textvariable=self.status_var, font=self.ui_font,
                     text_color="#d4d4de").pack(side="left")

        self.devices_btn = ctk.CTkButton(
            bar, text="Devices", command=lambda: list_input_devices(),
            width=80, height=30, corner_radius=8, font=self.ui_font,
            fg_color="transparent", hover_color=NEUTRAL,
            border_width=1, border_color=BORDER, text_color="#c9c9d4")
        self.devices_btn.pack(side="right", padx=(0, 8), pady=7)

    def _build_tabs(self):
        self.tabs = ctk.CTkTabview(
            self.root, fg_color=PANEL, corner_radius=12,
            command=self._on_tab_change,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color=ACCENT_HOVER)
        self.tabs.pack(fill="both", expand=True, padx=16, pady=0)
        stt_tab = self.tabs.add("🎙  Live STT")
        self._translator_tab_name = "🌐  Translator"
        trans_tab = self.tabs.add(self._translator_tab_name)
        file_tab = self.tabs.add("📁  Audio File")
        min_tab = self.tabs.add("📝  Minutes")
        self._build_stt_tab(stt_tab)
        self._build_translator_tab(trans_tab)
        self._build_file_tab(file_tab)
        self._build_minutes_tab(min_tab)

    def _build_stt_tab(self, tab):
        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.pack(fill="x", padx=4, pady=(6, 2))
        ctk.CTkLabel(bar, text="Live captions → transcript .txt (saved automatically)",
                     font=self.ui_font, text_color=TEXT_MUTED).pack(side="left", padx=(4, 0))
        self.save_btn = ctk.CTkButton(
            bar, text="💾  Save txt", command=self._on_save,
            width=104, height=34, corner_radius=10, font=self.ui_font,
            fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER)
        self.save_btn.pack(side="right")

        # Raw-audio archival. The engine reads config.SAVE_AUDIO once at start, so a mid-session
        # toggle can only take effect on the next Start — say so instead of silently doing nothing.
        self.save_audio_var = ctk.BooleanVar(value=config.SAVE_AUDIO)
        self.save_audio_switch = ctk.CTkSwitch(
            bar, text="🎧  Save audio", variable=self.save_audio_var,
            command=self._on_save_audio_toggle, font=self.ui_font,
            progress_color=ACCENT, text_color="#c9c9d4")
        self.save_audio_switch.pack(side="right", padx=(0, 12))

        # Where everything is written. Shows the folder name; the full path is in the toast.
        self.outdir_btn = ctk.CTkButton(
            bar, text="\U0001f4c2", command=self._on_pick_output_dir,
            width=170, height=34, corner_radius=10, font=self.ui_font,
            fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER)
        self.outdir_btn.pack(side="right", padx=(0, 12))
        self._refresh_outdir_btn()

        self.text = ctk.CTkTextbox(
            tab, font=self.mono_font, corner_radius=10,
            fg_color="#121218", text_color="#e8e8ee", wrap="word",
            border_spacing=6)
        self.text.pack(fill="both", expand=True, padx=4, pady=6)
        self.text.configure(state="disabled")
        self.text.tag_config("ts", foreground=TEXT_MUTED)

    def _build_translator_tab(self, tab):
        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.pack(fill="x", padx=4, pady=(6, 2))
        ctk.CTkLabel(bar, text="Translate to", font=self.ui_font,
                     text_color=TEXT_MUTED).pack(side="left")
        self.translation_lang_var = ctk.StringVar(value="Korean")
        self.translation_lang_menu = ctk.CTkOptionMenu(
            bar, values=list(translator.TARGET_LANGUAGES.keys()),
            variable=self.translation_lang_var,
            command=self._on_translation_lang_change,
            width=112, height=34, corner_radius=10, font=self.ui_font,
            fg_color=NEUTRAL, button_color=NEUTRAL,
            button_hover_color=NEUTRAL_HOVER)
        self.translation_lang_menu.pack(side="left", padx=(8, 0))
        self.translation_save_btn = ctk.CTkButton(
            bar, text="💾  Save", command=self._on_save_translation,
            width=104, height=34, corner_radius=10, font=self.ui_font,
            state="disabled", fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER)
        self.translation_save_btn.pack(side="left", padx=(8, 0))
        self.translation_hint = ctk.CTkLabel(
            bar, text="Live translation — every line as it's transcribed",
            font=self.ui_font, text_color=TEXT_MUTED)
        self.translation_hint.pack(side="left", padx=(12, 0))

        self.translation_text = ctk.CTkTextbox(
            tab, font=self.mono_font, corner_radius=10,
            fg_color="#121218", text_color="#e8e8ee", wrap="word",
            border_spacing=6)
        self.translation_text.pack(fill="both", expand=True, padx=4, pady=6)
        self.translation_text.configure(state="disabled")

    def _build_file_tab(self, tab):
        """Mode 2: rebuild a transcript from an audio FILE, after the meeting.

        Mode 1 (the toolbar Start/Stop) tags speakers by audio source, which is exact but only
        separates people who were on different sources -- an in-person meeting puts everyone on the
        one mic. This mode re-transcribes the audio and separates speakers from the sound itself,
        and because it works off audio-relative time it also gets timestamps that actually line up
        (the realtime ones are stamped when transcription finished, several seconds late).
        """
        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.pack(fill="x", padx=4, pady=(6, 2))
        self.file_pick_btn = ctk.CTkButton(
            bar, text="📁  Choose audio…", command=self._on_pick_file,
            width=150, height=34, corner_radius=10, font=self.ui_font,
            fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER)
        self.file_pick_btn.pack(side="left")
        self.file_run_btn = ctk.CTkButton(
            bar, text="✨  Rebuild", command=self._on_run_file,
            width=120, height=34, corner_radius=10, font=self.ui_bold,
            state="disabled", fg_color=ACCENT, hover_color=ACCENT_HOVER)
        self.file_run_btn.pack(side="left", padx=(8, 0))
        self.file_save_btn = ctk.CTkButton(
            bar, text="💾  Save", command=self._on_save_file,
            width=100, height=34, corner_radius=10, font=self.ui_font,
            state="disabled", fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER)
        self.file_save_btn.pack(side="left", padx=(8, 0))

        ctk.CTkLabel(bar, text="Speakers", font=self.ui_font,
                     text_color=TEXT_MUTED).pack(side="left", padx=(16, 6))
        # Defaults to 2, not Auto: naming the exact count is markedly more reliable than
        # threshold clustering (which collapsed two speakers into one at the wrong threshold during
        # testing), and a 1:1 interview -- the main use here -- is always 2.
        self.file_speakers_var = ctk.StringVar(value="2")
        ctk.CTkOptionMenu(
            bar, values=["Auto", "2", "3", "4", "5"], variable=self.file_speakers_var,
            width=84, height=34, corner_radius=10, font=self.ui_font,
            fg_color=NEUTRAL, button_color=NEUTRAL,
            button_hover_color=NEUTRAL_HOVER).pack(side="left")

        self.file_typo_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(bar, text="Fix typos", variable=self.file_typo_var,
                      font=self.ui_font, progress_color=ACCENT,
                      text_color="#c9c9d4").pack(side="left", padx=(14, 0))

        self.file_hint = ctk.CTkLabel(
            tab, text="Pick a recording (wav / m4a / mp3). Re-transcribes and separates speakers "
                      "from the audio — slower than live, but far more accurate.",
            font=self.ui_font, text_color=TEXT_MUTED, anchor="w")
        self.file_hint.pack(fill="x", padx=8, pady=(4, 0))

        self.file_text = ctk.CTkTextbox(
            tab, font=self.mono_font, corner_radius=10,
            fg_color="#121218", text_color="#e8e8ee", wrap="word",
            border_spacing=6)
        self.file_text.pack(fill="both", expand=True, padx=4, pady=6)
        self.file_text.configure(state="disabled")

    def _set_file_text(self, s, markdown=True):
        """Show `s`. Rendered as markdown by default; plain for error/status text."""
        self.file_md = s or ""
        if markdown:
            markdown_view.render(self.file_text, self.file_md, body_size=13)
            return
        self.file_text.configure(state="normal")
        self.file_text.delete("1.0", "end")
        if s:
            self.file_text.insert("end", s)
        self.file_text.configure(state="disabled")

    def _on_pick_file(self):
        path = filedialog.askopenfilename(
            title="Choose a recording",
            initialdir=os.path.abspath(config.OUTPUT_DIR),
            filetypes=[("Audio", "*.wav *.m4a *.mp3 *.aiff *.aif *.caf *.flac *.mp4"),
                       ("All files", "*.*")])
        if not path:
            return
        self.file_path = path
        self.file_run_btn.configure(state="normal")
        self.file_hint.configure(text=f"Ready: {os.path.basename(path)}")

    def _on_run_file(self):
        if self.file_busy or not getattr(self, "file_path", None):
            return
        if not diarize.is_available():
            self._set_file_text(
                "Speaker-separation models are missing.\n\n"
                f"Expected under: {os.path.abspath(config.DIARIZATION_DIR)}\n"
                "See config.py — the download URLs are in the batch-mode section.",
                markdown=False)
            return
        n = self.file_speakers_var.get()
        num_speakers = None if n == "Auto" else int(n)
        self.file_busy = True
        self.file_run_btn.configure(state="disabled")
        self.file_pick_btn.configure(state="disabled")
        self.file_save_btn.configure(state="disabled")
        self._set_file_text("")
        self.file_hint.configure(text="Working… (re-transcribe → separate speakers → clean up)")
        threading.Thread(target=self._file_worker,
                         args=(self.file_path, num_speakers, bool(self.file_typo_var.get())),
                         daemon=True).start()

    def _file_worker(self, path, num_speakers, fix_typos):
        try:
            result = batch.transcribe_file(
                path, num_speakers=num_speakers, fix_typos=fix_typos,
                on_status=lambda msg: self.file_queue.put(("fstatus", msg)))
            self.file_queue.put(("fdone", result))
        except Exception as e:  # noqa: BLE001
            self.file_queue.put(("ferror", str(e)))

    def _on_save_file(self):
        content = self.file_md.strip()   # the markdown, not the rendering shown in the widget
        if not content:
            self._toast("Nothing to save.")
            return
        base = os.path.splitext(os.path.basename(getattr(self, "file_path", "audio")))[0]
        path = self._ask_save_path(f"rebuilt_{base}.md")
        if not path:
            return
        _open_private(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        self._toast(f"Saved: {os.path.basename(path)}")

    def _build_minutes_tab(self, tab):
        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.pack(fill="x", padx=4, pady=(6, 2))
        self.make_btn = ctk.CTkButton(
            bar, text="✨  Generate", command=self._on_make_minutes,
            width=124, height=34, corner_radius=10, font=self.ui_bold,
            fg_color=ACCENT, hover_color=ACCENT_HOVER)
        self.make_btn.pack(side="left")
        self.minutes_save_btn = ctk.CTkButton(
            bar, text="💾  Save", command=self._on_save_minutes,
            width=124, height=34, corner_radius=10, font=self.ui_font,
            state="disabled", fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER)
        self.minutes_save_btn.pack(side="left", padx=(8, 0))
        self.minutes_hint = ctk.CTkLabel(
            bar, text="Full transcript → local LLM minutes",
            font=self.ui_font, text_color=TEXT_MUTED)
        self.minutes_hint.pack(side="left", padx=(12, 0))

        self.minutes_text = ctk.CTkTextbox(
            tab, font=self.mono_font, corner_radius=10,
            fg_color="#121218", text_color="#e8e8ee", wrap="word",
            border_spacing=6)
        self.minutes_text.pack(fill="both", expand=True, padx=4, pady=6)
        self.minutes_text.configure(state="disabled")

    # ---------- actions ----------
    def _clear_captions(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        self.session_lines = []
        self._utterances = []
        self._live_translate_idx = 0
        self._set_translation_text("")
        self.translation_save_btn.configure(state="disabled")
        self._set_file_text("")
        self.file_save_btn.configure(state="disabled")
        self._set_minutes_text("")
        self.minutes_save_btn.configure(state="disabled")

    def _set_translation_text(self, s):
        self.translation_text.configure(state="normal")
        self.translation_text.delete("1.0", "end")
        if s:
            self.translation_text.insert("end", s)
        self.translation_text.configure(state="disabled")

    def _set_minutes_text(self, s):
        self.minutes_md = s or ""
        self._minutes_last_render = 0.0
        self._render_minutes()

    def _set_status(self, msg, color=None):
        self.status_var.set(msg)
        if color:
            self.status_dot.configure(text_color=color)

    def _on_lang_change(self, choice):
        """Language dropdown change -> apply to config immediately. If running, takes effect from the next chunk."""
        lang = dict(LANG_CHOICES)[choice]
        config.LANGUAGE = lang
        if self.ctrl.running:
            self._toast(f"Language → {choice} (applies to next chunk)")

    def _on_model_change(self, choice):
        """Model dropdown change. If running, the worker reloads without interruption (a few seconds if cached)."""
        if self.ctrl.running:
            self.ctrl.request_model(choice)
            self._toast(f"Switching model → {choice} … (reloading)")

    def _on_start(self):
        lang = dict((n, v) for n, v in LANG_CHOICES)[self.lang_var.get()]
        # New session (New pressed, or first start) -> issue a timestamp = new file.
        # Otherwise (restart after Stop) -> keep the existing timestamp = append to the same file.
        if self.session_started is None:
            self.session_started = datetime.now()
        self.start_btn.configure(state="disabled")
        self.new_btn.configure(state="disabled")
        # Keep model/language enabled so they can be changed while running
        self.stop_btn.configure(state="normal")
        self._set_status("Starting…", GREEN)
        self.ctrl.start(self.model_var.get(), lang, self.session_started)

    def _on_stop(self):
        self._set_status("Stopping…", "#eab308")
        self.stop_btn.configure(state="disabled")
        self.ctrl.stop()

    def _on_new(self):
        if self.ctrl.running:
            self._toast("Stop first, then start a new session.")
            return
        self._clear_captions()
        self.session_started = None   # next Start = new file
        self._set_status("Idle", TEXT_MUTED)

    def _on_save_audio_toggle(self):
        config.SAVE_AUDIO = bool(self.save_audio_var.get())
        state = "on" if config.SAVE_AUDIO else "off"
        if self.ctrl.running:
            self._toast(f"Audio saving {state} — applies from the next session (Stop → Start).")
        else:
            self._toast(f"Audio saving {state}"
                        + (" · ~1.9 MB/min per source" if config.SAVE_AUDIO else ""))

    # ---------- where output is saved ----------
    def _refresh_outdir_btn(self):
        name = os.path.basename(config.OUTPUT_DIR.rstrip(os.sep)) or config.OUTPUT_DIR
        self.outdir_btn.configure(text=f"\U0001f4c2  {name}")

    def _use_output_dir(self, folder):
        """Adopt `folder` as where everything is saved -- now, and on every launch after.

        This is the whole "remember where I last saved" behaviour: it is called both from the
        folder button and from every Save dialog, so simply saving somewhere else once is
        enough to move the default there.
        """
        folder = os.path.abspath(folder)
        if folder == os.path.abspath(config.OUTPUT_DIR):
            return
        if not settings.is_usable(folder):
            self._toast(f"Cannot write to {folder}")
            return
        config.OUTPUT_DIR = folder
        settings.set_output_dir(folder)
        self._refresh_outdir_btn()

    def _on_pick_output_dir(self):
        folder = filedialog.askdirectory(
            title="Where to save transcripts, audio, minutes and translations",
            initialdir=config.OUTPUT_DIR, mustexist=False)
        if not folder:
            return
        self._use_output_dir(folder)
        if self.ctrl.running:
            # writer.py and audio_writer.py open their files at Start, so the session already
            # running keeps writing where it began -- say so instead of implying it moved.
            self._toast(f"Saving to {config.OUTPUT_DIR} from the next session "
                        f"(the running one keeps its current folder)")
        else:
            self._toast(f"Saving to {config.OUTPUT_DIR}")

    def _ask_save_path(self, default_name):
        """Ask where to write `default_name`, starting in the current output folder.

        Returns the chosen path, or None if the dialog was cancelled. Choosing a different
        folder makes it the default for everything saved afterwards.
        """
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save as",
            initialdir=config.OUTPUT_DIR,
            initialfile=default_name,
            defaultextension=os.path.splitext(default_name)[1],
            filetypes=[("Markdown", "*.md"), ("All files", "*.*")])
        if not path:
            return None
        self._use_output_dir(os.path.dirname(path))
        return path

    def _on_save(self):
        path = self.ctrl.save()
        if not path:
            self._toast("Nothing to save.")
            return
        # The folder, not the full path: the status bar is narrow, and where it landed is the
        # part that is worth confirming now that the folder is settable.
        where = os.path.basename(os.path.dirname(path)) or os.path.dirname(path)
        audio = self.ctrl.engine.audio_paths()
        if audio:
            names = ", ".join(sorted(audio))       # speaker labels, not full paths
            self._toast(f"Saved to {where}/: {os.path.basename(path)}  +  audio ({names})")
        else:
            self._toast(f"Saved to {where}/: {os.path.basename(path)}")

    # ---------- real-time translation ----------
    def _on_tab_change(self):
        """Opening the Translator tab turns on live translation; leaving it turns it off."""
        if self.tabs.get() == self._translator_tab_name:
            self._enable_live_translate()
        else:
            self.live_translate = False

    def _enable_live_translate(self):
        """Turn on live translation: every utterance is rendered in the target language."""
        self.live_translate = True
        self._ensure_live_translate_worker()
        target = self.translation_lang_var.get()
        if translator.is_available():
            self.translation_hint.configure(text=f"🔴 Live → {target}")
        else:
            # Same-language lines still render as-is; only cross-language lines need Ollama.
            self.translation_hint.configure(
                text=f"🔴 Live → {target} (translation needs: ollama serve)")
        self._pump_live_translate()   # flush any backlog transcribed before the tab was opened

    def _on_translation_lang_change(self, _choice=None):
        """Changing the target language re-renders the whole transcript in that language."""
        self._live_translate_idx = 0          # replay every utterance under the new target
        self._set_translation_text("")
        self.translation_save_btn.configure(state="disabled")
        if self.live_translate or self.tabs.get() == self._translator_tab_name:
            self._enable_live_translate()

    def _ensure_live_translate_worker(self):
        if self.live_translate_thread is None:
            self.live_translate_thread = threading.Thread(
                target=self._live_translate_worker, daemon=True)
            self.live_translate_thread.start()

    def _pump_live_translate(self):
        """Queue every not-yet-rendered utterance for the worker (with its detected language)."""
        target = self.translation_lang_var.get()
        while self._live_translate_idx < len(self._utterances):
            ts, speaker, lang, text = self._utterances[self._live_translate_idx]
            self._live_translate_idx += 1
            if text.strip():
                put_drop_oldest(self.live_translate_queue, (ts, speaker, lang, text, target))

    def _live_translate_worker(self):
        """Background: one utterance at a time. Same language -> keep as-is; else translate."""
        while True:
            ts, speaker, lang, text, target = self.live_translate_queue.get()
            if lang and lang == TARGET_LANG_CODE.get(target):
                out = text.strip()          # already in the target language -> no LLM call
            else:
                try:
                    out = translator.translate_line(text, target_language=target)
                except Exception as e:  # noqa: BLE001
                    out = f"[translate error: {e}]"
            who = f"{speaker} · " if speaker else ""
            block = f"{ts}  {who}{target}\n{out}\n\n"
            self.translation_queue.put(("ltline", block))

    def _on_save_translation(self):
        content = self.translation_text.get("1.0", "end").strip()
        if not content:
            self._toast("No translation to save.")
            return
        stamp = (self.session_started or datetime.now()).strftime("%Y-%m-%d_%H-%M")
        target = self.translation_lang_var.get().lower()
        path = self._ask_save_path(f"translation_{target}_{stamp}.md")
        if not path:
            return
        _open_private(path)   # 0600 before writing — same reasoning as writer.Writer
        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        self._toast(f"Translation saved: {os.path.basename(path)}")

    # ---------- minutes (local LLM) ----------
    def _on_make_minutes(self):
        if self.minutes_busy:
            return
        if not self.session_lines:
            self._toast("Need transcribed text first.")
            return
        if not minutes.is_available():
            self._toast("Ollama not running — run 'ollama serve' then retry")
            self.minutes_hint.configure(
                text=f"Ollama off: ollama serve + ollama pull {config.OLLAMA_MODEL}")
            return
        transcript = "\n".join(self.session_lines)
        when = (self.session_started.strftime("%Y-%m-%d %H:%M")
                if self.session_started else "")
        self.minutes_busy = True
        self.make_btn.configure(state="disabled")
        self.minutes_save_btn.configure(state="disabled")
        self.minutes_hint.configure(
            text=f"Generating… ({config.OLLAMA_MODEL})")
        self._set_minutes_text("")
        threading.Thread(
            target=self._minutes_worker, args=(transcript, when), daemon=True
        ).start()

    def _minutes_worker(self, transcript, when):
        try:
            minutes.generate_minutes(
                transcript, when=when,
                on_token=lambda tok: self.minutes_queue.put(("mtoken", tok)))
            self.minutes_queue.put(("mdone", None))
        except Exception as e:  # noqa: BLE001
            self.minutes_queue.put(("merror", str(e)))

    def _on_save_minutes(self):
        content = self.minutes_md.strip()   # the markdown, not the rendering shown in the widget
        if not content:
            self._toast("No minutes to save.")
            return
        stamp = (self.session_started or datetime.now()).strftime("%Y-%m-%d_%H-%M")
        path = self._ask_save_path(f"minutes_{stamp}.md")
        if not path:
            return
        _open_private(path)   # 0600 before writing — same reasoning as writer.Writer
        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        self._toast(f"Minutes saved: {os.path.basename(path)}")

    # ---------- render ----------
    def _speaker_tag(self, speaker):
        if speaker not in self._speaker_colors:
            color = SPEAKER_PALETTE[len(self._speaker_colors) % len(SPEAKER_PALETTE)]
            self._speaker_colors[speaker] = color
            self.text.tag_config(f"sp_{speaker}", foreground=color)
        return f"sp_{speaker}"

    def _append(self, ts, speaker, lang, text):
        self.text.configure(state="normal")
        self.text.insert("end", f"{ts}  ", ("ts",))
        who = f"{speaker} · " if speaker else ""
        tag = self._speaker_tag(speaker) if speaker else "ts"
        self.text.insert("end", f"{who}{lang}\n", (tag,))
        self.text.insert("end", f"{text}\n\n", (tag,))
        self.text.see("end")
        self.text.configure(state="disabled")
        who_txt = f"[{speaker}] " if speaker else ""
        self.session_lines.append(f"[{ts}] {who_txt}({lang}) {text}")
        self._utterances.append((ts, speaker, lang, text))
        if self.live_translate:
            self._pump_live_translate()

    def _append_minutes_token(self, tok):
        # Render as it streams, throttled. Rendering per token would re-parse a growing document
        # on every token; rendering ONLY at the end meant that any path which never reached the
        # "done" event -- a generation error, a dropped connection -- left the raw `##`/`**`
        # markdown on screen for good. Throttled re-render keeps the formatting correct at every
        # moment regardless of how generation ends.
        self.minutes_md += tok
        now = time.monotonic()
        if now - self._minutes_last_render < MINUTES_RENDER_INTERVAL_SEC:
            return
        self._minutes_last_render = now
        self._render_minutes()

    def _render_minutes(self):
        """Show self.minutes_md as formatted markdown.

        Guarded: this runs inside the Tk poll loop, and an exception escaping there would kill
        the `root.after` chain -- the whole UI would stop updating, with the raw markdown frozen
        on screen as the only symptom. On failure, fall back to the plain text so the content is
        never lost, and say why on stderr.
        """
        try:
            markdown_view.render(self.minutes_text, self.minutes_md, body_size=13)
        except Exception as e:  # noqa: BLE001 - display must degrade, never crash the poll loop
            print(f"[minutes] markdown render failed ({e!r}) - showing raw text",
                  file=sys.stderr, flush=True)
            self.minutes_text.configure(state="normal")
            self.minutes_text.delete("1.0", "end")
            self.minutes_text.insert("end", self.minutes_md)
            self.minutes_text.configure(state="disabled")
        self.minutes_text.see("end")

    def _append_translation_token(self, tok):
        self.translation_text.configure(state="normal")
        self.translation_text.insert("end", tok)
        self.translation_text.see("end")
        self.translation_text.configure(state="disabled")

    def _idle_buttons(self):
        self.start_btn.configure(state="normal")
        self.new_btn.configure(state="normal")
        self.model_menu.configure(state="normal")
        self.lang_menu.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def _toast(self, msg):
        """Briefly show a message in the status bar (simple toast substitute)."""
        self._set_status(msg)

    def _diag(self):
        """Main-thread health probe. Detects when the UI loop is being starved/blocked
        (the beach ball) and logs resource stats so the real cause is caught on the next run."""
        now = time.monotonic()
        gap = now - self._diag_last_poll
        self._diag_last_poll = now
        if gap > self._diag_max_gap:
            self._diag_max_gap = gap
        # A stall: main loop didn't run for far longer than the 100ms schedule.
        if gap > 0.6:
            aq = "-"
            try:
                cap = getattr(self.ctrl.engine, "capture", None)
                if cap is not None:
                    aq = cap.audio_queue.qsize()
            except Exception:
                pass
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # MB (macOS: bytes)
            print(f"[diag] MAIN-THREAD STALL {gap:.2f}s  audio_q={aq}  line_q={self.ctrl.line_queue.qsize()}  "
                  f"utter={len(self._utterances)}  rssMB={rss:.0f}", file=sys.stderr, flush=True)
        # Periodic heartbeat every ~15s so we see trend even without a stall.
        if now - self._diag_last_stats > 15.0:
            self._diag_last_stats = now
            aq = "-"
            try:
                cap = getattr(self.ctrl.engine, "capture", None)
                if cap is not None:
                    aq = cap.audio_queue.qsize()
            except Exception:
                pass
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
            print(f"[diag] alive running={self.ctrl.running}  audio_q={aq}  line_q={self.ctrl.line_queue.qsize()}  "
                  f"utter={len(self._utterances)}  maxgap={self._diag_max_gap:.2f}s  rssMB={rss:.0f}",
                  file=sys.stderr, flush=True)
            self._diag_max_gap = 0.0

    def _poll_queue(self):
        self._diag()
        try:
            while True:
                kind, payload = self.ctrl.line_queue.get_nowait()
                if kind == "line":
                    ts, speaker, lang, text = payload
                    self._append(ts, speaker, lang, text)
                elif kind == "status":
                    if payload == "Stopped":
                        self._set_status("Stopped", TEXT_MUTED)
                        self._idle_buttons()
                    elif payload.startswith("Transcribing"):
                        self._set_status(payload, GREEN)
                    else:
                        self._set_status(payload)
                elif kind == "error":
                    self._set_status("Error", "#ef4444")
                    self._idle_buttons()
                    self._toast(payload)
        except queue.Empty:
            pass

        try:
            while True:
                kind, payload = self.translation_queue.get_nowait()
                if kind == "ltline":
                    self._append_translation_token(payload)
                    self.translation_save_btn.configure(state="normal")
        except queue.Empty:
            pass

        try:
            while True:
                kind, payload = self.file_queue.get_nowait()
                if kind == "fstatus":
                    self.file_hint.configure(text=payload)
                elif kind == "fdone":
                    self._set_file_text(payload["markdown"])
                    self.file_busy = False
                    self.file_run_btn.configure(state="normal")
                    self.file_pick_btn.configure(state="normal")
                    self.file_save_btn.configure(state="normal")
                    n = len({t["speaker"] for t in payload["turns"] if t["speaker"] is not None})
                    self.file_hint.configure(
                        text=f"Done · {len(payload['turns'])} turns · {n} speaker(s) · 💾 Save")
                elif kind == "ferror":
                    self.file_busy = False
                    self.file_run_btn.configure(state="normal")
                    self.file_pick_btn.configure(state="normal")
                    self.file_hint.configure(text="Failed")
                    self._set_file_text(payload, markdown=False)
        except queue.Empty:
            pass

        try:
            while True:
                kind, payload = self.minutes_queue.get_nowait()
                if kind == "mtoken":
                    self._append_minutes_token(payload)
                elif kind == "mdone":
                    self._render_minutes()
                    self.minutes_busy = False
                    self.make_btn.configure(state="normal")
                    self.minutes_save_btn.configure(state="normal")
                    self.minutes_hint.configure(text="Done · 💾 Save to keep it")
                elif kind == "merror":
                    # Render whatever did arrive: a partial answer is still markdown, and leaving
                    # it raw is exactly the bug this path used to produce.
                    self._render_minutes()
                    self.minutes_busy = False
                    self.make_btn.configure(state="normal")
                    self.minutes_hint.configure(text="Failed")
                    self._toast(payload)
        except queue.Empty:
            pass

        self.root.after(100, self._poll_queue)

    def _on_close(self):
        if self.ctrl.running:
            self.ctrl.stop()
            self.root.after(300, self.root.destroy)
        else:
            self.root.destroy()


def main():
    root = ctk.CTk()
    App(root)
    # After Tk owns the NSApplication, never before -- see dock_icon.apply().
    dock_icon.apply()
    root.mainloop()


if __name__ == "__main__":
    main()
