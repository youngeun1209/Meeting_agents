"""STT transcription GUI — modern UI (customtkinter).

- Tab 1 🎙 Live STT: real-time scrolling captions colored per speaker
- Tab 2 ⚡ Live Summary: per-speaker live accumulation
- Tab 3 🌐 Translator: translate the transcript with a local LLM (Ollama)
- Tab 4 📝 Minutes: turn the transcript into minutes with a local LLM (Ollama)
- Start/Stop/New session, model & language selection, automatic txt saving
"""
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import os
import queue
import threading
from datetime import datetime

import customtkinter as ctk

import config
import minutes
import translator
from audio_capture import list_input_devices
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
SPEAKER_PALETTE = ["#7ec8ff", "#9cffb0", "#ffcf7e",
                   "#ff9ec8", "#c8a0ff", "#a0ffe8"]

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


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
        self._speaker_colors = {}
        self._live_summary = []      # [[speaker, [text, ...]], ...] merges consecutive same-speaker turns

        # --- real-time translation ---
        self._utterances = []                 # [(ts, speaker, lang, text), ...] raw, for live translate
        self._live_translate_idx = 0          # how many utterances have been queued for translation
        self.live_translate = False           # on when the Translator tab is open + Ollama is up
        self.live_translate_queue = queue.Queue()   # feed to the live-translate worker
        self.live_translate_thread = None

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
        live_tab = self.tabs.add("⚡  Live Summary")
        self._translator_tab_name = "🌐  Translator"
        trans_tab = self.tabs.add(self._translator_tab_name)
        min_tab = self.tabs.add("📝  Minutes")
        self._build_stt_tab(stt_tab)
        self._build_live_tab(live_tab)
        self._build_translator_tab(trans_tab)
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

        self.text = ctk.CTkTextbox(
            tab, font=self.mono_font, corner_radius=10,
            fg_color="#121218", text_color="#e8e8ee", wrap="word",
            border_spacing=6)
        self.text.pack(fill="both", expand=True, padx=4, pady=6)
        self.text.configure(state="disabled")
        self.text.tag_config("ts", foreground=TEXT_MUTED)

    def _build_live_tab(self, tab):
        hint = ctk.CTkLabel(
            tab, text="Live per-speaker accumulation (no LLM). Generate the polished version in the Minutes tab.",
            font=self.ui_font, text_color=TEXT_MUTED)
        hint.pack(fill="x", padx=8, pady=(8, 2))
        self.live_text = ctk.CTkTextbox(
            tab, font=self.mono_font, corner_radius=10,
            fg_color="#121218", text_color="#e8e8ee", wrap="word",
            border_spacing=6)
        self.live_text.pack(fill="both", expand=True, padx=4, pady=6)
        self.live_text.configure(state="disabled")

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
        self._live_summary = []
        self.live_text.configure(state="normal")
        self.live_text.delete("1.0", "end")
        self.live_text.configure(state="disabled")
        self._set_translation_text("")
        self.translation_save_btn.configure(state="disabled")
        self._set_minutes_text("")
        self.minutes_save_btn.configure(state="disabled")

    def _set_translation_text(self, s):
        self.translation_text.configure(state="normal")
        self.translation_text.delete("1.0", "end")
        if s:
            self.translation_text.insert("end", s)
        self.translation_text.configure(state="disabled")

    def _set_minutes_text(self, s):
        self.minutes_text.configure(state="normal")
        self.minutes_text.delete("1.0", "end")
        if s:
            self.minutes_text.insert("end", s)
        self.minutes_text.configure(state="disabled")

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

    def _on_save(self):
        path = self.ctrl.save()
        if path:
            self._toast(f"Saved: {os.path.basename(path)}")
        else:
            self._toast("Nothing to save.")

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
                self.live_translate_queue.put((ts, speaker, lang, text, target))

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
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        stamp = (self.session_started or datetime.now()).strftime("%Y-%m-%d_%H-%M")
        target = self.translation_lang_var.get().lower()
        path = os.path.join(config.OUTPUT_DIR, f"translation_{target}_{stamp}.md")
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
        content = self.minutes_text.get("1.0", "end").strip()
        if not content:
            self._toast("No minutes to save.")
            return
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        stamp = (self.session_started or datetime.now()).strftime("%Y-%m-%d_%H-%M")
        path = os.path.join(config.OUTPUT_DIR, f"minutes_{stamp}.md")
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
        self._update_live_summary(speaker, text)
        if self.live_translate:
            self._pump_live_translate()

    def _update_live_summary(self, speaker, text):
        """Accumulate per-speaker utterances live without an LLM. Merge consecutive same-speaker turns into one block."""
        if not text.strip():
            return
        who = speaker or "?"
        if self._live_summary and self._live_summary[-1][0] == who:
            self._live_summary[-1][1].append(text.strip())
        else:
            self._live_summary.append([who, [text.strip()]])
        self._render_live_summary()

    def _render_live_summary(self):
        blocks = []
        for who, texts in self._live_summary:
            blocks.append(f"● {who}\n   {' '.join(texts)}")
        self.live_text.configure(state="normal")
        self.live_text.delete("1.0", "end")
        self.live_text.insert("end", "\n\n".join(blocks))
        self.live_text.see("end")
        self.live_text.configure(state="disabled")

    def _append_minutes_token(self, tok):
        self.minutes_text.configure(state="normal")
        self.minutes_text.insert("end", tok)
        self.minutes_text.see("end")
        self.minutes_text.configure(state="disabled")

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

    def _poll_queue(self):
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
                kind, payload = self.minutes_queue.get_nowait()
                if kind == "mtoken":
                    self._append_minutes_token(payload)
                elif kind == "mdone":
                    self.minutes_busy = False
                    self.make_btn.configure(state="normal")
                    self.minutes_save_btn.configure(state="normal")
                    self.minutes_hint.configure(text="Done · edit then 💾 Save")
                elif kind == "merror":
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
    root.mainloop()


if __name__ == "__main__":
    main()
