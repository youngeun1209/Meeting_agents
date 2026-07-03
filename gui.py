"""STT transcription GUI — modern UI (customtkinter).

- Tab 1 🎙 Live STT: real-time scrolling captions colored per speaker
- Tab 2 📝 Minutes: turn the transcript into minutes with a local LLM (Ollama)
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
from audio_capture import MultiCapture, list_input_devices
from vad_buffer import VADBuffer
from writer import Writer

# Transcriber is heavy (model load) -> import/create it when Start is pressed.

MODEL_CHOICES = ["base", "small", "medium", "large-v3-turbo", "large-v3"]
LANG_CHOICES = [("Auto", None), ("Korean", "ko"), ("English", "en")]

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
    """Runs the pipeline on a background thread and passes caption lines to the GUI via a queue."""

    def __init__(self):
        self.line_queue = queue.Queue()    # ('line'|'status'|'error', payload)
        self.stop_event = threading.Event()
        self.worker = None
        self.capture = None
        self.writer = None
        self.running = False
        self._pending_model = None   # model swap request while running (string)

    def request_model(self, model_size):
        """Request a model swap while running. The worker reloads on the next loop."""
        self._pending_model = model_size

    def _emit_line(self, line, lang, text, ts, speaker):
        self.line_queue.put(("line", (ts, speaker, lang, text)))

    def _status(self, msg):
        self.line_queue.put(("status", msg))

    def start(self, model_size, language, session_started=None):
        if self.running:
            return
        self.stop_event.clear()
        self.running = True
        self.worker = threading.Thread(
            target=self._run, args=(model_size, language, session_started),
            daemon=True,
        )
        self.worker.start()

    def _run(self, model_size, language, session_started=None):
        try:
            config.MODEL_SIZE = model_size
            config.LANGUAGE = language
            self._status(f"Loading model: {model_size} …")
            from transcriber import Transcriber  # lazy import
            stt = Transcriber()

            # Starting again with the same session_started appends to the same file
            self.writer = Writer(on_line=self._emit_line, echo=False,
                                 started_at=session_started)
            self.capture = MultiCapture(config.SOURCES)
            self.capture.start()

            vads = {sp: VADBuffer() for sp in self.capture.speakers()}
            who = " + ".join(self.capture.speakers())
            current_model = model_size
            self._pending_model = None
            self._status(f"Transcribing · {who} · {current_model}")

            # Background model loader: keep transcribing with the old model until the new one is ready,
            # then swap the moment it finishes loading (no interruption).
            loader = {"thread": None, "model": None, "stt": None, "error": None}

            def _load(model_name, box):
                try:
                    config.MODEL_SIZE = model_name
                    box["stt"] = Transcriber()   # download+load (seconds to tens of seconds)
                    box["model"] = model_name
                except Exception as e:  # noqa: BLE001
                    box["error"] = str(e)

            while not self.stop_event.is_set():
                # 1) If there is a swap request and nothing is loading -> start a background load
                pending = self._pending_model
                if pending and pending != current_model and loader["thread"] is None:
                    self._pending_model = None
                    self._status(f"Loading {pending} … (still using {current_model})")
                    loader["thread"] = threading.Thread(
                        target=_load, args=(pending, loader), daemon=True)
                    loader["thread"].start()

                # 2) If the load finished -> swap right then
                if loader["thread"] is not None and not loader["thread"].is_alive():
                    if loader["error"]:
                        self._status(f"Model load failed: {loader['error']}")
                    elif loader["stt"] is not None:
                        stt = loader["stt"]
                        current_model = loader["model"]
                        self._status(f"Transcribing · {who} · {current_model}")
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

            for speaker, vad in vads.items():
                tail = vad.finalize()
                if tail is not None:
                    lang, text = stt.transcribe(tail)
                    self.writer.emit(lang, text, speaker=speaker)

        except Exception as e:  # noqa: BLE001
            self.line_queue.put(("error", f"Error: {e}"))
        finally:
            if self.capture is not None:
                self.capture.stop()
                self.capture = None
            if self.writer is not None:
                self.writer.close()
            self.running = False
            self._status("Stopped")

    def stop(self):
        self.stop_event.set()

    def save(self):
        if self.writer is None:
            return None
        return self.writer.save_txt()


class App:
    def __init__(self, root):
        self.root = root
        self.ctrl = STTController()
        self.session_lines = []
        self.session_started = None
        self.minutes_queue = queue.Queue()
        self.minutes_busy = False
        self._speaker_colors = {}
        self._live_summary = []      # [[speaker, [text, ...]], ...] merges consecutive same-speaker turns

        root.title("Live STT · Minutes")
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

        self.save_btn = ctk.CTkButton(
            bar, text="💾  Save txt", command=self._on_save, width=104, height=30,
            corner_radius=8, font=self.ui_font,
            fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER)
        self.save_btn.pack(side="right", padx=(0, 14), pady=7)
        self.devices_btn = ctk.CTkButton(
            bar, text="Devices", command=lambda: list_input_devices(),
            width=80, height=30, corner_radius=8, font=self.ui_font,
            fg_color="transparent", hover_color=NEUTRAL,
            border_width=1, border_color=BORDER, text_color="#c9c9d4")
        self.devices_btn.pack(side="right", padx=(0, 8), pady=7)

    def _build_tabs(self):
        self.tabs = ctk.CTkTabview(
            self.root, fg_color=PANEL, corner_radius=12,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color=ACCENT_HOVER)
        self.tabs.pack(fill="both", expand=True, padx=16, pady=0)
        stt_tab = self.tabs.add("🎙  Live STT")
        live_tab = self.tabs.add("⚡  Live Summary")
        min_tab = self.tabs.add("📝  Minutes")
        self._build_stt_tab(stt_tab)
        self._build_live_tab(live_tab)
        self._build_minutes_tab(min_tab)

    def _build_stt_tab(self, tab):
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
        self._live_summary = []
        self.live_text.configure(state="normal")
        self.live_text.delete("1.0", "end")
        self.live_text.configure(state="disabled")
        self._set_minutes_text("")
        self.minutes_save_btn.configure(state="disabled")

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
        self._update_live_summary(speaker, text)

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
