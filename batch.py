"""Mode 2: regenerate a clean, speaker-attributed transcript FROM AN AUDIO FILE (post-session).

Why this exists (and why it re-transcribes instead of reusing the realtime transcript)
----------------------------------------------------------------------------------------
The realtime transcript's line timestamps (writer.py) are stamped with `datetime.now()`
at the moment transcription FINISHED, which lags the actual speech by 5-10s -- fine for a
live readout, but useless for aligning against diarization segments (which are anchored to
real audio-relative time). So mode 2 does not touch the realtime text at all: it decodes
the given audio file, re-transcribes it from scratch with mlx_whisper (keeping each
segment's real audio-relative start/end), runs offline speaker diarization on the same
audio (diarize.py), and merges the two by time overlap. This is a one-shot batch job, not
a realtime path -- latency doesn't matter, accuracy does.

Pipeline
--------
1. Decode the input file to 16kHz mono float32 (any format ffmpeg/afconvert can read).
2. Re-transcribe the whole file with mlx_whisper, keeping each segment's start/end.
3. Apply transcriber.py's existing quality gates (hallucination / repetition / RMS) to each
   segment, exactly like the realtime path -- reused, not reimplemented.
4. Diarize the same audio (diarize.py) and assign each transcript segment the speaker whose
   diarization segments overlap it MOST by actual overlapping duration (not midpoint --
   midpoint misassigns long segments that cross a speaker boundary).
5. Merge consecutive same-speaker segments into turns.
6. Optionally run a final LLM typo-fix pass, reusing refine.py's Ollama client + chunking
   (long transcripts need the same windowing refine.py already solves). The prompt is
   typo-fix ONLY -- it must not re-assign speakers or drop/summarize content, since
   diarization already decided speaker identity from the audio and is authoritative here.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

import config
import diarize
from transcriber import is_hallucination, is_repetitive, rms

# --- typo-fix prompt (speaker assignment is NOT this prompt's job -- diarization already
# decided that from the audio, and this pass must not override it) -------------------------
_TYPO_FIX_INSTRUCTIONS = """The following is a speaker-attributed STT transcript. Each block is
already correctly split by speaker and by turn -- diarization decided that from the audio, and
it is authoritative. Your ONLY job is to fix obvious STT mis-transcriptions (typos / mis-heard
words) using context. Rules:
- Do NOT change, merge, split, or reassign any speaker label or turn boundary.
- Do NOT add, remove, reorder, or summarize any block.
- Preserve every timestamp range exactly as given.
- Only fix obvious mis-transcriptions. Keep proper nouns / technical terms recognizable (correct
  a garbled STT rendering of a real term once you are confident, not left garbled).
- Write the output in the SAME language as the transcript.
- Output format: markdown, same block structure as the input, one block per turn, each formatted
  exactly as:
  **[MM:SS-MM:SS] <speaker label>**
  <corrected text of that turn>
  (blank line between blocks)."""

_TYPO_FIX_TEMPLATE = """{instructions}
{context_section}
--- transcript segment to fix (do not repeat any context above in your output) ---
{raw_chunk}
--- end of segment ---

Fixed transcript (markdown turn blocks only, no preamble):"""

_TYPO_FIX_CONTEXT_TEMPLATE = """
--- context: already-fixed trailing turns from just before this segment (for continuity only --
do NOT reprint these in your output) ---
{tail}
"""

# Reuse refine.py's exact chunk-sizing knobs/heuristic so long transcripts window the same way.
import refine as _refine  # noqa: E402  (after the prompt constants, which don't depend on it)

# WAV format mlx_whisper / diarize.py expect: 16kHz, mono, 16-bit PCM.
_TARGET_RATE = config.SAMPLE_RATE  # 16000


def _is_already_target_wav(path):
    """True if `path` is a .wav already at 16kHz/mono/16-bit -- skip conversion if so."""
    if not path.lower().endswith(".wav"):
        return False
    try:
        with wave.open(path, "rb") as r:
            return (r.getframerate() == _TARGET_RATE
                    and r.getnchannels() == 1
                    and r.getsampwidth() == 2)
    except (wave.Error, EOFError, OSError):
        return False


def _decode_to_float32(path):
    """Any audio file -> (16kHz mono float32 array, path to a 16k/mono/16-bit WAV, temp_or_None).

    Converts via the macOS built-in `afconvert` (no new pip dependency) into a temp WAV unless the
    input is already exactly 16k/mono/16-bit. The WAV path is returned as well because diarization
    runs in a child process and takes a file rather than a 230MB pickled array; the third element
    is the temp file the CALLER must delete (None when the input was usable as-is).
    """
    tmp_path = None
    if True:
        if _is_already_target_wav(path):
            read_path = path
        else:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            # -d LEI16@16000 = little-endian 16-bit PCM at 16kHz; -c 1 = mono.
            result = subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", path, tmp_path],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"[batch] afconvert failed on {path!r}: {result.stderr.strip()}")
            read_path = tmp_path

        with wave.open(read_path, "rb") as r:
            if r.getsampwidth() != 2:
                raise RuntimeError(
                    f"[batch] unexpected sample width after conversion: {r.getsampwidth()}")
            raw = r.readframes(r.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, read_path, tmp_path


def _transcribe_segments(audio_f32_16k):
    """Re-transcribe the whole file with mlx_whisper. Returns (language, list of segment dicts
    with audio-relative start/end/text), filtered through transcriber.py's quality gates."""
    import mlx_whisper  # lazy import, same as transcriber.py
    import mlx.core as mx

    repo = config.MLX_MODEL_MAP.get(config.MODEL_SIZE)
    if repo is None:
        raise RuntimeError(
            f"No mlx model mapping for '{config.MODEL_SIZE}'. "
            f"Add it to config.MLX_MODEL_MAP or change MODEL_SIZE.")

    # Cap the Metal buffer cache for the whole run. The realtime path trims the pool between
    # chunks (transcriber.py), but here the entire file is one mlx_whisper.transcribe() call --
    # it loops over 30-second windows internally, with no point for us to trim between them. A
    # two-hour file is ~240 of those windows, so an uncapped pool grows all run and pushes a
    # long job into swap. set_cache_limit bounds it from the outside instead; it is a cache, so
    # capping it costs a little re-allocation, never correctness.
    mx.set_cache_limit(config.MLX_CACHE_LIMIT_BYTES)

    result = mlx_whisper.transcribe(
        audio_f32_16k,
        path_or_hf_repo=repo,
        language=config.LANGUAGE,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        verbose=None,
    )
    language = result.get("language") or (config.LANGUAGE or "")

    segments = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start, end = float(seg["start"]), float(seg["end"])
        # Quality gates: same ones the realtime path applies (transcriber.py), reused not copied.
        chunk_audio = audio_f32_16k[int(start * _TARGET_RATE):int(end * _TARGET_RATE)]
        if rms(chunk_audio) < config.MIN_RMS:
            continue
        if is_hallucination(text) or is_repetitive(text):
            continue
        segments.append({"start": start, "end": end, "text": text})

    return language, segments


def _assign_speakers(segments, diar_segments):
    """Assign each transcript segment the speaker whose diarization segments overlap it most
    by actual overlap DURATION (not midpoint -- midpoint misassigns long segments that cross
    a speaker boundary). No overlap -> speaker None."""
    for seg in segments:
        s, e = seg["start"], seg["end"]
        overlap_by_speaker = {}
        for ds, de, spk in diar_segments:
            overlap = min(e, de) - max(s, ds)
            if overlap > 0:
                overlap_by_speaker[spk] = overlap_by_speaker.get(spk, 0.0) + overlap
        seg["speaker"] = (max(overlap_by_speaker, key=overlap_by_speaker.get)
                           if overlap_by_speaker else None)
    return segments


def _merge_turns(segments):
    """Merge consecutive same-speaker segments into turns: {start, end, speaker, text}."""
    turns = []
    for seg in segments:
        if turns and turns[-1]["speaker"] == seg["speaker"]:
            turns[-1]["end"] = seg["end"]
            turns[-1]["text"] = (turns[-1]["text"] + " " + seg["text"]).strip()
        else:
            turns.append({"start": seg["start"], "end": seg["end"],
                          "speaker": seg["speaker"], "text": seg["text"]})
    return turns


def _fmt_mmss(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"


def _speaker_label(speaker):
    if speaker is None:
        return "화자 ?"
    return f"화자 {speaker + 1}" if isinstance(speaker, int) else f"화자 {speaker}"


def _render_markdown(turns):
    blocks = []
    for t in turns:
        # Bold header, matching both the typo-fix prompt's required format and
        # refine._split_output_blocks -- so output is identical whether or not the typo-fix
        # pass ran, and stays re-parseable.
        header = f"**[{_fmt_mmss(t['start'])}-{_fmt_mmss(t['end'])}] {_speaker_label(t['speaker'])}**"
        blocks.append(f"{header}\n{t['text']}")
    return "\n\n".join(blocks)


def _fix_typos(turns, on_status=None):
    """Final LLM pass: fix STT typos only, reusing refine.py's Ollama client + chunking so long
    transcripts window the same way. Returns corrected turns (parsed back out of the markdown),
    or the original turns unchanged if Ollama is unavailable (reported via on_status)."""
    if not _refine.is_available():
        if on_status:
            on_status("Ollama unavailable -- skipping typo-fix pass (transcript is still valid).")
        return turns

    # Build pseudo-utterances in refine.py's chunking shape so _build_chunks can reuse its
    # token-budget logic unmodified.
    pseudo = [{"ts": None, "speaker": t["speaker"], "lang": None,
               "text": t["text"],
               "raw": f"**[{_fmt_mmss(t['start'])}-{_fmt_mmss(t['end'])}] "
                      f"{_speaker_label(t['speaker'])}**\n{t['text']}"}
              for t in turns]
    chunks = _refine._build_chunks(pseudo)

    fixed_blocks = []
    prev_tail_raw = []
    for chunk in chunks:
        raw_chunk = "\n\n".join(u["raw"] for u in chunk)
        context_section = (
            _TYPO_FIX_CONTEXT_TEMPLATE.format(tail="\n\n".join(u["raw"] for u in prev_tail_raw))
            if prev_tail_raw else ""
        )
        prompt = _TYPO_FIX_TEMPLATE.format(
            instructions=_TYPO_FIX_INSTRUCTIONS,
            context_section=context_section,
            raw_chunk=raw_chunk,
        )
        try:
            output = _refine._stream_generate(
                prompt, temperature=0.1, num_ctx=config.OLLAMA_NUM_CTX)
        except RuntimeError as e:
            if on_status:
                on_status(f"Typo-fix pass failed ({e}) -- keeping untouched transcript.")
            return turns
        if output:
            fixed_blocks.append(output)
        prev_tail_raw = chunk[-_refine.CONTEXT_TURNS:]

    fixed_text = "\n\n".join(fixed_blocks).strip()

    # Parse the fixed markdown back into turns, preserving original start/end/speaker order
    # (the prompt is forbidden from reordering/adding/removing blocks, so this is a 1:1 zip;
    # fall back to the untouched turns if the count doesn't line up, rather than guess).
    fixed_bodies = [b.split("\n", 1)[1].strip() if "\n" in b else ""
                    for b in _refine._split_output_blocks(fixed_text)]
    if len(fixed_bodies) != len(turns):
        if on_status:
            on_status("Typo-fix pass returned a mismatched block count -- keeping untouched transcript.")
        return turns

    return [{**t, "text": body} for t, body in zip(turns, fixed_bodies)]


def transcribe_file(path, num_speakers=None, on_status=None, on_segment=None, fix_typos=True):
    """
    Regenerate a clean, speaker-attributed transcript from an audio file (post-session).

    Returns a dict: {"turns": [{"start", "end", "speaker", "text"}, ...],
                      "language": str, "markdown": str}
    """
    def status(msg):
        if on_status:
            on_status(msg)

    status(f"Decoding {os.path.basename(path)} …")
    audio, wav_path, tmp_path = _decode_to_float32(path)
    try:
        duration = len(audio) / _TARGET_RATE
        mins = duration / 60.0

        # Both heavy stages run roughly at real time on this hardware, so a long recording takes
        # about as long as it lasts. Say so up front instead of showing a frozen-looking label.
        status(f"Transcribing {mins:.1f} min of audio … (roughly {max(1, int(mins * 0.25))} min)")
        language, segments = _transcribe_segments(audio)
        if on_segment:
            for seg in segments:
                on_segment(seg)

        # Out-of-process: sherpa-onnx holds the GIL for the whole run and would otherwise freeze
        # the UI outright (measured: 139s solid for a 3-minute file).
        def diar_progress(done, total):
            if total:
                status(f"Separating speakers … {int(done * 100 / total)}%")
        status("Separating speakers … 0%")
        diar_segments = diarize.diarize_file(
            wav_path, num_speakers=num_speakers, on_progress=diar_progress)

        status("Merging transcript with speakers …")
        segments = _assign_speakers(segments, diar_segments)
        turns = _merge_turns(segments)

        if fix_typos:
            status("Fixing typos (local LLM) …")
            turns = _fix_typos(turns, on_status=on_status)

        markdown = _render_markdown(turns)
        status("Done.")
        return {"turns": turns, "language": language, "markdown": markdown}
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mode 2: regenerate a clean, speaker-attributed transcript from an audio file.")
    parser.add_argument("audiofile", help="path to an audio file (.wav/.m4a/.mp3/.aiff/...)")
    parser.add_argument("-o", "--output", help="write the markdown transcript to this path")
    parser.add_argument("--speakers", type=int, default=None,
                         help="optional hint: exact number of distinct speakers")
    parser.add_argument("--no-typo-fix", action="store_true",
                         help="skip the final Ollama typo-fix pass")
    args = parser.parse_args()

    if not diarize.is_available():
        print(diarize.MODEL_DOWNLOAD_INFO, file=sys.stderr)
        sys.exit(1)

    result = transcribe_file(
        args.audiofile,
        num_speakers=args.speakers,
        on_status=lambda msg: print(f"[batch] {msg}", file=sys.stderr),
        fix_typos=not args.no_typo_fix,
    )

    print(result["markdown"])

    if args.output:
        # 0600 -- a rebuilt transcript is the full meeting content, same as writer.Writer's.
        os.close(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600))
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result["markdown"] + "\n")
        print(f"[batch] saved: {args.output}", file=sys.stderr)
