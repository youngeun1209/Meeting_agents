"""Transcript translation — calls a local LLM (Ollama).

Uses the same local Ollama HTTP API path as minutes generation.
"""
import json
import urllib.error
import urllib.request

import config

TARGET_LANGUAGES = {
    "Korean": "Korean",
    "English": "English",
    "Chinese": "Chinese",
}

PROMPT_TEMPLATE = """The following is a raw real-time STT transcript of a meeting/lecture.
It mixes timestamps, speaker tags, language tags, and possible STT typos.

Translate it into {target_language}.

Format:
- Preserve timestamps.
- Preserve [Me] and [Others] speaker tags.
- Preserve technical terms, product names, people names, and acronyms when translating them would be unnatural.
- Keep the same line-by-line structure as much as possible.
- Fix obvious STT typos only when the intended meaning is clear from context.

Rules:
- Do not summarize.
- Do not add information that is not in the transcript.
- If a line is already in {target_language}, keep it natural and lightly corrected.

--- STT transcript ---
{transcript}
--- end of transcript ---

Translation:"""


LINE_PROMPT_TEMPLATE = """Translate the following single line of meeting/lecture speech into {target_language}.

Rules:
- Output only the translation. No quotes, no notes, no explanation.
- Keep technical terms, product names, people names, and acronyms untranslated when translating them would be unnatural.
- If the line is already in {target_language}, lightly correct it and return it.

Line: {text}
Translation:"""


def _stream_generate(prompt, temperature, num_ctx, on_token=None):
    """POST /api/generate (streaming). Returns the full text. Raises RuntimeError on failure."""
    payload = json.dumps({
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }).encode("utf-8")

    req = urllib.request.Request(
        config.OLLAMA_URL + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    parts = []
    try:
        with urllib.request.urlopen(req, timeout=config.OLLAMA_TIMEOUT) as resp:
            for raw in resp:
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                if "error" in obj:
                    raise RuntimeError(obj["error"])
                tok = obj.get("response", "")
                if tok:
                    parts.append(tok)
                    if on_token is not None:
                        on_token(tok)
                if obj.get("done"):
                    break
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama connection failed: {e}. Check that `ollama serve` is running "
            f"and that the model '{config.OLLAMA_MODEL}' has been pulled."
        )
    return "".join(parts).strip()


def translate_line(text, target_language="Korean", on_token=None):
    """
    Translate a single utterance (str) -> translated line (str), for real-time translation.
    Uses a short prompt + small context so each line returns fast. Raises RuntimeError on failure.
    """
    if not text.strip():
        return ""
    if target_language not in TARGET_LANGUAGES:
        raise RuntimeError(f"Unsupported target language: {target_language}")
    prompt = LINE_PROMPT_TEMPLATE.format(
        target_language=TARGET_LANGUAGES[target_language],
        text=text.strip(),
    )
    return _stream_generate(prompt, temperature=0.1, num_ctx=2048, on_token=on_token)


def is_available():
    """Check whether the Ollama server is up."""
    try:
        req = urllib.request.Request(config.OLLAMA_URL + "/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def translate_transcript(transcript, target_language="Korean", on_token=None):
    """
    STT transcript (str) -> translated transcript (str).
    on_token: streaming callback (partial text). If None, only the finished text is returned.
    Raises RuntimeError on failure.
    """
    if not transcript.strip():
        raise RuntimeError("Nothing to translate.")
    if target_language not in TARGET_LANGUAGES:
        raise RuntimeError(f"Unsupported target language: {target_language}")

    prompt = PROMPT_TEMPLATE.format(
        target_language=TARGET_LANGUAGES[target_language],
        transcript=transcript,
    )
    return _stream_generate(
        prompt, temperature=0.1, num_ctx=config.OLLAMA_NUM_CTX, on_token=on_token)
