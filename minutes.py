"""Minutes generation — calls a local LLM (Ollama).

Uses the Ollama HTTP API (localhost:11434). No extra pip dependency (urllib).
Requires: `ollama serve` running + the model (config.OLLAMA_MODEL) pulled.
"""
import json
import urllib.error
import urllib.request

import config

PROMPT_TEMPLATE = """The following is a raw real-time STT transcript of a meeting/lecture.
It mixes timestamps and language tags, and sentences may be cut off or contain typos.

Turn it into clean meeting minutes. Write the minutes in the SAME language as the
transcript (Korean transcript -> Korean minutes, English transcript -> English minutes;
if mixed, use the dominant language). Format:

## Meeting Minutes
- **Date**: {when}
- **Note**: auto-generated from STT (may contain errors)

### Summary
(3-6 bullets summarizing the whole thing)

### Discussion
(subheadings by topic + bullets. Group by topic rather than chronologically)

### Decisions
(what was agreed/decided. "None" if nothing)

### Action Items
- [ ] (with owner/deadline if available)

Rules:
- Do not invent anything not in the transcript.
- Fix STT typos naturally from context. Keep proper nouns / technical terms in their original form.
- Use the [Me]/[Others] speaker tags on each line to distinguish who said what,
  and who requested/decided what. State the responsible speaker for each action item.

--- STT transcript ---
{transcript}
--- end of transcript ---

Minutes:"""


def is_available():
    """Check whether the Ollama server is up."""
    try:
        req = urllib.request.Request(config.OLLAMA_URL + "/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def generate_minutes(transcript, when="", on_token=None):
    """
    STT transcript (str) -> minutes (str).
    on_token: streaming callback (partial text). If None, only the finished text is returned.
    Raises RuntimeError on failure.
    """
    if not transcript.strip():
        raise RuntimeError("Nothing to summarize.")

    prompt = PROMPT_TEMPLATE.format(when=when or "(not specified)", transcript=transcript)
    payload = json.dumps({
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.2, "num_ctx": config.OLLAMA_NUM_CTX},
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
