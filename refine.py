"""Post-session transcript refinement — calls a local LLM (Ollama).

Problem this solves
--------------------
`config.SOURCES` normally tags mic audio as "Me" and system audio as "Others".
In practice the system-audio source frequently captures nothing (permission
not granted, the other party on a call that doesn't route through system
output, etc.), so every line in the raw transcript ends up tagged "[Me]" even
though it was really a two-person conversation. On top of that, real-time STT
on Korean speech mis-hears technical terms constantly (기터부 -> GitHub,
채찔 PT -> ChatGPT, 이이지 -> EEG, 폴로우 -> 플로우, 500MHz -> 500MB, ...).

`refine_transcript()` is a post-session pass (NOT real-time) that re-reads a
saved transcript, and asks the LLM to (a) fix obvious STT typos and (b)
re-derive speaker turns from conversational cues when the recorded tags are
useless, while never inventing, summarizing, or dropping content. It reuses
the exact Ollama HTTP client pattern from `minutes.py` / `translator.py`
(same `/api/generate` streaming call, same `is_available()` shape, same
`urllib`-only dependency, same RuntimeError-on-failure contract).

Chunking strategy (the hard part)
----------------------------------
`config.OLLAMA_NUM_CTX` is 8192 tokens, shared between prompt + generated
output. A real meeting transcript is easily 15-30k+ tokens, so the whole
thing cannot go in one call. The transcript is parsed into individual
utterances (one per source line) and packed into non-overlapping windows on
UTTERANCE boundaries only — a single STT line is never split mid-sentence.
Each window's raw-text token budget is sized so that
    prompt overhead + context carry-over + window + generation headroom
stays comfortably under `OLLAMA_NUM_CTX` (see `CHUNK_TOKEN_BUDGET`).

The problem with simply doing this independently per window is that speaker
labels invented in one window (e.g. "화자 A" for whoever is speaking first)
have no relationship to the labels invented in the next window — the model
has no memory between calls. To keep labels consistent ACROSS windows, each
chunk after the first carries forward, as pure context (explicitly marked
"do not repeat this in your output"):
  1. the last `CONTEXT_TURNS` RAW utterance lines of the previous window
     (so the model has the literal original text near the boundary), and
  2. the last `CONTEXT_OUTPUT_BLOCKS` markdown blocks of the PREVIOUS
     window's *resolved* output (so the model sees exactly which speaker
     label / turn boundary was already assigned to that trailing text).
Because the next window's new content picks up immediately after that
overlap, the model is told "continue seamlessly using the same 화자 A /
화자 B assignment you see in the context" instead of re-deciding from
scratch, which keeps the speaker identity stable end-to-end. Chunks are
processed strictly sequentially (each depends on the previous chunk's
actual output), and their outputs are concatenated in order for the final
result.

Token estimation is a cheap heuristic (`_estimate_tokens`) rather than a
real tokenizer: CJK characters are counted heavier (~1.3 tokens/char) than
other characters (~1 token/3.5 chars), which is enough to size chunks safely
without adding a tokenizer dependency.

Speaker-tag reliability
------------------------
Before chunking, `refine_transcript` checks whether the parsed utterances
carry more than one distinct speaker tag. If everything is tagged
identically (the "system audio captured nothing" failure mode above), the
prompt switches to "infer speakers from conversational cues" mode and labels
turns 화자 A / 화자 B (only introducing a third label if the conversation
clearly has one). If the original tags already look reliable (e.g. both
[Me] and [Others] appear), the prompt instead asks the model to preserve and
lightly clean up the existing tags.
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request

import config

# --- line parsing -----------------------------------------------------------
# Matches writer.py's emitted format `[HH:MM:SS] [speaker] (lang) text`, and
# tolerates the no-speaker and PRINT_LANG=False variants:
#   [15:00:57] [Me] (ko) text...
#   [15:00:57] (ko) text...            (no speaker tag)
#   [15:00:57] [Me] text...            (PRINT_LANG=False)
#   [15:00:57] text...                 (no speaker, PRINT_LANG=False)
LINE_RE = re.compile(
    r"^\[(?P<ts>\d{2}:\d{2}:\d{2})\]\s*"
    r"(?:\[(?P<speaker>[^\]]+)\]\s*)?"
    r"(?:\((?P<lang>[a-zA-Z]{2,5})\)\s*)?"
    r"(?P<text>.*)$"
)

# --- chunking knobs -----------------------------------------------------------
# Reserve headroom out of OLLAMA_NUM_CTX for: the fixed instruction text, the
# carried-forward context block, and the model's generated output (a refined
# turn is roughly as long as the source text, plus light markdown).
_RESERVED_PROMPT_TOKENS = 700
_RESERVED_CONTEXT_TOKENS = 400
_RESERVED_OUTPUT_FRACTION = 0.4
CHUNK_TOKEN_BUDGET = max(
    800,
    int(config.OLLAMA_NUM_CTX
        - _RESERVED_PROMPT_TOKENS
        - _RESERVED_CONTEXT_TOKENS
        - config.OLLAMA_NUM_CTX * _RESERVED_OUTPUT_FRACTION),
)
CONTEXT_TURNS = 6          # raw trailing lines carried forward as context
CONTEXT_OUTPUT_BLOCKS = 3  # resolved markdown blocks carried forward as context

INSTRUCTIONS = """The following is a raw real-time STT transcript of a meeting/conversation.
It may contain STT typos and mis-heard technical terms, and speaker tags may be missing,
inconsistent, or (in the worst case) all identical even though multiple people were speaking.

Rewrite it into a clean, speaker-attributed transcript. Rules:
- Preserve every timestamp exactly as given.
- NEVER invent, summarize, or drop any content. Every idea in the source must survive in the output.
- Only fix obvious mis-transcriptions (STT typos / mis-heard words) using context. Keep proper
  nouns / technical terms recognizable (e.g. a garbled STT rendering of "GitHub" or "ChatGPT"
  should be corrected to the real term once you are confident, not left garbled).
- You may split or merge turns where a single speaker's utterance was cut into multiple STT lines,
  or a line actually contains two speakers' turns.
- CRITICAL: every sentence you output must be a lightly-corrected copy of an actual sentence from
  the source segment, attributed to whichever speaker actually said it. Do NOT write new sentences
  that describe, paraphrase, or summarize "what 화자 A said" in the third person — that is exactly
  the kind of invention/summarizing that is forbidden. If you cannot tell which of two speakers a
  line belongs to, keep it under your best-guess speaker rather than rewriting it as commentary.
- Write the output in the SAME language as the transcript.
- Output format: markdown, one block per speaker turn, each block formatted exactly as:
  **[HH:MM:SS-HH:MM:SS] <speaker label>**
  <cleaned text of that turn>
  (blank line between blocks). The timestamp range is the first and last original timestamp
  covered by that turn.
- If you are not confident about a speaker change, still make a call but append " (⚠ 화자 불확실)"
  to that block's speaker label instead of silently guessing with full confidence."""

RELIABLE_SPEAKER_HINT = """Speaker tags in the source are reliable (they come from separate audio
sources per speaker) — preserve them as the speaker label for each turn, only fixing obvious
tagging glitches, and lightly clean up the label text itself if needed."""

UNRELIABLE_SPEAKER_HINT = """All source speaker tags are identical or missing, so they cannot be
trusted as-is (a known capture failure mode: only one audio source actually recorded). Infer turn
boundaries and speaker identity purely from conversational cues — question/answer pairs, backchannels
like "네" / "맞아요" / "그쵸", topic ownership, self-references. Label the two speakers 화자 A and
화자 B (use 화자 C only if a third speaker is clearly evident from the content).
Write the labels EXACTLY as "화자 A" / "화자 B" in Korean. Never write "Speaker A", "A", or any
other variant -- the transcript is split into several LLM calls and the labels must match across
all of them so the same person keeps the same name from start to finish.
Keep turns at conversational granularity: a question and its answer are two DIFFERENT turns and
must never share one block. Do not merge many minutes of back-and-forth into a single block."""

CHUNK_PROMPT_TEMPLATE = """{instructions}

{speaker_hint}
{num_speakers_hint}
{context_section}
--- new transcript segment to refine (do not repeat any of the context above in your output) ---
{raw_chunk}
--- end of new segment ---

Refined transcript (markdown turn blocks only, no preamble):"""

CONTEXT_SECTION_TEMPLATE = """--- context: already-refined trailing turns from just before this segment \
(for continuity only — do NOT reprint these in your output; continue the SAME speaker labeling) ---
{resolved_tail}

--- original raw lines matching that context (for reference only) ---
{raw_tail}
"""


def _estimate_tokens(text):
    """Cheap token-count heuristic: CJK chars are ~1.3 tokens each, everything
    else averages ~1 token per 3.5 chars. Good enough for chunk sizing."""
    cjk = sum(
        1 for ch in text
        if "぀" <= ch <= "ヿ"   # kana
        or "㄰" <= ch <= "㆏"   # hangul jamo
        or "가" <= ch <= "힣"   # hangul syllables
        or "一" <= ch <= "鿿"   # CJK unified ideographs
    )
    other = len(text) - cjk
    return int(cjk * 1.3 + other / 3.5) + 1


def parse_transcript(transcript_text):
    """
    Raw transcript text -> list of utterance dicts:
      {"ts": "HH:MM:SS", "speaker": str|None, "lang": str|None, "text": str, "raw": original line}
    Skips blank lines and `# ...` header/comment lines. Lines that don't match the
    expected format are kept verbatim (ts=None) so nothing from the source is silently lost.
    """
    utterances = []
    for line in transcript_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = LINE_RE.match(stripped)
        if not m:
            utterances.append({"ts": None, "speaker": None, "lang": None,
                                "text": stripped, "raw": stripped})
            continue
        utterances.append({
            "ts": m.group("ts"),
            "speaker": m.group("speaker"),
            "lang": m.group("lang"),
            "text": m.group("text").strip(),
            "raw": stripped,
        })
    return utterances


def _speaker_tags_reliable(utterances):
    tags = {u["speaker"] for u in utterances if u["speaker"]}
    return len(tags) > 1


def _build_chunks(utterances):
    """Pack utterances into non-overlapping windows on utterance boundaries,
    each window's raw text staying under CHUNK_TOKEN_BUDGET tokens."""
    chunks = []
    current, current_tokens = [], 0
    for u in utterances:
        line_tokens = _estimate_tokens(u["raw"])
        if current and current_tokens + line_tokens > CHUNK_TOKEN_BUDGET:
            chunks.append(current)
            current, current_tokens = [], 0
        current.append(u)
        current_tokens += line_tokens
    if current:
        chunks.append(current)
    return chunks


def _split_output_blocks(output_text):
    """Split a chunk's markdown output into its per-turn blocks (blank-line separated)."""
    return [b for b in re.split(r"\n\s*\n", output_text.strip()) if b.strip()]


def _stream_generate(prompt, temperature, num_ctx, on_token=None):
    """POST /api/generate (streaming). Returns the full text. Raises RuntimeError on failure.
    Same client pattern as minutes.generate_minutes / translator._stream_generate."""
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


def is_available():
    """Check whether the Ollama server is up."""
    try:
        req = urllib.request.Request(config.OLLAMA_URL + "/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


# Models drift on label wording across separate calls (observed on qwen2.5:7b: one chunk came
# back as "Speaker A" after several chunks of "화자 A"), which silently breaks the whole point of
# carrying context forward -- the reader can no longer tell it is the same person. The prompt asks
# for the canonical form, but asking is not a guarantee across N independent generations, so
# normalize deterministically as well. Applied to each chunk BEFORE it is carried forward as
# context, so the correction also reinforces the next chunk instead of only fixing the final text.
_LABEL_VARIANTS = re.compile(
    r"(?<=\*\*)(?P<pre>\[[^\]]*\]\s*)(?:speaker|spkr|화자)\s*(?P<who>[A-Ca-c1-3])",
    re.IGNORECASE)
_LETTER = {"1": "A", "2": "B", "3": "C"}


def _normalize_speaker_labels(text):
    """Force every block's speaker label to the canonical Korean `화자 A/B/C` form."""
    def sub(m):
        who = m.group("who").upper()
        return f"{m.group('pre')}화자 {_LETTER.get(who, who)}"
    return _LABEL_VARIANTS.sub(sub, text)


def refine_transcript(transcript_text, on_token=None, num_speakers=None):
    """
    Raw STT transcript (str) -> refined, speaker-attributed markdown (str).
    on_token: streaming callback (partial text), fired for every chunk in order.
              If None, only the finished text is returned.
    num_speakers: optional int hint ("there are exactly N speakers").
    Raises RuntimeError on failure (empty transcript, Ollama unreachable, etc.).
    """
    utterances = parse_transcript(transcript_text)
    if not utterances:
        raise RuntimeError("Nothing to refine.")

    reliable = _speaker_tags_reliable(utterances)
    speaker_hint = RELIABLE_SPEAKER_HINT if reliable else UNRELIABLE_SPEAKER_HINT
    num_speakers_hint = (
        f"There are exactly {num_speakers} distinct speakers in this conversation."
        if num_speakers else ""
    )

    chunks = _build_chunks(utterances)
    results = []
    prev_raw_tail = []
    prev_resolved_tail = []

    for chunk in chunks:
        raw_chunk = "\n".join(u["raw"] for u in chunk)

        if prev_raw_tail:
            context_section = CONTEXT_SECTION_TEMPLATE.format(
                resolved_tail="\n\n".join(prev_resolved_tail),
                raw_tail="\n".join(u["raw"] for u in prev_raw_tail),
            )
        else:
            context_section = ""

        prompt = CHUNK_PROMPT_TEMPLATE.format(
            instructions=INSTRUCTIONS,
            speaker_hint=speaker_hint,
            num_speakers_hint=num_speakers_hint,
            context_section=context_section,
            raw_chunk=raw_chunk,
        )

        output = _stream_generate(
            prompt, temperature=0.2, num_ctx=config.OLLAMA_NUM_CTX, on_token=on_token)
        if not output:
            raise RuntimeError("Ollama returned an empty response for a transcript chunk.")
        output = _normalize_speaker_labels(output)
        results.append(output)

        prev_raw_tail = chunk[-CONTEXT_TURNS:]
        blocks = _split_output_blocks(output)
        prev_resolved_tail = blocks[-CONTEXT_OUTPUT_BLOCKS:]

    return "\n\n".join(results).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refine a raw STT transcript: fix typos and re-group by speaker.")
    parser.add_argument("transcript", help="path to a transcripts/transcript_*.txt file")
    parser.add_argument("-o", "--output", help="write the refined markdown to this path")
    parser.add_argument("--speakers", type=int, default=None,
                         help="optional hint: exact number of distinct speakers")
    args = parser.parse_args()

    if not is_available():
        print(f"error: Ollama is not reachable at {config.OLLAMA_URL}. "
              f"Run `ollama serve` and `ollama pull {config.OLLAMA_MODEL}` first.",
              file=sys.stderr)
        sys.exit(1)

    with open(args.transcript, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        result = refine_transcript(
            text,
            on_token=lambda tok: print(tok, end="", flush=True),
            num_speakers=args.speakers,
        )
    except RuntimeError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        sys.exit(1)

    print()  # trailing newline after the streamed output

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result + "\n")
        print(f"[refine] saved: {args.output}", file=sys.stderr)
