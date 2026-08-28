"""Render markdown into a Tk/customtkinter textbox as *formatting* instead of raw syntax.

The LLM tabs (Minutes, Audio File) get markdown back from the model, and a Tk textbox has no
markdown support -- so `## Heading`, `**bold**` and `- item` were being shown literally, syntax
characters and all. This turns them into real headings, bold runs and bullets.

Deliberately a small subset -- headings, bold, italic, inline code, bullets, numbered lists,
blockquotes and horizontal rules. That is what the minutes/transcript prompts actually produce.
Tables are left as monospace text: Tk has no column model, and faking one by padding spaces
breaks the moment the font isn't monospace or the window is narrow.

The widget is display-only, so the ORIGINAL markdown must be kept by the caller for saving to
disk -- what you read back out of the textbox is the rendered text, with the syntax gone.
"""
import re

# (tag, font spec, extra options). Sizes step down so hierarchy is visible without being shouty.
_HEADING_TAGS = {
    1: ("md_h1", ("Helvetica", 19, "bold")),
    2: ("md_h2", ("Helvetica", 16, "bold")),
    3: ("md_h3", ("Helvetica", 14, "bold")),
    4: ("md_h4", ("Helvetica", 13, "bold")),
}

_BULLET = "•"
_RULE = "─" * 40

# Inline spans, ordered: code first so ** inside `code` is left alone.
_INLINE = [
    ("md_code",   re.compile(r"`([^`\n]+)`")),
    ("md_bold",   re.compile(r"\*\*([^*\n]+)\*\*")),
    ("md_italic", re.compile(r"(?<![*\w])[*_]([^*_\n]+)[*_](?![*\w])")),
]


def _tk(textbox):
    """The underlying tk.Text.

    CTkTextbox.tag_config refuses a `font` option outright ("incompatible with scaling"), so tags
    have to be configured on the real Tk widget. Scaling is then ours to handle -- see _scale.
    """
    return getattr(textbox, "_textbox", textbox)


def _scale(textbox):
    """CTk's widget scaling factor, so manually-set tag fonts still track display scaling."""
    try:
        return float(textbox._get_widget_scaling())
    except Exception:  # noqa: BLE001 - a plain tk.Text has no scaling concept
        return 1.0


def configure_tags(textbox, body_family="Menlo", body_size=13,
                   heading_color="#ffffff", muted="#8b8b99", accent="#a5b4fc"):
    """Set up the tags this module renders with. Call once per render."""
    t = _tk(textbox)
    k = _scale(textbox)
    sz = lambda n: max(1, int(round(n * k)))   # noqa: E731
    for _lvl, (tag, (family, size, weight)) in _HEADING_TAGS.items():
        t.tag_config(tag, font=(family, sz(size), weight),
                     foreground=heading_color, spacing1=sz(10), spacing3=sz(4))
    t.tag_config("md_bold", font=(body_family, sz(body_size), "bold"))
    t.tag_config("md_italic", font=(body_family, sz(body_size), "italic"))
    t.tag_config("md_code", font=(body_family, sz(body_size)), foreground=accent)
    t.tag_config("md_bullet", foreground=accent)
    t.tag_config("md_quote", foreground=muted, lmargin1=sz(18), lmargin2=sz(18))
    t.tag_config("md_rule", foreground=muted)
    t.tag_config("md_body", font=(body_family, sz(body_size)))


def _insert_inline(textbox, text, base_tags):
    """Insert one line, converting inline spans to tags and dropping their syntax characters."""
    # Find the earliest match among all inline patterns, emit the text before it plain, then the
    # span itself tagged, then continue after it. Repeating from the cut point keeps nesting simple.
    while text:
        best = None
        for tag, pattern in _INLINE:
            m = pattern.search(text)
            if m and (best is None or m.start() < best[1].start()):
                best = (tag, m)
        if best is None:
            textbox.insert("end", text, base_tags)
            return
        tag, m = best
        if m.start():
            textbox.insert("end", text[:m.start()], base_tags)
        textbox.insert("end", m.group(1), base_tags + (tag,))
        text = text[m.end():]


def render(textbox, markdown, body_family="Menlo", body_size=13):
    """Replace the textbox contents with `markdown`, rendered. Leaves the widget disabled."""
    configure_tags(textbox, body_family=body_family, body_size=body_size)
    textbox.configure(state="normal")
    textbox.delete("1.0", "end")

    in_fence = False
    for line in (markdown or "").split("\n"):
        stripped = line.strip()

        # Fenced code block: emit verbatim, no inline parsing, and don't render the fence itself.
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            textbox.insert("end", line + "\n", ("md_code",))
            continue

        if not stripped:
            textbox.insert("end", "\n")
            continue

        # Horizontal rule
        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            textbox.insert("end", _RULE + "\n", ("md_rule",))
            continue

        # Heading
        m = re.match(r"(#{1,6})\s+(.*)", stripped)
        if m:
            level = min(len(m.group(1)), 4)
            tag, _font = _HEADING_TAGS[level]
            _insert_inline(textbox, m.group(2), (tag,))
            textbox.insert("end", "\n", (tag,))
            continue

        # Blockquote
        if stripped.startswith(">"):
            _insert_inline(textbox, stripped.lstrip("> ").rstrip(), ("md_quote",))
            textbox.insert("end", "\n", ("md_quote",))
            continue

        # Bullet / numbered list -- keep the original indent so nesting still reads
        indent = len(line) - len(line.lstrip())
        m = re.match(r"[-*+]\s+(.*)", stripped)
        if m:
            textbox.insert("end", " " * indent + f"  {_BULLET} ", ("md_bullet",))
            _insert_inline(textbox, m.group(1), ("md_body",))
            textbox.insert("end", "\n")
            continue
        m = re.match(r"(\d+)[.)]\s+(.*)", stripped)
        if m:
            textbox.insert("end", " " * indent + f"  {m.group(1)}. ", ("md_bullet",))
            _insert_inline(textbox, m.group(2), ("md_body",))
            textbox.insert("end", "\n")
            continue

        _insert_inline(textbox, line, ("md_body",))
        textbox.insert("end", "\n")

    textbox.configure(state="disabled")
