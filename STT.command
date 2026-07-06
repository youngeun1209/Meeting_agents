#!/bin/bash
# Double-click launcher for the Meeting STT app.
# Finds a Python 3, then hands off to the setup wizard (setup_app.py), which
# installs anything missing and launches the real GUI. No terminal knowledge
# needed — just double-click this file in Finder.
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

# Pick a python3 that actually has Tkinter (Homebrew python often lacks it).
PY=""
for cand in \
    /usr/local/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/*/bin/python3 \
    /usr/bin/python3 \
    python3 \
    /opt/homebrew/bin/python3; do
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c 'import tkinter' >/dev/null 2>&1; then PY="$cand"; break; fi
done

if [ -z "$PY" ]; then
    osascript -e 'display alert "No Python with Tk found" message "Install Python 3 from https://www.python.org/downloads/ (it includes Tkinter). If you use Homebrew Python: brew install python-tk"'
    exit 1
fi

exec "$PY" setup_app.py
