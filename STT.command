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

# Close this Terminal window once the app is up.
#
# The .app cannot launch Python itself: an ad-hoc signed bundle is refused READ access to a
# project folder under ~/Documents (measured: READ_OK=no), so it hands off to Terminal, whose
# TCC grant does work -- see "Meeting STT.app/Contents/MacOS/launch". That leaves a Terminal
# window sitting there for the rest of the session, which is just noise.
#
# The window is matched by this shell's own tty, so only the window we are running in closes,
# never one the user opened themselves. It is scheduled in the background and fires after the
# shell has exited: closing a window whose process is still running makes Terminal put up a
# "close anyway?" prompt, and waiting for "[Process completed]" avoids it. The GUI survives
# because setup_app.py starts it in its own session (start_new_session=True).
MY_TTY="$(tty)"
(
    sleep 2
    /usr/bin/osascript -e "
        tell application \"Terminal\"
            repeat with w in windows
                repeat with t in tabs of w
                    if tty of t is \"$MY_TTY\" then close w saving no
                end repeat
            end repeat
        end tell" >/dev/null 2>&1
) &

"$PY" setup_app.py
