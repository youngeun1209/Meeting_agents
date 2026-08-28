"""Settings the user changes at runtime and expects to still be there next launch.

Right now that is one thing: where output is saved. The default is a `transcripts` folder next
to the app; pick a different folder once -- from the folder button or from any Save dialog --
and everything afterwards defaults to that folder, this launch and every launch after.

Stored as settings.json next to the app, matching where the app already keeps the rest of its
local state (.setup_done, gui_error.log). config.py holds the *defaults*; this holds the user's
deviations from them. Nothing here imports config -- config imports this.
"""
import json
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(APP_DIR, "settings.json")


def load():
    """The saved settings as a dict. Missing, unreadable or corrupt file -> {} (defaults win)."""
    try:
        with open(PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:  # noqa: BLE001 - a bad settings file must not stop the app
        print(f"[settings] ignoring unreadable {PATH}: {e!r}", file=sys.stderr)
        return {}


def set_value(key, value):
    """Persist one setting. Returns True if it reached disk."""
    data = load()
    data[key] = value
    try:
        tmp = PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, PATH)   # atomic: a crash mid-write cannot leave a half-written file
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[settings] could not write {PATH}: {e!r}", file=sys.stderr)
        return False


def is_usable(path):
    """True if `path` can hold output -- it exists (or can be created) and is writable."""
    try:
        os.makedirs(path, exist_ok=True)
        return os.access(path, os.W_OK)
    except Exception:  # noqa: BLE001
        return False


def output_dir(default):
    """Where to save: the remembered folder if it is still usable, otherwise `default`.

    Usability is checked rather than assumed because a remembered folder can sit on a drive
    that is no longer mounted. Failing every save from then on would be worse than going back
    to the folder next to the app and saying so.
    """
    saved = load().get("output_dir")
    if not saved:
        return default
    if is_usable(saved):
        return saved
    print(f"[settings] remembered output folder is unusable ({saved}) -> using {default}",
          file=sys.stderr)
    return default


def set_output_dir(path):
    """Remember `path` as where output goes from now on."""
    return set_value("output_dir", os.path.abspath(path))
