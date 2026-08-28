"""Set this process's macOS Dock icon.

gui.py runs as a plain `python gui.py` process, not from inside the .app bundle: the bundle's
executable deliberately hands off to Terminal (see "Meeting STT.app/Contents/MacOS/launch" for
why -- an ad-hoc signed app cannot hold a stable TCC grant). LaunchServices therefore never
associates the running process with the bundle, and the Dock shows the generic Python rocket
instead of the app icon.

Tk cannot fix this. Measured on Tk 8.6.14: `wm iconphoto` leaves NSApplication's
applicationIconImage pointer unchanged, and the Dock keeps the rocket. The Dock icon of a
already-running process is settable only through AppKit's -[NSApplication
setApplicationIconImage:], which is what this module calls -- via ctypes, so that three
message sends don't pull in pyobjc as a dependency.
"""
import ctypes
import ctypes.util
import os
import sys

ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "packaging", "AppIcon.icns")


def apply(path=ICON_PATH):
    """Point the Dock icon at `path` (.icns). Returns True if it was set.

    MUST be called only after Tk has been initialized. Tk installs its own NSApplication
    subclass (TKApplication); calling [NSApplication sharedApplication] before that creates a
    plain NSApplication instead, and Tk then dies on
    `-[NSApplication macOSVersion]: unrecognized selector`.

    Cosmetic by nature, so a failure here must never take the app down with it: every
    failure path returns False and says why on stderr.
    """
    if sys.platform != "darwin":
        return False
    if not os.path.isfile(path):
        print(f"[dock_icon] icon not found: {path}", file=sys.stderr)
        return False
    try:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        # objc_msgSend is variadic in the headers; ctypes needs one concrete prototype per
        # argument count, hence two casts of the same symbol rather than one shared function.
        send = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        send1 = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p)(("objc_msgSend", objc))

        def cls(name):
            return ctypes.c_void_p(objc.objc_getClass(name.encode()))

        def sel(name):
            return ctypes.c_void_p(objc.sel_registerName(name.encode()))

        nspath = ctypes.c_void_p(send1(
            cls("NSString"), sel("stringWithUTF8String:"),
            ctypes.cast(ctypes.c_char_p(path.encode("utf-8")), ctypes.c_void_p)))
        image = ctypes.c_void_p(send1(
            ctypes.c_void_p(send(cls("NSImage"), sel("alloc"))),
            sel("initWithContentsOfFile:"), nspath))
        if not image.value:
            print(f"[dock_icon] NSImage could not read {path}", file=sys.stderr)
            return False
        nsapp = ctypes.c_void_p(send(cls("NSApplication"), sel("sharedApplication")))
        send1(nsapp, sel("setApplicationIconImage:"), image)
        return True
    except Exception as e:  # noqa: BLE001 - a cosmetic icon must not break startup
        print(f"[dock_icon] could not set the Dock icon: {e!r}", file=sys.stderr)
        return False
