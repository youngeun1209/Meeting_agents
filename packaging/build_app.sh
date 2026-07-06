#!/bin/bash
# Rebuilds "Meeting STT.app" from source: renders the icon (mic + note),
# packs it into an .icns, and assembles the bundle around setup_app.py.
# Run:  bash packaging/build_app.sh
set -e
cd "$(dirname "$0")"                 # packaging/
ROOT="$(cd .. && pwd)"              # project folder (holds setup_app.py, gui.py)
APP="$ROOT/Meeting STT.app"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "1/4 rendering icon ..."
swiftc -O render_icon.swift -o "$WORK/render_icon"
"$WORK/render_icon" "$WORK/icon_1024.png"

echo "2/4 building icns ..."
mkdir "$WORK/AppIcon.iconset"
gen () { sips -z "$2" "$2" "$WORK/icon_1024.png" --out "$WORK/AppIcon.iconset/icon_$1.png" >/dev/null; }
gen 16x16 16;    gen 16x16@2x 32
gen 32x32 32;    gen 32x32@2x 64
gen 128x128 128; gen 128x128@2x 256
gen 256x256 256; gen 256x256@2x 512
gen 512x512 512; cp "$WORK/icon_1024.png" "$WORK/AppIcon.iconset/icon_512x512@2x.png"
iconutil -c icns "$WORK/AppIcon.iconset" -o "$WORK/AppIcon.icns"

echo "3/4 assembling bundle ..."
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$WORK/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
cp Info.plist "$APP/Contents/Info.plist"
cp launch "$APP/Contents/MacOS/launch"
chmod +x "$APP/Contents/MacOS/launch"
printf 'APPL????' > "$APP/Contents/PkgInfo"

echo "4/4 registering with Finder ..."
touch "$APP"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP" 2>/dev/null || true

echo "done -> $APP"
