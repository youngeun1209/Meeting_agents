#!/bin/bash
# Build the ScreenCaptureKit system-audio sidecar.
# Usage: bash native/build.sh  -> produces the native/sysaudio binary.
set -e
cd "$(dirname "$0")"

swiftc -O sysaudio.swift -o sysaudio \
    -target arm64-apple-macos13.0 \
    -framework ScreenCaptureKit \
    -framework AVFoundation \
    -framework CoreMedia

echo "build complete: $(pwd)/sysaudio"
echo "On first run a 'Screen Recording' permission prompt appears -> allow it or no audio is captured."
