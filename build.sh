#!/bin/bash
# Builds the plugin and the test harness on macOS -- the counterpart of build.ps1.
#
#   ./build.sh            Release build
#   ./build.sh --debug    Debug build
#   ./build.sh --clean    wipe build/ first
#
# Needs the Xcode command line tools (xcode-select --install) and CMake 3.20+
# (brew install cmake). Ninja is used when present, Make otherwise.
set -euo pipefail

CONFIG=Release
CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --debug) CONFIG=Debug ;;
    --clean) CLEAN=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

root="$(cd "$(dirname "$0")" && pwd)"
build="$root/build"

die() { echo "error: $*" >&2; exit 1; }

xcode-select -p >/dev/null 2>&1 || die "Xcode command line tools missing. Run: xcode-select --install"
command -v cmake >/dev/null 2>&1 || die "cmake not found. Run: brew install cmake"

# The same OpenFX commit the Windows build is made with, so the Support library
# compiled into the plugin is identical on both platforms.
OPENFX_COMMIT=3de640d6f645fe6e346acd57e568d8b0a5ae4574
if [ ! -f "$root/external/openfx/Support/Library/ofxsCore.cpp" ]; then
  echo "==> Fetching OpenFX SDK"
  rm -rf "$root/external/openfx"
  git init -q "$root/external/openfx"
  git -C "$root/external/openfx" remote add origin https://github.com/AcademySoftwareFoundation/openfx.git
  git -C "$root/external/openfx" fetch -q --depth 1 origin "$OPENFX_COMMIT"
  git -C "$root/external/openfx" checkout -q FETCH_HEAD
fi

if [ "$CLEAN" = 1 ]; then rm -rf "$build"; fi

generator="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then generator="Ninja"; fi

echo "==> Configuring ($CONFIG, $generator)"
cmake -S "$root" -B "$build" -G "$generator" -DCMAKE_BUILD_TYPE="$CONFIG"

echo "==> Building"
cmake --build "$build" --parallel "$(sysctl -n hw.ncpu)"

payload="$build/bundle/TextAnimator.ofx.bundle/Contents/MacOS/TextAnimator.ofx"
[ -f "$payload" ] || die "build finished but $payload is missing"

echo
echo "Built:"
echo "  $build/bundle/TextAnimator.ofx.bundle  ($(lipo -archs "$payload"))"
echo "  $build/segtest"
