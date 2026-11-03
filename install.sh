#!/bin/bash
# Installs TextAnimator.ofx.bundle into /Library/OFX/Plugins -- the macOS
# counterpart of install.ps1.
#
#   ./install.sh            install the last build
#   ./install.sh --build    build first
#   ./install.sh --force    quit DaVinci Resolve first instead of just warning
#
# Resolve scans for OFX plugins only at launch, so it must be restarted after
# installing. Writing to /Library needs administrator rights; sudo asks for
# your password when it does.
set -euo pipefail

DEST="/Library/OFX/Plugins"
BUILD=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --build) BUILD=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

repo="$(cd "$(dirname "$0")" && pwd)"
bundle="$repo/build/bundle/TextAnimator.ofx.bundle"
payload="$bundle/Contents/MacOS/TextAnimator.ofx"

step() { printf '\033[36m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m    %s\033[0m\n' "$*"; }
die()  { printf '\n\033[31mINSTALL FAILED\n%s\033[0m\n' "$*" >&2; exit 1; }

# Built as you, never under sudo: a root-owned build/ makes every later build fail.
if [ "$BUILD" = 1 ]; then
  step "Building..."
  "$repo/build.sh"
fi

step "Checking bundle..."
[ -d "$bundle" ] || die "Bundle not found at:
  $bundle
Run ./build.sh first, or re-run this with --build."
[ -f "$payload" ] || die "Bundle is malformed - missing Contents/MacOS/TextAnimator.ofx"
[ -f "$bundle/Contents/Info.plist" ] || die "Bundle is malformed - missing Contents/Info.plist"
ok "Found $(( $(stat -f %z "$payload") / 1024 )) KB plugin binary ($(lipo -archs "$payload"))."

# Resolve's process is "Resolve"; the app is "DaVinci Resolve".
if pgrep -x Resolve >/dev/null 2>&1; then
  if [ "$FORCE" = 1 ]; then
    step "Quitting DaVinci Resolve..."
    osascript -e 'quit app "DaVinci Resolve"' >/dev/null 2>&1 || true
    for _ in $(seq 1 30); do
      pgrep -x Resolve >/dev/null 2>&1 || break
      sleep 1
    done
    pgrep -x Resolve >/dev/null 2>&1 && die "Resolve did not quit within 30s. Quit it manually and re-run."
    ok "Quit."
  else
    printf '\033[33mWARNING: DaVinci Resolve is running. It only loads plugins at launch, so restart it after this.\033[0m\n'
  fi
fi

# Probe rather than assume: a writable folder needs no password prompt.
SUDO=""
if [ -e "$DEST" ]; then writable="$DEST"; else writable="$(dirname "$DEST")"; fi
if [ ! -w "$writable" ]; then
  SUDO="sudo"
  step "Administrator rights required for $DEST"
fi

step "Installing to $DEST"
target="$DEST/TextAnimator.ofx.bundle"
$SUDO mkdir -p "$DEST"
if [ -e "$target" ]; then
  ok "Replacing existing installation."
  $SUDO rm -rf "$target"
fi
# ditto, not cp: it keeps the bundle's code signature and extended attributes intact.
$SUDO ditto "$bundle" "$target"
# A bundle that arrived by download carries a quarantine flag, and macOS then
# refuses to let Resolve load it without saying why.
$SUDO xattr -dr com.apple.quarantine "$target" 2>/dev/null || true

installed="$target/Contents/MacOS/TextAnimator.ofx"
[ -f "$installed" ] || die "Copy appeared to succeed but $installed is missing."
[ "$(shasum -a 256 < "$payload")" = "$(shasum -a 256 < "$installed")" ] \
  || die "Installed binary does not match the built binary."
codesign --verify "$target" >/dev/null 2>&1 || die "The installed bundle fails codesign --verify."

ok "Verified: $installed"
echo
printf '\033[32mInstalled successfully.\033[0m\n'
echo
echo "Next:"
echo "  1. Start DaVinci Resolve."
echo "  2. Put a Text+ on its own track over any clip."
echo "  3. Apply OpenFX > Filters > Text > 'Text Animator' to the Text+ clip."
echo "  4. Turn on 'Show Detection' first to confirm the words are found."
