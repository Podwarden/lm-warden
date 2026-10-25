#!/bin/bash
# make-dmg.sh — pack a built, signed "LM Warden.app" into a DMG (Task 7, P8).
#
# Usage:
#   scripts/macos/make-dmg.sh <app> <version> <out.dmg>
#
# <app> is the bundled-Python app `make macos-bundle-python` leaves at
# dist/LM Warden.app: the release binary, an Info.plist from
# write-info-plist.sh, the embedded interpreter + deps + app/ under
# Contents/Resources/python, signed inside-out by sign-bundle.sh. This script
# does not build or sign anything (re-signing the ~600 MB tree here would only
# repeat what macos-bundle-python did); it refuses to pack what it is not sure
# of:
#
#   1. <app> must carry "Contents/Resources/python/bin/LM Warden Server" (the
#      renamed interpreter bundle-python.sh lays down, the path
#      ShellAppDelegate.bundledPythonPath() launches): a DMG without the
#      server's interpreter is a shell that cannot start the server.
#   2. Its CFBundleShortVersionString must equal <version>, so the DMG name
#      and the app inside it never disagree.
#   3. It is copied with `ditto` (keeps the signature, xattrs, symlinks) into a
#      staging dir as "LM Warden.app", and verify-bundle.sh checks THAT copy --
#      the bytes the DMG is made from. A bundle that no longer verifies never
#      reaches the image.
#   4. `hdiutil create` makes a read-write (UDRW) image with volume name
#      "LM Warden" in a scratch dir, and `hdiutil convert` turns it into the
#      shipped ULMO (LZMA) DMG, which is moved into place only once complete.
#      LZMA, not zlib: the same app measured ~134 MB as ULMO vs ~373 MB as UDZO
#      on the M1 (footprint plan: 128.3 vs 341.4 MB), and ULMO is read-only and
#      opens on macOS 14, the app's minimum. Two steps, not `create -format
#      ULMO`: the one-step create packs the same app to ~255 MB.
#
# Exit codes: 0 ok, 1 a gate or step failed (no DMG is written), 2 usage.
#
# Bash-3.2-safe (the M1 runner's /bin/bash is 3.2): no associative arrays, no
# `${x,,}`, no `mapfile`.
set -eu

usage() { echo "usage: make-dmg.sh <app> <version> <out.dmg>" >&2; exit 2; }
die() { echo "make-dmg.sh: $*" >&2; exit 1; }

[ $# -eq 3 ] && [ -n "$2" ] && [ -n "$3" ] || usage
app=$1
version=$2
out_dmg=$3
[ -d "$app/Contents" ] || usage

script_dir=$(cd "$(dirname "$0")" && pwd)
verify_bundle="$script_dir/verify-bundle.sh"

# Absolute paths — hdiutil and codesign both behave better with them.
app_abs=$(cd "$app" && pwd)
case "$out_dmg" in
    /*) out_dmg_abs=$out_dmg ;;
    *) out_dmg_abs=$(pwd)/$out_dmg ;;
esac

[ -x "$app_abs/Contents/Resources/python/bin/LM Warden Server" ] \
    || die "no bundled python in $app (Contents/Resources/python/bin/LM Warden Server); build it with 'make macos-bundle-python'"

plist="$app_abs/Contents/Info.plist"
[ -f "$plist" ] || die "no Info.plist in $app"
got=$(plutil -extract CFBundleShortVersionString raw -o - "$plist" 2>/dev/null) \
    || die "no CFBundleShortVersionString in $plist"
[ "$got" = "$version" ] || die "CFBundleShortVersionString is $got, not $version ($plist)"

# Stage in a temp dir so the DMG srcfolder is exactly the app.
staging=$(mktemp -d -t lmwarden-dmg)
# Scratch for the read-write image and the converted DMG: outside $staging (the
# srcfolder must hold only the app) and removed on any exit.
rw_dir=$(mktemp -d -t lmwarden-dmg-rw)
trap 'rm -rf "$staging" "$rw_dir"' EXIT
app_root="$staging/LM Warden.app"

ditto "$app_abs" "$app_root"

# Verify what ships. Under set -e a non-zero exit stops the script here, so no
# DMG is produced.
"$verify_bundle" "$app_root"

# ULMO = LZMA-compressed, read-only, single partition: ~60% smaller than the old
# UDZO (zlib), and it opens on macOS 14, the app's minimum. It is made in two
# steps -- create a UDRW image, then convert it to ULMO -- because a one-step
# `hdiutil create -format ULMO -srcfolder` compresses far worse (measured on the
# M1, same app: ~255 MB one-step vs ~134 MB two-step). The converted image is
# moved over <out.dmg> only once it is complete, so a failed step leaves no
# DMG. The volume name "LM Warden" is what appears in Finder.
hdiutil create \
    -volname "LM Warden" \
    -srcfolder "$staging" \
    -ov \
    -format UDRW \
    "$rw_dir/rw.dmg"
hdiutil convert "$rw_dir/rw.dmg" -format ULMO -o "$rw_dir/ulmo.dmg"
mv -f "$rw_dir/ulmo.dmg" "$out_dmg_abs"

# Report the bytes actually written (BSD stat: %z = size in bytes) so a CI log
# shows the shipped DMG's footprint.
echo "make-dmg.sh: wrote $out_dmg_abs ($(stat -f %z "$out_dmg_abs") bytes)"
