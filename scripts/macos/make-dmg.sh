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
#   4. `hdiutil create` makes a UDZO DMG with volume name "LM Warden".
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
trap 'rm -rf "$staging"' EXIT
app_root="$staging/LM Warden.app"

ditto "$app_abs" "$app_root"

# Verify what ships. Under set -e a non-zero exit stops the script here, so no
# DMG is produced.
"$verify_bundle" "$app_root"

# UDZO = compressed, read-only, single partition. -ov overwrites a stale
# output file. The volume name "LM Warden" is what appears in Finder.
hdiutil create \
    -volname "LM Warden" \
    -srcfolder "$staging" \
    -ov \
    -format UDZO \
    "$out_dmg_abs"

echo "make-dmg.sh: wrote $out_dmg_abs"
