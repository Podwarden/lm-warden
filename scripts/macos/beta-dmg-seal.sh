#!/bin/bash
# beta-dmg-seal.sh — the last gate on a BETA DMG, then its checksum (P8).
#
# Usage:
#   scripts/macos/beta-dmg-seal.sh <LM-Warden-<tag>-beta.dmg> <tag>
#
# OWNER DECISION 4 (P8): a beta DMG may be published before notarization. It
# is ad-hoc signed, labelled beta, offered as a GitHub pre-release download
# only, and it must never auto-update: the app inside carries no Sparkle key,
# so it never checks a feed, and no feed ever lists it. `make macos-beta-dmg`
# builds it with LMW_BETA_BUILD=1 and the Sparkle vars unset; this script
# checks the bytes that ship, so a build that slipped a key in (a make
# command-line override, a future writer) is caught here:
#
#   1. <dmg> must be named LM-Warden-<tag>-beta.dmg and <tag> must be a
#      strict release tag vYYYY.MM.DD.N (bundle-version.sh prints non-zero).
#      Exit 2 otherwise.
#   2. Any stale <dmg>.sha256 is removed first, so a refused DMG never keeps
#      a checksum (the publish job needs both files).
#   3. The DMG is attached read-only (no Finder window, private mount point);
#      the app's Info.plist must carry no SU* key at all and its
#      CFBundleShortVersionString must be <tag>. It is detached on every exit.
#   4. Only then: <dmg>.sha256, in `shasum -a 256` format
#      ("<hash>  <basename>"), so `shasum -a 256 -c` works next to the
#      download.
#
# Exit codes: 0 sealed, 1 refused (no checksum), 2 usage.
#
# Bash-3.2-safe (the M1 runner's /bin/bash is 3.2).
set -eu

usage() { echo "usage: beta-dmg-seal.sh <LM-Warden-<tag>-beta.dmg> <tag>" >&2; exit 2; }
die() { echo "beta-dmg-seal.sh: $*" >&2; exit 1; }

[ $# -eq 2 ] && [ -n "$1" ] && [ -n "$2" ] || usage
dmg=$1
tag=$2
script_dir=$(cd "$(dirname "$0")" && pwd)

bv=$(/bin/bash "$script_dir/bundle-version.sh" "$tag" 2>/dev/null) || bv=0
if [ "$bv" = "0" ]; then
    echo "beta-dmg-seal.sh: '$tag' is not a release tag (want vYYYY.MM.DD.N)" >&2
    usage
fi
name=$(basename "$dmg")
if [ "$name" != "LM-Warden-$tag-beta.dmg" ]; then
    echo "beta-dmg-seal.sh: '$name' is not LM-Warden-$tag-beta.dmg" >&2
    usage
fi
[ -f "$dmg" ] || die "no DMG at $dmg"

sums="$dmg.sha256"
rm -f "$sums"

mnt=$(mktemp -d -t lmw-beta-seal)
attached=0
cleanup() {
    if [ "$attached" = 1 ]; then
        hdiutil detach -quiet "$mnt" 2>/dev/null || hdiutil detach -force -quiet "$mnt" 2>/dev/null || true
    fi
    rmdir "$mnt" 2>/dev/null || true
}
trap cleanup EXIT

hdiutil attach -readonly -nobrowse -noautoopen -mountpoint "$mnt" "$dmg" >/dev/null \
    || die "could not attach $dmg"
attached=1

plist="$mnt/LM Warden.app/Contents/Info.plist"
[ -f "$plist" ] || die "no LM Warden.app/Contents/Info.plist on $name"

# Any Sparkle key, not just the two the writer knows: SUFeedURL,
# SUPublicEDKey, SUEnableAutomaticChecks, ... A beta carries none.
xml=$(plutil -convert xml1 -o - "$plist") || die "could not read $plist"
su=$(printf '%s\n' "$xml" | sed -n 's/.*<key>\(SU[A-Za-z]*\)<\/key>.*/\1/p' | tr '\n' ' ')
[ -z "$su" ] || die "$name carries Sparkle key(s): ${su% } -- a beta must never check an update feed"

got=$(plutil -extract CFBundleShortVersionString raw -o - "$plist" 2>/dev/null) \
    || die "no CFBundleShortVersionString in the app on $name"
[ "$got" = "$tag" ] || die "the app on $name is $got, not $tag"

hdiutil detach -quiet "$mnt" || die "could not detach $mnt"
attached=0

dir=$(cd "$(dirname "$dmg")" && pwd)
( cd "$dir" && shasum -a 256 "$name" > "$name.sha256.tmp" && mv "$name.sha256.tmp" "$name.sha256" ) \
    || { rm -f "$dir/$name.sha256.tmp"; die "could not write $sums"; }
echo "beta-dmg-seal.sh: sealed $name ($(cut -c1-12 "$sums")..., no Sparkle keys)"
