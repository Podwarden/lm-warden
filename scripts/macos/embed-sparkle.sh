#!/bin/bash
# embed-sparkle.sh — place the pinned Sparkle.framework in "LM Warden.app" (P8).
#
# Usage:
#   scripts/macos/embed-sparkle.sh <app> [--cache-dir <dir>] [--build-dir <dir>] [--offline]
#   scripts/macos/embed-sparkle.sh --fetch-only [--cache-dir <dir>] [--offline]
#   scripts/macos/embed-sparkle.sh [<app>] --print-plan [--cache-dir <dir>]
#
# DECISION (P8 sparkle-embed, 2026-10-21): Sparkle 2.10.0, the latest stable
# 2.x release (published 2026-09-13), consumed as upstream's SwiftPM binary
# artifact Sparkle-for-Swift-Package-Manager.zip, pinned by exact URL + sha256:
#   - macos/Package.swift links it as `.binaryTarget(url:checksum:)`, so the
#     merge gate's bare `swift build && swift test` resolves it with no
#     pre-step (a vendored path target would need a fetch before every swift
#     invocation), and SwiftPM itself refuses a zip whose sha256 differs.
#   - This script ships it: it fetches the SAME zip into <cache-dir> (default
#     ~/Library/Caches/lmw-macos-sparkle), checks the SAME sha256 (every run,
#     a cached copy included -- a mismatching cache file is refused, never
#     silently replaced), extracts the macOS slice of Sparkle.xcframework and
#     copies Sparkle.framework (symlinks preserved) into
#     <app>/Contents/Frameworks, the dir the binary's
#     LC_RPATH @executable_path/../Frameworks points at.
#   - The full release tarball (Sparkle-2.10.0.tar.xz) was not chosen: it
#     carries the same framework plus a test app and tools we do not ship, and
#     SwiftPM cannot consume it, so it would be a second pin for one binary.
# The sha256 is the value GitHub publishes as the asset digest and the one
# upstream's own Package.swift at tag 2.10.0 declares. tests/unit/
# test_embed_sparkle.py keeps the pins here equal to macos/Package.swift's.
#
# Checks before placing (any failure is exit 1 and nothing is placed):
#   - sha256 of the zip == SPARKLE_SHA256;
#   - the framework's CFBundleShortVersionString == SPARKLE_VERSION;
#   - the nested code Sparkle needs at update time is present:
#     Versions/B/{Sparkle,Autoupdate}, Versions/B/Updater.app,
#     Versions/B/XPCServices/{Downloader,Installer}.xpc, and Versions/Current
#     is the symlink to B;
#   - with --build-dir <dir> holding a Sparkle.framework (SwiftPM's copy, the
#     one the binary was linked against), its Versions/B/Sparkle is
#     byte-identical to the one placed: the app never ships a framework other
#     than the one it was built with.
# Signing is NOT done here: sign-bundle.sh signs the framework's nested code
# inside-out (XPCs, Updater.app, Autoupdate, then the framework) with the rest
# of the bundle, and verify-bundle.sh requires each of them validly signed.
#
# --fetch-only: fetch + verify the zip into the cache and stop.
# --print-plan: key=value lines of the resolved plan; touches nothing.
#
# Exit codes: 0 ok, 1 a step failed (download, checksum, version, layout,
# build mismatch, placement), 2 usage (including an <app> with no Contents/).
#
# Bash-3.2-safe (the M1 runner's /bin/bash is 3.2): no associative arrays, no
# ${x,,}, no mapfile.
set -eu

# --- the pin ---------------------------------------------------------------
SPARKLE_VERSION="2.10.0"
SPARKLE_URL="https://github.com/sparkle-project/Sparkle/releases/download/${SPARKLE_VERSION}/Sparkle-for-Swift-Package-Manager.zip"
SPARKLE_SHA256="17e28312b8e18ab7cdbbe09a6fb28cc55a5479ec6c371dbc07cdecd2a14fd959"
# Where the macOS framework sits inside the zip.
SPARKLE_SLICE="Sparkle.xcframework/macos-arm64_x86_64/Sparkle.framework"

USAGE="usage: embed-sparkle.sh <app>|--fetch-only [--cache-dir <dir>] [--build-dir <dir>] [--offline] [--print-plan]"

die() { echo "embed-sparkle.sh: $*" >&2; exit 1; }
usage() { echo "$USAGE" >&2; exit 2; }

app=""
cache_dir="${HOME}/Library/Caches/lmw-macos-sparkle"
build_dir=""
offline=0
fetch_only=0
print_plan=0
while [ $# -gt 0 ]; do
    case $1 in
        --cache-dir) [ $# -ge 2 ] || usage; cache_dir=$2; shift 2 ;;
        --build-dir) [ $# -ge 2 ] || usage; build_dir=$2; shift 2 ;;
        --offline) offline=1; shift ;;
        --fetch-only) fetch_only=1; shift ;;
        --print-plan) print_plan=1; shift ;;
        -*) usage ;;
        *) [ -z "$app" ] || usage; app=$1; shift ;;
    esac
done

archive="$cache_dir/Sparkle-for-Swift-Package-Manager-${SPARKLE_VERSION}.zip"

if [ $print_plan = 1 ]; then
    echo "sparkle_version=$SPARKLE_VERSION"
    echo "sparkle_url=$SPARKLE_URL"
    echo "sparkle_sha256=$SPARKLE_SHA256"
    echo "archive=$archive"
    [ -z "$app" ] || echo "dest=$app/Contents/Frameworks/Sparkle.framework"
    exit 0
fi

if [ $fetch_only = 0 ]; then
    [ -n "$app" ] || usage
    [ -d "$app/Contents" ] || { echo "embed-sparkle.sh: not an app bundle (no Contents/): $app" >&2; usage; }
fi

sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }

# --- 1. fetch + verify -----------------------------------------------------
mkdir -p "$cache_dir"
if [ -f "$archive" ]; then
    got=$(sha256_of "$archive")
    [ "$got" = "$SPARKLE_SHA256" ] || die "checksum mismatch for cached $archive: got $got, want $SPARKLE_SHA256 (refusing; delete the file to re-fetch)"
    echo "fetch: cached $archive (sha256 ok)"
else
    [ $offline = 0 ] || die "--offline and no cached archive: $archive"
    part="$archive.part.$$"
    rm -f "$part"
    curl -fsSL --proto '=https' --retry 3 -o "$part" "$SPARKLE_URL" || { rm -f "$part"; die "download failed: $SPARKLE_URL"; }
    got=$(sha256_of "$part")
    [ "$got" = "$SPARKLE_SHA256" ] || { rm -f "$part"; die "checksum mismatch for $SPARKLE_URL: got $got, want $SPARKLE_SHA256"; }
    mv "$part" "$archive"
    echo "fetch: downloaded $SPARKLE_URL (sha256 ok)"
fi
[ $fetch_only = 0 ] || exit 0

# --- 2. extract + check the layout -----------------------------------------
work=$(mktemp -d "${TMPDIR:-/tmp}/embed-sparkle.XXXXXX")
trap 'rm -rf "$work"' EXIT
ditto -x -k "$archive" "$work/x" || die "could not extract $archive"
src="$work/x/$SPARKLE_SLICE"
[ -d "$src" ] || die "no $SPARKLE_SLICE in $archive"

ver=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$src/Versions/B/Resources/Info.plist" 2>/dev/null) \
    || die "no CFBundleShortVersionString in the framework's Info.plist"
[ "$ver" = "$SPARKLE_VERSION" ] || die "framework version is $ver, want $SPARKLE_VERSION"

[ -L "$src/Versions/Current" ] && [ "$(readlink "$src/Versions/Current")" = "B" ] \
    || die "Versions/Current is not a symlink to B"
for rel in Versions/B/Sparkle Versions/B/Autoupdate \
           Versions/B/Updater.app/Contents/MacOS/Updater \
           Versions/B/XPCServices/Downloader.xpc/Contents/MacOS/Downloader \
           Versions/B/XPCServices/Installer.xpc/Contents/MacOS/Installer; do
    [ -f "$src/$rel" ] || die "framework is missing $rel"
    file -b "$src/$rel" | grep -q "Mach-O" || die "not a Mach-O: $rel"
done

# --- 3. the linked framework must be the shipped one -----------------------
if [ -n "$build_dir" ]; then
    linked="$build_dir/Sparkle.framework/Versions/B/Sparkle"
    [ -f "$linked" ] || die "--build-dir has no Sparkle.framework: $build_dir"
    cmp -s "$linked" "$src/Versions/B/Sparkle" \
        || die "the Sparkle the app was linked against ($linked) differs from pinned $SPARKLE_VERSION"
    echo "build: $linked matches the pinned framework"
fi

# --- 4. place ----------------------------------------------------------------
dest_dir="$app/Contents/Frameworks"
dest="$dest_dir/Sparkle.framework"
mkdir -p "$dest_dir"
rm -rf "$dest"
ditto "$src" "$dest" || die "could not copy Sparkle.framework into $dest_dir"
[ -L "$dest/Versions/Current" ] || die "placement lost the Versions/Current symlink"
echo "placed: Sparkle $SPARKLE_VERSION -> $dest"
