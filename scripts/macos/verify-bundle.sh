#!/bin/bash
# verify-bundle.sh — verify a signed .app is intact and validly signed (P8).
#
# Usage:
#   scripts/macos/verify-bundle.sh <app> [--require-sparkle] [--entitlements-dir <dir>]
#
# Follows sign-bundle.sh: it finds every nested Mach-O the same way (by
# `file -b` output containing "Mach-O", not by extension) and runs
# `codesign --verify --strict` on each, in path order (LC_ALL=C sort), then
# `codesign --verify --strict --deep` on <app> itself, which walks the whole
# tree. A file that is not in the bundle's seal (added after signing) fails its
# own verify; a file in the seal that was re-signed over breaks the seal and
# fails the whole-tree verify. Verification with --deep is fine; only
# *signing* with --deep is forbidden.
#
# Mach-O files are discovered by `file -b` output containing "Mach-O" (an
# extensionless binary is found, a text file named x.so is not).
#
# Output: the first failing item prints
# "verify-bundle.sh: not validly signed: <path relative to the app's parent
# dir>" to stderr and the script exits 1. On success it prints
# "verify-bundle.sh: verified <n> Mach-O + bundle" (n = number of Mach-O
# files found) and exits 0.
#
# Sparkle (P8): when the main executable (Info.plist CFBundleExecutable)
# links @rpath/Sparkle.framework, when Contents/Frameworks/Sparkle.framework
# exists, or with --require-sparkle (make macos-bundle-python passes it), the
# framework's nested code must be present and each piece must pass
# `codesign --verify --strict` as a unit, innermost first:
# XPCServices/Downloader.xpc, XPCServices/Installer.xpc, Updater.app,
# Autoupdate, then Sparkle.framework. A linking executable must also carry
# LC_RPATH @executable_path/../Frameworks (so @rpath resolves inside the
# bundle), and --require-sparkle fails an executable that does not link it.
# A missing piece prints "verify-bundle.sh: missing: <path>"; a bad signature
# the usual "not validly signed: <path>". On success one line
# "verify-bundle.sh: sparkle <n> nested items signed" precedes the summary.
#
# Hardened runtime + entitlements (P8 hardened-runtime): after the per-file
# validity pass, every Mach-O must carry the runtime flag (`codesign -dv`
# flags include "runtime"; else "verify-bundle.sh: no hardened runtime:
# <path>") and exactly the entitlements sign-bundle.sh gives its role, read
# from <dir> (default the repo's macos/entitlements): the main executable
# (Info.plist CFBundleExecutable) LMWarden.entitlements, the server interpreter
# Contents/Resources/python/bin/LM Warden Server LMWardenServer.entitlements,
# every other Mach-O (Sparkle's pieces included) none. Top-level key/value
# pairs are compared order-insensitively; a mismatch prints
# "verify-bundle.sh: entitlements differ from <file name|none>: <path>". On
# bundle that embeds Python (Contents/Resources/python exists) must have the
# server interpreter at that exact path as a Mach-O, so a renamed or moved
# interpreter cannot silently fall into the "none" role (else
# "verify-bundle.sh: server interpreter missing or not Mach-O: <path>"). On
# success one line "verify-bundle.sh: hardened runtime on <n> Mach-O,
# entitlements as committed (<k> pinned)" precedes the Sparkle line and the
# summary.
#
# Exit codes: 0 verified, 1 a Mach-O or the bundle failed verification (the
# path is named), 2 <app> missing or not a directory, a bad option (usage
# line), or an entitlements file missing.
#
# Bash-3.2-safe (the M1 runner's /bin/bash is 3.2): no associative arrays, no
# ${x,,}, no mapfile. Paths may contain spaces, so everything is quoted and
# find output is read with `while IFS= read -r`.
set -eu

usage="usage: verify-bundle.sh <app> [--require-sparkle] [--entitlements-dir <dir>]"
app=${1:-}
[ $# -eq 0 ] || shift
require_sparkle=0
script_dir=$(cd "$(dirname "$0")" && pwd)
ent_dir="$script_dir/../../macos/entitlements"
while [ $# -gt 0 ]; do
    case "$1" in
        --require-sparkle) require_sparkle=1; shift ;;
        --entitlements-dir)
            [ $# -ge 2 ] && [ -n "$2" ] || { app=""; break; }
            ent_dir=$2
            shift 2
            ;;
        *) app=""; break ;;
    esac
done
case "$app" in -*) app="" ;; esac
if [ -z "$app" ] || [ ! -d "$app" ]; then
    echo "$usage" >&2
    exit 2
fi
shell_ent="$ent_dir/LMWarden.entitlements"
server_ent="$ent_dir/LMWardenServer.entitlements"
for ent in "$shell_ent" "$server_ent"; do
    [ -f "$ent" ] || {
        echo "verify-bundle.sh: missing entitlements file: $ent" >&2
        exit 2
    }
done

# Normalized entitlements: one "key=value" line per top-level entry, sorted
# (LC_ALL=C), from an XML plist on stdin. Scalars only -- the committed sets
# are booleans; an array or dict value leaves its key as "key=?", which never
# matches a committed set, so it fails closed.
ent_pairs() {
    tr -d '\n\t' \
        | grep -oE '<key>[^<]*</key>|<true/>|<false/>|<string>[^<]*</string>|<integer>[^<]*</integer>|<array|<dict' \
        | awk '/^<key>/ { if (k != "") print k "=?"; k = $0; sub(/^<key>/, "", k); sub(/<\/key>$/, "", k); next }
               k == "" { next }
               /^<true\/>$/ { print k "=true"; k = ""; next }
               /^<false\/>$/ { print k "=false"; k = ""; next }
               /^<(string|integer)>/ { v = $0; sub(/^<[a-z]*>/, "", v); sub(/<\/[a-z]*>$/, "", v); print k "=" v; k = ""; next }
               { print k "=?"; k = "" }
               END { if (k != "") print k "=?" }' \
        | LC_ALL=C sort
}

app_abs=$(cd "$app" && pwd)
app_parent=$(cd "$app_abs/.." && pwd)
app_name=${app_abs##*/}
contents="$app_abs/Contents"

# Verify every Mach-O under Contents, in path order (LC_ALL=C). The first
# failing one is named (relative to the app's parent dir) and exits 1.
count=0
if [ -d "$contents" ]; then
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        [ -f "$f" ] || continue
        file -b "$f" | grep -q "Mach-O" || continue
        rel=${f#"$app_parent"/}
        codesign --verify --strict "$f" || {
            echo "verify-bundle.sh: not validly signed: $rel" >&2
            exit 1
        }
        count=$((count + 1))
    done <<EOF
$(find "$contents" -type f | LC_ALL=C sort)
EOF
fi

main_exe=""
if [ -f "$contents/Info.plist" ]; then
    exe_name=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$contents/Info.plist" 2>/dev/null || true)
    [ -z "$exe_name" ] || main_exe="$contents/MacOS/$exe_name"
fi

# Hardened runtime and the entitlement set per role (see the header).
shell_pairs=$(ent_pairs < "$shell_ent")
server_pairs=$(ent_pairs < "$server_ent")
server_path="$contents/Resources/python/bin/LM Warden Server"
rt_count=0
pinned=0
server_seen=0
if [ -d "$contents" ]; then
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        [ -f "$f" ] || continue
        file -b "$f" | grep -q "Mach-O" || continue
        rel=${f#"$app_parent"/}
        codesign -dv "$f" 2>&1 | grep -qE '^CodeDirectory .*flags=0x[0-9a-f]+\([^)]*runtime' || {
            echo "verify-bundle.sh: no hardened runtime: $rel" >&2
            exit 1
        }
        if [ "$f" = "$main_exe" ]; then
            want=$shell_pairs; label=${shell_ent##*/}; pinned=$((pinned + 1))
        elif [ "$f" = "$server_path" ]; then
            want=$server_pairs; label=${server_ent##*/}; pinned=$((pinned + 1)); server_seen=1
        else
            want=""; label=none
        fi
        have=$(codesign -d --entitlements - --xml "$f" 2>/dev/null | ent_pairs)
        [ "$have" = "$want" ] || {
            echo "verify-bundle.sh: entitlements differ from $label: $rel" >&2
            exit 1
        }
        rt_count=$((rt_count + 1))
    done <<EOF
$(find "$contents" -type f | LC_ALL=C sort)
EOF
fi
if [ -d "$contents/Resources/python" ] && [ $server_seen = 0 ]; then
    echo "verify-bundle.sh: server interpreter missing or not Mach-O: ${server_path#"$app_parent"/}" >&2
    exit 1
fi
echo "verify-bundle.sh: hardened runtime on $rt_count Mach-O, entitlements as committed ($pinned pinned)"

# Sparkle.framework's nested code (see the header), before the whole-tree check.
links_sparkle=0
if [ -n "$main_exe" ] && [ -f "$main_exe" ] \
    && otool -L "$main_exe" 2>/dev/null | grep -q '@rpath/Sparkle.framework/'; then
    links_sparkle=1
fi
fw="$contents/Frameworks/Sparkle.framework"
if [ $require_sparkle = 1 ] || [ $links_sparkle = 1 ] || [ -e "$fw" ]; then
    if [ $require_sparkle = 1 ] && [ $links_sparkle = 0 ]; then
        echo "verify-bundle.sh: main executable does not link @rpath/Sparkle.framework: ${main_exe#"$app_parent"/}" >&2
        exit 1
    fi
    if [ $links_sparkle = 1 ]; then
        otool -l "$main_exe" | awk '/cmd LC_RPATH/{r=1;next} r&&/ path /{print $2; r=0}' \
            | grep -qx '@executable_path/../Frameworks' || {
            echo "verify-bundle.sh: no LC_RPATH @executable_path/../Frameworks in ${main_exe#"$app_parent"/}" >&2
            exit 1
        }
    fi
    sparkle_count=0
    for item in "$fw/Versions/B/XPCServices/Downloader.xpc" \
                "$fw/Versions/B/XPCServices/Installer.xpc" \
                "$fw/Versions/B/Updater.app" \
                "$fw/Versions/B/Autoupdate" \
                "$fw"; do
        rel=${item#"$app_parent"/}
        [ -e "$item" ] || {
            echo "verify-bundle.sh: missing: $rel" >&2
            exit 1
        }
        codesign --verify --strict "$item" || {
            echo "verify-bundle.sh: not validly signed: $rel" >&2
            exit 1
        }
        sparkle_count=$((sparkle_count + 1))
    done
    echo "verify-bundle.sh: sparkle $sparkle_count nested items signed"
fi

# The bundle itself, last: --deep walks the nested code so the top-level
# Mach-Os and the whole tree are checked as one.
codesign --verify --strict --deep "$app_abs" || {
    echo "verify-bundle.sh: not validly signed: $app_name" >&2
    exit 1
}

echo "verify-bundle.sh: verified $count Mach-O + bundle"
