#!/bin/bash
# sign-bundle.sh — sign a .app inside-out, deepest first (Task 8).
#
# Usage:
#   scripts/macos/sign-bundle.sh <app> [identity] [--entitlements-dir <dir>]
#
# <identity> defaults to "-" (ad-hoc). Every item is signed with the hardened
# runtime (`--options runtime`), ad-hoc included, so a dev build runs under
# the same rules as a release; a real (Developer ID) identity adds
# `--timestamp` (ad-hoc cannot carry a secure timestamp).
#
# Entitlements (P8 hardened-runtime), by role, from <dir> (default: the
# repo's macos/entitlements, next to this script's ../../macos):
#   <app> itself (so its main executable)       LMWarden.entitlements
#   Contents/Resources/python/bin/LM Warden Server  LMWardenServer.entitlements
#   everything else (dylibs, .so, llama-server, Sparkle's XPCs, Updater.app,
#   Autoupdate, the framework)                  none
# Both files must exist in <dir> (exit 2 otherwise, before any signing). The
# committed set is the minimum found empirically on the M1 under an ad-hoc
# `-o runtime` signature: com.apple.security.cs.disable-library-validation
# alone, on both processes -- without it the interpreter's dlopen of a bundled
# .so (pydantic_core) and the shell's load of Sparkle.framework are refused
# ("mapping process and mapped file (non-platform) have different Team IDs").
# No allow-jit / allow-unsigned-executable-memory: MLX on the GPU, ctypes and
# cffi callbacks and llama-server (Metal) all run without them. verify-bundle.sh
# checks the result against the same files.
#
# Instead of `codesign --deep` (deprecated) this signs every nested Mach-O
# before the things that embed it, so each signature is over its final
# contents. Mach-O files are found by `file -b` output containing "Mach-O"
# (not by extension: an extensionless binary is found, a text file named
# x.so is not), and nested bundle directories (*.app, *.framework,
# *.bundle, *.xpc) are signed as units after their contents.
#
# Sparkle.framework (Contents/Frameworks, placed by embed-sparkle.sh) needs no
# special case: its symlinks (Sparkle.framework/Updater.app -> Versions/
# Current/...) are neither `-type f` nor `-type d`, so only the real
# Versions/B paths are signed, in the order Sparkle's own docs give --
# XPCServices/*.xpc (after their executables), Autoupdate, Updater.app, then
# the framework -- and the upstream entitlements are empty (Autoupdate's
# application-identifier is dropped by the re-sign, which a Developer ID
# build wants: there is no provisioning profile to back it).
#
# Every item is ordered by path depth (number of '/' components) descending,
# ties by path ascending (LC_ALL=C sort), and <app> itself is signed last.
# One line "signed: <path relative to the app's parent dir>" is printed per
# signing, in signing order, then a final "sign-bundle.sh: signed <n> items".
#
# Exit codes: 0 ok, 1 a codesign call failed (the path is named),
# 2 <app> missing or not a directory, a bad option (usage line), or an
# entitlements file missing.
#
# Bash-3.2-safe (the M1 runner's /bin/bash is 3.2): no associative arrays, no
# ${x,,}, no mapfile. Paths may contain spaces, so everything is quoted and
# find output is read with `while IFS= read -r`.
set -eu

usage="usage: sign-bundle.sh <app> [identity] [--entitlements-dir <dir>]"
script_dir=$(cd "$(dirname "$0")" && pwd)
ent_dir="$script_dir/../../macos/entitlements"
app=""
identity=""
npos=0
while [ $# -gt 0 ]; do
    case "$1" in
        --entitlements-dir)
            [ $# -ge 2 ] && [ -n "$2" ] || { echo "$usage" >&2; exit 2; }
            ent_dir=$2
            shift 2
            ;;
        --*) echo "$usage" >&2; exit 2 ;;
        *)
            npos=$((npos + 1))
            case $npos in
                1) app=$1 ;;
                2) identity=$1 ;;
                *) echo "$usage" >&2; exit 2 ;;
            esac
            shift
            ;;
    esac
done
if [ -z "$app" ] || [ ! -d "$app" ]; then
    echo "$usage" >&2
    exit 2
fi
identity=${identity:--}
[ -d "$ent_dir" ] && ent_dir=$(cd "$ent_dir" && pwd -P)
shell_ent="$ent_dir/LMWarden.entitlements"
server_ent="$ent_dir/LMWardenServer.entitlements"
for ent in "$shell_ent" "$server_ent"; do
    [ -f "$ent" ] || {
        echo "sign-bundle.sh: missing entitlements file: $ent" >&2
        exit 2
    }
done

app_abs=$(cd "$app" && pwd)
app_parent=$(cd "$app_abs/.." && pwd)
app_name=${app_abs##*/}
contents="$app_abs/Contents"

# Build the sign list. Each entry is "<depth>\t<path>": depth is the number
# of '/' in the path (deeper = larger), so `LC_ALL=C sort -t <tab> -k1,1nr -k2,2`
# orders deepest-first with ties broken by path ascending.
sign_list=""

# Mach-O files: regular files under Contents whose `file -b` output contains
# "Mach-O". The extension is irrelevant.
if [ -d "$contents" ]; then
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        [ -f "$f" ] || continue
        if file -b "$f" | grep -q "Mach-O"; then
            depth=${f//[!\/]/}
            sign_list="${sign_list}${#depth}	${f}
"
        fi
    done <<EOF
$(find "$contents" -type f)
EOF
fi

# Nested bundle directories under Contents.
if [ -d "$contents" ]; then
    while IFS= read -r d; do
        [ -n "$d" ] || continue
        [ -d "$d" ] || continue
        depth=${d//[!\/]/}
        sign_list="${sign_list}${#depth}	${d}
"
    done <<EOF
$(find "$contents" -type d \( -name '*.app' -o -name '*.framework' -o -name '*.bundle' -o -name '*.xpc' \))
EOF
fi

# Hardened runtime always; a secure timestamp only for a real identity.
sign_opts="--options runtime"
if [ "$identity" != "-" ]; then
    sign_opts="--options runtime --timestamp"
fi
server_path="$contents/Resources/python/bin/LM Warden Server"

count=0
while IFS="$(printf '\t')" read -r _depth path; do
    [ -n "$path" ] || continue
    if [ "$path" = "$server_path" ]; then
        # shellcheck disable=SC2086
        set -- $sign_opts --entitlements "$server_ent"
    else
        # shellcheck disable=SC2086
        set -- $sign_opts
    fi
    codesign --force --sign "$identity" "$@" "$path" || {
        echo "sign-bundle.sh: codesign failed: $path" >&2
        exit 1
    }
    rel=${path#"$app_parent"/}
    echo "signed: $rel"
    count=$((count + 1))
done <<EOF
$(printf '%s' "$sign_list" | LC_ALL=C sort -t "$(printf '\t')" -k1,1nr -k2,2)
EOF

# The app itself, always last.
# shellcheck disable=SC2086
codesign --force --sign "$identity" $sign_opts --entitlements "$shell_ent" "$app_abs" || {
    echo "sign-bundle.sh: codesign failed: $app_abs" >&2
    exit 1
}
echo "signed: $app_name"
count=$((count + 1))

echo "sign-bundle.sh: signed $count items"
