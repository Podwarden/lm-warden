#!/bin/bash
# bundle-webui.sh — put the web UI static export in "LM Warden.app" (P8, slice 4).
#
# Usage:
#   scripts/macos/bundle-webui.sh <app> [--work-dir <dir>]
#   scripts/macos/bundle-webui.sh <app> --export-dir <dir>
#   scripts/macos/bundle-webui.sh <app> --print-plan [--work-dir <dir>] [--export-dir <dir>]
#
# Lays the existing Next.js UI, built as a static export (LMW_UI_EXPORT=1,
# frontend/next.config.ts), into <app>/Contents/Resources/webui -- the path
# ShellAppDelegate.bundledWebUIDir() hands the server as VW_WEBUI_DIR, which
# app/main.py mounts at /ui (app/webui/mount.py). The UI is the same one the
# Linux product ships; nothing is redesigned for the window.
#
#   1. Refuses (exit 1), before anything is built, when `git status` shows
#      any uncommitted change under frontend/ (modified, staged, deleted or
#      untracked; ignored files such as node_modules/ .next/ out/ do not
#      count) -- the same owner rule bundle-python.sh applies to app/.
#   2. Writes frontend/ from `git archive HEAD frontend` into a fresh build
#      dir under <work-dir> (default .cache/macos-webui), so the export is
#      built from the committed tree only: no ignored .env.local, stale .next
#      cache or out/ of the working tree can reach it. Outside a git checkout
#      (a source tarball) frontend/ is copied as-is, without its ignored dirs.
#   3. Links the checkout's frontend/node_modules into it (it must exist: run
#      `npm ci` in frontend/, or scripts/farm/frontend-deps.sh on a farm
#      tree; this script never runs npm and never needs the registry token)
#      and runs `next build` with LMW_UI_EXPORT=1 and telemetry off.
#   4. Checks the export with verify-export-layout.sh (what the /ui mount
#      serves), refuses a symlink in it (a link in the signed bundle could
#      point outside it) and any file naming the build dir, the checkout or
#      the app's absolute path.
#   5. Replaces <app>/Contents/Resources/webui with it (staged next to it,
#      then moved), clears extended attributes, and checks index.html.
#
# --export-dir <dir> skips 1-3 and bundles an export built elsewhere (CI's
# Linux lane, a developer's frontend/out); steps 4-5 still apply. Its
# provenance is the caller's: nothing ties it to HEAD.
#
# It does NOT sign: `make macos-bundle-python` runs it after bundle-python.sh
# and before sign-bundle.sh, whose final bundle signature seals Resources/
# (the export holds no Mach-O), and verify-bundle.sh.
#
# --print-plan prints the resolved plan as key=value lines and touches nothing.
#
# Exit codes: 0 ok, 1 a step failed (dirty frontend/, no node_modules, build,
# layout, symlink, leaked path), 2 usage.
#
# Bash-3.2-safe (the M1 runner's /bin/bash is 3.2): no associative arrays, no
# ${x,,}, no mapfile.
set -eu

USAGE="usage: bundle-webui.sh <app> [--work-dir <dir>] [--export-dir <dir>] [--print-plan]"

die() { echo "bundle-webui.sh: $*" >&2; exit 1; }
usage() { echo "$USAGE" >&2; exit 2; }

ROOT=$(cd "$(dirname "$0")/../.." && pwd)

in_git() { git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; }

# require_clean_frontend: exit 1 if frontend/ differs from HEAD in any way git
# reports (modified, staged, deleted, untracked; not ignored).
require_clean_frontend() {
    in_git || return 0
    dirty=$(git -C "$ROOT" status --porcelain --untracked-files=all -- frontend) \
        || die "git status failed in $ROOT"
    [ -n "$dirty" ] || return 0
    {
        echo "bundle-webui.sh: uncommitted changes under frontend/ -- the bundle ships the UI built from HEAD;"
        echo "commit or discard them first:"
        printf '%s\n' "$dirty" | head -n 20 | sed 's/^/  /'
    } >&2
    exit 1
}

frontend_source() {
    if in_git; then
        echo "git archive $(git -C "$ROOT" rev-parse HEAD) frontend"
    else
        echo "tree $ROOT/frontend"
    fi
}

app=""
work_root="$ROOT/.cache/macos-webui"
export_dir=""
plan=0
while [ $# -gt 0 ]; do
    case $1 in
        --work-dir) [ $# -ge 2 ] && [ -n "$2" ] || usage; work_root=$2; shift 2 ;;
        --export-dir) [ $# -ge 2 ] && [ -n "$2" ] || usage; export_dir=$2; shift 2 ;;
        --print-plan) plan=1; shift ;;
        -h|--help) echo "$USAGE"; exit 0 ;;
        -*) usage ;;
        *) [ -z "$app" ] || usage; app=$1; shift ;;
    esac
done
[ -n "$app" ] || usage
if [ ! -d "$app/Contents" ]; then
    echo "bundle-webui.sh: not an app bundle (no Contents/): $app" >&2
    usage
fi
if [ -n "$export_dir" ] && [ ! -d "$export_dir" ]; then
    echo "bundle-webui.sh: --export-dir is not a directory: $export_dir" >&2
    exit 2
fi

app_abs=$(cd "$app" && pwd)
case $work_root in /*) ;; *) work_root=$(pwd)/$work_root ;; esac
dest="$app_abs/Contents/Resources/webui"

if [ $plan = 1 ]; then
    echo "dest=$dest"
    echo "env=VW_WEBUI_DIR=$dest"
    if [ -n "$export_dir" ]; then
        echo "source=export-dir $(cd "$export_dir" && pwd)"
    else
        echo "source=$(frontend_source)"
        echo "work_dir=$work_root"
        echo "node_modules=$ROOT/frontend/node_modules"
        echo "build=LMW_UI_EXPORT=1 NEXT_TELEMETRY_DISABLED=1 next build"
    fi
    echo "check=scripts/macos/verify-export-layout.sh"
    exit 0
fi

build=""
cleanup() { [ -z "$build" ] || rm -rf "$build"; }
trap cleanup EXIT

if [ -n "$export_dir" ]; then
    out=$(cd "$export_dir" && pwd)
    echo "==> bundle the prebuilt export $out"
else
    # --- 1. the dirty-tree gate, before anything is built --------------------
    require_clean_frontend
    nm="$ROOT/frontend/node_modules"
    [ -d "$nm" ] || die "no $nm: run \`npm ci\` in frontend/ (or scripts/farm/frontend-deps.sh <tree>) first"
    [ -x "$nm/.bin/next" ] || die "$nm has no .bin/next: re-run \`npm ci\` in frontend/"
    command -v node >/dev/null 2>&1 || die "node not found on PATH"

    # --- 2. frontend/ from HEAD ------------------------------------------------
    mkdir -p "$work_root"
    work_root=$(cd "$work_root" && pwd)
    build=$(mktemp -d "$work_root/build.XXXXXX")
    if in_git; then
        rev=$(git -C "$ROOT" rev-parse HEAD)
        echo "==> git archive $rev frontend"
        # Through a file, not a pipe: without pipefail a failed archive would
        # hand tar an empty stream and pass.
        tarf="$build/frontend.tar"
        git -C "$ROOT" archive --format=tar -o "$tarf" "$rev" frontend \
            || die "git archive $rev frontend failed"
        tar -xf "$tarf" -C "$build" || die "extracting frontend/ failed"
        rm -f "$tarf"
    else
        echo "==> copy frontend/ (not a git checkout)"
        (cd "$ROOT" && tar --exclude node_modules --exclude .next --exclude out -cf - frontend) \
            | tar -xf - -C "$build"
    fi
    [ -f "$build/frontend/package.json" ] || die "frontend/package.json missing after copy"

    # --- 3. the static export ----------------------------------------------------
    ln -s "$nm" "$build/frontend/node_modules"
    echo "==> LMW_UI_EXPORT=1 next build"
    (
        cd "$build/frontend"
        # ANALYZE would wrap the config in the bundle analyzer: never for a ship build.
        env -u ANALYZE LMW_UI_EXPORT=1 NEXT_TELEMETRY_DISABLED=1 ./node_modules/.bin/next build
    ) || die "next build (LMW_UI_EXPORT=1) failed"
    out="$build/frontend/out"
    [ -d "$out" ] || die "next build left no out/ (is output: 'export' gated on LMW_UI_EXPORT=1?)"
fi

# --- 4. checks ----------------------------------------------------------------
bash "$ROOT/scripts/macos/verify-export-layout.sh" "$out" || die "the export fails verify-export-layout.sh"
links=$(find "$out" -type l | head -n 20)
[ -z "$links" ] || die "symlinks in the export: $(printf '%s' "$links" | tr '\n' ' ')"
leaks=""
for p in ${build:+"$build"} "$ROOT" "$app_abs"; do
    hits=$(grep -rlF -- "$p" "$out" 2>/dev/null | head -n 20 || true)
    [ -z "$hits" ] || leaks="$leaks$hits
"
done
if [ -n "$leaks" ]; then
    echo "bundle-webui.sh: build-host paths in the export:" >&2
    printf '%s' "$leaks" | LC_ALL=C sort -u | sed "s|^$out/|  |" >&2
    exit 1
fi

# --- 5. into the bundle -------------------------------------------------------
mkdir -p "$app_abs/Contents/Resources"
stage="$app_abs/Contents/Resources/.webui.$$"
rm -rf "$stage"
cp -R "$out" "$stage" || { rm -rf "$stage"; die "copying the export failed"; }
if command -v xattr >/dev/null 2>&1; then
    xattr -cr "$stage"
fi
rm -rf "$dest"
mv "$stage" "$dest"
[ -f "$dest/index.html" ] || die "no index.html in $dest"
n=$(find "$dest" -type f | wc -l | tr -d ' ')
echo "bundle-webui.sh: web UI export ($n files) at ${dest#"$(dirname "$app_abs")"/}"
