#!/bin/bash
# bundle-python.sh — embed the server's Python in "LM Warden.app" (P8, first slice).
#
# Usage:
#   scripts/macos/bundle-python.sh <app> [--cache-dir <dir>] [--requirements <file>] [--offline]
#   scripts/macos/bundle-python.sh <app> --print-plan [--cache-dir <dir>] [--requirements <file>]
#   scripts/macos/bundle-python.sh --relocate-scripts <bin_dir> <old_prefix>     (test hook)
#   scripts/macos/bundle-python.sh --copy-app <dest_dir>                          (test hook)
#   scripts/macos/bundle-python.sh --rename-interpreter <bin_dir>                 (test hook)
#
# Lays a python-build-standalone CPython into <app>/Contents/Resources/python
# (the path ShellAppDelegate.bundledPythonPath() launches:
# Contents/Resources/python/bin/LM Warden Server) with the server's dependencies and the
# `app` package installed into its own site-packages (Metal design §3):
#
#   1. Fetches the PINNED interpreter (PBS_URL below, one exact release asset)
#      into <cache-dir> and checks it against PBS_SHA256. A cached tarball is
#      re-checked every run; a mismatch is never used (re-downloaded, or with
#      --offline a hard failure). Nothing unpinned is ever downloaded.
#   2. Extracts it into a staging dir under <cache-dir>.
#   3. `pip install --isolated --require-hashes --only-binary :all: --no-deps
#      -r <requirements>` (default requirements-macos.txt, the hash lock) into
#      that interpreter. Wheels only: nothing is compiled at install time.
#   4. Writes app/ into site-packages/app from `git archive HEAD app` -- the
#      committed tree, never the working tree. Before anything else (before the
#      download, even) the run refuses (exit 1) when `git status` shows any
#      uncommitted change under app/: modified, staged, deleted or untracked
#      (ignored files such as __pycache__ do not count), so a dirty checkout
#      can never ship unreviewed edits. Outside a git checkout (a source
#      tarball) app/ is copied as-is. pyproject.toml has no [build-system];
#      `pip install .` would fetch an unpinned setuptools into a
#      build-isolation env, which the hash lock forbids.
#   5. Relocatability: rewrites the console scripts pip wrote in bin/ (their
#      shebang names the staging interpreter) to a /bin/sh header that execs
#      the renamed interpreter next to the script with -I -B; uninstalls pip
#      (never `pip install` at runtime, §3); byte-compiles site-packages with
#      --invalidation-mode unchecked-hash and the staging prefix stripped, so the
#      .pyc never names a build path and are never rewritten inside the signed
#      bundle (the build interpreter itself always runs -B, and the stdlib is
#      compiled the same way); clears extended attributes (codesign refuses Finder detritus).
#   5b. Renames the interpreter (Metal design s3): bin/python3.12 becomes
#      bin/"LM Warden Server" -- the Mach-O itself, not a symlink and not a
#      second 18 MB copy, so it carries its own signature and Activity Monitor,
#      the Background Items list, the firewall dialog and `ps` name the
#      product, not "python3". The name is 16 bytes, exactly MAXCOMLEN, so the
#      kernel's p_comm keeps all of it. It stays in python/bin (not
#      Contents/MacOS): CPython finds its prefix from the executable's own dir
#      (../lib/python3.12/os.py) and -I ignores PYTHONHOME, so outside the tree
#      the stdlib is not found; and the mlx engine PATH is
#      dirname(sys.executable), which must stay the console-script dir. The
#      bin/python and bin/python3 symlinks and PBS's stub scripts (2to3, idle3,
#      pydoc3, python3-config, which exec bin/python3.12) are removed, so
#      nothing in the bundle starts an interpreter named "python"; the run
#      refuses (exit 1) if bin/ still holds a dangling link or a script that
#      execs python3.12.
#   6. Refuses (exit 1) if any file under the tree still contains the staging
#      dir, the cache dir, the repo root or the app's absolute path.
#   7. Moves the tree into <app>/Contents/Resources/python (replacing an old
#      one) and imports `ssl, sqlite3, app` with the moved, renamed
#      interpreter (-I -B).
#
# It does NOT sign: run scripts/macos/sign-bundle.sh <app> afterwards (it finds
# every embedded .so/.dylib/python by `file`, inside-out), then
# verify-bundle.sh. `make macos-bundle-python` runs all three.
#
# --print-plan prints the resolved plan as key=value lines and touches nothing.
# --copy-app runs step 4 alone (dirty-tree gate + git archive) into <dest_dir>.
# --rename-interpreter runs step 5b alone on <bin_dir>.
# --relocate-scripts rewrites, in <bin_dir>, every non-Mach-O file whose first
# two lines name <old_prefix> to the relocatable header (step 5 alone).
#
# Exit codes: 0 ok, 1 a step failed (dirty app/, checksum, download, pip,
# leaked path, rename, import), 2 usage.
#
# Bash-3.2-safe (the M1 runner's /bin/bash is 3.2): indexed arrays only, no
# associative arrays, no ${x,,}, no mapfile.
set -eu

# --- the pin --------------------------------------------------------------
# python-build-standalone release 20261003, CPython 3.12.15, install_only
# (not _stripped: keeps symbols for crash reports). sha256 is the GitHub
# release asset digest. Bumping it is a deliberate, reviewed edit.
PBS_RELEASE="20261003"
PBS_PYVER="3.12.15"
PBS_ASSET="cpython-${PBS_PYVER}+${PBS_RELEASE}-aarch64-apple-darwin-install_only.tar.gz"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/cpython-${PBS_PYVER}%2B${PBS_RELEASE}-aarch64-apple-darwin-install_only.tar.gz"
PBS_SHA256="316a463172740e71d8dca1f2730784e325f3f720941137b5d674d5801a632213"
PY_MINOR="3.12"
# The embedded interpreter's file (and so process) name. Must equal
# ShellAppDelegate.bundledInterpreterName (the product is "LM Warden").
SERVER_EXE="LM Warden Server"

USAGE="usage: bundle-python.sh <app> [--cache-dir <dir>] [--requirements <file>] [--offline] [--print-plan]"

die() { echo "bundle-python.sh: $*" >&2; exit 1; }
usage() { echo "$USAGE" >&2; exit 2; }

ROOT=$(cd "$(dirname "$0")/../.." && pwd)

# The relocatable console-script header. sh runs line 2 (`''` + `'exec'` ...);
# Python sees a comment and a string literal, then the original script body.
# It execs the renamed interpreter next to the script with -I (isolated) and
# -B (never write a .pyc into the signed bundle).
RELOC_HEADER_1='#!/bin/sh'
RELOC_HEADER_2="'''exec' \"\$(dirname -- \"\$(realpath -- \"\$0\")\")\"/'$SERVER_EXE' -I -B \"\$0\" \"\$@\""
RELOC_HEADER_3="' '''"

is_macho() {
    command -v file >/dev/null 2>&1 || return 1
    file -b "$1" | grep -q "Mach-O"
}

# relocate_scripts <bin_dir> <old_prefix>: rewrite pip's console scripts.
# pip (distlib) writes either `#!<prefix>/bin/python3.12` or, for a long or
# spaced path, a 3-line `#!/bin/sh` / `'''exec' "<prefix>/..."` / `' '''`
# header. Both are replaced by the RELOC header; the body is kept verbatim.
relocate_scripts() {
    bin_dir=$1
    old=$2
    n=0
    for f in "$bin_dir"/*; do
        [ -f "$f" ] && [ ! -L "$f" ] || continue
        is_macho "$f" && continue
        head -n 2 "$f" | grep -qF -- "$old" || continue
        tmp="$f.reloc.$$"
        {
            printf '%s\n%s\n%s\n' "$RELOC_HEADER_1" "$RELOC_HEADER_2" "$RELOC_HEADER_3"
            awk -v q="'" -v q3="'''" '
                NR == 1 { if ($0 ~ /^#!\/bin\/sh/) sh = 1; next }
                sh && NR == 2 && index($0, q3 "exec" q) == 1 { next }
                sh && NR == 3 && $0 == q " " q3 { next }
                { print }
            ' "$f"
        } >"$tmp"
        chmod 755 "$tmp"
        mv "$tmp" "$f"
        echo "relocated: ${f##*/}"
        n=$((n + 1))
    done
    echo "bundle-python.sh: relocated $n scripts"
}

# rename_interpreter <bin_dir>: step 5b. bin/python3.12 -> bin/$SERVER_EXE
# (a rename of the real file), then drop every other way to start "python".
rename_interpreter() {
    bin_dir=$1
    src="$bin_dir/python$PY_MINOR"
    [ -f "$src" ] && [ ! -L "$src" ] || die "no interpreter to rename: $src"
    [ ! -e "$bin_dir/$SERVER_EXE" ] || die "already exists: $bin_dir/$SERVER_EXE"
    mv "$src" "$bin_dir/$SERVER_EXE"
    for f in python python3 python3-config "python$PY_MINOR-config" \
        2to3 "2to3-$PY_MINOR" idle3 "idle$PY_MINOR" pydoc3 "pydoc$PY_MINOR"; do
        rm -f "$bin_dir/$f"
    done
    for f in "$bin_dir"/*; do
        if [ -L "$f" ] && [ ! -e "$f" ]; then
            die "dangling link left in bin/: ${f##*/}"
        fi
        [ -f "$f" ] || continue
        is_macho "$f" && continue
        if head -n 2 "$f" | grep -qF "python$PY_MINOR"; then
            die "bin/${f##*/} still execs python$PY_MINOR"
        fi
    done
    echo "bundle-python.sh: interpreter renamed to bin/$SERVER_EXE"
}

in_git() { git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; }

# require_clean_app: exit 1 if app/ differs from HEAD in any way git reports
# (modified, staged, deleted, untracked; not ignored).
require_clean_app() {
    in_git || return 0
    dirty=$(git -C "$ROOT" status --porcelain --untracked-files=all -- app) \
        || die "git status failed in $ROOT"
    [ -n "$dirty" ] || return 0
    {
        echo "bundle-python.sh: uncommitted changes under app/ -- the bundle ships app/ from HEAD;"
        echo "commit or discard them first:"
        printf '%s\n' "$dirty" | head -n 20 | sed 's/^/  /'
    } >&2
    exit 1
}

# copy_app <dest>: app/ exactly as committed at HEAD into <dest>/app.
copy_app() {
    require_clean_app
    if in_git; then
        rev=$(git -C "$ROOT" rev-parse HEAD)
        echo "==> git archive $rev app -> ${1##*/}"
        # Through a file, not a pipe: without pipefail a failed archive would
        # hand tar an empty stream and pass.
        tarf=$(mktemp "${TMPDIR:-/tmp}/bundle-app.XXXXXX")
        git -C "$ROOT" archive --format=tar -o "$tarf" "$rev" app || {
            rm -f "$tarf"
            die "git archive $rev app failed"
        }
        tar -xf "$tarf" -C "$1" || { rm -f "$tarf"; die "extracting app/ failed"; }
        rm -f "$tarf"
    else
        echo "==> copy app/ (not a git checkout)"
        (cd "$ROOT" && tar --exclude __pycache__ --exclude '*.pyc' -cf - app) | tar -xf - -C "$1"
    fi
}

app_source() {
    if in_git; then
        echo "git archive $(git -C "$ROOT" rev-parse HEAD) app"
    else
        echo "tree $ROOT/app"
    fi
}

sha256_of() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        sha256sum "$1" | awk '{print $1}'
    fi
}

# --- arguments ------------------------------------------------------------
if [ "${1:-}" = "--relocate-scripts" ]; then
    [ $# -eq 3 ] && [ -d "$2" ] && [ -n "$3" ] || {
        echo "usage: bundle-python.sh --relocate-scripts <bin_dir> <old_prefix>" >&2
        exit 2
    }
    relocate_scripts "$2" "$3"
    exit 0
fi
if [ "${1:-}" = "--rename-interpreter" ]; then
    [ $# -eq 2 ] && [ -d "$2" ] || {
        echo "usage: bundle-python.sh --rename-interpreter <bin_dir>" >&2
        exit 2
    }
    rename_interpreter "$2"
    exit 0
fi
if [ "${1:-}" = "--copy-app" ]; then
    [ $# -eq 2 ] && [ -d "$2" ] || {
        echo "usage: bundle-python.sh --copy-app <dest_dir>" >&2
        exit 2
    }
    copy_app "$2"
    exit 0
fi

app=""
cache_dir="$ROOT/.cache/macos-python"
req="$ROOT/requirements-macos.txt"
offline=0
plan=0
while [ $# -gt 0 ]; do
    case $1 in
        --cache-dir) [ $# -ge 2 ] && [ -n "$2" ] || usage; cache_dir=$2; shift 2 ;;
        --requirements) [ $# -ge 2 ] && [ -n "$2" ] || usage; req=$2; shift 2 ;;
        --offline) offline=1; shift ;;
        --print-plan) plan=1; shift ;;
        -h|--help) echo "$USAGE"; exit 0 ;;
        -*) usage ;;
        *) [ -z "$app" ] || usage; app=$1; shift ;;
    esac
done
[ -n "$app" ] || usage
if [ ! -d "$app/Contents" ]; then
    echo "bundle-python.sh: not an app bundle (no Contents/): $app" >&2
    usage
fi
[ -f "$req" ] || { echo "bundle-python.sh: requirements file not found: $req" >&2; exit 2; }

app_abs=$(cd "$app" && pwd)
req_abs=$(cd "$(dirname "$req")" && pwd)/${req##*/}
case $cache_dir in /*) ;; *) cache_dir=$(pwd)/$cache_dir ;; esac
dest="$app_abs/Contents/Resources/python"
tarball="$cache_dir/$PBS_ASSET"

PIP_ARGS=(install --isolated --no-input --disable-pip-version-check
    --require-hashes --only-binary :all: --no-deps
    --no-compile --no-warn-script-location
    --cache-dir "$cache_dir/pip" -r "$req_abs")

if [ $plan = 1 ]; then
    echo "url=$PBS_URL"
    echo "sha256=$PBS_SHA256"
    echo "tarball=$tarball"
    echo "dest=$dest"
    echo "python=$dest/bin/$SERVER_EXE"
    echo "interpreter_name=$SERVER_EXE"
    echo "site_packages=$dest/lib/python$PY_MINOR/site-packages"
    echo "requirements=$req_abs"
    echo "pip=python3 -I -m pip ${PIP_ARGS[*]}"
    echo "app_source=$(app_source)"
    exit 0
fi

# Refuse a dirty app/ before anything is downloaded or staged.
require_clean_app

mkdir -p "$cache_dir"
cache_dir=$(cd "$cache_dir" && pwd)
tarball="$cache_dir/$PBS_ASSET"

# --- 1. the pinned interpreter ----------------------------------------------
if [ -f "$tarball" ]; then
    got=$(sha256_of "$tarball")
    if [ "$got" != "$PBS_SHA256" ]; then
        if [ $offline = 1 ]; then
            die "sha256 mismatch for cached $tarball: expected $PBS_SHA256, got $got"
        fi
        echo "bundle-python.sh: cached tarball fails its checksum, re-downloading" >&2
        rm -f "$tarball"
    fi
fi
if [ ! -f "$tarball" ]; then
    [ $offline = 0 ] || die "--offline and $PBS_ASSET is not in $cache_dir"
    echo "==> fetch $PBS_URL"
    part="$tarball.part.$$"
    curl -fsSL --proto '=https' --tlsv1.2 --retry 3 -o "$part" "$PBS_URL" || {
        rm -f "$part"
        die "download failed: $PBS_URL"
    }
    got=$(sha256_of "$part")
    if [ "$got" != "$PBS_SHA256" ]; then
        rm -f "$part"
        die "sha256 mismatch for $PBS_URL: expected $PBS_SHA256, got $got"
    fi
    mv "$part" "$tarball"
fi
echo "==> $PBS_ASSET sha256 ok"

# --- 2. stage ---------------------------------------------------------------
stage=$(mktemp -d "$cache_dir/stage.XXXXXX")
trap 'rm -rf "$stage"' EXIT
tar -xzf "$tarball" -C "$stage"
spy="$stage/python"
[ -x "$spy/bin/python$PY_MINOR" ] || die "tarball has no python/bin/python$PY_MINOR"
site="$spy/lib/python$PY_MINOR/site-packages"

# --- 3. hash-locked deps ----------------------------------------------------
echo "==> pip install -r ${req_abs##*/} (--require-hashes --only-binary :all: --no-deps)"
"$spy/bin/python$PY_MINOR" -I -B -m pip "${PIP_ARGS[@]}" || die "pip install failed"

# --- 4. the app package -----------------------------------------------------
copy_app "$site"
[ -f "$site/app/main.py" ] || die "app/main.py missing after copy"

# --- 5. relocatability --------------------------------------------------------
relocate_scripts "$spy/bin" "$stage"
"$spy/bin/python$PY_MINOR" -I -B -m pip uninstall --isolated -y -q pip || die "pip uninstall failed"
# The build interpreter runs with -B throughout, so nothing it imports leaves a
# .pyc naming the staging dir. compileall then writes every .pyc on purpose:
# site-packages must compile cleanly; the stdlib is best-effort because it
# ships deliberately broken test inputs (lib2to3 / test badsyntax files).
echo "==> compileall stdlib + site-packages (unchecked-hash, prefix stripped)"
"$spy/bin/python$PY_MINOR" -I -B -m compileall -q -j 0 \
    --invalidation-mode unchecked-hash -s "$spy" -x '/site-packages/' \
    "$spy/lib/python$PY_MINOR" >/dev/null 2>&1 || true
"$spy/bin/python$PY_MINOR" -I -B -m compileall -q -j 0 \
    --invalidation-mode unchecked-hash -s "$spy" "$site" >/dev/null \
    || die "compileall failed"
if command -v xattr >/dev/null 2>&1; then
    xattr -cr "$spy"
fi

# --- 5b. the product-named interpreter ---------------------------------------
rename_interpreter "$spy/bin"

# --- 6. no build-host paths ---------------------------------------------------
leaks=""
for p in "$stage" "$cache_dir" "$ROOT" "$app_abs"; do
    hits=$(grep -rlF -- "$p" "$spy" 2>/dev/null | head -n 20 || true)
    [ -z "$hits" ] || leaks="$leaks$hits
"
done
if [ -n "$leaks" ]; then
    echo "bundle-python.sh: build-host paths left in the embedded tree:" >&2
    printf '%s' "$leaks" | LC_ALL=C sort -u | sed "s|^$stage/|  |" >&2
    exit 1
fi
echo "==> no build-host paths in the tree"

# --- 7. install into the bundle ---------------------------------------------
mkdir -p "$app_abs/Contents/Resources"
rm -rf "$dest"
mv "$spy" "$dest"
"$dest/bin/$SERVER_EXE" -I -B -c 'import ssl, sqlite3, app' || die "import check failed: $dest/bin/$SERVER_EXE"
echo "bundle-python.sh: embedded python $PBS_PYVER at ${dest#"$(dirname "$app_abs")"/}"
