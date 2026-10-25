#!/bin/bash
# bundle-python.sh — embed the server's Python in "LM Warden.app" (P8, first slice).
#
# Usage:
#   scripts/macos/bundle-python.sh <app> [--cache-dir <dir>] [--requirements <file>] [--offline]
#   scripts/macos/bundle-python.sh <app> --print-plan [--cache-dir <dir>] [--requirements <file>]
#   scripts/macos/bundle-python.sh --relocate-scripts <bin_dir> <old_prefix>     (test hook)
#   scripts/macos/bundle-python.sh --copy-app <dest_dir>                          (test hook)
#   scripts/macos/bundle-python.sh --rename-interpreter <bin_dir>                 (test hook)
#   scripts/macos/bundle-python.sh --prune <python_root> [--list <file>]          (test hook)
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
#   5a. Prunes the staged tree by the explicit, tested exclude list
#      scripts/macos/python-prune.txt (after the pip uninstall, before
#      compileall): one rule per line -- `path <glob>` removes every glob
#      match, `tree <dir> <name-glob>` removes every file or dir named
#      <name-glob> anywhere under <dir>. The list's rules are validated
#      before anything is deleted and the run refuses (exit 1) on a rule
#      that matches nothing (a stale list -- re-running the prune over an
#      already-pruned tree fails loudly, so a re-run starts from a fresh
#      stage), a malformed line, or a rule that would remove or cover a
#      protected path (the bin/ interpreter and llama-server pair,
#      libpython3.12.dylib, site-packages/app, the mlx and mlx_lm trees,
#      every transformers family's __init__/configuration_*/tokenization_*
#      and transformers' models/auto/ -- the modules bundle-import-guard.py
#      must be able to import).
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
#      one), imports `ssl, sqlite3, app` with the moved, renamed
#      interpreter (-I -B) and then runs scripts/macos/bundle-import-guard.py
#      under it: the import net (the P11 prune is gated on it) makes the
#      build fail closed when the prune removed a module the server imports.
#
# It does NOT sign: run scripts/macos/sign-bundle.sh <app> afterwards (it finds
# every embedded .so/.dylib/python by `file`, inside-out), then
# verify-bundle.sh. `make macos-bundle-python` runs all three.
#
# --print-plan prints the resolved plan as key=value lines and touches nothing.
# --copy-app runs step 4 alone (dirty-tree gate + git archive) into <dest_dir>.
# --rename-interpreter runs step 5b alone on <bin_dir>.
# --prune runs step 5a alone on <python_root> (default list:
# scripts/macos/python-prune.txt, override with --list).
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

# --- step 5a: the prune --------------------------------------------------------
#
# The list is scripts/macos/python-prune.txt: one rule per line, `#` and blank
# lines ignored, each rule:
#   path <glob>       remove every <python_root>/<glob> match
#                     (a bash glob; `*` spans one path segment)
#   tree <dir> <name> remove every file or dir named <name> under
#                     <python_root>/<dir> (find -name, -prune)
# A rule's operands must be relative (no leading /) and hold no `..` segment.
#
# Every rule is validated before the first removal, and a refusal (exit 1)
# removes nothing:
#   - a rule that matches nothing: the list is stale. A re-run of the build
#     stages a fresh tree, so this fails loudly instead of silently no-oping.
#   - a rule whose match is, or contains, a PROTECTED path: the interpreter
#     and the llama-server pair in bin/, the libpython dylib, site-packages/app
#     and the mlx / mlx_lm trees, and every transformers family's __init__.py,
#     configuration_*.py and tokenization_*.py (the modules
#     bundle-import-guard.py imports; step 7 is the backstop that turns the
#     build red if the net ever misses one).

# rel_is_protected <relpath>: the protected-path table, one entry per case.
# The `*/` branches cover a match that IS a protected dir (its children are
# all protected); the bare-name branches cover a match that IS a protected
# file.
rel_is_protected() {
    case $1 in
        bin | "bin/$SERVER_EXE" | bin/llama-server | bin/llama-server.build) return 0 ;;
        lib/libpython* | lib/libpython*/*) return 0 ;;
        lib/python*/site-packages/app | lib/python*/site-packages/app/*) return 0 ;;
        lib/python*/site-packages/mlx | lib/python*/site-packages/mlx/*) return 0 ;;
        lib/python*/site-packages/mlx_lm | lib/python*/site-packages/mlx_lm/*) return 0 ;;
        # transformers' models/auto/: every family's configuration_* and the
        # kept processing_auto import its modeling_auto / image_processing_auto
        # / video_processing_auto at module level. The tree rules skip it
        # (`! -path '*/models/auto/*'`); a rule that names it refuses.
        lib/python*/site-packages/transformers/models/auto | lib/python*/site-packages/transformers/models/auto/*) return 0 ;;
        lib/python*/site-packages/transformers/models/*/__init__.py) return 0 ;;
        lib/python*/site-packages/transformers/models/*/configuration_*.py) return 0 ;;
        lib/python*/site-packages/transformers/models/*/tokenization_*.py) return 0 ;;
    esac
    return 1
}

# match_hits_protected <python_root> <abs_match>: 0 if the match is, or
# CONTAINS, a protected path. A rule removes the whole match, so the check
# walks the match's descendants, not just its top: `path bin` covers the
# interpreter and the llama-server pair, and `path lib/python3.12` covers
# site-packages/app, the mlx / mlx_lm trees and every transformers family's
# __init__/configuration_*/tokenization_* files several levels deep.
match_hits_protected() {
    _root=$1
    _m=$2
    [ -e "$_m" ] || return 1
    # The descent is through an external file, not a pipe: a subshell could
    # not report a hit back (bash 3.2 has no lastpipe). For a file match,
    # find prints the file itself; for a dir match the whole subtree is
    # walked -- the protected table has file-level entries (a family's
    # __init__.py sits one level down, a site-packages package several), so a
    # shallow scan would miss a match that merely CONTAINS one.
    _hits=$(mktemp) || return 1
    # -print0 / read -d '' keep spaced paths intact. The top is always
    # checked; for a dir, every descendant is checked too.
    printf '%s\0' "$_m" >"$_hits"
    if [ -d "$_m" ]; then
        find "$_m" -mindepth 1 -print0 2>/dev/null >>"$_hits"
    fi
    while IFS= read -r -d '' _f; do
        [ -n "$_f" ] || continue
        _r=${_f#$_root}
        _r=${_r#/}
        rel_is_protected "$_r" && { rm -f "$_hits"; return 0; }
    done <"$_hits"
    rm -f "$_hits"
    return 1
}

# prune_python <python_root> <list_file>: validate the whole list (malformed
# line, stale rule, protected match), then remove. Prints `pruned: <relpath>`
# per removal and a final `pruned <n> paths (<KiB> KiB)` line. The list file
# is opened here and only here: the main flow (step 5a) and the --prune test
# hook both call this, so a throwaway repo that copies only bundle-python.sh
# keeps the two in sync.
prune_python() {
    root=$1
    listf=$2
    [ -d "$root" ] || die "python root not found: $root"
    [ -f "$listf" ] || die "prune list file not found: $listf"
    # --- validate: collect every match, refuse before the first removal ------
    # all_matches / stale_rules are fed through newlines, so an absolute
    # operand (which would survive as a path to an ancestor's sibling) is
    # checked here, once, before any find runs.
    all_matches=""
    stale_rules=""
    while IFS= read -r line || [ -n "$line" ]; do
        case $line in
            ''|\#*) continue ;;
        esac
        kind=${line%% *}
        rest=${line#* }
        [ "$kind" = "$line" ] && die "malformed prune rule (missing operand): '$line'"
        [ -n "$rest" ] || die "malformed prune rule (empty operand): '$line'"
        case $kind in
            path)
                [ "$rest" = "$line" ] && die "malformed prune rule (missing glob): '$line'"
                p=$rest
                case $p in /*) die "prune rule: absolute path not allowed: $p" ;; esac
                case /$p/ in *../*) die "prune rule: '..' segment not allowed: $p" ;; esac
                # find -path lets `*` span `/`, which would let `path lib/*.dylib`
                # also remove lib/<sub>/x.dylib; a bash glob's `*` spans one
                # segment, so keep only the matches at the pattern's own depth.
                # The match is <root>/<p>: its `/`-segment count is the pattern's
                # segments plus root's segments (the join adds no slash, it is
                # already counted on both sides). Count with split on "/" -- the
                # reliable form on this host's awk.
                n=$(printf '%s' "$p" | awk '{ print split($0, _a, "/") }')
                rseg=$(printf '%s' "$root" | awk '{ print split($0, _a, "/") }')
                out=$(find "$root" -path "$root/$p" -print -prune 2>/dev/null \
                    | awk -v want="$((n + rseg))" '{ if (split($0, _b, "/") == want) print }')
                [ -n "$out" ] || { stale_rules="$stale_rules$line
"; continue; }
                while IFS= read -r f; do
                    [ -n "$f" ] || continue
                    match_hits_protected "$root" "$f" && {
                        r=${f#"$root"/}
                        die "prune rule hits a protected path: $r ($line)"
                    }
                    all_matches="$all_matches$f
"
                done <<EOF
$out
EOF
                ;;
            tree)
                # dir is the first token; name is the rest (may contain a
                # space only if it is a wildcard; a plain name is one token).
                [ "$rest" = "$line" ] && die "malformed prune rule (missing operands): '$line'"
                dir=${rest%% *}
                name=${rest#* }
                [ "$name" = "$dir" ] && die "malformed prune rule (missing name glob): '$line'"
                case $dir in /*) die "prune rule: absolute path not allowed: $dir" ;; esac
                case /$dir/ in *../*) die "prune rule: '..' segment not allowed: $dir" ;; esac
                case $name in /*) die "prune rule: absolute name not allowed: $name" ;; esac
                [ -d "$root/$dir" ] || { stale_rules="$stale_rules$line
"; continue; }
                # `! -path '*/models/auto/*'` keeps transformers' models/auto/
                # dir: its modeling_auto.py / image_processing_auto.py /
                # video_processing_auto.py (and configuration_auto /
                # tokenization_auto) are what every family's configuration_*
                # module imports at module level, and the import guard imports
                # all of them. Without the exclusion a `tree .../models
                # modeling_*` rule would remove auto/ and the guard would fail
                # the build.
                out=$(find "$root/$dir" -name "$name" ! -path '*/models/auto/*' -print -prune 2>/dev/null)
                [ -n "$out" ] || { stale_rules="$stale_rules$line
"; continue; }
                while IFS= read -r f; do
                    [ -n "$f" ] || continue
                    match_hits_protected "$root" "$f" && {
                        r=${f#"$root"/}
                        die "prune rule hits a protected path: $r ($line)"
                    }
                    all_matches="$all_matches$f
"
                done <<EOF
$out
EOF
                ;;
            *) die "malformed prune rule (unknown kind '$kind'): '$line'" ;;
        esac
    done <"$listf"
    [ -z "$stale_rules" ] || {
        echo "bundle-python.sh: prune rule matched nothing (stale list; a rule no match):" >&2
        printf '%s' "$stale_rules" | sed 's/^/  /' >&2
        exit 1
    }
    [ -n "$all_matches" ] || die "prune list removed nothing (stale list)"
    # --- remove ----------------------------------------------------------------
    n=0
    kbytes=0
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        sz=$(du -sk "$f" 2>/dev/null | awk '{print $1}')
        [ -n "$sz" ] && kbytes=$((kbytes + sz))
        rm -rf "$f"
        echo "pruned: ${f#"$root"/}"
        n=$((n + 1))
    done <<EOF
$all_matches
EOF
    echo "bundle-python.sh: pruned $n paths ($kbytes KiB)"
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
if [ "${1:-}" = "--prune" ]; then
    # step 5a alone: <python_root>, optional --list <file> (default: the
    # committed exclude list, opened by prune_python, not here).
    prune_root=""
    prune_list=""
    shift
    while [ $# -gt 0 ]; do
        case $1 in
            --list) [ $# -ge 2 ] || {
                echo "usage: bundle-python.sh --prune <python_root> [--list <file>]" >&2
                exit 2
            }; prune_list=$2; shift 2 ;;
            -*) usage_p="usage: bundle-python.sh --prune <python_root> [--list <file>]";
               echo "$usage_p" >&2; exit 2 ;;
            *) [ -z "$prune_root" ] || {
                echo "usage: bundle-python.sh --prune <python_root> [--list <file>]" >&2
                exit 2
            }; prune_root=$1; shift ;;
        esac
    done
    [ -n "$prune_root" ] && [ -d "$prune_root" ] || {
        echo "usage: bundle-python.sh --prune <python_root> [--list <file>]" >&2
        exit 2
    }
    [ -n "$prune_list" ] || prune_list="$ROOT/scripts/macos/python-prune.txt"
    prune_python "$prune_root" "$prune_list"
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
# --- 5a. the prune (explicit list, validated before any removal) -------------
prune_python "$spy" "$ROOT/scripts/macos/python-prune.txt"
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
# The import net: every module the server needs on the Mac, imported by the
# bundled interpreter. The P11 prune is gated on this -- a pruned file the
# server still imports turns the build red, naming the module.
"$dest/bin/$SERVER_EXE" -I -B "$ROOT/scripts/macos/bundle-import-guard.py" || die "import guard failed: $dest/bin/$SERVER_EXE"
echo "bundle-python.sh: embedded python $PBS_PYVER at ${dest#"$(dirname "$app_abs")"/}"
