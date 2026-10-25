#!/bin/bash
# smoke-isolation.sh — the bundle smoke's isolation checks (F7).
#
# The old smoke asserted isolation by diffing a whole-tree snapshot of the
# operator's real data dir before and after the launch. That went red on the
# environment, not the code: an installed LM Warden.app (or a `git` checkout of
# the repo that lives under the real dir) writes vllm-warden.db there at any
# moment. This helper instead proves the property the smoke actually cares
# about — that the smoke's own processes never use the real dir — and is driven
# as a CLI by smoke-bundle.sh (or a test).
#
# Usage:
#   smoke-isolation.sh child-data-dir <recorded-file> <expected-dir> <real-dir>
#   smoke-isolation.sh open-under <real-dir> <pid>...
#
#   child-data-dir: the server child records the data dir it was launched with
#   (VW_DATA_DIR) into <recorded-file>. Exit 0 when the file's content equals
#   <expected-dir> exactly; exit 1 with a one-line reason naming the value when
#   the file is missing, empty, differs from <expected-dir>, or is <real-dir>
#   or under <real-dir>/.
#
#   open-under: exit 0 when none of <pids> has a file open at <real-dir> or
#   under <real-dir>/; exit 1 printing each offending path. A sibling whose
#   name only shares the prefix (e.g. "<real-dir> Backup/x", "<real-dir>2/x")
#   is not a hit, and a pid that has exited is clean (lsof non-zero, no
#   output). Bad usage (no pids, a relative dir) exits 2.
#
# Exit codes: 0 ok, 1 a check failed, 2 usage.
#
# Bash-3.2-safe (the M1's /bin/bash is 3.2): no mapfile, no associative
# arrays, no ${x,,}; safe under `set -eu`.
usage() {
    printf 'usage: smoke-isolation.sh child-data-dir <recorded-file> <expected-dir> <real-dir>\n' >&2
    printf '       smoke-isolation.sh open-under <real-dir> <pid>...\n' >&2
}

cmd=${1:-}
[ -n "$cmd" ] || { usage; exit 2; }
shift

case "$cmd" in
    child-data-dir)
        if [ "$#" -ne 3 ]; then
            usage
            exit 2
        fi
        recorded=$1
        expected=$2
        real=$3
        case "$expected" in
            /*) ;;
            *) usage; exit 2 ;;
        esac
        case "$real" in
            /*) ;;
            *) usage; exit 2 ;;
        esac
        # Strip one trailing slash so "dir/" matches the dir but not "dir//x".
        realn=${real%/}
        value=""
        if [ ! -f "$recorded" ]; then
            printf 'smoke-isolation: child-data-dir: %s is missing\n' "$recorded" >&2
            exit 1
        fi
        value=$(cat "$recorded")
        if [ -z "$value" ]; then
            printf 'smoke-isolation: child-data-dir: %s is empty\n' "$recorded" >&2
            exit 1
        fi
        if [ "$value" = "$realn" ]; then
            printf 'smoke-isolation: child-data-dir: child was launched with the real data dir %s\n' "$value" >&2
            exit 1
        fi
        case "$value" in
            "$realn"/*)
                printf 'smoke-isolation: child-data-dir: child data dir %s is under the real data dir %s\n' "$value" "$realn" >&2
                exit 1
                ;;
        esac
        if [ "$value" != "$expected" ]; then
            printf 'smoke-isolation: child-data-dir: child data dir is %s, want %s\n' "$value" "$expected" >&2
            exit 1
        fi
        exit 0
        ;;

    open-under)
        if [ "$#" -lt 2 ]; then
            usage
            exit 2
        fi
        real=$1
        shift
        case "$real" in
            /*) ;;
            *) usage; exit 2 ;;
        esac
        realn=${real%/}
        pids=""
        for pid in "$@"; do
            case "$pid" in
                ''|*[!0-9]*) usage; exit 2 ;;
            esac
            if [ -z "$pids" ]; then
                pids=$pid
            else
                pids="$pids,$pid"
            fi
        done
        # lsof lists every open file path of the pids; a pid that has exited
        # just yields no lines (and a non-zero status), which is clean.
        # 2>/dev/null: lsof writes a "no more processes" note to stderr, which
        # would otherwise be captured as a path line.
        paths=$(lsof -Fn -p "$pids" 2>/dev/null | sed -n 's/^n//p')
        hit=0
        oldIFS=$IFS
        IFS=$'\n'
        for path in $paths; do
            [ -n "$path" ] || continue
            if [ "$path" = "$realn" ]; then
                printf '%s\n' "$path"
                hit=1
                continue
            fi
            case "$path" in
                "$realn"/*)
                    printf '%s\n' "$path"
                    hit=1
                    ;;
            esac
        done
        IFS=$oldIFS
        [ "$hit" -eq 1 ] || exit 0
        exit 1
        ;;

    *)
        usage
        exit 2
        ;;
esac
