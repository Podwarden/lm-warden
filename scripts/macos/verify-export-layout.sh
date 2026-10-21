#!/bin/bash
# verify-export-layout.sh — check a Next static export against the app/webui mount.
#
# Usage:
#   scripts/macos/verify-export-layout.sh [<out-dir>]
#
# <out-dir> defaults to frontend/out (resolved relative to the repo root, so
# the script works from any cwd).
#
# app/webui mounts the export with StaticFiles(html=True) at /ui, i.e.
# /ui/<path> serves <out>/<path>. That topology is only sound when the export
# was built with the basePath baked in (Next inlines it into every asset URL)
# and carries no server-route artifacts. Fail (exit 1, one line per problem)
# when any of the following is violated:
#
#   * <out>/index.html exists (the /ui root page)
#   * <out>/_next/static/ exists (the asset root the html references)
#   * <out>/ui/ is NOT a directory (basePath must stay a URL prefix, not a
#     directory level — a ui/ dir would shadow it on disk)
#   * every /ui/_next/... src/href in every <out>/**/*.html resolves to a
#     file under <out>/_next/...
#   * no html references a bare /_next/... path (basePath not baked in; the
#     /ui mount would 404 it)
#   * no ref under /ui/_next/ carries a . or .. path segment (it would
#     resolve outside <out>/_next/ on disk)
#   * no api/ directory at the export root (server route artifact, illegal
#     under output: 'export')
#
# Counts: html = regular files named *.html; assets = regular files under
# <out>/_next/static/.
#
# Exit codes: 0 ok (one line "export layout ok: <n> html, <m> assets"),
# 1 problems found, 2 missing out dir.
#
# Bash-3.2-safe (the M1 runner's /bin/bash is 3.2): no associative arrays, no
# ${x,,}, no mapfile, no grep -P. Runs on Linux too.
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)

out=${1:-$repo_root/frontend/out}
if [ ! -d "$out" ]; then
    echo "verify-export-layout.sh: no such directory: $out" >&2
    exit 2
fi

problems=""
report() {
    problems="${problems}$1
"
}

[ -f "$out/index.html" ] || report "missing: $out/index.html"
[ -d "$out/_next/static" ] || report "missing: $out/_next/static/"
[ -e "$out/ui" ] && report "ui/ exists at the export root (basePath must not be a directory level): $out/ui/"
[ -d "$out/api" ] && report "api/ server route artifact: $out/api/"

# Extract every src="..." / href="..." value, one per line, in document
# order. A line may hold more than one attribute (a Next export is one long
# line: <link href> tags come before <script src> tags), so take the
# leftmost match of either attribute and rescan from just past it.
refs_of() {
    awk '
        {
            rest = $0
            while (match(rest, /(src|href)="[^"]*"/)) {
                m = substr(rest, RSTART, RLENGTH)
                i = index(m, "=\"")
                print substr(m, i + 2, length(m) - i - 2)
                rest = substr(rest, RSTART + RLENGTH)
            }
        }
    ' "$1"
}

html_count=0
while IFS= read -r f; do
    [ -n "$f" ] || continue
    html_count=$((html_count + 1))
    while IFS= read -r ref; do
        [ -n "$ref" ] || continue
        # The mount serves the path; a ?query or #fragment is not on disk.
        path=${ref%%\?*}
        path=${path%%#*}
        if case "$path" in /ui/_next/*) true ;; *) false ;; esac; then
            if case "/$path/" in */../* | */./*) true ;; *) false ;; esac; then
                # A dot segment would resolve outside <out>/_next on disk
                # (or outside <out> altogether); the mount refuses it.
                report "ref escapes the export's _next/ in $f: $ref"
            else
                # /ui/<path> serves <out>/<path> — check <out>/_next/... on disk.
                target="$out/${path#/ui/}"
                [ -f "$target" ] || report "unresolved ref in $f: $ref"
            fi
        elif case "$path" in /_next/*) true ;; *) false ;; esac; then
            # Bare /_next/... means the basePath was not baked in; the /ui
            # mount would serve it 404.
            report "ref not under /ui (basePath not baked) in $f: $ref"
        fi
    done <<EOF
$(refs_of "$f")
EOF
done <<EOF
$(find "$out" -type f -name '*.html')
EOF

asset_count=0
if [ -d "$out/_next/static" ]; then
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        asset_count=$((asset_count + 1))
    done <<EOF
$(find "$out/_next/static" -type f)
EOF
fi

if [ -n "$problems" ]; then
    printf '%s' "$problems"
    exit 1
fi
echo "export layout ok: $html_count html, $asset_count assets"
