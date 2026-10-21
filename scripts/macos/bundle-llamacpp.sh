#!/bin/bash
# bundle-llamacpp.sh — build a Metal llama-server and place it in "LM Warden.app" (P5).
#
# Usage:
#   scripts/macos/bundle-llamacpp.sh <app> [--cache-dir <dir>] [--jobs <n>] [--offline]
#   scripts/macos/bundle-llamacpp.sh --build-only [--cache-dir <dir>] [--jobs <n>] [--offline]
#   scripts/macos/bundle-llamacpp.sh [<app>] --print-plan [--cache-dir <dir>]
#   scripts/macos/bundle-llamacpp.sh --check-links <binary>                 (test hook)
#
# DECISION (Metal design s6): BUILD from a pinned upstream source tag, never the
# upstream macOS release binary (dynamically linked libggml*/libllama dylibs, a
# build-tree rpath, and a third-party signature inside our notarized bundle)
# and never Homebrew (not shippable). The pin is the GitHub source tarball of
# tag LLAMACPP_TAG, checked against LLAMACPP_SHA256, and -- where git is
# available -- its embedded commit id against LLAMACPP_COMMIT. The tag is the
# SAME one the Linux Dockerfile builds (ARG LLAMACPP_BUILD) and the one
# tests/fixtures/llamacpp/ was captured at; tests/unit/test_bundle_llamacpp.py
# fails if they drift.
#
# A tarball has no .git, so cmake/build-info.cmake would stamp "build 0,
# commit unknown". The Dockerfile solves that with a full clone; here the
# top-level CMakeLists' own override hooks (LLAMA_BUILD_NUMBER /
# LLAMA_BUILD_COMMIT, `if (NOT DEFINED ...)`) are fed the pinned values, so
# --version reads exactly as the Linux build's does:
# "version: <x> (build 11719, commit 5aeac37a3)".
#
# Steps:
#   1. Fetches the PINNED CMake (Kitware's macos-universal release tarball,
#      CMAKE_URL / CMAKE_SHA256) into <cache-dir> -- the M1 has no cmake, and a
#      Homebrew cmake would be an unpinned build input. Nothing unpinned is
#      ever downloaded; a cached tarball is re-checked every run.
#   2. Fetches the pinned llama.cpp source tarball and extracts it.
#   3. Configures and builds ONLY the llama-server target, arm64, Release:
#        -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON  the Metal kernels are
#            assembled into a __DATA,__ggml_metallib section and compiled by the
#            GPU driver at load: no default.metallib beside the binary, no
#            `xcrun metal` (full Xcode) at build time.
#        -DBUILD_SHARED_LIBS=OFF  libllama/libggml/libmtmd linked in: no
#            @rpath dylibs, so no rpath to get wrong.
#        -DGGML_NATIVE=OFF        no -mcpu=native: the binary must run on any
#            Apple silicon, not just the builder's (Apple clang's arm64 default
#            is already apple-m1).
#        -DGGML_OPENMP=OFF        a Homebrew libomp would otherwise be found
#            and linked from /opt/homebrew.
#        -DLLAMA_OPENSSL=OFF      no HTTPS: the -hf/-mu download path is never
#            used (paths.py resolves the file), and Homebrew openssl would leak
#            in. (b11719 has no LLAMA_CURL; the HTTP client is cpp-httplib.)
#        -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF  no npm and no
#            build-time download of llama.cpp's own chat UI from a Hugging Face
#            bucket (unpinned); the warden has its own UI and proxies the API.
#        -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0  matches requirements-macos.txt.
#      The build tree is cached under <cache-dir>, keyed by the tag and a hash
#      of the configure flags, so a re-run with the same pin does not rebuild.
#   4. The link gate: `otool -L` must list only /usr/lib/ and /System/Library/
#      entries (the macOS analogue of the Dockerfile's `ldd | ! grep 'not
#      found'`), there must be no LC_RPATH, the only architecture must be
#      arm64, and the __ggml_metallib section must be present. Any failure is
#      exit 1 -- the binary is never placed.
#   5. Copies it to <app>/Contents/Resources/python/bin/llama-server, next to
#      the embedded interpreter and mlx_lm.server: that dir is the first entry
#      of the apple engine PATH (app/runtime/backends/llamacpp/env.py), so the
#      backend's bare argv[0] "llama-server" resolves to it with no launcher
#      setting. Run AFTER bundle-python.sh (which replaces python/ wholesale)
#      and BEFORE sign-bundle.sh (it signs the new Mach-O inside-out like every
#      other); the gate is re-run on the placed copy. Beside it goes
#      llama-server.build, one line holding LLAMACPP_TAG: the apple profile's
#      version source (app/runtime/backends/llamacpp/version.py), since no
#      Dockerfile stamps VW_LLAMACPP_BUILD on the Mac.
#
# --build-only runs steps 1-4 and leaves the binary in the cache (lets a
# time-capped remote run split the compile from the bundling).
# --print-plan prints the resolved plan as key=value lines and touches nothing.
# --check-links runs the step-4 gate alone on <binary>.
#
# Exit codes: 0 ok, 1 a step failed (checksum, download, commit id, build,
# link gate, placement), 2 usage.
#
# Bash-3.2-safe (the M1 runner's /bin/bash is 3.2): indexed arrays only, no
# associative arrays, no ${x,,}, no mapfile.
set -eu

# --- the pins -------------------------------------------------------------
# llama.cpp: the Dockerfile's ARG LLAMACPP_BUILD and the fixtures' tag.
LLAMACPP_TAG="b11719"
LLAMACPP_BUILD_NUMBER="11719"
LLAMACPP_COMMIT="5aeac37a35b44fde4bdffaec9159f2fcb253b622"
LLAMACPP_URL="https://github.com/ggml-org/llama.cpp/archive/refs/tags/${LLAMACPP_TAG}.tar.gz"
LLAMACPP_SHA256="19487eb80fe875e6ead9551195e589ed4371b7dd3a6f32d9979fd1cbfa451589"
# CMake 3.31.8 (llama.cpp asks for 3.14...3.28 policy range); sha256 is the
# value in Kitware's own cmake-3.31.8-SHA-256.txt for this asset.
CMAKE_VERSION="3.31.8"
CMAKE_ASSET="cmake-${CMAKE_VERSION}-macos-universal.tar.gz"
CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_ASSET}"
CMAKE_SHA256="d1449f969c54d5c00886d5b643340d493dfb3c81cb39ee29b35453395c11ebf7"

CMAKE_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_OSX_ARCHITECTURES=arm64
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
    -DGGML_METAL=ON
    -DGGML_METAL_EMBED_LIBRARY=ON
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_NATIVE=OFF
    -DGGML_OPENMP=OFF
    -DLLAMA_OPENSSL=OFF
    -DLLAMA_BUILD_UI=OFF
    -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_SERVER=ON
    "-DLLAMA_BUILD_NUMBER=${LLAMACPP_BUILD_NUMBER}"
    "-DLLAMA_BUILD_COMMIT=${LLAMACPP_COMMIT:0:9}"
)

BIN_NAME="llama-server"
USAGE="usage: bundle-llamacpp.sh <app>|--build-only [--cache-dir <dir>] [--jobs <n>] [--offline] [--print-plan]"

die() { echo "bundle-llamacpp.sh: $*" >&2; exit 1; }
usage() { echo "$USAGE" >&2; exit 2; }

ROOT=$(cd "$(dirname "$0")/../.." && pwd)

sha256_of() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        sha256sum "$1" | awk '{print $1}'
    fi
}

# check_links <binary>: the step-4 gate. Prints one "links: <dylib>" line per
# dependency and exits the script 1 on the first violation.
check_links() {
    bin=$1
    [ -f "$bin" ] || die "link gate: no such file: $bin"
    file -b "$bin" | grep -q "Mach-O" || die "link gate: not a Mach-O: $bin"
    archs=$(lipo -archs "$bin" 2>/dev/null) || die "link gate: lipo failed on $bin"
    [ "$archs" = "arm64" ] || die "link gate: architectures are '$archs', want arm64 only: $bin"
    # otool -L prints "<bin>:" then one indented "<path> (compatibility ...)"
    # per load command; the first line is the file itself.
    deps=$(otool -L "$bin" | tail -n +2 | awk '{print $1}')
    bad=""
    while IFS= read -r d; do
        [ -n "$d" ] || continue
        case $d in
            /usr/lib/*|/System/Library/*) echo "links: $d" ;;
            *) bad="$bad $d" ;;
        esac
    done <<EOF
$deps
EOF
    [ -z "$bad" ] || die "link gate: non-system dylib(s) in $bin:$bad"
    if otool -l "$bin" | grep -q "LC_RPATH"; then
        die "link gate: $bin carries an LC_RPATH (a static build needs none)"
    fi
    echo "bundle-llamacpp.sh: link gate ok ($(printf '%s\n' "$deps" | grep -c .) system dylibs/frameworks): $bin"
}

# check_metallib <binary>: the Metal kernels are embedded, not a side file.
check_metallib() {
    otool -l "$1" | grep -q "sectname __ggml_metallib" \
        || die "no __ggml_metallib section in $1: GGML_METAL_EMBED_LIBRARY did not take"
    echo "bundle-llamacpp.sh: Metal library embedded (__DATA,__ggml_metallib)"
}

# fetch <url> <sha256> <dest>: download once into the cache, verify every run.
fetch() {
    f_url=$1 f_want=$2 f_dest=$3
    if [ -f "$f_dest" ]; then
        got=$(sha256_of "$f_dest")
        if [ "$got" != "$f_want" ]; then
            [ $offline = 0 ] || die "sha256 mismatch for cached $f_dest: expected $f_want, got $got"
            echo "bundle-llamacpp.sh: cached ${f_dest##*/} fails its checksum, re-downloading" >&2
            rm -f "$f_dest"
        fi
    fi
    if [ ! -f "$f_dest" ]; then
        [ $offline = 0 ] || die "--offline and ${f_dest##*/} is not in $cache_dir"
        echo "==> fetch $f_url"
        part="$f_dest.part.$$"
        curl -fsSL --proto '=https' --tlsv1.2 --retry 3 -o "$part" "$f_url" || {
            rm -f "$part"
            die "download failed: $f_url"
        }
        got=$(sha256_of "$part")
        if [ "$got" != "$f_want" ]; then
            rm -f "$part"
            die "sha256 mismatch for $f_url: expected $f_want, got $got"
        fi
        mv "$part" "$f_dest"
    fi
    echo "==> ${f_dest##*/} sha256 ok"
}

# --- arguments ------------------------------------------------------------
if [ "${1:-}" = "--check-links" ]; then
    [ $# -eq 2 ] || { echo "usage: bundle-llamacpp.sh --check-links <binary>" >&2; exit 2; }
    check_links "$2"
    exit 0
fi

app=""
build_only=0
cache_dir="${HOME}/Library/Caches/lmw-macos-llamacpp"
jobs=""
offline=0
plan=0
while [ $# -gt 0 ]; do
    case $1 in
        --cache-dir) [ $# -ge 2 ] && [ -n "$2" ] || usage; cache_dir=$2; shift 2 ;;
        --jobs) [ $# -ge 2 ] && [ -n "$2" ] || usage; jobs=$2; shift 2 ;;
        --offline) offline=1; shift ;;
        --build-only) build_only=1; shift ;;
        --print-plan) plan=1; shift ;;
        -h|--help) echo "$USAGE"; exit 0 ;;
        -*) usage ;;
        *) [ -z "$app" ] || usage; app=$1; shift ;;
    esac
done
case $jobs in ''|*[!0-9]*) [ -z "$jobs" ] || usage ;; esac
if [ $build_only = 0 ] && [ $plan = 0 ]; then
    [ -n "$app" ] || usage
fi
if [ -n "$app" ] && [ ! -d "$app/Contents" ]; then
    echo "bundle-llamacpp.sh: not an app bundle (no Contents/): $app" >&2
    usage
fi
case $cache_dir in /*) ;; *) cache_dir=$(pwd)/$cache_dir ;; esac

# The build key: a change to any configure flag (or the pin) is a new tree.
flags_str="${CMAKE_FLAGS[*]}"
key=$(printf '%s\n' "$LLAMACPP_TAG" "$LLAMACPP_SHA256" "$CMAKE_VERSION" "$flags_str" | sha256_of /dev/stdin | cut -c1-12)
src_tar="$cache_dir/llama.cpp-${LLAMACPP_TAG}.tar.gz"
src_dir="$cache_dir/src-${LLAMACPP_TAG}"
build_dir="$cache_dir/build-${LLAMACPP_TAG}-${key}"
built="$build_dir/bin/$BIN_NAME"
dest=""
[ -z "$app" ] || dest="$(cd "$app" && pwd)/Contents/Resources/python/bin/$BIN_NAME"
# app/runtime/backends/llamacpp/version.py STAMP_NAME: the tag, beside the binary.
STAMP_NAME="llama-server.build"
stamp=""
[ -z "$dest" ] || stamp="${dest%/*}/$STAMP_NAME"

if [ $plan = 1 ]; then
    echo "decision=build-from-source"
    echo "tag=$LLAMACPP_TAG"
    echo "commit=$LLAMACPP_COMMIT"
    echo "url=$LLAMACPP_URL"
    echo "sha256=$LLAMACPP_SHA256"
    echo "cmake_url=$CMAKE_URL"
    echo "cmake_sha256=$CMAKE_SHA256"
    echo "cmake_flags=$flags_str"
    echo "build_dir=$build_dir"
    echo "binary=$built"
    echo "dest=$dest"
    echo "stamp=$stamp"
    exit 0
fi

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] \
    || die "builds on Apple silicon only (this is $(uname -s)/$(uname -m))"

mkdir -p "$cache_dir"
cache_dir=$(cd "$cache_dir" && pwd)

if [ -x "$built" ] && [ -f "$build_dir/.lmw-built" ]; then
    echo "==> cached build $built (key $key)"
else
    # --- 1. the pinned cmake ---------------------------------------------
    fetch "$CMAKE_URL" "$CMAKE_SHA256" "$cache_dir/$CMAKE_ASSET"
    cmake_root="$cache_dir/cmake-${CMAKE_VERSION}"
    if [ ! -x "$cmake_root/CMake.app/Contents/bin/cmake" ]; then
        rm -rf "$cmake_root" "$cmake_root.tmp"
        mkdir -p "$cmake_root.tmp"
        tar -xzf "$cache_dir/$CMAKE_ASSET" -C "$cmake_root.tmp" --strip-components 1
        mv "$cmake_root.tmp" "$cmake_root"
    fi
    CMAKE="$cmake_root/CMake.app/Contents/bin/cmake"

    # --- 2. the pinned source --------------------------------------------
    fetch "$LLAMACPP_URL" "$LLAMACPP_SHA256" "$src_tar"
    if command -v git >/dev/null 2>&1; then
        got_commit=$(gzip -dc "$src_tar" | git get-tar-commit-id) || die "no commit id in $src_tar"
        [ "$got_commit" = "$LLAMACPP_COMMIT" ] \
            || die "commit id mismatch in $src_tar: expected $LLAMACPP_COMMIT, got $got_commit"
        echo "==> commit $got_commit ok"
    fi
    if [ ! -f "$src_dir/CMakeLists.txt" ]; then
        rm -rf "$src_dir" "$src_dir.tmp"
        mkdir -p "$src_dir.tmp"
        tar -xzf "$src_tar" -C "$src_dir.tmp" --strip-components 1
        mv "$src_dir.tmp" "$src_dir"
    fi

    # --- 3. configure + build llama-server only --------------------------
    [ -n "$jobs" ] || jobs=$(sysctl -n hw.ncpu)
    echo "==> cmake configure ($LLAMACPP_TAG, key $key)"
    "$CMAKE" -S "$src_dir" -B "$build_dir" "${CMAKE_FLAGS[@]}" || die "cmake configure failed"
    echo "==> cmake build --target $BIN_NAME -j$jobs"
    "$CMAKE" --build "$build_dir" --target "$BIN_NAME" -j "$jobs" || die "cmake build failed"
    [ -x "$built" ] || die "build produced no $built"
    touch "$build_dir/.lmw-built"
fi

# --- 4. the link gate on the built binary ----------------------------------
check_links "$built"
check_metallib "$built"
"$built" --version 2>&1 | grep -q "build ${LLAMACPP_BUILD_NUMBER}, commit ${LLAMACPP_COMMIT:0:9}" \
    || die "$built --version does not report build $LLAMACPP_BUILD_NUMBER / commit ${LLAMACPP_COMMIT:0:9}: $("$built" --version 2>&1 | tail -n 1)"
echo "==> $("$built" --version 2>&1 | grep -m1 '^version')"

if [ $build_only = 1 ]; then
    echo "bundle-llamacpp.sh: built $built (not placed: --build-only)"
    exit 0
fi

# --- 5. place it next to the embedded interpreter --------------------------
[ -d "${dest%/*}" ] || die "no ${dest%/*}: run bundle-python.sh first (it lays python/ wholesale)"
cp "$built" "$dest.tmp.$$"
chmod 755 "$dest.tmp.$$"
xattr -c "$dest.tmp.$$" 2>/dev/null || true
mv "$dest.tmp.$$" "$dest"
check_links "$dest"
# The build stamp: the version module reads it on the apple profile, where no
# Dockerfile sets VW_LLAMACPP_BUILD. One line, the pinned tag -- written from
# LLAMACPP_TAG so the pin stays single-sourced here.
printf '%s\n' "$LLAMACPP_TAG" > "$stamp.tmp.$$"
mv "$stamp.tmp.$$" "$stamp"
echo "bundle-llamacpp.sh: placed $dest, stamped $LLAMACPP_TAG in ${stamp##*/} (sign it with sign-bundle.sh)"
