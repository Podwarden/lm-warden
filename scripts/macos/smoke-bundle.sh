#!/bin/bash
# smoke-bundle.sh — scripted GUI smoke of the LM Warden.app bundle (P7).
#
# Usage:
#   scripts/macos/smoke-bundle.sh --python <abs interpreter>
#
# <abs interpreter> must import the app under -I (`python -I -c 'import
# app.main'`), because the shell starts its server child as
# `python -I -B -m uvicorn app.main:app`.
#
# What it does, in order:
#   1. swift build -c release, then assembles a bundle in a temp dir with
#      write-info-plist.sh bundle id com.podwarden.lmwarden.smoke and the
#      pinned Sparkle.framework (embed-sparkle.sh, default cache). The
#      interpreter goes in as a bash wrapper at
#      Contents/Resources/python/bin/LM Warden Server -- the renamed
#      interpreter's path, ShellAppDelegate.bundledPythonPath() -- (exec of
#      <abs interpreter>), then
#      sign-bundle.sh signs it ad-hoc.
#   2. Seeds the smoke defaults domain so first run is done, then launches the
#      app with NO arguments (GUI mode) through `open`.
#   3. Waits for the menu-bar process and its server child. The child's pid,
#      argv, VW_SHELL_TOKEN and VW_DATA_DIR (the data dir it was launched with)
#      are recorded by the wrapper into the smoke state dir (the token and
#      data-dir files are 0600 and never printed); the port is the value after
#      `--port` in that argv, so nothing is guessed. Then GET /healthz (200,
#      "ok": true) and GET /api/shell/status (401 without the token, 200 with
#      it).
#   4. Quits through osascript (`tell application id ... to quit`), asserts the
#      app exits within 20 s, nothing is left in the child's process group and
#      /healthz stops answering; launches again and does the same with SIGTERM.
#   5. Asserts isolation: the real `com.podwarden.lmwarden` Keychain item's
#      attributes are unchanged (never its secret), the smoke used its own data
#      dir, and the smoke's own processes never touched the real data dir. The
#      server child records the data dir it was launched with (VW_DATA_DIR) and
#      the smoke checks it is the smoke dir, not the real one (smoke-isolation.sh
#      child-data-dir); it also checks the app and server child hold no file
#      open in the real dir (smoke-isolation.sh open-under). A whole-dir
#      before/after snapshot is not used: an installed app (or a git checkout of
#      the repo under the real dir) may write vllm-warden.db there at any moment,
#      which would fail the gate on the environment, not the code.
#   Teardown (an EXIT trap, also on failure) kills any smoke process left,
#   deletes the smoke Keychain items, the smoke defaults domain, the smoke data
#   dir and the temp dir. If the lock was lost (the holder died) it still kills
#   this smoke's own processes and removes its temp dir, but leaves the shared
#   Keychain item, defaults domain and data dir alone: a new smoke may hold the
#   lock and own them by then, and every smoke cleans them at its start.
#
# Isolation is by bundle identity: the smoke bundle id gives its own Keychain
# service (KeychainStore.service(forBundleIdentifier:)), its own UserDefaults
# domain and its own data dir (AppPaths.applicationSupport(base:
# bundleIdentifier:)), so the shipped bundle's behaviour is unchanged.
#
# The whole smoke runs under an exclusive lock on $LMW_GUI_SMOKE_LOCK (default
# /Users/Shared/lmw-gui-smoke.lock, created mode 0666 so any user can take it),
# because the smoke bundle id, Keychain service and defaults domain are fixed:
# two M1 slots must never run the smoke at once. The script re-execs itself
# under `portable.py lock` (an exclusive flock held for the child's lifetime)
# before any real work; the re-exec'd child sees _LMW_SMOKE_LOCK_HELD=<lock>
# and does not take the lock again (the marker is unset right after, so the
# app never inherits it). If the lock holder alone dies, a watcher SIGTERMs
# this shell (normal teardown) within about a second, so the smoke never runs
# unlocked. LMW_SMOKE_LOCK_PROBE=1 exits 0 right after taking the lock (test
# hook: it prints `smoke: lock held <lock>` and does no real work);
# LMW_SMOKE_LOCK_PROBE=hold also prints `smoke: probe pid <pid>` and
# `smoke: probe watch pid <pid>` and then idles until signalled (test hook for
# the dead-holder watcher); LMW_SMOKE_LOCK_PROBE=teardown installs the real
# teardown, starts one stand-in app process (printed as `smoke: probe app pid
# <pid>`) and idles until signalled; its cleanup_smoke_state only logs
# `smoke: cleaning the shared smoke state` and deletes nothing (test hook for
# the lost-lock teardown).
#
# Exit codes: 0 ok, 1 an assertion failed, 2 usage / bad interpreter,
# 77 no GUI session for this user or not Apple Silicon.
#
# Bash-3.2-safe (the M1's /bin/bash is 3.2): no mapfile, no associative
# arrays, no ${x,,}.
set -eu

SMOKE_ID=com.podwarden.lmwarden.smoke
REAL_ID=com.podwarden.lmwarden
EXIT_LIMIT=20
HEALTH_LIMIT=120

usage="usage: smoke-bundle.sh --python <abs interpreter with the app importable under -I>"
python=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --python) python=${2:-}; shift 2 || { echo "$usage" >&2; exit 2; } ;;
        *) echo "$usage" >&2; exit 2 ;;
    esac
done
case "$python" in
    /*) ;;
    *) echo "$usage" >&2; exit 2 ;;
esac
if [ ! -x "$python" ]; then
    echo "$usage" >&2
    exit 2
fi
repo=$(cd "$(dirname "$0")/../.." && pwd)
lock=${LMW_GUI_SMOKE_LOCK:-/Users/Shared/lmw-gui-smoke.lock}
# Re-exec under the lock unless this shell already holds this very lock (the
# re-exec below sets _LMW_SMOKE_LOCK_HELD to the lock path for its child only).
if [ "${_LMW_SMOKE_LOCK_HELD:-}" != "$lock" ]; then
    ( umask 000; : >> "$lock" ) 2>/dev/null || true
    _LMW_SMOKE_LOCK_HELD=$lock exec "$python" -I "$repo/scripts/farm/portable.py" lock "$lock" -- /bin/bash "$repo/scripts/macos/smoke-bundle.sh" --python "$python"
fi
unset _LMW_SMOKE_LOCK_HELD
# The lock lives only as long as its holder (portable.py lock, our parent). If
# the holder alone is killed the flock is gone while this shell would keep
# running, and a second smoke could start beside it. A watcher polls this
# shell's ppid once a second; when it is no longer the holder (we were
# reparented, so the holder died) it sends this shell SIGTERM, which runs the
# normal teardown. The watcher stops on its own once this shell is gone, and
# teardown kills it first.
lock_holder=$PPID
smoke_shell=$$
(
    trap - EXIT INT TERM HUP
    while :; do
        p=$(ps -o ppid= -p "$smoke_shell" 2>/dev/null | tr -d ' ')
        [ -n "$p" ] || exit 0
        if [ "$p" != "$lock_holder" ]; then
            printf 'smoke: FAIL: lock holder %s died; aborting the smoke\n' "$lock_holder" >&2
            kill -TERM "$smoke_shell" 2>/dev/null
            exit 0
        fi
        sleep 1
    done
) </dev/null >/dev/null &
lock_watch=$!
stop_lock_watch() { kill "$lock_watch" 2>/dev/null || true; }
# Exits before teardown is installed (no app.main, exit 77, set -e) stop it too.
trap stop_lock_watch EXIT
case "${LMW_SMOKE_LOCK_PROBE:-}" in
    1)
        printf 'smoke: lock held %s\n' "$lock"
        stop_lock_watch
        exit 0
        ;;
    hold)
        printf 'smoke: lock held %s\n' "$lock"
        printf 'smoke: probe pid %s\n' "$$"
        printf 'smoke: probe watch pid %s\n' "$lock_watch"
        trap 'exit 1' INT TERM HUP
        while :; do sleep 1; done
        ;;
esac
# LMW_SMOKE_LOCK_PROBE=teardown skips these checks: it builds and launches
# nothing, and its unit tests also run where there is no GUI session.
if [ "${LMW_SMOKE_LOCK_PROBE:-}" != teardown ]; then
    if ! "$python" -I -c 'import app.main' >/dev/null 2>&1; then
        echo "smoke-bundle.sh: $python cannot import app.main under -I" >&2
        exit 2
    fi

    if [ "$(uname -m)" != arm64 ]; then
        echo "smoke-bundle.sh: not Apple Silicon" >&2
        exit 77
    fi
    if [ "$(stat -f %Su /dev/console 2>/dev/null)" != "$(id -un)" ]; then
        echo "smoke-bundle.sh: no GUI session for $(id -un)" >&2
        exit 77
    fi
fi

started=$(date +%s)
appsupport="$HOME/Library/Application Support"
real_data="$appsupport/LM Warden"
smoke_data="$appsupport/$SMOKE_ID"
lsregister=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister

work=$(mktemp -d "${TMPDIR:-/tmp}/lmw-smoke.XXXXXX")
work=$(cd "$work" && pwd -P)
app="$work/LM Warden.app"
exe="$app/Contents/MacOS/LMWarden"
state="$work/state"
mkdir -p "$state"
chmod 700 "$state"
pgids=""

log() { printf 'smoke: %s\n' "$*"; }
fail() { printf 'smoke: FAIL: %s\n' "$*" >&2; exit 1; }

# Pids of processes whose command starts with the smoke executable path.
app_pids() {
    ps -axo pid=,command= | while read -r pid cmd; do
        case "$cmd" in
            "$exe"*) echo "$pid" ;;
        esac
    done
}

# Pids in process group $1.
group_pids() {
    ps -axo pgid=,pid= | awk -v g="$1" '$1 == g { print $2 }'
}

delete_smoke_keychain() {
    n=0
    while security delete-generic-password -s "$SMOKE_ID" >/dev/null 2>&1; do
        n=$((n + 1))
        [ "$n" -lt 20 ] || break
    done
}

cleanup_smoke_state() {
    log "cleaning the shared smoke state"
    # The teardown probe never touches the real smoke Keychain item, defaults
    # domain or data dir: a real smoke may be running on this host.
    [ "${LMW_SMOKE_LOCK_PROBE:-}" != teardown ] || return 0
    delete_smoke_keychain
    defaults delete "$SMOKE_ID" >/dev/null 2>&1 || true
    rm -f "$HOME/Library/Preferences/$SMOKE_ID.plist"
    rm -rf "$HOME/Library/Saved Application State/$SMOKE_ID.savedState"
    rm -rf "$smoke_data"
}

teardown() {
    status=$?
    set +e
    stop_lock_watch
    for pid in $(app_pids); do
        kill -KILL "$pid" 2>/dev/null
    done
    for g in $pgids; do
        kill -KILL -- "-$g" 2>/dev/null
    done
    [ -x "$lsregister" ] && "$lsregister" -u "$app" >/dev/null 2>&1
    # The shared smoke state (Keychain item, defaults domain, data dir) is only
    # ours while we hold the lock. If the holder died (we were reparented) a new
    # smoke may already own the lock and that state, so leave it: every smoke
    # cleans it at its start, under the lock. An unreadable ppid counts as lost.
    now_ppid=$(ps -o ppid= -p "$smoke_shell" 2>/dev/null | tr -d ' ')
    if [ "$now_ppid" = "$lock_holder" ]; then
        cleanup_smoke_state
    else
        log "lock lost (holder $lock_holder gone): killed this smoke's own processes, left the shared smoke state to the next smoke"
    fi
    rm -rf "$work"
    if [ "$status" -eq 0 ]; then
        log "PASS in $(( $(date +%s) - started )) s"
    fi
    exit "$status"
}
trap teardown EXIT
trap 'exit 1' INT TERM HUP

if [ "${LMW_SMOKE_LOCK_PROBE:-}" = teardown ]; then
    # Test hook: one stand-in app process whose command starts with $exe (so
    # app_pids finds it exactly as it finds the real menu-bar app), then idle
    # until signalled; the EXIT trap runs the real teardown.
    ( exec -a "$exe" sleep 3600 ) </dev/null >/dev/null 2>&1 &
    printf 'smoke: lock held %s\n' "$lock"
    printf 'smoke: probe pid %s\n' "$$"
    printf 'smoke: probe watch pid %s\n' "$lock_watch"
    printf 'smoke: probe app pid %s\n' "$!"
    while :; do sleep 1; done
fi

# A previous run that died hard may have left state: start from clean.
cleanup_smoke_state

# Isolation baseline: attributes only (never -w), so no secret and no prompt.
# The real data dir is not snapshotted before/after: it may be written by an
# installed app (or a git checkout of the repo that lives under it) at any
# moment, and the isolation checks in launch_and_probe prove the smoke's own
# processes never use it instead.
real_keychain_before=$(security find-generic-password -s "$REAL_ID" 2>&1 || true)

# --- 1. build and assemble -------------------------------------------------
log "swift build -c release"
swift build -c release --package-path "$repo/macos" >"$work/build.log" 2>&1 || {
    tail -20 "$work/build.log" >&2
    fail "swift build"
}
bin_dir=$(swift build -c release --package-path "$repo/macos" --show-bin-path)
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources/python/bin"
cp "$bin_dir/LMWarden" "$exe"
"$repo/scripts/macos/write-info-plist.sh" "$app/Contents/Info.plist" smoke LMWarden "$SMOKE_ID"
# The binary links @rpath/Sparkle.framework (P8): without the framework in
# Contents/Frameworks dyld refuses to start it. Same pinned zip and cache as
# make macos-app; --build-dir checks it is the copy SwiftPM linked.
"$repo/scripts/macos/embed-sparkle.sh" "$app" --build-dir "$bin_dir" >"$work/sparkle.log" 2>&1 || {
    cat "$work/sparkle.log" >&2
    fail "embed-sparkle.sh"
}

# The interpreter wrapper. It records its pid (= the server child's, after the
# exec), its argv and the per-launch shell token for this script, then execs
# the real interpreter with the argv unchanged.
wrapper="$app/Contents/Resources/python/bin/LM Warden Server"
{
    printf '#!/bin/bash\n'
    printf 'umask 077\n'
    printf 'state=%q\n' "$state"
    printf 'printf %%s "${VW_SHELL_TOKEN:-}" > "$state/child.token"\n'
    printf 'printf %%s "${VW_DATA_DIR:-}" > "$state/child.datadir"\n'
    printf 'for a in "$@"; do printf "%%s\\n" "$a"; done > "$state/child.args"\n'
    printf 'echo $$ > "$state/child.pid.tmp" && mv "$state/child.pid.tmp" "$state/child.pid"\n'
    printf 'exec %q "$@"\n' "$python"
} >"$wrapper"
chmod 755 "$wrapper"
"$repo/scripts/macos/sign-bundle.sh" "$app" >/dev/null

defaults write "$SMOKE_ID" lmw.firstRunCompleted -bool true
defaults write "$SMOKE_ID" lmw.openAtLogin -bool false

# --- 2.-4. launch, probe, quit ---------------------------------------------
child_pid=""
port=""

launch_and_probe() {
    rm -f "$state/child.pid" "$state/child.args" "$state/child.token" "$state/child.datadir"
    open -n -g "$app"
    app_pid=""
    i=0
    while [ -z "$app_pid" ]; do
        app_pid=$(app_pids | head -1)
        i=$((i + 1))
        [ "$i" -le 30 ] || fail "menu-bar process did not start"
        [ -n "$app_pid" ] || sleep 1
    done
    i=0
    while [ ! -f "$state/child.pid" ]; do
        i=$((i + 1))
        [ "$i" -le 60 ] || fail "server child did not start"
        sleep 1
    done
    child_pid=$(cat "$state/child.pid")
    pgids="$pgids $child_pid"
    child_pgid=$(ps -o pgid= -p "$child_pid" | tr -d ' ')
    [ "$child_pgid" = "$child_pid" ] || fail "child is not its own process group leader"
    ppid=$(ps -o ppid= -p "$child_pid" | tr -d ' ')
    [ "$ppid" = "$app_pid" ] || fail "server child's parent is not the menu-bar process"
    port=$(awk 'f { print; exit } $0 == "--port" { f = 1 }' "$state/child.args")
    case "$port" in
        ''|*[!0-9]*) fail "no --port in the child's argv" ;;
    esac
    log "menu-bar pid $app_pid, server child pid $child_pid, port $port"

    i=0
    until body=$(curl -fsS -m 2 "http://127.0.0.1:$port/healthz" 2>/dev/null); do
        i=$((i + 1))
        [ "$i" -le "$HEALTH_LIMIT" ] || fail "/healthz never answered 200"
        kill -0 "$child_pid" 2>/dev/null || fail "server child exited before /healthz"
        sleep 1
    done
    case "$body" in
        *'"ok":true'*|*'"ok": true'*) ;;
        *) fail "/healthz body has no \"ok\": true" ;;
    esac
    log "/healthz 200 after ~${i} s"

    # Isolation: the child must run against the smoke data dir, not the real
    # one, and the smoke's own processes must hold nothing open in the real dir.
    # Each failure goes through fail with the helper's one-line reason.
    "$repo/scripts/macos/smoke-isolation.sh" child-data-dir "$state/child.datadir" "$smoke_data/LM Warden" "$real_data" ||
        fail "child-data-dir: server child is not isolated from the real data dir"
    "$repo/scripts/macos/smoke-isolation.sh" open-under "$real_data" "$app_pid" "$child_pid" ||
        fail "open-under: a smoke process has a file open in the real data dir"

    code=$(curl -sS -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/api/shell/status")
    [ "$code" = 401 ] || fail "/api/shell/status without the token: $code, want 401"
    [ -s "$state/child.token" ] || fail "the child got no VW_SHELL_TOKEN"
    # The header goes through a 0600 file so the token is never on a command line.
    (umask 077; printf 'Authorization: Bearer %s\n' "$(cat "$state/child.token")" >"$state/auth.header")
    code=$(curl -sS -m 5 -o "$state/status.json" -w '%{http_code}' -H @"$state/auth.header" "http://127.0.0.1:$port/api/shell/status")
    rm -f "$state/auth.header"
    [ "$code" = 200 ] || fail "/api/shell/status with the token: $code, want 200"
    "$python" -I -c 'import json, sys; json.load(open(sys.argv[1]))' "$state/status.json" ||
        fail "/api/shell/status body is not JSON"
    log "/api/shell/status 401 without the token, 200 with it"
}

assert_clean_exit() {
    how=$1
    i=0
    while kill -0 "$app_pid" 2>/dev/null; do
        i=$((i + 1))
        [ "$i" -le "$EXIT_LIMIT" ] || fail "$how: the app did not exit within $EXIT_LIMIT s"
        sleep 1
    done
    log "$how: app exited within ~${i} s"
    j=0
    while [ -n "$(group_pids "$child_pid")" ]; do
        j=$((j + 1))
        if [ "$j" -gt "$EXIT_LIMIT" ]; then
            for pid in $(group_pids "$child_pid"); do
                ps -o pid=,ppid=,stat=,command= -p "$pid" >&2
            done
            curl -sS -m 2 -o /dev/null -w 'healthz: %{http_code}\n' "http://127.0.0.1:$port/healthz" >&2 || true
            fail "$how: processes left in the child's group after $EXIT_LIMIT s"
        fi
        sleep 1
    done
    [ -z "$(app_pids)" ] || fail "$how: a smoke app process is still running"
    if curl -fsS -m 2 -o /dev/null "http://127.0.0.1:$port/healthz" 2>/dev/null; then
        fail "$how: /healthz still answers on port $port"
    fi
    # The group is empty: forget it, so teardown never signals a pgid that a
    # process this script did not start may have reused since.
    pgids=$(printf ' %s ' $pgids | sed "s/ $child_pid / /")
    log "$how: nothing left running, /healthz down"
}

launch_and_probe
osascript -e "tell application id \"$SMOKE_ID\" to quit" >/dev/null || fail "osascript quit failed"
assert_clean_exit "osascript quit"

launch_and_probe
kill -TERM "$app_pid"
assert_clean_exit "SIGTERM"

# --- 5. isolation ------------------------------------------------------------
real_keychain_after=$(security find-generic-password -s "$REAL_ID" 2>&1 || true)
[ "$real_keychain_before" = "$real_keychain_after" ] || fail "the real $REAL_ID Keychain item changed"
[ -d "$smoke_data/LM Warden" ] || fail "the smoke data dir $smoke_data/LM Warden was not used"
log "isolation: real Keychain item unchanged, the smoke's processes never used the real data dir $real_data"
