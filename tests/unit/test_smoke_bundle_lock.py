"""The smoke bundle runs under a host-wide lock (smoke-bundle-lock).

``scripts/macos/smoke-bundle.sh`` re-execs itself under
``portable.py lock`` (an exclusive ``flock``) before doing any real work, so
two M1 slots can never run the GUI smoke at once: the smoke's bundle id,
Keychain service and defaults domain are fixed, so a second concurrent run
would corrupt the first. These tests exercise the probe mode only
(``LMW_SMOKE_LOCK_PROBE=1``): the script takes the lock, prints
``smoke: lock held <path>`` and exits 0 without building or launching
anything; ``LMW_SMOKE_LOCK_PROBE=hold`` instead idles after taking the lock
so a test can kill the lock holder underneath it. None of them point the lock
at ``/Users/Shared``.

The interpreter is ``sys.executable``: the probe exits before the script's
``import app.main`` check, so it need not import the app.

The lock watcher and the teardown find the lock holder and the smoke's app
processes with ``ps`` (``-o ppid=``, ``-axo``). The CI deps image
(python:3.11-slim) has no ``ps``, so the hold/teardown tests skip there with a
reason; they run on the bare linux host and the M1, where ``ps`` is present.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "macos" / "smoke-bundle.sh"

# Without ps the watcher cannot read the smoke shell's ppid and the teardown
# cannot list the app's pids, so these legs would test nothing real.
needs_ps = pytest.mark.skipif(
    shutil.which("ps") is None,
    reason="the lock watcher and teardown need ps (absent from the CI deps image; runs on the bare host and the M1)",
)


@pytest.fixture()
def app_python() -> str:
    """The absolute interpreter passed as --python (the probe never imports the app)."""
    return sys.executable


def _probe_env(tmp_path: Path, lock: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["LMW_GUI_SMOKE_LOCK"] = str(lock)
    env["LMW_SMOKE_LOCK_PROBE"] = "1"
    return env


def _run_probe(lock: Path, python: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(SCRIPT), "--python", python],
        cwd=REPO,
        env=_probe_env(lock.parent, lock),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_probe_takes_the_lock(tmp_path: Path, app_python: str) -> None:
    lock = tmp_path / "lmw-gui-smoke.lock"
    done = _run_probe(lock, app_python)
    assert f"smoke: lock held {lock}" in done.stdout, (done.returncode, done.stdout, done.stderr)
    assert done.returncode == 0, done.stderr
    assert lock.is_file()


def test_waits_while_lock_is_held(tmp_path: Path, app_python: str) -> None:
    import fcntl
    import time

    lock = tmp_path / "lmw-gui-smoke.lock"
    held = open(lock, "a+")
    fcntl.flock(held.fileno(), fcntl.LOCK_EX)
    proc = subprocess.Popen(
        ["/bin/bash", str(SCRIPT), "--python", app_python],
        cwd=REPO,
        env=_probe_env(tmp_path, lock),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(2)
        # Still blocked on the flock: not exited, and nothing on stdout yet
        # (a non-blocking read of the raw pipe: no "lock held" before release).
        assert proc.poll() is None, "script exited before the lock was released"
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        os.set_blocking(fd, False)
        try:
            early = os.read(fd, 4096)
        except BlockingIOError:
            early = b""
        os.set_blocking(fd, True)
        assert b"lock held" not in early, early
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        out, _ = proc.communicate(timeout=30)
        assert proc.returncode == 0
        assert f"smoke: lock held {lock}" in early.decode() + out
    finally:
        if proc.poll() is None:
            proc.kill()
        held.close()


def test_lock_file_created_world_writable(tmp_path: Path, app_python: str) -> None:
    lock = tmp_path / "lmw-gui-smoke.lock"
    assert not lock.exists()
    done = _run_probe(lock, app_python)
    assert done.returncode == 0, done.stderr
    assert stat.S_IMODE(lock.stat().st_mode) == 0o666


def test_bad_python_exits_before_lock(tmp_path: Path) -> None:
    for py in ("python3", "/nope/no-such-python"):
        lock = tmp_path / f"lmw-{py.replace('/', '-')}.lock"
        assert not lock.exists()
        done = subprocess.run(
            ["/bin/bash", str(SCRIPT), "--python", py],
            cwd=REPO,
            env=_probe_env(tmp_path, lock),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert done.returncode == 2, (py, done.returncode, done.stderr)
        assert not lock.exists(), f"lock created for bad interpreter {py}"


def test_lock_marker_not_inherited() -> None:
    # The re-exec marker must be cleared after the re-exec line and before the
    # app-import check, so the app never inherits it.
    text = SCRIPT.read_text()
    exec_i = text.index('exec "$python" -I')
    unset_i = text.index("unset _LMW_SMOKE_LOCK_HELD")
    import_i = text.index("import app.main")
    assert exec_i < unset_i < import_i


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_is_free(lock: Path) -> bool:
    import fcntl

    with open(lock, "a+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return True


def _start_probe(
    tmp_path: Path, app_python: str, mode: str, want: tuple[str, ...]
) -> tuple[subprocess.Popen[bytes], Path, dict[str, int], bytes]:
    """Start LMW_SMOKE_LOCK_PROBE=<mode>; return (holder proc, lock, printed pids, stdout so far).

    ``want`` names the ``smoke: probe <name> pid <pid>`` lines to wait for
    (``""`` is the smoke shell's own ``smoke: probe pid <pid>``).
    """
    import select
    import time

    lock = tmp_path / "lmw-gui-smoke.lock"
    env = _probe_env(tmp_path, lock)
    env["LMW_SMOKE_LOCK_PROBE"] = mode
    proc = subprocess.Popen(
        ["/bin/bash", str(SCRIPT), "--python", app_python],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    buf = b""
    pids: dict[str, int] = {}
    deadline = time.monotonic() + 30
    while any(w not in pids for w in want) and time.monotonic() < deadline:
        ready, _, _ = select.select([proc.stdout], [], [], 1)
        if ready:
            chunk = os.read(proc.stdout.fileno(), 4096)
            if not chunk:
                break
            buf += chunk
        for line in buf.decode().splitlines():
            m = re.fullmatch(r"smoke: probe (?:(\w+) )?pid (\d+)", line)
            if m:
                pids[m.group(1) or ""] = int(m.group(2))
    if any(w not in pids for w in want):
        proc.kill()
        proc.wait()
        proc.stdout.close()
        raise AssertionError(buf)
    return proc, lock, pids, buf


def _start_hold(tmp_path: Path, app_python: str) -> tuple[subprocess.Popen[bytes], Path, int, int]:
    """Start LMW_SMOKE_LOCK_PROBE=hold; return (holder proc, lock, smoke shell pid, watcher pid)."""
    proc, lock, pids, _ = _start_probe(tmp_path, app_python, "hold", ("", "watch"))
    return proc, lock, pids[""], pids["watch"]


def _reap(proc: subprocess.Popen[bytes], *pids: int) -> None:
    """Kill only the processes this test started (the holder and the pids it printed)."""
    import signal

    if proc.poll() is None:
        proc.kill()
        proc.wait()
    if proc.stdout is not None:
        proc.stdout.close()
    for pid in pids:
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)


@needs_ps
def test_smoke_aborts_when_lock_holder_dies(tmp_path: Path, app_python: str) -> None:
    """Killing only the lock holder must not leave the smoke running unlocked.

    LMW_SMOKE_LOCK_PROBE=hold takes the lock, prints the smoke shell's pid and
    then idles like a long smoke would. The holder stays alive for a few watcher
    polls first (a healthy smoke is never aborted, and the watcher keeps
    polling past its first look). Then the test SIGKILLs the lock holder
    (``portable.py lock``, the pid ``bash`` exec'd into), which frees the flock
    at once; the smoke shell must notice and exit, so a second smoke cannot
    start while the orphan still runs.
    """
    import signal
    import time

    proc, lock, smoke_pid, watch_pid = _start_hold(tmp_path, app_python)
    try:
        assert smoke_pid != proc.pid  # the smoke shell is the lock holder's child
        time.sleep(2.5)  # several 1 s watcher polls with a live holder
        assert _alive(smoke_pid), "watcher aborted a smoke whose lock holder is alive"
        assert _alive(watch_pid), "watcher stopped polling while the smoke runs"
        assert not _lock_is_free(lock), "lock not held while the smoke runs"

        os.kill(proc.pid, signal.SIGKILL)  # the lock holder only
        proc.wait(timeout=10)

        deadline = time.monotonic() + 10
        while _alive(smoke_pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        free = _lock_is_free(lock)
        assert not _alive(
            smoke_pid
        ), f"smoke pid {smoke_pid} still running after its lock holder died (lock free: {free})"
    finally:
        _reap(proc, smoke_pid, watch_pid)


@needs_ps
def test_lock_watcher_exits_when_the_smoke_shell_is_killed(tmp_path: Path, app_python: str) -> None:
    """A SIGKILLed smoke shell runs no teardown; its watcher must still stop on its own."""
    import signal
    import time

    proc, lock, smoke_pid, watch_pid = _start_hold(tmp_path, app_python)
    try:
        assert _alive(watch_pid)
        os.kill(smoke_pid, signal.SIGKILL)
        proc.wait(timeout=10)  # the holder sees its child die and releases the lock
        deadline = time.monotonic() + 5
        while _alive(watch_pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not _alive(watch_pid), f"lock watcher {watch_pid} still polling a dead smoke shell"
        assert _lock_is_free(lock)
    finally:
        _reap(proc, smoke_pid, watch_pid)


def _wait_dead(pid: int, limit: float) -> bool:
    import time

    deadline = time.monotonic() + limit
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    return not _alive(pid)


@needs_ps
def test_lost_lock_teardown_kills_own_app_but_keeps_shared_state(
    tmp_path: Path, app_python: str
) -> None:
    """A smoke that lost its lock kills its own app but leaves the shared smoke state alone.

    The shared Keychain item, defaults domain and data dir belong to whichever
    smoke holds the lock now: a new smoke may have taken it in the second the
    watcher needs to notice, so wiping them would corrupt that run. The next
    smoke cleans them at its start anyway (test_next_smoke_cleans_state_at_start).
    LMW_SMOKE_LOCK_PROBE=teardown installs the real teardown, starts a stand-in
    app process that app_pids() finds by the smoke executable path, and idles.
    """
    import signal

    proc, lock, pids, buf = _start_probe(tmp_path, app_python, "teardown", ("", "watch", "app"))
    smoke_pid, watch_pid, app_pid = pids[""], pids["watch"], pids["app"]
    try:
        assert _alive(app_pid)
        os.kill(proc.pid, signal.SIGKILL)  # the lock holder only
        proc.wait(timeout=10)
        assert _wait_dead(
            smoke_pid, 10
        ), f"smoke pid {smoke_pid} still running after its lock holder died"
        assert proc.stdout is not None
        out = (buf + proc.stdout.read()).decode()
        assert _wait_dead(app_pid, 5), f"teardown left its own app pid {app_pid} running"
        assert "cleaning the shared smoke state" not in out, out
        assert "smoke: lock lost" in out, out
    finally:
        _reap(proc, smoke_pid, watch_pid, app_pid)


@needs_ps
def test_locked_teardown_cleans_shared_state(tmp_path: Path, app_python: str) -> None:
    """With the lock still held, an aborted smoke's teardown kills its app and cleans the state."""
    import signal

    proc, lock, pids, buf = _start_probe(tmp_path, app_python, "teardown", ("", "watch", "app"))
    smoke_pid, watch_pid, app_pid = pids[""], pids["watch"], pids["app"]
    try:
        os.kill(smoke_pid, signal.SIGTERM)
        assert proc.wait(timeout=15) == 1  # the holder passes on the smoke's exit status
        assert proc.stdout is not None
        out = (buf + proc.stdout.read()).decode()
        assert _wait_dead(app_pid, 5), f"teardown left its own app pid {app_pid} running"
        assert "cleaning the shared smoke state" in out, out
        assert "smoke: lock lost" not in out, out
        assert _lock_is_free(lock)
    finally:
        _reap(proc, smoke_pid, watch_pid, app_pid)


@needs_ps
def test_unreadable_parent_teardown_keeps_shared_state(
    tmp_path: Path, app_python: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A teardown that cannot read its own ppid treats the lock as lost.

    A ``ps`` shim on PATH starts failing every ``-o ppid=`` query once a flag
    file exists (the holder is still alive and still holds the flock). The
    teardown must then kill its app and skip cleanup_smoke_state: it cannot
    prove it still holds the lock, and the next smoke cleans at its start.
    """
    import signal

    shim = tmp_path / "shim"
    shim.mkdir()
    flag = tmp_path / "ps-fails"
    (shim / "ps").write_text(
        "#!/bin/bash\n"
        f"if [ -e '{flag}' ]; then case \" $* \" in *ppid=*) exit 1 ;; esac; fi\n"
        'exec /bin/ps "$@"\n'
    )
    (shim / "ps").chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ.get('PATH', '')}")
    proc, lock, pids, buf = _start_probe(tmp_path, app_python, "teardown", ("", "watch", "app"))
    smoke_pid, watch_pid, app_pid = pids[""], pids["watch"], pids["app"]
    try:
        flag.touch()
        os.kill(smoke_pid, signal.SIGTERM)
        assert proc.wait(timeout=15) == 1
        assert proc.stdout is not None
        out = (buf + proc.stdout.read()).decode()
        assert _wait_dead(app_pid, 5), f"teardown left its own app pid {app_pid} running"
        assert "cleaning the shared smoke state" not in out, out
        assert "smoke: lock lost" in out, out
    finally:
        _reap(proc, smoke_pid, watch_pid, app_pid)


def test_next_smoke_cleans_state_at_start() -> None:
    """A lost-lock teardown may skip the cleanup only because every smoke cleans at its start.

    The start-of-run cleanup_smoke_state call must run under the lock (after
    the re-exec and the teardown trap) and before the build or any launch.
    """
    text = SCRIPT.read_text()
    exec_i = text.index('exec "$python" -I')
    trap_i = text.index("trap teardown EXIT")
    start_i = text.index("\ncleanup_smoke_state\n", trap_i)
    build_i = text.index('log "swift build -c release"')
    launch_i = text.index("\nlaunch_and_probe\n")
    assert exec_i < trap_i < start_i < build_i < launch_i
