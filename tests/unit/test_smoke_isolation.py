"""The bundle smoke proves isolation from ITS OWN processes, not by diffing the
operator's real data dir (F7).

The old smoke took a whole-tree ``find ... stat`` snapshot of the real
``~/Library/Application Support/LM Warden`` before it launched, then asserted
the snapshot was unchanged afterwards. An installed LM Warden.app running on the
M1 (or a ``git`` checkout of the repo that lives under that dir) writes
``vllm-warden.db`` there at any moment, so the gate went red on the
environment, not the code.

Instead the smoke now proves the property it actually cares about: that the
smoke's own processes never touch the real dir. Two checks do that.

``smoke-isolation.sh child-data-dir <recorded> <expected> <real>``: the server
child records the data dir it was launched with (``VW_DATA_DIR``); this asserts
that recorded dir is exactly the smoke's dir, never the real one (and never
under it). The wrapper writes it to a 0600 file.

``smoke-isolation.sh open-under <real> <pid>...``: a process that has a file
open under the real dir is using it. This lists every open path of the named
pids via ``lsof`` and fails if any is the real dir or under it. A sibling whose
name only shares the real dir's prefix (``<real> Backup``, ``<real>2``) is not a
hit, and a pid that has already exited is clean, not an error.

These tests drive both subcommands as a CLI (``/bin/bash``) and assert the
static invariants of ``smoke-bundle.sh``. None of them needs the GUI smoke to
run.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "scripts" / "macos" / "smoke-isolation.sh"
BUNDLE = REPO / "scripts" / "macos" / "smoke-bundle.sh"

needs_lsof = pytest.mark.skipif(
    shutil.which("lsof") is None,
    reason="open-under inspects open file descriptors with lsof (absent from the CI deps image; runs on the bare host and the M1)",
)


def _helper(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(HELPER), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _child_data_dir(
    tmp_path: Path, value: str | None, expected: Path, real: Path
) -> subprocess.CompletedProcess[str]:
    """Record ``value`` (or no file at all when ``value`` is None) and check it."""
    recorded = tmp_path / "child.datadir"
    if value is not None:
        recorded.write_text(value)
    return _helper("child-data-dir", str(recorded), str(expected), str(real))


def _start_standin(
    tmp_path: Path, open_under: Path | None, tag: str
) -> tuple[subprocess.Popen[bytes], Path | None]:
    """Start a stand-in that, when given, holds a file open under ``open_under``.

    The stand-in stands in for the server child: it is a long-lived process that
    (like the real child) keeps a file in its data dir open. ``open_under`` is
    the dir to create and keep a file open in (the smoke dir for the clean leg,
    the real dir for the leak leg). Returns (proc, held_path_or_None). The held
    file is opened by this process (not the child) and kept open for the child's
    lifetime via an inherited fd, so lsof lists it on the stand-in deterministically.
    """
    import time

    if open_under is not None:
        open_under.mkdir(parents=True, exist_ok=True)
        hold = open_under / f"hold-{tag}.dat"
        fh = hold.open("a")
        # Open the file in THIS process, then fork the stand-in with the fd
        # inherited (pass_fds) so lsof -p <pid> lists hold under open_under for
        # the child's whole life. Close our copy so only the stand-in holds it.
        proc = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(fh.fileno(),),
        )
        fh.close()
        time.sleep(0.3)
        return proc, hold
    proc = subprocess.Popen(["sleep", "30"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.2)
    return proc, None


def _stop(proc: subprocess.Popen[bytes]) -> None:
    proc.kill()
    proc.wait()


# --- (a) child-data-dir ------------------------------------------------------


def test_child_datadir_exact_expected_passes(tmp_path: Path) -> None:
    expected = (tmp_path / "smoke" / "LM Warden").resolve()
    real = (tmp_path / "real").resolve()
    done = _child_data_dir(tmp_path, str(expected), expected, real)
    assert done.returncode == 0, (done.returncode, done.stdout, done.stderr)


def test_child_datadir_missing_file_fails(tmp_path: Path) -> None:
    expected = (tmp_path / "smoke" / "LM Warden").resolve()
    real = (tmp_path / "real").resolve()
    done = _child_data_dir(tmp_path, None, expected, real)
    assert done.returncode == 1, (done.returncode, done.stdout, done.stderr)
    assert "smoke-isolation:" in done.stderr, done.stderr


def test_child_datadir_empty_file_fails(tmp_path: Path) -> None:
    expected = (tmp_path / "smoke" / "LM Warden").resolve()
    real = (tmp_path / "real").resolve()
    done = _child_data_dir(tmp_path, "", expected, real)
    assert done.returncode == 1, (done.returncode, done.stdout, done.stderr)
    assert "smoke-isolation:" in done.stderr, done.stderr


def test_child_datadir_is_real_dir_fails(tmp_path: Path) -> None:
    expected = (tmp_path / "smoke" / "LM Warden").resolve()
    real = (tmp_path / "real").resolve()
    done = _child_data_dir(tmp_path, str(real), expected, real)
    assert done.returncode == 1, (done.returncode, done.stdout, done.stderr)
    assert "smoke-isolation:" in done.stderr, done.stderr


def test_child_datadir_is_under_real_dir_fails(tmp_path: Path) -> None:
    expected = (tmp_path / "smoke" / "LM Warden").resolve()
    real = (tmp_path / "real").resolve()
    done = _child_data_dir(tmp_path, str(real / "sub"), expected, real)
    assert done.returncode == 1, (done.returncode, done.stdout, done.stderr)
    assert "smoke-isolation:" in done.stderr, done.stderr


def test_child_datadir_trailing_slash_is_not_exact(tmp_path: Path) -> None:
    expected = (tmp_path / "smoke" / "LM Warden").resolve()
    real = (tmp_path / "real").resolve()
    # The exact value with a trailing slash is not the recorded dir: a dir is
    # recorded without one, so this must be rejected as a mismatch.
    done = _child_data_dir(tmp_path, str(expected) + "/", expected, real)
    assert done.returncode == 1, (done.returncode, done.stdout, done.stderr)
    assert "smoke-isolation:" in done.stderr, done.stderr


def test_child_datadir_other_dir_fails(tmp_path: Path) -> None:
    expected = (tmp_path / "smoke" / "LM Warden").resolve()
    real = (tmp_path / "real").resolve()
    done = _child_data_dir(tmp_path, str(tmp_path / "elsewhere"), expected, real)
    assert done.returncode == 1, (done.returncode, done.stdout, done.stderr)
    assert "smoke-isolation:" in done.stderr, done.stderr


def test_child_datadir_bad_usage_exits_2(tmp_path: Path) -> None:
    # Not exactly four arguments: usage, not a check failure.
    done = _helper("child-data-dir", str(tmp_path / "a"), str(tmp_path / "b"))
    assert done.returncode == 2, (done.returncode, done.stdout, done.stderr)


# --- (b) open-under ----------------------------------------------------------


@needs_lsof
def test_open_under_nothing_open_passes(tmp_path: Path) -> None:
    real = (tmp_path / "real").resolve()
    standin, _ = _start_standin(tmp_path, None, "clean")
    try:
        done = _helper("open-under", str(real), str(standin.pid))
        assert done.returncode == 0, (done.returncode, done.stdout, done.stderr)
    finally:
        _stop(standin)


@needs_lsof
def test_open_under_file_under_real_fails(tmp_path: Path) -> None:
    real = (tmp_path / "real").resolve()
    standin, hold = _start_standin(tmp_path, real / "nested", "leak")
    try:
        done = _helper("open-under", str(real), str(standin.pid))
        assert done.returncode == 1, (done.returncode, done.stdout, done.stderr)
        # Each offending path is named.
        assert str(hold) in done.stdout, (done.stdout, done.stderr)
    finally:
        _stop(standin)


@needs_lsof
def test_open_under_sibling_prefix_is_not_a_hit(tmp_path: Path) -> None:
    """A dir whose name only shares the real dir's prefix must not be flagged."""
    real = (tmp_path / "real").resolve()
    sibling = (tmp_path / "real Backup").resolve()
    standin, _ = _start_standin(tmp_path, sibling / "inner", "sib")
    try:
        done = _helper("open-under", str(real), str(standin.pid))
        assert done.returncode == 0, (done.returncode, done.stdout, done.stderr)
    finally:
        _stop(standin)


@needs_lsof
def test_open_under_dead_pid_is_clean(tmp_path: Path) -> None:
    import time

    real = (tmp_path / "real").resolve()
    # A pid that has exited: lsof returns non-zero with no output, which is
    # clean. Hold the pid briefly (sleep 0.3) so it is dead, not mid-reap.
    gone = subprocess.Popen(["sleep", "0.3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    gone.wait()
    done = _helper("open-under", str(real), str(gone.pid))
    assert done.returncode == 0, (done.returncode, done.stdout, done.stderr)


def test_open_under_bad_usage_exits_2(tmp_path: Path) -> None:
    real = (tmp_path / "real").resolve()
    # No pids: usage.
    done = _helper("open-under", str(real))
    assert done.returncode == 2, (done.returncode, done.stdout, done.stderr)
    # Relative dir: usage.
    done = _helper("open-under", "real", "1")
    assert done.returncode == 2, (done.returncode, done.stdout, done.stderr)


# --- (e) concurrent writer red-first leg -------------------------------------


@needs_lsof
def test_concurrent_writer_into_real_dir_stays_isolated(tmp_path: Path) -> None:
    """A process writing the real dir during the smoke must not make it red.

    The whole point of F7: a concurrent writer to the real data dir (an
    installed LM Warden writing vllm-warden.db) is the operator's traffic, not
    the smoke's. The smoke checks only its own processes, so it stays green:
    child-data-dir holds the smoke dir and the stand-in holds nothing under the
    real dir, even while another process appends to the real dir.
    """
    import time

    real = (tmp_path / "real").resolve()
    real.mkdir(parents=True)
    smoke = (tmp_path / "smoke" / "LM Warden").resolve()
    db = real / "vllm-warden.db"

    writer = subprocess.Popen(
        ["bash", "-c", f"for i in $(seq 1 50); do echo $i >> '{db}'; sleep 0.05; done"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    standin, _ = _start_standin(tmp_path, smoke, "concurrent")
    try:
        time.sleep(0.3)  # let the writer do a couple of appends
        datadir = tmp_path / "child.datadir"
        datadir.write_text(str(smoke))
        assert _child_data_dir(tmp_path, str(smoke), smoke, real).returncode == 0
        done = _helper("open-under", str(real), str(standin.pid))
        assert done.returncode == 0, (done.returncode, done.stdout, done.stderr)
        time.sleep(0.3)
    finally:
        _stop(standin)
        writer.kill()
        writer.wait()
    # The real dir was indeed written by the concurrent writer during the leg.
    assert db.exists() and db.stat().st_size > 0


# --- (g) static invariants + bash -n -----------------------------------------


def test_bundle_drops_snapshot_and_uses_helper() -> None:
    text = BUNDLE.read_text()
    assert "snapshot_real_data" not in text
    assert "child.datadir" in text
    assert "smoke-isolation.sh" in text
    # The old whole-dir before/after comparison is gone: no baseline capture and
    # no "changed" assertion on the real dir.
    assert "real_data_before" not in text
    assert "real data dir $real_data changed" not in text


def test_wrapper_records_vw_data_dir() -> None:
    text = BUNDLE.read_text()
    assert re.search(r"VW_DATA_DIR", text), "wrapper must record VW_DATA_DIR"


@pytest.mark.parametrize("script", [BUNDLE, HELPER], ids=["smoke-bundle", "smoke-isolation"])
def test_bundle_and_helper_pass_bash_n(script: Path) -> None:
    # One file per run: `bash -n a b` parses only a (b is just $1), so a
    # syntax error in the helper would pass a two-file call.
    done = subprocess.run(
        ["/bin/bash", "-n", str(script)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, (done.returncode, done.stdout, done.stderr)
