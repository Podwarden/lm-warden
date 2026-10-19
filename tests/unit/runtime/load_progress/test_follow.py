"""Tests for ``LogFollower`` — Task 3 of the model-load-progress plan.

The follower starts at the CURRENT END of the (reused) engine log file, returns
new text on each ``read_new()``, reopens from offset 0 on size drop or inode
change, and tolerates a missing file (returns "" and keeps trying). One read is
capped at 4 MiB so a burst cannot balloon memory; the remainder is held for the
next call.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.runtime.load_progress.follow import LogFollower

# 4 MiB read cap, verbatim from the brief.
_READ_CAP = 4 * 1024 * 1024


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _append(path: Path, text: str | bytes) -> None:
    if isinstance(text, bytes):
        with open(path, "ab") as f:
            f.write(text)
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)


def _inode(path: Path) -> int | None:
    try:
        return os.stat(path).st_ino
    except OSError:
        return None


def test_starts_at_end_of_reused_log(tmp_path: Path) -> None:
    """A log file is reused across runs (run_marker.py). The follower must
    start at the file's CURRENT END so a stale tail from a previous run is
    not re-read on the first call."""
    log = tmp_path / "engine.log"
    _write(log, "old run line 1\nold run line 2\n")
    f = LogFollower(log)
    # First read at end-of-file returns nothing — no new text yet.
    assert f.read_new() == ""
    # Append new content; that is what the next call must return.
    _append(log, "new line A\nnew line B\n")
    assert f.read_new() == "new line A\nnew line B\n"


def test_reopens_after_truncation(tmp_path: Path) -> None:
    """Truncation (size drop) must reopen from offset 0 — a fresh run starting
    from scratch, not from a now-undersized offset."""
    log = tmp_path / "engine.log"
    _write(log, "first content\n")
    f = LogFollower(log)
    _append(log, "appended\n")
    assert f.read_new() == "appended\n"

    # Truncate and rewrite smaller.
    _write(log, "fresh\n")
    # Size dropped below our last offset → reopen from 0 → read the new body.
    assert f.read_new() == "fresh\n"


def test_reopens_after_rotation_rename(tmp_path: Path) -> None:
    """logrotate moves the file aside and creates a new one. The inode
    changes, so the follower must reopen the new file from offset 0."""
    log = tmp_path / "engine.log"
    _write(log, "old body line\n")
    f = LogFollower(log)
    assert f.read_new() == ""

    # Rotate: rename old, create a brand-new file at the same path.
    rotated = tmp_path / "engine.log.1"
    os.rename(log, rotated)
    _write(log, "brand new body\n")

    assert _inode(log) != _inode(rotated)
    assert f.read_new() == "brand new body\n"


def test_follower_waits_for_missing_file(tmp_path: Path) -> None:
    """If the file does not exist yet (engine not started, log path chosen
    but no file written), read_new() returns "" and does not raise — the
    caller keeps polling until the file appears."""
    log = tmp_path / "does_not_exist_yet.log"
    assert not log.exists()
    f = LogFollower(log)
    assert f.read_new() == ""
    assert f.read_new() == ""

    # File appears later; the next call picks it up from offset 0.
    _write(log, "first line after spawn\n")
    assert f.read_new() == "first line after spawn\n"


def test_follower_caps_one_read_at_4mib(tmp_path: Path) -> None:
    """One read_new() must cap at 4 MiB. The remainder is held for the next
    call, so a burst cannot balloon memory but is not lost."""
    log = tmp_path / "engine.log"
    f = LogFollower(log)

    # Write 10 MiB in one go (more than one cap).
    chunk = "A" * (10 * 1024 * 1024)
    _write(log, chunk)

    first = f.read_new()
    assert len(first) == _READ_CAP, f"first read capped at {_READ_CAP}, got {len(first)}"
    # The remainder (10 MiB - 4 MiB) is held; subsequent calls drain it.
    second = f.read_new()
    assert len(second) == _READ_CAP
    third = f.read_new()
    assert len(third) == 10 * 1024 * 1024 - 2 * _READ_CAP
    # And a final call returns nothing more.
    assert f.read_new() == ""


def test_follower_preserves_carriage_returns(tmp_path: Path) -> None:
    """tqdm progress lines use ``\\r`` to repaint. The follower must keep
    ``\\r`` and pass it through verbatim (decoded UTF-8 with errors="replace")."""
    log = tmp_path / "engine.log"
    f = LogFollower(log)
    _write(log, "loading:  3%|██        | 3/100 [00:01<00:32,  3.0it/s]\r")
    assert f.read_new() == "loading:  3%|██        | 3/100 [00:01<00:32,  3.0it/s]\r"


def test_follower_replaces_invalid_utf8(tmp_path: Path) -> None:
    """Engine logs sometimes contain stray bytes (binary-ish CUDA output).
    Decode with errors="replace" so a bad byte never raises."""
    log = tmp_path / "engine.log"
    f = LogFollower(log)
    with open(log, "wb") as fp:
        fp.write(b"good\xff\xfe bytes\n")
    out = f.read_new()
    assert "good" in out and "bytes" in out
    # No exception; replacement chars are acceptable.
    assert isinstance(out, str)


def test_multibyte_char_split_across_reads_is_intact(tmp_path: Path) -> None:
    """A UTF-8 multibyte character that straddles two reads (because the engine
    has not finished writing it yet) must be reassembled, not split into two
    U+FFFD. The follower holds incomplete trailing bytes in an incremental
    decoder for the next read.

    U+3000 (IDEOGRAPHIC SPACE) is ``\\xe3\\x80\\x80`` — a 3-byte sequence. We
    write the first two bytes, read, then append the third plus more text.
    """
    log = tmp_path / "engine.log"
    f = LogFollower(log)
    # Write "ab" + first two bytes of U+3000.
    with open(log, "wb") as fp:
        fp.write(b"ab\xe3\x80")
    first = f.read_new()
    assert first == "ab", f"incomplete trailing bytes held, not emitted yet; got {first!r}"
    # Append the third byte of U+3000 + "cd".
    _append(log, b"\x80cd")
    second = f.read_new()
    assert second == "　cd", f"split char reassembled; got {second!r}"
    # Concatenated reads == the full text, no U+FFFD anywhere.
    full = first + second
    assert full == "ab　cd"
    assert "�" not in full


def test_multibyte_split_at_4mib_cap(tmp_path: Path) -> None:
    """A 3-byte UTF-8 character straddling the 4 MiB read-cap boundary must
    survive intact across the two reads — the first read holds the incomplete
    trailing bytes for the second."""
    log = tmp_path / "engine.log"
    f = LogFollower(log)
    # Build content: (4 MiB - 1) bytes of 'a', then U+3000 (3 bytes), then "cd".
    # The first read caps at 4 MiB: it gets 4 MiB of 'a' + the first byte of
    # U+3000. The incremental decoder holds that one byte; the next read gets
    # the remaining 2 bytes + "cd" and reassembles U+3000.
    char = "　"  # \xe3\x80\x80
    prefix = "a" * (_READ_CAP - 1)
    payload = prefix + char + "cd"
    _write(log, payload)

    first = f.read_new()
    assert len(first) == _READ_CAP - 1, (
        f"first read returns complete chars only (incomplete trailing byte "
        f"held); got {len(first)} chars"
    )
    assert first == prefix
    second = f.read_new()
    assert second == char + "cd", f"split char reassembled on second read; got {second!r}"
    full = first + second
    assert full == payload
    assert "�" not in full


def test_follower_decoder_resets_on_reopen(tmp_path: Path) -> None:
    """When the follower reopens from offset 0 (truncation/rotation), the
    incremental decoder must be RESET so leftover bytes from the previous
    file do not corrupt the new file's first characters."""
    log = tmp_path / "engine.log"
    f = LogFollower(log)
    # Write content ending with an incomplete multibyte sequence.
    with open(log, "wb") as fp:
        fp.write(b"ab\xe3\x80")
    first = f.read_new()
    assert first == "ab"  # incomplete bytes held in decoder

    # Truncate to a size smaller than our offset (4) so the size-drop
    # detection fires and the follower reopens from 0 with a decoder reset.
    _write(log, "hi\n")  # 3 bytes < 4 byte offset → reopen from 0
    second = f.read_new()
    assert second == "hi\n", f"decoder reset on reopen; got {second!r}"
    assert "�" not in second


def test_close_closes_the_handle(tmp_path: Path) -> None:
    """``close()`` closes the underlying file handle deterministically."""
    log = tmp_path / "engine.log"
    _write(log, "existing content\n")
    f = LogFollower(log)
    # The follower starts at the file's current end, so the first read
    # returns "".  Append new content so the next read opens the handle.
    _append(log, "new content\n")
    assert f.read_new() == "new content\n"
    # The internal handle is now open.
    assert f._fp is not None
    f.close()
    # The handle is closed.
    assert f._fp is None


def test_close_is_safe_to_call_twice(tmp_path: Path) -> None:
    """``close()`` must be a no-op when already closed — safe to call from a
    ``finally`` cleanup path that may run after the follower was already
    closed on a clean exit."""
    log = tmp_path / "engine.log"
    _write(log, "existing\n")
    f = LogFollower(log)
    _append(log, "new\n")
    f.read_new()
    f.close()
    # Second call must not raise.
    f.close()
    assert f._fp is None


def test_close_safe_when_file_never_opened(tmp_path: Path) -> None:
    """``close()`` on a follower whose file never existed (handle never
    opened) must be a no-op, not raise."""
    log = tmp_path / "nonexistent.log"
    f = LogFollower(log)
    assert f._fp is None
    f.close()
    assert f._fp is None


def test_read_new_after_close_reopens_gracefully(tmp_path: Path) -> None:
    """After ``close()``, ``read_new`` reopens the file from the start offset
    (the file's size at construction time) if the file still exists, and
    returns any new content written since.  If there is no new content, it
    returns "".  This makes ``close`` safe to call from a ``finally`` cleanup
    path — a subsequent ``read_new`` (if it ever happens) degrades to a
    graceful reopen, not a crash."""
    log = tmp_path / "engine.log"
    # Start with existing content; the follower captures the size as
    # _start_offset.
    _write(log, "existing\n")
    f = LogFollower(log)
    # Read new content to open the handle.
    _append(log, "chunk A\n")
    assert f.read_new() == "chunk A\n"
    f.close()
    assert f._fp is None
    # After close, read_new reopens from _start_offset (the size at
    # construction = 9 bytes for "existing\n").  "chunk A\n" was appended
    # after construction, so it is past _start_offset.  But we already read
    # it before closing — the offset advanced to 9 + 8 = 17.  After close,
    # _fp is None and _offset is still 17 (close does not reset it).  The
    # next read_new sees _fp is None → opens from _start_offset (9), not
    # from _offset (17).  So it re-reads from 9 and returns "chunk A\n"
    # again.  This is the documented behaviour: read_new after close
    # reopens from the start offset and may re-read content.  In practice
    # the watcher is never resumed after cleanup, so this is a safety net.
    result = f.read_new()
    # The reopen from _start_offset re-reads "chunk A\n" (the content
    # between _start_offset and the current end).
    assert result == "chunk A\n"
