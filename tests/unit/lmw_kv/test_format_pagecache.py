from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from lmw_kv import format as format_
from lmw_kv.format import HEADER_LEN, SegmentWriter, read_record
from tests.live.kv.pagecache import fs_type, resident_bytes

NS = b"N" * 32
KEY = b"\x02" * 32
BK1 = b"\x11" * 32
BK2 = b"\x22" * 32


def test_append_waits_for_writeback_then_drops_each_record(tmp_path, monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(format_, "_sync_range", lambda fd, o, n: calls.append(("sync", o, n)))
    monkeypatch.setattr(
        os, "posix_fadvise", lambda fd, o, n, a: calls.append(("adv", o, n, a)), raising=False
    )
    w = SegmentWriter(tmp_path, NS, KEY)
    a = w.append(BK1, "g", b"x" * 5000)
    b = w.append(BK2, "g", b"y" * 7000)
    i = calls.index(("sync", a.offset, a.length))
    assert calls[i + 1] == ("adv", a.offset, a.length, os.POSIX_FADV_DONTNEED)
    j = calls.index(("sync", b.offset, b.length))
    assert calls[j + 1] == ("adv", b.offset, b.length, os.POSIX_FADV_DONTNEED)


def test_roll_drops_the_header_after_writeback(tmp_path, monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(format_, "_sync_range", lambda fd, o, n: calls.append(("sync", o, n)))
    monkeypatch.setattr(
        os, "posix_fadvise", lambda fd, o, n, a: calls.append(("adv", o, n, a)), raising=False
    )
    w = SegmentWriter(tmp_path, NS, KEY)
    w.append(BK1, "g", b"x")
    assert calls[:2] == [
        ("sync", 0, HEADER_LEN),
        ("adv", 0, HEADER_LEN, os.POSIX_FADV_DONTNEED),
    ]


def test_read_record_reads_the_range_and_drops_it(tmp_path, monkeypatch):
    w = SegmentWriter(tmp_path, NS, KEY)
    loc = w.append(BK1, "g", b"payload" * 100)
    w.close()
    calls: list[tuple] = []
    monkeypatch.setattr(format_, "_sync_range", lambda fd, o, n: calls.append(("sync", o, n)))
    monkeypatch.setattr(
        os, "posix_fadvise", lambda fd, o, n, a: calls.append(("adv", o, n, a)), raising=False
    )
    assert read_record(tmp_path, loc, KEY, BK1, "g") == b"payload" * 100
    assert ("adv", 0, 0, os.POSIX_FADV_RANDOM) in calls
    assert ("adv", 0, HEADER_LEN, os.POSIX_FADV_DONTNEED) in calls
    assert ("adv", loc.offset, loc.length, os.POSIX_FADV_DONTNEED) in calls
    assert not any(c[0] == "sync" for c in calls)


def test_sync_range_falls_back_to_fdatasync(tmp_path, monkeypatch):
    monkeypatch.setattr(format_, "_LIBC_SYNC_FILE_RANGE", None)
    seen: list[int] = []
    monkeypatch.setattr(os, "fdatasync", lambda fd: seen.append(fd))
    format_._sync_range(3, 0, 10)
    assert seen == [3]


def test_drop_range_never_raises(tmp_path, monkeypatch):
    def boom(*a):
        raise OSError("nope")

    monkeypatch.setattr(format_, "_sync_range", boom)
    monkeypatch.setattr(os, "posix_fadvise", boom, raising=False)
    format_.drop_range(3, 0, 10, written=True)
    format_.drop_range(3, 0, 10, written=False)


def test_without_fadvise_records_still_round_trip(tmp_path, monkeypatch):
    monkeypatch.delattr(os, "posix_fadvise", raising=False)
    w = SegmentWriter(tmp_path, NS, KEY)
    loc = w.append(BK1, "g", b"hello")
    assert read_record(tmp_path, loc, KEY, BK1, "g") == b"hello"
    w.close()
    assert read_record(tmp_path, loc, KEY, BK1, "g") == b"hello"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux page cache")
def test_written_and_read_segments_are_not_resident(tmp_path: Path):
    if fs_type(tmp_path) in ("tmpfs", "overlay", "ramfs"):
        pytest.skip("needs a disk-backed filesystem")
    w = SegmentWriter(tmp_path, NS, KEY)
    locs = [w.append(bytes([i]) * 32, "g", os.urandom(1 << 20)) for i in range(32)]
    for i, loc in enumerate(locs):
        assert read_record(tmp_path, loc, KEY, bytes([i]) * 32, "g") is not None
    assert sum(resident_bytes(p) for p in tmp_path.glob("*.seg")) <= 1 << 20


def test_read_record_into_reads_the_range_and_drops_it(tmp_path, monkeypatch):
    from lmw_kv.format import ReadBuffer, read_record_into

    w = SegmentWriter(tmp_path, NS, KEY)
    loc = w.append(BK1, "g", b"payload" * 100)
    w.close()
    calls: list[tuple] = []
    monkeypatch.setattr(
        os, "posix_fadvise", lambda fd, o, n, a: calls.append(("adv", o, n, a)), raising=False
    )
    out = bytearray(700)
    assert read_record_into(tmp_path, loc, KEY, BK1, "g", memoryview(out), ReadBuffer()) == 700
    assert bytes(out) == b"payload" * 100
    assert ("adv", 0, 0, os.POSIX_FADV_RANDOM) in calls
    assert ("adv", 0, HEADER_LEN, os.POSIX_FADV_DONTNEED) in calls
    assert ("adv", loc.offset, loc.length, os.POSIX_FADV_DONTNEED) in calls


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux page cache")
def test_records_read_into_a_buffer_are_not_resident(tmp_path: Path):
    from lmw_kv.format import ReadBuffer, read_record_into

    if fs_type(tmp_path) in ("tmpfs", "overlay", "ramfs"):
        pytest.skip("needs a disk-backed filesystem")
    w = SegmentWriter(tmp_path, NS, KEY)
    locs = [w.append(bytes([i]) * 32, "g", os.urandom(1 << 20)) for i in range(32)]
    w.close()
    buf, out = ReadBuffer(), memoryview(bytearray(1 << 20))
    for i, loc in enumerate(locs):
        assert read_record_into(tmp_path, loc, KEY, bytes([i]) * 32, "g", out, buf) == 1 << 20
    assert sum(resident_bytes(p) for p in tmp_path.glob("*.seg")) <= 1 << 20
