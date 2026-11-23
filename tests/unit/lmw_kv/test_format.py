from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from lmw_kv.format import Location, SegmentWriter, load_or_create_data_key, read_record

NS_A = b"A" * 32
NS_B = b"B" * 32
KEY = b"\x01" * 32


def _k(i: int) -> bytes:
    return bytes([i]) * 32


def test_round_trip(tmp_path: Path) -> None:
    w = SegmentWriter(tmp_path, NS_A, KEY)
    items = [(_k(1), "full", b"alpha" * 10), (_k(2), "swa", b"beta"), (_k(3), "full", b"")]
    locs = [w.append(k, g, p) for k, g, p in items]
    w.close()
    for (k, g, p), loc in zip(items, locs, strict=True):
        assert read_record(tmp_path, loc, KEY, k, g) == p


def test_wrong_expectations_are_a_miss(tmp_path: Path) -> None:
    w = SegmentWriter(tmp_path, NS_A, KEY)
    loc = w.append(_k(1), "full", b"data")
    w.close()
    assert read_record(tmp_path, loc, KEY, _k(2), "full") is None
    assert read_record(tmp_path, loc, KEY, _k(1), "swa") is None
    assert read_record(tmp_path, loc, b"\x02" * 32, _k(1), "full") is None


def test_tamper_is_a_miss(tmp_path: Path) -> None:
    w = SegmentWriter(tmp_path, NS_A, KEY)
    loc = w.append(_k(1), "full", b"payload-bytes")
    w.close()
    seg = tmp_path / "000001.seg"
    orig = seg.read_bytes()

    # flip a payload (ciphertext) byte; the trailing CRC is recomputed-free, so the
    # CRC check alone must already reject it, and so would the AEAD tag
    end = loc.offset + loc.length
    b = bytearray(orig)
    b[end - 4 - 20] ^= 0x01
    seg.write_bytes(bytes(b))
    assert read_record(tmp_path, loc, KEY, _k(1), "full") is None

    # flip the stored block key (just after the u32 length)
    b = bytearray(orig)
    b[loc.offset + 4] ^= 0x01
    seg.write_bytes(bytes(b))
    assert read_record(tmp_path, loc, KEY, _k(1), "full") is None

    seg.write_bytes(orig)
    assert read_record(tmp_path, loc, KEY, _k(1), "full") == b"payload-bytes"


def test_bad_header_or_missing_file_is_a_miss(tmp_path: Path) -> None:
    w = SegmentWriter(tmp_path, NS_A, KEY)
    loc = w.append(_k(1), "full", b"x")
    w.close()
    seg = tmp_path / "000001.seg"
    orig = seg.read_bytes()
    seg.write_bytes(b"XXXXXXXX" + orig[8:])
    assert read_record(tmp_path, loc, KEY, _k(1), "full") is None
    seg.write_bytes(orig[:-3])  # truncated
    assert read_record(tmp_path, loc, KEY, _k(1), "full") is None
    seg.unlink()
    assert read_record(tmp_path, loc, KEY, _k(1), "full") is None
    assert read_record(tmp_path, Location(9, 0, 10), KEY, _k(1), "full") is None


def test_wrong_namespace_is_a_miss(tmp_path: Path) -> None:
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    wa = SegmentWriter(dir_a, NS_A, KEY)
    loc_a = wa.append(_k(1), "full", b"secret")
    wa.close()
    wb = SegmentWriter(dir_b, NS_B, KEY)
    wb.append(_k(9), "full", b"other-ns-data")
    wb.close()
    # ns A's location read against ns B's file set: never returns data
    assert read_record(dir_b, loc_a, KEY, _k(1), "full") is None
    # the same bytes moved under B's file name still do not decrypt for B's key
    (dir_b / "000001.seg").write_bytes((dir_a / "000001.seg").read_bytes())
    assert read_record(dir_b, loc_a, KEY, _k(1), "full") == b"secret"  # same file, same ns A
    # splice A's record into a file whose header says B: AAD (ns id) mismatches
    hdr_b = bytearray((dir_a / "000001.seg").read_bytes())
    hdr_b[10:42] = NS_B
    (dir_b / "000001.seg").write_bytes(bytes(hdr_b))
    assert read_record(dir_b, loc_a, KEY, _k(1), "full") is None


def test_segment_rolls_over(tmp_path: Path) -> None:
    w = SegmentWriter(tmp_path, NS_A, KEY, segment_bytes=300)
    locs = [w.append(_k(i), "full", bytes([i]) * 100) for i in range(1, 6)]
    w.close()
    names = sorted(p.name for p in tmp_path.glob("*.seg"))
    assert len(names) > 1
    assert names[0] == "000001.seg"
    assert all(len(n) == len("000001.seg") and n[:6].isdigit() for n in names)
    assert {loc.segment for loc in locs} == {int(n[:6]) for n in names}
    for i, loc in enumerate(locs, start=1):
        assert read_record(tmp_path, loc, KEY, _k(i), "full") == bytes([i]) * 100


def test_reopen_continues_with_new_segment(tmp_path: Path) -> None:
    w = SegmentWriter(tmp_path, NS_A, KEY)
    loc1 = w.append(_k(1), "full", b"one")
    w.close()
    w = SegmentWriter(tmp_path, NS_A, KEY)
    loc2 = w.append(_k(2), "full", b"two")
    w.close()
    assert loc2.segment > loc1.segment
    assert read_record(tmp_path, loc1, KEY, _k(1), "full") == b"one"
    assert read_record(tmp_path, loc2, KEY, _k(2), "full") == b"two"


def test_data_key_created_once_with_mode_0600(tmp_path: Path) -> None:
    k1 = load_or_create_data_key(tmp_path)
    k2 = load_or_create_data_key(tmp_path)
    assert len(k1) == 32
    assert k1 == k2
    assert stat.S_IMODE(os.stat(tmp_path / "data.key").st_mode) == 0o600
    assert [p.name for p in tmp_path.iterdir()] == ["data.key"]  # no temp litter


def test_corrupt_data_key_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "data.key").write_bytes(b"short")
    with pytest.raises(RuntimeError):
        load_or_create_data_key(tmp_path)
