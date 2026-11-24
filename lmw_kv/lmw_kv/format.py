"""Encrypted append-only segment files for the NVMe tier.

Segment file ``NNNNNN.seg`` (a plain counter, never a hash)::

    header: magic "LMWKVSEG" | u16 version | 32B ns id | 12B header nonce
    record: u32 total_len | 32B block key | u8 len(group) + group | 12B nonce
            | AES-256-GCM(payload, AAD = ns id + block key + group) | u32 crc32

``total_len`` counts the whole record including itself and the CRC; the CRC
covers everything before it. Any mismatch on read is a miss, never an error.
"""

from __future__ import annotations

import ctypes
import errno
import logging
import os
import secrets
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

log = logging.getLogger("lmw_kv.format")

MAGIC = b"LMWKVSEG"
FORMAT_VERSION = 1
HEADER_LEN = len(MAGIC) + 2 + 32 + 12
KEY_FILENAME = "data.key"
_NONCE_LEN = 12
_TAG_LEN = 16
_MIN_RECORD = 4 + 32 + 1 + _NONCE_LEN + _TAG_LEN + 4
_LINK_UNSUPPORTED = {
    errno.EPERM,
    errno.EACCES,
    errno.ENOTSUP,
    errno.EOPNOTSUPP,
    errno.ENOSYS,
    errno.EMLINK,
    errno.EXDEV,
}


# sync_file_range flags: WAIT_BEFORE | WRITE | WAIT_AFTER
_SYNC_FILE_RANGE_FLAGS = 1 | 2 | 4


def _load_sync_file_range() -> ctypes._NamedFuncPointer | None:
    try:
        fn = ctypes.CDLL(None, use_errno=True).sync_file_range
        fn.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint]
        fn.restype = ctypes.c_int
        return fn
    except (OSError, AttributeError):
        return None


_LIBC_SYNC_FILE_RANGE = _load_sync_file_range()
_drop_error_logged = False


def _sync_range(fd: int, offset: int, length: int) -> None:
    """Wait for writeback of one byte range, without a device cache flush when libc allows."""
    fn = _LIBC_SYNC_FILE_RANGE
    if fn is None:
        os.fdatasync(fd)
        return
    if fn(fd, offset, length, _SYNC_FILE_RANGE_FLAGS) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def drop_range(fd: int, offset: int, length: int, *, written: bool) -> None:
    """Drop a byte range from the page cache. Never raises.

    ``written``: the range was just written, so wait for its writeback first
    (DONTNEED leaves dirty pages cached). A no-op where ``posix_fadvise`` is missing.
    """
    global _drop_error_logged
    advise = getattr(os, "posix_fadvise", None)
    if advise is None:
        return
    try:
        if written:
            _sync_range(fd, offset, length)
        advise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except OSError as e:
        if not _drop_error_logged:
            _drop_error_logged = True
            log.debug("page cache drop failed: %s", e)


@dataclass(frozen=True)
class Location:
    segment: int
    offset: int
    length: int


def record_size(group: str, payload_len: int) -> int:
    """Bytes one record takes in a segment file (see the layout above)."""
    return 4 + 32 + 1 + len(group.encode("utf-8")) + _NONCE_LEN + payload_len + _TAG_LEN + 4


def _seg_path(directory: Path, segment: int) -> Path:
    return directory / f"{segment:06d}.seg"


def _existing_segments(directory: Path) -> list[int]:
    return sorted(int(p.name[:6]) for p in directory.glob("??????.seg") if p.name[:6].isdigit())


class SegmentWriter:
    """Appends records to counter-named segments, rolling at ``segment_bytes``."""

    def __init__(
        self,
        dir: Path,
        ns_id: bytes,
        data_key: bytes,
        segment_bytes: int = 64 << 20,
        *,
        first_segment: int = 1,
    ) -> None:
        """``first_segment``: the lowest number a new segment may take, so a caller that
        deleted the newest files never sees their numbers again."""
        if len(ns_id) != 32 or len(data_key) != 32:
            raise ValueError("ns_id and data_key must be 32 bytes")
        self._dir = dir
        self._ns_id = ns_id
        self._aead = AESGCM(data_key)
        self._segment_bytes = segment_bytes
        self._f: BinaryIO | None = None
        self._segment = 0
        self._size = 0
        existing = _existing_segments(dir)
        self._next = max((existing[-1] + 1) if existing else 1, first_segment)

    def _roll(self) -> None:
        self.close()
        self._segment = self._next
        self._next += 1
        # "xb": never reuse or overwrite a segment file
        f = open(_seg_path(self._dir, self._segment), "xb")  # noqa: SIM115
        header = MAGIC + struct.pack("<H", FORMAT_VERSION) + self._ns_id + secrets.token_bytes(12)
        f.write(header)
        f.flush()
        drop_range(f.fileno(), 0, len(header), written=True)
        self._f = f
        self._size = len(header)

    def append(self, block_key: bytes, group: str, payload: bytes) -> Location:
        g = group.encode("utf-8")
        if len(block_key) != 32 or len(g) > 255:
            raise ValueError("block key must be 32 bytes and group at most 255 bytes")
        nonce = secrets.token_bytes(_NONCE_LEN)
        ct = self._aead.encrypt(nonce, payload, self._ns_id + block_key + g)
        body = block_key + bytes([len(g)]) + g + nonce + ct
        total = 4 + len(body) + 4
        if self._f is None or (
            self._size > HEADER_LEN and self._size + total > self._segment_bytes
        ):
            self._roll()
        assert self._f is not None
        head = struct.pack("<I", total) + body
        record = head + struct.pack("<I", zlib.crc32(head))
        offset = self._size
        self._f.write(record)
        self._f.flush()
        drop_range(self._f.fileno(), offset, total, written=True)
        self._size += total
        return Location(self._segment, offset, total)

    def close(self) -> None:
        if self._f is not None:
            self._f.flush()
            os.fsync(self._f.fileno())
            self._f.close()
            self._f = None


class ReadBuffer:
    """A reusable buffer for raw records, so a reading thread allocates (and page-faults)
    once instead of per record. One per thread: it is not locked."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def view(self, n: int) -> memoryview:
        if len(self._buf) < n:
            self._buf = bytearray(n)
        return memoryview(self._buf)[:n]


def _read_raw(dir: Path, loc: Location, rec: memoryview) -> bytes | None:
    """The segment header, with the record read into ``rec``; ``None`` on any I/O error.

    The segment is opened with ``POSIX_FADV_RANDOM`` (no readahead past the record) and
    both ranges are dropped from the page cache after the read (spec 4.5).
    """
    try:
        fd = os.open(_seg_path(dir, loc.segment), os.O_RDONLY)
    except OSError:
        return None
    try:
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            except OSError:
                pass
        header = os.pread(fd, HEADER_LEN, 0)
        if hasattr(os, "preadv"):
            got = os.preadv(fd, [rec], loc.offset)
        else:  # pragma: no cover - platforms without preadv
            data = os.pread(fd, loc.length, loc.offset)
            got = len(data)
            rec[:got] = data
        drop_range(fd, 0, HEADER_LEN, written=False)
        drop_range(fd, loc.offset, loc.length, written=False)
    except OSError:
        return None
    finally:
        os.close(fd)
    return header if got == loc.length else None


def _checked(
    header: bytes, rec: memoryview, loc: Location, expect_key: bytes, expect_group: str
) -> tuple[bytes, memoryview, bytes] | None:
    """(nonce, ciphertext with tag, AAD) of a record whose frame, CRC, key and group
    check out; ``None`` otherwise. Views into ``rec``, never copies."""
    if len(header) != HEADER_LEN:
        return None
    if header[:8] != MAGIC or struct.unpack("<H", header[8:10])[0] != FORMAT_VERSION:
        return None
    (total,) = struct.unpack("<I", rec[:4])
    if total != loc.length:
        return None
    (crc,) = struct.unpack("<I", rec[-4:])
    if zlib.crc32(rec[:-4]) != crc:
        return None
    key = bytes(rec[4:36])
    glen = rec[36]
    group = bytes(rec[37 : 37 + glen])
    nonce_at = 37 + glen
    nonce = bytes(rec[nonce_at : nonce_at + _NONCE_LEN])
    ct = rec[nonce_at + _NONCE_LEN : -4]
    if key != expect_key or group != expect_group.encode("utf-8") or len(ct) < _TAG_LEN:
        return None
    return nonce, ct, header[10:42] + key + group


def read_record(
    dir: Path, loc: Location, data_key: bytes, expect_key: bytes, expect_group: str
) -> bytes | None:
    """Decrypt one record; ``None`` on any mismatch or damage."""
    if loc.length < _MIN_RECORD:
        return None
    rec = memoryview(bytearray(loc.length))
    header = _read_raw(dir, loc, rec)
    got = None if header is None else _checked(header, rec, loc, expect_key, expect_group)
    if got is None:
        return None
    nonce, ct, aad = got
    try:
        return AESGCM(data_key).decrypt(nonce, ct, aad)
    except InvalidTag:
        return None


def read_record_into(
    dir: Path,
    loc: Location,
    data_key: bytes,
    expect_key: bytes,
    expect_group: str,
    out: memoryview,
    buf: ReadBuffer,
) -> int | None:
    """``read_record`` decrypting into ``out`` (the payload's first bytes): the payload
    length, or ``None`` on any mismatch or damage, or when ``out`` is too small.

    The same checks in the same order (frame, CRC, key, group, then the GCM tag), with
    the raw record read into ``buf`` and no other copy. Large reads, the CRC and the
    decrypt release the GIL, so several threads can run this at once. On a
    ``None`` the bytes left in ``out`` are undefined.
    """
    if loc.length < _MIN_RECORD:
        return None
    rec = buf.view(loc.length)
    header = _read_raw(dir, loc, rec)
    got = None if header is None else _checked(header, rec, loc, expect_key, expect_group)
    if got is None:
        return None
    nonce, ct, aad = got
    n = len(ct) - _TAG_LEN
    if len(out) < n:
        return None
    aead = AESGCM(data_key)
    try:
        if hasattr(aead, "decrypt_into"):
            aead.decrypt_into(nonce, ct, aad, out[:n])
        else:  # pragma: no cover - cryptography older than 45
            out[:n] = aead.decrypt(nonce, ct, aad)
    except InvalidTag:
        return None
    return n


def _parse_key(path: Path) -> bytes:
    try:
        key = path.read_bytes()
    except OSError:
        key = b""
    if len(key) != 32:
        raise RuntimeError(f"KV data key {path} is empty or corrupt; restore it from backup")
    return key


def load_or_create_data_key(nvme_dir: Path) -> bytes:
    """32 random bytes in ``<nvme_dir>/data.key`` (0600), created race-safely."""
    path = nvme_dir / KEY_FILENAME
    if path.exists():
        return _parse_key(path)
    key = secrets.token_bytes(32)
    fd, tmp_name = tempfile.mkstemp(dir=nvme_dir, prefix=f".{KEY_FILENAME}.")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(key)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return _parse_key(path)  # another starter won the race: use its key
        except OSError as e:
            if e.errno not in _LINK_UNSUPPORTED:
                raise
            try:
                efd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                return _parse_key(path)
            try:
                with os.fdopen(efd, "wb") as f:
                    f.write(key)
                    f.flush()
                    os.fsync(f.fileno())
            except BaseException:
                path.unlink(missing_ok=True)
                raise
    finally:
        tmp.unlink(missing_ok=True)
    return key
