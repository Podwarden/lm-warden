"""Encrypted append-only segment files for the NVMe tier.

Segment file ``NNNNNN.seg`` (a plain counter, never a hash)::

    header: magic "LMWKVSEG" | u16 version | 32B ns id | 12B header nonce
    record: u32 total_len | 32B block key | u8 len(group) + group | 12B nonce
            | AES-256-GCM(payload, AAD = ns id + block key + group) | u32 crc32

``total_len`` counts the whole record including itself and the CRC; the CRC
covers everything before it. Any mismatch on read is a miss, never an error.
"""

from __future__ import annotations

import errno
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
        self._size += total
        return Location(self._segment, offset, total)

    def close(self) -> None:
        if self._f is not None:
            self._f.flush()
            os.fsync(self._f.fileno())
            self._f.close()
            self._f = None


def read_record(
    dir: Path, loc: Location, data_key: bytes, expect_key: bytes, expect_group: str
) -> bytes | None:
    """Decrypt one record; ``None`` on any mismatch or damage."""
    if loc.length < _MIN_RECORD:
        return None
    try:
        with open(_seg_path(dir, loc.segment), "rb") as f:
            header = f.read(HEADER_LEN)
            f.seek(loc.offset)
            rec = f.read(loc.length)
    except OSError:
        return None
    if len(header) != HEADER_LEN or len(rec) != loc.length:
        return None
    if header[:8] != MAGIC or struct.unpack("<H", header[8:10])[0] != FORMAT_VERSION:
        return None
    ns_id = header[10:42]
    (total,) = struct.unpack("<I", rec[:4])
    if total != loc.length:
        return None
    (crc,) = struct.unpack("<I", rec[-4:])
    if zlib.crc32(rec[:-4]) != crc:
        return None
    key = rec[4:36]
    glen = rec[36]
    group = rec[37 : 37 + glen]
    nonce_at = 37 + glen
    nonce = rec[nonce_at : nonce_at + _NONCE_LEN]
    ct = rec[nonce_at + _NONCE_LEN : -4]
    if key != expect_key or group != expect_group.encode("utf-8") or len(ct) < _TAG_LEN:
        return None
    try:
        return AESGCM(data_key).decrypt(nonce, ct, ns_id + key + group)
    except InvalidTag:
        return None


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
