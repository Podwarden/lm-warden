"""The cobs2 network key: 32 random bytes persisted next to the database.

Every content id is an HMAC under this key, so regenerating it would change
every id. A missing file is created; an empty or corrupt one is an error.
"""

from __future__ import annotations

import errno
import os
import secrets
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

KEY_FILENAME = "cobs2_network_key"
_EPOCH = 1
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
class NetworkKey:
    epoch: int
    key: bytes

    def __repr__(self) -> str:  # never leak the key
        return f"NetworkKey(epoch={self.epoch}, key=<redacted>)"


def _parse(path: Path) -> NetworkKey:
    bad = RuntimeError(
        f"cobs2 network key file {path} is empty or corrupt; restore it from backup "
        "(deleting it changes every cobs2 id)"
    )
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
    except (UnicodeDecodeError, OSError):
        raise bad from None
    if not lines:
        raise bad
    version, sep, hexkey = lines[0].strip().partition(":")
    if version != "v1" or not sep:
        raise bad
    try:
        key = bytes.fromhex(hexkey)
    except ValueError:
        raise bad from None
    if len(key) != 32:
        raise bad
    return NetworkKey(epoch=_EPOCH, key=key)


def _parse_retry(path: Path) -> NetworkKey:
    """Parse, retrying briefly: a racing starter may be mid-write."""
    for _ in range(20):
        try:
            return _parse(path)
        except RuntimeError:
            time.sleep(0.05)
    return _parse(path)


def network_key_path(db_path: Path) -> Path:
    """Where the network key lives: ``cobs2_network_key`` next to the DB."""
    return db_path.parent / KEY_FILENAME


def load_or_create_network_key(db_path: Path) -> NetworkKey:
    path = network_key_path(db_path)
    if path.exists():
        return _parse_retry(path)
    if not db_path.parent.exists():
        raise RuntimeError(f"cannot persist cobs2 network key: data dir {db_path.parent} missing")
    key = secrets.token_bytes(32)
    fd, tmp_name = tempfile.mkstemp(dir=db_path.parent, prefix=f".{KEY_FILENAME}.")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(f"v1:{key.hex()}\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return _parse_retry(path)  # another starter won the race: use its key
        except OSError as e:
            if e.errno not in _LINK_UNSUPPORTED:
                raise
            # no hard links on this filesystem: exclusive create + write instead
            try:
                efd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                return _parse_retry(path)
            try:
                with os.fdopen(efd, "w") as f:
                    f.write(f"v1:{key.hex()}\n")
                    f.flush()
                    os.fsync(f.fileno())
            except BaseException:
                path.unlink(missing_ok=True)
                raise
    finally:
        tmp.unlink(missing_ok=True)
    try:
        dfd = os.open(db_path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass
    return NetworkKey(epoch=_EPOCH, key=key)
