"""Debug capture (owner request 2026-10-28): a key's opt-in, time-limited
recording of the exact request bodies and the raw response, so requests can be
replayed before and after a prompt change (the prompt diet experiment).

Unlike the rest of LM Warden this stores TEXT, so it is off unless a key's
``capture_until`` is in the future, readable by the admin only, and bounded:
one gzip file per request (named by its history-row id) on the data volume,
oldest deleted first once the directory passes ``max_bytes``. Writes run on a
worker thread; a failure never touches the request.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import anyio.to_thread

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
#: How often (seconds) the size cap is checked; a scan walks every file.
ENFORCE_EVERY_S = 60.0


def capturing(capture_until: float | None, now: float) -> bool:
    return capture_until is not None and capture_until > now


class CaptureStore:
    def __init__(self, root: Path, *, max_bytes: int) -> None:
        self.root = Path(root)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._last_enforce = 0.0

    def _path(self, rid: str, ts: float) -> Path:
        day = time.strftime("%Y%m%d", time.gmtime(ts))
        return self.root / day / f"{rid}.json.gz"

    def put_sync(self, record: dict[str, Any]) -> None:
        rid = str(record.get("id", ""))
        if not _ID_RE.match(rid):
            return
        ts = float(record.get("ts", time.time()))
        p = self._path(rid, ts)
        data = gzip.compress(json.dumps(record, separators=(",", ":")).encode(), compresslevel=6)
        with self._lock:
            p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            if time.monotonic() - self._last_enforce >= ENFORCE_EVERY_S:
                self._last_enforce = time.monotonic()
                self._enforce_locked()

    async def put(self, record: dict[str, Any]) -> None:
        """Best effort, off the event loop; never raises."""
        try:
            await anyio.to_thread.run_sync(self.put_sync, record)
        except Exception:  # noqa: BLE001 — capture must never fail a request
            logger.warning("capture: write failed", exc_info=True)

    def get(self, rid: str) -> dict[str, Any] | None:
        if not _ID_RE.match(rid):
            return None
        for p in sorted(self.root.glob(f"*/{rid}.json.gz"), reverse=True):
            try:
                with gzip.open(p, "rb") as f:
                    out: dict[str, Any] = json.loads(f.read())
                    return out
            except (OSError, ValueError):
                continue
        return None

    def enforce(self) -> None:
        with self._lock:
            self._enforce_locked()

    def _enforce_locked(self) -> None:
        files = []
        total = 0
        for p in self.root.glob("*/*.json.gz"):
            try:
                st = p.stat()
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, p))
            total += st.st_size
        if total <= self.max_bytes:
            return
        for _mt, size, p in sorted(files):
            try:
                p.unlink()
            except OSError:
                continue
            total -= size
            if total <= self.max_bytes:
                break
