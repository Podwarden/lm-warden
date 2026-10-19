"""Tail-follow a reused engine log file without blocking.

The engine log (``{logs_dir}/{model_id}.log``) is opened O_APPEND and reused by
every load attempt (see ``app/runtime/engine/run_marker.py``), so it accumulates
the output of many runs. A follower that wants the CURRENT run's text must start
at the file's CURRENT END — otherwise it would re-read a previous run's stale
tail on the first call.

``LogFollower`` does exactly that, plus the rotation/truncation/missing-file
behaviour a long-lived reader needs:

- starts at the current end of the file (first ``read_new()`` returns ""),
- returns new text decoded UTF-8 with ``errors="replace"`` and keeps ``\\r``,
- returns "" if the file does not exist yet (caller keeps polling),
- reopens from offset 0 on size drop or inode change (logrotate / truncate),
- caps one read at 4 MiB and holds the remainder for the next call so a burst
  cannot balloon memory.

**Incremental UTF-8 decode:** the engine writes bytes asynchronously and a read
may land in the middle of a multibyte character (either because the 4 MiB cap
cut it, or because the engine has not finished writing it yet). A naive
``bytes.decode`` would turn the incomplete trailing bytes into U+FFFD on this
read and the continuation bytes into another U+FFFD on the next — corrupting
one character into two replacement chars. The follower uses one
``codecs.getincrementaldecoder("utf-8")(errors="replace")`` per instance and
decodes with ``final=False`` so incomplete trailing bytes are held for the next
read. The decoder is RESET when the follower reopens from offset 0
(truncation/rotation/recreate) so leftover bytes from the previous file do not
corrupt the new file's first characters.

All I/O is synchronous and tiny (one ``read`` per call). The caller is the
load-progress loop, which already runs file reads off the event loop; a
non-blocking ``read_new`` is the right shape here.
"""

from __future__ import annotations

import codecs
import logging
import os
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)

# One read is capped at 4 MiB so a chatty burst cannot balloon memory. The
# remainder is held for the next call (we seek past what we returned).
_READ_CAP = 4 * 1024 * 1024


class LogFollower:
    """Tail-follow a log file, starting at the current end.

    The "current end" is the file's size at construction time (the moment the
    load-progress loop decided to follow this log). Text written after that
    is returned by ``read_new``; the stale tail of a prior run that was
    already on disk when the follower was built is not.

    The follower remembers the (inode, offset) it last read to so it can
    detect rotation (inode change) and truncation (size drop) and reopen from
    offset 0 in either case.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        # Size of the file at construction time (0 if it didn't exist). The
        # first successful open seeks here so we start at the "current end"
        # and never replay a prior run's stale tail.
        self._start_offset = self._current_size()
        self._inode: int | None = None
        self._offset: int = 0
        self._fp: IO[bytes] | None = None  # binary file handle, lazily opened
        # One incremental UTF-8 decoder per follower instance. Incomplete
        # trailing bytes (a multibyte char split across reads) are held here
        # and reassembled on the next read. Reset on reopen-from-0.
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def _current_size(self) -> int:
        try:
            return os.stat(self._path).st_size
        except (FileNotFoundError, OSError):
            return 0

    def _reset_decoder(self) -> None:
        """Reset the incremental decoder. Called on every reopen from offset 0
        (truncation/rotation/recreate) so leftover bytes from the previous file
        do not corrupt the new file's first characters."""
        self._decoder.reset()

    def _open(self, *, seek_to: int) -> None:
        """Open the file and seek to ``seek_to``. Record the inode.

        If the file doesn't exist, leave ``_fp`` None — caller will poll again.
        On reopen (seek_to == 0 for rotation/truncation) the decoder is reset.
        """
        try:
            st = os.stat(self._path)
        except (FileNotFoundError, OSError):
            self._inode = None
            self._offset = 0
            return
        # Open for read in binary; we decode ourselves so we can use
        # errors="replace" and keep \r.
        fp = open(self._path, "rb")  # noqa: SIM115 — held for the follower's life
        self._fp = fp
        self._inode = st.st_ino
        fp.seek(seek_to)
        self._offset = seek_to
        # Reset the incremental decoder whenever we reopen from a new start
        # (rotation/truncation/recreate). The first open (from _start_offset)
        # also resets — there is no prior state to carry over.
        self._reset_decoder()

    def read_new(self) -> str:
        """Return new text since the last call. Never raises.

        - If the file doesn't exist yet, returns "".
        - If the file rotated (inode changed) or was truncated (size < offset),
          reopens from offset 0.
        - One read is capped at 4 MiB; the remainder is held for the next call.
        - Decodes UTF-8 with ``errors="replace"`` and keeps ``\\r``.
        """
        try:
            return self._read_new_inner()
        except OSError as exc:
            # Any I/O error degrades to "no new text" rather than propagating;
            # the load-progress loop polls again in a few seconds.
            logger.debug("LogFollower read failed for %s: %s", self._path, exc)
            return ""

    def _read_new_inner(self) -> str:
        # First call: open the file and seek to the "current end" captured at
        # construction time, so a prior run's stale tail is not replayed.
        if self._fp is None:
            self._open(seek_to=self._start_offset)
            if self._fp is None:
                # File didn't exist; caller will poll again.
                return ""
            return self._read_chunk_and_advance()

        # We already have an open file. Check for rotation / truncation.
        try:
            st = os.stat(self._path)
        except (FileNotFoundError, OSError):
            # File went away. Close our handle; next call reopens when it
            # reappears (from _start_offset, which was 0 for a missing file).
            self._close()
            return ""

        if st.st_ino != self._inode:
            # Rotated (logrotate renamed + created a new file). Reopen from 0.
            self._close()
            self._open(seek_to=0)
            return self._read_chunk_and_advance() if self._fp is not None else ""

        if st.st_size < self._offset:
            # Truncated in place. Reopen from 0.
            self._close()
            self._open(seek_to=0)
            return self._read_chunk_and_advance() if self._fp is not None else ""

        # Same file, same or larger size: read from our last offset.
        self._fp.seek(self._offset)
        return self._read_chunk_and_advance()

    def _read_chunk_and_advance(self) -> str:
        """Read up to _READ_CAP bytes from the current cursor, decode
        incrementally, and advance the offset by what we actually read.

        Incomplete trailing bytes (a multibyte UTF-8 character split across
        reads) are held in the incremental decoder and reassembled on the next
        call — they are NOT turned into U+FFFD.
        """
        assert self._fp is not None
        data = self._fp.read(_READ_CAP)
        if not data:
            # No new bytes this time. Do NOT flush the decoder — the held
            # bytes may be completed by the next read. Returning "" is fine:
            # the caller polls again and the next read may supply the rest.
            return ""
        # Advance the offset by bytes read so the next call continues from
        # here. The decoder may hold a few incomplete trailing bytes; they
        # are not lost — they will be prepended to the next read internally.
        self._offset += len(data)
        # final=False: incomplete trailing bytes are buffered inside the
        # decoder, not emitted as U+FFFD.
        return self._decoder.decode(data, final=False)

    def _close(self) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            except OSError:
                pass
            self._fp = None

    def close(self) -> None:
        """Close the file handle deterministically.  Safe to call twice
        (or when the file was never opened) — a no-op if already closed.

        Called from the watcher's ``finally`` cleanup path so the handle is
        released immediately when the watcher task is cancelled, rather than
        waiting for GC.  ``read_new`` after ``close`` reopens the file from
        the start offset if the file still exists (the follower's rotation
        logic handles a ``None`` handle on the next call), or returns "" if
        the file is gone.
        """
        self._close()
