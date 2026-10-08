"""Config-gated, per-token-scoped request/response content logger.

Tapped into the proxy forward path (``app/proxy/routes.py::_forward``) to
capture full prompt + completion content for a small allowlist of tokens on a
shared server — the diagnostic tool for chasing runaway / degenerate
generations where we need to see exactly what came out.

Contract:

* **Disabled by default.** When ``settings.content_log_enabled`` is False the
  caller's fast-path gate (:func:`should_log`) short-circuits before any of
  this runs, so the forward path stays byte-identical to a build without this
  module — no extra awaits, no extra work.
* **Hard-scoped.** Even when enabled, a request is logged only if its token id
  is in ``settings.content_log_tokens``. There is deliberately no "log
  everything" mode: this runs on shared production.
* **Never breaks the response, and never delays another one.** Every public
  entry point swallows exceptions and at most logs a warning. The caller does
  the write only after the response content is in hand and the scheduler slot
  is released — so the *logged* request is already finished with. The write
  itself runs on a worker thread (:func:`write_entry` is a coroutine and
  awaits ``anyio.to_thread.run_sync``), because the image runs a single
  uvicorn worker: a synchronous ~80 KB disk write on the event loop stalled
  every OTHER in-flight request, SSE chunk pumping included, once per logged
  request.
* **Strictly ordered, never interleaved.** A worker pool is exactly the
  cross-writer interleaving the single uvicorn worker used to rule out. Each
  record's place in the file is therefore fixed on the event loop, before the
  offload (:func:`_enqueue`), and the worker drains that queue under
  :data:`_write_lock`. Order in the file is submission order.
* **Owner-only on disk.** The log directory is created 0700 and the file 0600,
  applied at creation rather than left to the process umask — see
  :func:`_open_append`. The destination sits on the persistent data volume
  alongside the SQLite database, and nothing rotates or prunes it.
* **Bounded, and fail-safe when the bound is reached.** Two caps, and neither
  has an "unlimited" value that an operator could reach for by accident:

  - ``settings.content_log_max_chars`` caps one RECORD — the prompt and the
    completion separately. A negative value is not "no limit": it is clamped
    to 0 (capture nothing) in :func:`_cap` and at config-parse time, which is
    also where the operator gets told. See :func:`_cap`.
  - ``settings.content_log_max_bytes`` (``VW_CONTENT_LOG_MAX_BYTES``, 512 MiB)
    caps the FILE. On reaching it the sink stops writing and warns once. It
    deliberately does not rotate and does not delete — see :func:`_drain`.
    ``<= 0`` switches the cap off, matching ``VW_ENGINE_LOG_MAX_BYTES``.

One JSON object per line (JSONL).
"""

import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import TextIO

import anyio.to_thread

from app.config import Settings

log = logging.getLogger("vllm_warden.content_log")

# The records hold users' prompts and completions verbatim, so the directory
# and the file are created owner-only rather than at whatever the process
# umask happens to be (0o022 on a stock image, which would publish every
# prompt as 0o644). Both modes are handed to mkdir(2)/open(2) so they apply at
# creation — there is no window in which the file exists world-readable — and
# neither has group or other bits for a umask to widen; a umask can only
# narrow them further.
_DIR_MODE = 0o700
_FILE_MODE = 0o600

# --- The serialising sink -------------------------------------------------
#
# _pending holds already-serialised records in submission order as
# (path, max_bytes, line) triples. It is appended to ON THE EVENT LOOP (a
# deque append is atomic under the GIL, and the loop is the only appender) and
# drained by whichever worker thread wins _write_lock. That split is the whole
# ordering guarantee: the thread pool decides who *writes*, never who comes
# first in the file.
#
# max_bytes rides along per record rather than being read from a module global
# because settings are per-request and frozen; the path does too, so a drain
# can never mix two destinations into one open() (tests, and any future
# per-token destination, rely on that).
_pending: deque[tuple[Path, int, str]] = deque()
_write_lock = threading.Lock()

# --- Failure suppression --------------------------------------------------
#
# The failure this module causes is a full volume, and the old handler emitted
# a full traceback per logged request — adding write pressure to a disk that
# is already out of space, once per request, for as long as the logger stays
# armed. The first failure is logged in full; identical repeats are counted
# and suppressed, and the count is reported when a write next succeeds so a
# transient fault cannot silence the warning permanently.
_last_failure: str | None = None
_suppressed: int = 0
# The size ceiling warns once per breach in the same way, with its own flag so
# a write failure and a full file cannot mask each other.
_ceiling_warned: bool = False


def _reset_state_for_tests() -> None:
    """Clear the module-level sink state. Test-support only.

    The queue, the lock's history, the failure-suppression counters and the
    ceiling flag are all process-global by design — they have to be, to
    serialise and to deduplicate across requests. Tests that exercise the
    failure and ceiling paths must not leak that state into each other.
    """
    global _last_failure, _suppressed, _ceiling_warned
    with _write_lock:
        _pending.clear()
        _last_failure = None
        _suppressed = 0
        _ceiling_warned = False


def should_log(settings, token_id) -> bool:
    """Cheap gate: True only when logging is enabled AND this token is allowlisted.

    Called first in the hot path so a disabled logger costs one attribute read
    plus, at most, a set-membership test — no I/O, no parsing.
    """
    return (
        settings.content_log_enabled
        and token_id is not None
        and token_id in settings.content_log_tokens
    )


def _cap(text: str | None, max_chars: int) -> str | None:
    """Truncate ``text`` to ``max_chars``, appending a marker noting how much
    was dropped. ``None`` passes through unchanged.

    **There is no value that means "no limit".** This used to read
    ``if max_chars >= 0 and ...``, which made ``VW_CONTENT_LOG_MAX_CHARS=-1``
    an undocumented escape hatch writing every record in full — flatly
    contradicting this module's contract, and -1 is exactly the value an
    operator reaches for, since ``VW_REQUEST_MAX_WALL_S=0`` does mean "no cap"
    a few lines away in the same ``.env.example``.

    A character count has no negative reading, so a negative one is clamped to
    0: capture nothing, keep the record's shape and its metadata. That is the
    fail-safe direction — an operator who meant "unlimited" sees empty prompts
    at once and fixes the value, where the old behaviour handed them unbounded
    records and an eventual full volume with no warning at all.
    ``load_settings`` clamps and warns as well, so the value in ``Settings`` is
    never negative; this stays defensive because ``Settings`` is also
    constructed directly.
    """
    if text is None:
        return None
    limit = max(max_chars, 0)
    if len(text) > limit:
        dropped = len(text) - limit
        return text[:limit] + f"…[truncated {dropped} chars]"
    return text


def parse_nonstream(content: bytes, is_chat: bool) -> tuple[str | None, str | None]:
    """Extract ``(completion_text, finish_reason)`` from a non-streaming body.

    ``is_chat`` selects ``choices[0].message.content`` (chat) vs
    ``choices[0].text`` (/completions). Best-effort: any parse failure yields
    ``(None, None)`` rather than raising.
    """
    try:
        data = json.loads(content)
        choice = (data.get("choices") or [{}])[0] or {}
        if is_chat:
            completion = (choice.get("message") or {}).get("content")
        else:
            completion = choice.get("text")
        return completion, choice.get("finish_reason")
    except Exception:
        return None, None


def parse_sse_finish(line: bytes) -> str | None:
    """Return ``choices[0].finish_reason`` from one SSE ``data:`` frame, if any.

    Companion to ``routes._parse_sse_delta`` — the streaming tap tracks the
    last non-null finish_reason so the JSONL record can carry it. Only called
    from the gated (logging-enabled) path, so the extra parse never touches the
    disabled hot path.
    """
    if not line.lstrip().startswith(b"data:"):
        return None
    payload = line.split(b"data:", 1)[1].strip()
    if payload == b"[DONE]":
        return None
    try:
        ev = json.loads(payload)
        return (ev.get("choices") or [{}])[0].get("finish_reason")
    except Exception:
        return None


def _open_append(path: Path) -> TextIO:
    """Open ``path`` for append, creating it owner-only if it is not there.

    An **existing** directory or file keeps the mode it has: this runs on every
    logged request, and an operator who deliberately widened the file (a log
    shipper reading it as another user, say) should not have that undone once
    per request by a best-effort diagnostic path. The consequence is that a
    file created by a build from before this hardening keeps its old, umask-
    derived mode — the changelog tells operators to chmod that one file once.

    ``parents=True`` creates any *intermediate* directories at the default
    mode; only the log directory itself is forced to ``_DIR_MODE``. The
    shipped path (``/data/logs/content.jsonl``) has no intermediates — /data
    is the volume mount — so this only applies to a deep custom
    ``VW_CONTENT_LOG_PATH``.
    """
    directory = path.parent
    if not directory.is_dir():
        directory.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    # O_CREAT applies _FILE_MODE at creation and is ignored when the file
    # already exists, which is exactly the "create tight, leave existing
    # alone" rule above — no chmod, and so no world-readable window.
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, _FILE_MODE)
    return os.fdopen(fd, "a", encoding="utf-8")


def _enqueue(path: Path, max_bytes: int, line: str) -> None:
    """Fix this record's place in the file. Called on the event loop.

    Everything after this point runs on a worker thread and may be reordered
    by the pool; nothing before it can be, because the loop is single-threaded.
    So this call, not the write, is the ordering point.
    """
    _pending.append((path, max_bytes, line))


def _warn_write_failed(path: Path, exc: BaseException) -> None:
    """First failure in full; identical repeats counted, not logged.

    Keyed on the exception's type and message, so a *different* fault after a
    run of ENOSPC still gets its own full report rather than disappearing into
    the suppression.

    Deliberately does NOT take :data:`_write_lock`, unlike its counterpart
    :func:`_note_write_succeeded`: this one is reached from the event loop
    (``write_entry``'s handler), and blocking there on a lock a worker holds
    across disk I/O would put back the very stall this module was fixed to
    avoid. The counters are therefore racy between the loop and a worker, and
    they are allowed to be — the worst outcome is an off-by-one in a
    diagnostic count, or one warning more or less than the ideal.
    """
    global _last_failure, _suppressed
    signature = f"{type(exc).__name__}: {exc}"
    if signature == _last_failure:
        _suppressed += 1
        return
    _last_failure = signature
    _suppressed = 0
    log.warning(
        "content_log: failed to write entry to %s — further identical "
        "failures will be suppressed until one succeeds",
        path,
        exc_info=exc,
    )


def _note_write_succeeded(path: Path) -> None:
    """Clear the suppression, reporting what it swallowed.

    Without the reset a transient fault would silence the warning for the life
    of the process; without the count the operator would never learn how many
    records the outage cost them.
    """
    global _last_failure, _suppressed
    if _last_failure is None:
        return
    if _suppressed:
        log.warning(
            "content_log: writing to %s recovered; %d further identical "
            "failure(s) were suppressed",
            path,
            _suppressed,
        )
    _last_failure = None
    _suppressed = 0


def _drain() -> None:
    """Write every queued record, in order, under one lock. Worker thread.

    Runs the whole queue rather than one record so that concurrent callers
    coalesce into a single open/append/close instead of one apiece, and so a
    record can never be written before one submitted ahead of it. Consecutive
    records bound for the same file share the open; a change of destination
    starts a new one.

    **The size ceiling is fail-safe: it stops, it does not rotate.** The
    engine log (``LocalSubprocessDriver._rotate``, ``VW_ENGINE_LOG_MAX_BYTES``)
    keeps one ``.1`` generation, which is right for a log nobody depends on
    the history of. It is wrong here. The hazard is an outage — the SQLite
    database shares this volume, and the 2026-06-15 ENOSPC incident took it
    down — so the real budget must be the number in the variable, and a single
    generation would quietly make it 2x. And the records are the incidents the
    operator armed the logger to capture: deleting them to make room for newer
    ones destroys the evidence the file exists for. Hitting the ceiling is a
    condition for a human to act on, so it stops and says so, once.
    """
    global _ceiling_warned
    with _write_lock:
        fh: TextIO | None = None
        open_path: Path | None = None
        # Tracked rather than re-stat()ed per record: the handle is buffered,
        # so st_size lags the writes, and a queue drained in one go would
        # otherwise sail past the ceiling by the length of the queue.
        size = 0
        last_written: Path | None = None
        try:
            while _pending:
                path, max_bytes, line = _pending.popleft()
                if fh is not None and path != open_path:
                    fh.close()
                    fh = None
                    open_path = None
                if fh is None:
                    size = _size_of(path)
                if max_bytes > 0 and size >= max_bytes:
                    _warn_ceiling(path, max_bytes)
                    # Drop this record and everything else queued for the same
                    # file: holding them would trade a bounded file for an
                    # unbounded queue.
                    _discard_for(path)
                    continue
                if fh is None:
                    fh = _open_append(path)
                    open_path = path
                blob = line + "\n"
                fh.write(blob)
                size += len(blob.encode("utf-8"))
                last_written = path
                # Writing again means the operator cleared the file; re-arm the
                # ceiling warning so the next breach is reported too.
                _ceiling_warned = False
            if last_written is not None:
                _note_write_succeeded(last_written)
        finally:
            if fh is not None:
                fh.close()


def _warn_ceiling(path: Path, max_bytes: int) -> None:
    """One warning per breach, naming the file and the ceiling."""
    global _ceiling_warned
    if _ceiling_warned:
        return
    _ceiling_warned = True
    log.warning(
        "content_log: %s has reached the VW_CONTENT_LOG_MAX_BYTES ceiling of "
        "%d bytes — content logging to it has STOPPED. Nothing is rotated and "
        "nothing is deleted: archive or remove the file, or turn "
        "VW_CONTENT_LOG_ENABLED off, to resume.",
        path,
        max_bytes,
    )


def _size_of(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _discard_for(path: Path) -> None:
    """Drop every queued record bound for ``path``, keeping the rest in order."""
    kept = [item for item in _pending if item[0] != path]
    _pending.clear()
    _pending.extend(kept)


async def write_entry(
    settings: Settings,
    *,
    token_id: str | None,
    model_id: str | None,
    served_name: str | None,
    stream: bool,
    max_tokens: int | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    finish_reason: str | None,
    prompt: str | None,
    completion: str | None,
    extra: dict | None = None,
) -> None:
    """Append one JSONL record. Best-effort — never raises into the caller.

    ``extra`` merges additional top-level keys into the record (used by the
    runaway detector to attach the trip signal and the tokens-in-think count);
    it never overrides the core fields above.

    A coroutine, and it must be awaited. Serialising the record is in-memory
    work and stays on the loop — it is also the point at which this record's
    place in the file is fixed. Only the open/write/close crosses to a worker
    thread, because on a single-uvicorn-worker image that syscall sequence
    used to stall every other in-flight request, SSE streams included.
    """
    path = settings.content_log_path
    try:
        record = {
            "ts": time.time(),
            "token_id": token_id,
            "model_id": model_id,
            "served_name": served_name,
            "stream": stream,
            "max_tokens": max_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "prompt": _cap(prompt, settings.content_log_max_chars),
            "completion": _cap(completion, settings.content_log_max_chars),
        }
        if extra:
            for k, v in extra.items():
                record.setdefault(k, v)
        line = json.dumps(record, ensure_ascii=False)
        _enqueue(path, settings.content_log_max_bytes, line)
        # The only await. Everything above is in-memory and stays on the loop;
        # the file I/O — the part that used to stall every other request —
        # happens over there.
        await anyio.to_thread.run_sync(_drain)
    except Exception as exc:
        # Diagnostic logging must never break the proxied response, and its
        # own failure mode must not amplify: full detail once, then counted.
        _warn_write_failed(path, exc)
