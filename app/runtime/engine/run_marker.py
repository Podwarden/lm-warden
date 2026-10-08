"""Run-boundary sentinels for the per-model engine log.

The engine log (``{logs_dir}/{model_id}.log``) is opened O_APPEND and reused by
every load attempt, so it accumulates the output of many runs. That is what the
operator wants -- comparing attempt N-1 with attempt N is the whole point of
keeping the file -- but it broke automated diagnosis: ``_read_engine_log_tail``
handed the last 200 lines to ``diagnose_engine_log``, and a *previous* run's
traceback sitting inside that window produced a confident, specific and WRONG
explanation for the current failure, which then replaced the accurate generic
message (#234; observed twice live on 2026-09-03 -- a successful-through-
profiling run reported as "GPU ran out of memory", and a run with
``max_model_len=32768`` reported verbatim with the previous run's
"wants 262144 tokens").

The fix is a delimiter, not truncation: every driver writes one sentinel line
into the log when it opens it for a spawn, and the diagnosis reader starts from
the LAST sentinel. History stays in the file; diagnosis sees only this run.

This lives in ``app/runtime/engine/`` (next to the drivers, above any single
backend) because it is a property of how a run is *launched*, not of vLLM's or
llama.cpp's log grammar -- both benefit unchanged.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime

# The literal an operator greps for when splitting a log by run.
RUN_SENTINEL_PREFIX = "===== vllm-warden run "

# Deliberately ``search``-not-``match`` semantics at the call site: if a
# previous run's final chunk had no trailing newline the sentinel can end up
# glued to the tail of that line, and we still want the boundary recognised.
# Kept loose on the id/timestamp so the format can gain fields without the
# reader (which may be running against logs written by an older build) losing
# the ability to split them.
_SENTINEL_RE = re.compile(r"=====\s+vllm-warden\s+run\s+\S+\s+started\s+\S+\s+=====\s*$")


def format_run_sentinel(run_id: str | None = None, *, now: datetime | None = None) -> str:
    """Build the sentinel line for one spawn (no trailing newline).

    ``run_id`` defaults to a fresh uuid4 hex; the timestamp is UTC ISO-8601 so
    the boundary is unambiguous, unlike vLLM's own ``MM-DD HH:MM:SS`` lines
    which carry no year and no zone.
    """
    rid = run_id or uuid.uuid4().hex
    ts = (now or datetime.now(UTC)).isoformat()
    return f"{RUN_SENTINEL_PREFIX}{rid} started {ts} ====="


def is_run_sentinel(line: str) -> bool:
    """True if ``line`` is (or ends with) a run-boundary sentinel."""
    return _SENTINEL_RE.search(line) is not None


def tail_since_last_run(text: str, *, max_lines: int) -> str:
    """Return at most ``max_lines`` of ``text`` from the last run boundary on.

    Legacy/foreign logs (written before sentinels existed, or by a driver that
    does not write them) contain no sentinel: we then fall back to exactly the
    old behaviour -- the last ``max_lines`` lines -- because a *possibly* stale
    diagnosis still beats no diagnosis at all, and returning "" would silently
    delete the feature for every log already on disk.

    The sentinel itself is dropped: it is warden bookkeeping, not engine output,
    and no backend grammar should ever have to know about it.
    """
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if is_run_sentinel(lines[i]):
            # Only the current run, still capped: a very chatty run must not
            # hand the parser an unbounded string.
            return "\n".join(lines[i + 1 :][-max_lines:])
    return "\n".join(lines[-max_lines:])
