"""Drop accounting for the proxy's request-path bookkeeping.

R43 (issue #279, stage 1): the proxy's own SQLite ledger writes (the
token ``last_used_at`` touch in ``require_bearer`` and the per-request
counters batch in ``_record_counters``) can fail with
``sqlite3.OperationalError: database is locked`` under real
concurrency. That failure must never fail the paid /v1 request that
produced it: the caller catches ONLY ``sqlite3.Error`` around the
ledger write, calls :func:`note_dropped`, and the request completes
with its normal answer.

The counter is process-global and keyed by call site, so an operator
can see WHICH ledger write is dropping and how often. It is the
in-process home of the ``vw_bookkeeping_dropped_total`` counter (one
series per call site).

#294 exports it: :func:`snapshot` is served by the admin-only
``GET /api/stats/v2/bookkeeping`` and drives the Stats page's "stats writes
dropped" line. Next to each count it keeps the requests whose accounting
was lost, when the call site knows that number. Both reset when the process
restarts (``since_epoch`` says when that was).
"""

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_dropped: dict[str, int] = {}
# Requests lost per call site. A site that has dropped once without knowing
# the number is None for good: unknown is never summed into a 0.
_lost: dict[str, int | None] = {}
_lock = threading.Lock()
_STARTED = time.time()


def dropped_total() -> dict[str, int]:
    """Snapshot of the drop counts, keyed by call site."""
    with _lock:
        return dict(_dropped)


def snapshot() -> dict[str, Any]:
    """The counters as the admin endpoint serves them.

    ``{"since_epoch", "sites": {site: {"count", "requests_lost"}}, "total"}``;
    a ``requests_lost`` of None means unknown, and makes the total unknown.
    """
    with _lock:
        sites = {
            site: {"count": n, "requests_lost": _lost.get(site)} for site, n in _dropped.items()
        }
    lost: int | None = 0
    for v in sites.values():
        site_lost = v["requests_lost"]
        lost = None if lost is None or site_lost is None else lost + site_lost
    return {
        "since_epoch": _STARTED,
        "sites": sites,
        "total": {"count": sum(v["count"] for v in sites.values()), "requests_lost": lost},
    }


def note_dropped(
    call_site: str,
    exc: BaseException,
    *,
    requests: int | None = None,
    log: bool = True,
) -> None:
    """Count one dropped bookkeeping write and log it.

    ``requests`` is how many requests' accounting the drop lost; None when
    the caller cannot tell.

    ``log=False`` is for a caller that logs its own, more specific warning
    (the ledger names the keys and requests lost).

    Never raises: the point of this module is that the caller can
    afford to run it on the failure path.
    """
    with _lock:
        _dropped[call_site] = _dropped.get(call_site, 0) + 1
        prev = _lost.get(call_site, 0)
        _lost[call_site] = None if prev is None or requests is None else prev + requests
    if log:
        logger.warning("dropped %s bookkeeping write: %s", call_site, exc)
