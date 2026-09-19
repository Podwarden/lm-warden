import json
import time
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth.deps import require_jwt
from app.db.database import open_db
from app.stats import request_history
from app.system.gpu import gpu_probe_health

router = APIRouter()

_RANGE_TO_MINUTES = {"1h": 60, "6h": 360, "24h": 1440, "7d": 10080}


def _parse_models(models: str | None) -> list[str] | None:
    """Parse the ``?models=`` selection. ``None`` means "no filter".

    The distinction that matters is ABSENT vs EMPTY, and they are not the same
    question:

      ``?models`` absent      the whole deployment -- every model, every card.
                              The pre-selection behaviour, unchanged.
      ``?models=a,b``         only those models, and only the cards they hold.
      ``?models=`` (empty)    400.

    An empty selection could plausibly be read as "everything" or as "nothing",
    and silently choosing one is how a UI bug becomes a wrong number instead of
    an error. The UI's own rule is that at least one model stays selected, so an
    empty string reaching here is a client defect and is reported as one.

    Order and duplicates are normalised away: the value is a set of ids, and
    two clients that spell the same selection differently must not produce two
    cache entries or two different answers.
    """
    if models is None:
        return None
    ids = sorted({m.strip() for m in models.split(",") if m.strip()})
    if not ids:
        raise HTTPException(
            status_code=400,
            detail=(
                "empty models selection; omit ?models entirely to include every "
                "model, or name at least one"
            ),
        )
    return ids


def _decode_gpu_indices(raw) -> list[int]:
    """``models.gpu_indices`` (JSON list stored as TEXT) -> list[int].

    Never raises. A malformed or absent value yields ``[]``, because this is
    read by a dashboard: a bad row must narrow what a chart covers, visibly,
    rather than 500 a page whose whole job is to report on the deployment.
    """
    try:
        parsed = json.loads(raw) if raw else []
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [g for g in parsed if isinstance(g, int)]


async def _resolve_selection(db, ids: list[str]) -> list[int]:
    """Validate ``ids`` against the models table; return their GPU indices.

    Raises 400 for an id this deployment does not have. A silently-dropped id
    would give the operator a chart that answers a narrower question than the
    one their checkboxes describe -- the worst possible failure for a page whose
    entire job is to say which models a number covers.

    ``models.gpu_indices`` is a JSON list stored as TEXT. The union of those
    lists is what makes a GPU-side filter meaningful: VRAM, utilisation and
    power are per-CARD facts with no model column, so "this selection's VRAM" is
    exactly "the VRAM of the cards this selection occupies".
    """
    placeholders = ",".join("?" for _ in ids)
    cur = await db.execute(
        f"SELECT id, gpu_indices FROM models WHERE id IN ({placeholders})",
        tuple(ids),
    )
    rows = await cur.fetchall()
    found = {r[0] for r in rows}
    missing = [i for i in ids if i not in found]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"unknown model id(s): {', '.join(missing)}"
        )
    gpus: set[int] = set()
    for _mid, raw in rows:
        gpus.update(_decode_gpu_indices(raw))
    return sorted(gpus)


def _since_minute(range_: str) -> int:
    if range_ not in _RANGE_TO_MINUTES:
        raise HTTPException(status_code=400, detail="invalid range")
    now_min = int(time.time() // 60)
    return now_min - _RANGE_TO_MINUTES[range_]


def _validate_range(range_: str) -> int:
    """Return retention-bounded minutes for ``range_``. Raises 400 on unknown.

    Accepts ``1h``, ``6h``, ``24h``, ``7d``. S7 (#124) — shared by every
    ``/api/stats/v2/*`` endpoint so frontend range-picker errors hit a
    single, predictable validation site.
    """
    if range_ not in _RANGE_TO_MINUTES:
        raise HTTPException(
            status_code=400,
            detail=f"invalid range '{range_}'; allowed: {sorted(_RANGE_TO_MINUTES)}",
        )
    return _RANGE_TO_MINUTES[range_]


@router.get("/api/stats/models")
async def stats_models(
    request: Request, range: str = "24h", _user: str = Depends(require_jwt)
):
    since = _since_minute(range)
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        cur = await db.execute(
            "SELECT model_id, minute, requests, prompt_tokens, completion_tokens "
            "FROM model_samples WHERE minute >= ? ORDER BY minute ASC",
            (since,),
        )
        rows = await cur.fetchall()
    return [
        {"model_id": r[0], "minute": r[1], "requests": r[2],
         "prompt_tokens": r[3], "completion_tokens": r[4]}
        for r in rows
    ]


@router.get("/api/stats/gpus")
async def stats_gpus(
    request: Request, range: str = "24h", _user: str = Depends(require_jwt)
):
    since = _since_minute(range)
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        cur = await db.execute(
            "SELECT gpu_index, minute, utilization_pct, memory_used_mib, "
            "memory_total_mib, name "
            "FROM gpu_samples WHERE minute >= ? ORDER BY minute ASC, gpu_index ASC",
            (since,),
        )
        rows = await cur.fetchall()
    # `name` is NULL on rows written before migration 0013; UI must fall back
    # to "GPU N". Surface it explicitly so the frontend doesn't have to probe.
    return [
        {"gpu_index": r[0], "minute": r[1], "utilization_pct": r[2],
         "memory_used_mib": r[3], "memory_total_mib": r[4], "name": r[5]}
        for r in rows
    ]


# ============================================================================
# S7 (#124) — /api/stats/v2 endpoints. Coexist with v1 (CTO decision #7); v1
# stays untouched and is the existing dashboard's stable contract. v2 is the
# new richer shape consumed by the redesigned stats UI in dev-2's slice.
#
# Shape contract is the source of truth for the frontend handoff (see the MR
# description / dev-2 handoff note). Keys are stable; new fields may be added
# but existing ones won't be renamed or dropped without a follow-up issue.
# ============================================================================


@router.get("/api/stats/v2/overview")
async def stats_v2_overview(
    request: Request,
    range: str = "24h",
    models: str | None = None,
    _user: str = Depends(require_jwt),
):
    """Aggregate dashboard payload for the stats page, for a selection of models.

    ``models`` is a comma-separated list of model ids. Absent means the whole
    deployment (the original behaviour, byte-for-byte). Present, it narrows
    EVERY number in the response -- see ``_parse_models``.

    WHAT "NARROWS" MEANS PER SERIES, because the three data sources carry
    different dimensions and pretending otherwise would produce a chart that
    silently answers a different question than its checkboxes:

      tokens / tps   per MODEL. ``model_samples`` has a model_id column, so the
                     filter is exact and a selection is a true partition:
                     selecting every model sums to the unfiltered total.
      vram / util /  per CARD. Those tables have no model column at all, so the
      power          selection is resolved to the union of the selected models'
                     ``gpu_indices`` and the cards are filtered by that. "This
                     model's VRAM" is, and can only be, "the VRAM of the cards
                     it occupies" -- which is also the number an operator asking
                     "will another model fit beside it" needs.

    A selected model holding no cards contributes none, so a selection of only
    such rows yields empty GPU series rather than silently widening to the box.

    Returns:
      {
        "range": "24h",
        "now_minute": int,
        "since_minute": int,
        "selected_model_ids": [str, ...] | None,   # echo; null when unfiltered
        "selected_gpu_indices": [int, ...] | None, # the cards it resolved to
        "current": {
          "vram_used_mib": int | None,
          "vram_total_mib": int | None,
          "vram_pct": int | None,    # 0..100, rounded
          "gpu_util_pct": int | None,  # max across GPUs at most-recent minute
          "power_w": float | None,   # sum across GPUs at most-recent minute
          "tps": float,              # tokens-per-second over last full minute
                                     # (prompt + completion)
          # The GPU probe right now (#255). While it is "failing" the four
          # GPU numbers and power_w are null: the newest sample is from
          # before the probe broke, and showing it as current would hide
          # that the warden has lost its GPUs.
          "gpu_probe": {"state": "unknown"|"ok"|"failing"|"absent",
                        "error": str | None},
        },
        "active_models": [
          {"id": str, "served_model_name": str, "gpu_indices": [int, ...]}, ...
        ],
        "series": {
          "vram": [{"minute": int, "used_mib": int, "total_mib": int}, ...],
          "util": [{"minute": int, "max_pct": int}, ...],
          "power": [{"minute": int, "watts": float}, ...],     # sum / minute
          "tokens": [{"minute": int, "prompt": int, "completion": int}, ...],
        }
      }

    NOTE ON THE TOKEN SOURCE. The tokens series and ``tps`` come from
    ``model_samples``, not ``token_usage_minute``. They were the latter, which
    is keyed by API token and has no model dimension -- so it cannot be filtered
    by model at all, and using it for the unfiltered case while using
    ``model_samples`` for the filtered one would make "every model selected"
    disagree with "no filter". One source keeps the selection a partition. The
    numbers move slightly: ``model_samples`` also counts requests made without
    an API token, which ``token_usage_minute`` deliberately skips so the NULL
    key does not accumulate orphan rows. ``/api/stats/v2/tokens-per-key`` still
    reads ``token_usage_minute`` -- that endpoint asks a per-key question, which
    is the one that table answers.
    """
    _validate_range(range)
    since = _since_minute(range)
    now_min = int(time.time() // 60)
    settings = request.app.state.settings
    selected = _parse_models(models)
    async with open_db(settings.db_path) as db:
        # Resolved first: it validates the ids (400 on unknown) before any
        # aggregation runs, so a bad request never costs four table scans.
        gpu_indices = (
            await _resolve_selection(db, selected) if selected is not None else None
        )
        # Bound into every card-scoped query below. EMPTY when unfiltered, so
        # the unfiltered path executes exactly the SQL it always did rather
        # than a filtered query that happens to select everything.
        if gpu_indices is None:
            gpu_where, gpu_args = "", ()
        elif gpu_indices:
            gpu_where = f" AND gpu_index IN ({','.join('?' for _ in gpu_indices)})"
            gpu_args = tuple(gpu_indices)
        else:
            # A selection that occupies no card. `IN ()` is a syntax error in
            # SQLite, and a bare 1=1 would silently widen to the whole box --
            # the exact substitution this route exists to prevent.
            gpu_where, gpu_args = " AND 0", ()
        # power_samples names the column gpu_idx, not gpu_index.
        pwr_where = gpu_where.replace("gpu_index", "gpu_idx")
        if selected is None:
            model_where, model_args = "", ()
        else:
            model_where = f" AND model_id IN ({','.join('?' for _ in selected)})"
            model_args = tuple(selected)
        # ---- series: GPU util + VRAM, aggregated across the SELECTED cards --
        cur = await db.execute(
            "SELECT minute, "
            "       MAX(utilization_pct) AS max_util, "
            "       SUM(memory_used_mib) AS used_mib, "
            "       SUM(memory_total_mib) AS total_mib "
            "FROM gpu_samples WHERE minute >= ?" + gpu_where + " "
            "GROUP BY minute ORDER BY minute ASC",
            (since, *gpu_args),
        )
        gpu_rows = await cur.fetchall()

        # ---- series: power.draw, summed across the SELECTED cards ----------
        # Each row in power_samples is (gpu_idx, minute, watts_sum, samples)
        # — to get per-minute average watts we first average within each GPU
        # (watts_sum/samples) then sum across the cards in scope.
        cur = await db.execute(
            "SELECT minute, SUM(watts_sum / NULLIF(samples, 0)) AS box_watts "
            "FROM power_samples WHERE minute >= ?" + pwr_where + " "
            "GROUP BY minute ORDER BY minute ASC",
            (since, *gpu_args),
        )
        power_rows = await cur.fetchall()

        # ---- series: tokens, summed across the SELECTED models per minute ---
        cur = await db.execute(
            "SELECT minute, "
            "       COALESCE(SUM(prompt_tokens), 0) AS prompt, "
            "       COALESCE(SUM(completion_tokens), 0) AS completion "
            "FROM model_samples WHERE minute >= ?" + model_where + " "
            "GROUP BY minute ORDER BY minute ASC",
            (since, *model_args),
        )
        token_rows = await cur.fetchall()

        # ---- current snapshot (most recent minute in window) ----------------
        # The inner MAX(minute) carries the same filter as the outer query: the
        # "latest minute" has to be the latest minute THIS SELECTION reported,
        # or a card that stopped sampling would read as zero against a minute
        # only its neighbours reached.
        cur = await db.execute(
            "SELECT SUM(memory_used_mib), SUM(memory_total_mib), MAX(utilization_pct) "
            "FROM gpu_samples WHERE minute = ("
            "  SELECT MAX(minute) FROM gpu_samples WHERE minute >= ?" + gpu_where +
            ")" + gpu_where,
            (since, *gpu_args, *gpu_args),
        )
        cur_gpu = await cur.fetchone() or (None, None, None)

        cur = await db.execute(
            "SELECT SUM(watts_sum / NULLIF(samples, 0)) "
            "FROM power_samples WHERE minute = ("
            "  SELECT MAX(minute) FROM power_samples WHERE minute >= ?" + pwr_where +
            ")" + pwr_where,
            (since, *gpu_args, *gpu_args),
        )
        cur_power = (await cur.fetchone() or (None,))[0]

        cur = await db.execute(
            "SELECT COALESCE(SUM(prompt_tokens), 0) + COALESCE(SUM(completion_tokens), 0) "
            "FROM model_samples WHERE minute = ("
            "  SELECT MAX(minute) FROM model_samples WHERE minute >= ?" + model_where +
            ")" + model_where,
            (since, *model_args, *model_args),
        )
        last_min_tokens = (await cur.fetchone() or (0,))[0] or 0

        # ---- active model list ---------------------------------------------
        # ``models.status='loaded'`` AND ``model_runtime`` row exists, same as
        # header/routes_api.py::_active_models. NOT filtered by the selection:
        # this list is what the selector itself is built from, so narrowing it
        # would make a deselected model impossible to select again.
        cur = await db.execute(
            "SELECT m.id, m.served_model_name, m.gpu_indices "
            "FROM models m JOIN model_runtime r ON r.model_id = m.id "
            "WHERE m.status = 'loaded' "
            "ORDER BY m.served_model_name ASC"
        )
        active_rows = await cur.fetchall()

    used_mib = int(cur_gpu[0] or 0)
    total_mib = int(cur_gpu[1] or 0)
    vram_used: int | None = used_mib
    vram_total: int | None = total_mib
    vram_pct: int | None = int(round(100.0 * used_mib / total_mib)) if total_mib else 0
    util_pct: int | None = int(cur_gpu[2] or 0)
    power_w: float | None = float(cur_power) if cur_power is not None else None
    probe = gpu_probe_health()
    if probe.state == "failing":
        vram_used = vram_total = vram_pct = util_pct = None
        power_w = None
    # TPS = total tokens in last full minute / 60s. Floor at 0.
    tps = float(last_min_tokens) / 60.0 if last_min_tokens else 0.0

    return {
        "range": range,
        "now_minute": now_min,
        "since_minute": since,
        # Echoed so the page can state what its numbers cover instead of
        # relying on the checkboxes still matching the response in flight.
        # Null, not the full id list, when unfiltered: "every model" and "these
        # N models that happen to be all of them" answer different questions
        # about the cards, and the difference must survive the round trip.
        "selected_model_ids": selected,
        "selected_gpu_indices": gpu_indices,
        "current": {
            "vram_used_mib": vram_used,
            "vram_total_mib": vram_total,
            "vram_pct": vram_pct,
            "gpu_util_pct": util_pct,
            "power_w": power_w,
            "tps": tps,
            "gpu_probe": {"state": probe.state, "error": probe.error},
        },
        "active_models": [
            {
                "id": r[0],
                "served_model_name": r[1],
                # So the selector can say WHICH cards a checkbox brings in --
                # the operator's next question after "why did VRAM drop when I
                # deselected that model".
                "gpu_indices": _decode_gpu_indices(r[2]),
            }
            for r in active_rows
        ],
        "series": {
            "vram": [
                {"minute": r[0], "used_mib": int(r[2] or 0), "total_mib": int(r[3] or 0)}
                for r in gpu_rows
            ],
            "util": [
                {"minute": r[0], "max_pct": int(r[1] or 0)} for r in gpu_rows
            ],
            "power": [
                {"minute": r[0], "watts": float(r[1])} for r in power_rows
                if r[1] is not None
            ],
            "tokens": [
                {"minute": r[0], "prompt": int(r[1]), "completion": int(r[2])}
                for r in token_rows
            ],
        },
    }


@router.get("/api/stats/v2/tokens-per-key")
async def stats_v2_tokens_per_key(
    request: Request, range: str = "24h", _user: str = Depends(require_jwt)
):
    """Per-API-key token usage over the range.

    JOIN ``token_usage_minute`` onto ``api_tokens`` so the response carries
    the human-readable name (``name``) alongside the opaque ``token_id``.
    Tokens that haven't been used in the window are omitted (no row → no entry,
    keeps the response bounded by activity rather than the token catalog).

    Returns:
      {
        "range": "24h",
        "since_minute": int,
        "rows": [
          {
            "token_id": str,
            "name": str,           # api_tokens.name (or "(unknown)" for orphan)
            "prefix": str | None,  # api_tokens.prefix, helps disambiguate
            "requests": int,
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,   # prompt + completion (server-computed)
          }, ...
        ]
      }

    Rows are sorted by total_tokens DESC so the heaviest keys are first.
    """
    _validate_range(range)
    since = _since_minute(range)
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        # LEFT JOIN: keep rows even if the api_tokens entry has been deleted —
        # surfacing the orphan token_id is more useful than silently dropping
        # historical usage that's still in the rollup.
        cur = await db.execute(
            "SELECT u.token_id, "
            "       COALESCE(t.name, '(unknown)') AS name, "
            "       t.prefix AS prefix, "
            "       SUM(u.requests) AS requests, "
            "       SUM(u.prompt_tokens) AS prompt_tokens, "
            "       SUM(u.completion_tokens) AS completion_tokens "
            "FROM token_usage_minute u "
            "LEFT JOIN api_tokens t ON t.id = u.token_id "
            "WHERE u.minute >= ? "
            "GROUP BY u.token_id, t.name, t.prefix "
            "ORDER BY (SUM(u.prompt_tokens) + SUM(u.completion_tokens)) DESC, "
            "         u.token_id ASC",
            (since,),
        )
        rows = await cur.fetchall()
    return {
        "range": range,
        "since_minute": since,
        "rows": [
            {
                "token_id": r[0],
                "name": r[1],
                "prefix": r[2],
                "requests": int(r[3] or 0),
                "prompt_tokens": int(r[4] or 0),
                "completion_tokens": int(r[5] or 0),
                "total_tokens": int((r[4] or 0) + (r[5] or 0)),
            }
            for r in rows
        ],
    }


# ============================================================================
# Per-request history -- /api/stats/v2/requests and /api/stats/v2/latency.
#
# Both read the request_history table (app/stats/request_history.py), which
# the proxy feeds on every completed /v1 request. They honour the same
# ``range`` and ``models`` rules as the other v2 endpoints, and every response
# carries a ``coverage`` block so the page can say precisely what a window
# covers -- "history begins 3 h ago" is a different statement from "no
# requests in the last 24 h", and a chart that shows a narrower span than the
# button promises, without saying so, is the failure this page was rebuilt to
# remove.
# ============================================================================

_REQUESTS_LIMIT_DEFAULT = 2000
_REQUESTS_LIMIT_MAX = 5000
_LATENCY_LAST_N_DEFAULT = 500
_LATENCY_LAST_N_MAX = 5000


async def _validate_selection(db: aiosqlite.Connection, selected: list[str] | None) -> None:
    """400 on an unknown model id, exactly as the overview does.

    The selector is built from loaded models, so an id it cannot offer is a
    client defect; a history row for a since-deleted model is reachable only
    without a filter, which is the honest outcome for a model nobody can
    select any more.
    """
    if selected is not None:
        await _resolve_selection(db, selected)


def _coverage(
    *, earliest: float | None, since: float, retention_days: int, max_rows: int
) -> dict[str, Any]:
    """What the store can say about a window starting at ``since``.

    ``covers_window`` is True only when history reaches back to ``since`` AND
    the configured retention is at least as long as the window -- a 7d window
    against a 1-day retention can never be served in full, and the page should
    say that rather than draw one day under a "7d" heading.
    """
    window_s = max(0.0, time.time() - since)
    retention_ok = retention_days * 86400.0 >= window_s
    reaches_back = earliest is not None and earliest <= since
    return {
        "earliest_epoch": earliest,
        "retention_days": retention_days,
        "max_rows": max_rows,
        "covers_window": bool(retention_ok and reaches_back),
    }


@router.get("/api/stats/v2/requests")
async def stats_v2_requests(
    request: Request,
    range: str = "1h",
    models: str | None = None,
    limit: int = _REQUESTS_LIMIT_DEFAULT,
    _user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """Completed requests in the window, newest first, for the requests chart.

    Returns:
      {
        "range": "1h",
        "since_epoch": float,
        "now_epoch": float,
        "selected_model_ids": [str, ...] | None,
        "total": int,          # rows in the window for the selection
        "stride": int,         # 1 = every row; k = every k-th row (see below)
        "requests": [ {finished_record fields + finished_at}, ... ],
        "coverage": {
          "earliest_epoch": float | None,   # when history begins, store-wide
          "retention_days": int,
          "max_rows": int,
          "covers_window": bool,
        },
      }

    ``limit`` caps the rows returned (default 2000, max 5000). When the window
    holds more, every k-th row is returned rather than the newest ``limit``:
    the chart has a TIME axis, and a newest-N cut would leave its left side
    empty while the heading claimed the whole window. ``stride`` says which k,
    and ``total`` says how many rows the window really holds.
    """
    _validate_range(range)
    since = time.time() - _RANGE_TO_MINUTES[range] * 60.0
    settings = request.app.state.settings
    selected = _parse_models(models)
    limit = max(1, min(int(limit), _REQUESTS_LIMIT_MAX))
    async with open_db(settings.db_path) as db:
        await _validate_selection(db, selected)
        window = await request_history.query_window(
            db, since=since, model_ids=selected, limit=limit
        )
        earliest = await request_history.earliest_finished_at(db)
    return {
        "range": range,
        "since_epoch": since,
        "now_epoch": time.time(),
        "selected_model_ids": selected,
        "total": window.total,
        "stride": window.stride,
        "requests": window.rows,
        "coverage": _coverage(
            earliest=earliest,
            since=since,
            retention_days=int(settings.request_history_retention_days),
            max_rows=int(settings.request_history_max_rows),
        ),
    }


@router.get("/api/stats/v2/latency")
async def stats_v2_latency(
    request: Request,
    range: str = "1h",
    models: str | None = None,
    basis: str = "window",
    n: int = _LATENCY_LAST_N_DEFAULT,
    _user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """TTFT, per-request mean ITL and duration distributions from the store.

    Two bases, and the response says which:

      basis=window   every request that finished inside ``range``. The same
                     scope as the rest of the page.
      basis=last     the newest ``n`` requests regardless of time (default
                     500, max 5000). A fixed count does not go silent when
                     traffic is thin, which makes it the better statistic for
                     latency on a quiet deployment; ``span_s`` says how far
                     back those ``n`` reach so the page can state it.

    All three series are the PROXY's measurements, identical for every
    backend. ``itl`` is the mean inter-token gap of each request -- a
    per-request statistic, not the engine's per-token histogram -- and is
    labelled that way wherever it is shown. Quantiles are exact, from the
    samples; ``buckets`` use vLLM's own edges (e2e for duration) in the
    engine-histogram wire shape so the existing renderer draws them.

    Returns:
      {
        "basis": "window" | "last", "range": str, "n": int | None,
        "since_epoch": float | None,     # window basis only
        "selected_model_ids": [...] | None,
        "count": int, "span_s": float | None,
        "oldest_epoch": float | None, "newest_epoch": float | None,
        "ttft": Dist, "itl": Dist, "duration": Dist,
        "coverage": {... as /api/stats/v2/requests ...},
      }
      Dist = {"count", "p50", "p90", "p99", "mean",
              "buckets": {"le": [..., null], "counts": [...cumulative],
                          "count", "sum"}}
    """
    _validate_range(range)
    if basis not in ("window", "last"):
        raise HTTPException(
            status_code=400, detail=f"invalid basis '{basis}'; allowed: ['last', 'window']"
        )
    n = max(1, min(int(n), _LATENCY_LAST_N_MAX))
    since = time.time() - _RANGE_TO_MINUTES[range] * 60.0
    settings = request.app.state.settings
    selected = _parse_models(models)
    async with open_db(settings.db_path) as db:
        await _validate_selection(db, selected)
        if basis == "window":
            # No row cap here: the distribution must be over EVERY request in
            # the window, not a sample of it. Four columns a row keeps a full
            # 7d at the retention cap well under a second.
            rows = await request_history.query_window_stats(
                db, since=since, model_ids=selected
            )
        else:
            rows = await request_history.query_last(db, n=n, model_ids=selected)
        earliest = await request_history.earliest_finished_at(db)
    summary = request_history.latency_summary(rows)
    return {
        "basis": basis,
        "range": range,
        "n": n if basis == "last" else None,
        "since_epoch": since if basis == "window" else None,
        "selected_model_ids": selected,
        **summary,
        "coverage": _coverage(
            earliest=earliest,
            since=since,
            retention_days=int(settings.request_history_retention_days),
            max_rows=int(settings.request_history_max_rows),
        ),
    }


_THROUGHPUT_BASES = ("request", "wallclock")
# model_samples is pruned on a fixed 7-day retention by the stats pruner
# (app/runtime/stats_pruner.py:RETENTION_MINUTES); unlike request_history it
# has no configurable setting and no row cap.
_SAMPLES_RETENTION_DAYS = 7


async def _wallclock_rates(
    db: aiosqlite.Connection, *, since_minute: int, selected: list[str] | None
) -> tuple[list[float], list[float], float | None]:
    """Per-minute prefill and generation tok/s, idle minutes included as zero.

    Returns ``(prefill, generation, earliest_epoch)``.

    The zero-fill is what makes this basis answer "how much work did the box
    do" rather than "how fast was it when it was busy": a minute with no row
    in ``model_samples`` is a minute the deployment served nothing, and
    averaging only over the busy minutes would report the engine's speed under
    a heading that claims to describe the window.

    It is bounded by the store on the left and by the clock on the right:

      left   ``MIN(minute)`` over the WHOLE table, so a 7d window over a
             deployment that booted an hour ago does not report six days of
             idle minutes nobody has evidence for. Unfiltered by the selection
             on purpose -- a quiet model over a busy hour really was idle for
             it, and filtering here would hide exactly the thing being asked.
      right  the last COMPLETE minute. The current minute is still being
             written and would read as a partial, artificially slow one.
    """
    where = ""
    args: tuple[str, ...] = ()
    if selected is not None:
        where = f" AND model_id IN ({','.join('?' for _ in selected)})"
        args = tuple(selected)
    cur = await db.execute(
        "SELECT minute, COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0) "
        "FROM model_samples WHERE minute >= ?" + where + " GROUP BY minute",
        (since_minute, *args),
    )
    by_minute = {int(r[0]): (float(r[1]), float(r[2])) for r in await cur.fetchall()}

    cur = await db.execute("SELECT MIN(minute) FROM model_samples")
    earliest_minute = (await cur.fetchone() or (None,))[0]
    if earliest_minute is None:
        return [], [], None
    earliest_epoch = float(earliest_minute) * 60.0
    first = max(int(since_minute), int(earliest_minute))
    last = int(time.time() // 60) - 1
    if last < first:
        return [], [], earliest_epoch
    prefill: list[float] = []
    generation: list[float] = []
    for minute in range(first, last + 1):
        prompt, completion = by_minute.get(minute, (0.0, 0.0))
        prefill.append(prompt / 60.0)
        generation.append(completion / 60.0)
    return prefill, generation, earliest_epoch


@router.get("/api/stats/v2/throughput")
async def stats_v2_throughput(
    request: Request,
    range: str = "1h",
    models: str | None = None,
    basis: str = "request",
    _user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """Prefill and generation tokens per second, as average / max / mode.

    This exists because the overview's single ``current.tps`` is
    ``(prompt + completion) / 60`` over the last full minute -- one number
    blending two quantities that move independently. Prefill is compute-bound
    and runs in the hundreds or thousands of tok/s; generation is
    memory-bandwidth-bound and runs in the tens. Their average describes
    neither, and it changes when the prompt-to-completion ratio changes even
    though the hardware did not.

    Two bases, and the response says which:

      basis=request    per-request engine speed from ``request_history``:
                       ``prompt_tokens / ttft`` and
                       ``(completion_tokens - 1) / (duration - ttft)``.
                       It does NOT answer "how fast is the rig", and the
                       panel defaults to wallclock for that reason. It is
                       wrong in both directions on a real workload: the prompt
                       figure counts prefix-cached tokens the engine never
                       computed (measured ~26k tok/s where the hardware can do
                       roughly 1/13th of that), and the decode figure is one
                       request's share of a batched engine rather than the
                       engine's aggregate. What it IS good for is comparing
                       requests with each other. Also note the ENGINE's own
                       waiting queue is inside TTFT and invisible from the
                       proxy; the WARDEN's admission queue is not -- that is
                       ``queued_s``, migration 0032.
      basis=wallclock  tokens counted per minute from ``model_samples``, with
                       idle minutes included as zero. Answers "how much work
                       did the box do".

    ``mode`` is the modal value over bins ~5% wide (see
    ``request_history.mode_relative``) and is null when the sample is too
    small or nothing repeats. On a bimodal prefill series it is the statistic
    worth reading; ``max`` there is usually a cache artefact.

    Returns:
      {
        "basis": "request" | "wallclock",
        "range": str,
        "since_epoch": float,
        "selected_model_ids": [str, ...] | None,
        "prefill":    {"count": int, "avg": float|None,
                       "max": float|None, "mode": float|None},
        "generation": {... same ...},
        "coverage": {... as /api/stats/v2/requests, for the store this basis
                     reads ...},
      }
    """
    _validate_range(range)
    if basis not in _THROUGHPUT_BASES:
        raise HTTPException(
            status_code=400,
            detail=f"invalid basis '{basis}'; allowed: {sorted(_THROUGHPUT_BASES)}",
        )
    since = time.time() - _RANGE_TO_MINUTES[range] * 60.0
    settings = request.app.state.settings
    selected = _parse_models(models)
    async with open_db(settings.db_path) as db:
        await _validate_selection(db, selected)
        if basis == "request":
            rows = await request_history.query_window_stats(
                db, since=since, model_ids=selected
            )
            prefill = [
                v for v in (request_history.prefill_tps_of(r) for r in rows)
                if v is not None
            ]
            generation = [
                v for v in (request_history.gen_tps_of(r) for r in rows)
                if v is not None
            ]
            earliest = await request_history.earliest_finished_at(db)
            retention_days = int(settings.request_history_retention_days)
            max_rows = int(settings.request_history_max_rows)
        else:
            prefill, generation, earliest = await _wallclock_rates(
                db, since_minute=int(since // 60), selected=selected
            )
            # The minute rollup has a fixed retention and no row cap, so
            # coverage reports the store this basis actually read rather than
            # request_history's, which governs nothing here.
            retention_days = _SAMPLES_RETENTION_DAYS
            max_rows = 0
    return {
        "basis": basis,
        "range": range,
        "since_epoch": since,
        "selected_model_ids": selected,
        "prefill": request_history.rate_summary(prefill),
        "generation": request_history.rate_summary(generation),
        "coverage": _coverage(
            earliest=earliest,
            since=since,
            retention_days=retention_days,
            max_rows=max_rows,
        ),
    }


@router.get("/api/stats/live/finished")
async def stats_live_finished(
    request: Request,
    models: str | None = None,
    limit: int = 50,
    _user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """Requests that have COMPLETED, newest first. Superseded.

    Kept for a UI image older than this API: ui and api ship separately, and
    the previous stats page polls this for its finished table. It now reads
    the persisted store, so ``retained_seconds`` is the configured retention
    rather than the old ring's 15 minutes. New code reads
    ``/api/stats/v2/requests``, which carries the window and its coverage.

    ``models`` follows the same absent/empty rules as every other stats
    endpoint: absent means the whole deployment, empty is a client defect and
    a 400.
    """
    selection = _parse_models(models)
    settings = request.app.state.settings
    if getattr(request.app.state, "request_history", None) is None:
        # A build without the store answers honestly rather than 500ing.
        return {"requests": [], "retained_seconds": 0, "available": False}
    limit = max(1, min(int(limit), 200))
    async with open_db(settings.db_path) as db:
        rows = await request_history.query_last(db, n=limit, model_ids=selection)
    return {
        "requests": rows,
        "retained_seconds": int(settings.request_history_retention_days) * 86400,
        "available": True,
    }
