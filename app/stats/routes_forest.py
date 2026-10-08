"""Session forest API (spec 2026-10-05 §4, §10.3).

The app runs ONE uvicorn worker, so a slow handler here stalls every proxied stream. Building the
forest is pure Python over up to the whole retention window (seconds at 200k rows), so:

- The load, JSON decode, build and serialization run in a worker thread (``asyncio.to_thread``)
  with their own short-lived ``sqlite3`` connection. One thread hop does SQL, decode and build
  together; with aiosqlite the rows would come back to the loop and be decoded there, which is
  most of the load cost. The thread still shares the GIL, but CPython hands the GIL back every
  5 ms, so the loop keeps serving streams instead of stopping for the whole build.
- One process-level ``_Store`` holds the forest of the last WHOLE_S (7 d), one ``forest.TokenForest``
  per token, shared by every viewer and range (rows are held once; re-review 2, R2). It is refreshed
  single-flight at most every ``FOREST_TTL_S``. An admin's response composes every token's part, a
  key holder's only its own token's.
- A refresh is incremental (review N3): it loads only the rows committed since the last refresh (by
  rowid) and drops those that slid out of the window; only the trees whose sessions changed are
  rebuilt and re-serialized, every other tree is reused as JSON. Every ``FULL_REBUILD_S``, after the
  table shrank, or after the history pruner deleted rows that may be inside the window (its
  generation counter, R3), everything is rebuilt and re-anchored.
- A refresh is transactional (R1): if it fails, nothing is published and the store starts over
  with a full build on the next refresh.
- Held rows are bounded (re-review 3, N1): a full build gives each token at most its max-min fair
  share of ``MAX_HELD_ROWS`` (its newest rows), loading one token at a time, so the rows held never
  exceed the budget, during the build or after. A token over its share is served from those rows,
  stable, with ``truncated_before`` in the response; it is never rebuilt per request. New rows that
  take the store past the budget (plus ``HELD_SLACK``) move the largest tokens' floors up.
- Readers see one consistent snapshot (N2): a refresh publishes an immutable ``_View`` in one
  assignment; composing never reads the refresh's working state.
- The JSON is gzip-compressed for clients that accept it, on this route only (a global GZip
  middleware would buffer the proxied SSE streams).
- Whole trees (follow-up C1): the range picks trees, it never trims them. Every range up to 7 d
  shares one state built over the last WHOLE_S (7 d), and a request sends the trees with a turn
  finished in [now - range, now], each with every turn of those 7 d. So a range switch or the
  window's slide never cuts a tree the viewer already sees.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import sqlite3
import struct
import time
import zlib
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jwt as pyjwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from app.auth.deps import bearer_token, require_jwt, token_refusal
from app.auth.jwt import decode, mint_forest
from app.db.database import open_db
from app.db.repos.tokens import TokenRepo, TokenRow
from app.proxy.auth import require_bearer
from app.stats import forest, request_history

router = APIRouter()

RANGES = {
    "1h": 3600,
    "6h": 6 * 3600,
    "24h": 86400,
    "48h": 2 * 86400,
    "7d": 7 * 86400,
}  # nothing above WHOLE_S: the forest never holds rows older than 7 d (re-review 2, R2)
#: Trees are built over at least this much history and sent whole (see the module docstring).
WHOLE_S = 7 * 86400
SINCE_OVERLAP_S = 5.0
FLOWER_RESEND_S = 600.0
TPUT_TTL_S = 60.0
FOREST_TTL_S = 3.0  # a cached state answers every request for this long
FULL_REBUILD_S = (
    6 * 3600.0
)  # then everything is rebuilt and re-anchored (the window's slide is incremental)
MAX_CACHED_KEYS = 32  # composed per-viewer responses kept (cheap: they share the store's parts)
CACHE_IDLE_S = 600.0  # a composed state nobody asked for in this long is dropped
#: The store holds at most this many rows; past it the least recently used tokens' rows are evicted
#: (at ~600 B a held row, about 180 MB).
MAX_HELD_ROWS = 300_000
#: New rows may take the store this far over the budget before the largest tokens' oldest rows are dropped.
HELD_SLACK = 0.05
MAX_NEW_ROWS = 50_000  # more new rows than this in one refresh: rebuild everything instead
GZIP_LEVEL = 5
FOREST_TTL_MIN = 12 * 60
_TPUT_FIELDS = (
    "variant_id",
    "ttft_s",
    "duration_s",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
)
_FIELDS = (
    "id", "finished_at", "duration_s", "ttft_s", "token_id", "token_name", "model_id", "variant_id",
    "prompt_tokens", "completion_tokens", "cached_tokens", "finish_reason", "session_key",
    "parent_session_key", "batch_id", "tools_in", "tools_out",
)  # fmt: skip


@dataclass(frozen=True)
class ForestViewer:
    #: None = admin (every key); otherwise only this api_tokens row id.
    token_id: str | None


async def forest_viewer(request: Request) -> ForestViewer:
    tok = bearer_token(request)
    if tok and not tok.startswith("vw"):
        try:
            claims = decode(tok, request.app.state.jwt_secret)
        except pyjwt.PyJWTError:
            claims = {}
        if claims.get("typ") == "forest":
            async with open_db(request.app.state.settings.db_path) as db:
                row = await TokenRepo(db).get(str(claims.get("sub")))
                cur = await db.execute("SELECT datetime('now')")
                fetched = await cur.fetchone()
                assert fetched is not None  # SELECT datetime('now') always yields one row
                (now,) = fetched
            if row is None or row.scope != "inference":
                raise HTTPException(401, "unknown token")
            refusal = token_refusal(row, now)
            if refusal is not None:
                raise refusal
            return ForestViewer(token_id=row.id)
    await require_jwt(request)
    return ForestViewer(token_id=None)


@router.post("/api/forest/login")
async def forest_login(request: Request, row: TokenRow = Depends(require_bearer)) -> dict[str, Any]:
    """Exchange an inference API key for a forest-only JWT (see ``mint_forest``)."""
    return {
        "token": mint_forest(row.id, request.app.state.jwt_secret, FOREST_TTL_MIN),
        "expires_in": FOREST_TTL_MIN * 60,
    }


def _now() -> float:
    return time.time()


def _tools_in_ok(e: Any) -> bool:
    return (
        isinstance(e, list)
        and len(e) == 3
        and isinstance(e[0], str)
        and isinstance(e[1], int | float)
        and not isinstance(e[1], bool)
        and isinstance(e[2], bool)
    )


def _tools_out_ok(e: Any) -> bool:
    return isinstance(e, list) and len(e) == 2 and isinstance(e[0], str) and isinstance(e[1], str)


def _decode(v: Any, ok: Callable[[Any], bool]) -> list[Any]:
    """A tools_in/tools_out column: the well-formed entries of a JSON list, nothing else.

    One malformed row must not 500 the forest for everyone, so entries of the wrong shape are
    dropped here and the assembly only ever sees lists of the shape it indexes into.
    """
    if not v:
        return []
    try:
        x = json.loads(v)
    except (TypeError, ValueError):
        return []
    return [e for e in x if ok(e)] if isinstance(x, list) else []


def _dumps(v: Any) -> bytes:
    # Same settings as FastAPI's JSONResponse.
    return json.dumps(v, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def _connect(db_path: Path) -> sqlite3.Connection:
    """A read-only-by-pragma connection for the worker thread (WAL: readers never block)."""
    db = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    db.execute("PRAGMA query_only = ON")
    return db


def _token_clause(tokens: Iterable[str]) -> tuple[str, list[Any]]:
    named = sorted(t for t in tokens if t != "-")
    conds, args = [], []
    if named:
        conds.append(f"token_id IN ({','.join('?' * len(named))})")
        args += named
    if "-" in tokens:  # forest.token_of: rows without a token
        conds.append("token_id IS NULL OR token_id IN ('', '-')")
    return " AND (" + " OR ".join(conds) + ")", args


def _load(
    db: sqlite3.Connection,
    ws: float,
    token_id: str | None,
    tokens: set[str] | None,
    upto_rowid: int | None = None,
    newest: int | None = None,
) -> list[forest._Row]:
    """Rows of the window, decoded into held (slim) rows; only ``tokens``' rows when given, and only
    rows committed up to ``upto_rowid`` when given. Streamed: no list of full dict rows is built."""
    sql = f"SELECT {', '.join(_FIELDS)} FROM request_history WHERE finished_at >= ?"
    args: list[Any] = [ws]
    if upto_rowid is not None:
        sql += " AND rowid <= ?"
        args.append(upto_rowid)
    if token_id is not None:
        sql += " AND token_id = ?"
        args.append(token_id)
    if tokens is not None:
        clause, targs = _token_clause(tokens)
        sql += clause
        args += targs
    if (
        newest is not None
    ):  # only the newest rows (a token over its share of the budget), oldest first
        rows = [
            _slim(row)
            for row in db.execute(sql + " ORDER BY finished_at DESC LIMIT ?", [*args, newest])
        ]
        rows.reverse()
        return rows
    return [_slim(row) for row in db.execute(sql + " ORDER BY finished_at", args)]


def _counts(db: sqlite3.Connection, ws: float, upto_rowid: int) -> dict[str, int]:
    """Rows in the window per forest token (one indexed pass)."""
    out: dict[str, int] = {}
    for tok, n in db.execute(
        "SELECT token_id, COUNT(*) FROM request_history"
        " WHERE finished_at >= ? AND rowid <= ? GROUP BY token_id",
        [ws, upto_rowid],
    ):
        k = forest.token_of({"token_id": tok})
        out[k] = out.get(k, 0) + n
    return out


def _shares(counts: dict[str, int], budget: int) -> dict[str, int]:
    """Max-min fair shares of ``budget`` rows: a token with fewer rows than its fair share keeps them all and leaves the
    rest to the others; the larger ones get equal shares of what remains."""
    out: dict[str, int] = {}
    left, todo = budget, sorted(counts.items(), key=lambda kv: kv[1])
    for i, (tok, n) in enumerate(todo):
        take = min(n, left // max(1, len(todo) - i))
        out[tok] = take
        left -= take
    return out


def _slim(row: tuple[Any, ...]) -> forest._Row:
    d = dict(zip(_FIELDS, row, strict=True))
    d["tools_in"] = _decode(d["tools_in"], _tools_in_ok)
    d["tools_out"] = _decode(d["tools_out"], _tools_out_ok)
    r = forest._Row(d)
    r.token_id = d["token_id"]
    return r


def _load_new(
    db: sqlite3.Connection, after_rowid: int, upto_rowid: int, ws: float, token_id: str | None
) -> list[forest._Row]:
    """The window's rows committed since the last refresh, decoded.

    By rowid (commit order), not finished_at: the history writer commits in batches up to a
    second late, and a finished_at cut would need an overlap that rebuilds the latest token on
    every refresh even when nothing new arrived. Rowids only grow here (prune deletes the oldest).
    """
    sql = (
        f"SELECT {', '.join(_FIELDS)} FROM request_history"
        " WHERE rowid > ? AND rowid <= ? AND finished_at >= ?"
    )
    args: list[Any] = [after_rowid, upto_rowid, ws]
    if token_id is not None:
        sql += " AND token_id = ?"
        args.append(token_id)
    return [_slim(row) for row in db.execute(sql + " ORDER BY finished_at", args)]


def _throughput_sync(db_path: Path, now: float) -> dict[str, dict[str, Any]]:
    """7-day per-variant throughput, global (not per viewer)."""
    with closing(_connect(db_path)) as db:
        cur = db.execute(
            f"SELECT {', '.join(_TPUT_FIELDS)} FROM request_history WHERE finished_at >= ?",
            [now - 7 * 86400],
        )
        rows = [dict(zip(_TPUT_FIELDS, row, strict=True)) for row in cur]
    return forest.model_throughput(rows)


@dataclass(frozen=True)
class _Tree:
    id: str
    start: float
    last_at: float
    json: bytes


@dataclass(frozen=True)
class _Part:
    """One token's slice of the forest, times relative to its state's ``anchor``."""

    trees: tuple[_Tree, ...]
    flowers: tuple[list[Any], ...]
    variants: dict[str, str]


@dataclass
class _State:
    """One viewer's composed response state (its tokens' parts of the store, at one store refresh)."""

    anchor: float  # every relative time in this state is against this (the response's t0)
    ws: float  # window start at this refresh
    at: float  # when this refresh read the database (NOT the response's "now": see _head)
    hw: float | None  # newest finished_at seen: the response's cursor
    parts: dict[str, _Part]
    ids: frozenset[str]
    new_ids: frozenset[str]  # ids the previous state of this key did not have
    from_scratch: bool  # no previous state existed: a delta cannot be trusted
    rebuilt: tuple[str, ...] | None  # tokens rebuilt by this refresh; None = all of them
    models: dict[str, dict[str, Any]]
    used: float = 0.0
    #: the full response after its head, rendered once per state and range
    tails: dict[str, _Tail] = field(default_factory=dict)
    #: the store refresh this state was composed from
    rev: int = 0
    #: over the held-row budget, some of this viewer's oldest rows are not held: trees before this time may miss wood
    truncated_before: float | None = None


@dataclass
class _Tok:
    """One token in the store: its forest, its part, its newest finish and, when the token is over its share of the
    held-row budget, the time before which its rows are not held (its oldest trees then miss that wood)."""

    tf: forest.TokenForest
    part: _Part | None
    hw: float | None
    floor: float | None = None


@dataclass(frozen=True)
class _View:
    """What readers see: one consistent snapshot, published by a refresh in one assignment (re-review 3, N2)."""

    anchor: float
    ws: float
    at: float
    rev: int
    rebuilt: tuple[str, ...] | None
    max_len: dict[str, Any]
    parts: dict[str, _Part]
    hws: dict[str, float | None]
    floors: dict[str, float]


@dataclass
class _Store:
    """The process-level forest of the last WHOLE_S, one entry per token (see the module docstring).

    Only the refresh (one at a time, in a worker thread) touches ``toks`` and the cursor; readers only ever read
    ``view``, which each successful refresh replaces whole.
    """

    anchor: float | None = None
    full_at: float = float("-inf")
    max_rowid: int = 0
    gen: int = -1
    toks: dict[str, _Tok] = field(default_factory=dict)
    rev: int = 0
    view: _View | None = None

    def held_rows(self) -> int:
        return sum(len(t.tf) for t in self.toks.values())


def _make_tree(t: dict[str, Any]) -> _Tree:
    return _Tree(t["id"], t["start"], t["last_at"], _dumps(t))


def _build_part(tf: forest.TokenForest) -> _Part:
    b = tf.build(make=_make_tree)
    return _Part(
        trees=tuple(made for _, made in b["trees"]),
        flowers=tuple(b["flowers"]),
        variants=tf.variant_models(),
    )


def _full_build(
    db: sqlite3.Connection, ws: float, max_rowid: int, anchor: float
) -> dict[str, _Tok]:
    """Every token of the window, each up to its fair share of MAX_HELD_ROWS (its newest rows): loaded and built one
    token at a time, so the rows held never exceed the budget, during the build or after (re-review 3, N1)."""
    counts = _counts(db, ws, max_rowid)
    shares = _shares(counts, MAX_HELD_ROWS)
    toks: dict[str, _Tok] = {}
    for tok, n in sorted(counts.items(), key=lambda kv: kv[1]):
        share = shares[tok]
        if share <= 0:
            continue
        cut = share < n
        rows = _load(db, ws, None, {tok}, max_rowid, newest=share if cut else None)
        if not rows:
            continue
        tf = forest.TokenForest(tok, anchor)
        tf.add(rows)
        toks[tok] = _Tok(
            tf,
            _build_part(tf),
            max(float(r.finished_at) for r in rows),
            float(rows[0].finished_at) if cut else None,
        )
    return toks


def _rebalance(store: _Store) -> set[str]:
    """New rows took the store over its budget (by more than the slack): the tokens over their fair share lose their
    oldest held rows (their floor moves up). Returns the tokens changed."""
    if store.held_rows() <= MAX_HELD_ROWS * (1 + HELD_SLACK):
        return set()
    shares = _shares({k: len(t.tf) for k, t in store.toks.items()}, MAX_HELD_ROWS)
    changed = set()
    for k, t in store.toks.items():
        n = len(t.tf)
        if n <= shares[k]:
            continue
        times = t.tf.finished_times()
        t.floor = times[n - shares[k]]
        t.tf.prune(t.floor)
        changed.add(k)
    return changed


def _store_refresh(db_path: Path, store: _Store, now: float) -> None:
    """Brings the store up to date and publishes a new view (in a worker thread; one at a time).

    On failure nothing is published (readers keep the last view) and the store is reset, so the next refresh is a full
    build (R1)."""
    ws = now - WHOLE_S
    try:
        with closing(_connect(db_path)) as db:
            max_len = {r[0]: r[1] for r in db.execute("SELECT id, max_model_len FROM models")}
            (max_rowid,) = db.execute(
                "SELECT COALESCE(MAX(rowid), 0) FROM request_history"
            ).fetchone()
            gen = request_history.prune_generation()
            full = (
                store.anchor is None
                or now - store.full_at >= FULL_REBUILD_S
                or max_rowid < store.max_rowid
                or gen != store.gen
            )
            new: list[forest._Row] = []
            if not full:
                new = _load_new(db, store.max_rowid, max_rowid, ws, None)
                full = len(new) > MAX_NEW_ROWS
            if full:
                store.toks = {}  # the old rows go before the new ones come: the peak is one budget
                anchor = ws
                store.toks = _full_build(db, ws, max_rowid, anchor)
                changed: set[str] = set(store.toks)
            else:
                assert store.anchor is not None
                anchor = store.anchor
                changed = set()
                by_token: dict[str, list[forest._Row]] = {}
                for r in new:
                    by_token.setdefault(forest.token_of({"token_id": r.token_id}), []).append(r)
                for tok, rows in by_token.items():
                    t = store.toks.get(tok)
                    if t is None:
                        t = store.toks[tok] = _Tok(forest.TokenForest(tok, anchor), None, None)
                    t.tf.add(rows)
                    hw = max(float(r.finished_at) for r in rows)
                    t.hw = hw if t.hw is None else max(t.hw, hw)
                    changed.add(tok)
        if not full:
            for tok, t in store.toks.items():
                if t.tf.prune(max(ws, t.floor or ws)):
                    changed.add(tok)
            changed |= _rebalance(store)
            for tok in list(store.toks):
                t = store.toks[tok]
                if not len(t.tf):
                    del store.toks[tok]
                elif tok in changed or t.part is None:
                    t.part = _build_part(t.tf)
    except BaseException:
        store.anchor = (
            None  # half-applied: the next refresh starts over; readers keep the last view
        )
        store.toks = {}
        raise
    store.anchor, store.max_rowid, store.gen = anchor, max_rowid, gen
    if full:
        store.full_at = now
    store.rev += 1
    store.view = _View(
        anchor=anchor,
        ws=ws,
        at=now,
        rev=store.rev,
        rebuilt=None if full else tuple(sorted(k for k in changed if k in store.toks)),
        max_len=max_len,
        parts={k: t.part for k, t in store.toks.items() if t.part is not None},
        hws={k: t.hw for k, t in store.toks.items()},
        floors={k: t.floor for k, t in store.toks.items() if t.floor is not None},
    )


def _compose(
    view: _View, token_id: str | None, prev: _State | None, tput: dict[str, dict[str, Any]]
) -> _State:
    """One viewer's state from a published view: every token's part (admin) or its own token's."""
    if token_id is None:
        keys = list(view.parts)
    else:
        tok = forest.token_of({"token_id": token_id})
        keys = [tok] if tok in view.parts else []
    parts = {k: view.parts[k] for k in keys}
    hws = [h for k in keys if (h := view.hws.get(k)) is not None]
    hw = max(hws) if hws else None
    if prev is not None and prev.hw is not None:
        hw = prev.hw if hw is None else max(hw, prev.hw)
    ids = frozenset(t.id for p in parts.values() for t in p.trees)
    seen = {v: m for p in parts.values() for v, m in p.variants.items()}
    rebuilt = view.rebuilt if view.rebuilt is None else tuple(k for k in view.rebuilt if k in parts)
    floors = [view.floors[k] for k in keys if k in view.floors]
    return _State(
        anchor=view.anchor,
        ws=view.ws,
        at=view.at,
        hw=hw,
        parts=parts,
        ids=ids,
        new_ids=ids - prev.ids if prev is not None else ids,
        from_scratch=prev is None,
        rebuilt=rebuilt,
        models={
            v: {**s, "max_model_len": view.max_len.get(seen[v])}
            for v, s in tput.items()
            if v in seen
        },
        used=view.at,
        rev=view.rev,
        truncated_before=max(floors) if floors else None,
    )


@dataclass(frozen=True)
class _Tail:
    """A body after its head (``,"trees":[...]...}``), raw and as a final raw-deflate stream."""

    raw: bytes
    deflate: bytes


def _tail(raw: bytes) -> _Tail:
    c = zlib.compressobj(GZIP_LEVEL, zlib.DEFLATED, -15)
    return _Tail(raw, c.compress(raw) + c.flush())


def _head(state: _State, range_name: str, now: float) -> bytes:
    """The body up to (not including) its trees, without the closing brace.

    ``now`` is the time of the request, not of the cached state: the client runs its clock on
    its own and uses ``now`` only to estimate the skew, so a ``now`` up to FOREST_TTL_S old
    would read as the server's clock going back and forth.
    """
    head = {"range": range_name, "t0": state.anchor, "now": now, "models": state.models}
    if state.truncated_before is not None:
        head["truncated_before"] = state.truncated_before
    return _dumps(head)[:-1]


def _render_tail(state: _State, since: float | None, ws: float | None = None) -> bytes:
    """The body after its head (in a worker thread: serializing is CPU too).

    ``ws``: the range's window start. The trees with a turn finished at or after it are sent,
    whole; flowers from an hour before it on. None: everything the state holds.
    """
    cut = state.ws if ws is None else ws
    trees = sorted(
        (t for p in state.parts.values() for t in p.trees if t.last_at >= cut),
        key=lambda t: (t.start, t.id),
    )
    flowers = sorted(
        f for p in state.parts.values() for f in p.flowers if f[0] + state.anchor >= cut - 3600
    )
    ids = [t.id for t in trees]
    full = True
    if since is not None:
        delta = [t for t in trees if t.last_at > since - SINCE_OVERLAP_S]
        # Complete only if every id the client may lack is in the delta itself. Otherwise (a tree
        # id that changed without growing, or no previous state to compare with) send everything.
        full = state.from_scratch or bool((state.new_ids & set(ids)) - {t.id for t in delta})
        if not full:
            trees = delta
            flowers = [f for f in flowers if f[0] + state.anchor > since - FLOWER_RESEND_S]
    cursor = state.hw if state.hw is not None else (since if since is not None else cut)
    tail = _dumps({"flowers": flowers, "ids": ids, "cursor": cursor, "full": full})
    return b',"trees":[' + b",".join(t.json for t in trees) + b"]," + tail[1:]


def _range_ws(state: _State, range_name: str) -> float:
    """The range's window start as of the state's refresh (so a state's renders are stable)."""
    return state.at - RANGES[range_name]


def _render(state: _State, range_name: str, since: float | None, gz: bool, now: float) -> bytes:
    """A delta response, rendered for this request alone."""
    body = _head(state, range_name, now) + _render_tail(state, since, _range_ws(state, range_name))
    return gzip.compress(body, GZIP_LEVEL) if gz else body


_GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff"  # deflate, no name, mtime 0


def _gzip_spliced(head: bytes, tail: _Tail) -> bytes:
    """One gzip member: ``head`` in stored (uncompressed) deflate blocks, then the cached tail's
    own deflate stream. A stored block ends on a byte boundary and the tail never refers back
    past its own start, so the result is a valid deflate stream; only the CRC is recomputed.
    The head is a few hundred bytes, so this costs a CRC pass, not a recompression."""
    out = [_GZIP_HEADER]
    for i in range(0, len(head), 0xFFFF):
        chunk = head[i : i + 0xFFFF]
        out.append(b"\x00" + struct.pack("<HH", len(chunk), len(chunk) ^ 0xFFFF) + chunk)
    out.append(tail.deflate)
    crc = zlib.crc32(tail.raw, zlib.crc32(head))
    out.append(struct.pack("<II", crc, (len(head) + len(tail.raw)) & 0xFFFFFFFF))
    return b"".join(out)


class ForestCache:
    """Per-app forest states and the throughput cache, each refreshed single-flight."""

    def __init__(self) -> None:
        self.store = _Store()
        self._lock = asyncio.Lock()
        self._states: OrderedDict[tuple[str, str | None], _State] = OrderedDict()
        self._flights: dict[Any, asyncio.Future[Any]] = {}
        self._tput: tuple[float, dict[str, dict[str, Any]]] | None = None

    async def _single(self, key: Any, factory: Callable[[], Awaitable[Any]]) -> Any:
        fut = self._flights.get(key)
        if fut is None:
            fut = asyncio.ensure_future(factory())
            self._flights[key] = fut

            def _done(f: asyncio.Future[Any], k: Any = key) -> None:
                if self._flights.get(k) is f:
                    del self._flights[k]
                if not f.cancelled():
                    f.exception()  # retrieved: a failure with no waiter left is not "unhandled"

            fut.add_done_callback(_done)
        # shield: a client that disconnects must not cancel the build the others wait for.
        return await asyncio.shield(fut)

    async def throughput(self, db_path: Path) -> dict[str, dict[str, Any]]:
        now = _now()
        if self._tput is not None and now - self._tput[0] < TPUT_TTL_S:
            return self._tput[1]

        async def work() -> dict[str, dict[str, Any]]:
            value = await asyncio.to_thread(_throughput_sync, db_path, now)
            self._tput = (now, value)
            return value

        result: dict[str, dict[str, Any]] = await self._single("tput", work)
        return result

    async def state(self, key: tuple[str, str | None], db_path: Path) -> _State:
        """The composed state for ``key`` = (range, viewer token or None for the admin)."""
        now = _now()
        for k in [k for k, s in self._states.items() if now - s.used > CACHE_IDLE_S]:
            del self._states[k]
        view = self.store.view
        if view is None or now - view.at >= FOREST_TTL_S:
            await self._single("store", lambda: self._refresh_store(db_path))
        tput = await self.throughput(db_path)
        # no await from here on, and one published view: a consistent snapshot, one state per refresh
        view = self.store.view
        assert view is not None
        st = self._states.get(key)
        if st is not None and st.rev == view.rev:
            st.used = now
            self._states.move_to_end(key)
            return st
        new = _compose(view, key[1], st, tput)
        self._states[key] = new
        self._states.move_to_end(key)
        while len(self._states) > MAX_CACHED_KEYS:
            self._states.popitem(last=False)
        return new

    async def _refresh_store(self, db_path: Path) -> None:
        async with self._lock:
            v = self.store.view
            if v is not None and _now() - v.at < FOREST_TTL_S:
                return
            await asyncio.to_thread(_store_refresh, db_path, self.store, _now())


def _cache(app: Any) -> ForestCache:
    cache = getattr(app.state, "forest_cache", None)
    if cache is None:
        cache = app.state.forest_cache = ForestCache()
    return cache


def _accepts_gzip(header: str) -> bool:
    """Accept-Encoding check: an explicit ``gzip`` entry wins over ``*``; a q of 0 means no gzip."""
    qs: dict[str, float] = {}
    for part in header.split(","):
        name, _, params = part.strip().partition(";")
        name = name.strip().lower()
        if name not in ("gzip", "*"):
            continue
        q = 1.0
        for param in params.split(";"):
            key, _, val = param.strip().partition("=")
            if key.strip().lower() == "q":
                try:
                    q = float(val.strip())
                except ValueError:
                    q = 0.0
        qs[name] = q
    return qs.get("gzip", qs.get("*", 0.0)) > 0


@router.get("/api/stats/forest", response_model=None)
async def stats_forest(
    request: Request,
    range: str = "48h",
    since: float | None = None,
    viewer: ForestViewer = Depends(forest_viewer),
) -> Response:
    """Session forest.

    Client contract (also in documents/API.md, "Session forest"):

    - Times in trees and flowers are relative to this response's ``t0`` (absolute = rel + t0).
      ``t0`` can change between responses, so a client holding trees from an earlier response
      rebases them by the change in ``t0``. The window ``[now - range, now]`` picks the trees with
      a turn finished in it; each is sent whole (every turn of the last 7 d), never cut at the edge.
    - ``ids`` always lists every current tree, in or out of this response. Drop any tree you
      hold whose id is not in it.
    - Poll with ``since=<cursor of the last response>``. A delta holds the whole trees with a
      turn finished after ``since - 5 s`` (replace by id) and every flower that started after
      ``since - 600 s``: flowers have no ids, so replace all flowers with t >= since - 600.
    - Tree ids can change without the tree growing (a spell's id follows its first in-window
      session; a day turning busy regroups its spells). The client rule: **if ``full`` is true or
      ``ids`` contains an id you don't hold, refetch without ``since``.** A ``full: true``
      response already carries every tree and flower, so it can stand in for that refetch.
    """
    if range not in RANGES:
        raise HTTPException(400, f"range must be one of {sorted(RANGES)}")
    db_path = request.app.state.settings.db_path
    # every range shares the store built over WHOLE_S: trees are sent whole
    state = await _cache(request.app).state(("7d", viewer.token_id), db_path)
    gz = _accepts_gzip(request.headers.get("accept-encoding", ""))
    now = _now()
    if since is None:
        tail = state.tails.get(range)
        if tail is None:
            ws = _range_ws(state, range)
            tail = state.tails[range] = await asyncio.to_thread(
                lambda: _tail(_render_tail(state, None, ws))
            )
        head = _head(state, range, now)
        # The CRC pass is over the whole body: off the loop too.
        body = await asyncio.to_thread(_gzip_spliced, head, tail) if gz else head + tail.raw
    else:
        body = await asyncio.to_thread(_render, state, range, since, gz, now)
    headers = {"Vary": "Accept-Encoding", "Cache-Control": "no-store"}
    if gz:
        headers["Content-Encoding"] = "gzip"
    return Response(body, media_type="application/json", headers=headers)
