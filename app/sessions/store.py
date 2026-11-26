"""Bounded, indexed reads for the sessions pages. Synchronous ``sqlite3``: the
service runs them in a worker thread. Every query is scoped by token and bounded
by time or ``LIMIT``; ids, sizes and counts only."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

ROOT_DEPTH = 8
MAX_LIST_ROWS = 200_000

_COLS = (
    "id",
    "token_id",
    "token_name",
    "model_id",
    "model",
    "finished_at",
    "duration_s",
    "queued_s",
    "ttft_s",
    "finish_reason",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "reusable_tokens",
    "cache_outcome",
    "cache_outcome_own",
    "cobs2_divergence_cause",
    "cobs2_divergence_cause_own",
    "dp_rank",
    "session_key",
    "parent_session_key",
    "cobs2_session",
    "cobs2_conv",
    "cobs2_node",
    "cobs2_parent_node",
    "cobs2_parent_req",
    "cobs2_edge",
    "cobs2_compaction_request",
    "cobs2_shape",
    "cobs2_toolset",
    "tools_out",
    "relayed",
    "upstream",
    "served_gpu_tokens",
    "served_ram_tokens",
    "served_nvme_tokens",
    "served_remote_tokens",
    "load_s",
    "node_id",
    "session_source",
    "reusable_tokens_own",
    "cached_ttft_est_tokens",
    "diverged_at_own",
    "kv_isolated",
    "lens_break",
    "lens_cause",
    "lens_span_class",
    "lens_span_line",
    "lens_lost_tokens",
    "lens_cf_tokens",
    "lens_other_breaks",
    "cache_write_tokens",
    "cache_marks",
    "lens_repeat_results",
    "max_tokens_req",
    "reasoning_tokens",
    "reply_chars",
    "calls_n",
    "calls_bad",
    "tools_lint",
    "lens_version",
    "tools_in",
    "guard",
)
_SELECT = "SELECT " + ", ".join(_COLS) + " FROM request_history"


@dataclass(frozen=True)
class Row:
    id: str
    token_id: str | None
    token_name: str | None
    model_id: str
    model: str
    started_at: float
    finished_at: float
    duration_s: float
    queued_s: float | None
    ttft_s: float | None
    finish_reason: str | None
    prompt_tokens: int
    cached_tokens: int | None
    completion_tokens: int
    reusable_tokens: int | None
    cache_outcome: str | None
    cache_outcome_own: str | None
    divergence_cause: str | None
    divergence_cause_own: str | None
    dp_rank: int | None
    session_key: str | None
    parent_session_key: str | None
    cobs2_session: str | None
    cobs2_conv: str | None
    cobs2_node: str | None
    cobs2_parent_node: str | None
    cobs2_parent_req: str | None
    cobs2_edge: str | None
    cobs2_compaction_request: bool
    shape: dict[str, Any] | None
    toolset: str | None
    #: ``(family, name_hmac8)``, or ``(family, name_hmac8, args8)`` from Task 1.4 on.
    tools_out: tuple[tuple[str, ...], ...]
    relayed: bool
    upstream: str | None
    served_gpu_tokens: int | None
    served_ram_tokens: int | None
    served_nvme_tokens: int | None
    served_remote_tokens: int | None
    load_s: float | None
    node_id: str | None
    session_source: str | None = None
    reusable_tokens_own: int | None = None
    cached_ttft_est_tokens: int | None = None
    diverged_at_own: int | None = None
    #: The turn ran isolated (0060): True / False, None on rows written before it.
    kv_isolated: bool | None = None
    #: Prompt Lens (0063): layer 1 codes and counts, relay cache writes and
    #: marks, layer 3 counts (``calls_bad`` / ``tools_lint`` as stored JSON).
    #: None on rows written before it or when not computed.
    lens_break: str | None = None
    lens_cause: str | None = None
    lens_span_class: str | None = None
    lens_span_line: int | None = None
    lens_lost_tokens: int | None = None
    lens_cf_tokens: int | None = None
    lens_other_breaks: int | None = None
    cache_write_tokens: int | None = None
    cache_marks: int | None = None
    lens_repeat_results: int | None = None
    max_tokens_req: int | None = None
    reasoning_tokens: int | None = None
    reply_chars: int | None = None
    calls_n: int | None = None
    calls_bad: str | None = None
    tools_lint: str | None = None
    #: Prompt version (0065): ``h:<name>`` or 8 hex; None when not computed.
    lens_version: str | None = None
    #: As stored JSON: ``[[family, tokens, failed], ...]`` (0045) and the guard's
    #: ``{"m","b","f":[[rule, count, new]],"r"}`` (0055). Codes and counts only.
    tools_in: str | None = None
    guard: str | None = None


@dataclass(frozen=True)
class ListRow:
    id: str
    token_id: str | None
    token_name: str | None
    model: str
    started_at: float
    finished_at: float
    prompt_tokens: int
    cached_tokens: int | None
    completion_tokens: int
    cobs2_session: str
    parent_session_key: str | None
    cobs2_conv: str | None
    cobs2_edge: str | None
    relayed: bool
    has_tools: bool
    model_id: str = ""
    session_source: str | None = None
    #: Shape chars and (dup, boil, reason) dead weight; None when not recorded.
    chars: int | None = None
    dw: tuple[int, int, int] | None = None
    toolset: str | None = None
    called: tuple[str, ...] = ()
    reusable_tokens: int | None = None
    reusable_tokens_own: int | None = None
    cache_outcome: str | None = None
    cache_outcome_own: str | None = None
    cached_ttft_est_tokens: int | None = None
    cobs2_parent_req: str | None = None
    cobs2_parent_node: str | None = None
    diverged_at_own: int | None = None
    ttft_s: float | None = None


def _shape(v: Any) -> dict[str, Any] | None:
    if not isinstance(v, str):
        return None
    try:
        d = json.loads(v)
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def _tools_out(v: Any) -> tuple[tuple[str, ...], ...]:
    """The stored entries: ``[family, hash8]`` (older rows) or ``[family, hash8,
    args8]`` (Task 1.4); index ``[0]`` and ``[1]``, never unpack two."""
    if not isinstance(v, str):
        return ()
    try:
        d = json.loads(v)
    except ValueError:
        return ()
    if not isinstance(d, list):
        return ()
    return tuple(tuple(str(x) for x in e[:3]) for e in d if isinstance(e, list) and len(e) >= 2)


def _row(t: tuple[Any, ...]) -> Row:
    d = dict(zip(_COLS, t, strict=True))
    fin = float(d["finished_at"])
    dur = float(d["duration_s"] or 0.0)
    return Row(
        id=d["id"],
        token_id=d["token_id"],
        token_name=d["token_name"],
        model_id=d["model_id"],
        model=d["model"],
        started_at=fin - dur,
        finished_at=fin,
        duration_s=dur,
        queued_s=d["queued_s"],
        ttft_s=d["ttft_s"],
        finish_reason=d["finish_reason"],
        prompt_tokens=int(d["prompt_tokens"] or 0),
        cached_tokens=d["cached_tokens"],
        completion_tokens=int(d["completion_tokens"] or 0),
        reusable_tokens=d["reusable_tokens"],
        cache_outcome=d["cache_outcome"],
        cache_outcome_own=d["cache_outcome_own"],
        divergence_cause=d["cobs2_divergence_cause"],
        divergence_cause_own=d["cobs2_divergence_cause_own"],
        dp_rank=d["dp_rank"],
        session_key=d["session_key"],
        parent_session_key=d["parent_session_key"],
        cobs2_session=d["cobs2_session"],
        cobs2_conv=d["cobs2_conv"],
        cobs2_node=d["cobs2_node"],
        cobs2_parent_node=d["cobs2_parent_node"],
        cobs2_parent_req=d["cobs2_parent_req"],
        cobs2_edge=d["cobs2_edge"],
        cobs2_compaction_request=bool(d["cobs2_compaction_request"]),
        shape=_shape(d["cobs2_shape"]),
        toolset=d["cobs2_toolset"],
        tools_out=_tools_out(d["tools_out"]),
        relayed=bool(d["relayed"]),
        upstream=d["upstream"],
        served_gpu_tokens=d["served_gpu_tokens"],
        served_ram_tokens=d["served_ram_tokens"],
        served_nvme_tokens=d["served_nvme_tokens"],
        served_remote_tokens=d["served_remote_tokens"],
        load_s=d["load_s"],
        node_id=d["node_id"],
        session_source=d["session_source"],
        reusable_tokens_own=d["reusable_tokens_own"],
        cached_ttft_est_tokens=d["cached_ttft_est_tokens"],
        diverged_at_own=d["diverged_at_own"],
        kv_isolated=None if d["kv_isolated"] is None else bool(d["kv_isolated"]),
        lens_break=d["lens_break"],
        lens_cause=d["lens_cause"],
        lens_span_class=d["lens_span_class"],
        lens_span_line=d["lens_span_line"],
        lens_lost_tokens=d["lens_lost_tokens"],
        lens_cf_tokens=d["lens_cf_tokens"],
        lens_other_breaks=d["lens_other_breaks"],
        cache_write_tokens=d["cache_write_tokens"],
        cache_marks=d["cache_marks"],
        lens_repeat_results=d["lens_repeat_results"],
        max_tokens_req=d["max_tokens_req"],
        reasoning_tokens=d["reasoning_tokens"],
        reply_chars=d["reply_chars"],
        calls_n=d["calls_n"],
        calls_bad=d["calls_bad"],
        tools_lint=d["tools_lint"],
        lens_version=d["lens_version"],
        tools_in=d["tools_in"],
        guard=d["guard"],
    )


def lookup_ref(db: sqlite3.Connection, sess_ref: str) -> tuple[str, str] | None:
    """(token_id, cobs2_session) of a session lookup ref, from one indexed row."""
    r = db.execute(
        "SELECT token_id, cobs2_session FROM request_history WHERE cobs2_sess_ref = ?"
        " ORDER BY finished_at DESC LIMIT 1",
        (sess_ref,),
    ).fetchone()
    if r is None or r[1] is None:
        return None
    return (r[0], r[1])


def _has_rows(db: sqlite3.Connection, token_id: str | None, session: str) -> bool:
    return (
        db.execute(
            "SELECT 1 FROM request_history WHERE token_id IS ? AND cobs2_session = ? LIMIT 1",
            (token_id, session),
        ).fetchone()
        is not None
    )


def _parent_of(db: sqlite3.Connection, token_id: str | None, session: str) -> str | None:
    r = db.execute(
        "SELECT parent_session_key FROM request_history"
        " WHERE token_id IS ? AND cobs2_session = ? AND parent_session_key IS NOT NULL"
        " ORDER BY finished_at DESC LIMIT 1",
        (token_id, session),
    ).fetchone()
    return r[0] if r is not None else None


def root_of(db: sqlite3.Connection, token_id: str | None, cobs2_session: str) -> str:
    """The topmost ancestor (with rows, same token) of a session, at most
    ``ROOT_DEPTH`` steps up; cycle-safe."""
    cur = cobs2_session
    seen = {cur}
    for _ in range(ROOT_DEPTH):
        parent = _parent_of(db, token_id, cur)
        if parent is None or parent in seen or not _has_rows(db, token_id, parent):
            break
        seen.add(parent)
        cur = parent
    return cur


#: No `cobs2_session IS NOT NULL` term (filtered in Python): with it, and no
#: ANALYZE statistics (production never runs ANALYZE), the planner picks the
#: (token, session) index and scans every session row of the key.
_DESC_SQL = (
    "SELECT DISTINCT cobs2_session FROM request_history"
    " INDEXED BY idx_request_history_parent_session"
    " WHERE token_id IS ? AND parent_session_key = ?"
)


def descendants(db: sqlite3.Connection, token_id: str | None, root: str) -> list[str]:
    """The root and its child sessions, breadth-first, at most ``ROOT_DEPTH`` levels."""
    out = [root]
    seen = {root}
    level = [root]
    for _ in range(ROOT_DEPTH):
        nxt: list[str] = []
        for s in level:
            for (child,) in db.execute(_DESC_SQL, (token_id, s)):
                if child is not None and child not in seen:
                    seen.add(child)
                    out.append(child)
                    nxt.append(child)
        if not nxt:
            break
        level = nxt
    return out


def _ROWS_SQL_FOR(n: int) -> str:  # noqa: N802 -- named in the plan
    marks = ", ".join("?" for _ in range(max(1, n)))
    return (
        _SELECT + f" WHERE token_id IS ? AND cobs2_session IN ({marks})"
        " ORDER BY finished_at DESC LIMIT ?"
    )


def rows_for(
    db: sqlite3.Connection, token_id: str | None, sessions: Sequence[str], limit: int
) -> tuple[list[Row], bool]:
    """The newest ``limit`` rows of the sessions, oldest first, and whether
    more exist."""
    if not sessions:
        return [], False
    # One indexed query per session (an `IN (...)` over several falls back to
    # the token index without statistics), then the newest `limit` overall.
    got: list[tuple[Any, ...]] = []
    for sess in dict.fromkeys(sessions):
        got += db.execute(_ROWS_SQL_FOR(1), (token_id, sess, int(limit) + 1)).fetchall()
    fin = _COLS.index("finished_at")
    got.sort(key=lambda t: t[fin], reverse=True)
    truncated = len(got) > limit
    rows = [_row(t) for t in got[:limit]]
    rows.reverse()
    return rows, truncated


def anchor_row(db: sqlite3.Connection, token_id: str | None, sess: str, prefix: str) -> Row | None:
    """The row of a session whose id starts with ``prefix`` (a segment's name), by primary key."""
    t = db.execute(
        _SELECT
        + " WHERE id >= ? AND id < ? AND cobs2_session = ? AND token_id IS ? ORDER BY id LIMIT 1",
        (prefix, prefix + "g", sess, token_id),
    ).fetchone()
    return _row(t) if t is not None else None


def rows_around(
    db: sqlite3.Connection, token_id: str | None, sess: str, at: float, side: int
) -> tuple[list[Row], bool, bool]:
    """Up to ``side`` rows of a session finishing at or before ``at`` and up to ``side`` after,
    oldest first; whether more exist before, and after."""
    q = (
        _SELECT
        + " WHERE token_id IS ? AND cobs2_session = ? AND finished_at {op} ? ORDER BY finished_at {d} LIMIT ?"
    )
    before = db.execute(q.format(op="<=", d="DESC"), (token_id, sess, at, side + 1)).fetchall()
    after = db.execute(q.format(op=">", d="ASC"), (token_id, sess, at, side + 1)).fetchall()
    more_before, more_after = len(before) > side, len(after) > side
    rows = [_row(t) for t in reversed(before[:side])] + [_row(t) for t in after[:side]]
    return rows, more_before, more_after


def request_row(db: sqlite3.Connection, request_id: str) -> Row | None:
    t = db.execute(_SELECT + " WHERE id = ?", (request_id,)).fetchone()
    return _row(t) if t is not None else None


def owned_ids(db: sqlite3.Connection, token_id: str, ids: Collection[str]) -> set[str]:
    """Which of ``ids`` are requests of ``token_id``: a primary-key lookup each."""
    want = sorted(set(ids))
    out: set[str] = set()
    for i in range(0, len(want), 500):
        part = want[i : i + 500]
        q = f"SELECT id FROM request_history WHERE token_id = ? AND id IN ({','.join('?' * len(part))})"
        out.update(r[0] for r in db.execute(q, (token_id, *part)))
    return out


def toolsets(
    db: sqlite3.Connection, digests: Collection[str]
) -> dict[str, list[tuple[str, str, int]]]:
    out: dict[str, list[tuple[str, str, int]]] = {}
    for d in set(digests):
        r = db.execute("SELECT tools FROM cobs2_toolsets WHERE digest = ?", (d,)).fetchone()
        if r is None:
            continue
        try:
            entries = json.loads(r[0])
        except ValueError:
            continue
        out[d] = [
            (str(e[0]), str(e[1]), int(e[2]))
            for e in entries
            if isinstance(e, list) and len(e) == 3 and isinstance(e[2], int)
        ]
    return out


def tool_names(
    db: sqlite3.Connection, token_id: str | None, hmacs: Collection[str]
) -> dict[str, str]:
    """Plain names for one token's tools only (owner decision: names, never
    descriptions); a primary-key lookup per hash."""
    if token_id is None:
        return {}
    out: dict[str, str] = {}
    for h in set(hmacs):
        r = db.execute(
            "SELECT name FROM session_tools WHERE token_id = ? AND tool_hmac = ?", (token_id, h)
        ).fetchone()
        if r is not None:
            out[h] = r[0]
    return out


def recent_toolset_calls(
    db: sqlite3.Connection, token_id: str | None, toolset: str, limit: int = 20
) -> dict[str, int]:
    """Per tool hash, the calls across the key's last ``limit`` sessions that carried
    ``toolset`` (analysis spec 7.2 item 1). Scoped by ``token_id`` (A-1) and bounded
    by ``limit`` sessions; counts only, no names and no text."""
    if token_id is None or limit <= 0:
        return {}
    rows = db.execute(
        "SELECT json_extract(e.value, '$[1]'), COUNT(*)"
        " FROM request_history r,"
        " json_each(CASE WHEN json_valid(r.tools_out) AND json_type(r.tools_out) = 'array'"
        " THEN r.tools_out ELSE '[]' END) e"
        " WHERE r.token_id = ? AND json_type(e.value) = 'array' AND r.cobs2_session IN ("
        "  SELECT cobs2_session FROM request_history"
        "  WHERE token_id = ? AND cobs2_toolset = ? AND cobs2_session IS NOT NULL"
        "  GROUP BY cobs2_session ORDER BY MAX(finished_at) DESC LIMIT ?)"
        " GROUP BY 1",
        (token_id, token_id, toolset, int(limit)),
    ).fetchall()
    return {str(h): int(n) for h, n in rows if isinstance(h, str) and h}


_LIST_COLS = (
    "id, token_id, token_name, model, finished_at, duration_s, prompt_tokens, cached_tokens,"
    " completion_tokens, cobs2_session, parent_session_key, cobs2_conv, cobs2_edge, relayed,"
    " (tools_out IS NOT NULL AND tools_out <> '[]'),"
    " model_id, session_source,"
    " CASE WHEN json_valid(cobs2_shape) THEN json_extract(cobs2_shape, '$.chars') END,"
    " CASE WHEN json_valid(cobs2_shape) THEN json_extract(cobs2_shape, '$.dw.dup') END,"
    " CASE WHEN json_valid(cobs2_shape) THEN json_extract(cobs2_shape, '$.dw.boil') END,"
    " CASE WHEN json_valid(cobs2_shape) THEN json_extract(cobs2_shape, '$.dw.reason') END,"
    " cobs2_toolset, CASE WHEN tools_out <> '[]' THEN tools_out END,"
    " reusable_tokens, reusable_tokens_own, cache_outcome, cache_outcome_own, cached_ttft_est_tokens,"
    " cobs2_parent_req, cobs2_parent_node, diverged_at_own, ttft_s"
)


def _int_or_none(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def list_rows(
    db: sqlite3.Connection, *, since: float, token_id: str | None, limit: int = MAX_LIST_ROWS
) -> tuple[list[ListRow], bool]:
    """A light projection of the window's session rows, newest first; ``token_id``
    None = every key (admin)."""
    sql = f"SELECT {_LIST_COLS} FROM request_history WHERE finished_at >= ? AND cobs2_session IS NOT NULL"
    args: list[Any] = [since]
    if token_id is not None:
        sql += " AND token_id = ?"
        args.append(token_id)
    got = db.execute(sql + " ORDER BY finished_at DESC LIMIT ?", (*args, int(limit) + 1)).fetchall()
    truncated = len(got) > limit
    out = [
        ListRow(
            id=t[0],
            token_id=t[1],
            token_name=t[2],
            model=t[3],
            started_at=float(t[4]) - float(t[5] or 0.0),
            finished_at=float(t[4]),
            prompt_tokens=int(t[6] or 0),
            cached_tokens=t[7],
            completion_tokens=int(t[8] or 0),
            cobs2_session=t[9],
            parent_session_key=t[10],
            cobs2_conv=t[11],
            cobs2_edge=t[12],
            relayed=bool(t[13]),
            has_tools=bool(t[14]),
            model_id=t[15] or "",
            session_source=t[16],
            chars=_int_or_none(t[17]),
            dw=(
                (t[18], t[19], t[20])
                if all(_int_or_none(x) is not None for x in (t[18], t[19], t[20]))
                else None
            ),
            toolset=t[21],
            called=tuple(e[1] for e in _tools_out(t[22])),
            reusable_tokens=t[23],
            reusable_tokens_own=t[24],
            cache_outcome=t[25],
            cache_outcome_own=t[26],
            cached_ttft_est_tokens=t[27],
            cobs2_parent_req=t[28],
            cobs2_parent_node=t[29],
            diverged_at_own=t[30],
            ttft_s=t[31],
        )
        for t in got[:limit]
    ]
    return out, truncated


def oldest_finished_at(db: sqlite3.Connection) -> float | None:
    r = db.execute("SELECT MIN(finished_at) FROM request_history").fetchone()
    return float(r[0]) if r is not None and r[0] is not None else None


LLAMACPP_DEFAULT_SLOTS = 4


@dataclass(frozen=True)
class ModelEngine:
    backend: str
    dp_size: int
    dp_affinity: bool
    #: llama.cpp server slots (R7); 4 for other backends.
    slots: int
    #: Whether the backend reports cached prompt tokens (MLX does not).
    cache_known: bool


def llamacpp_slots(backend: str, max_num_seqs: int | None, extra_args: Sequence[str]) -> int:
    if backend != "llamacpp":
        return LLAMACPP_DEFAULT_SLOTS
    if isinstance(max_num_seqs, int) and max_num_seqs > 0:
        return max_num_seqs
    args = list(extra_args)
    for i, a in enumerate(args):
        val: str | None = None
        if a in ("--parallel", "-np") and i + 1 < len(args):
            val = args[i + 1]
        elif a.startswith("--parallel="):
            val = a.split("=", 1)[1]
        if val is not None:
            try:
                n = int(val)
            except ValueError:
                continue
            if n > 0:
                return n
    return LLAMACPP_DEFAULT_SLOTS


def model_engines(db: sqlite3.Connection, model_ids: Collection[str]) -> dict[str, ModelEngine]:
    """Engine facts per model id (absent: no models row any more)."""
    from app.runtime.backends import registry

    out: dict[str, ModelEngine] = {}
    for mid in set(model_ids):
        r = db.execute(
            "SELECT backend, data_parallel_size, dp_affinity_enabled, max_num_seqs, extra_args"
            " FROM models WHERE id = ?",
            (mid,),
        ).fetchone()
        if r is None:
            continue
        backend = r[0] or registry.DEFAULT_BACKEND
        try:
            extra = json.loads(r[4]) if r[4] else []
        except ValueError:
            extra = []
        if not isinstance(extra, list):
            extra = []
        try:
            cache_known = registry.get(backend).capabilities.cached_tokens_reporting != "none"
        except Exception:  # noqa: BLE001 -- an unknown backend: say unknown
            cache_known = False
        out[mid] = ModelEngine(
            backend=backend,
            dp_size=int(r[1] or 1),
            dp_affinity=bool(r[2]),
            slots=llamacpp_slots(backend, r[3], [str(x) for x in extra]),
            cache_known=cache_known,
        )
    return out
