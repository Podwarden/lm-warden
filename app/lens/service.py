"""Prompt Lens request findings: the scope-checked read behind the request view
(analysis Task 1.8, spec 16.1).

``request_findings`` finds the ``request_history`` row a Prompt Lens id stands
for, loads that request's session (bounded, the row's own key only), runs
``analyse_session`` with what is cheap to know, and keeps the findings that
name the request's turn. A key holder (``token_id`` set) gets ``own=True``
analysis over its own rows only (A-1): the own lens, no fleet figure, no
admin-only cause. Anything not the caller's -- another key's id, an unknown or
evicted one -- is None, so the routes answer one identical 404.

What the answer holds (A-10): codes, counts, sizes, positions, keyed tool
hashes, turn numbers; ``part`` places a finding on the request view; ``names``
maps the tool hashes the findings carry to the plain names the caller already
sees elsewhere (a key holder: its own ``session_tools`` only). Never text.

Context passed to the analysis: the toolsets, the key's recent calls per tool
(``store.recent_toolset_calls``, scoped by token), the learned cold prefill
rate (the key's own for a key holder), the model's ``max_model_len`` and the
guard log's roles (``guard_where_of``, the row's key only, bounded), and per
changed system prompt line the key's own earlier sessions that changed the
same line (``cross_sessions``: the row's key only, bounded, spec 4.5). Not
known here, so None: the route's modelled tool limit (C-09 stays off).
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import time
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field, replace
from typing import Any

from app.lens import prices, ranking, scoring, trends, wontfix
from app.lens.analyse import (
    AnalyseContext,
    SessionAnalysis,
    analyse_session,
    findings_for_request,
)
from app.lens.catalogue import RULES, Finding, Saving
from app.lens.response import guard_where_of
from app.lens.store import LensStore, SessionScoreRow, encode_cursor
from app.lens.turns import LensTurn
from app.lens.versions import lineage_ids, version_label
from app.lens.wontfix import WontfixRepo
from app.sessions import store
from app.sessions.harness import session_harness
from app.sessions.model import MAX_TURNS, own_lens, segment
from app.sessions.refs import is_fallback, parse_ref, sess_ref, session_id
from app.sessions.service import _cold_rates, _session_rows
from app.stats.routes_forest import ForestViewer, _connect

logger = logging.getLogger(__name__)

#: Ids the routes accept: a Prompt Lens id (16 hex) or a stored request id.
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
#: Layer 1 causes only the admin may see (app/proxy/prompt_lens_scope.py).
ADMIN_ONLY_CAUSES = frozenset({"misrouted"})
#: Guard log rows read for one session, at most.
MAX_GUARD_ROWS = 10_000
#: Toolsets whose recent calls are read, at most (the session's most used).
MAX_TOOLSETS = 8
#: Seconds around the session that the guard log is read for.
_GUARD_SLACK_S = 60.0
#: E-01 across sessions (spec 4.5): how far back the key's earlier sessions
#: are looked for, the rows read at most (newest first, on the
#: ``(token_id, finished_at)`` index), and the lines counted at most.
CROSS_WINDOW_S = 24 * 3600.0
MAX_CROSS_ROWS = 20_000
MAX_CROSS_LINES = 32
_SYS_CAUSES = ("sys_noise", "sys_changed")

_CALL_RULES = frozenset({"R-04", "R-05", "R-06"})
_TOOLS_RULES = frozenset({"C-01", "C-08", "C-09"})
#: E rules (and C-06) that sit on the part reuse stopped at; the default when
#: the turn has no stored break.
_BREAK_RULES: Mapping[str, str | None] = {
    "E-01": "sys",
    "C-06": "sys",
    "E-02": "tools",
    "E-03": "tools",
    "E-04": None,
    "E-05": None,
    "E-06": None,
    "E-07": "sys",
    "E-08": None,
    "E-09": None,
}
_PART = re.compile(r"(sys|tools|node:\d+)")


def _valid_part(p: str | None) -> str | None:
    return p if isinstance(p, str) and _PART.fullmatch(p) else None


def _call_part(f: Finding, t: LensTurn) -> str | None:
    """``call:<k>``: the first call of the finding's tool in the turn's reply.
    For R-04, the first call of that tool judged bad, at the index its stored
    verdict names (I4: with one tool called twice in parallel, the second
    call bad is ``call:1``). A verdict stored before the index (k None) falls
    back to the tool's first call in ``tools_out``."""
    if f.tool is None:
        return None
    if f.rule == "R-04":
        mine = [k for k, h, _code in t.calls_bad or () if h == f.tool]
        if not mine:
            return None
        known = [k for k in mine if k is not None]
        if known:
            return f"call:{min(known)}"
    for k, e in enumerate(t.tools_out):
        if len(e) >= 2 and e[1] == f.tool:
            return f"call:{k}"
    return None


def part_of(f: Finding, t: LensTurn, where: Mapping[str, int] | None) -> str | None:
    """Where on the request view ``f`` sits for turn ``t``: ``sys``, ``tools``,
    ``node:<i>``, ``call:<k>``, or None (the strip only)."""
    if f.rule in _BREAK_RULES:
        part = _valid_part(t.brk)
        if part is None and "message" in f.counts:
            part = f"node:{int(f.counts['message'])}"
        return part or _BREAK_RULES[f.rule]
    if f.rule in _TOOLS_RULES:
        return "tools"
    if f.rule == "C-03":
        return "sys"
    if f.rule in _CALL_RULES:
        return _call_part(f, t)
    if f.rule == "R-09":
        return "sys" if (where or {}).get("system", 0) > 0 else None
    return None


# -- reads (worker thread) ------------------------------------------------------------


def _kept_request(db: sqlite3.Connection, req_id: str) -> tuple[str, str] | None:
    """(token id, stored request id) of a kept capture."""
    try:
        r = db.execute(
            "SELECT token_id, request_id FROM prompt_lens_kept WHERE req_id = ?", (req_id,)
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return (str(r[0]), str(r[1])) if r is not None and r[1] else None


def _find_row(
    db: sqlite3.Connection,
    req_id: str,
    token_id: str | None,
    held: tuple[str, str | None] | None,
) -> store.Row | None:
    """The row ``req_id`` names -- a request in memory, a kept capture, or a
    stored request id -- if the caller may see it."""
    candidates: list[tuple[str | None, str]] = []
    if held is not None and held[1]:
        candidates.append((held[0], held[1]))
    kept = _kept_request(db, req_id)
    if kept is not None:
        candidates.append(kept)
    candidates.append((None, req_id))
    for owner, rid in candidates:
        if token_id is not None and owner is not None and owner != token_id:
            return None
        row = store.request_row(db, rid)
        if row is not None:
            if token_id is not None and row.token_id != token_id:
                return None
            return row
    return None


def _rows_of(db: sqlite3.Connection, secret: str, row: store.Row) -> list[store.Row]:
    """The request's session as the sessions page reads it (bounded by MAX_TURNS)."""
    if not row.cobs2_session:
        return [row]
    got = _session_rows(db, secret, row.token_id, row.cobs2_session, anchor=row.id, seg=None)
    rows = got[0] if got is not None and got[0] else [row]
    if not any(r.id == row.id for r in rows):
        rows = store.rows_around(
            db, row.token_id, row.cobs2_session, row.finished_at, MAX_TURNS // 2
        )[0]
    return rows or [row]


def _for_key_holder(
    db: sqlite3.Connection, token_id: str, rows: Sequence[store.Row], *, strip_admin: bool = True
) -> list[store.Row]:
    """The own lens (``own_lens``: no fleet figure, no lineage to another key's
    request) and no admin-only cause."""
    owned = {r.id for r in rows}
    ext = {r.cobs2_parent_req for r in rows if r.cobs2_parent_req} - owned
    if ext:
        owned |= store.owned_ids(db, token_id, ext)
    out = []
    for r in rows:
        r = own_lens(r, owned)
        if strip_admin and r.lens_cause in ADMIN_ONLY_CAUSES:
            r = replace(r, lens_cause=None)
        out.append(r)
    return out


def _recent_calls(
    db: sqlite3.Connection, token_id: str | None, rows: Sequence[store.Row]
) -> dict[str, int]:
    use: dict[str, int] = {}
    for r in rows:
        if r.toolset:
            use[r.toolset] = use.get(r.toolset, 0) + 1
    out: dict[str, int] = {}
    for ts in sorted(use, key=lambda d: -use[d])[:MAX_TOOLSETS]:
        for h, n in store.recent_toolset_calls(db, token_id, ts).items():
            out[h] = max(out.get(h, 0), n)
    return out


def _max_model_len(db: sqlite3.Connection, model_id: str) -> int | None:
    r = db.execute("SELECT max_model_len FROM models WHERE id = ?", (model_id,)).fetchone()
    v = r[0] if r is not None else None
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None


def _guard_where(
    db: sqlite3.Connection, token_id: str | None, rows: Sequence[store.Row]
) -> dict[str, dict[str, int]]:
    """Request id -> guard ``where`` code -> sanitized values, from the guard
    log of the rows' own key, over the session's time span (bounded)."""
    if token_id is None or not rows:
        return {}
    ids = {r.id for r in rows}
    lo = min(r.started_at for r in rows) - _GUARD_SLACK_S
    hi = max(r.finished_at for r in rows) + _GUARD_SLACK_S
    got = db.execute(
        "SELECT request_id, role, kind FROM guard_events"
        " WHERE token_id = ? AND ts >= ? AND ts <= ? AND action = 'sanitized' LIMIT ?",
        (token_id, lo, hi, MAX_GUARD_ROWS),
    ).fetchall()
    out: dict[str, dict[str, int]] = {}
    for rid, role, kind in got:
        code = guard_where_of(role, kind)
        if rid in ids and code is not None:
            w = out.setdefault(rid, {})
            w[code] = w.get(code, 0) + 1
    return out


def cross_sessions(
    db: sqlite3.Connection, row: store.Row, rows: Sequence[store.Row]
) -> dict[int, int]:
    """Span line -> the DISTINCT earlier sessions of the row's own key (never
    this session) whose request changed the same system prompt line
    (``sys_noise`` / ``sys_changed``), for the lines this session changed.
    Bounded: the key's newest ``MAX_CROSS_ROWS`` rows that finished in the
    ``CROSS_WINDOW_S`` before this request started. Counts only (A-1: the
    row's key, which for a key holder is its own)."""
    token = row.token_id
    if not token:
        return {}
    lines = sorted(
        {
            r.lens_span_line
            for r in rows
            if r.lens_cause in _SYS_CAUSES
            and isinstance(r.lens_span_line, int)
            and not isinstance(r.lens_span_line, bool)
        }
    )[:MAX_CROSS_LINES]
    if not lines:
        return {}
    marks = ", ".join("?" * len(lines))
    got = db.execute(
        "SELECT lens_span_line, COUNT(DISTINCT cobs2_session) FROM ("
        "  SELECT lens_cause, lens_span_line, cobs2_session FROM request_history"
        "  WHERE token_id = ? AND finished_at >= ? AND finished_at < ?"
        "  ORDER BY finished_at DESC LIMIT ?)"
        f" WHERE lens_cause IN (?, ?) AND lens_span_line IN ({marks})"
        " AND cobs2_session IS NOT NULL AND cobs2_session <> ?"
        " GROUP BY lens_span_line",
        (
            token,
            row.started_at - CROSS_WINDOW_S,
            row.started_at,
            MAX_CROSS_ROWS,
            *_SYS_CAUSES,
            *lines,
            row.cobs2_session or "",
        ),
    ).fetchall()
    return {int(ln): int(n) for ln, n in got if isinstance(ln, int) and isinstance(n, int)}


def _guard_json(
    f: Finding, turns: Sequence[LensTurn], where: Mapping[str, Mapping[str, int]]
) -> dict[str, list[str]]:
    """R-09's rule ids and roles, from the guard data only (never from text)."""
    mine = [t for t in turns if t.n in f.turns]
    rules = {r for t in mine for r, c, n in t.guard or () if c > 0 or n > 0}
    codes = {c for t in mine for c, v in (where.get(t.req_id) or {}).items() if v > 0}
    return {"rules": sorted(rules), "where": sorted(codes)}


def analysis_context(
    db: sqlite3.Connection,
    state: Any,
    row: store.Row,
    rows: Sequence[store.Row],
    *,
    viewer_token: str | None,
) -> tuple[AnalyseContext, dict[str, dict[str, int]]]:
    """What ``analyse_session`` needs beyond the rows (see the module docstring)
    and the guard roles per request, for ``row``'s key."""
    where = _guard_where(db, row.token_id, rows)
    rate = _cold_rates(state, {row.model_id}, ForestViewer(viewer_token)).get(row.model_id)
    ctx = AnalyseContext(
        toolsets=store.toolsets(db, {r.toolset for r in rows if r.toolset}),
        recent_calls=_recent_calls(db, row.token_id, rows),
        prefill_rate=rate,
        max_model_len=_max_model_len(db, row.model_id),
        route_tool_limit=None,  # no modelled limit per route yet: C-09 stays off
        cross_sessions=cross_sessions(db, row, rows),
        guard_where=where,
    )
    return ctx, where


def _work(
    db_path: Any,
    state: Any,
    secret: str,
    req_id: str,
    token_id: str | None,
    held: tuple[str, str | None] | None,
) -> dict[str, Any] | None:
    with closing(_connect(db_path)) as db:
        row = _find_row(db, req_id, token_id, held)
        if row is None:
            return None
        rows = _rows_of(db, secret, row)
        if token_id is not None:
            rows = _for_key_holder(db, token_id, rows)
        ctx, where = analysis_context(db, state, row, rows, viewer_token=token_id)
        try:
            a = analyse_session(rows, own=token_id is not None, ctx=ctx)
        except Exception as exc:  # noqa: BLE001 -- the view shows none rather than fail
            logger.warning("lens: findings failed: %s", type(exc).__name__)
            return {"turn": None, "findings": [], "names": {}}
        turn = next((t for t in a.turns if t.req_id == row.id), None)
        found = findings_for_request(a, row.id)
        out = []
        for f in found:
            d = f.to_json()
            if token_id is not None and d["cause"] in ADMIN_ONLY_CAUSES:
                d["cause"] = None
            d["part"] = part_of(f, turn, where.get(row.id)) if turn is not None else None
            d["guard"] = _guard_json(f, a.turns, where) if f.rule == "R-09" else None
            out.append(d)
        names = store.tool_names(db, row.token_id, {f.tool for f in found if f.tool})
    return {"turn": turn.n if turn is not None else None, "findings": out, "names": names}


async def request_findings_body(
    state: Any, *, token_id: str | None, req_id: str
) -> dict[str, Any] | None:
    """The findings route's answer: ``{"turn", "findings", "names"}``, or None
    (404) when ``req_id`` is not the caller's to read. ``token_id=None`` is
    the admin (any key)."""
    if not _ID.fullmatch(req_id):
        return None
    sections = getattr(state, "prompt_lens_sections", None)
    held = sections.request_of(req_id) if sections is not None else None
    return await asyncio.to_thread(
        _work, state.settings.db_path, state, state.settings.cookie_secret, req_id, token_id, held
    )


async def request_findings(
    state: Any, *, token_id: str | None, req_id: str
) -> list[dict[str, Any]] | None:
    """The request's findings (``Finding.to_json()`` plus ``part``), or None (404)."""
    body = await request_findings_body(state, token_id=token_id, req_id=req_id)
    return None if body is None else list(body["findings"])


# -- scores and the scorecard (analysis Tasks 2.3, 2.4) ---------------------------------

#: Sessions listed at most per page.
MAX_LIST = 200
#: How far back the session list reads when the caller gives no window.
DEFAULT_SINCE_S = 30 * 24 * 3600.0
_INPUT_LAYERS = frozenset({"E", "C"})


@dataclass(frozen=True)
class ScoredSession:
    """One session scored: the analysis, the scores, the priced findings and the
    labels the list and the scorecard show. Codes and numbers only (A-10)."""

    key_id: str
    sess_ref: str
    rows: Sequence[store.Row]
    analysis: SessionAnalysis
    scores: dict[str, scoring.Score]
    terms: Mapping[str, Any]
    findings: list[Finding]
    top: list[Finding]
    notice: Finding | None
    card: Any
    where: dict[str, dict[str, int]]
    model: str
    route: str
    harness: str
    started: float
    last_at: float
    #: The session's newest prompt version (``h:<name>`` or 8 hex), None when unknown.
    version: str | None = None
    #: The won't-fix marks applied to this scoring: rule -> ``key`` or ``admin``.
    marks: Mapping[str, str] = field(default_factory=dict)
    #: The scores with the marks counted, only when ``marks`` is not empty (spec 11.4).
    plain: dict[str, scoring.Score] | None = None

    def to_row(self) -> SessionScoreRow:
        fired: dict[str, list[str]] = {}
        for f in self.findings:
            ids = fired.setdefault(RULES[f.rule].layer, [])
            if f.rule not in ids:
                ids.append(f.rule)
        return SessionScoreRow.of(
            key_id=self.key_id,
            sess_ref=self.sess_ref,
            started=self.started,
            last_at=self.last_at,
            model=self.model,
            route=self.route,
            harness=self.harness,
            turns=len(self.rows),
            scores=self.scores,
            terms=self.terms,
            coverage={str(k): v for k, v in self.analysis.coverage.items()},
            fired=fired,
            score_version=scoring.SCORE_VERSION,
            version=self.version,
        )


def _decode_rate(rows: Sequence[store.Row]) -> float | None:
    """The session's decode rate: completion tokens per second after the first token."""
    toks = secs = 0.0
    for r in rows:
        if (
            r.ttft_s is not None
            and r.duration_s
            and r.duration_s > r.ttft_s
            and r.completion_tokens > 0
        ):
            toks += r.completion_tokens
            secs += r.duration_s - r.ttft_s
    return toks / secs if secs > 0 else None


def price_findings(
    findings: Sequence[Finding],
    card: Any,
    *,
    prefill_rate: float | None,
    decode_rate: float | None,
    cached_share: float,
) -> None:
    """Fill each finding's seconds and money (spec 12.2), in place: input-side
    savings (E, C) as prefill seconds at the learned rate, output-side (R) as
    decode seconds at the session's rate; money only with a card."""
    for f in findings:
        s = f.saving
        if s.tokens <= 0:
            continue
        layer = RULES[f.rule].layer
        out_side = layer == "R"
        prefill_s, decode_s = s.prefill_s, s.decode_s
        if out_side and decode_s is None and decode_rate:
            decode_s = s.tokens / decode_rate
        if not out_side and prefill_s is None and prefill_rate:
            prefill_s = s.tokens / prefill_rate
        f.saving = prices.price_saving(
            Saving(s.tokens, prefill_s, decode_s, s.money),
            card,
            cached_share=cached_share,
            kind="output" if out_side else "waste" if layer == "C" else "reuse",
        )


def score_rows(
    db: sqlite3.Connection,
    state: Any,
    rows: Sequence[store.Row],
    *,
    sid: str,
    strip_admin: bool,
    marks: Mapping[str, str] | None = None,
) -> ScoredSession | None:
    """Analyse and score one session's rows on the key's OWN lens (A-1: a key
    holder's scores never depend on another key's traffic; the admin reads the
    same figures). ``sid`` is the sessions page's session id and seeds the
    range. ``marks`` are the won't-fix rules (rule -> who marked it): they are
    left out of the scores, their findings are muted and never ranked, and the
    scores with them counted are kept beside (``plain``). None for no rows."""
    if not rows:
        return None
    last = max(rows, key=lambda r: (r.finished_at, r.id))
    key = last.token_id or ""
    own = _for_key_holder(db, key, rows, strip_admin=strip_admin) if key else list(rows)
    ctx, where = analysis_context(db, state, last, own, viewer_token=key or None)
    a = analyse_session(own, own=True, ctx=ctx)
    terms: dict[str, Any] = {str(k): t.to_json() for k, t in scoring.terms_of(a).items()}
    exclude = frozenset(marks or ())
    scores = scoring.score_session(a, seed=sid, exclude=exclude)
    plain = scoring.score_session(a, seed=sid) if exclude else None
    for f in a.findings:
        f.muted = f.rule in exclude
    card = prices.card_for(
        db,
        relayed=bool(last.relayed),
        upstream=last.upstream,
        model=last.model,
        model_id=last.model_id,
    )
    e_turns = scoring.e_turns(a)
    prompt = sum(max(t.prompt_tokens, 0) for t in e_turns)
    ideal = sum(t.cf_tokens or 0 for t in e_turns)
    cached_share = min(1.0, ideal / prompt) if prompt > 0 else 1.0
    price_findings(
        a.findings,
        card,
        prefill_rate=ctx.prefill_rate,
        decode_rate=_decode_rate(rows),
        cached_share=cached_share,
    )
    return ScoredSession(
        key_id=key,
        sess_ref=sid,
        rows=own,
        analysis=a,
        scores=scores,
        terms=terms,
        findings=a.findings,
        top=ranking.top_fixes(a.findings, card),
        notice=ranking.pinned_notice(a.findings),
        card=card,
        where=where,
        model=last.model,
        route="relay" if last.relayed else "local",
        harness=session_harness(r.session_source for r in rows),
        started=min(r.started_at for r in rows),
        last_at=max(r.finished_at for r in rows),
        version=newest_version(rows),
        marks=dict(marks or {}),
        plain=plain,
    )


def newest_version(rows: Sequence[store.Row]) -> str | None:
    """The prompt version of the session's newest request that has one."""
    for r in sorted(rows, key=lambda r: (r.finished_at, r.id), reverse=True):
        if r.lens_version:
            return r.lens_version
    return None


def session_units(
    db: sqlite3.Connection, secret: str, token_id: str | None, sess: str
) -> list[tuple[str, list[store.Row]]]:
    """The session units of a cobs2 session as the sessions page names them:
    one ``(session id, rows)`` for a harness session's whole tree, one per
    idle-separated segment for a fallback session."""
    if is_fallback(sess):
        ref = sess_ref(secret, token_id, sess)
        rows = store.rows_for(db, token_id, [sess], MAX_TURNS)[0]
        if ref is None or not rows:
            return []
        return [(session_id(ref, part[0].id), part[-MAX_TURNS:]) for part in segment(rows)]
    root = store.root_of(db, token_id, sess)
    ref = sess_ref(secret, token_id, root)
    if ref is None:
        return []
    rows = store.rows_for(db, token_id, store.descendants(db, token_id, root), MAX_TURNS)[0]
    return [(session_id(ref), rows)] if rows else []


def _resolve(
    db: sqlite3.Connection, secret: str, ref: str, token_id: str | None
) -> tuple[str, list[store.Row]] | None:
    """The rows of the session ``ref`` (the sessions page's ``s_...`` id), if the
    caller may read them: a key holder only its own (one answer for foreign and
    unknown, A-1)."""
    parsed = parse_ref(ref)
    if parsed is None or parsed.kind != "session":
        return None
    found = store.lookup_ref(db, parsed.ref)
    if found is None:
        return None
    key, sess = found
    if token_id is not None and key != token_id:
        return None
    got = _session_rows(db, secret, key, sess, anchor=None, seg=parsed.segment)
    if got is None or not got[0]:
        return None
    rows, _trunc, sid, _root = got
    return sid, list(rows)


_MARKS = ("reused", "recomputed", "new")


def _turn_strip(a: SessionAnalysis) -> list[dict[str, Any]]:
    """One bar per turn (spec 12.1 item 5): tokens reused, recomputed (what the
    cache could have served but did not) and new (appended this turn), and the
    rule ids of the findings naming the turn."""
    out = []
    for t in a.turns:
        prompt = max(t.prompt_tokens, 0)
        reused = min(max(t.reused_own or 0, 0), prompt)
        if t.cf_tokens is not None and t.edge != "fork":
            new = max(prompt - t.cf_tokens, 0)
            recomputed = max(prompt - reused - new, 0)
        else:
            new, recomputed = prompt - reused, 0
        out.append(
            {
                "n": t.n,
                "req_id": t.req_id,
                "reused": reused,
                "recomputed": recomputed,
                "new": new,
                "markers": list(a.markers.get(t.n, [])),
            }
        )
    return out


def _cost(ss: ScoredSession) -> dict[str, Any]:
    """The session's figures of spec 5.2: tokens always, seconds with a rate,
    money with a card and its date."""
    a = ss.analysis
    prompt = sum(max(t.prompt_tokens, 0) for t in a.turns)
    reused = sum(max(t.reused_own or 0, 0) for t in a.turns)
    fixable = sum(
        t.lost_tokens or 0 for t in scoring.e_turns(a) if scoring.E_RULE_OF_CAUSE.get(t.cause or "")
    )
    out: dict[str, Any] = {
        "prompt_tokens": prompt,
        "reused_tokens": reused,
        "recomputed_tokens": max(prompt - reused, 0),
        "fixable_lost_tokens": fixable,
        "card": None,
        "actual_input": None,
        "saving": None,
        "saving_pct": None,
    }
    card = ss.card
    if card is not None:
        e_turns = scoring.e_turns(a)
        pe = sum(max(t.prompt_tokens, 0) for t in e_turns)
        ideal = sum(t.cf_tokens or 0 for t in e_turns)
        actual = prices.session_input_cost(prompt, card)
        saved = prices.price_saving(
            Saving(fixable, None, None, None),
            card,
            cached_share=min(1.0, ideal / pe) if pe else 1.0,
        ).money
        out["card"] = {
            "provider": card.provider,
            "model": card.model,
            "currency": card.currency,
            "effective_from": card.effective_from,
        }
        out["actual_input"] = actual
        out["saving"] = saved
        out["saving_pct"] = round(100 * saved / actual) if actual and saved is not None else None
    return out


def _finding_json(
    f: Finding,
    where: Mapping[str, Mapping[str, int]],
    a: SessionAnalysis,
    strip: bool,
    marks: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    d = f.to_json()
    if f.muted:
        d["muted_by"] = (marks or {}).get(f.rule)
    if strip and d["cause"] in ADMIN_ONLY_CAUSES:
        d["cause"] = None
    d["layer"] = RULES[f.rule].layer
    d["guard"] = _guard_json(f, a.turns, where) if f.rule == "R-09" else None
    return d


def _with_marks(ss: ScoredSession) -> dict[str, dict[str, Any]]:
    """Each score as JSON; where won't-fix marks left rules out of a score, its
    ``inputs["with_marks"]`` is the value with them counted (spec 11.4)."""
    out = {k: v.to_json() for k, v in ss.scores.items()}
    for k, plain in (ss.plain or {}).items():
        js = out[k]
        if js.get("scored") and plain.scored and plain.value != js.get("value"):
            js["inputs"] = {**js.get("inputs", {}), "with_marks": plain.value}
    return out


def scorecard_body(
    ss: ScoredSession,
    *,
    strip_admin: bool,
    names: Mapping[str, str],
    version: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    a = ss.analysis
    scores = _with_marks(ss)
    not_scored = {k: v.reason for k, v in ss.scores.items() if not v.scored and k != "overall"}
    return {
        "session": {
            "harness": ss.harness,
            "model": ss.model,
            "route": ss.route,
            "turns": len(ss.rows),
            "duration_s": max(ss.last_at - ss.started, 0.0),
            "version": (version or {}).get("label"),
            "version_id": ss.version,
            "version_since": (version or {}).get("since"),
            # The admin marks a won't-fix rule for the session's key; a key holder's
            # marks name no key (the sign-in's is used), so it is not sent to them.
            **({} if strip_admin else {"key_id": ss.key_id}),
        },
        "scores": scores,
        "top": [_finding_json(f, ss.where, a, strip_admin, ss.marks) for f in ss.top],
        "notice": (
            _finding_json(ss.notice, ss.where, a, strip_admin, ss.marks) if ss.notice else None
        ),
        "findings": [_finding_json(f, ss.where, a, strip_admin, ss.marks) for f in ss.findings],
        "not_scored": not_scored,
        "turns": _turn_strip(a),
        "cost": _cost(ss),
        "score_version": scoring.SCORE_VERSION,
        "weights": dict(scoring.WEIGHTS),
        "review": None,
        "names": dict(names),
    }


def _key_and_version(rows: Sequence[store.Row]) -> tuple[str, str | None]:
    last = max(rows, key=lambda r: (r.finished_at, r.id))
    return last.token_id or "", newest_version(rows)


def _scorecard_work(
    db_path: Any, state: Any, secret: str, token_id: str | None, ref: str
) -> dict[str, Any] | None:
    with closing(_connect(db_path)) as db:
        got = _resolve(db, secret, ref, token_id)
        if got is None:
            return None
        sid, rows = got
        try:
            key, version = _key_and_version(rows)
            marks = WontfixRepo(db).active_marks(key, version) if key else {}
            ss = score_rows(db, state, rows, sid=sid, strip_admin=token_id is not None, marks=marks)
        except Exception as exc:  # noqa: BLE001 -- the page shows none rather than fail
            logger.warning("lens: scorecard failed: %s", type(exc).__name__)
            return None
        if ss is None:
            return None
        hashes = {f.tool for f in ss.findings if f.tool}
        names = store.tool_names(db, ss.key_id, hashes) if hashes else {}
        label = (
            version_label(
                db, lineage_ids(db, ss.key_id), ss.model, ss.version, first_seen=ss.started
            )
            if ss.version and ss.key_id
            else None
        )
        return scorecard_body(ss, strip_admin=token_id is not None, names=names, version=label)


async def scorecard(state: Any, *, token_id: str | None, sess_ref: str) -> dict[str, Any] | None:
    """The scorecard of one session (spec 12.1), computed on read, or None
    (404): unknown, evicted, or not the caller's (a key holder reads its own
    key's sessions only, A-1; ``token_id=None`` is the admin)."""
    return await asyncio.to_thread(
        _scorecard_work,
        state.settings.db_path,
        state,
        state.settings.cookie_secret,
        token_id,
        sess_ref,
    )


def _key_names(db: sqlite3.Connection, ids: set[str]) -> dict[str, str]:
    """``api_tokens.name`` per id; a deleted key is simply absent."""
    if not ids:
        return {}
    marks = ", ".join("?" * len(ids))
    try:
        got = db.execute(f"SELECT id, name FROM api_tokens WHERE id IN ({marks})", sorted(ids))
        return {i: n for i, n in got.fetchall()}
    except sqlite3.OperationalError:
        return {}


def _list_json(
    r: SessionScoreRow, *, admin: bool, names: Mapping[str, str] | None = None
) -> dict[str, Any]:
    sc = r.scores()
    out: dict[str, Any] = {
        "session_id": r.sess_ref,
        "model": r.model,
        "route": r.route,
        "harness": r.harness,
        "turns": r.turns,
        "started": r.started,
        "last_at": r.last_at,
        "overall": r.overall,
        "e": r.e,
        "c": r.c,
        "r": r.r,
        "confidence": r.confidence.get("overall"),
        "reasons": {k: v.reason for k, v in sc.items() if not v.scored},
        "score_version": r.score_version,
        # Its own key's name for a key holder, any key's for the admin (A-1).
        "key_name": (names or {}).get(r.key_id),
    }
    if admin:
        out["key_id"] = r.key_id
    return out


def _list_work(
    db_path: Any,
    token_id: str | None,
    key: Sequence[str] | None,
    since: float,
    model: Sequence[str] | None,
    harness: Sequence[str] | None,
    limit: int,
    cursor: str | None,
) -> dict[str, Any]:
    with closing(_connect(db_path)) as db:
        try:
            rows = LensStore(db).sessions(
                token_id if token_id is not None else key,
                since=since,
                model=model,
                harness=harness,
                limit=limit + 1,
                cursor=cursor,
            )
        except sqlite3.OperationalError:
            rows = []
        page = rows[:limit]
        names = _key_names(db, {r.key_id for r in page})
    return {
        "sessions": [_list_json(r, admin=token_id is None, names=names) for r in page],
        "next_cursor": encode_cursor(page[-1]) if len(rows) > limit and page else None,
    }


async def session_list(
    state: Any,
    *,
    token_id: str | None,
    since: float | None = None,
    model: str | Sequence[str] | None = None,
    limit: int = 50,
    cursor: str | None = None,
    key: str | Sequence[str] | None = None,
    harness: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """The scored sessions, newest first (a session appears once it has been
    quiet for ``lens_quiet_s``). A key holder gets its own key's; ``key`` (one
    id or several) is the admin's filter and is ignored for a key holder (A-1).
    ``model`` and ``harness`` take one value or several (any of them)."""

    def many(v: str | Sequence[str] | None) -> list[str] | None:
        if v is None:
            return None
        return [x for x in ([v] if isinstance(v, str) else v) if x] or None

    now = time.time()
    return await asyncio.to_thread(
        _list_work,
        state.settings.db_path,
        token_id,
        many(key) if token_id is None else None,
        since if since is not None else now - DEFAULT_SINCE_S,
        many(model),
        many(harness),
        max(1, min(int(limit), MAX_LIST)),
        cursor,
    )


# -- trends, versions and won't fix (analysis Tasks 3.2, 3.3) ---------------------------

#: How far back trends and versions read when the caller gives no window.
TRENDS_SINCE_S = 90 * 24 * 3600.0
#: Stored session rows read for one trends answer, at most (newest first).
MAX_TREND_ROWS = 5_000
#: Versions listed, at most (the newest).
MAX_VERSIONS = 50


def _model_rows(
    db: sqlite3.Connection, key: str, since: float, model: str | None
) -> tuple[list[str], str | None, list[SessionScoreRow]]:
    """The key's chain, the model the answer is about (``model``, else the newest
    session's) and that model's stored rows since ``since``, with the key's
    won't-fix marks applied. Oldest first."""
    chain = lineage_ids(db, key)
    lens = LensStore(db)
    try:
        rows = lens.scores_for(chain, since=since, limit=MAX_TREND_ROWS)
    except sqlite3.OperationalError:
        rows = []
    if model is None and rows:
        model = rows[0].model  # newest first
    rows = [r for r in rows if r.model == model] if model else []
    marks = WontfixRepo(db).marks(key)
    rows = wontfix.apply_all(reversed(rows), marks)
    return chain, model, rows


def _labels(
    db: sqlite3.Connection, chain: Sequence[str], model: str | None
) -> dict[str, dict[str, Any]]:
    """Version id -> its label and first-seen time, over the chain's whole history
    on ``model`` (so a label does not move with the window)."""
    if not model:
        return {}
    got = db.execute(
        f"SELECT version, MIN(started) FROM lens_session_scores WHERE key_id IN ({','.join('?' * len(chain))})"
        " AND model = ? AND version IS NOT NULL GROUP BY version ORDER BY 2 ASC, 1 ASC",
        (*chain, model),
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    n = 0
    for v, first in got:
        v = str(v)
        if v.startswith("h:"):
            out[v] = {"label": v[2:], "since": float(first)}
        else:
            n += 1
            out[v] = {"label": f"v{n}", "since": float(first)}
    return out


def _version_list(
    rows: Sequence[SessionScoreRow], labels: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by: dict[str, list[SessionScoreRow]] = {}
    for r in rows:
        if r.version is not None:
            by.setdefault(r.version, []).append(r)
    order = sorted(by, key=lambda v: (min(r.started for r in by[v]), v))
    out: list[dict[str, Any]] = []
    prev: set[str] = set()
    for v in order[-MAX_VERSIONS:]:
        grp = by[v]
        fired = trends.fired_rules(grp)
        counts: dict[str, int] = {}
        for r in grp:
            for rule in {i for ids in r.rules.values() if isinstance(ids, list) for i in ids}:
                counts[str(rule)] = counts.get(str(rule), 0) + 1
        lab = labels.get(v) or {"label": v[2:] if v.startswith("h:") else v, "since": None}
        out.append(
            {
                "id": v,
                "label": lab["label"],
                "since": lab["since"] if lab["since"] is not None else min(r.started for r in grp),
                "last_at": max(r.last_at for r in grp),
                "n": len(grp),
                "medians": {
                    m: (
                        trends.median(x for r in grp if (x := trends.value_of(r, m)) is not None)
                        if len(grp) >= trends.MIN_BUCKET
                        and sum(1 for r in grp if trends.value_of(r, m) is not None)
                        >= trends.MIN_BUCKET
                        else None
                    )
                    for m in trends.METRICS
                },
                "rules": dict(sorted(counts.items())),
                "resolved": sorted(prev - fired),
                "new": sorted(fired - prev) if out else [],
            }
        )
        prev = fired
    return out


def _trends_work(
    db_path: Any, key: str, by: str, model: str | None, since: float, tz: int
) -> dict[str, Any]:
    with closing(_connect(db_path)) as db:
        chain, model, rows = _model_rows(db, key, since, model)
        labels = _labels(db, chain, model)
        try:
            models = [
                str(m)
                for (m,) in db.execute(
                    f"SELECT DISTINCT model FROM lens_session_scores WHERE key_id IN ({','.join('?' * len(chain))})"
                    " AND last_at >= ? ORDER BY model",
                    (*chain, since),
                ).fetchall()
            ]
        except sqlite3.OperationalError:
            models = []
    pts = trends.trend_series(rows, by=by, tz_offset_min=tz)  # type: ignore[arg-type]
    points = []
    for p in pts:
        d = {
            "bucket": p.bucket,
            "n": p.n,
            "medians": p.medians,
            "band": list(p.band) if p.band else None,
            "start": p.start,
        }
        if by == "version":
            d["label"] = (labels.get(p.bucket) or {}).get("label", p.bucket)
        points.append(d)
    return {
        "by": by,
        "model": model,
        "models": models,
        "points": points,
        "versions": _version_list(rows, labels),
        "weights": dict(scoring.WEIGHTS),
        "score_version": scoring.SCORE_VERSION,
    }


async def trends_body(
    state: Any,
    *,
    key: str,
    by: str = "week",
    model: str | None = None,
    since: float | None = None,
    tz_offset_min: int = 0,
) -> dict[str, Any]:
    """Medians per day, week or prompt version of the key's scored sessions (its
    whole rotation chain), with the won't-fix marks applied, plus the versions
    seen. ``key`` is the key holder's own or the admin's choice: the route decides."""
    now = time.time()
    return await asyncio.to_thread(
        _trends_work,
        state.settings.db_path,
        key,
        by,
        model,
        since if since is not None else now - TRENDS_SINCE_S,
        tz_offset_min,
    )


def _versions_work(db_path: Any, key: str, model: str | None, since: float) -> dict[str, Any]:
    with closing(_connect(db_path)) as db:
        chain, model, rows = _model_rows(db, key, since, model)
        labels = _labels(db, chain, model)
    return {"model": model, "versions": _version_list(rows, labels)}


async def versions_body(
    state: Any, *, key: str, model: str | None = None, since: float | None = None
) -> dict[str, Any]:
    now = time.time()
    return await asyncio.to_thread(
        _versions_work,
        state.settings.db_path,
        key,
        model,
        since if since is not None else now - TRENDS_SINCE_S,
    )


def _compare_work(
    db_path: Any, key: str, a: str, b: str, model: str | None, since: float
) -> dict[str, Any] | None:
    with closing(_connect(db_path)) as db:
        if model is None:
            # The model version ``b`` was last seen on.
            r = db.execute(
                f"SELECT model FROM lens_session_scores WHERE version = ? AND key_id IN"
                f" ({','.join('?' * len(lineage_ids(db, key)))}) ORDER BY last_at DESC LIMIT 1",
                (b, *lineage_ids(db, key)),
            ).fetchone()
            model = str(r[0]) if r is not None else None
        chain, model, rows = _model_rows(db, key, since, model)
        labels = _labels(db, chain, model)
    ra = [r for r in rows if r.version == a]
    rb = [r for r in rows if r.version == b]
    if not ra or not rb:
        return None
    c = trends.compare_versions(ra, rb, a=a, b=b)
    return {
        "model": model,
        "a": a,
        "b": b,
        "a_label": (labels.get(a) or {}).get("label", a),
        "b_label": (labels.get(b) or {}).get("label", b),
        "n_a": c.n_a,
        "n_b": c.n_b,
        "diffs": c.diffs,
        "verdict": c.verdict,
        "like_for_like": c.like_for_like,
        "note": c.note,
        "resolved": c.resolved,
        "new": c.new,
        "p": c.p,
    }


async def compare_body(
    state: Any, *, key: str, a: str, b: str, model: str | None = None, since: float | None = None
) -> dict[str, Any] | None:
    """Version ``b`` against version ``a`` of the key (same model), or None when
    either has no scored session."""
    now = time.time()
    return await asyncio.to_thread(
        _compare_work,
        state.settings.db_path,
        key,
        a,
        b,
        model,
        since if since is not None else now - TRENDS_SINCE_S,
    )


def _mark_json(m: wontfix.Mark) -> dict[str, Any]:
    return {"rule": m.rule, "version": m.version, "by": m.by, "at": m.at}


def _rw_connect(db_path: Any) -> sqlite3.Connection:
    """A writable connection (``_connect`` is read-only), for the won't-fix marks."""
    db = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    db.execute("PRAGMA busy_timeout = 30000")
    return db


def _wontfix_work(
    db_path: Any, key: str, action: str, rule: str, version: str | None, by: str
) -> tuple[bool, list[dict[str, Any]]]:
    with closing(_rw_connect(db_path) if action != "list" else _connect(db_path)) as db:
        repo = WontfixRepo(db)
        changed = True
        if action == "mark":
            changed = repo.mark(key, rule, version, by)
        elif action == "unmark":
            changed = repo.unmark(key, rule, version)
        return changed, [_mark_json(m) for m in repo.marks(key)]


async def wontfix_marks(state: Any, *, key: str) -> list[dict[str, Any]]:
    _ok, marks = await asyncio.to_thread(
        _wontfix_work, state.settings.db_path, key, "list", "", None, "key"
    )
    return marks


async def wontfix_change(
    state: Any, *, key: str, action: str, rule: str, version: str | None, by: str
) -> tuple[bool, list[dict[str, Any]]]:
    """Mark or unmark; (whether anything changed, the key's marks after)."""
    return await asyncio.to_thread(
        _wontfix_work, state.settings.db_path, key, action, rule, version, by
    )


def _key_known(db_path: Any, key: str) -> bool:
    with closing(_connect(db_path)) as db:
        return db.execute("SELECT 1 FROM api_tokens WHERE id = ?", (key,)).fetchone() is not None


async def key_known(state: Any, key: str) -> bool:
    return await asyncio.to_thread(_key_known, state.settings.db_path, key)


def _quartiles(vals: Sequence[int]) -> dict[str, int | None]:
    s = sorted(vals)
    if len(s) < trends.MIN_BUCKET:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    return {f"p{q}": round(trends._quantile(s, q / 100)) for q in (10, 25, 50, 75, 90)}


def _fleet_work(db_path: Any, model: str | None, since: float) -> dict[str, Any]:
    with closing(_connect(db_path)) as db:
        try:
            rows = LensStore(db).scores_for(None, since=since, model=model, limit=20_000)
        except sqlite3.OperationalError:
            rows = []
    out: dict[str, Any] = {
        "model": model,
        "sessions": len(rows),
        "keys": len({r.key_id for r in rows}),
    }
    for m in trends.METRICS:
        vals = [v for r in rows if (v := trends.value_of(r, m)) is not None]
        out[m] = {"n": len(vals), **_quartiles(vals)}
    return out


async def fleet_body(
    state: Any, *, model: str | None = None, since: float | None = None
) -> dict[str, Any]:
    """The distribution of the stored scores over every key (admin only, D-4).
    Stored scores carry no won't-fix marks, so a mark cannot move the fleet."""
    now = time.time()
    return await asyncio.to_thread(
        _fleet_work,
        state.settings.db_path,
        model,
        since if since is not None else now - TRENDS_SINCE_S,
    )
