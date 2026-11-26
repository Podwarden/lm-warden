"""Prompt Lens turn model: one stored request as the analysis reads it.

``turns_of`` adapts ``request_history`` rows (``app.sessions.store.Row``) to
``LensTurn``: codes, counts, sizes and keyed hashes, never text (A-3, A-10).
Pure: no I/O. A stored field that is missing or malformed reads as None
("not known"), never as 0 or as a cause.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.lens.hot import own_reuse
from app.sessions.costs import tokens_per_char
from app.sessions.store import Row


@dataclass(frozen=True)
class LensTurn:
    """One stored request of a session, ``n`` its 1-based place in start order.

    ``reused_own``/``outcome`` follow the lens asked for (``turns_of(own=)``):
    the key's own reuse and outcome, or the fleet's. ``tools_in`` is the number
    of offered tools; ``dead_weight`` is ``(dup, boil, reason)`` in tokens at
    the turn's own tokens per char; ``gap_s`` is the pause since the previous
    turn of the same lane ended. ``results`` are the tool results the request
    carried, ``(family, tokens, failed)`` each (``tools_in``); ``guard`` the
    guard's ``(rule, values, new)`` per rule. ``max_model_len`` is not stored on
    a row: the session analysis fills it when it has it.
    """

    req_id: str
    n: int
    lane: str | None
    edge: str | None
    prompt_tokens: int
    completion_tokens: int
    reused_own: int | None
    cause: str | None
    brk: str | None
    span_class: str | None
    span_line: int | None
    lost_tokens: int | None
    cf_tokens: int | None
    finish_reason: str | None
    max_tokens_req: int | None
    reasoning_tokens: int | None
    reply_chars: int | None
    calls_n: int | None
    #: ``(k, hash8, code)`` of the reply's non-ok calls, ``k`` the call's
    #: index in the reply (None: leaked markup, or a row stored before the
    #: index as ``[hash8, code]``); None: not computed.
    calls_bad: tuple[tuple[int | None, str, str], ...] | None
    #: As stored: ``(family, hash8)`` or ``(family, hash8, args8)`` per call.
    tools_out: tuple[tuple[str, ...], ...]
    tools_in: int | None
    shape: dict[str, Any] | None
    toolset: str | None
    tools_lint: tuple[int, ...] | None
    dead_weight: tuple[int, int, int] | None
    repeat_results: int | None
    ttft_s: float | None
    queued_s: float | None
    gap_s: float | None
    relayed: bool
    upstream: str | None
    guard: tuple[tuple[str, int, int], ...] | None = None
    max_model_len: int | None = None
    # -- beyond the brief's list, read by the rules ---------------------------
    outcome: str | None = None
    cached_tokens: int | None = None
    cache_marks: int | None = None
    cache_write_tokens: int | None = None
    model: str = ""
    started_at: float = 0.0
    tokens_per_char: float = 0.25
    results: tuple[tuple[str, int, bool], ...] | None = None


def _int(v: object) -> int | None:
    if isinstance(v, bool) or not isinstance(v, int):
        return None
    return v


def _json(v: Any) -> Any:
    if not isinstance(v, str):
        return None
    try:
        return json.loads(v)
    except ValueError:
        return None


def _calls_bad(v: Any) -> tuple[tuple[int | None, str, str], ...] | None:
    """``[k, hash8, code]`` entries, and the old ``[hash8, code]`` pairs as
    ``k = None``; anything else is skipped."""
    d = _json(v)
    if not isinstance(d, list):
        return None
    out: list[tuple[int | None, str, str]] = []
    for e in d:
        if not isinstance(e, list):
            continue
        if len(e) == 2 and isinstance(e[0], str) and isinstance(e[1], str):
            out.append((None, e[0], e[1]))
        elif len(e) >= 3 and isinstance(e[1], str) and isinstance(e[2], str):
            out.append((_int(e[0]), e[1], e[2]))
    return tuple(out)


def _lint(v: Any) -> tuple[int, ...] | None:
    d = _json(v)
    if not isinstance(d, list) or not all(_int(x) is not None for x in d):
        return None
    return tuple(int(x) for x in d)


def _results(v: Any) -> tuple[tuple[str, int, bool], ...] | None:
    d = _json(v)
    if not isinstance(d, list):
        return None
    out: list[tuple[str, int, bool]] = []
    for e in d:
        if isinstance(e, list) and len(e) >= 3 and isinstance(e[0], str):
            tok = _int(e[1])
            out.append((e[0], max(tok or 0, 0), e[2] is True))
    return tuple(out)


def _guard(v: Any) -> tuple[tuple[str, int, int], ...] | None:
    d = _json(v)
    f = d.get("f") if isinstance(d, dict) else None
    if not isinstance(f, list):
        return None
    out: list[tuple[str, int, int]] = []
    for e in f:
        if isinstance(e, list) and len(e) >= 3 and isinstance(e[0], str):
            c, n = _int(e[1]), _int(e[2])
            if c is not None and n is not None:
                out.append((e[0], max(c, 0), max(n, 0)))
    return tuple(out)


def _dead_weight(shape: dict[str, Any] | None, tpc: float) -> tuple[int, int, int] | None:
    dw = (shape or {}).get("dw")
    if not isinstance(dw, dict):
        return None
    vals = [_int(dw.get(k)) for k in ("dup", "boil", "reason")]
    if any(v is None for v in vals):
        return None
    d, b, r = (round(max(v or 0, 0) * tpc) for v in vals)
    return d, b, r


def _turn(r: Row, n: int, gap: float | None, own: bool) -> LensTurn:
    shape = r.shape if isinstance(r.shape, dict) else None
    chars = _int((shape or {}).get("chars"))
    tpc = tokens_per_char(r.prompt_tokens, chars)
    if own:
        reused = own_reuse(r.cached_tokens, r.reusable_tokens_own)
        outcome = r.cache_outcome_own
    else:
        reused = _int(r.cached_tokens)
        outcome = r.cache_outcome
    return LensTurn(
        req_id=r.id,
        n=n,
        lane=r.cobs2_conv,
        edge=r.cobs2_edge,
        prompt_tokens=r.prompt_tokens,
        completion_tokens=r.completion_tokens,
        reused_own=reused,
        cause=r.lens_cause,
        brk=r.lens_break,
        span_class=r.lens_span_class,
        span_line=_int(r.lens_span_line),
        lost_tokens=_int(r.lens_lost_tokens),
        cf_tokens=_int(r.lens_cf_tokens),
        finish_reason=r.finish_reason,
        max_tokens_req=_int(r.max_tokens_req),
        reasoning_tokens=_int(r.reasoning_tokens),
        reply_chars=_int(r.reply_chars),
        calls_n=_int(r.calls_n),
        calls_bad=_calls_bad(r.calls_bad),
        tools_out=r.tools_out,
        tools_in=_int((shape or {}).get("tools")),
        shape=shape,
        toolset=r.toolset,
        tools_lint=_lint(r.tools_lint),
        dead_weight=_dead_weight(shape, tpc),
        repeat_results=_int(r.lens_repeat_results),
        ttft_s=r.ttft_s,
        queued_s=r.queued_s,
        gap_s=gap,
        relayed=bool(r.relayed),
        upstream=r.upstream,
        guard=_guard(r.guard),
        outcome=outcome,
        cached_tokens=_int(r.cached_tokens),
        cache_marks=_int(r.cache_marks),
        cache_write_tokens=_int(r.cache_write_tokens),
        model=r.model,
        started_at=r.started_at,
        tokens_per_char=tpc,
        results=_results(r.tools_in),
    )


def turns_of(rows: Sequence[Row], *, own: bool) -> list[LensTurn]:
    """The session's turns in start order (ties by request id), numbered from 1.
    ``own=True`` reads the key's own lens (what a key holder may see, A-1)."""
    ordered = sorted(rows, key=lambda r: (r.started_at, r.id))
    last_end: dict[str, float] = {}
    out: list[LensTurn] = []
    for i, r in enumerate(ordered):
        gap: float | None = None
        if r.cobs2_conv is not None:
            prev = last_end.get(r.cobs2_conv)
            if prev is not None:
                gap = max(0.0, r.started_at - prev)
            last_end[r.cobs2_conv] = max(prev or r.finished_at, r.finished_at)
        out.append(_turn(r, i + 1, gap, own))
    return out
