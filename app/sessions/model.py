"""The pure session read model: lanes (one per conversation), main / subagent /
compaction lanes with known, inferred or compaction parents, side calls,
lineage gaps, totals and unused tools. Rows in, plain JSON out; O(rows) with
dict indexes. Built from persisted rows only, so it survives a restart."""

from __future__ import annotations

import bisect
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, TypeVar

from app.cache_obs.classify import DIVERGENCE_CAUSES
from app.sessions.costs import (
    Anatomy,
    DeadWeight,
    anatomy_json,
    anatomy_of,
    cache_counts,
    dead_weight_json,
    dead_weight_turn_json,
    recomputed_s,
    saved_vs_plain_s,
    served_s,
    share_of,
    tokens_per_char,
    turn_dead_weight,
    where_of,
)
from app.sessions.harness import session_harness
from app.sessions.pauses import PauseMark, PauseTurn, lens, mark_pauses, summarize_pauses
from app.sessions.refs import is_fallback, sess_ref, session_id
from app.sessions.replay import ReplayTurn, replay
from app.sessions.store import LLAMACPP_DEFAULT_SLOTS, ListRow, ModelEngine, Row

SEGMENT_GAP_S = 1800.0
INFER_WINDOW_S = 600.0
MAX_TURNS = 5000
#: Tool families whose call dispatches a subagent (versioned rule).
DISPATCH_FAMILIES = frozenset({"agent"})
DISPATCH_VERSION = 1
TOOL_ROWS_PAGE = 50
ACTIVE_S = 120.0
_MAX_TOOLS = 128


class RowLike(Protocol):
    @property
    def started_at(self) -> float: ...
    @property
    def finished_at(self) -> float: ...


_T = TypeVar("_T", bound=RowLike)


def segment(rows: Sequence[_T], gap_s: float = SEGMENT_GAP_S) -> list[list[_T]]:
    """Split time-ordered rows where the idle gap after everything so far
    exceeds ``gap_s``."""
    out: list[list[_T]] = []
    last_end = float("-inf")
    for r in sorted(rows, key=lambda x: x.started_at):
        if not out or r.started_at - last_end > gap_s:
            out.append([])
        out[-1].append(r)
        last_end = max(last_end, r.finished_at)
    return out


_RowT = TypeVar("_RowT", Row, ListRow)


def own_lens(row: _RowT, owned: Collection[str]) -> _RowT:
    """A key holder's view of one of its rows (admins never pass through here).

    The fleet-lens cache figures are matched against EVERY key's prompts, so
    they would tell a key holder that another key sent the same prefix: they
    go. Rows stored before the lineage was scoped per key (key_scope) can name
    a parent that is not one of ``owned`` (another key's request with the same
    client session id, or one whose owner is unknown): that parent goes too,
    with everything it caused (the edge, the conversation id it handed down,
    the shape's break depth, the own-lens compacted / harness_rewrite
    relabel), so the row reads as a new body with a plain diverged / cold miss."""
    ch: dict[str, Any] = {"reusable_tokens": None, "cache_outcome": None}
    if isinstance(row, Row):
        ch["divergence_cause"] = None
    preq, pnode = row.cobs2_parent_req, row.cobs2_parent_node
    if (preq is not None and preq not in owned) or (preq is None and pnode is not None):
        ch.update(cobs2_edge="new", cobs2_parent_req=None, cobs2_parent_node=None, cobs2_conv=None)
        if row.cache_outcome_own in DIVERGENCE_CAUSES:
            ch["cache_outcome_own"] = "diverged" if row.diverged_at_own is not None else "cold"
        if isinstance(row, Row):
            ch["divergence_cause_own"] = None
            if row.shape is not None and row.shape.get("brk") is not None:
                ch["shape"] = {**row.shape, "brk": None}
    return replace(row, **ch)


def is_side_call(turns: int, has_tools: bool, first_edge: str | None) -> bool:
    """A harness's one-shot call (title, topic, summary prefix): one turn, no
    tool use, and not a compaction continuation."""
    return turns == 1 and not has_tools and first_edge != "compacted_from"


def _ratio(row: Row) -> float:
    chars = (row.shape or {}).get("chars")
    return tokens_per_char(row.prompt_tokens, chars if isinstance(chars, int) else None)


def _tools(
    rows: Sequence[Row],
    toolsets: Mapping[str, list[tuple[str, str, int]]],
    tool_names: Mapping[str, str],
) -> dict[str, Any]:
    offered: dict[str, tuple[str, int]] = {}
    offered_turns: dict[str, int] = {}
    ratios: dict[str, list[float]] = {}
    called: dict[str, int] = {}
    truncated = False
    for r in rows:
        for e in r.tools_out:  # [family, hash8] or [family, hash8, args8]
            called[e[1]] = called.get(e[1], 0) + 1
    known: list[tuple[Row, list[tuple[str, str, int]]]] = []
    for r in rows:
        if r.toolset is None or r.toolset not in toolsets:
            continue
        entries = toolsets[r.toolset]
        known.append((r, entries))
        if len(entries) >= _MAX_TOOLS:
            truncated = True
        ratio = _ratio(r)
        for h, fam, chars in entries:
            offered[h] = (fam, chars)
            offered_turns[h] = offered_turns.get(h, 0) + 1
            ratios.setdefault(h, []).append(ratio)
    unused = {h for h in offered if h not in called}
    per_turn: list[float] = []
    prompt_sum = 0
    for r, entries in known:
        ratio = _ratio(r)
        per_turn.append(sum(chars * ratio for h, _f, chars in entries if h in unused))
        prompt_sum += r.prompt_tokens
    total = sum(per_turn)
    rows_out: list[dict[str, Any]] = [
        {
            "id": h,
            "name": tool_names.get(h),
            "family": fam,
            "chars": chars,
            "tokens": round(chars * (sum(ratios[h]) / len(ratios[h]))),
            "called": called.get(h, 0),
            "offered_turns": offered_turns.get(h, 0),
        }
        for h, (fam, chars) in offered.items()
    ]
    rows_out.sort(key=lambda x: (-x["tokens"], x["id"]))
    return {
        "offered": len(offered),
        "called": len([h for h in offered if h in called]),
        "unused": len(unused),
        "unused_tokens_per_turn": (total / len(per_turn)) if per_turn else None,
        "unused_tokens_total": total if per_turn else None,
        "unused_share": (total / prompt_sum) if prompt_sum else None,
        "estimated": True,
        "truncated": truncated,
        "rows": rows_out,
    }


def turn_json(
    row: Row, *, conversation_id: str | None, turn: int, lineage_unknown: bool, side_call: bool
) -> dict[str, Any]:
    prefill = row.ttft_s
    gen = max(0.0, row.duration_s - row.ttft_s) if row.ttft_s is not None else None
    return {
        "request_id": row.id,
        "conversation_id": conversation_id,
        "turn": turn,
        "started_at": row.started_at,
        "finished_at": row.finished_at,
        "edge": row.cobs2_edge,
        "node": row.cobs2_node,
        "parent_node": row.cobs2_parent_node,
        "parent_request_id": row.cobs2_parent_req,
        "prompt_tokens": row.prompt_tokens,
        "cached_tokens": row.cached_tokens,
        "generated_tokens": row.completion_tokens,
        "reusable_tokens": row.reusable_tokens,
        "reusable_tokens_own": row.reusable_tokens_own,
        "cache_outcome": row.cache_outcome,
        "cache_outcome_own": row.cache_outcome_own,
        "divergence_cause": row.divergence_cause,
        "divergence_cause_own": row.divergence_cause_own,
        "dp_rank": row.dp_rank,
        "queued_s": row.queued_s,
        "ttft_s": row.ttft_s,
        "prefill_s": prefill,
        "gen_s": gen,
        "finish_reason": row.finish_reason,
        "shape": row.shape,
        "tiers": {
            "gpu": row.served_gpu_tokens,
            "ram": row.served_ram_tokens,
            "nvme": row.served_nvme_tokens,
            "remote": row.served_remote_tokens,
            "load_s": row.load_s,
        },
        # Recorded with the turn, so an old turn reads right after the key switches.
        "isolated": row.kv_isolated,
        "node_id": row.node_id,
        "relayed": row.relayed,
        "upstream": row.upstream,
        "model": row.model,
        "lineage_unknown": lineage_unknown,
        "side_call": side_call,
    }


@dataclass(frozen=True)
class ListContext:
    toolsets: Mapping[str, list[tuple[str, str, int]]]
    engines: Mapping[str, ModelEngine]
    cold_rates: Mapping[str, float | None]


@dataclass(frozen=True)
class _C:
    """One row, reduced to what the cost figures need (from a Row or a ListRow)."""

    id: str
    lane: str | None
    started: float
    finished: float
    prompt: int
    #: cached prompt tokens, None when unknown (no value, or the engine cannot report)
    cached: int | None
    #: cached, else the TTFT estimate (pauses, R9)
    cached_any: int | None
    relayed: bool
    model_id: str
    chars: int | None
    dw: tuple[int, int, int] | None
    toolset: str | None
    called: tuple[str, ...]
    reusable: int | None
    outcome: str | None
    rate: float | None
    engine: ModelEngine | None


def _dw_tuple(v: Any) -> tuple[int, int, int] | None:
    if isinstance(v, dict) and all(isinstance(v.get(k), int) for k in ("dup", "boil", "reason")):
        return (v["dup"], v["boil"], v["reason"])
    return None


def _c(
    *,
    rid: str,
    lane: str | None,
    started: float,
    finished: float,
    prompt: int,
    cached_tokens: int | None,
    ttft_est: int | None,
    relayed: bool,
    model_id: str,
    chars: int | None,
    dw: tuple[int, int, int] | None,
    toolset: str | None,
    called: tuple[str, ...],
    reusable: int | None,
    reusable_own: int | None,
    outcome: str | None,
    outcome_own: str | None,
    engines: Mapping[str, ModelEngine],
    cold_rates: Mapping[str, float | None],
) -> _C:
    eng = engines.get(model_id)
    known = cached_tokens is not None and (eng is None or eng.cache_known)
    reu, out = lens(
        reusable=reusable, reusable_own=reusable_own, outcome=outcome, outcome_own=outcome_own
    )
    return _C(
        id=rid,
        lane=lane,
        started=started,
        finished=finished,
        prompt=prompt,
        cached=cached_tokens if known else None,
        cached_any=cached_tokens if cached_tokens is not None else ttft_est,
        relayed=relayed,
        model_id=model_id,
        chars=chars,
        dw=dw,
        toolset=toolset,
        called=called,
        reusable=reu,
        outcome=out,
        rate=cold_rates.get(model_id) if not relayed else None,
        engine=eng,
    )


def _c_of_row(r: Row, engines: Mapping[str, ModelEngine], rates: Mapping[str, float | None]) -> _C:
    chars = (r.shape or {}).get("chars")
    return _c(
        rid=r.id,
        lane=r.cobs2_conv,
        started=r.started_at,
        finished=r.finished_at,
        prompt=r.prompt_tokens,
        cached_tokens=r.cached_tokens,
        ttft_est=r.cached_ttft_est_tokens,
        relayed=r.relayed,
        model_id=r.model_id,
        chars=chars if isinstance(chars, int) else None,
        dw=_dw_tuple((r.shape or {}).get("dw")),
        toolset=r.toolset,
        called=tuple(e[1] for e in r.tools_out),
        reusable=r.reusable_tokens,
        reusable_own=r.reusable_tokens_own,
        outcome=r.cache_outcome,
        outcome_own=r.cache_outcome_own,
        engines=engines,
        cold_rates=rates,
    )


def _c_of_list(r: ListRow, lane: str | None, ctx: ListContext) -> _C:
    return _c(
        rid=r.id,
        lane=lane,
        started=r.started_at,
        finished=r.finished_at,
        prompt=r.prompt_tokens,
        cached_tokens=r.cached_tokens,
        ttft_est=r.cached_ttft_est_tokens,
        relayed=r.relayed,
        model_id=r.model_id,
        chars=r.chars,
        dw=r.dw,
        toolset=r.toolset,
        called=r.called,
        reusable=r.reusable_tokens,
        reusable_own=r.reusable_tokens_own,
        outcome=r.cache_outcome,
        outcome_own=r.cache_outcome_own,
        engines=ctx.engines,
        cold_rates=ctx.cold_rates,
    )


def _unused_chars(
    cs: Sequence[_C], toolsets: Mapping[str, list[tuple[str, str, int]]]
) -> dict[str, int]:
    """Per turn: chars of its offered tools that ``cs`` never called (R2)."""
    called = {h for c in cs for h in c.called}
    per_set: dict[str, int] = {}  # one sum per toolset: the called set is fixed here
    out: dict[str, int] = {}
    for c in cs:
        ts = c.toolset
        if ts is not None and ts in toolsets:
            n = per_set.get(ts)
            if n is None:
                n = per_set[ts] = sum(ch for h, _f, ch in toolsets[ts] if h not in called)
            out[c.id] = n
    return out


@dataclass
class _Analysis:
    turn_dw: dict[str, DeadWeight]
    sess_dw: list[DeadWeight]
    anatomy: dict[str, Anatomy]


def _analyse(cs: Sequence[_C], toolsets: Mapping[str, list[tuple[str, str, int]]]) -> _Analysis:
    lanes: dict[Any, list[_C]] = {}
    for i, c in enumerate(sorted(cs, key=lambda x: (x.started, x.id))):
        lanes.setdefault(c.lane if c.lane is not None else ("solo", i), []).append(c)
    conv_unused: dict[str, int] = {}
    for lc in lanes.values():
        conv_unused.update(_unused_chars(lc, toolsets))
    sess_unused = _unused_chars(cs, toolsets)
    turn_dw: dict[str, DeadWeight] = {}
    anat: dict[str, Anatomy] = {}
    sess_dw: list[DeadWeight] = []
    for lc in lanes.values():
        prev: int | None = None
        for c in lc:
            d = turn_dead_weight(
                prompt_tokens=c.prompt,
                chars=c.chars,
                dw=c.dw,
                unused_tool_chars=conv_unused.get(c.id),
            )
            turn_dw[c.id] = d
            su = sess_unused.get(c.id)
            sess_dw.append(
                d
                if su == conv_unused.get(c.id)
                else turn_dead_weight(
                    prompt_tokens=c.prompt, chars=c.chars, dw=c.dw, unused_tool_chars=su
                )
            )
            anat[c.id] = anatomy_of(
                prompt=c.prompt, cached=c.cached, prev_prompt=prev, dead=d.total
            )
            prev = c.prompt
    return _Analysis(turn_dw, sess_dw, anat)


def _marks(cs: Sequence[_C]) -> dict[str, PauseMark]:
    m = mark_pauses(
        [
            PauseTurn(
                c.lane,
                c.started,
                c.finished,
                c.relayed,
                c.cached_any,
                c.reusable,
                c.outcome,
                c.rate,
            )
            for c in cs
        ]
    )
    return {c.id: mk for c, mk in zip(cs, m, strict=True)}


def _sum_opt(vals: Iterable[float | None]) -> float | None:
    out: float | None = None
    for v in vals:
        if v is not None:
            out = (out or 0.0) + v
    return out


def _times(cs: Sequence[_C]) -> dict[str, float | None]:
    local = [c for c in cs if not c.relayed]
    return {
        "served": _sum_opt(served_s(c.cached, c.rate) for c in local),
        "recomputed": _sum_opt(recomputed_s(c.prompt, c.cached, c.rate) for c in local),
        "saved": _sum_opt(saved_vs_plain_s(c.cached, c.engine, c.rate) for c in local),
    }


def _replicas(cs: Sequence[_C]) -> int | None:
    seen = Counter(c.engine.dp_size for c in cs if c.engine is not None and not c.relayed)
    return seen.most_common(1)[0][0] if seen else None


def _replay(cs: Sequence[_C]) -> dict[str, Any] | None:
    local = [c for c in cs if not c.relayed]
    engs = Counter(c.engine for c in local if c.engine is not None)
    slots = engs.most_common(1)[0][0].slots if engs else LLAMACPP_DEFAULT_SLOTS
    return replay(
        [ReplayTurn(c.lane, c.started, c.prompt, c.cached, c.rate, c.engine) for c in local],
        slots=slots,
    )


def _anatomy_sum(an: Iterable[Anatomy]) -> dict[str, int]:
    c = d = r = n = u = 0.0
    for a in an:
        c += a.cached
        d += a.dead
        r += a.recomputed
        n += a.new
        u += a.unknown
    return anatomy_json(Anatomy(c, d, r, n, u))


def _conv_costs(lr: Sequence[Row], an: _Analysis, c_by_id: Mapping[str, _C]) -> dict[str, Any]:
    cs = [c_by_id[r.id] for r in lr]
    return {
        "dead_weight": dead_weight_json([an.turn_dw[r.id] for r in lr]),
        "anatomy": _anatomy_sum(an.anatomy[r.id] for r in lr),
        "saved_vs_vanilla_s": _times(cs)["saved"],
    }


def build_session(
    rows: Sequence[Row],
    *,
    session_id: str,
    token_name: str | None,
    truncated: bool,
    oldest_data_at: float | None,
    cold_rates: Mapping[str, float | None],
    toolsets: Mapping[str, list[tuple[str, str, int]]],
    tool_names: Mapping[str, str],
    root_session: str,
    engines: Mapping[str, ModelEngine] | None = None,
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: (r.started_at, r.id))
    by_id = {r.id: r for r in rows}
    engines = engines or {}
    cs = [_c_of_row(r, engines, cold_rates) for r in rows]
    an = _analyse(cs, toolsets)
    marks = _marks(cs)
    times = _times(cs)
    by_node: dict[str, Row] = {}
    for r in rows:
        if r.cobs2_node:
            by_node.setdefault(r.cobs2_node, r)
    lanes: dict[str | None, list[Row]] = {}
    for r in rows:
        lanes.setdefault(r.cobs2_conv, []).append(r)

    side: set[str] = set()
    for conv, lr in lanes.items():
        if conv is None:
            continue
        if is_side_call(len(lr), any(x.tools_out for x in lr), lr[0].cobs2_edge):
            side.add(conv)

    # Dispatch candidates (inference): rows of non-side lanes with an agent call.
    disp = [
        r
        for r in rows
        if r.cobs2_conv not in side and any(e[0] in DISPATCH_FAMILIES for e in r.tools_out)
    ]
    disp_starts = [r.started_at for r in disp]

    parents: dict[str, dict[str, Any]] = {}
    kinds: dict[str, str] = {}
    for conv, lr in lanes.items():
        if conv is None or conv in side:
            continue
        first = lr[0]
        prow: Row | None = None
        link: str | None = None
        if first.cobs2_edge == "compacted_from":
            prow = by_id.get(first.cobs2_parent_req or "") or by_node.get(
                first.cobs2_parent_node or ""
            )
            link = "compaction"
        elif first.parent_session_key and first.cobs2_session != root_session:
            prow = by_id.get(first.cobs2_parent_req or "")
            if prow is None:
                cands = [
                    r
                    for r in rows
                    if r.cobs2_session == first.parent_session_key
                    and r.started_at < first.started_at
                ]
                prow = cands[-1] if cands else None
            link = "known"
        elif first.cobs2_edge in ("subagent", "new") and (
            len(lr) >= 2 or any(x.tools_out for x in lr)
        ):
            hi = bisect.bisect_left(disp_starts, first.started_at)
            for r in reversed(disp[:hi]):
                if r.cobs2_conv == conv:
                    continue
                if r.finished_at >= first.started_at - INFER_WINDOW_S:
                    prow = r
                break
            link = "inferred"
        if prow is not None and prow.cobs2_conv != conv:
            parents[conv] = {
                "conversation_id": prow.cobs2_conv,
                "request_id": prow.id,
                "link": link,
            }
            kinds[conv] = "compaction" if link == "compaction" else "subagent"

    order = [c for c in lanes if c is not None and c not in side]
    order.sort(key=lambda c: (lanes[c][0].started_at, -len(lanes[c])))
    parentless = [c for c in order if c not in parents]
    main = parentless[0] if parentless else None
    labels: dict[str, str] = {}
    comp_n: dict[str, tuple[str, int]] = {}
    sub_k = 0
    conv_k = 1
    for c in order:
        if c == main:
            labels[c] = "main"
            kinds[c] = "main"
        elif c not in parents:
            conv_k += 1
            labels[c] = f"conversation {conv_k}"
            kinds[c] = "conversation"
        elif kinds[c] == "subagent":
            sub_k += 1
            labels[c] = f"subagent #{sub_k}"
    for c in order:  # compaction labels need the parent's label: start order
        if kinds.get(c) != "compaction":
            continue
        pc = parents[c]["conversation_id"]
        base, n = comp_n.get(pc, (labels.get(pc, "conversation"), 0))
        comp_n[c] = (base, n + 1)
        labels[c] = f"{base} after compaction" + (f" {n + 1}" if n else "")

    turns: list[dict[str, Any]] = []
    unknown_by_conv: dict[str | None, bool] = {}
    for conv, lr in lanes.items():
        is_side = conv in side
        for i, r in enumerate(lr):
            unknown = False
            if (
                r.cobs2_parent_node
                and r.cobs2_parent_req not in by_id
                and r.cobs2_parent_node not in by_node
            ):
                unknown = True
            if i > 0 and r.cobs2_edge == "new":
                unknown = True
            if r.cobs2_edge is None and r.cobs2_node is None:
                unknown = True
            unknown_by_conv[conv] = unknown_by_conv.get(conv, False) or unknown
            turns.append(
                turn_json(
                    r, conversation_id=conv, turn=i + 1, lineage_unknown=unknown, side_call=is_side
                )
            )
    turns.sort(key=lambda t: (t["started_at"], t["request_id"]))
    c_by_id = {c.id: c for c in cs}
    for t in turns:
        rid = t["request_id"]
        mk = marks.get(rid)
        t["dead_weight"] = dead_weight_turn_json(an.turn_dw[rid])
        t["anatomy"] = anatomy_json(an.anatomy[rid])
        t["cache_known"] = c_by_id[rid].cached is not None
        t["pause_s"] = mk.pause_s if mk is not None else None
        t["evicted_on_resume"] = bool(mk is not None and mk.evicted)
        t["lost_tokens"] = mk.lost_tokens if mk is not None and mk.evicted else None
        t["lost_s"] = mk.lost_s if mk is not None and mk.evicted else None

    def sums(lr: Sequence[Row]) -> dict[str, Any]:
        return {
            "first_at": min(x.started_at for x in lr),
            "last_at": max(x.finished_at for x in lr),
            "turns": len(lr),
            "prefill_tokens": sum(x.prompt_tokens for x in lr if not x.relayed),
            "cached_tokens": sum(x.cached_tokens or 0 for x in lr if not x.relayed),
            "generated_tokens": sum(x.completion_tokens for x in lr),
            "relayed": any(x.relayed for x in lr),
        }

    convs: list[dict[str, Any]] = []
    for c in ([main] if main else []) + [c for c in order if c != main]:
        lr = lanes[c]
        convs.append(
            {
                "id": c,
                "label": labels.get(c, c),
                "kind": kinds.get(c, "conversation"),
                "parent": parents.get(c),
                **sums(lr),
                "lineage_unknown": unknown_by_conv.get(c, False),
                "tools": _tools(lr, toolsets, tool_names),
                **_conv_costs(lr, an, c_by_id),
            }
        )
    if None in lanes:
        lr = lanes[None]
        convs.append(
            {
                "id": None,
                "label": "unlinked",
                "kind": "unlinked",
                "parent": None,
                **sums(lr),
                "lineage_unknown": True,
                "tools": _tools(lr, toolsets, tool_names),
                **_conv_costs(lr, an, c_by_id),
            }
        )

    local = [r for r in rows if not r.relayed]
    rel = [r for r in rows if r.relayed]
    prefill = sum(r.prompt_tokens for r in local)
    cached = sum(r.cached_tokens or 0 for r in local)
    served = times["served"]
    started = min((r.started_at for r in rows), default=None)
    last = max((r.finished_at for r in rows), default=None)
    totals = {
        "started_at": started,
        "last_at": last,
        "wall_s": (last - started) if started is not None and last is not None else None,
        "requests": len(rows),
        "conversations": len(order),
        "side_calls": sum(len(lanes[c]) for c in side),
        "compactions": sum(1 for c in order if kinds.get(c) == "compaction"),
        "prefill_tokens": prefill,
        "cached_tokens": cached,
        # Over the local requests that report cache data only (verified finding #67).
        "cached_share": _local_share(local),
        "cache_share": share_of(cache_counts((c.prompt, c.cached) for c in cs)),
        "generated_tokens": sum(r.completion_tokens for r in rows),
        "cache_served_s": served,
        "prompt_tokens": sum(r.prompt_tokens for r in rows),
        "recomputed_s": times["recomputed"],
        "saved_vs_vanilla_s": times["saved"],
        "replicas": _replicas(cs),
        "where": where_of(len(local), len(rel)),
        "relayed": (
            {
                "requests": len(rel),
                "prefill_tokens": sum(r.prompt_tokens for r in rel),
                "cached_tokens": sum(r.cached_tokens or 0 for r in rel),
                "upstream": next((r.upstream for r in rel if r.upstream), None),
            }
            if rel
            else None
        ),
    }
    head = rows[0] if rows else None
    return {
        "session_id": session_id,
        "token_id": head.token_id if head else None,
        "token_name": token_name,
        "models": sorted({r.model for r in rows}),
        "truncated": truncated,
        "oldest_data_at": oldest_data_at,
        "totals": totals,
        "tools": _tools(rows, toolsets, tool_names),
        "harness": session_harness(r.session_source for r in rows),
        "dead_weight": dead_weight_json(an.sess_dw),
        "anatomy": _anatomy_sum(an.anatomy.values()),
        "engine_replay": _replay(cs),
        "pauses": summarize_pauses(marks.values()),
        "conversations": convs,
        "side_calls": [
            {
                "request_id": r.id,
                "started_at": r.started_at,
                "model": r.model,
                "prompt_tokens": r.prompt_tokens,
            }
            for c in side
            for r in lanes[c]
        ],
        "turns": turns,
    }


def _local_share(local: Sequence[Row] | Sequence[ListRow]) -> float | None:
    """Cached over prompt tokens of the local requests that report cache data; None when none do."""
    known = [r for r in local if r.cached_tokens is not None]
    p = sum(r.prompt_tokens for r in known)
    return (sum(min(r.cached_tokens or 0, r.prompt_tokens) for r in known) / p) if p else None


def _list_item(
    rows: Sequence[ListRow],
    *,
    sid: str,
    now: float,
    ctx: ListContext | None = None,
    marks: Mapping[str, PauseMark] | None = None,
    cs: Sequence[_C] | None = None,
    fallback: bool = False,
) -> dict[str, Any]:
    convs: dict[str | None, list[ListRow]] = {}
    for r in sorted(rows, key=lambda x: x.started_at):
        convs.setdefault(r.cobs2_conv, []).append(r)
    n_conv = 0
    comps = 0
    for c, lr in convs.items():
        if c is None:
            continue
        if is_side_call(len(lr), any(x.has_tools for x in lr), lr[0].cobs2_edge):
            continue
        n_conv += 1
        if lr[0].cobs2_edge == "compacted_from":
            comps += 1
    local = [r for r in rows if not r.relayed]
    prefill = sum(r.prompt_tokens for r in local)
    cached = sum(r.cached_tokens or 0 for r in local)
    started = min(r.started_at for r in rows)
    last = max(r.finished_at for r in rows)
    first = rows[0]
    return {
        "session_id": sid,
        "token_id": first.token_id,
        "token_name": first.token_name,
        "models": sorted({r.model for r in rows}),
        "started_at": started,
        "last_at": last,
        "duration_s": last - started,
        "requests": len(rows),
        "conversations": n_conv,
        "prefill_tokens": prefill,
        "cached_tokens": cached,
        "cached_share": _local_share(local),
        "relayed_requests": sum(1 for r in rows if r.relayed),
        "generated_tokens": sum(r.completion_tokens for r in rows),
        "compactions": comps,
        "active": now - last <= ACTIVE_S,
        **_list_costs(rows, ctx or ListContext({}, {}, {}), marks or {}, cs, fallback),
    }


def _list_costs(
    rows: Sequence[ListRow],
    ctx: ListContext,
    marks: Mapping[str, PauseMark],
    cs: Sequence[_C] | None = None,
    fallback: bool = False,
) -> dict[str, Any]:
    if cs is None:
        cs = [_c_of_list(r, r.cobs2_conv, ctx) for r in rows]
    an = _analyse(cs, ctx.toolsets)
    times = _times(cs)
    dw = dead_weight_json(an.sess_dw)
    mk = [marks[r.id] for r in rows if r.id in marks]
    # A fallback session's page starts after a 30 min gap, so a resume across
    # one is not this item's (I3); the cross-session summary still counts it.
    ev = [
        m
        for m in mk
        if m.evicted and not (fallback and m.pause_s is not None and m.pause_s > SEGMENT_GAP_S)
    ]
    ck = cache_counts((c.prompt, c.cached) for c in cs)
    local = sum(1 for r in rows if not r.relayed)
    return {
        "harness": session_harness(r.session_source for r in rows),
        "where": where_of(local, len(rows) - local),
        "prompt_tokens": sum(r.prompt_tokens for r in rows),
        "anatomy": _anatomy_sum(an.anatomy.values()),
        "dead_weight_share": dw["share"],
        "dead_weight_tokens": dw["total_tokens"],
        "dead_weight_measured": dw["turns_measured"] > 0,
        "saved_vs_vanilla_s": times["saved"],
        "replicas": _replicas(cs),
        "recomputed_s": times["recomputed"],
        "cache_served_s": times["served"],
        "cache_known": any(c.cached is not None for c in cs if not c.relayed),
        "cache_share": share_of(ck),
        "_ck": ck,
        "evicted_resumes": len(ev),
        "evicted_lost_s": _sum_opt(m.lost_s for m in ev),
        "_marks": mk,
    }


def summarize_list(
    rows: Sequence[ListRow],
    *,
    secret: str,
    root_of: Callable[[str, str], str],
    now: float,
    ctx: ListContext | None = None,
) -> list[dict[str, Any]]:
    """Sessions list items, newest activity first: child sessions fold into
    their root; session-less (fallback) sessions split on idle gaps."""
    ctx = ctx or ListContext({}, {}, {})
    # Pause marks over lanes keyed (token, session, conversation), before the
    # fallback split (R13): a long pause in session-less traffic still counts.
    all_cs = [
        _c_of_list(
            r, f"{r.token_id}|{r.cobs2_session}|{r.cobs2_conv}" if r.cobs2_conv else None, ctx
        )
        for r in rows
    ]
    marks = _marks(all_cs)
    c_by_id = {c.id: c for c in all_cs}
    roots: dict[tuple[Any, str], str] = {}
    groups: dict[tuple[Any, str], list[ListRow]] = {}
    for r in rows:
        key = (r.token_id, r.cobs2_session)
        if is_fallback(r.cobs2_session):
            root = r.cobs2_session
        else:
            known = roots.get(key)
            if known is None:
                known = roots[key] = root_of(r.token_id or "", r.cobs2_session)
            root = known
        groups.setdefault((r.token_id, root), []).append(r)
    items: list[dict[str, Any]] = []
    for (tok, root), grp in groups.items():
        ref = sess_ref(secret, tok, root)
        if ref is None:
            continue
        if is_fallback(root):
            for seg in segment(grp):
                items.append(
                    _list_item(
                        seg,
                        sid=session_id(ref, seg[0].id),
                        now=now,
                        ctx=ctx,
                        marks=marks,
                        cs=[c_by_id[r.id] for r in seg],
                        fallback=True,
                    )
                )
        else:
            items.append(
                _list_item(
                    grp,
                    sid=session_id(ref),
                    now=now,
                    ctx=ctx,
                    marks=marks,
                    cs=[c_by_id[r.id] for r in grp],
                )
            )
    items.sort(key=lambda i: -i["last_at"])
    return items
