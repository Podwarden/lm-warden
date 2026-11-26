"""Prompt Lens layer 1: why a request did not reuse its prefix (lens spec v1 4.2).

``evaluate`` is a pure function of ids, digests, probes and counts: no I/O, no
text kept (A-3, A-10). It names one primary cause per request (the first
broken part decides it), counts the other breaks, and gives the counterfactual
reuse and the tokens lost to the miss (v1 5.1-5.2). When the evidence cannot
attribute a cause (e.g. a continuing turn whose parent is no longer in memory)
it returns the empty verdict: never a guessed ``new_conversation`` or
``history_rewritten``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from app.cobs2.lineage import Observation
from app.lens.probes import PROBE_ORDER, first_changed_line, span_class

#: The 18 codes of v1 4.2, in evaluation order. Two departures inside the
#: groups, both because a code would otherwise be wrong: the body probes are
#: tried most specific first (see ``_body_cause``) and, on a relay,
#: ``provider_below_min`` is tried before ``provider_no_marks`` (spec 11.1
#: E-06's guard: below the minimum, marks would not have helped).
CAUSES: tuple[str, ...] = (
    "compacted",
    "fork",
    "sys_noise",
    "sys_changed",
    "tools_reordered",
    "tools_key_order",
    "tools_added_end",
    "tools_changed",
    "history_rewritten",
    "history_noise",
    "reasoning_dropped",
    "reasoning_kept",
    "media_changed",
    "new_conversation",
    "server_evicted",
    "misrouted",
    "provider_no_marks",
    "provider_below_min",
)

#: Causes the prompt author controls (spec 10.2). ``compacted`` counts only
#: with a changed head: a compaction that keeps the head has a counterfactual of
#: the head alone, so it loses nothing when the head was reused.
APP_FIXABLE: frozenset[str] = frozenset(
    {
        "sys_noise",
        "sys_changed",
        "tools_reordered",
        "tools_key_order",
        "tools_added_end",
        "tools_changed",
        "history_rewritten",
        "history_noise",
        "reasoning_dropped",
        "reasoning_kept",
        "media_changed",
        "provider_no_marks",
        "compacted",
    }
)

RenderOrder = Literal["sys_first", "tools_first"]
Probes = tuple[bytes, bytes, bytes, bytes]

_NOISE = PROBE_ORDER.index("noise")
_NO_REASONING = PROBE_ORDER.index("no_reasoning")
_TEXT_ONLY = PROBE_ORDER.index("text_only")
_WS = PROBE_ORDER.index("ws")


@dataclass(frozen=True)
class CauseInputs:
    """Everything ``evaluate`` reads. Every field is optional; a missing one
    narrows what can be attributed, it never raises.

    ``*_parent`` head fields are the lineage parent's (for ``edge = new``: the
    recent head of the same key, ``app.lens.heads``). ``msgs_now`` are this
    request's 8-byte short msgs (``lineage.short_msg``); ``probes_now`` are the
    probes of ``msgs_now[b]``, ``b`` the first index where it differs from
    ``obs.parent_msgs``. ``sys_text_line(n)`` returns line ``n`` (1-based) of the
    current system prompt; only the span class is kept.
    """

    edge: str | None = None
    obs: Observation | None = None
    sys_now: str | None = None
    sys_parent: str | None = None
    loose_sys_now: str | None = None
    loose_sys_parent: str | None = None
    #: Line digests (a tuple, or ``probes.PackedDigests``: any sequence of 8-byte digests).
    sys_digests_now: Sequence[bytes] | None = None
    sys_digests_parent: Sequence[bytes] | None = None
    tools_now: str | None = None
    tools_parent: str | None = None
    loose_tools_now: str | None = None
    loose_tools_parent: str | None = None
    #: Per-tool hashes (``probes.PackedDigests``, or any sequence of 8-byte
    #: digests); None above ``schema_lint.MAX_TOOLS`` tools or not computed.
    tool_hashes_now: Sequence[bytes] | None = None
    tool_hashes_parent: Sequence[bytes] | None = None
    msgs_now: tuple[bytes, ...] = ()
    probes_now: Probes | None = None
    #: Tokens per body message, in order.
    msg_tokens: tuple[int, ...] = ()
    #: Tokens of the head (system prompt and tools).
    head_tokens: int = 0
    own_outcome: str | None = None
    gap_s: float | None = None
    relayed_to: str | None = None
    #: ``cache_control`` marks counted on a route that needs them (None: not counted).
    cache_marks: int | None = None
    provider_min: int | None = None
    sys_text_line: Callable[[int], str] | None = None
    #: The tool list's share of ``head_tokens`` (None: the split is unknown and
    #: a reuse that stops between sys and tools counts 0 for the head).
    tools_tokens: int | None = None
    #: Tokens this request actually reused (own lens); None: lost is unknown.
    actual_tokens: int | None = None
    #: Whether the message at the break carries reasoning now; tells
    #: ``reasoning_kept`` from ``reasoning_dropped`` (None: dropped, the usual case).
    reasoning_now: bool | None = None
    #: True when the body has no prior turn (no assistant and no tool
    #: message). ``new_conversation`` needs it: a multi-turn body
    #: with ``edge = new`` is a conversation whose parent left memory (restart,
    #: TTL), not a new one. None (unknown) or False: never ``new_conversation``.
    first_turn: bool | None = None


@dataclass(frozen=True)
class LensVerdict:
    brk: str | None
    cause: str | None
    span_class: str | None
    span_line: int | None
    counterfactual_tokens: int | None
    lost_tokens: int | None
    #: Further broken parts after the first; None when not attributed.
    other_breaks: int | None


#: The unattributed verdict: every field None (M2), so an unattributed row
#: stores no lens field at all.
EMPTY = LensVerdict(None, None, None, None, None, None, None)


@dataclass(frozen=True)
class _HeadBreak:
    part: str  # "sys" | "tools"
    cause: str
    line: int | None = None
    cls: str | None = None


def evaluate(ci: CauseInputs, *, render_order: RenderOrder) -> LensVerdict:
    """The layer 1 verdict for one request. Never raises: an internal error
    gives the empty verdict (fail-open, A-5)."""
    try:
        return _evaluate(ci, render_order)
    except Exception:  # noqa: BLE001 - the lens never breaks a request
        return EMPTY


# -- internals ---------------------------------------------------------------


def _tok(v: object) -> int:
    """A token count as a non-negative int (anything else counts 0)."""
    if isinstance(v, bool) or not isinstance(v, int | float):
        return 0
    try:
        return max(int(v), 0)
    except (ValueError, OverflowError):  # nan, inf
        return 0


def _lost(ci: CauseInputs, cf: int | None) -> int | None:
    if cf is None or ci.actual_tokens is None:
        return None
    return max(0, cf - _tok(ci.actual_tokens))


def _differs(a: object, b: object) -> bool:
    """Both sides known and different."""
    return a is not None and b is not None and a != b


def _equal(a: object, b: object) -> bool:
    """Both sides known and equal."""
    return a is not None and b is not None and a == b


def _sys_break(ci: CauseInputs) -> _HeadBreak | None:
    if not _differs(ci.sys_now, ci.sys_parent):
        return None
    cause = "sys_noise" if _equal(ci.loose_sys_now, ci.loose_sys_parent) else "sys_changed"
    line: int | None = None
    cls: str | None = None
    if ci.sys_digests_now is not None and ci.sys_digests_parent is not None:
        fc = first_changed_line(ci.sys_digests_parent, ci.sys_digests_now)
        line = fc[0] if fc else None
    if line is not None and ci.sys_text_line is not None:
        try:
            text = ci.sys_text_line(line)
            cls = span_class(text) if isinstance(text, str) else None
        except Exception:  # noqa: BLE001 - the text is optional evidence
            cls = None
    return _HeadBreak("sys", cause, line, cls)


def _tools_break(ci: CauseInputs) -> _HeadBreak | None:
    if not _differs(ci.tools_now, ci.tools_parent):
        return None
    hn, hp = ci.tool_hashes_now, ci.tool_hashes_parent
    loose_equal = _equal(ci.loose_tools_now, ci.loose_tools_parent)
    if hn is not None and hp is not None:
        if hn != hp and sorted(hn) == sorted(hp):
            return _HeadBreak("tools", "tools_reordered")
        if loose_equal:
            return _HeadBreak("tools", "tools_key_order")
        if len(hn) > len(hp) and tuple(hn[: len(hp)]) == tuple(hp):
            return _HeadBreak("tools", "tools_added_end")
        return _HeadBreak("tools", "tools_changed")
    return _HeadBreak("tools", "tools_key_order" if loose_equal else "tools_changed")


def _head_breaks(ci: CauseInputs, order: RenderOrder) -> list[_HeadBreak]:
    s, t = _sys_break(ci), _tools_break(ci)
    seq = (t, s) if order == "tools_first" else (s, t)
    return [b for b in seq if b is not None]


def _counterfactual(
    ci: CauseInputs,
    order: RenderOrder,
    heads: list[_HeadBreak],
    primary: str | None,
    parent_msgs: tuple[bytes, ...],
    diffs: list[int],
) -> int:
    """Tokens of the prefix shared with the parent when the ``primary`` broken
    part is treated as equal: walk sys and tools in render order, then the body
    messages the parent had, and stop at the next broken part (v1 5.1)."""
    head = _tok(ci.head_tokens)
    tools = min(_tok(ci.tools_tokens), head) if ci.tools_tokens is not None else None
    part_tok = {"tools": tools, "sys": head - tools if tools is not None else None}
    broken = {b.part for b in heads}
    parts = ("tools", "sys") if order == "tools_first" else ("sys", "tools")
    for i, part in enumerate(parts):
        if part in broken and part != primary:
            # Stops inside the head: count the part before it only when the split is known.
            if i == 0:
                return 0
            first = part_tok[parts[0]]
            return first if first is not None else 0
    total = head
    stop = {d for d in diffs if f"node:{d}" != primary}
    toks = ci.msg_tokens
    for i in range(min(len(parent_msgs), len(ci.msgs_now))):
        if i in stop:
            break
        total += _tok(toks[i]) if i < len(toks) else 0
    return total


def _body_cause(ci: CauseInputs, obs: Observation) -> str | None:
    """The first probe that equalises the parent's and this request's message
    names the cause. Tried most specific first: the ``noise`` probe is the loose
    form, which also drops reasoning and collapses whitespace, so in
    ``PROBE_ORDER`` it would hide ``reasoning_*`` and ``ws``. No probes on
    either side: no cause (never a guessed ``history_rewritten``)."""
    pp, pn = obs.parent_probes, ci.probes_now
    if pp is None or pn is None or len(pp) != len(PROBE_ORDER) or len(pn) != len(PROBE_ORDER):
        return None
    if pp[_WS] == pn[_WS]:
        return "history_noise"
    if pp[_NO_REASONING] == pn[_NO_REASONING]:
        return "reasoning_kept" if ci.reasoning_now is True else "reasoning_dropped"
    if pp[_TEXT_ONLY] == pn[_TEXT_ONLY]:
        return "media_changed"
    if pp[_NOISE] == pn[_NOISE]:
        return "history_noise"
    return "history_rewritten"


def _stable_cause(ci: CauseInputs, cf: int) -> str | None:
    """Server and provider causes, for a prompt that did not break."""
    outcome = ci.own_outcome
    relayed = bool(ci.relayed_to)
    missed = outcome is not None and outcome != "hit"
    provider: list[str] = []
    if missed and ci.provider_min is not None and cf < _tok(ci.provider_min):
        provider.append("provider_below_min")
    if missed and relayed and ci.cache_marks is not None and _tok(ci.cache_marks) == 0:
        provider.append("provider_no_marks")
    server: list[str] = []
    if outcome == "lost":
        server.append("server_evicted")
    elif outcome == "misrouted":
        server.append("misrouted")
    # On a relay the provider's rules explain a miss first; on our own engines
    # the server does.
    seq = provider + server if relayed else server + provider
    return seq[0] if seq else None


def _head_verdict(
    ci: CauseInputs,
    order: RenderOrder,
    heads: list[_HeadBreak],
    parent_msgs: tuple[bytes, ...],
    diffs: list[int],
) -> LensVerdict:
    p = heads[0]
    cf = _counterfactual(ci, order, heads, p.part, parent_msgs, diffs)
    other = len(heads) - 1 + len(diffs)
    return LensVerdict(p.part, p.cause, p.cls, p.line, cf, _lost(ci, cf), other)


def _evaluate(ci: CauseInputs, order: RenderOrder) -> LensVerdict:
    obs = ci.obs
    edge = ci.edge if ci.edge is not None else (obs.edge if obs is not None else None)
    pm: tuple[bytes, ...] = tuple(obs.parent_msgs) if obs is not None else ()
    now = ci.msgs_now
    diffs = [i for i in range(min(len(pm), len(now))) if pm[i] != now[i]]

    if edge == "compacted_from":
        heads = _head_breaks(ci, order)
        cf = _tok(ci.head_tokens)
        brk = heads[0].part if heads else "node:0"
        return LensVerdict(brk, "compacted", None, None, cf, _lost(ci, cf), max(len(heads) - 1, 0))

    if edge == "fork":
        if not pm:
            return LensVerdict(None, "fork", None, None, None, None, 0)
        b = diffs[0] if diffs else len(pm)
        cf = _counterfactual(ci, order, [], None, pm, diffs)
        return LensVerdict(f"node:{b}", "fork", None, None, cf, _lost(ci, cf), 0)

    if edge == "new":
        heads = _head_breaks(ci, order)
        if heads:
            # The key's recent head (same loose sys and tools) differs: the
            # head comparison stands on its own, whatever the body holds.
            return _head_verdict(ci, order, heads, (), [])
        if ci.first_turn is not True:
            # Review Focus 1: a body with earlier turns and no parent in memory
            # (or an unknown body) is not attributed.
            return EMPTY
        if ci.sys_parent is None:
            # No recent head of this key: nothing to compare the head with.
            return LensVerdict(None, "new_conversation", None, None, None, None, 0)
        cf = _tok(ci.head_tokens)
        return LensVerdict(None, "new_conversation", None, None, cf, _lost(ci, cf), 0)

    if edge == "subagent":
        heads = _head_breaks(ci, order)
        return _head_verdict(ci, order, heads, (), []) if heads else EMPTY

    if edge != "continue" or obs is None or obs.parent is None or not pm:
        # Review Focus 1: no parent in memory, no attribution.
        return EMPTY

    heads = _head_breaks(ci, order)
    if heads:
        return _head_verdict(ci, order, heads, pm, diffs)
    if diffs:
        b = diffs[0]
        cause = _body_cause(ci, obs)
        if cause is None:
            return EMPTY
        cf = _counterfactual(ci, order, [], f"node:{b}", pm, diffs)
        return LensVerdict(f"node:{b}", cause, None, None, cf, _lost(ci, cf), len(diffs) - 1)
    if obs.loose_match_beyond_exact:
        # The exact chain differs but no part we can see does (e.g. the head
        # ids were not passed): the change is real, its place unknown.
        return EMPTY
    cf = _counterfactual(ci, order, [], None, pm, [])
    return LensVerdict(None, _stable_cause(ci, cf), None, None, cf, _lost(ci, cf), 0)
