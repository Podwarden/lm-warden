"""G3: the per-request kvsig parts event (spec 2026-11-22 prompt_lens dock).

``build_kvsig`` is pure: ids, the lineage observation, the cache-observation
chain and its two prefix matches, and the body's layout (sizes and roles from
``prompt_lens_capture.body_layout``) in; the event body out. It reads no text
and works on index ranges (bisection, then at most ``MAX_KVSIG_PARTS`` slice
sums), so it runs on the event loop at any body size. The content passes (ids,
head part ids, the layout) ran in ``_cobs2_compute``: in a worker thread for a
body over 256 KiB, on the loop below that, as cobs2 always has.

``loose_depth`` in the event is ``max(exact_depth, loose depth)``: the deepest
node known at least loosely (an exact node is loosely known too). Nodes
``exact_depth + 1 .. loose_depth`` are the loose-only ones.

Part states, in order of precedence:

* ``exact`` / ``loose``: this session's own lineage knows the node (the
  request's own key, own conversation), exactly or in its loose form.
* ``other``: another key's request made the part's bytes cached. Read from the
  cache-observation prefix index (``app.cache_obs.index``), approximately; see
  ``_other_span``.
* ``new``: none of the above (including "cached, but it might be this key's").

Key holders never see ``other`` (ruling G-R11, superseding G-F9): it would
confirm that another key sent exactly this prefix, an oracle on other keys'
prompts. ``build_kvsig_pair`` adds ``parts_key``, the parts built as if no
other key had cached anything, whenever ``other`` appears; the key holder's
stream sends those (app/proxy/prompt_lens_scope.py), so neither a state nor a part
boundary or size shows where another key's span ended (``_cap`` cuts runs, so
an ``other`` run would move the cuts). The admin sees ``other``; ``parts_key``
never leaves the server.
"""

from __future__ import annotations

import bisect
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from app.cache_obs.canonical import _SEP, CHUNK_BYTES, Chain
from app.cache_obs.classify import bytes_to_tokens
from app.cache_obs.index import Match
from app.cobs2.ids import HeadPartIds, Ids
from app.cobs2.lineage import Observation

#: Most parts one event carries (sys and tools included).
MAX_KVSIG_PARTS = 64
#: The loose form's profile name, as the open spec names it.
PROFILE = "loose/1"
#: Role names a part's ``roles`` may carry; anything else counts as "unknown"
#: (a role is client text and is never echoed).
KNOWN_ROLES = frozenset({"system", "developer", "user", "assistant", "tool", "function"})


def _s8(id_hex: str | None) -> str:
    """The short id: 8 hex chars after the epoch byte."""
    return id_hex[2:10] if id_hex else ""


def _role(r: str) -> str:
    return r if r in KNOWN_ROLES else "unknown"


def _roles(names: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in names:
        k = _role(r)
        out[k] = out.get(k, 0) + 1
    return out


def _elem_end(chain: Chain, k: int) -> int:
    """One past the last byte of element ``k`` (its separator excluded)."""
    if k + 1 < len(chain.elem_starts):
        return chain.elem_starts[k + 1] - len(_SEP)
    return chain.total_bytes


def _other_span(
    chain: Chain | None, match_all: Match | None, match_own: Match | None
) -> tuple[int, int] | None:
    """``(lo, hi)``: an element ending at ``e`` with ``lo <= e <= hi`` was made
    cached by another key, or None when nothing can be said.

    Approximate, and only ever in the safe direction (a part another key
    cached may read ``new``; a part this key may have sent never reads
    ``other``):

    * The index matches whole ``CHUNK_BYTES`` chunks, so ``best_bytes`` is
      floored to a chunk boundary. Whatever ends at or before ``match_all``'s
      figure was sent, byte for byte, by some key: ``hi`` is exact.
    * This key's own identical prefix can run past ``match_own``'s figure by up
      to ``CHUNK_BYTES - 1`` bytes (its last, partial chunk never matches).
      ``lo`` adds a whole chunk, so an element inside that slack reads ``new``.
    * Without an own figure (no own match was made) nothing is ``other``.
    * Not caught: the shared index is one LRU; this key's own entries can be
      evicted while another key's requests keep the all-keys entries alive. A
      part then reads ``other`` though this key sent it too, but another key
      did send it, so the label is true, only not exclusive.
    * Element bytes are the cache-observation canonical form (``chain_of``),
      not the cobs2 canonical form the node ids hash; both walk the raw
      ``messages`` list in order, so raw index ``r`` is element
      ``r + msg_offset`` in both.
    """
    if chain is None or match_all is None or match_own is None:
        return None
    lo = match_own.best_bytes + CHUNK_BYTES
    hi = match_all.best_bytes
    return (lo, hi) if lo <= hi else None


def _part(
    p: str, lo: int, hi: int, chars: int, roles: dict[str, int], x: str, lx: str, state: str
) -> dict[str, Any]:
    return {
        "p": p,
        "from": lo,
        "to": hi,
        "chars": chars,
        "roles": roles,
        "exact8": x,
        "loose8": lx,
        "state": state,
    }


_Run = tuple[int, int, str]  # (first node, last node, state), inclusive


def _state_runs(
    first: int,
    last: int,
    loose_depth: int,
    ends: Callable[[int], int] | None,
    n_ends: int,
    span: tuple[int, int] | None,
) -> list[_Run]:
    """The states of nodes ``first..last`` as contiguous runs, in O(log n): the
    loose run is an index range, and a node's element end grows with its index
    (raw indices increase), so the ``other`` nodes are one range found by
    bisection. ``ends(i)`` is node ``i``'s element end, defined for
    ``i < n_ends``. At most four runs: loose, new, other, new."""
    runs: list[_Run] = []
    if first > last:
        return runs
    if loose_depth >= first:
        lt = min(loose_depth, last)
        runs.append((first, lt, "loose"))
        first = lt + 1
        if first > last:
            return runs
    o_lo = o_hi = -1
    top = min(last, n_ends - 1)
    if span is not None and ends is not None and first <= top:
        lo, hi = span
        a, b = first, top + 1  # first node whose end >= lo
        while a < b:
            mid = (a + b) // 2
            if ends(mid) >= lo:
                b = mid
            else:
                a = mid + 1
        c, d = a, top + 1  # first node whose end > hi
        while c < d:
            mid = (c + d) // 2
            if ends(mid) > hi:
                d = mid
            else:
                c = mid + 1
        if a < c:
            o_lo, o_hi = a, c - 1
    if o_lo < 0:
        runs.append((first, last, "new"))
        return runs
    if first < o_lo:
        runs.append((first, o_lo - 1, "new"))
    runs.append((o_lo, o_hi, "other"))
    if o_hi < last:
        runs.append((o_hi + 1, last, "new"))
    return runs


def _cap(runs: list[_Run], budget: int) -> list[_Run]:
    """Cut the runs into at most ``budget`` parts, never across a state. Fewer
    nodes than ``budget``: one part each. Else the newest ``budget // 2`` nodes
    stay single and the older ones share the rest of the budget, each run at
    least one part plus a share by length, each cut into near-equal parts
    (the oldest a node wider). Works on index ranges: no per-node object."""
    budget = max(1, budget)
    n = sum(b - a + 1 for a, b, _ in runs)
    if n <= budget:
        return [(i, i, st) for a, b, st in runs for i in range(a, b + 1)]
    singles = budget // 2
    cut = runs[-1][1] - singles + 1  # nodes >= cut stay single
    older: list[_Run] = []
    newest: list[_Run] = []
    for a, b, st in runs:
        if a < cut:
            older.append((a, min(b, cut - 1), st))
        if b >= cut:
            newest.extend((i, i, st) for i in range(max(a, cut), b + 1))
    room = budget - len(newest)
    if len(older) > room:
        # Unreachable with at most four runs; kept so the cap always holds.
        # Mixed runs merge as "new": never claiming a cache state.
        k = len(older) - room + 1
        older = [(older[0][0], older[k - 1][1], "new"), *older[k:]]
    lens = [b - a + 1 for a, b, _ in older]
    total = sum(lens)
    spare = room - len(older)
    shares = [spare * x / total for x in lens]
    alloc = [1 + int(x) for x in shares]
    left = room - sum(alloc)
    for j in sorted(range(len(older)), key=lambda j: shares[j] - int(shares[j]), reverse=True)[
        :left
    ]:
        alloc[j] += 1
    out: list[_Run] = []
    for (a, b, st), k in zip(older, alloc, strict=True):
        length = b - a + 1
        k = min(k, length)
        w, extra = divmod(length, k)
        pos = a
        for j in range(k):
            step = w + (1 if j < extra else 0)
            out.append((pos, pos + step - 1, st))
            pos += step
    out.extend(newest)
    return out


def build_kvsig_pair(**kw: Any) -> dict[str, Any]:
    """``build_kvsig(**kw)``, plus ``parts_key`` when a part reads ``other``: the
    parts of ``build_kvsig`` with no all-keys match, the key holder's view
    (module docstring). Without an ``other`` part the two builds are the same,
    so the second one runs only then."""
    ev = build_kvsig(**kw)
    if any(p.get("state") == "other" for p in ev.get("parts", ())):
        ev["parts_key"] = build_kvsig(**{**kw, "match_all": None})["parts"]
    return ev


def build_kvsig(
    *,
    ids: Ids | None,
    head_parts: HeadPartIds | None,
    obs: Observation | None,
    chain: Chain | None,
    match_all: Match | None,
    match_own: Match | None,
    prompt_tokens: int,
    raw_body_index: Sequence[int],
    parts_chars: Sequence[int],
    sys_chars: int,
    tools_chars: int,
    roles: Sequence[str],
    reason: str | None,
    raw_head_index: Sequence[int] | None = None,
    head_roles: Sequence[str] = (),
) -> dict[str, Any]:
    """The kvsig event body (the caller adds ``type``, ``req_id``,
    ``token_id``, ``ts``, ``session_id`` and ``turn``).

    ``raw_body_index[i]`` is body node ``i``'s index in the raw ``messages``
    list; ``raw_head_index`` the head's (default: every raw index the chain
    has that is not a body index). Sizes and roles come from ``body_layout``.
    Unavailable (``reason`` set, or no ids) is ``{"available": False,
    "reason": ..., "parts": []}``."""
    if reason is not None or ids is None:
        return {"available": False, "reason": reason or "error", "parts": []}

    n = len(ids.nodes)
    exact_depth = obs.exact_depth if obs is not None else -1
    # An exact node is loosely known too (the raw lineage figure can be lower,
    # e.g. without loose ids); the event reports the deepest known either way.
    loose_depth = max(exact_depth, obs.loose_depth) if obs is not None else -1
    span = _other_span(chain, match_all, match_own)

    def covered(elems: Sequence[int]) -> bool:
        if span is None or chain is None or not elems:
            return False
        n_elems = len(chain.elem_starts)
        if any(k < 0 or k >= n_elems for k in elems):
            return False
        end = max(_elem_end(chain, k) for k in elems)
        return span[0] <= end <= span[1]

    def head_state(elems: Sequence[int]) -> str:
        if exact_depth >= 0:
            return "exact"
        if loose_depth >= 0:
            return "loose"
        return "other" if covered(elems) else "new"

    parts: list[dict[str, Any]] = []
    off = chain.msg_offset if chain is not None else 0
    if sys_chars > 0:
        if raw_head_index is None:
            body_set = set(raw_body_index)
            n_raw = len(chain.elem_starts) - off if chain is not None else 0
            raw_head_index = [r for r in range(n_raw) if r not in body_set]
        parts.append(
            _part(
                "sys",
                -1,
                -1,
                sys_chars,
                _roles(head_roles),
                _s8(head_parts.sys if head_parts else None),
                _s8(head_parts.loose_sys if head_parts else None),
                head_state([r + off for r in raw_head_index]),
            )
        )
    if tools_chars > 0:
        parts.append(
            _part(
                "tools",
                -1,
                -1,
                tools_chars,
                {},
                _s8(head_parts.tools if head_parts else None),
                _s8(head_parts.loose_tools if head_parts else None),
                head_state([0] if off == 1 else []),
            )
        )

    n_chars = len(parts_chars)
    n_roles = len(roles)

    def node_part(a: int, b: int, state: str) -> dict[str, Any]:
        # Slices sum and count at C speed; a run's ids are its last node's
        # (a node id fixes its whole prefix).
        rc = Counter(roles[a : min(b + 1, n_roles)])
        rs: dict[str, int] = {}
        for k, v in rc.items():
            r = _role(k)
            rs[r] = rs.get(r, 0) + v
        return _part(
            "nodes",
            a,
            b,
            sum(parts_chars[a : min(b + 1, n_chars)]),
            rs,
            _s8(ids.nodes[b]),
            _s8(ids.loose_nodes[b] if b < len(ids.loose_nodes) else None),
            state,
        )

    top = min(exact_depth, n - 1)
    if top >= 0:
        parts.append(node_part(0, top, "exact"))
    ends: Callable[[int], int] | None = None
    n_ends = 0
    if chain is not None:
        n_elems = len(chain.elem_starts)
        # Nodes whose element exists in the chain (a prefix: raw indices grow).
        n_ends = bisect.bisect_left(raw_body_index, n_elems - off)
        n_ends = min(n_ends, n)

        ch = chain

        def _end(i: int) -> int:
            return _elem_end(ch, raw_body_index[i] + off)

        ends = _end

    runs = _state_runs(top + 1, n - 1, loose_depth, ends, n_ends, span)
    for a, b, st in _cap(runs, MAX_KVSIG_PARTS - len(parts)):
        parts.append(node_part(a, b, st))

    def reusable(m: Match | None) -> int | None:
        if m is None:
            return None
        return bytes_to_tokens(m.best_bytes, m.total_bytes, prompt_tokens)

    return {
        "available": True,
        "profile": PROFILE,
        "head8": _s8(ids.head),
        "conv8": _s8(obs.conv) if obs is not None and obs.conv else None,
        "edge": obs.edge if obs is not None else None,
        "parent_request_id": obs.parent_req if obs is not None else None,
        "exact_depth": exact_depth,
        "loose_depth": loose_depth,
        "n_nodes": n,
        "reusable_own": reusable(match_own),
        "reusable_any": reusable(match_all),
        "parts": parts,
    }
