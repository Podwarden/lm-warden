"""Session forest assembly: request_history rows → trees (spec 2026-10-05 §4).

Pure functions over plain dict rows, so they are tested without a database.
Turn wire format: [t, ctx, gen, cached, tools_in, tools_out_families, finish, merged_n],
with t = request start (finished_at - duration_s) relative to t0.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

SESSION_GAP_S = 1200.0  # heuristic chain: next turn within 20 min
SPELL_SPAN_S = 3 * 3600.0
SPELL_GAP_S = 3600.0
LOD_CAP = 2000  # turns of a session sent one by one; later turns go in fixed blocks
LOD_BLOCK = 8
MAX_OPEN_CHAINS = 64


def tree_id(token_id: str, kind: str, key: str) -> str:
    return f"{token_id}:{kind}:{key}"


def _start(r: dict[str, Any] | _Row) -> float:
    return float(r["finished_at"]) - float(r.get("duration_s") or 0.0)


def _turn(r: dict[str, Any], t0: float) -> list[Any]:
    tout = [x[0] for x in (r.get("tools_out") or []) if isinstance(x, list | tuple) and x]
    return [
        round(_start(r) - t0, 1),
        int(r.get("prompt_tokens") or 0),
        int(r.get("completion_tokens") or 0),
        r.get("cached_tokens"),
        [list(x) if isinstance(x, tuple) else x for x in r.get("tools_in") or []],
        tout,
        r.get("finish_reason"),
        1,
    ]


def merge_turns(
    turns: list[list[Any]], cap: int = LOD_CAP, block: int = LOD_BLOCK
) -> list[list[Any]]:
    """Prefix-stable level of detail: the first ``cap`` turns go one by one, later ones in fixed blocks of ``block``.

    Grouping depends only on a turn's index, never on the session's length, so a new turn never regroups the turns
    a viewer already sees grown (the forest's live view must not reshape a tree when it grows). The trailing,
    incomplete block is sent one by one until it fills.
    """
    if len(turns) <= cap:
        return turns
    out = list(turns[:cap])
    rest = turns[cap:]
    full = len(rest) - len(rest) % block
    for i in range(0, full, block):
        g = rest[i : i + block]
        tin = [x for t in g for x in t[4]][:6]
        tout = [x for t in g for x in t[5]][:6]
        cached = [t[3] for t in g if t[3] is not None]
        out.append(
            [
                g[-1][0],
                max(t[1] for t in g),
                sum(t[2] for t in g),
                (sum(cached) // len(cached)) if cached else None,
                tin,
                tout,
                g[-1][6],
                sum(t[7] for t in g),
            ]
        )
    out.extend(rest[full:])
    return out


def merged_index(i: int, n: int, cap: int = LOD_CAP, block: int = LOD_BLOCK) -> int:
    """Index in ``merge_turns`` output of the session's ``i``-th turn (of ``n``)."""
    if i < cap:
        return i
    full = (n - cap) - (n - cap) % block
    j = i - cap
    return cap + (j // block if j < full else full // block + (j - full))


def _sessions_for_token(rows: Sequence[dict[str, Any] | _Row]) -> list[dict[str, Any]]:
    """Sessions of one token: by session_key, else the prompt-chain heuristic."""
    keyed: dict[str, list[dict[str, Any] | _Row]] = defaultdict(list)
    loose: list[dict[str, Any] | _Row] = []
    for r in rows:
        (keyed[r["session_key"]] if r.get("session_key") else loose).append(r)
    sessions = [
        {
            "id": k,
            "rows": v,
            "parent": next(
                (x.get("parent_session_key") for x in v if x.get("parent_session_key")), None
            ),
        }
        for k, v in keyed.items()
    ]
    open_: list[dict[str, Any]] = []
    used = set(keyed)
    for r in sorted(loose, key=lambda r: (_start(r), str(r.get("id") or ""))):
        t, p = _start(r), int(r.get("prompt_tokens") or 0)
        best = None
        for s in open_:
            last = s["rows"][-1]
            lp = int(last.get("prompt_tokens") or 0)
            if (
                t - _start(last) < SESSION_GAP_S
                and lp * 0.85 <= p <= lp + 90_000
                and (best is None or lp > int(best["rows"][-1].get("prompt_tokens") or 0))
            ):
                best = s
        if best is None:
            # Two unkeyed chains can start in the same 0.1 s (parallel agent calls): the first
            # row's id keeps them apart, and is stable for as long as that row is in the window.
            hid = f"h:{round(t, 1)}:{str(r.get('id') or '')[:8]}"
            while hid in used:
                hid += "+"
            used.add(hid)
            best = {"id": hid, "rows": [], "parent": None}
            open_.append(best)
            sessions.append(best)
        best["rows"].append(r)
        open_ = [s for s in open_ if t - _start(s["rows"][-1]) < SESSION_GAP_S]
        if len(open_) > MAX_OPEN_CHAINS:
            open_.sort(key=lambda s: _start(s["rows"][-1]))
            open_ = open_[-MAX_OPEN_CHAINS:]
    for s in sessions:
        s["rows"].sort(key=_start)
        s["start"] = _start(s["rows"][0])
    return sessions


def _is_flower(s: dict[str, Any]) -> bool:
    if len(s["rows"]) != 1 or s.get("children"):
        return False
    r = s["rows"][0]
    return not (r.get("tools_out") or []) and r.get("finish_reason") != "tool_calls"


def _wire_session(s: dict[str, Any], by_id: dict[str, dict[str, Any]], t0: float) -> dict[str, Any]:
    turns = merge_turns([_turn(r, t0) for r in s["rows"]])
    children = []
    for c in s.get("children", []):
        n_rows = len(s["rows"])
        at = sum(1 for r in s["rows"] if _start(r) <= c["start"]) - 1
        at_merged = merged_index(max(0, at), n_rows)
        w = _wire_session(c, by_id, t0)
        w["at"] = max(0, min(at_merged, len(turns) - 1))
        children.append(w)
    return {
        "id": s["id"],
        "variant": s["rows"][0].get("variant_id"),
        "turns": turns,
        "children": children,
    }


def _descendants(s: dict[str, Any]) -> list[dict[str, Any]]:
    return [s] + [d for c in s.get("children", []) for d in _descendants(c)]


def _break_cycles(sessions: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> None:
    for s in sessions:
        path: list[dict[str, Any]] = []
        cur: dict[str, Any] | None = s
        while cur is not None and all(cur is not x for x in path):
            path.append(cur)
            cur = by_id.get(cur["parent"]) if cur["parent"] else None
        if cur is not None:
            cyc = path[next(i for i, x in enumerate(path) if x is cur) :]
            min(cyc, key=lambda x: x["start"])["parent"] = None


def _traits(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    sessions = [d for s in sessions for d in _descendants(s)]
    rows = [r for s in sessions for r in s["rows"]]
    firsts = [int(s["rows"][0].get("prompt_tokens") or 0) for s in sessions]
    ctx = sum(int(r.get("prompt_tokens") or 0) for r in rows)
    gen = sum(int(r.get("completion_tokens") or 0) for r in rows)
    later = [
        r
        for s in sessions
        for r in s["rows"][1:]
        if r.get("cached_tokens") is not None and r.get("prompt_tokens")
    ]
    thrash = (
        (sum(1 for r in later if r["cached_tokens"] / r["prompt_tokens"] < 0.3) / len(later))
        if later
        else 0.0
    )
    tins = [x for r in rows for x in (r.get("tools_in") or []) if isinstance(x, list | tuple)]
    fail = (sum(1 for x in tins if len(x) > 2 and x[2] is True) / len(tins)) if tins else 0.0
    kids = sum(len(s.get("children", [])) for s in sessions)  # sessions is already flattened
    return {
        "sys0": round(sum(firsts) / max(1, len(firsts))),
        "ctx_gen": round(ctx / max(1, gen), 1),
        "mean_gen": round(gen / max(1, len(rows))),
        "thrash": round(thrash, 3),
        "fail": round(fail, 3),
        "fanout": round(kids / max(1, len(sessions)), 3),
    }


def _all_rows(s: dict[str, Any]) -> list[dict[str, Any]]:
    own: list[dict[str, Any]] = s["rows"]
    return own + [r for c in s.get("children", []) for r in _all_rows(c)]


def _group_trees(token: str, roots: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Trees of one token: a batch per ``X-Batch-Id``, otherwise spells of sessions.

    Grouping depends only on session start times, which never change, so a tree id, once issued, keeps its
    sessions: a new session only starts a new spell or joins the open one. (A busy day is no longer regrouped
    into one "day" tree — the live view must never swap trees the viewer already sees grown; spec §10.1.)
    A spell ends when the next session starts more than SPELL_GAP_S after the previous session's start, or more
    than SPELL_SPAN_S after the spell's first.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rest: list[dict[str, Any]] = []
    for s in roots:
        b = next((r.get("batch_id") for r in s["rows"] if r.get("batch_id")), None)
        (groups[tree_id(token, "batch", b)] if b else rest).append(s)
    cur: list[dict[str, Any]] = []
    first = prev = 0.0
    for s in sorted(rest, key=lambda x: x["start"]):
        if cur and (s["start"] - first > SPELL_SPAN_S or s["start"] - prev > SPELL_GAP_S):
            groups[tree_id(token, "spell", str(round(first, 1)))] = cur
            cur = []
        if not cur:
            first = s["start"]
        cur.append(s)
        prev = s["start"]
    if cur:
        groups[tree_id(token, "spell", str(round(first, 1)))] = cur
    return list(groups.items())


def token_of(r: dict[str, Any]) -> str:
    """The forest's per-token partition key (rows without a token share "-")."""
    return str(r.get("token_id") or "-")


def build_forest(rows: list[dict[str, Any]], *, t0: float) -> dict[str, Any]:
    trees, flowers = [], []
    for part in build_parts(rows, t0=t0).values():
        trees += part["trees"]
        flowers += part["flowers"]
    trees.sort(key=lambda t: t["start"])
    flowers.sort()
    return {"trees": trees, "flowers": flowers}


def build_parts(rows: list[dict[str, Any]], *, t0: float) -> dict[str, dict[str, list[Any]]]:
    """The forest of each token separately: ``{token: {"trees": [...], "flowers": [...]}}``.

    A token's trees and flowers depend only on that token's rows, which is what lets the route
    rebuild just the tokens that changed. Built through ``TokenForest`` from scratch, so an
    incremental forest (rows added and pruned over time) and this one are the same code.
    """
    by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_token[token_of(r)].append(r)
    out: dict[str, dict[str, list[Any]]] = {}
    for token, trows in by_token.items():
        tf = TokenForest(token, t0)
        tf.add(trows)
        b = tf.build()
        out[token] = {"trees": [t for t, _ in b["trees"]], "flowers": b["flowers"]}
    return out


def _tree_sig(group: list[dict[str, Any]]) -> tuple[Any, ...]:
    """What a tree's wire form depends on, besides t0: its sessions (with their subagents), each by id, row count and
    first and last row. Rows never change once written, so an equal signature is an equal tree."""
    return tuple(
        (d["id"], len(d["rows"]), d["rows"][0].get("id"), d["rows"][-1].get("id"))
        for s in group
        for d in _descendants(s)
    )


def _build_tree(
    token: str, tid: str, group: list[dict[str, Any]], by_id: dict[str, dict[str, Any]], t0: float
) -> dict[str, Any]:
    """One tree's wire form (the expensive part: every turn of its sessions)."""
    end = max(_start(r) for s in group for r in _all_rows(s))
    last = max(float(r["finished_at"]) for s in group for r in _all_rows(s))
    return {
        "id": tid,
        "key": group[0]["rows"][0].get("token_name") or token,
        "start": round(group[0]["start"] - t0, 1),
        "end": round(end - t0, 1),
        "last": round(last - t0, 1),
        "last_at": last,
        "traits": _traits(group),
        "sessions": [_wire_session(s, by_id, t0) for s in group],
    }


_ROW_FIELDS = (
    "id", "finished_at", "duration_s", "token_name", "model_id", "variant_id", "prompt_tokens",
    "completion_tokens", "cached_tokens", "finish_reason", "session_key", "parent_session_key",
    "batch_id", "tools_in", "tools_out", "token_id",
)  # fmt: skip
_INTERN = (
    "token_name",
    "model_id",
    "variant_id",
    "finish_reason",
    "session_key",
    "parent_session_key",
    "batch_id",
)


class _Row:
    """A held row, slim (review N3: a 7 d forest holds every row of its window): the fields the assembly reads,
    repeated strings interned, empty lists shared. Read like the dict rows (``get``, ``[]``)."""

    __slots__ = _ROW_FIELDS

    # Annotation-only (no class attribute, so no clash with the slots): lets the type checker see the slot fields.
    id: Any
    finished_at: Any
    duration_s: Any
    token_name: Any
    model_id: Any
    variant_id: Any
    prompt_tokens: Any
    completion_tokens: Any
    cached_tokens: Any
    finish_reason: Any
    session_key: Any
    parent_session_key: Any
    batch_id: Any
    tools_in: Any
    tools_out: Any
    token_id: Any

    def __init__(self, d: dict[str, Any]) -> None:
        for f in _ROW_FIELDS:
            v = d.get(f)
            if f in _INTERN and isinstance(v, str):
                v = sys.intern(v)
            elif f in ("tools_in", "tools_out"):
                # tuples of tuples of atoms: the garbage collector stops tracking them (a held 7 d window would
                # otherwise add several tracked objects per row to every full collection)
                v = tuple(tuple(x) if isinstance(x, list | tuple) else x for x in v) if v else ()
            setattr(self, f, v)

    def get(self, k: str, default: Any = None) -> Any:
        v = getattr(self, k, None)
        return default if v is None else v

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)


class TokenForest:
    """One token's forest, kept up to date row by row (review N3).

    Rows are added as they are written and pruned as they leave the window; ``build`` regroups the sessions (cheap: one
    entry per session) and rebuilds only the trees whose sessions changed, reusing every other tree as it was built
    (and whatever the caller made of it: ``build(make=...)`` caches ``make(tree)``, e.g. the serialized JSON).

    The result equals ``build_parts`` over the rows currently held: sessions keep the order of their first row by
    ``finished_at`` (as rows loaded ``ORDER BY finished_at`` give them), and a session's rows are in start order.
    Rows without a session key (the prompt-chain heuristic) are regrouped from scratch on every build; they are few.
    """

    def __init__(self, token: str, t0: float) -> None:
        self.token = token
        self.t0 = t0
        #: session key -> its rows (start order) and the finished_at / parent of its earliest-finished row
        self.keyed: dict[str, dict[str, Any]] = {}
        self.loose: list[dict[str, Any] | _Row] = []
        self._built: dict[str, tuple[tuple[Any, ...], Any]] = {}
        self.variants: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))

    def __len__(self) -> int:
        return sum(len(v["rows"]) for v in self.keyed.values()) + len(self.loose)

    def add(self, rows: Sequence[dict[str, Any] | _Row]) -> None:
        for r in sorted(
            (x if isinstance(x, _Row) else _Row(x) for x in rows),
            key=lambda x: float(x["finished_at"]),
        ):
            if r.get("variant_id"):
                self.variants[r["variant_id"]][r.get("model_id")] += 1
            k = r.get("session_key")
            if not k:
                self.loose.append(r)
                continue
            fin = float(r["finished_at"])
            s = self.keyed.get(k)
            if s is None:
                s = self.keyed[k] = {"rows": [], "fmin": fin, "parent": None, "pfin": float("inf")}
            st = s["rows"]
            # start order; a row finishing later goes after equal starts (a stable sort of finished_at order)
            t = _start(r)
            lo, hi = 0, len(st)
            if not st or _start(st[-1]) <= t:
                lo = hi
            while lo < hi:
                mid = (lo + hi) // 2
                if _start(st[mid]) <= t:
                    lo = mid + 1
                else:
                    hi = mid
            st.insert(lo, r)
            s["fmin"] = min(s["fmin"], fin)
            if r.get("parent_session_key") and fin < s["pfin"]:
                s["parent"], s["pfin"] = r["parent_session_key"], fin

    def finished_times(self) -> list[float]:
        """Every held row's finish time, ascending."""
        out = [float(r["finished_at"]) for s in self.keyed.values() for r in s["rows"]]
        out += [float(r["finished_at"]) for r in self.loose]
        out.sort()
        return out

    def prune(self, ws: float) -> bool:
        """Drops the rows that finished before ``ws``; True when any was dropped."""
        dropped = False
        for k in [k for k, s in self.keyed.items() if s["fmin"] < ws]:
            s = self.keyed[k]
            keep = [r for r in s["rows"] if float(r["finished_at"]) >= ws]
            for r in s["rows"]:
                if float(r["finished_at"]) < ws and r.get("variant_id"):
                    self._unvariant(r)
            dropped = True
            if not keep:
                del self.keyed[k]
                continue
            s["rows"] = keep
            s["fmin"] = min(float(r["finished_at"]) for r in keep)
            withp = [r for r in keep if r.get("parent_session_key")]
            first = min(withp, key=lambda r: float(r["finished_at"])) if withp else None
            s["parent"] = first["parent_session_key"] if first else None
            s["pfin"] = float(first["finished_at"]) if first else float("inf")
        if any(float(r["finished_at"]) < ws for r in self.loose):
            for r in self.loose:
                if float(r["finished_at"]) < ws and r.get("variant_id"):
                    self._unvariant(r)
            self.loose = [r for r in self.loose if float(r["finished_at"]) >= ws]
            dropped = True
        return dropped

    def _unvariant(self, r: dict[str, Any]) -> None:
        v = self.variants[r["variant_id"]]
        v[r.get("model_id")] -= 1
        if v[r.get("model_id")] <= 0:
            del v[r.get("model_id")]
        if not v:
            del self.variants[r["variant_id"]]

    def variant_models(self) -> dict[str, str]:
        """variant id -> its model (the last one seen, as the route's map did)."""
        return {v: next(reversed(m)) for v, m in self.variants.items() if m}

    def sessions(self) -> list[dict[str, Any]]:
        order = sorted(
            self.keyed.items(), key=lambda kv: (kv[1]["fmin"], kv[1]["rows"][0].get("id") or "")
        )
        out = [
            {"id": k, "rows": s["rows"], "parent": s["parent"], "start": _start(s["rows"][0])}
            for k, s in order
        ]
        return out + _sessions_for_token(self.loose) if self.loose else out

    def build(self, make: Any = None) -> dict[str, list[Any]]:
        """``{"trees": [(tree, made), ...], "flowers": [...]}``; ``made`` is ``make(tree)``, cached. With a ``make``, a
        reused tree comes back as ``(None, made)``: only ``made`` is kept between builds."""
        t0, token = self.t0, self.token
        sessions = self.sessions()
        by_id = {s["id"]: s for s in sessions}
        _break_cycles(sessions, by_id)
        roots, flowers = [], []
        for s in sessions:
            par = by_id.get(s["parent"]) if s["parent"] else None
            if par is not None and par is not s:
                par.setdefault("children", []).append(s)
        for s in sessions:
            par = by_id.get(s["parent"]) if s["parent"] else None
            if par is not None and par is not s:
                continue
            if _is_flower(s):
                r = s["rows"][0]
                flowers.append(
                    [
                        round(_start(r) - t0, 1),
                        int(r.get("prompt_tokens") or 0),
                        int(r.get("completion_tokens") or 0),
                    ]
                )
            else:
                roots.append(s)
        trees: list[tuple[dict[str, Any], Any]] = []
        built: dict[str, tuple[tuple[Any, ...], Any]] = {}
        for tid, group in _group_trees(token, roots):
            group.sort(key=lambda s: s["start"])
            sig = _tree_sig(group)
            hit = self._built.get(tid)
            if hit is not None and hit[0] == sig:
                tree, made = hit[1]
            else:
                tree = _build_tree(token, tid, group, by_id, t0)
                made = make(tree) if make is not None else None
            # with a `make`, only what it made is kept (the wire dict of every tree would double the memory)
            built[tid] = (sig, (None if make is not None else tree, made))
            trees.append((tree, made))
        self._built = built
        return {"trees": trees, "flowers": flowers}


def _p75(xs: list[float]) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return float(xs[min(len(xs) - 1, int(0.75 * (len(xs) - 1) + 0.5))])


def model_throughput(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    pre: dict[str, list[float]] = defaultdict(list)
    dec: dict[str, list[float]] = defaultdict(list)
    n: dict[str, int] = defaultdict(int)
    for r in rows:
        v = r.get("variant_id")
        ttft, dur = r.get("ttft_s"), float(r.get("duration_s") or 0)
        p, c, g = (
            int(r.get("prompt_tokens") or 0),
            int(r.get("cached_tokens") or 0),
            int(r.get("completion_tokens") or 0),
        )
        if not v or not ttft or ttft <= 0:
            continue
        used = False
        if p - c > 0:
            pre[v].append((p - c) / ttft)
            used = True
        if g > 1 and dur > ttft:
            dec[v].append(g / (dur - ttft))
            used = True
        n[v] += 1 if used else 0
    # n: requests that contributed a sample (to either rate), so a client can weigh a p75 of 3.
    return {
        v: {"prefill_tps": _p75(pre.get(v, [])), "decode_tps": _p75(dec.get(v, [])), "n": n[v]}
        for v in set(pre) | set(dec)
    }
