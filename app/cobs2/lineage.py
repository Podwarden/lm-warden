"""Session lineage and edge detection (cobs2 plan 3, task 4).

Pure and in-memory. Only ids and digests are kept. A session's last reply is
reduced to a keyed bottom-k sketch (64 hash values) computed from text passed
in; the text is never stored. Every lookup is scoped by ``(model_id, scope)``
so one scope never sees another's nodes. ``token_scope`` is stored raw, so
callers must pass a digest of the token id, never the token itself.
"""

from __future__ import annotations

import hashlib
import struct
import time
from collections import OrderedDict
from collections.abc import Callable, Collection
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from app.cobs2.ids import Ids

Edge = Literal["new", "continue", "fork", "compacted_from", "subagent"]

#: Version 1. Matched case-insensitively against the loose last user text.
#: The Cline phrase is from memory of its summarize prompt (unverified); it is
#: deliberately a long specific phrase rather than a bare word like "condense".
COMPACTION_FINGERPRINTS: tuple[str, ...] = (
    # Claude Code
    "summary of the conversation so far",
    "create a detailed summary of the conversation",
    # OpenCode
    "provide a detailed prompt for continuing our conversation",
    "summarize the conversation so far",
    # Cline
    "the user has explicitly asked you to create a detailed summary",
)

_SKETCH_K = 64
_SHINGLE = 5
_MAX_SHINGLES = 2000
_OVERLAP_MIN = 0.5
_HEAD_WINDOW_S = 600.0
_MAX_CHILDREN = 16


@lru_cache(maxsize=64)
def _sketch_key(scope: bytes) -> bytes:
    return hashlib.blake2b(b"cobs2-sketch" + scope, digest_size=32).digest()


def sketch(text: str, scope: bytes) -> bytes | None:
    """Keyed bottom-k sketch: the 64 smallest 64-bit hashes of 5-word shingles.

    At most the first 2000 shingles (a contiguous prefix: positional sampling
    breaks the estimate when the two texts differ in length). Returns packed sorted uint64s, or
    None for empty text. The text itself is never retained.
    """
    words = text.split()
    if not words:
        return None
    if len(words) <= _SHINGLE:
        picks = [" ".join(words)]
    else:
        total = len(words) - _SHINGLE + 1
        picks = [" ".join(words[i : i + _SHINGLE]) for i in range(min(total, _MAX_SHINGLES))]
    key = _sketch_key(scope)
    vals = {
        int.from_bytes(
            hashlib.blake2b(w.encode("utf-8", "replace"), digest_size=8, key=key).digest(), "little"
        )
        for w in picks
    }
    low = sorted(vals)[:_SKETCH_K]
    return struct.pack(f"<{len(low)}Q", *low)


def strip_analysis(text: str) -> str:
    """Drop a leading ``<analysis>...</analysis>`` block (Claude Code's compaction
    reply opens with one and strips it from the post-compaction message)."""
    t = text.lstrip()
    if t.startswith("<analysis>"):
        j = t.find("</analysis>")
        if j >= 0:
            return t[j + len("</analysis>") :]
    return text


def _unpack(b: bytes) -> set[int]:
    return set(struct.unpack(f"<{len(b) // 8}Q", b))


def similarity(a: bytes, b: bytes) -> float:
    """Jaccard estimate from the union of two bottom-k sketches."""
    sa, sb = _unpack(a), _unpack(b)
    union = sorted(sa | sb)[:_SKETCH_K]
    if not union:
        return 0.0
    return sum(1 for x in union if x in sa and x in sb) / len(union)


@dataclass(frozen=True)
class Observation:
    edge: Edge
    parent: str | None
    #: True when the loose chain matched deeper than the exact chain (the
    #: harness rewrote history, e.g. dropped reasoning): Task 6's harness_rewrite.
    loose_match_beyond_exact: bool = False
    #: The session identity used (pass it back to ``remember_reply``).
    session: str = ""
    #: The request id that produced ``parent`` (when observe was given ids).
    parent_req: str | None = None
    #: The deepest body index whose EXACT node this session knows (-1: none).
    exact_depth: int = -1
    #: The conversation this request belongs to: the parent's when it continues
    #: or forks one (so a lane survives a harness rewriting its first message),
    #: else the conv id it was observed with.
    conv: str | None = None


@dataclass
class _Node:
    parent: str | None
    loose_head: str
    ts: float
    reply_msg: str | None = None
    #: The request that produced this node (an interior node inherits the
    #: request of the node before it: the turn it extends).
    req: str | None = None
    #: The conversation the node belongs to (see Observation.conv).
    conv: str | None = None


@dataclass
class _Loose:
    exact: str
    ts: float
    children: dict[str, None]


@dataclass
class _Tip:
    loose_head: str
    node: str
    ts: float
    sig: bytes | None = None
    compaction: bool = False
    #: The sketch of the reply WITH its leading <analysis> block, kept only when
    #: it had one: a harness that keeps the analysis in the post-compaction
    #: message matches this one instead of ``sig``.
    sig_raw: bytes | None = None


@dataclass
class _Reply:
    node: str
    ts: float


@dataclass
class _Session:
    ts: float
    tips: list[_Tip]


_Key = tuple[Any, ...]
_MAX_TIPS = 4


def fallback_session(token_scope: str | None, head: str) -> str:
    """Session identity when the harness sends none: digest of (token, loose head)."""
    h = hashlib.sha256(b"cobs2-session|" + (token_scope or "").encode() + b"|" + head.encode())
    return "f:" + h.hexdigest()[:32]


class Lineage:
    """Lineage edges are per session; nothing is continued across sessions."""

    def __init__(
        self,
        max_nodes: int = 50_000,
        ttl_s: float = 21_600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max_nodes
        self._ttl = ttl_s
        self._clock = clock
        # (model, scope, session, node)
        self._nodes: OrderedDict[_Key, _Node] = OrderedDict()
        # (model, scope, session, loose node): latest exact node + loose children
        self._loose: OrderedDict[_Key, _Loose] = OrderedDict()
        self._sessions: OrderedDict[_Key, _Session] = OrderedDict()
        self._heads: OrderedDict[_Key, float] = OrderedDict()
        # (model, scope, session, reply msg id) -> the node whose reply it was: a
        # later request carrying that reply continues that node, whatever the
        # harness rewrote before it.
        self._replies: OrderedDict[_Key, _Reply] = OrderedDict()

    # -- public ----------------------------------------------------------
    @staticmethod
    def is_compaction_request(last_user_text_loose: str) -> bool:
        t = last_user_text_loose.lower()
        return any(f in t for f in COMPACTION_FINGERPRINTS)

    def remember_reply(
        self,
        *,
        model_id: str,
        scope: bytes,
        session: str,
        node: str,
        reply_ids: Ids,
        reply_text_loose: str | None = None,
    ) -> None:
        now = self._clock()
        entry = self._nodes.get((model_id, scope, session, node))
        if entry is None or now - entry.ts > self._ttl:
            return
        if reply_ids.msgs:
            entry.reply_msg = reply_ids.msgs[0]
            rk = (model_id, scope, session, reply_ids.msgs[0])
            self._replies[rk] = _Reply(node, now)
            self._replies.move_to_end(rk)
            self._evict()
        sess = self._sessions.get((model_id, scope, session))
        if sess is None or reply_text_loose is None:
            return
        for tip in sess.tips:
            if tip.loose_head == entry.loose_head and tip.node == node:
                stripped = strip_analysis(reply_text_loose)
                tip.sig = sketch(stripped, scope)
                tip.sig_raw = (
                    sketch(reply_text_loose, scope) if stripped != reply_text_loose else None
                )

    def observe(
        self,
        *,
        model_id: str,
        scope: bytes,
        session_key: str | None,
        parent_session_key: str | None,
        ids: Ids,
        last_user_text_loose: str,
        first_body_text_loose: str | Callable[[], str],
        token_scope: str | None = None,
        inflight_heads: Collection[str] | None = None,
        request_id: str | None = None,
        conv: str | None = None,
    ) -> Observation:
        now = self._clock()
        self._gc(now)
        # The LOOSE head, like the tips and the replica key: a changed reminder
        # or date line in the system prompt keeps the session.
        skey = session_key or fallback_session(token_scope, ids.loose_head)
        n = len(ids.nodes)
        if n == 0:
            return Observation("new", None, False, skey)
        ms = (model_id, scope, skey)

        d, hid, beyond, k = self._find(ms, ids)
        edge: Edge
        parent: str | None
        pms = ms
        if d >= 0:
            assert hid is not None
            owner = self._nodes.get((*ms, hid))
            if d + 1 >= n:
                # Fully known (identical retry): report the node's own parent.
                edge, parent = "continue", (owner.parent if owner is not None else None)
            else:
                lo = self._loose.get((*ms, ids.loose_nodes[d]))
                if lo is not None and lo.children and ids.loose_nodes[d + 1] not in lo.children:
                    edge = "fork"
                else:
                    edge = "continue"
                parent = hid
        elif (anchor := self._reply_anchor(ms, now, ids)) is not None:
            # No known prefix, but the body carries a reply this session saw:
            # the harness rewrote what came before it (e.g. the first message).
            edge, parent, beyond = "continue", anchor, True
        else:
            beyond = False
            edge, parent = self._classify_collapsed(
                ms,
                now,
                session_key is None,
                parent_session_key,
                ids,
                first_body_text_loose,
                inflight_heads,
            )
            if edge == "subagent" and parent_session_key is not None:
                pms = (model_id, scope, parent_session_key)
        pnode = self._nodes.get((*pms, parent)) if parent is not None else None
        parent_req = pnode.req if pnode is not None else None
        out_conv = conv
        if edge in ("continue", "fork") and pnode is not None and pnode.conv is not None:
            out_conv = pnode.conv
        self._record(
            ms, now, ids, self.is_compaction_request(last_user_text_loose), request_id, out_conv
        )
        return Observation(edge, parent, beyond, skey, parent_req, k, out_conv)

    def _reply_anchor(self, ms: _Key, now: float, ids: Ids) -> str | None:
        """The node whose reply is the deepest message of this body that this
        session saw as a reply (within the TTL), or None."""
        for i in range(len(ids.msgs) - 1, 0, -1):
            r = self._replies.get((*ms, ids.msgs[i]))
            if r is not None and now - r.ts <= self._ttl and (*ms, r.node) in self._nodes:
                return r.node
        return None

    # -- internals -------------------------------------------------------
    def _find(self, ms: _Key, ids: Ids) -> tuple[int, str | None, bool, int]:
        """Deepest known index, its exact node id, whether loose beat exact, and
        the deepest index whose exact node is known."""
        n = len(ids.nodes)
        kl = -1
        if len(ids.loose_nodes) == n:
            for i in range(n - 1, -1, -1):
                if (*ms, ids.loose_nodes[i]) in self._loose:
                    kl = i
                    break
        k = -1
        for i in range(n - 1, -1, -1):
            if (*ms, ids.nodes[i]) in self._nodes:
                k = i
                break
        if k >= 0 and k >= kl:
            return k, ids.nodes[k], False, k
        if kl >= 0:
            exact = self._loose[(*ms, ids.loose_nodes[kl])].exact
            # Beyond only when the exact id really differs: an unchanged
            # resend whose exact nodes the LRU or TTL dropped (while the loose
            # entries survived) matches here too, and is no rewrite.
            return kl, exact, ids.nodes[kl] != exact, k
        return -1, None, False, k

    def _classify_collapsed(
        self,
        ms: _Key,
        now: float,
        windowed: bool,
        parent_session_key: str | None,
        ids: Ids,
        first_text: str | Callable[[], str],
        inflight_heads: Collection[str] | None,
    ) -> tuple[Edge, str | None]:
        model_id, scope, _skey = ms
        # Rule 3: collapsed body in a known session; only the tips with this
        # loose head (compaction keeps system and tools) are considered.
        sess = self._sessions.get(ms)
        if sess is not None:
            tips = [
                t
                for t in sess.tips
                if t.loose_head == ids.loose_head and (not windowed or now - t.ts <= _HEAD_WINDOW_S)
            ]
            for t in reversed(tips):
                if t.compaction:
                    return "compacted_from", t.node
            if any(t.sig is not None for t in tips):
                # Lazy: the first message's loose text is only built on this
                # (collapsed, sketched) path, never on a plain continue.
                text = first_text() if callable(first_text) else first_text
                cur = sketch(text, scope)
                if cur is not None:
                    best: tuple[float, _Tip] | None = None
                    for t in tips:
                        if t.sig is None:
                            continue
                        sim = similarity(cur, t.sig)
                        if t.sig_raw is not None:
                            sim = max(sim, similarity(cur, t.sig_raw))
                        if sim >= _OVERLAP_MIN and (best is None or sim > best[0]):
                            best = (sim, t)
                    if best is not None:
                        return "compacted_from", best[1].node
            # Cross-head: some harnesses send the summarize request under their
            # own summarizer system prompt (another head). Only for a head this
            # session already has, only a fingerprint-flagged tip (never plain
            # overlap), and only the shape "this head's last turn, then the
            # summarize call": the flagged tip is the session's newest and comes
            # directly after this head's newest tip, within the window. A match
            # consumes the flag, so a later collapsed request (another subagent
            # task, a second summarize under that head) never reuses it.
            if tips and len(sess.tips) >= 2:
                flagged = sess.tips[-1]
                if (
                    flagged.compaction
                    and flagged.loose_head != ids.loose_head
                    and sess.tips[-2] is tips[-1]
                    and now - flagged.ts <= _HEAD_WINDOW_S
                ):
                    flagged.compaction = False
                    return "compacted_from", flagged.node
        # Rule 4: subagent (a head never seen in this scope).
        if (model_id, scope, ids.head) not in self._heads:
            if parent_session_key is not None:
                p = self._sessions.get((model_id, scope, parent_session_key))
                return "subagent", (p.tips[-1].node if p and p.tips else None)
            if inflight_heads and any(h != ids.head for h in inflight_heads):
                return "subagent", None
        return "new", None

    def _record(
        self,
        ms: _Key,
        now: float,
        ids: Ids,
        is_compaction: bool,
        request_id: str | None = None,
        conv: str | None = None,
    ) -> None:
        nodes, loose = self._nodes, self._loose
        n = len(ids.nodes)
        use_loose = len(ids.loose_nodes) == n
        prev_req = request_id
        for i in range(n):
            nk = (*ms, ids.nodes[i])
            e = nodes.get(nk)
            if e is None:
                e = nodes[nk] = _Node(
                    ids.nodes[i - 1] if i else None, ids.loose_head, now, req=prev_req, conv=conv
                )
            else:
                e.ts = now
                nodes.move_to_end(nk)
            if i == n - 1 and request_id is not None:
                e.req = request_id
            if i == n - 1 and conv is not None:
                e.conv = conv
            prev_req = e.req
            if use_loose:
                lk = (*ms, ids.loose_nodes[i])
                lo = loose.get(lk)
                if lo is None:
                    lo = loose[lk] = _Loose(ids.nodes[i], now, {})
                else:
                    lo.exact = ids.nodes[i]
                    lo.ts = now
                    loose.move_to_end(lk)
                if i + 1 < n:
                    nxt = ids.loose_nodes[i + 1]
                    if nxt not in lo.children:
                        if len(lo.children) >= _MAX_CHILDREN:
                            del lo.children[next(iter(lo.children))]
                        lo.children[nxt] = None
        sess = self._sessions.get(ms)
        if sess is None:
            sess = self._sessions[ms] = _Session(now, [])
        else:
            sess.ts = now
            self._sessions.move_to_end(ms)
        last = ids.nodes[-1]
        old = next((t for t in sess.tips if t.loose_head == ids.loose_head), None)
        keep = old is not None and old.node == last
        sig = old.sig if keep and old is not None else None
        sig_raw = old.sig_raw if keep and old is not None else None
        if old is not None:
            sess.tips.remove(old)
        sess.tips.append(_Tip(ids.loose_head, last, now, sig, is_compaction, sig_raw))
        del sess.tips[:-_MAX_TIPS]
        hk = (ms[0], ms[1], ids.head)
        self._heads[hk] = now
        self._heads.move_to_end(hk)
        self._evict()

    def _gc(self, now: float) -> None:
        cutoff = now - self._ttl
        for od in (self._nodes, self._loose, self._sessions, self._replies):
            while od:
                first = next(iter(od))
                if od[first].ts >= cutoff:
                    break
                del od[first]
        while self._heads:
            first = next(iter(self._heads))
            if self._heads[first] >= cutoff:
                break
            del self._heads[first]

    def _evict(self) -> None:
        for od in (self._nodes, self._loose, self._sessions, self._heads, self._replies):
            while len(od) > self._max:
                od.popitem(last=False)
