"""Prompt Lens layer 1 on the hot path (lens spec v1 4.3, analysis plan Task 1.3).

Three steps, each fail-open (A-5): the caller wraps every call and a fault
drops only the lens fields, never the request.

1. ``prepare`` runs inside ``routes._cobs2_compute`` (in a worker thread for a
   big body): the head's line digests and per-tool hashes (memoised per
   distinct ``sys`` / ``tools`` id), the body's short msgs, sizes and roles, and
   the probes of every message this process has not probed yet (memoised by
   exact msg id, like the loose ids: in a conversation only the new tail costs).
2. ``link`` runs right after ``Lineage.observe`` (which recorded each new node's
   probes through ``LensPre.probes_at``): it finds the parent's head (the
   request that produced the parent node, or the key's recent head for a new
   session), feeds ``RecentHeads`` and takes the break message's probes.
   ``LensState.admit`` then evaluates the structural verdict for the kvsig
   event; the system prompt line it classifies is read from the current body
   and dropped, and the body references go with it.
3. ``LensState.final`` evaluates again at the finish, with the request's own
   outcome and reuse (lost tokens, server and provider causes) for the row.

Ids, keyed digests, codes and counts only: nothing here keeps or logs text
(A-3, A-10). Every memory is bounded.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from app.cobs2 import ids as ids_mod
from app.cobs2.canonical import _HEAD_ROLES, Canon
from app.cobs2.ids import Ids
from app.cobs2.keys import NetworkKey
from app.cobs2.lineage import Observation, short_msg
from app.cobs2.shape import _parts_chars
from app.lens import causes, probes, schema_lint
from app.lens import heads as heads_mod
from app.lens import versions as versions_mod
from app.lens.causes import LensVerdict, RenderOrder
from app.lens.heads import RecentHeads

Probes = tuple[bytes, bytes, bytes, bytes]

#: Probes memo, as many entries as the kvsig loose memo (I3: a smaller memo
#: re-probes every long session's whole body once a few dozen are live). An
#: entry is one 32-byte bytes (the four probes back to back) under a 17-byte
#: id: about 200 bytes with the dict's own share, 13 MB full.
PROBES_MAX = ids_mod._LOOSE_MEMO_MAX
#: Line digests per distinct ``sys`` id (lens spec v1 4.3 item 3), packed
#: (``probes.PackedDigests``: one bytes object, 8 bytes a digest), at most
#: ``heads.MAX_DIGESTS`` (16 KB) each.
SYS_IDS_MAX = 4_096
#: Per-tool hashes per distinct ``tools`` id: packed (8 bytes a tool), at
#: most ``schema_lint.MAX_TOOLS`` tools (4 KB) each; None above that.
TOOLS_IDS_MAX = 4_096
#: Schema lint per distinct ``tools`` id (Task 1.4): a tuple of small ints
#: (one per tool, at most ``schema_lint.MAX_TOOLS``).
TOOLS_LINT_MAX = 4_096
#: Tool-result content hashes per exact msg id (Task 1.4, repeat results):
#: 8 bytes, or ``b""`` for a message that holds no tool result. Sized like
#: the probes memo (I3): about 150 bytes an entry, 10 MB full.
RESULTS_MAX = ids_mod._LOOSE_MEMO_MAX


def _memo_key(msg_id: str) -> bytes:
    """A msg id's bytes (17 for a hex id; half its str's size): the memo key."""
    try:
        return bytes.fromhex(msg_id)
    except ValueError:
        return msg_id.encode()


class _LRU:
    """A locked, bounded LRU (prepare runs on the loop and in worker threads)."""

    def __init__(self, maxlen: int) -> None:
        self._max = maxlen
        self._od: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, k: Any) -> Any:
        with self._lock:
            v = self._od.get(k)
            if v is not None:
                self._od.move_to_end(k)
            return v

    def put(self, k: Any, v: Any) -> None:
        with self._lock:
            self._od[k] = v
            self._od.move_to_end(k)
            while len(self._od) > self._max:
                self._od.popitem(last=False)

    def __len__(self) -> int:
        return len(self._od)

    def clear(self) -> None:
        with self._lock:
            self._od.clear()


def clear_memos() -> None:
    """Empty the process-wide memos (tests: each starts from a cold process)."""
    for m in (_PROBES, _SYS_DIGESTS, _TOOL_HASHES, _TOOLS_LINT, _RESULTS, _VERSIONS):
        m.clear()


#: exact msg id (its bytes) -> its four probes packed into 32 bytes. The id
#: is keyed over the message and the cache_salt, the probes over the same: a
#: hit is exact.
_PROBES = _LRU(PROBES_MAX)
#: exact sys id (hex) -> line digests.
_SYS_DIGESTS = _LRU(SYS_IDS_MAX)
#: exact tools id (hex) -> per-tool definition hashes.
_TOOL_HASHES = _LRU(TOOLS_IDS_MAX)
#: (exact sys id, exact tools id) -> the detected prompt version (8 hex).
_VERSIONS = _LRU(SYS_IDS_MAX)
#: exact tools id (hex) -> per-tool lint flags (``schema_lint.lint_tools``).
_TOOLS_LINT = _LRU(TOOLS_LINT_MAX)
#: exact msg id (its bytes) -> its tool result's content hash (``b""``: no result).
_RESULTS = _LRU(RESULTS_MAX)


@dataclass(frozen=True)
class HeadSnap:
    """One request's head: the exact and loose ``sys`` / ``tools`` ids, the
    per-tool hashes and the system prompt's line digests (None: not computed)."""

    sys: str
    tools: str
    loose_sys: str
    loose_tools: str
    tool_hashes: probes.PackedDigests | None
    digests: probes.PackedDigests | None


class ParentHeads:
    """Request id -> the head it was sent with, so a turn finds its lineage
    parent's head through ``Observation.parent_req``. A node id is chained from
    the exact head, so every request whose body held the parent node sent the
    same head. Bounded: ``max_reqs`` request ids, each holding only the
    ``(sys, tools)`` key of one of at most ``max_heads`` snapshots; a request
    whose snapshot was evicted resolves to None. A snapshot packs its line
    digests (at most ``heads.MAX_DIGESTS`` x 8 = 16 KB) and its per-tool
    hashes (at most ``schema_lint.MAX_TOOLS`` x 8 = 4 KB, None above), so the
    worst case is about 4,096 x 20 KB = 80 MB plus 16,384 small keys. Ids
    and digests only."""

    def __init__(self, max_reqs: int = 16_384, max_heads: int = 4_096) -> None:
        self._reqs: OrderedDict[str, tuple[str, str]] = OrderedDict()
        self._snaps: OrderedDict[tuple[str, str], HeadSnap] = OrderedDict()
        self._max_reqs = max_reqs
        self._max_heads = max_heads
        self._lock = threading.Lock()

    def put(self, req_id: str, snap: HeadSnap) -> None:
        k = (snap.sys, snap.tools)
        with self._lock:
            shared = self._snaps.get(k)
            if shared is None or shared != snap:
                self._snaps[k] = snap
            self._snaps.move_to_end(k)
            while len(self._snaps) > self._max_heads:
                self._snaps.popitem(last=False)
            self._reqs[req_id] = k
            self._reqs.move_to_end(req_id)
            while len(self._reqs) > self._max_reqs:
                self._reqs.popitem(last=False)

    def get(self, req_id: str) -> HeadSnap | None:
        with self._lock:
            k = self._reqs.get(req_id)
            return self._snaps.get(k) if k is not None else None

    def __len__(self) -> int:
        return len(self._reqs)

    def snapshots(self) -> int:
        """Live snapshots (the memory bound)."""
        return len(self._snaps)


def _msg_text(m: dict[str, Any]) -> str:
    return "\n".join(
        p["v"] for p in m["parts"] if p.get("t") == "text" and isinstance(p.get("v"), str)
    )


def sys_text(head: Sequence[dict[str, Any]]) -> str:
    """The system prompt as the lens numbers its lines: each head message's
    text parts, one per line, the messages one after the other."""
    return "\n".join(_msg_text(m) for m in head)


def sys_line(text: str, n: int) -> str | None:
    """Piece ``n`` (1-based) of ``text`` as ``probes.sys_line_digests`` cuts it:
    a line (without its ending) for the first ``MAX_LINES``, then fixed chunks."""
    if n < 1:
        return None
    lines, rest = probes.sys_pieces(text)
    if n <= probes.MAX_LINES:
        return lines[n - 1] if n <= len(lines) else None
    i = (n - probes.MAX_LINES - 1) * probes.CHUNK_CHARS
    return rest[i : i + probes.CHUNK_CHARS] if i < len(rest) else None


class LensPre:
    """What ``prepare`` computed. ``_raw`` and ``_head`` reference the current
    body (no copy) until ``release``: the probes fallback and the one line the
    span class reads."""

    __slots__ = (
        "snap",
        "msgs",
        "msg_ids",
        "msg_chars",
        "sys_chars",
        "tools_chars",
        "first_turn",
        "reasoning_at",
        "tools_lint",
        "repeat_chars",
        "version",
        "_raw",
        "_head",
        "_key",
        "_salt",
    )

    def __init__(
        self,
        snap: HeadSnap,
        msgs: tuple[bytes, ...],
        msg_ids: tuple[str, ...],
        msg_chars: tuple[int, ...],
        sys_chars: int,
        tools_chars: int,
        first_turn: bool | None,
        reasoning_at: frozenset[int],
        raw: list[dict[str, Any]] | None,
        head: list[dict[str, Any]] | None,
        key: NetworkKey,
        salt: bytes,
        tools_lint: tuple[int, ...] | None = None,
        repeat_chars: int | None = None,
        version: str | None = None,
    ) -> None:
        self.snap = snap
        self.msgs = msgs
        self.msg_ids = msg_ids
        self.msg_chars = msg_chars
        self.sys_chars = sys_chars
        self.tools_chars = tools_chars
        self.first_turn = first_turn
        self.reasoning_at = reasoning_at
        self.tools_lint = tools_lint
        self.repeat_chars = repeat_chars
        self.version = version
        self._raw = raw
        self._head = head
        self._key = key
        self._salt = salt

    def probes_at(self, i: int) -> Probes:
        """Body message ``i``'s probes: the memo, else computed (and memoised)
        while the body is held. Raises when neither is possible (the lineage
        then records the node without probes)."""
        mid = _memo_key(self.msg_ids[i])
        packed: bytes | None = _PROBES.get(mid)
        if packed is not None:
            return (packed[0:8], packed[8:16], packed[16:24], packed[24:32])
        raw = self._raw
        if raw is None:
            raise LookupError("no probes")
        p = probes.message_probes(raw[i], self._key, self._salt)
        _PROBES.put(mid, b"".join(p))
        return p

    def line(self, n: int) -> str:
        """Line ``n`` of the current system prompt (read, never kept). Raises
        when the body is gone or has no such line: no span class then."""
        head = self._head
        text = sys_line(sys_text(head), n) if head is not None else None
        if text is None:
            raise LookupError("no such line")
        return text

    def release(self) -> None:
        self._raw = None
        self._head = None


def _body_raw(body: dict[str, Any], n: int) -> list[dict[str, Any]] | None:
    """The raw body messages, in canonical order (the same head/body split as
    ``canonicalize``); None when they do not line up with the ``n`` canonical ones."""
    out: list[dict[str, Any]] = []
    for m in body.get("messages") or ():
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if (role if isinstance(role, str) else "") not in _HEAD_ROLES:
            out.append(m)
    return out if len(out) == n else None


def _tools_lint(body: dict[str, Any], tools_id: str) -> tuple[int, ...] | None:
    """The body's tool lint, once per distinct tools id. None when it failed."""
    got: tuple[int, ...] | None = _TOOLS_LINT.get(tools_id)
    if got is None:
        try:
            got = tuple(schema_lint.lint_tools(body.get("tools")))
            _TOOLS_LINT.put(tools_id, got)
        except Exception:  # noqa: BLE001 -- one field, never the pass
            return None
    return got


def repeat_chars(
    msgs: Sequence[dict[str, Any]], msg_ids: Sequence[str], key: NetworkKey, salt: bytes
) -> int:
    """Chars of the tool results whose content already came back in an earlier
    tool-result message of this prompt (analysis spec 7.2 item 2, ruling R1):
    each repeat counts once, the first occurrence never. Hashes are memoised
    by exact msg id, so a conversation's old results are memo hits."""
    seen: set[bytes] = set()
    n = 0
    for m, mid in zip(msgs, msg_ids, strict=True):
        if m.get("role") != "tool":
            continue
        mk = _memo_key(mid)
        h: bytes | None = _RESULTS.get(mk)
        parts = next((p for p in m["parts"] if p.get("t") == "tool_result"), None)
        if h is None:
            sub = parts.get("parts") if parts is not None else None
            h = probes.result_hash(sub, key, salt) if isinstance(sub, list) and sub else b""
            _RESULTS.put(mk, h)
        if not h:
            continue
        if h in seen:
            sub = parts.get("parts") if parts is not None else None
            n += _parts_chars(sub) if isinstance(sub, list) else 0
        else:
            seen.add(h)
    return n


def _first_turn(roles: Sequence[str]) -> bool | None:
    """True when the body has no prior turn of any kind: no assistant and no
    tool message (an agent loop's only user message is its first one, so
    "nothing before the last user message" would read a tool loop as new).
    An Anthropic body reaches the lens translated, its ``tool_result`` blocks
    as ``tool`` messages, so they count too. None without a user message
    (unknown: never a new conversation)."""
    if "user" not in roles:
        return None
    return "assistant" not in roles and "tool" not in roles


def prepare(
    body: dict[str, Any],
    canon: Canon,
    ids: Ids,
    key: NetworkKey,
    salt: bytes,
) -> LensPre:
    """The lens part of the kvsig pass (worker-thread safe). Raises on a bug;
    the caller fails open. A digest, hash or probe that fails alone is left out.
    The head part ids are memoised by head id (``ids.head_part_ids``): the
    kvsig event's own call, when Prompt Lens is on, is a memo hit."""
    head = ids_mod.head_part_ids(canon, key, salt, ids.head)
    hashes = _TOOL_HASHES.get(head.tools)
    if hashes is None:
        try:
            hashes = probes.tool_def_hashes(canon.tools, key, salt)
            if hashes is not None:  # None (over MAX_TOOLS) is a length check: no memo
                _TOOL_HASHES.put(head.tools, hashes)
        except Exception:  # noqa: BLE001 -- one field, never the pass
            hashes = None
    digests = _SYS_DIGESTS.get(head.sys)
    if digests is None:
        try:
            digests = probes.PackedDigests.of(
                probes.sys_line_digests(sys_text(canon.head), key, salt)[: heads_mod.MAX_DIGESTS]
            )
            _SYS_DIGESTS.put(head.sys, digests)
        except Exception:  # noqa: BLE001
            digests = None
    version: str | None = None
    if digests is not None:
        vk = (head.sys, head.tools)
        version = _VERSIONS.get(vk)
        if version is None:
            try:
                version = versions_mod.version_id(
                    digests,
                    versions_mod.volatile_lines(sys_text(canon.head)),
                    head.tools,
                    key,
                    salt,
                )
                _VERSIONS.put(vk, version)
            except Exception:  # noqa: BLE001 -- one field, never the pass
                version = None
    msgs = canon.messages
    raw = _body_raw(body, len(msgs))
    try:
        rep_chars: int | None = repeat_chars(msgs, ids.msgs, key, salt)
    except Exception:  # noqa: BLE001 -- one field, never the pass
        rep_chars = None
    pre = LensPre(
        HeadSnap(head.sys, head.tools, head.loose_sys, head.loose_tools, hashes, digests),
        tuple(short_msg(m) for m in ids.msgs),
        ids.msgs,
        tuple(_parts_chars(m["parts"]) for m in msgs),
        sum(_parts_chars(m["parts"]) for m in canon.head),
        sum(len(t) for t in canon.tools if isinstance(t, str)),
        _first_turn([m["role"] for m in msgs]),
        frozenset(
            i for i, m in enumerate(msgs) if any(p.get("t") == "reasoning" for p in m["parts"])
        ),
        raw,
        canon.head,
        key,
        salt,
        tools_lint=_tools_lint(body, head.tools),
        repeat_chars=rep_chars,
        version=version,
    )
    if raw is not None:
        # Here, in the kvsig pass's thread: the messages this process has not
        # probed yet (a conversation's new tail), so the lineage's per-node
        # probes and the break message's are memo hits on the loop.
        for i in range(len(msgs)):
            if _PROBES.get(_memo_key(ids.msgs[i])) is None:
                try:
                    pre.probes_at(i)
                except Exception:  # noqa: BLE001 -- that node goes without probes
                    pass
    return pre


def _apportion(
    prompt_tokens: int, sys_chars: int, tools_chars: int, msg_chars: Sequence[int]
) -> dict[str, Any]:
    """Token counts per part, from the prompt's token count shared by chars."""
    total = sys_chars + tools_chars + sum(msg_chars)
    if prompt_tokens <= 0 or total <= 0:
        return {"msg_tokens": tuple(0 for _ in msg_chars), "head_tokens": 0, "tools_tokens": 0}
    f = prompt_tokens / total
    return {
        "msg_tokens": tuple(int(c * f) for c in msg_chars),
        "head_tokens": int((sys_chars + tools_chars) * f),
        "tools_tokens": int(tools_chars * f),
    }


class LensState:
    """The lens inputs between admission and the finish. ``adm`` is the
    admission verdict (the kvsig event's), None until ``admit``."""

    __slots__ = (
        "ci",
        "order",
        "sys_chars",
        "tools_chars",
        "msg_chars",
        "tools_lint",
        "repeat_chars",
        "version",
        "adm",
        "_pre",
    )

    def __init__(self, ci: causes.CauseInputs, order: RenderOrder, pre: LensPre) -> None:
        self.ci = ci
        self.order: RenderOrder = order
        self.sys_chars = pre.sys_chars
        self.tools_chars = pre.tools_chars
        self.msg_chars = pre.msg_chars
        self.tools_lint = pre.tools_lint
        self.repeat_chars = pre.repeat_chars
        self.version = pre.version
        self.adm: LensVerdict | None = None
        self._pre: LensPre | None = pre

    def repeat_tokens(self, prompt_tokens: int) -> int | None:
        """``lens_repeat_results``: the repeated results' chars at the
        request's own tokens per char. None without a token count."""
        rc = self.repeat_chars
        total = self.sys_chars + self.tools_chars + sum(self.msg_chars)
        if rc is None or prompt_tokens <= 0 or total <= 0:
            return None
        return int(rc * prompt_tokens / total)

    def _tokens(self, prompt_tokens: int) -> dict[str, Any]:
        return _apportion(prompt_tokens, self.sys_chars, self.tools_chars, self.msg_chars)

    def admit(self, prompt_tokens: int) -> LensVerdict:
        """The structural verdict (outcome unknown). Classifies the changed
        system prompt line from the current body, then lets the body go."""
        pre = self._pre
        try:
            line: Callable[[int], str] | None = pre.line if pre is not None else None
            ci = replace(self.ci, **self._tokens(prompt_tokens), sys_text_line=line)
            self.adm = causes.evaluate(ci, render_order=self.order)
            return self.adm
        finally:
            if pre is not None:
                pre.release()
            self._pre = None

    def final(
        self,
        *,
        prompt_tokens: int,
        own_outcome: str | None,
        actual_tokens: int | None,
        gap_s: float | None = None,
        relayed_to: str | None = None,
        cache_marks: int | None = None,
        provider_min: int | None = None,
    ) -> LensVerdict:
        """The row's verdict: with the own outcome and reuse. The span class
        comes from the admission verdict (the body is gone by now)."""
        ci = replace(
            self.ci,
            **self._tokens(prompt_tokens),
            own_outcome=own_outcome,
            actual_tokens=actual_tokens,
            gap_s=gap_s,
            relayed_to=relayed_to,
            cache_marks=cache_marks,
            provider_min=provider_min,
            sys_text_line=None,
        )
        v = causes.evaluate(ci, render_order=self.order)
        adm = self.adm
        if (
            adm is not None
            and v.span_class is None
            and v.span_line is not None
            and (v.brk, v.span_line) == (adm.brk, adm.span_line)
        ):
            v = replace(v, span_class=adm.span_class)
        if prompt_tokens <= 0:
            # No token count: the figures would read 0, which is a claim.
            v = replace(v, counterfactual_tokens=None, lost_tokens=None)
        return v


#: The fallback compares only system prompts this close (Task 1.3 ruling):
#: at most this many differing lines, or this share of them, whichever is larger.
CLOSE_LINES = 3
CLOSE_SHARE = 0.05


def close_enough(a: Sequence[bytes] | None, b: Sequence[bytes] | None) -> bool:
    """Whether two line-digest sets differ in at most ``CLOSE_LINES`` lines or
    ``CLOSE_SHARE`` of them (whichever is larger), and differ at all."""
    if a is None or b is None or not a or not b:
        return False
    fc = probes.first_changed_line(a, b)
    if fc is None:
        return False
    return fc[1] <= max(CLOSE_LINES, CLOSE_SHARE * max(len(a), len(b)))


def link(
    pre: LensPre,
    obs: Observation,
    *,
    scope: bytes,
    model_id: str,
    request_id: str | None,
    heads: RecentHeads | None,
    parents: ParentHeads | None,
    now: float,
    render_order: RenderOrder,
) -> LensState:
    """The lineage-side inputs for ``obs`` (on the loop, after the observe)."""
    snap = pre.snap
    if parents is not None and request_id:
        parents.put(request_id, snap)
    edge = obs.edge
    parent: HeadSnap | None = None
    if edge == "continue" and not obs.loose_match_beyond_exact and obs.parent is not None:
        # The exact chain matched: the parent node is chained from this head.
        parent = snap
    elif edge in ("continue", "fork", "compacted_from"):
        if parents is not None and obs.parent_req:
            parent = parents.get(obs.parent_req)
    elif edge in ("new", "subagent") and heads is not None:
        # Cross-session (lens spec 4.6): this key's recent head with the same
        # loose system prompt and tool list; only sys can differ.
        e = heads.lookup(scope, model_id, snap.loose_sys, snap.tools, now)
        if e is not None:
            parent = replace(snap, sys=e.sys_id, digests=e.digests)
        elif pre.first_turn is True:
            # Spec 4.5 fallback: the key's last head with the same tools, when
            # its system prompt differs only in a few lines (a path, a date).
            # Only for a new conversation's body (M6): a body with earlier
            # turns whose parent left memory stays unattributed.
            f = heads.lookup_by_tools(scope, model_id, snap.tools, now)
            if f is not None and f.loose_sys and close_enough(f.digests, snap.digests):
                parent = replace(snap, sys=f.sys_id, loose_sys=f.loose_sys, digests=f.digests)
    if heads is not None and snap.digests is not None:
        # Never without digests: an empty set would later locate a false line 1.
        heads.remember(scope, model_id, snap.loose_sys, snap.tools, snap.sys, snap.digests, now)
    pm, now_msgs = obs.parent_msgs, pre.msgs
    b = next((i for i in range(min(len(pm), len(now_msgs))) if pm[i] != now_msgs[i]), None)
    probes_now: Probes | None = None
    if b is not None:
        try:
            probes_now = pre.probes_at(b)
        except Exception:  # noqa: BLE001 -- the break goes without probes
            probes_now = None
    ci = causes.CauseInputs(
        edge=edge,
        obs=obs,
        sys_now=snap.sys,
        sys_parent=parent.sys if parent is not None else None,
        loose_sys_now=snap.loose_sys,
        loose_sys_parent=parent.loose_sys if parent is not None else None,
        sys_digests_now=snap.digests,
        sys_digests_parent=parent.digests if parent is not None else None,
        tools_now=snap.tools,
        tools_parent=parent.tools if parent is not None else None,
        loose_tools_now=snap.loose_tools,
        loose_tools_parent=parent.loose_tools if parent is not None else None,
        tool_hashes_now=snap.tool_hashes,
        tool_hashes_parent=parent.tool_hashes if parent is not None else None,
        msgs_now=now_msgs,
        probes_now=probes_now,
        first_turn=pre.first_turn,
        reasoning_now=(b in pre.reasoning_at) if b is not None else None,
    )
    return LensState(ci, render_order, pre)


def apply(live: Any, v: LensVerdict) -> None:
    """Copy a verdict onto a ``LiveRequest`` (the row's lens fields)."""
    live.lens_break = v.brk
    live.lens_cause = v.cause
    live.lens_span_class = v.span_class
    live.lens_span_line = v.span_line
    live.lens_cf_tokens = v.counterfactual_tokens
    live.lens_lost_tokens = v.lost_tokens
    live.lens_other_breaks = v.other_breaks


def own_reuse(cached: int | None, reusable_own: int | None) -> int | None:
    """Tokens this request reused from its own key's earlier requests: the
    engine's figure, capped by what the key's own history could give."""
    if cached is None or isinstance(cached, bool):
        return None
    c = max(int(cached), 0)
    if isinstance(reusable_own, int) and not isinstance(reusable_own, bool):
        return min(c, max(reusable_own, 0))
    return c


# -- relays (Anthropic Messages) ---------------------------------------------


def relay_outcome(usage_known: bool, cache_read: int) -> str | None:
    """A relayed turn's own outcome from the upstream's usage."""
    if not usage_known:
        return None
    return "hit" if cache_read > 0 else "cold"


def provider_min(model: str) -> int:
    """Anthropic's minimum cacheable prefix (spec section 6): 2,048 tokens on
    Haiku models, 1,024 on the others."""
    return 2_048 if "haiku" in (model or "").lower() else 1_024


def _marked(b: Any) -> bool:
    return isinstance(b, dict) and isinstance(b.get("cache_control"), dict)


def cache_marks_of(body: Any) -> int | None:
    """``cache_control`` marks in an Anthropic Messages body: the top level
    (automatic caching), system blocks, tools, message content blocks and the
    blocks inside a tool result. None when ``body`` is not a JSON object."""
    if not isinstance(body, dict):
        return None
    n = 1 if _marked(body) else 0
    sys = body.get("system")
    if isinstance(sys, list):
        n += sum(1 for b in sys if _marked(b))
    tools = body.get("tools")
    if isinstance(tools, list):
        n += sum(1 for t in tools if _marked(t))
    for m in body.get("messages") or ():
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            continue
        for b in content:
            if _marked(b):
                n += 1
            inner = b.get("content") if isinstance(b, dict) else None
            if isinstance(inner, list):
                n += sum(1 for x in inner if _marked(x))
    return n
