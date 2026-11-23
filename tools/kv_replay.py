"""KV trace build and replay for the sharing and flood gates (G9, G10; Plan 2 spec 8).

A trace is JSON lines, one request per line, in time order::

    {"t": 12.5, "key": "k03", "isolated": false, "priority": 5, "tokens": [151644, ...]}

``t`` is seconds from the first request, ``key`` a name made up for the trace (never a
secret or a key id), ``tokens`` the prompt as token ids. A trace holds no text, but
token ids decode back to it: keep a trace built from recordings where the recordings are.

Commands (``lmw_kv`` must be importable: run from the repo root with
``PYTHONPATH=lmw_kv``; every location is an argument, there are no defaults)::

    python tools/kv_replay.py build --recordings DIR --tokenizer PATH --out trace.jsonl
        [--key-by session|auth] [--priority 5] [--all-status]

  Wire recordings (one JSON file per request: ``ts``, ``path``, ``request_headers``,
  ``request_body`` {``text`` or ``b64``}, ``status``) to a trace, offline. Chat
  Completions, Responses and Anthropic Messages bodies are rendered in ChatML (system
  and tools, then each message; reasoning left out, as chat templates drop it) and
  tokenized with ``PATH`` (a Hugging Face ``tokenizer.json``). The rendering is not the
  engine's chat template, but it is stable: a turn that repeats the last one's messages
  repeats its tokens, so prefix reuse is kept. ``--key-by session`` (default) makes each
  harness session its own key (session id headers); ``auth`` one key per Authorization
  value. Keys are numbered in order of first use; header values are never written, and
  a request without that header gets a key of its own (``unnamed`` in the summary).
  Failed requests (status not 2xx) and files that are not recordings are left out
  unless ``--all-status`` (failed ones only).

    python tools/kv_replay.py sim --trace trace.jsonl --policy lru|fair --ram-blocks N
        --nvme-segments M --segment-blocks S --block-size B [--anonymous] [--workdir DIR]
        [--flood KEY [--factor 10] [--flood-victim KEY] [--flood-share tier|traffic]]
        [--gap-s 300]

  Replays the trace on an in-process ``Store`` in a temporary directory and prints
  ``{"ram_tokens", "nvme_tokens", "recomputed_tokens", "by_key": {...}}``. There is no
  GPU cache model: every request looks up its whole blocks (all but the last token's),
  loads the leading stored ones and saves the rest. RAM holds ``N`` blocks, NVMe ``M``
  segment files of ``S`` blocks. Single-threaded and one replay per process at a time
  (it sets store module globals while it runs). ``by_key`` adds, per key, ``turns``,
  ``repeat_turns`` / ``repeat_served`` (turns whose first block the key sent before, and
  those that reused every such leading block) and ``resumed_turns`` / ``served_turns``
  (the repeats that come more than ``--gap-s`` after the key's previous turn: what G10
  scores, since closer turns hit the GPU prefix cache in production; ``GAP_S`` says why
  300 s), and ``served_store_turns``, the resumed turns RAM and NVMe alone served (equal
  to ``served_turns`` here, which has no GPU cache). ``--flood`` adds ``flood()`` first: ``factor`` times the flooder's share of the
  larger tier (``--flood-share tier``, spec 5.3) or of the trace's tokens (``traffic``),
  inside every resume pause of ``--flood-victim`` (or spread over the whole trace).
  ``--anonymous`` saves without tenants (sharing only): with ``lru`` it runs on the step
  1 store too, for G9's "same hits as step 1".

    python tools/kv_replay.py live --trace trace.jsonl --url URL --keys keys.json
        --model NAME --block-size B [--admin-token-env VAR] [--timeout 600]
        [--flood KEY --flood-share-tokens T [--factor 10] [--flood-victim KEY]] [--gap-s 300]

  Sends each entry as ``POST /v1/completions`` with the token-id prompt and
  ``max_tokens: 1``, in order and without pacing, under the bearer of its key in
  ``keys.json`` (``{"k01": {"token": "...", "token_id": "..."}}``, the flooder's too; a
  key's isolation and priority are its settings on the server, not the trace's). Each
  turn is then read back from its history row (``GET /api/requests/{X-LMW-Request-Id}``,
  ``tiers``) and scored per turn as ``sim`` does, the GPU's tokens counting as reused in
  ``served_turns`` and not in ``served_store_turns`` (G10 counts the second). A row whose
  gpu, ram and nvme are all null counts in ``unscored``, not as a miss.
  It adds each key's ``GET /api/tokens/{id}/cache`` 24 h window difference and ``GET
  /api/kv/usage``. ``--flood-share-tokens`` is the flooder's tier share in tokens (the
  server's tier sizes are not in tokens). The admin token is read from the environment
  variable (default ``LMW_ADMIN_TOKEN``); no token is printed. ``--timeout`` is per
  HTTP request, in seconds.

The ``build`` test needs ``tokenizers``; it comes with ``transformers`` in
requirements-dev.txt, and the test skips where it is missing.

G9: replay the recorded trace (spec 8.1) with ``--policy lru`` (equal to ``--anonymous``
on the step 1 store) and ``--policy fair``; fair's ``ram_tokens + nvme_tokens`` must be
at least 95% of lru's (report ``ram_tokens`` too: the sum cannot see RAM demoted to
NVMe). G10: ``--flood`` a new key at factor 10 with ``--flood-victim`` the resumed
session's key; its ``served_store_turns / resumed_turns`` must be at least 0.9 with
``fair`` (report ``lru``). In ``live``, ``--flood-share-tokens`` is the tier divided by
the keys holding blocks during the flood (spec 5.3: with 8 keys, 10x is 1.25 tiers).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import itertools
import json
import math
import mmap
import os
import random
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from lmw_kv import store as store_mod
from lmw_kv.connector_logic import block_keys_for, max_load_blocks
from lmw_kv.format import HEADER_LEN, record_size
from lmw_kv.keys import Namespace, root_key
from lmw_kv.store import Store, StoreConfig

Entry = dict[str, Any]
Op = tuple[Any, ...]

#: Keys the block keys of a replay; any constant does (the trace never leaves the replay).
REPLAY_SECRET = b"lmw-kv-replay\x00v1"
GROUP = "kv"
GROUP_BYTES = 1000  # one page per block in the RAM file
CLOCK_BASE = 1_000_000.0
#: Token ids of flood prompts: inside every production vocabulary.
FLOOD_TOKENS = (1000, 100_000)
SESSION_HEADERS = (
    "x-claude-code-session-id",
    "x-session-id",
    "session-id",
    "x-opencode-session",
    "thread-id",
)
COUNTS = ("ram_tokens", "nvme_tokens", "recomputed_tokens")
#: A turn more than this long after its key's previous turn is a resume (G10). In
#: production the GPU prefix cache serves turns seconds apart; in the recorded trace 194
#: of 208 same-key gaps are under 30 s and 201 under 5 min, while the spec's resumes come
#: every 30 min. 5 min separates bursts from resumes with room on both sides.
GAP_S = 300.0
#: The proxy's response header naming the request's history row (app/proxy/request_id.py).
REQUEST_ID_HEADER = "X-LMW-Request-Id"


# ---- replay -------------------------------------------------------------------------


def replay_ops(
    store: Store, ops: Iterable[Op], *, results: list[Any] | None = None
) -> dict[str, int]:
    """Apply operations to ``store`` in order; returns token totals.

    Operations (``opts`` is a dict, every entry optional)::

        ("save", ns_id, key, parent, scope, opts)   begin_save; opts: tenant, scope_tag
        ("commit", ns_id, key)  ("abort", ns_id, key)  ("flush",)
        ("lookup", ns_id, keys)
        ("load", ns_id, keys, opts)  begin_load; opts: tenant, block_tokens, prompt_tokens,
                                     hold (a name: end_load waits for ("end", name))
        ("end", name)

    A load counts ``block_tokens`` (default 1) per block it served, from RAM or from
    NVMe, and ``prompt_tokens - served`` as recomputed when ``prompt_tokens`` is given.
    Each result (the begin_save offsets or None, the lookup count, a load's
    ``(blocks, nvme_blocks)``, else None) is appended to ``results`` before the next
    operation is read, so a lazy ``ops`` can depend on it. Empty tenant and scope tag
    are not passed: the step 1 store (no tenants) replays ``lru`` operations too.
    """
    totals = {"ram_tokens": 0, "nvme_tokens": 0, "recomputed_tokens": 0}
    held: dict[str, int] = {}
    out: list[Any] = results if results is not None else []
    for op in ops:
        kind = op[0]
        got: Any = None
        if kind == "save":
            _, ns_id, key, parent, scope, opts = op
            got = store.begin_save(
                ns_id, key, parent, scope, **_tagged(opts, "tenant", "scope_tag")
            )
        elif kind == "commit":
            store.commit_save(op[1], op[2])
        elif kind == "abort":
            store.abort_save(op[1], op[2])
        elif kind == "flush":
            store.flush()
        elif kind == "lookup":
            got = store.lookup(op[1], op[2])
        elif kind == "load":
            _, ns_id, keys, opts = op
            h = store.begin_load(ns_id, keys, **_tagged(opts, "tenant"))
            n, nv = len(h.slots), h.nvme_blocks
            bt = int(opts.get("block_tokens", 1))
            totals["ram_tokens"] += (n - nv) * bt
            totals["nvme_tokens"] += nv * bt
            if "prompt_tokens" in opts:
                totals["recomputed_tokens"] += int(opts["prompt_tokens"]) - n * bt
            hold = opts.get("hold")
            if hold is None:
                store.end_load(h.id)
            else:
                held[hold] = h.id
            got = (n, nv)
        elif kind == "end":
            hid = held.pop(op[1], None)
            if hid is not None:
                store.end_load(hid)
        else:
            raise ValueError(f"unknown operation {kind!r}")
        out.append(got)
    for hid in held.values():
        store.end_load(hid)
    return totals


def _tagged(opts: Mapping[str, Any], *names: str) -> dict[str, Any]:
    return {n: opts[n] for n in names if opts.get(n)}


def _tenant(key: str) -> bytes:
    return hashlib.sha256(b"lmw-kv-replay-tenant\x00" + key.encode()).digest()[:16]


def _scope(key: str) -> bytes:
    return hashlib.sha256(b"lmw-kv-replay-scope\x00" + key.encode()).digest()


class _Clock:
    """Trace time, strictly increasing between calls (no two reads tie)."""

    def __init__(self) -> None:
        self.now = CLOCK_BASE
        self._n = itertools.count()

    def time(self) -> float:
        return self.now + next(self._n) * 1e-6

    @staticmethod
    def monotonic() -> float:
        return time.monotonic()


@contextlib.contextmanager
def _sim_store(
    clock: _Clock, ram_blocks: int, nvme_segments: int, segment_blocks: int, workdir: Path | None
) -> Iterator[Store]:
    """A store in a temporary directory, its segment size and clock set (restored after)."""
    seg = HEADER_LEN + segment_blocks * record_size(GROUP, GROUP_BYTES)  # S blocks exactly
    old_seg, old_time = store_mod.SEGMENT_BYTES, getattr(store_mod, "time")  # noqa: B009
    store_mod.SEGMENT_BYTES = seg  # read when a segment writer opens
    setattr(store_mod, "time", clock)  # noqa: B010  # the step 1 store ages segments by it
    try:
        with tempfile.TemporaryDirectory(prefix="kv-replay-", dir=workdir) as tmp:
            slot = -(-GROUP_BYTES // mmap.PAGESIZE) * mmap.PAGESIZE
            cfg = StoreConfig(
                ram_path=Path(tmp) / "ram.pool",
                ram_bytes=ram_blocks * slot,
                nvme_dir=Path(tmp) / "nvme" if nvme_segments > 0 else None,
                nvme_cap_bytes=nvme_segments * seg,
            )
            if hasattr(cfg, "clock"):
                cfg.clock = clock.time
            s = Store(cfg)
            try:
                yield s
            finally:
                s.close()
    finally:
        store_mod.SEGMENT_BYTES = old_seg
        setattr(store_mod, "time", old_time)  # noqa: B010


def _policy(mode: str, trace: Sequence[Entry], anonymous: bool) -> Any:
    from lmw_kv.fair import Policy, TenantPolicy

    tenants = {} if anonymous else {
        _tenant(e["key"]): TenantPolicy(weight=int(e.get("priority", 5)) + 1) for e in trace
    }  # fmt: skip
    if mode == "fair":
        return Policy(mode="fair", tenants=tenants)
    return Policy(mode="lru", tenants=tenants)


class TurnScore:
    """Per key: which turns repeat a prefix the key sent before, and which were served.

    A turn repeats when its first loadable block is one this key sent before; its
    expected blocks are the leading loadable blocks this key sent before. It is
    *resumed* when it also comes more than ``gap_s`` after the key's previous turn:
    turns closer together are served by the engine's GPU prefix cache in production,
    so only resumed turns test the RAM and NVMe tiers (G10). A turn is served when the
    tokens it reused cover its expected blocks.
    """

    def __init__(self, block_size: int, gap_s: float) -> None:
        self.block_size = block_size
        self.gap_s = gap_s
        self._seen: dict[str, set[bytes]] = {}
        self._last: dict[str, float] = {}
        self.rows: dict[str, dict[str, Any]] = {}

    def expect(self, key: str, t: float, tokens: Sequence[int]) -> tuple[int, bool]:
        """(expected blocks, resumed) for a turn about to run; remembers its blocks."""
        b = self.block_size
        ids = block_keys_for(REPLAY_SECRET, b"", b"\x00" * 32, tokens, b)
        loadable = ids[: max_load_blocks(len(tokens), b)]
        mine = self._seen.setdefault(key, set())
        expected = next((i for i, k in enumerate(loadable) if k not in mine), len(loadable))
        prev = self._last.get(key)
        resumed = expected > 0 and prev is not None and float(t) - prev > self.gap_s
        mine.update(ids)
        self._last[key] = float(t)
        return expected, resumed

    def row(self, key: str) -> dict[str, Any]:
        return self.rows.setdefault(
            key,
            {c: 0 for c in (*COUNTS, "turns", "repeat_turns", "repeat_served",
                            "resumed_turns", "served_turns", "served_store_turns")},
        )  # fmt: skip

    def score(
        self,
        key: str,
        expected: int,
        resumed: bool,
        reused_tokens: int,
        store_tokens: int | None = None,
    ) -> None:
        """Score one turn. ``store_tokens`` is the part of ``reused_tokens`` that came
        from RAM or NVMe (None: all of it, as in ``sim``, which has no GPU cache)."""
        r = self.row(key)
        r["turns"] += 1
        if expected <= 0:
            return
        need = expected * self.block_size
        ok = int(reused_tokens >= need)
        r["repeat_turns"] += 1
        r["repeat_served"] += ok
        if resumed:
            r["resumed_turns"] += 1
            r["served_turns"] += ok
            store = reused_tokens if store_tokens is None else store_tokens
            r["served_store_turns"] += int(store >= need)


def sim(
    trace: Iterable[Entry],
    *,
    policy: str,
    ram_blocks: int,
    nvme_segments: int,
    segment_blocks: int,
    block_size: int,
    anonymous: bool = False,
    workdir: Path | None = None,
    gap_s: float = GAP_S,
) -> dict[str, Any]:
    """Replay ``trace`` on an in-process store (module docstring, ``sim``).

    The store's files live in a temporary directory under ``workdir`` (the system's
    temporary directory when None); a memory file system there makes a replay many
    times faster, since the store syncs every segment and manifest write.

    Not thread-safe: it sets ``lmw_kv.store.SEGMENT_BYTES`` and the store module's
    ``time`` for the whole process while it runs (restored after). Requests run one at
    a time, in trace order, against a store whose only other thread is its writer.
    """
    if policy not in ("lru", "fair"):
        raise ValueError('policy: "lru" or "fair"')
    if anonymous and policy != "lru":
        raise ValueError("anonymous replays are lru only (fair needs tenants)")
    entries = list(trace)
    clock = _Clock()
    turns = TurnScore(block_size, gap_s)
    with _sim_store(clock, ram_blocks, nvme_segments, segment_blocks, workdir) as s:
        if hasattr(s, "set_policy"):
            s.set_policy(_policy(policy, entries, anonymous))
        elif policy != "lru":
            raise ValueError("this store has no fair policy")
        ns = Namespace(
            model_repo="replay/model", revision_sha="r", tokenizer_sha="t", kv_dtype="fp8",
            block_size=block_size, attention_layout="hnd", kv_groups=(GROUP,), tp=1, pp=1,
            engine_family="vllm", layout_version=1,
        )  # fmt: skip
        ns_id = s.register_namespace(ns, {GROUP: GROUP_BYTES})
        results: list[Any] = []
        ops = _entry_ops(entries, ns_id, block_size, anonymous, clock, results, turns)
        totals = replay_ops(s, ops, results=results)
    return {**totals, "by_key": dict(sorted(turns.rows.items()))}


def _entry_ops(
    entries: Sequence[Entry],
    ns_id: bytes,
    block_size: int,
    anonymous: bool,
    clock: _Clock,
    results: list[Any],
    turns: TurnScore,
) -> Iterator[Op]:
    """Each entry's lookup, load and saves; scores each turn from the results."""
    for e in entries:
        key, tokens = str(e["key"]), e["tokens"]
        clock.now = CLOCK_BASE + float(e["t"])
        isolated = bool(e.get("isolated", False))
        salt = _scope(key) if isolated else b""
        tag = b""
        if isolated:
            from lmw_kv.scopes import scope_tag

            tag = scope_tag(salt)
        tenant = b"" if anonymous else _tenant(key)
        keys = block_keys_for(REPLAY_SECRET, salt, ns_id, tokens, block_size)
        loadable = keys[: max_load_blocks(len(tokens), block_size)]
        expected, resumed = turns.expect(key, e["t"], tokens)
        yield ("lookup", ns_id, loadable)
        found = int(results[-1])
        opts = {"tenant": tenant, "block_tokens": block_size, "prompt_tokens": len(tokens)}
        yield ("load", ns_id, loadable[:found], opts)
        n, nv = results[-1]
        row = turns.row(key)
        row["ram_tokens"] += (n - nv) * block_size
        row["nvme_tokens"] += nv * block_size
        row["recomputed_tokens"] += len(tokens) - n * block_size
        turns.score(key, expected, resumed, n * block_size)
        parent = root_key(REPLAY_SECRET, salt, ns_id)
        for k in keys:
            yield ("save", ns_id, k, parent, salt, {"tenant": tenant, "scope_tag": tag})
            if results[-1] is not None:
                yield ("commit", ns_id, k)
                # Each write lands before the next save: whether a block is on NVMe yet
                # decides victims, and the writer thread's pace must not.
                yield ("flush",)
            parent = k


# ---- flood --------------------------------------------------------------------------


def tier_share_tokens(
    trace: Iterable[Entry], flooder: str, tier_blocks: int, block_size: int
) -> float:
    """The flooder's share of a tier of ``tier_blocks`` blocks, in tokens (spec 5.3).

    ``tier * w / sum(w)`` with ``w = priority + 1`` over every key in the trace and the
    flooder (priority 5). The spec counts only the keys holding blocks at the time, so
    this share is the smallest the flooder could have: the flood is not overstated.
    """
    weights = {str(e["key"]): int(e.get("priority", 5)) + 1 for e in trace}
    weights[flooder] = 6
    return tier_blocks * block_size * weights[flooder] / sum(weights.values())


def flood(
    trace: Iterable[Entry],
    key: str,
    factor: float,
    *,
    share_tokens: float | None = None,
    victim: str | None = None,
    gap_s: float = GAP_S,
) -> list[Entry]:
    """``trace`` with a flooding ``key`` added: long distinct prompts at ``factor`` times
    its share.

    G10's scenario: give ``share_tokens`` (``tier_share_tokens``) and the ``victim``.
    Then the flooder sends ``factor * share_tokens`` tokens inside every pause of the
    victim's longer than ``gap_s`` (the turns that resume), spread evenly between its
    two turns. Without a victim the prompts are spread over the whole trace; without
    ``share_tokens`` the share is the trace's tokens over its keys (the flooder counted).
    Each prompt is as long as the trace's longest and shares no block with any other;
    a prompt at the same time as a trace entry comes after it. Deterministic.
    """
    entries = list(trace)
    if not entries:
        return []
    if share_tokens is None:
        keys = {e["key"] for e in entries} | {key}
        share_tokens = sum(len(e["tokens"]) for e in entries) / len(keys)
    length = max(len(e["tokens"]) for e in entries)
    count = max(1, math.ceil(factor * share_tokens / length))
    if victim is None:
        t0, t1 = min(e["t"] for e in entries), max(e["t"] for e in entries)
        times = [t0 + (i + 0.5) * (t1 - t0) / count for i in range(count)]
    else:
        mine = [float(e["t"]) for e in entries if e["key"] == victim]
        times = []
        for a, b in itertools.pairwise(mine):
            if b - a > gap_s:
                times += [a + (j + 1) * (b - a) / (count + 1) for j in range(count)]
    lo, hi = FLOOD_TOKENS
    added = []
    for i, t in enumerate(times):
        rng = random.Random(f"lmw-kv-flood:{key}:{i}")
        added.append(
            {
                "t": t,
                "key": key,
                "isolated": False,
                "priority": 5,
                "tokens": [rng.randrange(lo, hi) for _ in range(length)],
            }
        )
    tagged = [(e["t"], 0, n, e) for n, e in enumerate(entries)]
    tagged += [(e["t"], 1, n, e) for n, e in enumerate(added)]
    return [e for *_, e in sorted(tagged, key=lambda x: x[:3])]


# ---- build --------------------------------------------------------------------------


def read_recordings(directory: Path, *, all_status: bool = False) -> list[dict[str, Any]]:
    """The recordings in ``directory`` (``*.json``) as ``{ts, headers, body}``, by time.

    ``headers`` are lower-cased (a list of pairs or an object). A file that is not a
    recording (no numeric ``ts``, unreadable, malformed headers), a body that is not a
    JSON object, and (unless ``all_status``) a request that did not succeed, is left out.
    """
    out: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(rec, dict) or rec.get("method", "POST") != "POST":
            continue
        status = rec.get("status")
        if not all_status and not (isinstance(status, int) and 200 <= status < 300):
            continue
        ts = rec.get("ts")
        if isinstance(ts, bool) or not isinstance(ts, int | float):
            continue
        rb = rec.get("request_body") or {}
        text = rb.get("text") if isinstance(rb, dict) else None
        try:
            if not text and isinstance(rb, dict) and rb.get("b64"):
                text = base64.b64decode(rb["b64"]).decode("utf-8", "replace")
            body = json.loads(text) if isinstance(text, str) and text else None
        except ValueError:  # bad base64 or JSON (binascii.Error is a ValueError)
            continue
        if not isinstance(body, dict):
            continue
        headers = _headers(rec.get("request_headers"))
        if headers is None:
            continue
        out.append({"ts": float(ts), "headers": headers, "body": body})
    out.sort(key=lambda r: r["ts"])
    return out


def _headers(raw: Any) -> dict[str, str] | None:
    """Recorded headers as a lower-cased dict; None when they are neither pairs nor an object."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(k).lower(): str(v) for k, v in raw.items()}
    if isinstance(raw, list) and all(isinstance(p, list | tuple) and len(p) == 2 for p in raw):
        return {str(k).lower(): str(v) for k, v in raw}
    return None


def _text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "".join(_text(p) for p in x)
    if isinstance(x, dict):
        kind = x.get("type")
        if kind in ("thinking", "redacted_thinking", "reasoning"):
            return ""
        if kind in ("text", "input_text", "output_text"):
            return str(x.get("text", ""))
        if kind == "tool_use":
            call = {"name": x.get("name"), "arguments": x.get("input")}
            return "<tool_call>\n" + json.dumps(call) + "\n</tool_call>"
        if kind == "tool_result":
            return "<tool_response>\n" + _text(x.get("content")) + "\n</tool_response>"
        if kind in ("image", "image_url", "input_image"):
            return "<|vision_pad|>"
        return json.dumps(x, sort_keys=True)
    return json.dumps(x)


def _messages(body: Mapping[str, Any]) -> list[tuple[str, str]]:
    msgs: list[tuple[str, str]] = []
    system = _text(body.get("system")) or _text(body.get("instructions"))
    tools = body.get("tools")
    if tools:
        system += "\n\n# Tools\n" + "\n".join(json.dumps(t) for t in tools)
    if system:
        msgs.append(("system", system))
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = "system" if m.get("role") == "developer" else str(m.get("role", "user"))
        text = _text(m.get("content"))
        for call in m.get("tool_calls") or []:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            text += "\n<tool_call>\n" + json.dumps(fn) + "\n</tool_call>"
        msgs.append((role, text))
    inp = body.get("input")
    if isinstance(inp, str):
        msgs.append(("user", inp))
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                continue
            kind = item.get("type", "message")
            if kind == "message":
                role = "system" if item.get("role") == "developer" else str(item.get("role"))
                msgs.append((role, _text(item.get("content"))))
            elif kind == "function_call":
                call = {"name": item.get("name"), "arguments": item.get("arguments")}
                msgs.append(("assistant", "<tool_call>\n" + json.dumps(call) + "\n</tool_call>"))
            elif kind == "function_call_output":
                out = "<tool_response>\n" + _text(item.get("output")) + "\n</tool_response>"
                msgs.append(("tool", out))
            elif kind != "reasoning":
                msgs.append(("user", _text(item)))
    return msgs


def render(body: Mapping[str, Any]) -> str:
    """A request body as ChatML text, ending with the assistant's open turn."""
    parts = [f"<|im_start|>{role}\n{text}<|im_end|>\n" for role, text in _messages(body)]
    return "".join(parts) + "<|im_start|>assistant\n"


def build_trace(
    records: Iterable[Mapping[str, Any]],
    encode: Callable[[str], list[int]],
    *,
    key_by: str = "session",
    priority: int = 5,
) -> list[Entry]:
    """Recordings (``read_recordings``) to trace entries; keys numbered by first use.

    A request without the header that names its key (no session id, or no
    Authorization) gets a key of its own: nothing says it shares a key with another.
    """
    names: dict[str, str] = {}  # the header value -> its name, in memory only
    out: list[Entry] = []
    t0: float | None = None
    for n, r in enumerate(records):
        headers = r["headers"]
        if key_by == "auth":
            ident = headers.get("authorization", "")
        else:
            ident = next((headers[h] for h in SESSION_HEADERS if headers.get(h)), "")
        ident = ident or f"\x00unnamed:{n}"  # never a header value
        name = names.setdefault(ident, f"k{len(names) + 1:02d}")
        t0 = r["ts"] if t0 is None else t0
        out.append(
            {
                "t": round(r["ts"] - t0, 3),
                "key": name,
                "isolated": False,
                "priority": priority,
                "tokens": list(encode(render(r["body"]))),
            }
        )
    return out


def _load_tokenizer(path: Path) -> Callable[[str], list[int]]:
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(path))
    return lambda text: list(tk.encode(text, add_special_tokens=False).ids)


# ---- live ---------------------------------------------------------------------------


def live(
    trace: Iterable[Entry],
    keys: Mapping[str, Mapping[str, str]],
    *,
    model: str,
    admin_token: str,
    client: Any,
    block_size: int,
    gap_s: float = GAP_S,
    wait_ms: int = 5000,
) -> dict[str, Any]:
    """Send ``trace`` through a running server (module docstring, ``live``).

    ``client`` is an ``httpx.Client`` with the server as its ``base_url``. Each turn is
    read back by its ``X-LMW-Request-Id`` from ``GET /api/requests/{id}`` (its history
    row's ``tiers``) and scored as ``sim`` scores it, with the engine's GPU tokens
    counted as reused too; ``ram_tokens`` and ``nvme_tokens`` say what the store served.
    """
    admin = {"Authorization": f"Bearer {admin_token}"}
    entries = list(trace)
    used = sorted({str(e["key"]) for e in entries})
    missing = [k for k in used if k not in keys]
    if missing:
        raise ValueError(f"keys.json has no entry for {len(missing)} trace keys")

    def window(key: str) -> dict[str, int]:
        r = client.get(f"/api/tokens/{keys[key]['token_id']}/cache", headers=admin)
        r.raise_for_status()
        w = r.json()["windows"]["24h"]
        return {c: int(w.get(c) or 0) for c in ("gpu", "ram", "nvme", "recomputed")}

    before = {k: window(k) for k in used}
    turns = TurnScore(block_size, gap_s)
    sent: list[tuple[str, int, bool, int, str | None]] = []
    cached = dict.fromkeys(used, 0)
    errors = 0
    for e in entries:
        k = str(e["key"])
        expected, resumed = turns.expect(k, e["t"], e["tokens"])
        body = {"model": model, "prompt": list(e["tokens"]), "max_tokens": 1}
        r = client.post(
            "/v1/completions", json=body, headers={"Authorization": f"Bearer {keys[k]['token']}"}
        )
        if r.status_code != 200:
            errors += 1
            continue
        details = (r.json().get("usage") or {}).get("prompt_tokens_details") or {}
        cached[k] += int(details.get("cached_tokens") or 0)
        sent.append((k, expected, resumed, len(e["tokens"]), r.headers.get(REQUEST_ID_HEADER)))
    unscored = 0
    for k, expected, resumed, n, rid in sent:
        tiers = _turn_tiers(client, admin, rid, wait_ms)
        if tiers is None:
            unscored += 1
            continue
        gpu, ram, nvme = (int(tiers.get(t) or 0) for t in ("gpu", "ram", "nvme"))
        row = turns.row(k)
        row["ram_tokens"] += ram
        row["nvme_tokens"] += nvme
        row["recomputed_tokens"] += max(0, n - gpu - ram - nvme)
        row["gpu_tokens"] = row.get("gpu_tokens", 0) + gpu
        turns.score(k, expected, resumed, gpu + ram + nvme, store_tokens=ram + nvme)
    after = {k: window(k) for k in used}
    usage = client.get("/api/kv/usage", headers=admin)
    usage.raise_for_status()
    for k in used:
        row = turns.row(k)
        row["cached_tokens"] = cached[k]
        row["window_24h"] = {c: after[k][c] - before[k][c] for c in after[k]}
    return {
        "requests": len(entries),
        "errors": errors,
        "unscored": unscored,
        "by_key": dict(sorted(turns.rows.items())),
        "usage": usage.json(),
    }


def _turn_tiers(client: Any, admin: Mapping[str, str], rid: str | None, wait_ms: int) -> Any:
    """A turn's ``tiers`` from its history row, or None (no id, not found, no tiers, or
    gpu, ram and nvme all null: the row says nothing about what served the turn)."""
    if not rid:
        return None
    r = client.get(f"/api/requests/{rid}", params={"wait_ms": wait_ms}, headers=admin)
    if r.status_code != 200:
        return None
    tiers = r.json().get("tiers")
    if not isinstance(tiers, dict) or all(tiers.get(t) is None for t in ("gpu", "ram", "nvme")):
        return None
    return tiers


# ---- command line -------------------------------------------------------------------


def read_trace(path: Path) -> list[Entry]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _add_flood_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--flood", metavar="KEY", help="add a flooding key (G10)")
    p.add_argument("--factor", type=float, default=10.0)
    p.add_argument("--flood-victim", metavar="KEY", help="flood inside this key's resume pauses")
    p.add_argument("--gap-s", type=float, default=GAP_S, help="a longer pause is a resume")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="KV trace build and replay (gates G9, G10)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="wire recordings to a trace")
    b.add_argument("--recordings", type=Path, required=True)
    b.add_argument("--tokenizer", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--key-by", choices=("session", "auth"), default="session")
    b.add_argument("--priority", type=int, default=5)
    b.add_argument("--all-status", action="store_true")
    s = sub.add_parser("sim", help="replay a trace on an in-process store")
    s.add_argument("--trace", type=Path, required=True)
    s.add_argument("--policy", choices=("lru", "fair"), required=True)
    s.add_argument("--ram-blocks", type=int, required=True)
    s.add_argument("--nvme-segments", type=int, required=True)
    s.add_argument("--segment-blocks", type=int, required=True)
    s.add_argument("--block-size", type=int, required=True)
    _add_flood_args(s)
    s.add_argument("--flood-share", choices=("tier", "traffic"), default="tier")
    s.add_argument("--anonymous", action="store_true")
    s.add_argument("--workdir", type=Path, help="where the store's files go (a tmpfs is fastest)")
    lv = sub.add_parser("live", help="replay a trace through a running server")
    lv.add_argument("--trace", type=Path, required=True)
    lv.add_argument("--url", required=True)
    lv.add_argument("--keys", type=Path, required=True)
    lv.add_argument("--model", required=True)
    lv.add_argument("--block-size", type=int, required=True)
    _add_flood_args(lv)
    lv.add_argument("--flood-share-tokens", type=float, help="the flooder's tier share, tokens")
    lv.add_argument("--admin-token-env", default="LMW_ADMIN_TOKEN")
    lv.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        recs = read_recordings(a.recordings, all_status=a.all_status)
        hdrs = ("authorization",) if a.key_by == "auth" else SESSION_HEADERS
        unnamed = sum(1 for r in recs if not any(r["headers"].get(h) for h in hdrs))
        trace = build_trace(
            recs, _load_tokenizer(a.tokenizer), key_by=a.key_by, priority=a.priority
        )
        with a.out.open("w") as f:
            for e in trace:
                f.write(json.dumps(e, separators=(",", ":")) + "\n")
        summary = {"requests": len(trace), "keys": len({e["key"] for e in trace})}
        print(json.dumps({**summary, "unnamed": unnamed}))
        return 0
    trace = read_trace(a.trace)
    if a.cmd == "sim":
        if a.flood:
            share = None
            if a.flood_share == "tier":
                tier = max(a.ram_blocks, a.nvme_segments * a.segment_blocks)
                share = tier_share_tokens(trace, a.flood, tier, a.block_size)
            trace = flood(
                trace, a.flood, a.factor, share_tokens=share, victim=a.flood_victim, gap_s=a.gap_s
            )
        got = sim(
            trace, policy=a.policy, ram_blocks=a.ram_blocks, nvme_segments=a.nvme_segments,
            segment_blocks=a.segment_blocks, block_size=a.block_size, anonymous=a.anonymous,
            workdir=a.workdir, gap_s=a.gap_s,
        )  # fmt: skip
        print(json.dumps(got, sort_keys=True))
        return 0
    import httpx

    token = os.environ.get(a.admin_token_env, "")
    if not token:
        print(f"set the admin token in ${a.admin_token_env}", file=sys.stderr)
        return 2
    if a.flood:
        if a.flood_share_tokens is None:
            print("--flood needs --flood-share-tokens on a live server", file=sys.stderr)
            return 2
        trace = flood(
            trace, a.flood, a.factor, share_tokens=a.flood_share_tokens,
            victim=a.flood_victim, gap_s=a.gap_s,
        )  # fmt: skip
    keys = json.loads(a.keys.read_text())
    with httpx.Client(base_url=a.url, timeout=a.timeout) as client:
        got = live(
            trace, keys, model=a.model, admin_token=token, client=client,
            block_size=a.block_size, gap_s=a.gap_s,
        )  # fmt: skip
    print(json.dumps(got, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
