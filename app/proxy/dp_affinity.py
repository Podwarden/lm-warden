"""Cache-affine replica routing for data-parallel vLLM models (#286).

vLLM's internal load balancer spreads requests across the data-parallel
replicas by load, so a conversation's prefix cache is rebuilt on whichever rank
it lands on. This module pins a conversation to one rank (a stable hash of an
affinity key) and passes the choice to vLLM in ``X-data-parallel-rank``,
spilling to the least-loaded rank when the home rank is saturated.

Everything here is pure except ``DpRoutingState``, one in-process object (single
uvicorn worker, like ``PriorityScheduler``) that keeps the per-rank in-flight
counts and lifetime counters. The hash is blake2b, never Python's salted
``hash()``: the same key must map to the same rank across restarts.
"""

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from app.proxy.session_id import (
    extract_session,
    first_user_text,
)

RANK_HEADER = "X-data-parallel-rank"

Decision = Literal["sticky", "placed", "spilled", "client_pinned", "balanced", "unrouted"]
KeySource = Literal[
    "claude_code_header",
    "anthropic_metadata",
    "x_session_id",
    "x_session_affinity",
    "session_id_header",
    "prompt_cache_key",
    "openai_user",
    "client_request_id",
    "prompt_hash",
    "none",
]


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup that works on plain dicts too."""
    want = name.lower()
    for k, v in headers.items():
        if k.lower() == want:
            return v
    return None


def affinity_key(
    body_json: dict[str, Any],
    headers: Mapping[str, str],
    token_id: str | None,
    *,
    explicit: str | None = None,
) -> tuple[str | None, KeySource]:
    """The session id (shared extractor, app/proxy/session_id.py) when the
    request carries one, else a hash of the conversation's opening turns."""
    info = extract_session(body_json, headers, explicit)
    if info.id and info.source:
        return info.id, info.source  # type: ignore[return-value]
    text = first_user_text(body_json)
    if text:
        return (
            f"{token_id or ''}\x00{text}",
            "prompt_hash",
        )
    return None, "none"


def session_id_of(
    body_json: Any, headers: Mapping[str, str], explicit_user_id: str | None
) -> str | None:
    """The client's session id for the dashboard, or None (see session_id.py).
    A prompt-hash affinity key is deliberately not a session id."""
    return extract_session(body_json, headers, explicit_user_id).id


def home_rank(key: str, dp: int) -> int:
    digest = hashlib.blake2b(key.encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dp


def _least_loaded(dp: int, in_flight: Mapping[int, int]) -> int:
    best = 0
    best_n = in_flight.get(0, 0)
    for r in range(1, dp):
        n = in_flight.get(r, 0)
        if n < best_n:
            best, best_n = r, n
    return best


def choose_rank(
    key: str | None,
    dp: int,
    in_flight: Mapping[int, int],
    threshold: int,
    *,
    home: int | None = None,
) -> tuple[int, Decision]:
    """``home`` overrides the hash home with a rank the session already owns."""
    if key is None:
        return _least_loaded(dp, in_flight), "balanced"
    if home is None:
        home = home_rank(key, dp)
    if in_flight.get(home, 0) >= threshold:
        target = _least_loaded(dp, in_flight)
        if target != home:
            return target, "spilled"
    return home, "sticky"


def client_rank(headers: Mapping[str, str], dp: int) -> int | None:
    raw = _header(headers, RANK_HEADER)
    if raw is None:
        return None
    try:
        r = int(raw.strip())
    except ValueError:
        return None
    return r if 0 <= r < dp else None


#: How much a replica's KV-cache fill (0..1) counts, in requests, when ranking
#: replicas for a NEW session: a replica at 100% KV scores like 8 extra requests.
KV_WEIGHT = 8.0
#: A KV reading older than this is ignored (in-flight counts only).
KV_MAX_AGE_S = 15.0
#: Each live session on a replica counts as this many requests when placing a
#: NEW one. vLLM's KV gauge only counts blocks held by RUNNING requests, so a
#: replica holding many idle, prefix-cached sessions looks empty to it.
LIVE_WEIGHT = 1.0
PLACEMENT_MAX_KEYS = 8192
PLACEMENT_TTL_S = 1800.0
#: A session counts as "live" on its replica for this long after its last use.
PLACEMENT_LIVE_S = 300.0


class SessionPlacement:
    """Which replica each conversation (affinity key) lives on.

    Hashing a key to a rank is sticky but blind to load: with few sessions it
    leaves replicas idle while others queue and evict. Instead a session is
    PLACED on the least-loaded replica the first time it is seen and then stays
    there, so its prefix cache is reused exactly as before.

    Bounded LRU per model (``PLACEMENT_MAX_KEYS``), entries expire
    ``PLACEMENT_TTL_S`` after last use, and ``epoch`` (variant + engine load
    generation) scopes them to one engine run, like ``PrefixMemory``. Keys are
    stored as 16-byte blake2b digests, never raw. In-process, single worker.
    """

    def __init__(
        self,
        *,
        max_keys: int = PLACEMENT_MAX_KEYS,
        ttl_s: float = PLACEMENT_TTL_S,
        live_s: float = PLACEMENT_LIVE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max_keys
        self._ttl = ttl_s
        self._live = live_s
        self._clock = clock
        self._m: dict[str, OrderedDict[tuple[str, bytes], tuple[int, float]]] = {}

    @staticmethod
    def _k(epoch: str, key: str) -> tuple[str, bytes]:
        return (
            epoch,
            hashlib.blake2b(key.encode("utf-8", "surrogatepass"), digest_size=16).digest(),
        )

    def lookup(self, model_id: str, epoch: str, key: str, dp: int) -> int | None:
        """The rank this session owns, refreshing its recency; None if unknown,
        expired, or no longer a valid rank."""
        d = self._m.get(model_id)
        if d is None:
            return None
        k = self._k(epoch, key)
        hit = d.get(k)
        if hit is None:
            return None
        rank, at = hit
        now = self._clock()
        if now - at > self._ttl or not 0 <= rank < dp:
            del d[k]
            return None
        d[k] = (rank, now)
        d.move_to_end(k)
        return rank

    def assign(self, model_id: str, epoch: str, key: str, rank: int) -> None:
        d = self._m.setdefault(model_id, OrderedDict())
        k = self._k(epoch, key)
        d[k] = (rank, self._clock())
        d.move_to_end(k)
        while len(d) > self._max:
            d.popitem(last=False)

    def live_counts(self, model_id: str, epoch: str | None = None) -> dict[int, int]:
        """Sessions with activity in the last ``live_s`` seconds, per rank. The
        map is recency-ordered, so the scan stops at the first stale entry."""
        d = self._m.get(model_id)
        counts: dict[int, int] = {}
        if not d:
            return counts
        floor = self._clock() - self._live
        for (ep, _), (rank, at) in reversed(d.items()):
            if at < floor:
                break
            if epoch is None or ep == epoch:
                counts[rank] = counts.get(rank, 0) + 1
        return counts

    def place(
        self,
        model_id: str,
        epoch: str,
        dp: int,
        in_flight: Mapping[int, int],
        kv: Mapping[int, float] | None,
    ) -> int:
        """Least loaded: in_flight + KV_WEIGHT x kv fill + LIVE_WEIGHT x live
        sessions; ties go to the rank with the fewest live sessions, then the
        lowest index. The KV term is used only when EVERY rank has a reading (a
        partial scrape would favour the unread ranks). A session just placed
        counts as live at once, so a burst of new keys spreads. O(dp)."""
        live = self.live_counts(model_id, epoch)
        use_kv = kv is not None and all(r in kv for r in range(dp))
        best = 0
        best_score: tuple[float, int] | None = None
        for r in range(dp):
            load: float = in_flight.get(r, 0) + LIVE_WEIGHT * live.get(r, 0)
            if use_kv and kv is not None:
                load += KV_WEIGHT * max(0.0, min(1.0, kv[r]))
            score = (round(load, 3), live.get(r, 0))
            if best_score is None or score < best_score:
                best, best_score = r, score
        return best

    def reset(self, model_id: str) -> None:
        self._m.pop(model_id, None)


@dataclass
class RankCounters:
    in_flight: int = 0
    placed: int = 0
    sticky: int = 0
    spilled_in: int = 0
    client_pinned: int = 0


@dataclass
class _ModelState:
    ranks: dict[int, RankCounters] = field(default_factory=dict)
    balanced: int = 0
    unrouted: int = 0
    since: str | None = None

    def rank(self, r: int) -> RankCounters:
        c = self.ranks.get(r)
        if c is None:
            c = self.ranks[r] = RankCounters()
        return c


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class DpRoutingState:
    """Per-model, per-rank in-flight counts and lifetime routing counters."""

    def __init__(self, placement: SessionPlacement | None = None) -> None:
        self._models: dict[str, _ModelState] = {}
        self.placement = placement or SessionPlacement()

    def route(
        self,
        model_id: str,
        *,
        dp: int,
        key: str | None,
        threshold: int,
        pinned: int | None,
        affinity_enabled: bool,
        epoch: str = "",
        kv: Mapping[int, float] | None = None,
    ) -> tuple[int | None, Decision]:
        """Decide the rank. ``pinned`` is a valid client rank (counted on it
        only). Increments ``in_flight`` for a non-None rank; the caller must
        ``release`` exactly once."""
        st = self._models.setdefault(model_id, _ModelState())
        if st.since is None:
            st.since = _now_iso()
        if pinned is not None:
            c = st.rank(pinned)
            c.in_flight += 1
            c.client_pinned += 1
            return pinned, "client_pinned"
        if not affinity_enabled or dp <= 1:
            st.unrouted += 1
            return None, "unrouted"
        in_flight = {r: c.in_flight for r, c in st.ranks.items()}
        home: int | None = None
        new_session = False
        if key is not None:
            # Place on first sight, then stick. Any failure here falls back to
            # the plain hash home (the previous behaviour).
            try:
                home = self.placement.lookup(model_id, epoch, key, dp)
                if home is None:
                    new_session = True
                    home = self.placement.place(model_id, epoch, dp, in_flight, kv)
                    self.placement.assign(model_id, epoch, key, home)
            except Exception:  # noqa: BLE001 -- routing must never fail a request
                home, new_session = None, False
        rank, decision = choose_rank(key, dp, in_flight, threshold, home=home)
        if new_session and decision == "sticky":
            decision = "placed"
        c = st.rank(rank)
        c.in_flight += 1
        if decision == "placed":
            c.placed += 1
        elif decision == "sticky":
            c.sticky += 1
        elif decision == "spilled":
            c.spilled_in += 1
        else:
            st.balanced += 1
        return rank, decision

    def release(self, model_id: str, rank: int | None) -> None:
        if rank is None:
            return
        st = self._models.get(model_id)
        if st is None:
            return
        c = st.ranks.get(rank)
        if c is not None and c.in_flight > 0:
            c.in_flight -= 1

    def snapshot(self, model_id: str, dp: int, epoch: str | None = None) -> dict[str, Any]:
        """``epoch`` limits ``assigned_sessions`` to the current engine run
        (display only; None counts every run)."""
        st = self._models.get(model_id) or _ModelState()
        ranks = []
        assigned = self.placement.live_counts(model_id, epoch)
        for r in range(dp):
            c = st.ranks.get(r) or RankCounters()
            ranks.append(
                {
                    "rank": r,
                    "in_flight": c.in_flight,
                    "assigned_sessions": assigned.get(r, 0),
                    "placed": c.placed,
                    "sticky": c.sticky,
                    "spilled_in": c.spilled_in,
                    "client_pinned": c.client_pinned,
                }
            )
        return {
            "since": st.since,
            "totals": {
                "in_flight": sum(x["in_flight"] for x in ranks),
                "placed": sum(x["placed"] for x in ranks),
                "sticky": sum(x["sticky"] for x in ranks),
                "spilled": sum(x["spilled_in"] for x in ranks),
                "client_pinned": sum(x["client_pinned"] for x in ranks),
                "balanced": st.balanced,
                "unrouted": st.unrouted,
            },
            "ranks": ranks,
        }

    def reset(self, model_id: str) -> None:
        self._models.pop(model_id, None)
        self.placement.reset(model_id)


class PrefixMemory:
    """What the proxy last sent a conversation to which replica, so an in-flight
    request's prefix-cache hit can be ESTIMATED.

    vLLM reports ``cached_tokens`` only on a stream's final usage chunk, so a
    request still running has no measured hit. The best available proxy: the
    same conversation (affinity key) last had a prompt of N tokens processed on
    the same rank, so about min(N, prompt) tokens are probably still cached
    there. Only the PROMPT is remembered: reasoning is stripped from re-sent
    history, so the previous completion is not part of the next prefix. A
    shorter new prompt means the conversation diverged or was compacted, and
    gives no estimate.

    ``epoch`` scopes an entry to one engine run (variant id + the supervisor's
    load generation), so a restart or variant swap cannot inherit a cache that
    no longer exists. Keys are stored as a 16-byte blake2b digest, never raw
    (they can be an Anthropic user_id). Bounded LRU with a TTL; in-process like
    ``DpRoutingState``.
    """

    def __init__(
        self,
        *,
        max_entries: int = 4096,
        ttl_s: float = 900.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max_entries
        self._ttl = ttl_s
        self._clock = clock
        self._d: OrderedDict[tuple[str, str, bytes], tuple[int | None, int, int, float]] = (
            OrderedDict()
        )

    @staticmethod
    def _k(model_id: str, key: str, epoch: str) -> tuple[str, str, bytes]:
        digest = hashlib.blake2b(key.encode("utf-8", "surrogatepass"), digest_size=16).digest()
        return (model_id, epoch, digest)

    def remember(
        self,
        model_id: str,
        key: str,
        rank: int | None,
        prompt_tokens: int,
        engine_prompt: int | None = None,
        *,
        epoch: str = "",
    ) -> None:
        """``prompt_tokens`` is the warden's own pre-forward count;
        ``engine_prompt`` the engine's, when it reported one (else the warden's)."""
        k = self._k(model_id, key, epoch)
        warden = max(0, int(prompt_tokens))
        engine = warden if engine_prompt is None else max(0, int(engine_prompt))
        self._d[k] = (rank, warden, engine, self._clock())
        self._d.move_to_end(k)
        while len(self._d) > self._max:
            self._d.popitem(last=False)

    def estimate(
        self,
        model_id: str,
        key: str,
        rank: int | None,
        prompt_tokens: int,
        *,
        epoch: str = "",
    ) -> int | None:
        return self.estimate_with_gap(model_id, key, rank, prompt_tokens, epoch=epoch)[0]

    def estimate_with_gap(
        self,
        model_id: str,
        key: str,
        rank: int | None,
        prompt_tokens: int,
        *,
        epoch: str = "",
    ) -> tuple[int | None, float | None]:
        """Tokens of the prompt likely cached on ``rank``, in ENGINE units;
        None = no basis. ``prompt_tokens`` is the warden's pre-forward count and
        is only compared with the previous warden count (like with like): a
        tokenizer-less GGUF is counted from characters, ~30% below the engine's
        figure, so comparing it with a remembered engine count would always
        read as "shrunk"."""
        k = self._k(model_id, key, epoch)
        hit = self._d.get(k)
        if hit is None:
            return None, None
        mem_rank, prev_warden, prev_engine, at = hit
        gap = self._clock() - at
        if gap > self._ttl:
            del self._d[k]
            return None, None
        if mem_rank != rank or prompt_tokens < prev_warden or prev_warden <= 0:
            return None, None
        self._d.move_to_end(k)
        return min(prev_engine, round(prompt_tokens * prev_engine / prev_warden)), gap

    def __len__(self) -> int:
        return len(self._d)
