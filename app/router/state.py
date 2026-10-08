"""In-process router state (#287): rule-set cache, breakers, counters, ring.

One instance on ``app.state.router``. Process-local on purpose, like
``PriorityScheduler`` and ``DpRoutingState``: the warden runs a single uvicorn
worker. Nothing here may raise into a proxied request -- the mutators swallow
and debug-log their own errors.
"""

import asyncio
import logging
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.db.repos.router_rules import RouterRuleRepo
from app.db.repos.settings import SettingsRepo
from app.router.rules import ROUTER_KEYS, RouterConfig, RuleSet, RuleSpec, config_from_kv

logger = logging.getLogger(__name__)

Route = Literal["local", "passthrough", "fallback", "refused", "error"]
_ROUTES: tuple[str, ...] = ("local", "passthrough", "fallback", "refused", "error")
RING_SIZE = 200
SAMPLE_SIZE = 512
MODEL_IN_MAX = 128
#: A half-open probe that never reports back (its stream was abandoned before
#: it started) stops blocking the next probe after its own local timeout plus
#: this margin; ``PROBE_TTL_S`` is the fallback when no timeout is given.
PROBE_TTL_S = 120.0
PROBE_MARGIN_S = 30.0


@dataclass
class Decision:
    ts: float
    path: str
    model_in: str | None
    model_out: str | None
    rule_id: str | None
    route: Route
    reason: str | None
    status: int
    latency_ms: int
    ttfb_ms: int | None
    stream: bool
    token_name: str | None


@dataclass
class _Breaker:
    consecutive: int = 0
    open_until: float | None = None
    last_reason: str | None = None
    failures: int = 0
    probe_claim: int | None = None  # the half-open probe in flight, if any
    probe_expires: float = 0.0


@dataclass
class _RuleCounters:
    local: int = 0
    fallback: int = 0
    refused: int = 0


def _iso(ts: float, *, millis: bool = False) -> str:
    dt = datetime.fromtimestamp(ts, UTC)
    if millis:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _pct(samples: "deque[int] | list[int]", p: int) -> int | None:
    """Nearest-rank percentile; None without samples."""
    if not samples:
        return None
    ordered = sorted(samples)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def _pcts(samples: "deque[int] | list[int]") -> dict[str, int | None]:
    return {"p50": _pct(samples, 50), "p95": _pct(samples, 95)}


class RouterState:
    def __init__(self, *, now: Callable[[], float] = time.time) -> None:
        self._now = now
        self._dirty = True
        self._ruleset: RuleSet | None = None
        self._version = 0
        self._load_lock = asyncio.Lock()
        self._probe_seq = 0
        self._clear()

    def _clear(self) -> None:
        self._breakers: dict[str, _Breaker] = {}
        self._since: float | None = None
        self._totals: dict[str, int] = dict.fromkeys(_ROUTES, 0)
        self._by_reason: dict[str, int] = {}
        self._error_reasons: dict[str, int] = {}
        self._rule_counters: dict[str, _RuleCounters] = {}
        self._rule_patterns: dict[str, str] = {}
        self._rule_latency: dict[str, deque[int]] = {}
        self._rule_ttfb: dict[str, deque[int]] = {}
        self._pt_requests = 0
        self._pt_errors = 0
        self._pt_latency: deque[int] = deque(maxlen=SAMPLE_SIZE)
        self._pt_ttfb: deque[int] = deque(maxlen=SAMPLE_SIZE)
        self._pt_in = 0
        self._pt_out = 0
        self._ring: deque[Decision] = deque(maxlen=RING_SIZE)

    # -- rule-set cache ----------------------------------------------------

    def invalidate(self) -> None:
        """Mark the cached rule set stale (every control-API write calls this)."""
        self._dirty = True

    async def ruleset(self, db_path: Path) -> RuleSet:
        """The current rule set; reloaded from the DB only while dirty."""
        rs = self._ruleset
        if rs is not None and not self._dirty:
            return rs
        async with self._load_lock:
            rs = self._ruleset
            if rs is not None and not self._dirty:
                return rs
            # Clear BEFORE loading so an invalidate() during the load is not lost.
            self._dirty = False
            try:
                rs = await self._load(db_path)
            except BaseException:
                self._dirty = True
                raise
            self._ruleset = rs
            return rs

    async def _load(self, db_path: Path) -> RuleSet:
        async with open_db(db_path) as db:
            kv = await SettingsRepo(db).get_many(list(ROUTER_KEYS))
            rows = await RouterRuleRepo(db).list_all()
            models = ModelRepo(db)
            served: dict[str, str | None] = {}
            for r in rows:
                if r.target_model_id not in served:
                    m = await models.get(r.target_model_id)
                    served[r.target_model_id] = m.served_model_name if m else None
        self._version += 1
        return RuleSet(
            config=config_from_kv(kv),
            rules=tuple(
                RuleSpec(
                    id=r.id,
                    position=r.position,
                    pattern=r.pattern,
                    target_model_id=r.target_model_id,
                    enabled=r.enabled,
                    fallback=r.fallback,
                    strip_thinking=r.strip_thinking,
                    min_max_tokens=r.min_max_tokens,
                )
                for r in rows
            ),
            served_names=served,
            version=self._version,
        )

    # -- breakers ----------------------------------------------------------

    def breaker_allows(self, model_id: str) -> bool:
        """Peek: would a request be admitted right now? Claims nothing."""
        b = self._breakers.get(model_id)
        if b is None or b.open_until is None:
            return True
        now = self._now()
        if now < b.open_until:
            return False
        return b.probe_claim is None or now >= b.probe_expires

    def breaker_admit(self, model_id: str, *, timeout_s: float | None = None) -> int | None:
        """Admit one request. None: refused. 0: closed, no claim. >0: you are
        the half-open probe (decision 15: only the next request is) and must
        end up in ``record_success`` / ``record_failure`` / ``release_probe``.
        The claim lapses after ``timeout_s`` (the probe's local timeout) plus a
        margin, so a hung engine cannot admit a second probe early."""
        b = self._breakers.get(model_id)
        if b is None or b.open_until is None:
            return 0
        now = self._now()
        if now < b.open_until:
            return None
        if b.probe_claim is not None and now < b.probe_expires:
            return None
        self._probe_seq += 1
        b.probe_claim = self._probe_seq
        b.probe_expires = now + (PROBE_TTL_S if timeout_s is None else timeout_s + PROBE_MARGIN_S)
        return self._probe_seq

    def release_probe(self, model_id: str, claim: int) -> None:
        """Give the probe slot back without an outcome (idempotent, per claim)."""
        b = self._breakers.get(model_id)
        if b is not None and b.probe_claim == claim:
            b.probe_claim = None

    def record_success(self, model_id: str) -> None:
        b = self._breakers.get(model_id)
        if b is not None:
            b.consecutive = 0
            b.open_until = None
            b.probe_claim = None

    def record_failure(self, model_id: str, reason: str, *, threshold: int, open_s: int) -> bool:
        """Count a breaker failure; True when this one opened (or re-opened) it."""
        b = self._breakers.setdefault(model_id, _Breaker())
        now = self._now()
        b.consecutive += 1
        b.failures += 1
        b.last_reason = reason
        b.probe_claim = None
        half_open = b.open_until is not None and now >= b.open_until
        currently_open = b.open_until is not None and not half_open
        if currently_open:
            return False
        if half_open or b.consecutive >= threshold:
            b.open_until = now + open_s
            return True
        return False

    def _breaker_state(self, b: _Breaker | None) -> str:
        if b is None or b.open_until is None:
            return "closed"
        return "open" if self._now() < b.open_until else "half_open"

    # -- recording ---------------------------------------------------------

    def record(self, d: Decision, *, rule_pattern: str | None = None) -> None:
        """Fold one decision into counters, samples and the ring. Never raises."""
        try:
            latency = int(d.latency_ms)
            ttfb = None if d.ttfb_ms is None else int(d.ttfb_ms)
            route = d.route
            if route not in self._totals:
                return
            if self._since is None:
                self._since = self._now()
            self._ring.append(d)
            self._totals[route] += 1
            if d.reason:
                # by_reason answers "why did a rule fall back / refuse"; reasons
                # on other routes (local/passthrough/error) are errors, kept apart.
                bucket = (
                    self._by_reason if route in ("fallback", "refused") else self._error_reasons
                )
                bucket[d.reason] = bucket.get(d.reason, 0) + 1
            if d.rule_id is not None:
                rid = d.rule_id
                if rule_pattern is not None:
                    self._rule_patterns[rid] = rule_pattern
                rc = self._rule_counters.setdefault(rid, _RuleCounters())
                if route == "local":
                    rc.local += 1
                elif route == "fallback":
                    rc.fallback += 1
                elif route == "refused":
                    rc.refused += 1
                if route in ("local", "fallback"):
                    self._rule_latency.setdefault(rid, deque(maxlen=SAMPLE_SIZE)).append(latency)
                    if ttfb is not None:
                        self._rule_ttfb.setdefault(rid, deque(maxlen=SAMPLE_SIZE)).append(ttfb)
            if route in ("passthrough", "error"):
                self._pt_requests += 1
                if route == "error" or d.status >= 400:
                    self._pt_errors += 1
                self._pt_latency.append(latency)
                if ttfb is not None:
                    self._pt_ttfb.append(ttfb)
        except Exception:
            logger.debug("router: record failed", exc_info=True)

    def add_passthrough_tokens(self, input_tokens: int | None, output_tokens: int | None) -> None:
        """Best-effort Anthropic usage sniffed from a passthrough response."""
        try:
            self._pt_in += int(input_tokens or 0)
            self._pt_out += int(output_tokens or 0)
        except Exception:
            logger.debug("router: add_passthrough_tokens failed", exc_info=True)

    # -- reads -------------------------------------------------------------

    def snapshot(self, ruleset: RuleSet | None) -> dict[str, Any]:
        """The ``RouterStatsOut`` dict."""
        rules: list[dict[str, Any]] = []
        targets: list[dict[str, Any]] = []
        if ruleset is not None:
            seen: set[str] = set()
            for r in ruleset.rules:
                rc = self._rule_counters.get(r.id) or _RuleCounters()
                served = ruleset.served_names.get(r.target_model_id)
                rules.append(
                    {
                        "rule_id": r.id,
                        "pattern": r.pattern,
                        "target_model_id": r.target_model_id,
                        "target_served_name": served,
                        "local": rc.local,
                        "fallback": rc.fallback,
                        "refused": rc.refused,
                        "latency_ms": _pcts(self._rule_latency.get(r.id, [])),
                        "ttfb_ms": _pcts(self._rule_ttfb.get(r.id, [])),
                    }
                )
                if r.target_model_id in seen:
                    continue
                seen.add(r.target_model_id)
                b = self._breakers.get(r.target_model_id)
                targets.append(
                    {
                        "model_id": r.target_model_id,
                        "served_name": served,
                        "breaker": self._breaker_state(b),
                        "consecutive_failures": b.consecutive if b else 0,
                        "open_until": _iso(b.open_until) if b and b.open_until else None,
                        "last_reason": b.last_reason if b else None,
                        "failures": b.failures if b else 0,
                    }
                )
        return {
            "enabled": bool(ruleset.config.enabled) if ruleset is not None else False,
            "since": _iso(self._since) if self._since is not None else None,
            "totals": dict(self._totals),
            "by_reason": dict(self._by_reason),
            "error_reasons": dict(self._error_reasons),
            "rules": rules,
            "targets": targets,
            "passthrough": {
                "requests": self._pt_requests,
                "errors": self._pt_errors,
                "latency_ms": _pcts(self._pt_latency),
                "ttfb_ms": _pcts(self._pt_ttfb),
                "input_tokens": self._pt_in,
                "output_tokens": self._pt_out,
            },
        }

    def decisions(self, limit: int) -> list[dict[str, Any]]:
        """Newest first, at most ``limit`` (the ring holds 200)."""
        out: list[dict[str, Any]] = []
        for d in reversed(self._ring):
            if len(out) >= max(0, limit):
                break
            out.append(
                {
                    "ts": _iso(d.ts, millis=True),
                    "path": d.path,
                    "model_in": d.model_in[:MODEL_IN_MAX] if d.model_in is not None else None,
                    "model_out": d.model_out,
                    "rule_id": d.rule_id,
                    "route": d.route,
                    "reason": d.reason,
                    "status": d.status,
                    "latency_ms": d.latency_ms,
                    "ttfb_ms": d.ttfb_ms,
                    "stream": d.stream,
                    "token_name": d.token_name,
                }
            )
        return out

    def reset(self) -> None:
        """Clear counters, samples, breakers and the ring (not the rule cache)."""
        self._clear()


__all__ = ["Decision", "Route", "RouterConfig", "RouterState"]
