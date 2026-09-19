"""Load pattern: calibration, then one round (saturate -> burst -> drain ->
collect rows). See the plan, section 4, for the algorithm this follows.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypeVar

from tests.live._admin import AdminSession
from tests.live._env import LiveConfig
from tests.live._openai import OpenAIClient, RequestTiming
from tests.live._prompts import make_prompt

T = TypeVar("T")


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


@dataclass
class CalibrationResult:
    decode_rate_tps: float
    base_tokens: int
    step_tokens: int


@dataclass
class ProbeToken:
    label: str
    priority: int
    token_id: str
    #: I6 / repr hygiene: never let a bearer token reach a traceback, an
    #: assertion message, or a log line via the dataclass's default repr.
    plaintext: str = field(repr=False)


@dataclass
class RoundResult:
    index: int
    server_start: str | None = None
    saturation_snapshot: dict[str, Any] | None = None
    burst_snapshot: dict[str, Any] | None = None
    #: filler index (0..N-1, matches its stagger position) -> timing
    filler_timings: dict[int, RequestTiming] = field(default_factory=dict)
    probe_timings: dict[str, RequestTiming] = field(default_factory=dict)
    #: probe label -> finished_record dict (or None if it never appeared)
    finished_rows: dict[str, dict | None] = field(default_factory=dict)
    t_send: dict[str, float] = field(default_factory=dict)
    t_burst_end: float | None = None
    #: Excluded from clean-round assertions (A-D). True for foreign traffic
    #: contamination AND for setup_failure rounds (see below) -- both mean
    #: "this round proves nothing," which is why every A-D check is skipped.
    tainted: bool = False
    taint_reason: str | None = None
    #: I5: True iff the taint is OUR harness/environment failing (could not
    #: saturate, a probe of ours jumped the burst, the round timed out, or
    #: server rows never showed up) rather than mere foreign-traffic
    #: contamination. A setup_failure round is a HARD test failure; a
    #: foreign-only taint is excluded and reported, not failed on.
    setup_failure: bool = False
    errors: list[str] = field(default_factory=list)


class RoundTimeoutError(RuntimeError):
    pass


async def _poll_until(
    step: Callable[[], Awaitable[tuple[bool, T]]],
    *,
    interval_s: float,
    timeout_s: float,
    on_timeout: str,
    describe: Callable[[T], str] | None = None,
) -> T:
    """Call `step()` -> (done, value) until done or timeout_s elapses.

    I6: the timeout message never embeds `value` itself -- an
    /api/stats/requests snapshot carries other operators' token names and
    client IPs, and an admin-call exception's `str()` can carry the request
    URL (hostname). `describe`, when given, converts the last value into a
    pre-sanitized one-line summary (counts only) for the failure message.
    """
    deadline = time.monotonic() + timeout_s
    last_value: T | None = None
    while True:
        done, last_value = await step()
        if done:
            return last_value
        if time.monotonic() >= deadline:
            detail = ""
            if describe is not None:
                try:
                    detail = f" ({describe(last_value)})"
                except Exception:  # noqa: BLE001 -- a bad summarizer must not mask the real error
                    detail = ""
            raise RoundTimeoutError(f"{on_timeout}{detail}")
        await asyncio.sleep(interval_s)


async def calibrate(
    oai: OpenAIClient, *, cfg: LiveConfig, filler_token: str
) -> CalibrationResult:
    """One streaming probe-shaped request on the filler token: measures
    single-stream decode rate and confirms the engine accepts `ignore_eos`.
    Raises on a non-200 or a frameless response -- the one unverified engine
    capability the plan calls out (section 4.2)."""
    prompt = make_prompt(seed=0, target_tokens=64, tag="calib")
    timing = await oai.stream_chat(
        label="calib",
        token=filler_token,
        model=cfg.model,
        prompt=prompt,
        max_tokens=64,
        priority_seed=0,
        ignore_eos=True,
        temperature=0.0,
    )
    if timing.status != 200 or timing.error:
        # Status code only -- `timing.error` can be the raw engine response
        # BODY (or an exception's str()), neither vetted for what it might
        # contain; never let it reach a report or a raised message.
        raise RuntimeError(f"calibration request failed: status={timing.status}")
    if timing.t_first_frame is None or timing.t_done is None:
        raise RuntimeError("calibration request produced no frames")
    elapsed = timing.t_done - timing.t_first_frame
    n_generated = (timing.usage or {}).get("completion_tokens")
    if not n_generated:
        n_generated = max(1, timing.n_frames - 1)
    rate = n_generated / elapsed if elapsed > 0 else float(n_generated)
    base_tokens = min(max(math.ceil(rate * cfg.hold_s), 64), cfg.filler_max_tokens)
    step_tokens = min(max(math.ceil(rate * cfg.step_s), 8), 64)
    # Cap the STAGGER (not just each individual filler's clamped max_tokens)
    # so base + (N-1)*step never exceeds filler_max_tokens. Without this, a
    # high decode rate or a long HOLD_S pins base_tokens at (or near) the
    # cap and every filler after i=0 gets clamped to the SAME max_tokens by
    # run_round's per-filler `min(...)` -- the stagger silently collapses to
    # nothing and fillers finish in a herd instead of incrementally, which
    # is the one thing the whole saturate/burst/drain shape depends on.
    n = max(1, cfg.max_inflight)
    span = cfg.filler_max_tokens - base_tokens
    max_step_for_span = span // (n - 1) if n > 1 else step_tokens
    if step_tokens > max_step_for_span:
        step_tokens = max(1, max_step_for_span)
    return CalibrationResult(decode_rate_tps=rate, base_tokens=base_tokens, step_tokens=step_tokens)


async def run_round(
    *,
    index: int,
    cfg: LiveConfig,
    admin: AdminSession,
    oai: OpenAIClient,
    calib: CalibrationResult,
    model_row_id: str | None,
    filler: ProbeToken,
    probes: list[ProbeToken],
) -> RoundResult:
    result = RoundResult(index=index)
    n_fillers = cfg.max_inflight
    own_ids = {p.token_id for p in probes} | {filler.token_id}

    # 1. Quiet check. A single transient admin-API error must not abort the
    # round -- only the deadline does.
    async def _quiet() -> tuple[bool, dict | None]:
        try:
            snap = await admin.live_requests()
        except Exception:  # noqa: BLE001 -- transient; _poll_until's deadline bounds the retry
            return False, None
        n = sum(1 for r in snap["requests"] if r["model"] == cfg.model)
        return n == 0, snap

    try:
        snap = await _poll_until(
            _quiet,
            interval_s=1.0,
            timeout_s=30.0,
            on_timeout="round did not go quiet",
            describe=lambda s: f"count={s['count']}" if s else "admin API unreachable",
        )
    except RoundTimeoutError as exc:
        result.errors.append(str(exc))
        result.tainted = True
        result.taint_reason = "not quiet before round start"
        return result
    result.server_start = snap["ts"]

    filler_tasks: list[asyncio.Task] = []
    probe_tasks: dict[str, asyncio.Task] = {}

    async def _cancel_all() -> None:
        for t in (*filler_tasks, *probe_tasks.values()):
            if not t.done():
                t.cancel()
        if filler_tasks or probe_tasks:
            await asyncio.gather(*filler_tasks, *probe_tasks.values(), return_exceptions=True)

    try:
        # 2. Saturate.
        for i in range(n_fillers):
            max_tokens = min(calib.base_tokens + i * calib.step_tokens, cfg.filler_max_tokens)
            prompt = make_prompt(
                seed=index * 10_000 + i, target_tokens=cfg.filler_prompt_tokens, tag=f"filler{i}"
            )
            filler_tasks.append(
                asyncio.create_task(
                    oai.stream_chat(
                        label=f"filler{i}",
                        token=filler.plaintext,
                        model=cfg.model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        priority_seed=i,
                        ignore_eos=True,
                        temperature=0.0,
                    )
                )
            )

        # 3. Confirm saturation.
        plateau_start: float | None = None

        async def _saturated() -> tuple[bool, dict | None]:
            nonlocal plateau_start
            try:
                s = await admin.live_requests()
            except Exception:  # noqa: BLE001 -- transient; deadline bounds the retry
                return False, None
            rows = [r for r in s["requests"] if r["model"] == cfg.model]
            n = len(rows)
            all_filler = bool(rows) and all(r["token_id"] == filler.token_id for r in rows)
            if n == n_fillers and all_filler:
                plateau_start = None
                return True, s
            if n < n_fillers:
                if plateau_start is None:
                    plateau_start = time.monotonic()
            else:
                plateau_start = None
            return False, s

        try:
            sat_snap = await _poll_until(
                _saturated,
                interval_s=0.2,
                timeout_s=cfg.saturation_timeout_s,
                on_timeout="could not saturate",
                describe=lambda s: (
                    f"count={sum(1 for r in s['requests'] if r['model'] == cfg.model)}/{n_fillers}"
                    if s
                    else "admin API unreachable"
                ),
            )
        except RoundTimeoutError as exc:
            await _cancel_all()
            hint = ""
            if plateau_start is not None and time.monotonic() - plateau_start >= 10:
                hint = " -- plateau stable below N for 10s: VW_LIVE_MAX_INFLIGHT is probably too high"
            elif plateau_start is None:
                hint = " -- never plateaued: VW_LIVE_MAX_INFLIGHT is probably too low"
            result.errors.append(f"{exc}{hint}")
            result.tainted = True
            result.setup_failure = True  # I5: hard failure, not a silent exclusion
            result.taint_reason = "saturation failure"
            return result

        sat_rows = [r for r in sat_snap["requests"] if r["model"] == cfg.model]
        result.saturation_snapshot = {"count": len(sat_rows), "expected": n_fillers}

        # 4. Burst -- ascending priority order, ties in label order (the
        # reverse of what strict priority must produce).
        for p in probes:
            result.t_send[p.label] = time.monotonic()
            seed = index * 10_000 + 5000 + abs(hash(p.label)) % 1000
            prompt = make_prompt(seed=seed, target_tokens=cfg.probe_prompt_tokens, tag=p.label)
            probe_tasks[p.label] = asyncio.create_task(
                oai.stream_chat(
                    label=p.label,
                    token=p.plaintext,
                    model=cfg.model,
                    prompt=prompt,
                    max_tokens=cfg.probe_max_tokens,
                    priority_seed=seed,
                    ignore_eos=True,
                    temperature=0.0,
                )
            )
            await asyncio.sleep(cfg.burst_gap_ms / 1000)
        result.t_burst_end = time.monotonic()
        await asyncio.sleep(0.1)
        burst_snap = await admin.live_requests()
        burst_rows = [r for r in burst_snap["requests"] if r["model"] == cfg.model]
        own_burst_rows = [r for r in burst_rows if r["token_id"] in own_ids]
        foreign_burst_rows = [r for r in burst_rows if r["token_id"] not in own_ids]
        all_filler_among_ours = bool(own_burst_rows) and all(
            r["token_id"] == filler.token_id for r in own_burst_rows
        )
        result.burst_snapshot = {"count": len(burst_rows), "all_filler": all_filler_among_ours}
        if foreign_burst_rows:
            # I5: foreign traffic alone is an EXCLUDABLE taint, not a hard
            # failure -- it means someone/something outside our control was
            # admitted, not that our scheduler misbehaved.
            result.tainted = True
            result.taint_reason = f"foreign traffic at burst: {len(foreign_burst_rows)} row(s)"
        if len(own_burst_rows) != n_fillers or not all_filler_among_ours:
            # One of OUR OWN probes was admitted before the whole burst was
            # sent -- that is a real scheduler/harness bug, always a hard
            # failure regardless of any foreign contamination above.
            result.tainted = True
            result.setup_failure = True
            extra = "A2 violated: one of our own probes was admitted before the burst finished"
            result.taint_reason = (
                f"{result.taint_reason}; {extra}" if result.taint_reason else extra
            )

        # 5. Drain.
        all_results = await asyncio.wait_for(
            asyncio.gather(*filler_tasks, *probe_tasks.values(), return_exceptions=True),
            timeout=cfg.round_timeout_s,
        )
    except TimeoutError:
        await _cancel_all()
        result.errors.append("round timed out waiting for fillers+probes to finish")
        result.tainted = True
        result.setup_failure = True  # I5: hard failure
        result.taint_reason = "round timeout"
        return result
    except Exception as exc:  # noqa: BLE001 -- catch-all so tasks are never leaked mid-round
        await _cancel_all()
        result.errors.append(f"unexpected error during round: {type(exc).__name__}")
        result.tainted = True
        result.setup_failure = True
        result.taint_reason = f"unexpected error: {type(exc).__name__}"
        return result

    n_f = len(filler_tasks)
    for i, r in enumerate(all_results[:n_f]):
        if isinstance(r, BaseException):
            result.errors.append(f"filler{i}: {type(r).__name__}")
        else:
            result.filler_timings[i] = r
    for label, r in zip(probe_tasks.keys(), all_results[n_f:], strict=False):
        if isinstance(r, BaseException):
            result.errors.append(f"{label}: {type(r).__name__}")
        else:
            result.probe_timings[label] = r

    # 6. Collect server rows (writer flushes >=1s after the first record).
    await asyncio.sleep(2.0)
    probe_ids = {p.token_id for p in probes}
    filler_id = filler.token_id
    server_start_dt = _parse_iso(result.server_start) if result.server_start else None

    async def _collected() -> tuple[bool, dict]:
        try:
            data = await admin.finished(model_row_id=model_row_id, limit=200)
        except Exception as exc:  # noqa: BLE001 -- transient poll failure, keep polling
            # I6: exception TYPE only -- str(exc) on an httpx error can carry
            # the full request URL (hostname), which must never reach the
            # report.
            return False, {"error": type(exc).__name__, "rows": [], "by_token": {}}
        rows = [
            r
            for r in data["requests"]
            # Parsed comparison, not string: datetime.isoformat() omits the
            # microsecond component when it is exactly zero, which makes
            # naive string comparison order some timestamps wrong (a
            # fractional-second value can sort BEFORE a whole-second one it
            # actually follows, since '.' < 'Z' lexicographically).
            if r.get("started_iso")
            and server_start_dt is not None
            and _parse_iso(r["started_iso"]) >= server_start_dt
        ]
        by_token: dict[str | None, list[dict]] = {}
        for r in rows:
            by_token.setdefault(r["token_id"], []).append(r)
        probes_ok = all(len(by_token.get(pid, [])) >= 1 for pid in probe_ids)
        fillers_ok = len(by_token.get(filler_id, [])) >= n_fillers
        return (probes_ok and fillers_ok), {"rows": rows, "by_token": by_token}

    try:
        collected = await _poll_until(
            _collected,
            interval_s=1.0,
            timeout_s=15.0,
            on_timeout="finished rows never appeared",
            describe=lambda c: (
                f"probes_seen={sum(1 for k in c['by_token'] if k in probe_ids)}/{len(probe_ids)} "
                f"fillers_seen={len(c['by_token'].get(filler_id, []))}/{n_fillers}"
            ),
        )
    except RoundTimeoutError as exc:
        result.errors.append(str(exc))
        result.tainted = True
        result.setup_failure = True  # I5: hard failure
        result.taint_reason = "missing finished rows"
        return result

    by_token = collected["by_token"]
    for p in probes:
        rows = by_token.get(p.token_id, [])
        result.finished_rows[p.label] = rows[0] if rows else None
    result.finished_rows["_fillers"] = by_token.get(filler_id, [])  # type: ignore[assignment]

    # 7. Foreign check over the finished-rows window too (the burst-time
    # check in step 4 only sees a single instant).
    foreign_rows = [r for r in collected["rows"] if r["token_id"] not in own_ids]
    if foreign_rows:
        result.tainted = True
        extra = f"foreign traffic: {len(foreign_rows)} row(s) from other tokens"
        result.taint_reason = f"{result.taint_reason}; {extra}" if result.taint_reason else extra

    return result
