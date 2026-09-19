"""Pure functions: admission order, FIFO, spread, Spearman, aggregate stats.

No network, no filesystem, no wall-clock reads -- everything here takes
already-collected data and returns verdicts, so it is exercised by ordinary
unit tests (tests/unit/live/test_live_analysis.py) alongside the app's own
fast suite. See the plan, section 6, for what each check proves and why.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProbeRecord:
    label: str
    priority: int
    send_order: int  #: index in send order (ascending priority, ties by label)
    t_send: float  #: client monotonic clock
    t_burst_end: float  #: client monotonic clock, shared across the round
    started_iso: str | None  #: server clock, None if never admitted
    queued_s: float | None  #: server-measured scheduler wait
    finish_reason: str | None
    orphan: bool
    status: int | None
    client_ttft_s: float | None = None
    client_done_s: float | None = None


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def admission_order(probes: list[ProbeRecord]) -> list[ProbeRecord]:
    """Probes sorted by server `started_iso` ascending. A probe never
    admitted (no started_iso) sorts last."""

    def key(p: ProbeRecord) -> tuple[int, datetime]:
        if p.started_iso:
            return (0, _parse_iso(p.started_iso))
        return (1, datetime.min)

    return sorted(probes, key=key)


def check_a3_saturated(probes: list[ProbeRecord], *, slack_s: float = 0.25) -> list[str]:
    """A3: each probe was still queued when the whole burst had been sent."""
    violations = []
    for p in probes:
        if p.queued_s is None:
            violations.append(f"{p.label}: queued_s missing")
            continue
        burst_wait = p.t_burst_end - p.t_send
        if p.queued_s < burst_wait - slack_s:
            violations.append(
                f"{p.label}: queued_s={p.queued_s:.3f} < burst_wait-slack="
                f"{burst_wait - slack_s:.3f} (engine was not saturated: raise "
                "VW_LIVE_MAX_INFLIGHT or VW_LIVE_HOLD_S)"
            )
    return violations


def check_b_strict_priority(adm: list[ProbeRecord]) -> list[str]:
    """B: priority is non-increasing in admission order. Reports every
    adjacent violating pair (zero tolerance)."""
    violations = []
    for a, b in zip(adm, adm[1:], strict=False):
        if a.priority < b.priority:
            delta = ""
            if a.started_iso and b.started_iso:
                delta_s = (_parse_iso(b.started_iso) - _parse_iso(a.started_iso)).total_seconds()
                delta = f" (Delta_started_iso={delta_s:.6f}s)"
            violations.append(
                f"{a.label}(p{a.priority}) admitted before {b.label}(p{b.priority})"
                f"{delta}: priority order violated"
            )
    return violations


def check_b_prime_client_order(adm: list[ProbeRecord]) -> list[str]:
    """B': cross-check from an independent clock. admit_client = t_send +
    queued_s; for every pair whose server gap > 0.1s the client-derived
    order must agree."""
    violations = []
    for a, b in zip(adm, adm[1:], strict=False):
        if not a.started_iso or not b.started_iso:
            continue
        gap = (_parse_iso(b.started_iso) - _parse_iso(a.started_iso)).total_seconds()
        if gap <= 0.1:
            continue
        if a.queued_s is None or b.queued_s is None:
            continue
        admit_a = a.t_send + a.queued_s
        admit_b = b.t_send + b.queued_s
        if admit_a > admit_b:
            violations.append(
                f"{a.label} vs {b.label}: server says {a.label} admitted first "
                f"(gap {gap:.3f}s) but client-derived admit time disagrees"
            )
    return violations


def check_c_fifo_within_tier(adm: list[ProbeRecord]) -> list[str]:
    """C: within each priority tier with >=2 probes, admission order equals
    send order, and the tier's probes are contiguous in admission order
    (nothing of ours interleaves)."""
    violations = []
    by_priority: dict[int, list[int]] = {}
    for idx, p in enumerate(adm):
        by_priority.setdefault(p.priority, []).append(idx)
    for prio, positions in sorted(by_priority.items()):
        tier_probes = [p for p in adm if p.priority == prio]
        if len(tier_probes) < 2:
            continue
        send_sorted = sorted(tier_probes, key=lambda p: p.send_order)
        if [p.label for p in tier_probes] != [p.label for p in send_sorted]:
            violations.append(f"priority {prio}: admission order != send order")
        contiguous = positions == list(range(positions[0], positions[0] + len(positions)))
        if not contiguous:
            violations.append(f"priority {prio}: probes not contiguous in admission order")
    return violations


def check_d1_queued_increasing_across_tiers(adm: list[ProbeRecord]) -> list[str]:
    """D1: queued_s strictly increases across tiers in admission order --
    min(queued_s) of a tier admitted later must exceed max(queued_s) of the
    tier admitted just before it."""
    violations: list[str] = []
    groups: list[tuple[int, list[float]]] = []
    for p in adm:
        if p.queued_s is None:
            continue
        if groups and groups[-1][0] == p.priority:
            groups[-1][1].append(p.queued_s)
        else:
            groups.append((p.priority, [p.queued_s]))
    for i in range(1, len(groups)):
        prev_max = max(groups[i - 1][1])
        cur_min = min(groups[i][1])
        if cur_min <= prev_max:
            violations.append(
                f"tier p{groups[i][0]} min queued_s={cur_min:.3f} <= "
                f"tier p{groups[i - 1][0]} max queued_s={prev_max:.3f}"
            )
    return violations


def check_d2_spread(adm: list[ProbeRecord], *, min_spread_s: float) -> tuple[bool, float]:
    qs = [p.queued_s for p in adm if p.queued_s is not None]
    if len(qs) < 2:
        return False, 0.0
    spread = qs[-1] - qs[0]
    return spread >= min_spread_s, spread


def check_d3_first_admitted(
    adm: list[ProbeRecord],
    *,
    hold_s: float,
    prefill_allowance_s: float = 10.0,
    filler0_duration_s: float | None = None,
) -> list[str]:
    """D3: the first-admitted probe waited only for the first slot release
    -- smallest queued_s, bounded by the ACTUAL measured duration of filler0
    (the shortest-held filler, whose completion is what frees that first
    slot) plus a prefill allowance. Falls back to the configured `hold_s`
    only when filler0's real duration isn't available (e.g. it errored) --
    `hold_s` is merely the target the calibration step aimed for, and can
    read low if the engine decoded faster than calibration measured."""
    if not adm or adm[0].queued_s is None:
        return ["no admitted probe with queued_s to check D3"]
    first = adm[0]
    reference = filler0_duration_s if filler0_duration_s is not None else hold_s
    bound = reference + prefill_allowance_s
    violations = []
    if first.queued_s > bound:
        violations.append(f"{first.label}: queued_s={first.queued_s:.3f} > bound {bound:.3f}")
    for other in adm[1:]:
        if other.queued_s is not None and other.queued_s <= first.queued_s:
            violations.append(f"{first.label} not strictly smallest queued_s (vs {other.label})")
    return violations


def check_d4_basic(probes: list[ProbeRecord]) -> list[str]:
    violations = []
    for p in probes:
        if p.queued_s is None or p.queued_s <= 0.05:
            violations.append(f"{p.label}: queued_s not > 0.05s ({p.queued_s})")
        if p.finish_reason not in ("stop", "length"):
            violations.append(f"{p.label}: finish_reason={p.finish_reason!r}")
        if p.orphan:
            violations.append(f"{p.label}: orphan=True")
        if p.status != 200:
            violations.append(f"{p.label}: status={p.status}")
    return violations


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation, pure Python (no scipy dependency in
    tests/). Ties get average rank. None if fewer than 2 points or either
    series has zero variance."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg_rank
            i = j + 1
        return r

    rx = ranks(xs)
    ry = ranks(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry, strict=False))
    den_x = sum((a - mean_rx) ** 2 for a in rx) ** 0.5
    den_y = sum((b - mean_ry) ** 2 for b in ry) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    s = sorted(values)

    def pct(p: float) -> float:
        idx = min(len(s) - 1, int(round(p * (len(s) - 1))))
        return s[idx]

    return {"p50": pct(0.5), "p95": pct(0.95), "max": s[-1]}
