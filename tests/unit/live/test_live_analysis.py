"""Pure-logic unit tests for tests/live/_analysis.py -- no network, fast.

These exercise the assertion logic the live test relies on against
hand-built fixture rows, independent of any real deployment.
"""

from __future__ import annotations

from tests.live import _analysis as analysis
from tests.live._analysis import ProbeRecord


def _rec(
    label: str,
    priority: int,
    send_order: int,
    *,
    started_iso: str | None,
    queued_s: float | None,
    t_send: float = 0.0,
    t_burst_end: float = 100.0,
    finish_reason: str | None = "stop",
    orphan: bool = False,
    status: int | None = 200,
) -> ProbeRecord:
    return ProbeRecord(
        label=label,
        priority=priority,
        send_order=send_order,
        t_send=t_send,
        t_burst_end=t_burst_end,
        started_iso=started_iso,
        queued_s=queued_s,
        finish_reason=finish_reason,
        orphan=orphan,
        status=status,
    )


def test_admission_order_sorts_by_started_iso():
    a = _rec("a", 9, 0, started_iso="2026-01-01T00:00:00.000002Z", queued_s=1.0)
    b = _rec("b", 5, 1, started_iso="2026-01-01T00:00:00.000001Z", queued_s=0.5)
    order = analysis.admission_order([a, b])
    assert [r.label for r in order] == ["b", "a"]


def test_admission_order_puts_never_admitted_last():
    a = _rec("a", 9, 0, started_iso=None, queued_s=None)
    b = _rec("b", 5, 1, started_iso="2026-01-01T00:00:00.000001Z", queued_s=0.5)
    order = analysis.admission_order([a, b])
    assert [r.label for r in order] == ["b", "a"]


def test_check_b_strict_priority_passes_when_non_increasing():
    adm = [
        _rec("p9", 9, 0, started_iso="2026-01-01T00:00:00.000001Z", queued_s=1.0),
        _rec("p5", 5, 1, started_iso="2026-01-01T00:00:00.000002Z", queued_s=2.0),
        _rec("p0", 0, 2, started_iso="2026-01-01T00:00:00.000003Z", queued_s=3.0),
    ]
    assert analysis.check_b_strict_priority(adm) == []


def test_check_b_strict_priority_flags_violation():
    adm = [
        _rec("p0", 0, 0, started_iso="2026-01-01T00:00:00.000001Z", queued_s=1.0),
        _rec("p9", 9, 1, started_iso="2026-01-01T00:00:00.000002Z", queued_s=2.0),
    ]
    violations = analysis.check_b_strict_priority(adm)
    assert len(violations) == 1
    assert "p0(p0) admitted before p9(p9)" in violations[0]


def test_check_b_prime_client_order_agrees_with_server():
    adm = [
        _rec("a", 9, 0, started_iso="2026-01-01T00:00:00.000000Z", queued_s=1.0, t_send=0.0),
        _rec("b", 5, 1, started_iso="2026-01-01T00:00:01.000000Z", queued_s=1.5, t_send=0.1),
    ]
    assert analysis.check_b_prime_client_order(adm) == []


def test_check_b_prime_client_order_flags_disagreement():
    # Server says a admitted first (gap > 0.1s), but a's client-derived
    # admit time (t_send + queued_s) is LATER than b's.
    adm = [
        _rec("a", 9, 0, started_iso="2026-01-01T00:00:00.000000Z", queued_s=5.0, t_send=0.0),
        _rec("b", 5, 1, started_iso="2026-01-01T00:00:01.000000Z", queued_s=0.1, t_send=0.0),
    ]
    violations = analysis.check_b_prime_client_order(adm)
    assert len(violations) == 1


def test_check_c_fifo_within_tier_passes_in_send_order():
    adm = [
        _rec("p0a", 0, 0, started_iso="2026-01-01T00:00:00.000001Z", queued_s=1.0),
        _rec("p0b", 0, 1, started_iso="2026-01-01T00:00:00.000002Z", queued_s=1.1),
    ]
    assert analysis.check_c_fifo_within_tier(adm) == []


def test_check_c_fifo_within_tier_flags_out_of_order():
    adm = [
        _rec("p0b", 0, 1, started_iso="2026-01-01T00:00:00.000001Z", queued_s=1.0),
        _rec("p0a", 0, 0, started_iso="2026-01-01T00:00:00.000002Z", queued_s=1.1),
    ]
    violations = analysis.check_c_fifo_within_tier(adm)
    assert any("admission order != send order" in v for v in violations)


def test_check_c_fifo_within_tier_flags_non_contiguous():
    adm = [
        _rec("p0a", 0, 0, started_iso="2026-01-01T00:00:00.000001Z", queued_s=1.0),
        _rec("p5", 5, 2, started_iso="2026-01-01T00:00:00.000002Z", queued_s=1.05),
        _rec("p0b", 0, 1, started_iso="2026-01-01T00:00:00.000003Z", queued_s=1.1),
    ]
    violations = analysis.check_c_fifo_within_tier(adm)
    assert any("not contiguous" in v for v in violations)


def test_check_d1_queued_increasing_across_tiers_passes():
    adm = [
        _rec("p9", 9, 0, started_iso="2026-01-01T00:00:00.000001Z", queued_s=0.5),
        _rec("p5", 5, 1, started_iso="2026-01-01T00:00:00.000002Z", queued_s=1.5),
        _rec("p0", 0, 2, started_iso="2026-01-01T00:00:00.000003Z", queued_s=3.0),
    ]
    assert analysis.check_d1_queued_increasing_across_tiers(adm) == []


def test_check_d1_queued_increasing_across_tiers_flags_violation():
    adm = [
        _rec("p9", 9, 0, started_iso="2026-01-01T00:00:00.000001Z", queued_s=2.0),
        _rec("p0", 0, 1, started_iso="2026-01-01T00:00:00.000002Z", queued_s=1.0),
    ]
    violations = analysis.check_d1_queued_increasing_across_tiers(adm)
    assert len(violations) == 1


def test_check_d2_spread():
    adm = [
        _rec("a", 9, 0, started_iso="x", queued_s=0.5),
        _rec("b", 0, 1, started_iso="x", queued_s=2.5),
    ]
    ok, spread = analysis.check_d2_spread(adm, min_spread_s=1.0)
    assert ok is True
    assert spread == 2.0

    ok2, spread2 = analysis.check_d2_spread(adm, min_spread_s=3.0)
    assert ok2 is False


def test_check_d3_first_admitted_within_bound():
    adm = [
        _rec("p9a", 9, 0, started_iso="x", queued_s=5.0),
        _rec("p9b", 9, 1, started_iso="x", queued_s=6.0),
    ]
    assert analysis.check_d3_first_admitted(adm, hold_s=8.0) == []


def test_check_d3_first_admitted_over_bound():
    adm = [_rec("p9a", 9, 0, started_iso="x", queued_s=100.0)]
    violations = analysis.check_d3_first_admitted(adm, hold_s=8.0, prefill_allowance_s=10.0)
    assert any("bound" in v for v in violations)


def test_check_d3_uses_filler0_duration_over_hold_s_when_given():
    # hold_s (8.0) alone would make this fail (20 > 8+10=18); the ACTUAL
    # measured filler0 duration (say the engine decoded slower than
    # calibration predicted) raises the bound and the round passes.
    adm = [_rec("p9a", 9, 0, started_iso="x", queued_s=20.0)]
    assert analysis.check_d3_first_admitted(adm, hold_s=8.0, filler0_duration_s=15.0) == []
    # hold_s fallback still applies when filler0's duration isn't available.
    violations = analysis.check_d3_first_admitted(adm, hold_s=8.0, filler0_duration_s=None)
    assert any("bound" in v for v in violations)


def test_check_d4_basic_flags_each_issue():
    bad = _rec(
        "p0", 0, 0, started_iso="x", queued_s=0.01, finish_reason="error", orphan=True, status=500
    )
    violations = analysis.check_d4_basic([bad])
    assert len(violations) == 4  # queued_s, finish_reason, orphan, status


def test_check_d4_basic_passes_clean_row():
    good = _rec("p0", 0, 0, started_iso="x", queued_s=1.0)
    assert analysis.check_d4_basic([good]) == []


def test_check_a3_saturated_passes():
    r = _rec("p0", 0, 0, started_iso="x", queued_s=10.0, t_send=0.0, t_burst_end=1.5)
    assert analysis.check_a3_saturated([r]) == []


def test_check_a3_saturated_flags_undersaturation():
    r = _rec("p0", 0, 0, started_iso="x", queued_s=0.01, t_send=0.0, t_burst_end=5.0)
    violations = analysis.check_a3_saturated([r])
    assert len(violations) == 1
    assert "not saturated" in violations[0]


def test_spearman_perfect_negative_correlation():
    priorities = [9.0, 7.0, 5.0, 3.0, 0.0]
    ttft = [0.1, 0.3, 0.5, 0.8, 1.2]
    rho = analysis.spearman(priorities, ttft)
    assert rho is not None
    assert rho < -0.99


def test_spearman_no_variance_returns_none():
    assert analysis.spearman([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) is None


def test_spearman_too_few_points_returns_none():
    assert analysis.spearman([1.0], [2.0]) is None


def test_percentiles_basic():
    p = analysis.percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
    assert p["p50"] == 3.0
    assert p["max"] == 5.0


def test_percentiles_empty():
    assert analysis.percentiles([]) == {"p50": 0.0, "p95": 0.0, "max": 0.0}
