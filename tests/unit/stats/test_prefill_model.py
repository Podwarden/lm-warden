"""The learned prefill model: rate, calibration, idle-gap decay, queue-ahead, causes."""

import sqlite3

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.proxy.dp_affinity import PrefixMemory
from app.proxy.request_registry import LiveRequest
from app.stats import prefill_model as pm
from app.stats.live_requests import _ahead_of, _serialize, slow_cause


def row(**kw):
    base = dict(
        model_id="m",
        prompt_tokens=10_000,
        cached_tokens=2_000,
        cache_est_tokens=None,
        ttft_s=4.0,
        inflight_same_rank_at_start=0,
        gap_s=None,
    )
    base.update(kw)
    return base


# ---- rate ------------------------------------------------------------------


def test_rate_is_the_median_of_fresh_over_ttft():
    rows = [row(ttft_s=t) for t in [2.0, 4.0, 8.0] * 7]  # 21 rows
    rate, n = pm.learned_rate(rows)
    assert n == 21
    assert rate == 8_000 / 4.0  # median ttft 4 s, fresh 8000


def test_rate_ignores_contended_tiny_unmeasured_and_non_stream_rows():
    good = [row() for _ in range(20)]
    noise = [
        row(inflight_same_rank_at_start=1, ttft_s=0.1),  # queue, not engine
        row(inflight_same_rank_at_start=None, ttft_s=0.1),  # not measured
        row(prompt_tokens=1_000, cached_tokens=900, ttft_s=0.1),  # fresh < 512
        row(cached_tokens=None, ttft_s=0.1),  # engine did not report the hit
        row(ttft_s=None),  # non-stream
        row(ttft_s=0),
    ]
    rate, n = pm.learned_rate(good + noise)
    assert n == 20 and rate == 2_000.0


def test_rate_needs_min_samples_else_none():
    rate, n = pm.learned_rate([row() for _ in range(19)])
    assert rate is None and n == 19


def test_state_falls_back_to_the_hint():
    st = pm.PrefillModelState()
    assert st.rate_for("m", 2000.0) == (2000.0, "hint")
    st.replace(pm.build([row() for _ in range(25)]))
    assert st.rate_for("m", 2000.0) == (2000.0, "learned")  # 8000/4
    st.replace(pm.build([row(ttft_s=2.0) for _ in range(25)]))
    assert st.rate_for("m", 99.0) == (4000.0, "learned")
    assert st.rate_for("other", 99.0) == (99.0, "hint")


# ---- calibration -----------------------------------------------------------


def test_calibration_counts_right_when_measured_reaches_80_percent():
    rows = [
        row(cache_est_tokens=1000, cached_tokens=800),  # right (boundary)
        row(cache_est_tokens=1000, cached_tokens=799),  # wrong
        row(cache_est_tokens=1000, cached_tokens=1500),  # right
        row(cache_est_tokens=None, cached_tokens=5),  # no estimate: skipped
        row(cache_est_tokens=1000, cached_tokens=None),  # not measured: skipped
    ]
    c = pm.calibration(rows)
    assert c["samples"] == 3 and abs(c["hit_ratio"] - 2 / 3) < 1e-9
    assert pm.calibration([])["hit_ratio"] is None


# ---- idle-gap buckets and decay ---------------------------------------------


def test_gap_buckets_and_decay():
    rows = []
    for _ in range(12):  # 5-15m: estimate mostly wrong
        rows.append(row(gap_s=400, cache_est_tokens=1000, cached_tokens=100))
    for _ in range(12):  # 0-30s: fine
        rows.append(row(gap_s=10, cache_est_tokens=1000, cached_tokens=990))
    for _ in range(3):  # 2-5m: too few rows to act on
        rows.append(row(gap_s=200, cache_est_tokens=1000, cached_tokens=0))
    buckets = {b["label"]: b for b in pm.gap_buckets(rows)}
    assert buckets["5-15m"]["hit_ratio"] == 0.0 and buckets["5-15m"]["samples"] == 12
    assert buckets["0-30s"]["hit_ratio"] == 1.0
    assert buckets["30-120s"]["hit_ratio"] is None
    st = pm.PrefillModelState()
    st.replace(pm.build(rows))
    assert st.decay_for("m", 400.0) == 0.0
    assert st.decay_for("m", 10.0) is None
    assert st.decay_for("m", 200.0) is None  # < MIN_BUCKET_SAMPLES
    assert st.decay_for("m", 5000.0) is None  # beyond every bucket
    assert st.decay_for("m", None) is None
    assert st.decay_for("zzz", 400.0) is None


def test_decay_factor_is_the_bucket_ratio():
    rows = [
        row(gap_s=400, cache_est_tokens=1000, cached_tokens=(900 if i < 3 else 0))
        for i in range(12)
    ]
    st = pm.PrefillModelState()
    st.replace(pm.build(rows))
    assert st.decay_for("m", 400.0) == 3 / 12


def test_prefix_memory_reports_the_gap():
    t = [100.0]
    m = PrefixMemory(clock=lambda: t[0])
    m.remember("m", "k", None, 1000, 1000)
    t[0] = 160.0
    est, gap = m.estimate_with_gap("m", "k", None, 1500)
    assert est == 1000 and gap == 60.0
    assert m.estimate_with_gap("m", "none", None, 1500) == (None, None)


# ---- refresh from the database ------------------------------------------------


async def test_refresh_reads_request_history(tmp_path):
    db_path = tmp_path / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    with sqlite3.connect(db_path) as db:
        for i in range(22):
            db.execute(
                "INSERT INTO request_history(id, finished_at, model_id, model, prompt_tokens, "
                "completion_tokens, duration_s, ttft_s, orphan, started_iso, cached_tokens, "
                "inflight_same_rank_at_start) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"r{i}", 90_000.0 + i, "m", "m", 10_000, 5, 5.0, 4.0, 0, "x", 2_000, 0),
            )
        db.execute(  # outside the 24 h window
            "INSERT INTO request_history(id, finished_at, model_id, model, prompt_tokens, "
            "completion_tokens, duration_s, orphan, started_iso) VALUES "
            "('old', 1.0, 'm', 'm', 1, 1, 1, 0, 'x')"
        )
        db.commit()
    st = pm.PrefillModelState()
    n = await pm.refresh(st, db_path, now=100_000.0)
    assert n == 22
    assert st.rate_for("m", 1.0) == (2000.0, "learned")
    snap = st.snapshot(2000.0)
    assert snap["models"]["m"]["samples"] == 22 and snap["updated_at"] is not None


# ---- queue-ahead and causes ------------------------------------------------------


def live(rid, *, started, prompt=10_000, est=0, rank=None, phase="prefill", model="m"):
    return LiveRequest(
        id=rid,
        token_id="t",
        token_name="n",
        client_ip="1",
        model=model,
        model_row_id=model,
        path="/v1/chat/completions",
        prompt_tokens=prompt,
        max_model_len=100_000,
        started_monotonic=started,
        started_iso="x",
        phase=phase,
        dp_rank=rank,
        cache_est_tokens=est or None,
    )


def test_ahead_counts_earlier_prefills_on_the_same_replica_only():
    me = live("me", started=10.0, prompt=8_000, est=2_000)
    others = [
        live("a", started=1.0, prompt=5_000, est=1_000),  # ahead: 4000 fresh
        live("b", started=2.0, prompt=3_000),  # ahead: 3000
        live("c", started=11.0),  # later: not ahead
        live("d", started=3.0, phase="answering"),  # already past prefill
        live("e", started=4.0, rank=1),  # other replica
        live("f", started=5.0, model="other"),  # other model
        live("g", started=6.0, phase="queued"),  # not admitted
    ]
    assert _ahead_of(me, [me, *others]) == (2, 7_000)


def test_ahead_single_replica_and_dp_rank_equality():
    a = live("a", started=1.0, rank=None)
    b = live("b", started=2.0, rank=None)
    assert _ahead_of(b, [a, b]) == (1, 10_000)
    r3 = live("r3", started=1.0, rank=3)
    me3 = live("me3", started=2.0, rank=3)
    me0 = live("me0", started=2.0, rank=0)
    assert _ahead_of(me3, [r3, me3, me0]) == (1, 10_000)
    assert _ahead_of(me0, [r3, me3, me0]) == (0, 0)


def test_serialize_expected_start_uses_the_rate():
    me = live("me", started=10.0, prompt=8_000, est=2_000)  # own fresh 6000
    row_ = _serialize(me, 12.0, 1000.0, ahead=(2, 7_000), rate_source="learned", est_accuracy=0.9)
    assert row_["expected_start_s"] == 13.0  # (7000 + 6000) / 1000
    assert row_["ahead_count"] == 2 and row_["ahead_fresh_tokens"] == 7_000
    assert row_["prefill_rate_source"] == "learned" and row_["est_accuracy"] == 0.9
    assert row_["dp_rank"] is None
    assert row_["slow_cause"] == "queued"


def test_cause_rule_boundaries():
    kw = dict(slow=False, ahead_count=1, expected_start_s=10.0, cache_est_pct=None)
    # within 1.5 x expected + 3 -> queued, inclusive at the boundary
    assert slow_cause(elapsed_s=18.0, **kw) == "queued"
    assert slow_cause(elapsed_s=18.1, **kw) is None
    assert slow_cause(elapsed_s=18.1, **{**kw, "slow": True}) == "slow"
    # alone, slow, mostly cached -> evicted_likely; below 0.5 -> plain slow
    alone = dict(slow=True, ahead_count=0, expected_start_s=1.0, elapsed_s=30.0)
    assert slow_cause(cache_est_pct=0.5, **alone) == "evicted_likely"
    assert slow_cause(cache_est_pct=0.49, **alone) == "slow"
    assert slow_cause(cache_est_pct=None, **alone) == "slow"
    # not slow, nobody ahead -> nothing to say
    assert (
        slow_cause(
            slow=False, ahead_count=0, expected_start_s=1.0, elapsed_s=1.0, cache_est_pct=0.9
        )
        is None
    )


def test_causes_only_apply_to_prefill_rows():
    ans = live("x", started=1.0, phase="answering")
    row_ = _serialize(ans, 999.0, 2000.0, ahead=(3, 9_000))
    assert row_["slow_cause"] is None and row_["expected_start_s"] is None
    assert row_["ahead_count"] == 0


def _cold(prompt, ttft, **kw):
    r = dict(
        model_id="m",
        prompt_tokens=prompt,
        cached_tokens=None,
        ttft_s=ttft,
        inflight_same_rank_at_start=0,
        reusable_tokens=0,
        cache_est_tokens=None,
        gap_s=None,
    )
    r.update(kw)
    return r


def test_cold_rate_is_median_of_unreported_cold_alone_rows():
    rows = [_cold(4000, 1.0)] * 20 + [_cold(4000, 1.0, reusable_tokens=3000)] * 50
    rate, n = pm.cold_rate(rows)
    assert (rate, n) == (4000.0, 20)


def test_cold_rate_ignores_measured_contended_and_tiny():
    rows = (
        [_cold(4000, 1.0, cached_tokens=0)] * 30
        + [_cold(4000, 1.0, inflight_same_rank_at_start=2)] * 30
        + [_cold(100, 0.01)] * 30
    )
    assert pm.cold_rate(rows) == (None, 0)


def test_state_exposes_cold_rate():
    st = pm.PrefillModelState()
    st.replace(pm.build([_cold(4000, 1.0)] * 20))
    assert st.cold_rate_for("m") == 4000.0
    assert st.cold_rate_for("nope") is None
