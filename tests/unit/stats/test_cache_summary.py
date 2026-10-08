"""request_history.query_cache_summary -- the per-model cache observation."""

from __future__ import annotations

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.stats.request_history import query_cache_summary


async def _db(tmp_path):
    p = tmp_path / "t.db"
    async with open_db(p) as db:
        await apply_migrations(db)
    return p


async def _ins(db, i, **kw):
    row = dict(
        id=f"r{i}",
        finished_at=1000.0 + i,
        model_id="m",
        model="m",
        token_id="tA",
        token_name="a",
        prompt_tokens=1000,
        completion_tokens=1,
        duration_s=1.0,
        orphan=0,
        started_iso="",
        cached_tokens=None,
        reusable_tokens=None,
        reusable_tokens_own=None,
        cache_outcome=None,
        cache_outcome_own=None,
        diverged_at=None,
        diverged_at_own=None,
        cached_source=None,
        cached_ttft_est_tokens=None,
        dp_rank=None,
    )
    row.update(kw)
    cols = ",".join(row)
    await db.execute(
        f"INSERT INTO request_history({cols}) VALUES ({','.join('?' * len(row))})",
        tuple(row.values()),
    )


async def test_sums_outcomes_and_sources(tmp_path):
    p = await _db(tmp_path)
    async with open_db(p) as db:
        await _ins(
            db,
            1,
            cached_tokens=900,
            reusable_tokens=1000,
            cache_outcome="hit",
            cached_source="engine",
        )
        await _ins(
            db,
            2,
            cached_tokens=0,
            reusable_tokens=1000,
            cache_outcome="lost",
            cached_source="engine",
        )
        await _ins(
            db,
            3,
            cached_ttft_est_tokens=500,
            reusable_tokens=800,
            cache_outcome="partial",
            cached_source="estimated",
        )
        await _ins(db, 4)  # nothing known
        await db.commit()
        [s] = await query_cache_summary(db, since=0, model_ids=None)
    assert s["requests"] == 4 and s["prompt_tokens"] == 4000
    assert s["cached_tokens"] == 1400 and s["reusable_tokens"] == 2800
    assert (s["measured_requests"], s["estimated_requests"]) == (2, 1)
    assert s["outcomes"] == {
        "hit": 1,
        "partial": 1,
        "lost": 1,
        "misrouted": 0,
        "diverged": 0,
        "cold": 0,
    }


async def test_own_lens_uses_own_columns_and_hides_diverging(tmp_path):
    p = await _db(tmp_path)
    async with open_db(p) as db:
        await _ins(
            db,
            1,
            cached_tokens=0,
            reusable_tokens=1000,
            reusable_tokens_own=0,
            cache_outcome="misrouted",
            cache_outcome_own="cold",
            cached_source="engine",
        )
        await _ins(
            db, 2, token_id="tB", cached_tokens=0, cache_outcome="diverged", cached_source="engine"
        )
        await db.commit()
        [s] = await query_cache_summary(db, since=0, model_ids=None, token_id="tA")
    assert s["requests"] == 1 and s["outcomes"]["cold"] == 1 and s["outcomes"]["misrouted"] == 0
    assert s["reusable_tokens"] == 0 and "top_diverging" not in s


async def test_top_diverging_for_operator(tmp_path):
    p = await _db(tmp_path)
    async with open_db(p) as db:
        for i in range(3):
            await _ins(
                db,
                i,
                token_id="tB",
                token_name="b",
                cache_outcome="diverged",
                cached_tokens=0,
                cached_source="engine",
            )
        await db.commit()
        [s] = await query_cache_summary(db, since=0, model_ids=None)
    assert s["top_diverging"] == [{"token_id": "tB", "token_name": "b", "requests": 3}]


async def test_efficiency_inputs_apply_rule_one_per_row(tmp_path):
    # 50 rows the index missed but the engine served (C > R) plus 50 rows the
    # index knew but the engine lost: half the reusable work was reused.
    p = await _db(tmp_path)
    async with open_db(p) as db:
        for i in range(50):
            await _ins(
                db,
                i,
                prompt_tokens=10_000,
                cached_tokens=9000,
                reusable_tokens=0,
                cached_source="engine",
            )
        for i in range(50, 100):
            await _ins(
                db,
                i,
                prompt_tokens=10_000,
                cached_tokens=0,
                reusable_tokens=9000,
                cached_source="engine",
            )
        await db.commit()
        [s] = await query_cache_summary(db, since=0, model_ids=None)
    assert s["reused_tokens"] == 450_000
    assert s["reusable_tokens"] == 900_000
    assert s["reused_tokens"] / s["reusable_tokens"] == 0.5


async def test_pre_0046_row_is_out_of_both_efficiency_sides(tmp_path):
    p = await _db(tmp_path)
    async with open_db(p) as db:
        await _ins(db, 1, cached_tokens=800, reusable_tokens=1000, cached_source="engine")
        # before migration 0046: C measured, R never computed
        await _ins(db, 2, cached_tokens=900, reusable_tokens=None, cached_source="engine")
        await db.commit()
        [s] = await query_cache_summary(db, since=0, model_ids=None)
    assert (s["reused_tokens"], s["reusable_tokens"]) == (800, 1000)
    assert s["cached_tokens"] == 1700  # "from cache" still counts every known C


async def test_known_prompt_tokens_skip_unknown_fact(tmp_path):
    p = await _db(tmp_path)
    async with open_db(p) as db:
        await _ins(db, 1, prompt_tokens=1000, cached_tokens=500, cached_source="engine")
        await _ins(db, 2, prompt_tokens=9000)  # backend never reported: unknown, not 0
        await db.commit()
        [s] = await query_cache_summary(db, since=0, model_ids=None)
    assert s["prompt_tokens"] == 10_000
    assert s["known_prompt_tokens"] == 1000
    assert s["cached_tokens"] / s["known_prompt_tokens"] == 0.5


async def test_by_rank_uses_the_same_efficiency_inputs(tmp_path):
    p = await _db(tmp_path)
    async with open_db(p) as db:
        await _ins(db, 1, dp_rank=0, cached_tokens=900, reusable_tokens=0, cached_source="engine")
        await _ins(db, 2, dp_rank=0, cached_tokens=0, reusable_tokens=900, cached_source="engine")
        await _ins(db, 3, dp_rank=0, cached_tokens=700, reusable_tokens=None)
        await db.commit()
        [s] = await query_cache_summary(db, since=0, model_ids=None)
    [r0] = s["by_rank"]
    assert (r0["reused_tokens"], r0["reusable_tokens"]) == (900, 1800)
    assert r0["cached_tokens"] == 1600 and r0["requests"] == 3


async def test_estimate_never_raises_the_reusable_count(tmp_path):
    # A cold request on a "none" backend whose TTFT estimate cleared the noise
    # floor (C_est=400, R=0) must not read as 400/400 reused: rule 1 is for
    # measured C only. A measured C > R still raises R.
    p = await _db(tmp_path)
    async with open_db(p) as db:
        await _ins(db, 1, cached_tokens=900, reusable_tokens=1000, cached_source="engine")
        await _ins(db, 2, cached_tokens=600, reusable_tokens=0, cached_source="engine")
        await _ins(db, 3, cached_ttft_est_tokens=400, reusable_tokens=0, cached_source="estimated")
        # estimate above R: capped at R, so this row reads 100%, never more
        await _ins(
            db, 4, cached_ttft_est_tokens=700, reusable_tokens=500, cached_source="estimated"
        )
        await _ins(
            db, 5, cached_ttft_est_tokens=200, reusable_tokens=800, cached_source="estimated"
        )
        await db.commit()
        [s] = await query_cache_summary(db, since=0, model_ids=None)
    assert s["reused_tokens"] == 900 + 600 + 0 + 500 + 200
    assert s["reusable_tokens"] == 1000 + 600 + 0 + 500 + 800
    assert s["cached_tokens"] == 2800  # "from cache" keeps the raw estimate


async def test_estimated_rows_cannot_lift_efficiency_above_their_own_ratio(tmp_path):
    p = await _db(tmp_path)
    async with open_db(p) as db:
        for i in range(10):
            await _ins(
                db, i, cached_ttft_est_tokens=300, reusable_tokens=1000, cached_source="estimated"
            )
        for i in range(10, 20):  # cold, estimate above the floor
            await _ins(
                db, i, cached_ttft_est_tokens=500, reusable_tokens=0, cached_source="estimated"
            )
        await db.commit()
        [s] = await query_cache_summary(db, since=0, model_ids=None)
    assert s["reused_tokens"] / s["reusable_tokens"] == 0.3


async def test_by_rank_caps_estimates_too(tmp_path):
    p = await _db(tmp_path)
    async with open_db(p) as db:
        await _ins(
            db,
            1,
            dp_rank=0,
            cached_ttft_est_tokens=400,
            reusable_tokens=0,
            cached_source="estimated",
        )
        await _ins(
            db,
            2,
            dp_rank=0,
            cached_ttft_est_tokens=100,
            reusable_tokens=1000,
            cached_source="estimated",
        )
        await db.commit()
        [s] = await query_cache_summary(db, since=0, model_ids=None)
    [r0] = s["by_rank"]
    assert (r0["reused_tokens"], r0["reusable_tokens"]) == (100, 1000)
    assert (s["reused_tokens"], s["reusable_tokens"]) == (100, 1000)


async def test_prefill_saved_is_null_when_no_fact_is_known(tmp_path):
    from app.stats.request_history import decorate_cache_rows

    p = await _db(tmp_path)
    async with open_db(p) as db:
        await _ins(db, 1, reusable_tokens=1000)
        await db.commit()
        rows = await query_cache_summary(db, since=0, model_ids=None)
        await decorate_cache_rows(db, rows, learned=None, hint=1000.0)
    assert rows[0]["prefill_saved_s"] is None
    assert rows[0]["rate_source"] == "hint"


async def test_deep_diverged_counts_mid_prompt_breaks_that_still_reused(tmp_path):
    # A rewrite deep inside a long prompt leaves a long reusable prefix, so it
    # reads hit or partial; diverged_at is the only trace of the waste.
    p = await _db(tmp_path)
    async with open_db(p) as db:
        await _ins(db, 1, cache_outcome="hit", diverged_at=3)
        await _ins(db, 2, cache_outcome="partial", diverged_at=-1)
        await _ins(db, 3, cache_outcome="diverged", diverged_at=0)  # already its own outcome
        await _ins(db, 4, cache_outcome="lost", diverged_at=5)
        await _ins(db, 5, cache_outcome="hit")
        await db.commit()
        [s] = await query_cache_summary(db, since=0, model_ids=None)
    assert s["deep_diverged"] == 2


async def test_deep_diverged_own_lens_uses_own_columns_only(tmp_path):
    p = await _db(tmp_path)
    async with open_db(p) as db:
        # fleet-scope break only (other keys' messages): not this key's to see
        await _ins(db, 1, cache_outcome="hit", cache_outcome_own="hit", diverged_at=3)
        await _ins(db, 2, cache_outcome="hit", cache_outcome_own="partial", diverged_at_own=2)
        await _ins(db, 3, token_id="tB", cache_outcome_own="hit", diverged_at_own=1)
        await db.commit()
        [s] = await query_cache_summary(db, since=0, model_ids=None, token_id="tA")
    assert s["deep_diverged"] == 1
