# tests/unit/cache_obs/test_record.py
from app.cache_obs.index import Match
from app.cache_obs.record import observe_fields, ttft_estimate

KEYS = {
    "reusable_tokens",
    "reusable_tokens_fleet",
    "reusable_tokens_own",
    "cache_outcome",
    "cache_outcome_own",
    "diverged_at",
    "diverged_at_own",
    "cached_source",
    "cached_ttft_est_tokens",
}


def m(by_rank, total=10_000, div=None):
    return Match(by_rank, max(by_rank.values(), default=0), total, div)


def f(**kw):
    base = dict(
        prompt=10_000,
        cached=None,
        reporting="usage",
        ttft_s=None,
        cold_rate=None,
        match_all=None,
        match_own=None,
        rank=None,
        rank_known=True,
    )
    base.update(kw)
    return observe_fields(**base)


def test_keys_are_exactly_the_columns():
    assert set(f()) == KEYS


def test_measured_hit_on_served_rank():
    out = f(cached=9_500, match_all=m({2: 10_000}), match_own=m({2: 10_000}), rank=2)
    assert out["cached_source"] == "engine"
    assert out["reusable_tokens"] == 10_000 and out["cache_outcome"] == "hit"
    assert out["cache_outcome_own"] == "hit"


def test_misrouted_when_prefix_on_other_rank():
    out = f(cached=0, match_all=m({5: 9_000}), match_own=m({5: 9_000}), rank=2)
    assert out["reusable_tokens"] == 0 and out["reusable_tokens_fleet"] == 9_000
    assert out["cache_outcome"] == "misrouted"
    assert out["cache_outcome_own"] == "cold"  # own lens never says misrouted


def test_unrouted_observation_never_makes_a_routed_miss_misrouted():
    # #289: unrouted requests are observed under rank None; for a ROUTED
    # request that bit is not "another replica", so it stays out of R_fleet.
    out = f(cached=0, match_all=m({None: 9_000}), match_own=m({None: 9_000}), rank=2)
    assert out["reusable_tokens"] == 0 and out["reusable_tokens_fleet"] == 0
    assert out["cache_outcome"] == "cold"


def test_routed_fleet_still_counts_other_ranks_next_to_unrouted():
    out = f(cached=0, match_all=m({None: 9_000, 5: 4_000}), rank=2)
    assert out["reusable_tokens_fleet"] == 4_000 and out["cache_outcome"] == "misrouted"


def test_single_replica_keeps_rank_none_in_fleet():
    out = f(cached=9_500, match_all=m({None: 10_000}), rank=None, rank_known=True)
    assert out["reusable_tokens"] == 10_000 and out["reusable_tokens_fleet"] == 10_000


def test_unknown_rank_uses_fleet_and_never_misroutes():
    out = f(cached=0, match_all=m({5: 9_000}), rank=None, rank_known=False)
    assert out["reusable_tokens"] == 9_000 and out["cache_outcome"] == "lost"


def test_usage_backend_without_report_is_null_not_zero():
    out = f(cached=None, match_all=m({None: 9_000}))
    assert out["cached_source"] is None and out["cache_outcome"] is None


def test_none_backend_estimates_from_ttft():
    out = f(reporting="none", ttft_s=0.5, cold_rate=4_000.0, match_all=m({None: 9_000}))
    # 10_000 - 0.5 * 4_000 = 8_000
    assert out["cached_ttft_est_tokens"] == 8_000 and out["cached_source"] == "estimated"
    assert out["cache_outcome"] == "partial"


def test_ttft_estimate_needs_a_learned_rate():
    assert ttft_estimate(10_000, 0.5, None) is None
    assert ttft_estimate(10_000, None, 4_000.0) is None
    assert ttft_estimate(10_000, 5.0, 4_000.0) == 0  # clamped


def test_own_lens_uses_its_own_match():
    out = f(
        cached=0,
        match_all=m({2: 10_000}),
        match_own=m({2: 1_000}),
        rank=2,
    )
    assert out["reusable_tokens"] == 10_000
    assert out["reusable_tokens_own"] == 1_000
    assert out["cache_outcome"] == "lost"
    assert out["cache_outcome_own"] == "lost"
    out2 = f(cached=1_000, match_all=m({2: 10_000}), match_own=m({2: 1_000}), rank=2)
    assert out2["cache_outcome"] == "partial"
    assert out2["cache_outcome_own"] == "hit"


def test_none_backend_cold_request_is_cold_not_hit():
    # 2000 - 0.8 * 2000 = 400 estimated, nothing reusable: a slow-ish cold prefill
    out = f(
        prompt=2000,
        reporting="none",
        ttft_s=0.8,
        cold_rate=2000.0,
        match_all=m({None: 0}, total=2000),
        match_own=m({None: 0}, total=2000),
    )
    assert out["cached_ttft_est_tokens"] == 400
    assert out["cache_outcome"] == "cold" and out["cache_outcome_own"] == "cold"


def test_estimate_below_noise_floor_is_zero():
    # 10% of P: within prefill-speed noise, read as nothing served from cache
    assert ttft_estimate(10_000, 9_000 / 4_000.0, 4_000.0) == 0  # est 1_000 = 10%
    assert ttft_estimate(10_000, 2.0, 4_200.0) == 1_600  # 16% stays


def test_diverged_at_own_comes_from_the_own_match():
    # The fleet-scope break point can rest on other keys' messages; the own
    # lens gets its own, matched against this key's prompts only.
    out = f(
        cached=9_000,
        match_all=m({None: 9_500}, div=3),
        match_own=m({None: 9_500}, div=None),
        rank=None,
    )
    assert out["diverged_at"] == 3 and out["diverged_at_own"] is None
    out = f(cached=9_000, match_all=m({None: 9_500}, div=3), match_own=m({None: 9_500}, div=2))
    assert out["diverged_at_own"] == 2
