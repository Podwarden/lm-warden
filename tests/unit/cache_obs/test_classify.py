# tests/unit/cache_obs/test_classify.py
import pytest

from app.cache_obs.classify import bytes_to_tokens, classify


def c(cached, r, fleet=None, div=None, mis=True):
    return classify(
        cached=cached,
        reusable=r,
        reusable_fleet=r if fleet is None else fleet,
        diverged_at=div,
        allow_misrouted=mis,
    )


def test_null_fact_is_null():
    assert c(None, 5000) is None


def test_null_potential_is_null():
    assert classify(cached=10, reusable=None, reusable_fleet=None, diverged_at=None) is None


def test_c_above_r_is_hit():
    assert c(9000, 0) == "hit"


def test_misrouted_when_other_replica_had_it():
    assert c(0, 0, fleet=8000) == "misrouted"


def test_misrouted_suppressed_in_own_lens():
    assert c(0, 0, fleet=8000, mis=False) == "cold"


def test_diverged_and_cold():
    assert c(0, 100, div=0) == "diverged"
    assert c(0, 100) == "cold"


@pytest.mark.parametrize(
    "cached,expected", [(9000, "hit"), (8100, "hit"), (5000, "partial"), (500, "lost"), (0, "lost")]
)
def test_hit_partial_lost(cached, expected):
    assert c(cached, 9000) == expected


def test_bytes_to_tokens():
    assert bytes_to_tokens(500, 1000, 4000) == 2000
    assert bytes_to_tokens(0, 0, 4000) == 0


def test_estimated_c_above_r_is_not_promoted_to_hit():
    # rule 1 (R := C) trusts only a MEASURED C
    assert (
        classify(
            cached=400, reusable=0, reusable_fleet=0, diverged_at=None, cached_is_estimate=True
        )
        == "cold"
    )
    assert classify(cached=400, reusable=0, reusable_fleet=0, diverged_at=None) == "hit"
