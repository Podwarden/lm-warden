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


# --- #301: block-aware classification ------------------------------------
# pw-prod's hybrid Qwen3.5 runs 784-token KV blocks; the engine serves whole
# blocks only, so a perfect hit leaves the trailing partial block uncached.


def cb(cached, r, block, fleet=None, div=None, mis=True):
    return classify(
        cached=cached,
        reusable=r,
        reusable_fleet=r if fleet is None else fleet,
        diverged_at=div,
        allow_misrouted=mis,
        block_size=block,
    )


def test_probe_numbers_read_hit_with_784_blocks():
    # the live probe: 6,445-token prompt twice, 6272 = 8 x 784 cached
    assert cb(6272, 6445, 784) == "hit"


def test_short_prompt_full_hit_is_not_partial_with_784_blocks():
    # 1568 = 2 x 784 is everything the engine can serve of 2000; 78 % of R
    assert c(1568, 2000) == "partial"
    assert cb(1568, 2000, 784) == "hit"


def test_one_block_lost_of_eight_is_partial():
    assert cb(5488, 6445, 784) == "partial"  # 7 of 8 blocks


def test_under_one_block_is_cold_never_lost():
    assert c(0, 500) == "lost"
    assert cb(0, 500, 784) == "cold"


def test_whole_blocks_missed_is_lost():
    assert cb(0, 6445, 784) == "lost"


def test_misrouted_needs_a_whole_block_elsewhere():
    assert cb(0, 0, 784, fleet=500) == "cold"
    assert cb(0, 0, 784, fleet=8000) == "misrouted"


def test_under_one_block_keeps_diverged():
    assert cb(0, 500, 784, div=3) == "diverged"


def test_measured_c_above_r_is_still_a_hit():
    assert cb(1568, 1000, 784) == "hit"


@pytest.mark.parametrize("block", [None, 16])
@pytest.mark.parametrize(
    "cached,r,expected",
    [
        (9000, 9000, "hit"),
        (8100, 9000, "hit"),
        (5000, 9000, "partial"),
        (500, 9000, "lost"),
        (0, 9000, "lost"),
        (0, 500, "lost"),
        (0, 100, "cold"),
        # quantising to 16 would make this 260 >= 0.9 * 288, a hit
        (260, 300, "partial"),
    ],
)
def test_unknown_or_default_block_is_unchanged(block, cached, r, expected):
    assert c(cached, r) == expected
    assert cb(cached, r, block) == expected
