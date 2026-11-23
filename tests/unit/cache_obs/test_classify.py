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


# --- the engine drops the last matched block (EAGLE/MTP speculative decoding) ---
# lmwarden.com, 2026-11-22: vLLM 0.32, hybrid Qwen, 1,584-token blocks, MTP with
# one speculative token. Measured: cached = (floor(matched / 1584) - 1) x 1584.
# Numbers below are from the controlled experiment (cinv/exp_results.jsonl).
B = 1584


def cd(cached, r, block=B, drop=True, fleet=None, div=None, mis=True, unit=None):
    return classify(
        cached=cached,
        reusable=r,
        reusable_fleet=r if fleet is None else fleet,
        diverged_at=div,
        allow_misrouted=mis,
        block_size=block,
        drops_last_block=drop,
        match_unit=unit,
    )


@pytest.mark.parametrize(
    "cached,matched",
    [
        # the same prompt sent twice, nothing in between
        (1584, 4394),
        (3168, 6086),
        (6336, 8888),
        # an append-only chain: each turn matched the previous turn's prompt
        (1584, 4020),
        (4752, 7078),
        (7920, 10036),
    ],
)
def test_everything_the_engine_can_serve_is_a_hit(cached, matched):
    assert cd(cached, matched) == "hit"
    # before the fix these read partial: the dropped block counted as reusable
    assert cd(cached, matched, drop=False) == "partial"


@pytest.mark.parametrize("matched", [1014, 1632, 2006, 2508, 3167])
def test_under_two_blocks_is_nothing_to_reuse(matched):
    # one whole block or less: the engine drops it, so 0 is all it can serve
    assert cd(0, matched) == "cold"


def test_exactly_two_blocks_serves_one():
    assert cd(1584, 3168) == "hit"
    assert cd(0, 3168) == "lost"


def test_misses_beyond_the_engine_limit_keep_their_labels():
    assert cd(0, 8888) == "lost"  # 4 blocks servable, none served: evicted
    assert cd(3168, 8888) == "partial"  # 2 of 4
    assert cd(0, 2006, div=5) == "diverged"


def test_fleet_loses_the_block_too():
    assert cd(0, 0, fleet=3000) == "cold"  # one block elsewhere: nothing servable
    assert cd(0, 0, fleet=4394) == "misrouted"


def test_measured_cached_above_the_index_is_still_a_hit():
    assert cd(4752, 3000) == "hit"
    # production rows: the index read just under two blocks, the engine
    # served one, so it had matched two; never "nothing to reuse"
    assert cd(1584, 3143) == "hit"
    assert cd(1584, 3004) == "hit"


@pytest.mark.parametrize("block", [None, 16])
def test_drop_is_ignored_without_large_blocks(block):
    # llama.cpp / MLX (no block size) and classic 16-token pages: unchanged
    assert cd(0, 500, block=block) == "lost"
    assert cd(5000, 9000, block=block) == "partial"
    assert cd(8100, 9000, block=block) == "hit"


# --- --prefix-match-unit N: the engine matches every N tokens (N | B) and,
# with MTP, drops one unit rather than one block (owner-approved for
# lmwarden.com 2026-11-22, benchmark scratchpad/mtp-block-benchmark.md).


def test_unit_equal_to_block_is_todays_rule():
    assert cd(1584, 4394, unit=B) == cd(1584, 4394) == "hit"
    assert cd(0, 3000, unit=B) == "cold"


def test_unit_16_with_mtp_expects_all_but_one_unit():
    # 4,394 matched -> 4,384 on 16-token steps, minus one unit for the drafter
    assert cd(4368, 4394, unit=16) == "hit"
    # the old block rule's 1,584 is now a real shortfall
    assert cd(1584, 4394, unit=16) == "partial"
    assert cd(0, 4394, unit=16) == "lost"
    # a prompt under two blocks is reusable now
    assert cd(0, 2006, unit=16) == "lost"
    assert cd(1984, 2006, unit=16) == "hit"


def test_unit_16_without_mtp_expects_every_whole_unit():
    assert cd(4384, 4394, unit=16, drop=False) == "hit"
    assert cd(3168, 4394, unit=16, drop=False) == "partial"  # fell back to a block boundary


def test_disabled_eagle_drop_with_block_unit_keeps_the_last_block():
    # disable_eagle_block_drop: true arrives here as drop=False
    assert cd(3168, 4394, drop=False) == "hit"
    assert cd(1584, 4394, drop=False) == "partial"


def test_unit_is_ignored_on_small_pages_and_unknown_blocks():
    assert cd(0, 500, block=None, unit=16) == "lost"
    assert cd(260, 300, block=16, unit=16) == "partial"
