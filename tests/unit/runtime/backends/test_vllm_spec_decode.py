"""Does a vLLM engine launched with these args drop the last matched
prefix-cache block on every hit? (vLLM 0.32.0: EAGLE-like speculative
methods recompute the tokens before the generation point, so the KV cache
coordinator pops one whole block off each hit unless
``disable_eagle_block_drop`` is set.)"""

import pytest

from app.runtime.backends.vllm.spec_decode import drops_last_cached_block

MTP = '{"method":"mtp","num_speculative_tokens":1}'


@pytest.mark.parametrize(
    "args",
    [
        ["--speculative-config", MTP],  # lmwarden.com's production engine
        [f"--speculative-config={MTP}"],
        ["--speculative_config", MTP],
        ["--speculative-config", '{"method": "eagle3", "model": "x/y"}'],
        ["--speculative-config", '{"method": "eagle"}'],
        ["--speculative-config", '{"method": "dflash"}'],
        ["--speculative-config", '{"method": "dspark"}'],
        # model-specific MTP names are normalised to "mtp" by vLLM
        ["--speculative-config", '{"method": "qwen3_5_mtp"}'],
        ["--speculative-config.method", "mtp"],
        ["--speculative-config.method=mtp"],
        ["--max-num-seqs", "8", "--speculative-config", MTP, "--enable-prefix-caching"],
    ],
)
def test_eagle_like_speculative_methods_drop_the_last_block(args):
    assert drops_last_cached_block(args) is True


@pytest.mark.parametrize(
    "args",
    [
        [],
        None,
        ["--enable-prefix-caching"],
        ["--speculative-config", '{"method":"ngram","num_speculative_tokens":4}'],
        ["--speculative-config", '{"method":"draft_model","model":"x"}'],
        # the experimental opt-out keeps the block
        ["--speculative-config", '{"method":"mtp","disable_eagle_block_drop":true}'],
        [
            "--speculative-config.method",
            "mtp",
            "--speculative-config.disable_eagle_block_drop",
            "true",
        ],
        # unparseable or method unknown: no evidence, keep today's rules
        ["--speculative-config", "{not json"],
        ["--speculative-config", '{"num_speculative_tokens":1}'],
        ["--speculative-config"],
    ],
)
def test_other_configs_keep_the_last_block(args):
    assert drops_last_cached_block(args) is False


# --- --prefix-match-unit (vLLM 0.32 CacheConfig.prefix_match_unit) ----------
# Prefixes match every N tokens instead of every physical block; N must divide
# the block size. Unset (or unusable) means the block size.
from app.runtime.backends.vllm.spec_decode import prefix_match_unit  # noqa: E402


@pytest.mark.parametrize(
    "args,block,expected",
    [
        (["--prefix-match-unit", "16"], 1584, 16),
        (["--prefix-match-unit=16"], 1584, 16),
        (["--prefix_match_unit", "32"], 1056, 32),
        ([], 1584, 1584),  # today's lmwarden.com engine
        (None, 1584, 1584),
        (["--prefix-match-unit", "0"], 1584, 1584),
        (["--prefix-match-unit", "100"], 1584, 1584),  # does not divide the block
        (["--prefix-match-unit", "3168"], 1584, 1584),  # above the block
        (["--prefix-match-unit", "x"], 1584, 1584),
        (["--prefix-match-unit"], 1584, 1584),
    ],
)
def test_prefix_match_unit(args, block, expected):
    assert prefix_match_unit(args, block) == expected
