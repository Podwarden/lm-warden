"""Deterministic random-word prompt generator -- defeats prefix caching.

Each prompt is unique text built from a per-request seed, sized to a target
token budget via a fixed tokens/word estimate (no tokenizer dependency in
tests/). Pure function, no I/O -- covered by tests/unit/live.
"""

from __future__ import annotations

import random

_WORDS = (
    "amber apex arcane basalt beacon cobalt comet drift ember flint "
    "forge glacier harbor helix ion jade kite lattice meadow nebula "
    "opal orbit pivot quartz raven ridge signal spruce talon umbra "
    "vertex wharf yonder zephyr anchor bramble cinder dune ferry "
    "granite husk ivory jetty knoll lumen marsh nectar onyx plume "
    "quiver rune sable thicket urchin vapor willow yarrow zinc"
).split()

#: Rough words-per-token budget for untokenized English word salad. Not
#: precise (no tokenizer dependency here) -- calibration and the
#: max_model_len headroom check both allow slack around it.
TOKENS_PER_WORD = 1.35


def make_prompt(*, seed: int, target_tokens: int, tag: str) -> str:
    """A unique prompt of roughly ``target_tokens`` tokens for this ``seed``.

    ``tag`` (the probe/filler label) is embedded so a human skimming server
    logs can tell requests apart.
    """
    rng = random.Random(seed)
    n_words = max(8, int(target_tokens / TOKENS_PER_WORD))
    body = " ".join(rng.choice(_WORDS) for _ in range(n_words))
    return f"[{tag}#{seed}] Repeat back the following tokens verbatim, then stop: {body}"
