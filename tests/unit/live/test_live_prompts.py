"""Pure-logic unit tests for tests/live/_prompts.py -- no network."""

from __future__ import annotations

from tests.live._prompts import make_prompt


def test_make_prompt_is_deterministic_for_same_seed():
    a = make_prompt(seed=42, target_tokens=100, tag="p0a")
    b = make_prompt(seed=42, target_tokens=100, tag="p0a")
    assert a == b


def test_make_prompt_differs_across_seeds():
    a = make_prompt(seed=1, target_tokens=100, tag="p0a")
    b = make_prompt(seed=2, target_tokens=100, tag="p0a")
    assert a != b


def test_make_prompt_differs_across_seeds_even_with_shared_prefix_tag():
    # Defeating prefix caching depends on the WORD body differing, not just
    # the bracketed tag -- assert the bodies (after the tag) diverge too.
    a = make_prompt(seed=1, target_tokens=200, tag="filler0")
    b = make_prompt(seed=2, target_tokens=200, tag="filler0")
    body_a = a.split("verbatim, then stop: ", 1)[1]
    body_b = b.split("verbatim, then stop: ", 1)[1]
    assert body_a != body_b


def test_make_prompt_scales_roughly_with_target_tokens():
    short = make_prompt(seed=1, target_tokens=50, tag="x")
    long = make_prompt(seed=1, target_tokens=500, tag="x")
    assert len(long.split()) > len(short.split())


def test_make_prompt_embeds_tag_and_seed():
    p = make_prompt(seed=7, target_tokens=50, tag="p9a")
    assert "[p9a#7]" in p


def test_make_prompt_never_empty_even_for_tiny_target():
    p = make_prompt(seed=1, target_tokens=1, tag="tiny")
    # Minimum word floor keeps this from degenerating to nothing.
    assert len(p.split()) >= 8
