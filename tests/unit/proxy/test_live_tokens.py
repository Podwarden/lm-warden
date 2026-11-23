"""B1 (stats redesign): live out-token counting and the 5 s decode speed.

While a stream runs, the live row's ``completion_tokens`` must count TOKENS of
every generated channel (content, reasoning, tool-call name and arguments), not
SSE frames of the content channel only. A thinking model used to read 0 until it
started answering, and a speculative-decoding engine (several tokens a frame)
read low.
"""

from __future__ import annotations

import asyncio

from app.proxy.live_tokens import LiveOutCounter, decode_tps


def _count_chars(text: str) -> int:
    return len(text)


async def _acount(text: str) -> int:
    return len(text)


def test_counter_counts_every_channel_as_tokens():
    c = LiveOutCounter()
    c.feed("abc", None)  # reasoning
    c.feed("de", None)  # tool-call arguments
    c.feed("", None)
    n = asyncio.run(c.count(_acount))
    assert n == 5
    c.feed("fgh", None)
    assert asyncio.run(c.count(_acount)) == 8
    assert c.frames == 3


def test_counter_prefers_the_engines_own_usage():
    c = LiveOutCounter()
    c.feed("abcdef", None)
    c.feed("", 42)
    assert asyncio.run(c.count(_acount)) == 42


def test_counter_falls_back_to_frames_when_counting_fails():
    async def boom(_t: str) -> int:
        raise RuntimeError("tokenizer gone")

    c = LiveOutCounter()
    c.feed("abc", None)
    c.feed("de", None)
    assert asyncio.run(c.count(boom)) == 2


def test_decode_tps_over_the_last_five_seconds():
    # first token at t=10; 50 tok/s for 10 s, sampled every 0.5 s
    samples = [(10.0 + i * 0.5, int(i * 0.5 * 50)) for i in range(1, 21)]
    assert round(decode_tps(samples, first_token=10.0, now=20.0)) == 50


def test_decode_tps_uses_first_token_as_origin_when_young():
    samples = [(11.0, 30), (12.0, 60)]
    assert round(decode_tps(samples, first_token=10.0, now=12.0)) == 30


def test_decode_tps_none_before_first_token_and_zero_when_stalled():
    assert decode_tps([], first_token=None, now=5.0) is None
    # no new token for more than 5 s: the request is stalled, not fast
    assert decode_tps([(10.0, 100)], first_token=5.0, now=16.0) == 0.0


def test_an_early_zero_usage_frame_does_not_pin_the_count():
    """Review finding 6: an engine figure is a floor, never a ceiling."""
    c = LiveOutCounter()
    c.feed("abc", 0)
    assert asyncio.run(c.count(_acount)) == 3
    c.feed("defg", 1)
    assert asyncio.run(c.count(_acount)) == 7


def test_per_frame_usage_skips_tokenizing():
    calls: list[str] = []

    async def counting(t: str) -> int:
        calls.append(t)
        return len(t)

    c = LiveOutCounter()
    c.feed("abc", 3)
    c.feed("de", 5)
    assert asyncio.run(c.count(counting)) == 5
    assert calls == []
