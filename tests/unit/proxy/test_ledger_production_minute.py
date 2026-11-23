"""B7 / owner decision D3: tokens are booked to the minutes they were produced.

A 10-minute generation used to land as one spike in the minute it finished.
Now the prompt (and its cached part) is booked to the minute of the first
token, the completion is spread over the minutes from the first token to the
finish in proportion to the time spent in each, and the request itself counts
in the finish minute. Totals stay exact.
"""

from __future__ import annotations

from app.proxy.ledger import Ledger, spread_over_minutes
from tests.unit.proxy.test_ledger import _q, db_path  # noqa: F401 -- fixture


def test_spread_is_exact_and_proportional():
    # first token at 60*10 + 30, finish at 60*13 + 30: 30 s, 60, 60, 30 s
    out = spread_over_minutes(600, start=630.0, end=810.0)
    assert sum(out.values()) == 600
    assert out == {10: 100, 11: 200, 12: 200, 13: 100}


def test_spread_within_one_minute_and_zero_span():
    assert spread_over_minutes(7, start=605.0, end=650.0) == {10: 7}
    assert spread_over_minutes(7, start=650.0, end=650.0) == {10: 7}
    assert spread_over_minutes(0, start=600.0, end=900.0) == {}


async def test_long_generation_is_spread_over_its_minutes(db_path):  # noqa: F811
    led = Ledger(db_path, clock=lambda: 60 * 1013 + 30)
    led.record_request(
        model_id="m",
        token_id="t",
        prompt_tokens=1000,
        completion_tokens=600,
        cached_tokens=400,
        variant_id=None,
        produced_s=180.0,  # first token at 60*1010 + 30
    )
    await led.flush()
    rows = await _q(
        db_path,
        "SELECT minute, requests, prompt_tokens, completion_tokens, cached_tokens, "
        "cached_measured_requests FROM model_samples ORDER BY minute",
    )
    assert rows == [
        (1010, 0, 1000, 100, 400, 1),
        (1011, 0, 0, 200, 0, 0),
        (1012, 0, 0, 200, 0, 0),
        (1013, 1, 0, 100, 0, 0),
    ]
    usage = await _q(
        db_path,
        "SELECT minute, requests, prompt_tokens, completion_tokens FROM token_usage_minute "
        "ORDER BY minute",
    )
    assert usage == [(1010, 0, 1000, 100), (1011, 0, 0, 200), (1012, 0, 0, 200), (1013, 1, 0, 100)]
    # The lifetime counter is not per minute and stays one row.
    assert await _q(db_path, "SELECT requests, completion_tokens FROM counters") == [(1, 600)]


async def test_without_a_first_token_everything_books_at_finish(db_path):  # noqa: F811
    led = Ledger(db_path, clock=lambda: 60 * 1000 + 5)
    led.record_request(
        model_id="m",
        token_id="t",
        prompt_tokens=10,
        completion_tokens=5,
        cached_tokens=None,
        variant_id=None,
    )
    await led.flush()
    assert await _q(
        db_path, "SELECT minute, requests, prompt_tokens, completion_tokens FROM model_samples"
    ) == [(1000, 1, 10, 5)]


def test_produced_span_streamed_and_not_streamed():
    """Non-streamed requests book by the slot's span (review finding 7): the
    prompt at the slot's minute, the reply spread to the finish."""
    from types import SimpleNamespace

    from app.proxy.routes import _produced_s

    assert (
        _produced_s(SimpleNamespace(first_token_monotonic=95.0, started_monotonic=90.0), 100.0)
        == 5.0
    )
    assert (
        _produced_s(SimpleNamespace(first_token_monotonic=None, started_monotonic=40.0), 100.0)
        == 60.0
    )
    assert _produced_s(None, 100.0) is None
