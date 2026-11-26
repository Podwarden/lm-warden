"""leanprompt's measured score (spec 14.1; leanprompt DESIGN section 8)."""

from __future__ import annotations

from app.lens.leanprompt_score import ScoreRow, leanprompt_score


def row(**kw: object) -> ScoreRow:
    base = dict(
        prompt_tokens=1000, cached_tokens=900, completion_tokens=100, reusable_tokens=900,
        continues=True, finish_reason="stop", bad_call=False,
    )  # fmt: skip
    return ScoreRow(**{**base, **kw})  # type: ignore[arg-type]


def test_under_50_calls_is_not_scored_and_has_no_number() -> None:
    out = leanprompt_score([row()] * 49)
    assert out == {
        "scored": False,
        "reason": "49 of 50 calls needed for a measured score",
        "calls": 49,
    }


def test_the_formula_and_its_inputs() -> None:
    out = leanprompt_score([row()] * 50)
    # efficiency 1.0 (all reusable served), stability 0.9, no waste: 100 x (0.6 x 0.9 + 0.4) = 94
    assert out["prefill_efficiency"] == 1.0 and out["stability"] == 0.9
    assert out["generation_waste"] == 0.0 and out["leanprompt"] == 94
    assert out["kind"] == "measured" and out["calls"] == 50


def test_waste_counts_truncated_and_malformed_calls() -> None:
    rows = [row()] * 40 + [row(finish_reason="length")] * 5 + [row(bad_call=True)] * 5
    out = leanprompt_score(rows)
    assert out["generation_waste"] == 0.2
    assert out["leanprompt"] == 100 * 0 + int(100 * (0.6 * 0.9 + 0.4 * 0.8))


def test_unserved_reusable_tokens_lower_the_efficiency() -> None:
    out = leanprompt_score([row(cached_tokens=450)] * 50)
    assert out["prefill_efficiency"] == 0.5


def test_calls_that_do_not_continue_are_not_in_stability() -> None:
    rows = [row(continues=False, reusable_tokens=None, cached_tokens=None)] * 10 + [row()] * 40
    out = leanprompt_score(rows)
    assert out["continuing_calls"] == 40 and out["stability"] == 0.9


def test_no_continuing_calls_is_not_scored() -> None:
    out = leanprompt_score([row(continues=False)] * 60)
    assert out["scored"] is False and out["calls"] == 60
