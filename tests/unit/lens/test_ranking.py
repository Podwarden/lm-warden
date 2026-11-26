"""Ranking the fixes (analysis Task 2.2, spec 12.2)."""

from app.chat2.ledger import RateCard
from app.lens.catalogue import RULES, Finding, Saving
from app.lens.ranking import is_fix, pinned_notice, rank_key, top_fixes

CARD = RateCard(
    "c1", None, "prov", "m", 100_000, 400_000, None, 10_000, None, 0, "USD", "2026-11-21"
)


def f(rule, tokens, *, money=None, prefill=None, conf="high", muted=False, cause=None, **counts):
    return Finding(
        rule, (2, 3), dict(counts), None, cause, Saving(tokens, prefill, None, money), conf, muted
    )


def test_own_engine_without_card_ranks_in_seconds():
    a, b = f("C-02", 9_000, prefill=1.0), f("E-01", 3_000, prefill=4.0)
    assert [x.rule for x in top_fixes([a, b], None)] == ["E-01", "C-02"]


def test_money_ranks_when_a_card_is_known():
    a, b = f("E-01", 100, money=0.5), f("C-01", 5_000, money=0.1)
    assert [x.rule for x in top_fixes([b, a], CARD)] == ["E-01", "C-01"]


def test_without_money_or_seconds_tokens_rank():
    a, b = f("E-01", 100), f("C-01", 5_000)
    assert [x.rule for x in top_fixes([a, b], None)] == ["C-01", "E-01"]


def test_confidence_weights_the_rank():
    hi, lo = f("E-02", 1_000, prefill=1.0), f("E-01", 2_000, prefill=2.0, conf="low")
    assert rank_key(hi, None) == 1.0 and rank_key(lo, None) == 0.6
    assert [x.rule for x in top_fixes([lo, hi], None)] == ["E-02", "E-01"]


def test_pinned_and_unscored_are_never_ranked():
    nine = f("R-09", 0, info=1)
    fs = [
        nine,
        f("R-08", 9_999, prefill=9.0),
        f("C-05", 9_999, prefill=9.0),
        f("E-01", 10, prefill=1.0),
    ]
    assert [x.rule for x in top_fixes(fs, None)] == ["E-01"]
    assert pinned_notice(fs) is nine
    assert pinned_notice([f("E-01", 1)]) is None


def test_muted_and_server_causes_are_never_top():
    fs = [
        f("E-01", 9_000, prefill=9.0, muted=True),
        f("E-08", 9_000, prefill=9.0),
        f("R-10", 9_000, prefill=9.0, cause="server_load"),
        f("E-02", 5, prefill=0.1),
    ]
    assert [x.rule for x in top_fixes(fs, None)] == ["E-02"]
    assert not is_fix(fs[0])


def test_at_most_three_ties_by_rule_order():
    fs = [f(r, 1_000, prefill=1.0) for r in ("E-04", "E-02", "E-03", "E-01")]
    assert [x.rule for x in top_fixes(fs, None)] == ["E-01", "E-02", "E-03"]
    assert list(RULES).index("E-01") < list(RULES).index("E-02")
