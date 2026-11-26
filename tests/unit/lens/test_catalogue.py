"""Prompt Lens recommendation catalogue (analysis spec 11, 10.2)."""

import json

from app.lens.catalogue import RULES, Finding, Rule, Saving

E_IDS = [f"E-{i:02d}" for i in range(1, 10)]
C_IDS = [f"C-{i:02d}" for i in range(1, 10)]
R_IDS = [f"R-{i:02d}" for i in range(1, 11)]


def test_every_rule_has_threshold_guard_and_copy():
    assert {r.layer for r in RULES.values()} == {"E", "C", "R"}
    assert len(RULES) == 28 and all(r.message and r.fix for r in RULES.values())
    assert all(r.threshold and r.guard for r in RULES.values())


def test_rule_ids_layers_and_keys():
    assert list(RULES) == E_IDS + C_IDS + R_IDS
    for rid, r in RULES.items():
        assert isinstance(r, Rule) and r.id == rid and r.layer == rid[0]


def test_no_rule_advises_moving_dates_to_the_end_or_dropping_reasoning():  # A-6
    text = " ".join(r.message + " " + r.fix for r in RULES.values()).lower()
    assert "move the date to the end" not in text
    assert "drop reasoning" not in text and "drop the reasoning" not in text


def test_copy_and_lp_ids_are_the_specs():
    e01 = RULES["E-01"]
    assert e01.message == (
        "System prompt line {n} changes every {turn/session} ({class}); "
        "the {m} lines after it are recomputed"
    )
    assert e01.fix == (
        "make the value stable (a day, not a time; a fixed path) or move it to the "
        "newest user message; re-check answers (A-6)"
    )
    assert e01.lp == "PC-03, PC-02"
    assert RULES["E-06"].message == "Anthropic caches only at breakpoints; this session has none"
    assert RULES["C-01"].lp == "PC-05" and RULES["R-04"].lp == "GE-10"
    assert RULES["R-01"].lp == "GE-03, GE-05"
    assert RULES["R-09"].lp is None  # the spec's "none"
    assert RULES["R-05"].message == (
        "{tool} was called {k} times with the same arguments and got the same result"
    )


def test_scored_terms_follow_spec_10_2():
    # E: every E rule but the server and provider-minimum one (A-11).
    assert {r for r in E_IDS if not RULES[r].scored} == {"E-08"}
    # C: unused tools, duplicates and repeated results, boilerplate.
    assert {r for r in C_IDS if RULES[r].scored} == {"C-01", "C-02", "C-07"}
    # R: cuts, runaway, malformed, loops, failed retries, empty replies.
    assert {r for r in R_IDS if RULES[r].scored} == {
        "R-01",
        "R-02",
        "R-03",
        "R-04",
        "R-05",
        "R-06",
        "R-07",
    }


def test_finding_json_has_no_text():
    f = Finding("E-01", (1,), {"lost": 9}, None, "sys_changed", Saving(9, None, None, None), "high")
    assert set(f.to_json()) <= {
        "rule",
        "turns",
        "counts",
        "tool",
        "cause",
        "saving",
        "confidence",
        "muted",
    }


def test_finding_json_is_codes_and_numbers_only():
    f = Finding(
        "R-04",
        (7, 9),
        {"bad": 2, "calls": 6},
        "fa000003",
        "schema_undocumented",
        Saving(400, 0.5, None, 0.001),
        "medium",
        muted=True,
    )
    j = f.to_json()
    assert j == {
        "rule": "R-04",
        "turns": [7, 9],
        "counts": {"bad": 2, "calls": 6},
        "tool": "fa000003",
        "cause": "schema_undocumented",
        "saving": {"tokens": 400, "prefill_s": 0.5, "decode_s": None, "money": 0.001},
        "confidence": "medium",
        "muted": True,
    }
    json.dumps(j)  # serialisable as is

    def leaves(v):
        if isinstance(v, dict):
            for x in v.values():
                yield from leaves(x)
        elif isinstance(v, list):
            for x in v:
                yield from leaves(x)
        else:
            yield v

    for leaf in leaves(j):
        assert leaf is None or isinstance(leaf, bool | int | float) or len(leaf) <= 32
