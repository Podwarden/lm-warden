"""Won't fix: the mark store and marks on stored rows (analysis Task 3.3, spec 11.4)."""

from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from app.lens.store import SessionScoreRow
from app.lens.wontfix import (
    UNMARKABLE,
    WontfixError,
    WontfixRepo,
    apply_marks,
    exclude_for,
)

SQL = """
CREATE TABLE api_tokens (id TEXT PRIMARY KEY, rotated_from TEXT, created_at REAL DEFAULT 0);
CREATE TABLE lens_wontfix (
  key_id TEXT NOT NULL, rule TEXT NOT NULL, version TEXT NOT NULL DEFAULT '',
  by TEXT NOT NULL, at REAL NOT NULL, PRIMARY KEY (key_id, rule, version));
"""


@pytest.fixture
def repo() -> WontfixRepo:
    db = sqlite3.connect(":memory:")
    db.executescript(SQL)
    db.executemany(
        "INSERT INTO api_tokens VALUES (?, ?, ?)",
        [("old", None, 1), ("new", "old", 2), ("other", None, 3)],
    )
    return WontfixRepo(db)


def test_mark_is_idempotent_and_unmark_reports(repo):
    assert repo.mark("new", "C-01", None, "key") is True
    assert repo.mark("new", "C-01", None, "admin") is False  # keeps the first
    assert [(m.rule, m.version, m.by) for m in repo.marks("new")] == [("C-01", None, "key")]
    assert repo.unmark("new", "C-01", None) is True
    assert repo.unmark("new", "C-01", None) is False
    assert repo.marks("new") == []


def test_a_version_mark_and_an_all_versions_mark_are_two_marks(repo):
    repo.mark("new", "C-01", "v4id", "key")
    repo.mark("new", "C-01", None, "key")
    assert len(repo.marks("new")) == 2
    assert repo.unmark("new", "C-01", "v4id") and len(repo.marks("new")) == 1


def test_active_applies_to_the_version_or_all(repo):
    repo.mark("new", "C-01", "aaaa1111", "key")
    repo.mark("new", "E-01", None, "admin")
    assert repo.active("new", "aaaa1111") == {"C-01", "E-01"}
    assert repo.active("new", "bbbb2222") == {"E-01"}  # a limited mark stops on the next version
    assert repo.active("new", None) == {"E-01"}
    assert repo.active_marks("new", "aaaa1111") == {"C-01": "key", "E-01": "admin"}


def test_marks_follow_the_rotation_lineage_and_not_other_keys(repo):
    repo.mark("old", "C-01", None, "key")
    assert repo.active("new", None) == {"C-01"}
    assert repo.active("old", None) == {"C-01"}
    assert repo.active("other", None) == frozenset()
    assert repo.mark("new", "C-01", None, "key") is False  # already marked on the chain
    assert repo.unmark("new", "C-01", None) is True  # removes the predecessor's mark too
    assert repo.active("old", None) == frozenset()


@pytest.mark.parametrize("rule", ["R-09", "X-99", "", "c-01"])
def test_refused_rules(repo, rule):
    assert "R-09" in UNMARKABLE
    with pytest.raises(WontfixError):
        repo.mark("new", rule, None, "key")


def test_unknown_actor_is_refused(repo):
    with pytest.raises(WontfixError):
        repo.mark("new", "C-01", None, "someone")


def _row(**over) -> SessionScoreRow:
    terms = {
        "E": {
            "total": 10_000.0,
            "base": 0.0,
            "by_rule": {"E-01": 2_000.0, "E-02": 1_000.0},
            "turns": 10,
            "inputs": {"ideal": 10_000, "lost": 3_000},
            "excluded": [],
        },
        "C": {
            "total": 50_000.0,
            "base": 0.0,
            "by_rule": {"C-01": 5_000.0, "C-02": 2_500.0},
            "turns": 10,
            "inputs": {"waste": 7_500, "prompt": 50_000},
            "excluded": [],
        },
        "R": {
            "total": 20.0,
            "base": 4_000.0,
            "by_rule": {"R-04:turns": 4.0, "R-04:out": 800.0, "R-01:turns": 2.0, "R-01:out": 600.0},
            "turns": 20,
            "inputs": {"turns": 20, "bad": 6, "completion": 4_000, "waste_out": 1_400},
            "excluded": [],
        },
    }
    base = dict(
        key_id="k",
        sess_ref="s_1",
        started=1.0,
        last_at=2.0,
        model="m",
        route="local",
        harness="h",
        version="v1",
        turns=20,
        e=70,
        c=70,
        r=round(100 * (1 - 0.5 * 6 / 20 - 0.5 * 1400 / 4000)),
        overall=70,
        terms=terms,
    )
    return SessionScoreRow(**{**base, **over})


def test_apply_marks_recomputes_from_the_stored_terms():
    row = _row()
    assert apply_marks(row, frozenset()) is row
    e = apply_marks(row, frozenset({"E-01"}))
    assert e.e == 90 and e.c == row.c and e.r == row.r  # 100 * (1 - 1000/10000)
    c = apply_marks(row, frozenset({"C-01"}))
    assert c.c == 90  # 100 * (1 - 2 * 2500 / 50000)
    r = apply_marks(row, frozenset({"R-04"}))
    assert r.r == round(100 * (1 - 0.5 * 2 / 20 - 0.5 * 600 / 4000))


def test_apply_marks_leaves_a_layer_without_the_rule_untouched_and_mixes_the_overall():
    row = _row()
    out = apply_marks(row, frozenset({"E-01"}))
    # overall = weighted mean of E 90, C 70, R as stored, weights .45 / .15 / .40
    want = round((0.45 * 90 + 0.15 * 70 + 0.40 * row.r) / 1.0)
    assert out.overall == want
    assert apply_marks(row, frozenset({"E-77"})).overall == row.overall  # no such term


def test_apply_marks_never_scores_a_layer_that_was_not_scored():
    row = _row(e=None, overall=None)
    out = apply_marks(row, frozenset({"E-01", "C-01"}))
    assert out.e is None and out.overall is None and out.c == 90


def test_apply_marks_follows_the_rows_version():
    repo = WontfixRepo(sqlite3.connect(":memory:"))
    repo.db.executescript(SQL)
    repo.mark("k", "E-01", "v1", "key")
    marks = repo.marks("k")
    assert exclude_for(marks, "v1") == {"E-01"} and exclude_for(marks, "v2") == frozenset()
    assert replace(_row(), version="v2").version == "v2"
