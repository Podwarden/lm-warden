"""Prompt Lens score store (analysis Task 2.3): migration 0064, upsert in place,
the findings table, paging, retention."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.lens.catalogue import Finding, Saving
from app.lens.scoring import SCORE_VERSION, Score, Terms
from app.lens.store import LensStore, SessionScoreRow, decode_cursor, encode_cursor

SQL = Path(__file__).parents[3] / "app" / "db" / "sql" / "0064_lens_scores.sql"


@pytest.fixture
def db() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.executescript(SQL.read_text())
    return c


def score(layer, value, conf="high"):
    if value is None:
        return Score(layer, False, None, None, None, None, 2, "2 of 3 continuing turns")
    return Score(layer, True, value, value - 3, value + 2, conf, 24, None, {"lost": 10})


def row(ref="s_" + "a" * 32, key="k1", last_at=1000.0, turns=24, e=70) -> SessionScoreRow:
    scores = {
        "E": score("E", e),
        "C": score("C", 90),
        "R": score("R", None),
        "overall": score("overall", 78),
    }
    terms = {"E": Terms("E", 1000.0, 0.0, {"E-01": 300.0}).to_json()}
    return SessionScoreRow.of(
        key_id=key,
        sess_ref=ref,
        started=last_at - 600,
        last_at=last_at,
        model="fake-model",
        route="local",
        harness="other",
        turns=turns,
        scores=scores,
        terms=terms,
        coverage={"E": 1.0, "C": 1.0, "R": 0.5},
        fired={"E": ["E-01"]},
        score_version=SCORE_VERSION,
    )


def finding(rule="E-01", tool=None):
    return Finding(
        rule, (2, 3), {"turns": 2}, tool, "sys_changed", Saving(900, 1.5, None, 0.01), "high"
    )


def test_upsert_round_trip_and_scores(db):
    s = LensStore(db)
    s.upsert_session(row(), [finding(), finding("R-04", "fa000001")])
    got = s.session("k1", "s_" + "a" * 32)
    assert got is not None and (got.e, got.c, got.r, got.overall) == (70, 90, None, 78)
    sc = got.scores()
    assert sc["E"].scored and (sc["E"].lo, sc["E"].hi, sc["E"].confidence) == (67, 72, "high")
    assert not sc["R"].scored and sc["R"].reason == "2 of 3 continuing turns"
    assert got.terms["E"]["by_rule"] == {"E-01": 300.0}
    fs = s.findings("k1", "s_" + "a" * 32)
    assert [(f.rule, f.tool) for f in fs] == [("E-01", None), ("R-04", "fa000001")]
    assert fs[0].saving.money == 0.01


def test_resumed_session_is_updated_in_place(db):  # Review Focus 5
    s = LensStore(db)
    s.upsert_session(row(), [finding(), finding("R-04", "fa000001")])
    s.upsert_session(row(last_at=2000.0, turns=30, e=60), [finding()])
    n = db.execute("SELECT COUNT(*) FROM lens_session_scores").fetchone()[0]
    assert n == 1
    got = s.session("k1", "s_" + "a" * 32)
    assert (got.turns, got.e, got.last_at) == (30, 60, 2000.0)
    assert len(s.findings("k1", "s_" + "a" * 32)) == 1


def test_sessions_are_scoped_to_the_key_and_paged(db):
    s = LensStore(db)
    for i in range(5):
        s.upsert_session(row(ref=f"s_{i:032x}", last_at=1000.0 + i), [])
    s.upsert_session(row(ref="s_" + "f" * 32, key="k2", last_at=2000.0), [])
    mine = s.sessions("k1", since=0, limit=10)
    assert [r.last_at for r in mine] == [1004.0, 1003.0, 1002.0, 1001.0, 1000.0]
    assert len(s.sessions(None, since=0, limit=10)) == 6
    page1 = s.sessions("k1", since=0, limit=2)
    page2 = s.sessions("k1", since=0, limit=2, cursor=encode_cursor(page1[-1]))
    assert [r.last_at for r in page1 + page2] == [1004.0, 1003.0, 1002.0, 1001.0]
    assert s.sessions("k1", since=1003.0, limit=10)[-1].last_at == 1003.0
    assert s.sessions("k1", since=0, model="other-model") == []
    assert decode_cursor("garbage") is None


def test_tables_hold_numbers_and_codes_only(db):
    LensStore(db).upsert_session(row(), [finding()])
    cols = [r[1] for r in db.execute("PRAGMA table_info(lens_session_scores)")] + [
        r[1] for r in db.execute("PRAGMA table_info(lens_findings)")
    ]
    assert not {"text", "prompt", "reply", "content", "name"} & set(cols)
