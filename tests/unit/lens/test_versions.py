"""Prompt versions (analysis Task 3.1, spec 13.2)."""

import os
import sqlite3

import pytest

from app.cobs2.keys import NetworkKey
from app.lens.probes import sys_line_digests
from app.lens.versions import (
    client_version,
    lineage_ids,
    version_id,
    version_label,
    volatile_lines,
)

SALT = b"s" * 8


@pytest.fixture
def key() -> NetworkKey:
    return NetworkKey(epoch=1, key=os.urandom(32))


def _vid(key, text, tools="t1"):
    return version_id(sys_line_digests(text, key, SALT), volatile_lines(text), tools, key, SALT)


BASE = "You are a coding agent.\nBe brief.\nToday is 2026-11-20.\nWorking directory: /w/one"


def test_volatile_only_change_keeps_the_version(key):
    other = BASE.replace("2026-11-20", "2026-11-21").replace("/w/one", "/w/two")
    assert _vid(key, BASE) == _vid(key, other)


def test_stable_line_change_starts_a_new_version(key):
    assert _vid(key, BASE) != _vid(key, BASE.replace("Be brief.", "Be thorough."))


def test_added_or_removed_volatile_line_keeps_the_version(key):
    assert _vid(key, BASE) == _vid(key, BASE + "\nSession id: 1f0c2a9e7d414b8e9c3a")
    assert _vid(key, BASE) == _vid(key, BASE.replace("Today is 2026-11-20.\n", ""))


def test_toolset_change_starts_a_new_version(key):
    assert _vid(key, BASE, "t1") != _vid(key, BASE, "t2")
    assert _vid(key, BASE, None) != _vid(key, BASE, "t1")


def test_a_number_alone_is_not_volatile(key):
    # "limit 50" -> "limit 80" is a prompt edit, so it is a version.
    assert _vid(key, "Read at most 50 lines.") != _vid(key, "Read at most 80 lines.")


def test_the_id_is_8_hex_and_keyed(key):
    v = _vid(key, BASE)
    assert len(v) == 8 and int(v, 16) >= 0
    assert v != _vid(NetworkKey(epoch=1, key=os.urandom(32)), BASE)


def test_crlf_and_trailing_space_do_not_change_it(key):
    assert _vid(key, BASE) == _vid(key, BASE.replace("\n", "\r\n") + "\r\n")


def test_a_huge_line_is_bounded(key):
    big = "x" * 1_000_000
    assert volatile_lines(big) == frozenset()
    assert len(_vid(key, big)) == 8


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("v4", "h:v4"),
        ("  spring prompt 2  ", "h:spring prompt 2"),
        (None, None),
        ("", None),
        ("   ", None),
        ("x" * 65, None),
        ("x" * 64, "h:" + "x" * 64),
        ("bad\x00name", None),
        ("bad\nname", None),
        ("café", None),
    ],
)
def test_client_version_is_cleaned_like_a_batch_id(raw, want):
    assert client_version(raw) == want


def _db():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE api_tokens (id TEXT PRIMARY KEY, rotated_from TEXT, created_at REAL DEFAULT 0)"
    )
    db.execute(
        "CREATE TABLE lens_session_scores (key_id TEXT, sess_ref TEXT, started REAL, model TEXT,"
        " version TEXT)"
    )
    return db


def test_lineage_ids_walks_both_ways_and_survives_a_cycle():
    db = _db()
    db.executemany(
        "INSERT INTO api_tokens VALUES (?, ?, ?)", [("a", None, 1), ("b", "a", 2), ("c", "b", 3)]
    )
    assert lineage_ids(db, "b") == ["a", "b", "c"]
    assert lineage_ids(db, "c") == ["a", "b", "c"]
    assert lineage_ids(db, "zz") == ["zz"]
    db.execute("UPDATE api_tokens SET rotated_from = 'c' WHERE id = 'a'")
    assert sorted(lineage_ids(db, "a")) == ["a", "b", "c"]


def test_version_label_is_the_ordinal_by_first_seen_per_model():
    db = _db()
    db.executemany(
        "INSERT INTO lens_session_scores VALUES (?, ?, ?, ?, ?)",
        [
            ("a", "s1", 100.0, "m", "aaaa1111"),
            ("a", "s2", 300.0, "m", "bbbb2222"),
            ("b", "s3", 200.0, "m", "cccc3333"),  # the rotated successor, same chain
            ("a", "s4", 50.0, "other", "zzzz9999"),  # another model: not counted
        ],
    )
    chain = ["a", "b"]
    assert version_label(db, chain, "m", "aaaa1111") == {"label": "v1", "since": 100.0}
    assert version_label(db, chain, "m", "cccc3333") == {"label": "v2", "since": 200.0}
    assert version_label(db, chain, "m", "bbbb2222") == {"label": "v3", "since": 300.0}
    assert version_label(db, chain, "m", "new00000", first_seen=400.0) == {
        "label": "v4",
        "since": 400.0,
    }


def test_a_header_named_version_keeps_its_name_and_is_not_counted():
    db = _db()
    db.executemany(
        "INSERT INTO lens_session_scores VALUES (?, ?, ?, ?, ?)",
        [("a", "s1", 100.0, "m", "h:spring"), ("a", "s2", 200.0, "m", "aaaa1111")],
    )
    assert version_label(db, ["a"], "m", "h:spring") == {"label": "spring", "since": 100.0}
    assert version_label(db, ["a"], "m", "aaaa1111") == {"label": "v1", "since": 200.0}
    assert version_label(db, ["a"], "m", "h:new", first_seen=5.0) == {"label": "new", "since": 5.0}
