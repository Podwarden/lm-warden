"""``turns[].isolated`` comes from the history row, not from the key's current setting.

Spec 2026-11-21 (Plan 2) section 6: a turn records whether it ran isolated, so old turns
read right after the key switches.
"""

from __future__ import annotations

import sqlite3

from app.sessions.model import turn_json
from app.sessions.store import request_row
from tests.unit.sessions.test_store import _insert
from tests.unit.sessions.test_store import db as db  # noqa: F401  (fixture)


def _turn(db: sqlite3.Connection, rid: str) -> dict:  # noqa: F811
    row = request_row(db, rid)
    assert row is not None
    return turn_json(row, conversation_id=None, turn=1, lineage_unknown=False, side_call=False)


def test_a_turn_recorded_isolated_stays_isolated_after_the_key_switched(db):  # noqa: F811
    db.execute(
        "INSERT INTO api_tokens(id, name, prefix, hash, scope, kv_isolation) "
        "VALUES ('tok1', 'k', 'vw_x', 'h', 'inference', 'isolated')"
    )
    _insert(db, "r1", kv_isolated=1)
    db.execute("UPDATE api_tokens SET kv_isolation = 'sharing' WHERE id = 'tok1'")
    db.commit()
    _insert(db, "r2", fin=1010.0, kv_isolated=0)
    assert _turn(db, "r1")["isolated"] is True
    assert _turn(db, "r2")["isolated"] is False


def test_an_old_row_reads_isolated_none(db):  # noqa: F811
    _insert(db, "r0")  # written before 0060: the column is NULL
    assert _turn(db, "r0")["isolated"] is None
