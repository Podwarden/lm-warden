from datetime import UTC, datetime

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.tokens import TokenRepo

_SQLITE_UTC_FMT = "%Y-%m-%d %H:%M:%S"


@pytest.fixture
async def db(tmp_path):
    async with open_db(tmp_path / "vw.db") as conn:
        await apply_migrations(conn)
        yield conn


async def test_create_with_expiry(db):
    repo = TokenRepo(db)
    await repo.create(
        token_id="t1", name="ci", plaintext="vw_aaaabbbbccccddddeeeeffffgggg", expires_in_days=30
    )
    rows = await repo.list_all()
    assert len(rows) == 1
    row = rows[0]
    assert row.expires_at is not None
    assert row.rotated_at is None


async def test_create_never_expires_when_zero(db):
    repo = TokenRepo(db)
    await repo.create(
        token_id="t2", name="never", plaintext="vw_aaaabbbbccccddddeeeeffffhhhh", expires_in_days=0
    )
    rows = await repo.list_all()
    assert len(rows) == 1
    row = rows[0]
    assert row.expires_at is None


async def test_rotate_sets_pointers(db):
    repo = TokenRepo(db)
    # Create the "old" token
    await repo.create(token_id="old", name="original", plaintext="vw_aaaabbbbccccddddeeeeffffoooo")

    # Rotate it
    new_id, _new_plaintext, renamed_to = await repo.rotate("old", grace_hours=24)

    # Fetch old row
    all_rows = await repo.list_all()
    old_row = next(r for r in all_rows if r.id == "old")
    new_row = next(r for r in all_rows if r.id == new_id)

    # Old token should have rotated_at set and revoked_at scheduled (grace window)
    assert old_row.rotated_at is not None
    assert old_row.revoked_at is not None

    # New token should point back at old via rotated_from
    assert new_row.rotated_from == "old"
    # #150 — predecessor was renamed to "original (old 1)"; successor keeps "original".
    assert old_row.name == "original (old 1)"
    assert new_row.name == "original"
    assert renamed_to == "original (old 1)"


async def test_find_by_plaintext_returns_extended_fields(db):
    """find_by_plaintext must surface the new nullable rotation/expiry columns."""
    repo = TokenRepo(db)
    plaintext = "vw_aaaabbbbccccddddeeeeffffpppp"
    await repo.create(token_id="t3", name="ext", plaintext=plaintext, expires_in_days=30)

    row = await repo.find_by_plaintext(plaintext)
    assert row is not None
    assert row.expires_at is not None
    assert row.created_at is not None
    assert row.rotated_at is None
    assert row.rotated_from is None


async def test_rotate_new_token_is_independently_authable(db):
    """After rotation the successor can be looked up by its own plaintext."""
    repo = TokenRepo(db)
    await repo.create(token_id="old2", name="original", plaintext="vw_aaaabbbbccccddddeeeeffffqqqq")

    _new_id, new_plaintext, _renamed_to = await repo.rotate("old2", grace_hours=24)

    found = await repo.find_by_plaintext(new_plaintext)
    assert found is not None


async def test_rotate_grace_window_is_in_the_future(db):
    """The predecessor's revoked_at must be strictly after the moment rotate() was called."""
    repo = TokenRepo(db)
    await repo.create(token_id="old3", name="original", plaintext="vw_aaaabbbbccccddddeeeeffffrrrr")

    before = datetime.now(UTC)
    await repo.rotate("old3", grace_hours=24)

    all_rows = await repo.list_all()
    old_row = next(r for r in all_rows if r.id == "old3")
    assert old_row.revoked_at is not None

    revoked_at_dt = datetime.strptime(old_row.revoked_at, _SQLITE_UTC_FMT).replace(tzinfo=UTC)
    # Grace is 24 h; before is ≈ now — so revoked_at must be well after `before` (allow 1 s slop).
    assert revoked_at_dt > before


async def test_rotate_atomicity_both_rows_consistent(db):
    """After rotate() both successor and marked predecessor are visible in one list_all() read."""
    repo = TokenRepo(db)
    await repo.create(token_id="old4", name="original", plaintext="vw_aaaabbbbccccddddeeeeffffssss")

    new_id, _plaintext, _renamed_to = await repo.rotate("old4", grace_hours=1)

    all_rows = await repo.list_all()
    ids = {r.id for r in all_rows}
    assert "old4" in ids
    assert new_id in ids

    new_row = next(r for r in all_rows if r.id == new_id)
    old_row = next(r for r in all_rows if r.id == "old4")
    assert new_row.rotated_from == "old4"
    assert old_row.rotated_at is not None
