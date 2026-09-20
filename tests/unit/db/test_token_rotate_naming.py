"""#150 — token rotation renames old row to ``"{name} (old N)"`` and mints
a successor with the ORIGINAL name.

Direct ``TokenRepo.rotate()`` coverage so a regression in the repo layer
shows up here before the higher-level endpoint tests get involved.
"""
from datetime import timedelta

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.tokens import TokenRepo, sqlite_utc_in, sqlite_utc_now


@pytest.fixture
async def db(tmp_path):
    async with open_db(tmp_path / "vw.db") as conn:
        await apply_migrations(conn)
        yield conn


async def test_first_rotation_renames_old_to_old_1_and_new_keeps_name(db):
    repo = TokenRepo(db)
    await repo.create("a", "prod-bot", "vw_" + "1" * 32)

    new_id, _plaintext, renamed_to = await repo.rotate("a")

    assert renamed_to == "prod-bot (old 1)"
    rows = {r.id: r for r in await repo.list_all()}
    assert rows["a"].name == "prod-bot (old 1)"
    assert rows[new_id].name == "prod-bot"
    assert rows[new_id].rotated_from == "a"
    assert rows["a"].rotated_at is not None


async def test_second_rotation_advances_to_old_2(db):
    """First rotate burns ``(old 1)``; second rotate must allocate ``(old 2)``."""
    repo = TokenRepo(db)
    await repo.create("a", "prod-bot", "vw_" + "1" * 32)

    new1_id, _p1, renamed1 = await repo.rotate("a")
    assert renamed1 == "prod-bot (old 1)"

    new2_id, _p2, renamed2 = await repo.rotate(new1_id)
    assert renamed2 == "prod-bot (old 2)"

    rows = {r.id: r for r in await repo.list_all()}
    # The (old 1) row from the first rotation is untouched.
    assert rows["a"].name == "prod-bot (old 1)"
    # The previously-active row is now (old 2).
    assert rows[new1_id].name == "prod-bot (old 2)"
    # The freshly-minted row keeps the original name.
    assert rows[new2_id].name == "prod-bot"


async def test_max_plus_one_handles_gaps(db):
    """If ``(old 1)`` was deleted but ``(old 5)`` exists, next slot is 6 — we
    do NOT reuse gaps (cf. ``_next_old_suffix`` docstring on monotonicity)."""
    repo = TokenRepo(db)
    # Hand-place a stale "(old 5)" row to simulate a long-running cluster
    # where intermediate rotations were pruned.
    await repo.create("stale", "prod-bot (old 5)", "vw_" + "9" * 32)
    await repo.create("a", "prod-bot", "vw_" + "1" * 32)

    _new_id, _plaintext, renamed_to = await repo.rotate("a")
    assert renamed_to == "prod-bot (old 6)"


async def test_rotate_of_already_rotated_token_is_rejected(db):
    """Repo-layer guard: already-rotated rows raise ValueError so the route
    can translate to 409 (#150 AC 'rotate of an already-rotated token
    rejected with 4xx')."""
    repo = TokenRepo(db)
    await repo.create("a", "prod-bot", "vw_" + "1" * 32)
    await repo.rotate("a")  # marks "a" rotated_at

    with pytest.raises(ValueError, match="already rotated"):
        await repo.rotate("a")


async def test_old_row_keeps_secret_working_during_grace(db):
    """#150 explicitly preserves the old token's auth during grace —
    rotating must NOT NULL-out the old hash, prefix, or otherwise break
    its lookup-by-plaintext."""
    repo = TokenRepo(db)
    plaintext_old = "vw_" + "a" * 32
    await repo.create("a", "prod-bot", plaintext_old)

    await repo.rotate("a", grace_hours=24)

    # The old plaintext still resolves to a row — bearer auth (#114) will
    # then check revoked_at, but the row itself is intact.
    found = await repo.find_by_plaintext(plaintext_old)
    assert found is not None
    assert found.id == "a"
    assert found.name == "prod-bot (old 1)"
    # priority preserved on the old row (#150 spec).
    assert found.priority == 5  # default carried through


async def test_special_chars_in_name_are_treated_as_literals(db):
    """A token name containing SQL wildcards (``_``, ``%``) must not collide
    with other names via LIKE-pattern interpretation in ``_next_old_suffix``."""
    repo = TokenRepo(db)
    # Hand-place a stale row that would falsely match if we treated ``_`` as
    # the LIKE wildcard (single-char match).
    await repo.create("decoy", "prodXbot (old 9)", "vw_" + "8" * 32)
    await repo.create("a", "prod_bot", "vw_" + "1" * 32)

    _new_id, _plaintext, renamed_to = await repo.rotate("a")
    # If ``_`` were not escaped, ``prodXbot (old 9)`` would match and the
    # next slot would be 10. With proper escaping, only literal
    # ``prod_bot (old N)`` matches and the next slot is 1.
    assert renamed_to == "prod_bot (old 1)"


async def test_concurrent_rotate_does_not_fork_the_chain(db, monkeypatch):
    """M7 (final-review 2026-09-19): a concurrent rotate can fork the chain.

    ``rotate()``'s own top-of-function check (`pred_rotated_at is not
    None`) only catches a rotation that had ALREADY committed by the time
    our SELECT ran. It does nothing for two rotations racing past that
    SELECT at the same time -- both see ``rotated_at IS NULL``, both then
    rename the predecessor, INSERT a successor and mark it rotated. Without
    a guard on the final UPDATE, both commit and the predecessor ends up
    with two successors.

    Simulated deterministically (no real second connection/thread needed):
    ``_next_old_suffix`` is the last thing ``rotate()`` calls before its own
    writes, i.e. exactly the gap between its SELECT and its write phase. We
    monkeypatch it to run and COMMIT a full competing rotation right there,
    reproducing the interleaving a real race would produce.
    """
    repo = TokenRepo(db)
    await repo.create("old", "prod-bot", "vw_" + "1" * 32)
    real_suffix = repo._next_old_suffix

    async def racing_suffix(name: str, scope: str) -> int:
        # A second, independent rotate() of the SAME predecessor lands and
        # commits here -- before our own caller writes anything.
        await db.execute(
            "UPDATE api_tokens SET name = ? WHERE id = 'old'",
            (f"{name} (old 1)",),
        )
        await db.execute(
            "INSERT INTO api_tokens"
            "(id, name, prefix, hash, scope, rotated_from, created_by) "
            "VALUES ('racer-successor', ?, 'vwa_race', 'x', ?, 'old', 'admin')",
            (name, scope),
        )
        await db.execute(
            "UPDATE api_tokens SET rotated_at = ?, revoked_at = ? WHERE id = 'old'",
            (sqlite_utc_now(), sqlite_utc_in(timedelta(hours=1))),
        )
        await db.commit()
        return await real_suffix(name, scope)

    monkeypatch.setattr(repo, "_next_old_suffix", racing_suffix)

    with pytest.raises(ValueError, match="already rotated"):
        await repo.rotate("old")

    # The racer's successor is the ONLY one -- our own (losing) rename and
    # INSERT must have been rolled back, not left half-applied.
    cur = await db.execute("SELECT id FROM api_tokens WHERE rotated_from = 'old'")
    successor_ids = [r[0] for r in await cur.fetchall()]
    assert successor_ids == ["racer-successor"]

    cur = await db.execute("SELECT name, rotated_at FROM api_tokens WHERE id = 'old'")
    name, rotated_at = await cur.fetchone()
    assert name == "prod-bot (old 1)"  # the racer's rename stuck
    assert rotated_at is not None
