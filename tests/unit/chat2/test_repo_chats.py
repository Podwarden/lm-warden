import sqlite3
from pathlib import Path

import pytest

from app.chat2.repo import DEFAULT_SETTINGS, Chat2Repo
from app.db.database import open_db
from tests.conftest import seed_admin_user


def _users(db_path: Path) -> tuple[int, int]:
    seed_admin_user(db_path, username="alice", password="pw-alice-1")
    seed_admin_user(db_path, username="bob", password="pw-bob-1")
    c = sqlite3.connect(db_path)
    ids = {u: i for i, u in c.execute("SELECT id, username FROM users")}
    return ids["alice"], ids["bob"]


@pytest.mark.asyncio
async def test_create_list_get_scoped_by_user(tmp_data_dir, client) -> None:
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    alice, bob = _users(db_path)
    async with open_db(db_path) as db:
        repo = Chat2Repo(db)
        chat = await repo.create_chat(alice, model="m", settings=DEFAULT_SETTINGS)
        assert chat.title == "New chat" and chat.settings["temperature"] == 0.7
        assert [c.id for c in await repo.list_chats(alice)] == [chat.id]
        assert await repo.list_chats(bob) == []
        assert await repo.get_chat(bob, chat.id) is None
        assert (await repo.get_chat(alice, chat.id)) == chat


@pytest.mark.asyncio
async def test_update_delete_and_defaults(tmp_data_dir, client) -> None:
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    alice, bob = _users(db_path)
    async with open_db(db_path) as db:
        repo = Chat2Repo(db)
        chat = await repo.create_chat(alice, model=None, settings=DEFAULT_SETTINGS)
        upd = await repo.update_chat(
            alice,
            chat.id,
            title="Renamed",
            title_source="user",
            settings={**DEFAULT_SETTINGS, "temperature": 0.2},
        )
        assert upd and upd.title == "Renamed" and upd.settings["temperature"] == 0.2
        assert upd.title_prev is None
        # `title_prev` follows the same dynamic-SET rule as every other field:
        # only written when passed, so a later rename (or a settings-only
        # PATCH) can never wipe the undo target.
        undo = await repo.update_chat(alice, chat.id, title="Auto", title_prev="Renamed")
        assert undo and undo.title == "Auto" and undo.title_prev == "Renamed"
        kept = await repo.update_chat(alice, chat.id, title="Again")
        assert kept and kept.title == "Again" and kept.title_prev == "Renamed"
        assert await repo.update_chat(bob, chat.id, title="x") is None
        assert await repo.delete_chat(bob, chat.id) is False
        assert await repo.delete_chat(alice, chat.id) is True
        assert await repo.get_defaults(alice) is None
        await repo.put_defaults(alice, model="m2", settings={**DEFAULT_SETTINGS, "top_p": 0.9})
        model, settings = await repo.get_defaults(alice)  # type: ignore[misc]
        assert model == "m2" and settings["top_p"] == 0.9
        await repo.create_chat(alice, model=None, settings=DEFAULT_SETTINGS)
        await repo.create_chat(alice, model=None, settings=DEFAULT_SETTINGS)
        assert await repo.delete_all_chats(alice) == 2
        assert await repo.list_chats(alice) == []


@pytest.mark.asyncio
async def test_replace_auto_title_loses_to_a_concurrent_user_rename(tmp_data_dir, client) -> None:
    """The title task decides to rename, then writes -- and a user rename can
    land in between (the call sits at the lowest proxy priority, so that window
    is seconds wide, not microseconds).

    Re-reading `title_source` before the write narrows it but cannot close it.
    The `AND title_source = 'auto'` predicate does: the user's title wins by
    construction, and the caller can see that it did.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    alice, bob = _users(db_path)
    async with open_db(db_path) as db:
        repo = Chat2Repo(db)
        chat = await repo.create_chat(alice, model=None, settings=DEFAULT_SETTINGS)
        await repo.update_chat(alice, chat.id, title="First topic")

        # the reassessment displaces the old title in ONE statement
        assert (
            await repo.replace_auto_title(alice, chat.id, title="Second topic", keep_prev=True)
            is True
        )
        row = await repo.get_chat(alice, chat.id)
        assert row and row.title == "Second topic" and row.title_prev == "First topic"

        # the user renames -> title_source flips, and the next reassessment is
        # refused at the SQL level rather than by a racy read-then-write
        await repo.update_chat(alice, chat.id, title="Mine", title_source="user")
        assert (
            await repo.replace_auto_title(alice, chat.id, title="Third topic", keep_prev=True)
            is False
        )
        row = await repo.get_chat(alice, chat.id)
        # untouched -- note this raw `update_chat` did not pass
        # `clear_title_prev`, which is the route's job (see
        # `test_user_rename_clears_title_prev`), so the old undo target
        # survives here. What matters is that the refused reassessment wrote
        # NOTHING: not the title, and not `title_prev` either.
        assert row and row.title == "Mine" and row.title_prev == "First topic"

        # ...and it is scoped by user like every other write in this repo
        assert await repo.replace_auto_title(bob, chat.id, title="X", keep_prev=True) is False


@pytest.mark.asyncio
async def test_replace_auto_title_can_skip_title_prev(tmp_data_dir, client) -> None:
    """The FIRST titling replaces the first-line fallback, which is not a title
    anyone chose -- offering to undo back to it would be noise."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    alice, _ = _users(db_path)
    async with open_db(db_path) as db:
        repo = Chat2Repo(db)
        chat = await repo.create_chat(alice, model=None, settings=DEFAULT_SETTINGS)
        await repo.update_chat(alice, chat.id, title="how do i mount this")
        assert (
            await repo.replace_auto_title(
                alice, chat.id, title="Mounting a volume", keep_prev=False
            )
            is True
        )
        row = await repo.get_chat(alice, chat.id)
        assert row and row.title == "Mounting a volume" and row.title_prev is None


@pytest.mark.asyncio
async def test_user_rename_clears_title_prev(tmp_data_dir, client) -> None:
    """Once the user picks a title, the machine-generated one it displaced is
    no longer a sensible thing to offer as an undo target."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    alice, _ = _users(db_path)
    async with open_db(db_path) as db:
        repo = Chat2Repo(db)
        chat = await repo.create_chat(alice, model=None, settings=DEFAULT_SETTINGS)
        await repo.replace_auto_title(alice, chat.id, title="Auto two", keep_prev=True)
        upd = await repo.update_chat(
            alice, chat.id, title="Mine", title_source="user", clear_title_prev=True
        )
        assert upd and upd.title == "Mine" and upd.title_prev is None
