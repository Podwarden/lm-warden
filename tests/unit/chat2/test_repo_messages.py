import sqlite3

import pytest

from app.chat2.repo import DEFAULT_SETTINGS, Chat2Repo
from app.db.database import open_db
from tests.conftest import seed_admin_user


def _uid(db_path) -> int:
    seed_admin_user(db_path, username="alice", password="pw-alice-1")
    return sqlite3.connect(db_path).execute("SELECT id FROM users").fetchone()[0]


@pytest.mark.asyncio
async def test_append_allocates_seq_and_lists_in_order(tmp_data_dir, client) -> None:
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    uid = _uid(db_path)
    async with open_db(db_path) as db:
        repo = Chat2Repo(db)
        chat = await repo.create_chat(uid, model="m", settings=DEFAULT_SETTINGS)
        u = await repo.append_message(
            chat.id,
            role="user",
            parts=[{"type": "text", "text": "hi"}],
            settings_snapshot=DEFAULT_SETTINGS,
        )
        a = await repo.append_message(
            chat.id,
            role="assistant",
            parts=[{"type": "text", "text": "hello"}],
            settings_snapshot=DEFAULT_SETTINGS,
            model="m",
            usage={"prompt": 3, "completion": 2},
            finish_reason="stop",
        )
        assert (u.seq, a.seq) == (1, 2)
        rows = await repo.list_messages(chat.id)
        assert [r.role for r in rows] == ["user", "assistant"]
        assert rows[1].usage == {"prompt": 3, "completion": 2}


@pytest.mark.asyncio
async def test_delete_tail_assistant_frees_seq_and_drops_tool_rows(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    uid = _uid(db_path)
    async with open_db(db_path) as db:
        repo = Chat2Repo(db)
        chat = await repo.create_chat(uid, model="m", settings=DEFAULT_SETTINGS)
        await repo.append_message(
            chat.id,
            role="user",
            parts=[{"type": "text", "text": "q"}],
            settings_snapshot=DEFAULT_SETTINGS,
        )
        await repo.append_message(
            chat.id,
            role="assistant",
            parts=[
                {
                    "type": "tool_call",
                    "id": "c1",
                    "name": "present_options",
                    "args": {"options": []},
                    "argsText": "{}",
                    "client": True,
                }
            ],
            settings_snapshot=DEFAULT_SETTINGS,
            finish_reason="tool_calls",
        )
        await repo.append_message(
            chat.id,
            role="tool",
            parts=[
                {"type": "tool_result", "callId": "c1", "result": {"selected": ["a"]}, "ok": True}
            ],
            settings_snapshot=DEFAULT_SETTINGS,
        )
        assert await repo.delete_tail_assistant(chat.id) == 2
        assert [r.role for r in await repo.list_messages(chat.id)] == ["user"]
        assert await repo.delete_tail_assistant(chat.id) is None


@pytest.mark.asyncio
async def test_delete_tail_assistant_is_noop_when_not_trailing(tmp_data_dir, client) -> None:
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    uid = _uid(db_path)
    async with open_db(db_path) as db:
        repo = Chat2Repo(db)
        chat = await repo.create_chat(uid, model="m", settings=DEFAULT_SETTINGS)
        await repo.append_message(
            chat.id,
            role="user",
            parts=[{"type": "text", "text": "q1"}],
            settings_snapshot=DEFAULT_SETTINGS,
        )
        await repo.append_message(
            chat.id,
            role="assistant",
            parts=[{"type": "text", "text": "a1"}],
            settings_snapshot=DEFAULT_SETTINGS,
            finish_reason="stop",
        )
        await repo.append_message(
            chat.id,
            role="user",
            parts=[{"type": "text", "text": "q2"}],
            settings_snapshot=DEFAULT_SETTINGS,
        )
        assert await repo.delete_tail_assistant(chat.id) is None
        rows = await repo.list_messages(chat.id)
        assert [(r.seq, r.role) for r in rows] == [(1, "user"), (2, "assistant"), (3, "user")]


@pytest.mark.asyncio
async def test_fork_copies_prefix_and_appends_edit(tmp_data_dir, client) -> None:
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    uid = _uid(db_path)
    async with open_db(db_path) as db:
        repo = Chat2Repo(db)
        chat = await repo.create_chat(uid, model="m", settings=DEFAULT_SETTINGS)
        for i, role in enumerate(["user", "assistant", "user", "assistant"], start=1):
            await repo.append_message(
                chat.id,
                role=role,
                parts=[{"type": "text", "text": f"t{i}"}],
                settings_snapshot=DEFAULT_SETTINGS,
            )
        # edit message seq 3 -> copy seq < 3, append edited user message
        fork = await repo.fork(uid, chat.id, at_seq=3, edited_text="edited")
        assert fork and fork.forked_from_chat_id == chat.id and fork.forked_at_seq == 3
        rows = await repo.list_messages(fork.id)
        assert [(r.seq, r.role, r.parts[0]["text"]) for r in rows] == [
            (1, "user", "t1"),
            (2, "assistant", "t2"),
            (3, "user", "edited"),
        ]
        # fork from assistant seq 2 without edit -> copy seq <= 2
        fork2 = await repo.fork(uid, chat.id, at_seq=2)
        assert [r.seq for r in await repo.list_messages(fork2.id)] == [1, 2]  # type: ignore
        assert await repo.fork(uid + 1, chat.id, at_seq=2) is None


@pytest.mark.asyncio
async def test_fork_is_atomic_on_failure(tmp_data_dir, client, monkeypatch) -> None:
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    uid = _uid(db_path)
    async with open_db(db_path) as db:
        repo = Chat2Repo(db)
        chat = await repo.create_chat(uid, model="m", settings=DEFAULT_SETTINGS)
        await repo.append_message(
            chat.id,
            role="user",
            parts=[{"type": "text", "text": "t1"}],
            settings_snapshot=DEFAULT_SETTINGS,
        )
        await repo.append_message(
            chat.id,
            role="assistant",
            parts=[{"type": "text", "text": "t2"}],
            settings_snapshot=DEFAULT_SETTINGS,
        )

        original_append = Chat2Repo.append_message
        calls = {"n": 0}

        async def flaky_append(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            return await original_append(self, *args, **kwargs)

        monkeypatch.setattr(Chat2Repo, "append_message", flaky_append)

        chats_before = {c.id for c in await repo.list_chats(uid)}
        with pytest.raises(RuntimeError):
            await repo.fork(uid, chat.id, at_seq=2)

        chats_after = {c.id for c in await repo.list_chats(uid)}
        assert chats_after == chats_before
        assert [r.role for r in await repo.list_messages(chat.id)] == ["user", "assistant"]
