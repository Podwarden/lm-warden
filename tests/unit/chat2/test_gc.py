import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from app.chat2 import storage
from app.chat2.gc import collect_once
from tests.conftest import seed_admin_user


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _seed(db_path, data_dir, *, sha, created, message_id="m", size=10):
    # _seed is called more than once per test (e.g. two attachments for the
    # same user in the LRU test) — seed_admin_user isn't idempotent (UNIQUE
    # constraint on username), so swallow the repeat-insert error.
    try:
        seed_admin_user(db_path, username="alice", password="pw-alice-1")
    except sqlite3.IntegrityError:
        pass
    c = sqlite3.connect(db_path)
    (uid,) = c.execute("SELECT id FROM users").fetchone()
    c.execute(
        "INSERT OR IGNORE INTO chats(id,user_id,title,settings_json,created_at,updated_at) "
        "VALUES ('c1',?,'t','{}',?,?)",
        (uid, created, created),
    )
    c.execute(
        "INSERT INTO attachments(id,user_id,chat_id,message_id,kind,mime,size_bytes,sha256,"
        "created_at) VALUES (?,?,?,?,'image','image/png',?,?,?)",
        (f"a-{sha[:6]}", uid, "c1", message_id, size, sha, created),
    )
    c.commit()
    p = storage.file_path(data_dir, uid, sha, "png")
    storage.write_file(p, b"x" * size)
    return uid, p


@pytest.mark.asyncio
async def test_orphans_both_directions_and_stale_drafts(tmp_data_dir, client) -> None:
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    now = datetime.now(UTC)
    uid, kept = _seed(db_path, tmp_data_dir, sha="a" * 64, created=_iso(now))
    # file without row
    storage.write_file(storage.file_path(tmp_data_dir, uid, "b" * 64, "png"), b"zz")
    # row without file
    c = sqlite3.connect(db_path)
    c.execute(
        "INSERT INTO attachments(id,user_id,chat_id,message_id,kind,mime,size_bytes,sha256,"
        "created_at) VALUES ('nofile',?, 'c1','m','image','image/png',1,?,?)",
        (uid, "c" * 64, _iso(now)),
    )
    # stale draft (message_id NULL, 8 days old)
    c.execute(
        "INSERT INTO attachments(id,user_id,chat_id,message_id,kind,mime,size_bytes,sha256,"
        "created_at) VALUES ('draft',?, 'c1',NULL,'image','image/png',1,?,?)",
        (uid, "d" * 64, _iso(now - timedelta(days=8))),
    )
    c.commit()
    storage.write_file(storage.file_path(tmp_data_dir, uid, "d" * 64, "png"), b"d")
    rep = await collect_once(client.app.state.settings, now=_iso(now))
    assert (rep.files_without_rows, rep.rows_without_files, rep.stale_drafts) == (1, 1, 1)
    assert kept.exists()
    assert not storage.file_path(tmp_data_dir, uid, "b" * 64, "png").exists()
    assert not storage.file_path(tmp_data_dir, uid, "d" * 64, "png").exists()


@pytest.mark.asyncio
async def test_ttl_eviction_keeps_row_sets_evicted_at(tmp_data_dir, client) -> None:
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    old = datetime.now(UTC) - timedelta(days=91)
    uid, path = _seed(db_path, tmp_data_dir, sha="e" * 64, created=_iso(old))
    rep = await collect_once(client.app.state.settings, now=_iso(datetime.now(UTC)))
    assert rep.evicted_ttl == 1 and not path.exists()
    c = sqlite3.connect(db_path)
    (ev,) = c.execute("SELECT evicted_at FROM attachments WHERE sha256 = ?", ("e" * 64,)).fetchone()
    assert ev is not None


@pytest.mark.asyncio
async def test_lru_eviction_near_quota(tmp_data_dir, client) -> None:
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    now = datetime.now(UTC)
    uid, p_old = _seed(
        db_path, tmp_data_dir, sha="1" * 64, created=_iso(now - timedelta(days=2)), size=60
    )
    _, p_new = _seed(db_path, tmp_data_dir, sha="2" * 64, created=_iso(now), size=60)
    s = client.app.state.settings
    object.__setattr__(s, "chat_quota_user_bytes", 100)  # used 120 > 90% → evict oldest until < 80
    rep = await collect_once(s, now=_iso(now))
    assert rep.evicted_lru == 1 and not p_old.exists() and p_new.exists()


@pytest.mark.asyncio
async def test_lru_eviction_never_evicts_a_live_draft(tmp_data_dir, client) -> None:
    """Spec §3: "Eviction never touches drafts younger than the draft TTL."

    The oldest group here is a draft the user is still composing with, so LRU
    must skip it and evict the next-oldest linked group instead — otherwise the
    composer's thumbnail vanishes and the turn dies on `attachment_missing`.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    now = datetime.now(UTC)
    uid, p_draft = _seed(
        db_path,
        tmp_data_dir,
        sha="3" * 64,
        created=_iso(now - timedelta(days=3)),
        message_id=None,
        size=60,
    )
    _, p_linked = _seed(
        db_path, tmp_data_dir, sha="4" * 64, created=_iso(now - timedelta(days=1)), size=60
    )
    s = client.app.state.settings
    object.__setattr__(s, "chat_quota_user_bytes", 100)  # 120 used > 90% → LRU runs
    rep = await collect_once(s, now=_iso(now))
    assert rep.evicted_lru == 1
    assert p_draft.exists(), "a draft younger than DRAFT_TTL_DAYS must never be LRU-evicted"
    assert not p_linked.exists()
    c = sqlite3.connect(db_path)
    (ev,) = c.execute("SELECT evicted_at FROM attachments WHERE sha256 = ?", ("3" * 64,)).fetchone()
    assert ev is None


@pytest.mark.asyncio
async def test_a_row_with_an_unknown_mime_is_skipped_not_fatal(tmp_data_dir, client) -> None:
    """One row whose mime left `ALLOWED_IMAGE_MIMES` (whitelist narrowed, hand-
    edited row) used to KeyError out of `collect_once` and abort the whole pass
    — every later row, every later user. It must be skipped and logged instead,
    with the rest of the pass unaffected."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    now = datetime.now(UTC)
    uid, _kept = _seed(db_path, tmp_data_dir, sha="5" * 64, created=_iso(now))
    c = sqlite3.connect(db_path)
    # a stale draft in an unsupported mime + a stale draft in a good one: the
    # good one must still be collected despite the bad row sorting first.
    c.execute(
        "INSERT INTO attachments(id,user_id,chat_id,message_id,kind,mime,size_bytes,sha256,"
        "created_at) VALUES ('badmime',?, 'c1',NULL,'image','image/gif',1,?,?)",
        (uid, "6" * 64, _iso(now - timedelta(days=8))),
    )
    c.execute(
        "INSERT INTO attachments(id,user_id,chat_id,message_id,kind,mime,size_bytes,sha256,"
        "created_at) VALUES ('okdraft',?, 'c1',NULL,'image','image/png',1,?,?)",
        (uid, "7" * 64, _iso(now - timedelta(days=8))),
    )
    c.commit()
    storage.write_file(storage.file_path(tmp_data_dir, uid, "7" * 64, "png"), b"z")
    rep = await collect_once(client.app.state.settings, now=_iso(now))
    assert rep.stale_drafts == 2
    assert not storage.file_path(tmp_data_dir, uid, "7" * 64, "png").exists()
