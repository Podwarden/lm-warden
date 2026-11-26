"""Migration 0060: per-key KV sharing columns, the model weight and ``kv_retired_scopes``.

Covers the plan 2 rulings that bind the schema:
  * O1: every key, new and existing, starts as ``sharing``.
  * No token row is ever left with an empty ``kv_tenant_id``: the tenant tag derivation fails
    closed on an empty id. Existing rows are backfilled by the migration itself, rows made by
    ``TokenRepo.create`` get their own id, and a raw INSERT gets one from the trigger.
  * Rotation copies the whole KV policy, tenant id included, so a successor keeps its cache.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from app.db import migrations
from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import _MODEL_COLS, ModelRepo, ModelRow
from app.db.repos.tokens import _POLICY_COLS, TokenRepo, TokenRow, hash_token

KV_TOKEN_COLS = ("kv_isolation", "kv_tenant_id", "kv_epoch", "kv_max_pct_ram", "kv_max_pct_nvme")


def _q(db_path: Path, sql: str, args: tuple = ()) -> list[tuple]:
    with sqlite3.connect(db_path) as c:
        return c.execute(sql, args).fetchall()


async def _db_at_0058(
    tmp_data_dir: Path, tmp_path: Path, monkeypatch, *, chains: bool = False
) -> Path:
    """A database migrated to 0058, as lmwarden.com is before the deploy."""
    old_sql = tmp_path / "sql"
    old_sql.mkdir()
    for p in migrations.SQL_DIR.glob("*.sql"):
        if p.name < "0060":
            shutil.copy(p, old_sql / p.name)
    db_path = tmp_data_dir / "vllm-warden.db"
    real_sql = migrations.SQL_DIR
    monkeypatch.setattr(migrations, "SQL_DIR", old_sql)
    async with open_db(db_path) as db:
        await apply_migrations(db)
        for tid, scope, revoked in (
            ("tok-a", "inference", None),
            ("tok-b", "inference", "2999-01-01 00:00:00"),  # rotated, in its grace window
            ("tok-c", "inference", "2000-01-01 00:00:00"),  # long dead
            ("adm-1", "admin", None),
        ):
            plain = f"vw_{tid}_secret_secret_secret"
            await db.execute(
                "INSERT INTO api_tokens(id, name, prefix, hash, scope, revoked_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (tid, tid, plain[:8], hash_token(plain), scope, revoked),
            )
        if chains:
            for tid, prev, revoked in (
                ("root", None, "2999-01-01 00:00:00"),
                ("succ-1", "root", "2999-01-01 00:00:00"),
                ("succ-2", "succ-1", None),
                ("orphan", None, None),
            ):
                await db.execute(
                    "INSERT INTO api_tokens(id, name, prefix, hash, rotated_from, revoked_at) "
                    "VALUES (?, ?, 'vw_x', ?, ?, ?)",
                    (tid, tid, f"h-{tid}", prev, revoked),
                )
        await db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "extra_args, status, extra_env) VALUES ('m1', 'm1', 'o/r', 'main', '[0]', '[]', "
            "'registered', '{}')"
        )
        await db.commit()
    monkeypatch.setattr(migrations, "SQL_DIR", real_sql)
    return db_path


async def _upgrade(db_path: Path) -> None:
    async with open_db(db_path) as db:
        await apply_migrations(db)


async def test_0060_backfills_every_existing_key_as_sharing_with_its_own_tenant(
    tmp_data_dir, tmp_path, monkeypatch
):
    db_path = await _db_at_0058(tmp_data_dir, tmp_path, monkeypatch)
    await _upgrade(db_path)
    rows = _q(
        db_path,
        "SELECT id, kv_isolation, kv_tenant_id, kv_epoch, kv_max_pct_ram, kv_max_pct_nvme "
        "FROM api_tokens ORDER BY id",
    )
    assert rows == [
        ("adm-1", "sharing", "adm-1", 0, None, None),
        ("tok-a", "sharing", "tok-a", 0, None, None),
        ("tok-b", "sharing", "tok-b", 0, None, None),
        ("tok-c", "sharing", "tok-c", 0, None, None),
    ]
    assert _q(db_path, "SELECT COUNT(*) FROM api_tokens WHERE kv_tenant_id = ''") == [(0,)]
    assert _q(db_path, "SELECT kv_store_weight FROM models") == [(1,)]


async def test_0060_backfill_keeps_a_rotation_chain_in_one_family(
    tmp_data_dir, tmp_path, monkeypatch
):
    db_path = await _db_at_0058(tmp_data_dir, tmp_path, monkeypatch, chains=True)
    await _upgrade(db_path)
    assert _q(db_path, "SELECT id, kv_tenant_id FROM api_tokens ORDER BY id") == [
        ("adm-1", "adm-1"),
        ("orphan", "orphan"),  # its predecessor was deleted: rotated_from is NULL
        ("root", "root"),  # rotated, in its grace window
        ("succ-1", "root"),  # rotated again, in its grace window
        ("succ-2", "root"),  # the live head of the chain
        ("tok-a", "tok-a"),
        ("tok-b", "tok-b"),
        ("tok-c", "tok-c"),
    ]


async def test_0060_upgrade_is_idempotent(tmp_data_dir, tmp_path, monkeypatch):
    db_path = await _db_at_0058(tmp_data_dir, tmp_path, monkeypatch)
    await _upgrade(db_path)
    await _upgrade(db_path)  # the runner records 0060 and never replays it
    assert _q(
        db_path, "SELECT COUNT(*) FROM schema_migrations WHERE filename = '0060_kv_per_key.sql'"
    ) == [(1,)]


async def test_0060_schema_columns_and_retired_scopes_table(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    await _upgrade(db_path)
    tok = {r[1]: (r[2], r[3], r[4]) for r in _q(db_path, "PRAGMA table_info(api_tokens)")}
    assert tok["kv_isolation"] == ("TEXT", 1, "'sharing'")
    assert tok["kv_tenant_id"] == ("TEXT", 1, "''")
    assert tok["kv_epoch"] == ("INTEGER", 1, "0")
    assert tok["kv_max_pct_ram"] == ("INTEGER", 0, None)
    assert tok["kv_max_pct_nvme"] == ("INTEGER", 0, None)
    models = {r[1]: (r[2], r[3], r[4]) for r in _q(db_path, "PRAGMA table_info(models)")}
    assert models["kv_store_weight"] == ("INTEGER", 1, "1")
    hist = {r[1]: r[2] for r in _q(db_path, "PRAGMA table_info(request_history)")}
    assert hist["kv_isolated"] == "INTEGER"
    assert hist["kv_iso_cost_tokens"] == "INTEGER"
    retired = {
        r[1]: (r[2], r[3], r[5]) for r in _q(db_path, "PRAGMA table_info(kv_retired_scopes)")
    }
    assert retired == {
        "scope_tag": ("TEXT", 0, 1),
        "kv_tenant_id": ("TEXT", 1, 0),
        "retired_at": ("TEXT", 1, 0),
    }


async def test_a_new_key_gets_its_own_id_as_tenant_and_defaults_to_sharing(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        repo = TokenRepo(db)
        await repo.create("tok-new", "n", "vw_nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
        await repo.create(
            "tok-iso", "i", "vw_iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii", kv_isolation="isolated"
        )
        new = await repo.get("tok-new")
        iso = await repo.get("tok-iso")
    assert new is not None and iso is not None
    assert (new.kv_isolation, new.kv_tenant_id, new.kv_epoch) == ("sharing", "tok-new", 0)
    assert (new.kv_max_pct_ram, new.kv_max_pct_nvme) == (None, None)
    assert (iso.kv_isolation, iso.kv_tenant_id) == ("isolated", "tok-iso")


async def test_a_raw_insert_still_gets_a_tenant_id(tmp_data_dir):
    """Every insert path, not only the repo: an empty tenant id must never reach a row."""
    db_path = tmp_data_dir / "vllm-warden.db"
    await _upgrade(db_path)
    with sqlite3.connect(db_path) as c:
        c.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) "
            "VALUES ('raw-1', 'r', 'vw_rrrrr', 'h-raw-1', 'inference')"
        )
        c.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope, kv_tenant_id) "
            "VALUES ('raw-2', 'r2', 'vw_rrrrr', 'h-raw-2', 'inference', 'family-x')"
        )
        c.commit()
    assert _q(db_path, "SELECT id, kv_tenant_id FROM api_tokens ORDER BY id") == [
        ("raw-1", "raw-1"),
        ("raw-2", "family-x"),
    ]


async def test_rotate_carries_the_kv_policy_and_tenant_to_the_successor(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        repo = TokenRepo(db)
        await repo.create("tok-a", "a", "vw_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        await db.execute(
            "UPDATE api_tokens SET kv_isolation = 'isolated', kv_epoch = 3, "
            "kv_max_pct_ram = 40, kv_max_pct_nvme = 25 WHERE id = 'tok-a'"
        )
        await db.commit()
        new_id, _plain, _renamed = await repo.rotate("tok-a", grace_hours=1)
        succ = await repo.get(new_id)
    assert succ is not None and new_id != "tok-a"
    assert (
        succ.kv_isolation,
        succ.kv_tenant_id,
        succ.kv_epoch,
        succ.kv_max_pct_ram,
        succ.kv_max_pct_nvme,
    ) == ("isolated", "tok-a", 3, 40, 25)


def test_kv_columns_are_policy_and_appended_last():
    assert set(KV_TOKEN_COLS) <= set(_POLICY_COLS)
    names = [f for f in TokenRow.__dataclass_fields__]
    # In migration order: 0059's godmode_self_view, these (0060), then only
    # 0061's keep_captures (Prompt Lens), then 0066's per-key permissions.
    i = names.index("godmode_self_view")
    assert tuple(names[i + 1 : i + 1 + len(KV_TOKEN_COLS)]) == KV_TOKEN_COLS
    assert names[i + 1 + len(KV_TOKEN_COLS) :] == ["keep_captures", "prompt_lens", "engine_facts"]
    assert [c.strip() for c in _MODEL_COLS.split(",")][-1] == "kv_store_weight"
    assert list(ModelRow.__dataclass_fields__)[-1] == "kv_store_weight"


async def test_model_row_round_trips_kv_store_weight(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        repo = ModelRepo(db)
        base = dict(
            served_model_name="m",
            hf_repo="o/r",
            hf_revision="main",
            gpu_indices=[0],
            tensor_parallel_size=1,
            dtype=None,
            max_model_len=None,
            gpu_memory_utilization=0.9,
            trust_remote_code=False,
            extra_args=[],
            extra_env={},
            status="registered",
            pulled_bytes=0,
            pulled_total=None,
            last_error=None,
        )
        await repo.insert(ModelRow(id="m-default", **base))  # type: ignore[arg-type]
        await repo.insert(
            ModelRow(id="m-heavy", **{**base, "served_model_name": "m2"}, kv_store_weight=7)  # type: ignore[arg-type]
        )
        default = await repo.get("m-default")
        heavy = await repo.get("m-heavy")
    assert default is not None and default.kv_store_weight == 1
    assert heavy is not None and heavy.kv_store_weight == 7


@pytest.mark.parametrize("bad", ["", "SHARING", "open"])
async def test_kv_isolation_has_no_unknown_values_from_create(tmp_data_dir, bad):
    """``create`` refuses a value that is neither ``sharing`` nor ``isolated``."""
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        with pytest.raises(ValueError):
            await TokenRepo(db).create(
                "t", "t", "vw_tttttttttttttttttttttttttttttttt", kv_isolation=bad
            )


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO api_tokens(id, name, prefix, hash, kv_isolation) "
        "VALUES ('x', 'x', 'vw_x', 'h-x', 'open')",
        "UPDATE api_tokens SET kv_isolation = 'SHARING' WHERE id = 'tok-1'",
    ],
)
async def test_the_database_refuses_an_unknown_kv_isolation(tmp_data_dir, sql):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await TokenRepo(db).create("tok-1", "t", "vw_tttttttttttttttttttttttttttttttt")
    with sqlite3.connect(db_path) as c, pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        c.execute(sql)
    assert _q(db_path, "SELECT id, kv_isolation FROM api_tokens") == [("tok-1", "sharing")]


async def test_a_tenant_id_can_never_be_blanked(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await TokenRepo(db).create("tok-1", "t", "vw_tttttttttttttttttttttttttttttttt")
    with sqlite3.connect(db_path) as c:
        with pytest.raises(sqlite3.IntegrityError, match="kv_tenant_id must not be empty"):
            c.execute("UPDATE api_tokens SET kv_tenant_id = '' WHERE id = 'tok-1'")
        c.execute("UPDATE api_tokens SET kv_tenant_id = 'family-y' WHERE id = 'tok-1'")
        c.commit()
    assert _q(db_path, "SELECT kv_tenant_id FROM api_tokens") == [("family-y",)]
