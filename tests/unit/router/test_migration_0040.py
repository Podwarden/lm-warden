"""Migration 0040: ``router_rules`` + ``api_tokens.anthropic_relay`` (#287)."""

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.tokens import TokenRepo


async def _info(db, table):
    return {r[1]: r for r in await (await db.execute(f"PRAGMA table_info({table})")).fetchall()}


@pytest.mark.asyncio
async def test_router_rules_columns(tmp_path):
    async with open_db(tmp_path / "t.db") as db:
        await apply_migrations(db)
        info = await _info(db, "router_rules")
        idx = await (await db.execute("PRAGMA index_list(router_rules)")).fetchall()
    # PRAGMA row: (cid, name, type, notnull, dflt_value, pk)
    assert info["id"][5] == 1
    for col in ("position", "pattern", "target_model_id", "enabled", "fallback"):
        assert info[col][3] == 1, col
    for col in ("strip_thinking", "min_max_tokens", "created_at", "updated_at"):
        assert info[col][3] == 1, col
    assert info["enabled"][4] == "1"
    assert info["fallback"][4] == "1"
    assert info["strip_thinking"][4] == "1"
    assert info["min_max_tokens"][4] == "0"
    assert any("position" in r[1] for r in idx)


@pytest.mark.asyncio
async def test_anthropic_relay_column_defaults_to_zero(tmp_path):
    async with open_db(tmp_path / "t.db") as db:
        await apply_migrations(db)
        info = await _info(db, "api_tokens")
    assert info["anthropic_relay"][3] == 1
    assert info["anthropic_relay"][4] == "0"


@pytest.mark.asyncio
async def test_pre_0040_token_row_decodes_relay_off(tmp_path):
    async with open_db(tmp_path / "t.db") as db:
        await apply_migrations(db)
        await db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) "
            "VALUES ('t1', 'old', 'vw_aaaaa', 'h1', 'inference')"
        )
        await db.commit()
        row = await TokenRepo(db).get("t1")
    assert row is not None
    assert row.anthropic_relay == 0
