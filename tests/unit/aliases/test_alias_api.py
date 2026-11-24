"""Model aliases: the model_aliases table and the /api/aliases admin API."""

import asyncio
from pathlib import Path

import pytest

from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from tests.conftest import csrf_header, jwt_login


def _model(mid: str, name: str, *, status: str = "registered") -> ModelRow:
    return ModelRow(
        id=mid,
        served_model_name=name,
        hf_repo="a/b",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        status=status,
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
        extra_env={},
    )


def _seed_models(db_path: Path, *models: ModelRow) -> None:
    async def go() -> None:
        async with open_db(db_path) as db:
            repo = ModelRepo(db)
            for m in models:
                await repo.insert(m)

    asyncio.run(go())


@pytest.fixture
def api(client, seeded_db):
    _seed_models(seeded_db, _model("m1", "qwen"), _model("m2", "llama", status="loaded"))
    return client, {**jwt_login(client), **csrf_header(client)}, seeded_db


def test_create_list_update_delete(api):
    c, h, _ = api
    r = c.post(
        "/api/aliases",
        json={
            "name": "lmw-default",
            "targets": ["m1", "m2"],
            "guard": "sanitize",
            "thinking_policy": "off",
            "params": {"max_tokens": {"value": 16384, "mode": "cap"}},
        },
        headers=h,
    )
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["name"] == "lmw-default"
    assert out["guard"] == "sanitize"
    assert out["thinking_policy"] == "off"
    assert out["params"] == {"max_tokens": {"value": 16384, "mode": "cap"}}
    assert out["enabled"] is True
    assert out["created_at"] and out["updated_at"]
    assert [t["id"] for t in out["targets"]] == ["m1", "m2"]
    assert [t["served_model_name"] for t in out["targets"]] == ["qwen", "llama"]
    assert [t["status"] for t in out["targets"]] == ["registered", "loaded"]

    r = c.get("/api/aliases", headers=h)
    assert r.status_code == 200, r.text
    aliases = r.json()["aliases"]
    assert [a["name"] for a in aliases] == ["lmw-default"]

    r = c.get(f"/api/aliases/{out['id']}", headers=h)
    assert r.status_code == 200
    assert r.json()["id"] == out["id"] and r.json()["name"] == "lmw-default"

    r = c.patch(
        f"/api/aliases/{out['id']}",
        json={"thinking_policy": "on", "enabled": False, "targets": ["m2"]},
        headers=h,
    )
    assert r.status_code == 200, r.text
    assert r.json()["thinking_policy"] == "on"
    assert r.json()["enabled"] is False
    assert [t["id"] for t in r.json()["targets"]] == ["m2"]

    r = c.delete(f"/api/aliases/{out['id']}", headers=h)
    assert r.status_code == 204, r.text
    assert c.get(f"/api/aliases/{out['id']}", headers=h).status_code == 404
    assert c.get("/api/aliases", headers=h).json()["aliases"] == []


def test_delete_missing_alias_404(api):
    c, h, _ = api
    r = c.delete("/api/aliases/nope", headers=h)
    assert r.status_code == 404


def test_duplicate_name_409(api):
    c, h, _ = api
    r = c.post("/api/aliases", json={"name": "dup", "targets": ["m1"]}, headers=h)
    assert r.status_code == 201, r.text
    r = c.post("/api/aliases", json={"name": "dup", "targets": ["m2"]}, headers=h)
    assert r.status_code == 409, r.text

    r = c.post("/api/aliases", json={"name": "other", "targets": ["m1"]}, headers=h)
    assert r.status_code == 201, r.text
    r = c.patch(f"/api/aliases/{r.json()['id']}", json={"name": "dup"}, headers=h)
    assert r.status_code == 409, r.text


def test_alias_may_share_a_model_name(api):
    c, h, _ = api
    r = c.post("/api/aliases", json={"name": "qwen", "targets": ["m1"]}, headers=h)
    assert r.status_code == 201, r.text


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"max_tokens": {"value": 0, "mode": "cap"}}, 422),
        ({"temperature": {"value": "hot", "mode": "force"}}, 422),
        ({"top_p": {"value": 0.9, "mode": "cap"}}, 201),
        ({"foo": {"value": 1, "mode": "force"}}, 422),
        ({"chat_template_kwargs.enable_thinking": {"value": False, "mode": "force"}}, 201),
        ({"reasoning_effort": {"value": "low", "mode": "cap"}}, 422),
        ({"seed": {"value": True, "mode": "force"}}, 422),
        ({"seed": {"value": 7, "mode": "cap"}}, 422),
        ({"seed": {"value": 7, "mode": "force"}}, 201),
        ({"min_p": {"value": 1.5, "mode": "default"}}, 422),
        ({"reasoning_effort": {"value": "bogus", "mode": "force"}}, 422),
        ({"max_tokens": {"value": 16, "mode": "cap"}}, 201),
        ({"chat_template_kwargs.enable_thinking": {"value": "yes", "mode": "default"}}, 201),
    ],
)
def test_params_validation(api, params, expected):
    c, h, _ = api
    r = c.post(
        "/api/aliases",
        json={"name": "p", "targets": ["m1"], "params": params},
        headers=h,
    )
    assert r.status_code == expected, r.text
    if expected == 422:
        key = next(iter(params))
        assert key in r.text, r.text


def test_create_rejects_unknown_fields(api):
    c, h, _ = api
    r = c.post("/api/aliases", json={"name": "x", "targets": ["m1"], "typo": 1}, headers=h)
    assert r.status_code == 422, r.text
    assert "typo" in r.text


def test_unknown_target_422(api):
    c, h, _ = api
    r = c.post("/api/aliases", json={"name": "t", "targets": ["nope"]}, headers=h)
    assert r.status_code == 422, r.text
    assert "nope" in r.text

    r = c.post("/api/aliases", json={"name": "ok", "targets": ["m1"]}, headers=h)
    assert r.status_code == 201, r.text
    r = c.patch(f"/api/aliases/{r.json()['id']}", json={"targets": ["nope"]}, headers=h)
    assert r.status_code == 422, r.text
    assert "nope" in r.text


def test_duplicate_targets_422(api):
    c, h, _ = api
    r = c.post(
        "/api/aliases",
        json={"name": "t", "targets": ["m1", "m1"]},
        headers=h,
    )
    assert r.status_code == 422, r.text


def test_deleting_model_drops_it_from_aliases(api):
    c, h, _ = api
    r = c.post(
        "/api/aliases",
        json={"name": "target", "targets": ["m1", "m2"]},
        headers=h,
    )
    assert r.status_code == 201, r.text
    alias_id = r.json()["id"]
    r = c.delete("/api/models/m1", headers=h)
    assert r.status_code == 204, r.text
    r = c.get(f"/api/aliases/{alias_id}", headers=h)
    assert r.status_code == 200, r.text
    assert [t["id"] for t in r.json()["targets"]] == ["m2"]
    r = c.get("/api/aliases", headers=h)
    assert [t["id"] for t in r.json()["aliases"][0]["targets"]] == ["m2"]


def test_requires_admin(client, seeded_db):
    # No Authorization header, no session cookie: CSRF is satisfied by a fresh
    # anonymous cookie+token (like the router control API's 401 test), then
    # require_jwt refuses with 401.
    client.cookies.clear()
    csrf = csrf_header(client)
    for m, p in [
        ("GET", "/api/aliases"),
        ("POST", "/api/aliases"),
        ("GET", "/api/aliases/nope"),
        ("PATCH", "/api/aliases/nope"),
        ("DELETE", "/api/aliases/nope"),
    ]:
        r = client.request(m, p, json={}, headers=csrf)
        assert r.status_code == 401, (m, p)


# -- the repo and the shared cache -------------------------------------------


async def test_repo_roundtrip_and_drop_target(tmp_data_dir):
    from app.db.database import open_db
    from app.db.migrations import apply_migrations
    from app.db.repos.model_aliases import AliasRow, ModelAliasRepo
    from app.proxy import model_cache as mc

    path = tmp_data_dir / "vllm-warden.db"
    async with open_db(path) as conn:
        await apply_migrations(conn)
        repo = ModelAliasRepo(conn)
        gen = mc.generation()
        row = await repo.create(AliasRow(id="a1", name="one", targets=["m1", "m2"]))
        assert row.id == "a1"
        assert row.targets == ["m1", "m2"]
        assert row.guard == "inherit"
        assert row.enabled is True
        assert await repo.get_by_name("one") is not None
        assert mc.generation() > gen

        updated = await repo.update("a1", {"guard": "block", "enabled": False})
        assert updated is not None
        assert updated.guard == "block" and updated.enabled is False
        assert mc.generation() > gen

        assert await repo.drop_target("m1") == 1
        assert (await repo.get("a1")).targets == ["m2"]
        assert mc.generation() > gen

        assert await repo.delete("a1") is True
        assert await repo.get("a1") is None
        assert await repo.delete("a1") is False


async def test_alias_cache_invalidated_by_writes_and_never_shared(tmp_data_dir):
    from app.db.database import open_db
    from app.db.migrations import apply_migrations
    from app.db.repos.model_aliases import AliasRow, ModelAliasRepo
    from app.proxy import model_cache as mc

    path = tmp_data_dir / "vllm-warden.db"
    async with open_db(path) as conn:
        await apply_migrations(conn)
        repo = ModelAliasRepo(conn)
        await repo.create(AliasRow(id="a1", name="one", targets=["m1"]))

        cache = mc.ModelCache()
        assert [a.name for a in await mc.all_aliases(cache, path)] == ["one"]

        # A write drops the snapshot for the next read.
        await repo.update("a1", {"guard": "sanitize"})
        assert [a.guard for a in await mc.all_aliases(cache, path)] == ["sanitize"]

        # Mutating one returned list cannot corrupt the cached snapshot.
        again = await mc.all_aliases(cache, path)
        again[0].targets.append("m2")
        assert (await mc.all_aliases(cache, path))[0].targets == ["m1"]

        # An absent name is a cached None, not a miss.
        assert await mc.alias_by_name(cache, path, "gone") is None
        await repo.create(AliasRow(id="a2", name="gone", targets=["m1"]))
        assert {a.id for a in await mc.all_aliases(cache, path)} == {"a1", "a2"}
