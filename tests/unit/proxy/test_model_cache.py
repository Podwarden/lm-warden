"""Model row cache (#293): unit behaviour and the /v1 resolution path."""

from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from app.proxy import model_cache as mc
from tests.unit.proxy.test_dp_routing_forward import AUTH, _post, _ready


def _row(id_="m1", served="m1", status="loaded") -> ModelRow:
    return ModelRow(
        id=id_,
        served_model_name=served,
        hf_repo="org/repo",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype=None,
        max_model_len=None,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        status=status,
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
        extra_env={},
    )


def test_hit_within_ttl_and_expiry():
    now = [0.0]
    c = mc.ModelCache(ttl_s=5.0, clock=lambda: now[0])
    c.put([_row()], mc.generation())
    assert c.by_served_name("m1").id == "m1"
    assert c.by_id("m1").served_model_name == "m1"
    assert c.by_served_name("nope") is None
    now[0] = 5.1
    assert c.by_served_name("m1") is mc.MISS
    assert c.rows() is mc.MISS


def test_invalidate_all_drops_every_instance():
    a, b = mc.ModelCache(), mc.ModelCache()
    a.put([_row()], mc.generation())
    b.put([_row()], mc.generation())
    mc.invalidate_all()
    assert a.rows() is mc.MISS and b.by_id("m1") is mc.MISS


def test_generation_snapshot_before_read_makes_stale_put_dead():
    c = mc.ModelCache()
    gen = mc.generation()
    mc.invalidate_all()
    c.put([_row()], gen)
    assert c.rows() is mc.MISS


def test_get_returns_a_deep_copy():
    c = mc.ModelCache()
    c.put([_row()], mc.generation())
    got = c.by_id("m1")
    got.gpu_indices.append(7)
    got.status = "failed"
    again = c.by_id("m1")
    assert again.gpu_indices == [0] and again.status == "loaded"
    rows = c.rows()
    rows.clear()
    assert len(c.rows()) == 1


#: The request-path modules. Background tasks of the app under test (the
#: watchdog's restore-on-boot pass, for one) also read the models table, and
#: under load one of them can land inside a test, so only reads made on behalf
#: of a request are counted.
_REQUEST_PATH = ("app/proxy/", "app/router/")


def _count_reads(monkeypatch):
    import sys

    calls = []

    def on_request_path():
        f = sys._getframe(2)
        while f is not None:
            if any(p in f.f_code.co_filename for p in _REQUEST_PATH):
                return True
            f = f.f_back
        return False

    for name in ("list_all", "get", "get_by_served_name"):
        orig = getattr(ModelRepo, name)

        def make(orig):
            async def counted(self, *a, **kw):
                if on_request_path():
                    calls.append(1)
                return await orig(self, *a, **kw)

            return counted

        monkeypatch.setattr(ModelRepo, name, make(orig))
    return calls


def _repo_call(client, tmp_data_dir, method, *args, **kw):
    async def go():
        async with open_db(str(tmp_data_dir / "vllm-warden.db")) as db:
            return await getattr(ModelRepo(db), method)(*args, **kw)

    return client.portal.call(go)


def test_two_requests_make_one_model_read(tmp_data_dir, client, monkeypatch):
    _ready(client, tmp_data_dir)
    calls = _count_reads(monkeypatch)
    assert _post(client)[0].status_code == 200
    assert _post(client)[0].status_code == 200
    assert len(calls) == 1


def test_v1_models_shares_the_cache(tmp_data_dir, client, monkeypatch):
    _ready(client, tmp_data_dir)
    calls = _count_reads(monkeypatch)
    assert _post(client)[0].status_code == 200
    r = client.get("/v1/models", headers=AUTH)
    assert [m["id"] for m in r.json()["data"]] == ["qwen"]
    assert len(calls) == 1


def test_status_change_is_visible_on_the_next_request(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    assert _post(client)[0].status_code == 200
    # The supervisor's status writes go through ModelRepo.update_status.
    _repo_call(client, tmp_data_dir, "update_status", "qwen", "loading")
    assert _post(client)[0].status_code == 404
    assert client.get("/v1/models", headers=AUTH).json()["data"] == []
    _repo_call(client, tmp_data_dir, "update_status", "qwen", "loaded")
    assert _post(client)[0].status_code == 200
    _repo_call(client, tmp_data_dir, "update_status", "qwen", "failed", "boom")
    assert _post(client)[0].status_code == 404


def test_unload_is_visible_on_the_next_request(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    assert _post(client)[0].status_code == 200
    _repo_call(client, tmp_data_dir, "update_status", "qwen", "pulled")
    assert _post(client)[0].status_code == 404


def test_rename_is_visible_on_the_next_request(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    assert _post(client)[0].status_code == 200
    _repo_call(client, tmp_data_dir, "update_fields", "qwen", {"served_model_name": "qwen-b"})
    assert _post(client)[0].status_code == 404
    body = {"model": "qwen-b", "messages": [{"role": "user", "content": "hi"}]}
    assert _post(client, body)[0].status_code == 200


def test_delete_is_visible_on_the_next_request(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    assert _post(client)[0].status_code == 200
    _repo_call(client, tmp_data_dir, "delete", "qwen")
    assert _post(client)[0].status_code == 404


def test_status_change_racing_an_inflight_read_wins(tmp_data_dir, client, monkeypatch):
    _ready(client, tmp_data_dir)
    real = ModelRepo.list_all

    async def read_then_fail(self):
        rows = await real(self)
        # The supervisor marks the model failed after this request read the
        # table, before it caches it.
        async with open_db(str(tmp_data_dir / "vllm-warden.db")) as db2:
            await ModelRepo(db2).update_status("qwen", "failed", "engine died")
        return rows

    monkeypatch.setattr(ModelRepo, "list_all", read_then_fail)
    _post(client)  # may be 200: it read the row before the failure
    monkeypatch.setattr(ModelRepo, "list_all", real)
    assert _post(client)[0].status_code == 404


def test_cache_failure_falls_open_to_a_db_read(tmp_data_dir, client, monkeypatch):
    _ready(client, tmp_data_dir)

    def boom(*a, **kw):
        raise RuntimeError("cache broken")

    monkeypatch.setattr(mc.ModelCache, "by_served_name", boom)
    monkeypatch.setattr(mc.ModelCache, "put", boom)
    assert _post(client)[0].status_code == 200
