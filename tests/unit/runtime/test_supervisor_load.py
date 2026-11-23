from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.runtime.gpu_ownership import GpuConflict
from app.runtime.supervisor import EnginePinUnsupported, Supervisor
from tests.fakes.fake_engine import FakeDriver


class _Settings:
    data_dir = "/data"
    hf_token_path = "/data/hf-token"
    hf_cache_dir = "/root/.cache/huggingface"
    load_max_s = 7200.0


class _M:
    id = "qwen"
    hf_repo = "Qwen/Qwen3.5-9B"
    hf_revision = "main"
    served_model_name = "qwen3.5-9b"
    gpu_indices = [1, 2]
    tensor_parallel_size = 2
    max_model_len = 8192
    dtype = "auto"
    gpu_memory_utilization = 0.90
    extra_env = {}


@pytest.mark.asyncio
async def test_load_claims_gpus_and_spawns_subprocess(tmp_path):
    settings = _Settings()
    settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_text("hf_xxx")

    sup = Supervisor(settings)
    fake_proc = MagicMock()
    fake_proc.pid = 99999
    fake_proc.returncode = None

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake_proc)) as spawn:
        await sup.load(_M(), port=10001)

    assert sup.gpus.owner_of(1) == "qwen"
    assert sup.gpus.owner_of(2) == "qwen"
    assert "qwen" in sup._handles
    kwargs = spawn.call_args.kwargs
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1,2"
    assert "VW_ADMIN_PASSWORD" not in kwargs["env"]
    assert kwargs["start_new_session"] is True


@pytest.mark.asyncio
async def test_load_resolves_engine_image_from_model_channel(tmp_path):
    # A model carrying an engine axis (channel + vLLM version) gets its
    # image resolved and handed to the driver on spec.image (#161-min).
    settings = _Settings()
    settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_text("hf_xxx")
    driver = FakeDriver()
    sup = Supervisor(settings, driver=driver)
    m = _M()
    m.engine_channel = "cuda-stable"
    m.engine_vllm_version = "0.20.0"
    await sup.load(m, port=10001)
    assert driver.spawned[0].image == "vllm/vllm-openai:v0.20.0"


@pytest.mark.asyncio
async def test_load_without_engine_axis_leaves_image_none(tmp_path):
    # Legacy models (no engine axis) leave spec.image None so the driver
    # falls back to its default — the in-container path is unaffected.
    settings = _Settings()
    settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_text("hf_xxx")
    driver = FakeDriver()
    sup = Supervisor(settings, driver=driver)
    await sup.load(_M(), port=10001)
    assert driver.spawned[0].image is None


class _SubprocessStyleDriver(FakeDriver):
    """FakeDriver that, like the real in-container subprocess driver, cannot
    honor an engine-image pin."""

    supports_engine_image = False


@pytest.mark.asyncio
async def test_load_refuses_pin_under_subprocess_driver(tmp_path):
    # A driver that cannot swap the engine image + a model carrying an engine
    # pin => load() raises EnginePinUnsupported BEFORE any GPU is claimed, so
    # the pin is never silently discarded (the #177 bug) and no GPU is held.
    settings = _Settings()
    settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_text("hf_xxx")
    driver = _SubprocessStyleDriver()
    sup = Supervisor(settings, driver=driver)
    m = _M()
    m.engine_channel = "cuda-stable"
    m.engine_vllm_version = "0.21.0"

    with pytest.raises(EnginePinUnsupported) as exc:
        await sup.load(m, port=10001)

    # The pinned image is named in the message so the operator can act.
    assert "0.21.0" in str(exc.value)
    # No GPU claimed, no driver spawn, no handle registered.
    assert sup.gpus.owner_of(1) is None
    assert sup.gpus.owner_of(2) is None
    assert driver.spawned == []
    assert "qwen" not in sup._handles


@pytest.mark.asyncio
async def test_load_legacy_model_ok_under_subprocess_driver(tmp_path):
    # A legacy model with no engine axis (image resolves to None) loads fine
    # under the subprocess driver — no regression for the default path.
    settings = _Settings()
    settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_text("hf_xxx")
    driver = _SubprocessStyleDriver()
    sup = Supervisor(settings, driver=driver)
    await sup.load(_M(), port=10001)
    assert driver.spawned[0].image is None
    assert sup.gpus.owner_of(1) == "qwen"


@pytest.mark.asyncio
async def test_load_pin_ok_under_docker_driver(tmp_path):
    # A driver that CAN swap the engine image honors the pin — load succeeds
    # and the resolved image reaches the driver on spec.image.
    settings = _Settings()
    settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_text("hf_xxx")
    driver = FakeDriver()
    driver.supports_engine_image = True
    sup = Supervisor(settings, driver=driver)
    m = _M()
    m.engine_channel = "cuda-stable"
    m.engine_vllm_version = "0.20.0"
    await sup.load(m, port=10001)
    assert driver.spawned[0].image == "vllm/vllm-openai:v0.20.0"
    assert sup.gpus.owner_of(1) == "qwen"


@pytest.mark.asyncio
async def test_load_conflict_releases_gpus(tmp_path):
    settings = _Settings()
    settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_text("hf_xxx")
    sup = Supervisor(settings)

    m1 = _M()
    m1.id = "m1"
    m1.gpu_indices = [0, 1]
    m2 = _M()
    m2.id = "m2"
    m2.gpu_indices = [1, 2]

    fake_proc = MagicMock()
    fake_proc.pid = 1
    fake_proc.returncode = None
    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake_proc)):
        await sup.load(m1, port=10001)
        with pytest.raises(GpuConflict):
            await sup.load(m2, port=10002)

    assert sup.gpus.owner_of(0) == "m1"
    assert sup.gpus.owner_of(1) == "m1"
    assert sup.gpus.owner_of(2) is None


@pytest.mark.asyncio
async def test_failed_load_releases_gpus(tmp_path):
    settings = _Settings()
    settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_text("hf_xxx")
    sup = Supervisor(settings)
    m = _M()

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=OSError("nope"))):
        with pytest.raises(OSError):
            await sup.load(m, port=10001)

    assert sup.gpus.owner_of(1) is None
    assert sup.gpus.owner_of(2) is None
    assert "qwen" not in sup._handles


@pytest.mark.asyncio
async def test_load_increments_get_generation_per_attempt(tmp_path):
    """#275 follow-up — a load runner (routes_api.start_engine) reads
    ``get_generation()`` right after ``load()`` returns, then compares it
    again before writing a terminal status, so a superseded attempt can tell
    it is no longer current. The counter must be monotonic per model_id,
    start above 0 (0 means "never loaded"), and survive an unload/reload
    cycle without resetting or repeating a value.
    """
    settings = _Settings()
    settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_text("hf_xxx")
    driver = FakeDriver()
    sup = Supervisor(settings, driver=driver)
    m = _M()
    m.id = "gen-model"

    assert sup.get_generation(m.id) == 0

    await sup.load(m, port=10001)
    assert sup.get_generation(m.id) == 1

    await sup.unload(m.id, force=True)

    await sup.load(m, port=10002)
    assert sup.get_generation(m.id) == 2


@pytest.mark.asyncio
async def test_load_reports_the_spawned_wrapper_pid_to_the_hook(tmp_path):
    """#278 slice 3 — the watchdog's session ledger must learn a wrapper pid at
    spawn time, not at the next tick: a wrapper that dies within one tick is
    never sampled by remember_wrapper_pids, and without the spawn-time record
    the orphans it fathered are unattributable. The supervisor reports through
    a hook so it never imports the watchdog."""
    settings = _Settings()
    settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_text("hf_xxx")
    driver = FakeDriver()
    seen: list[tuple[str, int]] = []

    def hook(model_id: str, pid: int) -> None:
        seen.append((model_id, pid))

    sup = Supervisor(settings, driver=driver, on_wrapper_spawn=hook)
    m = _M()

    await sup.load(m, port=10001)

    assert seen == [("qwen", 4242)]


# --- #278 slice 1: the per-model restart/load claim ---------------------------
#
# The watchdog's restart and an operator /load both claim the model before
# touching it; the loser backs off. The claim is synchronous, per-model, and
# released only by its holder.


def test_claim_grants_to_first_holder_and_holds_for_others():
    sup = Supervisor(_Settings())
    assert sup.claim("qwen", "watchdog") is None, "the first claimant wins"
    assert sup.claim("qwen", "load") == "watchdog", "a second holder hears who holds"
    assert sup.claim("other", "load") is None, "the claim is per-model"


def test_claim_is_idempotent_for_the_same_holder():
    sup = Supervisor(_Settings())
    assert sup.claim("qwen", "watchdog") is None
    assert sup.claim("qwen", "watchdog") is None, "re-claiming is how _restart re-enters"


def test_release_claim_only_clears_for_the_holding_holder():
    sup = Supervisor(_Settings())
    sup.claim("qwen", "watchdog")
    sup.release_claim("qwen", "load")
    assert sup.claim("qwen", "load") == "watchdog", "a non-holder must not free the claim"
    sup.release_claim("qwen", "watchdog")
    assert sup.claim("qwen", "load") is None, "the holder's release frees the model"


class _FakeKv:
    def __init__(self, ok=True):
        self.calls = 0
        self.ok = ok
        self.gen = 1
        self.up = ok
        self.listeners = []

    async def ensure_running(self):
        self.calls += 1
        return self.ok

    def generation(self):
        return self.gen

    def healthy(self):
        return self.up

    def on_restart(self, cb):
        self.listeners.append(cb)


SHA = "c" * 40


async def _load_with_kv(tmp_path, kv_store_value, kv, *, cached=True, spawned=None):
    settings = _Settings()
    settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    settings.db_path = tmp_path / "w.db"  # kv_launch_kwargs derives the key path from it
    settings.hf_cache_dir = str(tmp_path / "hf")
    if cached:  # the pulled model: refs/main names the commit the engine loads
        refs = tmp_path / "hf" / "models--Qwen--Qwen3.5-9B" / "refs"
        refs.mkdir(parents=True)
        (refs / "main").write_text(SHA)
    (tmp_path / "hf-token").write_text("hf_xxx")
    model = _M()
    model.kv_store = kv_store_value
    sup = Supervisor(settings, kv_store=kv)
    fake_proc = MagicMock()
    fake_proc.pid = 99999
    fake_proc.returncode = None
    spawn = AsyncMock(return_value=fake_proc)
    with patch("asyncio.create_subprocess_exec", new=spawn):
        await sup.load(model, port=10001)
    if spawned is not None:
        spawned.extend(str(a) for a in spawn.call_args.args)
    return sup


@pytest.mark.asyncio
async def test_load_waits_for_the_kv_daemon_when_kv_store_is_on(tmp_path, caplog):
    kv = _FakeKv(ok=False)
    sup = await _load_with_kv(tmp_path, "ram", kv)
    assert kv.calls == 1
    assert "qwen" in sup._handles  # unhealthy daemon: still loads
    assert "KV tier off until it is reloaded" in caplog.text


@pytest.mark.asyncio
async def test_load_does_not_wait_when_kv_store_is_off(tmp_path):
    kv = _FakeKv()
    await _load_with_kv(tmp_path, "off", kv)
    assert kv.calls == 0


@pytest.mark.asyncio
async def test_load_refuses_kv_store_under_the_docker_driver(tmp_path):
    """F1: a sibling engine container cannot reach the daemon; refuse before any GPU claim."""
    from app.runtime.kv_store import KvTierUnsupported

    settings = _Settings()
    settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    settings.db_path = tmp_path / "w.db"
    settings.engine_driver = "docker"
    (tmp_path / "hf-token").write_text("hf_xxx")
    model = _M()
    model.kv_store = "ram"
    kv = _FakeKv()
    driver = FakeDriver()
    sup = Supervisor(settings, driver=driver, kv_store=kv)
    with pytest.raises(KvTierUnsupported, match="VW_ENGINE_DRIVER=docker"):
        await sup.load(model, port=10001)
    assert driver.spawned == []
    assert sup.gpus.owner_of(1) is None
    assert kv.calls == 0
    model.kv_store = "off"
    await sup.load(model, port=10001)
    assert len(driver.spawned) == 1


@pytest.mark.asyncio
async def test_kv_tier_is_on_then_off_until_reload_after_a_daemon_restart(tmp_path, caplog):
    """F3: a restarted daemon turns an engine's connector off for good; say so."""
    import logging

    kv = _FakeKv(ok=True)
    sup = await _load_with_kv(tmp_path, "ram", kv)
    assert sup.kv_tier("qwen") == {"state": "on", "reason": None, "generation": 1}
    kv.up = False
    assert sup.kv_tier("qwen")["state"] == "unavailable"
    kv.up, kv.gen = True, 2
    caplog.set_level(logging.WARNING, logger="app.runtime.supervisor")
    for cb in kv.listeners:
        cb(2)
    tier = sup.kv_tier("qwen")
    assert tier["state"] == "off_until_reload"
    assert "restarted" in tier["reason"]
    assert sup.kv_tiers() == {"qwen": tier}
    assert any(
        "qwen" in r.getMessage() and "until it is reloaded" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )
    await sup.unload("qwen", force=True)
    assert sup.kv_tier("qwen") is None


@pytest.mark.asyncio
async def test_kv_tier_is_off_until_reload_when_the_daemon_was_down_at_load(tmp_path):
    kv = _FakeKv(ok=False)
    sup = await _load_with_kv(tmp_path, "ram", kv)
    kv.up = True  # the daemon comes up later: this engine's connector is already off
    tier = sup.kv_tier("qwen")
    assert tier["state"] == "off_until_reload"
    assert "not healthy" in tier["reason"]


@pytest.mark.asyncio
async def test_kv_tier_is_none_for_a_model_without_the_tier(tmp_path):
    sup = await _load_with_kv(tmp_path, "off", _FakeKv())
    assert sup.kv_tier("qwen") is None
    assert sup.kv_tiers() == {}


def test_kv_store_admin_route_reports_daemon_and_tiers(client, tmp_data_dir):
    """F3: GET /api/system/kv-store shows the daemon generation and each engine's tier."""
    from tests.conftest import jwt_login, seed_admin_user

    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0])
    r = client.get("/api/system/kv-store", headers=jwt_login(client))
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["daemon"]) == {
        "healthy",
        "generation",
        "ram_requested_bytes",
        "ram_granted_bytes",
    }
    assert body["models"] == {}


def test_kv_store_admin_route_reports_requested_and_granted_ram(client, tmp_data_dir):
    """The daemon may shrink the pool to fit free memory: the route shows both sizes."""
    from types import SimpleNamespace

    from tests.conftest import jwt_login, seed_admin_user

    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0])
    client.app.state.kv_store = SimpleNamespace(
        healthy=lambda: True,
        generation=lambda: 2,
        ram=lambda: {"requested_bytes": 16 << 30, "granted_bytes": 6 << 30},
    )
    r = client.get("/api/system/kv-store", headers=jwt_login(client))
    assert r.status_code == 200, r.text
    d = r.json()["daemon"]
    assert d == {
        "healthy": True,
        "generation": 2,
        "ram_requested_bytes": 16 << 30,
        "ram_granted_bytes": 6 << 30,
    }


@pytest.mark.asyncio
async def test_load_keys_the_tier_by_the_resolved_commit(tmp_path):
    """F6: hf_revision 'main' becomes the commit in the cache, never the branch name."""
    import json

    argv: list[str] = []
    sup = await _load_with_kv(tmp_path, "ram", _FakeKv(), spawned=argv)
    extra = json.loads(argv[argv.index("--kv-transfer-config") + 1])["kv_connector_extra_config"]
    assert extra["revision"] == SHA and extra["tokenizer_revision"] == SHA
    assert sup.kv_tier("qwen")["state"] == "on"


@pytest.mark.asyncio
async def test_load_with_an_unresolvable_revision_runs_with_the_tier_off(tmp_path, caplog):
    """F6: no commit for 'main' in the cache: the engine loads without the connector."""
    import logging

    caplog.set_level(logging.WARNING, logger="app.runtime.supervisor")
    argv: list[str] = []
    sup = await _load_with_kv(tmp_path, "ram", _FakeKv(), cached=False, spawned=argv)
    assert "--kv-transfer-config" not in argv
    assert "qwen" in sup._handles
    tier = sup.kv_tier("qwen")
    assert tier["state"] == "off_until_reload" and "not in the model cache" in tier["reason"]
    assert any("KV tier off" in r.getMessage() for r in caplog.records)
