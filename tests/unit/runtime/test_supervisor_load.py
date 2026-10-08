from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.runtime.gpu_ownership import GpuConflict
from app.runtime.supervisor import EnginePinUnsupported, Supervisor
from tests.fakes.fake_engine import FakeDriver


class _Settings:
    data_dir = "/data"
    hf_token_path = "/data/hf-token"
    hf_cache_dir = "/root/.cache/huggingface"


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
