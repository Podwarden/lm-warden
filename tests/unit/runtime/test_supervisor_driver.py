import pytest

from app.runtime.supervisor import ModelState, Supervisor
from tests.fakes.fake_engine import FakeDriver


class _Settings:
    def __init__(self, tmp_path):
        self.data_dir = str(tmp_path)
        self.hf_token_path = str(tmp_path / "tok")
        self.hf_cache_dir = str(tmp_path / "hf-cache")
        self.load_max_s = 7200.0


class _M:
    id = "qwen"
    hf_repo = "Qwen/Q"
    hf_revision = "main"
    served_model_name = "q"
    gpu_indices = [0]
    tensor_parallel_size = 1
    max_model_len = 4096
    dtype = "auto"
    gpu_memory_utilization = 0.9
    extra_env = {}
    extra_args = []


@pytest.mark.asyncio
async def test_load_uses_injected_driver(tmp_path):
    drv = FakeDriver()
    sup = Supervisor(_Settings(tmp_path), driver=drv)
    await sup.load(_M(), port=8001)
    assert len(drv.spawned) == 1
    assert drv.spawned[0].port == 8001
    assert sup.get_state("qwen") is ModelState.LOADING
    assert sup.get_pid("qwen") == 4242


@pytest.mark.asyncio
async def test_unload_terminates_via_driver(tmp_path):
    drv = FakeDriver()
    sup = Supervisor(_Settings(tmp_path), driver=drv)
    await sup.load(_M(), port=8001)
    await sup.unload("qwen", force=True)
    assert sup.is_running("qwen") is False


@pytest.mark.asyncio
async def test_get_host_falls_back_to_loopback_for_legacy_driver(tmp_path):
    # FakeDriver predates the engine_host protocol method; the supervisor
    # must degrade to loopback rather than crashing (the in-container
    # subprocess default).
    drv = FakeDriver()
    sup = Supervisor(_Settings(tmp_path), driver=drv)
    await sup.load(_M(), port=8001)
    assert sup.get_host("qwen") == "127.0.0.1"


@pytest.mark.asyncio
async def test_get_host_uses_driver_engine_host(tmp_path):
    # When the driver implements engine_host (the docker driver), the
    # supervisor records and returns whatever host it names so the probe /
    # warmup / proxy reach the engine container, not the API's own loopback.
    class _DnsDriver(FakeDriver):
        def engine_host(self, model_id: str) -> str:
            return f"vllm-warden-engine-{model_id}"

    sup = Supervisor(_Settings(tmp_path), driver=_DnsDriver())
    await sup.load(_M(), port=8001)
    assert sup.get_host("qwen") == "vllm-warden-engine-qwen"


@pytest.mark.asyncio
async def test_get_host_cleared_on_unload(tmp_path):
    sup = Supervisor(_Settings(tmp_path), driver=FakeDriver())
    await sup.load(_M(), port=8001)
    await sup.unload("qwen", force=True)
    assert sup.get_host("qwen") is None


# -- engine ports are returned to the pool on every way an engine goes away (verified finding #3)

import asyncio  # noqa: E402

from app.runtime.port_alloc import PortAllocator  # noqa: E402


@pytest.mark.asyncio
async def test_unload_returns_the_engines_port(tmp_path):
    ports = PortAllocator(start=10000, end=10000)
    sup = Supervisor(_Settings(tmp_path), driver=FakeDriver(), ports=ports)
    p = ports.allocate()
    await sup.load(_M(), port=p)
    await sup.unload("qwen", force=True)
    assert ports.allocate() == p


@pytest.mark.asyncio
async def test_an_engine_that_exits_returns_its_port(tmp_path):
    ports = PortAllocator(start=10000, end=10000)
    drv = FakeDriver()
    sup = Supervisor(_Settings(tmp_path), driver=drv, ports=ports)
    p = ports.allocate()
    await sup.load(_M(), port=p)
    drv.handles["qwen"]._finish(1)
    for _ in range(20):
        await asyncio.sleep(0)
    assert ports.allocate() == p


@pytest.mark.asyncio
async def test_many_load_unload_cycles_never_exhaust_a_one_port_pool(tmp_path):
    ports = PortAllocator(start=10000, end=10000)
    sup = Supervisor(_Settings(tmp_path), driver=FakeDriver(), ports=ports)
    for _ in range(50):
        await sup.load(_M(), port=ports.allocate())
        await sup.unload("qwen", force=True)
